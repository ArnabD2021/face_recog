"""
Robot Brain — Unified
======================
Voice + Face Recognition + WebSocket dashboard + MJPEG camera stream

Ports:
  8765 — WebSocket  (brain events → dashboard)
  8766 — HTTP       (MJPEG camera stream → dashboard)

Requirements:
  pip install anthropic pyserial vosk websockets sounddevice numpy opencv-contrib-python ultralytics
"""

import asyncio
import json
import logging
import threading
import time
import serial
import serial.tools.list_ports
# import sounddevice as sd  # voice input disabled
import numpy as np
import websockets
import anthropic
import cv2
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from datetime import datetime
# from vosk import Model, KaldiRecognizer  # voice input disabled
from ultralytics import YOLO
import pyttsx3
import pickle
import urllib.request

# ── Config ────────────────────────────────────────────────────────────────────

VOSK_MODEL_PATH   = "vosk-model-small-en-us-0.15"
ANTHROPIC_API_KEY = "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"          # paste your sk-ant-... key here
ARDUINO_PORT      = "COM3"
ARDUINO_BAUD      = 9600
WS_HOST           = "localhost"
WS_PORT           = 8765
CAM_PORT          = 8766        # MJPEG stream port
SAMPLE_RATE       = 16000
MIC_DEVICE        = 1
CAMERA_INDEX      = 1           # 0 = built-in, 1 = external
WAKE_WORDS        = ["robot", "row bot", "row but", "ro bot", "ro but", "robat", "rowbot"]
ENABLE_FACE       = True
PEOPLE_FILE       = "people.json"

TWILIO_FACE_URL   = "https://ai-robotics-2574.twil.io/airobotics-customer-lookup"
TWILIO_APPID      = "itexps_customer_lookup"
TWILIO_SECRET_KEY = "9d7e3c4a1f8b6e2d5c"

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("brain")

# ── Shared camera frame (written by camera thread, read by HTTP server) ───────
# This is a global variable that holds the latest JPEG frame as bytes.
# The camera thread updates it; the HTTP server reads it.

latest_frame_lock  = threading.Lock()
latest_frame_bytes = None   # raw JPEG bytes of the most recent camera frame


# ── WebSocket broadcast ───────────────────────────────────────────────────────

ws_clients: set = set()
ws_command_handler    = None   # set by run_brain() once handle_command is ready
ws_enrollment_handler = None   # set by run_brain() to handle dashboard enrollment responses
enrollment_ref        = None   # global reference so ws_handler can re-send enrollment_ask on reconnect


async def broadcast(event_type: str, text: str, extra: dict = None):
    payload = {
        "type":      event_type,
        "text":      text,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **(extra or {}),
    }
    log.info("[%s] %s", event_type.upper(), text)
    if ws_clients:
        msg = json.dumps(payload)
        await asyncio.gather(
            *[ws.send(msg) for ws in list(ws_clients)],
            return_exceptions=True,
        )


async def ws_handler(websocket):
    global ws_command_handler, ws_enrollment_handler
    ws_clients.add(websocket)
    log.info("Dashboard connected (%d total)", len(ws_clients))

    # If enrollment is already waiting for a name, immediately send the prompt
    # to any newly connected (or reconnected) dashboard client
    if enrollment_ref and enrollment_ref.state == "asking":
        try:
            await websocket.send(json.dumps({
                "type":      "enrollment_ask",
                "text":      "Unknown face — who is this?",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }))
        except Exception:
            pass

    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)
                mtype = msg.get("type")
                if mtype == "command" and ws_command_handler:
                    text = msg.get("text", "").strip()
                    if text:
                        await broadcast("hearing", f"Dashboard: '{text}'")
                        await ws_command_handler(text)
                elif mtype == "enroll_response":
                    phone = msg.get("phone", "").strip()
                    name  = msg.get("name", "").strip()
                    if phone and name and ws_enrollment_handler:
                        await ws_enrollment_handler(phone, name)
                elif mtype == "enroll_dismiss":
                    if enrollment_ref and enrollment_ref.state == "asking":
                        enrollment_ref.state = "idle"
                        enrollment_ref._unknown_since = None
            except Exception as e:
                log.warning("Bad dashboard message: %s", e)
    finally:
        ws_clients.discard(websocket)


# ── MJPEG HTTP server ─────────────────────────────────────────────────────────
#
# MJPEG works like this:
#   1. Browser requests /video
#   2. Server sends a special header saying "this is a multipart stream"
#   3. Server keeps sending JPEG frames separated by a boundary string
#   4. Browser renders each frame as it arrives — looks like live video
#
# This is the same format used by IP cameras and webcam servers.

MJPEG_BOUNDARY = b"--frame"

class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass   # silence HTTP access logs

    def do_GET(self):
        if self.path.startswith("/video"):
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary=frame")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                while True:
                    with latest_frame_lock:
                        frame = latest_frame_bytes

                    if frame is not None:
                        # Each frame is sent as a MIME part with its own headers
                        self.wfile.write(MJPEG_BOUNDARY + b"\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()

                    time.sleep(1 / 24)   # ~24 fps

            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass   # browser closed the tab
        else:
            self.send_response(404)
            self.end_headers()


def start_mjpeg_server():
    """Run the MJPEG HTTP server in its own thread."""
    HTTPServer.allow_reuse_address = True
    server = HTTPServer(("localhost", CAM_PORT), MJPEGHandler)
    log.info("MJPEG camera stream on http://localhost:%d/video", CAM_PORT)
    server.serve_forever()


# ── Camera + face recognition thread ─────────────────────────────────────────
#
# This runs in a background thread. It:
#   1. Reads frames from the webcam
#   2. Runs face detection on each frame
#   3. Draws boxes and labels on the frame
#   4. Encodes to JPEG and stores in latest_frame_bytes for the MJPEG server
#   5. Broadcasts face events to the dashboard via asyncio

SFACE_MODEL        = "face_recognition_sface_2021dec.onnx"
YUNET_MODEL        = "face_detection_yunet_2023mar.onnx"
ENCODINGS_FILE     = "face_encodings.pkl"
SFACE_THRESHOLD    = 0.45    # cosine score above this = same person

reload_encodings_event = threading.Event()


def load_people() -> dict:
    p = Path(PEOPLE_FILE)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def save_people(data: dict):
    Path(PEOPLE_FILE).write_text(json.dumps(data, indent=2))


def generate_face_id() -> str:
    date_str = datetime.now().strftime("%Y%m%d")
    counter  = len(PEOPLE) + 1
    return f"FACE_{date_str}_{counter:04d}"


PEOPLE = load_people()


class EnrollmentManager:
    STABLE_SECS    = 3.0   # seconds an unknown face must be present before asking
    SAMPLES_NEEDED = 30    # face crops to capture before training
    NO_FACE_GRACE  = 1.5   # seconds face can flicker away without resetting timer

    def __init__(self):
        self.state          = "idle"   # idle | asking | capturing | training
        self.pending_key    = None     # e.g. "john_doe"
        self.pending_name   = None     # e.g. "John Doe"
        self.pending_phone  = None     # e.g. "8477499559"
        self.pending_twilio = {}       # Twilio lookup result from handle_enroll_response
        self.frames         = []       # captured numpy face crops
        self._unknown_since = None     # time.time() stamp
        self._no_face_since = None     # time.time() stamp for grace period
        self._asking_at     = None     # time.time() stamp when asking started


def camera_thread(loop, broadcast_fn, command_fn, enroll_mgr, say_fn, enroll_done_fn):
    global latest_frame_bytes, PEOPLE

    sface_rec       = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")
    yunet_det       = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (640, 480), 0.5, 0.3, 5000)
    known_encodings = {}

    def load_recognizer():
        nonlocal known_encodings
        enc_path = Path(ENCODINGS_FILE)
        if enc_path.exists():
            with open(str(enc_path), "rb") as f:
                known_encodings = pickle.load(f)
            log.info("Face encodings loaded (%d people)", len(known_encodings))
        else:
            known_encodings = {}
            log.warning("No face encodings — detection only")

    load_recognizer()

    yolo = YOLO("yolov8n.pt")
    log.info("YOLO object detection model loaded")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        log.error("Could not open camera index %d", CAMERA_INDEX)
        asyncio.run_coroutine_threadsafe(
            broadcast_fn("planning", f"Camera not found (index {CAMERA_INDEX})"), loop
        )
        return

    log.info("Camera started on index %d", CAMERA_INDEX)
    greeted      = set()
    last_seen    = None
    last_objects = set()

    def emit(event_type, text):
        asyncio.run_coroutine_threadsafe(broadcast_fn(event_type, text), loop)

    while True:
        if reload_encodings_event.is_set():
            reload_encodings_event.clear()
            PEOPLE = load_people()
            load_recognizer()
            greeted.clear()
            emit("planning", "Face encodings reloaded")

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        h_frame, w_frame = frame.shape[:2]
        yunet_det.setInputSize((w_frame, h_frame))
        _, yunet_faces = yunet_det.detect(frame)

        current_seen = None
        any_unknown  = False

        if yunet_faces is not None:
            for face in yunet_faces:
                x = max(0, int(face[0]))
                y = max(0, int(face[1]))
                w = min(int(face[2]), w_frame - x)
                h = min(int(face[3]), h_frame - y)

                person_key, confidence = None, 0.0

                if known_encodings:
                    try:
                        aligned    = sface_rec.alignCrop(frame, face)
                        query_feat = sface_rec.feature(aligned)
                        best_key, best_score = None, -1.0
                        for pk, feats in known_encodings.items():
                            for kf in feats:
                                # ensure C-contiguous float32 so cv2.match doesn't reject pickled arrays
                                kf_clean = np.ascontiguousarray(kf, dtype=np.float32)
                                score = float(sface_rec.match(
                                    query_feat, kf_clean, cv2.FaceRecognizerSF_FR_COSINE
                                ))
                                if score > best_score:
                                    best_score = score
                                    best_key   = pk
                        if best_score >= SFACE_THRESHOLD:
                            person_key = best_key
                            confidence = round(best_score * 100, 1)
                        else:
                            confidence = round(max(0.0, best_score * 100), 1)
                            if best_key:
                                log.info("Best match: %s %.3f (need %.3f)", best_key, best_score, SFACE_THRESHOLD)
                    except Exception as exc:
                        log.warning("SFace match error: %s", exc)

                if person_key is not None:
                    person = PEOPLE.get(person_key, {})
                    name   = person.get("full_name", person_key)
                    color  = (0, 220, 120)
                    label  = f"{name} {confidence:.0f}%"
                    current_seen = person_key
                    if last_seen != person_key:
                        emit("hearing", f"Face detected: {name} ({confidence:.0f}% confidence)")
                    if person_key not in greeted:
                        greeted.add(person_key)
                        emit("thinking", f"Recognized {name} — greeting")
                        asyncio.run_coroutine_threadsafe(
                            command_fn(f"You just recognized {name}. Say hello."), loop
                        )
                else:
                    color        = (60, 60, 220)
                    label        = f"Unknown {confidence:.0f}%"
                    current_seen = "unknown"
                    any_unknown  = True
                    if last_seen != "unknown":
                        emit("hearing", "Unknown face detected")

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.rectangle(frame, (x, y-28), (x+w, y), color, -1)
                cv2.putText(frame, label, (x+6, y-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # ── Enrollment state machine ──────────────────────────────────────────
        if any_unknown:
            # At least one face in frame is unrecognised — run the enrollment timer
            enroll_mgr._no_face_since = None
            if enroll_mgr._unknown_since is None:
                enroll_mgr._unknown_since = time.time()
            elif (time.time() - enroll_mgr._unknown_since >= EnrollmentManager.STABLE_SECS
                  and enroll_mgr.state == "idle"):
                enroll_mgr.state      = "asking"
                enroll_mgr._asking_at = time.time()
                enroll_mgr._unknown_since = None
                emit("planning", "Unknown face — asking for name")
                asyncio.run_coroutine_threadsafe(
                    broadcast_fn("enrollment_ask", "Unknown face — who is this?"), loop
                )
                say_fn("Hi there! I don't recognize you. Please enter your name and phone number on the screen.")
        elif current_seen is None:
            # No face at all — start grace-period countdown
            if enroll_mgr._no_face_since is None:
                enroll_mgr._no_face_since = time.time()
            elif time.time() - enroll_mgr._no_face_since >= EnrollmentManager.NO_FACE_GRACE:
                enroll_mgr._unknown_since = None
                enroll_mgr._no_face_since = None
                if enroll_mgr.state in ("asking", "capturing"):
                    enroll_mgr.state = "idle"
                    enroll_mgr.frames = []
                    emit("planning", "Face left — enrollment cancelled")
        else:
            # Every detected face is a known person — clear timers only when no unknown is present
            enroll_mgr._unknown_since = None
            enroll_mgr._no_face_since = None

        # Capture SFace feature vectors while enrolling (no image files written)
        if enroll_mgr.state == "capturing":
            if yunet_faces is not None and len(yunet_faces) > 0:
                try:
                    aligned = sface_rec.alignCrop(frame, yunet_faces[0])
                    feat    = sface_rec.feature(aligned).copy()  # .copy() prevents stale OpenCV buffer reference
                    enroll_mgr.frames.append(feat)
                except Exception as e:
                    log.warning("Capture frame skipped: %s", e)
            progress = len(enroll_mgr.frames)
            label = f"ENROLLING: {enroll_mgr.pending_name or '?'}  {progress}/{EnrollmentManager.SAMPLES_NEEDED}"
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (0, 160, 220), -1)
            cv2.putText(frame, label, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if progress > 0 and progress % 10 == 0:
                emit("planning", f"Enrolling {enroll_mgr.pending_name}: {progress}/{EnrollmentManager.SAMPLES_NEEDED} frames")
            if progress >= EnrollmentManager.SAMPLES_NEEDED:
                enroll_mgr.state = "training"
                enroll_done_fn()

        if last_seen is not None and current_seen is None:
            emit("hearing", "Face left camera view")

        last_seen = current_seen

        # ── Object detection ──────────────────────────────────────────────────
        results = yolo(frame, verbose=False)[0]
        current_objects = set()

        for box in results.boxes:
            cls_id          = int(box.cls[0])
            label           = yolo.names[cls_id]
            conf            = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            current_objects.add(label)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
            cv2.rectangle(frame, (x1, y1 - 22), (x2, y1), (255, 165, 0), -1)
            cv2.putText(frame, f"{label} {conf:.0%}", (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        for obj in current_objects - last_objects:
            log.debug("Object appeared: %s", obj)
        for obj in last_objects - current_objects:
            log.debug("Object left frame: %s", obj)

        last_objects = current_objects

        _, jpeg = cv2.imencode(".jpg", frame)
        with latest_frame_lock:
            latest_frame_bytes = jpeg.tobytes()


# ── Arduino ───────────────────────────────────────────────────────────────────

class Arduino:
    def __init__(self):
        self.ser = None

    def connect(self, port=None, baud=ARDUINO_BAUD):
        if port is None:
            port = self._auto_detect()
        if port is None:
            log.warning("No Arduino found — commands logged only")
            return
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            log.info("Arduino connected on %s", port)
        except Exception as e:
            log.warning("Could not connect to Arduino: %s", e)

    def _auto_detect(self):
        for p in serial.tools.list_ports.comports():
            desc = (p.description or "").lower()
            if "arduino" in desc or "ch340" in desc or "cp210" in desc:
                return p.device
        return None

    def send(self, command: str):
        if self.ser and self.ser.is_open:
            self.ser.write((command.strip() + "\n").encode())


# ── LLM Brain ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the intelligent brain of a small wheeled robot connected to a Windows computer.
You help the user by controlling the robot, reading files, scheduling reminders, and answering questions.

Use your tools to fulfill every request. Guidelines:
- Always call the speak tool to give a verbal response — never reply with plain text alone.
- When asked about files, call list_files or read_file first, then speak a concise summary.
- Keep spoken responses short and conversational (1-3 sentences).
- For face recognition events, greet the person warmly using speak.
- Chain tools as needed: you can move the robot AND speak at the same time.
"""

TOOLS = [
    {
        "name": "move",
        "description": "Move the robot in a direction for a given duration.",
        "input_schema": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["forward", "backward", "left", "right", "stop"],
                    "description": "Direction to move, or stop."
                },
                "duration_ms": {
                    "type": "integer",
                    "description": "How long to move in milliseconds. Omit or use 0 for stop."
                }
            },
            "required": ["direction"]
        }
    },
    {
        "name": "speak",
        "description": "Make the robot say something out loud to the user.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "What the robot should say."}
            },
            "required": ["text"]
        }
    },
    {
        "name": "list_files",
        "description": "List files and folders inside a directory on the computer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute directory path to list, e.g. C:/Users/dasar/Desktop"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "read_file",
        "description": "Read the text contents of a file on the computer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the file."}
            },
            "required": ["path"]
        }
    },
    {
        "name": "schedule_task",
        "description": "Schedule a reminder or task to run after a delay.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task":          {"type": "string",  "description": "Description of what to do when the timer fires."},
                "delay_seconds": {"type": "integer", "description": "Delay in seconds before the task runs."}
            },
            "required": ["task", "delay_seconds"]
        }
    },
    {
        "name": "get_weather",
        "description": "Get the current weather for any city or location.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name or location, e.g. 'Cupertino' or 'New York'"}
            },
            "required": ["location"]
        }
    },
    {
        "name": "open_app",
        "description": "Open an application or file on the Windows computer using its name or path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "App name (e.g. notepad, chrome) or full file path to open."}
            },
            "required": ["target"]
        }
    },
]


class Brain:
    def __init__(self):
        self.client       = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY or None)
        self.history      = []
        self.tool_executor = None  # set by run_brain() after all callbacks are ready

    async def think(self, transcript: str) -> dict:
        self.history.append({"role": "user", "content": transcript})
        await broadcast("thinking", f"Processing: '{transcript}'")

        messages  = list(self.history[-10:])
        commands  = []
        final_reply = ""

        # Tool-use loop: Claude calls tools until stop_reason == "end_turn"
        while True:
            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                for block in response.content:
                    if hasattr(block, "text"):
                        final_reply = block.text
                break

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    await broadcast("thinking", f"Tool: {block.name}({list(block.input.keys())})")
                    if self.tool_executor:
                        result_text = await self.tool_executor(block.name, block.input, commands)
                    else:
                        result_text = "Tool executor not ready."
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     result_text,
                    })
                messages.append({"role": "user", "content": tool_results})
            else:
                break

        # Persist updated history
        self.history = messages

        return {
            "reasoning": f"Used tools: {[b.name for b in response.content if hasattr(b, 'name') and b.type == 'tool_use']}",
            "commands":  commands,
            "reply":     final_reply,
            "schedule":  None,
        }


# ── Speaker ───────────────────────────────────────────────────────────────────

class Speaker:
    def __init__(self):
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", 165)
        # Voices: David (male) or Zira (female) — swap the name below
        voices = self._engine.getProperty("voices")
        zira = next((v for v in voices if "Zira" in v.name), None)
        if zira:
            self._engine.setProperty("voice", zira.id)
        self._lock = threading.Lock()

    def _speak_sync(self, text: str):
        with self._lock:
            self._engine.say(text)
            self._engine.runAndWait()

    async def say(self, text: str):
        await asyncio.get_running_loop().run_in_executor(None, self._speak_sync, text)


# ── Scheduler ─────────────────────────────────────────────────────────────────

class Scheduler:
    def add(self, task_desc: str, delay_seconds: int, callback):
        async def _run():
            await asyncio.sleep(delay_seconds)
            await broadcast("planning", f"Timer fired: {task_desc}")
            await callback(f"Timer alert: {task_desc}")
        asyncio.create_task(_run())
        log.info("Scheduled '%s' in %ds", task_desc, delay_seconds)


# ── Voice listener (disabled) ─────────────────────────────────────────────────
# class Listener: ...  (commented out — voice input disabled for now)
# def strip_wake_word: ... (commented out)


# ── Main brain loop ───────────────────────────────────────────────────────────

async def run_brain():
    global enrollment_ref
    arduino    = Arduino()
    brain      = Brain()
    speaker    = Speaker()
    sched      = Scheduler()
    enrollment = EnrollmentManager()
    enrollment_ref = enrollment
    loop       = asyncio.get_event_loop()

    arduino.connect(ARDUINO_PORT)
    await broadcast("planning", "Robot brain started")

    global ws_command_handler, ws_enrollment_handler

    async def handle_command(transcript: str):
        result    = await brain.think(transcript)
        reasoning = result.get("reasoning", "")
        commands  = result.get("commands", [])
        reply     = result.get("reply", "")
        schedule  = result.get("schedule")

        await broadcast("thinking", reasoning)

        for cmd in commands:
            await broadcast("doing", f"-> {cmd}")
            arduino.send(cmd)
            if cmd.upper().startswith("SPEAK "):
                log.info("Speak: %s", cmd[6:])

        if reply:
            await broadcast("hearing", f"Reply: {reply}")

        if schedule:
            task_desc     = schedule.get("task", "Task")
            delay_seconds = int(schedule.get("delay_seconds", 60))
            sched.add(task_desc, delay_seconds, handle_command)
            await broadcast("planning", f"Scheduled '{task_desc}' in {delay_seconds}s")

    async def tool_executor(name: str, inputs: dict, commands: list) -> str:
        import subprocess

        if name == "move":
            direction  = inputs["direction"].upper()
            duration   = inputs.get("duration_ms", 1000)
            cmd        = "STOP" if direction == "STOP" else f"{direction} {duration}"
            commands.append(cmd)
            await broadcast("doing", f"Motor: {cmd}")
            arduino.send(cmd)
            return f"Queued motor command: {cmd}"

        if name == "speak":
            text = inputs["text"]
            commands.append(f"SPEAK {text}")
            await broadcast("doing", text)
            await speaker.say(text)
            return "Speaking."

        if name == "list_files":
            path = inputs["path"]
            try:
                p       = Path(path).expanduser()
                entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
                lines   = [("[DIR] " if e.is_dir() else "[FILE]") + " " + e.name for e in entries]
                result  = "\n".join(lines) if lines else "(empty)"
                await broadcast("planning", f"Listed {len(lines)} items in {path}")
                return result
            except Exception as exc:
                return f"Error listing {path}: {exc}"

        if name == "read_file":
            path = inputs["path"]
            try:
                p = Path(path).expanduser()
                if p.stat().st_size > 50_000:
                    return f"File too large ({p.stat().st_size} bytes). Max 50 KB."
                content = p.read_text(encoding="utf-8", errors="replace")
                await broadcast("planning", f"Read {path} ({len(content)} chars)")
                return content
            except Exception as exc:
                return f"Error reading {path}: {exc}"

        if name == "schedule_task":
            task  = inputs["task"]
            delay = int(inputs["delay_seconds"])
            sched.add(task, delay, handle_command)
            await broadcast("planning", f"Scheduled '{task}' in {delay}s")
            return f"Scheduled."

        if name == "get_weather":
            location = inputs["location"]
            try:
                import urllib.request
                url  = f"https://wttr.in/{urllib.request.quote(location)}?format=j1"
                with urllib.request.urlopen(url, timeout=5) as r:
                    data    = json.loads(r.read())
                current = data["current_condition"][0]
                temp_f  = current["temp_F"]
                desc    = current["weatherDesc"][0]["value"]
                feels_f = current["FeelsLikeF"]
                humidity = current["humidity"]
                await broadcast("planning", f"Weather fetched for {location}")
                return f"{location}: {desc}, {temp_f}°F (feels like {feels_f}°F), humidity {humidity}%"
            except Exception as exc:
                return f"Could not fetch weather for {location}: {exc}"

        if name == "open_app":
            target = inputs["target"]
            try:
                subprocess.Popen(["cmd", "/c", "start", "", target], shell=False)
                await broadcast("doing", f"Opened: {target}")
                return f"Opened {target}."
            except Exception as exc:
                return f"Error opening {target}: {exc}"

        return f"Unknown tool: {name}"

    brain.tool_executor = tool_executor
    ws_command_handler  = handle_command

    async def handle_enroll_response(phone: str, name: str):
        """Called when the dashboard submits name + phone for an unknown face."""
        if not phone or not name:
            return
        if enrollment.state in ("capturing", "training"):
            return
        key = name.lower().replace(" ", "_")
        enrollment.pending_key    = key
        enrollment.pending_name   = name
        enrollment.pending_phone  = phone
        enrollment.pending_twilio = {}
        enrollment.state          = "capturing"
        enrollment.frames         = []
        await broadcast("planning", f"Starting enrollment for {name}...")
        await speaker.say(f"Great to meet you, {name}! Hold still while I learn your face.")

    ws_enrollment_handler = handle_enroll_response

    async def register_face_twilio(face_id: str, name: str, phone: str, email: str) -> dict:
        """Create or update a Knack record with the person's face_id via Twilio."""
        payload = json.dumps({
            "appid":     TWILIO_APPID,
            "secretkey": TWILIO_SECRET_KEY,
            "face_id":   face_id,
            "name":      name,
            "phone":     phone or "",
            "email":     email or "",
        }).encode("utf-8")

        def _post():
            req = urllib.request.Request(
                TWILIO_FACE_URL,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))

        try:
            result = await loop.run_in_executor(None, _post)
            log.info("Twilio face register %s → %s", face_id, result)
            return result
        except Exception as exc:
            log.warning("Twilio face register failed: %s", exc)
            return {}

    async def finish_enrollment():
        global PEOPLE
        name     = enrollment.pending_name
        phone    = enrollment.pending_phone
        features = enrollment.frames[:]

        def _reset_enrollment():
            enrollment.state          = "idle"
            enrollment.pending_key    = None
            enrollment.pending_name   = None
            enrollment.pending_phone  = None
            enrollment.pending_twilio = {}
            enrollment.frames         = []
            enrollment._unknown_since = None
            enrollment._no_face_since = None

        if not features:
            await broadcast("planning", "Enrollment failed — no face features captured")
            await speaker.say("Sorry, I couldn't capture your face. Please try again.")
            _reset_enrollment()
            return

        confirmed_name  = name
        confirmed_email = None
        confirmed_role  = "Visitor"
        key     = confirmed_name.lower().replace(" ", "_")
        face_id = generate_face_id()

        try:
            enc_path = Path(ENCODINGS_FILE)
            enc_map  = {}
            if enc_path.exists():
                with open(str(enc_path), "rb") as f:
                    enc_map = pickle.load(f)
            enc_map[key] = features
            with open(ENCODINGS_FILE, "wb") as f:
                pickle.dump(enc_map, f)
            log.info("Saved %d feature vectors for %s", len(features), confirmed_name)
        except Exception as exc:
            log.error("Failed to save encodings: %s", exc)
            await broadcast("planning", f"Enrollment save error: {exc}")
            await speaker.say("Sorry, I had a problem saving your face. Please try again.")
            _reset_enrollment()
            return

        try:
            PEOPLE[key] = {
                "full_name":   confirmed_name,
                "role":        confirmed_role,
                "phone":       phone,
                "email":       confirmed_email,
                "face_id":     face_id,
                "enrolled_at": datetime.now().isoformat(timespec="seconds"),
                "twilio_raw":  {},
            }
            save_people(PEOPLE)
            await register_face_twilio(face_id, confirmed_name, phone or "", confirmed_email or "")

            reload_encodings_event.set()
            await broadcast("planning", f"Enrolled {confirmed_name} successfully ({len(features)} samples)")

            profile_lines = [f"Name: {confirmed_name}"]
            if phone:
                profile_lines.append(f"Phone: {phone}")
            if confirmed_email:
                profile_lines.append(f"Email: {confirmed_email}")
            if confirmed_role and confirmed_role != "Visitor":
                profile_lines.append(f"Role: {confirmed_role}")
            await broadcast("profile", " | ".join(profile_lines))

            speech = f"All set! Welcome, {confirmed_name}!"
            await speaker.say(speech)
        finally:
            _reset_enrollment()

    def say_sync(text: str):
        asyncio.run_coroutine_threadsafe(speaker.say(text), loop)

    def enroll_done():
        asyncio.run_coroutine_threadsafe(finish_enrollment(), loop)

    # Start camera thread
    if ENABLE_FACE:
        t = threading.Thread(
            target=camera_thread,
            args=(loop, broadcast, handle_command, enrollment, say_sync, enroll_done),
            daemon=True,
        )
        t.start()
        await broadcast("planning", f"Camera thread started (index {CAMERA_INDEX})")

    # Voice input disabled — dashboard text input and face enrollment handle all interaction
    while True:
        await asyncio.sleep(3600)


async def main():
    # Start MJPEG server in background thread
    mjpeg_thread = threading.Thread(target=start_mjpeg_server, daemon=True)
    mjpeg_thread.start()

    # Start WebSocket server
    ws_server = await websockets.serve(ws_handler, WS_HOST, WS_PORT, reuse_address=True)
    log.info("WebSocket on ws://%s:%d", WS_HOST, WS_PORT)

    await asyncio.gather(ws_server.serve_forever(), run_brain())


if __name__ == "__main__":
    asyncio.run(main())
