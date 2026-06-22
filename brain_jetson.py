"""
Robot Brain — Jetson Nano Edition
===================================
All optimizations applied for NVIDIA Jetson Nano:
  - TensorRT YOLO (yolov8n.engine) with automatic fallback to .pt
  - CUDA-accelerated YuNet face detection + SFace face recognition
  - YOLO frame-skip (runs every YOLO_SKIP frames, boxes drawn every frame)
  - Piper neural TTS → espeak fallback (no pyttsx3/Windows SAPI)
  - GStreamer camera support for both USB and CSI (ribbon-cable) cameras
  - Linux serial port for Arduino (/dev/ttyUSB0)
  - open_app uses xdg-open instead of Windows cmd

One-time setup on the Nano (run these before starting):
  sudo nvpmodel -m 0
  sudo jetson_clocks
  sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile
  sudo mkswap /swapfile && sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
  sudo apt install espeak libespeak-dev
  pip install anthropic websockets pyserial ultralytics

Export YOLO to TensorRT (run once on the Nano after PyTorch is installed):
  python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='engine', half=True, device=0)"

Optional better TTS voice (download model from github.com/rhasspy/piper/releases):
  pip install piper-tts
  # Place en_US-lessac-medium.onnx in the same folder as this file

Ports:
  8765 — WebSocket  (brain events → dashboard)
  8766 — HTTP       (MJPEG camera stream → dashboard)
"""

import asyncio
import json
import logging
import subprocess
import threading
import time
import serial
import serial.tools.list_ports
import numpy as np
import websockets
import anthropic
import cv2
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import pickle
import urllib.request

# ── Config ────────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = ""

# ── Hardware ──────────────────────────────────────────────────────────────────
ARDUINO_PORT  = "/dev/ttyUSB0"   # change to /dev/ttyACM0 if not found  (run: ls /dev/tty*)
ARDUINO_BAUD  = 9600
CAMERA_INDEX  = 0                # USB camera index (ignored when USE_CSI_CAMERA = True)
USE_CSI_CAMERA = False           # True = CSI ribbon-cable camera, False = USB

# ── Resolution ────────────────────────────────────────────────────────────────
FRAME_W = 640                    # reduce to 416 if CPU/GPU is still struggling
FRAME_H = 480

# ── Network ───────────────────────────────────────────────────────────────────
WS_HOST  = "localhost"
WS_PORT  = 8765
CAM_PORT = 8766

# ── Misc ──────────────────────────────────────────────────────────────────────
ENABLE_FACE = True
PEOPLE_FILE = "people.json"
WAKE_WORDS  = ["robot", "row bot", "row but", "ro bot", "ro but", "robat", "rowbot"]

# ── Face recognition ──────────────────────────────────────────────────────────
SFACE_MODEL     = "face_recognition_sface_2021dec.onnx"
YUNET_MODEL     = "face_detection_yunet_2023mar.onnx"
ENCODINGS_FILE  = "face_encodings.pkl"
SFACE_THRESHOLD = 0.45

# ── YOLO ──────────────────────────────────────────────────────────────────────
YOLO_MODEL = "yolov8n.engine"    # TensorRT — export from yolov8n.pt first (see header)
YOLO_SKIP  = 3                   # run YOLO every N frames; boxes are redrawn every frame

# ── Twilio / Knack ────────────────────────────────────────────────────────────
TWILIO_FACE_URL   = "https://ai-robotics-2574.twil.io/airobotics-customer-lookup"
TWILIO_APPID      = "itexps_customer_lookup"
TWILIO_SECRET_KEY = "9d7e3c4a1f8b6e2d5c"

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("brain")

# ── Shared camera frame ───────────────────────────────────────────────────────

latest_frame_lock  = threading.Lock()
latest_frame_bytes = None

# ── WebSocket state ───────────────────────────────────────────────────────────

ws_clients: set = set()
ws_command_handler    = None
ws_enrollment_handler = None
enrollment_ref        = None


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
                msg   = json.loads(raw)
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

MJPEG_BOUNDARY = b"--frame"


class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path.startswith("/video"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    with latest_frame_lock:
                        frame = latest_frame_bytes
                    if frame is not None:
                        self.wfile.write(MJPEG_BOUNDARY + b"\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    time.sleep(1 / 24)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass
        else:
            self.send_response(404)
            self.end_headers()


def start_mjpeg_server():
    HTTPServer.allow_reuse_address = True
    server = HTTPServer(("localhost", CAM_PORT), MJPEGHandler)
    log.info("MJPEG camera stream on http://localhost:%d/video", CAM_PORT)
    server.serve_forever()


# ── People store ──────────────────────────────────────────────────────────────

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

reload_encodings_event = threading.Event()


# ── Enrollment state machine ──────────────────────────────────────────────────

class EnrollmentManager:
    STABLE_SECS    = 3.0
    SAMPLES_NEEDED = 30
    NO_FACE_GRACE  = 1.5

    def __init__(self):
        self.state          = "idle"
        self.pending_key    = None
        self.pending_name   = None
        self.pending_phone  = None
        self.pending_twilio = {}
        self.frames         = []
        self._unknown_since = None
        self._no_face_since = None
        self._asking_at     = None


# ── Camera + face recognition thread ─────────────────────────────────────────

def camera_thread(loop, broadcast_fn, command_fn, enroll_mgr, say_fn, enroll_done_fn):
    global latest_frame_bytes, PEOPLE

    # ── Detect CUDA support ───────────────────────────────────────────────────
    _cuda_ok     = cv2.cuda.getCudaEnabledDeviceCount() > 0
    _dnn_backend = cv2.dnn.DNN_BACKEND_CUDA   if _cuda_ok else cv2.dnn.DNN_BACKEND_DEFAULT
    _dnn_target  = cv2.dnn.DNN_TARGET_CUDA    if _cuda_ok else cv2.dnn.DNN_TARGET_CPU
    log.info("OpenCV DNN backend: %s", "CUDA" if _cuda_ok else "CPU")

    # ── Face models ───────────────────────────────────────────────────────────
    sface_rec = cv2.FaceRecognizerSF.create(SFACE_MODEL, "", _dnn_backend, _dnn_target)
    yunet_det = cv2.FaceDetectorYN.create(
        YUNET_MODEL, "", (FRAME_W, FRAME_H), 0.5, 0.3, 5000, _dnn_backend, _dnn_target
    )
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

    # ── YOLO — try TensorRT engine first, fall back to .pt ───────────────────
    _trt = Path(YOLO_MODEL)
    _pt  = Path("yolov8n.pt")
    if _trt.exists():
        yolo = YOLO(str(_trt))
        log.info("YOLO loaded from TensorRT engine (%s)", YOLO_MODEL)
    elif _pt.exists():
        yolo = YOLO(str(_pt))
        log.warning("TensorRT engine not found — using PyTorch model (slower). "
                    "Export with: python3 -c \"from ultralytics import YOLO; "
                    "YOLO('yolov8n.pt').export(format='engine', half=True, device=0)\"")
    else:
        log.error("No YOLO model found. Place yolov8n.pt or yolov8n.engine in the working directory.")
        return

    # ── Camera ────────────────────────────────────────────────────────────────
    if USE_CSI_CAMERA:
        gst_pipeline = (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM),width={FRAME_W},height={FRAME_H},framerate=30/1 ! "
            f"nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
            f"video/x-raw,format=BGR ! appsink"
        )
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        log.info("Camera: CSI via GStreamer")
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        log.info("Camera: USB index %d", CAMERA_INDEX)

    if not cap.isOpened():
        log.error("Could not open camera")
        asyncio.run_coroutine_threadsafe(
            broadcast_fn("planning", "Camera not found — check index or CSI connection"), loop
        )
        return

    greeted          = set()
    last_seen        = None
    last_objects     = set()
    current_objects  = set()
    last_yolo_boxes  = []       # redrawn every frame even on skipped YOLO ticks
    yolo_frame_count = 0

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
            enroll_mgr._unknown_since = None
            enroll_mgr._no_face_since = None

        # ── Face capture for enrollment ───────────────────────────────────────
        if enroll_mgr.state == "capturing":
            if yunet_faces is not None and len(yunet_faces) > 0:
                try:
                    aligned = sface_rec.alignCrop(frame, yunet_faces[0])
                    feat    = sface_rec.feature(aligned).copy()
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

        # ── Object detection (YOLO — runs every YOLO_SKIP frames) ─────────────
        yolo_frame_count += 1
        if yolo_frame_count % YOLO_SKIP == 0:
            results = yolo(frame, verbose=False)[0]
            current_objects = set()
            last_yolo_boxes = []
            for box in results.boxes:
                cls_id          = int(box.cls[0])
                lbl             = yolo.names[cls_id]
                conf            = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                current_objects.add(lbl)
                last_yolo_boxes.append((x1, y1, x2, y2, lbl, conf))
            for obj in current_objects - last_objects:
                log.debug("Object appeared: %s", obj)
            for obj in last_objects - current_objects:
                log.debug("Object left frame: %s", obj)
            last_objects = current_objects

        # Draw YOLO boxes from most recent detection (smooth even on skipped frames)
        for x1, y1, x2, y2, lbl, conf in last_yolo_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
            cv2.rectangle(frame, (x1, y1 - 22), (x2, y1), (255, 165, 0), -1)
            cv2.putText(frame, f"{lbl} {conf:.0%}", (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
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

SYSTEM_PROMPT = """You are the intelligent brain of a small wheeled robot running on an NVIDIA Jetson Nano (Linux).
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
                "path": {"type": "string", "description": "Absolute directory path to list, e.g. /home/jetson/robot"}
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
        "description": "Open an application or file on the Linux computer using xdg-open.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "App name or full file path to open, e.g. /home/jetson/file.pdf"}
            },
            "required": ["target"]
        }
    },
]


class Brain:
    def __init__(self):
        self.client        = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY or None)
        self.history       = []
        self.tool_executor = None

    async def think(self, transcript: str) -> dict:
        self.history.append({"role": "user", "content": transcript})
        await broadcast("thinking", f"Processing: '{transcript}'")

        messages    = list(self.history[-10:])
        commands    = []
        final_reply = ""

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

        self.history = messages

        return {
            "reasoning": f"Used tools: {[b.name for b in response.content if hasattr(b, 'name') and b.type == 'tool_use']}",
            "commands":  commands,
            "reply":     final_reply,
            "schedule":  None,
        }


# ── Speaker ───────────────────────────────────────────────────────────────────
# Uses piper (neural TTS) if en_US-lessac-medium.onnx is present,
# otherwise falls back to espeak. No pyttsx3 / Windows SAPI.

class Speaker:
    def __init__(self):
        self._lock       = threading.Lock()
        self._piper_model = Path("en_US-lessac-medium.onnx")
        if self._piper_model.exists():
            log.info("TTS: piper (%s)", self._piper_model)
        else:
            log.info("TTS: espeak (place en_US-lessac-medium.onnx here for better quality)")

    def _speak_sync(self, text: str):
        # Sanitise text so the shell doesn't choke on quotes or special chars
        safe = text.replace('"', "'").replace('`', "'").replace('$', '')
        with self._lock:
            try:
                if self._piper_model.exists():
                    subprocess.run(
                        f'echo "{safe}" | piper --model {self._piper_model} '
                        f'--output_raw | aplay -r 22050 -f S16_LE -t raw -',
                        shell=True, check=False, timeout=30
                    )
                else:
                    subprocess.run(
                        ["espeak", "-s", "150", "-v", "en", safe],
                        check=False, timeout=30
                    )
            except Exception as exc:
                log.warning("TTS failed: %s", exc)

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
    await broadcast("planning", "Robot brain started (Jetson Nano)")

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
        if name == "move":
            direction = inputs["direction"].upper()
            duration  = inputs.get("duration_ms", 1000)
            cmd       = "STOP" if direction == "STOP" else f"{direction} {duration}"
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
            return "Scheduled."

        if name == "get_weather":
            location = inputs["location"]
            try:
                url = f"https://wttr.in/{urllib.request.quote(location)}?format=j1"
                with urllib.request.urlopen(url, timeout=5) as r:
                    data = json.loads(r.read())
                current  = data["current_condition"][0]
                temp_f   = current["temp_F"]
                desc     = current["weatherDesc"][0]["value"]
                feels_f  = current["FeelsLikeF"]
                humidity = current["humidity"]
                await broadcast("planning", f"Weather fetched for {location}")
                return f"{location}: {desc}, {temp_f}°F (feels like {feels_f}°F), humidity {humidity}%"
            except Exception as exc:
                return f"Could not fetch weather for {location}: {exc}"

        if name == "open_app":
            target = inputs["target"]
            try:
                subprocess.Popen(["xdg-open", target])
                await broadcast("doing", f"Opened: {target}")
                return f"Opened {target}."
            except Exception as exc:
                return f"Error opening {target}: {exc}"

        return f"Unknown tool: {name}"

    brain.tool_executor = tool_executor
    ws_command_handler  = handle_command

    async def handle_enroll_response(phone: str, name: str):
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

            await speaker.say(f"All set! Welcome, {confirmed_name}!")
        finally:
            _reset_enrollment()

    def say_sync(text: str):
        asyncio.run_coroutine_threadsafe(speaker.say(text), loop)

    def enroll_done():
        asyncio.run_coroutine_threadsafe(finish_enrollment(), loop)

    if ENABLE_FACE:
        t = threading.Thread(
            target=camera_thread,
            args=(loop, broadcast, handle_command, enrollment, say_sync, enroll_done),
            daemon=True,
        )
        t.start()
        await broadcast("planning", f"Camera thread started")

    while True:
        await asyncio.sleep(3600)


async def main():
    mjpeg_thread = threading.Thread(target=start_mjpeg_server, daemon=True)
    mjpeg_thread.start()

    ws_server = await websockets.serve(ws_handler, WS_HOST, WS_PORT, reuse_address=True)
    log.info("WebSocket on ws://%s:%d", WS_HOST, WS_PORT)

    await asyncio.gather(ws_server.serve_forever(), run_brain())


if __name__ == "__main__":
    asyncio.run(main())
