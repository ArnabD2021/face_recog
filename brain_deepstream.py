"""
Robot Brain — DeepStream Edition (Jetson Nano)
================================================
Uses NVIDIA DeepStream SDK for the entire video pipeline:
  - GStreamer handles camera capture with hardware H.264 decode
  - nvtracker assigns stable IDs to faces and objects across frames
  - nvdsosd draws bounding boxes GPU-side (no cv2.rectangle in Python)
  - YuNet (CUDA) runs in probe 1 to detect faces → adds NvDsObjectMeta
  - SFace (CUDA) runs in probe 2 to recognize faces → adds display meta
  - YOLO (TensorRT) runs in probe 1 every YOLO_SKIP frames → adds NvDsObjectMeta
  - asyncio handles WebSocket, Claude, enrollment (identical to jetson edition)

Pipeline:
  camera → nvstreammux
         → [probe-detect: YuNet + YOLO → add object metas]
         → nvtracker  (assigns stable track IDs)
         → [probe-recognize: SFace → add display metas + drive enrollment]
         → nvdsosd    (draws all boxes GPU-side)
         → nvvideoconvert → videoconvert → appsink (→ MJPEG server)

One-time setup on Jetson Nano (JetPack 4.6):
  sudo nvpmodel -m 0 && sudo jetson_clocks
  sudo apt install deepstream-6.0 espeak libespeak-dev
  sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile
  sudo mkswap /swapfile && sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
  pip install anthropic websockets pyserial

  # Install pyds (DeepStream Python bindings)
  cd /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps
  pip install ./bindings/dist/pyds-1.1*.whl

  # Export YOLO to TensorRT (run once)
  python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='engine', half=True, device=0)"

  # Optional neural TTS (place model in same folder as this file)
  pip install piper-tts   # then download en_US-lessac-medium.onnx

Ports: 8765 WebSocket, 8766 MJPEG
"""

import asyncio
import json
import logging
import pickle
import subprocess
import threading
import time
import urllib.request

import cv2
import numpy as np
import serial
import serial.tools.list_ports
import websockets
import anthropic

from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
import pyds

# ── Config ────────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = "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"

# ── Hardware ──────────────────────────────────────────────────────────────────
ARDUINO_PORT  = "/dev/ttyUSB0"
ARDUINO_BAUD  = 9600
CAMERA_DEVICE = "/dev/video0"     # USB camera device node
USE_CSI       = False             # True = CSI ribbon camera (nvarguscamerasrc)
FRAME_W       = 640
FRAME_H       = 480

# ── Network ───────────────────────────────────────────────────────────────────
WS_HOST  = "localhost"
WS_PORT  = 8765
CAM_PORT = 8766

# ── Features ──────────────────────────────────────────────────────────────────
ENABLE_FACE = True
PEOPLE_FILE = "people.json"
WAKE_WORDS  = ["robot", "row bot", "row but", "ro bot", "ro but", "robat", "rowbot"]

# ── Face recognition ──────────────────────────────────────────────────────────
SFACE_MODEL     = "face_recognition_sface_2021dec.onnx"
YUNET_MODEL     = "face_detection_yunet_2023mar.onnx"
ENCODINGS_FILE  = "face_encodings.pkl"
SFACE_THRESHOLD = 0.45
FACE_CLASS_ID   = 999   # synthetic class ID we assign to YuNet face detections

# ── YOLO ──────────────────────────────────────────────────────────────────────
YOLO_MODEL = "yolov8n.engine"
YOLO_SKIP  = 3

# ── Twilio ────────────────────────────────────────────────────────────────────
TWILIO_FACE_URL   = "https://ai-robotics-2574.twil.io/airobotics-customer-lookup"
TWILIO_APPID      = "itexps_customer_lookup"
TWILIO_SECRET_KEY = "9d7e3c4a1f8b6e2d5c"

# ── DeepStream tracker ────────────────────────────────────────────────────────
DS_TRACKER_LIB = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
DS_TRACKER_CFG = "configs/tracker.yml"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("brain")

# ── Shared frame buffer (GStreamer → MJPEG server) ────────────────────────────
latest_frame_lock  = threading.Lock()
latest_frame_bytes: bytes = None

# ── WebSocket clients ─────────────────────────────────────────────────────────
ws_clients: set = set()
ws_command_handler    = None
ws_enrollment_handler = None
enrollment_ref        = None

# ── asyncio ↔ GStreamer bridge ────────────────────────────────────────────────
_async_loop:  asyncio.AbstractEventLoop = None
_event_queue: asyncio.Queue             = None

# De-duplicate probe events so we don't flood the queue every frame
_last_event_ts: dict = {}
_EVENT_GAP = 2.0   # minimum seconds between identical event types


def _post_event(event: dict):
    """Thread-safe: called from GStreamer probe thread, posts into asyncio queue."""
    key = event["type"] + ":" + event.get("key", "") + ":" + event.get("label", "")
    now = time.time()
    if now - _last_event_ts.get(key, 0) < _EVENT_GAP:
        return
    _last_event_ts[key] = now
    if _async_loop and _event_queue:
        _async_loop.call_soon_threadsafe(_event_queue.put_nowait, event)


# ── Inference objects (global, initialized in run_brain before GStreamer starts)
_sface_rec       = None
_yunet_det       = None
_yolo            = None
_known_encodings: dict = {}
_enc_lock         = threading.Lock()
_yolo_tick        = 0

reload_encodings_event = threading.Event()

PEOPLE: dict = {}

# ── Enrollment finish callback ref (set in run_brain, called from probe) ──────
_finish_enrollment_fn = None


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
    return f"FACE_{datetime.now().strftime('%Y%m%d')}_{len(PEOPLE)+1:04d}"


def _reload_encodings():
    global _known_encodings, PEOPLE
    PEOPLE = load_people()
    enc_path = Path(ENCODINGS_FILE)
    with _enc_lock:
        if enc_path.exists():
            with open(str(enc_path), "rb") as f:
                _known_encodings = pickle.load(f)
            log.info("Encodings reloaded (%d people)", len(_known_encodings))
        else:
            _known_encodings = {}


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
        self.frames         = []
        self._unknown_since = None
        self._no_face_since = None


# ── DeepStream configs (written to disk on startup) ───────────────────────────

def _write_configs():
    Path("configs").mkdir(exist_ok=True)
    Path("configs/tracker.yml").write_text("""\
NvDCF:
  useUniqueID: 1
  maxTargetsPerStream: 30
  minDetectorConfidence: 0.3
  terminalAge: 60
  probationAge: 3
  minVisibilityRatio: 0.2
""")


# ── DeepStream probe 1: detect faces + objects, inject object metas ───────────

def _probe_detect(pad, info, _u):
    """
    Fires on mux srcpad (before tracker). Runs YuNet and YOLO, injects
    NvDsObjectMeta for each detection so nvtracker can assign stable IDs.
    """
    global _yolo_tick

    gst_buffer = info.get_buffer()
    if gst_buffer is None:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame    = batch_meta.frame_meta_list

    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        # Pull the frame into a numpy array (RGBA, from NVMM surface)
        try:
            n_frame    = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            frame_copy = np.array(n_frame, copy=True, order="C")
            frame_bgr  = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGR)
        except Exception as exc:
            log.debug("probe_detect frame extract: %s", exc)
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
            continue

        h, w = frame_bgr.shape[:2]

        # ── YuNet face detection ──────────────────────────────────────────────
        if _yunet_det is not None:
            _yunet_det.setInputSize((w, h))
            try:
                _, faces = _yunet_det.detect(frame_bgr)
                if faces is not None:
                    for face in faces:
                        fx = max(0, int(face[0]))
                        fy = max(0, int(face[1]))
                        fw = min(int(face[2]), w - fx)
                        fh = min(int(face[3]), h - fy)
                        conf = float(face[14]) if face.shape[0] > 14 else 0.9

                        obj = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
                        obj.class_id              = FACE_CLASS_ID
                        obj.confidence            = conf
                        obj.rect_params.left      = fx
                        obj.rect_params.top       = fy
                        obj.rect_params.width     = fw
                        obj.rect_params.height    = fh
                        obj.rect_params.border_width = 0
                        # Stash the raw YuNet row index so probe 2 can re-run alignCrop
                        # We encode x,y into misc_obj_info as a breadcrumb
                        obj.misc_obj_info[0] = fx
                        obj.misc_obj_info[1] = fy
                        pyds.nvds_add_obj_meta_to_frame(frame_meta, obj, None)
            except Exception as exc:
                log.debug("YuNet probe error: %s", exc)

        # ── YOLO object detection (every YOLO_SKIP frames) ───────────────────
        _yolo_tick += 1
        if _yolo is not None and _yolo_tick % YOLO_SKIP == 0:
            try:
                results = _yolo(frame_bgr, verbose=False)[0]
                for box in results.boxes:
                    cls_id          = int(box.cls[0])
                    conf            = float(box.conf[0])
                    lbl             = _yolo.names.get(cls_id, str(cls_id))
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    obj = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
                    obj.class_id              = cls_id
                    obj.confidence            = conf
                    obj.rect_params.left      = x1
                    obj.rect_params.top       = y1
                    obj.rect_params.width     = x2 - x1
                    obj.rect_params.height    = y2 - y1
                    obj.rect_params.border_width = 0
                    pyds.nvds_add_obj_meta_to_frame(frame_meta, obj, None)

                    _post_event({"type": "object", "label": lbl, "conf": round(conf, 2)})
            except Exception as exc:
                log.debug("YOLO probe error: %s", exc)

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


# ── DeepStream probe 2: recognize faces, drive enrollment, add display metas ──

def _probe_recognize(pad, info, u_data):
    """
    Fires on tracker srcpad (after tracker assigns stable IDs).
    Reads face objects (class_id == FACE_CLASS_ID), runs SFace recognition,
    adds NvDsDisplayMeta for colored boxes + labels, drives enrollment state.
    """
    gst_buffer = info.get_buffer()
    if gst_buffer is None:
        return Gst.PadProbeReturn.OK

    enrollment: EnrollmentManager = u_data

    if reload_encodings_event.is_set():
        reload_encodings_event.clear()
        _reload_encodings()

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame    = batch_meta.frame_meta_list

    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        # Pull frame for SFace alignCrop
        try:
            n_frame    = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
            frame_copy = np.array(n_frame, copy=True, order="C")
            frame_bgr  = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGR)
        except Exception as exc:
            log.debug("probe_recognize frame extract: %s", exc)
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
            continue

        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_rects  = 0
        display_meta.num_labels = 0

        any_unknown    = False
        has_any_face   = False

        with _enc_lock:
            encs_snapshot = dict(_known_encodings)

        l_obj = frame_meta.obj_meta_list
        while l_obj:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            cls_id = obj_meta.class_id

            # ── Face object ───────────────────────────────────────────────────
            if cls_id == FACE_CLASS_ID:
                has_any_face = True
                rx   = int(obj_meta.rect_params.left)
                ry   = int(obj_meta.rect_params.top)
                rw   = int(obj_meta.rect_params.width)
                rh   = int(obj_meta.rect_params.height)
                rcon = float(obj_meta.confidence)

                # Reconstruct the 15-element YuNet face vector for alignCrop
                face_vec = np.array(
                    [rx, ry, rw, rh, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, rcon],
                    dtype=np.float32,
                )

                person_key, confidence = None, 0.0
                aligned_feat = None

                if _sface_rec is not None and encs_snapshot:
                    try:
                        aligned      = _sface_rec.alignCrop(frame_bgr, face_vec)
                        query_feat   = _sface_rec.feature(aligned)
                        aligned_feat = (aligned, query_feat)

                        best_key, best_score = None, -1.0
                        for pk, feats in encs_snapshot.items():
                            for kf in feats:
                                score = float(_sface_rec.match(
                                    query_feat,
                                    np.ascontiguousarray(kf, dtype=np.float32),
                                    cv2.FaceRecognizerSF_FR_COSINE,
                                ))
                                if score > best_score:
                                    best_score, best_key = score, pk
                        if best_score >= SFACE_THRESHOLD:
                            person_key = best_key
                            confidence = round(best_score * 100, 1)
                        else:
                            confidence = round(max(0.0, best_score * 100), 1)
                    except Exception as exc:
                        log.debug("SFace error: %s", exc)

                # Decide color and label
                if person_key:
                    person = PEOPLE.get(person_key, {})
                    name   = person.get("full_name", person_key)
                    label  = f"{name} {confidence:.0f}%"
                    color  = (0.0, 0.86, 0.47, 1.0)   # green  (R,G,B,A  0-1)
                    _post_event({"type": "recognized", "key": person_key,
                                 "confidence": confidence})
                else:
                    label      = f"Unknown {confidence:.0f}%"
                    color      = (0.24, 0.24, 0.86, 1.0)  # blue
                    any_unknown = True

                # Draw rectangle via display meta
                if display_meta.num_rects < 16:
                    rect = display_meta.rect_params[display_meta.num_rects]
                    rect.left   = rx
                    rect.top    = ry
                    rect.width  = rw
                    rect.height = rh
                    rect.border_width           = 2
                    rect.border_color.red       = color[0]
                    rect.border_color.green     = color[1]
                    rect.border_color.blue      = color[2]
                    rect.border_color.alpha     = color[3]
                    rect.has_bg_color           = 0
                    display_meta.num_rects += 1

                # Draw label
                if display_meta.num_labels < 16:
                    tp = display_meta.text_params[display_meta.num_labels]
                    tp.display_text = label
                    tp.x_offset     = rx
                    tp.y_offset     = max(0, ry - 28)
                    tp.font_params.font_name  = "Serif"
                    tp.font_params.font_size  = 10
                    tp.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
                    tp.set_bg_clr        = 1
                    tp.text_bg_clr.red   = color[0]
                    tp.text_bg_clr.green = color[1]
                    tp.text_bg_clr.blue  = color[2]
                    tp.text_bg_clr.alpha = 0.8
                    display_meta.num_labels += 1

                # Collect enrollment frame
                if enrollment.state == "capturing" and aligned_feat is not None:
                    try:
                        _, query_feat = aligned_feat
                        enrollment.frames.append(query_feat.copy())
                        n = len(enrollment.frames)
                        if n > 0 and n % 10 == 0:
                            _post_event({"type": "enroll_progress",
                                         "count": n,
                                         "total": EnrollmentManager.SAMPLES_NEEDED})
                        if n >= EnrollmentManager.SAMPLES_NEEDED:
                            enrollment.state = "training"
                            if _async_loop and _finish_enrollment_fn:
                                _async_loop.call_soon_threadsafe(
                                    _async_loop.create_task,
                                    _finish_enrollment_fn(),
                                )
                    except Exception as exc:
                        log.debug("Enrollment frame: %s", exc)

            # ── Non-face YOLO object: draw orange box ─────────────────────────
            elif display_meta.num_rects < 16:
                color = (1.0, 0.65, 0.0, 1.0)
                rect  = display_meta.rect_params[display_meta.num_rects]
                rect.left   = int(obj_meta.rect_params.left)
                rect.top    = int(obj_meta.rect_params.top)
                rect.width  = int(obj_meta.rect_params.width)
                rect.height = int(obj_meta.rect_params.height)
                rect.border_width           = 2
                rect.border_color.red       = color[0]
                rect.border_color.green     = color[1]
                rect.border_color.blue      = color[2]
                rect.border_color.alpha     = color[3]
                rect.has_bg_color           = 0
                display_meta.num_rects += 1

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        # ── Enrollment state machine ──────────────────────────────────────────
        if any_unknown:
            enrollment._no_face_since = None
            if enrollment._unknown_since is None:
                enrollment._unknown_since = time.time()
            elif (time.time() - enrollment._unknown_since >= EnrollmentManager.STABLE_SECS
                  and enrollment.state == "idle"):
                enrollment.state          = "asking"
                enrollment._unknown_since = None
                _post_event({"type": "enrollment_ask"})
        elif not has_any_face:
            if enrollment._no_face_since is None:
                enrollment._no_face_since = time.time()
            elif time.time() - enrollment._no_face_since >= EnrollmentManager.NO_FACE_GRACE:
                enrollment._unknown_since = None
                enrollment._no_face_since = None
                if enrollment.state in ("asking", "capturing"):
                    enrollment.state  = "idle"
                    enrollment.frames = []
                    _post_event({"type": "enrollment_cancelled"})
        else:
            enrollment._unknown_since = None
            enrollment._no_face_since = None

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


# ── appsink: grab rendered frame → MJPEG ──────────────────────────────────────

def _on_new_sample(sink, _):
    global latest_frame_bytes
    sample = sink.emit("pull-sample")
    if sample is None:
        return Gst.FlowReturn.OK
    buf  = sample.get_buffer()
    caps = sample.get_caps()
    ok, map_info = buf.map(Gst.MapFlags.READ)
    if ok:
        s = caps.get_structure(0)
        w = s.get_value("width")
        h = s.get_value("height")
        arr = np.frombuffer(map_info.data, dtype=np.uint8).reshape((h, w, 3))
        _, jpeg = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with latest_frame_lock:
            latest_frame_bytes = jpeg.tobytes()
        buf.unmap(map_info)
    return Gst.FlowReturn.OK


# ── GStreamer pipeline builder ────────────────────────────────────────────────

def _build_pipeline(enrollment: EnrollmentManager) -> Gst.Pipeline:
    def make(factory, name):
        el = Gst.ElementFactory.make(factory, name)
        if el is None:
            raise RuntimeError(
                f"GStreamer element '{factory}' not found. "
                "Is deepstream-6.0 installed and on the plugin path?"
            )
        pipeline.add(el)
        return el

    pipeline = Gst.Pipeline.new("robot-ds")

    # ── Source ────────────────────────────────────────────────────────────────
    if USE_CSI:
        src      = make("nvarguscamerasrc", "src")
        caps_src = make("capsfilter", "caps_src")
        caps_src.set_property("caps", Gst.Caps.from_string(
            f"video/x-raw(memory:NVMM),width={FRAME_W},height={FRAME_H},"
            f"framerate=30/1,format=NV12"
        ))
    else:
        src      = make("v4l2src", "src")
        src.set_property("device", CAMERA_DEVICE)
        caps_src = make("capsfilter", "caps_src")
        caps_src.set_property("caps", Gst.Caps.from_string(
            f"video/x-raw,width={FRAME_W},height={FRAME_H},framerate=30/1"
        ))

    conv_to_nvmm = make("nvvideoconvert", "conv_to_nvmm")
    caps_nvmm    = make("capsfilter", "caps_nvmm")
    caps_nvmm.set_property("caps", Gst.Caps.from_string(
        "video/x-raw(memory:NVMM),format=NV12"
    ))

    # ── Streammux ─────────────────────────────────────────────────────────────
    mux = make("nvstreammux", "mux")
    mux.set_property("batch-size",           1)
    mux.set_property("width",                FRAME_W)
    mux.set_property("height",               FRAME_H)
    mux.set_property("batched-push-timeout", 33333)
    mux.set_property("live-source",          1)

    # ── Tracker ───────────────────────────────────────────────────────────────
    tracker = make("nvtracker", "tracker")
    tracker.set_property("ll-lib-file",           DS_TRACKER_LIB)
    tracker.set_property("ll-config-file",        DS_TRACKER_CFG)
    tracker.set_property("enable-batch-process",  True)

    # ── On-screen display ─────────────────────────────────────────────────────
    osd = make("nvdsosd", "osd")
    osd.set_property("process-mode", 0)   # 0=CPU, 1=GPU (GPU needs EGL display)
    osd.set_property("display-text",  True)

    # ── Output chain: NVMM → BGR → appsink ───────────────────────────────────
    conv_out  = make("nvvideoconvert", "conv_out")
    caps_bgrx = make("capsfilter", "caps_bgrx")
    caps_bgrx.set_property("caps", Gst.Caps.from_string("video/x-raw,format=BGRx"))

    conv_bgr  = make("videoconvert", "conv_bgr")
    caps_bgr  = make("capsfilter", "caps_bgr")
    caps_bgr.set_property("caps", Gst.Caps.from_string("video/x-raw,format=BGR"))

    sink = make("appsink", "sink")
    sink.set_property("emit-signals", True)
    sink.set_property("max-buffers",  1)
    sink.set_property("drop",         True)
    sink.set_property("sync",         False)

    # ── Link source chain → mux via request pad ───────────────────────────────
    src.link(caps_src)
    caps_src.link(conv_to_nvmm)
    conv_to_nvmm.link(caps_nvmm)
    sink_pad = mux.get_request_pad("sink_0")
    caps_nvmm.get_static_pad("src").link(sink_pad)

    # ── Link mux → tracker → osd → output chain ──────────────────────────────
    mux.link(tracker)
    tracker.link(osd)
    osd.link(conv_out)
    conv_out.link(caps_bgrx)
    caps_bgrx.link(conv_bgr)
    conv_bgr.link(caps_bgr)
    caps_bgr.link(sink)

    # ── Attach probes ─────────────────────────────────────────────────────────
    # Probe 1: on mux src pad — detect faces + objects before tracker
    mux.get_static_pad("src").add_probe(
        Gst.PadProbeType.BUFFER, _probe_detect, None
    )
    # Probe 2: on tracker src pad — recognize faces, add display meta, enrollment
    tracker.get_static_pad("src").add_probe(
        Gst.PadProbeType.BUFFER, _probe_recognize, enrollment
    )

    # ── appsink callback ──────────────────────────────────────────────────────
    sink.connect("new-sample", _on_new_sample, None)

    return pipeline


def _run_gst(enrollment: EnrollmentManager):
    Gst.init(None)
    _write_configs()
    pipeline = _build_pipeline(enrollment)

    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_message(_, msg):
        if msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            log.error("GStreamer error: %s | %s", err, dbg)
        elif msg.type == Gst.MessageType.EOS:
            log.warning("GStreamer: end of stream")

    bus.connect("message", on_message)

    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        log.error("GStreamer pipeline failed to start — check camera and DeepStream install")
        return

    log.info("DeepStream pipeline playing")
    GLib.MainLoop().run()
    pipeline.set_state(Gst.State.NULL)


# ── MJPEG HTTP server ─────────────────────────────────────────────────────────

class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_GET(self):
        if not self.path.startswith("/video"):
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            while True:
                with latest_frame_lock:
                    frame = latest_frame_bytes
                if frame:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                    self.wfile.write(frame + b"\r\n")
                    self.wfile.flush()
                time.sleep(1 / 24)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass


def start_mjpeg_server():
    HTTPServer.allow_reuse_address = True
    server = HTTPServer(("localhost", CAM_PORT), MJPEGHandler)
    log.info("MJPEG on http://localhost:%d/video", CAM_PORT)
    server.serve_forever()


# ── WebSocket ─────────────────────────────────────────────────────────────────

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
                        enrollment_ref.state          = "idle"
                        enrollment_ref._unknown_since = None
            except Exception as e:
                log.warning("Bad ws message: %s", e)
    finally:
        ws_clients.discard(websocket)


# ── Arduino ───────────────────────────────────────────────────────────────────

class Arduino:
    def __init__(self):
        self.ser = None

    def connect(self, port=ARDUINO_PORT, baud=ARDUINO_BAUD):
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            log.info("Arduino on %s", port)
        except Exception as e:
            log.warning("Arduino not connected: %s", e)

    def send(self, cmd: str):
        if self.ser and self.ser.is_open:
            self.ser.write((cmd.strip() + "\n").encode())


# ── LLM brain ─────────────────────────────────────────────────────────────────

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
    {"name": "move", "description": "Move the robot in a direction.",
     "input_schema": {"type": "object",
         "properties": {
             "direction":   {"type": "string", "enum": ["forward","backward","left","right","stop"]},
             "duration_ms": {"type": "integer"}},
         "required": ["direction"]}},
    {"name": "speak", "description": "Make the robot say something out loud.",
     "input_schema": {"type": "object",
         "properties": {"text": {"type": "string"}},
         "required": ["text"]}},
    {"name": "list_files", "description": "List files in a directory.",
     "input_schema": {"type": "object",
         "properties": {"path": {"type": "string", "description": "Absolute path, e.g. /home/jetson/robot"}},
         "required": ["path"]}},
    {"name": "read_file", "description": "Read the text contents of a file.",
     "input_schema": {"type": "object",
         "properties": {"path": {"type": "string"}},
         "required": ["path"]}},
    {"name": "schedule_task", "description": "Schedule a reminder after a delay.",
     "input_schema": {"type": "object",
         "properties": {
             "task":          {"type": "string"},
             "delay_seconds": {"type": "integer"}},
         "required": ["task", "delay_seconds"]}},
    {"name": "get_weather", "description": "Get current weather for a location.",
     "input_schema": {"type": "object",
         "properties": {"location": {"type": "string"}},
         "required": ["location"]}},
    {"name": "open_app", "description": "Open an application or file with xdg-open.",
     "input_schema": {"type": "object",
         "properties": {"target": {"type": "string"}},
         "required": ["target"]}},
]


class Brain:
    def __init__(self):
        self.client        = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
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
                results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    await broadcast("thinking", f"Tool: {block.name}")
                    out = (await self.tool_executor(block.name, block.input, commands)
                           if self.tool_executor else "Tool executor not ready.")
                    results.append({"type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": out})
                messages.append({"role": "user", "content": results})
            else:
                break

        self.history = messages
        return {"commands": commands, "reply": final_reply}


# ── Speaker (piper → espeak fallback) ────────────────────────────────────────

class Speaker:
    def __init__(self):
        self._lock        = threading.Lock()
        self._piper_model = Path("en_US-lessac-medium.onnx")
        log.info("TTS: %s", "piper" if self._piper_model.exists() else "espeak")

    def _speak_sync(self, text: str):
        safe = text.replace('"', "'").replace("`", "'").replace("$", "")
        with self._lock:
            try:
                if self._piper_model.exists():
                    subprocess.run(
                        f'echo "{safe}" | piper --model {self._piper_model} '
                        f'--output_raw | aplay -r 22050 -f S16_LE -t raw -',
                        shell=True, check=False, timeout=30,
                    )
                else:
                    subprocess.run(
                        ["espeak", "-s", "150", "-v", "en", safe],
                        check=False, timeout=30,
                    )
            except Exception as exc:
                log.warning("TTS failed: %s", exc)

    async def say(self, text: str):
        await asyncio.get_running_loop().run_in_executor(None, self._speak_sync, text)


# ── Scheduler ─────────────────────────────────────────────────────────────────

class Scheduler:
    def add(self, task: str, delay: int, callback):
        async def _run():
            await asyncio.sleep(delay)
            await broadcast("planning", f"Timer: {task}")
            await callback(f"Timer alert: {task}")
        asyncio.create_task(_run())


# ── Main ──────────────────────────────────────────────────────────────────────

async def run_brain():
    global _async_loop, _event_queue, enrollment_ref
    global _sface_rec, _yunet_det, _yolo, _known_encodings, PEOPLE
    global ws_command_handler, ws_enrollment_handler, _finish_enrollment_fn

    _async_loop  = asyncio.get_event_loop()
    _event_queue = asyncio.Queue()

    PEOPLE = load_people()

    # ── Init CUDA-backed face models ──────────────────────────────────────────
    _cuda_ok     = cv2.cuda.getCudaEnabledDeviceCount() > 0
    _dnn_backend = cv2.dnn.DNN_BACKEND_CUDA  if _cuda_ok else cv2.dnn.DNN_BACKEND_DEFAULT
    _dnn_target  = cv2.dnn.DNN_TARGET_CUDA   if _cuda_ok else cv2.dnn.DNN_TARGET_CPU
    log.info("cv2 DNN: %s", "CUDA" if _cuda_ok else "CPU")

    _sface_rec = cv2.FaceRecognizerSF.create(SFACE_MODEL, "", _dnn_backend, _dnn_target)
    _yunet_det = cv2.FaceDetectorYN.create(
        YUNET_MODEL, "", (FRAME_W, FRAME_H), 0.5, 0.3, 5000, _dnn_backend, _dnn_target
    )
    _reload_encodings()

    # ── Init YOLO (TRT → .pt fallback) ───────────────────────────────────────
    trt_path = Path(YOLO_MODEL)
    pt_path  = Path("yolov8n.pt")
    if trt_path.exists():
        _yolo = YOLO(str(trt_path))
        log.info("YOLO: TensorRT (%s)", YOLO_MODEL)
    elif pt_path.exists():
        _yolo = YOLO(str(pt_path))
        log.warning("YOLO: PyTorch fallback — run export for TRT speedup")
    else:
        log.error("No YOLO model — place yolov8n.pt or yolov8n.engine here")

    arduino    = Arduino()
    brain      = Brain()
    speaker    = Speaker()
    sched      = Scheduler()
    greeted:   set = set()

    enrollment     = EnrollmentManager()
    enrollment_ref = enrollment

    arduino.connect()
    await broadcast("planning", "Robot brain started (DeepStream edition)")

    # ── Tool executor ─────────────────────────────────────────────────────────

    async def handle_command(transcript: str):
        result   = await brain.think(transcript)
        commands = result.get("commands", [])
        reply    = result.get("reply", "")
        for cmd in commands:
            await broadcast("doing", f"-> {cmd}")
            arduino.send(cmd)
        if reply:
            await broadcast("hearing", f"Reply: {reply}")

    ws_command_handler = handle_command

    async def tool_executor(name: str, inputs: dict, commands: list) -> str:
        if name == "move":
            direction = inputs["direction"].upper()
            duration  = inputs.get("duration_ms", 1000)
            cmd       = "STOP" if direction == "STOP" else f"{direction} {duration}"
            commands.append(cmd)
            await broadcast("doing", f"Motor: {cmd}")
            arduino.send(cmd)
            return f"Queued: {cmd}"

        if name == "speak":
            text = inputs["text"]
            commands.append(f"SPEAK {text}")
            await broadcast("doing", text)
            await speaker.say(text)
            return "Speaking."

        if name == "list_files":
            path = inputs["path"]
            try:
                entries = sorted(Path(path).expanduser().iterdir(),
                                 key=lambda e: (e.is_file(), e.name.lower()))
                lines   = [("[DIR] " if e.is_dir() else "[FILE]") + " " + e.name
                           for e in entries]
                await broadcast("planning", f"Listed {len(lines)} items in {path}")
                return "\n".join(lines) if lines else "(empty)"
            except Exception as exc:
                return f"Error listing {path}: {exc}"

        if name == "read_file":
            path = inputs["path"]
            try:
                p = Path(path).expanduser()
                if p.stat().st_size > 50_000:
                    return "File too large (>50 KB)."
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
                c = data["current_condition"][0]
                return (f"{location}: {c['weatherDesc'][0]['value']}, {c['temp_F']}°F "
                        f"(feels {c['FeelsLikeF']}°F), humidity {c['humidity']}%")
            except Exception as exc:
                return f"Could not fetch weather: {exc}"

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

    # ── Enrollment handlers ───────────────────────────────────────────────────

    async def handle_enroll_response(phone: str, name: str):
        if not phone or not name:
            return
        if enrollment.state in ("capturing", "training"):
            return
        enrollment.pending_key   = name.lower().replace(" ", "_")
        enrollment.pending_name  = name
        enrollment.pending_phone = phone
        enrollment.state         = "capturing"
        enrollment.frames        = []
        await broadcast("planning", f"Enrolling {name}…")
        await speaker.say(f"Great to meet you, {name}! Hold still while I learn your face.")

    ws_enrollment_handler = handle_enroll_response

    async def register_face_twilio(face_id: str, name: str, phone: str, email: str):
        payload = json.dumps({
            "appid": TWILIO_APPID, "secretkey": TWILIO_SECRET_KEY,
            "face_id": face_id, "name": name,
            "phone": phone or "", "email": email or "",
        }).encode()

        def _post():
            req = urllib.request.Request(
                TWILIO_FACE_URL, data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                return json.loads(r.read())

        try:
            return await _async_loop.run_in_executor(None, _post)
        except Exception as exc:
            log.warning("Twilio register failed: %s", exc)
            return {}

    async def finish_enrollment():
        global PEOPLE
        name     = enrollment.pending_name
        phone    = enrollment.pending_phone
        features = enrollment.frames[:]

        def _reset():
            enrollment.state          = "idle"
            enrollment.pending_key    = None
            enrollment.pending_name   = None
            enrollment.pending_phone  = None
            enrollment.frames         = []
            enrollment._unknown_since = None
            enrollment._no_face_since = None

        if not features:
            await broadcast("planning", "Enrollment failed — no face data captured")
            await speaker.say("Sorry, I couldn't capture your face. Please try again.")
            _reset()
            return

        key     = (name or "unknown").lower().replace(" ", "_")
        face_id = generate_face_id()

        # Save face encodings
        try:
            enc_map: dict = {}
            enc_path = Path(ENCODINGS_FILE)
            if enc_path.exists():
                with open(str(enc_path), "rb") as f:
                    enc_map = pickle.load(f)
            enc_map[key] = features
            with open(ENCODINGS_FILE, "wb") as f:
                pickle.dump(enc_map, f)
            log.info("Saved %d vectors for %s", len(features), name)
        except Exception as exc:
            log.error("Encodings save failed: %s", exc)
            await broadcast("planning", f"Enrollment save error: {exc}")
            await speaker.say("Sorry, I had a problem saving your face.")
            _reset()
            return

        try:
            PEOPLE[key] = {
                "full_name":   name,
                "role":        "Visitor",
                "phone":       phone,
                "email":       None,
                "face_id":     face_id,
                "enrolled_at": datetime.now().isoformat(timespec="seconds"),
                "twilio_raw":  {},
            }
            save_people(PEOPLE)
            await register_face_twilio(face_id, name, phone or "", "")
            reload_encodings_event.set()
            await broadcast("planning", f"Enrolled {name} successfully ({len(features)} samples)")
            profile = f"Name: {name}"
            if phone:
                profile += f" | Phone: {phone}"
            await broadcast("profile", profile)
            await speaker.say(f"All set! Welcome, {name}!")
        finally:
            _reset()

    _finish_enrollment_fn = finish_enrollment

    # ── Start GStreamer in background thread ──────────────────────────────────
    if ENABLE_FACE:
        gst_thread = threading.Thread(target=_run_gst, args=(enrollment,), daemon=True)
        gst_thread.start()
        await broadcast("planning", "DeepStream pipeline starting…")

    # ── Event consumer: GStreamer probe events → app logic ────────────────────
    async def event_consumer():
        while True:
            event = await _event_queue.get()
            etype = event["type"]

            if etype == "recognized":
                key    = event["key"]
                conf   = event.get("confidence", 0)
                person = PEOPLE.get(key, {})
                name   = person.get("full_name", key)
                await broadcast("hearing", f"Recognized: {name} ({conf:.0f}%)")
                if key not in greeted:
                    greeted.add(key)
                    await handle_command(f"You just recognized {name}. Say hello.")

            elif etype == "enrollment_ask":
                if enrollment.state == "asking":
                    await broadcast("enrollment_ask", "Unknown face — who is this?")
                    await speaker.say(
                        "Hi there! I don't recognize you. "
                        "Please enter your name and phone number on the screen."
                    )

            elif etype == "enroll_progress":
                n     = event.get("count", 0)
                total = event.get("total", EnrollmentManager.SAMPLES_NEEDED)
                await broadcast("planning", f"Enrolling: {n}/{total} frames")

            elif etype == "enrollment_cancelled":
                await broadcast("planning", "Enrollment cancelled — face left frame")

            elif etype == "object":
                label = event.get("label", "")
                conf  = event.get("conf", 0)
                await broadcast("hearing", f"Object: {label} ({conf:.0%})")

    await asyncio.gather(
        event_consumer(),
        asyncio.sleep(float("inf")),
    )


async def main():
    threading.Thread(target=start_mjpeg_server, daemon=True).start()
    ws_server = await websockets.serve(ws_handler, WS_HOST, WS_PORT, reuse_address=True)
    log.info("WebSocket on ws://%s:%d", WS_HOST, WS_PORT)
    await asyncio.gather(ws_server.serve_forever(), run_brain())


if __name__ == "__main__":
    asyncio.run(main())
