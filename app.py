
import os
import cv2
import math
import wave
import struct
import base64
import tempfile
import platform
import subprocess
import threading
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from ultralytics import YOLO


# ============================================================
# PPE Final GUI App - Helmet Recall Version
# Uses only models/best.pt
# GUI style: dashboard + sidebar + cards + original/result columns
# ============================================================

st.set_page_config(page_title="PPE Safety Monitoring Dashboard", page_icon="🦺", layout="wide")

MODEL_PATH = "models/best.pt"
RESULTS_DIR = "results"
LOG_PATH = "incident_log.csv"
ALARM_PATH = os.path.join(RESULTS_DIR, "danger_alarm.wav")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# CSS — PPE Safety Monitoring Dashboard
# ============================================================
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; height: 0; }
.block-container {
    padding-top: 0.6rem;
    padding-bottom: 1.5rem;
    max-width: 1280px;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #14532D 0%, #166534 100%);
}
[data-testid="stSidebar"] * { color: #F0FDF4 !important; }

.dash-header {
    background: linear-gradient(90deg, #14532D 0%, #166534 55%, #15803D 100%);
    border-radius: 14px;
    padding: 22px 28px;
    margin-bottom: 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 14px;
    box-shadow: 0 4px 18px rgba(20, 83, 45, 0.25);
}
.dash-header h1 {
    color: #fff;
    font-size: 28px;
    font-weight: 800;
    margin: 0 0 4px 0;
    letter-spacing: -0.02em;
}
.dash-header p {
    color: #DCFCE7;
    font-size: 14px;
    margin: 0;
}
.header-pills { display: flex; gap: 10px; flex-wrap: wrap; }
.pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 700;
    color: #fff;
}
.pill-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: rgba(255,255,255,0.9);
}
.pill-green { background: #22C55E; }
.pill-yellow { background: #EAB308; color: #422006; }
.pill-yellow .pill-dot { background: #422006; }
.pill-red { background: #EF4444; }

.dash-card {
    background: #fff;
    border-radius: 14px;
    padding: 18px 20px;
    border: 1px solid #E5E7EB;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
    margin-bottom: 16px;
}
.dash-card-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
}
.dash-card-head h3 {
    margin: 0;
    font-size: 17px;
    font-weight: 800;
    color: #111827;
}
.dash-card-head span {
    font-size: 12px;
    color: #6B7280;
    font-weight: 600;
}
.status-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    margin-bottom: 8px;
    background: #FAFAFA;
    font-size: 14px;
}
.status-row:last-child { margin-bottom: 0; }
.status-row .label { color: #374151; font-weight: 600; }
.status-row .value { font-weight: 800; }
.val-green { color: #16A34A; }
.val-blue { color: #2563EB; }
.val-red { color: #DC2626; }

.risk-block {
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 8px;
    border: 2px solid transparent;
}
.risk-block:last-child { margin-bottom: 0; }
.risk-block h4 { margin: 0 0 4px 0; font-size: 15px; font-weight: 800; }
.risk-block p { margin: 0; font-size: 12px; color: #4B5563; }
.risk-l1 { background: #DCFCE7; border-color: #86EFAC; }
.risk-l2 { background: #FEF9C3; border-color: #FDE047; }
.risk-l3 { background: #FEE2E2; border-color: #FCA5A5; }
.risk-active { box-shadow: 0 0 0 3px rgba(22, 163, 74, 0.35); }
.risk-active-l2 { box-shadow: 0 0 0 3px rgba(234, 179, 8, 0.45); }
.risk-active-l3 { box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.4); animation: blink 1s linear infinite; }
@keyframes blink { 50% { opacity: .65; } }

.upload-zone {
    border: 2px dashed #D1D5DB;
    border-radius: 12px;
    padding: 36px 20px;
    text-align: center;
    color: #6B7280;
    background: #F9FAFB;
    min-height: 200px;
}
.upload-zone .icon { font-size: 36px; margin-bottom: 8px; }
.upload-zone strong { display: block; color: #374151; font-size: 15px; margin-bottom: 4px; }

.scan-banner-safe {
    background: #DCFCE7;
    color: #166534;
    border: 2px solid #22C55E;
    padding: 14px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: 800;
    text-align: center;
    margin: 12px 0;
}
.scan-banner-danger {
    background: #FEE2E2;
    color: #991B1B;
    border: 2px solid #EF4444;
    padding: 14px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: 800;
    text-align: center;
    margin: 12px 0;
    animation: blink 1s linear infinite;
}

.incident-table-wrap {
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    overflow: hidden;
}
.incident-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}
.incident-table th {
    background: #F3F4F6;
    color: #374151;
    font-weight: 800;
    text-align: left;
    padding: 10px 14px;
    border-bottom: 1px solid #E5E7EB;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0.04em;
}
.incident-table td {
    padding: 10px 14px;
    border-bottom: 1px solid #F3F4F6;
    color: #111827;
}
.incident-table tr:last-child td { border-bottom: none; }
.incident-empty {
    padding: 28px;
    text-align: center;
    color: #9CA3AF;
    font-size: 14px;
}
.badge-safe { color: #16A34A; font-weight: 800; }
.badge-danger { color: #DC2626; font-weight: 800; }

.ppe-pill {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
}
.ppe-pill-green { background: #DCFCE7; color: #166534; }
.ppe-pill-yellow { background: #FEF9C3; color: #854D0E; }
.ppe-pill-red { background: #FEE2E2; color: #991B1B; }
.risk-0 { color: #16A34A; font-weight: 800; }
.risk-50 { color: #CA8A04; font-weight: 800; }
.risk-75, .risk-100 { color: #DC2626; font-weight: 800; }
.action-allowed { color: #16A34A; font-weight: 800; }
.action-denied { color: #DC2626; font-weight: 800; }

div[data-testid="stFileUploader"] section {
    border: 2px dashed #D1D5DB;
    border-radius: 12px;
    padding: 8px;
    background: #F9FAFB;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# Alarm
# ============================================================
def create_alarm():
    sample_rate = 44100
    frequency = 980
    duration = 0.28
    repeat = 6
    silence = 0.10
    audio = []

    for _ in range(repeat):
        for i in range(int(sample_rate * duration)):
            audio.append(0.65 * math.sin(2 * math.pi * frequency * i / sample_rate))
        for _ in range(int(sample_rate * silence)):
            audio.append(0)

    with wave.open(ALARM_PATH, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for sample in audio:
            wav.writeframes(struct.pack("<h", int(sample * 32767)))


def ensure_alarm():
    if not os.path.exists(ALARM_PATH):
        create_alarm()


def play_alarm_server():
    ensure_alarm()

    def _play():
        try:
            system = platform.system().lower()
            if system == "darwin":
                subprocess.Popen(["afplay", ALARM_PATH])
            elif system == "windows":
                import winsound
                winsound.PlaySound(ALARM_PATH, winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                print("\a")
        except Exception:
            print("\a")

    threading.Thread(target=_play, daemon=True).start()


def play_alarm_browser():
    ensure_alarm()
    with open(ALARM_PATH, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    components.html(
        f"""
        <audio autoplay>
            <source src="data:audio/wav;base64,{b64}" type="audio/wav">
        </audio>
        """,
        height=0,
    )


def alarm_once(key):
    if st.session_state.get("last_alarm_key") == key:
        return
    st.session_state["last_alarm_key"] = key
    play_alarm_server()
    play_alarm_browser()


# ============================================================
# Model
# ============================================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found. Please put best.pt at models/best.pt")
        st.stop()
    return YOLO(MODEL_PATH)


model = load_model()


# ============================================================
# Helpers
# ============================================================
def normalize_label(x):
    return str(x).lower().strip().replace("-", "_").replace(" ", "_")


def area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0


def nms(dets, thr=0.45):
    dets = sorted(dets, key=lambda d: d["confidence"], reverse=True)
    keep = []
    for d in dets:
        duplicate = False
        for k in keep:
            if d["class"] == k["class"] and iou(d["box"], k["box"]) > thr:
                duplicate = True
                break
        if not duplicate:
            keep.append(d)
    return keep


def crop_box(pil_img, box):
    w, h = pil_img.size
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array(pil_img.crop((x1, y1, x2, y2)).convert("RGB"))


def high_vis_score(rgb):
    if rgb is None or rgb.size == 0:
        return 0.0
    if rgb.shape[0] * rgb.shape[1] < 300:
        return 0.0

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    yellow_orange = (h >= 12) & (h <= 38) & (s >= 50) & (v >= 105)
    lime_green = (h >= 39) & (h <= 88) & (s >= 35) & (v >= 95)
    red_orange = (h <= 10) & (s >= 70) & (v >= 110)

    safety = yellow_orange | lime_green | red_orange
    useful = v >= 65

    if useful.sum() == 0:
        return 0.0
    return float(safety.sum() / useful.sum())


# Deployment note: all crop detections must return to the original image
# coordinate system before cross-crop deduplication can be meaningful.
def infer_one(pil_img, x0, y0, x1, y1, raw_conf, imgsz, augment):
    crop = pil_img.crop((x0, y0, x1, y1))
    result = model(crop, conf=raw_conf, imgsz=imgsz, augment=augment, verbose=False)[0]
    dets = []

    if result.boxes is None:
        return dets

    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = normalize_label(result.names[cls_id])
        conf = float(box.conf[0])
        bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().tolist()

        if label not in ["helmet", "vest", "no_helmet", "no_vest"]:
            continue

        dets.append({
            "class": label,
            "confidence": conf,
            "box": [bx1 + x0, by1 + y0, bx2 + x0, by2 + y0],
        })

    return dets


def raw_detect(pil_img, raw_conf, imgsz, augment, helmet_recall_mode, use_tiling):
    w, h = pil_img.size
    dets = []

    # Full image
    dets += infer_one(pil_img, 0, 0, w, h, raw_conf, imgsz, augment)

    # Helmet recall mode: run extra crops because helmets are usually small.
    if helmet_recall_mode:
        # top 70% crop
        dets += infer_one(pil_img, 0, 0, w, int(h * 0.75), raw_conf, max(imgsz, 1280), augment)

        # center crop
        dets += infer_one(
            pil_img,
            int(w * 0.10),
            int(h * 0.05),
            int(w * 0.90),
            int(h * 0.85),
            raw_conf,
            max(imgsz, 1280),
            augment,
        )

    # Optional 2x2 tiles
    if use_tiling and min(w, h) >= 500:
        xs = [0, w // 2, w]
        ys = [0, h // 2, h]
        for yi in range(2):
            for xi in range(2):
                dets += infer_one(
                    pil_img,
                    xs[xi],
                    ys[yi],
                    xs[xi + 1],
                    ys[yi + 1],
                    raw_conf,
                    max(imgsz, 1280),
                    augment,
                )

    return nms(dets, 0.45)


def filter_detections(pil_img, raw, helmet_th, vest_th, negative_th, use_negative, strict_vest, hi_vis_th):
    accepted = []
    rejected = []

    for d in raw:
        d = dict(d)
        label = d["class"]
        conf = d["confidence"]

        if label == "helmet":
            if conf >= helmet_th:
                d["reason"] = "accepted helmet"
                accepted.append(d)
            else:
                d["reason"] = "helmet below threshold"
                rejected.append(d)

        elif label == "vest":
            score = high_vis_score(crop_box(pil_img, d["box"]))
            d["high_vis_score"] = score

            if conf < vest_th:
                d["reason"] = "vest below threshold"
                rejected.append(d)
            elif strict_vest and score < hi_vis_th:
                d["reason"] = "rejected: not high-vis vest"
                rejected.append(d)
            else:
                d["reason"] = "accepted vest"
                accepted.append(d)

        elif use_negative and label in ["no_helmet", "no_vest"]:
            if conf >= negative_th:
                d["reason"] = "accepted negative evidence"
                accepted.append(d)
            else:
                d["reason"] = "negative below threshold"
                rejected.append(d)
        else:
            d["reason"] = "negative ignored"
            rejected.append(d)

    return accepted, rejected


def decide(accepted, use_negative):
    labels = [d["class"] for d in accepted]
    has_helmet = "helmet" in labels
    has_vest = "vest" in labels
    has_no_helmet = "no_helmet" in labels
    has_no_vest = "no_vest" in labels

    if use_negative:
        is_safe = has_helmet and has_vest and not has_no_helmet and not has_no_vest
    else:
        is_safe = has_helmet and has_vest

    if is_safe:
        return "SAFE", "Helmet and safety vest detected", 0, "Access Allowed"

    missing = []
    if not has_helmet or has_no_helmet:
        missing.append("helmet")
    if not has_vest or has_no_vest:
        missing.append("safety vest")

    return "DANGER", "Missing or invalid " + " and ".join(missing), 100, "Access Denied"


def draw(pil_img, dets):
    img = np.array(pil_img).copy()
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for d in dets:
        label = d["class"]
        conf = d["confidence"]
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        color = (0, 180, 0) if label in ["helmet", "vest"] else (0, 0, 230)

        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 3)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        top = max(0, y1 - th - 12)
        cv2.rectangle(bgr, (x1, top), (x1 + tw + 8, y1), color, -1)
        cv2.putText(bgr, text, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def df_dets(dets):
    rows = []
    for d in dets:
        rows.append({
            "class": d.get("class"),
            "confidence": round(d.get("confidence", 0), 3),
            "high_vis_score": round(d.get("high_vis_score", 0), 3) if "high_vis_score" in d else "",
            "reason": d.get("reason", ""),
            "x1": int(d["box"][0]),
            "y1": int(d["box"][1]),
            "x2": int(d["box"][2]),
            "y2": int(d["box"][3]),
        })
    return pd.DataFrame(rows)


def status_html(status, message):
    if status == "SAFE":
        return f"<div class='scan-banner-safe'>✅ SAFE: {message}</div>"
    return f"<div class='scan-banner-danger'>🚨 DANGER: {message}</div>"


def init_session_state():
    defaults = {
        "camera_online": False,
        "alarm_active": False,
        "workers_detected": 0,
        "compliant_count": 0,
        "violation_count": 0,
        "current_risk_level": 1,
        "gate_status": "Open",
        "last_scan_status": None,
        "last_scan_message": "",
        "upload_status": "Waiting for upload",
        "current_workers": [],
        "media_key": None,
        "last_annotated": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_scan_state():
    st.session_state["current_workers"] = []
    st.session_state["workers_detected"] = 0
    st.session_state["compliant_count"] = 0
    st.session_state["violation_count"] = 0
    st.session_state["current_risk_level"] = 1
    st.session_state["gate_status"] = "Open"
    st.session_state["alarm_active"] = False
    st.session_state["camera_online"] = False
    st.session_state["upload_status"] = "Waiting for upload"
    st.session_state["last_scan_status"] = None
    st.session_state["last_scan_message"] = ""
    st.session_state["last_annotated"] = None
    st.session_state["last_scan_sig"] = None
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)


def media_key_from_upload(uploaded, cam):
    if uploaded is not None:
        kind = "video" if uploaded.type and uploaded.type.startswith("video") else "image"
        return (kind, uploaded.name, uploaded.size)
    if cam is not None:
        return ("cam", cam.size, getattr(cam, "file_id", id(cam)))
    return None


def sync_media_clear(uploaded, cam):
    key = media_key_from_upload(uploaded, cam)
    prev = st.session_state.get("media_key")
    if prev is not None and key is None:
        reset_scan_state()
    elif prev is not None and key is not None and prev != key:
        reset_scan_state()
    st.session_state["media_key"] = key
    return key


def gate_status_from_level(level):
    if level == 1:
        return "Open"
    if level == 2:
        return "Locked"
    return "Locked"


# This groups PPE evidence by horizontal proximity; it is NOT a person
# detector or tracker. Crowds/background equipment can break this heuristic.
def cluster_detections_into_workers(dets, img_w, min_gap_ratio=0.16):
    if not dets:
        return []

    sorted_dets = sorted(dets, key=lambda d: (d["box"][0] + d["box"][2]) / 2)
    min_gap = max(img_w * min_gap_ratio, 60)
    clusters = [[sorted_dets[0]]]
    last_cx = (sorted_dets[0]["box"][0] + sorted_dets[0]["box"][2]) / 2

    for det in sorted_dets[1:]:
        cx = (det["box"][0] + det["box"][2]) / 2
        if cx - last_cx > min_gap:
            clusters.append([det])
        else:
            clusters[-1].append(det)
        last_cx = cx

    return clusters


def evaluate_worker_cluster(worker_num, cluster, use_negative):
    labels = [d["class"] for d in cluster]
    has_helmet = "helmet" in labels
    has_vest = "vest" in labels
    has_no_helmet = "no_helmet" in labels
    has_no_vest = "no_vest" in labels

    if use_negative and (has_no_helmet or has_no_vest):
        if has_no_helmet and has_no_vest:
            return _worker_row(worker_num, 3, "No helmet, no vest", 75, "Denied", "No helmet and no vest detected", "DANGER")
        if has_no_vest:
            return _worker_row(worker_num, 2, "No vest", 50, "Denied", "No vest detected", "DANGER")
        return _worker_row(worker_num, 3, "No helmet", 75, "Denied", "No helmet detected", "DANGER")

    if has_helmet and has_vest:
        return _worker_row(worker_num, 1, "Safe", 0, "Allowed", "Helmet and vest detected", "SAFE")

    if not has_helmet and not has_vest:
        return _worker_row(worker_num, 3, "No helmet, no vest", 75, "Denied", "No helmet and no vest detected", "DANGER")

    if has_helmet and not has_vest:
        return _worker_row(worker_num, 2, "No vest", 50, "Denied", "No vest detected", "DANGER")

    if not has_helmet and has_vest:
        return _worker_row(worker_num, 3, "No helmet", 75, "Denied", "No helmet detected", "DANGER")

    return _worker_row(worker_num, 3, "No helmet, no vest", 75, "Denied", "PPE incomplete", "DANGER")


def _worker_row(worker_num, level, ppe_label, risk, action, message, status):
    return {
        "id": f"Worker {worker_num}",
        "level": level,
        "ppe_label": ppe_label,
        "risk_score": risk,
        "action": action,
        "message": message,
        "status": status,
        "ppe_pill_class": {1: "ppe-pill-green", 2: "ppe-pill-yellow", 3: "ppe-pill-red"}[level],
    }


def build_workers_from_detections(pil_img, accepted, use_negative):
    w, _ = pil_img.size
    clusters = cluster_detections_into_workers(accepted, w)
    if not clusters:
        status, message, risk, action = decide(accepted, use_negative)
        level = 1 if status == "SAFE" else (2 if risk == 50 else 3)
        if status == "SAFE":
            return [_worker_row(1, 1, "Safe", 0, "Allowed", message, status)]
        if "vest" in message.lower() and "helmet" not in message.lower().split(" and ")[0]:
            ppe = "No vest"
            level, risk = 2, 50
        else:
            ppe = "No helmet, no vest"
            level, risk = 3, 75
        return [_worker_row(1, level, ppe, risk, "Denied" if status == "DANGER" else "Allowed", message, status)]

    return [evaluate_worker_cluster(i, c, use_negative) for i, c in enumerate(clusters, 1)]


def apply_workers_to_dashboard(workers, source_label):
    st.session_state["current_workers"] = workers
    st.session_state["workers_detected"] = len(workers)
    st.session_state["compliant_count"] = sum(1 for w in workers if w["level"] == 1)
    st.session_state["violation_count"] = sum(1 for w in workers if w["level"] >= 2)
    max_level = max((w["level"] for w in workers), default=1)
    st.session_state["current_risk_level"] = max_level
    st.session_state["gate_status"] = gate_status_from_level(max_level)
    st.session_state["alarm_active"] = st.session_state["violation_count"] > 0
    st.session_state["camera_online"] = True
    st.session_state["upload_status"] = f"Scan complete — {len(workers)} worker(s)"
    if workers:
        worst = max(workers, key=lambda w: w["level"])
        st.session_state["last_scan_status"] = worst["status"]
        st.session_state["last_scan_message"] = worst["action"]


def render_header(camera_online, alert_count, alarm_active):
    cam_cls = "pill-green" if camera_online else "pill-yellow"
    cam_txt = "Camera Online" if camera_online else "Camera Offline"
    alert_cls = "pill-yellow" if alert_count > 0 else "pill-green"
    alarm_cls = "pill-red" if alarm_active else "pill-green"
    alarm_txt = "Alarm: ON" if alarm_active else "Alarm: OFF"
    st.markdown(
        f"""
        <div class="dash-header">
            <div>
                <h1>PPE Safety Monitoring Dashboard</h1>
                <p>Gate A access control and live compliance overview</p>
            </div>
            <div class="header-pills">
                <span class="pill {cam_cls}"><span class="pill-dot"></span>{cam_txt}</span>
                <span class="pill {alert_cls}"><span class="pill-dot"></span>Alerts: {alert_count}</span>
                <span class="pill {alarm_cls}"><span class="pill-dot"></span>{alarm_txt}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_rows(rows):
    html = ""
    for label, value, color_class in rows:
        html += (
            f'<div class="status-row">'
            f'<span class="label">{label}</span>'
            f'<span class="value {color_class}">{value}</span>'
            f"</div>"
        )
    return html


def render_risk_response(active_level):
    levels = [
        (1, "risk-l1", "risk-active", "Level 1: Green", "Allow entry"),
        (2, "risk-l2", "risk-active-l2", "Level 2: Yellow", "Warn supervisor before entry"),
        (3, "risk-l3", "risk-active-l3", "Level 3: Red", "Deny entry and trigger alarm"),
    ]
    html = ""
    for lvl, base_cls, active_cls, title, subtitle in levels:
        cls = f"risk-block {base_cls}"
        if lvl == active_level:
            cls += f" {active_cls}"
        html += f'<div class="{cls}"><h4>{title}</h4><p>{subtitle}</p></div>'
    return html


def incident_log_html(workers=None):
    workers = workers if workers is not None else st.session_state.get("current_workers", [])
    if not workers:
        return '<div class="incident-empty">No incidents for current scan. Upload an image or video to begin.</div>'

    rows = ""
    for w in workers:
        risk_cls = f"risk-{w['risk_score']}"
        action_cls = "action-allowed" if w["action"] == "Allowed" else "action-denied"
        rows += (
            f"<tr>"
            f"<td><strong>{w['id']}</strong></td>"
            f'<td><span class="ppe-pill {w["ppe_pill_class"]}">{w["ppe_label"]}</span></td>'
            f'<td class="{risk_cls}">{w["risk_score"]}</td>'
            f'<td class="{action_cls}">{w["action"]}</td>'
            f"</tr>"
        )

    return (
        '<div class="incident-table-wrap">'
        '<table class="incident-table">'
        "<thead><tr><th>ID</th><th>PPE STATUS</th><th>RISK SCORE</th><th>ACTION</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


# Current-scan export: to_csv below replaces the file, rather than appending
# an audit history. The public README intentionally documents that boundary.
def save_scan_log(workers, source_label):
    if not workers:
        if os.path.exists(LOG_PATH):
            os.remove(LOG_PATH)
        return
    rows = []
    for w in workers:
        rows.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Input Type": source_label,
            "Worker ID": w["id"],
            "PPE Status": w["ppe_label"],
            "Risk Score": w["risk_score"],
            "Access Action": w["action"],
            "Message": w["message"],
        })
    pd.DataFrame(rows).to_csv(LOG_PATH, index=False)


def run_detection(pil_img, raw_conf, imgsz, augment, helmet_recall_mode, use_tiling, helmet_th, vest_th, negative_th, use_negative, strict_vest, hi_vis_th):
    raw = raw_detect(pil_img, raw_conf, imgsz, augment, helmet_recall_mode, use_tiling)
    accepted, rejected = filter_detections(pil_img, raw, helmet_th, vest_th, negative_th, use_negative, strict_vest, hi_vis_th)
    workers = build_workers_from_detections(pil_img, accepted, use_negative)
    status, message, risk, action = decide(accepted, use_negative)
    annotated = draw(pil_img, accepted)
    return raw, accepted, rejected, workers, status, message, risk, action, annotated


# ============================================================
# UI — PPE Safety Monitoring Dashboard
# ============================================================
init_session_state()

st.sidebar.title("Detection Settings")
raw_conf = st.sidebar.slider("Raw model confidence", 0.001, 0.50, 0.005, 0.001)
imgsz = st.sidebar.select_slider("Inference image size", [640, 768, 960, 1280], value=1280)
helmet_th = st.sidebar.slider("Helmet threshold", 0.001, 0.90, 0.03, 0.001)
vest_th = st.sidebar.slider("Vest threshold", 0.01, 0.95, 0.50, 0.01)
helmet_recall_mode = st.sidebar.toggle("Helmet recall mode", value=True)
use_tiling = st.sidebar.toggle("Use multi-crop detection", value=True)
augment = st.sidebar.toggle("Use YOLO augment inference", value=True)
strict_vest = st.sidebar.toggle("Reject non-high-vis vest", value=False)
hi_vis_th = st.sidebar.slider("High-vis vest score threshold", 0.00, 0.40, 0.04, 0.005)
use_negative = st.sidebar.toggle("Use no_helmet / no_vest in final decision", value=False)
negative_th = st.sidebar.slider("no_helmet / no_vest threshold", 0.01, 0.95, 0.80, 0.01)
alarm_enabled = st.sidebar.toggle("Automatic alarm enabled", value=True)
audio_alarm_enabled = st.sidebar.toggle("Audio alarm enabled", value=True)

st.sidebar.markdown("---")
if st.sidebar.button("Test Alarm"):
    play_alarm_server()
    play_alarm_browser()
if st.sidebar.button("Clear Incident Log"):
    reset_scan_state()
    st.session_state["media_key"] = None
    st.rerun()

render_header(
    st.session_state["camera_online"],
    st.session_state["violation_count"],
    st.session_state["alarm_active"],
)

left_col, right_col = st.columns([1.05, 0.95], gap="medium")

with left_col:
    st.markdown(
        f"""
        <div class="dash-card">
            <div class="dash-card-head">
                <h3>Live Cam</h3>
                <span>{st.session_state["upload_status"]}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    input_mode = st.radio(
        "Input source",
        ["Upload image / video", "Webcam snapshot"],
        horizontal=True,
        label_visibility="collapsed",
    )

    image = None
    uploaded = None
    cam = None
    uploaded_video = None
    source_label = ""

    if input_mode == "Upload image / video":
        uploaded = st.file_uploader(
            "Upload video or image",
            type=["jpg", "jpeg", "png", "mp4", "mov", "avi"],
            label_visibility="collapsed",
            key="main_uploader",
        )
        sync_media_clear(uploaded, None)
        if uploaded is not None:
            st.session_state["upload_status"] = "Processing upload..."
            if uploaded.type and uploaded.type.startswith("video"):
                uploaded_video = uploaded
                source_label = "Video"
            else:
                image = Image.open(uploaded).convert("RGB")
                source_label = "Image"
    else:
        cam = st.camera_input("Webcam snapshot", label_visibility="collapsed", key="main_camera")
        sync_media_clear(None, cam)
        if cam is not None:
            image = Image.open(cam).convert("RGB")
            st.session_state["upload_status"] = "Webcam snapshot captured"
            source_label = "Webcam"

    preview = st.empty()
    status_banner = st.empty()
    detail_expander = st.expander("Detection details", expanded=False)

    scan_sig = st.session_state.get("media_key")
    last_sig = st.session_state.get("last_scan_sig")

    if image is not None and scan_sig != last_sig:
        with st.spinner("Running PPE detection..."):
            raw, accepted, rejected, workers, status, message, risk, action, annotated = run_detection(
                image, raw_conf, imgsz, augment, helmet_recall_mode, use_tiling,
                helmet_th, vest_th, negative_th, use_negative, strict_vest, hi_vis_th,
            )

        apply_workers_to_dashboard(workers, source_label)
        st.session_state["last_annotated"] = annotated
        st.session_state["last_scan_sig"] = scan_sig
        save_scan_log(workers, source_label)

        if st.session_state["violation_count"] > 0 and alarm_enabled and audio_alarm_enabled:
            alarm_once(f"{source_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        with detail_expander:
            st.caption("Accepted PPE evidence")
            st.dataframe(df_dets(accepted), use_container_width=True)
            st.caption("Rejected / ignored")
            st.dataframe(df_dets(rejected) if rejected else pd.DataFrame(), use_container_width=True)

    elif uploaded_video is not None and scan_sig != last_sig:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp.write(uploaded_video.read())
        temp.close()

        cap = cv2.VideoCapture(temp.name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        out_path = os.path.join(RESULTS_DIR, f"ppe_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        progress = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = 0
        all_accepted = []
        last_annotated = None
        last_frame_workers = []

        with st.spinner("Processing video..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                raw, accepted, rejected, frame_workers, status, message, risk, action, annotated = run_detection(
                    pil_frame, raw_conf, imgsz, augment=False, helmet_recall_mode=False, use_tiling=False,
                    helmet_th=helmet_th, vest_th=vest_th, negative_th=negative_th,
                    use_negative=use_negative, strict_vest=strict_vest, hi_vis_th=hi_vis_th,
                )
                all_accepted.extend(accepted)
                last_annotated = annotated
                last_frame_workers = frame_workers
                writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                preview.image(annotated, use_container_width=True)
                idx += 1
                if total_frames > 0:
                    progress.progress(min(idx / total_frames, 1.0))

        cap.release()
        writer.release()

        workers = last_frame_workers

        apply_workers_to_dashboard(workers, "Video")
        st.session_state["last_annotated"] = last_annotated
        st.session_state["last_scan_sig"] = scan_sig
        save_scan_log(workers, "Video")

        if st.session_state["violation_count"] > 0 and alarm_enabled and audio_alarm_enabled:
            alarm_once(f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        with open(out_path, "rb") as f:
            st.download_button("Download processed video", f, file_name="ppe_detection_result.mp4", mime="video/mp4")

    workers = st.session_state.get("current_workers", [])
    annotated = st.session_state.get("last_annotated")

    if workers and annotated is not None:
        preview.image(annotated, use_container_width=True)
        worst = max(workers, key=lambda w: w["level"])
        status_banner.markdown(
            status_html(worst["status"], worst["message"]),
            unsafe_allow_html=True,
        )
    elif st.session_state.get("media_key") is None:
        preview.markdown(
            """
            <div class="upload-zone">
                <div class="icon">⬆</div>
                <strong>Upload video or image</strong>
                Supported source for PPE detection preview
            </div>
            """,
            unsafe_allow_html=True,
        )
        status_banner.empty()
    else:
        preview.info("Processing…")

    alarm_on = "ON" if st.session_state["alarm_active"] else "OFF"
    alarm_cls = "val-green" if not st.session_state["alarm_active"] else "val-red"
    audio_txt = "Enabled" if audio_alarm_enabled else "Disabled"
    audio_cls = "val-green" if audio_alarm_enabled else "val-red"
    gate_txt = st.session_state["gate_status"]
    gate_cls = "val-green" if gate_txt == "Open" else ("val-red" if gate_txt == "Closed" else "val-blue")

    st.markdown(
        f"""
        <div class="dash-card">
            <div class="dash-card-head"><h3>Alarm Status</h3></div>
            {render_metric_rows([
                ("Current alarm", alarm_on, alarm_cls),
                ("Audio alarm", audio_txt, audio_cls),
                ("Gate status", gate_txt, gate_cls),
            ])}
        </div>
        """,
        unsafe_allow_html=True,
    )

with right_col:
    workers = st.session_state["workers_detected"]
    compliant = st.session_state["compliant_count"]
    violations = st.session_state["violation_count"]
    risk_level = st.session_state["current_risk_level"]

    st.markdown(
        f"""
        <div class="dash-card">
            <div class="dash-card-head"><h3>Site Status</h3></div>
            {render_metric_rows([
                ("Workers detected", str(workers), "val-blue"),
                ("Compliant workers", str(compliant), "val-green"),
                ("Violations", str(violations), "val-red"),
            ])}
        </div>
        <div class="dash-card">
            <div class="dash-card-head"><h3>Risk Response</h3></div>
            {render_risk_response(risk_level)}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <div class="dash-card">
        <div class="dash-card-head">
            <h3>Incident Log</h3>
            <span>Access decisions for current scan</span>
        </div>
        {incident_log_html(st.session_state.get("current_workers", []))}
    </div>
    """,
    unsafe_allow_html=True,
)

if os.path.exists(LOG_PATH) and st.session_state.get("current_workers"):
    with open(LOG_PATH, "rb") as f:
        st.download_button("Download incident_log.csv", f, file_name="incident_log.csv", mime="text/csv")
