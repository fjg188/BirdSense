
import argparse
import signal
import threading
from queue import Queue, Empty, Full
from pathlib import Path

import uuid
from datetime import datetime, timezone
import requests
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from norfair import Detection as NFDetection, Tracker
from picamera2 import Picamera2
from libcamera import controls
from cachetools import TTLCache

# ───── Paths & constants ─────
HOME = Path.home()
PROJECT_DIR = HOME / "birdcam"
WEIGHTS_DIR = HOME / ".cache" / "ultralytics" / "hub" / "weights"
DET_WEIGHT = WEIGHTS_DIR / "yolo11n.pt"
CLS_WEIGHT = PROJECT_DIR / "models" / "best_swin_cub.pth"
CLASSES_FILE = PROJECT_DIR / "classes.txt"
LABELS = [ln.strip().split(" ", 1)[1] for ln in open(CLASSES_FILE)] if CLASSES_FILE.exists() else [str(i) for i in range(200)]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

API_BASE   = os.getenv("BIRDCAM_API",   "https://yh5oyjgccj.execute-api.us-east-2.amazonaws.com/default/")
TIMEOUT    = 40 # seconds

# ───── Utility functions ─────

def build_camera(width: int, height: int, fps: int):
    cam = Picamera2()
    config = cam.create_video_configuration(main={"size": (width, height)})
    cam.configure(config)
    cam.set_controls({"AfMode": controls.AfModeEnum.Continuous})
    cam.start()
    return cam

def get_signed_url(fname: str, content_type="image/jpeg") -> dict:
    """POST → /v1/upload-url  → {'upload_url', 'object_url', 'key'}"""
    resp = requests.post(
        f"{API_BASE}/v1/upload-url",
        json={"filename": fname, "content_type": content_type},
        headers=HEADERS, timeout=TIMEOUT
    )
    resp.raise_for_status()
    return resp.json()

def upload_image(signed: dict, data: bytes, content_type="image/jpeg") -> None:
    resp = requests.put(
        signed["upload_url"], data=data,
        headers={"Content-Type": content_type}, timeout=TIMEOUT
    )
    resp.raise_for_status()

def post_observation(species: str, conf: float, obj_url: str, ts: str) -> None:
    payload = {
        "species":    species,
        "confidence": round(conf, 4),
        "timestamp":  ts,
        "image_url":  obj_url
    }
    resp = requests.post(f"{API_BASE}/v1/observations",
                         json=payload, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()


def detections_to_norfair(result, conf_threshold=0.25):
    boxes = result.boxes
    detections = []
    if boxes is None:
        return detections
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    mask = confs >= conf_threshold
    xyxy = xyxy[mask]
    confs = confs[mask]
    if xyxy.size == 0:
        return detections
    centers = (xyxy[:, :2] + xyxy[:, 2:]) / 2
    for (cx, cy), (x1, y1, x2, y2), c in zip(centers, xyxy, confs):
        detections.append(
            NFDetection(np.array([cx, cy]), scores=np.array([c]), data={"box": (x1, y1, x2, y2)})
        )
    return detections


@torch.inference_mode()
def classify_crop(bgr_crop):
    return None, 0.0


# ───── Main application ─────

def main(args):
    detector = YOLO(str(DET_WEIGHT))
    detector.fuse()

    cam = build_camera(args.width, args.height, args.fps)

    frame_q: Queue[np.ndarray] = Queue(maxsize=4)
    det_q: Queue[tuple[np.ndarray, object]] = Queue(maxsize=4)

    running = threading.Event()
    running.set()

    # Thread 1: Capture
    def capture_loop():
        while running.is_set():
            frame = cam.capture_array()
            try:
                frame_q.put(frame, timeout=0.01)
            except Full:
                continue

    # Thread 2: Detect
    def detect_loop():
        while running.is_set():
            try:
                frame = frame_q.get(timeout=0.1)
            except Empty:
                continue
            res = detector.predict(
                frame,
                device=0 if DEVICE.type == "cuda" else "cpu",
                imgsz=args.imgsz,
                conf=args.conf,
                verbose=False,
            )
            try:
                det_q.put((frame, res[0]), timeout=0.01)
            except Full:
                continue

    cap_thread = threading.Thread(target=capture_loop, daemon=True)
    det_thread = threading.Thread(target=detect_loop, daemon=True)
    cap_thread.start()
    det_thread.start()

    tracker = Tracker(distance_function=lambda a, b: np.linalg.norm(a.points - b.points), distance_threshold=30)
    seen = TTLCache(maxsize=512, ttl=30)

    def handle_exit(sig, _):
        running.clear()

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    # Thread 3: Classify + display (main)
    while running.is_set():
        try:
            frame, det_result = det_q.get(timeout=0.1)
        except Empty:
            continue

        detections = detections_to_norfair(det_result, args.conf)
        tracked = tracker.update(detections)

        for trk in tracked:
            x1, y1, x2, y2 = trk.last_detection.data["box"]
            tid = trk.id

            if tid not in seen:
                crop = frame[int(y1): int(y2), int(x1): int(x2)]
                if crop.size:
                    cls_idx, cls_conf = classify_crop(crop)
                    seen[tid] = (cls_idx, cls_conf)

            cls_idx, cls_conf = seen.get(tid, (None, 0.0))
            label = LABELS[cls_idx] if cls_idx is not None else f"ID {tid}"
            txt = f"{label} {cls_conf:.2f}" if cls_idx is not None else label

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, txt, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("BirdCam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            running.clear()

    cap_thread.join()
    det_thread.join()
    cam.stop()
    cam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Three‑stage threaded birdcam pipeline")
    ap.add_argument("--width", type=int, default=1280, help="Frame width")
    ap.add_argument("--height", type=int, default=720, help="Frame height")
    ap.add_argument("--fps", type=int, default=30, help="Camera FPS")
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    args = ap.parse_args()
    main(args)
