"""
Raspberry Pi BirdCam — Headless Core
"""

from __future__ import annotations

import argparse
import io
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import requests
import timm
import torch
from PIL import Image
from picamera2 import Picamera2
from torchvision import transforms
from ultralytics import YOLO
from libcamera import controls

# Project paths and constants
HOME = Path.home()
PROJECT_DIR = HOME / "birdcam"
MODEL_PATH = PROJECT_DIR / "models" / "best_tinyvit_cub.pth"
CLASSES_FILE = PROJECT_DIR / "classes.txt"
WEIGHTS_DIR = HOME / ".cache/ultralytics/hub/weights"
DET_WEIGHT = WEIGHTS_DIR / "yolo11n.pt"
BACKBONE_NAME = "tiny_vit_21m_384"
IMG_SIZE = 384
FPS_DEFAULT = 8
MIN_DET_CONF_DEFAULT = 0.1  #adjust if needed
BACKEND_IMAGE_URL = "https://api.example.com/bird-image"
BACKEND_META_URL = "https://api.example.com/bird-meta"
API_KEY = "YOUR_API_KEY_HERE"

# Check required files
for path, desc in [(CLASSES_FILE, "classes file"), (DET_WEIGHT, "YOLO weights"), (MODEL_PATH, "classifier weights")]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {desc} at {path}")

LABELS = [ln.strip().split(' ',1)[1] for ln in open(CLASSES_FILE)]

# Initialize models
detector = YOLO(str(DET_WEIGHT))
detector.fuse()
state_dict = torch.load(MODEL_PATH, map_location="cpu")
classifier = timm.create_model(BACKBONE_NAME, pretrained=False, num_classes=len(LABELS))
classifier.load_state_dict(state_dict, strict=False)
classifier.eval()

# Preprocessing for classifier
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def detect(frame: Image.Image, min_conf: float) -> List[Tuple[List[int], float]]:
    # YOLO detection, filter for class 14 (bird)
    results = detector(frame, imgsz=480, conf=min_conf, verbose=False)
    detections: List[Tuple[List[int], float]] = []
    for res in results:
        for box, conf, cls in zip(res.boxes.xyxy, res.boxes.conf, res.boxes.cls):
            if int(cls) == 14:
                detections.append((list(map(int, box)), float(conf)))
    return detections


def classify(img: Image.Image) -> Tuple[str, float]:
    with torch.no_grad():
        probs = torch.softmax(classifier(preprocess(img).unsqueeze(0)), dim=1)
        conf, idx = torch.max(probs, dim=1)
    return LABELS[idx.item()], float(conf.item())


def upload_image(img: Image.Image) -> str:
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=90); buf.seek(0)
    r = requests.post(
        BACKEND_IMAGE_URL,
        files={"image": ("bird.jpg", buf, "image/jpeg")},
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    r.raise_for_status()
    return r.json().get("image_id", "")


def upload_meta(species: str, cls_conf: float, det_conf: float, img_id: str):
    r = requests.post(
        BACKEND_META_URL,
        json={
            "species": species,
            "classifier_conf": cls_conf,
            "detector_conf": det_conf,
            "image_id": img_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    r.raise_for_status()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--fps", type=float, default=FPS_DEFAULT)
    parser.add_argument("--conf", type=float, default=MIN_DET_CONF_DEFAULT)
    args = parser.parse_args()

    debug_preview = args.preview and bool(os.environ.get("DISPLAY"))
    print(f"DISPLAY={os.environ.get('DISPLAY')} {'(preview)' if debug_preview else '(headless)'}", flush=True)

    # Initialize camera
    picam = Picamera2()
    cfg = picam.create_video_configuration(main={"format": "BGR888", "size": (1280, 720)})
    picam.configure(cfg)
    picam.start()

    # Trigger autofocus
    #TODO fix autofocus
    try:
        picam.set_controls({
            "AfMode": controls.AfModeEnum.Continuous,
            "AfTrigger": controls.AfTrigger.Start
        })
        print("Autofocus: continuous mode enabled.")
    except Exception:
        print("Warning: autofocus not supported.")

    if debug_preview:
        cv2.namedWindow("BirdCam", cv2.WINDOW_NORMAL)

    interval = 1.0 / args.fps
    print("System ready — entering main loop…", flush=True)

    try:
        while True:
            tic = time.perf_counter()

            frame_bgr = picam.capture_array()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)

            # Detect birds
            dets = detect(frame_pil, args.conf)
            if not dets:
                print("-- no detections --", flush=True)
            for (x1, y1, x2, y2), d_conf in dets:
                # Classify crop
                crop = frame_pil.crop((x1, y1, x2, y2))
                crop.show()
                species, c_conf = classify(crop)
                print(f" {species:20s} cls={c_conf:.2f} det={d_conf:.2f}", flush=True)
                # Draw bounding box
                if debug_preview:
                    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # Show preview
            if debug_preview:
                cv2.imshow("BirdCam", frame_rgb)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            # Throttle loop
            elapsed = time.perf_counter() - tic
            time.sleep(max(0.0, interval - elapsed))

    finally:
        picam.stop()
        if debug_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
