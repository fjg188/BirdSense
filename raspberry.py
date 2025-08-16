import argparse
import os
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
import re
import requests
from picamera2 import Picamera2
from ultralytics import YOLO
from libcamera import controls
from norfair import Detection, Paths, Tracker, Video

HOME = Path.home()
PROJECT_DIR = HOME / "birdcam"
WEIGHTS_DIR = HOME / ".cache" / "ultralytics" / "hub" / "weights"

DET_WEIGHT = WEIGHTS_DIR / "yolo11n.pt"
CLF_WEIGHT = PROJECT_DIR / "models" / "best_swin_cub.pth"
CLASSES_FILE = PROJECT_DIR / "classes.txt"
BACKBONE_NAME = "swin_base_patch4_window12_384"

#TODO make these "enviromental variabels" so they are hidden from github 
UPLOAD_URL_ENDPOINT = "REDACTED" 
SAVE_META_ENDPOINT   = "REDACTED"
TIMEOUT = 15

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LABELS = [ln.strip().split(" ", 1)[1] for ln in open(CLASSES_FILE)] if CLASSES_FILE.exists() else [str(i) for i in range(200)]

# Detection model

detector = YOLO(str(DET_WEIGHT))
detector.fuse()

# Classification model (Swin + BS head)

class BSHead(nn.Module):
    def __init__(self, in_ch: int, n_cls: int, k: int = 64):
        super().__init__()
        self.cls = nn.Conv2d(in_ch, n_cls, 1)
        self.k = k

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        logits_map = self.cls(feat)
        logits = logits_map.flatten(2).topk(self.k, dim=2).values.mean(-1)
        return logits


class SwinBS(nn.Module):
    def __init__(self, num_classes: int = 200):
        super().__init__()
        self.backbone = timm.create_model(BACKBONE_NAME, pretrained=False, num_classes=0)
        self.in_ch = self.backbone.num_features
        self.bs = BSHead(self.in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.forward_features(x)
        if feat.dim() == 3:
            b, l, c = feat.shape
            s = int(l ** 0.5)
            feat = feat.view(b, s, s, c).permute(0, 3, 1, 2).contiguous()
        elif feat.dim() == 4:
            if feat.shape[1] != self.in_ch and feat.shape[-1] == self.in_ch:
                feat = feat.permute(0, 3, 1, 2).contiguous()
        else:
            raise RuntimeError(f"Unexpected feature shape: {feat.shape}")
        return self.bs(feat)


classifier = SwinBS(len(LABELS)).to(DEVICE)
state = torch.load(str(CLF_WEIGHT), map_location=DEVICE)
classifier.load_state_dict(state, strict=True)
classifier.eval()

# Pre‑processing

INPUT_SIZE = 384
mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


@torch.no_grad()
def preprocess(img: np.ndarray) -> torch.Tensor:
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    # img is RGB here
    x = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
    x = (x - mean) / std
    return x


@torch.no_grad()
def classify(img_rgb: np.ndarray):
    logits = classifier(preprocess(img_rgb))
    prob = torch.softmax(logits, dim=1)
    conf, pred = prob.max(dim=1)
    return pred.item(), conf.item()


def detections_to_norfair(res) -> List[Detection]:

    if isinstance(res, list):
        res = res[0]

    if res.boxes is None or len(res.boxes) == 0:
        return []

    xywh = res.boxes.xywh
    conf = res.boxes.conf
    cls = res.boxes.cls.int()

    dets: List[Detection] = []
    for (xc, yc, w, h), p, c in zip(xywh, conf, cls):
        if c != 14:  # not a bird skip
            continue

        x1 = (xc - w / 2).item();
        y1 = (yc - h / 2).item()
        x2 = (xc + w / 2).item();
        y2 = (yc + h / 2).item()

        points = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
        scores = np.array([p.item(), p.item()], dtype=np.float32)

        dets.append(Detection(points=points, scores=scores, label=14))

    return dets

def _get_upload_url():
    r = requests.get(UPLOAD_URL_ENDPOINT, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data["uploadURL"], data["Key"]

def _put_image(upload_url, img_bgr):
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    headers = {"Content-Type": "image/jpeg"}
    r = requests.put(upload_url, data=buf.tobytes(), headers=headers, timeout=TIMEOUT)
    r.raise_for_status()

def _save_meta(species, conf, key):
    payload = {
        "species": species,
        "confidence": float(round(conf, 4)),
        "image": key
    }
    r = requests.post(SAVE_META_ENDPOINT, json=payload, timeout=TIMEOUT)
    r.raise_for_status()

def upload_classification(crop_bgr, species, confidence):
    upload_url, key = _get_upload_url()
    _put_image(upload_url, crop_bgr)
    _save_meta(species, confidence, key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--fps", type=float, default=8)
    parser.add_argument("--conf", type=float, default=0.1)
    args = parser.parse_args()

    preview = args.preview and bool(os.environ.get("DISPLAY"))

    cam = Picamera2()
    cfg = cam.create_video_configuration(main={"format": "BGR888", "size": (1280, 720)})
    cam.configure(cfg)
    cam.start()
    try:
        cam.set_controls({
            "AfMode": controls.AfModeEnum.Continuous,
            "AfSpeed": controls.AfSpeedEnum.Fast,
        })

        cam.set_controls({
            "AeEnable": True,
            "AeMeteringMode": controls.AeMeteringModeEnum.CentreWeighted,
        })

    except Exception:
        pass

    if preview:
        cv2.namedWindow("BirdCam", cv2.WINDOW_NORMAL)

    interval = 1.0 / args.fps

    tracker = Tracker(
        distance_function='iou',
        distance_threshold= float(.7)
    )
    seen = set()

    try:
        while True:
            tic = time.perf_counter()

            frame_bgr = cam.capture_array()
            if frame_bgr is None:
                raise RuntimeError("Camera frame grab failed")

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            results = detector.predict(frame_rgb, device=str(DEVICE), conf=args.conf, verbose=False)
            centroids = detections_to_norfair(results[0])

            tracked_objects = tracker.update(detections=centroids)


            for object in tracked_objects:
                if object.id in seen:
                    continue

                seen.add(object.id)

                (x1, y1), (x2, y2) = object.estimate.astype(int)

                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, frame_rgb.shape[1]), min(y2, frame_rgb.shape[0])
                crop_bgr = frame_bgr[y1:y2, x1:x2]
                crop_rgb = frame_rgb[y1:y2, x1:x2]

                if crop_bgr.size == 0:
                    continue
                idx, score = classify(crop_bgr)
                if score > 0.3:
                    label = LABELS[idx] if idx < len(LABELS) else str(idx)
                    print(f"species: {label}  confidence: {score:.2f}")
                    cv2.putText(frame_rgb, f"{label}:{score:.2f}", (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    try:
                        name = re.sub(r'^\d+\s*[\.\s]?\s*', '', label.strip())
                        name = name.replace('_', ' ')

                        upload_classification(crop_rgb, name, round(score * 100))
                    except Exception as e:
                        print(e)
                        print('upload failed')

            if preview:
                cv2.imshow("BirdCam", frame_rgb)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            elapsed = time.perf_counter() - tic
            if elapsed < interval:
                time.sleep(interval - elapsed)
    finally:
        cam.stop()
        if preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
