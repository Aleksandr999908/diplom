#!/usr/bin/env python3
"""
Извлекает скелеты из data/raw/<класс>/*.mp4 с помощью RTMPose ONNX.
Результат: data/processed/skeletons/<класс>/<stem>.npy и manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exercise_recognition.config import (
    DATA_PROCESSED,
    DATA_RAW,
    RTMPOSE_ONNX,
    SKELETON_EXTRACT_MAX_FRAMES,
    configure_stdio_utf8,
)
from exercise_recognition.rtmpose_onnx import RTMPoseONNX
from exercise_recognition.skeleton import normalize_skeleton_sequence

VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".webm", ".mov"}


def _pad_pose(k: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(k, dtype=np.float32).reshape(-1, 2)
    sc = np.asarray(s, dtype=np.float32).reshape(-1)
    if xy.shape[0] < 17:
        xy = np.vstack([xy, np.zeros((17 - xy.shape[0], 2), dtype=np.float32)])
    if sc.shape[0] < 17:
        sc = np.pad(sc, (0, 17 - sc.shape[0]), constant_values=0.0)
    return xy[:17], sc[:17]


def iter_videos(raw_dir: Path) -> list[tuple[Path, str, str | None]]:
    out: list[tuple[Path, str, str | None]] = []
    for class_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        label = class_dir.name
        for vid in sorted(p for p in class_dir.iterdir() if p.is_file()):
            if vid.suffix.lower() in VIDEO_SUFFIXES:
                out.append((vid, label, None))
        for athlete_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
            athlete_id = athlete_dir.name
            for vid in sorted(p for p in athlete_dir.iterdir() if p.is_file()):
                if vid.suffix.lower() in VIDEO_SUFFIXES:
                    out.append((vid, label, athlete_id))
    return out


def extract_video(pose: RTMPoseONNX, path: Path, max_frames: int = 160) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return np.zeros((0, 17, 3), dtype=np.float32)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames * 2
    step = max(1, total // max_frames)
    keypoints: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            k, s = pose.infer(frame)
            xy, sc = _pad_pose(k, s)
            keypoints.append(xy)
            scores.append(sc)
        idx += 1
        if len(keypoints) >= max_frames:
            break
    cap.release()
    if not keypoints:
        return np.zeros((0, 17, 3), dtype=np.float32)
    return normalize_skeleton_sequence(
        np.stack(keypoints, axis=0), np.stack(scores, axis=0)
    )


def video_id_for_manifest(raw_dir: Path, vid: Path, label: str) -> str:
    try:
        return vid.relative_to(raw_dir / label).with_suffix("").as_posix()
    except ValueError:
        return vid.stem


def main() -> None:
    configure_stdio_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=SKELETON_EXTRACT_MAX_FRAMES)
    ap.add_argument("--limit-videos", type=int, default=0, help="0 = все")
    args = ap.parse_args()

    if not RTMPOSE_ONNX.is_file():
        print(f"Нет модели позы: {RTMPOSE_ONNX}. Запустите: python scripts/download_model.py")
        sys.exit(1)

    skel_root = DATA_PROCESSED / "skeletons"
    skel_root.mkdir(parents=True, exist_ok=True)

    pose = RTMPoseONNX(str(RTMPOSE_ONNX), device="cpu")
    pairs = iter_videos(DATA_RAW)
    if args.limit_videos:
        pairs = pairs[: args.limit_videos]

    manifest: list[dict] = []
    for vid, label, athlete_id in tqdm(pairs, desc="videos"):
        out_dir = skel_root / label if athlete_id is None else skel_root / label / athlete_id
        out_dir.mkdir(parents=True, exist_ok=True)
        arr = extract_video(pose, vid, max_frames=args.max_frames)
        if arr.shape[0] < 8:
            continue
        out_path = out_dir / f"{vid.stem}.npy"
        np.save(out_path, arr)
        rel = out_path.relative_to(ROOT).as_posix()
        item = {
            "path": rel,
            "label_name": label,
            "video_id": video_id_for_manifest(DATA_RAW, vid, label),
            "source_video": vid.relative_to(DATA_RAW).as_posix()
            if vid.is_relative_to(DATA_RAW)
            else vid.name,
        }
        if athlete_id is not None:
            item["athlete_id"] = athlete_id
            item["person_id"] = athlete_id
        manifest.append(item)

    man_path = DATA_PROCESSED / "manifest.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Записано сэмплов: {len(manifest)}, manifest: {man_path}")


if __name__ == "__main__":
    main()
