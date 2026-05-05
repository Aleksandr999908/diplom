#!/usr/bin/env python3
"""Скачивает RTMPose-m (ONNX) в models/rtmpose-m.onnx."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from urllib.request import urlretrieve

from exercise_recognition.config import (
    MODELS_DIR,
    RTMPOSE_DOWNLOAD_URL,
    RTMPOSE_ONNX,
    configure_stdio_utf8,
)


def main() -> None:
    configure_stdio_utf8()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if RTMPOSE_ONNX.is_file():
        print(f"Уже есть: {RTMPOSE_ONNX}")
        return
    print(f"Загрузка {RTMPOSE_DOWNLOAD_URL} …")
    urlretrieve(RTMPOSE_DOWNLOAD_URL, RTMPOSE_ONNX)
    print(f"Сохранено: {RTMPOSE_ONNX}")


if __name__ == "__main__":
    main()
