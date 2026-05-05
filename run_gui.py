#!/usr/bin/env python3
"""Запуск десктопного приложения."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from exercise_recognition.config import configure_stdio_utf8
from exercise_recognition.gui_app import main

if __name__ == "__main__":
    configure_stdio_utf8()
    main()
