#!/usr/bin/env python3
"""
Экспорт классификатора скелета Shift-GCN в ONNX.

Вход: (N, C, T, V, M), как в train.py / pipeline (M=1).
По умолчанию выход — только логиты класса упражнения (удобно для TensorRT / onnxruntime).

Пример:
  python scripts/export_classifier_onnx.py --checkpoint models/shift_gcn_best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exercise_recognition.classifier_loader import build_classifier
from exercise_recognition.config import (
    MODELS_DIR,
    NUM_JOINTS,
    T_FRAMES,
    TRAINING_META,
    classifier_weights_path,
    configure_stdio_utf8,
    inference_class_names,
)


class LogitsOnlyWrapper(nn.Module):
    """Один тензор логитов класса (для совместимости с большинством рантаймов)."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.inner(x)
        return o[0] if isinstance(o, tuple) else o


class MultiTaskOnnxWrapper(nn.Module):
    """Три выхода: logits, phase_logits, err_logits (только если модель multi_task)."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        o = self.inner(x)
        if isinstance(o, tuple) and len(o) == 3:
            return o[0], o[1], o[2]
        raise RuntimeError("Модель без multi_task: используйте экспорт без --multi-outputs")


def main() -> None:
    configure_stdio_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Путь к .pt (по умолчанию — из training_meta или classifier_weights_path())",
    )
    ap.add_argument(
        "--output",
        type=str,
        default="",
        help="Куда сохранить .onnx (по умолчанию models/<stem>.onnx рядом с .pt)",
    )
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument(
        "--multi-outputs",
        action="store_true",
        help="Три выхода (класс + фаза + ошибка); только для multi_task модели",
    )
    ap.add_argument(
        "--update-meta",
        action="store_true",
        help="Дописать поле onnx в models/training_meta.json",
    )
    ap.add_argument("--verify", action="store_true", help="Проверить ONNX и один прогон onnxruntime")
    args = ap.parse_args()

    ck_path = Path(args.checkpoint) if args.checkpoint else classifier_weights_path()
    if not ck_path.is_file():
        print(f"Нет checkpoint: {ck_path}", flush=True)
        sys.exit(1)

    classes = inference_class_names()
    if not classes:
        print("Пустой список классов: добавьте папки в data/raw или обучите модель.", flush=True)
        sys.exit(1)

    try:
        state = torch.load(ck_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(ck_path, map_location="cpu")

    device = torch.device("cpu")
    model, ic = build_classifier(state, classes, device)
    model.eval()

    multi = bool(getattr(model, "multi_task", False))
    if args.multi_outputs:
        if not multi:
            print(
                "Модель не multi_task — экспорт с тремя выходами невозможен. "
                "Обучите с --multi-task.",
                flush=True,
            )
            sys.exit(1)
        wrapped: nn.Module = MultiTaskOnnxWrapper(model)
        out_names = ["logits", "phase_logits", "error_logits"]
    else:
        wrapped = LogitsOnlyWrapper(model)
        out_names = ["logits"]

    t, v = T_FRAMES, NUM_JOINTS
    dummy = torch.randn(1, ic, t, v, 1, dtype=torch.float32, device=device)

    out_onnx = Path(args.output) if args.output else ck_path.with_suffix(".onnx")
    out_onnx.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapped,
        dummy,
        str(out_onnx),
        input_names=["skeleton"],
        output_names=out_names,
        opset_version=args.opset,
        dynamo=False,
        do_constant_folding=True,
    )
    print(f"Сохранено: {out_onnx}  (C={ic}, T={t}, V={v}, outputs={out_names})", flush=True)

    if args.update_meta and TRAINING_META.is_file():
        try:
            with TRAINING_META.open(encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            meta = {}
        rel = out_onnx.relative_to(ROOT).as_posix()
        meta["onnx"] = rel
        meta["onnx_outputs"] = out_names
        with TRAINING_META.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"Обновлён {TRAINING_META}", flush=True)

    if args.verify:
        try:
            import onnx
            from onnx import checker

            m = onnx.load(str(out_onnx))
            checker.check_model(m)
            print("onnx.checker: OK", flush=True)
        except Exception as e:
            print(f"onnx проверка: {e}", flush=True)
            sys.exit(1)

        try:
            import numpy as np
            import onnxruntime as ort

            sess = ort.InferenceSession(
                str(out_onnx), providers=["CPUExecutionProvider"]
            )
            inp = {sess.get_inputs()[0].name: dummy.numpy()}
            outs = sess.run(None, inp)
            for i, name in enumerate(out_names):
                arr = outs[i]
                print(f"  {name}: shape={arr.shape} dtype={arr.dtype}", flush=True)
            print("onnxruntime: OK", flush=True)
        except Exception as e:
            print(f"onnxruntime: {e}", flush=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
