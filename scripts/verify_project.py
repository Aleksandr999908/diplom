#!/usr/bin/env python3
"""
Проверка окружения и согласованности проекта (зависимости, модели, manifest, веса).

Запуск из корня репозитория:
  python scripts/verify_project.py
  python scripts/verify_project.py --strict   # нет .pt или веса не C=16 как в train (боевой режим)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _print_status(ok: bool, msg: str) -> None:
    tag = "[OK]" if ok else "[!!]"
    print(f"{tag} {msg}")


def main() -> int:
    from exercise_recognition.config import (
        DATA_ERRORS,
        DATA_PROCESSED,
        DATA_RAW,
        FAULT_TRAINING_META,
        LGB_MANUAL_META,
        MANIFEST_ERRORS,
        MODEL_IN_CHANNELS,
        NUM_JOINTS,
        PROJECT_ROOT,
        RTMPOSE_ONNX,
        TRAINING_META,
        classifier_weights_path,
        configure_stdio_utf8,
        exercise_class_names,
        fault_classifier_weights_path,
        inference_class_names,
        inference_exercise_backend,
        lgb_manual_booster_path,
        manual_tcn_checkpoint_path,
        read_fault_training_meta,
        read_lgb_manual_meta,
        read_manual_tcn_meta,
        read_training_meta,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Ошибка, если нет .pt или каналы весов ≠ MODEL_IN_CHANNELS (как в обучении)",
    )
    args = ap.parse_args()

    configure_stdio_utf8()
    exit_code = 0
    strict = bool(args.strict)

    # Зависимости
    for mod in ("numpy", "cv2", "torch", "scipy", "PIL", "onnxruntime"):
        try:
            __import__(mod if mod != "PIL" else "PIL.Image")
            _print_status(True, f"импорт {mod}")
        except ImportError as e:
            _print_status(False, f"импорт {mod}: {e}")
            exit_code = 1

    try:
        import onnx  # noqa: F401

        _print_status(True, "импорт onnx (опционально, для export --verify)")
    except ImportError:
        _print_status(False, "onnx не установлен (pip install onnx) — export --verify недоступен")

    # RTMPose
    _print_status(RTMPOSE_ONNX.is_file(), f"RTMPose ONNX: {RTMPOSE_ONNX}")
    if not RTMPOSE_ONNX.is_file():
        exit_code = 1

    # Manifest
    manifest = DATA_PROCESSED / "manifest.json"
    if manifest.is_file():
        try:
            with manifest.open(encoding="utf-8") as f:
                items = json.load(f)
            n = len(items) if isinstance(items, list) else 0
            _print_status(n > 0, f"manifest.json: {n} записей")
            if n == 0:
                exit_code = 1
        except (OSError, json.JSONDecodeError) as e:
            _print_status(False, f"manifest.json: {e}")
            exit_code = 1
    else:
        _print_status(False, f"нет {manifest} — выполните scripts/extract_skeletons.py")
        exit_code = 1

    # Классы
    classes = exercise_class_names()
    _print_status(len(classes) > 0, f"классов упражнений: {len(classes)}")
    if not classes:
        exit_code = 1

    # training_meta.json
    meta = read_training_meta()
    if meta is None:
        _print_status(False, f"нет или битый {TRAINING_META}")
        exit_code = 1
    else:
        wrel = meta.get("weights")
        mic = meta.get("in_channels")
        ma = meta.get("model_arch", "?")
        eb = inference_exercise_backend()
        _print_status(
            True,
            f"training_meta: model_arch={ma!r}, in_channels(meta)={mic!r}, exercise_backend={eb!r}",
        )
        if eb == "lightgbm":
            lm = read_lgb_manual_meta()
            lgb_ok = lm is not None and lgb_manual_booster_path(lm) is not None
            _print_status(lgb_ok, f"LightGBM: {LGB_MANUAL_META.name} и файл модели")
            if strict and not lgb_ok:
                exit_code = 1
        elif eb == "tcn":
            tm = read_manual_tcn_meta()
            tcn_ok = tm is not None and manual_tcn_checkpoint_path(tm) is not None
            _print_status(tcn_ok, "manual TCN: meta и .pt")
            if strict and not tcn_ok:
                exit_code = 1
        else:
            if isinstance(wrel, str) and wrel.strip():
                wp = (PROJECT_ROOT / wrel.replace("\\", "/")).resolve()
                has = wp.is_file()
                _print_status(has, f"файл весов из meta: {wp.name} ({'есть' if has else 'нет — train.py'})")
                if not has and strict:
                    exit_code = 1
            else:
                _print_status(False, "в meta не указан weights")
                if strict:
                    exit_code = 1

        mcl = meta.get("classes")
        if isinstance(mcl, list) and mcl and DATA_RAW.is_dir():
            raw_sorted = sorted(
                p.name
                for p in DATA_RAW.iterdir()
                if p.is_dir() and not p.name.startswith(".")
            )
            meta_list = [str(x) for x in mcl if x]
            if raw_sorted != meta_list:
                if set(raw_sorted) != set(meta_list):
                    _print_status(
                        False,
                        "data/raw и training_meta.classes различаются по составу — "
                        "инференс всё равно по meta; для обучения синхронизируйте папки и train.py",
                    )
                else:
                    print(
                        "     → Порядок классов в data/raw ≠ training_meta; GUI берёт порядок из meta "
                        "(совпадает с .pt)."
                    )

    ic_names = inference_class_names()
    _print_status(
        len(ic_names) > 0,
        f"инференс: {len(ic_names)} классов ({'training_meta / .pt' if ic_names else 'нет'})",
    )

    # Фактический путь (с fallback)
    wpath = classifier_weights_path()
    has_w = wpath.is_file()
    _print_status(has_w, f"classifier_weights_path(): {wpath.name}")
    eb_chk = inference_exercise_backend()
    manual_ok = False
    if eb_chk == "lightgbm":
        lm2 = read_lgb_manual_meta()
        manual_ok = lm2 is not None and lgb_manual_booster_path(lm2) is not None
    elif eb_chk == "tcn":
        tm2 = read_manual_tcn_meta()
        manual_ok = tm2 is not None and manual_tcn_checkpoint_path(tm2) is not None
    elif eb_chk in ("manual_ensemble", "hybrid_ensemble"):
        lm2 = read_lgb_manual_meta()
        tm2 = read_manual_tcn_meta()
        manual_ok = (
            lm2 is not None
            and lgb_manual_booster_path(lm2) is not None
            and tm2 is not None
            and manual_tcn_checkpoint_path(tm2) is not None
        )
    if not has_w and strict and eb_chk == "torch":
        exit_code = 1
    if strict and eb_chk in ("lightgbm", "tcn", "manual_ensemble") and not manual_ok:
        exit_code = 1
    if strict and eb_chk == "hybrid_ensemble" and (not has_w or not manual_ok):
        exit_code = 1

    # Каналы в checkpoint vs код
    if has_w and eb_chk in ("torch", "hybrid_ensemble"):
        try:
            import torch

            try:
                ck = torch.load(wpath, map_location="cpu", weights_only=False)
            except TypeError:
                ck = torch.load(wpath, map_location="cpu")
            if isinstance(ck, dict):
                ckc = ck.get("classes")
                if isinstance(ckc, list) and ckc and ic_names:
                    ck_list = [str(x) for x in ckc if x]
                    if ck_list != ic_names:
                        _print_status(
                            False,
                            "список classes в checkpoint ≠ инференс (training_meta) — подставятся неверные подписи классов",
                        )
                        if strict:
                            exit_code = 1
            sd = ck.get("model", ck)
            if "data_bn.weight" in sd:
                ic_ck = int(sd["data_bn.weight"].shape[0]) // NUM_JOINTS
                ok_ch = ic_ck in (6, 16)
                _print_status(
                    ok_ch,
                    f"каналы в весах (GCN): C={ic_ck} (целевые для нового обучения: {MODEL_IN_CHANNELS})",
                )
                if not ok_ch:
                    exit_code = 1
                elif ic_ck != MODEL_IN_CHANNELS:
                    print(
                        "     → Веса C=6 (legacy motion); датасет/train — C=16. "
                        "Переобучите: python scripts/train.py"
                    )
                    if strict:
                        exit_code = 1
            elif "embed.weight" in sd:
                _print_status(
                    False,
                    "checkpoint старой неподдерживаемой модели больше не используется",
                )
                print("     → Переобучите Shift-GCN: python scripts/train.py")
                exit_code = 1
        except Exception as e:
            _print_status(False, f"чтение checkpoint: {e}")
            exit_code = 1

    # Опционально: классификатор типа ошибки (train_fault.py)
    fmeta = read_fault_training_meta()
    if fmeta is None:
        _print_status(
            True,
            f"fault_training_meta: нет {FAULT_TRAINING_META} (опционально: train_fault.py)",
        )
    else:
        fw = fault_classifier_weights_path()
        fc = fmeta.get("classes")
        nfc = len(fc) if isinstance(fc, list) else 0
        has_fw = fw is not None and fw.is_file()
        _print_status(
            has_fw and nfc > 0,
            f"классификатор ошибок: {fw.name if fw else '?'} "
            f"({nfc} классов fault) — {'OK' if has_fw else 'нет файла'}",
        )
        if has_fw and nfc > 0:
            try:
                import torch

                try:
                    fck = torch.load(fw, map_location="cpu", weights_only=False)
                except TypeError:
                    fck = torch.load(fw, map_location="cpu")
                fsd = fck.get("model", fck)
                if isinstance(fsd, dict) and "data_bn.weight" in fsd:
                    fic = int(fsd["data_bn.weight"].shape[0]) // NUM_JOINTS
                    ok_f = fic == MODEL_IN_CHANNELS
                    _print_status(
                        ok_f,
                        f"  fault C_in={fic} (ожидается {MODEL_IN_CHANNELS})",
                    )
            except Exception as e:
                _print_status(False, f"  fault checkpoint: {e}")
    if DATA_ERRORS.is_dir():
        _print_status(True, f"папка «Ошибки»: {DATA_ERRORS} → extract_errors.py")
    else:
        _print_status(True, f"нет {DATA_ERRORS} (опционально)")
    me = MANIFEST_ERRORS.is_file()
    _print_status(
        me,
        f"manifest_errors.json: {'есть — можно train_fault.py' if me else 'нет (опционально)'}",
    )

    if exit_code == 0:
        print("\nИтог: базовые проверки пройдены (без --strict отсутствие .pt допустимо).")
        print(
            "Проверка обучения без эпох: "
            "python scripts/train.py --one-batch --device cpu",
        )
    else:
        print("\nИтог: есть проблемы — см. [!!] выше.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
