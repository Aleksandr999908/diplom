"""Пути и гиперпараметры проекта."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def configure_stdio_utf8() -> None:
    """Снижает UnicodeEncodeError и битый вывод argparse на Windows (cp1251)."""
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        reconf = getattr(stream, "reconfigure", None)
        if callable(reconf):
            try:
                reconf(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass

# Корень репозитория: .../дипломный проект
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_local_env() -> None:
    """Подхватывает PROJECT_ROOT/.env (KEY=value) без зависимости python-dotenv."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        key, _, val = s.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key:
            os.environ.setdefault(key, val)


# Загрузка .env при импорте пакета (run_gui и скрипты подхватят ключи локально)
load_local_env()

DATA_RAW = PROJECT_ROOT / "data" / "raw"
# Видео с демонстрацией типичных ошибок: data/Ошибки/<класс>/<категория>/*.mp4
DATA_ERRORS = PROJECT_ROOT / "data" / "Ошибки"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MANIFEST_MAIN = DATA_PROCESSED / "manifest.json"
MANIFEST_ERRORS = DATA_PROCESSED / "manifest_errors.json"
MODELS_DIR = PROJECT_ROOT / "models"
FAULT_TRAINING_META = MODELS_DIR / "fault_training_meta.json"
FAULT_GCN_WEIGHTS = MODELS_DIR / "fault_gcn_best.pt"
RTMPOSE_ONNX = MODELS_DIR / "rtmpose-m.onnx"
SHIFTGCN_WEIGHTS = MODELS_DIR / "shift_gcn_best.pt"
TRAINING_META = MODELS_DIR / "training_meta.json"
LGB_MANUAL_META = MODELS_DIR / "lgb_manual_meta.json"
MANUAL_TCN_META = MODELS_DIR / "manual_tcn_meta.json"
MANUAL_TCN_WEIGHTS = MODELS_DIR / "manual_tcn_best.pt"


def _meta_weights_path(rel: str) -> Path:
    """Путь к весам из JSON (и с «/», и с «\\» на Windows)."""
    return (PROJECT_ROOT / rel.replace("\\", "/")).resolve()


def read_fault_training_meta() -> dict | None:
    """Метаданные models/fault_training_meta.json (классификатор типа ошибки)."""
    if not FAULT_TRAINING_META.is_file():
        return None
    try:
        with FAULT_TRAINING_META.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def fault_classifier_weights_path() -> Path | None:
    """Путь к fault .pt из fault_training_meta или models/fault_gcn_best.pt."""
    meta = read_fault_training_meta()
    if meta is not None:
        rel = meta.get("weights")
        if isinstance(rel, str) and rel.strip():
            p = _meta_weights_path(rel.strip())
            if p.is_file():
                return p
    return FAULT_GCN_WEIGHTS if FAULT_GCN_WEIGHTS.is_file() else None


def read_training_meta() -> dict | None:
    """Содержимое models/training_meta.json или None."""
    if not TRAINING_META.is_file():
        return None
    try:
        with TRAINING_META.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def read_lgb_manual_meta() -> dict | None:
    """Метаданные LightGBM по ручным признакам (scripts/train_manual_lgb.py)."""
    if not LGB_MANUAL_META.is_file():
        return None
    try:
        with LGB_MANUAL_META.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def read_manual_tcn_meta() -> dict | None:
    """Метаданные 1D-CNN по углам (scripts/train_manual_tcn.py)."""
    if not MANUAL_TCN_META.is_file():
        return None
    try:
        with MANUAL_TCN_META.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def lgb_manual_booster_path(meta: dict) -> Path | None:
    rel = meta.get("weights")
    if not isinstance(rel, str) or not rel.strip():
        return None
    p = (PROJECT_ROOT / rel.replace("\\", "/")).resolve()
    return p if p.is_file() else None


def manual_tcn_checkpoint_path(meta: dict | None = None) -> Path | None:
    if meta:
        rel = meta.get("weights")
        if isinstance(rel, str) and rel.strip():
            p = (PROJECT_ROOT / rel.replace("\\", "/")).resolve()
            if p.is_file():
                return p
    return MANUAL_TCN_WEIGHTS if MANUAL_TCN_WEIGHTS.is_file() else None


def inference_exercise_backend() -> str:
    """
    Какой классификатор упражнения использовать:
    torch | lightgbm | tcn | manual_ensemble | hybrid_ensemble.
    Задаётся в models/training_meta.json ключом exercise_backend:
    lightgbm | lgb | tcn | manual_tcn | manual_ensemble | hybrid_ensemble
    (по умолчанию torch).
    """
    meta = read_training_meta()
    if meta is None:
        return "torch"
    eb = meta.get("exercise_backend")
    if not isinstance(eb, str):
        return "torch"
    e = eb.strip().lower().replace("-", "_")
    if e in ("lightgbm", "lgb", "lightgbm_manual"):
        return "lightgbm"
    if e in ("tcn", "tcn_manual", "manual_tcn", "1dcnn"):
        return "tcn"
    if e in ("manual_ensemble", "ensemble_manual", "lgb_tcn", "lightgbm_tcn"):
        return "manual_ensemble"
    if e in (
        "hybrid_ensemble",
        "hybrid",
        "gcn_lgb_tcn",
        "shiftgcn_lgb_tcn",
        "torch_manual_ensemble",
    ):
        return "hybrid_ensemble"
    return "torch"


def classifier_arch_from_meta(meta: dict | None) -> dict | None:
    """Архитектура классификатора: classifier_arch либо устаревшее gcn_arch."""
    if not meta:
        return None
    ca = meta.get("classifier_arch")
    if isinstance(ca, dict):
        return ca
    ga = meta.get("gcn_arch")
    return ga if isinstance(ga, dict) else None


def classifier_weights_path() -> Path:
    """Актуальный checkpoint из training_meta.json или shift_gcn_best.pt."""
    meta = read_training_meta()
    if meta is not None:
        rel = meta.get("weights")
        if isinstance(rel, str) and rel.strip():
            p = _meta_weights_path(rel)
            if p.is_file():
                return p
    return SHIFTGCN_WEIGHTS


def classifier_ensemble_extra_paths() -> list[Path]:
    """Дополнительные .pt для усреднения логитов (ключ training_meta ensemble_weights)."""
    meta = read_training_meta()
    if meta is None:
        return []
    raw = meta.get("ensemble_weights")
    if not isinstance(raw, list):
        return []
    primary = classifier_weights_path().resolve()
    out: list[Path] = []
    for rel in raw:
        if not isinstance(rel, str) or not rel.strip():
            continue
        p = _meta_weights_path(rel.strip()).resolve()
        if p.is_file() and p != primary:
            out.append(p)
    return out


SESSION_DB = PROJECT_ROOT / "data" / "sessions.db"

NUM_JOINTS = 17
T_FRAMES = 64
# Длительность непрерывного захвата в GUI (камера / файл), сек.
GUI_CAPTURE_MAX_SECONDS = 10.0
# Воспроизведение файла: 1.0 = по заявленному FPS; больше 1 — быстрее.
GUI_FILE_PLAYBACK_SPEED = 1.0
# Для стабильности классификации поза считается на каждом обработанном кадре файла.
GUI_FILE_POSE_EVERY_N_FRAMES = 1
# Сколько поз сохранять при субдискретизации ролика (как в scripts/extract_skeletons.py).
SKELETON_EXTRACT_MAX_FRAMES = 192
# В .npy: x, y, confidence; в модель: joint(6)+bone(6)+угловые скорости(4×тайл по V) = 16
SKELETON_NPY_DIM = 3
MODEL_IN_CHANNELS = 16
# Сглаживание координат перед признаками: none | oneuro | savgol | both
SKELETON_SMOOTH_METHOD = "oneuro"
# Кадры с очень плохим средним confidence интерполируются перед ручными фичами/TCN.
# Для текущего RTMPose mean frame confidence ~=0.42; 0.5 отбрасывает слишком много кадров.
MANUAL_FRAME_CONF_THRESHOLD = 0.1
# Ресемплинг времени: linear | adaptive_mean
SKELETON_RESAMPLE_MODE = "adaptive_mean"
SIMCC_SPLIT_RATIO = 2.0

# Препроцессинг как в MMPose RTMPose demo (W, H)
MODEL_INPUT_SIZE = (192, 256)

RTMPOSE_DOWNLOAD_URL = (
    "https://huggingface.co/bukuroo/RTMPose-ONNX/resolve/main/rtmpose-m.onnx"
)


def exercise_class_names() -> list[str]:
    """Имена классов: папки в data/raw; если пусто — список из models/training_meta.json (после обучения)."""
    names: list[str] = []
    if DATA_RAW.is_dir():
        names = sorted(
            p.name for p in DATA_RAW.iterdir() if p.is_dir() and not p.name.startswith(".")
        )
    if names:
        return names
    meta = read_training_meta()
    if meta is not None:
        c = meta.get("classes")
        if isinstance(c, list):
            return [str(x) for x in c if x]
    return []


def inference_class_names() -> list[str]:
    """
    Порядок меток для GUI/инференса — как при обучении (иначе argmax → неверное название).

    Раньше GUI брал только sorted(data/raw), что расходилось с training_meta / .pt после
    смены папок или отличий порядка. Сначала training_meta, затем поле classes в весах,
    и только потом папки raw.
    """
    meta = read_training_meta()
    if meta is not None:
        c = meta.get("classes")
        if isinstance(c, list) and c:
            return [str(x) for x in c if x]
    wp = classifier_weights_path()
    if wp.is_file():
        import torch

        try:
            ck = torch.load(str(wp), map_location="cpu", weights_only=False)
        except TypeError:
            ck = torch.load(str(wp), map_location="cpu")
        if isinstance(ck, dict):
            c = ck.get("classes")
            if isinstance(c, list) and c:
                return [str(x) for x in c if x]
    be = inference_exercise_backend()
    if be in ("lightgbm", "manual_ensemble", "hybrid_ensemble"):
        lm = read_lgb_manual_meta()
        if lm is not None:
            c = lm.get("classes")
            if isinstance(c, list) and c:
                return [str(x) for x in c if x]
    if be in ("tcn", "manual_ensemble", "hybrid_ensemble"):
        tm = read_manual_tcn_meta()
        if tm is not None:
            c = tm.get("classes")
            if isinstance(c, list) and c:
                return [str(x) for x in c if x]
    return exercise_class_names()
