"""Датасет последовательностей скелетов для обучения Shift-GCN."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import (
    NUM_JOINTS,
    PROJECT_ROOT,
    SKELETON_NPY_DIM,
    SKELETON_RESAMPLE_MODE,
    SKELETON_SMOOTH_METHOD,
    T_FRAMES,
    exercise_class_names,
)
from .signal_filters import smooth_skeleton_sequence
from .skeleton import (
    build_shiftgcn_features,
    flip_skeleton_horizontal,
    resample_time,
    resample_time_adaptive_mean,
    sequence_to_tensor,
)


class SkeletonSequenceDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        augment: bool = False,
        t_frames: int = T_FRAMES,
        smooth_method: str | None = None,
        resample_mode: str | None = None,
        multi_task: bool = False,
    ):
        mp = Path(manifest_path).resolve()
        if mp.name == "manifest_errors.json":
            raise ValueError(
                "manifest_errors.json только для обучения типов ошибок "
                "(scripts/train_fault.py + FaultSkeletonDataset). "
                "Для класса упражнения используйте manifest.json."
            )
        with mp.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        self.augment = augment
        self.t_frames = t_frames
        self.smooth_method = smooth_method if smooth_method is not None else SKELETON_SMOOTH_METHOD
        self.resample_mode = resample_mode if resample_mode is not None else SKELETON_RESAMPLE_MODE
        self.idx_to_class = exercise_class_names()
        self.class_to_idx = {n: i for i, n in enumerate(self.idx_to_class)}
        self.items = [x for x in raw if x["label_name"] in self.class_to_idx]
        self.multi_task = multi_task

    def __len__(self) -> int:
        return len(self.items)

    def _resample(self, seq: np.ndarray) -> np.ndarray:
        if self.resample_mode == "adaptive_mean":
            return resample_time_adaptive_mean(seq, self.t_frames)
        return resample_time(seq, self.t_frames)

    def __getitem__(
        self, i: int
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        it = self.items[i]
        seq = np.load(PROJECT_ROOT / it["path"])
        if seq.ndim != 3 or seq.shape[1] != NUM_JOINTS or seq.shape[2] != SKELETON_NPY_DIM:
            raise ValueError(f"Bad shape in {it['path']}: {seq.shape}")

        if self.augment and seq.shape[0] > self.t_frames + 2:
            win_hi = min(seq.shape[0], max(self.t_frames * 2, int(seq.shape[0] * 0.85)))
            win = random.randint(self.t_frames + 1, win_hi)
            win = min(win, seq.shape[0])
            start = random.randint(0, seq.shape[0] - win)
            seq = seq[start : start + win].copy()

        seq = smooth_skeleton_sequence(seq, method=self.smooth_method)

        if self.augment and random.random() < 0.5:
            noise = np.random.normal(0, 0.01, seq.shape).astype(np.float32)
            seq = seq + noise
        if self.augment and random.random() < 0.42:
            seq = flip_skeleton_horizontal(seq)
        if self.augment and random.random() < 0.06:
            seq[:, :, 1] *= -1.0

        seq = self._resample(seq)

        if self.augment and random.random() < 0.22:
            sh = random.uniform(-0.11, 0.11)
            x0 = seq[:, :, 0].copy()
            y0 = seq[:, :, 1].copy()
            seq[:, :, 0] = x0 + sh * y0

        if self.augment and random.random() < 0.42:
            seq[:, :, :2] *= random.uniform(0.86, 1.16)

        if self.augment and random.random() < 0.30:
            th = random.uniform(-0.14, 0.14)
            c0, s0 = math.cos(th), math.sin(th)
            x0 = seq[:, :, 0].copy()
            y0 = seq[:, :, 1].copy()
            seq[:, :, 0] = x0 * c0 - y0 * s0
            seq[:, :, 1] = x0 * s0 + y0 * c0

        if self.augment and seq.shape[0] > self.t_frames + 4 and random.random() < 0.16:
            seq = np.ascontiguousarray(seq[::-1].copy())

        if self.augment and seq.shape[0] > 6 and random.random() < 0.18:
            n_drop = random.randint(1, 2)
            for _ in range(n_drop):
                j = random.randint(0, NUM_JOINTS - 1)
                seq[:, j, :] = 0.0

        if self.augment and seq.shape[0] > 14 and random.random() < 0.12:
            w = random.randint(2, min(6, seq.shape[0] // 3))
            s0 = random.randint(0, seq.shape[0] - w)
            seq[s0 : s0 + w] *= random.uniform(0.15, 0.55)

        full = build_shiftgcn_features(seq.astype(np.float32))
        x = sequence_to_tensor(full)
        y = self.class_to_idx[it["label_name"]]
        if not self.multi_task:
            return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

        from .stages.form_errors import detect_faults
        from .stages.phases import estimate_phase, phase_bucket_id

        ph, _ = estimate_phase(it["label_name"], seq)
        pid = phase_bucket_id(ph)
        last = seq[-1]
        errs = detect_faults(it["label_name"], ph, last[:, :2], last[:, 2])
        ef = 1 if errs else 0
        return (
            torch.from_numpy(x),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(pid, dtype=torch.long),
            torch.tensor(ef, dtype=torch.long),
        )
