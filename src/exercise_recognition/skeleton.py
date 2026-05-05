"""Нормализация скелета COCO-17 и утилиты для последовательностей."""

from __future__ import annotations

import numpy as np
import torch

# Индексы COCO (как в MMPose body)
L_SH, R_SH = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANK, R_ANK = 15, 16
L_ELB, R_ELB = 7, 8
L_WR, R_WR = 9, 10
NOSE = 0

# Дерево костей COCO-17: parent[j] — индекс родителя, −1 у корня (левое бедро).
COCO_PARENT: tuple[int, ...] = (
    5,  # 0 nose — опора на плечо (упрощённо, для связного дерева)
    0,
    0,
    1,
    2,
    11,
    12,
    5,
    6,
    7,
    8,
    -1,
    11,
    11,
    12,
    13,
    14,
)

# Пары левый/правый для зеркалирования по вертикали (COCO-17 body).
COCO_LR_SWAP_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
)


def flip_skeleton_horizontal(seq: np.ndarray) -> np.ndarray:
    """
    Горизонтальное зеркало в нормализованных координатах: −x и обмен L/R суставов.
    seq: (T, V, C). Для C>=6: инверсия x и vx у joint-блока (0..5) и bone-блока (6..11);
    каналы 12–15 (угловые скорости) меняются местами по смыслу L/R: 0=L локоть, 1=R, 2=L колено, 3=R.
    """
    s = np.asarray(seq, dtype=np.float32).copy()
    s[..., 0] *= -1.0
    c = s.shape[-1]
    if c >= 6:
        s[..., 3] *= -1.0
    if c >= 12:
        s[..., 6] *= -1.0
        s[..., 9] *= -1.0
    if c >= 16:
        s[..., [12, 13, 14, 15]] = s[..., [13, 12, 15, 14]]
    for i, j in COCO_LR_SWAP_PAIRS:
        s[:, [i, j], :] = s[:, [j, i], :]
    return s


def angle_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Угол ABC в градусах."""
    ba = a - b
    bc = c - b
    n1 = np.linalg.norm(ba) * np.linalg.norm(bc)
    if n1 < 1e-8:
        return float("nan")
    cos = np.clip(np.dot(ba, bc) / n1, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def _xy_conf_17(xy: np.ndarray, scores: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    xy_arr = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    if xy_arr.shape[0] < 17:
        pad = np.zeros((17 - xy_arr.shape[0], 2), dtype=np.float32)
        xy_arr = np.vstack([xy_arr, pad])
    xy_arr = xy_arr[:17]
    if scores is None:
        conf = np.ones((17,), dtype=np.float32)
    else:
        conf = np.asarray(scores, dtype=np.float32).reshape(-1)
        if conf.shape[0] < 17:
            conf = np.pad(conf, (0, 17 - conf.shape[0]), constant_values=0.0)
        conf = conf[:17]
    return xy_arr, conf


def _robust_positive(values: np.ndarray, fallback: float = 1.0) -> float:
    vals = np.asarray(values, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals) & (vals > 1e-6)]
    if vals.size == 0:
        return float(fallback)
    return float(np.median(vals))


def _normalize_xy_with_reference(
    xy: np.ndarray,
    conf: np.ndarray,
    *,
    scale: float,
    shoulder_phi: float,
    center_xy: np.ndarray | None = None,
    force_right_positive: bool = True,
) -> np.ndarray:
    if center_xy is None:
        center_xy = (xy[L_HIP] + xy[R_HIP]) * 0.5
    centered_xy = xy - np.asarray(center_xy, dtype=np.float32).reshape(2)
    cp, sp = float(np.cos(shoulder_phi)), float(np.sin(shoulder_phi))
    x0 = centered_xy[:, 0]
    y0 = centered_xy[:, 1]
    xr = (cp * x0 + sp * y0) / max(float(scale), 1e-6)
    yr = (-sp * x0 + cp * y0) / max(float(scale), 1e-6)
    centered = np.stack([xr, yr], axis=-1)
    if force_right_positive and centered[R_SH, 0] < centered[L_SH, 0]:
        centered[:, 0] *= -1.0
    return np.concatenate([centered, conf[:, None]], axis=-1).astype(np.float32)


def normalize_frame(xy: np.ndarray, scores: np.ndarray | None = None) -> np.ndarray:
    """
    Нормализация одного кадра: центр mid-hip, scale по длине торса, плечи горизонтально.

    Для офлайн-датасета лучше использовать normalize_skeleton_sequence: там scale/поворот
    берутся как медианы по видео и не дрожат от кадра к кадру.
    """
    xy_arr, conf = _xy_conf_17(xy, scores)
    mid_hip = (xy_arr[L_HIP] + xy_arr[R_HIP]) * 0.5
    mid_shoulder = (xy_arr[L_SH] + xy_arr[R_SH]) * 0.5
    torso = float(np.linalg.norm(mid_shoulder - mid_hip))
    shoulder_w = float(np.linalg.norm(xy_arr[L_SH] - xy_arr[R_SH]))
    scale = torso if torso > 1e-6 else max(shoulder_w, 1.0)
    sh_vec = xy_arr[R_SH] - xy_arr[L_SH]
    phi = float(np.arctan2(float(sh_vec[1]), float(sh_vec[0]) + 1e-8))
    return _normalize_xy_with_reference(xy_arr, conf, scale=scale, shoulder_phi=phi)


def normalize_frame_v2(xy: np.ndarray, scores: np.ndarray | None = None) -> np.ndarray:
    """
    Нормализация v2: центр — середина бёдер; масштаб — геом. среднее торса и ширины плеч;
    поворот 2D: линия плеч выравнивается по горизонтали (Procrustes-подобно для торса).
    Возвращает (17, 3): x, y, confidence.
    """
    xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
    if xy.shape[0] < 17:
        pad = np.zeros((17 - xy.shape[0], 2), dtype=np.float32)
        xy = np.vstack([xy, pad])
    if scores is None:
        conf = np.ones((17,), dtype=np.float32)
    else:
        conf = np.asarray(scores, dtype=np.float32).reshape(-1)
        if conf.shape[0] < 17:
            conf = np.pad(conf, (0, 17 - conf.shape[0]), constant_values=0.0)

    mid_hip = (xy[L_HIP] + xy[R_HIP]) * 0.5
    shoulders_mid = (xy[L_SH] + xy[R_SH]) * 0.5
    torso = float(np.linalg.norm(shoulders_mid - mid_hip)) + 1e-6
    shoulder_w = float(np.linalg.norm(xy[L_SH] - xy[R_SH])) + 1e-6
    scale = float(0.55 * torso + 0.45 * shoulder_w)

    centered = xy - mid_hip
    sh_vec = xy[R_SH] - xy[L_SH]
    sh_norm = float(np.linalg.norm(sh_vec)) + 1e-8
    vx, vy = float(sh_vec[0] / sh_norm), float(sh_vec[1] / sh_norm)
    phi = float(np.arctan2(vy, vx))
    cp, sp = float(np.cos(phi)), float(np.sin(phi))
    x0 = centered[:, 0]
    y0 = centered[:, 1]
    xr = (cp * x0 + sp * y0) / scale
    yr = (-sp * x0 + cp * y0) / scale
    centered = np.stack([xr, yr], axis=-1)

    out = np.concatenate([centered, conf[:, None]], axis=-1)
    return out.astype(np.float32)


def normalize_skeleton_sequence(
    keypoints: np.ndarray,
    scores: np.ndarray | None = None,
    *,
    conf_threshold: float = 0.25,
) -> np.ndarray:
    """
    Нормализация всего видео одним reference center, scale и rotation.

    keypoints: (T,17,2) или (T,17,3). Если scores не передан и C>=3, confidence берётся
    из третьего канала. Центр, scale и угол плеч берутся как медианы по валидным
    кадрам. Так сохраняется вертикальная траектория таза/плеч внутри упражнения.
    """
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1] < 1:
        raise ValueError(f"Ожидалась форма (T,V,C), получено {arr.shape}")
    if arr.shape[2] >= 3 and scores is None:
        xy_raw = arr[:, :, :2]
        sc_raw = arr[:, :, 2]
    else:
        xy_raw = arr[:, :, :2]
        if scores is None:
            sc_raw = np.ones(arr.shape[:2], dtype=np.float32)
        else:
            sc_raw = np.asarray(scores, dtype=np.float32)
    frames_xy: list[np.ndarray] = []
    frames_conf: list[np.ndarray] = []
    for t in range(arr.shape[0]):
        xy_t, conf_t = _xy_conf_17(xy_raw[t], sc_raw[t])
        frames_xy.append(xy_t)
        frames_conf.append(conf_t)
    if not frames_xy:
        return np.zeros((0, 17, 3), dtype=np.float32)
    xy = np.stack(frames_xy, axis=0).astype(np.float32)
    conf = np.stack(frames_conf, axis=0).astype(np.float32)

    mid_hip = (xy[:, L_HIP] + xy[:, R_HIP]) * 0.5
    mid_shoulder = (xy[:, L_SH] + xy[:, R_SH]) * 0.5
    torso = np.linalg.norm(mid_shoulder - mid_hip, axis=1)
    shoulder_vec = xy[:, R_SH] - xy[:, L_SH]
    shoulder_w = np.linalg.norm(shoulder_vec, axis=1)
    valid_torso = (
        (conf[:, L_HIP] >= conf_threshold)
        & (conf[:, R_HIP] >= conf_threshold)
        & (conf[:, L_SH] >= conf_threshold)
        & (conf[:, R_SH] >= conf_threshold)
        & np.isfinite(torso)
        & (torso > 1e-6)
    )
    scale = _robust_positive(torso[valid_torso], fallback=_robust_positive(shoulder_w, 1.0))

    if np.any(valid_torso):
        center_xy = np.median(mid_hip[valid_torso], axis=0).astype(np.float32)
    else:
        center_xy = np.median(mid_hip, axis=0).astype(np.float32)

    valid_sh = (
        (conf[:, L_SH] >= conf_threshold)
        & (conf[:, R_SH] >= conf_threshold)
        & np.isfinite(shoulder_vec).all(axis=1)
        & (shoulder_w > 1e-6)
    )
    if np.any(valid_sh):
        # Median direction on the unit circle, stable for the whole video.
        unit = shoulder_vec[valid_sh] / np.maximum(shoulder_w[valid_sh, None], 1e-6)
        ref = np.median(unit, axis=0)
        shoulder_phi = float(np.arctan2(ref[1], ref[0] + 1e-8))
    else:
        shoulder_phi = 0.0

    out = np.zeros((xy.shape[0], 17, 3), dtype=np.float32)
    for t in range(xy.shape[0]):
        out[t] = _normalize_xy_with_reference(
            xy[t],
            conf[t],
            scale=scale,
            shoulder_phi=shoulder_phi,
            center_xy=center_xy,
        )
    return out


def resample_time(x: np.ndarray, target_len: int) -> np.ndarray:
    """x: (T, V, C) → (target_len, V, C) линейная интерполяция по времени."""
    t, v, c = x.shape
    if t == target_len:
        return x
    if t == 0:
        return np.zeros((target_len, v, c), dtype=np.float32)
    old_idx = np.linspace(0, t - 1, t)
    new_idx = np.linspace(0, t - 1, target_len)
    out = np.zeros((target_len, v, c), dtype=np.float32)
    for i in range(v):
        for j in range(c):
            out[:, i, j] = np.interp(new_idx, old_idx, x[:, i, j])
    return out


def resample_time_adaptive_mean(x: np.ndarray, target_len: int) -> np.ndarray:
    """Среднее по непересекающимся временным бинам (сохраняет контраст быстрый/медленный лучше линейной интерполяции)."""
    t, v, c = x.shape
    if t == target_len:
        return x
    if t == 0:
        return np.zeros((target_len, v, c), dtype=np.float32)
    out = np.zeros((target_len, v, c), dtype=np.float32)
    for i in range(target_len):
        t0 = int(i * t / target_len)
        t1 = int((i + 1) * t / target_len)
        t1 = max(t1, t0 + 1)
        t1 = min(t1, t)
        out[i] = np.mean(x[t0:t1], axis=0)
    return out.astype(np.float32)


def bones_from_joints(seq_xyz: np.ndarray) -> np.ndarray:
    """Векторы костей p_j - p_parent(j); (T, V, 3) как bx, by, conf."""
    seq_xyz = np.asarray(seq_xyz, dtype=np.float32)
    t, v, c = seq_xyz.shape
    out = np.zeros((t, v, 3), dtype=np.float32)
    if c < 3:
        return out
    pos = seq_xyz[..., :2]
    conf = seq_xyz[..., 2]
    for j in range(v):
        pj = COCO_PARENT[j]
        if pj < 0 or pj >= v:
            continue
        b = pos[:, j, :] - pos[:, pj, :]
        out[:, j, 0] = b[:, 0]
        out[:, j, 1] = b[:, 1]
        out[:, j, 2] = np.minimum(conf[:, j], conf[:, pj])
    return out


def _limb_angle_series_deg(seq_xyz: np.ndarray, a: int, b: int, c: int) -> np.ndarray:
    t = seq_xyz.shape[0]
    ang = np.zeros(t, dtype=np.float32)
    for i in range(t):
        ang[i] = angle_2d(seq_xyz[i, a, :2], seq_xyz[i, b, :2], seq_xyz[i, c, :2])
        if not np.isfinite(ang[i]):
            ang[i] = 0.0
    return ang


def angular_velocity_limbs_deg(seq_xyz: np.ndarray) -> np.ndarray:
    """(T, 4): Δугол/кадр для локтей и коленей (градусы)."""
    le = _limb_angle_series_deg(seq_xyz, L_SH, L_ELB, L_WR)
    re = _limb_angle_series_deg(seq_xyz, R_SH, R_ELB, R_WR)
    lk = _limb_angle_series_deg(seq_xyz, L_HIP, L_KNEE, L_ANK)
    rk = _limb_angle_series_deg(seq_xyz, R_HIP, R_KNEE, R_ANK)
    t = seq_xyz.shape[0]
    out = np.zeros((t, 4), dtype=np.float32)
    if t > 1:
        out[1:, 0] = le[1:] - le[:-1]
        out[1:, 1] = re[1:] - re[:-1]
        out[1:, 2] = lk[1:] - lk[:-1]
        out[1:, 3] = rk[1:] - rk[:-1]
    return out


def build_shiftgcn_features(seq_xyz: np.ndarray) -> np.ndarray:
    """
    Полный вход Shift-GCN v3: joint+скорость (6) + bone+скорость (6) + угловые скорости 4 суставов (тайл по V).
    seq_xyz: (T, V, 3) нормализованные координаты.
    Возвращает (T, V, 16).
    """
    seq_xyz = np.asarray(seq_xyz, dtype=np.float32)
    t, v, _ = seq_xyz.shape
    j6 = append_temporal_motion(seq_xyz)
    b3 = bones_from_joints(seq_xyz)
    b6 = append_temporal_motion(b3)
    av = angular_velocity_limbs_deg(seq_xyz)
    av_tile = np.broadcast_to(av[:, None, :], (t, v, 4)).copy()
    return np.concatenate([j6, b6, av_tile], axis=-1).astype(np.float32)


def sequence_to_tensor(seq: np.ndarray) -> np.ndarray:
    """(T, V, C) → (C, T, V) для модели."""
    return np.transpose(seq, (2, 0, 1)).astype(np.float32)


def append_temporal_motion(seq_xyz: np.ndarray) -> np.ndarray:
    """
    Добавляет скорость по времени: (T, V, 3) → (T, V, 6).
    Каналы: x, y, conf, vx, vy, conf_joint на шаге.
    """
    seq_xyz = np.asarray(seq_xyz, dtype=np.float32)
    t, v, c = seq_xyz.shape
    if c != 3:
        raise ValueError(f"append_temporal_motion ожидает C=3, получено {c}")
    vel = np.zeros((t, v, 3), dtype=np.float32)
    if t > 1:
        vel[1:, :, :2] = seq_xyz[1:, :, :2] - seq_xyz[:-1, :, :2]
        vel[1:, :, 2] = np.minimum(seq_xyz[1:, :, 2], seq_xyz[:-1, :, 2])
    return np.concatenate([seq_xyz, vel], axis=-1)


def torch_horizontal_flip_gcn_input(x: torch.Tensor) -> torch.Tensor:
    """
    Тест-тайм / обучение: то же зеркало, что flip_skeleton_horizontal.
    x: (N, C, T, V, M), M=1 у Shift-GCN. C=16: joint6+bone6+ang4 (см. build_shiftgcn_features).
    """
    x2 = x.clone()
    x2[:, 0, ...] *= -1.0
    if x2.size(1) > 3:
        x2[:, 3, ...] *= -1.0
    if x2.size(1) >= 12:
        x2[:, 6, ...] *= -1.0
        x2[:, 9, ...] *= -1.0
    if x2.size(1) >= 16:
        tmp12 = x2[:, 12, ...].clone()
        x2[:, 12, ...] = x2[:, 13, ...]
        x2[:, 13, ...] = tmp12
        tmp14 = x2[:, 14, ...].clone()
        x2[:, 14, ...] = x2[:, 15, ...]
        x2[:, 15, ...] = tmp14
    if x.dim() != 5:
        raise ValueError(f"ожидается (N,C,T,V,M), получено {tuple(x.shape)}")
    for i, j in COCO_LR_SWAP_PAIRS:
        tmp = x2[:, :, :, i, :].clone()
        x2[:, :, :, i, :] = x2[:, :, :, j, :]
        x2[:, :, :, j, :] = tmp
    return x2
