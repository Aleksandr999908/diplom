"""Склейка RTMPose + классификатор скелета + эвристики для одного окна кадров."""

from __future__ import annotations

import pickle

import numpy as np
import torch

from .classifier_loader import build_classifier
from .config import (
    MODEL_IN_CHANNELS,
    PROJECT_ROOT,
    SKELETON_RESAMPLE_MODE,
    SKELETON_SMOOTH_METHOD,
    T_FRAMES,
    inference_exercise_backend,
    lgb_manual_booster_path,
    manual_tcn_checkpoint_path,
    read_lgb_manual_meta,
    read_manual_tcn_meta,
    read_training_meta,
)
from .heuristics import blend_probs, geometry_prior_vector, technique_hints
from .rtmpose_onnx import RTMPoseONNX
from .signal_filters import smooth_skeleton_sequence
from .skeleton_gradcam import gradcam_joint_scores_safe
from .skeleton import (
    append_temporal_motion,
    build_shiftgcn_features,
    resample_time,
    resample_time_adaptive_mean,
    sequence_to_tensor,
    torch_horizontal_flip_gcn_input,
)


def _forward_logits(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    o = model(x)
    return o[0] if isinstance(o, tuple) else o


def _forward_split(
    model: torch.nn.Module, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    o = model(x)
    if isinstance(o, tuple):
        return o[0], o[1], o[2]
    return o, None, None


def _ensemble_val_acc_weights(checkpoint: dict | object) -> float:
    """Невесомая метка качества модели для взвешенного ансамбля логитов."""
    if not isinstance(checkpoint, dict):
        return 0.0
    raw = checkpoint.get("val_acc")
    try:
        v = float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, v)


def _normalize_ensemble_weights(scores: list[float]) -> np.ndarray:
    a = np.asarray(scores, dtype=np.float64)
    if a.size == 0:
        return a
    s = float(a.sum())
    if s < 1e-12:
        a = np.ones_like(a)
        s = float(a.sum())
    return a / s


class ExercisePipeline:
    def __init__(
        self,
        onnx_path: str,
        weights_path: str,
        class_names: list[str],
        device: str | None = None,
        *,
        ensemble_weights: list[str] | None = None,
        exercise_backend: str | None = None,
    ):
        self.class_names = class_names
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_device = torch.device(dev)
        self.pose = RTMPoseONNX(onnx_path, device="cpu")

        be_raw = exercise_backend if exercise_backend is not None else inference_exercise_backend()
        be = str(be_raw).strip().lower()
        if be in ("auto", ""):
            be = inference_exercise_backend()
        if be not in ("torch", "lightgbm", "tcn", "manual_ensemble", "hybrid_ensemble"):
            raise ValueError(
                "exercise_backend: ожидалось torch | lightgbm | tcn | "
                f"manual_ensemble | hybrid_ensemble, получено {be_raw!r}"
            )
        self._exercise_backend: str = be
        train_meta = read_training_meta() or {}
        self._manual_t_frames: int = T_FRAMES
        self._lgb_t_frames: int = T_FRAMES
        self._tcn_t_frames: int = T_FRAMES
        self._manual_conf_threshold: float = 0.3
        self._tcn_include_dynamics: bool = False
        self._lgb_booster = None
        self._lgb_best_iter: int | None = None
        self._lgb_feature_names: list[str] | None = None
        self._tcn_model: torch.nn.Module | None = None
        self._manual_ensemble_lgb_weight: float = 0.5
        self._manual_ensemble_method: str = "average"
        self._manual_stacking_model = None
        self._hybrid_gcn_weight: float = float(train_meta.get("hybrid_gcn_weight", 0.6))
        self.models: list[torch.nn.Module] = []
        self.model: torch.nn.Module | None = None
        self._ensemble_logit_weights = np.asarray([1.0], dtype=np.float64)
        self._ic = MODEL_IN_CHANNELS

        if be in ("lightgbm", "manual_ensemble", "hybrid_ensemble"):
            try:
                import lightgbm as lgb
            except ImportError as e:
                raise ImportError("Установите lightgbm: pip install lightgbm") from e

            meta = read_lgb_manual_meta()
            if meta is None:
                raise FileNotFoundError(
                    "Нет models/lgb_manual_meta.json — выполните scripts/train_manual_lgb.py"
                )
            bp = lgb_manual_booster_path(meta)
            if bp is None:
                raise FileNotFoundError(
                    "В lgb_manual_meta.json указан несуществующий файл weights (.txt модели)"
                )
            m_cls = [str(x) for x in meta.get("classes", []) if x]
            if m_cls != list(class_names):
                raise ValueError(
                    "Классы pipeline ≠ lgb_manual_meta.json (порядок и имена должны совпадать с обучением)"
                )
            self._manual_t_frames = int(meta.get("t_frames", T_FRAMES))
            self._lgb_t_frames = self._manual_t_frames
            self._manual_conf_threshold = float(meta.get("conf_threshold", 0.3))
            raw_feature_names = meta.get("feature_names")
            self._lgb_feature_names = (
                [str(x) for x in raw_feature_names]
                if isinstance(raw_feature_names, list)
                else None
            )
            # model_file на Windows часто падает на путях с кириллицей; строка модели читается через Python.
            self._lgb_booster = lgb.Booster(model_str=bp.read_text(encoding="utf-8"))
            bi = meta.get("best_iteration")
            self._lgb_best_iter = int(bi) if bi is not None else None
            nf_meta = meta.get("n_features")
            nf_booster = int(self._lgb_booster.num_feature())
            if isinstance(nf_meta, int) and nf_meta != nf_booster:
                raise ValueError(
                    f"Число признаков в meta ({nf_meta}) ≠ модель LightGBM ({nf_booster})"
                )
            if be in ("manual_ensemble", "hybrid_ensemble"):
                self._manual_ensemble_lgb_weight = float(
                    train_meta.get("manual_ensemble_lgb_weight", 0.5)
                )
                self._manual_ensemble_method = str(
                    train_meta.get("manual_ensemble_method", "average")
                ).strip().lower()
                stack_rel = train_meta.get("manual_stacking")
                if self._manual_ensemble_method == "stacking" and isinstance(stack_rel, str) and stack_rel.strip():
                    stack_path = (PROJECT_ROOT / stack_rel.replace("\\", "/")).resolve()
                    if stack_path.is_file():
                        with stack_path.open("rb") as f:
                            payload = pickle.load(f)
                        if isinstance(payload, dict):
                            st_classes = [str(x) for x in payload.get("classes", []) if x]
                            if st_classes and st_classes != list(class_names):
                                raise ValueError("Классы stacking meta-model ≠ classes pipeline")
                            self._manual_stacking_model = payload.get("model")
        if be in ("tcn", "manual_ensemble", "hybrid_ensemble"):
            meta = read_manual_tcn_meta()
            if meta is None:
                raise FileNotFoundError(
                    "Нет models/manual_tcn_meta.json — выполните scripts/train_manual_tcn.py"
                )
            ck_path = manual_tcn_checkpoint_path(meta)
            if ck_path is None:
                raise FileNotFoundError("Не найден checkpoint TCN (weights в meta или manual_tcn_best.pt)")
            m_cls = [str(x) for x in meta.get("classes", []) if x]
            if m_cls != list(class_names):
                raise ValueError(
                    "Классы pipeline ≠ manual_tcn_meta.json (порядок и имена должны совпадать с обучением)"
                )
            try:
                st = torch.load(
                    str(ck_path), map_location=self.torch_device, weights_only=False
                )
            except TypeError:
                st = torch.load(str(ck_path), map_location=self.torch_device)
            if not isinstance(st, dict) or "model" not in st:
                raise ValueError(f"Неверный формат checkpoint TCN: {ck_path}")
            in_ch = int(st["in_ch"])
            base_ch = int(st.get("base_ch", 64))
            dropout = float(st.get("dropout", 0.25))
            blocks = int(st.get("blocks", 4))
            from .manual_tcn import SmallTCN

            tm = SmallTCN(
                in_ch, len(class_names), base=base_ch, dropout=dropout, blocks=blocks
            ).to(self.torch_device)
            tm.load_state_dict(st["model"], strict=True)
            tm.eval()
            self._tcn_model = tm
            self._manual_t_frames = int(meta.get("t_frames", T_FRAMES))
            self._tcn_t_frames = self._manual_t_frames
            self._manual_conf_threshold = float(meta.get("conf_threshold", 0.3))
            self._tcn_include_dynamics = bool(meta.get("include_dynamics", in_ch > 18))
        if be in ("torch", "hybrid_ensemble"):
            if not (weights_path or "").strip():
                raise ValueError(
                    "Для exercise_backend=torch/hybrid_ensemble нужен путь к .pt (weights_path)"
                )
            paths = [weights_path] + list(ensemble_weights or [])
            self.models: list[torch.nn.Module] = []
            ics: list[int] = []
            val_scores: list[float] = []
            for wp in paths:
                try:
                    state = torch.load(
                        wp, map_location=self.torch_device, weights_only=False
                    )
                except TypeError:
                    state = torch.load(wp, map_location=self.torch_device)
                val_scores.append(_ensemble_val_acc_weights(state))
                m, ic = build_classifier(state, class_names, self.torch_device)
                m.eval()
                self.models.append(m)
                ics.append(ic)
            self._ensemble_logit_weights = _normalize_ensemble_weights(val_scores)
            if len(set(ics)) > 1:
                raise ValueError(
                    f"Ансамбль: разное число входных каналов в весах {ics}; нужны модели с одинаковым C_in."
                )
            self._ic = ics[0]
            self.model = self.models[0]

        self.last_probs: np.ndarray | None = None
        self.fault_model: torch.nn.Module | None = None
        self.fault_class_names: list[str] = []
        self._load_fault_classifier()

    def _load_fault_classifier(self) -> None:
        from .config import fault_classifier_weights_path, read_fault_training_meta

        meta = read_fault_training_meta()
        if meta is None:
            return
        wpath = fault_classifier_weights_path()
        if wpath is None or not wpath.is_file():
            return
        cl = meta.get("classes")
        if not isinstance(cl, list) or not cl:
            return
        fc = [str(x) for x in cl if x]
        try:
            st = torch.load(
                str(wpath), map_location=self.torch_device, weights_only=False
            )
        except TypeError:
            st = torch.load(str(wpath), map_location=self.torch_device)
        fm, fic = build_classifier(st, fc, self.torch_device)
        if fic != self._ic:
            return
        fm.eval()
        self.fault_model = fm
        self.fault_class_names = fc

    @property
    def has_fault_classifier(self) -> bool:
        return self.fault_model is not None

    def _manual_resampled(self, seq: np.ndarray, t_frames: int | None = None) -> np.ndarray:
        """Тот же сглаживание/ресемпл, что в ручных признаках (длина T из meta)."""
        from .manual_skeleton_features import confidence_gate_sequence
        from .config import MANUAL_FRAME_CONF_THRESHOLD

        tf = int(t_frames if t_frames is not None else self._manual_t_frames)
        seq_g = confidence_gate_sequence(
            seq.astype(np.float32),
            joint_conf_threshold=self._manual_conf_threshold,
            frame_conf_threshold=MANUAL_FRAME_CONF_THRESHOLD,
        )
        seq_s = smooth_skeleton_sequence(seq_g, method=SKELETON_SMOOTH_METHOD)
        if SKELETON_RESAMPLE_MODE == "adaptive_mean":
            return resample_time_adaptive_mean(seq_s, tf)
        return resample_time(seq_s, tf)

    def _preprocess_to_tensor(self, seq: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
        seq_s = smooth_skeleton_sequence(seq.astype(np.float32), method=SKELETON_SMOOTH_METHOD)
        if SKELETON_RESAMPLE_MODE == "adaptive_mean":
            seq_r = resample_time_adaptive_mean(seq_s, T_FRAMES)
        else:
            seq_r = resample_time(seq_s, T_FRAMES)
        if self._ic >= 16:
            seq_feat = build_shiftgcn_features(seq_r)
        elif self._ic == 6:
            seq_feat = append_temporal_motion(seq_r)
        else:
            seq_feat = seq_r
        x = sequence_to_tensor(seq_feat)
        xt = torch.from_numpy(x).unsqueeze(0).to(self.torch_device).unsqueeze(-1)
        return seq_r, xt

    def _tta_split_for_model(
        self, model: torch.nn.Module, xt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        l1, ph1, er1 = _forward_split(model, xt)
        if self._ic >= 6:
            xt2 = torch_horizontal_flip_gcn_input(xt)
            l2, ph2, er2 = _forward_split(model, xt2)
            logits = 0.5 * (l1 + l2)
            phase_logits = (
                0.5 * (ph1 + ph2) if ph1 is not None and ph2 is not None else None
            )
            err_logits = (
                0.5 * (er1 + er2) if er1 is not None and er2 is not None else None
            )
        else:
            logits, phase_logits, err_logits = l1, ph1, er1
        return logits, phase_logits, err_logits

    def _predict_torch_distribution(
        self, seq: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        seq_r, xt = self._preprocess_to_tensor(seq)
        log_stack: list[torch.Tensor] = []
        ph_stack: list[torch.Tensor] = []
        er_stack: list[torch.Tensor] = []
        for m in self.models:
            lg, ph, er = self._tta_split_for_model(m, xt)
            log_stack.append(lg)
            if ph is not None:
                ph_stack.append(ph)
            if er is not None:
                er_stack.append(er)
        w_t = torch.as_tensor(
            self._ensemble_logit_weights,
            dtype=log_stack[0].dtype,
            device=log_stack[0].device,
        ).view(-1, 1, 1)
        logits = (torch.stack(log_stack, dim=0) * w_t).sum(dim=0)
        if ph_stack:
            if len(ph_stack) == len(log_stack):
                phase_logits = (torch.stack(ph_stack, dim=0) * w_t).sum(dim=0)
            else:
                phase_logits = torch.stack(ph_stack, dim=0).mean(dim=0)
        else:
            phase_logits = None
        if er_stack:
            if len(er_stack) == len(log_stack):
                err_logits = (torch.stack(er_stack, dim=0) * w_t).sum(dim=0)
            else:
                err_logits = torch.stack(er_stack, dim=0).mean(dim=0)
        else:
            err_logits = None
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        phase_probs = (
            torch.softmax(phase_logits, dim=-1).squeeze(0).cpu().numpy()
            if phase_logits is not None
            else None
        )
        err_probs = (
            torch.softmax(err_logits, dim=-1).squeeze(0).cpu().numpy()
            if err_logits is not None
            else None
        )
        return probs.astype(np.float64), seq_r, phase_probs, err_probs

    def _predict_lgb_manual_probs(self, seq_r: np.ndarray) -> np.ndarray:
        from .manual_skeleton_features import extract_manual_feature_vector_from_preprocessed

        assert self._lgb_booster is not None
        vec, names = extract_manual_feature_vector_from_preprocessed(
            seq_r, conf_threshold=self._manual_conf_threshold
        )
        if self._lgb_feature_names is not None and len(self._lgb_feature_names) != len(names):
            name_to_i = {name: i for i, name in enumerate(names)}
            missing = [name for name in self._lgb_feature_names if name not in name_to_i]
            if missing:
                raise RuntimeError(
                    f"Текущий extractor не содержит признаков из LGBM meta: {missing[:5]}"
                )
            vec = np.asarray([vec[name_to_i[name]] for name in self._lgb_feature_names], dtype=np.float32)
        vec = vec.reshape(1, -1)
        if self._lgb_best_iter is not None:
            raw = self._lgb_booster.predict(vec, num_iteration=self._lgb_best_iter)
        else:
            raw = self._lgb_booster.predict(vec)
        proba = np.asarray(raw, dtype=np.float64)
        probs = proba.reshape(-1) if proba.ndim == 2 and proba.shape[0] == 1 else proba.ravel()
        n_cls = len(self.class_names)
        if probs.size != n_cls:
            raise RuntimeError(f"LightGBM вернул {probs.size} вероятностей, ожидалось {n_cls}")
        return probs

    def _predict_tcn_manual_probs(self, seq_r: np.ndarray) -> np.ndarray:
        from .manual_skeleton_features import angle_channels_from_preprocessed

        assert self._tcn_model is not None
        ch, _ = angle_channels_from_preprocessed(
            seq_r,
            include_dynamics=self._tcn_include_dynamics,
            conf_threshold=self._manual_conf_threshold,
        )
        x = torch.from_numpy(ch).unsqueeze(0).to(self.torch_device, dtype=torch.float32)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        logits = self._tcn_model(x)
        return torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy().astype(np.float64)

    def _predict_manual_ensemble_probs(self, seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        seq_r = self._manual_resampled(seq, self._lgb_t_frames)
        lgb_probs = self._predict_lgb_manual_probs(seq_r)
        tcn_probs = self._predict_tcn_manual_probs(
            self._manual_resampled(seq, self._tcn_t_frames)
        )
        if self._manual_ensemble_method == "stacking" and self._manual_stacking_model is not None:
            x_stack = np.concatenate([lgb_probs, tcn_probs], axis=0).reshape(1, -1)
            probs = np.asarray(
                self._manual_stacking_model.predict_proba(x_stack), dtype=np.float64
            ).reshape(-1)
        else:
            w = float(np.clip(self._manual_ensemble_lgb_weight, 0.0, 1.0))
            probs = w * lgb_probs + (1.0 - w) * tcn_probs
        probs = probs / max(float(np.sum(probs)), 1e-12)
        return probs, seq_r

    @torch.inference_mode()
    def predict_fault_hint(
        self, seq: np.ndarray, exercise_name: str
    ) -> tuple[str, float] | None:
        """
        По обученному train_fault.py: тип ошибки (fault_id) при фиксированном упражнении.
        Вероятности режутся до классов вида «упражнение__fault_id».
        """
        if (
            self.fault_model is None
            or not self.fault_class_names
            or not (exercise_name or "").strip()
            or exercise_name.strip() in ("—", "…")
        ):
            return None
        ex = exercise_name.strip()
        prefix = f"{ex}__"
        idxs = [i for i, n in enumerate(self.fault_class_names) if n.startswith(prefix)]
        if not idxs:
            return None
        seq_r, xt = self._preprocess_to_tensor(seq)
        logits, _, _ = self._tta_split_for_model(self.fault_model, xt)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        sub = torch.stack([probs[i] for i in idxs])
        sub = sub / sub.sum().clamp(min=1e-12)
        j = int(torch.argmax(sub))
        full = self.fault_class_names[idxs[j]]
        fault_id = full[len(prefix) :] if full.startswith(prefix) else full
        return fault_id, float(sub[j].item())

    @torch.inference_mode()
    def predict_class_distribution(
        self, seq: np.ndarray
    ) -> tuple[str, float, bool, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        """
        Этап B: классификация упражнения (+ эвристический blend при низкой уверенности).
        seq: (T_raw, 17, 3) нормализованные кости.
        Дополнительно: распределения по фазе и «ошибке», если модель обучена с multi_task.
        """
        phase_probs: np.ndarray | None
        err_probs: np.ndarray | None

        if self._exercise_backend == "lightgbm":
            seq_r = self._manual_resampled(seq, self._lgb_t_frames)
            probs = self._predict_lgb_manual_probs(seq_r)
            phase_probs = None
            err_probs = None
        elif self._exercise_backend == "tcn":
            seq_r = self._manual_resampled(seq, self._tcn_t_frames)
            probs = self._predict_tcn_manual_probs(seq_r)
            phase_probs = None
            err_probs = None
        elif self._exercise_backend == "manual_ensemble":
            probs, seq_r = self._predict_manual_ensemble_probs(seq)
            phase_probs = None
            err_probs = None
        elif self._exercise_backend == "hybrid_ensemble":
            gcn_probs, seq_r, phase_probs, err_probs = self._predict_torch_distribution(seq)
            manual_probs, _manual_seq_r = self._predict_manual_ensemble_probs(seq)
            w = float(np.clip(self._hybrid_gcn_weight, 0.0, 1.0))
            probs = w * gcn_probs + (1.0 - w) * manual_probs
            probs = probs / max(float(np.sum(probs)), 1e-12)
        else:
            probs, seq_r, phase_probs, err_probs = self._predict_torch_distribution(seq)

        geom = geometry_prior_vector(seq_r, self.class_names)
        ent = float(-np.sum(probs * np.log(probs + 1e-12)))
        h_max = float(np.log(max(2, len(probs))))
        uncertain = min(1.0, max(0.0, ent / max(1e-6, h_max)))
        mx_nn = float(np.max(probs))
        # Выше вес сети: геометрия часто путает близкие классы (тяги, жимы), хотя GCN уже различает.
        w_nn = 0.52 + 0.34 * (1.0 - uncertain)
        if mx_nn > 0.62:
            w_nn = min(0.92, w_nn + 0.07)
        fused = w_nn * probs + (1.0 - w_nn) * geom
        fused = fused / max(1e-12, float(fused.sum()))
        i_nn = int(np.argmax(probs))
        i_gm = int(np.argmax(geom))
        if i_nn != i_gm:
            gap = float(probs[i_nn] - probs[i_gm])
            if gap < 0.14 and mx_nn < 0.58:
                fused = 0.74 * fused + 0.26 * geom
                fused = fused / max(1e-12, float(fused.sum()))
        p2, used_rules = blend_probs(fused, seq_r, self.class_names, nn_weight=0.62)
        if p2.size > 1:
            order = np.argsort(p2)
            margin = float(p2[order[-1]] - p2[order[-2]])
            if margin < 0.12 and float(np.max(p2)) < 0.55:
                p2 = 0.62 * p2 + 0.38 * geom
                p2 = p2 / max(1e-12, float(p2.sum()))
        self.last_probs = p2
        idx = int(np.argmax(p2))
        name = self.class_names[idx]
        conf = float(p2[idx])
        return name, conf, used_rules, p2, seq_r, phase_probs, err_probs

    def gradcam_joint_importance(self, seq: np.ndarray, class_idx: int) -> np.ndarray | None:
        """Важность суставов для выбранного класса; только при C>=6."""
        if self.model is None or self._ic < 6 or class_idx < 0:
            return None
        _, xt = self._preprocess_to_tensor(seq)
        x = xt.squeeze(0).squeeze(-1).detach().cpu().numpy()
        self.model.eval()
        return gradcam_joint_scores_safe(self.model, x, self.torch_device, class_idx)

    @torch.inference_mode()
    def predict_sequence(self, seq: np.ndarray) -> tuple[str, float, str, bool]:
        name, conf, used_rules, _, seq_r, _, _ = self.predict_class_distribution(seq)
        hint = technique_hints(name, seq_r)
        return name, conf, hint, used_rules
