"""
Tkinter + OpenCV: этапы A–E
A — поза (RTMPose), B — класс упражнения (Shift-GCN), C — фаза, D–E — ошибки и подсветка суставов.
"""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

if TYPE_CHECKING:
    from .pipeline import ExercisePipeline
    from .stages.types import FormFault

from .config import (
    GUI_CAPTURE_MAX_SECONDS,
    GUI_FILE_PLAYBACK_SPEED,
    GUI_FILE_POSE_EVERY_N_FRAMES,
    RTMPOSE_ONNX,
    SESSION_DB,
    SKELETON_EXTRACT_MAX_FRAMES,
    T_FRAMES,
    classifier_ensemble_extra_paths,
    classifier_weights_path,
    inference_class_names,
    inference_exercise_backend,
    lgb_manual_booster_path,
    manual_tcn_checkpoint_path,
    read_lgb_manual_meta,
    read_manual_tcn_meta,
)

# Меньше кадров, чем T_FRAMES//2 — быстрее появляются B–E при живом видео
MIN_SKELETON_FRAMES_FOR_ANALYSIS = max(20, (T_FRAMES * 3) // 8)
# Файл обрабатываем последовательно; камера ниже всё равно берёт только последний кадр.
_GUI_FILE_FRAMES_PER_TICK = 1
from .database import connect, end_session, init_schema, log_event, start_session
from .rtmpose_onnx import RTMPoseONNX
from .skeleton import normalize_skeleton_sequence
from .stages import FrameAnalysis, MultiStageAnalyzer
from .stages.joint_labels import joints_ru


def format_phase_for_ui(phase: str, detail: str) -> tuple[str, str]:
    """Чёткая подпись: что за фаза видна на видео сейчас (по последнему окну скелета)."""
    ph = (phase or "—").strip() or "—"
    det = (detail or "").strip()
    line1 = f"Сейчас на видео — {ph}"
    line2 = f"По скелету: {det}" if det else ""
    return line1, line2


COCO_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def fault_joint_short_labels(faults: list["FormFault"]) -> dict[int, str]:
    """Краткие подписи у ключевых точек (кириллица через PIL)."""
    labels: dict[int, str] = {}
    for f in faults:
        lab = (f.short_label or "").strip()
        if not lab:
            lab = "ошибка" if f.severity == "error" else "внимание"
        for j in f.joints:
            prev = labels.get(j)
            if prev is None:
                labels[j] = lab
            elif lab and lab not in prev and len(prev) + len(lab) < 30:
                labels[j] = f"{prev}, {lab}"
    return labels


def draw_pose_with_issues(
    bgr: np.ndarray,
    kpts: np.ndarray,
    scores: np.ndarray,
    fault_joints: set[int],
    joint_labels: dict[int, str] | None = None,
    joint_importance: np.ndarray | None = None,
    thr: float = 0.3,
) -> None:
    """Этап E: зелёный скелет; суставы с ошибками — кольцо и заливка."""
    fault_joints = fault_joints or set()
    for i, j in COCO_EDGES:
        if scores[i] > thr and scores[j] > thr:
            p1 = tuple(kpts[i].astype(int))
            p2 = tuple(kpts[j].astype(int))
            col = (0, 255, 0)
            if i in fault_joints and j in fault_joints:
                col = (0, 80, 255)
            elif i in fault_joints or j in fault_joints:
                col = (0, 180, 255)
            cv2.line(bgr, p1, p2, col, 2, cv2.LINE_AA)
    for i in range(len(kpts)):
        if scores[i] <= thr:
            continue
        pt = tuple(kpts[i].astype(int))
        if i in fault_joints:
            cv2.circle(bgr, pt, 10, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(bgr, pt, 4, (120, 120, 255), -1, cv2.LINE_AA)
        else:
            cv2.circle(bgr, pt, 3, (0, 128, 255), -1, cv2.LINE_AA)
            if joint_importance is not None and i < len(joint_importance):
                wgt = float(joint_importance[i])
                if wgt > 0.04:
                    rr = max(5, int(4 + 14.0 * wgt))
                    cv2.circle(bgr, pt, rr, (0, 200, 255), 1, cv2.LINE_AA)

    if joint_labels:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_im)
        font: ImageFont.ImageFont | ImageFont.FreeTypeFont
        try:
            font = ImageFont.truetype("segoeui.ttf", 13)
        except OSError:
            try:
                font = ImageFont.truetype("arial.ttf", 13)
            except OSError:
                font = ImageFont.load_default()
        for ji, text in joint_labels.items():
            if ji >= len(kpts) or scores[ji] <= thr or not text.strip():
                continue
            x, y = float(kpts[ji][0]), float(kpts[ji][1])
            draw.text((x + 7, y - 16), text, fill=(255, 85, 40), font=font)
        bgr[:, :, :] = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Контроль техники: поза → класс → фаза → ошибки (A–E)")
        self.geometry("1520x920")
        try:
            self.state("zoomed")
        except tk.TclError:
            pass

        self.classes = inference_class_names()
        # Для файла нужен небольшой буфер без потери порядка; камера ниже всё равно сбрасывает backlog.
        self.frame_q: queue.Queue[tuple[np.ndarray | None, int]] = queue.Queue(maxsize=96)
        self._video_target_w = 1280
        self._video_target_h = 720
        self.stop_capture = threading.Event()
        self.capture_thread: threading.Thread | None = None
        self.pipeline: ExercisePipeline | None = None
        self.pose_infer: RTMPoseONNX | None = None
        self.analyzer: MultiStageAnalyzer | None = None
        self.session_id: int | None = None

        self.buf: list[np.ndarray] = []
        # Файл: та же схема, что extract_skeletons (шаг по номеру кадра, до SKELETON_EXTRACT_MAX_FRAMES).
        self._file_subsample_buf: list[np.ndarray] = []
        self._file_step: int = 1
        self._file_read_idx: int = 0
        self._display_frame_idx: int = 0
        self._next_file_sample_idx: int = 0
        self._last_log_t = 0.0
        self._last_log_name = ""
        self._capture_ended_timelimit = False
        self._active_video_path: str | None = None
        self._last_analysis: FrameAnalysis | None = None
        self._llm_seq: np.ndarray | None = None
        self._llm_busy = False

        top = ttk.Frame(self, padding=8)
        top.pack(fill=tk.X)
        ttk.Button(top, text="Камера", command=self.start_camera).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Видео…", command=self.open_video).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Стоп", command=self.stop_stream).pack(side=tk.LEFT, padx=4)
        self.var_log = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Лог в SQLite", variable=self.var_log).pack(side=tk.LEFT, padx=8)

        self.lbl_status = ttk.Label(top, text="Инициализация…")
        self.lbl_status.pack(side=tk.RIGHT)

        self._video_paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        self._video_paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        video_frame = ttk.Frame(self._video_paned)
        self._video_paned.add(video_frame, weight=4)
        _vbg = self._theme_flat_bg()
        self.video_label = tk.Label(
            video_frame,
            background=_vbg,
            highlightthickness=0,
            borderwidth=0,
        )
        self.video_label.pack(fill=tk.BOTH, expand=True)
        self.video_label.bind("<Configure>", self._on_video_panel_configure)

        side = ttk.Frame(self._video_paned, width=380)
        self._video_paned.add(side, weight=1)
        wrap = 360

        ttk.Label(side, text="Этап A: скелет (RTMPose)", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        ttk.Label(side, text="Отображается на видео", foreground="gray").pack(anchor=tk.W)

        ttk.Label(side, text="Этап B: классификация упражнения", font=("Segoe UI", 11, "bold")).pack(
            anchor=tk.W, pady=(10, 0)
        )
        self.lbl_class = ttk.Label(side, text="—", font=("Segoe UI", 13))
        self.lbl_class.pack(anchor=tk.W, pady=2)
        self.lbl_conf = ttk.Label(side, text="", wraplength=wrap)
        self.lbl_conf.pack(anchor=tk.W)
        self.lbl_blend = ttk.Label(side, text="", wraplength=wrap, foreground="gray")
        self.lbl_blend.pack(anchor=tk.W)

        ttk.Label(
            side,
            text="Этап C: фаза движения (что сейчас на видео)",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor=tk.W, pady=(12, 0))
        self.lbl_phase = ttk.Label(
            side, text="Сейчас на видео — …", font=("Segoe UI", 12, "bold"), wraplength=wrap
        )
        self.lbl_phase.pack(anchor=tk.W, pady=2)
        self.lbl_phase_d = ttk.Label(side, text="", wraplength=wrap, foreground="gray")
        self.lbl_phase_d.pack(anchor=tk.W)
        self.lbl_reps = ttk.Label(side, text="", wraplength=wrap, font=("Segoe UI", 11))
        self.lbl_reps.pack(anchor=tk.W, pady=(4, 0))
        self.lbl_aux = ttk.Label(side, text="", wraplength=wrap, foreground="gray")
        self.lbl_aux.pack(anchor=tk.W, pady=(2, 0))

        ttk.Label(side, text="Этап D: ошибка техники", font=("Segoe UI", 11, "bold")).pack(
            anchor=tk.W, pady=(12, 0)
        )
        ttk.Label(
            side,
            text="Этап E: ошибки (красный); кольца по важности суставов — Grad-CAM (встроено)",
            font=("Segoe UI", 9),
            foreground="gray",
        ).pack(anchor=tk.W)
        self.txt_faults = tk.Text(side, height=14, width=42, wrap=tk.WORD, state=tk.DISABLED)
        self.txt_faults.pack(fill=tk.X, pady=4)

        ttk.Label(side, text="Техника и как исправить", font=("Segoe UI", 10, "bold")).pack(
            anchor=tk.W, pady=(8, 0)
        )
        self.txt_hint = tk.Text(side, height=9, width=42, wrap=tk.WORD, state=tk.DISABLED)
        self.txt_hint.pack(fill=tk.X, pady=2)

        ttk.Label(side, text="ИИ-тренер (LLM по скелету)", font=("Segoe UI", 10, "bold")).pack(
            anchor=tk.W, pady=(10, 0)
        )
        llm_row = ttk.Frame(side)
        llm_row.pack(fill=tk.X)
        ttk.Button(llm_row, text="Получить совет", command=self._request_llm_advice).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Label(
            llm_row,
            text=".env: LLM_PROVIDER=chad, CHAD_API_KEY",
            font=("Segoe UI", 8),
            foreground="gray",
        ).pack(side=tk.LEFT)
        self.txt_llm = tk.Text(side, height=8, width=42, wrap=tk.WORD, state=tk.DISABLED)
        self.txt_llm.pack(fill=tk.X, pady=(2, 0))

        self._init_models()
        self.after(12, self._drain_loop)
        self.after(200, self._apply_initial_sash)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _theme_flat_bg(self) -> str:
        """Цвет фона как у ttk-панелей (убирает «чёрный квадрат» под letterbox видео)."""
        st = ttk.Style(self)
        for key in ("TFrame", "TLabel", "TPanedwindow"):
            bg = st.lookup(key, "background")
            if bg and str(bg).strip():
                return str(bg)
        try:
            return self.cget("bg")
        except tk.TclError:
            return "SystemButtonFace"

    def _apply_initial_sash(self) -> None:
        """Больше места под видео; разделитель можно двигать мышью."""
        try:
            w = max(500, int(self.winfo_width()))
            self._video_paned.sashpos(0, int(w * 0.72))
        except tk.TclError:
            pass

    def _on_video_panel_configure(self, event: tk.Event) -> None:
        if event.widget is not self.video_label:
            return
        w, h = int(event.width), int(event.height)
        if w > 8 and h > 8:
            self._video_target_w = w
            self._video_target_h = h

    def _infer_pose_tracked(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Только полный кадр — как в extract_skeletons.py. ROI-кроп ломал масштаб RTMPose
        и давал другие координаты, чем при обучении.
        """
        assert self.pose_infer is not None
        return self.pose_infer.infer(frame)

    def _init_models(self) -> None:
        try:
            if not RTMPOSE_ONNX.is_file():
                messagebox.showerror(
                    "Нет RTMPose ONNX",
                    f"Выполните:\npython scripts/download_model.py\n\nОжидался файл:\n{RTMPOSE_ONNX}",
                )
                self.lbl_status.config(text="Нет модели позы")
                return
            backend = inference_exercise_backend()
            wpath = classifier_weights_path()
            from .pipeline import ExercisePipeline

            loaded = False
            if self.classes and backend == "hybrid_ensemble":
                lm = read_lgb_manual_meta()
                tm = read_manual_tcn_meta()
                has_lgb = lm is not None and lgb_manual_booster_path(lm) is not None
                has_tcn = tm is not None and manual_tcn_checkpoint_path(tm) is not None
                if wpath.is_file() and has_lgb and has_tcn:
                    try:
                        extra = [str(p) for p in classifier_ensemble_extra_paths()]
                        self.pipeline = ExercisePipeline(
                            str(RTMPOSE_ONNX),
                            str(wpath),
                            self.classes,
                            ensemble_weights=extra or None,
                            exercise_backend="hybrid_ensemble",
                        )
                        loaded = True
                    except Exception as e:
                        self.pipeline = None
                        messagebox.showwarning(
                            "Hybrid ensemble",
                            f"Не удалось загрузить Shift-GCN+LGBM+TCN:\n{e}",
                        )
                else:
                    self.pipeline = None
                    messagebox.showwarning(
                        "Hybrid ensemble",
                        "В training_meta указано exercise_backend=hybrid_ensemble, но нет "
                        "Shift-GCN .pt, lgb_manual_meta.json / manual_tcn_meta.json "
                        "или файлов моделей.\n"
                        "Запустите train.py, train_manual_lgb.py, train_manual_tcn.py "
                        "и tune_manual_ensemble.py",
                    )
            elif self.classes and backend == "manual_ensemble":
                lm = read_lgb_manual_meta()
                tm = read_manual_tcn_meta()
                has_lgb = lm is not None and lgb_manual_booster_path(lm) is not None
                has_tcn = tm is not None and manual_tcn_checkpoint_path(tm) is not None
                if has_lgb and has_tcn:
                    try:
                        self.pipeline = ExercisePipeline(
                            str(RTMPOSE_ONNX),
                            "",
                            self.classes,
                            ensemble_weights=None,
                            exercise_backend="manual_ensemble",
                        )
                        loaded = True
                    except Exception as e:
                        self.pipeline = None
                        messagebox.showwarning(
                            "Manual ensemble",
                            f"Не удалось загрузить ансамбль LGBM+TCN:\n{e}",
                        )
                else:
                    self.pipeline = None
                    messagebox.showwarning(
                        "Manual ensemble",
                        "В training_meta указано exercise_backend=manual_ensemble, но нет "
                        "lgb_manual_meta.json / manual_tcn_meta.json или файлов моделей.\n"
                        "Запустите train_manual_lgb.py, train_manual_tcn.py и tune_manual_ensemble.py",
                    )
            elif self.classes and backend == "lightgbm":
                lm = read_lgb_manual_meta()
                if lm is not None and lgb_manual_booster_path(lm) is not None:
                    try:
                        self.pipeline = ExercisePipeline(
                            str(RTMPOSE_ONNX),
                            "",
                            self.classes,
                            ensemble_weights=None,
                            exercise_backend="lightgbm",
                        )
                        loaded = True
                    except Exception as e:
                        self.pipeline = None
                        messagebox.showwarning(
                            "LightGBM",
                            f"Не удалось загрузить ручной классификатор:\n{e}\n\n"
                            "Проверьте models/lgb_manual_meta.json и pip install lightgbm",
                        )
                else:
                    self.pipeline = None
                    messagebox.showwarning(
                        "LightGBM",
                        "В training_meta указано exercise_backend=lightgbm, но нет "
                        "lgb_manual_meta.json или файла модели.\n"
                        "Запустите: python scripts/train_manual_lgb.py",
                    )
            elif self.classes and backend == "tcn":
                tm = read_manual_tcn_meta()
                if tm is not None and manual_tcn_checkpoint_path(tm) is not None:
                    try:
                        self.pipeline = ExercisePipeline(
                            str(RTMPOSE_ONNX),
                            "",
                            self.classes,
                            ensemble_weights=None,
                            exercise_backend="tcn",
                        )
                        loaded = True
                    except Exception as e:
                        self.pipeline = None
                        messagebox.showwarning(
                            "TCN",
                            f"Не удалось загрузить manual TCN:\n{e}",
                        )
                else:
                    self.pipeline = None
                    messagebox.showwarning(
                        "TCN",
                        "В training_meta указано exercise_backend=tcn, но нет "
                        "manual_tcn_meta.json или весов.\n"
                        "Запустите: python scripts/train_manual_tcn.py",
                    )
            elif wpath.is_file() and self.classes:
                try:
                    extra = [str(p) for p in classifier_ensemble_extra_paths()]
                    self.pipeline = ExercisePipeline(
                        str(RTMPOSE_ONNX),
                        str(wpath),
                        self.classes,
                        ensemble_weights=extra or None,
                        exercise_backend="torch",
                    )
                    loaded = True
                except Exception as e:
                    self.pipeline = None
                    self.pose_infer = RTMPoseONNX(str(RTMPOSE_ONNX), device="cpu")
                    self.analyzer = MultiStageAnalyzer(None, self.classes)
                    self.lbl_status.config(text="Поза OK; классификатор не загрузился")
                    messagebox.showwarning(
                        "Классификатор",
                        f"Не удалось загрузить веса ({wpath.name}):\n{e}\n\n"
                        "Этапы B–E будут по геометрии. Переобучите модель под текущие признаки:\n"
                        "python scripts/train.py",
                    )
            if loaded:
                assert self.pipeline is not None
                self.pose_infer = self.pipeline.pose
                self.analyzer = MultiStageAnalyzer(self.pipeline)
                tag = (
                    "LightGBM (ручные признаки)"
                    if backend == "lightgbm"
                    else (
                        "Shift-GCN+LGBM+TCN"
                        if backend == "hybrid_ensemble"
                        else (
                            "LGBM+TCN (manual ensemble)"
                            if backend == "manual_ensemble"
                            else ("TCN (углы)" if backend == "tcn" else wpath.name)
                        )
                    )
                )
                sfx = " + сеть ошибок" if self.pipeline.has_fault_classifier else ""
                self.lbl_status.config(text=f"Готово: A–E ({tag}){sfx}")
            elif self.classes:
                self.pose_infer = RTMPoseONNX(str(RTMPOSE_ONNX), device="cpu")
                self.analyzer = MultiStageAnalyzer(None, self.classes)
                self.lbl_status.config(
                    text="A–E по геометрии (обучите модель → models/*.pt, см. training_meta.json)"
                )
            else:
                self.pose_infer = RTMPoseONNX(str(RTMPOSE_ONNX), device="cpu")
                self.lbl_status.config(
                    text="Только поза: нет классов (папки в data/raw или models/training_meta.json)"
                )
        except Exception as e:
            messagebox.showerror("Ошибка моделей", str(e))
            self.lbl_status.config(text="Ошибка загрузки")

    def _set_status_for_video_path(self, path: str | None) -> None:
        """Подсказка в строке состояния: камера или файл."""
        if path is None:
            self.lbl_status.config(text="Камера")
            return
        self.lbl_status.config(text=f"Видео: {Path(path).name}")

    def start_camera(self) -> None:
        self.stop_stream()
        self._active_video_path = None
        self._file_step = 1
        self._file_read_idx = 0
        self._display_frame_idx = 0
        self._next_file_sample_idx = 0
        self._file_subsample_buf.clear()
        self._set_status_for_video_path(None)
        self._capture_ended_timelimit = False
        self.stop_capture.clear()
        self.capture_thread = threading.Thread(target=self._capture_loop, args=(0,), daemon=True)
        self.capture_thread.start()

    def open_video(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mkv"), ("All", "*.*")])
        if not path:
            return
        self.stop_stream()
        self._active_video_path = path
        nframes = 0
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        cap.release()
        # В GUI важнее быстрый старт анализа: берём каждую обработанную позу,
        # а длину окна ограничиваем SKELETON_EXTRACT_MAX_FRAMES ниже.
        self._file_step = 1
        self._file_read_idx = 0
        self._display_frame_idx = 0
        self._next_file_sample_idx = 0
        self._file_subsample_buf.clear()
        self._set_status_for_video_path(path)
        self._capture_ended_timelimit = False
        self.stop_capture.clear()
        self.capture_thread = threading.Thread(target=self._capture_loop, args=(path,), daemon=True)
        self.capture_thread.start()

    def _put_frame_drop_old(self, frame: np.ndarray | None, frame_idx: int = -1) -> None:
        """Кладёт кадр без накопления задержки: если очередь полная, старый кадр выбрасывается."""
        item = (frame, int(frame_idx))
        try:
            self.frame_q.put(item, timeout=0.002)
            return
        except queue.Full:
            pass
        try:
            _ = self.frame_q.get_nowait()
        except queue.Empty:
            pass
        try:
            self.frame_q.put_nowait(item)
        except queue.Full:
            pass

    def _put_frame_ordered(self, frame: np.ndarray | None, frame_idx: int = -1) -> None:
        """Для видеофайла: сохраняет порядок кадров и притормаживает чтение, если GUI занят."""
        item = (frame, int(frame_idx))
        while not self.stop_capture.is_set():
            try:
                self.frame_q.put(item, timeout=0.05)
                return
            except queue.Full:
                continue

    def _capture_loop(self, source: int | str) -> None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self._put_frame_drop_old(None)
            return
        is_file = isinstance(source, str)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_delay = (1.0 / fps) if is_file and 1.5 <= fps <= 120.0 else 0.0
        t0 = time.monotonic()
        frame_idx = 0
        while not self.stop_capture.is_set():
            if not is_file and time.monotonic() - t0 >= GUI_CAPTURE_MAX_SECONDS:
                self._capture_ended_timelimit = True
                break
            t_read = time.monotonic()
            ok, frame = cap.read()
            if not ok:
                break
            if is_file:
                self._put_frame_ordered(frame, frame_idx)
            else:
                self._put_frame_drop_old(frame, -1)
            frame_idx += 1
            if frame_delay > 0.0:
                spd = float(GUI_FILE_PLAYBACK_SPEED)
                spd = max(0.42, min(2.5, spd))
                delay = frame_delay / spd
                spent = time.monotonic() - t_read
                rem = delay - spent
                if rem > 0:
                    time.sleep(rem)
        cap.release()
        if is_file:
            self._put_frame_ordered(None)
        else:
            self._put_frame_drop_old(None)

    def stop_stream(self) -> None:
        self.stop_capture.set()
        self.buf.clear()
        self._file_subsample_buf.clear()
        self._file_read_idx = 0
        self._display_frame_idx = 0
        self._next_file_sample_idx = 0
        self._last_analysis = None
        self._llm_seq = None
        if self.analyzer is not None:
            self.analyzer.reset_temporal_smoothing()
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)

    def _drain_loop(self) -> None:
        """
        Камера: только последний кадр — меньше задержка.
        Файл: все кадры по порядку — иначе ломается шаг субдискретизации как в датасете.
        """
        stream_ended = False
        try:
            if self._active_video_path is not None:
                for _ in range(_GUI_FILE_FRAMES_PER_TICK):
                    try:
                        frame, frame_idx = self.frame_q.get_nowait()
                    except queue.Empty:
                        break
                    if frame is None:
                        stream_ended = True
                        break
                    self._process_frame(frame, frame_idx)
            else:
                latest: np.ndarray | None = None
                latest_idx = -1
                while True:
                    try:
                        frame, frame_idx = self.frame_q.get_nowait()
                    except queue.Empty:
                        break
                    if frame is None:
                        stream_ended = True
                        break
                    latest = frame
                    latest_idx = frame_idx
                if latest is not None:
                    self._process_frame(latest, latest_idx)
        except queue.Empty:
            pass
        alive = self.capture_thread and self.capture_thread.is_alive()
        if alive and self._active_video_path is not None:
            interval = 4
        elif alive:
            interval = 10
        else:
            interval = 50
        self.after(interval, self._drain_loop)
        if stream_ended:
            if self._capture_ended_timelimit:
                self.lbl_status.config(
                    text=f"Захват завершён ({GUI_CAPTURE_MAX_SECONDS:.0f} с)"
                )
                self._capture_ended_timelimit = False
            else:
                self.lbl_status.config(text="Поток остановлен")

    def _set_faults_text(self, analysis: FrameAnalysis | None, ready: bool) -> None:
        self.txt_faults.config(state=tk.NORMAL)
        self.txt_faults.delete("1.0", tk.END)
        if not ready:
            self.txt_faults.insert(
                tk.END,
                "Накопление окна кадров… Этапы B–E активируются после достаточной длины ряда.",
            )
        elif analysis is None:
            self.txt_faults.insert(tk.END, "—")
        elif not analysis.faults:
            self.txt_faults.insert(
                tk.END,
                "Техника без явных ошибок по текущим правилам.\n"
                "Продолжайте контролировать темп, амплитуду и дыхание.",
            )
        else:
            lines: list[str] = []
            for f in analysis.faults:
                sev = "Внимание" if f.severity == "warning" else "Ошибка"
                jn = joints_ru(f.joints)
                lines.append(f"• [{sev}] {f.message}")
                if f.confidence_score > 0:
                    lines.append(f"  Уверенность: {f.confidence_score:.0%}")
                lines.append(f"  Суставы: {jn}")
                if f.moment:
                    lines.append(f"  Момент: {f.moment}")
                if (f.fix_hint or "").strip():
                    lines.append(f"  Как исправить: {f.fix_hint.strip()}")
                lines.append("")
            self.txt_faults.insert(tk.END, "\n".join(lines).strip())
        self.txt_faults.config(state=tk.DISABLED)

    def _request_llm_advice(self) -> None:
        """Запрос к LLM: контекст из окна скелета + этапы B–E (в фоне, без блокировки UI)."""
        if self._llm_busy:
            return
        an = self._last_analysis
        seq = self._llm_seq
        if an is None or seq is None:
            messagebox.showinfo(
                "ИИ-тренер",
                "Сначала запустите камеру или видео и дождитесь анализа (этапы B–E).",
            )
            return
        self._llm_busy = True
        self.txt_llm.config(state=tk.NORMAL)
        self.txt_llm.delete("1.0", tk.END)
        self.txt_llm.insert(tk.END, "Запрос к модели…")
        self.txt_llm.config(state=tk.DISABLED)

        an_snap = an
        seq_copy = np.array(seq, copy=True)

        def work() -> None:
            try:
                from .llm_coach import build_llm_payload, request_coaching_text

                payload = build_llm_payload(an_snap, seq_copy)
                text = request_coaching_text(payload)
                self.after(0, lambda t=text: self._llm_advice_done_ok(t))
            except Exception as e:
                self.after(0, lambda err=str(e): self._llm_advice_done_err(err))

        threading.Thread(target=work, daemon=True).start()

    def _llm_advice_done_ok(self, text: str) -> None:
        self._llm_busy = False
        self.txt_llm.config(state=tk.NORMAL)
        self.txt_llm.delete("1.0", tk.END)
        self.txt_llm.insert(tk.END, text)
        self.txt_llm.config(state=tk.DISABLED)

    def _llm_advice_done_err(self, err: str) -> None:
        self._llm_busy = False
        self.txt_llm.config(state=tk.NORMAL)
        self.txt_llm.delete("1.0", tk.END)
        self.txt_llm.insert(tk.END, f"Не удалось получить ответ.\n\n{err}")
        self.txt_llm.config(state=tk.DISABLED)

    def _process_frame(self, frame: np.ndarray, frame_idx: int = -1) -> None:
        display = frame.copy()
        analysis: FrameAnalysis | None = None
        fault_joints: set[int] = set()
        ready_bcde = False
        file_frame_idx: int | None = None
        analyze_pose = True
        if self._active_video_path is not None:
            file_frame_idx = int(frame_idx) if frame_idx >= 0 else self._display_frame_idx
            self._display_frame_idx += 1
            stride = max(1, int(GUI_FILE_POSE_EVERY_N_FRAMES))
            analyze_pose = (file_frame_idx % stride) == 0

        if analyze_pose and self.pose_infer is not None and self.analyzer is not None:
            k, s = self._infer_pose_tracked(frame)
            xy = np.asarray(k, dtype=np.float32).reshape(-1, 2)
            sc = np.asarray(s, dtype=np.float32).reshape(-1)
            if xy.shape[0] < 17:
                xy = np.vstack([xy, np.zeros((17 - xy.shape[0], 2), dtype=np.float32)])
            if sc.shape[0] < 17:
                sc = np.pad(sc, (0, 17 - sc.shape[0]), constant_values=0.0)
            raw_pose = np.concatenate([xy[:17], sc[:17, None]], axis=-1).astype(np.float32)
            if self._active_video_path is not None:
                current_idx = int(file_frame_idx or 0)
                self._file_read_idx = current_idx + 1
                if current_idx >= self._next_file_sample_idx:
                    self._file_subsample_buf.append(raw_pose)
                    if len(self._file_subsample_buf) > SKELETON_EXTRACT_MAX_FRAMES:
                        self._file_subsample_buf.pop(0)
                    self._next_file_sample_idx = current_idx + max(1, self._file_step)
                seq_src = self._file_subsample_buf
            else:
                self.buf.append(raw_pose)
                if len(self.buf) > T_FRAMES * 2:
                    self.buf = self.buf[-T_FRAMES * 2 :]
                seq_src = self.buf
            if len(seq_src) >= MIN_SKELETON_FRAMES_FOR_ANALYSIS:
                ready_bcde = True
                seq = normalize_skeleton_sequence(np.stack(seq_src, axis=0))
                analysis = self.analyzer.full_analysis(seq, self._active_video_path)
                self._last_analysis = analysis
                self._llm_seq = seq.copy()
                fault_joints = analysis.fault_joints
                jlab = fault_joint_short_labels(analysis.faults)
                gcam = analysis.gradcam_joints
            else:
                jlab = {}
                gcam = None
            draw_pose_with_issues(
                display,
                k,
                s,
                fault_joints,
                joint_labels=jlab if jlab else None,
                joint_importance=gcam,
            )
        elif analyze_pose and self.pose_infer is not None:
            k, s = self._infer_pose_tracked(frame)
            draw_pose_with_issues(display, k, s, set())
        else:
            pass

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        fh, fw = rgb.shape[0], rgb.shape[1]
        tw = max(320, self._video_target_w - 6)
        th = max(240, self._video_target_h - 6)
        scale = min(tw / fw, th / fh)
        nw = max(1, int(round(fw * scale)))
        nh = max(1, int(round(fh * scale)))
        if (nw, nh) != (fw, fh):
            im = im.resize((nw, nh), Image.Resampling.BILINEAR)
        photo = ImageTk.PhotoImage(image=im)
        self.video_label.configure(image=photo)
        self.video_label.image = photo

        if analysis is not None:
            self.lbl_class.config(text=analysis.exercise)
            self.lbl_conf.config(text=f"Уверенность (B): {analysis.exercise_conf:.2f}")
            if analysis.used_rule_blend:
                ck_hint = classifier_weights_path()
                blend_txt = (
                    f"Класс (B): эвристика по позе — нужен checkpoint ({ck_hint.name}, см. training_meta.json)"
                    if self.pipeline is None
                    else "К этапу B добавлены эвристики (низкая уверенность сети)"
                )
            else:
                blend_txt = ""
            self.lbl_blend.config(text=blend_txt)
            p1, p2 = format_phase_for_ui(analysis.phase, analysis.phase_detail)
            self.lbl_phase.config(text=p1)
            self.lbl_phase_d.config(text=p2)
            self.lbl_reps.config(
                text=f"Повторов в окне (оценка): {analysis.rep_estimate}"
            )
            aux_parts: list[str] = []
            if analysis.nn_phase_id >= 0:
                aux_parts.append(
                    f"Сеть — фаза: {analysis.nn_phase_label} ({analysis.nn_phase_conf:.2f})"
                )
            if analysis.nn_error_prob > 1e-5:
                aux_parts.append(f"риск ошибки (сеть): {analysis.nn_error_prob:.2f}")
            if (analysis.nn_fault_label or "").strip():
                aux_parts.append(
                    f"тип ошибки (сеть): {analysis.nn_fault_label} ({analysis.nn_fault_conf:.2f})"
                )
            self.lbl_aux.config(text=" · ".join(aux_parts))
            self._set_faults_text(analysis, ready_bcde)
            self.txt_hint.config(state=tk.NORMAL)
            self.txt_hint.delete("1.0", tk.END)
            self.txt_hint.insert(tk.END, analysis.hint_general)
            self.txt_hint.config(state=tk.DISABLED)

            if self.var_log.get() and self.session_id is not None and analysis.exercise != "—":
                now = time.time()
                if analysis.exercise != self._last_log_name or now - self._last_log_t >= 2.0:
                    notes = f"{analysis.phase} | " + " | ".join(f.message for f in analysis.faults)[:400]
                    conn = connect(SESSION_DB)
                    log_event(conn, self.session_id, analysis.exercise, analysis.exercise_conf, notes=notes)
                    conn.close()
                    self._last_log_t = now
                    self._last_log_name = analysis.exercise
        else:
            self.lbl_class.config(text="—" if not ready_bcde else "…")
            self.lbl_conf.config(text="")
            self.lbl_blend.config(text="")
            if not ready_bcde:
                self.lbl_phase.config(
                    text="Сейчас на видео — собираем кадры…",
                )
                self.lbl_phase_d.config(
                    text="Как только накопится окно скелета, здесь появится фаза движения.",
                )
            else:
                self.lbl_phase.config(text="Сейчас на видео — …")
                self.lbl_phase_d.config(text="")
            self.lbl_reps.config(text="")
            self.lbl_aux.config(text="")
            self._set_faults_text(None, ready_bcde)
            self.txt_hint.config(state=tk.NORMAL)
            self.txt_hint.delete("1.0", tk.END)
            if self.pose_infer is not None and self.analyzer is None:
                self.txt_hint.insert(
                    tk.END,
                    "Нет списка классов: добавьте упражнения в data/raw или файл models/training_meta.json.",
                )
            self.txt_hint.config(state=tk.DISABLED)

    def on_close(self) -> None:
        self.stop_stream()
        if self.session_id is not None:
            conn = connect(SESSION_DB)
            end_session(conn, self.session_id)
            conn.close()
        self.destroy()


def main() -> None:
    conn = connect(SESSION_DB)
    init_schema(conn)
    app = App()
    app.session_id = start_session(conn)
    conn.close()
    app.mainloop()


if __name__ == "__main__":
    main()
