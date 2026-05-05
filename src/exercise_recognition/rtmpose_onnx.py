"""
RTMPose-m через ONNX Runtime.
Препроцессинг и декодирование SimCC по примеру MMPose
projects/rtmpose/examples/onnxruntime/main.py
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from .config import MODEL_INPUT_SIZE, SIMCC_SPLIT_RATIO


def bbox_xyxy2cs(
    bbox: np.ndarray, padding: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]
    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding
    if dim == 1:
        center = center[0]
        scale = scale[0]
    return center, scale


def _fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float) -> np.ndarray:
    w, h = np.hsplit(bbox_scale, [1])
    return np.where(
        w > h * aspect_ratio,
        np.hstack([w, w / aspect_ratio]),
        np.hstack([h * aspect_ratio, h]),
    )


def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ pt


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direction = a - b
    return b + np.r_[-direction[1], direction[0]]


def get_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size: Tuple[int, int],
    shift: Tuple[float, float] = (0.0, 0.0),
    inv: bool = False,
) -> np.ndarray:
    shift = np.array(shift)
    src_w = scale[0]
    dst_w, dst_h = output_size
    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0.0, src_w * -0.5]), rot_rad)
    dst_dir = np.array([0.0, dst_w * -0.5])
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])
    if inv:
        return cv2.getAffineTransform(np.float32(dst), np.float32(src))
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def top_down_affine(
    input_size: Tuple[int, int],
    bbox_scale: np.ndarray,
    bbox_center: np.ndarray,
    img: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    w, h = input_size
    warp_size = (int(w), int(h))
    bbox_scale = _fix_aspect_ratio(bbox_scale, aspect_ratio=w / h)
    warp_mat = get_warp_matrix(bbox_center, bbox_scale, 0, output_size=(w, h))
    img = cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)
    return img, bbox_scale


def preprocess_image(
    img: np.ndarray, input_size: Tuple[int, int] | None = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_size = input_size or MODEL_INPUT_SIZE
    img_shape = img.shape[:2]
    bbox = np.array([0, 0, img_shape[1], img_shape[0]], dtype=np.float32)
    center, scale = bbox_xyxy2cs(bbox, padding=1.25)
    resized_img, scale = top_down_affine(input_size, scale, center, img)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    resized_img = (resized_img.astype(np.float32) - mean) / std
    return resized_img, center, scale


def get_simcc_maximum(
    simcc_x: np.ndarray, simcc_y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    N, K, Wx = simcc_x.shape
    simcc_x = simcc_x.reshape(N * K, -1)
    simcc_y = simcc_y.reshape(N * K, -1)
    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)
    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x
    locs[vals <= 0.0] = -1
    locs = locs.reshape(N, K, 2)
    vals = vals.reshape(N, K)
    return locs, vals


def decode_simcc(
    simcc_x: np.ndarray, simcc_y: np.ndarray, simcc_split_ratio: float
) -> Tuple[np.ndarray, np.ndarray]:
    if simcc_x.ndim == 2:
        simcc_x = simcc_x[None, ...]
        simcc_y = simcc_y[None, ...]
    keypoints, scores = get_simcc_maximum(simcc_x, simcc_y)
    keypoints = keypoints / simcc_split_ratio
    return keypoints, scores


def postprocess_outputs(
    outputs: List[np.ndarray],
    model_input_size: Tuple[int, int],
    center: np.ndarray,
    scale: np.ndarray,
    simcc_split_ratio: float = SIMCC_SPLIT_RATIO,
) -> Tuple[np.ndarray, np.ndarray]:
    simcc_x, simcc_y = outputs[0], outputs[1]
    keypoints, scores = decode_simcc(simcc_x, simcc_y, simcc_split_ratio)
    w, h = float(model_input_size[0]), float(model_input_size[1])
    wh = np.array([w, h], dtype=np.float32)
    keypoints = keypoints / wh * scale + center - scale / 2
    return keypoints[0], scores[0]


class RTMPoseONNX:
    def __init__(self, onnx_path: str, device: str = "cpu"):
        providers = (
            ["CPUExecutionProvider"]
            if device == "cpu"
            else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        inp = self.session.get_inputs()[0]
        shape = inp.shape
        if len(shape) >= 4 and shape[2] and shape[3]:
            self._h, self._w = int(shape[2]), int(shape[3])
        else:
            self._h, self._w = MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0]
        self.model_input_size = (self._w, self._h)
        self._out_names = [o.name for o in self.session.get_outputs()]

    def infer(
        self,
        bgr: np.ndarray,
        roi_xyxy: tuple[float, float, float, float] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Кадр BGR → ключевые точки (K,2), скоры (K,).
        Если задан roi_xyxy=(x1,y1,x2,y2), поза считается по вырезке (онлайн-трекинг без полного кадра).
        """
        if roi_xyxy is None:
            resized, center, scale = preprocess_image(bgr, self.model_input_size)
            chw = resized.transpose(2, 0, 1)[None, ...].astype(np.float32)
            feeds = {self.session.get_inputs()[0].name: chw}
            outputs = self.session.run(self._out_names, feeds)
            kpts, sc = postprocess_outputs(
                outputs, self.model_input_size, center, scale, SIMCC_SPLIT_RATIO
            )
            return kpts, sc

        h, w = bgr.shape[:2]
        x1, y1, x2, y2 = roi_xyxy
        x1i = int(max(0, min(x1, w - 2)))
        y1i = int(max(0, min(y1, h - 2)))
        x2i = int(min(w, max(x2, x1i + 2)))
        y2i = int(min(h, max(y2, y1i + 2)))
        roi = bgr[y1i:y2i, x1i:x2i]
        if roi.size == 0 or roi.shape[0] < 8 or roi.shape[1] < 8:
            return self.infer(bgr, roi_xyxy=None)
        resized, center, scale = preprocess_image(roi, self.model_input_size)
        chw = resized.transpose(2, 0, 1)[None, ...].astype(np.float32)
        feeds = {self.session.get_inputs()[0].name: chw}
        outputs = self.session.run(self._out_names, feeds)
        kpts, sc = postprocess_outputs(
            outputs, self.model_input_size, center, scale, SIMCC_SPLIT_RATIO
        )
        kpts = kpts + np.array([float(x1i), float(y1i)], dtype=np.float32)
        return kpts, sc
