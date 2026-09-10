"""Minimal SCRFD ONNX runtime wrapper.

The decoding follows the public InsightFace SCRFD implementation. This file is a
small inference-only adaptation so the project does not need to install the full
InsightFace Python package.

Upstream code: https://github.com/deepinsight/insightface
Upstream code license: MIT. Pretrained InsightFace model weights have separate
usage terms; see the project root LICENSE_NOTES.md.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    preds: list[np.ndarray] = []
    for i in range(0, distance.shape[1], 2):
        preds.append(points[:, 0] + distance[:, i])
        preds.append(points[:, 1] + distance[:, i + 1])
    return np.stack(preds, axis=-1)


class SCRFD:
    def __init__(self, model_file: str | Path, device: str = "cuda:0", input_size=(640, 640), nms_thresh=0.4):
        import onnxruntime as ort

        self.model_file = str(model_file)
        if not Path(self.model_file).is_file():
            raise FileNotFoundError(self.model_file)

        # ORT 1.19+ can reuse CUDA/cuDNN DLLs installed with PyTorch on Windows.
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:
                pass

        if device.lower() == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            device_id = int(device.split(":")[-1]) if ":" in device else int(device)
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    "CUDAExecutionProvider is unavailable in ONNX Runtime. "
                    "Run check_gpu.py and make sure onnxruntime-gpu is installed."
                )
            providers = [("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]

        self.session = ort.InferenceSession(self.model_file, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_mean = 127.5
        self.input_std = 128.0
        self.input_size = tuple(input_size)
        self.nms_thresh = float(nms_thresh)
        self.center_cache: dict[tuple[int, int, int, int], np.ndarray] = {}

        outputs = self.session.get_outputs()
        self.batched = len(outputs[0].shape) == 3
        self.use_kps = False
        self.num_anchors = 1
        if len(outputs) == 6:
            self.fmc = 3
            self.feat_strides = [8, 16, 32]
            self.num_anchors = 2
        elif len(outputs) == 9:
            self.fmc = 3
            self.feat_strides = [8, 16, 32]
            self.num_anchors = 2
            self.use_kps = True
        elif len(outputs) == 10:
            self.fmc = 5
            self.feat_strides = [8, 16, 32, 64, 128]
        elif len(outputs) == 15:
            self.fmc = 5
            self.feat_strides = [8, 16, 32, 64, 128]
            self.use_kps = True
        else:
            raise RuntimeError(f"Unsupported SCRFD output count: {len(outputs)}")

        if not self.use_kps:
            raise RuntimeError(
                "This project requires a SCRFD model with 5-point landmarks. "
                "Use det_2.5g.onnx from InsightFace buffalo_m."
            )

    def _forward(self, image: np.ndarray, threshold: float):
        blob = cv2.dnn.blobFromImage(
            image,
            1.0 / self.input_std,
            tuple(image.shape[:2][::-1]),
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )
        outs = self.session.run(self.output_names, {self.input_name: blob})
        h, w = blob.shape[2], blob.shape[3]
        scores_list, bboxes_list, kpss_list = [], [], []

        for idx, stride in enumerate(self.feat_strides):
            if self.batched:
                scores = outs[idx][0]
                bbox_preds = outs[idx + self.fmc][0] * stride
                kps_preds = outs[idx + self.fmc * 2][0] * stride
            else:
                scores = outs[idx]
                bbox_preds = outs[idx + self.fmc] * stride
                kps_preds = outs[idx + self.fmc * 2] * stride

            height, width = h // stride, w // stride
            key = (height, width, stride, self.num_anchors)
            anchor_centers = self.center_cache.get(key)
            if anchor_centers is None:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self.num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self.num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            scores = np.asarray(scores).reshape(-1)
            pos = np.where(scores >= threshold)[0]
            if pos.size == 0:
                continue
            bboxes = distance2bbox(anchor_centers, np.asarray(bbox_preds))
            kpss = distance2kps(anchor_centers, np.asarray(kps_preds)).reshape((-1, 5, 2))
            scores_list.append(scores[pos][:, None])
            bboxes_list.append(bboxes[pos])
            kpss_list.append(kpss[pos])

        return scores_list, bboxes_list, kpss_list

    @staticmethod
    def _nms(dets: np.ndarray, threshold: float) -> list[int]:
        x1, y1, x2, y2, scores = [dets[:, i] for i in range(5)]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep: list[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            ww = np.maximum(0.0, xx2 - xx1 + 1)
            hh = np.maximum(0.0, yy2 - yy1 + 1)
            inter = ww * hh
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]
        return keep

    def detect(self, image: np.ndarray, threshold: float = 0.5, max_num: int = 0):
        input_w, input_h = self.input_size
        im_ratio = float(image.shape[0]) / max(image.shape[1], 1)
        model_ratio = float(input_h) / input_w
        if im_ratio > model_ratio:
            new_h = input_h
            new_w = int(new_h / im_ratio)
        else:
            new_w = input_w
            new_h = int(new_w * im_ratio)
        det_scale = float(new_h) / image.shape[0]
        resized = cv2.resize(image, (new_w, new_h))
        canvas = np.zeros((input_h, input_w, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized

        scores_list, bboxes_list, kpss_list = self._forward(canvas, threshold)
        if not scores_list:
            return np.empty((0, 5), np.float32), np.empty((0, 5, 2), np.float32)

        scores = np.vstack(scores_list).reshape(-1)
        bboxes = np.vstack(bboxes_list) / det_scale
        kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores[:, None])).astype(np.float32)
        order = pre_det[:, 4].argsort()[::-1]
        pre_det = pre_det[order]
        kpss = kpss[order]
        keep = self._nms(pre_det, self.nms_thresh)
        det = pre_det[keep]
        kpss = kpss[keep]

        if max_num > 0 and len(det) > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            idx = np.argsort(area)[::-1][:max_num]
            det, kpss = det[idx], kpss[idx]
        return det, kpss
