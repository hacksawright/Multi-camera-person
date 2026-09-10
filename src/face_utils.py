from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

# Standard 5-point ArcFace/AdaFace template for 112x112 aligned faces.
_ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


@dataclass
class FaceSample:
    embedding: np.ndarray
    quality: float
    detector_score: float
    face_size: float
    blur: float


def align_face_bgr(frame_bgr: np.ndarray, kps: np.ndarray, output_size: int = 112) -> np.ndarray | None:
    src = np.asarray(kps, dtype=np.float32).reshape(5, 2)
    dst = _ARCFACE_DST.copy()
    if output_size != 112:
        dst *= float(output_size) / 112.0

    # LMEDS is stable for the five facial landmarks and avoids a dependency on skimage.
    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if matrix is None:
        return None
    aligned = cv2.warpAffine(
        frame_bgr,
        matrix,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return aligned


def laplacian_blur_score(face_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def face_quality(detector_score: float, face_size: float, blur: float) -> float:
    """Simple quality proxy used only for temporal aggregation.

    It intentionally does not fabricate face details; it prefers detections that are
    larger, sharper and higher-confidence.
    """
    size_factor = min(max(face_size / 80.0, 0.0), 1.0)
    blur_factor = min(max(blur / 150.0, 0.0), 1.0)
    return float(detector_score * np.sqrt(max(size_factor * blur_factor, 0.0)))


def aggregate_face_samples(samples: Iterable[FaceSample], top_k: int = 8) -> np.ndarray | None:
    ranked = sorted(samples, key=lambda s: s.quality, reverse=True)[:top_k]
    if not ranked:
        return None
    feats = np.stack([s.embedding for s in ranked]).astype(np.float32)
    weights = np.asarray([max(s.quality, 1e-4) for s in ranked], dtype=np.float32)
    weights /= weights.sum()
    out = np.sum(feats * weights[:, None], axis=0)
    norm = float(np.linalg.norm(out))
    if norm <= 1e-12:
        return None
    return (out / norm).astype(np.float32)
