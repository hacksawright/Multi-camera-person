from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .scrfd_runtime import SCRFD


@dataclass
class FaceDetection:
    bbox: np.ndarray
    score: float
    kps: np.ndarray


class FaceDetector:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        input_size: int = 640,
        threshold: float = 0.50,
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"SCRFD model not found: {path}. Run setup.ps1 first."
            )
        self.threshold = float(threshold)
        self.detector = SCRFD(path, device=device, input_size=(input_size, input_size))

    def detect(self, frame_bgr: np.ndarray) -> list[FaceDetection]:
        det, kps = self.detector.detect(frame_bgr, threshold=self.threshold)
        out: list[FaceDetection] = []
        for d, p in zip(det, kps):
            out.append(FaceDetection(bbox=d[:4].astype(np.float32), score=float(d[4]), kps=p.astype(np.float32)))
        return out
