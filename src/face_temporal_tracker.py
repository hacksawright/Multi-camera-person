from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FaceTemporalSnapshot:
    state: str
    bbox: np.ndarray | None
    kps: np.ndarray | None
    score: float
    hits: int
    misses: int
    last_detection_frame: int | None

    @property
    def visible(self) -> bool:
        return self.state in {"VISIBLE", "COASTING"} and self.bbox is not None


class FaceTemporalTracker:
    """Small per-person face tracker used to remove detector flicker.

    SCRFD remains the source of *observations*. Between detector calls we only
    propagate the last face geometry with the motion of the enclosing person
    box. Predicted/coasting faces are never used as new recognition evidence.
    """

    def __init__(
        self,
        min_hits: int = 2,
        max_misses: int = 5,
        ema_alpha: float = 0.40,
    ) -> None:
        self.min_hits = max(int(min_hits), 1)
        self.max_misses = max(int(max_misses), 0)
        self.ema_alpha = float(np.clip(ema_alpha, 0.05, 1.0))

        self._bbox: np.ndarray | None = None
        self._kps: np.ndarray | None = None
        self._score = 0.0
        self._hits = 0
        self._misses = 0
        self._state = "ABSENT"
        self._last_detection_frame: int | None = None
        self._person_box: np.ndarray | None = None

    @staticmethod
    def _center_size(box: np.ndarray) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = [float(x) for x in box]
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2), max(x2 - x1, 1.0), max(y2 - y1, 1.0))

    def update_person(self, person_box: np.ndarray) -> None:
        """Propagate face geometry using enclosing person-box motion."""
        new_box = np.asarray(person_box, dtype=np.float32).reshape(4)
        if self._person_box is not None and self._bbox is not None:
            ocx, ocy, ow, oh = self._center_size(self._person_box)
            ncx, ncy, nw, nh = self._center_size(new_box)
            sx = float(np.clip(nw / ow, 0.85, 1.15))
            sy = float(np.clip(nh / oh, 0.85, 1.15))

            def transform(points: np.ndarray) -> np.ndarray:
                p = np.asarray(points, dtype=np.float32).copy()
                p[..., 0] = ncx + (p[..., 0] - ocx) * sx
                p[..., 1] = ncy + (p[..., 1] - ocy) * sy
                return p

            corners = np.asarray([
                [self._bbox[0], self._bbox[1]],
                [self._bbox[2], self._bbox[3]],
            ], dtype=np.float32)
            corners = transform(corners)
            self._bbox = np.asarray([
                corners[0, 0], corners[0, 1], corners[1, 0], corners[1, 1]
            ], dtype=np.float32)
            if self._kps is not None:
                self._kps = transform(self._kps)
        self._person_box = new_box

    def observe(self, bbox: np.ndarray, kps: np.ndarray, score: float, frame: int) -> FaceTemporalSnapshot:
        bbox = np.asarray(bbox, dtype=np.float32).reshape(4)
        kps = np.asarray(kps, dtype=np.float32).reshape(5, 2)
        if self._bbox is None:
            self._bbox = bbox.copy()
            self._kps = kps.copy()
        else:
            a = self.ema_alpha
            self._bbox = ((1.0 - a) * self._bbox + a * bbox).astype(np.float32)
            if self._kps is None:
                self._kps = kps.copy()
            else:
                self._kps = ((1.0 - a) * self._kps + a * kps).astype(np.float32)

        self._score = float(score)
        self._hits += 1
        self._misses = 0
        self._last_detection_frame = int(frame)
        self._state = "VISIBLE" if self._hits >= self.min_hits else "TENTATIVE"
        return self.snapshot()

    def miss(self) -> FaceTemporalSnapshot:
        if self._bbox is None:
            self._state = "ABSENT"
            return self.snapshot()
        self._misses += 1
        if self._hits < self.min_hits:
            self._state = "ABSENT"
            self._bbox = None
            self._kps = None
            self._hits = 0
            self._score = 0.0
        elif self._misses <= self.max_misses:
            self._state = "COASTING"
        else:
            self._state = "LOST"
            self._bbox = None
            self._kps = None
            self._hits = 0
            self._misses = 0
            self._score = 0.0
        return self.snapshot()

    def snapshot(self) -> FaceTemporalSnapshot:
        return FaceTemporalSnapshot(
            state=self._state,
            bbox=None if self._bbox is None else self._bbox.copy(),
            kps=None if self._kps is None else self._kps.copy(),
            score=float(self._score),
            hits=int(self._hits),
            misses=int(self._misses),
            last_detection_frame=self._last_detection_frame,
        )
