from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .prototypes import normalize


@dataclass(frozen=True)
class FaceRankResult:
    top1_person_id: int | None
    top1_score: float | None
    top2_person_id: int | None
    top2_score: float | None
    margin: float | None
    accepted: bool
    reason: str

    @property
    def top1_label(self) -> str:
        return f"P{self.top1_person_id:03d}" if self.top1_person_id is not None else ""

    @property
    def top2_label(self) -> str:
        return f"P{self.top2_person_id:03d}" if self.top2_person_id is not None else ""


class AnonymousFaceRecognizer:
    """Open-set anonymous face recognizer over Pxxx prototype banks.

    The recognizer is intentionally stateless. PersonMemory owns temporal voting,
    segment binding and prototype learning. This class only ranks a query face
    against existing face-anchored person profiles and exposes top1/top2/margin
    diagnostics so the video can prove when AdaFace actually influenced identity.
    """

    def __init__(self, threshold: float = 0.56, margin: float = 0.04) -> None:
        self.threshold = float(threshold)
        self.margin_threshold = float(margin)

    def rank(self, embedding: np.ndarray, banks: Mapping[int, Sequence[np.ndarray]]) -> FaceRankResult:
        q = normalize(np.asarray(embedding, dtype=np.float32))
        scored: list[tuple[float, int]] = []
        for pid, vectors in banks.items():
            best = None
            for vec in vectors:
                v = normalize(np.asarray(vec, dtype=np.float32))
                score = float(q @ v)
                best = score if best is None else max(best, score)
            if best is not None:
                scored.append((best, int(pid)))
        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            return FaceRankResult(None, None, None, None, None, False, "no_face_profiles")

        top1_score, top1_pid = scored[0]
        if len(scored) > 1:
            top2_score, top2_pid = scored[1]
            margin = float(top1_score - top2_score)
        else:
            top2_score, top2_pid = None, None
            margin = None

        if top1_score < self.threshold:
            return FaceRankResult(top1_pid, top1_score, top2_pid, top2_score, margin, False, "below_threshold")
        if margin is not None and margin < self.margin_threshold:
            return FaceRankResult(top1_pid, top1_score, top2_pid, top2_score, margin, False, "low_margin")
        return FaceRankResult(top1_pid, top1_score, top2_pid, top2_score, margin, True, "accepted")
