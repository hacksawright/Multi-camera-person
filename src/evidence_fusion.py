from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class IdentityEvidence:
    identity_id: int
    face_scores: list[float] = field(default_factory=list)
    body_scores: list[float] = field(default_factory=list)
    motion_scores: list[float] = field(default_factory=list)
    face_support: int = 0
    body_support: int = 0

    @property
    def face_score(self) -> float | None:
        return max(self.face_scores) if self.face_scores else None

    @property
    def body_score(self) -> float | None:
        return max(self.body_scores) if self.body_scores else None

    @property
    def motion_score(self) -> float | None:
        return max(self.motion_scores) if self.motion_scores else None


@dataclass(frozen=True)
class FusionDecision:
    identity_id: int | None
    state: str
    reason: str
    face_score: float | None
    body_score: float | None
    motion_score: float | None
    fusion_score: float | None
    margin: float | None
    face_support: int
    body_support: int


class TrackletEvidenceFusion:
    """Single policy for combining face, body and motion evidence."""

    def __init__(
        self,
        *,
        face_threshold: float = .56,
        weak_face_threshold: float = .45,
        body_threshold: float = .84,
        margin: float = .04,
        weak_face_weight: float = .35,
        body_weight: float = .55,
        face_weight: float = .70,
    ) -> None:
        self.face_threshold = float(face_threshold)
        self.weak_face_threshold = float(weak_face_threshold)
        self.body_threshold = float(body_threshold)
        self.margin_threshold = float(margin)
        self.weak_face_weight = float(weak_face_weight)
        self.body_weight = float(body_weight)
        self.face_weight = float(face_weight)

    def decide(self, evidence: Iterable[IdentityEvidence], current_id: int | None = None) -> FusionDecision:
        candidates = list(evidence)
        if not candidates:
            return FusionDecision(None, "UNKNOWN", "NO_EVIDENCE", None, None, None, None, None, 0, 0)

        scored: list[tuple[float, IdentityEvidence, str]] = []
        for item in candidates:
            face = item.face_score
            body = item.body_score
            motion = item.motion_score
            face_strong = face is not None and face >= self.face_threshold and item.face_support >= 2
            face_weak = face is not None and face >= self.weak_face_threshold and item.face_support >= 3
            body_strong = body is not None and body >= self.body_threshold and item.body_support >= 3
            if face_strong and body_strong:
                score = self.face_weight * face + self.body_weight * body
                reason = "FACE_BODY"
            elif face_strong:
                score = self.face_weight * face
                reason = "FACE_ONLY"
            elif body_strong:
                score = self.body_weight * body
                reason = "BODY_ONLY_PROVISIONAL"
            elif face_weak and body is not None:
                score = self.weak_face_weight * face + self.body_weight * body
                reason = "WEAK_FACE_BODY"
            else:
                continue
            if motion is not None:
                score += .15 * motion
            scored.append((float(score), item, reason))

        if not scored:
            return FusionDecision(None, "UNKNOWN", "INSUFFICIENT_EVIDENCE", None, None, None, None, None, 0, 0)
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best, reason = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else None
        margin = best_score - second_score if second_score is not None else None
        if margin is not None and margin < self.margin_threshold:
            return FusionDecision(
                current_id, "CONFLICT", "LOW_MARGIN", best.face_score, best.body_score,
                best.motion_score, best_score, margin, best.face_support, best.body_support,
            )
        state = "CONFIRMED" if reason in {"FACE_BODY", "FACE_ONLY"} else "PROVISIONAL"
        return FusionDecision(
            best.identity_id, state, reason, best.face_score, best.body_score,
            best.motion_score, best_score, margin, best.face_support, best.body_support,
        )
