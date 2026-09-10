from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

TrackKey = tuple[str, int]


@dataclass(frozen=True)
class TrackSpan:
    first: int
    last: int


@dataclass
class FaceObservation:
    frame: int
    embedding: np.ndarray
    quality: float
    tier: str
    detector_score: float
    face_size: float
    blur: float
    feature_norm: float
    pose: str
    yaw_proxy: float
    pitch_proxy: float
    roll_deg: float


@dataclass
class FacePrototype:
    embedding: np.ndarray
    quality: float
    pose: str
    count: int


@dataclass
class IdentityEvent:
    frame: int
    state: str
    employee_id: Optional[str]
    score: Optional[float]
    margin: Optional[float]
    quality: Optional[float]
    reason: str


@dataclass
class TrackSummary:
    key: TrackKey
    span: TrackSpan
    face_prototypes: list[FacePrototype] = field(default_factory=list)
    body_prototypes: list[np.ndarray] = field(default_factory=list)
    confirmed_employee: Optional[str] = None
    final_state: str = "UNIDENTIFIED"
    events: list[IdentityEvent] = field(default_factory=list)
