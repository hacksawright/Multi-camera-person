from __future__ import annotations

from dataclasses import dataclass, field

from .gallery import GalleryHit
from .models import IdentityEvent


@dataclass
class TrackIdentityState:
    state: str = "UNIDENTIFIED"
    employee_id: str | None = None
    candidate_id: str | None = None
    candidate_votes: int = 0
    candidate_weight: float = 0.0
    conflict_streak: int = 0
    conflict_candidate_id: str | None = None
    conflict_candidate_votes: int = 0
    conflict_candidate_weight: float = 0.0
    events: list[IdentityEvent] = field(default_factory=list)


class IdentityManager:
    """Per-local-track open-set identity state machine.

    A confirmed identity is intentionally sticky. A few contradictory frames
    are logged as conflicts but cannot silently rename a track.
    """

    def __init__(
        self,
        min_confirmations: int = 3,
        min_confirmation_weight: float = 1.45,
        min_support_quality: float = 0.32,
        strong_quality: float = 0.68,
    ) -> None:
        self.min_confirmations = int(min_confirmations)
        self.min_confirmation_weight = float(min_confirmation_weight)
        self.min_support_quality = float(min_support_quality)
        self.strong_quality = float(strong_quality)
        self.states: dict[int, TrackIdentityState] = {}

    def ensure(self, track_id: int, frame: int) -> TrackIdentityState:
        if track_id not in self.states:
            state = TrackIdentityState()
            state.events.append(IdentityEvent(frame, "UNIDENTIFIED", None, None, None, None, "track_created"))
            self.states[track_id] = state
        return self.states[track_id]

    def update(self, track_id: int, frame: int, hit: GalleryHit | None, quality: float, tier: str) -> TrackIdentityState:
        st = self.ensure(track_id, frame)
        if hit is None or tier == "reject" or quality < self.min_support_quality:
            return st

        if st.state == "CONFIRMED":
            if hit.accepted and hit.name == st.employee_id:
                st.conflict_streak = 0
                # Do not spam an event every frame; confirmation is sticky.
                return st
            if hit.accepted and hit.name != st.employee_id:
                st.conflict_streak += 1
                if st.conflict_candidate_id == hit.name:
                    st.conflict_candidate_votes += 1
                    st.conflict_candidate_weight += max(float(quality), 0.05)
                else:
                    st.conflict_candidate_id = hit.name
                    st.conflict_candidate_votes = 1
                    st.conflict_candidate_weight = max(float(quality), 0.05)
                enough_switch = (
                    st.conflict_candidate_votes >= self.min_confirmations
                    and st.conflict_candidate_weight >= self.min_confirmation_weight
                )
                if enough_switch:
                    old = st.employee_id
                    st.employee_id = st.conflict_candidate_id
                    st.candidate_id = st.employee_id
                    st.state = "CONFIRMED"
                    st.events.append(IdentityEvent(
                        frame, "REASSIGNED", st.employee_id, hit.score, hit.margin, quality,
                        f"multi_frame_conflict_reassigned:{old}",
                    ))
                    st.conflict_candidate_id = None
                    st.conflict_candidate_votes = 0
                    st.conflict_candidate_weight = 0.0
                else:
                    st.events.append(IdentityEvent(frame, "CONFLICT", st.employee_id, hit.score, hit.margin, quality,
                                                   f"candidate:{hit.name}"))
            return st

        if not hit.accepted:
            return st

        weight = max(float(quality), 0.05)
        if st.candidate_id != hit.name:
            st.candidate_id = hit.name
            st.candidate_votes = 1
            st.candidate_weight = weight
            st.state = "CANDIDATE"
            st.events.append(IdentityEvent(frame, "CANDIDATE", hit.name, hit.score, hit.margin, quality, "new_candidate"))
        else:
            st.candidate_votes += 1
            st.candidate_weight += weight

        enough = st.candidate_votes >= self.min_confirmations and st.candidate_weight >= self.min_confirmation_weight
        strong_fast_path = (
            tier == "strong"
            and quality >= self.strong_quality
            and st.candidate_votes >= 2
            and st.candidate_weight >= 1.35
        )
        if enough or strong_fast_path:
            st.state = "CONFIRMED"
            st.employee_id = st.candidate_id
            st.events.append(IdentityEvent(frame, "CONFIRMED", st.employee_id, hit.score, hit.margin, quality,
                                           "multi_frame_face_confirmed"))
        return st

    def final(self, track_id: int) -> TrackIdentityState:
        return self.states.get(track_id, TrackIdentityState())
