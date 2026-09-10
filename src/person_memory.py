from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import json
import math

import cv2
import numpy as np

from .models import TrackKey
from .prototypes import normalize
from .anonymous_face_recognizer import AnonymousFaceRecognizer, FaceRankResult


TRUSTED_FACE_TIERS = {"support", "strong"}


@dataclass
class FaceMemoryItem:
    embedding: np.ndarray
    quality: float
    pose: str
    tier: str
    support: int = 1
    core: bool = False
    first_frame: int = 0
    last_frame: int = 0
    sample_file: str | None = None


@dataclass
class BodyMemoryItem:
    embedding: np.ndarray
    quality: float
    support: int = 1
    core: bool = False
    first_frame: int = 0
    last_frame: int = 0
    sample_file: str | None = None


@dataclass
class TrackSegment:
    start_frame: int
    end_frame: int | None
    person_id: int | None
    reason: str
    authority: str


@dataclass
class FaceAuthorityResult:
    frame: int
    person_id: int | None
    decision: str
    accepted: bool
    top1_person_id: int | None = None
    top1_score: float | None = None
    top2_person_id: int | None = None
    top2_score: float | None = None
    margin: float | None = None
    candidate_votes: int = 0
    candidate_needed: int = 0
    pose: str = ""
    tier: str = ""
    quality: float = 0.0

    def row_fields(self) -> dict:
        return {
            "face_id_frame": self.frame,
            "face_id_person": f"P{self.person_id:03d}" if self.person_id is not None else "",
            "face_id_decision": self.decision,
            "face_id_accepted": self.accepted,
            "face_id_top1": f"P{self.top1_person_id:03d}" if self.top1_person_id is not None else "",
            "face_id_top1_score": self.top1_score if self.top1_score is not None else "",
            "face_id_top2": f"P{self.top2_person_id:03d}" if self.top2_person_id is not None else "",
            "face_id_top2_score": self.top2_score if self.top2_score is not None else "",
            "face_id_margin": self.margin if self.margin is not None else "",
            "face_id_candidate_votes": self.candidate_votes,
            "face_id_candidate_needed": self.candidate_needed,
            "face_id_pose": self.pose,
            "face_id_tier": self.tier,
            "face_id_quality": self.quality,
        }


@dataclass
class TrackBinding:
    key: TrackKey
    first_frame: int
    last_frame: int
    person_id: int | None = None
    bind_reason: str | None = None
    safe_after_frame: int = -1
    ambiguous_frames: int = 0
    face_observations: int = 0
    body_observations: int = 0
    last_bbox: np.ndarray | None = None
    prev_bbox: np.ndarray | None = None
    prev_bbox_frame: int | None = None
    last_body: np.ndarray | None = None
    last_body_frame: int | None = None
    appearance_jump_until: int = -1
    strong_face_conflict: bool = False
    segments: list[TrackSegment] = field(default_factory=list)
    face_candidate_pid: int | None = None
    face_candidate_votes: int = 0
    face_candidate_last_frame: int = -1
    new_face_candidate: np.ndarray | None = None
    new_face_votes: int = 0
    new_face_last_frame: int = -1
    body_candidate_pid: int | None = None
    body_candidate_votes: int = 0
    body_candidate_last_frame: int = -1
    identity_change_until: int = -1
    identity_change_count: int = 0

    @property
    def age(self) -> int:
        return max(int(self.last_frame) - int(self.first_frame) + 1, 1)


@dataclass
class PersonProfile:
    person_id: int
    employee_id: str | None = None
    members: list[TrackKey] = field(default_factory=list)
    face_bank: list[FaceMemoryItem] = field(default_factory=list)
    body_bank: list[BodyMemoryItem] = field(default_factory=list)
    last_seen: dict[str, int] = field(default_factory=dict)
    last_bbox: dict[str, np.ndarray] = field(default_factory=dict)
    prev_bbox: dict[str, np.ndarray] = field(default_factory=dict)
    prev_bbox_frame: dict[str, int] = field(default_factory=dict)
    face_anchor_count: int = 0
    body_observations: int = 0
    face_observations: int = 0
    reacquire_count: int = 0
    conflict_count: int = 0
    recent_face_embedding: np.ndarray | None = None
    recent_face_quality: float = 0.0
    recent_face_pose: str = ""
    recent_face_frame: int = -1
    recent_face_camera: str = ""

    def face_core(self) -> list[FaceMemoryItem]:
        core = [x for x in self.face_bank if x.tier in TRUSTED_FACE_TIERS and x.core]
        if core:
            return core
        trusted = [x for x in self.face_bank if x.tier in TRUSTED_FACE_TIERS]
        repeated = [x for x in trusted if x.support >= 2]
        return repeated if repeated else trusted

    def body_core(self) -> list[BodyMemoryItem]:
        core = [x for x in self.body_bank if x.core]
        return core if core else self.body_bank

    def maturity(self) -> str:
        face = len(self.face_core())
        body = len(self.body_core())
        members = len(self.members)
        if self.employee_id:
            return "CONFIRMED"
        if self.face_anchor_count >= 2 and body >= 3:
            return "STABLE"
        if face >= 1 or body >= 2 or members >= 2:
            return "LEARNING"
        return "NEW"

    def confidence(self) -> float:
        face = min(len(self.face_core()) / 3.0, 1.0)
        body = min(len(self.body_core()) / 5.0, 1.0)
        members = min(len(self.members) / 3.0, 1.0)
        reacq = min(self.reacquire_count / 2.0, 1.0)
        emp = 1.0 if self.employee_id else 0.0
        return float(np.clip(0.34 * face + 0.24 * body + 0.16 * members + 0.16 * reacq + 0.10 * emp, 0.0, 1.0))


@dataclass
class PersonDecision:
    frame: int
    camera: str
    track_id: int
    person_id: int | None
    reason: str
    accepted: bool
    score: float | None = None
    margin: float | None = None
    face_score: float | None = None
    body_score: float | None = None
    motion_score: float | None = None


class RepresentativeStore:
    """Stores only representative face/person crops for demo and later enrollment.

    Files are intentionally small in count. Memory embeddings remain the source of
    matching truth; JPGs are human-inspectable examples for FACE/PERSON profiles.
    """

    def __init__(self, root: str | Path, face_per_pose: int = 3, body_slots: int = 8) -> None:
        self.root = Path(root)
        self.face_per_pose = max(int(face_per_pose), 1)
        self.body_slots = max(int(body_slots), 1)
        self.root.mkdir(parents=True, exist_ok=True)

    def _person_dir(self, pid: int) -> Path:
        p = self.root / f"P{pid:03d}"
        (p / "face").mkdir(parents=True, exist_ok=True)
        (p / "body").mkdir(parents=True, exist_ok=True)
        return p

    def save_face(self, pid: int, pose: str, slot: int, image: np.ndarray) -> str | None:
        if image is None or image.size == 0:
            return None
        p = self._person_dir(pid) / "face" / f"{pose}_{slot + 1:02d}.jpg"
        if cv2.imwrite(str(p), image):
            return str(p.relative_to(self.root.parent))
        return None

    def save_body(self, pid: int, slot: int, image: np.ndarray) -> str | None:
        if image is None or image.size == 0:
            return None
        p = self._person_dir(pid) / "body" / f"view_{slot + 1:02d}.jpg"
        if cv2.imwrite(str(p), image):
            return str(p.relative_to(self.root.parent))
        return None

    def write_metadata(self, profile: PersonProfile) -> None:
        p = self._person_dir(profile.person_id) / "metadata.json"
        payload = {
            "person_id": f"P{profile.person_id:03d}",
            "employee_id": profile.employee_id,
            "maturity": profile.maturity(),
            "confidence": profile.confidence(),
            "members": [{"camera": c, "track_id": int(t)} for c, t in profile.members],
            "face_samples": [
                {
                    "pose": x.pose,
                    "tier": x.tier,
                    "quality": x.quality,
                    "support": x.support,
                    "core": x.core,
                    "sample_file": x.sample_file,
                }
                for x in profile.face_bank
            ],
            "body_samples": [
                {
                    "quality": x.quality,
                    "support": x.support,
                    "core": x.core,
                    "sample_file": x.sample_file,
                }
                for x in profile.body_bank
            ],
        }
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class PersonMemory:
    """Face-first person memory.

    Invariants:
      1. Body/motion can never move an already-bound local track to another P.
         Trusted face authority may split the track into a new P segment without
         rewriting earlier history.
      2. Brief local-track breaks are handled by a recent-person lease first
         (motion/geometry, optionally body). No global body search is performed.
      3. Trusted face evidence is the only biometric cue allowed to identify a
         person across long gaps or to confirm EMPxxx.
      4. Body embeddings are learned as same-session appearance and are used only
         for short-gap continuity; they do not define permanent identity.
      5. Ambiguous/crossing frames pause learning to reduce memory contamination.
    """

    def __init__(
        self,
        *,
        sample_root: str | Path,
        short_gap_frames: int = 45,
        motion_only_gap_frames: int = 6,
        new_person_after_frames: int = 12,
        ambiguity_grace_frames: int = 8,
        face_match_threshold: float = 0.56,
        face_match_margin: float = 0.04,
        face_learn_quality: float = 0.32,
        face_match_confirmations: int = 2,
        face_new_confirmations: int = 2,
        face_candidate_max_gap: int = 16,
        face_recent_anchor_frames: int = 300,
        body_short_gap_near: float = 0.76,
        body_short_gap_far: float = 0.84,
        body_match_margin: float = 0.05,
        body_learn_quality: float = 0.35,
        body_novelty_threshold: float = 0.91,
        body_jump_threshold: float = 0.48,
        max_face_per_pose: int = 4,
        max_body_views: int = 8,
        require_face_before_person: bool = True,
    ) -> None:
        self.short_gap_frames = max(int(short_gap_frames), 1)
        self.motion_only_gap_frames = max(int(motion_only_gap_frames), 1)
        self.new_person_after_frames = max(int(new_person_after_frames), 1)
        self.ambiguity_grace_frames = max(int(ambiguity_grace_frames), 0)
        self.face_match_threshold = float(face_match_threshold)
        self.face_match_margin = float(face_match_margin)
        self.face_learn_quality = float(face_learn_quality)
        self.face_match_confirmations = max(int(face_match_confirmations), 1)
        self.face_new_confirmations = max(int(face_new_confirmations), 1)
        self.face_candidate_max_gap = max(int(face_candidate_max_gap), 1)
        self.face_recent_anchor_frames = max(int(face_recent_anchor_frames), 1)
        self.face_recognizer = AnonymousFaceRecognizer(self.face_match_threshold, self.face_match_margin)
        self.body_short_gap_near = float(body_short_gap_near)
        self.body_short_gap_far = float(body_short_gap_far)
        self.body_match_margin = float(body_match_margin)
        self.body_learn_quality = float(body_learn_quality)
        self.body_novelty_threshold = float(body_novelty_threshold)
        self.body_jump_threshold = float(body_jump_threshold)
        self.max_face_per_pose = max(int(max_face_per_pose), 1)
        self.max_body_views = max(int(max_body_views), 1)
        self.require_face_before_person = bool(require_face_before_person)

        self.profiles: dict[int, PersonProfile] = {}
        self.tracks: dict[TrackKey, TrackBinding] = {}
        self.decisions: list[PersonDecision] = []
        self.events: list[dict] = []
        self._next_person_id = 1
        self.store = RepresentativeStore(sample_root, face_per_pose=max_face_per_pose, body_slots=max_body_views)
        self._face_diagnostics: dict[TrackKey, FaceAuthorityResult] = {}

    # ------------------------------ public views ------------------------------
    def profile_for_track(self, key: TrackKey) -> PersonProfile | None:
        st = self.tracks.get(key)
        return self.profiles.get(st.person_id) if st and st.person_id is not None else None

    def state_for_track(self, key: TrackKey) -> TrackBinding | None:
        return self.tracks.get(key)

    def public_label(self, key: TrackKey) -> str:
        st = self.tracks.get(key)
        if st is None or st.person_id is None:
            return "P:---"
        return f"P:{st.person_id:03d}"

    def mapping(self) -> dict[TrackKey, int]:
        return {k: st.person_id for k, st in self.tracks.items() if st.person_id is not None}

    def face_diagnostic(self, key: TrackKey) -> FaceAuthorityResult | None:
        return self._face_diagnostics.get(key)

    def import_embedding_records(self, records: list[object]) -> None:
        """Restore reusable P profiles from a previous embedding store."""
        grouped: dict[int, list[object]] = defaultdict(list)
        for record in records:
            pid = getattr(record, "person_id", None)
            if pid is not None:
                grouped[int(pid)].append(record)
        for pid, items in grouped.items():
            profile = self.profiles.setdefault(pid, PersonProfile(pid))
            for item in items:
                kind = getattr(item, "kind", "")
                emb = normalize(np.asarray(item.embedding, dtype=np.float32))
                quality = float(getattr(item, "quality", 0.0))
                frame = int(getattr(item, "frame", 0))
                if kind == "face" and quality >= self.face_learn_quality:
                    self._add_face(profile, emb, quality, getattr(item, "pose", "front"), "support", frame)
                elif kind == "body" and quality >= self.body_learn_quality:
                    self._add_body(profile, emb, quality, frame)
            profile.face_anchor_count = sum(1 for x in profile.face_bank if x.tier in TRUSTED_FACE_TIERS)
        if self.profiles:
            self._next_person_id = max(self.profiles) + 1

    def reconcile_track_faces(self, key: TrackKey, observations: list[object]) -> int | None:
        """Re-evaluate a completed local track against all known face profiles.

        Online binding is provisional. This pass uses independent observations
        collected across the tracklet and may create a new segment when another
        profile wins consistently. Raw observations are never rewritten.
        """
        st = self.tracks.get(key)
        if st is None or not observations:
            return st.person_id if st is not None else None
        usable = [x for x in observations if getattr(x, "kind", "") == "face" and float(getattr(x, "quality", 0.0)) >= self.face_learn_quality]
        if len(usable) < 2:
            return st.person_id

        if len(self.profiles) < 2:
            return st.person_id

        votes: dict[int, list[float]] = defaultdict(list)
        current_pid = st.person_id
        for obs in usable:
            query = normalize(np.asarray(obs.embedding, dtype=np.float32))
            ranked: list[tuple[float, int]] = []
            for pid, profile in self.profiles.items():
                # Never let a provisional profile win against itself. A new
                # track must first be compared with existing profiles before
                # its own observations are promoted into a new anchor.
                if pid == current_pid and not profile.face_core():
                    continue
                vectors = self._face_bank_vectors(profile, key[0], int(obs.frame))
                if not vectors:
                    continue
                ranked.append((max(float(query @ normalize(v)) for v in vectors), pid))
            ranked.sort(reverse=True)
            if not ranked:
                continue
            best_score, best_pid = ranked[0]
            second = ranked[1][0] if len(ranked) > 1 else -1.0
            if best_score >= self.face_match_threshold and best_score - second >= self.face_match_margin:
                votes[best_pid].append(best_score)

        if not votes:
            # No prior identity matched. Now it is safe to promote this
            # provisional track's mutually consistent observations.
            if current_pid is not None:
                current = self.profiles[current_pid]
                if not current.face_core():
                    ranked = sorted(usable, key=lambda x: float(x.quality), reverse=True)
                    first = normalize(np.asarray(ranked[0].embedding, dtype=np.float32))
                    agreeing = [x for x in ranked[1:] if float(first @ normalize(x.embedding)) >= 0.72]
                    if agreeing:
                        self._learn_face_sample(
                            current, key, int(ranked[0].frame), first,
                            float(ranked[0].quality), getattr(ranked[0], "pose", "front"),
                            "support", None,
                        )
            return st.person_id
        target_pid, scores = max(votes.items(), key=lambda item: (len(item[1]), sum(item[1]) / len(item[1])))
        mean_score = sum(scores) / len(scores)
        if target_pid == st.person_id or len(scores) < 2:
            return st.person_id
        if mean_score < self.face_match_threshold or len(scores) / len(usable) < 0.5:
            return st.person_id
        target = self.profiles[target_pid]
        start = int(min(getattr(x, "frame", st.first_frame) for x in usable if target_pid in votes))
        self.replace_track_identity(st, target, "offline_face_reconciliation", score=mean_score)
        self.events.append({
            "frame": start, "camera": key[0], "track_id": key[1],
            "event": "offline_face_reconciled", "person_id": target_pid,
            "score": mean_score, "votes": len(scores),
        })
        return target_pid

    def person_id_at(self, key: TrackKey, frame: int) -> int | None:
        st = self.tracks.get(key)
        if st is None:
            return None
        for seg in reversed(st.segments):
            if int(frame) >= seg.start_frame and (seg.end_frame is None or int(frame) <= seg.end_frame):
                return seg.person_id
        return st.person_id

    # ------------------------------ track state -------------------------------
    def ensure_track(self, key: TrackKey, frame: int) -> TrackBinding:
        st = self.tracks.get(key)
        if st is None:
            st = TrackBinding(key=key, first_frame=int(frame), last_frame=int(frame))
            self.tracks[key] = st
            self.events.append({
                "frame": int(frame), "camera": key[0], "track_id": key[1],
                "event": "track_created", "person_id": None,
            })
        return st

    def touch_track(self, key: TrackKey, frame: int, bbox: np.ndarray, *, ambiguous: bool = False) -> TrackBinding:
        st = self.ensure_track(key, frame)
        st.last_frame = max(st.last_frame, int(frame))
        b = np.asarray(bbox, dtype=np.float32).reshape(4).copy()
        if st.last_bbox is not None:
            st.prev_bbox = st.last_bbox.copy()
            st.prev_bbox_frame = int(frame) - 1
        st.last_bbox = b
        if ambiguous:
            st.ambiguous_frames += 1
            st.safe_after_frame = max(st.safe_after_frame, int(frame) + self.ambiguity_grace_frames)
        if st.person_id is not None:
            self._update_profile_geometry(st, frame)
        else:
            # Crossing pauses LEARNING, not continuity. A short tracker blink is
            # exactly when the recent-person lease is most valuable.
            self._try_motion_only_lease(st, frame, ambiguous=ambiguous)
        return st

    def maybe_seed_person(self, key: TrackKey, frame: int) -> int | None:
        st = self.ensure_track(key, frame)
        if st.person_id is not None:
            return st.person_id
        if self.require_face_before_person:
            return None
        if st.age < self.new_person_after_frames or int(frame) < st.safe_after_frame:
            return None
        # Do not mint a new P while a recently-lost P is still a plausible
        # continuation. This prevents P013 -> P015 during a brief crossing.
        recent = self._recent_candidates(st, frame)
        if recent and max(ms for _p, _gap, ms in recent) >= 0.45:
            return None
        p = self._new_profile(st, frame, reason="stable_track_seed")
        return p.person_id

    def _bind(self, st: TrackBinding, profile: PersonProfile, frame: int, reason: str, *, reacquire: bool = False) -> None:
        # Motion/body may bind only an unbound local track. Only trusted FACE
        # authority is allowed to split/rebind an already-bound local track.
        if st.person_id is not None:
            return
        self._set_binding(st, profile, frame, reason, authority="continuity", reacquire=reacquire, allow_rebind=False)

    def _set_binding(
        self, st: TrackBinding, profile: PersonProfile, frame: int, reason: str, *,
        authority: str, reacquire: bool = False, allow_rebind: bool = False,
    ) -> bool:
        old_pid = st.person_id
        if old_pid is not None and old_pid != profile.person_id and not allow_rebind:
            return False
        if old_pid == profile.person_id:
            return True
        if st.segments and st.segments[-1].end_frame is None:
            st.segments[-1].end_frame = max(int(frame) - 1, st.segments[-1].start_frame)
        st.segments.append(TrackSegment(int(frame), None, profile.person_id, reason, authority))
        st.person_id = profile.person_id
        st.bind_reason = reason
        if st.key not in profile.members:
            profile.members.append(st.key)
        if reacquire:
            profile.reacquire_count += 1
        self._update_profile_geometry(st, frame)
        event = "face_segment_rebind" if old_pid is not None and old_pid != profile.person_id else "bind"
        self.events.append({
            "frame": int(frame), "camera": st.key[0], "track_id": st.key[1],
            "event": event, "person_id": profile.person_id, "from_person_id": old_pid,
            "reason": reason, "authority": authority, "reacquire": bool(reacquire),
        })
        return True

    def _new_profile(self, st: TrackBinding, frame: int, reason: str) -> PersonProfile:
        p = PersonProfile(person_id=self._next_person_id)
        self._next_person_id += 1
        self.profiles[p.person_id] = p
        self._bind(st, p, frame, reason, reacquire=False)
        self.decisions.append(PersonDecision(
            frame=int(frame), camera=st.key[0], track_id=st.key[1], person_id=p.person_id,
            reason="new_person", accepted=True,
        ))
        return p

    def replace_track_identity(
        self, st: TrackBinding, profile: PersonProfile, reason: str, *,
        score: float | None = None, authority: str | None = None, frame: int | None = None,
    ) -> None:
        """Replace the final assignment for a completed tracklet without overlapping segments."""
        old_pid = st.person_id
        segment_authority = authority or ("face" if "face" in reason or "fusion" in reason else "continuity")
        st.segments = [TrackSegment(st.first_frame, None, profile.person_id, reason, segment_authority)]
        st.person_id = profile.person_id
        st.bind_reason = reason
        if st.key not in profile.members:
            profile.members.append(st.key)
        if old_pid != profile.person_id:
            profile.reacquire_count += 1
            if frame is not None:
                st.identity_change_until = max(st.identity_change_until, int(frame) + 30)
            st.identity_change_count += 1
        self.events.append({
            "frame": int(frame) if frame is not None else st.first_frame, "camera": st.key[0], "track_id": st.key[1],
            "event": "track_identity_replaced", "from_person_id": old_pid,
            "person_id": profile.person_id, "reason": reason, "score": score,
        })

    def _update_profile_geometry(self, st: TrackBinding, frame: int) -> None:
        if st.person_id is None or st.last_bbox is None:
            return
        p = self.profiles[st.person_id]
        cam = st.key[0]
        if cam in p.last_bbox:
            p.prev_bbox[cam] = p.last_bbox[cam].copy()
            p.prev_bbox_frame[cam] = p.last_seen.get(cam, int(frame) - 1)
        p.last_bbox[cam] = st.last_bbox.copy()
        p.last_seen[cam] = int(frame)

    def _person_active_this_frame(self, pid: int, camera: str, frame: int, except_key: TrackKey | None = None) -> bool:
        for k, st in self.tracks.items():
            if except_key is not None and k == except_key:
                continue
            if k[0] == camera and st.person_id == pid and st.last_frame == int(frame):
                return True
        return False

    # -------------------------- short-gap continuity --------------------------
    @staticmethod
    def _center_size(box: np.ndarray) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = [float(v) for v in box]
        return 0.5 * (x1 + x2), 0.5 * (y1 + y2), max(x2 - x1, 1.0), max(y2 - y1, 1.0)

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
        aa = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
        bb = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
        return float(inter / max(aa + bb - inter, 1e-6))

    def _predict_bbox(self, p: PersonProfile, camera: str, frame: int) -> np.ndarray | None:
        last = p.last_bbox.get(camera)
        last_f = p.last_seen.get(camera)
        if last is None or last_f is None:
            return None
        prev = p.prev_bbox.get(camera)
        prev_f = p.prev_bbox_frame.get(camera)
        if prev is None or prev_f is None or last_f <= prev_f:
            return last.copy()
        dt = float(last_f - prev_f)
        gap = float(int(frame) - last_f)
        velocity = (last - prev) / max(dt, 1.0)
        pred = last + velocity * gap
        # Avoid exploding prediction across noisy bbox changes.
        lc = np.asarray(self._center_size(last))
        pc = np.asarray(self._center_size(pred))
        max_shift = max(lc[2], lc[3]) * max(1.5, gap * 0.6)
        shift = np.hypot(pc[0] - lc[0], pc[1] - lc[1])
        if shift > max_shift:
            scale = max_shift / max(shift, 1e-6)
            dx = (pc[0] - lc[0]) * scale
            dy = (pc[1] - lc[1]) * scale
            w, h = lc[2], lc[3]
            cx, cy = lc[0] + dx, lc[1] + dy
            pred = np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dtype=np.float32)
        return pred.astype(np.float32)

    def _motion_score(self, p: PersonProfile, camera: str, frame: int, new_box: np.ndarray) -> float | None:
        pred = self._predict_bbox(p, camera, frame)
        if pred is None:
            return None
        pcx, pcy, pw, ph = self._center_size(pred)
        ncx, ncy, nw, nh = self._center_size(new_box)
        diag = max(math.hypot(pw, ph), 1.0)
        dist = math.hypot(ncx - pcx, ncy - pcy) / diag
        center = math.exp(-2.8 * dist)
        iou = self._iou(pred, new_box)
        scale = math.exp(-abs(math.log(max((nw * nh) / max(pw * ph, 1.0), 1e-6))))
        return float(np.clip(0.55 * center + 0.30 * iou + 0.15 * scale, 0.0, 1.0))

    def _recent_candidates(self, st: TrackBinding, frame: int) -> list[tuple[PersonProfile, int, float]]:
        if st.last_bbox is None:
            return []
        cam = st.key[0]
        out: list[tuple[PersonProfile, int, float]] = []
        for p in self.profiles.values():
            last = p.last_seen.get(cam)
            if last is None:
                continue
            gap = int(frame) - int(last)
            if gap <= 0 or gap > self.short_gap_frames:
                continue
            if self._person_active_this_frame(p.person_id, cam, frame, except_key=st.key):
                continue
            ms = self._motion_score(p, cam, frame, st.last_bbox)
            if ms is not None:
                out.append((p, gap, ms))
        return out

    def _try_motion_only_lease(self, st: TrackBinding, frame: int, *, ambiguous: bool = False) -> bool:
        if st.person_id is not None or st.last_bbox is None:
            return False
        candidates = [(p, gap, ms) for p, gap, ms in self._recent_candidates(st, frame) if gap <= self.motion_only_gap_frames]
        candidates.sort(key=lambda x: x[2], reverse=True)
        if not candidates:
            return False
        best_p, _gap, best = candidates[0]
        second = candidates[1][2] if len(candidates) > 1 else 0.0
        margin = best - second
        # Pure motion is intentionally allowed only for very short gaps and a
        # clear winner. This solves detector/tracker blinks without global ReID.
        min_score = 0.86 if ambiguous else 0.80
        min_margin = 0.16 if ambiguous else 0.12
        if best >= min_score and margin >= min_margin:
            self._bind(st, best_p, frame, "short_gap_motion_lease", reacquire=True)
            self.decisions.append(PersonDecision(
                frame=int(frame), camera=st.key[0], track_id=st.key[1], person_id=best_p.person_id,
                reason="short_gap_motion_lease", accepted=True, score=best, margin=margin,
                motion_score=best,
            ))
            return True
        return False

    # ------------------------------ face authority ------------------------------
    def _face_bank_vectors(self, p: PersonProfile, camera: str, frame: int) -> list[np.ndarray]:
        vectors = [x.embedding for x in p.face_core()]
        if (
            p.recent_face_embedding is not None
            and p.recent_face_camera == camera
            and 0 <= int(frame) - int(p.recent_face_frame) <= self.face_recent_anchor_frames
        ):
            vectors.append(p.recent_face_embedding)
        return vectors

    def _rank_face(self, key: TrackKey, frame: int, embedding: np.ndarray) -> FaceRankResult:
        st = self.tracks.get(key)
        current_pid = st.person_id if st is not None else None
        banks: dict[int, list[np.ndarray]] = {}
        for pid, p in self.profiles.items():
            # A P already active on another local track in the same frame should
            # not normally be stolen. The current P is always allowed so AdaFace
            # can explicitly verify continuity.
            if pid != current_pid and self._person_active_this_frame(pid, key[0], frame, except_key=key):
                continue
            vectors = self._face_bank_vectors(p, key[0], frame)
            if vectors:
                banks[pid] = vectors
        return self.face_recognizer.rank(embedding, banks)

    def _set_face_diag(
        self, key: TrackKey, frame: int, *, decision: str, accepted: bool,
        rank: FaceRankResult | None, votes: int, pose: str, tier: str, quality: float,
    ) -> FaceAuthorityResult:
        st = self.tracks.get(key)
        out = FaceAuthorityResult(
            frame=int(frame),
            person_id=st.person_id if st is not None else None,
            decision=decision,
            accepted=bool(accepted),
            top1_person_id=rank.top1_person_id if rank else None,
            top1_score=rank.top1_score if rank else None,
            top2_person_id=rank.top2_person_id if rank else None,
            top2_score=rank.top2_score if rank else None,
            margin=rank.margin if rank else None,
            candidate_votes=int(votes),
            candidate_needed=self.face_match_confirmations,
            pose=pose,
            tier=tier,
            quality=float(quality),
        )
        self._face_diagnostics[key] = out
        return out

    def _vote_face_pid(self, st: TrackBinding, pid: int, frame: int) -> int:
        if st.face_candidate_pid == pid and 0 <= int(frame) - st.face_candidate_last_frame <= self.face_candidate_max_gap:
            st.face_candidate_votes += 1
        else:
            st.face_candidate_pid = int(pid)
            st.face_candidate_votes = 1
        st.face_candidate_last_frame = int(frame)
        return st.face_candidate_votes

    def _reset_face_pid_vote(self, st: TrackBinding) -> None:
        st.face_candidate_pid = None
        st.face_candidate_votes = 0
        st.face_candidate_last_frame = -1

    def _vote_new_face(self, st: TrackBinding, embedding: np.ndarray, frame: int) -> int:
        e = normalize(embedding).astype(np.float32)
        keep = False
        if st.new_face_candidate is not None and 0 <= int(frame) - st.new_face_last_frame <= self.face_candidate_max_gap:
            sim = float(e @ st.new_face_candidate)
            keep = sim >= 0.72
        if keep:
            w = min(st.new_face_votes, 4)
            st.new_face_candidate = normalize((w * st.new_face_candidate + e) / (w + 1)).astype(np.float32)
            st.new_face_votes += 1
        else:
            st.new_face_candidate = e
            st.new_face_votes = 1
        st.new_face_last_frame = int(frame)
        return st.new_face_votes

    def _reset_new_face_vote(self, st: TrackBinding) -> None:
        st.new_face_candidate = None
        st.new_face_votes = 0
        st.new_face_last_frame = -1

    def _learn_face_sample(
        self, p: PersonProfile, key: TrackKey, frame: int, embedding: np.ndarray,
        quality: float, pose: str, tier: str, aligned_crop: np.ndarray | None,
    ) -> None:
        changed, slot = self._add_face(p, embedding, quality, pose, tier, frame)
        p.face_observations += 1
        if tier in TRUSTED_FACE_TIERS:
            p.face_anchor_count += 1
            p.recent_face_embedding = normalize(embedding).astype(np.float32)
            p.recent_face_quality = float(quality)
            p.recent_face_pose = pose
            p.recent_face_frame = int(frame)
            p.recent_face_camera = key[0]
        if changed and aligned_crop is not None:
            sample = self.store.save_face(p.person_id, pose, slot, aligned_crop)
            if sample:
                pose_items = [x for x in p.face_bank if x.pose == pose]
                if 0 <= slot < len(pose_items):
                    pose_items[slot].sample_file = sample
            self.store.write_metadata(p)

    def observe_face(
        self,
        key: TrackKey,
        frame: int,
        embedding: np.ndarray,
        quality: float,
        pose: str,
        tier: str,
        *,
        aligned_crop: np.ndarray | None = None,
        gallery_employee: str | None = None,
        gallery_accepted: bool = False,
    ) -> FaceAuthorityResult:
        """Run anonymous AdaFace recognition and apply FACE authority.

        Trusted face is the only cue allowed to correct P on an already-bound
        local track. The correction starts a new track segment at `frame`; it
        never rewrites earlier history. Body/motion cannot perform this rebind.
        """
        st = self.ensure_track(key, frame)
        st.face_observations += 1
        trusted = tier in TRUSTED_FACE_TIERS and quality >= self.face_learn_quality

        if not trusted:
            return self._set_face_diag(
                key, frame, decision="SKIP_WEAK_FACE", accepted=False, rank=None,
                votes=0, pose=pose, tier=tier, quality=quality,
            )

        rank = self._rank_face(key, frame, embedding)
        current_pid = st.person_id

        # 1) AdaFace recognizes an existing anonymous P. Require temporal
        # confirmation before changing P, except when it simply verifies the
        # already-bound P.
        if rank.accepted and rank.top1_person_id is not None:
            target_pid = int(rank.top1_person_id)
            target = self.profiles[target_pid]
            if current_pid == target_pid:
                self._reset_face_pid_vote(st)
                self._reset_new_face_vote(st)
                self._learn_face_sample(target, key, frame, embedding, quality, pose, tier, aligned_crop)
                if gallery_accepted and gallery_employee:
                    self.confirm_employee(key, frame, gallery_employee, source="gallery_face")
                return self._set_face_diag(
                    key, frame, decision="FACE_VERIFIED_CURRENT", accepted=True, rank=rank,
                    votes=self.face_match_confirmations, pose=pose, tier=tier, quality=quality,
                )

            votes = self._vote_face_pid(st, target_pid, frame)
            self._reset_new_face_vote(st)
            if votes >= self.face_match_confirmations:
                old_pid = st.person_id
                self._set_binding(
                    st, target, frame, "face_authority_rebind", authority="face",
                    reacquire=True, allow_rebind=True,
                )
                if old_pid is not None and old_pid != target_pid:
                    st.strong_face_conflict = True
                    self.events.append({
                        "frame": int(frame), "camera": key[0], "track_id": key[1],
                        "event": "face_authority_corrected_segment",
                        "from_person_id": old_pid, "to_person_id": target_pid,
                        "face_score": rank.top1_score, "margin": rank.margin,
                    })
                self.decisions.append(PersonDecision(
                    frame=int(frame), camera=key[0], track_id=key[1], person_id=target_pid,
                    reason="face_authority_rebind" if old_pid is not None else "trusted_face_reacquire",
                    accepted=True, score=rank.top1_score, margin=rank.margin, face_score=rank.top1_score,
                ))
                self._reset_face_pid_vote(st)
                self._learn_face_sample(target, key, frame, embedding, quality, pose, tier, aligned_crop)
                if gallery_accepted and gallery_employee:
                    self.confirm_employee(key, frame, gallery_employee, source="gallery_face")
                return self._set_face_diag(
                    key, frame, decision="FACE_REBOUND_EXISTING_P", accepted=True, rank=rank,
                    votes=self.face_match_confirmations, pose=pose, tier=tier, quality=quality,
                )

            return self._set_face_diag(
                key, frame, decision="FACE_MATCH_CANDIDATE", accepted=False, rank=rank,
                votes=votes, pose=pose, tier=tier, quality=quality,
            )

        # 2) No existing face profile passed open-set threshold/margin.
        self._reset_face_pid_vote(st)
        if st.person_id is not None:
            current = self.profiles[st.person_id]
            if not current.face_core():
                # A provisional continuity P is not promoted to a face identity
                # from one frame. Require the same multi-frame face confirmation
                # used when creating a brand-new anonymous face identity.
                votes = self._vote_new_face(st, embedding, frame)
                if votes < self.face_new_confirmations:
                    return self._set_face_diag(
                        key, frame, decision="FACE_ANCHOR_CANDIDATE", accepted=False, rank=rank,
                        votes=votes, pose=pose, tier=tier, quality=quality,
                    )
                self._learn_face_sample(current, key, frame, embedding, quality, pose, tier, aligned_crop)
                self._reset_new_face_vote(st)
                if gallery_accepted and gallery_employee:
                    self.confirm_employee(key, frame, gallery_employee, source="gallery_face")
                return self._set_face_diag(
                    key, frame, decision="FACE_ANCHORED_CURRENT_P", accepted=True, rank=rank,
                    votes=self.face_new_confirmations, pose=pose, tier=tier, quality=quality,
                )

            # Existing face-anchored P + unmatched trusted face: do NOT pollute
            # that P. Treat it as a possible tracker switch, pause learning, and
            # keep identity unchanged until a known P is positively recognized.
            current.conflict_count += 1
            st.strong_face_conflict = True
            st.safe_after_frame = max(st.safe_after_frame, int(frame) + self.ambiguity_grace_frames * 2)
            self.events.append({
                "frame": int(frame), "camera": key[0], "track_id": key[1],
                "event": "trusted_face_unknown_conflict", "person_id": current.person_id,
                "top1_person_id": rank.top1_person_id, "top1_score": rank.top1_score,
                "margin": rank.margin,
            })
            return self._set_face_diag(
                key, frame, decision="FACE_UNKNOWN_CONFLICT_HOLD_P", accepted=False, rank=rank,
                votes=0, pose=pose, tier=tier, quality=quality,
            )

        # 3) Unbound track with a trusted but unseen face. Build a short
        # multi-frame candidate before minting a new anonymous face identity.
        votes = self._vote_new_face(st, embedding, frame)
        if votes < self.face_new_confirmations:
            return self._set_face_diag(
                key, frame, decision="NEW_FACE_CANDIDATE", accepted=False, rank=rank,
                votes=votes, pose=pose, tier=tier, quality=quality,
            )

        p = self._new_profile(st, frame, reason="new_face_identity")
        self._reset_new_face_vote(st)
        self._learn_face_sample(p, key, frame, embedding, quality, pose, tier, aligned_crop)
        self.decisions.append(PersonDecision(
            frame=int(frame), camera=key[0], track_id=key[1], person_id=p.person_id,
            reason="new_face_identity", accepted=True,
        ))
        if gallery_accepted and gallery_employee:
            self.confirm_employee(key, frame, gallery_employee, source="gallery_face")
        return self._set_face_diag(
            key, frame, decision="NEW_FACE_IDENTITY", accepted=True, rank=rank,
            votes=self.face_new_confirmations, pose=pose, tier=tier, quality=quality,
        )

    def _add_face(self, p: PersonProfile, embedding: np.ndarray, quality: float, pose: str, tier: str, frame: int) -> tuple[bool, int]:
        e = normalize(embedding).astype(np.float32)
        group = [x for x in p.face_bank if x.pose == pose]
        if group:
            sims = [float(e @ x.embedding) for x in group]
            best_i = int(np.argmax(sims))
            if sims[best_i] >= 0.72:
                item = group[best_i]
                old_quality = float(item.quality)
                # Once a pose has repeated support, near-identical frames are
                # verification evidence only. Do not keep EMA-updating the
                # canonical face with the same view.
                if item.support >= 2 and sims[best_i] >= 0.88:
                    item.last_frame = int(frame)
                    return False, best_i
                w = min(item.support, 6)
                item.embedding = normalize((w * item.embedding + e) / (w + 1)).astype(np.float32)
                item.support += 1
                item.quality = max(item.quality, float(quality))
                item.last_frame = int(frame)
                item.core = item.core or (item.support >= 2 and item.tier in TRUSTED_FACE_TIERS)
                if tier in TRUSTED_FACE_TIERS:
                    item.tier = tier
                return float(quality) > old_quality + 0.05, best_i
        if len(group) < self.max_face_per_pose:
            item = FaceMemoryItem(e, float(quality), pose, tier, 1, tier == "strong", int(frame), int(frame))
            p.face_bank.append(item)
            return True, len(group)
        slot = min(range(len(group)), key=lambda i: group[i].quality)
        weakest = group[slot]
        if float(quality) > weakest.quality + 0.08:
            weakest.embedding = e
            weakest.quality = float(quality)
            weakest.tier = tier
            weakest.support = 1
            weakest.core = tier == "strong"
            weakest.first_frame = int(frame)
            weakest.last_frame = int(frame)
            return True, slot
        return False, 0

    # ------------------------------ body memory -------------------------------
    @staticmethod
    def _best_body_score(embedding: np.ndarray, p: PersonProfile) -> float | None:
        bank = p.body_core()
        if not bank:
            return None
        q = normalize(embedding)
        return max(float(q @ x.embedding) for x in bank)

    def observe_body(
        self,
        key: TrackKey,
        frame: int,
        embedding: np.ndarray,
        quality: float,
        *,
        crop: np.ndarray | None = None,
        ambiguous: bool = False,
    ) -> int | None:
        st = self.ensure_track(key, frame)
        st.body_observations += 1
        e = normalize(embedding).astype(np.float32)

        if st.person_id is None and not ambiguous and st.last_bbox is not None:
            candidates: list[tuple[float, PersonProfile, int, float, float]] = []
            for p, gap, motion in self._recent_candidates(st, frame):
                body = self._best_body_score(e, p)
                if body is None:
                    continue
                threshold = self.body_short_gap_near if gap <= 15 else self.body_short_gap_far
                if body < threshold or motion < 0.30:
                    continue
                score = 0.58 * motion + 0.42 * body
                candidates.append((score, p, gap, body, motion))
            candidates.sort(key=lambda x: x[0], reverse=True)
            if candidates:
                score, p, _gap, body, motion = candidates[0]
                second = candidates[1][0] if len(candidates) > 1 else 0.0
                margin = score - second
                if margin >= self.body_match_margin:
                    if st.body_candidate_pid == p.person_id and int(frame) - st.body_candidate_last_frame <= self.face_candidate_max_gap:
                        st.body_candidate_votes += 1
                    else:
                        st.body_candidate_pid = p.person_id
                        st.body_candidate_votes = 1
                    st.body_candidate_last_frame = int(frame)
                if margin >= self.body_match_margin and st.body_candidate_votes >= 3:
                    self._bind(st, p, frame, "short_gap_motion_body", reacquire=True)
                    self.decisions.append(PersonDecision(
                        frame=int(frame), camera=key[0], track_id=key[1], person_id=p.person_id,
                        reason="short_gap_motion_body", accepted=True, score=score, margin=margin,
                        body_score=body, motion_score=motion,
                    ))
                    st.body_candidate_pid = None
                    st.body_candidate_votes = 0
                    st.body_candidate_last_frame = -1
            else:
                st.body_candidate_pid = None
                st.body_candidate_votes = 0
                st.body_candidate_last_frame = -1

        if st.person_id is None:
            self.maybe_seed_person(key, frame)
        if st.person_id is None:
            return None

        p = self.profiles[st.person_id]
        if ambiguous or int(frame) < st.safe_after_frame or float(quality) < self.body_learn_quality:
            return p.person_id

        # Detect abrupt appearance jumps inside the same L. This does not rename
        # the person; it pauses learning because the local tracker may have switched.
        if st.last_body is not None and st.last_body_frame is not None:
            local_sim = float(e @ st.last_body)
            if local_sim < self.body_jump_threshold:
                st.appearance_jump_until = max(st.appearance_jump_until, int(frame) + self.ambiguity_grace_frames)
                self.events.append({
                    "frame": int(frame), "camera": key[0], "track_id": key[1],
                    "event": "body_jump_learning_paused", "person_id": p.person_id,
                    "similarity": local_sim,
                })
                # Do not make a suspected tracker switch the new continuity
                # reference. The sample remains diagnostic-only until a stable
                # observation confirms the appearance change.
                return p.person_id
        st.last_body = e.copy()
        st.last_body_frame = int(frame)
        if int(frame) < st.appearance_jump_until or st.strong_face_conflict:
            return p.person_id

        changed, slot = self._add_body(p, e, quality, frame)
        p.body_observations += 1
        if changed and crop is not None:
            sample = self.store.save_body(p.person_id, slot, crop)
            if sample and 0 <= slot < len(p.body_bank):
                p.body_bank[slot].sample_file = sample
            self.store.write_metadata(p)
        return p.person_id

    def _add_body(self, p: PersonProfile, e: np.ndarray, quality: float, frame: int) -> tuple[bool, int]:
        if p.body_bank:
            sims = [float(e @ x.embedding) for x in p.body_bank]
            best_i = int(np.argmax(sims))
            if sims[best_i] >= self.body_novelty_threshold:
                item = p.body_bank[best_i]
                if item.support >= 2 and sims[best_i] >= 0.93:
                    item.last_frame = int(frame)
                    return False, best_i
                old_quality = float(item.quality)
                w = min(item.support, 8)
                item.embedding = normalize((w * item.embedding + e) / (w + 1)).astype(np.float32)
                item.support += 1
                item.quality = max(item.quality, float(quality))
                item.last_frame = int(frame)
                item.core = item.core or item.support >= 3
                return float(quality) > old_quality + 0.08, best_i
        if len(p.body_bank) < self.max_body_views:
            p.body_bank.append(BodyMemoryItem(e.copy(), float(quality), 1, False, int(frame), int(frame)))
            return True, len(p.body_bank) - 1
        # Replace only weak singleton view with clearly better sample.
        candidate_indices = [i for i, x in enumerate(p.body_bank) if x.support <= 1]
        if candidate_indices:
            i = min(candidate_indices, key=lambda j: p.body_bank[j].quality)
            if float(quality) > p.body_bank[i].quality + 0.10:
                p.body_bank[i] = BodyMemoryItem(e.copy(), float(quality), 1, False, int(frame), int(frame))
                return True, i
        return False, 0

    # Face corrections are segment-level only. We intentionally do not
    # merge/rewrite entire historical profiles here; earlier frames remain auditable.

    # --------------------------- employee authority ---------------------------
    def confirm_employee(self, key: TrackKey, frame: int, employee_id: str, *, source: str) -> None:
        st = self.ensure_track(key, frame)
        if st.person_id is None:
            p = self._new_profile(st, frame, reason="employee_face_seed")
        else:
            p = self.profiles[st.person_id]
        if p.employee_id is None:
            p.employee_id = employee_id
            self.events.append({
                "frame": int(frame), "camera": key[0], "track_id": key[1],
                "event": "employee_confirmed", "person_id": p.person_id,
                "employee_id": employee_id, "source": source,
            })
        elif p.employee_id != employee_id:
            p.conflict_count += 1
            self.events.append({
                "frame": int(frame), "camera": key[0], "track_id": key[1],
                "event": "employee_face_reassigned", "person_id": p.person_id,
                "existing_employee_id": p.employee_id, "new_employee_id": employee_id,
            })
            p.employee_id = employee_id

    # ------------------------------- exporting --------------------------------
    def payload(self) -> dict:
        profiles = []
        for p in sorted(self.profiles.values(), key=lambda x: x.person_id):
            profiles.append({
                "person_id": f"P{p.person_id:03d}",
                "employee_id": p.employee_id,
                "maturity": p.maturity(),
                "confidence": p.confidence(),
                "reacquire_count": p.reacquire_count,
                "face_anchor_count": p.face_anchor_count,
                "face_observations": p.face_observations,
                "body_observations": p.body_observations,
                "conflict_count": p.conflict_count,
                "recent_face": {
                    "frame": p.recent_face_frame,
                    "camera": p.recent_face_camera,
                    "pose": p.recent_face_pose,
                    "quality": p.recent_face_quality,
                } if p.recent_face_embedding is not None else None,
                "members": [{"camera": c, "track_id": int(t)} for c, t in p.members],
                "face_memory": [
                    {
                        "pose": x.pose, "tier": x.tier, "quality": x.quality,
                        "support": x.support, "core": x.core,
                        "first_frame": x.first_frame, "last_frame": x.last_frame,
                        "sample_file": x.sample_file,
                    }
                    for x in p.face_bank
                ],
                "body_memory": [
                    {
                        "quality": x.quality, "support": x.support, "core": x.core,
                        "first_frame": x.first_frame, "last_frame": x.last_frame,
                        "sample_file": x.sample_file,
                    }
                    for x in p.body_bank
                ],
            })
        tracks = []
        for st in sorted(self.tracks.values(), key=lambda x: (x.key[0], x.first_frame, x.key[1])):
            tracks.append({
                "camera": st.key[0], "track_id": st.key[1],
                "person_id": f"P{st.person_id:03d}" if st.person_id is not None else None,
                "bind_reason": st.bind_reason,
                "first_frame": st.first_frame, "last_frame": st.last_frame,
                "age": st.age,
                "face_observations": st.face_observations,
                "body_observations": st.body_observations,
                "ambiguous_frames": st.ambiguous_frames,
                "strong_face_conflict": st.strong_face_conflict,
                "segments": [
                    {
                        "start_frame": seg.start_frame, "end_frame": seg.end_frame,
                        "person_id": f"P{seg.person_id:03d}" if seg.person_id is not None else None,
                        "reason": seg.reason, "authority": seg.authority,
                    }
                    for seg in st.segments
                ],
            })
        decisions = [d.__dict__ for d in self.decisions]
        return {
            "mode": "face_anchor_authority_v3",
            "profiles": profiles,
            "tracks": tracks,
            "decisions": decisions,
            "events": self.events,
        }

    def save_metadata_all(self) -> None:
        for p in self.profiles.values():
            self.store.write_metadata(p)
