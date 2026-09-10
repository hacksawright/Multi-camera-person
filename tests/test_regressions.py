from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from src.embedding_store import EmbeddingRecord, EmbeddingStore
from src.identity_manager import IdentityManager
from src.identity_database import IdentityDatabase
from src.person_memory import PersonMemory, PersonProfile
from src.evidence_fusion import IdentityEvidence, TrackletEvidenceFusion
from src.gallery import GalleryHit


def test_embedding_store_round_trip_and_query():
    with tempfile.TemporaryDirectory() as root:
        store = EmbeddingStore(Path(root))
        store.add(EmbeddingRecord("f1", "face", "cam3", 10, 4, np.array([2.0, 0.0]), .9, 1, "front"))
        store.save()
        loaded = EmbeddingStore.load(root)
        result = loaded.query(np.array([1.0, 0.0]), kind="face")
        assert result[0][0].record_id == "f1"
        assert result[0][1] > .99


def test_confirmed_identity_can_be_reassigned_by_repeated_evidence():
    manager = IdentityManager(min_confirmations=2, min_confirmation_weight=.5)
    a = GalleryHit("EMP001", .8, .1, .7, True, .55, .07)
    b = GalleryHit("EMP002", .8, .1, .7, True, .55, .07)
    manager.update(1, 1, a, .8, "strong")
    manager.update(1, 2, a, .8, "strong")
    manager.update(1, 3, b, .8, "strong")
    manager.update(1, 4, b, .8, "strong")
    assert manager.final(1).employee_id == "EMP002"


def test_identity_database_round_trip_query_and_assignment():
    with tempfile.TemporaryDirectory() as root:
        with IdentityDatabase(Path(root) / "identity.sqlite") as db:
            tracklet = db.upsert_tracklet("cam3", 7, 10)
            db.add_detection(tracklet, 10, np.array([1, 2, 30, 80]), .9, False)
            db.add_observation(
                tracklet, 10, "face", np.array([2.0, 0.0]), .8,
                pose="front", model_name="face", model_version="v1",
            )
            db.add_observation(
                tracklet, 11, "face", np.array([2.0, 0.0]), .8,
                pose="front", model_name="face", model_version="v1",
            )
            db.ensure_identity(3)
            db.assign(tracklet, 3, 10, "CONFIRMED", "test")
            db.promote_tracklet_references(tracklet, 3)
            db.ensure_identity(4)
            db.assign(tracklet, 4, 10, "CONFIRMED", "same_frame_correction")
            hits = db.search_identities([np.array([1.0, 0.0])], "face", "face", "v1")
            assert hits[0].identity_id == 4
            assert hits[0].score > .99
            assert hits[0].support == 1
            invalid = db.connection.execute(
                "SELECT COUNT(*) FROM assignments WHERE valid_to < valid_from"
            ).fetchone()[0]
            assert invalid == 0
            decision = TrackletEvidenceFusion().decide([
                IdentityEvidence(4, face_scores=[.8, .81], face_support=2)
            ])
            db.add_fusion_decision(tracklet, 10, decision)
            assert db.connection.execute("select count(*) from fusion_decisions").fetchone()[0] == 1
            assert db.integrity_check() == "ok"


def test_identity_database_separates_reused_track_ids_between_sessions():
    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / "identity.sqlite"
        with IdentityDatabase(path, session_id="run-a") as db:
            first = db.upsert_tracklet("cam3", 1, 0)
        with IdentityDatabase(path, session_id="run-b") as db:
            second = db.upsert_tracklet("cam3", 1, 0)
        assert first != second


def test_canonical_database_export_isolated_from_source():
    with tempfile.TemporaryDirectory() as root:
        source = Path(root) / "source.sqlite"
        destination = Path(root) / "canonical.sqlite"
        with IdentityDatabase(source, session_id="source") as db:
            tracklet = db.upsert_tracklet("cam3", 1, 1)
            db.add_observation(tracklet, 1, "face", np.array([1.0, 0.0]), .8, model_name="face", model_version="v1")
            db.ensure_identity(1)
            db.assign(tracklet, 1, 1, "VERIFIED", "test")
            db.promote_tracklet_references(tracklet, 1)
            stats = db.export_canonical(destination)
            assert stats["observations"] == 1
        assert source.is_file() and destination.is_file()
        with IdentityDatabase(destination) as db:
            assert db.canonical_stats()["observations"] == 1
            assert db.integrity_check() == "ok"


def test_canonical_database_is_read_only():
    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / "canonical.sqlite"
        with IdentityDatabase(path) as db:
            db.ensure_identity(1)
        with IdentityDatabase(path, readonly=True) as db:
            try:
                db.ensure_identity(2)
            except Exception:
                pass
            else:
                raise AssertionError("read-only canonical DB accepted a write")


def test_reconciliation_replaces_overlapping_segment_history():
    with tempfile.TemporaryDirectory() as root:
        memory = PersonMemory(sample_root=root, new_person_after_frames=1, require_face_before_person=False)
        key = ("cam3", 9)
        memory.touch_track(key, 10, np.array([0, 0, 20, 80], dtype=np.float32))
        old = memory.maybe_seed_person(key, 10)
        target_key = ("cam3", 3)
        memory.touch_track(target_key, 0, np.array([50, 0, 70, 80], dtype=np.float32))
        target = memory.maybe_seed_person(target_key, 0)
        state = memory.state_for_track(key)
        memory.replace_track_identity(state, memory.profiles[target], "test")
        assert old != target
        assert len(state.segments) == 1
        assert state.segments[0].start_frame == state.first_frame
        assert state.segments[0].person_id == target


def test_body_match_requires_temporal_confirmation():
    with tempfile.TemporaryDirectory() as root:
        memory = PersonMemory(sample_root=root, new_person_after_frames=100, require_face_before_person=False)
        old = ("cam3", 1)
        new = ("cam3", 2)
        box = np.array([100, 100, 150, 250], dtype=np.float32)
        memory.touch_track(old, 0, box)
        old_pid = memory.maybe_seed_person(old, 0)  # no seed because age gate
        memory.touch_track(old, 100, box)
        old_pid = memory.maybe_seed_person(old, 100)
        memory.observe_body(old, 100, np.array([1.0, 0.0, 0.0]), 1.0)
        memory.touch_track(new, 107, box, ambiguous=True)
        assert memory.observe_body(new, 107, np.array([1.0, 0.0, 0.0]), 1.0) is None
        assert memory.observe_body(new, 108, np.array([1.0, 0.0, 0.0]), 1.0) is None
        memory.observe_body(new, 109, np.array([1.0, 0.0, 0.0]), 1.0)
        assert memory.state_for_track(new).person_id == old_pid


def test_evidence_fusion_requires_independent_support_and_exposes_reason():
    fusion = TrackletEvidenceFusion()
    decision = fusion.decide([
        IdentityEvidence(2, face_scores=[.49, .48, .47], body_scores=[.90], face_support=3, body_support=3),
        IdentityEvidence(3, face_scores=[.85, .84], body_scores=[.60], face_support=2, body_support=1),
    ])
    assert decision.identity_id == 3
    assert decision.state == "CONFIRMED"
    assert decision.reason == "FACE_ONLY"


def test_evidence_fusion_rejects_ambiguous_candidates():
    fusion = TrackletEvidenceFusion()
    decision = fusion.decide([
        IdentityEvidence(2, face_scores=[.70, .69], face_support=2),
        IdentityEvidence(3, face_scores=[.69, .68], face_support=2),
    ])
    assert decision.state == "CONFLICT"
    assert decision.reason == "LOW_MARGIN"


def test_repeated_face_and_body_views_do_not_drift_prototypes():
    with tempfile.TemporaryDirectory() as root:
        memory = PersonMemory(sample_root=root, require_face_before_person=False)
        profile = PersonProfile(1)
        memory._add_face(profile, np.array([1.0, 0.0]), .8, "front", "support", 1)
        memory._add_face(profile, np.array([1.0, 0.0]), .8, "front", "support", 2)
        before_face = profile.face_bank[0].embedding.copy()
        memory._add_face(profile, np.array([.999, .01]), .9, "front", "support", 3)
        assert profile.face_bank[0].support == 2
        assert float(profile.face_bank[0].embedding @ before_face) > .999
        memory._add_body(profile, np.array([1.0, 0.0]), .8, 1)
        memory._add_body(profile, np.array([1.0, 0.0]), .8, 2)
        before_body = profile.body_bank[0].embedding.copy()
        memory._add_body(profile, np.array([.999, .01]), .9, 3)
        assert profile.body_bank[0].support == 2
        assert float(profile.body_bank[0].embedding @ before_body) > .999
