from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def _csv(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    a = p.parse_args()
    root = Path(a.run_dir)

    tracks = _csv(root / "tracks.csv")
    faces = _csv(root / "face_observations.csv")
    temporal = _csv(root / "face_temporal.csv")
    face_ids = _csv(root / "face_identity_diagnostics.csv")
    summaries = json.loads((root / "track_summaries.json").read_text(encoding="utf-8")) if (root / "track_summaries.json").is_file() else []
    events = json.loads((root / "identity_events.json").read_text(encoding="utf-8")) if (root / "identity_events.json").is_file() else []
    mem_path = root / "person_memory.json"
    memory = json.loads(mem_path.read_text(encoding="utf-8")) if mem_path.is_file() else {"profiles": [], "tracks": [], "decisions": [], "events": []}

    local = {(r["camera"], int(r["track_id"])) for r in tracks}
    pids = {str(r.get("person_id", "")) for r in tracks if r.get("person_id")}
    # Count unbound local tracks, not every unbound frame row.
    unbound = {
        (r["camera"], int(r["track_id"]))
        for r in tracks
        if not r.get("person_id")
    }
    bound_keys = {
        (r["camera"], int(r["track_id"]))
        for r in tracks
        if r.get("person_id")
    }
    unbound -= bound_keys
    state_counts = Counter(r.get("identity_state", "") for r in tracks)

    profiles = memory.get("profiles", [])
    decisions = memory.get("decisions", [])
    mem_events = memory.get("events", [])
    mem_tracks = memory.get("tracks", [])

    print("local_tracks                :", len(local))
    print("person_ids                  :", len(pids) if pids else len(profiles))
    print("unbound_tracks              :", len(unbound) if tracks else sum(x.get("person_id") is None for x in mem_tracks))
    print("confirmed_profiles          :", sum(bool(x.get("employee_id")) for x in profiles))
    print("face_anchored_profiles      :", sum(int(x.get("face_anchor_count", 0)) > 0 for x in profiles))
    print("face_observations           :", len(faces))
    print("identity_states             :", dict(state_counts))

    bind_reasons = Counter(x.get("bind_reason") or "UNBOUND" for x in mem_tracks)
    print("person_bind_reasons         :", dict(bind_reasons))
    decision_reasons = Counter(x.get("reason", "") for x in decisions)
    print("person_decisions            :", dict(decision_reasons))
    reacq = sum(int(x.get("reacquire_count", 0)) for x in profiles)
    print("person_reacquisitions       :", reacq)
    print("short_gap_motion_reacquire  :", decision_reasons.get("short_gap_motion_lease", 0))
    print("short_gap_body_reacquire    :", decision_reasons.get("short_gap_motion_body", 0))
    print("trusted_face_reacquire      :", decision_reasons.get("trusted_face_reacquire", 0))

    maturity = Counter(x.get("maturity", "") for x in profiles)
    print("person_maturity             :", dict(maturity))
    if profiles:
        confs = [float(x.get("confidence", 0.0)) for x in profiles]
        print("person_conf_mean            :", round(sum(confs) / len(confs), 3))
        face_views = [len(x.get("face_memory", [])) for x in profiles]
        body_views = [len(x.get("body_memory", [])) for x in profiles]
        print("face_views_per_P_mean       :", round(sum(face_views) / len(face_views), 2))
        print("face_views_per_P_max        :", max(face_views) if face_views else 0)
        print("body_views_per_P_mean       :", round(sum(body_views) / len(body_views), 2))
        print("body_views_per_P_max        :", max(body_views) if body_views else 0)
        saved_face = sum(bool(v.get("sample_file")) for x in profiles for v in x.get("face_memory", []))
        saved_body = sum(bool(v.get("sample_file")) for x in profiles for v in x.get("body_memory", []))
        print("saved_face_examples         :", saved_face)
        print("saved_body_examples         :", saved_body)

    mem_event_counts = Counter(x.get("event", "") for x in mem_events)
    print("memory_events               :", dict(mem_event_counts))
    print("strong_face_tracker_conflict:", mem_event_counts.get("strong_face_tracker_conflict", 0))
    print("face_segment_rebinds        :", mem_event_counts.get("face_segment_rebind", 0))
    print("face_authority_corrections  :", mem_event_counts.get("face_authority_corrected_segment", 0))
    print("trusted_face_unknown_conflict:", mem_event_counts.get("trusted_face_unknown_conflict", 0))
    print("body_jump_learning_paused   :", mem_event_counts.get("body_jump_learning_paused", 0))

    ambiguous = sum(int(x.get("ambiguous_frames", 0) or 0) for x in summaries)
    body_obs = sum(int(x.get("online_body_observations", 0) or 0) for x in summaries)
    face_obs = sum(int(x.get("online_face_observations", 0) or 0) for x in summaries)
    print("online_body_observations    :", body_obs)
    print("online_face_observations    :", face_obs)
    print("memory_ambiguous_frames     :", ambiguous)

    conflicts = [e for e in events if str(e.get("reason", "")).startswith("conflict_ignored")]
    print("ignored_gallery_conflicts   :", len(conflicts))

    if faces:
        qs = [float(x["quality"]) for x in faces]
        tiers = Counter(x["tier"] for x in faces)
        poses = Counter(x["pose"] for x in faces)
        accepted = [x for x in faces if str(x.get("accepted", "")).lower() == "true"]
        print("face_quality_mean           :", round(sum(qs) / len(qs), 3))
        print("face_tiers                  :", dict(tiers))
        print("face_poses                  :", dict(poses))
        print("gallery_accepted_obs        :", len(accepted))


    if face_ids:
        decisions = Counter(x.get("face_id_decision", "") for x in face_ids)
        accepted = sum(str(x.get("face_id_accepted", "")).lower() == "true" for x in face_ids)
        scored = []
        margins = []
        for x in face_ids:
            try:
                if x.get("face_id_top1_score", "") != "":
                    scored.append(float(x["face_id_top1_score"]))
                if x.get("face_id_margin", "") != "":
                    margins.append(float(x["face_id_margin"]))
            except ValueError:
                pass
        print("face_id_diagnostics         :", len(face_ids))
        print("face_id_decisions           :", dict(decisions))
        print("face_id_accepted            :", accepted)
        print("face_id_score_mean          :", round(sum(scored) / len(scored), 3) if scored else 0.0)
        print("face_id_margin_mean         :", round(sum(margins) / len(margins), 3) if margins else 0.0)

    if temporal:
        attempts = len(temporal)
        hits = sum(str(x.get("detected", "")).lower() == "true" for x in temporal)
        states = Counter(x.get("state", "") for x in temporal)
        print("face_detector_attempts      :", attempts)
        print("face_detector_hits          :", hits)
        print("face_detector_hit_rate      :", round(hits / attempts, 3) if attempts else 0.0)
        print("face_temporal_states        :", dict(states))
        print("face_coasting_frames        :", states.get("COASTING", 0))


if __name__ == "__main__":
    main()
