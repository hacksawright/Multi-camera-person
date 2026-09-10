from __future__ import annotations

import argparse
import bisect
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__import__("os").environ.get("PIT_WORKSPACE_ROOT", str(PROJECT_ROOT.parent))).resolve()
ASSETS_ROOT = Path(__import__("os").environ.get("PIT_ASSETS_ROOT", str(PROJECT_ROOT))).resolve()

_LOCAL_REID = PROJECT_ROOT / "third_party" / "deep-person-reid"
if (_LOCAL_REID / "torchreid" / "__init__.py").is_file():
    _local_reid_str = str(_LOCAL_REID)
    if _local_reid_str in sys.path:
        sys.path.remove(_local_reid_str)
    sys.path.insert(0, _local_reid_str)

from .adaface_quality import AdaFaceQualityEmbedder
from .person_memory import PersonMemory
from .face_detector import FaceDetector
from .face_quality import compute_quality, laplacian_blur_score
from .face_temporal_tracker import FaceTemporalTracker
from .face_utils import align_face_bgr
from .gallery import GalleryHit, build_gallery, match_gallery
from .head_roi import detect_best_face_in_person
from .identity_manager import IdentityManager
from .embedding_store import EmbeddingRecord, EmbeddingStore
from .identity_database import IdentityDatabase
from .evidence_fusion import IdentityEvidence, TrackletEvidenceFusion
from .models import FaceObservation, IdentityEvent, TrackKey, TrackSpan, TrackSummary
from .prototypes import aggregate_face_observations, normalize, select_body_prototypes
from .reid_embedder import TorchReIDEmbedder


def torch_device(device_arg: str) -> str:
    if device_arg.lower() == "cpu":
        return "cpu"
    if device_arg.startswith("cuda"):
        return device_arg
    return f"cuda:{device_arg}"


def camera_name(source: str, idx: int) -> str:
    if source.isdigit():
        return f"cam{source}"
    clean = source.split("?")[0].rstrip("/")
    return Path(clean).stem.strip() or f"cam{idx}"


def safe_crop(frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 - x1 < 12 or y2 - y1 < 24:
        return None
    return frame[y1:y2, x1:x2].copy()


def hit_fields(hit: GalleryHit | None) -> dict:
    if hit is None:
        return {
            "top1": "", "top1_score": "", "top2_score": "", "margin": "", "accepted": False,
            "threshold": "", "margin_threshold": "",
        }
    return {
        "top1": hit.name or "",
        "top1_score": hit.score,
        "top2_score": hit.second_score,
        "margin": hit.margin,
        "accepted": hit.accepted,
        "threshold": hit.threshold,
        "margin_threshold": hit.margin_threshold,
    }


def _balanced_face_insert(bank: list[FaceObservation], obs: FaceObservation, max_total: int) -> None:
    bank.append(obs)
    if len(bank) <= max_total:
        return
    per_pose_min = max(2, max_total // 4)
    keep: list[FaceObservation] = []
    leftovers: list[FaceObservation] = []
    for pose in ("front", "left", "right"):
        group = sorted((x for x in bank if x.pose == pose), key=lambda x: x.quality, reverse=True)
        keep.extend(group[:per_pose_min])
        leftovers.extend(group[per_pose_min:])
    keep = keep[:max_total]
    if len(keep) < max_total:
        leftovers.sort(key=lambda x: x.quality, reverse=True)
        keep.extend(leftovers[:max_total - len(keep)])
    bank[:] = keep


def _reservoir_embedding(bank: list[np.ndarray], emb: np.ndarray, seen: int, limit: int, track_id: int) -> None:
    emb = normalize(emb).astype(np.float32)
    if len(bank) < limit:
        bank.append(emb)
        return
    j = int(((seen * 1103515245 + track_id * 12345) & 0x7FFFFFFF) % max(seen, 1))
    if j < limit:
        bank[j] = emb


def _snapshot_to_row(row: dict, snapshot) -> None:
    row["face_state"] = snapshot.state
    row["face_score"] = float(snapshot.score) if snapshot.bbox is not None else ""
    if snapshot.bbox is None:
        row.update({"face_x1": "", "face_y1": "", "face_x2": "", "face_y2": ""})
    else:
        row.update({
            "face_x1": float(snapshot.bbox[0]), "face_y1": float(snapshot.bbox[1]),
            "face_x2": float(snapshot.bbox[2]), "face_y2": float(snapshot.bbox[3]),
        })


def _ambiguity_flags(boxes: np.ndarray) -> list[bool]:
    """Mark crowded/crossing boxes where continual memory learning should pause."""
    n = len(boxes)
    out = [False] * n
    if n <= 1:
        return out
    for i in range(n):
        ax1, ay1, ax2, ay2 = [float(v) for v in boxes[i]]
        aa = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
        for j in range(i + 1, n):
            bx1, by1, bx2, by2 = [float(v) for v in boxes[j]]
            bb = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
            if inter <= 0:
                continue
            iou = inter / max(aa + bb - inter, 1e-6)
            ios = inter / max(min(aa, bb), 1e-6)
            if iou >= 0.50 or ios >= 0.75:
                out[i] = True
                out[j] = True
    return out


def _body_quality(box: np.ndarray, frame_shape: tuple[int, int, int], det_conf: float) -> float:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = max(x2 - x1, 0.0), max(y2 - y1, 0.0)
    size = float(np.clip(min(bh / max(h * 0.35, 1.0), bw / max(w * 0.10, 1.0)), 0.0, 1.0))
    clipped = float(x1 <= 1 or y1 <= 1 or x2 >= w - 2 or y2 >= h - 2)
    return float(np.clip(0.60 * float(det_conf) + 0.40 * size - 0.10 * clipped, 0.0, 1.0))


def _memory_overlay(memory: PersonMemory, key: TrackKey) -> tuple[str, str]:
    label = memory.public_label(key)
    st = memory.state_for_track(key)
    if st is None or st.person_id is None:
        return label, "MEM:UNBOUND"
    ident = memory.profiles.get(st.person_id)
    if ident is None:
        return label, "MEM:BOUND"
    return label, f"MEM:{ident.maturity()} {ident.confidence():.2f}"


def _identity_text(state) -> str:
    if state.state == "CONFIRMED" and state.employee_id:
        return state.employee_id
    if state.state == "CANDIDATE" and state.candidate_id:
        return f"CAND:{state.candidate_id}"
    return "UNKNOWN"


def process_camera(
    source: str,
    cam: str,
    output_dir: Path,
    args: argparse.Namespace,
    face_detector: FaceDetector,
    face_embedder: AdaFaceQualityEmbedder,
    reid_embedder: TorchReIDEmbedder,
    gallery,
    memory: PersonMemory,
    embedding_store: EmbeddingStore,
    identity_db: IdentityDatabase,
    canonical_db: IdentityDatabase,
) -> tuple[
    list[dict], dict[int, list[np.ndarray]], dict[int, list[FaceObservation]], dict[int, TrackSpan],
    dict[int, IdentityManager], list[dict], list[dict], list[dict]
]:
    print(f"[track] {cam}: {source}")
    model = YOLO(args.yolo)
    rows: list[dict] = []
    body_bank: dict[int, list[np.ndarray]] = defaultdict(list)
    body_seen: dict[int, int] = defaultdict(int)
    face_bank: dict[int, list[FaceObservation]] = defaultdict(list)
    first_seen: dict[int, int] = {}
    last_seen: dict[int, int] = {}
    identity_managers: dict[int, IdentityManager] = {}
    face_trackers: dict[int, FaceTemporalTracker] = {}
    face_debug: list[dict] = []
    temporal_debug: list[dict] = []
    db_face_candidates: dict[int, dict[int, tuple[int, float, int]]] = defaultdict(dict)
    db_face_query_debug: list[dict] = []

    results = model.track(
        source=int(source) if source.isdigit() else source,
        stream=True,
        persist=True,
        tracker=str(Path(args.tracker).resolve()),
        classes=[0],
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        half=(torch_device(args.device).startswith("cuda") and not args.no_half),
        verbose=False,
    )

    preview_path = output_dir / f"{cam}_local_identity.mp4"
    writer = None
    frame_idx = -1

    for frame_idx, result in enumerate(results):
        orig = result.orig_img
        frame = orig.copy()
        if writer is None:
            h, w = frame.shape[:2]
            fps = 25.0
            if not source.isdigit():
                cap = cv2.VideoCapture(source)
                source_fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if source_fps and source_fps > 1:
                    fps = source_fps
            writer = cv2.VideoWriter(str(preview_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        boxes_obj = result.boxes
        if boxes_obj is None or boxes_obj.id is None or len(boxes_obj) == 0:
            writer.write(frame)
            continue

        boxes = boxes_obj.xyxy.detach().cpu().numpy()
        ids = boxes_obj.id.detach().cpu().numpy().astype(int)
        confs = boxes_obj.conf.detach().cpu().numpy()
        ambiguous = _ambiguity_flags(boxes)
        current_rows: dict[int, dict] = {}
        box_by_tid: dict[int, np.ndarray] = {}
        ambiguity_by_tid: dict[int, bool] = {}
        conf_by_tid: dict[int, float] = {}

        for box, tid_raw, score, is_ambiguous in zip(boxes, ids, confs, ambiguous):
            tid = int(tid_raw)
            key = (cam, tid)
            first_seen.setdefault(tid, frame_idx)
            last_seen[tid] = frame_idx
            box_by_tid[tid] = box
            ambiguity_by_tid[tid] = bool(is_ambiguous)
            conf_by_tid[tid] = float(score)

            if tid not in identity_managers:
                identity_managers[tid] = IdentityManager(
                    min_confirmations=args.confirm_frames,
                    min_confirmation_weight=args.confirm_weight,
                    min_support_quality=args.identity_min_quality,
                    strong_quality=args.strong_quality,
                )
                identity_managers[tid].ensure(tid, frame_idx)
            if tid not in face_trackers:
                face_trackers[tid] = FaceTemporalTracker(
                    min_hits=args.face_temporal_min_hits,
                    max_misses=args.face_temporal_max_misses,
                    ema_alpha=args.face_temporal_alpha,
                )
            face_trackers[tid].update_person(box)
            memory.touch_track(key, frame_idx, box, ambiguous=bool(is_ambiguous))
            db_tracklet_id = identity_db.upsert_tracklet(cam, tid, frame_idx)
            identity_db.add_detection(db_tracklet_id, frame_idx, box, float(score), bool(is_ambiguous))

            x1, y1, x2, y2 = [float(v) for v in box]
            row = {
                "camera": cam, "frame": frame_idx, "track_id": tid,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": float(score),
                "memory_ambiguous": bool(is_ambiguous),
            }
            _snapshot_to_row(row, face_trackers[tid].snapshot())
            rows.append(row)
            current_rows[tid] = row

        # ------------------------- online OSNet evidence -------------------------
        if args.reid_sample_every > 0 and frame_idx % args.reid_sample_every == 0:
            body_items: list[tuple[int, np.ndarray, float, bool]] = []
            body_crops: list[np.ndarray] = []
            for box, tid_raw, score, is_ambiguous in zip(boxes, ids, confs, ambiguous):
                tid = int(tid_raw)
                # Skip expensive learning around crossings; this is also the
                # period most likely to contain a tracker identity switch.
                if is_ambiguous:
                    continue
                crop = safe_crop(orig, box)
                if crop is None:
                    continue
                q = _body_quality(box, orig.shape, float(score))
                if q < args.memory_min_body_quality:
                    continue
                body_items.append((tid, box, q, bool(is_ambiguous)))
                body_crops.append(crop)
            if body_crops:
                feats = reid_embedder.encode(body_crops)
                for item, feat, crop in zip(body_items, feats, body_crops):
                    tid, _box, q, is_ambiguous = item
                    body_seen[tid] += 1
                    _reservoir_embedding(body_bank[tid], feat, body_seen[tid], args.max_reid_embeddings, tid)
                    memory.observe_body(
                        (cam, tid), frame_idx, feat, q,
                        crop=crop,
                        ambiguous=is_ambiguous,
                    )
                    embedding_store.add(EmbeddingRecord(
                        f"body:{cam}:{tid}:{frame_idx}", "body", cam, frame_idx, tid,
                        feat, q, memory.state_for_track((cam, tid)).person_id,
                    ))
                    identity_db.add_observation(
                        identity_db.upsert_tracklet(cam, tid, frame_idx), frame_idx, "body", feat, q,
                        pose="unknown", model_name=args.reid_name, model_version=Path(args.reid_model).name,
                        metadata={
                            "ambiguous": is_ambiguous,
                            "crop_width": int(crop.shape[1]), "crop_height": int(crop.shape[0]),
                            "box_width": float(item[1][2] - item[1][0]),
                            "box_height": float(item[1][3] - item[1][1]),
                            "clipped": bool(item[1][0] <= 1 or item[1][1] <= 1),
                            "view": "unknown",
                        },
                    )

        # -------------------------- SCRFD + AdaFace -----------------------------
        detection_frame = args.face_every > 0 and frame_idx % args.face_every == 0
        if detection_frame:
            pending: list[tuple[int, object, np.ndarray, float, float]] = []
            for box, tid_raw in zip(boxes, ids):
                tid = int(tid_raw)
                tracker = face_trackers[tid]
                preferred = tracker.snapshot().bbox
                face = detect_best_face_in_person(orig, box, face_detector, preferred_bbox=preferred)
                if face is None:
                    snap = tracker.miss()
                    temporal_debug.append({
                        "camera": cam, "frame": frame_idx, "track_id": tid, "detected": False,
                        "state": snap.state, "hits": snap.hits, "misses": snap.misses,
                        "detector_score": "", "face_size": "",
                    })
                    continue

                fw = float(face.bbox[2] - face.bbox[0])
                fh = float(face.bbox[3] - face.bbox[1])
                fsize = min(fw, fh)
                snap = tracker.observe(face.bbox, face.kps, face.score, frame_idx)
                temporal_debug.append({
                    "camera": cam, "frame": frame_idx, "track_id": tid, "detected": True,
                    "state": snap.state, "hits": snap.hits, "misses": snap.misses,
                    "detector_score": float(face.score), "face_size": float(fsize),
                })

                # Detection continuity and recognition evidence are separate.
                if fsize < args.min_face_detect_size:
                    continue
                aligned = align_face_bgr(orig, face.kps)
                if aligned is None:
                    continue
                blur = laplacian_blur_score(aligned)
                pending.append((tid, face, aligned, fsize, blur))

            if pending:
                feats, norms = face_embedder.encode([x[2] for x in pending])
                for item, feat, norm in zip(pending, feats, norms):
                    tid, face, _aligned, fsize, blur = item
                    q = compute_quality(face.score, fsize, blur, float(norm), face.kps, norm_ref=args.adaface_norm_ref)
                    hit = None
                    if q.tier != "reject":
                        obs = FaceObservation(
                            frame_idx, feat, q.score, q.tier, q.detector_score, q.face_size, q.blur, q.feature_norm,
                            q.pose.label, q.pose.yaw_proxy, q.pose.pitch_proxy, q.pose.roll_deg,
                        )
                        _balanced_face_insert(face_bank[tid], obs, args.max_face_samples)
                        if gallery:
                            hit = match_gallery(
                                feat, gallery, q.score,
                                base_threshold=args.identity_threshold,
                                base_margin=args.identity_margin,
                            )
                            identity_managers[tid].update(tid, frame_idx, hit, q.score, q.tier)

                        id_state_now = identity_managers[tid].final(tid)
                        face_auth = memory.observe_face(
                            (cam, tid), frame_idx, feat, q.score, q.pose.label, q.tier,
                            aligned_crop=_aligned,
                            gallery_employee=id_state_now.employee_id,
                            gallery_accepted=(id_state_now.state == "CONFIRMED" and bool(id_state_now.employee_id)),
                        )
                        embedding_store.add(EmbeddingRecord(
                            f"face:{cam}:{tid}:{frame_idx}", "face", cam, frame_idx, tid,
                            feat, q.score, memory.state_for_track((cam, tid)).person_id,
                            q.pose.label,
                        ))
                        identity_db.add_observation(
                            identity_db.upsert_tracklet(cam, tid, frame_idx), frame_idx, "face", feat, q.score,
                            pose=q.pose.label, feature_norm=float(norm), model_name="adaface_ir50",
                            model_version=Path(args.adaface_checkpoint).name,
                            metadata={
                                "tier": q.tier, "detector_score": q.detector_score,
                                "face_size": q.face_size, "blur": q.blur,
                                "yaw": q.pose.yaw_proxy, "pitch": q.pose.pitch_proxy,
                                "roll": q.pose.roll_deg,
                            },
                        )
                        # A good face is the strongest identity event. Capture
                        # the synchronized body view at the same frame so body
                        # learning is tied to the face-confirmed person rather
                        # than to clothing-only frames sampled independently.
                        if q.tier in {"support", "strong"} and frame_idx % args.reid_sample_every != 0:
                            body_crop = safe_crop(orig, box_by_tid[tid])
                            if body_crop is not None:
                                body_quality = _body_quality(box_by_tid[tid], orig.shape, conf_by_tid[tid])
                                body_features = reid_embedder.encode([body_crop])
                                if len(body_features):
                                    body_feature = body_features[0]
                                    memory.observe_body(
                                        (cam, tid), frame_idx, body_feature, body_quality,
                                        crop=body_crop, ambiguous=ambiguity_by_tid.get(tid, False),
                                    )
                                    identity_db.add_observation(
                                        identity_db.upsert_tracklet(cam, tid, frame_idx), frame_idx,
                                        "body", body_feature, body_quality, pose="unknown",
                                        model_name=args.reid_name, model_version=Path(args.reid_model).name,
                                        metadata={
                                            "face_synchronized": True,
                                            "face_quality": q.score,
                                            "crop_width": int(body_crop.shape[1]),
                                            "crop_height": int(body_crop.shape[0]),
                                            "view": "unknown",
                                        },
                                    )
                        if not ambiguity_by_tid.get(tid, False):
                            db_hits = canonical_db.search_identities(
                                [feat], "face", "adaface_ir50", Path(args.adaface_checkpoint).name,
                                exclude_tracklet_id=None, limit=2,
                            )
                            if db_hits:
                                best = db_hits[0]
                                second_score = db_hits[1].score if len(db_hits) > 1 else -1.0
                                current = memory.state_for_track((cam, tid))
                                weak_face = q.tier == "weak"
                                match_threshold = args.person_face_weak_threshold if weak_face else args.person_face_threshold
                                needed_votes = args.person_face_weak_confirmations if weak_face else args.person_face_confirmations
                                db_face_query_debug.append({
                                    "camera": cam, "frame": frame_idx, "track_id": tid,
                                    "tier": q.tier, "top1": best.identity_id,
                                    "score": best.score, "margin": best.score - second_score,
                                    "support": best.support, "threshold": match_threshold,
                                    "decision": "CANDIDATE" if best.score >= match_threshold else "BELOW_THRESHOLD",
                                })
                                if (
                                    frame_idx >= (current.identity_change_until if current is not None else -1)
                                    and
                                    best.identity_id in memory.profiles
                                    and (current is None or best.identity_id != current.person_id)
                                    and best.score >= match_threshold
                                ):
                                    candidates = db_face_candidates[tid]
                                    previous = candidates.get(best.identity_id)
                                    if previous and frame_idx - previous[2] <= args.person_face_candidate_gap:
                                        votes = previous[0] + 1
                                        score_sum = previous[1] + best.score
                                    else:
                                        votes = 1
                                        score_sum = best.score
                                    candidates[best.identity_id] = (votes, score_sum, frame_idx)
                                    ranked_votes = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)
                                    vote_margin = ranked_votes[0][1][0] - (ranked_votes[1][1][0] if len(ranked_votes) > 1 else 0)
                                    if (
                                        ranked_votes[0][0] == best.identity_id
                                        and votes >= needed_votes
                                        and vote_margin >= 2
                                        and current is not None
                                    ):
                                        memory.replace_track_identity(
                                            current, memory.profiles[best.identity_id],
                                            "online_db_face_reconciliation", score=score_sum / votes, authority="face",
                                            frame=frame_idx,
                                        )
                                        identity_db.ensure_identity(best.identity_id, state="VERIFIED")
                                        identity_db.assign(
                                            identity_db.upsert_tracklet(cam, tid, frame_idx), best.identity_id,
                                            frame_idx, "CONFIRMED", "online_db_face_reconciliation",
                                            score_sum / votes, vote_margin,
                                        )
                                        db_face_candidates.pop(tid, None)
                    else:
                        face_auth = None

                    anon_fields = face_auth.row_fields() if face_auth is not None else {}
                    face_debug.append({
                        "camera": cam, "frame": frame_idx, "track_id": tid,
                        "detector_score": q.detector_score, "face_size": q.face_size, "blur": q.blur,
                        "feature_norm": q.feature_norm, "quality": q.score, "tier": q.tier,
                        "pose": q.pose.label, "yaw_proxy": q.pose.yaw_proxy, "pitch_proxy": q.pose.pitch_proxy,
                        "roll_deg": q.pose.roll_deg, **hit_fields(hit), **anon_fields,
                    })

        # ------------------------- resolve + render online -----------------------
        for box, tid_raw in zip(boxes, ids):
            tid = int(tid_raw)
            key = (cam, tid)
            id_state = identity_managers[tid].final(tid)
            if id_state.state == "CONFIRMED" and id_state.employee_id:
                memory.confirm_employee(key, frame_idx, id_state.employee_id, source="gallery_multi_frame")
            memory.maybe_seed_person(key, frame_idx)
            online_state = memory.state_for_track(key)
            db_tracklet_id = identity_db.upsert_tracklet(cam, tid, frame_idx)
            online_pid = online_state.person_id if online_state is not None else None
            if online_pid is not None:
                identity_db.ensure_identity(online_pid)
            identity_db.assign(
                db_tracklet_id, online_pid, frame_idx, "PROVISIONAL",
                online_state.bind_reason if online_state and online_state.bind_reason else "unbound",
            )

            snap = face_trackers[tid].snapshot()
            row = current_rows[tid]
            _snapshot_to_row(row, snap)
            mem_label, mem_sub = _memory_overlay(memory, key)
            mem_state = memory.state_for_track(key)
            profile = memory.profile_for_track(key)
            row["online_person_id"] = mem_label
            row["online_memory_state"] = "BOUND" if mem_state and mem_state.person_id is not None else "UNBOUND"
            row["person_id"] = f"P{mem_state.person_id:03d}" if mem_state and mem_state.person_id is not None else ""
            row["person_bind_reason"] = mem_state.bind_reason if mem_state else ""
            row["memory_maturity"] = profile.maturity() if profile is not None else "UNBOUND"
            row["memory_confidence"] = round(profile.confidence(), 6) if profile is not None else 0.0
            diag = memory.face_diagnostic(key)
            if diag is not None:
                row.update(diag.row_fields())
                row["face_id_age_frames"] = int(frame_idx) - int(diag.frame)

            if snap.visible and snap.bbox is not None:
                fx1, fy1, fx2, fy2 = [int(round(v)) for v in snap.bbox]
                face_text = "FACE" if snap.state == "VISIBLE" else "FACE~"
                if diag is not None and diag.pose:
                    face_text += f" {diag.pose.upper()}"
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 1)
                cv2.putText(frame, face_text, (fx1, max(15, fy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.38, (0, 255, 0), 1, cv2.LINE_AA)

            x1, y1, x2, y2 = [int(v) for v in box]
            label = f"{mem_label} L:{tid} {_identity_text(id_state)}"
            sub = f"{mem_sub} FACE:{snap.state}"
            if diag is None:
                face_id_line = "FaceID:WAIT"
            elif diag.top1_person_id is not None and diag.top1_score is not None:
                margin_txt = f" m={diag.margin:.2f}" if diag.margin is not None else ""
                face_id_line = (
                    f"FaceID:P{diag.top1_person_id:03d} s={diag.top1_score:.2f}{margin_txt} "
                    f"{diag.decision} {diag.candidate_votes}/{max(diag.candidate_needed,1)}"
                )
            else:
                face_id_line = (
                    f"FaceID:NEW {diag.decision} "
                    f"{diag.candidate_votes}/{max(diag.candidate_needed,1)}"
                )
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 38)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, sub, (x1, max(20, y1 - 22)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.40, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, face_id_line, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.36, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(frame)

    if writer is not None:
        writer.release()
    spans = {tid: TrackSpan(first_seen[tid], last_seen[tid]) for tid in first_seen}
    print(
        f"[track] {cam}: frames={frame_idx + 1}, tracks={len(first_seen)}, face_tracks={len(face_bank)}, "
        f"person_ids={len(memory.profiles)}"
    )
    return rows, body_bank, face_bank, spans, identity_managers, face_debug, temporal_debug, db_face_query_debug


def choose_scrfd_model(value: str | None) -> Path:
    if value:
        p = Path(value)
        if not p.is_file():
            raise FileNotFoundError(p)
        return p
    preferred = ASSETS_ROOT / "models" / "scrfd" / "det_10g.onnx"
    fallback = ASSETS_ROOT / "models" / "scrfd" / "det_2.5g.onnx"
    if preferred.is_file():
        return preferred
    if fallback.is_file():
        print("[face] det_10g.onnx not found; reusing existing SCRFD 2.5G.")
        return fallback
    raise FileNotFoundError("No SCRFD model found. Run setup.ps1 first.")


def resolve_face_roi_input(scrfd_model: Path, requested: int) -> int:
    is_10g = "10g" in scrfd_model.name.lower()
    if requested <= 0:
        return 640 if is_10g else 320
    if is_10g and requested != 640:
        print(f"[face] SCRFD 10G expects 640x640; overriding ROI input {requested} -> 640.")
        return 640
    return int(requested)


def event_to_dict(e: IdentityEvent) -> dict:
    return {
        "frame": e.frame, "state": e.state, "employee_id": e.employee_id, "score": e.score,
        "margin": e.margin, "quality": e.quality, "reason": e.reason,
    }


def merge_render_events(track: TrackSummary, person_employee: str | None) -> list[IdentityEvent]:
    events = list(track.events)
    if person_employee and track.confirmed_employee is None:
        events.append(IdentityEvent(
            track.span.first, "REACQUIRED", person_employee, None, None, None,
            "person_memory_continuity",
        ))
    events.sort(key=lambda e: (e.frame, 0 if e.state == "REACQUIRED" else 1))
    return events


def state_at(events: list[IdentityEvent], frame: int) -> IdentityEvent | None:
    if not events:
        return None
    frames = [e.frame for e in events]
    idx = bisect.bisect_right(frames, frame) - 1
    return events[idx] if idx >= 0 else None


def render_identity(
    source: str,
    cam: str,
    rows: list[dict],
    mapping: dict[TrackKey, int],
    tracks: dict[TrackKey, TrackSummary],
    profiles: dict[int, object],
    memory: PersonMemory,
    output: Path,
) -> None:
    if source.isdigit():
        return
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_frame[int(row["frame"])].append(row)
    render_events: dict[TrackKey, list[IdentityEvent]] = {}
    for key, track in tracks.items():
        pid = mapping.get(key)
        ident = profiles.get(pid)
        render_events[key] = merge_render_events(track, getattr(ident, "employee_id", None))

    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for row in by_frame.get(idx, []):
            tid = int(row["track_id"])
            key = (cam, tid)
            pid_text = str(row.get("person_id", "") or "")
            if pid_text.startswith("P") and pid_text[1:].isdigit():
                pid = int(pid_text[1:])
            else:
                pid = memory.person_id_at(key, idx)
            mem_state = memory.state_for_track(key)
            ident = profiles.get(pid) if pid is not None else None
            event = state_at(render_events.get(key, []), idx)
            if event and event.employee_id:
                if event.state == "CONFIRMED":
                    identity_text = event.employee_id
                elif event.state == "REACQUIRED":
                    identity_text = f"{event.employee_id}?"
                else:
                    identity_text = f"{event.state}:{event.employee_id}"
            else:
                identity_text = "UNKNOWN"

            x1, y1, x2, y2 = [int(float(row[k])) for k in ("x1", "y1", "x2", "y2")]
            maturity = getattr(ident, "maturity", lambda: "PENDING")()
            confidence = getattr(ident, "confidence", lambda: 0.0)()
            public_id = f"P:{pid:03d}" if pid is not None else "P:---"
            label = f"{public_id} L:{tid} {identity_text}"
            status = "BOUND" if mem_state is not None and mem_state.person_id is not None else "UNBOUND"
            sub = f"MEM:{maturity} {confidence:.2f} {status} FACE:{row.get('face_state', 'ABSENT')}"
            top1 = str(row.get("face_id_top1", "") or "")
            score = row.get("face_id_top1_score", "")
            margin = row.get("face_id_margin", "")
            decision = str(row.get("face_id_decision", "") or "")
            votes = row.get("face_id_candidate_votes", "")
            needed = row.get("face_id_candidate_needed", "")
            if top1 and score != "":
                try:
                    score_txt = f"{float(score):.2f}"
                    margin_txt = f" m={float(margin):.2f}" if margin != "" else ""
                except (TypeError, ValueError):
                    score_txt, margin_txt = str(score), ""
                face_id_line = f"FaceID:{top1} s={score_txt}{margin_txt} {decision} {votes}/{needed}"
            elif decision:
                face_id_line = f"FaceID:NEW {decision} {votes}/{needed}"
            else:
                face_id_line = "FaceID:WAIT"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 38)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, sub, (x1, max(20, y1 - 22)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.40, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, face_id_line, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.36, (255, 255, 255), 1, cv2.LINE_AA)

            if row.get("face_state") in {"VISIBLE", "COASTING"} and row.get("face_x1", "") != "":
                fx1, fy1, fx2, fy2 = [int(float(row[k])) for k in ("face_x1", "face_y1", "face_x2", "face_y2")]
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 1)
        writer.write(frame)
        idx += 1
    cap.release()
    writer.release()


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Face-first person identity tracking")
    p.add_argument("--sources", nargs="+", required=True)
    p.add_argument("--output", default=str(PROJECT_ROOT / "runs" / "face_first_person_memory"))
    p.add_argument("--identity-store", default=None, help="Previous run embedding_store directory to restore P profiles")
    p.add_argument("--identity-db", default=None, help="SQLite identity database path; defaults to <output>/identity.sqlite")
    p.add_argument("--canonical-db", default=None, help="Read-only canonical reference DB")
    p.add_argument("--device", default="0")
    legacy_model = ASSETS_ROOT / "models" / "yolo" / "yolo11s.pt"
    p.add_argument("--yolo", default=str(legacy_model if legacy_model.is_file() else "yolo11s.pt"))
    p.add_argument("--tracker", default=str(PROJECT_ROOT / "tracker_botsort_reid.yaml"))
    # Important: tracker_low_thresh is 0.10. Detector postprocessing must not
    # remove those low-confidence observations before BoT-SORT receives them.
    p.add_argument("--conf", type=float, default=0.10)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--no-half", action="store_true")

    p.add_argument("--reid-model", default=str(ASSETS_ROOT / "models" / "osnet" / "osnet_x1_0_msmt17.pth"))
    p.add_argument("--reid-name", default="osnet_x1_0")
    p.add_argument("--reid-sample-every", type=int, default=4)
    p.add_argument("--max-reid-embeddings", type=int, default=128)
    p.add_argument("--body-prototypes", type=int, default=16)
    p.add_argument("--memory-min-body-quality", type=float, default=0.35)

    p.add_argument("--scrfd-model", default=None)
    p.add_argument("--face-roi-input", type=int, default=0, help="0=auto: 640 for SCRFD 10G, 320 for 2.5G")
    p.add_argument("--face-det-threshold", type=float, default=0.20)
    p.add_argument("--face-every", type=int, default=2)
    p.add_argument("--min-face-detect-size", type=float, default=16.0)
    p.add_argument("--max-face-samples", type=int, default=80)
    p.add_argument("--face-prototypes-per-pose", type=int, default=12)
    p.add_argument("--face-temporal-min-hits", type=int, default=2)
    p.add_argument("--face-temporal-max-misses", type=int, default=7)
    p.add_argument("--face-temporal-alpha", type=float, default=0.40)
    p.add_argument("--adaface-repo", default=str(PROJECT_ROOT / "third_party" / "AdaFace"))
    p.add_argument("--adaface-checkpoint", default=str(ASSETS_ROOT / "models" / "adaface" / "adaface_ir50_webface4m.ckpt"))
    p.add_argument("--adaface-norm-ref", type=float, default=20.0)

    # Optional real employee gallery. Without it, the exact same face pipeline
    # builds anonymous Pxxx face prototypes and representative crops.
    p.add_argument("--gallery", default=None)
    p.add_argument("--gallery-min-face-size", type=float, default=40.0)
    p.add_argument("--gallery-min-quality", type=float, default=0.42)
    p.add_argument("--identity-threshold", type=float, default=0.55)
    p.add_argument("--identity-margin", type=float, default=0.07)
    p.add_argument("--identity-min-quality", type=float, default=0.32)
    p.add_argument("--strong-quality", type=float, default=0.68)
    p.add_argument("--confirm-frames", type=int, default=3)
    p.add_argument("--confirm-weight", type=float, default=1.45)

    # P-memory: short continuity is separated from long-gap identity. Body is
    # only allowed to support short-gap continuity. Trusted face can reacquire
    # across long gaps and is the only cue that can confirm EMPxxx.
    p.add_argument("--person-short-gap", type=int, default=45)
    p.add_argument("--person-motion-only-gap", type=int, default=6)
    p.add_argument("--person-new-after", type=int, default=12)
    p.add_argument("--person-ambiguity-grace", type=int, default=8)
    p.add_argument("--person-face-threshold", type=float, default=0.56)
    p.add_argument("--person-face-margin", type=float, default=0.04)
    p.add_argument("--person-face-learn-quality", type=float, default=0.32)
    p.add_argument("--person-face-confirmations", type=int, default=2)
    p.add_argument("--person-face-weak-threshold", type=float, default=0.45)
    p.add_argument("--person-face-weak-confirmations", type=int, default=3)
    p.add_argument("--person-new-face-confirmations", type=int, default=2)
    p.add_argument("--person-face-candidate-gap", type=int, default=16)
    p.add_argument("--person-recent-face-frames", type=int, default=300)
    p.add_argument("--person-body-near", type=float, default=0.76)
    p.add_argument("--person-body-far", type=float, default=0.84)
    p.add_argument("--person-body-margin", type=float, default=0.05)
    p.add_argument("--person-body-novelty", type=float, default=0.91)
    p.add_argument("--person-body-jump", type=float, default=0.48)
    p.add_argument("--person-max-face-per-pose", type=int, default=8)
    p.add_argument("--person-max-body-views", type=int, default=24)
    p.add_argument("--require-face-before-person", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 76)
    print(" PERSON IDENTITY TRACKING - FACE ANCHOR AUTHORITY")
    print("=" * 76)
    print(f"[paths] project={PROJECT_ROOT}")
    print(f"[paths] workspace={WORKSPACE_ROOT}")
    print(f"[paths] assets={ASSETS_ROOT}")
    t0 = time.perf_counter()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch_device(args.device)
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("CUDA unavailable")
        print(f"[gpu] {torch.cuda.get_device_name(int(device.split(':')[-1]))}")

    reid_model = Path(args.reid_model)
    if not reid_model.is_file():
        raise FileNotFoundError(f"OSNet checkpoint not found: {reid_model}. Run setup.ps1 first.")
    scrfd_model = choose_scrfd_model(args.scrfd_model)
    face_roi_input = resolve_face_roi_input(scrfd_model, args.face_roi_input)
    print(f"[tracker] detector conf={args.conf:.2f}; low-score observations are preserved for tracker association")
    print(f"[face] SCRFD: {scrfd_model.name}, ROI input={face_roi_input}, det_threshold={args.face_det_threshold}")
    print(f"[face] temporal: min_hits={args.face_temporal_min_hits}, max_misses={args.face_temporal_max_misses}, alpha={args.face_temporal_alpha}")
    print(
        f"[person] P-memory: short_gap={args.person_short_gap}f, motion_only={args.person_motion_only_gap}f, "
        f"face={args.person_face_threshold:.2f}, body_near/far={args.person_body_near:.2f}/{args.person_body_far:.2f}"
    )
    print(
        f"[face-id] anonymous AdaFace: threshold={args.person_face_threshold:.2f}, margin={args.person_face_margin:.2f}, "
        f"confirm={args.person_face_confirmations} observations"
    )
    print("[person] authority: trusted FACE can split/correct P segments; BODY only supports short-gap continuity")

    face_detector = FaceDetector(str(scrfd_model), device=device, input_size=face_roi_input, threshold=args.face_det_threshold)
    face_embedder = AdaFaceQualityEmbedder(args.adaface_repo, args.adaface_checkpoint, device=device)
    reid_embedder = TorchReIDEmbedder(device=device, model_name=args.reid_name, model_path=str(reid_model))

    gallery = {}
    if args.gallery:
        gallery = build_gallery(
            args.gallery, face_detector, face_embedder,
            min_face_size=args.gallery_min_face_size,
            min_quality=args.gallery_min_quality,
            norm_ref=args.adaface_norm_ref,
        )
        print(f"[gallery] identities={len(gallery)}")
    else:
        print("[gallery] 0 identities: anonymous face profiles Pxxx will be learned")

    memory = PersonMemory(
        sample_root=out_dir / "person_memory",
        short_gap_frames=args.person_short_gap,
        motion_only_gap_frames=args.person_motion_only_gap,
        new_person_after_frames=args.person_new_after,
        ambiguity_grace_frames=args.person_ambiguity_grace,
        face_match_threshold=args.person_face_threshold,
        face_match_margin=args.person_face_margin,
        face_learn_quality=args.person_face_learn_quality,
        face_match_confirmations=args.person_face_confirmations,
        face_new_confirmations=args.person_new_face_confirmations,
        face_candidate_max_gap=args.person_face_candidate_gap,
        face_recent_anchor_frames=args.person_recent_face_frames,
        body_short_gap_near=args.person_body_near,
        body_short_gap_far=args.person_body_far,
        body_match_margin=args.person_body_margin,
        body_learn_quality=args.memory_min_body_quality,
        body_novelty_threshold=args.person_body_novelty,
        body_jump_threshold=args.person_body_jump,
        max_face_per_pose=args.person_max_face_per_pose,
        max_body_views=args.person_max_body_views,
        require_face_before_person=args.require_face_before_person,
    )
    embedding_store = EmbeddingStore(out_dir / "embedding_store")
    identity_db_path = Path(args.identity_db or (out_dir / "identity.sqlite")).resolve()
    if args.canonical_db and Path(args.canonical_db).resolve() == identity_db_path:
        raise SystemExit("--canonical-db and --identity-db must be different files")
    identity_db = IdentityDatabase(identity_db_path)
    canonical_db = IdentityDatabase(args.canonical_db, readonly=True) if args.canonical_db else identity_db
    evidence_fusion = TrackletEvidenceFusion(
        face_threshold=args.person_face_threshold,
        weak_face_threshold=args.person_face_weak_threshold,
        body_threshold=args.person_body_far,
        margin=args.person_face_margin,
    )
    restored_db_records = canonical_db.assigned_embedding_records()
    if restored_db_records:
        memory.import_embedding_records(restored_db_records)
        print(f"[identity-db] restored {len(restored_db_records)} assigned embeddings")
    if args.identity_store:
        previous_store = EmbeddingStore.load(args.identity_store)
        memory.import_embedding_records(previous_store.records)
        print(f"[identity] restored {len(previous_store.records)} embeddings from {args.identity_store}")

    all_rows: list[dict] = []
    rows_by_cam: dict[str, list[dict]] = {}
    sources_by_cam: dict[str, str] = {}
    tracks: dict[TrackKey, TrackSummary] = {}
    face_debug_all: list[dict] = []
    temporal_debug_all: list[dict] = []
    db_face_query_all: list[dict] = []

    for idx, source in enumerate(args.sources):
        cam = camera_name(source, idx)
        sources_by_cam[cam] = source
        rows, body_bank, face_bank, spans, managers, face_debug, temporal_debug, db_face_query = process_camera(
            source, cam, out_dir, args, face_detector, face_embedder, reid_embedder, gallery, memory, embedding_store, identity_db, canonical_db
        )
        rows_by_cam[cam] = rows
        all_rows.extend(rows)
        face_debug_all.extend(face_debug)
        temporal_debug_all.extend(temporal_debug)
        db_face_query_all.extend(db_face_query)

        for tid, span in spans.items():
            body_feats = body_bank.get(tid, [])
            body_protos = select_body_prototypes(
                np.stack(body_feats).astype(np.float32) if body_feats else np.empty((0, 512), dtype=np.float32),
                max_prototypes=args.body_prototypes,
            )
            face_protos = aggregate_face_observations(
                face_bank.get(tid, []),
                top_k_per_pose=args.face_prototypes_per_pose,
                min_quality=0.20,
            )
            st = managers[tid].final(tid)
            key = (cam, tid)
            tracks[key] = TrackSummary(
                key=key,
                span=span,
                face_prototypes=face_protos,
                body_prototypes=body_protos,
                confirmed_employee=st.employee_id if st.state == "CONFIRMED" else None,
                final_state=st.state,
                events=list(st.events),
            )

    # Persist provisional online assignments first. Raw observations remain
    # independent; these assignments may be replaced below.
    for key, st in memory.tracks.items():
        db_tracklet_id = identity_db.upsert_tracklet(key[0], key[1], st.last_frame)
        if st.person_id is not None:
            identity_db.ensure_identity(st.person_id)
        identity_db.assign(db_tracklet_id, st.person_id, st.first_frame, "PROVISIONAL", st.bind_reason or "unbound")

    # Reconcile completed tracklets after all observations are available. This
    # is deliberately before export/render so corrected segments reach every
    # downstream artifact consistently.
    for key in sorted(memory.tracks):
        track_observations = [x for x in embedding_store.records if x.camera == key[0] and x.track_id == key[1]]
        memory.reconcile_track_faces(key, track_observations)

    # Database-backed face/body fusion also covers long gaps where face-only
    # reconciliation has insufficient evidence.
    fusion_debug_by_key: dict[TrackKey, object] = {}
    for key, st in sorted(memory.tracks.items()):
        db_tracklet_id = identity_db.upsert_tracklet(key[0], key[1], st.last_frame)
        observations = [x for x in embedding_store.records if x.camera == key[0] and x.track_id == key[1]]
        face_vectors = [x.embedding for x in observations if x.kind == "face" and x.quality >= 0.20]
        body_vectors = [x.embedding for x in observations if x.kind == "body" and x.quality >= args.memory_min_body_quality]
        face_hits = canonical_db.search_identities(
            face_vectors, "face", "adaface_ir50", Path(args.adaface_checkpoint).name,
            exclude_tracklet_id=db_tracklet_id, limit=3,
        ) if face_vectors else []
        body_hits = canonical_db.search_identities(
            body_vectors, "body", args.reid_name, Path(args.reid_model).name,
            exclude_tracklet_id=db_tracklet_id, limit=3,
        ) if body_vectors else []
        evidence_by_pid: dict[int, IdentityEvidence] = {}
        for hit in face_hits:
            evidence_by_pid.setdefault(hit.identity_id, IdentityEvidence(hit.identity_id)).face_scores.append(hit.score)
            evidence_by_pid[hit.identity_id].face_support = hit.support
        for hit in body_hits:
            evidence_by_pid.setdefault(hit.identity_id, IdentityEvidence(hit.identity_id)).body_scores.append(hit.score)
            evidence_by_pid[hit.identity_id].body_support = hit.support
        decision = evidence_fusion.decide(evidence_by_pid.values(), current_id=st.person_id)
        fusion_debug_by_key[key] = decision
        identity_db.add_fusion_decision(db_tracklet_id, st.last_frame, decision)
        target_pid = decision.identity_id
        reason = f"db_{decision.reason.lower()}" if decision.identity_id is not None else ""
        score = decision.fusion_score
        if target_pid is not None and target_pid != st.person_id and target_pid in memory.profiles:
            overlapping = any(
                other.key[0] == key[0] and other.key != key and other.person_id == target_pid
                and other.first_frame <= st.last_frame and st.first_frame <= other.last_frame
                for other in memory.tracks.values()
            )
            if not overlapping:
                memory.replace_track_identity(
                    st, memory.profiles[target_pid], reason, score=score,
                    authority="face" if "face" in reason or "fusion" in reason else "continuity",
                )

    mapping = memory.mapping()
    profiles = memory.profiles
    for key, st in memory.tracks.items():
        db_tracklet_id = identity_db.upsert_tracklet(key[0], key[1], st.last_frame)
        for segment in sorted(st.segments, key=lambda item: item.start_frame):
            if segment.person_id is not None:
                identity_db.ensure_identity(segment.person_id)
            identity_db.assign(
                db_tracklet_id, segment.person_id, segment.start_frame,
                "CONFIRMED" if segment.authority == "face" else "PROVISIONAL",
                segment.reason,
            )
        if st.person_id is not None and memory.profiles[st.person_id].face_core():
            identity_db.promote_tracklet_references(db_tracklet_id, st.person_id)
    memory.save_metadata_all()
    embedding_store.save()
    print(f"[identity-db] integrity={identity_db.integrity_check()} path={identity_db.path}")
    identity_db.close()
    if canonical_db is not identity_db:
        canonical_db.close()

    export_rows: list[dict] = []
    render_event_cache: dict[TrackKey, list[IdentityEvent]] = {}
    for key, track in tracks.items():
        pid = mapping.get(key)
        profile = profiles.get(pid) if pid is not None else None
        render_event_cache[key] = merge_render_events(track, getattr(profile, "employee_id", None))

    for row in all_rows:
        key = (row["camera"], int(row["track_id"]))
        frame = int(row["frame"])
        resolved_pid = memory.person_id_at(key, frame)
        fusion_state = getattr(fusion_debug_by_key.get(key), "state", "")
        if fusion_state in {"CONFLICT", "UNKNOWN"}:
            resolved_pid = None
        pid = resolved_pid if resolved_pid is not None else None
        pid_text = f"P{pid:03d}" if pid is not None else ""
        st = memory.state_for_track(key)
        profile = profiles.get(pid) if pid is not None else None
        event = state_at(render_event_cache.get(key, []), frame)
        employee = getattr(profile, "employee_id", None)
        export_rows.append({
            **row,
            "person_id": pid_text,
            "identity": employee or (event.employee_id if event and event.employee_id else "UNKNOWN"),
            "identity_state": "CONFIRMED" if employee else (event.state if event else "UNIDENTIFIED"),
            "fusion_identity": getattr(fusion_debug_by_key.get(key), "identity_id", None),
            "fusion_state": getattr(fusion_debug_by_key.get(key), "state", ""),
            "fusion_reason": getattr(fusion_debug_by_key.get(key), "reason", ""),
            "fusion_face_score": getattr(fusion_debug_by_key.get(key), "face_score", None),
            "fusion_body_score": getattr(fusion_debug_by_key.get(key), "body_score", None),
            "fusion_score": getattr(fusion_debug_by_key.get(key), "fusion_score", None),
            "fusion_margin": getattr(fusion_debug_by_key.get(key), "margin", None),
            "identity_resolution_state": fusion_state,
        })
    save_csv(out_dir / "tracks.csv", export_rows)
    save_csv(out_dir / "face_observations.csv", face_debug_all)
    save_csv(out_dir / "face_temporal.csv", temporal_debug_all)
    save_csv(out_dir / "identity_db_face_queries.csv", db_face_query_all)
    face_id_rows = [{k: v for k, v in row.items() if k in {
        "camera", "frame", "track_id", "person_id", "face_state",
        "face_id_frame", "face_id_person", "face_id_decision", "face_id_accepted",
        "face_id_top1", "face_id_top1_score", "face_id_top2", "face_id_top2_score",
        "face_id_margin", "face_id_candidate_votes", "face_id_candidate_needed",
        "face_id_pose", "face_id_tier", "face_id_quality", "face_id_age_frames",
    }} for row in face_debug_all if row.get("face_id_decision")]
    save_csv(out_dir / "face_identity_diagnostics.csv", face_id_rows)

    events_payload = []
    summaries_payload = []
    for key, track in sorted(tracks.items()):
        pid = mapping.get(key)
        st = memory.state_for_track(key)
        profile = profiles.get(pid) if pid is not None else None
        for e in render_event_cache.get(key, []):
            events_payload.append({
                "camera": key[0], "track_id": key[1],
                "person_id": f"P{pid:03d}" if pid is not None else None,
                **event_to_dict(e),
            })
        summaries_payload.append({
            "camera": key[0], "track_id": key[1],
            "person_id": f"P{pid:03d}" if pid is not None else None,
            "bind_reason": st.bind_reason if st else None,
            "first_frame": track.span.first, "last_frame": track.span.last,
            "confirmed_employee": getattr(profile, "employee_id", None) or track.confirmed_employee,
            "final_state": track.final_state,
            "face_prototypes": [{"pose": x.pose, "quality": x.quality, "count": x.count} for x in track.face_prototypes],
            "body_prototypes": len(track.body_prototypes),
            "online_face_observations": st.face_observations if st else 0,
            "online_body_observations": st.body_observations if st else 0,
            "ambiguous_frames": st.ambiguous_frames if st else 0,
            "strong_face_conflict": st.strong_face_conflict if st else False,
        })
    (out_dir / "identity_events.json").write_text(json.dumps(events_payload, indent=2), encoding="utf-8")
    (out_dir / "track_summaries.json").write_text(json.dumps(summaries_payload, indent=2), encoding="utf-8")

    memory_payload = memory.payload()
    (out_dir / "person_memory.json").write_text(json.dumps(memory_payload, indent=2), encoding="utf-8")

    # Track-level embeddings remain available for diagnostics.
    face_npz = {}
    body_npz = {}
    for key, track in tracks.items():
        prefix = f"{key[0]}__{key[1]}"
        for i, proto in enumerate(track.face_prototypes):
            face_npz[f"{prefix}__{proto.pose}_{i}"] = proto.embedding
        for i, proto in enumerate(track.body_prototypes):
            body_npz[f"{prefix}__{i}"] = proto
    np.savez_compressed(out_dir / "face_prototypes.npz", **face_npz)
    np.savez_compressed(out_dir / "body_prototypes.npz", **body_npz)

    person_npz = {}
    for pid, profile in sorted(profiles.items()):
        prefix = f"P{pid:03d}"
        for i, proto in enumerate(profile.face_bank):
            person_npz[f"{prefix}__face__{proto.pose}_{i}"] = proto.embedding
        for i, proto in enumerate(profile.body_bank):
            person_npz[f"{prefix}__body__{i}"] = proto.embedding
    np.savez_compressed(out_dir / "person_memory_prototypes.npz", **person_npz)

    for cam, source in sources_by_cam.items():
        render_identity(
            source, cam, rows_by_cam[cam], mapping, tracks, profiles, memory,
            out_dir / f"{cam}_identity.mp4",
        )

    elapsed = time.perf_counter() - t0
    confirmed_profiles = sum(1 for x in profiles.values() if x.employee_id)
    reacq = sum(x.reacquire_count for x in profiles.values())
    face_anchored = sum(1 for x in profiles.values() if x.face_anchor_count > 0)
    unbound = sum(1 for x in memory.tracks.values() if x.person_id is None)
    print(
        f"[done] tracks={len(tracks)}, person_ids={len(profiles)}, unbound_tracks={unbound}, "
        f"face_anchored_profiles={face_anchored}, confirmed_profiles={confirmed_profiles}, gallery={len(gallery)}"
    )
    print(
        f"[done] short_gap_reacquisitions={reacq}, face_observations={len(face_debug_all)}, elapsed={elapsed:.1f}s"
    )
    print(f"[done] samples={out_dir / 'person_memory'}")
    print(f"[done] output={out_dir}")


if __name__ == "__main__":
    main()
