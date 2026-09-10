from __future__ import annotations

import numpy as np

from .models import FaceObservation, FacePrototype


def normalize(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(x))
    return x if n <= 1e-12 else x / n


def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    return float(np.dot(normalize(a), normalize(b)))


def aggregate_face_observations(
    observations: list[FaceObservation],
    top_k_per_pose: int = 6,
    min_quality: float = 0.20,
) -> list[FacePrototype]:
    out: list[FacePrototype] = []
    for pose in ("front", "left", "right"):
        group = [x for x in observations if x.pose == pose and x.quality >= min_quality]
        group.sort(key=lambda x: x.quality, reverse=True)
        group = group[:top_k_per_pose]
        if not group:
            continue
        feats = np.stack([normalize(x.embedding) for x in group]).astype(np.float32)
        prototype_quality = max(x.quality for x in group)
        weights = np.asarray([max(x.quality, 1e-3) for x in group], dtype=np.float32)

        # Robust outlier pruning: build a provisional centroid and discard samples
        # that disagree strongly with the better face observations in the same pose.
        centroid = normalize(np.sum(feats * weights[:, None], axis=0))
        sims = feats @ centroid
        if len(group) >= 3:
            keep = sims >= max(float(np.median(sims)) - 0.15, 0.20)
            if np.any(keep):
                feats = feats[keep]
                weights = weights[keep]
        weights /= max(float(weights.sum()), 1e-12)
        emb = normalize(np.sum(feats * weights[:, None], axis=0))
        out.append(FacePrototype(emb.astype(np.float32), float(prototype_quality), pose, int(len(feats))))
    return out


def select_body_prototypes(features: np.ndarray, max_prototypes: int = 4) -> list[np.ndarray]:
    if features.size == 0:
        return []
    feats = np.asarray(features, dtype=np.float32)
    feats /= np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12)
    selected: list[np.ndarray] = []
    # Keep a central prototype first, then diverse observations. This is more
    # robust than averaging every crop into one clothing-dominated vector.
    centroid = normalize(feats.mean(axis=0))
    selected.append(centroid.astype(np.float32))
    while len(selected) < max_prototypes and len(selected) < len(feats):
        sims = np.stack([feats @ p for p in selected], axis=1)
        distance_to_set = 1.0 - sims.max(axis=1)
        idx = int(np.argmax(distance_to_set))
        candidate = feats[idx]
        if max(float(candidate @ p) for p in selected) > 0.985:
            break
        selected.append(candidate.astype(np.float32))
    return selected
