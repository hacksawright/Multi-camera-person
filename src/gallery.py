from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .face_quality import compute_quality, laplacian_blur_score
from .models import FaceObservation, FacePrototype
from .prototypes import aggregate_face_observations, normalize


@dataclass
class GalleryIdentity:
    name: str
    prototypes: list[FacePrototype]
    image_count: int


@dataclass
class GalleryHit:
    name: str | None
    score: float
    second_score: float
    margin: float
    accepted: bool
    threshold: float
    margin_threshold: float


def _align_face_bgr(frame_bgr: np.ndarray, kps: np.ndarray, output_size: int = 112) -> np.ndarray | None:
    dst = np.array(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
        dtype=np.float32,
    )
    src = np.asarray(kps, dtype=np.float32).reshape(5, 2)
    m, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if m is None:
        return None
    return cv2.warpAffine(frame_bgr, m, (output_size, output_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def build_gallery(
    root: str,
    detector,
    embedder,
    min_face_size: float = 32.0,
    min_quality: float = 0.42,
    norm_ref: float = 20.0,
) -> dict[str, GalleryIdentity]:
    gallery_root = Path(root)
    if not gallery_root.is_dir():
        raise FileNotFoundError(f"Gallery directory not found: {gallery_root}")
    out: dict[str, GalleryIdentity] = {}
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for person_dir in sorted(p for p in gallery_root.iterdir() if p.is_dir()):
        pending: list[tuple[int, np.ndarray, object, np.ndarray, float, float]] = []
        for idx, path in enumerate(sorted(person_dir.iterdir())):
            if path.suffix.lower() not in valid_ext:
                continue
            image = cv2.imread(str(path))
            if image is None:
                continue
            faces = detector.detect(image)
            if not faces:
                continue
            face = max(faces, key=lambda f: float(f.score) * max(float((f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])), 1.0))
            fsize = min(float(face.bbox[2] - face.bbox[0]), float(face.bbox[3] - face.bbox[1]))
            if fsize < min_face_size:
                continue
            aligned = _align_face_bgr(image, face.kps)
            if aligned is None:
                continue
            blur = laplacian_blur_score(aligned)
            pending.append((idx, aligned, face, image, fsize, blur))
        if not pending:
            print(f"[gallery] {person_dir.name}: no usable faces")
            continue
        feats, norms = embedder.encode([x[1] for x in pending])
        observations: list[FaceObservation] = []
        for item, feat, norm in zip(pending, feats, norms):
            idx, _aligned, face, _image, fsize, blur = item
            q = compute_quality(face.score, fsize, blur, float(norm), face.kps, norm_ref=norm_ref)
            if q.score < min_quality or q.tier == "reject":
                continue
            observations.append(FaceObservation(idx, normalize(feat), q.score, q.tier, q.detector_score, q.face_size, q.blur,
                                               q.feature_norm, q.pose.label, q.pose.yaw_proxy, q.pose.pitch_proxy, q.pose.roll_deg))
        prototypes = aggregate_face_observations(observations, top_k_per_pose=8, min_quality=min_quality)
        if prototypes:
            out[person_dir.name] = GalleryIdentity(person_dir.name, prototypes, len(observations))
            print(f"[gallery] {person_dir.name}: images={len(observations)}, prototypes={len(prototypes)}")
    return out


def _identity_score(query: np.ndarray, ident: GalleryIdentity) -> float:
    q = normalize(query)
    return max(float(q @ p.embedding) for p in ident.prototypes)


def match_gallery(
    query: np.ndarray,
    gallery: dict[str, GalleryIdentity],
    quality: float,
    base_threshold: float = 0.55,
    base_margin: float = 0.07,
) -> GalleryHit | None:
    if not gallery:
        return None
    scored = sorted(((_identity_score(query, ident), name) for name, ident in gallery.items()), reverse=True)
    best_score, best_name = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else -1.0
    margin = best_score - second_score

    # Low-quality faces must clear a stricter bar instead of receiving a lower
    # threshold merely because the image is difficult.
    penalty = max(0.0, 0.65 - float(quality))
    threshold = float(base_threshold + 0.12 * penalty)
    margin_threshold = float(base_margin + 0.06 * penalty)
    accepted = best_score >= threshold and margin >= margin_threshold
    return GalleryHit(best_name, float(best_score), float(second_score), float(margin), accepted, threshold, margin_threshold)
