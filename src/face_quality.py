from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class PoseInfo:
    label: str
    yaw_proxy: float
    pitch_proxy: float
    roll_deg: float
    landmark_score: float


@dataclass(frozen=True)
class QualityInfo:
    score: float
    tier: str
    detector_score: float
    face_size: float
    blur: float
    feature_norm: float
    pose: PoseInfo


def _clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def laplacian_blur_score(face_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def estimate_pose(kps: np.ndarray) -> PoseInfo:
    """Cheap 5-landmark pose/geometry proxy.

    This is deliberately not a full head-pose solver. It is used only for
    quality weighting and pose-bucketed temporal aggregation.
    """
    p = np.asarray(kps, dtype=np.float32).reshape(5, 2)
    le, re, nose, lm, rm = p
    eye_vec = re - le
    eye_dist = max(float(np.linalg.norm(eye_vec)), 1e-6)
    eye_mid = (le + re) * 0.5
    mouth_mid = (lm + rm) * 0.5
    mouth_dist = max(float(np.linalg.norm(rm - lm)), 1e-6)
    vertical = max(float(np.linalg.norm(mouth_mid - eye_mid)), 1e-6)

    yaw_proxy = float((nose[0] - eye_mid[0]) / (0.5 * eye_dist))
    pitch_proxy = float(((nose[1] - eye_mid[1]) / vertical - 0.46) / 0.28)
    roll_deg = float(math.degrees(math.atan2(float(eye_vec[1]), float(eye_vec[0]))))

    # Landmark plausibility. Broad bounds are intentional for profile faces.
    eye_ratio = eye_dist / vertical
    mouth_ratio = mouth_dist / eye_dist
    geom = 1.0
    if eye_ratio < 0.55 or eye_ratio > 2.2:
        geom *= 0.55
    if mouth_ratio < 0.45 or mouth_ratio > 1.55:
        geom *= 0.65
    if nose[1] <= eye_mid[1] or nose[1] >= mouth_mid[1] + 0.35 * vertical:
        geom *= 0.55
    if abs(roll_deg) > 45:
        geom *= 0.70
    landmark_score = _clip01(geom)

    if yaw_proxy < -0.25:
        label = "left"
    elif yaw_proxy > 0.25:
        label = "right"
    else:
        label = "front"
    return PoseInfo(label, yaw_proxy, pitch_proxy, roll_deg, landmark_score)


def compute_quality(
    detector_score: float,
    face_size: float,
    blur: float,
    feature_norm: float,
    kps: np.ndarray,
    norm_ref: float = 20.0,
) -> QualityInfo:
    pose = estimate_pose(kps)

    # Face-pixel quality: 18 px is barely usable; 64+ px saturates.
    size_score = _clip01((float(face_size) - 18.0) / 46.0)
    # Blur is computed after alignment to 112x112. Log scaling is less brittle
    # than one fixed Laplacian threshold across camera distances.
    blur_score = _clip01(math.log1p(max(float(blur), 0.0)) / math.log1p(180.0))
    yaw_penalty = math.exp(-0.55 * min(abs(pose.yaw_proxy), 2.0))
    pitch_penalty = math.exp(-0.35 * min(abs(pose.pitch_proxy), 2.0))
    roll_penalty = math.exp(-0.35 * min(abs(pose.roll_deg) / 35.0, 2.0))
    pose_score = _clip01(yaw_penalty * pitch_penalty * roll_penalty)
    norm_score = _clip01(float(feature_norm) / max(float(feature_norm) + norm_ref, 1e-6))

    # Norm is intentionally only 10%: it is a useful AdaFace quality proxy,
    # but must be calibrated on real cameras before it can become a hard gate.
    q = (
        0.26 * _clip01(detector_score)
        + 0.23 * size_score
        + 0.14 * blur_score
        + 0.14 * pose_score
        + 0.13 * pose.landmark_score
        + 0.10 * norm_score
    )
    q = _clip01(q)

    if face_size < 18 or q < 0.18:
        tier = "reject"
    elif face_size < 32 or q < 0.45:
        tier = "weak"
    elif face_size < 48 or q < 0.68:
        tier = "support"
    else:
        tier = "strong"
    return QualityInfo(q, tier, float(detector_score), float(face_size), float(blur), float(feature_norm), pose)
