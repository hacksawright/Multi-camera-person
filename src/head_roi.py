from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RoiFace:
    bbox: np.ndarray
    score: float
    kps: np.ndarray


def head_roi(box: np.ndarray, frame_shape: tuple[int, int, int]) -> tuple[int, int, int, int] | None:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    pw, ph = x2 - x1, y2 - y1
    if pw < 12 or ph < 28:
        return None
    # Wider than the old ROI so a face that turns left/right does not leave the
    # search region. The temporal face prior below prevents the wider crop from
    # simply preferring a nearby person's face.
    rx1 = max(0, int(round(x1 - 0.30 * pw)))
    rx2 = min(w, int(round(x2 + 0.30 * pw)))
    ry1 = max(0, int(round(y1 - 0.16 * ph)))
    ry2 = min(h, int(round(y1 + 0.52 * ph)))
    if rx2 - rx1 < 12 or ry2 - ry1 < 12:
        return None
    return rx1, ry1, rx2, ry2


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / max(aa + bb - inter, 1e-6))


def detect_best_face_in_person(
    frame: np.ndarray,
    box: np.ndarray,
    detector,
    preferred_bbox: np.ndarray | None = None,
) -> RoiFace | None:
    roi = head_roi(box, frame.shape)
    if roi is None:
        return None
    x1, y1, x2, y2 = roi
    crop = frame[y1:y2, x1:x2]
    faces = detector.detect(crop)
    if not faces:
        return None

    candidates: list[tuple[float, object, np.ndarray]] = []
    px1, py1, px2, py2 = [float(v) for v in box]
    for face in faces:
        global_bbox = face.bbox.copy()
        global_bbox[[0, 2]] += x1
        global_bbox[[1, 3]] += y1
        fw = max(float(global_bbox[2] - global_bbox[0]), 1.0)
        fh = max(float(global_bbox[3] - global_bbox[1]), 1.0)
        fcx = 0.5 * float(global_bbox[0] + global_bbox[2])
        fcy = 0.5 * float(global_bbox[1] + global_bbox[3])
        # A face candidate must belong to the enclosing person detection. The
        # ROI is intentionally broad, so this containment check prevents a
        # neighbouring person's face from becoming identity evidence.
        if not (px1 <= fcx <= px2 and py1 <= fcy <= py2):
            continue
        base = float(face.score) * (fw * fh) ** 0.25
        if preferred_bbox is not None:
            pb = np.asarray(preferred_bbox, dtype=np.float32).reshape(4)
            iou = _iou(global_bbox, pb)
            gcx = fcx
            gcy = fcy
            pcx = 0.5 * float(pb[0] + pb[2])
            pcy = 0.5 * float(pb[1] + pb[3])
            pdiag = max(float(np.hypot(pb[2] - pb[0], pb[3] - pb[1])), 1.0)
            dist = float(np.hypot(gcx - pcx, gcy - pcy)) / pdiag
            temporal = 1.0 + 1.8 * iou + max(0.0, 0.8 - dist)
            base *= temporal
        candidates.append((base, face, global_bbox))

    if not candidates:
        return None
    _, face, bbox = max(candidates, key=lambda x: x[0])
    kps = face.kps.copy()
    kps[:, 0] += x1
    kps[:, 1] += y1
    return RoiFace(bbox.astype(np.float32), float(face.score), kps.astype(np.float32))
