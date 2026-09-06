"""2D skeleton calibration helpers for visualization."""

from __future__ import annotations

import numpy as np


COCO_ANKLE_INDICES = (15, 16)


def align_skeleton_to_keypoints(
    projected: np.ndarray,
    detected: np.ndarray,
    confidence: np.ndarray,
    min_confidence: float = 0.2,
    max_scale_change: float = 0.6,
) -> np.ndarray:
    """Fit a scale and translation from projected joints to ViTPose joints.

    The transform is deliberately limited to the visualization path. It removes
    monocular body-scale drift while keeping the 3D SMPL result untouched.
    Ankles receive extra weight so the rendered feet follow the detected floor
    contact instead of floating above it.
    """
    projected = np.asarray(projected, dtype=np.float32)
    detected = np.asarray(detected, dtype=np.float32)
    confidence = np.asarray(confidence, dtype=np.float32)

    if projected.shape != detected.shape or projected.ndim != 2 or projected.shape[1] != 2:
        raise ValueError("projected and detected must both have shape (J, 2)")
    if confidence.shape != (projected.shape[0],):
        raise ValueError("confidence must have shape (J,)")

    valid = (
        (confidence >= min_confidence)
        & np.isfinite(confidence)
        & np.isfinite(projected).all(axis=1)
        & np.isfinite(detected).all(axis=1)
    )
    if valid.sum() < 3:
        return projected.copy()

    weights = confidence[valid].copy()
    valid_indices = np.flatnonzero(valid)
    weights[np.isin(valid_indices, COCO_ANKLE_INDICES)] *= 4.0
    weights /= weights.sum()

    source = projected[valid]
    target = detected[valid]
    source_center = np.sum(source * weights[:, None], axis=0)
    target_center = np.sum(target * weights[:, None], axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    denominator = np.sum(weights * np.sum(source_centered**2, axis=1))
    if denominator <= 1e-6:
        return projected.copy()

    scale = np.sum(weights * np.sum(source_centered * target_centered, axis=1)) / denominator
    scale = float(np.clip(scale, 1.0 - max_scale_change, 1.0 + max_scale_change))
    translation = target_center - scale * source_center
    return projected * scale + translation