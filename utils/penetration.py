"""
penetration.py — Phát hiện khớp/xương xuyên thân và chỉnh trọng số fusion.

Ý tưởng:
    Nếu một khớp (hoặc xương) của nguồn A đi xuyên qua đoạn nối hai điểm trên thân,
    mà nguồn B (góc nhìn bổ sung) không xuyên, thì thay khớp đó bằng nguồn B.
    Pose fused vẫn bị phạt nặng nếu còn xuyên.
"""

from __future__ import annotations

import torch

from utils.geometry import interpolate_axis_angle

# COCO-17
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_ELBOW, _R_ELBOW = 7, 8
_L_WRIST, _R_WRIST = 9, 10
_L_HIP, _R_HIP = 11, 12
_L_KNEE, _R_KNEE = 13, 14
_L_ANKLE, _R_ANKLE = 15, 16
_NOSE = 0

# Đoạn "thân" — hai điểm trên cơ thể mà khớp khác không được cắt xuyên
TORSO_SEGMENTS = [
    (_L_SHOULDER, _R_SHOULDER),
    (_L_HIP, _R_HIP),
    (_L_SHOULDER, _L_HIP),
    (_R_SHOULDER, _R_HIP),
    (_L_SHOULDER, _R_HIP),
    (_R_SHOULDER, _L_HIP),
]

# Xương tay (thường xuyên bụng/ngực khi HMR4D sai chiều sâu)
LIMB_SEGMENTS = [
    (_L_SHOULDER, _L_ELBOW),
    (_L_ELBOW, _L_WRIST),
    (_R_SHOULDER, _R_ELBOW),
    (_R_ELBOW, _R_WRIST),
]

# Xương chân (kiểm tra va chạm chân trái vs chân phải)
LEFT_LEG_SEGMENTS = [(_L_HIP, _L_KNEE), (_L_KNEE, _L_ANKLE)]
RIGHT_LEG_SEGMENTS = [(_R_HIP, _R_KNEE), (_R_KNEE, _R_ANKLE)]

# Khớp cần kiểm tra "điểm cắt xuyên đoạn thân"
PROBE_JOINTS = (_L_ELBOW, _R_ELBOW, _L_WRIST, _R_WRIST, _L_KNEE, _R_KNEE, _L_ANKLE, _R_ANKLE)

# COCO → index body_pose SMPL (21 khớp, không gồm global_orient)
COCO_TO_SMPL_BODY = {
    5: 15, 6: 16,
    7: 17, 8: 18,
    9: 19, 10: 20,
    11: 0, 12: 1,
    13: 3, 14: 4,
    15: 6, 16: 7,
}


def _segment_segment_distance(
    p0: torch.Tensor,
    p1: torch.Tensor,
    q0: torch.Tensor,
    q1: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Khoảng cách gần nhất giữa hai đoạn thẳng. Input (B, 3) → (B,)."""
    u = p1 - p0
    v = q1 - q0
    w = p0 - q0
    a = (u * u).sum(dim=-1, keepdim=True)
    b = (u * v).sum(dim=-1, keepdim=True)
    c = (v * v).sum(dim=-1, keepdim=True)
    d = (u * w).sum(dim=-1, keepdim=True)
    e = (v * w).sum(dim=-1, keepdim=True)
    denom = a * c - b * b
    parallel = denom.abs() < eps
    s = torch.where(parallel, torch.zeros_like(a), (b * e - c * d) / denom.clamp(min=eps))
    t = torch.where(parallel, torch.zeros_like(a), (a * e - b * d) / denom.clamp(min=eps))
    s = s.clamp(0.0, 1.0)
    t = t.clamp(0.0, 1.0)
    closest_p = p0 + s * u
    closest_q = q0 + t * v
    return torch.linalg.norm(closest_p - closest_q, dim=-1)


def _point_segment_distance(
    point: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Khoảng cách từ điểm tới đoạn AB. (B, 3) → (B,)."""
    ab = b - a
    ab2 = (ab * ab).sum(dim=-1, keepdim=True).clamp(min=eps)
    t = ((point - a) * ab).sum(dim=-1, keepdim=True) / ab2
    t = t.clamp(0.0, 1.0)
    closest = a + t * ab
    return torch.linalg.norm(point - closest, dim=-1)


def joint_penetration_scores(
    joints_3d: torch.Tensor,
    radius_m: float = 0.10,
    sharpness: float = 30.0,
) -> torch.Tensor:
    """
    Điểm xuyên thân theo khớp COCO, trong (0, 1). Gần 1 = đang xuyên.

    joints_3d: (B, J, 3), J >= 17.
    """
    coco = joints_3d[:, :17, :]
    B = coco.shape[0]
    device = coco.device
    dtype = coco.dtype
    min_dist = [torch.full((B,), 1e6, device=device, dtype=dtype) for _ in range(17)]

    for j in PROBE_JOINTS:
        point = coco[:, j, :]
        for a_idx, b_idx in TORSO_SEGMENTS:
            if j in (a_idx, b_idx):
                continue
            dist = _point_segment_distance(point, coco[:, a_idx, :], coco[:, b_idx, :])
            min_dist[j] = torch.minimum(min_dist[j], dist)

    for i0, i1 in LIMB_SEGMENTS:
        p0, p1 = coco[:, i0, :], coco[:, i1, :]
        for t0, t1 in TORSO_SEGMENTS:
            if len({i0, i1, t0, t1}) < 4:
                continue
            dist = _segment_segment_distance(p0, p1, coco[:, t0, :], coco[:, t1, :])
            min_dist[i0] = torch.minimum(min_dist[i0], dist)
            min_dist[i1] = torch.minimum(min_dist[i1], dist)

    # Va chạm giữa 2 chân (chân trái vs chân phải)
    for i0, i1 in LEFT_LEG_SEGMENTS:
        p0, p1 = coco[:, i0, :], coco[:, i1, :]
        for j0, j1 in RIGHT_LEG_SEGMENTS:
            dist = _segment_segment_distance(p0, p1, coco[:, j0, :], coco[:, j1, :])
            min_dist[i0] = torch.minimum(min_dist[i0], dist)
            min_dist[i1] = torch.minimum(min_dist[i1], dist)
            min_dist[j0] = torch.minimum(min_dist[j0], dist)
            min_dist[j1] = torch.minimum(min_dist[j1], dist)

    head = coco[:, _NOSE, :]
    if joints_3d.shape[1] > 17:
        head = 0.5 * (head + joints_3d[:, 17, :])
    for j in (_L_WRIST, _R_WRIST, _L_ELBOW, _R_ELBOW):
        d_head = torch.linalg.norm(coco[:, j, :] - head, dim=-1)
        min_dist[j] = torch.minimum(min_dist[j], d_head)

    min_dist = torch.stack(min_dist, dim=1)
    scores = torch.sigmoid(sharpness * (radius_m - min_dist))
    scores = torch.where(min_dist < 1e5, scores, torch.zeros_like(scores))
    return scores


def coco_scores_to_smpl_body(coco_scores: torch.Tensor) -> torch.Tensor:
    """(B, 17) → (B, 21, 1) để nhân với mirror_weight."""
    smpl_scores = []
    for coco_idx, smpl_idx in COCO_TO_SMPL_BODY.items():
        while len(smpl_scores) <= smpl_idx:
            smpl_scores.append(coco_scores.new_zeros(coco_scores.shape[0]))
        smpl_scores[smpl_idx] = torch.maximum(
            smpl_scores[smpl_idx], coco_scores[:, coco_idx]
        )
    # Xuyên cẳng tay → cũng kéo khuỷu/vai cùng chuỗi
    while len(smpl_scores) <= 18:
        smpl_scores.append(coco_scores.new_zeros(coco_scores.shape[0]))
    smpl_scores[17] = torch.maximum(smpl_scores[17], coco_scores[:, _L_WRIST])
    smpl_scores[18] = torch.maximum(smpl_scores[18], coco_scores[:, _R_WRIST])
    while len(smpl_scores) < 21:
        smpl_scores.append(coco_scores.new_zeros(coco_scores.shape[0]))
    return torch.stack(smpl_scores, dim=1).unsqueeze(-1)


def penetration_aware_mirror_weight(
    learned_weight: torch.Tensor,
    joints_real: torch.Tensor,
    joints_mirror: torch.Tensor,
    radius_m: float = 0.10,
    sharpness: float = 30.0,
    replace_strength: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    learned_weight: (B, 21, 1) — càng lớn càng tin gương.

    Real xuyên + mirror sạch → trọng số gương → 1.
    Mirror xuyên + real sạch → trọng số gương → 0.
    """
    score_real = coco_scores_to_smpl_body(
        joint_penetration_scores(joints_real, radius_m, sharpness)
    )
    score_mirror = coco_scores_to_smpl_body(
        joint_penetration_scores(joints_mirror, radius_m, sharpness)
    )
    prefer_mirror = replace_strength * score_real * (1.0 - score_mirror)
    prefer_real = replace_strength * score_mirror * (1.0 - score_real)
    override = (prefer_mirror + prefer_real).clamp(0.0, 1.0)
    adjusted = (1.0 - override) * learned_weight + prefer_mirror
    return adjusted.clamp(0.0, 1.0), prefer_mirror, prefer_real


def compute_penetration_loss(
    joints_3d: torch.Tensor,
    radius_m: float = 0.10,
    sharpness: float = 30.0,
) -> torch.Tensor:
    """Phạt nặng khớp fused còn xuyên thân. Scalar."""
    scores = joint_penetration_scores(joints_3d, radius_m, sharpness)
    return torch.mean(scores ** 2)


def fuse_with_penetration(
    pose_real: torch.Tensor,
    pose_mirror: torch.Tensor,
    global_real: torch.Tensor,
    global_mirror: torch.Tensor,
    learned_weight: torch.Tensor,
    joints_real: torch.Tensor,
    joints_mirror: torch.Tensor,
    radius_m: float = 0.10,
    sharpness: float = 30.0,
    replace_strength: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    SLERP lại sau khi chỉnh trọng số theo xuyên thân.

    Returns:
        fused_bp (B, 63), fused_go (B, 3), adj_weight (B, 21, 1), prefer_mirror (B, 21, 1)
    """
    adj_w, prefer_mirror, _ = penetration_aware_mirror_weight(
        learned_weight,
        joints_real,
        joints_mirror,
        radius_m=radius_m,
        sharpness=sharpness,
        replace_strength=replace_strength,
    )
    fused_pose = interpolate_axis_angle(pose_real, pose_mirror, adj_w)
    fused_go = interpolate_axis_angle(
        global_real,
        global_mirror,
        adj_w.mean(dim=(1, 2), keepdim=False).unsqueeze(-1),
    )
    fused_bp = fused_pose.reshape(pose_real.shape[0], -1)
    return fused_bp, fused_go, adj_w, prefer_mirror
