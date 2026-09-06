"""
geometry.py — Phép biến đổi hình học cho 3D Pose và rotation (SO(3)).
"""

from __future__ import annotations

import torch


def align_by_pelvis(
    pose:          torch.Tensor,
    pelvis_index:  int = 0,
) -> torch.Tensor:
    """
    Căn chỉnh 3D pose bằng cách tịnh tiến gốc tọa độ về khớp xương chậu (pelvis).
    """
    if pose.dim() == 3:
        pelvis = pose[:, pelvis_index : pelvis_index + 1, :]  # (B, 1, 3)
    else:
        pelvis = pose[pelvis_index : pelvis_index + 1, :]     # (1, 3)
    return pose - pelvis


def axis_angle_to_quaternion(aa: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Axis-angle (..., 3) → quaternion wxyz (..., 4).
    Góc nhỏ dùng xấp xỉ ổn định số: q ≈ [1, aa/2].
    """
    angle = torch.linalg.norm(aa, dim=-1, keepdim=True)
    half = 0.5 * angle
    small = angle < 1e-6

    axis = aa / angle.clamp(min=eps)
    quat_full = torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)
    quat_small = torch.cat(
        [torch.ones_like(half), aa * 0.5],
        dim=-1,
    )
    return torch.where(small, quat_small, quat_full)


def quaternion_multiply(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Hamilton product, quaternion (..., 4) wxyz. Kết quả = q ⊗ r."""
    w1, x1, y1, z1 = q.unbind(dim=-1)
    w2, x2, y2, z2 = r.unbind(dim=-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def quaternion_to_axis_angle(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Quaternion wxyz (..., 4) → axis-angle (..., 3)."""
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp(min=eps)
    # Đảm bảo w >= 0 để góc nằm trong [-pi, pi]
    sign = torch.where(q[..., :1] < 0, -torch.ones_like(q[..., :1]), torch.ones_like(q[..., :1]))
    q = q * sign

    w = q[..., :1].clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    xyz = q[..., 1:]
    angle = 2.0 * torch.acos(w)
    sin_half = torch.sqrt((1.0 - w * w).clamp(min=eps))
    small = (angle < 1e-6).expand_as(xyz)
    axis = xyz / sin_half
    aa_full = axis * angle
    aa_small = xyz * 2.0
    return torch.where(small, aa_small, aa_full)


def compose_axis_angle(base: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """
    Kết hợp hai rotation axis-angle trên SO(3): R_out = R_base @ R_delta
    (delta trong local frame của khớp — đúng quy ước SMPL).
    """
    q_base = axis_angle_to_quaternion(base)
    q_delta = axis_angle_to_quaternion(delta)
    q_out = quaternion_multiply(q_base, q_delta)
    return quaternion_to_axis_angle(q_out)


def transfer_orientation(reference_source, reference_target, candidate_source):
    """Apply a source-frame rotation update in the target camera/world frame.

    R_target_new = R_target_ref @ R_source_ref.T @ R_source_new.
    The references must describe the same root orientation in the two frames.
    """
    q_source = axis_angle_to_quaternion(reference_source)
    inverse = q_source * q_source.new_tensor([1.0, -1.0, -1.0, -1.0])
    relative = quaternion_multiply(inverse, axis_angle_to_quaternion(candidate_source))
    return quaternion_to_axis_angle(quaternion_multiply(axis_angle_to_quaternion(reference_target), relative))


def interpolate_axis_angle(
    pose_a: torch.Tensor,
    pose_b: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Interpolate two rotations along the shortest path with stable nlerp."""
    q_a = axis_angle_to_quaternion(pose_a)
    q_b = axis_angle_to_quaternion(pose_b)
    dot = torch.sum(q_a * q_b, dim=-1, keepdim=True)
    q_b = torch.where(dot < 0.0, -q_b, q_b)
    q = (1.0 - alpha) * q_a + alpha * q_b
    return quaternion_to_axis_angle(q)


def rotation_geodesic_loss(
    pose_a: torch.Tensor,
    pose_b: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean shortest rotation error between two axis-angle tensors.

    weights: optional (..., 1) or same rank as per-joint error; joints with
    weight 0 are ignored (used when a joint was replaced by the other view).
    """
    q_a = axis_angle_to_quaternion(pose_a)
    q_b = axis_angle_to_quaternion(pose_b)
    cosine = torch.sum(q_a * q_b, dim=-1).abs().clamp(0.0, 1.0)
    per_joint = 1.0 - cosine.square()
    if weights is None:
        return torch.mean(per_joint)
    w = weights.squeeze(-1) if weights.ndim == per_joint.ndim + 1 else weights
    denom = w.sum().clamp_min(1e-8)
    return (per_joint * w).sum() / denom
