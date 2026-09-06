"""
mirror_view_correction.py — Gương làm góc nhìn bổ sung khi bị che khuất.

Khi góc nhìn gốc (real view) bị self-occlusion ở một số khớp, module này:
1. Phát hiện khớp nào bị occluded dựa trên belief map
2. Lấy rotation từ mirror view (đã unmirror) để thay thế hoặc blend
3. Chiếu cả 2 góc nhìn 2D (real + mirror) lên cùng hệ tọa độ để visualization

Ý tưởng cốt lõi:
    - Mirror view nhìn người từ góc khác → khớp bị che khuất ở real view
      có thể nhìn thấy rõ ở mirror view (và ngược lại)
    - Khi belief_real(j) < threshold → khớp j bị occluded → dùng mirror
    - Khi belief_mirror(j) < threshold → khớp j bị occluded ở mirror → giữ real
    - Cả hai đều occluded → giữ giá trị hiện tại (không có thông tin mới)
"""

from __future__ import annotations

import torch

from utils.geometry import interpolate_axis_angle


def detect_occluded_joints(
    belief: torch.Tensor,
    threshold: float = 0.3,
) -> torch.Tensor:
    """
    Phát hiện khớp nào bị che khuất dựa trên belief score.

    Args:
        belief    : (..., J) belief scores [0, 1]
        threshold : ngưỡng dưới đó coi là occluded

    Returns:
        occluded : (..., J) bool — True nếu khớp bị occluded
    """
    return belief < threshold


def apply_mirror_correction(
    real_pose: torch.Tensor,
    mirror_pose_unmirrored: torch.Tensor,
    belief_real: torch.Tensor,
    belief_mirror: torch.Tensor,
    occlusion_threshold: float = 0.3,
    blend_mode: str = "adaptive_slerp",
) -> torch.Tensor:
    """
    Sửa pose real bằng mirror view khi bị che khuất.

    Ba chế độ blend:
    - "replace": Thay thế trực tiếp khớp occluded bằng mirror (aggressive)
    - "slerp": SLERP cố định alpha=0.8 cho khớp occluded
    - "adaptive_slerp": Alpha tính theo belief ratio — mượt hơn

    Args:
        real_pose               : (..., 21, 3) body_pose axis-angle (hệ real)
        mirror_pose_unmirrored  : (..., 21, 3) mirror pose đã unmirror
        belief_real             : (..., 21) belief mỗi khớp cho real view
        belief_mirror           : (..., 21) belief mỗi khớp cho mirror view
        occlusion_threshold     : ngưỡng belief dưới đó coi là occluded
        blend_mode              : chiến lược blend

    Returns:
        corrected_pose : (..., 21, 3) pose đã được sửa
    """
    occluded_real = belief_real < occlusion_threshold
    occluded_mirror = belief_mirror < occlusion_threshold

    # Chỉ sửa khi: real bị occluded VÀ mirror KHÔNG bị occluded
    can_correct = occluded_real & (~occluded_mirror)

    if blend_mode == "replace":
        corrected = real_pose.clone()
        corrected[can_correct] = mirror_pose_unmirrored[can_correct]
        return corrected

    elif blend_mode == "slerp":
        # Alpha cố định cho khớp occluded
        alpha = torch.where(
            can_correct.unsqueeze(-1).expand_as(real_pose),
            torch.full_like(real_pose, 0.8),
            torch.zeros_like(real_pose),
        )
        return interpolate_axis_angle(real_pose, mirror_pose_unmirrored, alpha)

    elif blend_mode == "adaptive_slerp":
        # Alpha tỷ lệ với belief ratio — mượt hơn, không nhảy đột ngột
        eps = 1e-6
        total = belief_real + belief_mirror
        alpha_base = belief_mirror / total.clamp(min=eps)

        # Boost alpha khi real bị occluded, giảm khi mirror bị occluded
        alpha = torch.where(can_correct, alpha_base.clamp(min=0.6), alpha_base)
        alpha = torch.where(occluded_mirror, torch.zeros_like(alpha), alpha)

        return interpolate_axis_angle(
            real_pose, mirror_pose_unmirrored, alpha.unsqueeze(-1)
        )

    raise ValueError(f"Unknown blend_mode: {blend_mode}")


def apply_mirror_go_correction(
    real_go: torch.Tensor,
    mirror_go_unmirrored: torch.Tensor,
    belief_real_root: torch.Tensor,
    belief_mirror_root: torch.Tensor,
    occlusion_threshold: float = 0.3,
) -> torch.Tensor:
    """
    Sửa global_orient bằng mirror view khi root joints bị che khuất.

    Args:
        real_go                  : (N, 3) global_orient axis-angle
        mirror_go_unmirrored     : (N, 3) mirror global_orient đã unmirror
        belief_real_root         : (N,) belief root (trung bình vai+hông)
        belief_mirror_root       : (N,) belief root cho mirror

    Returns:
        corrected_go : (N, 3)
    """
    occluded = belief_real_root < occlusion_threshold
    mirror_ok = belief_mirror_root >= occlusion_threshold
    can_correct = occluded & mirror_ok

    eps = 1e-6
    total = belief_real_root + belief_mirror_root
    alpha = belief_mirror_root / total.clamp(min=eps)
    alpha = torch.where(can_correct, alpha.clamp(min=0.6), alpha)
    alpha = torch.where(~mirror_ok, torch.zeros_like(alpha), alpha)

    return interpolate_axis_angle(
        real_go, mirror_go_unmirrored, alpha.unsqueeze(-1)
    )


def compute_correction_diagnostics(
    belief_real: torch.Tensor,
    belief_mirror: torch.Tensor,
    occlusion_threshold: float = 0.3,
) -> dict[str, torch.Tensor]:
    """
    Thống kê diagnostic cho mirror correction.

    Returns:
        dict với:
        - occluded_real: (N, J) bool — khớp bị che khuất ở real
        - occluded_mirror: (N, J) bool — khớp bị che khuất ở mirror
        - corrected: (N, J) bool — khớp được sửa bởi mirror
        - n_corrected_per_frame: (N,) — số khớp được sửa mỗi frame
    """
    occ_real = belief_real < occlusion_threshold
    occ_mirror = belief_mirror < occlusion_threshold
    corrected = occ_real & (~occ_mirror)

    return {
        "occluded_real": occ_real,
        "occluded_mirror": occ_mirror,
        "corrected_by_mirror": corrected,
        "n_corrected_per_frame": corrected.sum(dim=-1),
    }
