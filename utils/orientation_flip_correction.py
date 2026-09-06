"""
orientation_flip_correction.py — Phát hiện và sửa lỗi lật 180° Hướng người (Front-Back / Depth Ambiguity).

Mô tả bài toán:
Các mô hình ước lượng 3D pose đơn lẻ (như HMR4D) thường bị bẫy bởi tính đối xứng
của keypoint 2D khi người quay lưng vào camera, dẫn đến việc khởi tạo `real_global_orient`
bị lật 180° (mô hình 3D quay mặt ra trước trong khi người thật quay lưng).

Module này cung cấp 3 cơ chế tự động phát hiện và sửa lỗi 180°:
1. `correct_front_back_flips_from_mirror`: So sánh tín hiệu khuôn mặt (Mũi/Mắt/Tai) giữa
   Real View và Mirror View. Nếu Mirror thấy rõ mặt nhưng Real bị che khuất mà Real GO
   vẫn hướng ra trước -> Đảo 180° GO.
2. `correct_chest_facing_from_keypoints`: Kiểm tra sự bất đồng giữa hướng ngực 3D (chest normal)
   và độ tin cậy của keypoints khuôn mặt 2D.
3. `correct_temporal_yaw_jumps`: Khử các bước nhảy 180° đột ngột giữa các frame liên tiếp.
"""

from __future__ import annotations

import torch
import numpy as np
from utils.geometry import compose_axis_angle, axis_angle_to_quaternion, rotation_geodesic_loss


def flip_global_orient_180_y(global_orient: torch.Tensor) -> torch.Tensor:
    """
    Xoay global_orient 180° quanh trục Y (Yaw flip 180°).
    global_orient: (..., 3) axis-angle tensor.
    """
    device = global_orient.device
    dtype = global_orient.dtype
    yaw_flip_aa = torch.tensor([0.0, np.pi, 0.0], device=device, dtype=dtype).expand_as(global_orient)
    return compose_axis_angle(global_orient, yaw_flip_aa)


def compute_chest_normal_z(joints_3d: torch.Tensor) -> torch.Tensor:
    """
    Tính thành phần Z của ma trận pháp tuyến ngực (chest normal vector).
    joints_3d: (B, 17+, 3) tọa độ 3D khớp COCO.
    
    Trong hệ tọa độ camera:
    - Z_normal < 0: Ngực quay về phía camera (Front view).
    - Z_normal > 0: Ngực quay ra xa camera (Back view).
    """
    # L_Shoulder = idx 5, R_Shoulder = idx 6
    # L_Hip = idx 11, R_Hip = idx 12
    l_sh = joints_3d[:, 5, :]
    r_sh = joints_3d[:, 6, :]
    l_hip = joints_3d[:, 11, :]
    r_hip = joints_3d[:, 12, :]

    mid_sh = 0.5 * (l_sh + r_sh)
    mid_hip = 0.5 * (l_hip + r_hip)

    spine_vec = mid_sh - mid_hip          # (B, 3) từ hông lên vai
    shoulder_vec = r_sh - l_sh            # (B, 3) từ vai trái sang vai phải

    # Chest normal = cross(shoulder_vec, spine_vec)
    chest_normal = torch.cross(shoulder_vec, spine_vec, dim=-1)  # (B, 3)
    norm = chest_normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    chest_normal = chest_normal / norm

    return chest_normal[:, 2]  # Z component


def correct_front_back_flips(
    real_go: torch.Tensor,
    mirror_go_unmirrored: torch.Tensor,
    kp2d_conf_real: torch.Tensor | None = None,
    kp2d_conf_mirror: torch.Tensor | None = None,
    joints_real_3d: torch.Tensor | None = None,
    face_indices: list[int] = [0, 1, 2, 3, 4],  # Nose, L_Eye, R_Eye, L_Ear, R_Ear
) -> torch.Tensor:
    """
    Phát hiện và sửa lỗi lật 180° cho real_global_orient.

    Args:
        real_go                 : (N, 3) axis-angle real global orient
        mirror_go_unmirrored    : (N, 3) axis-angle mirror global orient (đã unmirror)
        kp2d_conf_real          : (N, 17) confidence keypoints 2D view real
        kp2d_conf_mirror        : (N, 17) confidence keypoints 2D view mirror
        joints_real_3d          : (N, 17+, 3) joints 3D ước lượng từ real pose

    Returns:
        corrected_go : (N, 3) global orient đã được sửa lỗi lật 180°
    """
    N = real_go.shape[0]
    corrected_go = real_go.clone()

    # 1. So sánh tín hiệu mặt (Face belief) giữa Real và Mirror
    if kp2d_conf_real is not None and kp2d_conf_mirror is not None:
        conf_real_face = kp2d_conf_real[:, face_indices].mean(dim=-1)   # (N,)
        conf_mirror_face = kp2d_conf_mirror[:, face_indices].mean(dim=-1) # (N,)

        # Tính độ lệch góc xoay (geodesic distance) giữa real_go và mirror_go_unmirrored
        qa = axis_angle_to_quaternion(real_go)
        qb = axis_angle_to_quaternion(mirror_go_unmirrored)
        cosine = (qa * qb).sum(dim=-1).abs().clamp(-1.0, 1.0)
        angular_diff_deg = torch.rad2deg(2.0 * torch.acos(cosine))  # (N,)

        # Khi Real bị khuất mặt (conf_real < 0.35) nhưng Mirror nhìn rõ mặt (conf_mirror >= 0.35)
        # VÀ real_go khác biệt với mirror_go_unmirrored > 75°:
        # -> mirror_go_unmirrored mới là hướng chuẩn thực tế!
        mirror_has_face = (conf_real_face < 0.35) & (conf_mirror_face >= 0.35) & (angular_diff_deg > 75.0)
        corrected_go[mirror_has_face] = mirror_go_unmirrored[mirror_has_face]

    # 2. Kiểm tra Hướng Ngực (Chest Normal Z) vs Face confidence
    if joints_real_3d is not None and kp2d_conf_real is not None:
        conf_real_face = kp2d_conf_real[:, face_indices].mean(dim=-1)
        chest_z = compute_chest_normal_z(joints_real_3d)  # (N,)

        # Nếu conf_real_face < 0.25 (người thật quay lưng), nhưng chest_z < -0.15 (mô hình 3D quay mặt vào camera):
        # -> Đây là mâu thuẫn 180°! Đảo 180° GO.
        is_front_back_contradiction = (conf_real_face < 0.25) & (chest_z < -0.15)
        if is_front_back_contradiction.any():
            corrected_go[is_front_back_contradiction] = flip_global_orient_180_y(
                corrected_go[is_front_back_contradiction]
            )

    # 3. Khử các bước nhảy 180° đột ngột giữa các frame (Temporal Yaw Continuity)
    corrected_go = correct_temporal_yaw_jumps(corrected_go)

    return corrected_go


def correct_temporal_yaw_jumps(go_sequence: torch.Tensor, max_jump_deg: float = 100.0) -> torch.Tensor:
    """
    Khử các bước nhảy 180° Spurious Flip giữa các frame liên tiếp.
    Nếu frame t nhảy > 100° so với frame t-1, nhưng sau khi flip 180° lại gần frame t-1,
    thì flip frame t trở lại.
    """
    N = go_sequence.shape[0]
    if N < 2:
        return go_sequence

    seq = go_sequence.clone()
    for t in range(1, N):
        prev_go = seq[t - 1 : t]
        curr_go = seq[t : t + 1]

        qa = axis_angle_to_quaternion(prev_go)
        qb = axis_angle_to_quaternion(curr_go)
        diff_deg = torch.rad2deg(2.0 * torch.acos((qa * qb).sum(dim=-1).abs().clamp(-1.0, 1.0)))

        if diff_deg.item() > max_jump_deg:
            # Thử lật 180° curr_go
            flipped_curr = flip_global_orient_180_y(curr_go)
            qc = axis_angle_to_quaternion(flipped_curr)
            diff_flipped_deg = torch.rad2deg(2.0 * torch.acos((qa * qc).sum(dim=-1).abs().clamp(-1.0, 1.0)))

            # Nếu sau khi lật 180°, góc chênh với frame trước giảm mạnh (< 45°) -> lật luôn
            if diff_flipped_deg.item() < 45.0:
                seq[t] = flipped_curr[0]

    return seq
