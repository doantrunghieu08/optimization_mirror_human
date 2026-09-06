"""
mirror_geometry.py — Xử lý gương ở vị trí bất kỳ (arbitrary mirror plane).

Thay vì giả định gương luôn vuông góc trục X (mirror_axis_angle chỉ negate Y,Z),
module này:
1. Ước lượng trục phản xạ (mirror normal) từ tương quan giữa skeleton real và
   mirror — cho phép camera và gương đặt ở BẤT KỲ VỊ TRÍ nào.
2. Chuẩn hóa scale giữa skeleton real và mirror (ảnh gương xa hơn → 3D nhỏ hơn)
   để belief fusion không bị thiên lệch do chênh lệch kích thước.
3. Cung cấp hàm unmirror tổng quát dùng estimated mirror normal.

Lý thuyết:
    Phản xạ qua mặt phẳng có pháp tuyến n (unit vector) và qua gốc tọa độ:
        R_mirror = I - 2 * n * n^T   (Householder reflection)
    Axis-angle sau phản xạ:
        ω' = R_mirror @ ω   (phản chiếu trục xoay)
        θ' = -θ              (đảo chiều xoay — reflection đổi chirality)
    Kết hợp: ω'_aa = -R_mirror @ ω_aa   (với ω_aa = θ * axis)
"""

from __future__ import annotations

import torch
import numpy as np
from utils.smpl_utils import SMPL_FLIP_PAIRS


# ══════════════════════════════════════════════════════════════════════════
# 1. Ước lượng mirror normal từ skeleton pairs
# ══════════════════════════════════════════════════════════════════════════

def estimate_mirror_normal(
    joints_real: torch.Tensor,
    joints_mirror: torch.Tensor,
    method: str = "pca",
) -> torch.Tensor:
    """
    Ước lượng pháp tuyến mặt phẳng gương từ cặp skeleton real/mirror.

    Ý tưởng: Trung điểm giữa mỗi cặp khớp (real_j, mirror_j) đều nằm trên
    mặt phẳng gương. Pháp tuyến của mặt phẳng gương là hướng vuông góc với
    tập trung điểm này (PCA: eigenvector có eigenvalue nhỏ nhất).

    Args:
        joints_real   : (N, J, 3) — joints 3D ở hệ incam real
        joints_mirror : (N, J, 3) — joints 3D ở hệ incam mirror (raw, chưa unmirror)
        method        : "pca" hoặc "displacement"

    Returns:
        normal : (3,) unit vector — pháp tuyến ước lượng của mặt phẳng gương.
                 Mặc định trả về X-axis nếu dữ liệu không đủ / degenerate.
    """
    if method == "displacement":
        return _estimate_by_displacement(joints_real, joints_mirror)
    return _estimate_by_pca(joints_real, joints_mirror)


def _estimate_by_displacement(
    joints_real: torch.Tensor,
    joints_mirror: torch.Tensor,
) -> torch.Tensor:
    """
    Phương pháp đơn giản: hướng chênh lệch trung bình giữa real và mirror.
    Giả định gương đặt giữa hai cluster → displacement vector ≈ mirror normal.
    """
    diff = (joints_real - joints_mirror).mean(dim=(0, 1))  # (3,)
    norm = diff.norm()
    if norm < 1e-6:
        return torch.tensor([1.0, 0.0, 0.0], device=joints_real.device)
    return diff / norm


def _estimate_by_pca(
    joints_real: torch.Tensor,
    joints_mirror: torch.Tensor,
) -> torch.Tensor:
    """
    PCA trên tập trung điểm giữa real và mirror joints.
    Eigenvector có eigenvalue nhỏ nhất = pháp tuyến mặt phẳng gương.
    """
    midpoints = (joints_real + joints_mirror) / 2.0  # (N, J, 3)
    mid_flat = midpoints.reshape(-1, 3)  # (N*J, 3)

    # Center
    mean = mid_flat.mean(dim=0, keepdim=True)
    centered = mid_flat - mean

    # Covariance matrix
    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)

    # Eigendecomposition — eigenvector nhỏ nhất = normal
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    normal = eigenvectors[:, 0]  # smallest eigenvalue

    # Đảm bảo normal hướng từ mirror về real (dot product dương với displacement)
    diff = (joints_real - joints_mirror).mean(dim=(0, 1))
    if (normal @ diff) < 0:
        normal = -normal

    return normal


def estimate_mirror_normal_robust(
    joints_real: torch.Tensor,
    joints_mirror: torch.Tensor,
    agreement_cos_threshold: float = 0.9,
) -> torch.Tensor:
    """
    Ước lượng mirror normal đáng tin cậy hơn `estimate_mirror_normal`.

    `estimate_mirror_normal` có circular dependency: normal được suy ra từ
    chính joints 3D của pose HMR4D ban đầu (có thể còn nhiễu/sai), rồi normal
    đó lại dùng để sửa pose — nếu ước lượng lệch, unmirror_pose_general phản
    chiếu sai trục, gây lật tay/chân sai bên. Hàm này tính normal bằng CẢ HAI
    phương pháp độc lập (PCA + displacement) và chỉ tin kết quả khi chúng
    đồng thuận (cosine similarity cao); nếu bất đồng, coi ước lượng không
    đáng tin và fallback về giả định an toàn (gương vuông góc trục X) thay vì
    dùng một normal có thể sai.
    """
    normal_pca = _estimate_by_pca(joints_real, joints_mirror)
    normal_disp = _estimate_by_displacement(joints_real, joints_mirror)

    cos_sim = (normal_pca @ normal_disp).clamp(-1.0, 1.0)
    if cos_sim.abs() < agreement_cos_threshold:
        return torch.tensor(
            [1.0, 0.0, 0.0], device=joints_real.device, dtype=joints_real.dtype
        )

    if cos_sim < 0:
        normal_pca = -normal_pca
    combined = normal_pca + normal_disp
    return combined / combined.norm().clamp_min(1e-6)


# ══════════════════════════════════════════════════════════════════════════
# 2. Householder reflection cho axis-angle
# ══════════════════════════════════════════════════════════════════════════

def reflect_axis_angle(
    aa: torch.Tensor,
    normal: torch.Tensor,
) -> torch.Tensor:
    """
    Phản chiếu vector axis-angle qua mặt phẳng có pháp tuyến `normal`.

    Với reflection qua mặt phẳng pháp tuyến n:
        R = I - 2*n*n^T  (Householder matrix)
        ω' = -R @ ω      (đảo dấu vì reflection đổi chirality)

    Khi normal = [1,0,0] (X-axis), kết quả trùng với mirror_axis_angle()
    hiện tại: negate Y và Z, giữ nguyên X.

    Args:
        aa     : (..., 3) axis-angle vectors
        normal : (3,) unit normal vector

    Returns:
        reflected : (..., 3)
    """
    n = normal / normal.norm().clamp(min=1e-8)
    # Householder: R = I - 2*n*n^T
    # Component song song với n: proj = (aa · n) * n
    # Component vuông góc: perp = aa - proj
    # R @ aa = perp - proj = aa - 2*proj
    proj = (aa * n).sum(dim=-1, keepdim=True) * n
    reflected = aa - 2 * proj
    # Đảo dấu vì reflection đổi chirality (left-hand ↔ right-hand)
    return -reflected


def reflect_global_orientation(global_orient: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    """Physical reflection of the root: R' = S_normal @ R @ S_body_X.

    The left matrix lives in camera coordinates, the right one in the canonical
    body frame. Conjugating R by S_normal would mix these two frames.
    """
    from utils.geometry import axis_angle_to_quaternion, quaternion_multiply, quaternion_to_axis_angle
    from utils.smpl_utils import mirror_axis_angle
    n = normal / normal.norm().clamp_min(1e-8)
    q_normal = torch.cat([n.new_zeros(1), n])
    q_body_x = n.new_tensor([0.0, 1.0, 0.0, 0.0])
    q_offset = quaternion_multiply(q_normal, q_body_x)
    return quaternion_to_axis_angle(
        quaternion_multiply(q_offset, axis_angle_to_quaternion(mirror_axis_angle(global_orient)))
    )


def unmirror_pose_general(
    pose: torch.Tensor,
    normal: torch.Tensor,
) -> torch.Tensor:
    """
    Unmirror pose tổng quát cho gương ở bất kỳ vị trí nào.

    Thay thế unmirror_pose() (chỉ hoạt động khi gương vuông góc trục X).

    Args:
        pose   : (..., 22, 3) axis-angle cho 22 khớp SMPL
        normal : (3,) pháp tuyến mặt phẳng gương

    Returns:
        (..., 22, 3) pose đã được un-mirror.
    """
    from utils.smpl_utils import unmirror_pose
    reflected = unmirror_pose(pose)
    reflected[..., 0, :] = reflect_axis_angle(pose[..., 0, :], normal)
    return reflected


def remirror_body_pose_general(
    body_pose: torch.Tensor,
    normal: torch.Tensor,
) -> torch.Tensor:
    """
    Lật ngược lại body_pose (21 khớp) vào không gian gốc của gương.
    Tổng quát hóa remirror_body_pose() cho arbitrary mirror normal.
    """
    # Local joints always use the canonical body symmetry, independent of camera.
    from utils.smpl_utils import remirror_body_pose
    return remirror_body_pose(body_pose)


# ══════════════════════════════════════════════════════════════════════════
# 3. Chuẩn hóa scale giữa real/mirror skeleton
# ══════════════════════════════════════════════════════════════════════════

# Các đoạn xương dùng để tính scale (torso = stable reference)
_SCALE_BONES = [
    (5, 6),   # vai trái - vai phải
    (11, 12), # hông trái - hông phải
    (5, 11),  # vai trái - hông trái
    (6, 12),  # vai phải - hông phải
]


def compute_skeleton_scale_ratio(
    joints_real: torch.Tensor,
    joints_mirror: torch.Tensor,
) -> torch.Tensor:
    """
    Tính tỷ lệ scale trung bình giữa skeleton real và mirror.

    Dùng chiều dài xương torso (vai-hông) làm tham chiếu — phần thân người
    ít bị ảnh hưởng bởi tư thế nhất.

    Args:
        joints_real   : (N, J, 3) — COCO-17 joints, real view
        joints_mirror : (N, J, 3) — COCO-17 joints, mirror view

    Returns:
        scale_ratio : (N,) — tỷ lệ real/mirror cho mỗi frame.
                      Giá trị > 1 nghĩa mirror nhỏ hơn real.
    """
    eps = 1e-6
    ratios = []
    for (i, j) in _SCALE_BONES:
        len_real = (joints_real[:, i] - joints_real[:, j]).norm(dim=-1)
        len_mirror = (joints_mirror[:, i] - joints_mirror[:, j]).norm(dim=-1)
        ratio = len_real / len_mirror.clamp(min=eps)
        ratios.append(ratio)

    # Trung bình robust (median) để tránh outlier khi 1 xương bị ước lượng sai
    ratios_stack = torch.stack(ratios, dim=-1)  # (N, num_bones)
    return ratios_stack.median(dim=-1).values


def normalize_mirror_joints_scale(
    joints_mirror: torch.Tensor,
    scale_ratio: torch.Tensor,
) -> torch.Tensor:
    """
    Scale mirror joints lên để cùng kích thước với real joints.

    Args:
        joints_mirror : (N, J, 3)
        scale_ratio   : (N,) — tỷ lệ real/mirror

    Returns:
        joints_scaled : (N, J, 3)
    """
    # Scale quanh centroid của mirror skeleton
    centroid = joints_mirror.mean(dim=1, keepdim=True)  # (N, 1, 3)
    relative = joints_mirror - centroid
    scaled = relative * scale_ratio.unsqueeze(-1).unsqueeze(-1)
    return scaled + centroid
