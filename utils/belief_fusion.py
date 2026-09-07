"""Belief/reliability helpers for real and mirror observations.

Optimizer reliability is b_occ * b_det * b_reproj from the original
observation. Ray casting targets the mesh vertices supporting each COCO
surface landmark, never internal anatomical joint centres.

Sau khi có belief theo khung khớp COCO-17 (17 khớp), belief được ánh xạ sang
21 khớp body_pose SMPL (axis-angle) để dùng làm trọng số fusion SO(3)
(precision-weighted, mục 4): mỗi khớp SMPL được fuse bằng slerp giữa pose
real và pose mirror (đã un-mirror), trọng số tỉ lệ với belief của view đó.
"""

from __future__ import annotations

import torch

from utils.geometry import axis_angle_to_quaternion, interpolate_axis_angle
from utils.mesh_raycast import (
    compute_occlusion_belief_batch,
    surface_landmarks_from_regressor,
)

# Thứ tự khớp COCO-17, giống configs/keypoints3d_map.yml / keypoints2d_map.yml
COCO_JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
_NAME_TO_IDX = {name: idx for idx, name in enumerate(COCO_JOINT_NAMES)}
COCO_FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def flip_coco_belief(belief: torch.Tensor) -> torch.Tensor:
    """Swap anatomical sides for scalar COCO evidence; do not flip image pixels."""
    return belief[..., COCO_FLIP_IDX]

# Khớp "root" dùng để quyết định belief của global_orient (vai + hông)
_ROOT_JOINTS = [
    _NAME_TO_IDX["left_shoulder"], _NAME_TO_IDX["right_shoulder"],
    _NAME_TO_IDX["left_hip"], _NAME_TO_IDX["right_hip"],
]

# Ánh xạ 21 khớp body_pose SMPL (axis-angle, không gồm global_orient) → chỉ số
# (hoặc trung bình nhiều chỉ số) khớp COCO-17 tương ứng gần nhất, dùng để suy
# ra belief cho từng khớp SMPL từ belief đã tính trên COCO-17.
#   0 L_Hip, 1 R_Hip, 2 Spine1, 3 L_Knee, 4 R_Knee, 5 Spine2, 6 L_Ankle,
#   7 R_Ankle, 8 Spine3, 9 L_Foot, 10 R_Foot, 11 Neck, 12 L_Collar,
#   13 R_Collar, 14 Head, 15 L_Shoulder, 16 R_Shoulder, 17 L_Elbow,
#   18 R_Elbow, 19 L_Wrist, 20 R_Wrist
SMPL_BODY_TO_COCO = [
    (11,),           # 0  L_Hip
    (12,),           # 1  R_Hip
    (11, 12),        # 2  Spine1      -> trung bình hông
    (13,),           # 3  L_Knee
    (14,),           # 4  R_Knee
    (5, 6, 11, 12),  # 5  Spine2      -> trung bình vai + hông
    (15,),           # 6  L_Ankle
    (16,),           # 7  R_Ankle
    (5, 6),          # 8  Spine3      -> trung bình vai
    (15,),           # 9  L_Foot      -> không có khớp bàn chân riêng trong COCO
    (16,),           # 10 R_Foot
    (5, 6),          # 11 Neck
    (5,),            # 12 L_Collar
    (6,),            # 13 R_Collar
    (0,),            # 14 Head        -> nose
    (5,),            # 15 L_Shoulder
    (6,),            # 16 R_Shoulder
    (7,),            # 17 L_Elbow
    (8,),            # 18 R_Elbow
    (9,),            # 19 L_Wrist
    (10,),           # 20 R_Wrist
]


# ══════════════════════════════════════════════════════════════════════════
# 1. Belief theo từng nguồn evidence
# ══════════════════════════════════════════════════════════════════════════

def compute_detection_belief(confidence: torch.Tensor) -> torch.Tensor:
    """b_det: độ tin cậy 2D detector (ViTPose), clamp về [0, 1]. (..., 17)."""
    return confidence.clamp(0.0, 1.0)


def compute_reprojection_belief(
    projected_2d: torch.Tensor,
    detected_2d: torch.Tensor,
    valid_mask: torch.Tensor,
    sigma: float = 50.0,
) -> torch.Tensor:
    """
    b_reproj: nhất quán giữa joint 3D hiện tại reproject xuống ảnh và keypoint
    2D detect được. Cùng công thức Geman-McClure với calculate_reprojection_loss,
    nhưng trả belief = 1 - robust_error (thay vì loss), theo từng khớp.

    projected_2d, detected_2d : (..., 17, 2)
    valid_mask                : (..., 17) bool — False nếu depth <= 0 (sau lưng camera).
    """
    sq_error = (projected_2d - detected_2d).pow(2).sum(dim=-1)
    belief = (sigma ** 2) / (sq_error + sigma ** 2)
    return torch.where(valid_mask, belief, torch.zeros_like(belief))


def compute_occlusion_belief(
    joints_3d_view: torch.Tensor,
    vertices_view: torch.Tensor,
    faces,
    surface_regressor: torch.Tensor | None = None,
    camera_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    near_eps: float = 0.02,
    far_ratio: float = 0.985,
) -> torch.Tensor:
    """
    b_occ: ray-casting tới surface region COCO trên mesh của CHÍNH view đó.

    surface_regressor ánh xạ COCO-17 tới các vertex SMPL-X. Với legacy SMPL,
    landmark bề mặt gần nhất được dùng làm fallback.

    Không lan truyền gradient — dùng trong torch.no_grad() ở nơi gọi.
    """
    with torch.no_grad():
        landmarks, weights = surface_landmarks_from_regressor(
            joints_3d_view, vertices_view, surface_regressor
        )
        return compute_occlusion_belief_batch(
            landmarks, vertices_view, faces, surface_weights=weights,
            camera_origin=camera_origin, near_eps=near_eps, far_ratio=far_ratio,
        )


# ══════════════════════════════════════════════════════════════════════════
# 2. Kết hợp belief — Dempster–Shafer combination rule
# ══════════════════════════════════════════════════════════════════════════

def _ds_combine_pair(
    p1: torch.Tensor, a1: torch.Tensor,
    p2: torch.Tensor, a2: torch.Tensor,
    eps: float = 1e-6,
):
    """
    Dempster's rule cho 2 mass function trên khung {Present, Absent, Uncertain}.
    m(Uncertain) = 1 - m(Present) - m(Absent) (khung nhận thức đầy đủ = Uncertain).
    Trả về (p, a) — mass Present/Absent đã kết hợp và chuẩn hoá theo conflict K.
    """
    u1 = (1.0 - p1 - a1).clamp(min=0.0)
    u2 = (1.0 - p2 - a2).clamp(min=0.0)
    conflict = p1 * a2 + a1 * p2
    denom = (1.0 - conflict).clamp(min=eps)
    p = (p1 * p2 + p1 * u2 + u1 * p2) / denom
    a = (a1 * a2 + a1 * u2 + u1 * a2) / denom
    return p, a


def combine_beliefs_dempster_shafer(
    b_occ: torch.Tensor,
    b_det: torch.Tensor,
    b_reproj: torch.Tensor,
    occ_reliability: float = 0.999,
    soft_reliability: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Kết hợp 3 nguồn evidence bằng Dempster-Shafer combination rule.

    b_occ là evidence "cứng" (hình học, ray-casting): m(Present)=r_occ·b_occ,
    m(Absent)=r_occ·(1-b_occ). b_det/b_reproj là evidence "mềm" (thống kê,
    nhiễu) — chỉ có thể ủng hộ "Present" hoặc để "Uncertain", KHÔNG bao giờ
    khẳng định "Absent": m(Present)=r_soft·b, m(Absent)=0.

    `occ_reliability` gần 1 vì ray-casting gần như xác định tuyệt đối; còn
    `soft_reliability` < 1 (mặc định 0.5) để dù detector/reprojection có
    confidence tuyệt đối (=1), chúng cũng không đủ sức "cãi lại" một occlusion
    hình học đã xác nhận — tránh trường hợp suy biến (2 evidence mềm phiếu
    "present" thắng phiếu "absent" hình học chỉ vì đông hơn). Nếu không discount
    (reliability=1 cho tất cả), xung đột hoàn toàn (Zadeh's paradox) khiến
    kết quả không ổn định khi có 2/3 nguồn "chắc chắn" đối lập nhau.
    """
    p_occ, a_occ = occ_reliability * b_occ, occ_reliability * (1.0 - b_occ)
    p_det, a_det = soft_reliability * b_det, torch.zeros_like(b_det)
    p_reproj, a_reproj = soft_reliability * b_reproj, torch.zeros_like(b_reproj)

    p, a = _ds_combine_pair(p_occ, a_occ, p_det, a_det, eps)
    p, a = _ds_combine_pair(p, a, p_reproj, a_reproj, eps)
    return p.clamp(0.0, 1.0)


def combine_beliefs_weighted(
    b_occ: torch.Tensor,
    b_det: torch.Tensor,
    b_reproj: torch.Tensor,
    weights: tuple[float, float, float] = (0.5, 0.25, 0.25),
) -> torch.Tensor:
    """Lựa chọn đơn giản hơn (mục 3): trung bình có trọng số, chuẩn hoá [0, 1]."""
    w_occ, w_det, w_reproj = weights
    total_w = w_occ + w_det + w_reproj
    return (w_occ * b_occ + w_det * b_det + w_reproj * b_reproj) / total_w


def combine_view_belief(
    b_occ: torch.Tensor,
    b_det: torch.Tensor,
    b_reproj: torch.Tensor,
    method: str = "dempster_shafer",
) -> torch.Tensor:
    """belief(j) trong [0, 1] cho một view, đã kết hợp cả 3 nguồn evidence."""
    if method == "dempster_shafer":
        return combine_beliefs_dempster_shafer(b_occ, b_det, b_reproj)
    if method == "weighted":
        return combine_beliefs_weighted(b_occ, b_det, b_reproj)
    raise ValueError(f"Unknown belief combine method: {method}")


def smooth_belief_temporal(belief: torch.Tensor, kernel_size: int = 5, sigma: float = 1.2) -> torch.Tensor:
    """
    Làm mượt belief theo thời gian (dim 0 = frame) bằng 1D Gaussian.

    Occlusion/detection belief được tính ĐỘC LẬP mỗi frame (ray-casting +
    ViTPose confidence riêng từng frame) nên có thể dao động đột ngột dù
    chuyển động thực tế mượt — trọng số fusion/reprojection đổi theo dẫn tới
    khung xương rung giật (jitter) dù pose optimization đã có temporal loss.
    Làm mượt trực tiếp belief giảm bớt nguồn nhiễu này trước khi dùng.

    Args:
        belief: (N, ...) — N là số frame.
    """
    N = belief.shape[0]
    if N < 3:
        return belief

    half = min(kernel_size // 2, (N - 1) // 2)
    x = torch.arange(-half, half + 1, device=belief.device, dtype=belief.dtype)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()

    flat = belief.reshape(N, -1).transpose(0, 1).unsqueeze(1)  # (C, 1, N)
    padded = torch.nn.functional.pad(flat, (half, half), mode="replicate")
    smoothed = torch.nn.functional.conv1d(padded, kernel.view(1, 1, -1))
    return smoothed.squeeze(1).transpose(0, 1).reshape(belief.shape).clamp(0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════
# 3. Ánh xạ belief COCO-17 → 21 khớp body_pose SMPL + fusion SO(3)
# ══════════════════════════════════════════════════════════════════════════

def map_coco_belief_to_smpl_body(belief_coco17: torch.Tensor) -> torch.Tensor:
    """(..., 17) belief COCO -> (..., 21) belief cho từng khớp body_pose SMPL."""
    parts = [belief_coco17[..., list(idxs)].mean(dim=-1) for idxs in SMPL_BODY_TO_COCO]
    return torch.stack(parts, dim=-1)


# Chiều ngược của SMPL_BODY_TO_COCO — mỗi khớp COCO gom từ 1+ khớp SMPL body
# (khi nhiều khớp SMPL cùng trỏ tới 1 khớp COCO, ví dụ Spine1/2/3 đều dùng vai/hông,
# lấy trung bình).
_COCO_TO_SMPL_BODY: dict[int, list[int]] = {}
for _smpl_idx, _coco_idxs in enumerate(SMPL_BODY_TO_COCO):
    for _coco_idx in _coco_idxs:
        _COCO_TO_SMPL_BODY.setdefault(_coco_idx, []).append(_smpl_idx)


def map_smpl_body_scale_to_coco(scale21: torch.Tensor) -> torch.Tensor:
    """(..., 21) hệ số scale theo khớp SMPL body -> (..., 17) theo khớp COCO,
    ngược lại map_coco_belief_to_smpl_body (dùng để lan truyền cross-view trust
    tính ở không gian SMPL xuống belief COCO-17 dùng cho reprojection loss).
    Mắt/tai (COCO idx 1-4) không có khớp SMPL body nào ánh xạ tới -> scale = 1
    (không đổi), vì compute_cross_view_trust không áp dụng cho các điểm đó."""
    parts = []
    for coco_idx in range(len(COCO_JOINT_NAMES)):
        smpl_idxs = _COCO_TO_SMPL_BODY.get(coco_idx)
        if not smpl_idxs:
            parts.append(torch.ones_like(scale21[..., 0]))
        else:
            parts.append(scale21[..., smpl_idxs].mean(dim=-1))
    return torch.stack(parts, dim=-1)


def root_belief(belief_coco17: torch.Tensor) -> torch.Tensor:
    """Belief cho global_orient — trung bình belief của vai + hông."""
    return belief_coco17[..., _ROOT_JOINTS].mean(dim=-1)


# Parent index trong cây động học SMPL chuẩn, viết theo thứ tự 22 cột của
# pose_confidence_22 (0=root/global_orient, 1..21=body_pose joint 0..20 theo
# đúng thứ tự index-guide đầu losses/prior_losses.py). -1 = không có cha (root).
SMPL_KINEMATIC_PARENTS_22 = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
]


def _build_kinematic_adjacency_22(device: torch.device) -> torch.Tensor:
    """(22, 22) bool: True nếu j là chính i, cha của i, hoặc con trực tiếp của i."""
    n = len(SMPL_KINEMATIC_PARENTS_22)
    adjacency = torch.eye(n, dtype=torch.bool, device=device)
    for child, parent in enumerate(SMPL_KINEMATIC_PARENTS_22):
        if parent == -1:
            continue
        adjacency[child, parent] = True
        adjacency[parent, child] = True
    return adjacency


def propagate_confidence_to_kinematic_neighbors(confidence: torch.Tensor) -> torch.Tensor:
    """
    Lan truyền độ tin cậy THẤP sang khớp lân cận trên cây động học SMPL (1-hop).

    Một khớp bị "đoán bừa" (occlusion/depth ambiguity khiến belief cao giả —
    xem compute_cross_view_trust) thường kéo theo khớp cha/con trực tiếp cũng
    kém tin cậy: cùng vùng bị che khuất, hoặc lỗi rotation của khớp cha lan
    sang toàn bộ chuỗi động học phía con. Vì vậy không nên coi khớp lân cận
    đáng tin một cách độc lập với khớp đang xét.

    Với mỗi khớp i: confidence mới = min(confidence[i], confidence[cha(i)],
    confidence[từng con trực tiếp của i]) — chỉ 1 bước lan truyền, KHÔNG đệ quy
    nhiều khớp, để tránh 1 khớp xa bị thấp kéo sập confidence toàn thân.

    confidence: (..., 22) — cột 0=root/global_orient, 1..21=body_pose joint
                0..20 (theo SMPL_KINEMATIC_PARENTS_22 ở trên).
    Returns:    (..., 22) cùng shape, đã lan truyền.
    """
    n = confidence.shape[-1]
    if n != len(SMPL_KINEMATIC_PARENTS_22):
        raise ValueError(f"confidence phải có {len(SMPL_KINEMATIC_PARENTS_22)} khớp (root+21), nhận {n}")
    adjacency = _build_kinematic_adjacency_22(confidence.device)  # (22, 22)
    expanded = confidence.unsqueeze(-2).expand(*confidence.shape[:-1], n, n)  # (..., 22, 22)
    masked = torch.where(adjacency, expanded, expanded.new_full((), float("inf")))
    return masked.min(dim=-1).values


def compute_angular_discrepancy_penalty(
    pose_a: torch.Tensor,
    pose_b: torch.Tensor,
    threshold_deg: float = 90.0,
    transition_width: float = 15.0,
) -> torch.Tensor:
    """
    Tính hệ số phạt [0, 1] khi góc xoay giữa 2 pose quá lớn.

    Khi hai view cho ra ước lượng góc xoay khớp chênh lệch > threshold_deg,
    hệ số giảm dần về 0 (sigmoid), dùng để giảm belief mirror → fusion thiên
    về real, tránh pose trung gian cực đoan / vặn xoắn.

    Trường hợp sử dụng chính: gương đặt ở góc bất kỳ khiến global_orient từ
    mirror view (sau unmirror) rất khác với real view — nếu không giảm trọng số
    mirror, fusion sẽ tạo pose "trung gian" giữa 2 hướng nhìn hoàn toàn khác
    nhau, gây vặn xoắn / hành động cực đoan.

    pose_a, pose_b : (..., 3) axis-angle — cùng quy ước trái/phải.
    Returns        : (...,) scale factor trong [0, 1].
                     1.0 nếu angle << threshold (đồng thuận cao),
                     ~0.0 nếu angle >> threshold (bất đồng lớn).
    """
    q_a = axis_angle_to_quaternion(pose_a)
    q_b = axis_angle_to_quaternion(pose_b)
    dot = (q_a * q_b).sum(dim=-1).abs().clamp(0.0, 1.0)
    angle_deg = torch.rad2deg(2.0 * torch.acos(dot.clamp(max=1.0 - 1e-7)))
    return torch.sigmoid(-(angle_deg - threshold_deg) / transition_width)


def relax_discrepancy_penalty_when_real_absent(
    quality: torch.Tensor,
    belief_real: torch.Tensor,
) -> torch.Tensor:
    """
    Làm cho `compute_angular_discrepancy_penalty` linh hoạt theo ngữ cảnh thay
    vì áp một hệ số giảm CỐ ĐỊNH cho mọi khớp/frame bất kể real có tin cậy hay
    không: bất đồng góc xoay real/mirror chỉ đáng lo khi real THẬT SỰ có một
    ước lượng cạnh tranh đáng tin (belief_real cao) — nếu real đang occluded
    (belief_real ~ 0), không có "ứng viên real" nào để so bì, nên phạt mirror
    theo bất đồng góc lúc này là vô căn cứ (mirror là nguồn thông tin DUY NHẤT).

        scale = quality + (1 - quality) * (1 - belief_real)

    belief_real ~ 0 (occluded)  → scale ~ 1  (bỏ qua phạt, tin mirror hoàn toàn)
    belief_real ~ 1 (confident) → scale ~ quality (giữ nguyên mức phạt gốc)

    quality, belief_real : cùng shape, quality trong [0,1] (1=đồng thuận),
                            belief_real trong [0,1].
    Returns: scale trong [0,1], cùng shape.
    """
    relax = (1.0 - belief_real).clamp(0.0, 1.0)
    return quality + (1.0 - quality) * relax


def compute_cross_view_trust(
    det_real: torch.Tensor,
    det_mirror: torch.Tensor,
    pose_agreement: torch.Tensor,
    min_scale: float = 0.3,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Hệ số scale [min_scale, 1] cho belief_real / belief_mirror, dùng để phá vòng
    lặp tự-tham-chiếu của b_occ/b_reproj: cả hai được tính từ CHÍNH pose đang
    tối ưu của mỗi view, nên khi HMR4D đoán sai một khớp (ví dụ do occlusion/
    depth ambiguity ở real view), ray-casting không phát hiện được lỗi hình học
    nếu silhouette 2D vẫn khớp, và reprojection belief gần như luôn cao vì đó
    chính là đại lượng optimizer đang trực tiếp tối thiểu hoá — belief "tự nói
    dối là đúng". `compute_angular_discrepancy_penalty` (go_quality/bp_quality)
    trước đây chỉ dùng để hạ belief_mirror khi 2 view bất đồng (mặc định luôn
    tin real), không xử lý được trường hợp NGƯỢC LẠI: real đoán bừa còn mirror
    đúng.

    Độ tin cậy 2D detector (ViTPose, kp2d_conf) là bằng chứng ĐỘC LẬP duy nhất
    không phụ thuộc vào pose 3D đang tối ưu — dùng nó làm trọng tài khi hai view
    bất đồng mạnh (pose_agreement thấp), thay vì mặc định thiên vị real. Khi
    pose_agreement ~ 1 (đồng thuận), scale ~ 1 cho cả hai (không đổi gì).

    det_real, det_mirror : độ tin cậy detector 2D, cùng shape với pose_agreement.
    pose_agreement        : compute_angular_discrepancy_penalty(...) — 1 = đồng
                             thuận, ~0 = bất đồng mạnh.

    Returns: (scale_real, scale_mirror), cùng shape, mỗi phần tử trong [min_scale, 1].
    """
    total_det = (det_real + det_mirror).clamp(min=eps)
    trust_real = (det_real / total_det).clamp(min=min_scale)
    trust_mirror = (det_mirror / total_det).clamp(min=min_scale)
    scale_real = pose_agreement + (1.0 - pose_agreement) * trust_real
    scale_mirror = pose_agreement + (1.0 - pose_agreement) * trust_mirror
    return scale_real, scale_mirror


def fuse_pose_precision_weighted(
    pose_real: torch.Tensor,
    pose_mirror: torch.Tensor,
    belief_real: torch.Tensor,
    belief_mirror: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Precision-weighted fusion trên SO(3) (mục 4, thay cho công thức 3D-point
    vì hai camera real/mirror không được hiệu chỉnh chung một hệ tọa độ thế
    giới — chỉ có thể fuse góc xoay, không fuse trực tiếp vị trí 3D tuyệt đối):

        J_fused(j) = (b_real·real + b_mirror·mirror) / (b_real + b_mirror)

    áp dụng qua slerp trên quaternion tương ứng. Khi belief_real(j) = 0 (occluded),
    kết quả suy biến đúng thành pose_mirror(j).

    pose_real, pose_mirror     : (..., 21, 3) axis-angle.
    belief_real, belief_mirror : (..., 21).
    """
    total = belief_real + belief_mirror
    weight_mirror = torch.where(
        total > eps, belief_mirror / total.clamp(min=eps), torch.zeros_like(total)
    )
    return interpolate_axis_angle(pose_real, pose_mirror, weight_mirror.unsqueeze(-1))


def fuse_global_orient_precision_weighted(
    go_real: torch.Tensor,
    go_mirror: torch.Tensor,
    belief_real_root: torch.Tensor,
    belief_mirror_root: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Như fuse_pose_precision_weighted, cho global_orient (..., 3) với 1 belief scalar/frame."""
    total = belief_real_root + belief_mirror_root
    weight_mirror = torch.where(
        total > eps, belief_mirror_root / total.clamp(min=eps), torch.zeros_like(total)
    )
    return interpolate_axis_angle(go_real, go_mirror, weight_mirror.unsqueeze(-1))


# ══════════════════════════════════════════════════════════════════════════
# 4. Orchestrator: belief đầy đủ cho một view (occlusion + detection + reprojection)
# ══════════════════════════════════════════════════════════════════════════

def compute_full_view_belief(
    joints_3d_view: torch.Tensor,
    vertices_view: torch.Tensor,
    faces,
    kp2d_detected: torch.Tensor,
    kp2d_conf: torch.Tensor,
    projected_2d: torch.Tensor,
    valid_mask: torch.Tensor,
    surface_regressor: torch.Tensor | None = None,
    reproj_sigma: float = 50.0,
    combine_method: str = "dempster_shafer",
    occlusion_near_eps: float = 0.02,
    occlusion_far_ratio: float = 0.985,
) -> dict[str, torch.Tensor]:
    """
    Tính source reliability cố định từ surface visibility, detector và reprojection.

    joints_3d_view, vertices_view : hệ tọa độ "incam" của CHÍNH view này (mục 1/2).
    projected_2d, valid_mask      : joints_3d_view chiếu xuống ảnh qua K của view này.
    """
    b_occ = compute_occlusion_belief(
        joints_3d_view, vertices_view, faces,
        surface_regressor=surface_regressor,
        near_eps=occlusion_near_eps, far_ratio=occlusion_far_ratio,
    )
    b_det = compute_detection_belief(kp2d_conf)
    b_reproj = compute_reprojection_belief(projected_2d, kp2d_detected, valid_mask, sigma=reproj_sigma)
    # Independent evidence appears exactly once; low surface visibility lets
    # the other camera drive fusion/refit instead of trusting an occluded view.
    belief = b_occ * b_det * b_reproj
    return {
        "belief": belief,
        "b_occ": b_occ,
        "b_det": b_det,
        "b_reproj": b_reproj,
    }
