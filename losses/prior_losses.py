import math

import torch
from utils.geometry import axis_angle_to_quaternion, quaternion_multiply, rotation_geodesic_loss

# ══════════════════════════════════════════════════════════════════════════════
# SMPL body_pose index guide (21 joints, 0-indexed, không có global_orient)
#   0: L_Hip,  1: R_Hip,  2: Spine1
#   3: L_Knee, 4: R_Knee, 5: Spine2
#   6: L_Ankle,7: R_Ankle,8: Spine3
#   9: L_Foot, 10: R_Foot,11: Neck
#  12: L_Collar,13: R_Collar,14: Head
#  15: L_Shoulder,16: R_Shoulder
#  17: L_Elbow,  18: R_Elbow
#  19: L_Wrist,  20: R_Wrist
# ══════════════════════════════════════════════════════════════════════════════

# COCO-17 bone pairs cho Symmetry Loss
# Mỗi phần tử: ((trái_gốc, trái_ngọn), (phải_gốc, phải_ngọn))
COCO_SYMMETRIC_BONE_PAIRS = [
    ((5, 7),   (6, 8)),   # Vai → Khuỷu tay
    ((7, 9),   (8, 10)),  # Khuỷu → Cổ tay
    ((11, 13), (12, 14)), # Hông → Đầu gối
    ((13, 15), (14, 16)), # Đầu gối → Mắt cá chân
]

# COCO-17 tất cả các cạnh xương để tính Bone Length Stability
COCO_ALL_BONES = [
    (5, 7),   (7, 9),    # Tay trái
    (6, 8),   (8, 10),   # Tay phải
    (11, 13), (13, 15),  # Chân trái
    (12, 14), (14, 16),  # Chân phải
    (5, 6),              # Đòn vai
    (11, 12),            # Cạnh hông
    (5, 11),             # Thân trái
    (6, 12),             # Thân phải
]


# ══════════════════════════════════════════════════════════════════════════════
# Các hàm loss hiện có (giữ nguyên)
# ══════════════════════════════════════════════════════════════════════════════

# Vai/khuỷu/cổ tay (15-20) VÀ hông/gối/mắt cá chân (0,1,3,4,6,7) đã có loss ROM
# riêng (elbow_limits, shoulder_hyper, wrist_bend, anatomical_limits,
# joint_angle_limit) — loại khỏi pose_prior để tránh bị "kéo về 0" hai lần, vì
# khi tín hiệu reprojection yếu (mơ hồ độ sâu, camera quay lưng), lực kéo-về-0
# của pose_prior sẽ thắng và làm khớp duỗi thẳng đơ phi tự nhiên thay vì gập/
# bắt chéo tự nhiên theo tư thế thật (bug ban đầu phát hiện ở tay, cùng cơ chế
# cũng ảnh hưởng chân khi camera quay lưng làm gập gối bị mơ hồ độ sâu).
_POSE_PRIOR_EXCLUDE_JOINTS = (0, 1, 3, 4, 6, 7, 15, 16, 17, 18, 19, 20)


def compute_pose_prior_loss(body_pose, exclude_joints: tuple = _POSE_PRIOR_EXCLUDE_JOINTS):
    """
    Tính Loss phạt tư thế (Pose Prior L2 Regularization).
    Cách nhẹ và phổ biến nhất: Phạt trực tiếp trên độ lớn của góc xoay (axis-angle).
    Vì mô hình SMPL được thiết kế sao cho giá trị [0,0,0] là tư thế tự nhiên (T-pose/A-pose),
    việc phạt các góc xoay quá lớn sẽ đóng vai trò như "dây chun" kéo cơ thể về trạng thái 
    tự nhiên, ngăn không cho các khớp bị xoắn vặn cực đoan (gãy xương).
    Args:
        body_pose (torch.Tensor): Tensor chứa tham số pose, shape (batch_size, 63) hoặc (batch_size, 69)
        exclude_joints (tuple): chỉ số khớp (0-20) loại khỏi phạt — mặc định vai/khuỷu/cổ
            tay vì đã có loss ROM chuyên biệt, không cần thêm lực kéo-về-0 chung.

    Returns:
        torch.Tensor: Giá trị loss
    """
    pose = body_pose.view(body_pose.shape[0], -1, 3)
    if exclude_joints:
        keep = [j for j in range(pose.shape[1]) if j not in exclude_joints]
        pose = pose[:, keep, :]
    return torch.mean(pose ** 2)

def compute_shape_prior_loss(betas):
    """
    Tính Loss phạt hình dáng cơ thể (Shape Prior).
    Ngăn thuật toán làm biến dạng hình thể (quá béo, quá gầy, tay chân dài bất thường) 
    chỉ để cố gắng vươn tới khớp với điểm 2D. 
    
    Args:
        betas (torch.Tensor): Tensor chứa tham số shape, shape (batch_size, 10)
        
    Returns:
        torch.Tensor: Giá trị loss
    """
    return torch.mean(betas ** 2)

def compute_anatomical_limits_loss(body_pose):
    """
    Tính Loss phạt giới hạn sinh lý cụ thể (Anatomical/Angle Limits).
    Đầu gối người (khớp 4 và 5 trong SMPL, tương ứng index 3 và 4 trong body_pose) không thể gập cong ra phía trước.
    Trục gập chính của đầu gối trong SMPL là trục X (index 0).
    Gập tự nhiên (shin ra sau) tương ứng với góc DƯƠNG. Gập lỗi (bẻ ngược ra trước/flamingo) là góc ÂM.
    
    Args:
        body_pose (torch.Tensor): Tensor chứa tham số pose.
        
    Returns:
        torch.Tensor: Giá trị loss phạt
    """
    batch_size = body_pose.shape[0]
    
    # Đưa về dạng (batch_size, num_joints, 3) để dễ truy cập từng khớp
    pose_reshaped = body_pose.view(batch_size, -1, 3)
    
    loss = 0.0
    
    # Khớp L_Knee (index 3) và R_Knee (index 4) trong mảng body_pose 21 khớp
    left_knee_bend = pose_reshaped[:, 3, 0]  
    right_knee_bend = pose_reshaped[:, 4, 0] 
    
    # Chỉ phạt khi góc < 0 (bẻ gối ngược ra trước).
    # Dùng ReLU(-góc) để lấy phần âm. Nếu góc >= 0 (đứng thẳng hoặc gập tự nhiên), relu = 0 (không phạt).
    # Việc dùng exp như trước đó sẽ khiến model liên tục cố gập gối để làm exp(-góc) nhỏ nhất có thể, 
    # dẫn đến tình trạng không thể đứng thẳng.
    loss += torch.mean(torch.relu(-left_knee_bend) ** 2)
    loss += torch.mean(torch.relu(-right_knee_bend) ** 2)
    
    return loss

def estimate_height_coco17(joints_3d: torch.Tensor) -> torch.Tensor:
    """
    Chiều cao từ chuỗi xương COCO-17 (mét), không phụ thuộc trục đứng camera.
    joints_3d: (B, 17+, 3)
    """
    mid_sh = 0.5 * (joints_3d[:, 5, :] + joints_3d[:, 6, :])
    mid_hip = 0.5 * (joints_3d[:, 11, :] + joints_3d[:, 12, :])
    head = torch.linalg.norm(joints_3d[:, 0, :] - mid_sh, dim=-1)
    torso = torch.linalg.norm(mid_sh - mid_hip, dim=-1)
    l_leg = (
        torch.linalg.norm(joints_3d[:, 11, :] - joints_3d[:, 13, :], dim=-1)
        + torch.linalg.norm(joints_3d[:, 13, :] - joints_3d[:, 15, :], dim=-1)
    )
    r_leg = (
        torch.linalg.norm(joints_3d[:, 12, :] - joints_3d[:, 14, :], dim=-1)
        + torch.linalg.norm(joints_3d[:, 14, :] - joints_3d[:, 16, :], dim=-1)
    )
    return (head + torso + 0.5 * (l_leg + r_leg)) * 1.15


def compute_height_loss(
    joints_3d: torch.Tensor,
    target_height_m,                  # float hoặc torch.Tensor (B,) — GT per-frame
) -> torch.Tensor:
    """
    Ràng buộc chiều cao cơ thể qua chiều dài chuỗi xương (cùng công thức với GT).
    """
    estimated_height = estimate_height_coco17(joints_3d)
    return torch.mean((estimated_height - target_height_m) ** 2)


# ══════════════════════════════════════════════════════════════════════════════
# CÁC HÀM LOSS MỚI
# ══════════════════════════════════════════════════════════════════════════════

def compute_bone_symmetry_loss(joints_3d: torch.Tensor) -> torch.Tensor:
    """
    Bone Symmetry Loss — Phạt sự chênh lệch chiều dài xương trái/phải.

    Trên cơ thể người khỏe mạnh, xương tay/chân trái và phải phải có chiều dài
    gần bằng nhau. Nếu model tạo ra cơ thể bị "lệch" một bên (ví dụ: chân trái
    dài hơn chân phải 15cm), đó là kết quả phi sinh lý học cần bị phạt.

    Lợi ích:
      - Ngăn model biến dạng bất đối xứng để khớp với 2D keypoints bị nhiễu.
      - Đặc biệt hữu ích khi ảnh gương cho ra estimation không đều hai bên.

    Args:
        joints_3d (torch.Tensor): Tọa độ 3D khớp COCO, shape (B, 17, 3).

    Returns:
        torch.Tensor: Loss scalar (MSE giữa chiều dài xương trái và phải).
    """
    loss = joints_3d.new_zeros(1).squeeze()  # 0.0 trên cùng device

    for (l_prox, l_dist), (r_prox, r_dist) in COCO_SYMMETRIC_BONE_PAIRS:
        # Chiều dài xương bên trái và phải
        left_len  = torch.norm(joints_3d[:, l_prox, :] - joints_3d[:, l_dist, :], dim=-1)  # (B,)
        right_len = torch.norm(joints_3d[:, r_prox, :] - joints_3d[:, r_dist, :], dim=-1)  # (B,)
        loss = loss + torch.mean((left_len - right_len) ** 2)

    return loss


def compute_elbow_limits_loss(body_pose: torch.Tensor) -> torch.Tensor:
    """
        Elbow Limits Loss — Phạt khuỷu tay gập vượt quá ROM sinh lý.

        Hướng của axis-angle local không cố định giữa các output HMR4D / SMPL
        conventions. Vì vậy không được ép flexion về riêng trục X/Y/Z: làm vậy sẽ
        triệt tiêu một pose khuỷu tay gập hợp lệ. Chỉ giới hạn biên độ rotation
        tổng, còn pose/reprojection loss quyết định hướng gập.

    Args:
        body_pose (torch.Tensor): SMPL body pose, shape (B, 63).

    Returns:
        torch.Tensor: Loss scalar.
    """
    pose_reshaped = body_pose.view(body_pose.shape[0], -1, 3)

    left_elbow_angle = torch.linalg.vector_norm(pose_reshaped[:, 17, :], dim=-1)
    right_elbow_angle = torch.linalg.vector_norm(pose_reshaped[:, 18, :], dim=-1)

    # Phạt rotation vượt flexion tối đa sinh lý (150° — AAOS normal ROM).
    max_flexion_rad = math.radians(150)
    return (
        torch.mean(torch.relu(left_elbow_angle - max_flexion_rad) ** 2)
        + torch.mean(torch.relu(right_elbow_angle - max_flexion_rad) ** 2)
    )


def compute_mirror_consistency_loss(
    fused_body_pose: torch.Tensor,
    mirror_body_pose_unmirrored: torch.Tensor,
) -> torch.Tensor:
    """
    Mirror Consistency Loss — Ràng buộc fused pose nhất quán với cả 2 nguồn đầu vào.

    Hệ thống đã có `delta_reg` để ràng buộc fused ≈ real (người thật).
    Loss này bổ sung ràng buộc fused ≈ mirror (ảnh gương, đã un-mirror).
    Kết hợp cả hai, fused sẽ là điểm nằm giữa real và mirror — đây chính là
    điểm tối ưu của bài toán fusion.

    Tại sao cần thiết?
      Nếu chỉ có `delta_reg` (fused ≈ real), model có thể bỏ qua hoàn toàn
      thông tin từ góc nhìn gương và chỉ đơn giản copy pose người thật.
      Loss này buộc model phải "lắng nghe" cả hai góc nhìn.

    Args:
        fused_body_pose (torch.Tensor):
            Pose sau khi fusion, shape (B, 63).
        mirror_body_pose_unmirrored (torch.Tensor):
            Pose từ ảnh gương đã được un-mirror về hệ tọa độ người thật, shape (B, 63).
            Đây là `batch["mirror_body_pose"]` từ Dataset.

    Returns:
        torch.Tensor: Loss scalar (MSE giữa fused và mirror input trong cùng không gian).
    """
    fused = fused_body_pose.view(fused_body_pose.shape[0], -1, 3)
    mirror = mirror_body_pose_unmirrored.view(mirror_body_pose_unmirrored.shape[0], -1, 3)
    return rotation_geodesic_loss(fused, mirror)


def compute_bone_length_stability_loss(joints_3d: torch.Tensor) -> torch.Tensor:
    """
    Bone Length Stability Loss — Ràng buộc chiều dài xương không đổi qua các frame.

    Nguyên lý vật lý: Cơ thể người là vật cứng — chiều dài tay/chân không thay đổi
    từ frame này sang frame khác. Nếu model tạo ra frame này tay dài 60cm, frame sau
    tay ngắn 55cm, đó là kết quả sai hoàn toàn về mặt vật lý.

    Cách hoạt động:
      Xem batch dimension B như một chuỗi thời gian (time dimension).
      Tính chiều dài từng xương cho mỗi frame, sau đó phạt phương sai (variance)
      của chiều dài đó trên toàn bộ batch/sequence.
      → Variance = 0 nghĩa là chiều dài xương hoàn toàn ổn định qua thời gian.

    Lưu ý: Loss này hiệu quả nhất khi batch được lấy theo thứ tự frame liên tiếp.
    Nếu DataLoader shuffle, loss vẫn có tác dụng nhưng yếu hơn.

    Args:
        joints_3d (torch.Tensor): Tọa độ 3D khớp COCO, shape (B, 17, 3).

    Returns:
        torch.Tensor: Loss scalar (tổng variance chiều dài các xương).
    """
    if joints_3d.shape[0] < 2:
        return joints_3d.new_zeros(1).squeeze()

    loss = joints_3d.new_zeros(1).squeeze()

    for j1, j2 in COCO_ALL_BONES:
        # Chiều dài xương tại mỗi frame: (B,)
        bone_lengths = torch.norm(joints_3d[:, j1, :] - joints_3d[:, j2, :], dim=-1)
        # Phạt variance — nếu chiều dài ổn định, variance = 0
        loss = loss + torch.var(bone_lengths)

    return loss


def compute_temporal_smoothness_loss(joints_3d: torch.Tensor) -> torch.Tensor:
    """
    Temporal Smoothness Loss — Phạt gia tốc khớp lớn (Joint Acceleration Loss).

    Chuyển động tự nhiên của con người mượt mà và có gia tốc thấp.
    Gia tốc cao (giật cục giữa các frame) là dấu hiệu của nhiễu hoặc lỗi estimation.

    Công thức:
      velocity[t]     = joints[t] - joints[t-1]          → tốc độ
      acceleration[t] = velocity[t] - velocity[t-1]      → gia tốc
      L = mean(acceleration²)                             → phạt gia tốc lớn

    Tại sao phạt gia tốc thay vì tốc độ?
      Phạt tốc độ sẽ ngăn chuyển động nhanh (không muốn). Phạt gia tốc
      chỉ phạt sự THAY ĐỔI ĐỘT NGỘT của tốc độ (jerk), cho phép chuyển
      động nhanh nhưng mượt mà.

    Yêu cầu: batch_size >= 3 để tính được ít nhất 1 điểm gia tốc.

    Args:
        joints_3d (torch.Tensor): Chuỗi tọa độ 3D, shape (T, J, 3) trong đó
                                  T = số frame (batch_size được coi là time).

    Returns:
        torch.Tensor: Loss scalar. Trả về 0.0 nếu T < 3.
    """
    if joints_3d.ndim != 3 or joints_3d.shape[1] < 13 or joints_3d.shape[2] != 3:
        raise ValueError("joints_3d must have shape (T, J>=13, 3) in COCO order")
    if joints_3d.shape[0] < 3:
        return joints_3d.new_zeros(1).squeeze()

    # Pose smoothness must not fight camera/person translation.  Anchor every
    # frame at the COCO mid-hip before taking temporal differences.
    pelvis = joints_3d[:, 11:13].mean(dim=1, keepdim=True)
    joints_3d = joints_3d - pelvis
    velocity     = joints_3d[1:]  - joints_3d[:-1]   # (T-1, J, 3)
    acceleration = velocity[1:]   - velocity[:-1]     # (T-2, J, 3)

    return torch.mean(acceleration ** 2)


def compute_pose_temporal_smoothness_loss(
    global_orient: torch.Tensor,
    body_pose: torch.Tensor,
    confidence: torch.Tensor | None = None,
    min_weight: float = 0.3,
) -> torch.Tensor:
    """
    Pose Temporal Smoothness Loss — phạt trực tiếp "vận tốc" xoay (rotation
    velocity) giữa 2 frame liên tiếp, trên KHÔNG GIAN axis-angle (khác với
    compute_temporal_smoothness_loss vốn phạt gia tốc VỊ TRÍ 3D khớp).

    Lý do cần thêm: khớp có thể xoay tại chỗ (root xoay, hoặc khớp gần trục)
    mà vị trí 3D gần như không đổi — temporal loss trên joints_3d không bắt
    được kiểu "nhảy" này, khiến khung xương rung giật khi render dù loss vị
    trí đã thấp. Phạt trực tiếp trên rotation velocity xử lý đúng nguyên nhân.

    confidence (tùy chọn): (T, 22) độ tin cậy [0,1] mỗi khớp mỗi frame (vd.
    max(belief_real, belief_mirror) — root ở cột 0, 21 khớp body_pose sau đó).
    Khi có, những cặp frame mà CẢ HAI đầu đều được real/mirror xác nhận chắc
    chắn (confidence cao) sẽ bị giảm phạt vận tốc (gate = 1 - (1-min_weight)*conf),
    cho phép chuyển động nhanh/thật đi qua thay vì bị làm mượt che mất tín hiệu
    gương rõ ràng; những khớp/frame còn mơ hồ vẫn bị phạt đầy đủ (gate ~ 1).
    min_weight chặn dưới gate (không bao giờ tắt hẳn smoothing dù confidence=1).

    Args:
        global_orient (torch.Tensor): (T, 3) axis-angle.
        body_pose (torch.Tensor):     (T, 63) axis-angle (21 khớp).

    Returns:
        torch.Tensor: Loss scalar (0.0 nếu T < 2).
    """
    T = body_pose.shape[0]
    if T < 2:
        return body_pose.new_zeros(1).squeeze()

    bp = body_pose.view(T, -1, 3)
    go = global_orient.view(T, 1, 3)
    pose = torch.cat([go, bp], dim=1)  # (T, 22, 3)

    if confidence is None:
        return rotation_geodesic_loss(pose[1:], pose[:-1])

    q_a = axis_angle_to_quaternion(pose[1:])
    q_b = axis_angle_to_quaternion(pose[:-1])
    cosine = torch.sum(q_a * q_b, dim=-1).abs().clamp(0.0, 1.0)
    per_joint = 1.0 - cosine.square()  # (T-1, 22)

    pair_confidence = torch.minimum(confidence[1:], confidence[:-1])  # (T-1, 22)
    gate = 1.0 - (1.0 - min_weight) * pair_confidence
    return (per_joint * gate).mean()


def compute_pose_acceleration_smoothness_loss(
    global_orient: torch.Tensor,
    body_pose: torch.Tensor,
    confidence: torch.Tensor | None = None,
    min_weight: float = 0.3,
) -> torch.Tensor:
    """
    Pose Acceleration Smoothness Loss — phạt GIA TỐC xoay (đổi hướng vận tốc
    xoay đột ngột) trên SO(3), bổ sung cho compute_pose_temporal_smoothness_loss
    (loss đó chỉ phạt VẬN TỐC xoay giữa 2 frame liên tiếp — không chặn được
    trường hợp vận tốc xoay đổi dấu/đổi hướng liên tục giữa 3 frame, chính là
    nguyên nhân gây rung giật/jerk dù chuyển động tổng thể có vẻ mượt).

    Dùng hợp thành quaternion (không dùng hiệu Frobenius giữa 2 delta-rotation)
    vì hiệu trực tiếp không phản ánh đúng khoảng cách trên nhóm Lie SO(3).

    confidence (tùy chọn): (T, 22) độ tin cậy [0,1] mỗi khớp mỗi frame — cùng
    quy ước với compute_pose_temporal_smoothness_loss. Gate theo min-confidence
    của 3 frame liên quan tới mỗi gia tốc (t-1, t, t+1): chỉ giảm phạt khi CẢ
    BA frame đều được xác nhận chắc chắn.

    Args:
        global_orient (torch.Tensor): (T, 3) axis-angle.
        body_pose (torch.Tensor):     (T, 63) axis-angle (21 khớp).

    Returns:
        torch.Tensor: Loss scalar (0.0 nếu T < 3).
    """
    T = body_pose.shape[0]
    if T < 3:
        return body_pose.new_zeros(1).squeeze()

    bp = body_pose.view(T, -1, 3)
    go = global_orient.view(T, 1, 3)
    pose = torch.cat([go, bp], dim=1)  # (T, 22, 3)

    q = axis_angle_to_quaternion(pose)  # (T, 22, 4)
    q_conj = q.clone()
    q_conj[..., 1:] = -q_conj[..., 1:]

    # delta_q[t] = vận tốc xoay từ frame t sang t+1 (quaternion tương đối)
    delta_q = quaternion_multiply(q[1:], q_conj[:-1])  # (T-1, 22, 4)

    delta_conj = delta_q.clone()
    delta_conj[..., 1:] = -delta_conj[..., 1:]

    # rel[t] = lệch giữa 2 vận tốc xoay liên tiếp = gia tốc góc xoay
    rel = quaternion_multiply(delta_q[1:], delta_conj[:-1])  # (T-2, 22, 4)
    w = rel[..., 0].abs().clamp(max=1.0 - 1e-7)
    geodesic_dist = 2.0 * torch.acos(w)
    squared = geodesic_dist.pow(2)  # (T-2, 22)

    if confidence is None:
        return squared.mean()

    triple_confidence = torch.minimum(torch.minimum(confidence[:-2], confidence[1:-1]), confidence[2:])
    gate = 1.0 - (1.0 - min_weight) * triple_confidence
    return (squared * gate).mean()


def compute_head_collision_loss(joints_3d: torch.Tensor, threshold: float = 0.10) -> torch.Tensor:
    """
    Head Collision Loss — Phạt khi cổ tay hoặc khuỷu tay cắt xuyên qua đầu.
    
    Trong tư thế đưa tay lên đầu, model có xu hướng đẩy tay đâm xuyên qua hộp sọ 
    để tối ưu reprojection loss 2D do thiếu hiểu biết về thể tích 3D (volume).
    
    Args:
        joints_3d (torch.Tensor): Tọa độ 3D khớp COCO, shape (B, 17, 3).
        threshold (float): Khoảng cách tối thiểu (mét) từ tay đến mũi (COCO idx 0, xấp xỉ tâm đầu). 
                           Bán kính đầu người cộng thêm khoảng cách cổ tay khoảng 20cm (0.2).
                           
    Returns:
        torch.Tensor: Loss scalar.
    """
    head = joints_3d[:, 0, :]
    if joints_3d.shape[1] > 17:
        head = 0.5 * (head + joints_3d[:, 17, :])
    
    # Cổ tay (Wrist: L=9, R=10)
    l_wrist = joints_3d[:, 9, :]
    r_wrist = joints_3d[:, 10, :]
    
    # Khoảng cách
    d_l_wrist = torch.norm(l_wrist - head, dim=-1)  # (B,)
    d_r_wrist = torch.norm(r_wrist - head, dim=-1)
    
    # Phạt nếu khoảng cách nhỏ hơn threshold (xuyên qua đầu)
    loss = torch.mean(torch.relu(threshold - d_l_wrist) ** 2)
    loss = loss + torch.mean(torch.relu(threshold - d_r_wrist) ** 2)
    
    return loss


# ══════════════════════════════════════════════════════════════════════════════
# CÁC HÀM PHẠT TƯ THẾ QUÁI DỊ (GROTESQUE POSE PENALTIES)
# ══════════════════════════════════════════════════════════════════════════════

def compute_spine_twist_loss(
    body_pose: torch.Tensor,
    max_twist_rad: float = math.radians(15),
    max_total_twist_rad: float = math.radians(25),
) -> torch.Tensor:
    """
    Spine Twist Loss — Phạt xoắn cột sống quá mức trên trục Y (twist).

    Cột sống người có 3 đốt trong SMPL (Spine1=2, Spine2=5, Spine3=8).
    Tổng xoay trục (axial rotation) sinh lý tối đa của cột sống ngực-thắt lưng
    (thoracolumbar) vào khoảng 45-50° — nhưng đó là mức đồng thời TỐI ĐA ở MỌI
    đốt cùng lúc, hiếm khi xảy ra trong chuyển động thực. Giữ tổng chặt hơn
    (~25°) để chống "vặn xoắn cực đoan" (mỗi đốt xoay ~15° hợp lý riêng lẻ,
    nhưng cộng dồn cả 3 đốt theo cùng hướng lại tạo xoắn thân trên phi tự nhiên).

    Bổ sung phạt cả TỔNG XOẮN tích lũy qua cả 3 đốt — tránh trường hợp mỗi
    đốt xoay 15° nhưng cộng dồn lại làm thân trên xoay ngược so với hông.

    Args:
        body_pose (torch.Tensor): SMPL body pose, shape (B, 63).
        max_twist_rad (float): Giới hạn twist tối đa mỗi đốt (rad). Default 15°.
        max_total_twist_rad (float): Giới hạn tổng twist tích lũy 3 đốt (rad). Default 25°.

    Returns:
        torch.Tensor: Loss scalar.
    """
    pose = body_pose.view(body_pose.shape[0], -1, 3)

    # Spine1(2), Spine2(5), Spine3(8) — trục Y là twist trong local frame SMPL
    spine_indices = [2, 5, 8]
    loss = body_pose.new_zeros(1).squeeze()

    # 1. Phạt twist trên từng đốt riêng lẻ
    total_spine_y = body_pose.new_zeros(body_pose.shape[0])
    for idx in spine_indices:
        twist_y = pose[:, idx, 1]  # Thành phần Y = twist
        loss = loss + torch.mean(torch.relu(twist_y.abs() - max_twist_rad) ** 2)
        total_spine_y = total_spine_y + twist_y

    # 2. Phạt TỔNG TWIST tích lũy của cả 3 đốt cột sống
    loss = loss + torch.mean(torch.relu(total_spine_y.abs() - max_total_twist_rad) ** 2)

    return loss


def compute_hip_split_loss(
    body_pose: torch.Tensor,
    max_abduction_rad: float = math.radians(90),
) -> torch.Tensor:
    """
    Hip Split Loss — Phạt dạng háng (abduction) quá mức, tư thế xoạc.

    Trong SMPL, L_Hip (index 0) và R_Hip (index 1) kiểm soát chuyển động
    chân. Trục Z trong local frame tương ứng với abduction (dạng ra ngoài).
    Hip abduction sinh lý bình thường chỉ ~45°, nhưng vận động viên thể dục/
    vũ công có thể dạng rộng hơn nhiều (split hoàn toàn ~90° mỗi bên) —
    dùng 90° làm giới hạn tối đa thay vì giá trị "bình thường" 45° để không
    chặn các tư thế split/xoạc hợp lệ trong dance, nhưng vẫn chặn góc vượt
    quá giới hạn giải phẫu thật (>90° gần như luôn là lỗi estimation).

    Args:
        body_pose (torch.Tensor): SMPL body pose, shape (B, 63).
        max_abduction_rad (float): Giới hạn góc dạng háng tối đa (rad). Default 90°.

    Returns:
        torch.Tensor: Loss scalar.
    """
    pose = body_pose.view(body_pose.shape[0], -1, 3)

    # L_Hip = index 0, R_Hip = index 1 (trong body_pose 21 khớp)
    l_hip_z = pose[:, 0, 2]  # Trục Z = abduction
    r_hip_z = pose[:, 1, 2]  # Trục Z = abduction

    loss = torch.mean(torch.relu(l_hip_z.abs() - max_abduction_rad) ** 2)
    loss = loss + torch.mean(torch.relu(r_hip_z.abs() - max_abduction_rad) ** 2)

    return loss


def compute_shoulder_hyperextension_loss(
    body_pose: torch.Tensor,
    max_extension_rad: float = math.radians(180),
) -> torch.Tensor:
    """
    Shoulder Hyperextension Loss — Phạt vai bẻ ngược ra phía sau quá mức.

    Vai người (L_Shoulder=15, R_Shoulder=16 trong body_pose SMPL) có biên độ
    chuyển động rộng nhất cơ thể — flexion/abduction sinh lý tối đa đạt tới
    180° (giơ thẳng tay lên qua đầu, động tác múa phổ biến). Giá trị cũ hơn
    (2.0 rad/~115°, rồi 3.0 rad/~172°) từng khiến cánh tay giơ cao bị bẻ cụt/
    lệch khỏi tư thế thật vì thấp hơn giới hạn giải phẫu thật. Dùng đúng 180°
    (giới hạn sinh lý tối đa theo tài liệu goniometry lâm sàng chuẩn — AAOS
    normal ROM) thay vì số ước lượng.

    Args:
        body_pose (torch.Tensor): SMPL body pose, shape (B, 63).
        max_extension_rad (float): Giới hạn tổng góc xoay vai tối đa (rad). Default 180°.

    Returns:
        torch.Tensor: Loss scalar.
    """
    pose = body_pose.view(body_pose.shape[0], -1, 3)

    loss = body_pose.new_zeros(1).squeeze()

    # L_Shoulder = index 15, R_Shoulder = index 16 (trong body_pose 21 khớp)
    for idx in [15, 16]:
        angle = torch.norm(pose[:, idx, :], dim=-1)  # Tổng góc xoay (rad)
        loss = loss + torch.mean(torch.relu(angle - max_extension_rad) ** 2)

    return loss


def compute_wrist_bend_loss(
    body_pose: torch.Tensor,
    max_bend_rad: float = math.radians(90),
) -> torch.Tensor:
    """
    Wrist Bend Loss — Phạt cổ tay bẻ quá mức giới hạn sinh lý.

    Cổ tay người (L_Wrist=19, R_Wrist=20 trong body_pose SMPL) có biên độ
    gập tối đa (flexion ~80°, extension ~70-90° tùy người — AAOS normal ROM)
    — dùng giá trị lớn nhất trong khoảng đó (90°) làm giới hạn sinh lý tối đa.
    Bẻ cổ tay vượt quá giới hạn này tạo tư thế gãy tay.

    Args:
        body_pose (torch.Tensor): SMPL body pose, shape (B, 63).
        max_bend_rad (float): Giới hạn tổng góc bẻ cổ tay (rad). Default 90°.

    Returns:
        torch.Tensor: Loss scalar.
    """
    pose = body_pose.view(body_pose.shape[0], -1, 3)

    loss = body_pose.new_zeros(1).squeeze()

    # L_Wrist = index 19, R_Wrist = index 20 (trong body_pose 21 khớp)
    for idx in [19, 20]:
        angle = torch.norm(pose[:, idx, :], dim=-1)  # Tổng góc xoay (rad)
        loss = loss + torch.mean(torch.relu(angle - max_bend_rad) ** 2)

    return loss


def compute_hip_adduction_loss(
    body_pose: torch.Tensor,
    max_adduction_rad: float = math.radians(25),
) -> torch.Tensor:
    """
    Hip Adduction Loss — Phạt khép háng quá mức (vắt chân qua đường giữa).

    L_Hip (index 0 trong body_pose): -Z là khép háng.
    R_Hip (index 1 trong body_pose): +Z là khép háng.
    """
    pose = body_pose.view(body_pose.shape[0], -1, 3)
    l_hip_z = pose[:, 0, 2]
    r_hip_z = pose[:, 1, 2]

    loss = (
        torch.mean(torch.relu(-l_hip_z - max_adduction_rad) ** 2)
        + torch.mean(torch.relu(r_hip_z - max_adduction_rad) ** 2)
    )
    return loss


def compute_leg_crossing_loss(
    joints_3d: torch.Tensor,
    max_cross_margin_m: float = 0.04,
) -> torch.Tensor:
    """
    Leg Crossing Loss — Phạt vắt chân / chéo chân quá mức phi sinh lý.

    joints_3d: (B, 17+, 3) tọa độ 3D khớp COCO.
    - 11: L_Hip, 12: R_Hip
    - 13: L_Knee, 14: R_Knee
    - 15: L_Ankle, 16: R_Ankle

    Vectơ hông (v_hip) từ L_Hip trỏ sang R_Hip:
      v_hip = normalize(R_Hip - L_Hip)

    Tọa độ hình chiếu của L_Knee, L_Ankle theo trục v_hip tính từ L_Hip:
      d_l_knee = (L_Knee - L_Hip) · v_hip
      d_l_ankle = (L_Ankle - L_Hip) · v_hip

    Tương tự cho R_Knee, R_Ankle theo trục -v_hip tính từ R_Hip:
      d_r_knee = (R_Hip - R_Knee) · v_hip
      d_r_ankle = (R_Hip - R_Ankle) · v_hip
    """
    l_hip = joints_3d[:, 11, :]
    r_hip = joints_3d[:, 12, :]
    l_knee = joints_3d[:, 13, :]
    r_knee = joints_3d[:, 14, :]
    l_ankle = joints_3d[:, 15, :]
    r_ankle = joints_3d[:, 16, :]

    hip_vec = r_hip - l_hip
    hip_norm = hip_vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    v_hip = hip_vec / hip_norm  # (B, 3) trỏ từ L_Hip -> R_Hip

    d_l_knee = ((l_knee - l_hip) * v_hip).sum(dim=-1)
    d_l_ankle = ((l_ankle - l_hip) * v_hip).sum(dim=-1)

    d_r_knee = ((r_hip - r_knee) * v_hip).sum(dim=-1)
    d_r_ankle = ((r_hip - r_ankle) * v_hip).sum(dim=-1)

    loss = (
        torch.mean(torch.relu(d_l_knee - max_cross_margin_m) ** 2)
        + torch.mean(torch.relu(d_l_ankle - max_cross_margin_m) ** 2)
        + torch.mean(torch.relu(d_r_knee - max_cross_margin_m) ** 2)
        + torch.mean(torch.relu(d_r_ankle - max_cross_margin_m) ** 2)
    )
    return loss


# Bảng giới hạn ROM (radian) cho 21 khớp body_pose SMPL — index khớp với guide
# ở đầu file (0-20: Hip/Spine/Knee/Ankle/Foot/Neck/Collar/Head/Shoulder/Elbow/Wrist).
# Giá trị lấy theo giới hạn sinh lý TỐI ĐA (không phải giá trị "bình thường"
# trung bình) — chuẩn goniometry lâm sàng AAOS (American Academy of Orthopaedic
# Surgeons, normal joint ROM). Spine1/2/3 vẫn siết vì đây là góc TỔNG HỢP
# (norm cả 3 trục) — twist (xoắn) đã bị giới hạn riêng chặt hơn ở
# compute_spine_twist_loss (trục Y), nên giá trị ở đây chỉ cần đủ cho gập
# người (flexion) tự nhiên, không phải giới hạn xoắn.
JOINT_ROM_LIMITS = {
    0: math.radians(120), 1: math.radians(120),   # L_Hip, R_Hip — hip flexion max 120°
    2: math.radians(30),                          # Spine1 — 90° gập thân chia đều 3 đốt
    3: math.radians(150), 4: math.radians(150),   # L_Knee, R_Knee — knee flexion max 150°
    5: math.radians(30),                          # Spine2
    6: math.radians(50), 7: math.radians(50),     # L_Ankle, R_Ankle — plantarflexion max 50°
    8: math.radians(30),                          # Spine3
    9: math.radians(30), 10: math.radians(30),    # L_Foot, R_Foot (khớp phụ, không có chuẩn lâm sàng riêng)
    11: math.radians(50),                         # Neck — chia sẻ cervical rotation (~80°) với Head
    12: math.radians(25), 13: math.radians(25),   # L_Collar, R_Collar (khớp vai-đòn, biên độ nhỏ)
    14: math.radians(35),                         # Head — phần còn lại của cervical rotation
    15: math.radians(180), 16: math.radians(180), # L_Shoulder, R_Shoulder — flexion/abduction max 180°
    17: math.radians(150), 18: math.radians(150), # L_Elbow, R_Elbow — elbow flexion max 150°
    19: math.radians(90), 20: math.radians(90),   # L_Wrist, R_Wrist — wrist flexion/extension max 90°
}


def compute_joint_angle_limit_loss(
    body_pose: torch.Tensor,
    rom_limits: dict = JOINT_ROM_LIMITS,
) -> torch.Tensor:
    """
    Joint Angle Limit Loss — Phạt độ lớn góc xoay TỔNG HỢP (axis-angle norm,
    kết hợp cả 3 trục) vượt ngưỡng ROM sinh lý cho toàn bộ 21 khớp.

    Args:
        body_pose (torch.Tensor): SMPL body pose, shape (B, 63) hoặc (B, 21, 3).
        rom_limits (dict): joint_idx -> góc tối đa (rad).

    Returns:
        torch.Tensor: Loss scalar.
    """
    if body_pose.dim() == 2:
        body_pose = body_pose.view(body_pose.shape[0], 21, 3)

    total_loss = body_pose.new_zeros(1).squeeze()
    for joint_idx, theta_max in rom_limits.items():
        theta_norm = body_pose[:, joint_idx, :].norm(dim=-1)  # (B,)
        total_loss = total_loss + torch.relu(theta_norm - theta_max).pow(2).mean()

    return total_loss / max(len(rom_limits), 1)

