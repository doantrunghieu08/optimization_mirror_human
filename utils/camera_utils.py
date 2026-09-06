import torch

def project_3d_to_2d(joints_3d, camera_intrinsics, min_depth=1e-5):
    """
    Chiếu tọa độ 3D (X, Y, Z) xuống mặt phẳng 2D (u, v) bằng ma trận nội (Intrinsic Matrix K).
    
    Args:
        joints_3d (torch.Tensor): Tensor chứa tọa độ 3D khớp, shape (batch_size, num_joints, 3)
        camera_intrinsics (torch.Tensor): Ma trận camera K, shape (batch_size, 3, 3)
        min_depth (float): Khoảng cách z tối thiểu để coi là hợp lệ.
        
    Returns:
        tuple: (Tọa độ 2D shape (B, J, 2), Mask hợp lệ shape (B, J))
    """
    joints_3d_t = joints_3d.transpose(1, 2)
    projected_homogeneous = torch.bmm(camera_intrinsics, joints_3d_t).transpose(1, 2)
    
    depth = projected_homogeneous[..., 2:3]
    valid = depth > min_depth
    safe_depth = torch.where(valid, depth, torch.ones_like(depth))
    
    joints_2d_projected = projected_homogeneous[..., :2] / safe_depth
    
    return joints_2d_projected, valid.squeeze(-1)

def calculate_reprojection_loss(projected_2d, detected_2d, confidences, valid_depth_mask=None, sigma=100.0):
    """
    Tính loss (sai số) giữa tọa độ 2D chiếu xuống từ 3D và tọa độ 2D gốc detect được (như YOLO/ViTPose).
    Sử dụng Geman-McClure Robust Loss để loại bỏ ảnh hưởng của các điểm nhận diện nhầm xa (outliers).
    
    Args:
        projected_2d (torch.Tensor): Tọa độ 2D chiếu từ model 3D (batch_size, num_joints, 2)
        detected_2d (torch.Tensor): Tọa độ 2D nhận diện trên ảnh (batch_size, num_joints, 2)
        confidences (torch.Tensor): Độ tin cậy của tọa độ 2D nhận diện (batch_size, num_joints)
        valid_depth_mask (torch.Tensor): Mask các điểm có Z > 0, shape (batch_size, num_joints)
        sigma (float): Ngưỡng chịu đựng sai số. Lệch lớn hơn ngưỡng này, điểm đó sẽ bị bỏ qua.
        
    Returns:
        torch.Tensor: Giá trị loss
    """
    sq_error = torch.sum((projected_2d - detected_2d) ** 2, dim=-1)
    robust_error = sq_error / (sq_error + sigma ** 2)
    
    if valid_depth_mask is not None:
        valid = (confidences > 0) & valid_depth_mask
    else:
        valid = confidences > 0
        
    weighted_error = robust_error * confidences * valid.to(confidences.dtype)
    denominator = (confidences * valid.to(confidences.dtype)).sum().clamp_min(1e-8)
    
    return weighted_error.sum() / denominator
