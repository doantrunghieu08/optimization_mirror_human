"""
mesh_raycast.py — Phát hiện self-occlusion bằng ray-casting trên mesh SMPL.

Ý tưởng (xem HuongTiepCanMoi.md, mục 1):
    Với mỗi joint, bắn một tia (ray) từ tâm camera tới vị trí 3D của joint đó.
    Nếu tia này bị một mặt (face) khác của mesh cắt ngang TRƯỚC khi tới joint,
    joint được coi là bị che khuất bởi chính cơ thể (self-occlusion).

Quy ước hệ tọa độ:
    Cả view "real" và view "mirror" trong pipeline này đều đến từ hai lần chạy
    HMR4D độc lập (một lần trên ảnh thật, một lần trên ảnh phản chiếu qua
    gương). Mỗi lần chạy trả về pose ở dạng "incam": tâm camera nằm ở gốc tọa
    độ (0, 0, 0) và toàn bộ mesh/joints nằm ở phía dương trục Z. Vì vậy KHÔNG
    cần dựng "virtual camera" bằng ma trận phản xạ Householder như mô tả ở
    mục 2 của tài liệu — chỉ cần dựng mesh riêng cho mỗi view (bằng forward
    kinematics với params của đúng view đó) rồi bắn tia từ gốc tọa độ.

Dùng trimesh làm bộ tăng tốc ray–mesh intersection. Belief chính dùng các
vertex bề mặt tạo nên từng COCO landmark, không dùng tâm khớp nằm trong mesh.
"""

from __future__ import annotations

import numpy as np
import torch
import trimesh


def surface_landmarks_from_regressor(
    joints_3d: torch.Tensor,
    vertices: torch.Tensor,
    regressor: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return actual mesh vertices and weights for each COCO surface landmark."""
    if regressor is None:
        # ponytail: legacy SMPL has no COCO surface regressor; replace this
        # nearest-vertex fallback if that model becomes a production input.
        indices = torch.cdist(joints_3d, vertices).argmin(dim=-1)
        points = vertices.gather(1, indices[..., None].expand(-1, -1, 3))
        return points.unsqueeze(2), joints_3d.new_ones(joints_3d.shape[:2] + (1,))

    positive = regressor.to(device=vertices.device, dtype=vertices.dtype).clamp_min(0)
    count = int((positive > 0).sum(dim=-1).max().item())
    weights, indices = positive.topk(max(count, 1), dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return vertices[:, indices], weights


def compute_ray_visibility(
    surface_points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    camera_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    surface_eps: float = 0.02,
    far_ratio: float = 0.985,
) -> np.ndarray:
    """Continuous visibility of mesh-surface targets: 1=visible, 0=blocked."""
    targets = np.asarray(surface_points, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    origin = np.asarray(camera_origin, dtype=np.float64)
    directions = targets - origin[None]
    distances = np.linalg.norm(directions, axis=-1)
    valid = np.isfinite(directions).all(axis=-1) & (distances > 1e-8)
    visibility = np.zeros(len(targets), dtype=np.float64)
    visibility[valid] = 1.0
    if not valid.any():
        return visibility

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    valid_indices = np.flatnonzero(valid)
    locations, ray_indices, _ = mesh.ray.intersects_location(
        np.repeat(origin[None], len(valid_indices), axis=0),
        directions[valid] / distances[valid, None],
        multiple_hits=False,
    )
    if len(locations) == 0:
        return visibility

    hit_distances = np.linalg.norm(locations - origin[None], axis=-1)
    first_hit = np.full(len(valid_indices), np.inf)
    first_hit[ray_indices] = hit_distances
    target_distances = distances[valid]
    gap = target_distances - first_hit
    softness = np.maximum(target_distances * (1.0 - far_ratio), 1e-6)
    visibility[valid_indices] = 1.0 - np.clip((gap - surface_eps) / softness, 0.0, 1.0)
    return visibility


def compute_ray_occlusion(
    joints_3d: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    camera_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    near_eps: float = 0.02,
    far_ratio: float = 0.985,
) -> np.ndarray:
    """
    Ray-cast từ `camera_origin` tới từng joint trong `joints_3d` qua mesh (vertices, faces).

    Một joint bị coi là occluded nếu tồn tại giao điểm ray–mesh nằm trong
    khoảng (near_eps, far_ratio * khoảng_cách_tới_joint) — tức nằm TRƯỚC joint,
    không tính các giao điểm nằm ngay sát bề mặt da quanh chính joint đó
    (loại nhiễu "false positive" do mesh tự cắt chính bề mặt tại joint).

    Args:
        joints_3d     : (J, 3) tọa độ 3D các khớp, cùng hệ tọa độ với mesh.
        vertices      : (V, 3) đỉnh mesh SMPL.
        faces         : (F, 3) chỉ số mặt tam giác.
        camera_origin : tâm quang học của camera (mặc định gốc tọa độ, quy ước "incam").
        near_eps      : khoảng cách tối thiểu (m) trước khi tính là giao điểm hợp lệ.
        far_ratio     : tỉ lệ khoảng cách tới joint — giao điểm xa hơn ngưỡng này
                        (tức nằm ngay tại/gần bề mặt da của joint) bị bỏ qua.

    Returns:
        occluded : (J,) bool — True nếu joint bị self-occluded.
    """
    joints_3d = np.asarray(joints_3d, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    origin = np.asarray(camera_origin, dtype=np.float64)

    J = joints_3d.shape[0]
    occluded = np.zeros(J, dtype=bool)

    directions = joints_3d - origin[None, :]
    dists = np.linalg.norm(directions, axis=-1)
    safe_dists = np.clip(dists, 1e-8, None)
    unit_dirs = directions / safe_dists[:, None]

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    origins = np.repeat(origin[None, :], J, axis=0)

    locations, index_ray, _ = mesh.ray.intersects_location(
        origins, unit_dirs, multiple_hits=True
    )
    if len(locations) == 0:
        return occluded

    hit_dists = np.linalg.norm(locations - origin[None, :], axis=-1)
    for ray_idx in np.unique(index_ray):
        mask = index_ray == ray_idx
        far_limit = dists[ray_idx] * far_ratio
        valid_hit = (hit_dists[mask] > near_eps) & (hit_dists[mask] < far_limit)
        if np.any(valid_hit):
            occluded[ray_idx] = True

    return occluded


def compute_occlusion_belief_batch(
    surface_landmarks: torch.Tensor,
    vertices: torch.Tensor,
    faces: np.ndarray | torch.Tensor,
    surface_weights: torch.Tensor | None = None,
    camera_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    near_eps: float = 0.02,
    far_ratio: float = 0.985,
) -> torch.Tensor:
    """Weighted continuous visibility for (B, J, K, 3) surface regions."""
    if surface_landmarks.ndim == 3:
        surface_landmarks = surface_landmarks.unsqueeze(2)
    if surface_weights is None:
        surface_weights = surface_landmarks.new_ones(surface_landmarks.shape[1:3])

    landmarks_np = surface_landmarks.detach().cpu().numpy()
    verts_np = vertices.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy() if isinstance(faces, torch.Tensor) else np.asarray(faces)
    weights_np = surface_weights.detach().cpu().numpy()
    if weights_np.ndim == 2:
        weights_np = np.broadcast_to(weights_np[None], landmarks_np.shape[:3])

    B, J, _, _ = landmarks_np.shape
    belief = np.ones((B, J), dtype=np.float32)
    frame_range = range(B)
    if B > 20:
        try:
            from tqdm import tqdm
            frame_range = tqdm(frame_range, desc="ray-cast occlusion", leave=False)
        except ImportError:
            pass
    for b in frame_range:
        active = weights_np[b] > 0
        point_visibility = compute_ray_visibility(
            landmarks_np[b][active], verts_np[b], faces_np,
            camera_origin=camera_origin, surface_eps=near_eps, far_ratio=far_ratio,
        )
        visibility = np.zeros_like(weights_np[b], dtype=np.float64)
        visibility[active] = point_visibility
        belief[b] = (
            (visibility * weights_np[b]).sum(axis=-1)
            / np.maximum(weights_np[b].sum(axis=-1), 1e-8)
        )

    return torch.from_numpy(belief).to(
        device=surface_landmarks.device, dtype=surface_landmarks.dtype
    )
