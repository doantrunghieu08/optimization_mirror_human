import torch
import torch.nn as nn
import inspect

# Monkey-patch cho Python 3.11 vì thư viện chumpy cũ vẫn dùng getargspec
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool
    np.int = int
    np.float = float
    np.complex = complex
    np.object = object
    np.unicode = str
    np.str = str

import smplx
import os


def resolve_result_model_path(configured_path: str, result_model_type: str = "smpl") -> str:
    """Old result PKLs used SMPL; new PKLs explicitly record their body model."""
    if result_model_type not in {"smpl", "smplx"}:
        raise ValueError(f"Unsupported result body_model_type: {result_model_type}")
    configured_type = "smplx" if os.path.basename(configured_path).upper().startswith("SMPLX") else "smpl"
    if configured_type == result_model_type:
        return configured_path
    filename = "SMPLX_NEUTRAL.npz" if result_model_type == "smplx" else "SMPL_NEUTRAL.pkl"
    path = os.path.join(os.path.dirname(configured_path), filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Result uses {result_model_type}; matching body model missing: {path}")
    return path

# ============================================================
# Mapping SMPL 45 joints → 17 joints chuẩn COCO (ViTPose)
# (từ configs/keypoins3d_map.yml) — không gồm Neck
# ============================================================
SMPL_TO_COCO_IDX = [24, 26, 25, 28, 27, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]

# ============================================================
# Các cặp khớp đối xứng trái-phải trong SMPL (22 joints, 0-indexed)
# Dùng để "un-mirror" pose lấy từ ảnh gương
# ============================================================
SMPL_FLIP_PAIRS = [
    (1, 2),   # L_Hip   ↔ R_Hip
    (4, 5),   # L_Knee  ↔ R_Knee
    (7, 8),   # L_Ankle ↔ R_Ankle
    (10, 11), # L_Foot  ↔ R_Foot
    (13, 14), # L_Collar ↔ R_Collar
    (16, 17), # L_Shoulder ↔ R_Shoulder
    (18, 19), # L_Elbow  ↔ R_Elbow
    (20, 21), # L_Wrist  ↔ R_Wrist
]

def mirror_axis_angle(pose: torch.Tensor) -> torch.Tensor:
    """
    Áp dụng phép lật gương cho vector axis-angle.
    Với gương lật trục X, thành phần Y và Z của axis-angle sẽ bị đổi dấu.
    """
    mirrored = pose.clone()
    mirrored[..., 1] = -mirrored[..., 1]
    mirrored[..., 2] = -mirrored[..., 2]
    return mirrored


def unmirror_pose(pose: torch.Tensor) -> torch.Tensor:
    """
    Đảo ngược phép chiếu gương (reflection) cho pose dạng axis-angle.

    Khi camera nhìn vào gương, ảnh bị lật trái-phải (reflect theo trục X).
    Với quy ước HMR4D / SMPL hiện tại, phản xạ trục X trên axis-angle
    tương đương negate thành phần Y và Z (không phải X). Sau đó hoán đổi
    các cặp khớp trái ↔ phải.

    Args:
        pose: Tensor shape (..., 22, 3) — axis-angle cho 22 khớp SMPL.

    Returns:
        Tensor shape (..., 22, 3) — pose đã được un-mirror.
    """
    # Bước 1: Áp dụng phép lật cho các vector axis-angle
    pose = mirror_axis_angle(pose)

    # Bước 2: Hoán đổi cặp khớp trái ↔ phải
    for l_idx, r_idx in SMPL_FLIP_PAIRS:
        pose[..., l_idx, :], pose[..., r_idx, :] = (
            pose[..., r_idx, :].clone(),
            pose[..., l_idx, :].clone(),
        )

    return pose


def remirror_body_pose(body_pose: torch.Tensor) -> torch.Tensor:
    """
    Lật ngược lại body_pose (21 khớp) sau khi đã fuse, để trả về không gian gốc của gương.
    Vì body_pose không chứa global_orient (khớp 0), các index hoán đổi phải giảm đi 1.
    """
    # Bước 1: Áp dụng phép lật cho axis-angle
    bp = mirror_axis_angle(body_pose)
    
    # Bước 2: Hoán đổi cặp khớp trái phải (lưu ý trừ đi 1 vì mảng này không có global_orient)
    for l_idx, r_idx in SMPL_FLIP_PAIRS:
        if l_idx > 0 and r_idx > 0: # Bỏ qua global_orient nếu có
            bp[..., l_idx-1, :], bp[..., r_idx-1, :] = (
                bp[..., r_idx-1, :].clone(),
                bp[..., l_idx-1, :].clone(),
            )
            
    return bp


def unmirror_transl(transl: torch.Tensor) -> torch.Tensor:
    """
    Đảo ngược phép chiếu gương cho vector translation.

    Khi reflect theo trục X: (tx, ty, tz) → (-tx, ty, tz).

    Args:
        transl: Tensor shape (..., 3).

    Returns:
        Tensor shape (..., 3).
    """
    transl = transl.clone()
    transl[..., 0] = -transl[..., 0]
    return transl


class SMPLForwardPass(nn.Module):
    """
    SMPL/SMPL-X forward pass returning COCO-17 plus a shoulder-midpoint neck.
    SMPL-X uses GVHMR's surface landmark regressor and mean hand pose.
    """

    def __init__(self, model_path: str, device: torch.device, batch_size: int = 1):
        super().__init__()
        self.device = device
        self.model_path = model_path
        self.model_type = "smplx" if os.path.basename(model_path).upper().startswith("SMPLX") else "smpl"
        self.register_buffer("coco_regressor", None)
        if self.model_type == "smplx":
            regressor_path = os.path.join(os.path.dirname(model_path), "smplx_coco17_J_regressor.pt")
            # GVHMR uses COCO landmarks regressed from the surface, rather than
            # anatomical hip/shoulder joint centres or SMPL's face vertex indices.
            regressor = torch.load(regressor_path, map_location=device, weights_only=True)
            if regressor.shape != (17, 10475):
                raise ValueError(f"Expected a (17, 10475) SMPL-X COCO regressor: {regressor_path}")
            self.coco_regressor = regressor.float()
        self._smpl_cache = nn.ModuleDict()

        # Initialize default batch size
        self.smpl = self._get_smpl(int(batch_size))

    def _get_smpl(self, batch_size: int):
        key = str(batch_size)
        if key not in self._smpl_cache:
            extra = {"ext": "npz", "num_pca_comps": 12, "flat_hand_mean": False} if self.model_type == "smplx" else {}
            self._smpl_cache[key] = smplx.create(
                model_path=self.model_path,
                model_type=self.model_type,
                gender="neutral",
                batch_size=batch_size,
                **extra,
            ).to(self.device).requires_grad_(False)
        return self._smpl_cache[key]

    def forward(
        self,
        global_orient: torch.Tensor,
        body_pose: torch.Tensor,
        betas: torch.Tensor,
        transl: torch.Tensor,
        return_mesh: bool = False,
    ):
        """
        Args:
            global_orient : (B, 3)
            body_pose     : (B, 63)   — 21 khớp * 3
            betas         : (B, 10)
            transl        : (B, 3)
            return_mesh   : bool

        Returns:
            Nếu return_mesh = False:
                joints_coco: (B, 17, 3)
            Nếu return_mesh = True:
                joints_coco: (B, 17, 3)
                vertices: (B, 6890, 3)
                faces: (13776, 3)
        """
        B = global_orient.shape[0]
        # Keep full-sequence temporal losses while bounding the memory used by
        # SMPL-X skinning. Checkpoint only the differentiable joint passes.
        if B > 32:
            from torch.utils.checkpoint import checkpoint
            chunks = []
            for start in range(0, B, 32):
                args = tuple(value[start:start + 32] for value in (global_orient, body_pose, betas, transl))
                if not return_mesh and torch.is_grad_enabled() and any(value.requires_grad for value in args):
                    chunks.append(checkpoint(self.forward, *args, use_reentrant=False))
                else:
                    chunks.append(self.forward(*args, return_mesh=return_mesh))
            if return_mesh:
                return torch.cat([c[0] for c in chunks]), torch.cat([c[1] for c in chunks]), chunks[0][2]
            return torch.cat(chunks)
        self.smpl = self._get_smpl(B)

        # smplx mặc định yêu cầu 69 tham số cho body_pose (23 khớp x 3)
        # Nếu data chỉ có 63 (21 khớp), ta pad thêm 6 số 0 (2 khớp bàn tay)
        if self.model_type == "smpl" and body_pose.shape[-1] == 63:
            padding = torch.zeros(body_pose.shape[0], 6, device=body_pose.device, dtype=body_pose.dtype)
            body_pose = torch.cat([body_pose, padding], dim=-1)

        output = self.smpl(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            transl=transl,
            return_verts=return_mesh or self.coco_regressor is not None,
        )
        # output.joints: (B, 45, 3)
        if self.coco_regressor is not None:
            joints_coco = torch.einsum("jv,bvc->bjc", self.coco_regressor, output.vertices)
        else:
            joints_coco = output.joints[:, SMPL_TO_COCO_IDX, :].clone()
        
        # Add Neck joint (index 17) as midpoint of Left Shoulder (5) and Right Shoulder (6)
        neck = (joints_coco[:, 5:6, :] + joints_coco[:, 6:7, :]) / 2.0
        joints_coco = torch.cat([joints_coco, neck], dim=1) # (B, 18, 3)
        
        # Không chỉnh sửa thủ công vị trí ankle/knee — SMPL đã tự tính đúng
        # từ body_pose parameters. Bất kỳ hack alpha nào đều gây sai lệch giải phẫu.
        if return_mesh:
            return joints_coco, output.vertices, self.smpl.faces
        return joints_coco
