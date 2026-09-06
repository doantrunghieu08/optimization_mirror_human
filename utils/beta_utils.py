"""
beta_utils.py — Policy shape SMPL dùng chung train / inference / evaluate.
"""

from __future__ import annotations

import torch


def resolve_clip_betas(
    betas_all: torch.Tensor,
    beta_mode: str = "median",
    beta0_correction: float = 0.0,
    beta1_correction: float = 0.0,
    clamp_abs: float = 3.0,
    betas_mirror_all: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Tính betas clip-level shape (1, 10).

    Args:
        betas_all: (N, 10+) từ HMR4D (view real).
        beta_mode: "median" | "first"
        beta0_correction / beta1_correction: cộng thêm sau khi giảm theo mode.
            Để 0.0 khi so baseline MPJPE.
        betas_mirror_all: (N, 10+) từ HMR4D (view mirror), tuỳ chọn. Khi có,
            được gộp cùng real trước khi lấy median/first — real view có thể
            bị occlusion/quay lưng khiến HMR4D ước lượng sai body shape,
            mirror view bù lại thông tin đó thay vì bị bỏ qua hoàn toàn.
    """
    if beta_mode not in {"median", "first"}:
        raise ValueError("beta_mode must be 'median' or 'first'")

    betas = betas_all[..., :10]
    betas_mirror = betas_mirror_all[..., :10] if betas_mirror_all is not None else None

    if beta_mode == "first":
        resolved = betas[0:1].clone()
        if betas_mirror is not None:
            resolved = (resolved + betas_mirror[0:1]) / 2.0
    else:
        combined = torch.cat([betas, betas_mirror], dim=0) if betas_mirror is not None else betas
        resolved = combined.median(dim=0, keepdim=True).values

    resolved = resolved.clone()
    resolved[..., 0] += float(beta0_correction)
    resolved[..., 1] += float(beta1_correction)
    return resolved.clamp(-clamp_abs, clamp_abs)


def expand_clip_betas(clip_betas: torch.Tensor, num_frames: int) -> torch.Tensor:
    """(1, 10) → (N, 10)."""
    if clip_betas.ndim != 2 or clip_betas.shape[0] != 1:
        raise ValueError("clip_betas must have shape (1, 10)")
    return clip_betas.expand(num_frames, -1).contiguous()
