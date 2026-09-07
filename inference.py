"""
inference.py
============
Pipeline: surface-landmark ray-casting + evidence-weighted SMPLify-style pose
refit (xem utils/pose_refit.py).

Đầu ra: file .pkl chứa SMPL params cho mỗi frame — CÙNG schema với trước, để
tương thích render_video.py / evaluate.py.

Tách biệt hoàn toàn với việc render video (xem render_video.py).

Usage:
    python inference.py --config configs/default.yaml
    python inference.py --config configs/default.yaml --bypass_refit   # debug: dùng thẳng real pose, bỏ qua refit
"""

import os
import glob
import argparse
import yaml
import torch
import pickle

from utils.smpl_utils import SMPLForwardPass, unmirror_pose
from utils.beta_utils import resolve_clip_betas, expand_clip_betas
from utils.pose_refit import run_pose_refit
from utils.geometry import transfer_orientation


def run_inference(config, bypass_refit: bool = False) -> str:
    """
    Chạy surface-visibility refit trên toàn bộ sequence và lưu kết quả ra file PKL.

    Args:
        config        : dict config từ YAML.
        bypass_refit  : nếu True, bỏ qua ray-casting/belief/refit, dùng thẳng real pose (debug).

    Returns:
        Đường dẫn tới file PKL đã lưu.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smpl = SMPLForwardPass(
        model_path=config["model"]["smpl_model_path"],
        device=device,
    )

    # ── 1. Đường dẫn output ─────────────────────────────────────────────────
    output_dir = config.get("visualization", {}).get("output_dir", "output")
    os.makedirs(output_dir, exist_ok=True)
    output_pkl = os.path.join(output_dir, "inference_results.pkl")
    output_diagnostics_pkl = os.path.join(output_dir, "belief_diagnostics.pkl")

    # ── 2. Load dữ liệu đầu vào ──────────────────────────────────────────────
    real_dir = config["data"]["real_dir"]
    real_pt  = real_dir if real_dir.endswith(".pt") else os.path.join(real_dir, "hmr4d_results.pt")
    real_data = torch.load(real_pt, map_location="cpu", weights_only=True)

    mirror_dir = config["data"]["mirror_dir"]
    mirror_pt  = mirror_dir if mirror_dir.endswith(".pt") else os.path.join(mirror_dir, "hmr4d_results.pt")
    mirror_data = torch.load(mirror_pt, map_location="cpu", weights_only=True)

    def load_keypoints(pt_path: str, expected_frames: int) -> torch.Tensor | None:
        candidates = glob.glob(
            os.path.join(os.path.dirname(pt_path), "preprocess", "vitpose*.pt")
        )
        if not candidates:
            return None
        values = torch.load(candidates[0], map_location="cpu", weights_only=True)
        if values.ndim != 3 or values.shape[1:] != (17, 3):
            raise ValueError(f"Invalid ViTPose shape in {candidates[0]}: {tuple(values.shape)}")
        if values.shape[0] != expected_frames:
            raise ValueError(
                f"ViTPose frame mismatch in {candidates[0]}: "
                f"pose={expected_frames}, keypoints={values.shape[0]}"
            )
        return values

    n_real = real_data["smpl_params_incam"]["global_orient"].shape[0]
    n_mirror = mirror_data["smpl_params_incam"]["global_orient"].shape[0]
    if n_real == 0:
        raise ValueError("Inference inputs contain no frames")
    if n_real != n_mirror:
        raise ValueError(f"Frame count mismatch in inference: real={n_real}, mirror={n_mirror}")
    kp_real = load_keypoints(real_pt, n_real)
    kp_mirror = load_keypoints(mirror_pt, n_mirror)
    if (kp_real is None) != (kp_mirror is None):
        raise ValueError("ViTPose confidence must be available for both real and mirror inputs")
    if not bypass_refit and kp_real is None:
        raise ValueError(
            "Belief-theory refit requires ViTPose 2D keypoints for both real and mirror views "
            "(b_det/b_reproj cần chúng). Dùng --bypass_refit nếu chỉ muốn xuất pose real gốc."
        )

    # ── 3. Betas (shape) — cố định, không tối ưu ─────────────────────────────
    # Keep real shape by default so pose refinement has an unchanged baseline.
    betas_all = real_data["smpl_params_incam"]["betas"]
    betas_mirror_all = mirror_data["smpl_params_incam"]["betas"]
    beta_mode = config.get("data", {}).get("beta_mode", "median")
    beta_source = config.get("data", {}).get("beta_source", "real")
    if beta_source not in {"real", "both"}:
        raise ValueError("beta_source must be 'real' or 'both'")
    vis_cfg = config.get("visualization", {})
    max_frames = vis_cfg.get("max_frames", -1)
    if max_frames == 0 or max_frames < -1:
        raise ValueError("max_frames must be -1 or a positive integer")
    n_frames = n_real if max_frames == -1 else min(n_real, max_frames)

    def frame_betas(values: torch.Tensor, label: str) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] < 10 or values.shape[0] not in {1, n_real}:
            raise ValueError(
                f"Invalid {label} betas shape {tuple(values.shape)}; expected (1, >=10) "
                f"or ({n_real}, >=10)"
            )
        if not torch.isfinite(values).all():
            raise ValueError(f"{label} betas contain non-finite values")
        values = values[:, :10]
        return values[:1].expand(n_frames, -1).clone() if len(values) == 1 else values[:n_frames].clone()

    betas_real_frames = frame_betas(betas_all, "real")
    betas_mirror_frames = frame_betas(betas_mirror_all, "mirror")

    if bypass_refit:
        betas_policy = betas_real_frames
    elif beta_mode == "per_frame":
        betas_policy = betas_real_frames
        if beta_source == "both":
            betas_policy = (betas_policy + betas_mirror_frames) / 2.0
        betas_policy[..., 0] += float(vis_cfg.get("beta0_correction", 0.0))
        betas_policy[..., 1] += float(vis_cfg.get("beta1_correction", 0.0))
    else:
        betas_policy = resolve_clip_betas(
            betas_all,
            beta_mode=beta_mode,
            beta0_correction=vis_cfg.get("beta0_correction", 0.0),
            beta1_correction=vis_cfg.get("beta1_correction", 0.0),
            betas_mirror_all=betas_mirror_all if beta_source == 'both' else None,
        )
        betas_policy = expand_clip_betas(betas_policy, n_frames)
    print(f"Body shape betas ({beta_mode}, corrected): {betas_policy[0].numpy().round(3)}")

    # ── 4. Chuẩn bị tensor cho toàn bộ sequence (N frame) ────────────────────
    real_go     = real_data["smpl_params_incam"]["global_orient"][:n_frames].to(device)
    real_bp     = real_data["smpl_params_incam"]["body_pose"][:n_frames].to(device)
    real_transl = real_data["smpl_params_incam"]["transl"][:n_frames].to(device)
    K_real      = real_data["K_fullimg"][:n_frames].to(device)
    betas       = betas_policy[:n_frames].to(device)

    global_go = real_data["smpl_params_global"]["global_orient"][:n_frames].to(device)
    global_tr = real_data["smpl_params_global"]["transl"][:n_frames].to(device)

    mirror_go_raw     = mirror_data["smpl_params_incam"]["global_orient"][:n_frames].to(device)
    mirror_bp_raw     = mirror_data["smpl_params_incam"]["body_pose"][:n_frames].to(device)
    mirror_transl_raw = mirror_data["smpl_params_incam"]["transl"][:n_frames].to(device)
    K_mirror          = mirror_data["K_fullimg"][:n_frames].to(device)

    # Un-mirror pose ảnh gương về quy ước trái/phải của real — đây là NGUỒN
    # QUAN SÁT CỐ ĐỊNH thứ hai cho belief fusion (mục 2, 4 của tài liệu).
    full_mirror_pose = torch.cat(
        [mirror_go_raw.unsqueeze(1), mirror_bp_raw.view(-1, 21, 3)], dim=1
    )

    # Local SMPL/SMPL-X rotations use the canonical sagittal reflection.
    # Physical mirror geometry affects the camera root, not every local joint.
    # The refit derives a relative camera rotation from the original root pair.
    unmirrored = unmirror_pose(full_mirror_pose)
    mirror_go_unmirrored = unmirrored[:, 0, :]
    mirror_bp_unmirrored = unmirrored[:, 1:, :].reshape(-1, 63)

    kp2d_real = kp_conf_real = kp2d_mirror = kp_conf_mirror = None
    if kp_real is not None:
        kp2d_real       = kp_real[:n_frames, :, :2].to(device)
        kp_conf_real    = kp_real[:n_frames, :, 2].to(device)
        kp2d_mirror     = kp_mirror[:n_frames, :, :2].to(device)
        kp_conf_mirror  = kp_mirror[:n_frames, :, 2].to(device)

    # ── 5. Ray-casting occlusion + belief theory + SMPLify-style refit ───────
    diagnostics: dict = {}
    if bypass_refit:
        print("bypass_refit: exporting original real HMR4D pose and shape.")
        fused_go, fused_bp = real_go, real_bp
        fused_go_global = global_go
    else:
        refit_cfg = config.get("refit", {})
        print(f"Running surface-visibility refit on {n_frames} frames "
              f"(outer_iterations={refit_cfg.get('outer_iterations', 3)}, "
              f"inner_steps={refit_cfg.get('inner_steps', 150)})...")
        result = run_pose_refit(
            real_global_orient=real_go,
            real_body_pose=real_bp,
            mirror_global_orient_unmirrored=mirror_go_unmirrored,
            mirror_body_pose_unmirrored=mirror_bp_unmirrored,
            betas=betas,
            real_transl=real_transl,
            mirror_transl_raw=mirror_transl_raw,
            mirror_go_raw=mirror_go_raw,
            mirror_bp_raw=mirror_bp_raw,
            kp2d_real=kp2d_real, kp2d_conf_real=kp_conf_real, K_real=K_real,
            kp2d_mirror=kp2d_mirror, kp2d_conf_mirror=kp_conf_mirror, K_mirror=K_mirror,
            smpl=smpl,
            mirror_betas=betas,
            frame_rate=config.get("data", {}).get("frame_rate", 30.0),
            outer_iterations=refit_cfg.get("outer_iterations", 3),
            inner_steps=refit_cfg.get("inner_steps", 150),
            lr=refit_cfg.get("lr", 0.01),
            reproj_sigma=refit_cfg.get("reproj_sigma", 50.0),
            belief_combine=refit_cfg.get("belief_combine", "dempster_shafer"),
            loss_weights=refit_cfg.get("loss_weights"),
            convergence_deg=refit_cfg.get("convergence_deg", 0.5),
            occlusion_near_eps=refit_cfg.get("occlusion_near_eps", 0.02),
            occlusion_far_ratio=refit_cfg.get("occlusion_far_ratio", 0.985),
            post_smooth=refit_cfg.get("post_smooth", False),
            preserve_real_projection=refit_cfg.get("preserve_real_projection", True),
            acceptance_temporal_tolerance=refit_cfg.get("acceptance_temporal_tolerance", 1.05),
        )
        fused_go, fused_bp = result["global_orient"], result["body_pose"]
        diagnostics = result["diagnostics"]

        # Preserve the original camera-to-world transform for the 3D panel.
        fused_go_global = transfer_orientation(real_go, global_go, fused_go)

    # Overlay incam luôn dùng translation người thật — không interpolate mirror.
    results = {
        "body_model_type":      smpl.model_type,
        "global_orient":        fused_go.cpu().numpy(),
        "global_orient_global": fused_go_global.cpu().numpy(),
        "body_pose":            fused_bp.cpu().numpy(),
        "betas":                betas.cpu().numpy(),
        "transl_incam":         real_transl.cpu().numpy(),
        "transl_global":        global_tr.cpu().numpy(),
        "K_fullimg":            K_real.cpu().numpy(),
    }
    if kp_real is not None:
        results["kp2d_real"] = kp_real[:n_frames, :, :2].numpy()
        results["kp2d_conf_real"] = kp_real[:n_frames, :, 2].numpy()

    # Lưu belief vào PKL chính để render_video dùng cho panel dual 2D
    if diagnostics:
        if "belief_real" in diagnostics:
            results["belief_real"] = diagnostics["belief_real"].numpy()
        if "belief_mirror" in diagnostics:
            results["belief_mirror"] = diagnostics["belief_mirror"].numpy()

    with open(output_pkl, "wb") as f:
        pickle.dump(results, f)

    if diagnostics:
        with open(output_diagnostics_pkl, "wb") as f:
            pickle.dump({k: v.numpy() for k, v in diagnostics.items()}, f)
        print(f"Belief diagnostics saved to: {output_diagnostics_pkl}")

    print(f"Inference complete. Results saved to: {output_pkl}")
    return output_pkl


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Surface-visibility fusion + SMPLify-style refit — chỉ suy luận, không render video."
    )
    parser.add_argument("--config",        type=str, default="configs/default.yaml")
    parser.add_argument("--bypass_refit",  action="store_true",
                        help="Bỏ qua ray-casting/belief/refit, dùng thẳng real pose (debug).")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_inference(config, bypass_refit=args.bypass_refit)
