"""Refine a shared body pose from independent real/mirror observations.

Source meshes provide visibility and reprojection evidence for a single
belief-weighted pose anchor. The optimizer uses independent detector confidence
for its two reprojection losses, plus anatomical and temporal regularization.
Local rotations use canonical body reflection; root updates are transferred
through the original camera orientation pair. A fixed per-frame projection
check retains the best pose improving both views, including the real baseline.
Betas and translations stay fixed. Skinning is chunked without breaking the
full-sequence temporal objective.
"""

from __future__ import annotations

import torch

from utils.belief_fusion import (
    compute_angular_discrepancy_penalty,
    compute_cross_view_trust,
    compute_full_view_belief,
    flip_coco_belief,
    fuse_pose_precision_weighted,
    map_coco_belief_to_smpl_body,
    map_smpl_body_scale_to_coco,
    propagate_confidence_to_kinematic_neighbors,
    root_belief,
    smooth_belief_temporal,
)
from utils.camera_utils import calculate_reprojection_loss, project_3d_to_2d
from utils.geometry import axis_angle_to_quaternion, rotation_geodesic_loss, transfer_orientation
from utils.penetration import compute_penetration_loss
from utils.smpl_utils import SMPLForwardPass, mirror_axis_angle, remirror_body_pose
from losses.prior_losses import (
    compute_anatomical_limits_loss,
    compute_bone_length_stability_loss,
    compute_bone_symmetry_loss,
    compute_elbow_limits_loss,
    compute_head_collision_loss,
    compute_hip_adduction_loss,
    compute_hip_split_loss,
    compute_joint_angle_limit_loss,
    compute_leg_crossing_loss,
    compute_pose_acceleration_smoothness_loss,
    compute_pose_prior_loss,
    compute_pose_temporal_smoothness_loss,
    compute_shoulder_hyperextension_loss,
    compute_spine_twist_loss,
    compute_temporal_smoothness_loss,
    compute_wrist_bend_loss,
)

# w_temporal/w_bone_stability (không gian vị trí 3D) + w_pose_temporal (không
# gian axis-angle) cùng chống "nhảy khung xương" giữa các frame liên tiếp —
# w_pose_temporal đặt cao vì đây là nguyên nhân trực tiếp nhất của rung giật
# khi render (root/khớp xoay tại chỗ mà vị trí 3D gần như không đổi).
# w_spine_twist, w_joint_angle_limit, w_elbow, w_pose_prior tăng mạnh để
# ngăn Optimizer bẻ xoắn cột sống / vai / khuỷu tay phi thực tế.
DEFAULT_LOSS_WEIGHTS = {
    "w_reproj_real": 1.0,
    "w_reproj_mirror": 1.0,
    "w_pose_prior": 0.6,
    "w_pose_prior_min": 0.25,
    "w_joint_angle_limit": 1.5,
    "w_pose_accel": 2.0,
    "w_anatomical": 1.5,
    "w_elbow": 1.2,
    "w_head_collision": 1.0,
    "w_symmetry": 0.5,
    "w_bone_stability": 0.5,
    "w_temporal": 1.0,
    "w_pose_temporal": 3.0,
    "temporal_confidence_min_weight": 0.5,
    "w_penetration": 0.25,
    "w_anchor": 0.30,
    "w_spine_twist": 2.5,
    "w_hip_split": 1.0,
    "w_leg_crossing": 3.0,
    "w_hip_adduction": 2.0,
    "w_shoulder_hyper": 1.0,
    "w_wrist_bend": 0.8,
    "penetration_radius": 0.10,
    "penetration_sharpness": 30.0,
    "grad_clip_go": 0.5,
    "grad_clip_bp": 1.0,
}


def _get_pose_prior_weight(outer_iter: int, outer_iterations: int, w_min: float, w_max: float) -> float:
    """Tăng tuyến tính w_pose_prior qua các outer loop — tránh siết prior
    quá sớm khi pose khởi tạo còn xa tư thế tự nhiên (kẹt local minimum sai)."""
    if outer_iterations <= 1:
        return w_max
    ratio = outer_iter / (outer_iterations - 1)
    return w_min + ratio * (w_max - w_min)


def _mean_rotation_change_deg(go_a: torch.Tensor, bp_a: torch.Tensor, go_b: torch.Tensor, bp_b: torch.Tensor) -> float:
    """Góc xoay trung bình mỗi khớp (độ) giữa 2 pose — dùng cho log hội tụ."""
    N = bp_a.shape[0]
    pose_a = torch.cat([go_a.view(N, 1, 3), bp_a.view(N, 21, 3)], dim=1)
    pose_b = torch.cat([go_b.view(N, 1, 3), bp_b.view(N, 21, 3)], dim=1)
    qa = axis_angle_to_quaternion(pose_a)
    qb = axis_angle_to_quaternion(pose_b)
    cosine = (qa * qb).sum(dim=-1).abs().clamp(-1.0, 1.0)
    angle_deg = torch.rad2deg(2.0 * torch.acos(cosine))
    return angle_deg.mean().item()


def _forward_view_joints_and_mesh(
    smpl: SMPLForwardPass,
    global_orient: torch.Tensor,
    body_pose: torch.Tensor,
    betas: torch.Tensor,
    transl: torch.Tensor,
    return_mesh: bool,
):
    return smpl(global_orient=global_orient, body_pose=body_pose, betas=betas, transl=transl, return_mesh=return_mesh)


def _to_mirror_frame(
    global_orient: torch.Tensor,
    body_pose: torch.Tensor,
    mirror_normal: torch.Tensor | None = None,
    real_go_reference: torch.Tensor | None = None,
    mirror_go_reference: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chuyển pose hiện tại (hệ real) sang hệ tọa độ gốc của ảnh gương.
    Nếu mirror_normal được cung cấp, sử dụng Householder reflection cho arbitrary mirror plane."""
    N = body_pose.shape[0]
    mirrored_bp = remirror_body_pose(body_pose.view(N, 21, 3)).reshape(N, 63)
    if real_go_reference is not None and mirror_go_reference is not None:
        # Estimate only the inter-view rotation from the original root pair.
        # This preserves the raw mirror camera at initialization and propagates
        # root updates consistently in both the belief and reprojection passes.
        mirrored_go = transfer_orientation(
            mirror_axis_angle(real_go_reference), mirror_go_reference,
            mirror_axis_angle(global_orient),
        )
    elif mirror_normal is not None:
        from utils.mirror_geometry import reflect_global_orientation
        mirrored_go = reflect_global_orientation(global_orient, mirror_normal)
    else:
        mirrored_go = mirror_axis_angle(global_orient)
    return mirrored_go, mirrored_bp


def _compute_view_belief(
    smpl: SMPLForwardPass,
    global_orient: torch.Tensor,
    body_pose: torch.Tensor,
    betas: torch.Tensor,
    transl: torch.Tensor,
    kp2d: torch.Tensor,
    kp2d_conf: torch.Tensor,
    K: torch.Tensor,
    reproj_sigma: float,
    combine_method: str,
    occlusion_near_eps: float = 0.02,
    occlusion_far_ratio: float = 0.985,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        joints18, vertices, faces = _forward_view_joints_and_mesh(
            smpl, global_orient, body_pose, betas, transl, return_mesh=True
        )
        joints17 = joints18[:, :17, :]
        projected_2d, valid_mask = project_3d_to_2d(joints17, K)
        belief_info = compute_full_view_belief(
            joints17, vertices, faces,
            kp2d_detected=kp2d, kp2d_conf=kp2d_conf,
            projected_2d=projected_2d, valid_mask=valid_mask,
            reproj_sigma=reproj_sigma, combine_method=combine_method,
            occlusion_near_eps=occlusion_near_eps, occlusion_far_ratio=occlusion_far_ratio,
        )
    return belief_info


def run_pose_refit(
    real_global_orient: torch.Tensor,
    real_body_pose: torch.Tensor,
    mirror_global_orient_unmirrored: torch.Tensor,
    mirror_body_pose_unmirrored: torch.Tensor,
    betas: torch.Tensor,
    real_transl: torch.Tensor,
    mirror_transl_raw: torch.Tensor,
    mirror_go_raw: torch.Tensor,
    mirror_bp_raw: torch.Tensor,
    kp2d_real: torch.Tensor,
    kp2d_conf_real: torch.Tensor,
    K_real: torch.Tensor,
    kp2d_mirror: torch.Tensor,
    kp2d_conf_mirror: torch.Tensor,
    K_mirror: torch.Tensor,
    smpl: SMPLForwardPass,
    mirror_betas: torch.Tensor | None = None,
    frame_rate: float = 30.0,
    outer_iterations: int = 3,
    inner_steps: int = 150,
    lr: float = 0.01,
    reproj_sigma: float = 50.0,
    belief_combine: str = "dempster_shafer",
    loss_weights: dict | None = None,
    convergence_deg: float = 0.5,
    occlusion_near_eps: float = 0.02,
    occlusion_far_ratio: float = 0.985,
    mirror_normal: torch.Tensor | None = None,
    verbose: bool = True,
    post_smooth: bool = False,
    preserve_real_projection: bool = True,
) -> dict[str, torch.Tensor]:
    """Optimize all frames and return parameters plus source/acceptance diagnostics.

    The unmirrored body must use canonical X reflection and anatomical L/R swap.
    Root orientations from independent cameras are never averaged. The legacy
    mirror_global_orient_unmirrored and mirror_normal arguments remain accepted
    for callers; root transfer uses real_global_orient and mirror_go_raw.
    Diagnostic mirror beliefs are in real anatomical order; *_raw arrays and
    detector confidences are in their original view's COCO order.
    """
    N = real_body_pose.shape[0]
    w = {**DEFAULT_LOSS_WEIGHTS, **(loss_weights or {})}

    current_go = real_global_orient.clone()
    current_bp = real_body_pose.clone()
    prev_go = None
    prev_bp = None
    diagnostics: dict[str, torch.Tensor] = {}

    if N == 0 or outer_iterations < 1 or inner_steps < 1:
        raise ValueError("Refit requires frames, outer_iterations >= 1 and inner_steps >= 1")
    if not torch.isfinite(torch.tensor(frame_rate)) or frame_rate <= 0:
        raise ValueError("frame_rate must be a finite positive number")
    mirror_betas = betas if mirror_betas is None else mirror_betas
    temporal_scale = float(frame_rate / 30.0)

    # Reliability describes each ORIGINAL observation, not the current fused body.
    belief_real_info = _compute_view_belief(
        smpl, real_global_orient, real_body_pose, betas, real_transl,
        kp2d_real, kp2d_conf_real, K_real, reproj_sigma, belief_combine,
        occlusion_near_eps, occlusion_far_ratio,
    )
    belief_mirror_info = _compute_view_belief(
        smpl, mirror_go_raw, mirror_bp_raw, mirror_betas, mirror_transl_raw,
        kp2d_mirror, kp2d_conf_mirror, K_mirror, reproj_sigma, belief_combine,
        occlusion_near_eps, occlusion_far_ratio,
    )

    @torch.no_grad()
    def projection_errors(go, bp):
        mirror_go, mirror_bp = _to_mirror_frame(
            go, bp, real_go_reference=real_global_orient, mirror_go_reference=mirror_go_raw,
        )
        scores = []
        for view_go, view_bp, tr, targets, conf, K in (
            (go, bp, real_transl, kp2d_real, kp2d_conf_real, K_real),
            (mirror_go, mirror_bp, mirror_transl_raw, kp2d_mirror, kp2d_conf_mirror, K_mirror),
        ):
            joints = smpl(view_go, view_bp, betas, tr)[:, :17]
            xy, valid = project_3d_to_2d(joints, K)
            error = (xy - targets).norm(dim=-1)
            error = torch.where(valid & torch.isfinite(error), error, torch.full_like(error, 1e6))
            weights = conf.clamp(0.0, 1.0)
            scores.append((error * weights).sum(-1) / weights.sum(-1).clamp_min(1e-8))
        return torch.stack(scores, dim=-1)

    baseline_errors = projection_errors(current_go, current_bp)
    best_errors = baseline_errors.clone()
    best_go, best_bp = current_go.clone(), current_bp.clone()

    @torch.no_grad()
    def keep_improvements(go, bp):
        nonlocal best_go, best_bp, best_errors
        errors = projection_errors(go, bp)
        # A fixed detector metric prevents lowering belief to hide a regression.
        accept = torch.isfinite(errors).all(-1) & (errors <= best_errors + 1e-4).all(-1)
        accept &= errors.sum(-1) < best_errors.sum(-1)
        best_go = torch.where(accept[:, None], go.detach(), best_go)
        best_bp = torch.where(accept[:, None], bp.detach(), best_bp)
        best_errors = torch.where(accept[:, None], errors, best_errors)
        return errors

    for outer in range(outer_iterations):
        belief_real17 = belief_real_info["belief"].clone()
        belief_mirror17_raw = belief_mirror_info["belief"].clone()
        belief_mirror17 = flip_coco_belief(belief_mirror17_raw)
        # Làm mượt belief theo thời gian trước khi dùng — occlusion/detection
        # belief tính độc lập mỗi frame nên dễ dao động, gây trọng số fusion/
        # reprojection nhảy giữa các frame liên tiếp -> rung giật khung xương.
        if N >= 3:
            belief_real17 = smooth_belief_temporal(belief_real17)
            belief_mirror17 = smooth_belief_temporal(belief_mirror17)

        # ── 3. Belief-weighted precision fusion (SO(3)) làm điểm khởi tạo + anchor ──
        belief21_real = map_coco_belief_to_smpl_body(belief_real17)
        belief21_mirror = map_coco_belief_to_smpl_body(belief_mirror17)
        root_bel_real = root_belief(belief_real17)

        # Canonical body rotations are comparable after swapping L/R; camera
        # roots are not. Root disagreement must not suppress all mirror joints.
        with torch.no_grad():
            bp_quality = compute_angular_discrepancy_penalty(
                real_body_pose.view(N, 21, 3), mirror_body_pose_unmirrored.view(N, 21, 3),
                threshold_deg=90.0, transition_width=15.0,
            )
            det21_real = map_coco_belief_to_smpl_body(kp2d_conf_real)
            det21_mirror = map_coco_belief_to_smpl_body(flip_coco_belief(kp2d_conf_mirror))
            trust_real21, trust_mirror21 = compute_cross_view_trust(
                det21_real, det21_mirror, bp_quality,
            )
        belief21_real = belief21_real * trust_real21
        belief21_mirror = belief21_mirror * trust_mirror21
        belief_real17 = belief_real17 * map_smpl_body_scale_to_coco(trust_real21)
        # Expose the mirror anchor evidence in the original detector order.
        belief_mirror17_reproj = flip_coco_belief(
            belief_mirror17 * map_smpl_body_scale_to_coco(trust_mirror21)
        )
        joint_confidence = (torch.maximum(belief21_real, belief21_mirror) * bp_quality).detach()
        pose_confidence_22 = torch.cat([root_bel_real[:, None], joint_confidence], dim=-1).detach()
        pose_confidence_22 = propagate_confidence_to_kinematic_neighbors(pose_confidence_22)

        fused_bp_ref = fuse_pose_precision_weighted(
            real_body_pose.view(N, 21, 3), mirror_body_pose_unmirrored.view(N, 21, 3),
            belief21_real, belief21_mirror,
        ).reshape(N, 63)

        # Fuse once: a second blend overweights the mirror (50/50 becomes 25/75).
        fused_go_ref = real_global_orient
        init_go, init_bp = current_go, current_bp
        opt_go = init_go.clone().detach().requires_grad_(True)
        opt_bp = init_bp.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([opt_go, opt_bp], lr=lr)

        # BUG 1 FIX: warm-up w_pose_prior tuyến tính qua outer loop — bắt đầu yếu
        # (w_pose_prior_min) để không kẹt local minimum sai lúc pose còn xa tư thế
        # tự nhiên, siết dần lên w_pose_prior (max) ở outer loop cuối.
        current_w_pose_prior = _get_pose_prior_weight(outer, outer_iterations, w["w_pose_prior_min"], w["w_pose_prior"])

        for _step in range(inner_steps):
            optimizer.zero_grad()

            joints18_real = _forward_view_joints_and_mesh(smpl, opt_go, opt_bp, betas, real_transl, return_mesh=False)
            proj_real, valid_real = project_3d_to_2d(joints18_real[:, :17, :], K_real)
            # Keep all trusted detector constraints. A current mesh's visibility
            # cannot veto them without a circular loss/belief feedback loop.
            loss_reproj_real = calculate_reprojection_loss(
                proj_real, kp2d_real, kp2d_conf_real.clamp(0, 1), valid_depth_mask=valid_real, sigma=reproj_sigma
            )

            opt_mirrored_go, opt_mirrored_bp = _to_mirror_frame(
                opt_go, opt_bp, real_go_reference=real_global_orient, mirror_go_reference=mirror_go_raw,
            )
            joints18_mirror = _forward_view_joints_and_mesh(
                smpl, opt_mirrored_go, opt_mirrored_bp, betas, mirror_transl_raw, return_mesh=False
            )
            proj_mirror, valid_mirror = project_3d_to_2d(joints18_mirror[:, :17, :], K_mirror)
            loss_reproj_mirror = calculate_reprojection_loss(
                proj_mirror, kp2d_mirror, kp2d_conf_mirror.clamp(0, 1), valid_depth_mask=valid_mirror, sigma=reproj_sigma
            )

            loss_prior = compute_pose_prior_loss(opt_bp)
            loss_joint_rom = compute_joint_angle_limit_loss(opt_bp)
            loss_anatomical = compute_anatomical_limits_loss(opt_bp)
            loss_elbow = compute_elbow_limits_loss(opt_bp)
            loss_head = compute_head_collision_loss(joints18_real)
            loss_symmetry = compute_bone_symmetry_loss(joints18_real[:, :17, :])
            loss_bone_stability = (
                compute_bone_length_stability_loss(joints18_real[:, :17, :]) if N >= 2 else joints18_real.new_zeros(())
            )
            loss_temporal = (
                compute_temporal_smoothness_loss(joints18_real[:, :17, :]) if N >= 3 else joints18_real.new_zeros(())
            )
            loss_pose_temporal = (
                compute_pose_temporal_smoothness_loss(
                    opt_go, opt_bp, confidence=pose_confidence_22, min_weight=w["temporal_confidence_min_weight"]
                )
                if N >= 2
                else joints18_real.new_zeros(())
            )
            loss_pose_accel = (
                compute_pose_acceleration_smoothness_loss(
                    opt_go, opt_bp, confidence=pose_confidence_22, min_weight=w["temporal_confidence_min_weight"]
                )
                if N >= 3
                else joints18_real.new_zeros(())
            )
            loss_penetration = compute_penetration_loss(
                joints18_real, radius_m=w["penetration_radius"], sharpness=w["penetration_sharpness"]
            )
            # ── Grotesque pose penalties ──────────────────────────────────
            loss_spine_twist = compute_spine_twist_loss(opt_bp)
            loss_hip_split = compute_hip_split_loss(opt_bp)
            loss_leg_crossing = compute_leg_crossing_loss(joints18_real[:, :17, :])
            loss_hip_adduction = compute_hip_adduction_loss(opt_bp)
            loss_shoulder_hyper = compute_shoulder_hyperextension_loss(opt_bp)
            loss_wrist_bend = compute_wrist_bend_loss(opt_bp)

            # Neo cả global_orient (không chỉ body_pose) — thiếu ràng buộc này khiến
            # root rotation không có "điểm tựa" nào ngoài velocity-smoothing
            # (w_pose_temporal chỉ phạt CHÊNH LỆCH giữa 2 frame, một độ trôi góc
            # không đổi mỗi frame vẫn "mượt" theo velocity nhưng khiến khung xương
            # xoay liên tục qua cả sequence). Anchor về fused_go_ref chặn đứng trôi này.
            anchor_pose = torch.cat([opt_go.view(N, 1, 3), opt_bp.view(N, 21, 3)], dim=1)
            anchor_ref = torch.cat([fused_go_ref.view(N, 1, 3), fused_bp_ref.view(N, 21, 3)], dim=1)
            loss_anchor = rotation_geodesic_loss(anchor_pose, anchor_ref)

            total = (
                w["w_reproj_real"] * loss_reproj_real
                + w["w_reproj_mirror"] * loss_reproj_mirror
                + current_w_pose_prior * loss_prior
                + w["w_joint_angle_limit"] * loss_joint_rom
                + w["w_anatomical"] * loss_anatomical
                + w["w_elbow"] * loss_elbow
                + w["w_head_collision"] * loss_head
                + w["w_symmetry"] * loss_symmetry
                + w["w_bone_stability"] * loss_bone_stability
                + w["w_temporal"] * temporal_scale ** 4 * loss_temporal
                + w["w_pose_temporal"] * temporal_scale ** 2 * loss_pose_temporal
                + w["w_pose_accel"] * temporal_scale ** 4 * loss_pose_accel
                + w["w_penetration"] * loss_penetration
                + w["w_anchor"] * loss_anchor
                + w["w_spine_twist"] * loss_spine_twist
                + w["w_hip_split"] * loss_hip_split
                + w.get("w_leg_crossing", 3.0) * loss_leg_crossing
                + w.get("w_hip_adduction", 2.0) * loss_hip_adduction
                + w["w_shoulder_hyper"] * loss_shoulder_hyper
                + w["w_wrist_bend"] * loss_wrist_bend
            )
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite loss at outer={outer}: {total.item()}")

            total.backward()
            # BUG 6 FIX: clip riêng biệt cho opt_go (3 chiều) và opt_bp (63 chiều) —
            # trước đây clip chung khiến gradient của bên nhiều chiều hơn lấn át bên kia.
            torch.nn.utils.clip_grad_norm_([opt_go], max_norm=w["grad_clip_go"])
            torch.nn.utils.clip_grad_norm_([opt_bp], max_norm=w["grad_clip_bp"])
            optimizer.step()
            if preserve_real_projection and ((_step + 1) % 25 == 0 or _step + 1 == inner_steps):
                keep_improvements(opt_go, opt_bp)

        current_go = opt_go.detach()
        current_bp = opt_bp.detach()

        if preserve_real_projection:
            current_go, current_bp = best_go.clone(), best_bp.clone()
        diagnostics = {
            "belief_real": belief_real17.detach().cpu(),
            "belief_mirror": belief_mirror17.detach().cpu(),  # real anatomical labels
            "belief_mirror_raw": belief_mirror17_raw.detach().cpu(),
            "fusion_belief_mirror_raw": belief_mirror17_reproj.detach().cpu(),
            "reprojection_conf_real": kp2d_conf_real.clamp(0, 1).detach().cpu(),
            "reprojection_conf_mirror": kp2d_conf_mirror.clamp(0, 1).detach().cpu(),
            "bp_quality": bp_quality.detach().cpu(),
        }
        for key in ("b_occ", "b_det", "b_reproj"):
            diagnostics[key + "_real"] = belief_real_info[key].detach().cpu()
            diagnostics[key + "_mirror"] = belief_mirror_info[key].detach().cpu()
        change_deg = _mean_rotation_change_deg(prev_go, prev_bp, current_go, current_bp) if prev_go is not None else float("inf")
        if verbose:
            print(f"[pose_refit] outer {outer + 1}/{outer_iterations}: "
                  f"belief real/mirror={belief_real17.mean():.3f}/{belief_mirror17.mean():.3f}, "
                  f"pose change={change_deg:.3f} deg")
        if change_deg < convergence_deg:
            break
        prev_go, prev_bp = current_go.clone(), current_bp.clone()

    if post_smooth:
        current_go = smooth_global_orientation_sequence(current_go)
        current_bp = smooth_body_pose_sequence(current_bp)
    if preserve_real_projection:
        keep_improvements(current_go, current_bp)
        current_go, current_bp = best_go, best_bp
    final_errors = projection_errors(current_go, current_bp)
    diagnostics["reprojection_before_px"] = baseline_errors.cpu()
    diagnostics["reprojection_after_px"] = final_errors.cpu()
    diagnostics["refit_accepted"] = (
        ((current_bp - real_body_pose).abs().amax(-1) > 1e-6)
        | ((current_go - real_global_orient).abs().amax(-1) > 1e-6)
    ).cpu()
    if verbose:
        print(f"[pose_refit] reprojection real/mirror px: "
              f"{baseline_errors.mean(0).tolist()} -> {final_errors.mean(0).tolist()}; "
              f"accepted {int(diagnostics['refit_accepted'].sum())}/{N} frames")

    return {
        "global_orient": current_go,
        "body_pose": current_bp,
        "diagnostics": diagnostics,
    }


def gaussian_filter_quaternion_sequence(
    aa_sequence: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 1.2,
) -> torch.Tensor:
    """
    Lọc Gaussian 1D trên chuỗi quaternion theo thời gian (time dimension N)
    để loại bỏ hoàn toàn các rung giật tần số cao (high-frequency jitter).

    Args:
        aa_sequence: (N, J, 3) hoặc (N, 3) axis-angle.
        kernel_size: kích thước cửa sổ lọc (lẻ, ví dụ 5).
        sigma: độ rộng Gaussian.

    Returns:
        (N, J, 3) hoặc (N, 3) axis-angle đã làm mượt.
    """
    is_2d = aa_sequence.dim() == 2
    if is_2d:
        aa_sequence = aa_sequence.unsqueeze(1)  # (N, 1, 3)

    N, J, _ = aa_sequence.shape
    if N < 3:
        return aa_sequence.squeeze(1) if is_2d else aa_sequence

    # 1. Chuyển sang Quaternion (N, J, 4)
    q = axis_angle_to_quaternion(aa_sequence)

    # 2. Đảm bảo liên tục dấu của quaternion (q_t và q_{t-1} cùng bán cầu)
    q_cont = q.clone()
    for t in range(1, N):
        dot = (q_cont[t] * q_cont[t - 1]).sum(dim=-1, keepdim=True)
        q_cont[t] = torch.where(dot < 0, -q_cont[t], q_cont[t])

    # 3. Tạo 1D Gaussian Kernel
    half = kernel_size // 2
    x = torch.arange(-half, half + 1, device=aa_sequence.device, dtype=aa_sequence.dtype)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()

    # 4. Padding và Tích chập 1D
    # q_cont: (N, J, 4) -> (J * 4, 1, N)
    q_flat = q_cont.permute(1, 2, 0).reshape(J * 4, 1, N)
    q_padded = torch.nn.functional.pad(q_flat, (half, half), mode="replicate")
    q_smoothed_flat = torch.nn.functional.conv1d(q_padded, kernel.view(1, 1, -1))
    q_smoothed = q_smoothed_flat.view(J, 4, N).permute(2, 0, 1)  # (N, J, 4)

    # 5. Chuẩn hóa về Unit Quaternion và đổi lại Axis-Angle
    from utils.geometry import quaternion_to_axis_angle
    q_smoothed = q_smoothed / q_smoothed.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    aa_smoothed = quaternion_to_axis_angle(q_smoothed)

    return aa_smoothed.squeeze(1) if is_2d else aa_smoothed


def smooth_global_orientation_sequence(
    go_candidate: torch.Tensor,
    steps: int = 100,
    lr: float = 0.05,
    w_temporal: float = 1.5,
    w_accel: float = 1.0,
) -> torch.Tensor:
    """
    Làm mượt global_orient hệ "global" (world-frame, dùng cho overlay full-body).
    Kết hợp tối ưu Geodesic Velocity + Acceleration Loss và Lọc Gaussian 1D.
    """
    N = go_candidate.shape[0]
    if N < 2:
        return go_candidate.clone()

    ref = go_candidate.detach()
    opt_go = ref.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([opt_go], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        loss_anchor = rotation_geodesic_loss(opt_go.view(N, 1, 3), ref.view(N, 1, 3))
        loss_temporal = rotation_geodesic_loss(
            opt_go[1:].view(N - 1, 1, 3), opt_go[:-1].view(N - 1, 1, 3)
        )
        loss_accel = (
            compute_pose_acceleration_smoothness_loss(
                opt_go.view(N, 3), opt_go.new_zeros(N, 63)
            )
            if N >= 3
            else opt_go.new_zeros(())
        )
        total = loss_anchor + w_temporal * loss_temporal + w_accel * loss_accel
        total.backward()
        optimizer.step()

    out_go = opt_go.detach()
    return gaussian_filter_quaternion_sequence(out_go, kernel_size=5, sigma=1.2)


def smooth_body_pose_sequence(
    bp_candidate: torch.Tensor,
    steps: int = 100,
    lr: float = 0.03,
    w_temporal: float = 1.5,
    w_accel: float = 1.0,
) -> torch.Tensor:
    """
    Làm mượt body_pose (21 khớp) toàn sequence — chống giật khớp cục bộ.
    Kết hợp tối ưu Geodesic Velocity + Acceleration Loss và Lọc Gaussian 1D.
    """
    N = bp_candidate.shape[0]
    if N < 2:
        return bp_candidate.clone()

    ref = bp_candidate.detach()
    opt_bp = ref.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([opt_bp], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        loss_anchor = rotation_geodesic_loss(
            opt_bp.view(N, 21, 3), ref.view(N, 21, 3)
        )
        loss_temporal = rotation_geodesic_loss(
            opt_bp[1:].view(N - 1, 21, 3), opt_bp[:-1].view(N - 1, 21, 3)
        )
        loss_accel = (
            compute_pose_acceleration_smoothness_loss(
                opt_bp.new_zeros(N, 3), opt_bp.view(N, 63)
            )
            if N >= 3
            else opt_bp.new_zeros(())
        )
        total = loss_anchor + w_temporal * loss_temporal + w_accel * loss_accel
        total.backward()
        optimizer.step()

    out_bp = opt_bp.detach().view(N, 21, 3)
    out_bp = gaussian_filter_quaternion_sequence(out_bp, kernel_size=5, sigma=1.2)
    return out_bp.reshape(N, 63)
