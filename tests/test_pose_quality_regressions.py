"""Regression checks for the pose-fusion repair plan."""

import unittest
from unittest.mock import patch

import torch

from utils.belief_fusion import compute_full_view_belief
from utils.camera_utils import calculate_reprojection_loss
from utils.pose_refit import (
    DEFAULT_LOSS_WEIGHTS,
    _to_mirror_frame,
    estimate_fixed_inter_view_rotation,
    run_pose_refit,
)


def _mock_smpl(global_orient, body_pose, betas, transl, return_mesh=False):
    base = body_pose.new_zeros(len(body_pose), 18, 3)
    base[..., 0] = torch.linspace(-0.3, 0.3, 18, device=body_pose.device)
    base[..., 1] = torch.linspace(-0.5, 0.5, 18, device=body_pose.device)
    base[..., 2] = 3
    delta = torch.stack([body_pose[:, 6], body_pose[:, 6] * 0, body_pose[:, 6] * 0], -1)
    joints = base + delta[:, None]
    if return_mesh:
        return joints, body_pose.new_zeros(len(body_pose), 3, 3), [[0, 1, 2]]
    return joints


class PoseQualityRegressionTests(unittest.TestCase):
    def test_surface_visibility_enters_belief_without_doubling_evidence(self):
        confidence = torch.full((1, 17), 0.8)
        projected = torch.zeros(1, 17, 2)
        detected = torch.zeros_like(projected)
        detected[..., 0] = 50
        valid = torch.ones(1, 17, dtype=torch.bool)
        with patch(
            "utils.belief_fusion.compute_occlusion_belief",
            return_value=torch.full((1, 17), 0.25),
        ):
            result = compute_full_view_belief(
                torch.zeros(1, 17, 3),
                torch.zeros(1, 3, 3),
                [[0, 1, 2]],
                detected,
                confidence,
                projected,
                valid,
                reproj_sigma=50,
            )
        expected = 0.25 * confidence * 0.5
        torch.testing.assert_close(result["belief"], expected)
        self.assertAlmostEqual(float(result["b_occ"].mean()), 0.25)

    def test_fixed_inter_view_rotation_ignores_one_noisy_reference_frame(self):
        real = torch.zeros(5, 3)
        clean_mirror = torch.tensor([0.0, 0.4, 0.0]).expand(5, -1).clone()
        noisy_mirror = clean_mirror.clone()
        noisy_mirror[2] = torch.tensor([2.5, -0.3, 0.8])
        fixed, residual = estimate_fixed_inter_view_rotation(real, noisy_mirror)
        projected, _ = _to_mirror_frame(
            real, torch.zeros(5, 63), inter_view_rotation=fixed
        )
        torch.testing.assert_close(projected[[0, 1, 3, 4]], clean_mirror[[0, 1, 3, 4]], atol=1e-5, rtol=0)
        self.assertGreater(float(residual[2]), 30)

    def test_mirror_reprojection_keeps_gradient_to_real_candidate(self):
        candidate = torch.zeros(2, 3, requires_grad=True)
        fixed, _ = estimate_fixed_inter_view_rotation(
            torch.zeros(2, 3), torch.tensor([[0.0, 0.4, 0.0]]).expand(2, -1)
        )
        mirror, _ = _to_mirror_frame(
            candidate, torch.zeros(2, 63), inter_view_rotation=fixed
        )
        mirror.square().sum().backward()
        self.assertTrue(torch.isfinite(candidate.grad).all())
        self.assertGreater(float(candidate.grad.norm()), 0)

    def test_zero_real_reliability_produces_only_mirror_gradient(self):
        def gradient(include_zero_real):
            value = torch.tensor(1.0, requires_grad=True)
            projected = torch.stack([value, value * 0]).view(1, 1, 2)
            real = calculate_reprojection_loss(
                projected, torch.tensor([[[10.0, 0.0]]]), torch.zeros(1, 1)
            )
            mirror = calculate_reprojection_loss(
                projected, torch.zeros(1, 1, 2), torch.ones(1, 1)
            )
            (mirror + real if include_zero_real else mirror).backward()
            return value.grad

        torch.testing.assert_close(gradient(True), gradient(False))

    def test_sequence_acceptance_never_builds_alternating_frame_mosaic(self):
        frames = 4
        target_pose = torch.zeros(frames, 63)
        target_pose[:, 6] = torch.tensor([0.3, -0.3, 0.3, -0.3])
        targets = _mock_smpl(
            torch.zeros(frames, 3), target_pose, None, None
        )[:, :17, :2] * (100 / 3)

        def evidence(*args, **kwargs):
            n = len(args[1])
            return {key: torch.ones(n, 17) for key in ("belief", "b_occ", "b_det", "b_reproj")}

        weights = {key: 0.0 for key in DEFAULT_LOSS_WEIGHTS if key.startswith("w_")}
        weights.update(w_reproj_real=1.0, w_reproj_mirror=1.0)
        with patch("utils.pose_refit._compute_view_belief", side_effect=evidence):
            result = run_pose_refit(
                real_global_orient=torch.zeros(frames, 3),
                real_body_pose=torch.zeros(frames, 63),
                mirror_global_orient_unmirrored=torch.zeros(frames, 3),
                mirror_body_pose_unmirrored=torch.zeros(frames, 63),
                betas=torch.zeros(frames, 10),
                real_transl=torch.zeros(frames, 3),
                mirror_transl_raw=torch.zeros(frames, 3),
                mirror_go_raw=torch.zeros(frames, 3),
                mirror_bp_raw=torch.zeros(frames, 63),
                kp2d_real=targets,
                kp2d_conf_real=torch.ones(frames, 17),
                K_real=torch.diag(torch.tensor([100.0, 100.0, 1.0]))[None].expand(frames, -1, -1),
                kp2d_mirror=targets,
                kp2d_conf_mirror=torch.ones(frames, 17),
                K_mirror=torch.diag(torch.tensor([100.0, 100.0, 1.0]))[None].expand(frames, -1, -1),
                smpl=_mock_smpl,
                outer_iterations=1,
                inner_steps=25,
                loss_weights=weights,
                verbose=False,
            )

        torch.testing.assert_close(result["body_pose"], torch.zeros(frames, 63))
        self.assertFalse(bool(result["diagnostics"]["sequence_accepted"]))
        before = result["diagnostics"]["rotation_velocity_before_deg"]
        after = result["diagnostics"]["rotation_velocity_after_deg"]
        self.assertTrue((after <= before * 1.05 + 1e-4).all())


if __name__ == "__main__":
    unittest.main()
