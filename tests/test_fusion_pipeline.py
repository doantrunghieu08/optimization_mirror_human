"""Semantic regressions for camera frames, mirrored evidence and refit acceptance."""

import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from smplx.lbs import batch_rodrigues

from utils.belief_fusion import flip_coco_belief, compute_full_view_belief
from utils.geometry import transfer_orientation
from utils.mirror_geometry import reflect_global_orientation, unmirror_pose_general
from utils.pose_refit import DEFAULT_LOSS_WEIGHTS, _to_mirror_frame, run_pose_refit
from utils.smpl_utils import SMPLForwardPass, unmirror_pose


class CameraAndEvidenceTests(unittest.TestCase):
    def test_general_reflection_matches_physical_matrix_product(self):
        go = torch.tensor([[0.2, -0.4, 0.8], [0.0, 0.0, 0.0]])
        normal = torch.tensor([0.6, 0.0, 0.8])
        reflection = torch.eye(3) - 2 * normal[:, None] * normal[None, :]
        body_reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
        expected = reflection @ batch_rodrigues(go) @ body_reflection
        actual = batch_rodrigues(reflect_global_orientation(go, normal))
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)

    def test_camera_normal_does_not_change_local_rotations(self):
        pose = torch.randn(2, 22, 3) * 0.3
        actual = unmirror_pose_general(pose, torch.tensor([0.0, 0.0, 1.0]))
        torch.testing.assert_close(actual[:, 1:], unmirror_pose(pose)[:, 1:])

    def test_mirror_projection_preserves_original_camera_and_root_gradient(self):
        real = torch.tensor([[0.1, 0.2, -2.9]])
        mirror = torch.tensor([[2.8, 0.1, 0.8]])
        candidate = real.clone().requires_grad_(True)
        go, _ = _to_mirror_frame(candidate, torch.zeros(1, 63),
                                 real_go_reference=real, mirror_go_reference=mirror)
        torch.testing.assert_close(batch_rodrigues(go), batch_rodrigues(mirror), atol=1e-6, rtol=0)
        go.square().sum().backward()
        self.assertTrue(torch.isfinite(candidate.grad).all())
        self.assertGreater(float(candidate.grad.norm()), 0.1)

    def test_world_orientation_preserves_camera_to_world_rotation(self):
        real = torch.tensor([[0.2, -0.4, 0.8]])
        world = torch.tensor([[-0.3, 0.7, 0.5]])
        candidate = torch.tensor([[0.5, -0.1, 0.4]])
        converted = transfer_orientation(real, world, candidate)
        expected = batch_rodrigues(world) @ batch_rodrigues(real).transpose(-1, -2) @ batch_rodrigues(candidate)
        torch.testing.assert_close(batch_rodrigues(converted), expected, atol=1e-6, rtol=0)

    def test_mirror_evidence_follows_the_opposite_anatomical_side(self):
        raw = torch.zeros(1, 17)
        raw[0, 9] = 0.9  # mirror's left wrist is the person's right wrist
        aligned = flip_coco_belief(raw)
        self.assertAlmostEqual(float(aligned[0, 10]), 0.9, places=6)
        self.assertEqual(float(aligned[0, 9]), 0)
        torch.testing.assert_close(flip_coco_belief(aligned), raw)

    def test_visible_mesh_cannot_validate_missing_detection(self):
        with patch('utils.belief_fusion.compute_occlusion_belief', return_value=torch.ones(1, 17)):
            result = compute_full_view_belief(
                torch.zeros(1, 17, 3), torch.zeros(1, 3, 3), [[0, 1, 2]],
                torch.zeros(1, 17, 2), torch.zeros(1, 17),
                torch.zeros(1, 17, 2), torch.ones(1, 17, dtype=torch.bool),
            )
        self.assertEqual(float(result['belief'].sum()), 0)


class RefitAcceptanceTests(unittest.TestCase):
    @staticmethod
    def smpl(global_orient, body_pose, betas, transl, return_mesh=False):
        # A differentiable articulated observation whose spine-X rotation survives
        # the L/R reflection. This tests the optimizer independently of model assets.
        base = body_pose.new_zeros(len(body_pose), 18, 3)
        base[..., 0] = torch.linspace(-0.3, 0.3, 18, device=body_pose.device)
        base[..., 1] = torch.linspace(-0.5, 0.5, 18, device=body_pose.device)
        base[..., 2] = 3
        delta = torch.stack([body_pose[:, 6], body_pose[:, 6] * 0, body_pose[:, 6] * 0], -1)
        return base + delta[:, None]

    def run_refit(self, real_target, mirror_target, mirror_weight=1.0):
        def evidence(*args, **kwargs):
            return {key: torch.ones(1, 17) for key in ('belief', 'b_occ', 'b_det', 'b_reproj')}

        def target(amount):
            bp = torch.zeros(1, 63)
            bp[:, 6] = amount
            return self.smpl(torch.zeros(1, 3), bp, None, None)[:, :17, :2] * (100 / 3)

        w = {key: 0.0 for key in DEFAULT_LOSS_WEIGHTS if key.startswith('w_')}
        w.update(w_reproj_real=1.0, w_reproj_mirror=mirror_weight)
        with patch('utils.pose_refit._compute_view_belief', side_effect=evidence):
            return run_pose_refit(
                real_global_orient=torch.zeros(1, 3), real_body_pose=torch.zeros(1, 63),
                mirror_global_orient_unmirrored=torch.zeros(1, 3), mirror_body_pose_unmirrored=torch.zeros(1, 63),
                betas=torch.zeros(1, 10), real_transl=torch.zeros(1, 3), mirror_transl_raw=torch.zeros(1, 3),
                mirror_go_raw=torch.zeros(1, 3), mirror_bp_raw=torch.zeros(1, 63),
                kp2d_real=target(real_target), kp2d_conf_real=torch.ones(1, 17),
                kp2d_mirror=target(mirror_target), kp2d_conf_mirror=torch.ones(1, 17),
                K_real=torch.diag(torch.tensor([100., 100., 1.]))[None],
                K_mirror=torch.diag(torch.tensor([100., 100., 1.]))[None],
                smpl=self.smpl, outer_iterations=1, inner_steps=25, loss_weights=w, verbose=False,
            )

    def test_accepts_a_refit_that_improves_both_views(self):
        result = self.run_refit(0.2, 0.2)
        diag = result['diagnostics']
        self.assertTrue(bool(diag['refit_accepted'][0]))
        self.assertTrue((diag['reprojection_after_px'] < diag['reprojection_before_px']).all())

    def test_retains_real_pose_when_mirror_would_degrade_it(self):
        result = self.run_refit(0.0, 0.3, mirror_weight=10)
        torch.testing.assert_close(result['body_pose'], torch.zeros(1, 63), atol=0, rtol=0)
        self.assertFalse(bool(result['diagnostics']['refit_accepted'][0]))

    def test_mirror_evidence_uses_mirror_shape(self):
        seen_betas = []

        def evidence(*args, **kwargs):
            seen_betas.append(args[3].clone())
            return {key: torch.ones(1, 17) for key in ('belief', 'b_occ', 'b_det', 'b_reproj')}

        w = {key: 0.0 for key in DEFAULT_LOSS_WEIGHTS if key.startswith('w_')}
        with patch('utils.pose_refit._compute_view_belief', side_effect=evidence):
            run_pose_refit(
                real_global_orient=torch.zeros(1, 3), real_body_pose=torch.zeros(1, 63),
                mirror_global_orient_unmirrored=torch.zeros(1, 3), mirror_body_pose_unmirrored=torch.zeros(1, 63),
                betas=torch.zeros(1, 10), mirror_betas=torch.ones(1, 10),
                real_transl=torch.zeros(1, 3), mirror_transl_raw=torch.zeros(1, 3),
                mirror_go_raw=torch.zeros(1, 3), mirror_bp_raw=torch.zeros(1, 63),
                kp2d_real=torch.zeros(1, 17, 2), kp2d_conf_real=torch.ones(1, 17),
                kp2d_mirror=torch.zeros(1, 17, 2), kp2d_conf_mirror=torch.ones(1, 17),
                K_real=torch.eye(3)[None], K_mirror=torch.eye(3)[None], smpl=self.smpl,
                outer_iterations=1, inner_steps=1, loss_weights=w, verbose=False,
            )

        torch.testing.assert_close(seen_betas[0], torch.zeros(1, 10))
        torch.testing.assert_close(seen_betas[1], torch.ones(1, 10))


@unittest.skipUnless(Path('models/SMPLX_NEUTRAL.npz').exists()
                     and Path('models/smplx_coco17_J_regressor.pt').exists(), 'SMPL-X assets unavailable')
class SMPLXBatchingTests(unittest.TestCase):
    def test_gradients_survive_sequence_chunk_boundary(self):
        model = SMPLForwardPass('models/SMPLX_NEUTRAL.npz', torch.device('cpu'))
        bp = torch.zeros(33, 63, requires_grad=True)
        go = torch.zeros(33, 3)
        betas = torch.zeros(33, 10)
        transl = torch.zeros(33, 3)
        joints = model(go, bp, betas, transl)
        joints.square().sum().backward()
        bp_pair = bp.detach()[31:33].clone().requires_grad_(True)
        pair = model(go[31:33], bp_pair, betas[31:33], transl[31:33])
        pair.square().sum().backward()
        torch.testing.assert_close(joints.detach()[31:33], pair.detach(), atol=1e-6, rtol=0)
        torch.testing.assert_close(bp.grad[31:33], bp_pair.grad, atol=1e-5, rtol=1e-4)
        self.assertGreater(float(bp.grad[32].norm()), 0)


if __name__ == '__main__':
    unittest.main()
