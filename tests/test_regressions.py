import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dataloaders.dataset import flip_camera_intrinsics
from evaluate import evaluate_3d_error
from render_video import render_video
from utils.belief_fusion import (
    combine_beliefs_dempster_shafer,
    compute_detection_belief,
    compute_reprojection_belief,
    fuse_pose_precision_weighted,
    map_coco_belief_to_smpl_body,
    root_belief,
)
from utils.camera_utils import calculate_reprojection_loss, project_3d_to_2d
from utils.mesh_raycast import (
    compute_occlusion_belief_batch,
    compute_ray_occlusion,
    compute_ray_visibility,
)
from utils.mirror_geometry import (
    compute_skeleton_scale_ratio,
    estimate_mirror_normal_robust,
    normalize_mirror_joints_scale,
    reflect_axis_angle,
    unmirror_pose_general,
)
from utils.skeleton_alignment import align_skeleton_to_keypoints
from utils.smpl_utils import mirror_axis_angle, unmirror_pose
from losses.prior_losses import (
    compute_elbow_limits_loss,
    compute_hip_adduction_loss,
    compute_leg_crossing_loss,
    compute_temporal_smoothness_loss,
)


class GeometryRegressionTests(unittest.TestCase):
    def test_temporal_loss_ignores_per_frame_translation(self):
        joints = torch.randn(5, 17, 3)
        translation = torch.randn(5, 1, 3) * 10
        torch.testing.assert_close(
            compute_temporal_smoothness_loss(joints),
            compute_temporal_smoothness_loss(joints + translation),
        )

    def test_elbow_limit_allows_valid_hmr4d_y_axis_flexion(self):
        pose = torch.zeros(1, 63)
        pose[0, 17 * 3 + 1] = 1.5
        pose[0, 18 * 3 + 1] = 1.5
        self.assertEqual(float(compute_elbow_limits_loss(pose)), 0.0)

        pose[0, 17 * 3 + 1] = 3.0
        self.assertGreater(float(compute_elbow_limits_loss(pose)), 0.0)

    def test_leg_crossing_loss_penalizes_crossed_legs(self):
        joints = torch.zeros(1, 17, 3)
        joints[0, 11] = torch.tensor([-0.15, 0.0, 1.0])  # L_Hip
        joints[0, 12] = torch.tensor([0.15, 0.0, 1.0])   # R_Hip
        joints[0, 13] = torch.tensor([-0.15, -0.4, 1.0]) # L_Knee
        joints[0, 14] = torch.tensor([0.15, -0.4, 1.0])  # R_Knee
        joints[0, 15] = torch.tensor([-0.15, -0.8, 1.0]) # L_Ankle
        joints[0, 16] = torch.tensor([0.15, -0.8, 1.0])  # R_Ankle

        normal_loss = compute_leg_crossing_loss(joints)
        self.assertEqual(float(normal_loss), 0.0)

        joints[0, 13] = torch.tensor([0.25, -0.4, 1.0])  # L_Knee crossed
        joints[0, 15] = torch.tensor([0.25, -0.8, 1.0])  # L_Ankle crossed
        crossed_loss = compute_leg_crossing_loss(joints)
        self.assertGreater(float(crossed_loss), 0.0)

    def test_hip_adduction_loss_penalizes_extreme_adduction(self):
        pose = torch.zeros(1, 63)
        self.assertEqual(float(compute_hip_adduction_loss(pose)), 0.0)

        pose[0, 0 * 3 + 2] = -1.2  # ~68° adduction on L_Hip
        self.assertGreater(float(compute_hip_adduction_loss(pose)), 0.0)

    def test_mirror_pose_is_involution(self):
        pose = torch.arange(2 * 22 * 3, dtype=torch.float32).reshape(2, 22, 3)
        restored = unmirror_pose(unmirror_pose(pose))
        self.assertTrue(torch.equal(restored, pose))

    def test_projection_masks_non_positive_depth(self):
        joints = torch.tensor([[[1.0, 2.0, 1.0], [1.0, 2.0, -1.0]]])
        K = torch.eye(3).unsqueeze(0)
        projected, valid = project_3d_to_2d(joints, K)
        self.assertTrue(torch.isfinite(projected).all())
        self.assertTrue(torch.equal(valid, torch.tensor([[True, False]])))
        loss = calculate_reprojection_loss(
            projected, torch.zeros_like(projected), torch.ones(1, 2), valid
        )
        self.assertTrue(torch.isfinite(loss))

    def test_intrinsics_horizontal_flip(self):
        K = torch.tensor([[[100.0, 0.0, 20.0], [0.0, 100.0, 30.0], [0.0, 0.0, 1.0]]])
        flipped = flip_camera_intrinsics(K, 100)
        self.assertEqual(float(flipped[0, 0, 2]), 79.0)
        self.assertEqual(float(flipped[0, 1, 1]), 100.0)

    def test_skeleton_alignment_estimates_scale_and_ankle_contact(self):
        projected = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
            [0.5, 0.5], [0.0, 2.0], [1.0, 2.0], [0.0, 3.0],
            [1.0, 3.0], [0.0, 4.0], [1.0, 4.0], [0.0, 2.5],
            [1.0, 2.5], [0.0, 5.0], [1.0, 5.0], [0.0, 6.0],
            [1.0, 6.0],
        ], dtype=np.float32)
        target = projected * 1.2 + np.array([100.0, 200.0], dtype=np.float32)
        aligned = align_skeleton_to_keypoints(projected, target, np.ones(17, dtype=np.float32))
        np.testing.assert_allclose(aligned, target, atol=1e-5)
        np.testing.assert_allclose(aligned[[15, 16]], target[[15, 16]], atol=1e-5)

    def test_evaluator_rejects_frame_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            annotation = [{"id": 0, "keypoints3d": [[0.0, 0.0, 0.0, 1.0]] * 25}]
            (path / "000000.json").write_text(json.dumps(annotation), encoding="utf-8")
            (path / "000001.json").write_text(json.dumps(annotation), encoding="utf-8")
            prediction = np.zeros((1, 17, 3), dtype=np.float32)
            with self.assertRaisesRegex(ValueError, "frame count mismatch"):
                evaluate_3d_error(prediction, str(path))


class BeliefTheoryRegressionTests(unittest.TestCase):
    def test_occlusion_forces_belief_to_zero_regardless_of_other_evidence(self):
        b_occ = torch.tensor([0.0])
        b_det = torch.tensor([1.0])
        b_reproj = torch.tensor([1.0])
        combined = combine_beliefs_dempster_shafer(b_occ, b_det, b_reproj)
        self.assertLess(float(combined[0]), 0.05)

    def test_visible_joint_with_strong_evidence_gets_high_belief(self):
        b_occ = torch.tensor([1.0])
        b_det = torch.tensor([0.9])
        b_reproj = torch.tensor([0.9])
        combined = combine_beliefs_dempster_shafer(b_occ, b_det, b_reproj)
        self.assertGreater(float(combined[0]), 0.8)

    def test_detection_belief_is_clamped(self):
        conf = torch.tensor([-0.5, 0.5, 1.5])
        belief = compute_detection_belief(conf)
        self.assertTrue(torch.equal(belief, torch.tensor([0.0, 0.5, 1.0])))

    def test_reprojection_belief_penalizes_invalid_depth(self):
        proj = torch.zeros(1, 2, 2)
        det = torch.zeros(1, 2, 2)
        valid = torch.tensor([[True, False]])
        belief = compute_reprojection_belief(proj, det, valid, sigma=50.0)
        self.assertAlmostEqual(float(belief[0, 0]), 1.0, places=5)
        self.assertEqual(float(belief[0, 1]), 0.0)

    def test_fuse_pose_precision_weighted_falls_back_to_mirror_when_real_occluded(self):
        pose_real = torch.zeros(1, 21, 3)
        pose_mirror = torch.full((1, 21, 3), 0.3)
        belief_real = torch.zeros(1, 21)
        belief_mirror = torch.ones(1, 21)
        fused = fuse_pose_precision_weighted(pose_real, pose_mirror, belief_real, belief_mirror)
        torch.testing.assert_close(fused, pose_mirror, atol=1e-5, rtol=0.0)

    def test_map_coco_belief_to_smpl_body_shape_and_root_belief(self):
        belief17 = torch.rand(2, 17)
        belief21 = map_coco_belief_to_smpl_body(belief17)
        self.assertEqual(belief21.shape, (2, 21))
        root = root_belief(belief17)
        self.assertEqual(root.shape, (2,))
        expected_root = belief17[:, [5, 6, 11, 12]].mean(dim=-1)
        torch.testing.assert_close(root, expected_root)

    def test_global_orient_anchored_to_real(self):
        from utils.pose_refit import smooth_global_orientation_sequence
        real_go = torch.zeros(5, 3)
        mirror_go = torch.ones(5, 3) * 1.5
        smoothed = smooth_global_orientation_sequence(real_go)
        self.assertLess(float(torch.norm(smoothed - real_go)), 0.1)

    def test_angular_discrepancy_penalty_similar_poses(self):
        from utils.belief_fusion import compute_angular_discrepancy_penalty
        pose_a = torch.zeros(3, 3)
        pose_b = torch.zeros(3, 3) + 0.01
        penalty = compute_angular_discrepancy_penalty(pose_a, pose_b, threshold_deg=45.0)
        self.assertGreater(float(penalty.min()), 0.9)

    def test_angular_discrepancy_penalty_very_different_poses(self):
        from utils.belief_fusion import compute_angular_discrepancy_penalty
        pose_a = torch.zeros(3, 3)
        pose_b = torch.tensor([[3.14, 0.0, 0.0]] * 3)
        penalty = compute_angular_discrepancy_penalty(pose_a, pose_b, threshold_deg=45.0)
        self.assertLess(float(penalty.max()), 0.05)

    def test_angular_discrepancy_penalty_per_joint_batch(self):
        from utils.belief_fusion import compute_angular_discrepancy_penalty
        N = 4
        pose_a = torch.zeros(N, 21, 3)
        pose_b = torch.zeros(N, 21, 3)
        pose_b[:, 0, :] = 2.5
        penalty = compute_angular_discrepancy_penalty(pose_a, pose_b, threshold_deg=90.0)
        self.assertEqual(penalty.shape, (N, 21))
        self.assertLess(float(penalty[:, 0].mean()), 0.25)
        self.assertGreater(float(penalty[:, 1:].mean()), 0.9)


class MirrorGeometryRegressionTests(unittest.TestCase):
    def test_reflect_axis_angle_x_normal_matches_legacy_mirror_axis_angle(self):
        pose = torch.tensor([[0.3, -0.7, 1.1], [-1.2, 0.4, 0.9]])
        x_axis = torch.tensor([1.0, 0.0, 0.0])
        reflected = reflect_axis_angle(pose, x_axis)
        torch.testing.assert_close(reflected, mirror_axis_angle(pose))

    def test_unmirror_pose_general_is_involution_for_arbitrary_normal(self):
        pose = torch.arange(2 * 22 * 3, dtype=torch.float32).reshape(2, 22, 3)
        normal = torch.tensor([0.6, 0.8, 0.0])
        restored = unmirror_pose_general(unmirror_pose_general(pose.clone(), normal), normal)
        torch.testing.assert_close(restored, pose)

    def test_estimate_mirror_normal_robust_agrees_on_clean_x_axis_case(self):
        midpoints = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                                   [0.0, 0.0, 1.0], [0.0, 1.0, 1.0]]).unsqueeze(0)
        offset = torch.tensor([0.5, 0.0, 0.0])
        joints_real = midpoints + offset
        joints_mirror = midpoints - offset
        normal = estimate_mirror_normal_robust(joints_real, joints_mirror)
        self.assertGreater(float(normal.abs() @ torch.tensor([1.0, 0.0, 0.0])), 0.99)

    def test_estimate_mirror_normal_robust_falls_back_when_methods_disagree(self):
        midpoints = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                                   [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]).unsqueeze(0)
        offset = torch.tensor([0.5, 0.0, 0.0])
        joints_real = midpoints + offset
        joints_mirror = midpoints - offset
        normal = estimate_mirror_normal_robust(joints_real, joints_mirror)
        torch.testing.assert_close(normal, torch.tensor([1.0, 0.0, 0.0]))

    def test_skeleton_scale_normalization_matches_real_after_correction(self):
        joints_real = torch.zeros(1, 17, 3)
        joints_real[0, 5] = torch.tensor([-1.0, 1.0, 0.0])   # L_Shoulder
        joints_real[0, 6] = torch.tensor([1.0, 1.0, 0.0])    # R_Shoulder
        joints_real[0, 11] = torch.tensor([-1.0, -1.0, 0.0])  # L_Hip
        joints_real[0, 12] = torch.tensor([1.0, -1.0, 0.0])   # R_Hip
        joints_mirror = joints_real * 0.5

        scale_ratio = compute_skeleton_scale_ratio(joints_real, joints_mirror)
        torch.testing.assert_close(scale_ratio, torch.tensor([2.0]))

        scaled_mirror = normalize_mirror_joints_scale(joints_mirror, scale_ratio)
        corrected_ratio = compute_skeleton_scale_ratio(joints_real, scaled_mirror)
        torch.testing.assert_close(corrected_ratio, torch.tensor([1.0]), atol=1e-5, rtol=0.0)


class MeshRaycastRegressionTests(unittest.TestCase):
    def _cube_mesh(self):
        vertices = np.array([
            [-1, -1, 2], [1, -1, 2], [1, 1, 2], [-1, 1, 2],
            [-1, -1, 4], [1, -1, 4], [1, 1, 4], [-1, 1, 4],
        ], dtype=np.float64)
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # front (facing camera)
            [4, 6, 5], [4, 7, 6],  # back
            [0, 4, 5], [0, 5, 1],  # bottom
            [3, 2, 6], [3, 6, 7],  # top
            [0, 3, 7], [0, 7, 4],  # left
            [1, 5, 6], [1, 6, 2],  # right
        ], dtype=np.int64)
        return vertices, faces

    def test_joint_behind_mesh_is_occluded(self):
        vertices, faces = self._cube_mesh()
        joints = np.array([[0.0, 0.0, 6.0]])
        occluded = compute_ray_occlusion(joints, vertices, faces)
        self.assertTrue(bool(occluded[0]))

    def test_joint_in_front_of_mesh_is_not_occluded(self):
        vertices, faces = self._cube_mesh()
        joints = np.array([[0.0, 0.0, 1.0]])
        occluded = compute_ray_occlusion(joints, vertices, faces)
        self.assertFalse(bool(occluded[0]))

    def test_joint_off_axis_is_not_occluded(self):
        vertices, faces = self._cube_mesh()
        joints = np.array([[5.0, 5.0, 6.0]])
        occluded = compute_ray_occlusion(joints, vertices, faces)
        self.assertFalse(bool(occluded[0]))

    def test_surface_front_is_visible_and_back_is_occluded(self):
        vertices, faces = self._cube_mesh()
        visibility = compute_ray_visibility(
            np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 4.0]]),
            vertices,
            faces,
        )
        self.assertGreater(visibility[0], 0.99)
        self.assertLess(visibility[1], 0.01)

    def test_surface_region_returns_weighted_continuous_visibility(self):
        vertices, faces = self._cube_mesh()
        landmarks = torch.tensor([[[[0.0, 0.0, 2.0], [0.0, 0.0, 4.0]]]])
        weights = torch.tensor([[0.75, 0.25]])
        belief = compute_occlusion_belief_batch(
            landmarks, torch.tensor(vertices)[None], faces, surface_weights=weights
        )
        torch.testing.assert_close(belief, torch.tensor([[0.75]]), atol=1e-5, rtol=0)


class RenderResourceRegressionTests(unittest.TestCase):
    def test_render_video_uses_one_offscreen_renderer(self):
        source = inspect.getsource(render_video)
        self.assertEqual(source.count("pyrender.OffscreenRenderer("), 1)
        self.assertEqual(source.count("renderer.delete()"), 1)


if __name__ == "__main__":
    unittest.main()
