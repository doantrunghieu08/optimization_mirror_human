import unittest
import torch

from utils.belief_fusion import (
    combine_beliefs_dempster_shafer,
    compute_cross_view_trust,
    compute_full_view_belief,
    fuse_global_orient_precision_weighted,
    fuse_pose_precision_weighted,
)


class TestDiscrepancyFusion(unittest.TestCase):
    def test_dempster_shafer_veto_rule(self):
        """Test that occlusion (b_occ=0) heavily discounts present mass in Dempster-Shafer fusion."""
        b_occ = torch.tensor([0.0])  # occluded
        b_det = torch.tensor([1.0])  # confident 2D detector
        b_reproj = torch.tensor([1.0])  # perfect 2D reprojection

        combined = combine_beliefs_dempster_shafer(b_occ, b_det, b_reproj)
        self.assertLess(combined.item(), 0.1, "Occluded joint must have low belief despite confident detector")

    def test_compute_full_view_belief_uses_dempster_shafer(self):
        """Test compute_full_view_belief returns valid belief structure."""
        j3d = torch.zeros(1, 17, 3)
        verts = torch.zeros(1, 6890, 3)
        faces = torch.zeros(13776, 3, dtype=torch.long)
        kp2d_det = torch.zeros(1, 17, 2)
        kp2d_conf = torch.ones(1, 17)
        proj_2d = torch.zeros(1, 17, 2)
        valid_mask = torch.ones(1, 17, dtype=torch.bool)

        res = compute_full_view_belief(
            j3d, verts, faces, kp2d_det, kp2d_conf, proj_2d, valid_mask,
            combine_method="dempster_shafer"
        )
        self.assertIn("belief", res)
        self.assertIn("b_occ", res)
        self.assertIn("b_det", res)
        self.assertIn("b_reproj", res)
        self.assertEqual(res["belief"].shape, (1, 17))

    def test_fuse_global_orient_precision_weighted(self):
        """Test precision-weighted fusion on SO(3) root orientations."""
        go_real = torch.tensor([[0.0, 0.0, 0.0]])
        go_mirror = torch.tensor([[0.0, 0.5, 0.0]])
        b_real = torch.tensor([0.2])
        b_mirror = torch.tensor([0.8])

        fused = fuse_global_orient_precision_weighted(go_real, go_mirror, b_real, b_mirror)
        self.assertEqual(fused.shape, (1, 3))
        # Since mirror belief is 0.8 / 1.0 = 0.8, fused orient should be closer to mirror
        self.assertGreater(fused[0, 1].item(), 0.25)


if __name__ == "__main__":
    unittest.main()
