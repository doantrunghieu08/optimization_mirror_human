"""
evaluate.py — Đánh giá định lượng kết quả 3D Pose Estimation.

So sánh hai phương án:
    1. Original HMR4D — kết quả gốc từ HMR4D, chưa qua fusion.
    2. Fused Model    — kết quả sau khi qua pipeline fusion (belief-theory refit).

Metrics:
    MPJPE     : Mean Per-Joint Position Error (mm) — sau root alignment.
    PA-MPJPE  : Procrustes-Aligned MPJPE (mm) — sau Procrustes alignment.

Usage:
    python evaluate.py \\
        --gt_dir    inputs/GT \\
        --real_pt   inputs/real/hmr4d_results_p0.pt \\
        --fused_pkl visualizations/fused_keypoints_3d.pkl \\
        --smpl_model models/SMPL_NEUTRAL.pkl \\
        --output_csv output/evaluation_results.csv
"""

import argparse
import csv
import glob
import json
import os
import pickle

import numpy as np
import torch
import yaml

from utils.smpl_utils import SMPLForwardPass, resolve_result_model_path
from utils.beta_utils import resolve_clip_betas, expand_clip_betas


# ── Mapping COCO-17 (+ neck index 17 nếu có) → OpenPose-25 (GT JSON) ────────
COCO_TO_OP25: dict[int, int] = {
    0:  0,   # Nose
    1:  16,  # L_Eye
    2:  15,  # R_Eye
    3:  18,  # L_Ear
    4:  17,  # R_Ear
    5:  5,   # L_Shoulder
    6:  2,   # R_Shoulder
    7:  6,   # L_Elbow
    8:  3,   # R_Elbow
    9:  7,   # L_Wrist
    10: 4,   # R_Wrist
    11: 12,  # L_Hip
    12: 9,   # R_Hip
    13: 13,  # L_Knee
    14: 10,  # R_Knee
    15: 14,  # L_Ankle
    16: 11,  # R_Ankle
    17: 1,   # Neck
}

# Ngưỡng confidence tối thiểu để tính một khớp
_CONF_THRESHOLD = 0.1

# Person id mặc định (0 = người thật)
_DEFAULT_PERSON_ID = 0


# ═════════════════════════════════════════════════════════════════════════════
# Procrustes alignment
# ═════════════════════════════════════════════════════════════════════════════

def compute_similarity_transform(S1: np.ndarray, S2: np.ndarray) -> np.ndarray:
    """
    Tìm phép biến đổi Similarity (Scale, Rotation, Translation) khớp S1 → S2.

    Args:
        S1 : (N, 3) — Prediction.
        S2 : (N, 3) — Ground truth.

    Returns:
        S1_hat : (N, 3) — S1 sau khi căn chỉnh Procrustes về S2.
    """
    mu1 = S1.mean(axis=0, keepdims=True)
    mu2 = S2.mean(axis=0, keepdims=True)
    X1  = S1 - mu1
    X2  = S2 - mu2

    var1 = np.sum(X1 ** 2)
    if var1 < 1e-8:
        # Suy biến (tất cả khớp trùng 1 điểm) — không thể ước lượng scale/rotation đáng tin cậy.
        # Trả NaN để caller loại bỏ frame này khỏi PA-MPJPE thay vì tính ra sai số ảo.
        return np.full_like(X1, np.nan)

    K    = X1.T @ X2

    U, s, Vh = np.linalg.svd(K)
    V  = Vh.T
    Z  = np.eye(U.shape[0])
    if np.linalg.det(U @ V.T) < 0:
        Z[-1, -1] = -1
    R  = U @ Z @ V.T

    scale  = np.trace(Z @ np.diag(s)) / var1
    S1_hat = scale * X1 @ R + mu2
    return S1_hat


# ═════════════════════════════════════════════════════════════════════════════
# Core evaluation
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_3d_error(
    pred_joints_3d: np.ndarray,
    gt_json_dir:    str,
    person_id:      int = _DEFAULT_PERSON_ID,
) -> tuple[float, float]:
    """
    Tính MPJPE và PA-MPJPE giữa prediction và Ground Truth JSON.

    Args:
        pred_joints_3d : (N, 17+, 3) — Joints dự đoán, đơn vị mét. Cần ≥ 17 khớp COCO.
        gt_json_dir    : Đường dẫn đến thư mục chứa XXXXXX.json.
        person_id      : ID người cần đánh giá (0 = người thật).

    Returns:
        mean_mpjpe    : MPJPE trung bình (mm).
        mean_pampjpe  : PA-MPJPE trung bình (mm).
    """
    pred_joints_3d = np.asarray(pred_joints_3d)
    if pred_joints_3d.ndim != 3 or pred_joints_3d.shape[2] != 3:
        raise ValueError("pred_joints_3d must have shape (N, J, 3)")
    if pred_joints_3d.shape[1] < 17:
        raise ValueError("Evaluator requires at least 17 COCO joints")

    gt_files   = sorted(glob.glob(os.path.join(gt_json_dir, "*.json")))
    if len(pred_joints_3d) != len(gt_files):
        raise ValueError(
            f"Prediction/GT frame count mismatch: "
            f"pred={len(pred_joints_3d)}, gt={len(gt_files)}"
        )
    num_frames = len(gt_files)

    mpjpe_errors    = []
    pampjpe_errors  = []

    for i in range(num_frames):
        with open(gt_files[i], "r", encoding="utf-8") as f:
            gt_data = json.load(f)

        # Tìm annotation của person_id
        person = next(
            (annot for annot in gt_data if annot.get("id") == person_id),
            None,
        )
        if person is None:
            continue

        gt_keypoints = np.array(person["keypoints3d"], dtype=np.float32)  # (25, 4)

        # Chuyển đổi sang mm
        pred_mm = pred_joints_3d[i] * 1000.0
        gt_mm   = gt_keypoints[:, :3] * 1000.0

        # Root alignment tại MidHip
        pred_root = (pred_mm[11] + pred_mm[12]) / 2.0   # LHip + RHip
        gt_root   = gt_mm[8]                             # MidHip trong OP25

        pred_aligned = pred_mm - pred_root
        gt_aligned   = gt_mm   - gt_root

        # Lấy các khớp hợp lệ (theo mapping và confidence)
        valid_pred, valid_gt = [], []
        for coco_idx, op25_idx in COCO_TO_OP25.items():
            if coco_idx >= len(pred_aligned):
                continue
            conf = gt_keypoints[op25_idx, 3] if gt_keypoints.shape[1] == 4 else 1.0
            if (conf > _CONF_THRESHOLD
                    and np.isfinite(pred_aligned[coco_idx]).all()
                    and np.isfinite(gt_aligned[op25_idx]).all()):
                valid_pred.append(pred_aligned[coco_idx])
                valid_gt.append(gt_aligned[op25_idx])

        if not valid_pred:
            continue

        valid_pred = np.array(valid_pred)  # (K, 3)
        valid_gt   = np.array(valid_gt)    # (K, 3)

        # MPJPE
        mpjpe_errors.append(np.mean(np.linalg.norm(valid_pred - valid_gt, axis=1)))

        # PA-MPJPE
        if (len(valid_pred) >= 3
                and np.linalg.matrix_rank(valid_pred - valid_pred.mean(axis=0)) >= 2
                and np.linalg.matrix_rank(valid_gt - valid_gt.mean(axis=0)) >= 2):
            pred_pa = compute_similarity_transform(valid_pred, valid_gt)
            if np.isfinite(pred_pa).all():
                pampjpe_errors.append(np.mean(np.linalg.norm(pred_pa - valid_gt, axis=1)))

    if not mpjpe_errors:
        raise ValueError("No valid frames were available for evaluation")

    mean_mpjpe   = float(np.mean(mpjpe_errors))
    mean_pampjpe = float(np.mean(pampjpe_errors)) if pampjpe_errors else float("nan")

    print(f"  Frames considered: {num_frames}")
    print(f"  Frames evaluated : {len(mpjpe_errors)}")
    print(f"  MPJPE            : {mean_mpjpe:.2f} mm")
    print(f"  PA-MPJPE         : {mean_pampjpe:.2f} mm")
    return mean_mpjpe, mean_pampjpe


def _as_torch_matrix(value, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim == 3 and tensor.shape[1] == 1:
        tensor = tensor.squeeze(1)
    return tensor


def smpl_joints_from_params(
    smpl: SMPLForwardPass,
    global_orient,
    body_pose,
    betas,
    transl,
    device: torch.device,
) -> np.ndarray:
    """SMPL forward → (N, 18, 3) numpy, COCO-17 + neck."""
    go = _as_torch_matrix(global_orient, device)
    bp = _as_torch_matrix(body_pose, device)
    bt = _as_torch_matrix(betas, device)
    tr = _as_torch_matrix(transl, device)
    if bt.shape[0] == 1 and go.shape[0] > 1:
        bt = bt.expand(go.shape[0], -1)
    with torch.no_grad():
        return smpl(
            global_orient=go,
            body_pose=bp,
            betas=bt,
            transl=tr,
        ).cpu().numpy()


def load_eval_betas(real_data: dict, config: dict, device: torch.device) -> torch.Tensor:
    """Evaluate the original observation with its original shape."""
    return real_data["smpl_params_incam"]["betas"][..., :10].to(device)


def main(args: argparse.Namespace) -> None:
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.smpl_model or config["model"]["smpl_model_path"]
    smpl = SMPLForwardPass(model_path=model_path, device=device)

    results: dict[str, tuple[float, float]] = {}

    print("\n[ ORIGINAL HMR4D ]")
    real_data = torch.load(args.real_pt, map_location="cpu", weights_only=False)
    eval_betas = load_eval_betas(real_data, config, device)

    orig_joints = smpl_joints_from_params(
        smpl,
        real_data["smpl_params_incam"]["global_orient"],
        real_data["smpl_params_incam"]["body_pose"],
        eval_betas,
        real_data["smpl_params_incam"]["transl"],
        device,
    )
    orig_mpjpe, orig_pampjpe = evaluate_3d_error(orig_joints, args.gt_dir)
    results["Original HMR4D"] = (orig_mpjpe, orig_pampjpe)

    fused_pkl = args.fused_pkl

    if os.path.exists(fused_pkl):
        print("\n[ FUSED MODEL ]")
        with open(fused_pkl, "rb") as f:
            fused_data = pickle.load(f)

        if {"global_orient", "body_pose", "transl_incam"} <= set(fused_data):
            n_fused = np.asarray(fused_data["global_orient"]).shape[0]
            fused_betas = fused_data.get("betas", eval_betas[:n_fused])
            fused_model_path = resolve_result_model_path(model_path, fused_data.get("body_model_type", "smpl"))
            fused_smpl = smpl if fused_model_path == model_path else SMPLForwardPass(fused_model_path, device)
            fused_joints = smpl_joints_from_params(
                fused_smpl,
                fused_data["global_orient"],
                fused_data["body_pose"],
                fused_betas,
                fused_data["transl_incam"],
                device,
            )
        elif "joints_3d_incam" in fused_data:
            print("  WARNING: using stored joints_3d_incam (legacy PKL); beta policy may differ.")
            fused_joints = fused_data["joints_3d_incam"]
        else:
            raise KeyError(
                "Fused PKL must contain SMPL params "
                "(global_orient, body_pose, transl_incam) or joints_3d_incam"
            )

        fused_mpjpe, fused_pampjpe = evaluate_3d_error(fused_joints, args.gt_dir)
        results["Fused Model"] = (fused_mpjpe, fused_pampjpe)
    else:
        print(f"\n[evaluate] Không tìm thấy fused pkl tại: {fused_pkl}")

    # ── 3. Ghi CSV báo cáo ────────────────────────────────────────────────
    if len(results) > 0:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        with open(args.output_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Header
            writer.writerow(["Metric"] + list(results.keys()))
            # Rows
            writer.writerow(
                ["MPJPE (mm)"] + [round(v[0], 2) for v in results.values()]
            )
            writer.writerow(
                ["PA-MPJPE (mm)"] + [round(v[1], 2) for v in results.values()]
            )
        print(f"\n[evaluate] CSV report saved to: {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 3D Pose Estimation")
    parser.add_argument(
        "--config",
        type    = str,
        default = "configs/default.yaml",
        help    = "YAML dùng chung beta policy với inference",
    )
    parser.add_argument(
        "--gt_dir",
        type    = str,
        default = "inputs/GT",
        help    = "Thư mục chứa GT JSON 3D",
    )
    parser.add_argument(
        "--real_pt",
        type    = str,
        default = "inputs/real/hmr4d_results_p0.pt",
        help    = "File .pt HMR4D của real person",
    )
    parser.add_argument(
        "--fused_pkl",
        type    = str,
        default = "output/inference_results.pkl",
        help    = "File .pkl chứa kết quả fused",
    )
    parser.add_argument(
        "--smpl_model",
        type    = str,
        default = None,
        help    = "Đường dẫn body model (mặc định lấy từ config)",
    )
    parser.add_argument(
        "--output_csv",
        type    = str,
        default = "output/evaluation_results.csv",
        help    = "Đường dẫn file CSV đầu ra",
    )
    main(parser.parse_args())
