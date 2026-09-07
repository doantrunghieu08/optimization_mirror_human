"""Audit real/mirror reprojection, temporal motion and source-pose drift."""

import argparse
import json
import pickle
from pathlib import Path

import torch
import yaml

from utils.camera_utils import project_3d_to_2d
from utils.geometry import axis_angle_to_quaternion
from utils.pose_refit import _to_mirror_frame, estimate_fixed_inter_view_rotation
from utils.smpl_utils import SMPLForwardPass, resolve_result_model_path, unmirror_pose


def _input_path(value):
    path = Path(value)
    return path / "hmr4d_results.pt" if path.is_dir() else path


def _load_keypoints(pose_path):
    return torch.load(
        next((pose_path.parent / "preprocess").glob("vitpose*.pt")),
        map_location="cpu",
        weights_only=True,
    )


def _summary(values):
    if values.numel() == 0:
        return {"mean": None, "p95": None}
    return {
        "mean": round(float(values.mean()), 4),
        "p95": round(float(torch.quantile(values.flatten(), 0.95)), 4),
    }


def _rotation_metrics(global_orient, body_pose):
    pose = torch.cat([global_orient[:, None], body_pose.view(len(body_pose), 21, 3)], dim=1)
    q = axis_angle_to_quaternion(pose)
    velocity = torch.rad2deg(
        2 * torch.acos((q[1:] * q[:-1]).sum(-1).abs().clamp(max=1 - 1e-7))
    )
    acceleration = (velocity[1:] - velocity[:-1]).abs()
    return {"velocity_deg": _summary(velocity), "acceleration_deg": _summary(acceleration)}


def _pose_change(candidate, source):
    candidate_q = axis_angle_to_quaternion(candidate.view(len(candidate), -1, 3))
    source_q = axis_angle_to_quaternion(source.view(len(source), -1, 3))
    angle = torch.rad2deg(
        2 * torch.acos((candidate_q * source_q).sum(-1).abs().clamp(max=1 - 1e-7))
    )
    result = _summary(angle)
    result["spike_over_45_deg_ratio"] = round(float((angle > 45).float().mean()), 6)
    return result


def _view_metrics(smpl, global_orient, body_pose, betas, transl, K, keypoints):
    projections, valid_depth = [], []
    if len(betas) == 1:
        betas = betas.expand(len(global_orient), -1)
    with torch.no_grad():
        for start in range(0, len(global_orient), 32):
            stop = start + 32
            joints = smpl(
                global_orient[start:stop],
                body_pose[start:stop],
                betas[start:stop],
                transl[start:stop],
            )
            xy, valid = project_3d_to_2d(joints[:, :17], K[start:stop])
            projections.append(xy.cpu())
            valid_depth.append(valid.cpu())

    xy = torch.cat(projections)
    target = keypoints[: len(xy), :, :2]
    valid_depth = torch.cat(valid_depth)
    valid = valid_depth & torch.isfinite(xy).all(-1) & torch.isfinite(target).all(-1)
    error = (xy - target).norm(dim=-1)
    error = torch.where(valid & torch.isfinite(error), error, torch.zeros_like(error))
    confidence = keypoints[: len(xy), :, 2].clamp(0, 1)
    confidence = torch.where(valid & torch.isfinite(confidence), confidence, torch.zeros_like(confidence))
    denominator = confidence.sum()
    high = (confidence >= 0.5) & valid
    per_frame_den = confidence.sum(-1)
    per_frame = (error * confidence).sum(-1) / per_frame_den.clamp_min(1e-8)
    return {
        "confidence_weighted_error_px": (
            float((error * confidence).sum() / denominator) if denominator > 0 else None
        ),
        "high_confidence_error_px": float(error[high].mean()) if high.any() else None,
        "invalid_depth_ratio": round(float((~valid_depth).float().mean()), 6),
        "low_confidence_joint_ratio": round(float((confidence < 0.5).float().mean()), 6),
        "low_confidence_frame_ratio": round(float((confidence.mean(-1) < 0.5).float().mean()), 6),
        "per_frame_confidence_weighted_error_px": [
            round(float(value), 4) if den > 0 else None
            for value, den in zip(per_frame, per_frame_den)
        ],
        "per_joint_error_px": [
            round(float(error[:, joint][high[:, joint]].mean()), 3)
            if high[:, joint].any()
            else None
            for joint in range(17)
        ],
    }


def audit(config, result_paths):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = config["model"]["smpl_model_path"]
    real_path = _input_path(config["data"]["real_dir"])
    mirror_path = _input_path(config["data"]["mirror_dir"])
    real = torch.load(real_path, map_location="cpu", weights_only=True)
    mirror = torch.load(mirror_path, map_location="cpu", weights_only=True)
    kp_real, kp_mirror = _load_keypoints(real_path), _load_keypoints(mirror_path)
    raw_real, raw_mirror = real["smpl_params_incam"], mirror["smpl_params_incam"]
    fixed_rotation, residual = estimate_fixed_inter_view_rotation(
        raw_real["global_orient"], raw_mirror["global_orient"]
    )

    model_type = "smplx" if Path(model_path).name.upper().startswith("SMPLX") else "smpl"
    candidates = {
        "raw_hmr4d": {
            **raw_real,
            "transl_incam": raw_real["transl"],
            "K_fullimg": real["K_fullimg"],
            "body_model_type": model_type,
        }
    }
    for path in result_paths:
        with open(path, "rb") as stream:
            candidates[str(path)] = pickle.load(stream)

    mirror_source = unmirror_pose(
        torch.cat(
            [raw_mirror["global_orient"][:, None], raw_mirror["body_pose"].view(-1, 21, 3)],
            dim=1,
        )
    )[:, 1:].reshape(-1, 63)
    report, models = {}, {}
    for name, params in candidates.items():
        result_type = params.get("body_model_type", "smpl")
        if result_type not in models:
            models[result_type] = SMPLForwardPass(
                resolve_result_model_path(model_path, result_type), device
            )
        smpl = models[result_type]
        n = len(params["global_orient"])
        if n == 0 or n > min(len(kp_real), len(kp_mirror)):
            raise ValueError(
                f"Invalid frame count for {name}: result={n}, "
                f"real_keypoints={len(kp_real)}, mirror_keypoints={len(kp_mirror)}"
            )

        tensor = lambda key: torch.as_tensor(params[key][:n], device=device)
        go, body_pose, betas = tensor("global_orient"), tensor("body_pose"), tensor("betas")
        real_metrics = _view_metrics(
            smpl, go, body_pose, betas, tensor("transl_incam"), tensor("K_fullimg"), kp_real
        )
        mirror_go, mirror_body = _to_mirror_frame(
            go, body_pose, inter_view_rotation=fixed_rotation.to(device)
        )
        mirror_metrics = _view_metrics(
            smpl,
            mirror_go,
            mirror_body,
            betas,
            raw_mirror["transl"][:n].to(device),
            mirror["K_fullimg"][:n].to(device),
            kp_mirror,
        )
        report[name] = {
            "frames": n,
            "body_model_type": result_type,
            "real_reprojection": real_metrics,
            "mirror_reprojection": mirror_metrics,
            "rotation": _rotation_metrics(go, body_pose),
            "pose_change_vs_real_source_deg": _pose_change(
                body_pose.cpu(), raw_real["body_pose"][:n]
            ),
            "pose_change_vs_mirror_source_deg": _pose_change(
                body_pose.cpu(), mirror_source[:n]
            ),
            "inter_view_rotation_residual_deg": _summary(residual[:n]),
        }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--results", nargs="+", default=["output/inference_results.pkl"])
    parser.add_argument("--output", default="output/fusion_audit.json")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        report = audit(yaml.safe_load(stream), args.results)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
