"""Compare raw HMR4D and fused projections against the same ViTPose detections."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml

from utils.camera_utils import project_3d_to_2d
from utils.smpl_utils import SMPLForwardPass, resolve_result_model_path


def audit(config, result_paths):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = config['model']['smpl_model_path']
    real_path = Path(config['data']['real_dir'])
    if real_path.is_dir():
        real_path = real_path / 'hmr4d_results.pt'
    real = torch.load(real_path, map_location='cpu', weights_only=True)
    kp = torch.load(next((real_path.parent / 'preprocess').glob('vitpose*.pt')),
                    map_location='cpu', weights_only=True)
    raw = real['smpl_params_incam']
    model_type = 'smplx' if Path(model_path).name.upper().startswith('SMPLX') else 'smpl'
    candidates = {'raw_hmr4d': {**raw, 'transl_incam': raw['transl'],
                                'K_fullimg': real['K_fullimg'], 'body_model_type': model_type}}
    for path in result_paths:
        with open(path, 'rb') as stream:
            candidates[str(path)] = pickle.load(stream)
    report = {}
    models = {}
    for name, params in candidates.items():
        result_type = params.get('body_model_type', 'smpl')
        if result_type not in models:
            models[result_type] = SMPLForwardPass(resolve_result_model_path(model_path, result_type), device)
        smpl = models[result_type]
        n = len(params['global_orient'])
        if n == 0 or n > len(kp):
            raise ValueError(f"Invalid frame count for {name}: result={n}, keypoints={len(kp)}")
        projections = []
        valid_depth = []
        with torch.no_grad():
            for start in range(0, n, 32):
                get = lambda key: torch.as_tensor(params[key][start:start + 32], device=device)
                joints = smpl(get('global_orient'), get('body_pose'), get('betas'), get('transl_incam'))
                xy, valid = project_3d_to_2d(joints[:, :17], get('K_fullimg'))
                projections.append(xy.cpu())
                valid_depth.append(valid.cpu())
        xy = torch.cat(projections)
        target = kp[:n, :, :2]
        valid = torch.cat(valid_depth) & torch.isfinite(xy).all(-1) & torch.isfinite(target).all(-1)
        err = (xy - target).norm(dim=-1)
        err = torch.where(valid & torch.isfinite(err), err, torch.zeros_like(err))
        conf = kp[:n, :, 2].clamp(0, 1)
        conf = torch.where(valid & torch.isfinite(conf), conf, torch.zeros_like(conf))
        per_frame_den = conf.sum(-1)
        per_frame = (err * conf).sum(-1) / per_frame_den.clamp_min(1e-8)
        high = (conf >= 0.5) & valid
        body = high.clone()
        body[:, :5] = False
        weighted_den = conf.sum()
        weighted_error = float((err * conf).sum() / weighted_den) if weighted_den > 0 else None
        report[name] = {
            'frames': n,
            'body_model_type': result_type,
            'confidence_weighted_error_px': weighted_error,
            'high_confidence_error_px': float(err[high].mean()) if high.any() else None,
            'body_high_confidence_error_px': float(err[body].mean()) if body.any() else None,
            'per_frame_confidence_weighted_error_px': [
                round(float(value), 4) if denominator > 0 else None
                for value, denominator in zip(per_frame, per_frame_den)
            ],
            'per_joint_error_px': [
                round(float(err[:, j][high[:, j]].mean()), 3) if high[:, j].any() else None
                for j in range(17)
            ],
        }
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--results', nargs='+', default=['output/inference_results.pkl'])
    parser.add_argument('--output', default='output/fusion_audit.json')
    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as stream:
        report = audit(yaml.safe_load(stream), args.results)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({name: {key: value for key, value in row.items() if not key.startswith('per_')}
                      for name, row in report.items()}, indent=2))
