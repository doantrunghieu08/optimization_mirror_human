"""
api/config_utils.py — Build config riêng cho mỗi job từ configs/default.yaml.

Không ghi đè file configs/default.yaml dùng chung — mỗi job nhận 1 bản deep
copy với đường dẫn data/visualization trỏ vào runs/{job_id}/ của chính nó,
để nhiều job không tranh chấp input/output với nhau.
"""

from __future__ import annotations

import copy

import yaml

from api.jobs import Job

DEFAULT_CONFIG_PATH = "configs/default.yaml"


def load_default_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_job_config(job: Job, config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    config = copy.deepcopy(load_default_config(config_path))
    config["data"]["real_dir"] = str(job.inputs_dir / "real" / "hmr4d_results.pt")
    config["data"]["mirror_dir"] = str(job.inputs_dir / "mirror" / "hmr4d_results.pt")
    config["visualization"]["output_dir"] = str(job.output_dir)

    video_path = job.uploaded_files.get("video")
    if not video_path:
        video_candidates = list(job.inputs_dir.glob("input_video.*"))
        if video_candidates:
            video_path = str(video_candidates[0])
        else:
            video_path = str(job.inputs_dir / "input_video.mp4")

    config["visualization"]["input_video"] = str(video_path)
    config["visualization"]["output_video"] = str(job.output_dir / "fused_output.mp4")
    return config
