"""
render_video.py
===============
Đọc kết quả inference (file .pkl) và render ra video visualization.
Tách biệt hoàn toàn với quá trình suy luận (xem inference.py).

Pipeline:
    inference.py  →  inference_results.pkl  →  render_video.py  →  output.mp4

Usage:
    # Render sau khi đã chạy inference.py:
    python render_video.py --config configs/default.yaml

    # Hoặc chỉ định pkl và video cụ thể:
    python render_video.py --pkl output/inference_results.pkl \\
                           --input_video inputs/0_input_video.mp4 \\
                           --output_video output/fused_output.mp4

    # Chạy inference rồi render liền:
    python render_video.py --config configs/default.yaml --run_inference
"""

import os
import argparse
import yaml
import numpy as np
import cv2
import pickle
from tqdm import tqdm

from visualizations.mesh_renderer import render_smpl_mesh, render_smpl_mesh_overlay, PYRENDER_AVAILABLE
from visualizations.vis_utils import COCO_SKELETON
from utils.smpl_utils import SMPLForwardPass, resolve_result_model_path
from utils.camera_utils import project_3d_to_2d
import torch

if not PYRENDER_AVAILABLE:
    print("CẢNH BÁO: pyrender/trimesh chưa được cài — video sẽ chỉ hiện placeholder thay vì SMPL mesh.")


def _render_overlay_panel_mesh(frame: np.ndarray, vertices: np.ndarray, faces: np.ndarray, K_np: np.ndarray, renderer=None) -> np.ndarray:
    """Overlay SMPL mesh (không phải skeleton) lên frame video, dùng camera K thật."""
    return render_smpl_mesh_overlay(frame, vertices, faces, K_np, renderer=renderer)


def _render_global_panel_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    height: int,
    width: int,
    fixed_target_height: float,
    renderer=None,
) -> np.ndarray:
    """Render SMPL mesh từ góc nhìn 3D tự do (global view), camera framing cố định để tránh giật cục."""
    return render_smpl_mesh(
        vertices, faces,
        img_width=width, img_height=height,
        flip_y=False, add_floor=True,
        fixed_target_height=fixed_target_height,
        renderer=renderer,
    )


def _draw_dual_2d_skeleton(
    frame: np.ndarray,
    joints_2d_real: np.ndarray,
    joints_2d_mirror: np.ndarray | None = None,
    belief_real: np.ndarray | None = None,
    belief_mirror: np.ndarray | None = None,
    occlusion_threshold: float = 0.3,
) -> np.ndarray:
    """Vẽ skeleton 2D của cả real và mirror chồng lên nhau trên frame.

    Real skeleton: xanh lá (0, 255, 0)
    Mirror skeleton: xanh dương (255, 165, 0)
    Khớp occluded ở real: đỏ (0, 0, 255)
    Khớp được sửa bởi mirror: vàng (0, 255, 255)
    """
    img = frame.copy()
    h, w = img.shape[:2]

    # Vẽ mirror skeleton trước (dưới real)
    if joints_2d_mirror is not None:
        for (i, j) in COCO_SKELETON:
            if i >= len(joints_2d_mirror) or j >= len(joints_2d_mirror):
                continue
            pt1 = (int(joints_2d_mirror[i, 0]), int(joints_2d_mirror[i, 1]))
            pt2 = (int(joints_2d_mirror[j, 0]), int(joints_2d_mirror[j, 1]))
            # Kiểm tra bounds
            if 0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h:
                cv2.line(img, pt1, pt2, (255, 165, 0), 1, cv2.LINE_AA)
        for idx in range(min(len(joints_2d_mirror), 17)):
            pt = (int(joints_2d_mirror[idx, 0]), int(joints_2d_mirror[idx, 1]))
            if 0 <= pt[0] < w and 0 <= pt[1] < h:
                cv2.circle(img, pt, 3, (255, 165, 0), -1)

    # Vẽ real skeleton
    for (i, j) in COCO_SKELETON:
        if i >= len(joints_2d_real) or j >= len(joints_2d_real):
            continue
        pt1 = (int(joints_2d_real[i, 0]), int(joints_2d_real[i, 1]))
        pt2 = (int(joints_2d_real[j, 0]), int(joints_2d_real[j, 1]))
        if 0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h:
            cv2.line(img, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)

    # Vẽ khớp với màu theo trạng thái occlusion
    for idx in range(min(len(joints_2d_real), 17)):
        pt = (int(joints_2d_real[idx, 0]), int(joints_2d_real[idx, 1]))
        if not (0 <= pt[0] < w and 0 <= pt[1] < h):
            continue
        if belief_real is not None and idx < len(belief_real):
            if belief_real[idx] < occlusion_threshold:
                if belief_mirror is not None and idx < len(belief_mirror) and belief_mirror[idx] >= occlusion_threshold:
                    # Khớp được sửa bởi mirror — vàng
                    color = (0, 255, 255)
                    radius = 5
                else:
                    # Khớp occluded, mirror cũng không giúp được — đỏ
                    color = (0, 0, 255)
                    radius = 5
            else:
                color = (0, 255, 0)
                radius = 3
        else:
            color = (0, 255, 0)
            radius = 3
        cv2.circle(img, pt, radius, color, -1)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# Main render function
# ─────────────────────────────────────────────────────────────────────────────
def render_video(config: dict, pkl_path: str, input_video: str, output_video: str,
                 max_frames: int = -1) -> None:
    """
    Đọc file PKL (từ inference.py) và render ra video 2-panel.

    Panel 1 (trái): SMPL mesh overlay lên video gốc (incam space).
    Panel 2 (phải): SMPL mesh 3D render từ góc nhìn tự do (global space).

    Args:
        pkl_path    : đường dẫn file .pkl từ inference.py.
        input_video : video gốc để làm nền cho panel 1.
        output_video: đường dẫn file mp4 output.
        max_frames  : giới hạn số frame (-1 = không giới hạn).
    """
    # ── 1. Load kết quả inference ────────────────────────────────────────────
    print(f"Loading inference results: {pkl_path}")
    with open(pkl_path, "rb") as f:
        results = pickle.load(f)

    if max_frames == 0 or max_frames < -1:
        raise ValueError("max_frames must be -1 or a positive integer")
    required = ("global_orient", "body_pose", "betas", "transl_incam", "transl_global", "K_fullimg")
    missing = [key for key in required if key not in results]
    if missing:
        raise KeyError(f"Inference results missing required fields: {', '.join(missing)}")
    total_frames = results["global_orient"].shape[0]
    if total_frames == 0:
        raise ValueError("Inference results contain no frames")
    for key in required:
        length = len(results[key])
        if key == "betas" and length == 1:
            continue
        if length != total_frames:
            raise ValueError(f"Inference result frame mismatch: global_orient={total_frames}, {key}={length}")
    n_frames = total_frames if max_frames == -1 else min(total_frames, max_frames)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = resolve_result_model_path(
        config["model"]["smpl_model_path"], results.get("body_model_type", "smpl")
    )
    smpl = SMPLForwardPass(model_path=model_path, device=device)

    # Chiều cao tham chiếu cố định (tư thế T-pose, không phụ thuộc frame) để camera
    # panel global "look-at" ổn định — nếu tính động theo mỗi frame, vung tay/ngồi
    # xổm sẽ đổi bounding-box height và làm camera bị giật/nảy theo tư thế.
    with torch.no_grad():
        _, ref_verts_t, _ = smpl(
            torch.zeros((1, 3), device=device),
            torch.zeros((1, 63), device=device),
            torch.from_numpy(results["betas"][0:1]).to(device),
            torch.zeros((1, 3), device=device),
            return_mesh=True,
        )
    ref_verts_np = ref_verts_t[0].cpu().numpy()
    fixed_target_height = float(ref_verts_np[:, 1].max() - ref_verts_np[:, 1].min()) * 0.5

    # ── 2. Mở video đầu vào ──────────────────────────────────────────────────
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open input video: {input_video}")
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_frames > 0 and video_frames < n_frames:
        cap.release()
        raise ValueError(f"Input video is shorter than inference results: video={video_frames}, result={n_frames}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Input video has invalid dimensions: {width}x{height}")

    # ── 3. Mở VideoWriter output ──────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_video) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # 3 panels: overlay + dual 2D + 3D global
    out = cv2.VideoWriter(output_video, fourcc, fps, (width * 3, height))
    if not out.isOpened():
        cap.release()
        out.release()
        raise RuntimeError(f"Could not open output video writer: {output_video}")

    print(f"Rendering {n_frames} frames -> {output_video}")
    label1 = "SMPL Mesh Overlay"

    # Hai panel cùng kích thước nên dùng chung một EGL context.
    renderer = None
    try:
        if PYRENDER_AVAILABLE:
            import pyrender
            renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)

        for frame_idx in tqdm(range(n_frames)):
            # Đọc frame video gốc
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Could not read input video frame {frame_idx}")

            # Lấy dữ liệu frame
            real_go_t = torch.from_numpy(results["global_orient"][frame_idx:frame_idx+1]).to(device)
            global_go_t = torch.from_numpy(results["global_orient_global"][frame_idx:frame_idx+1]).to(device) if "global_orient_global" in results else real_go_t
            fused_bp_t = torch.from_numpy(results["body_pose"][frame_idx:frame_idx+1]).to(device)
            beta_idx = 0 if len(results["betas"]) == 1 else frame_idx
            betas_t = torch.from_numpy(results["betas"][beta_idx:beta_idx+1]).to(device)
            transl_incam_t = torch.from_numpy(results["transl_incam"][frame_idx:frame_idx+1]).to(device)
            transl_global_t = torch.from_numpy(results["transl_global"][frame_idx:frame_idx+1]).to(device)
            K_np = results["K_fullimg"][frame_idx]
            with torch.no_grad():
                _, verts_incam_t, faces_np = smpl(real_go_t, fused_bp_t, betas_t, transl_incam_t, return_mesh=True)
                _, verts_global_t, _ = smpl(global_go_t, fused_bp_t, betas_t, transl_global_t, return_mesh=True)

            verts_incam = verts_incam_t[0].cpu().numpy()
            verts_global = verts_global_t[0].cpu().numpy()

            # ── Panel 1: Overlay SMPL mesh lên video gốc ─────────────────────────────
            panel1 = _render_overlay_panel_mesh(frame=frame, vertices=verts_incam, faces=faces_np, K_np=K_np, renderer=renderer)
            cv2.putText(panel1, label1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # ── Panel 2: Dual 2D projection (real + mirror) ───────────────────
            with torch.no_grad():
                joints18_real = smpl(real_go_t, fused_bp_t, betas_t, transl_incam_t)
                proj_real_2d, _ = project_3d_to_2d(joints18_real[:, :17, :], torch.from_numpy(K_np).unsqueeze(0).to(device))

            proj_real_np = proj_real_2d[0].cpu().numpy()

            # Load belief diagnostics nếu có
            belief_real_frame = None
            belief_mirror_frame = None
            if "belief_real" in results:
                belief_real_frame = results["belief_real"][frame_idx]  # (17,)
            if "belief_mirror" in results:
                belief_mirror_frame = results["belief_mirror"][frame_idx]

            # ViTPose của real view; đây không phải projection/detection mirror.
            detected_real_2d = None
            if "kp2d_real" in results:
                detected_real_2d = results["kp2d_real"][frame_idx]  # (17, 2)

            panel2 = _draw_dual_2d_skeleton(
                frame.copy(), proj_real_np,
                joints_2d_mirror=detected_real_2d,
                belief_real=belief_real_frame,
                belief_mirror=belief_mirror_frame,
            )
            cv2.putText(panel2, "Real view: Green=Fused, Blue=Detected real 2D", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # Legend cho màu khớp
            cv2.putText(panel2, "Red=Low trust  Yellow=Mirror evidence", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # ── Panel 3: SMPL mesh 3D global view ───────────────────────────
            panel3 = _render_global_panel_mesh(verts_global, faces_np, height, width, fixed_target_height, renderer=renderer)
            cv2.putText(panel3, "3D Mesh Global View", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2)

            # ── Ghép và ghi frame ─────────────────────────────────────────
            combined = np.concatenate((panel1, panel2, panel3), axis=1)
            out.write(combined)
    finally:
        if renderer is not None:
            renderer.delete()
        cap.release()
        out.release()
    print(f"Video saved: {output_video}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render visualization video từ kết quả inference.")
    parser.add_argument("--config",       type=str, default="configs/default.yaml",
                        help="File config YAML.")
    parser.add_argument("--pkl",          type=str, default=None,
                        help="Đường dẫn file PKL (mặc định lấy từ config output_dir).")
    parser.add_argument("--input_video",  type=str, default=None,
                        help="Video gốc để overlay (mặc định lấy từ config).")
    parser.add_argument("--output_video", type=str, default=None,
                        help="Đường dẫn video output (mặc định lấy từ config).")
    parser.add_argument("--run_inference", action="store_true",
                        help="Chạy inference.py trước khi render (tiện lợi cho pipeline đầy đủ).")
    parser.add_argument("--bypass_refit", action="store_true",
                        help="Bypass pose refit khi --run_inference (debug).")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    vis_cfg = config.get("visualization", {})
    output_dir = vis_cfg.get("output_dir", "output")

    # Resolve paths
    pkl_path     = args.pkl          or os.path.join(output_dir, "inference_results.pkl")
    input_video  = args.input_video  or vis_cfg.get("input_video", "inputs/0_input_video.mp4")
    output_video = args.output_video or vis_cfg.get("output_video", os.path.join(output_dir, "fused_output.mp4"))
    max_frames   = vis_cfg.get("max_frames", -1)

    # Optionally chạy inference trước
    if args.run_inference or not os.path.exists(pkl_path):
        if not os.path.exists(pkl_path):
            print(f"PKL not found at {pkl_path}. Running inference first...")
        from inference import run_inference
        pkl_path = run_inference(config, bypass_refit=args.bypass_refit)

    render_video(config, pkl_path, input_video, output_video, max_frames=max_frames)
