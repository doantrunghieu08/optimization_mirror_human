import torch
import os
import glob
from torch.utils.data import Dataset, Sampler
from utils.smpl_utils import unmirror_pose, unmirror_transl
from utils.gt_utils import load_gt_heights_from_dir, FALLBACK_HEIGHT_M, normalize_optional_dir
from utils.beta_utils import resolve_clip_betas, expand_clip_betas


def flip_camera_intrinsics(K: torch.Tensor, image_width: float) -> torch.Tensor:
    """
    Biến đổi camera intrinsics khi ảnh bị lật ngang.
    Chỉ cx (principal point X) thay đổi.
    """
    flipped = K.clone()
    flipped[..., 0, 2] = image_width - 1 - K[..., 0, 2]
    return flipped


class ConsecutiveBatchSampler(Sampler):
    """
    Mỗi batch là một cửa sổ frame liên tiếp — bắt buộc cho temporal / bone-stability loss.
    Các cửa sổ không chồng nhau; thứ tự cửa sổ có thể shuffle (train).
    """

    def __init__(
        self,
        n_samples: int,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = True,
        seed: int | None = None,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self._epoch = 0
        if drop_last:
            n_windows = n_samples // batch_size
            self.starts = [i * batch_size for i in range(n_windows)]
        else:
            self.starts = list(range(0, n_samples, batch_size))

    def __iter__(self):
        starts = list(self.starts)
        if self.shuffle:
            g = torch.Generator()
            if self.seed is not None:
                g.manual_seed(self.seed + self._epoch)
            self._epoch += 1
            perm = torch.randperm(len(starts), generator=g).tolist()
            starts = [starts[i] for i in perm]
        for s in starts:
            end = min(s + self.batch_size, self.n_samples)
            yield list(range(s, end))

    def __len__(self) -> int:
        return len(self.starts)


class MirrorFusionDataset(Dataset):
    """
    Dataset cho bài toán Fusion Pose từ 2 nguồn: Người thật + Ảnh gương.

    Pipeline:
        - Nguồn 1 (real)  : hmr4d_results.pt chạy trên video người thật
        - Nguồn 2 (mirror): hmr4d_results.pt chạy trên video ảnh gương

    Dữ liệu trả về mỗi sample:
        - SMPL params của real và mirror (đã un-mirror) để đưa vào fusion model
        - 2D Keypoints từ ViTPose của cả 2 nguồn (để tính reprojection loss)
        - Camera intrinsics K của cả 2 nguồn
    """

    def __init__(
        self,
        real_dir: str,
        mirror_dir: str,
        is_train: bool = True,
        train_ratio: float = 0.8,
        gt_dir: str = None,           # Thư mục chứa GT JSON files (optional)
        zero_wrist: bool = False,
        allow_truncate: bool = False, # Cho phép truncate nếu số frame lệch
        image_width: int | None = None,
        mirror_intrinsics_mode: str = "raw",
        beta_mode: str = "median",
        allow_missing_vitpose: bool = False,
        gt_person_id: int = 0,
    ):
        """
        Args:
            real_dir   : Thư mục chứa hmr4d_results.pt + preprocess/vitpose.pt của người thật.
            mirror_dir : Thư mục chứa hmr4d_results.pt + preprocess/vitpose.pt của ảnh gương.
            is_train   : True → tập train, False → tập val.
            train_ratio: Tỉ lệ phân chia train/val.
            gt_dir     : Thư mục chứa XXXXXX.json GT keypoints3D (optional).
                         Nếu None → dùng fallback height cố định.
        """
        super().__init__()

        if mirror_intrinsics_mode not in {"raw", "flipped"}:
            raise ValueError("mirror_intrinsics_mode must be 'raw' or 'flipped'")
        if beta_mode not in {"median", "first", "per_frame"}:
            raise ValueError("beta_mode must be 'median', 'first', or 'per_frame'")

        gt_dir = normalize_optional_dir(gt_dir)
        self.allow_missing_vitpose = allow_missing_vitpose

        # ── Load dữ liệu người thật ───────────────────────────────────────
        # Hỗ trợ truyền vào trực tiếp file .pt hoặc thư mục
        real_pt = real_dir if real_dir.endswith('.pt') else os.path.join(real_dir, "hmr4d_results.pt")
        real_vp_list = glob.glob(os.path.join(os.path.dirname(real_pt), "preprocess", "vitpose*.pt"))
        real_vp = real_vp_list[0] if real_vp_list else os.path.join(os.path.dirname(real_pt), "preprocess", "vitpose.pt")

        # ── Load dữ liệu ảnh gương ────────────────────────────────────────
        mirror_pt = mirror_dir if mirror_dir.endswith('.pt') else os.path.join(mirror_dir, "hmr4d_results.pt")
        mirror_vp_list = glob.glob(os.path.join(os.path.dirname(mirror_pt), "preprocess", "vitpose*.pt"))
        mirror_vp = mirror_vp_list[0] if mirror_vp_list else os.path.join(os.path.dirname(mirror_pt), "preprocess", "vitpose.pt")

        for path in [real_pt, mirror_pt]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Khong tim thay: {path}")

        print(f"Loading real data   : {real_pt}")
        real_data   = torch.load(real_pt,   map_location="cpu", weights_only=False)

        print(f"Loading mirror data : {mirror_pt}")
        mirror_data = torch.load(mirror_pt, map_location="cpu", weights_only=False)

        # ── SMPL params — người thật ──────────────────────────────────────
        # Dùng smpl_params_incam (pose trong không gian camera) làm nguồn chính
        self.real_global_orient = real_data["smpl_params_incam"]["global_orient"]  # (N, 3)
        self.real_body_pose     = real_data["smpl_params_incam"]["body_pose"]      # (N, 63)
        self.real_betas         = real_data["smpl_params_incam"]["betas"]          # (N, 10)
        self.real_transl        = real_data["smpl_params_incam"]["transl"]         # (N, 3)
        self.K_real             = real_data["K_fullimg"]                           # (N, 3, 3)

        if beta_mode != "per_frame":
            clip_betas = resolve_clip_betas(self.real_betas, beta_mode=beta_mode)
            self.real_betas = expand_clip_betas(clip_betas, self.real_betas.shape[0])

        # ── SMPL params — ảnh gương (đã un-mirror) ───────────────────────
        # Ghép global_orient + body_pose thành (N, 22, 3) để un-mirror dễ hơn
        raw_mirror_go   = mirror_data["smpl_params_incam"]["global_orient"]  # (N, 3)
        raw_mirror_bp   = mirror_data["smpl_params_incam"]["body_pose"]      # (N, 63)
        raw_mirror_t    = mirror_data["smpl_params_incam"]["transl"]         # (N, 3)

        # Ghép thành (N, 22, 3) để unmirror_pose xử lý một lần
        full_mirror_pose = torch.cat(
            [raw_mirror_go.unsqueeze(1), raw_mirror_bp.view(-1, 21, 3)],
            dim=1,
        )  # (N, 22, 3)

        # Áp dụng un-mirror: flip trục X + hoán đổi cặp khớp trái ↔ phải
        unmirrored_pose = unmirror_pose(full_mirror_pose)  # (N, 22, 3)

        # Tách lại thành global_orient và body_pose
        self.mirror_global_orient = unmirrored_pose[:, 0, :]       # (N, 3)
        self.mirror_body_pose     = unmirrored_pose[:, 1:, :].reshape(-1, 63)  # (N, 63)
        self.mirror_betas         = mirror_data["smpl_params_incam"]["betas"]  # (N, 10)

        if zero_wrist:
            self.real_body_pose[:, 57:63] = 0.0
            self.mirror_body_pose[:, 57:63] = 0.0
        self.mirror_transl        = unmirror_transl(raw_mirror_t)              # (N, 3)
        raw_K_mirror = mirror_data["K_fullimg"]
        if mirror_intrinsics_mode == "flipped":
            if image_width is None:
                raise ValueError("image_width is required when mirror_intrinsics_mode='flipped'")
            self.K_mirror = flip_camera_intrinsics(raw_K_mirror, image_width)
        else:
            # The training path remirrors the fused pose into the raw mirror frame.
            self.K_mirror = raw_K_mirror.clone()

        self.kp2d_real, self.kp2d_conf_real = self._load_vitpose(real_vp, "real")
        self.kp2d_mirror, self.kp2d_conf_mirror = self._load_vitpose(mirror_vp, "mirror")

        # ── Sequence Length Validation ──────────────────────────────────────
        n_real = self.real_global_orient.shape[0]
        n_mirror = self.mirror_global_orient.shape[0]
        n_vp_real = self.kp2d_real.shape[0]
        n_vp_mirror = self.kp2d_mirror.shape[0]
        
        min_frames = min(n_real, n_mirror, n_vp_real, n_vp_mirror)
        if not (n_real == n_mirror == n_vp_real == n_vp_mirror):
            msg = f"Frame count mismatch (Real:{n_real}, Mirror:{n_mirror}, VP-R:{n_vp_real}, VP-M:{n_vp_mirror})."
            if not allow_truncate:
                raise ValueError(msg + " Set allow_truncate=True to silently truncate to min_frames.")
            print(f"[Dataset] Warning: {msg} Truncating to {min_frames}.")
            
            self.real_global_orient = self.real_global_orient[:min_frames]
            self.real_body_pose = self.real_body_pose[:min_frames]
            self.real_betas = self.real_betas[:min_frames]
            self.real_transl = self.real_transl[:min_frames]
            self.K_real = self.K_real[:min_frames]
            
            self.mirror_global_orient = self.mirror_global_orient[:min_frames]
            self.mirror_body_pose = self.mirror_body_pose[:min_frames]
            self.mirror_betas = self.mirror_betas[:min_frames]
            self.mirror_transl = self.mirror_transl[:min_frames]
            self.K_mirror = self.K_mirror[:min_frames]
            
            self.kp2d_real = self.kp2d_real[:min_frames]
            self.kp2d_conf_real = self.kp2d_conf_real[:min_frames]
            self.kp2d_mirror = self.kp2d_mirror[:min_frames]
            self.kp2d_conf_mirror = self.kp2d_conf_mirror[:min_frames]

        # ── GT Height per-frame ────────────────────────────────────────────
        N = self.K_real.shape[0]
        if gt_dir is not None and os.path.isdir(gt_dir):
            self.gt_height_m = load_gt_heights_from_dir(gt_dir, N, person_id=gt_person_id)  # (N,)
        else:
            if gt_dir is not None:
                print(f"[Dataset] WARNING: gt_dir='{gt_dir}' không tồn tại. "
                      f"Dùng fallback height={FALLBACK_HEIGHT_M}m.")
            self.gt_height_m = torch.full((N,), FALLBACK_HEIGHT_M, dtype=torch.float32)

        # ── Phân chia Train / Val ─────────────────────────────────────────
        N = self.K_real.shape[0]
        split = int(N * train_ratio)
        self.indices = list(range(0, split)) if is_train else list(range(split, N))

        print(f"Dataset ready ({'Train' if is_train else 'Val'}): {len(self.indices)} frames")

    def _load_vitpose(self, path: str, source: str):
        """Load vitpose.pt → tọa độ (N,17,2) và confidence (N,17)."""
        N = self.K_real.shape[0]
        if os.path.exists(path):
            vp = torch.load(path, map_location="cpu", weights_only=False)  # (N, 17, 3): [x, y, conf]
            return vp[..., :2], vp[..., 2]
        if self.allow_missing_vitpose:
            print(f"Warning: {path} not found. Using dummy 2D keypoints ({source}).")
            return torch.zeros(N, 17, 2), torch.zeros(N, 17)
        raise FileNotFoundError(
            f"Missing ViTPose file for {source}: {path}. "
            "Add the file or set data.allow_missing_vitpose=true (reprojection loss will be 0)."
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict:
        """
        Trả về 1 sample tại vị trí `item`.

        Returns:
            dict:
                real_global_orient  : (3,)    — Rotation gốc người thật
                real_body_pose      : (63,)   — 21 khớp axis-angle người thật
                real_betas          : (10,)   — Shape params người thật
                real_transl         : (3,)    — Translation người thật
                mirror_global_orient: (3,)    — Rotation gốc gương (đã un-mirror)
                mirror_body_pose    : (63,)   — 21 khớp axis-angle gương (đã un-mirror)
                mirror_betas        : (10,)   — Shape params gương
                mirror_transl       : (3,)    — Translation gương (đã un-mirror)
                kp2d_real           : (17, 2) — 2D keypoints ViTPose người thật
                kp2d_conf_real      : (17,)   — Confidence người thật
                kp2d_mirror         : (17, 2) — 2D keypoints ViTPose ảnh gương
                kp2d_conf_mirror    : (17,)   — Confidence ảnh gương
                K_real              : (3, 3)  — Camera intrinsics người thật
                K_mirror            : (3, 3)  — Camera intrinsics ảnh gương
                gt_height_m         : ()      — Chiều cao GT của frame này (mét, scalar)
        """
        idx = self.indices[item]

        return {
            # Người thật
            "real_global_orient":   self.real_global_orient[idx],
            "real_body_pose":       self.real_body_pose[idx],
            "real_betas":           self.real_betas[idx],
            "real_transl":          self.real_transl[idx],
            # Ảnh gương (đã un-mirror)
            "mirror_global_orient": self.mirror_global_orient[idx],
            "mirror_body_pose":     self.mirror_body_pose[idx],
            "mirror_betas":         self.mirror_betas[idx],
            "mirror_transl":        self.mirror_transl[idx],
            # 2D keypoints
            "kp2d_real":            self.kp2d_real[idx],
            "kp2d_conf_real":       self.kp2d_conf_real[idx],
            "kp2d_mirror":          self.kp2d_mirror[idx],
            "kp2d_conf_mirror":     self.kp2d_conf_mirror[idx],
            # Camera
            "K_real":               self.K_real[idx],
            "K_mirror":             self.K_mirror[idx],
            # GT Height
            "gt_height_m":          self.gt_height_m[idx],   # scalar ()
        }
