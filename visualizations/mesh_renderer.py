"""
mesh_renderer.py
================
Render SMPL mesh với pyrender — smooth shading, lighting, offscreen rendering.
Output: numpy image (H, W, 3) BGR cho OpenCV.
"""

import os
import numpy as np
import cv2

# Pyrender cần backend offscreen trên môi trường không có display (server / headless)
# Trên Windows với display thì không cần set, nhưng set cũng không hại
if "PYOPENGL_PLATFORM" in os.environ and os.environ["PYOPENGL_PLATFORM"] == "":
    del os.environ["PYOPENGL_PLATFORM"]

try:
    import pyrender
    import trimesh
    PYRENDER_AVAILABLE = True
except ImportError:
    PYRENDER_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# Checkerboard floor mesh
# ─────────────────────────────────────────────────────────────
def _make_checkerboard_floor(
    floor_y: float = 0.0,
    size: float = 6.0,
    n_tiles: int = 12,
    color1=(0.85, 0.78, 0.65),   # be sáng / cát
    color2=(0.55, 0.48, 0.38),   # nâu đất
):
    """Tạo sàn checkerboard ở độ cao floor_y."""
    tile = size / n_tiles
    verts, faces, colors = [], [], []

    for i in range(n_tiles):
        for j in range(n_tiles):
            x0 = -size / 2 + i * tile
            z0 = -size / 2 + j * tile
            x1, z1 = x0 + tile, z0 + tile

            base = len(verts)
            verts += [
                [x0, floor_y, z0],
                [x1, floor_y, z0],
                [x1, floor_y, z1],
                [x0, floor_y, z1],
            ]
            faces += [
                [base, base + 1, base + 2],
                [base, base + 2, base + 3],
            ]
            c = color1 if (i + j) % 2 == 0 else color2
            colors += [c, c, c, c]

    mesh = trimesh.Trimesh(
        vertices=np.array(verts, dtype=np.float32),
        faces=np.array(faces, dtype=np.int32),
        vertex_colors=np.array(colors, dtype=np.float32),
        process=False,
    )
    return mesh


# ─────────────────────────────────────────────────────────────
# Main render function
# ─────────────────────────────────────────────────────────────
def render_smpl_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    img_width: int = 640,
    img_height: int = 480,
    flip_y: bool = False,
    add_floor: bool = True,
    body_color=(0.88, 0.85, 0.82, 1.0),   # RGBA nhạt xám ấm
    camera_distance: float = 2.5,
    camera_elevation: float = 15,           # độ ngẩng camera (degrees)
    camera_azimuth: float = -20,            # góc xoay ngang (degrees)
    fixed_target_height: float = None,
    renderer=None,
) -> np.ndarray:
    """
    Render SMPL mesh ra ảnh numpy (H, W, 3) BGR dùng pyrender.

    Args:
        vertices  : (N, 3) float32 — SMPL mesh vertices
        faces     : (F, 3) int32   — triangle faces
        img_width : chiều rộng ảnh output
        img_height: chiều cao ảnh output
        flip_y    : True nếu vertices ở camera space (Y xuống), cần lật
        add_floor : True để thêm sàn checkerboard
        body_color: màu RGBA của body mesh
        camera_distance: khoảng cách camera tới mesh
        camera_elevation: góc ngẩng camera (degrees, dương = nhìn xuống)
        camera_azimuth: góc xoay ngang (degrees)
        fixed_target_height: nếu truyền vào, dùng chiều cao cố định này để camera
            "look-at" thay vì tính động theo bounding box mỗi frame — tránh camera
            bị "giật/nảy" theo tư thế (vung tay, ngồi xổm...) làm thay đổi chiều cao
            khung hình giữa các frame liên tiếp.
        renderer: pyrender.OffscreenRenderer đã tạo sẵn để tái sử dụng qua nhiều
            frame (tránh chi phí khởi tạo/hủy OpenGL context mỗi lần gọi). Nếu
            None, hàm sẽ tự tạo và hủy renderer tạm thời như trước.

    Returns:
        np.ndarray (H, W, 3) BGR — ảnh đã render, hoặc blank nếu pyrender không có.
    """
    if not PYRENDER_AVAILABLE:
        # Fallback: trả về ảnh trắng với text thông báo
        img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 230
        cv2.putText(img, "pyrender not installed", (20, img_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
        return img

    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)

    # ── 1. Chuẩn hoá hệ trục ────────────────────────────────
    # SMPL / pyrender dùng Y-up, right-hand coordinate system.
    # Incam: Y xuống → negate Y và Z để chuyển sang Y-up, Z-forward (OpenGL)
    if flip_y:
        vertices[:, 1] = -vertices[:, 1]  # lật Y
        vertices[:, 2] = -vertices[:, 2]  # lật Z (depth → -depth, OpenGL Z)

    # Center theo XZ (nằm ngang)
    centroid_xz = np.array([vertices[:, 0].mean(), 0.0, vertices[:, 2].mean()])
    vertices -= centroid_xz

    # Đưa chân chạm đất tại Y = 0 để sàn luôn cố định
    foot_y = vertices[:, 1].min()
    vertices[:, 1] -= foot_y
    floor_y = 0.0

    # ── 2. Tạo body mesh ────────────────────────────────────
    body_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    _ = body_trimesh.vertex_normals  # force compute smooth normals

    body_mesh = pyrender.Mesh.from_trimesh(
        body_trimesh,
        material=pyrender.MetallicRoughnessMaterial(
            baseColorFactor=body_color,
            metallicFactor=0.0,
            roughnessFactor=0.6,
            alphaMode="OPAQUE",
        ),
        smooth=True,
    )

    # ── 3. Scene (nền tối để dễ nhìn sàn và người) ─────────
    scene = pyrender.Scene(
        bg_color=[0.12, 0.12, 0.15, 1.0],  # nền xám xanh tối
        ambient_light=[0.35, 0.35, 0.35],
    )
    scene.add(body_mesh)

    # ── 4. Sàn checkerboard ─────────────────────────────────
    if add_floor:
        floor_mesh_tm = _make_checkerboard_floor(floor_y=floor_y)
        floor_mesh = pyrender.Mesh.from_trimesh(floor_mesh_tm, smooth=False)
        scene.add(floor_mesh)

    # ── 5. Lights ───────────────────────────────────────────
    body_center_y = vertices[:, 1].mean()
    # Key light (mạnh)
    key_light = pyrender.DirectionalLight(color=[1.0, 0.97, 0.92], intensity=4.0)
    key_pose = _look_at_pose(
        eye=np.array([2.0, 4.0, 2.0]),
        target=np.array([0.0, body_center_y, 0.0]),
    )
    scene.add(key_light, pose=key_pose)

    # Fill light (mềm hơn)
    fill_light = pyrender.DirectionalLight(color=[0.70, 0.80, 1.0], intensity=2.0)
    fill_pose = _look_at_pose(
        eye=np.array([-2.0, 2.5, -1.5]),
        target=np.array([0.0, body_center_y, 0.0]),
    )
    scene.add(fill_light, pose=fill_pose)

    # ── 6. Camera ───────────────────────────────────────────
    az_rad = np.deg2rad(camera_azimuth)
    el_rad = np.deg2rad(camera_elevation)

    eye = camera_distance * np.array([
        np.sin(az_rad) * np.cos(el_rad),
        np.sin(el_rad),
        np.cos(az_rad) * np.cos(el_rad),
    ])
    
    # Camera nhìn vào giữa thân người
    body_height = vertices[:, 1].max()
    target_height = fixed_target_height if fixed_target_height is not None else body_height * 0.5
    target = np.array([0.0, target_height, 0.0])
    eye += target

    cam_pose = _look_at_pose(eye=eye, target=target)

    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(45.0), aspectRatio=img_width / img_height)
    scene.add(camera, pose=cam_pose)

    # ── 7. Offscreen render ─────────────────────────────────
    owns_renderer = renderer is None
    try:
        if owns_renderer:
            renderer = pyrender.OffscreenRenderer(viewport_width=img_width, viewport_height=img_height)
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    except Exception as e:
        raise RuntimeError("Global mesh rendering failed") from e
    finally:
        if owns_renderer and renderer is not None:
            renderer.delete()

    # RGBA → BGR cho OpenCV
    img_bgr = cv2.cvtColor(color[:, :, :3], cv2.COLOR_RGB2BGR)
    return img_bgr


# ─────────────────────────────────────────────────────────────
# Utility: tính camera pose matrix từ eye và target
# ─────────────────────────────────────────────────────────────
def _look_at_pose(eye: np.ndarray, target: np.ndarray, up: np.ndarray = None) -> np.ndarray:
    """
    Tạo 4x4 camera pose matrix (world-space) từ eye, target, up.
    OpenGL convention: camera nhìn theo -Z.
    """
    if up is None:
        up = np.array([0.0, 1.0, 0.0])

    forward = eye - target
    norm = np.linalg.norm(forward)
    if norm < 1e-6:
        forward = np.array([0.0, 0.0, 1.0])
    else:
        forward = forward / norm

    right = np.cross(up, forward)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        up = np.array([1.0, 0.0, 0.0])
        right = np.cross(up, forward)
        right_norm = np.linalg.norm(right)
    right = right / right_norm

    up_corrected = np.cross(forward, right)

    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right
    pose[:3, 1] = up_corrected
    pose[:3, 2] = forward
    pose[:3, 3] = eye
    return pose

# ─────────────────────────────────────────────────────────────
# Render overlay onto image
# ─────────────────────────────────────────────────────────────
def render_smpl_mesh_overlay(
    image: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    body_color=(0.88, 0.85, 0.82, 1.0),
    renderer=None,
) -> np.ndarray:
    """
    Render SMPL mesh overlaid on an image using camera intrinsics K.

    Args:
        renderer: pyrender.OffscreenRenderer đã tạo sẵn để tái sử dụng qua nhiều
            frame. Nếu None, hàm sẽ tự tạo và hủy renderer tạm thời như trước.
    """
    if not PYRENDER_AVAILABLE:
        # Fallback
        img = image.copy()
        h, w = img.shape[:2]
        cv2.putText(img, "pyrender not installed", (20, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return img

    h, w = image.shape[:2]
    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)
    
    # SMPL to pyrender coordinates (Y-down camera to Y-up OpenGL)
    vertices[:, 1] = -vertices[:, 1]
    vertices[:, 2] = -vertices[:, 2]

    body_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    _ = body_trimesh.vertex_normals  # force compute smooth normals
    
    body_mesh = pyrender.Mesh.from_trimesh(
        body_trimesh,
        material=pyrender.MetallicRoughnessMaterial(
            baseColorFactor=body_color,
            metallicFactor=0.0,
            roughnessFactor=0.6,
            alphaMode="OPAQUE",
        ),
        smooth=True,
    )

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.4, 0.4, 0.4])
    scene.add(body_mesh)

    # Lighting
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    # Light from top-front
    light_pose = np.eye(4)
    light_pose[:3, :3] = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.707, 0.707],
        [0.0, -0.707, 0.707]
    ])
    scene.add(light, pose=light_pose)

    # Camera
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
    
    # Camera pose is identity because vertices are already in camera space
    # but we negated Y and Z, so we just use identity
    cam_pose = np.eye(4)
    scene.add(camera, pose=cam_pose)

    owns_renderer = renderer is None
    try:
        if owns_renderer:
            renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    except Exception as e:
        raise RuntimeError("Overlay mesh rendering failed") from e
    finally:
        if owns_renderer and renderer is not None:
            renderer.delete()

    # Alpha composite
    color_bgr = cv2.cvtColor(color[:, :, :3], cv2.COLOR_RGB2BGR)
    alpha = color[:, :, 3:] / 255.0
    
    output = (color_bgr * alpha + image * (1 - alpha)).astype(np.uint8)
    return output
