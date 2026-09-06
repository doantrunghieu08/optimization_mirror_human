import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- cần để đăng ký projection='3d'
import torch
import numpy as np
import cv2

# Định nghĩa các đoạn xương (Bones) nối các khớp của hệ 17 joints COCO
# Dùng để vẽ các đường nối tạo thành bộ xương
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Đầu / Mặt
    (5, 7), (7, 9),                  # Tay trái
    (6, 8), (8, 10),                 # Tay phải
    (5, 6),                          # Vai
    (5, 11), (6, 12),                # Thân trên
    (11, 12),                        # Hông
    (11, 13), (13, 15),              # Chân trái
    (12, 14), (14, 16),              # Chân phải
    (0, 17), (5, 17), (6, 17)        # Cổ nối với mũi và hai vai
]

def plot_3d_skeleton(joints_3d, title="3D Skeleton", ax=None, show=True):
    """
    Vẽ bộ xương 3D bằng Matplotlib.
    
    Args:
        joints_3d (np.ndarray or torch.Tensor): (17, 3) tọa độ x,y,z của 17 khớp
        title (str): Tiêu đề biểu đồ
        ax: Trục matplotlib (nếu muốn vẽ lên subplot có sẵn)
        show (bool): Nếu True sẽ gọi plt.show()
    """
    if isinstance(joints_3d, torch.Tensor):
        joints_3d = joints_3d.detach().cpu().numpy()
        
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
    plot_x = joints_3d[:, 0]
    plot_y = joints_3d[:, 2] # depth
    plot_z = joints_3d[:, 1] # up/down
    
    # Plot các khớp (màu xanh dương)
    ax.scatter(plot_x, plot_y, plot_z, c='b', marker='o', s=20)
    
    # Plot các đoạn xương (màu đỏ)
    for (i, j) in COCO_SKELETON:
        if i >= len(joints_3d) or j >= len(joints_3d):
            continue
        ax.plot([plot_x[i], plot_x[j]], [plot_y[i], plot_y[j]], [plot_z[i], plot_z[j]], c='r', linewidth=2)
        
    # Thiết lập góc nhìn và giới hạn trục (Giữ tỉ lệ 1:1:1 cho không bị méo người)
    max_range = np.array([plot_x.max()-plot_x.min(), plot_y.max()-plot_y.min(), plot_z.max()-plot_z.min()]).max() / 2.0
    mid_x = (plot_x.max()+plot_x.min()) * 0.5
    mid_y = (plot_y.max()+plot_y.min()) * 0.5
    mid_z = (plot_z.max()+plot_z.min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Z (Depth)')
    ax.set_zlabel('Y (Up)')
    ax.set_title(title)
    
    # Thiết lập góc nhìn trùng khớp với vị trí camera nhìn dọc theo trục Depth (Z)
    ax.view_init(elev=15, azim=-90)
    
    if show:
        plt.show()

def overlay_2d_skeleton(image, joints_2d, color=(0, 255, 0), thickness=2):
    """
    Vẽ 2D skeleton đè lên ảnh gốc.
    
    Args:
        image (np.ndarray): Ảnh BGR (ví dụ đọc bằng cv2.imread)
        joints_2d (np.ndarray): (17, 2) Tọa độ u, v trên pixel
        color: Màu của xương (mặc định xanh lục)
        
    Returns:
        np.ndarray: Ảnh đã được vẽ skeleton
    """
    if isinstance(joints_2d, torch.Tensor):
        joints_2d = joints_2d.detach().cpu().numpy()
        
    img_draw = image.copy()
    
    # Vẽ các khớp
    for (u, v) in joints_2d:
        cv2.circle(img_draw, (int(u), int(v)), 3, (0, 0, 255), -1) # Điểm khớp màu đỏ
        
    # Vẽ các xương
    for (i, j) in COCO_SKELETON:
        if i >= len(joints_2d) or j >= len(joints_2d):
            continue
        pt1 = (int(joints_2d[i, 0]), int(joints_2d[i, 1]))
        pt2 = (int(joints_2d[j, 0]), int(joints_2d[j, 1]))
        cv2.line(img_draw, pt1, pt2, color, thickness)
        
    return img_draw

def plot_3d_mesh(vertices, faces, title="3D Mesh", ax=None, show=True, center=True, flip_y=False):
    """
    Vẽ 3D mesh bằng Matplotlib plot_trisurf.
    
    Args:
        vertices: (N, 3) mesh vertices
        faces: (F, 3) triangle face indices
        title: tiêu đề biểu đồ
        ax: trục matplotlib
        show: nếu True sẽ gọi plt.show()
        center: nếu True sẽ center vertices về gốc toạ độ (cần thiết cho global view
                vì global_transl có thể rất lớn làm viewport bị sai scale)
        flip_y: nếu True sẽ negate trục Y trước khi plot — dùng cho incam view vì
                camera space có Y chỉ XUỐNG, cần đảo để người đứng thẳng khi hiển thị
    """
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()
        
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

    # Center vertices về gốc toạ độ để tránh viewport bị kéo giãn do global_transl lớn
    if center:
        centroid = vertices.mean(axis=0)
        vertices = vertices - centroid

    plot_x = vertices[:, 0]
    plot_y = vertices[:, 2]  # depth
    # Camera space (incam): Y trỏ XUỐNG → cần negate để người đứng thẳng
    # World space (global): Y trỏ LÊN → không cần negate
    plot_z = -vertices[:, 1] if flip_y else vertices[:, 1]

    if faces is not None and len(faces) > 0:
        # Dùng plot_trisurf để render mesh thật thay vì point cloud
        # Downsample faces nếu quá nhiều để tăng tốc độ render
        MAX_FACES = 3000
        if len(faces) > MAX_FACES:
            idx = np.random.choice(len(faces), MAX_FACES, replace=False)
            faces_vis = faces[idx]
        else:
            faces_vis = faces

        ax.plot_trisurf(
            plot_x, plot_y, plot_z,
            triangles=faces_vis,
            color='steelblue',
            alpha=0.85,
            linewidth=0,
            antialiased=True
        )
    else:
        # Fallback: point cloud nếu không có faces
        ax.scatter(plot_x, plot_y, plot_z, c='lightblue', s=0.5, alpha=0.6)
    
    max_range = np.array([
        plot_x.max() - plot_x.min(),
        plot_y.max() - plot_y.min(),
        plot_z.max() - plot_z.min()
    ]).max() / 2.0
    mid_x = (plot_x.max() + plot_x.min()) * 0.5
    mid_y = (plot_y.max() + plot_y.min()) * 0.5
    mid_z = (plot_z.max() + plot_z.min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Z (Depth)')
    ax.set_zlabel('Y (Up)')
    ax.set_title(title)
    
    # Thiết lập góc nhìn trùng khớp với vị trí camera nhìn dọc theo trục Depth (Z)
    ax.view_init(elev=15, azim=-90)
    
    if show:
        plt.show()
