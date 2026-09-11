# Nguồn Tham Khảo Lý Thuyết Sinh Lý Học Trong Mô Hình

## Dự Án: Optimization Mirror Human (3D Pose & Mesh Fusion)

---

## Tổng Quan

Mô hình sử dụng **5 nhóm lý thuyết sinh lý học** chính, mỗi nhóm có nguồn tham khảo rõ ràng từ y văn lâm sàng, papers kỹ thuật và sách giáo khoa giải phẫu học.

---

## 1. Giới Hạn Biên Độ Vận Động Khớp (Range of Motion — ROM)

> **Nguồn chính**: Tài liệu lâm sàng tiêu chuẩn AAOS

### 📖 Sách / Tài liệu Lâm sàng

| Tên tài liệu                                                                  | Nhà xuất bản / Tổ chức                     |            Năm            |
| :------------------------------------------------------------------------------- | :---------------------------------------------- | :-------------------------: |
| **"Joint Motion: Method of Measuring and Recording"**                      | American Academy of Orthopaedic Surgeons (AAOS) | 1965, tái bản nhiều lần |
| **"The Clinical Measurement of Joint Motion"**                             | American Academy of Orthopaedic Surgeons (AAOS) |            1994            |
| **"Measurement of Joint Motion: A Guide to Goniometry"** — Norkin & White | F.A. Davis Company                              |       2009 (4th ed.)       |

### Áp dụng cụ thể trong code ([losses/prior_losses.py](../losses/prior_losses.py))

```python
# JOINT_ROM_LIMITS — trích dẫn trực tiếp từ AAOS normal ROM
JOINT_ROM_LIMITS = {
    0: math.radians(120), 1: math.radians(120),   # Hip flexion max 120°
    3: math.radians(150), 4: math.radians(150),   # Knee flexion max 150°
    6: math.radians(50), 7: math.radians(50),     # Ankle plantarflexion max 50°
    15: math.radians(180), 16: math.radians(180), # Shoulder flexion/abduction max 180°
    17: math.radians(150), 18: math.radians(150), # Elbow flexion max 150°
    19: math.radians(90), 20: math.radians(90),   # Wrist flexion/extension max 90°
}
```

Từng giới hạn cụ thể và nguồn AAOS tương ứng:

| Khớp                                 | Giới hạn dùng trong code |          Chuẩn AAOS Normal ROM          | Hàm áp dụng                                                         |
| :------------------------------------ | :-------------------------: | :---------------------------------------: | :--------------------------------------------------------------------- |
| **Hip flexion** (Háng gập)    |            120°            |                 0–120°                 | `compute_joint_angle_limit_loss`                                     |
| **Hip abduction** (Háng dạng) |            90°            | 0–45° (bình thường); ~90° (athlete) | `compute_hip_split_loss`                                             |
| **Hip adduction** (Háng khép) |            25°            |                  0–30°                  | `compute_hip_adduction_loss`                                         |
| **Knee flexion** (Gối gập)    |            150°            |                 0–135°                 | `compute_anatomical_limits_loss`, `compute_joint_angle_limit_loss` |
| **Knee extension** (Gối duỗi) |         0° (cận)         |         Không duỗi quá thẳng         | `compute_anatomical_limits_loss`                                     |
| **Ankle plantarflexion**        |            50°            |                  0–50°                  | `compute_joint_angle_limit_loss`                                     |
| **Shoulder flexion/abduction**  |            180°            |                 0–180°                 | `compute_shoulder_hyperextension_loss`                               |
| **Elbow flexion**               |            150°            |                 0–150°                 | `compute_elbow_limits_loss`                                          |
| **Wrist flexion/extension**     |            90°            |    Flexion ~80°, Extension ~70–90°    | `compute_wrist_bend_loss`                                            |

> **Ghi chú:** Comment tại dòng 691–695 của `losses/prior_losses.py` ghi rõ: *"Giá trị lấy theo giới hạn sinh lý TỐI ĐA (không phải giá trị 'bình thường' trung bình) — chuẩn goniometry lâm sàng AAOS"*

---

## 2. Sinh Lý Học Cột Sống (Spine Biomechanics)

> **Nguồn chính**: Sách sinh cơ học cột sống và y văn chỉnh hình

### 📖 Sách / Papers

| Tên tài liệu                                                 | Tác giả                                    |    Năm    |
| :-------------------------------------------------------------- | :------------------------------------------- | :---------: |
| **"Clinical Anatomy of the Spine, Spinal Cord, and ANS"** | Cramer & Darby                               |    2014    |
| **"Clinical Biomechanics of the Spine"**                  | White & Panjabi                              |    1990    |
| **"Thoracolumbar Spine Biomechanics"**                    | Nhiều tác giả, J. of Bone & Joint Surgery | nhiều năm |

### Áp dụng trong code (`losses/prior_losses.py` — `compute_spine_twist_loss`)

Hàm `compute_spine_twist_loss` phạt xoắn vặn (axial rotation) cột sống ngực-thắt lưng:

```python
def compute_spine_twist_loss(body_pose, max_twist_rad=math.radians(15), max_total_twist_rad=math.radians(25)):
    """
    Tổng xoay trục (axial rotation) sinh lý tối đa của cột sống ngực-thắt lưng
    (thoracolumbar) vào khoảng 45-50° — nhưng đó là mức đồng thời TỐI ĐA ở MỌI
    đốt cùng lúc, hiếm khi xảy ra trong chuyển động thực.
    """
```

| Giới hạn                          | Giá trị áp dụng | Cơ sở sinh lý                               |
| :---------------------------------- | :-----------------: | :--------------------------------------------- |
| Twist mỗi đốt sống (Spine1/2/3) |     15°/đốt     | Thoracolumbar rotation phân bổ đều 3 đốt |
| Tổng twist 3 đốt                 |        25°        | Chống xoắn thân trên phi tự nhiên        |
| Spine flexion (gập người)        |     30°/đốt     | 90° tổng chia đều 3 đốt                  |

---

## 3. Mô Hình Cơ Thể Tham Số SMPL (Parametric Body Model)

> **Nguồn chính**: Paper gốc SIGGRAPH 2015

### 📄 Paper Gốc

> **Loper, M., Mahmood, N., Romero, J., Pons-Moll, G., & Black, M. J. (2015).**
> *"SMPL: A Skinned Multi-Person Linear Model."*
> **ACM Transactions on Graphics (Proc. SIGGRAPH Asia), 34(6), 248:1–248:16.**
> DOI: https://doi.org/10.1145/2816795.2818013

### Mô hình toán học

$$
M(\boldsymbol{\beta}, \boldsymbol{\theta}, \mathbf{\gamma}) = W\left(\mathbf{T}_p(\boldsymbol{\beta}, \boldsymbol{\theta}), J(\boldsymbol{\beta}), \boldsymbol{\theta}, \mathbf{W}\right) + \mathbf{\gamma}
$$

- **β** (Shape Betas, 10 chiều): Hình thể cơ thể — chiều cao, cân nặng, tỷ lệ chi thể
- **θ** (Pose Parameters, 72 chiều): 23 khớp × 3 axis-angle + 1 khớp gốc
- **γ** (Global Translation): Vị trí 3D trong không gian

### SMPL-X (phiên bản mở rộng) — dùng trong dự án

> **Pavlakos, G., Choutas, V., Ghorbani, N., Bolkart, T., Osman, A. A. A., Tzionas, D., & Black, M. J. (2019).**
> *"Expressive Body Capture: 3D Hands, Face, and Body from a Single Image."*
> **CVPR 2019.**

Dự án dùng `models/SMPLX_NEUTRAL.npz` (10475 đỉnh, mở rộng thêm ngón tay và khuôn mặt).

---

## 4. Sinh Lý Học Chuyển Động Người (Human Motion Biomechanics)

> **Nguồn chính**: Papers về temporal smoothness trong motion capture / pose estimation

### 4.1. Temporal Smoothness — Joint Acceleration Loss

Nguyên lý: *"Chuyển động tự nhiên của con người mượt mà và có gia tốc thấp"* — từ nghiên cứu biomechanics vận động người.

> **Ionescu, C., Papava, D., Olaru, V., & Sminchisescu, C. (2014).**
> *"Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments."*
> **IEEE TPAMI 36(7).**

> **Mehta, D., et al. (2017).**
> *"Vnect: Real-time 3D Human Pose Estimation with a Single RGB Camera."*
> **SIGGRAPH 2017.**

### 4.2. Kiểm Soát Xương Đối Xứng (Bone Symmetry)

> Nguyên lý giải phẫu: Xương tay/chân trái và phải ở người khỏe mạnh có chiều dài gần bằng nhau — từ giải phẫu học người cơ bản.

> **Moore, K. L., Dalley, A. F., & Agur, A. M. R. (2013).**
> *"Clinically Oriented Anatomy"* (7th edition). Lippincott Williams & Wilkins.

### 4.3. Bone Length Stability

> Nguyên lý vật lý cứng: *"Cơ thể người là vật cứng — chiều dài tay/chân không thay đổi từ frame này sang frame khác"* — từ biomechanics rigid body.

Áp dụng tại `compute_temporal_smoothness_loss`, `compute_bone_symmetry_loss`, `compute_bone_length_stability_loss`.

---

## 5. Phòng Chống Va Chạm Cơ Thể (Body Penetration / Self-Intersection)

> **Nguồn chính**: Papers về SMPLify và body fitting

### 📄 Papers Tham Khảo Trực Tiếp

> **Bogo, F., Kanazawa, A., Lassner, C., Gehler, P., Romero, J., & Black, M. J. (2016).**
> *"Keep It SMPL: Automatic Estimation of 3D Human Pose and Shape from a Single Image."*
> **ECCV 2016.** *(Bài báo gốc của **SMPLify** — nền tảng của toàn bộ pipeline optimization trong dự án)*

Dự án kế thừa phương pháp SMPLify-style iterative optimization:

- **Pose Prior**: L2 regularization trên axis-angle
- **Shape Prior**: Penalty trên β Betas
- **Reprojection Loss**: Khớp 3D chiếu về 2D phải khớp với keypoints detected

> **Lassner, C., Romero, J., Kiefel, M., Bogo, F., Black, M. J., & Gehler, P. V. (2017).**
> *"Unite the People: Closing the Loop Between 3D and 2D Human Representations."*
> **CVPR 2017.**

---

## 6. Lý Thuyết Toán Học Hỗ Trợ (Mathematical Frameworks)

> Không trực tiếp là sinh lý học, nhưng là công cụ để *mô hình hóa* và *thực thi* các ràng buộc sinh lý học.

### 6.1. Lý Thuyết Niềm Tin Dempster-Shafer (DST)

> **Dempster, A. P. (1967).** *"Upper and lower probabilities induced by a multivalued mapping."* Annals of Mathematical Statistics, 38(2), 325–339.

> **Shafer, G. (1976).** *"A Mathematical Theory of Evidence."* Princeton University Press.

Dùng để: Kết hợp bằng chứng hình học (ray-casting occlusion) với bằng chứng mềm (ViTPose confidence) để xác định view nào đáng tin cậy hơn — áp dụng tại [`utils/belief_fusion.py`](../utils/belief_fusion.py).

### 6.2. Phép Biến Đổi Householder (Gương)

> **Householder, A. S. (1958).** *"Unitary Triangularization of a Nonsymmetric Matrix."* Journal of the ACM, 5(4), 339–342.

Dùng để: Phản xạ axis-angle dưới gương bất kỳ có pháp tuyến **n** tùy ý — áp dụng tại [`utils/mirror_geometry.py`](../utils/mirror_geometry.py):

$$
R_{\text{mirror}} = I - 2\mathbf{n}\mathbf{n}^T
$$

### 6.3. SLERP trên SO(3) (Quaternion Interpolation)

> **Shoemake, K. (1985).** *"Animating Rotation with Quaternion Curves."* SIGGRAPH 1985 Proceedings.

Dùng để: Hòa trộn rotation từ real view và mirror view theo belief weight, tránh gimbal lock — áp dụng tại [`utils/belief_fusion.py`](../utils/belief_fusion.py).

### 6.4. Nhân Geman-McClure (Robust Loss)

> **Geman, S., & McClure, D. E. (1987).** *"Statistical Methods for Tomographic Image Reconstruction."* Bulletin ISI.

Dùng để: Robust reprojection loss chống outlier 2D keypoints bị nhiễu:

$$
b_{\text{reproj}} = \frac{\sigma^2}{\|\mathbf{x}_{2D}^{\text{proj}} - \mathbf{x}_{2D}^{\text{det}}\|^2 + \sigma^2}
$$

---

## 7. Tóm Tắt Bảng Nguồn → Code Mapping

| Lý thuyết sinh lý học                        | Nguồn tham khảo                          | File code                                          | Hàm                                                           |
| :----------------------------------------------- | :----------------------------------------- | :------------------------------------------------- | :------------------------------------------------------------- |
| Hip ROM (120° flexion, 90° abduction)          | AAOS Clinical ROM                          | [prior_losses.py](../losses/prior_losses.py)        | `compute_hip_split_loss`, `compute_joint_angle_limit_loss` |
| Knee flexion max 150°, chống gối gập ngược | AAOS Clinical ROM                          | [prior_losses.py](../losses/prior_losses.py)        | `compute_anatomical_limits_loss`                             |
| Ankle ROM 50°                                   | AAOS Clinical ROM                          | [prior_losses.py](../losses/prior_losses.py)        | `compute_joint_angle_limit_loss`                             |
| Shoulder flexion max 180°                       | AAOS Clinical ROM (Goniometry)             | [prior_losses.py](../losses/prior_losses.py)        | `compute_shoulder_hyperextension_loss`                       |
| Elbow flexion max 150°                          | AAOS Clinical ROM                          | [prior_losses.py](../losses/prior_losses.py)        | `compute_elbow_limits_loss`                                  |
| Wrist flexion/extension max 90°                 | AAOS Clinical ROM                          | [prior_losses.py](../losses/prior_losses.py)        | `compute_wrist_bend_loss`                                    |
| Thoracolumbar axial rotation                     | White & Panjabi (Spine Biomechanics)       | [prior_losses.py](../losses/prior_losses.py)        | `compute_spine_twist_loss`                                   |
| Xương đối xứng trái/phải                  | Moore et al. (Clinically Oriented Anatomy) | [prior_losses.py](../losses/prior_losses.py)        | `compute_bone_symmetry_loss`                                 |
| Rigid body — xương không đổi chiều dài   | Biomechanics rigid body                    | [prior_losses.py](../losses/prior_losses.py)        | `compute_bone_length_stability_loss`                         |
| Chuyển động mượt mà (gia tốc thấp)       | Human Motion Biomechanics                  | [prior_losses.py](../losses/prior_losses.py)        | `compute_temporal_smoothness_loss`                           |
| Mô hình SMPL cơ thể tham số                 | Loper et al. SIGGRAPH 2015                 | [models/](../models/)                               | Toàn bộ pipeline                                             |
| SMPLify-style optimization                       | Bogo et al. ECCV 2016                      | [utils/pose_refit.py](../utils/pose_refit.py)       | `run_pose_refit`                                             |
| Dempster-Shafer Belief Fusion                    | Shafer 1976; Dempster 1967                 | [utils/belief_fusion.py](../utils/belief_fusion.py) | `fuse_beliefs`                                               |

---

## 8. Tài Liệu Tham Khảo Đầy Đủ

### Sách Lâm Sàng / Giải Phẫu

1. **American Academy of Orthopaedic Surgeons (AAOS).** (1994). *The Clinical Measurement of Joint Motion.* American Academy of Orthopaedic Surgeons.
2. **Norkin, C. C., & White, D. J.** (2009). *Measurement of Joint Motion: A Guide to Goniometry* (4th ed.). F.A. Davis.
3. **Moore, K. L., Dalley, A. F., & Agur, A. M. R.** (2013). *Clinically Oriented Anatomy* (7th ed.). Lippincott Williams & Wilkins.
4. **White, A. A., & Panjabi, M. M.** (1990). *Clinical Biomechanics of the Spine* (2nd ed.). Lippincott.
5. **Cramer, G. D., & Darby, S. A.** (2014). *Clinical Anatomy of the Spine, Spinal Cord, and ANS* (3rd ed.). Elsevier.

### Papers Kỹ Thuật Máy Tính (Computer Vision / Graphics)

6. **Loper, M., Mahmood, N., Romero, J., Pons-Moll, G., & Black, M. J.** (2015). SMPL: A Skinned Multi-Person Linear Model. *ACM Trans. Graphics (Proc. SIGGRAPH Asia)*, 34(6), 248.
7. **Bogo, F., Kanazawa, A., Lassner, C., Gehler, P., Romero, J., & Black, M. J.** (2016). Keep It SMPL: Automatic Estimation of 3D Human Pose and Shape from a Single Image. *ECCV 2016*.
8. **Pavlakos, G., Choutas, V., Ghorbani, N., Bolkart, T., Osman, A. A. A., Tzionas, D., & Black, M. J.** (2019). Expressive Body Capture: 3D Hands, Face, and Body from a Single Image. *CVPR 2019*.
9. **Kanazawa, A., Black, M. J., Jacobs, D. W., & Malik, J.** (2018). End-to-end Recovery of Human Shape and Pose. *CVPR 2018*. *(HMR — tiền thân của HMR4D)*
10. **Ionescu, C., Papava, D., Olaru, V., & Sminchisescu, C.** (2014). Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments. *IEEE TPAMI*, 36(7).
11. **Mehta, D., et al.** (2017). Vnect: Real-time 3D Human Pose Estimation with a Single RGB Camera. *SIGGRAPH 2017*.
12. **Lassner, C., Romero, J., Kiefel, M., Bogo, F., Black, M. J., & Gehler, P. V.** (2017). Unite the People: Closing the Loop Between 3D and 2D Human Representations. *CVPR 2017*.

### Toán Học / Lý Thuyết

13. **Dempster, A. P.** (1967). Upper and lower probabilities induced by a multivalued mapping. *Annals of Mathematical Statistics*, 38(2), 325–339.
14. **Shafer, G.** (1976). *A Mathematical Theory of Evidence.* Princeton University Press.
15. **Shoemake, K.** (1985). Animating Rotation with Quaternion Curves. *Proc. SIGGRAPH '85*, 245–254.
16. **Householder, A. S.** (1958). Unitary Triangularization of a Nonsymmetric Matrix. *Journal of the ACM*, 5(4), 339–342.
17. **Geman, S., & McClure, D. E.** (1987). Statistical Methods for Tomographic Image Reconstruction. *Bulletin ISI*, 52(4).
