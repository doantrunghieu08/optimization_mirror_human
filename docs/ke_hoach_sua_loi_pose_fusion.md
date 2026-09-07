# Kế hoạch sửa lỗi logic Pose Fusion

Tài liệu này chuyển kết quả review codebase thành checklist triển khai. Mục tiêu là cải thiện pose 3D và độ ổn định theo thời gian, không chỉ giảm reprojection error 2D.

## 1. Baseline hiện tại

Dữ liệu kiểm tra: `401` frame, `30 FPS`, video `720 × 1280`.

| Chỉ số | HMR4D gốc | Sau fusion |
|---|---:|---:|
| Reprojection 2D có trọng số | 27.45 px | 13.40 px |
| Rotation liên-frame trung bình | 1.93° | 3.81° |
| Rotation liên-frame p95 | 5.64° | 11.98° |

Các dấu hiệu bất thường khác:

- Pose sau fusion lệch HMR4D trung bình `19.72°/joint`, p95 `43.99°`, lớn nhất `94.58°`.
- `401/401` frame được cơ chế acceptance chấp nhận.
- Ray-casting đánh dấu vai, hông và gối của real view là occluded ở `100%` frame.
- Inter-view transform suy ra từ HMR4D thay đổi trung bình `0.98°` mỗi frame dù camera/gương cố định.

Baseline phải được lưu lại trước mỗi thay đổi để tránh tối ưu một chỉ số nhưng làm hỏng chỉ số khác.

## 2. Thứ tự sửa

Thực hiện từng phase độc lập và chạy ablation sau mỗi phase. Không sửa đồng thời tất cả loss vì sẽ không biết thay đổi nào thực sự có tác dụng.

### Phase 1 — Ngừng dùng joint-center ray-casting làm visibility

**Root cause**

`utils/mesh_raycast.py` bắn tia từ camera tới tâm khớp nằm bên trong mesh. Tia sẽ cắt bề mặt da trước khi đến vai, hông, gối…, kể cả khi landmark đó nhìn thấy trên ảnh. Vì vậy kết quả không phải visibility của keypoint 2D.

**Thay đổi tối thiểu**

1. Trong `utils/belief_fusion.py::compute_full_view_belief`, tạm thời không đưa `b_occ` vào belief dùng để fusion.
2. Giữ `b_occ` trong diagnostics để so sánh, nhưng không cho nó tác động đến optimizer.
3. Belief tạm thời chỉ dùng detector confidence và residual của observation gốc:

   ```text
   belief = b_det * b_reproj
   ```

4. Bỏ phép nhân lặp lại evidence. Hiện `b_det` và `b_reproj` đã tham gia Dempster–Shafer rồi lại bị nhân thêm lần nữa.

**Bản hoàn chỉnh, chỉ làm sau khi Phase 1 chứng minh cần thiết**

- Gắn mỗi COCO landmark với một tập vertex/surface region tương ứng.
- Ray-cast tới surface landmark thay vì anatomical joint center.
- Visibility nên là continuous score dựa trên depth buffer hoặc khoảng cách tới giao điểm đầu tiên, không phải nhị phân cứng.

**Test bắt buộc**

- Với người đứng chính diện, vai và hông không được occluded toàn bộ frame.
- Một joint nằm thật sự sau torso phải có visibility thấp hơn joint cùng bên nằm phía trước.
- `compute_full_view_belief` không double-count `b_det`/`b_reproj`.

### Phase 2 — Không ghép best pose độc lập theo từng frame

**Root cause**

`utils/pose_refit.py::keep_improvements` lưu best pose riêng cho mỗi frame tại các step 25, 50, 75… Output cuối có thể ghép các frame lấy từ những trạng thái optimizer khác nhau, phá vỡ temporal loss vừa tối ưu.

**Thay đổi tối thiểu**

1. Đánh giá và accept/reject toàn sequence sau mỗi outer iteration.
2. Candidate chỉ được nhận nếu:
   - mean reprojection của real không tăng;
   - mean reprojection của mirror không tăng;
   - temporal rotation metric không tăng quá ngưỡng;
   - tổng objective tốt hơn best sequence trước đó.
3. Không dùng `torch.where(accept[:, None], ...)` theo frame để tạo pose mosaic.
4. Nếu sequence dài cần xử lý cục bộ, dùng window liên tục có overlap; không accept từng frame rời rạc.

**Test bắt buộc**

- Tạo candidate A tốt ở frame chẵn và candidate B tốt ở frame lẻ. Acceptance không được ghép A/B thành sequence rung.
- Rotation liên-frame của output không được lớn hơn candidate sequence đã được optimizer đánh giá.
- Test phải thất bại với implementation cũ.

### Phase 3 — Ổn định biến đổi real → mirror

**Root cause**

`_to_mirror_frame` dùng cặp `real_go_reference` và `mirror_go_reference` theo từng frame. Transform camera vì thế chứa trực tiếp noise của hai dự đoán HMR4D và không còn là extrinsic cố định.

**Thay đổi tối thiểu**

1. Tính relative rotation của tất cả frame hợp lệ một lần trước optimization.
2. Căn dấu quaternion và lấy robust quaternion mean/median trên các frame có shoulder/hip confidence cao.
3. Dùng một relative rotation cố định cho toàn clip.
4. Ghi `inter_view_rotation_residual_deg` vào diagnostics; frame residual lớn chỉ được giảm trọng số, không được tạo transform riêng.
5. Không fuse trực tiếp hai `global_orient` thuộc hai camera khác nhau. Chỉ fuse sau khi đã chuyển về cùng frame.

**Sửa đúng hình học về lâu dài**

Calibrate camera và mặt phẳng gương để có transform từ body trong real camera sang virtual mirror camera:

```text
X_mirror_camera = R_rm · X_real_camera + t_rm
```

Khi đã có transform này, cả rotation và translation của candidate phải được chuyển bằng cùng một phép biến đổi. Không lấy `mirror_transl_raw` độc lập theo frame làm extrinsic.

**Test bắt buộc**

- Một candidate bằng real reference phải tái tạo mirror reference trong sai số số học cho dữ liệu synthetic.
- Relative transform cố định không đổi khi thêm noise nhỏ vào một frame reference.
- Gradient từ mirror reprojection phải quay về candidate real pose.

### Phase 4 — Cho reliability điều khiển đúng reprojection loss

**Root cause**

Reprojection loss hiện chỉ dùng ViTPose confidence. Belief chủ yếu tác động lên anchor và temporal gate, nên view/joint kém tin cậy vẫn kéo optimizer gần như bình thường.

**Thay đổi**

1. Tạo trọng số cố định từ observation ban đầu, không tính lại từ candidate đang tối ưu:

   ```text
   weight_real  = detector_conf_real  * source_quality_real
   weight_mirror = detector_conf_mirror * source_quality_mirror
   ```

2. `source_quality` không được lấy từ reprojection của chính candidate hiện tại vì sẽ tạo vòng lặp tự xác nhận.
3. Chuẩn hóa loss riêng từng view trước khi nhân `w_reproj_real`/`w_reproj_mirror`, tránh view có nhiều joint hợp lệ hơn lấn át view kia.
4. Khi cả hai view có trọng số thấp, giữ pose nguồn hoặc dùng temporal prior; không kéo về zero pose.

**Test bắt buộc**

- Real confidence bằng 0 và mirror confidence bằng 1 phải tạo gradient chỉ từ mirror.
- Đổi candidate pose không được làm thay đổi source reliability cố định.
- Hai view giống nhau phải cho nghiệm giống single-view, không tạo thêm rotation.

### Phase 5 — Dùng cùng body shape trong evidence và optimization

**Root cause**

Mirror belief được tính bằng `mirror_betas`, còn mirror reprojection trong optimizer dùng `betas` của real. Reliability vì thế được đo trên một mesh khác với mesh được tối ưu.

**Thay đổi**

1. Chọn một clip-level beta duy nhất cho cùng một người.
2. Mặc định dùng median của real; chỉ gộp mirror sau khi kiểm chứng beta mirror không có systematic bias do reflection/crop.
3. Dùng beta đã chọn cho:
   - real evidence;
   - mirror evidence;
   - cả hai reprojection loss;
   - render và evaluate.
4. Không dùng `per_frame` beta nếu input model thực sự trả shape dao động theo frame.

**Test bắt buộc**

- Mock SMPL ghi nhận cùng tensor beta ở cả evidence và optimization.
- Output beta phải cố định theo clip khi `beta_mode=median`.

### Phase 6 — Loại các anatomical heuristic không đúng biểu diễn rotation

**Root cause**

Các loss đang coi thành phần X/Y/Z của axis-angle như flexion, twist và adduction. Axis-angle không phải Euler angle; khi joint xoay nhiều trục, từng thành phần không còn mang ý nghĩa giải phẫu độc lập.

**Thay đổi tối thiểu**

Tắt trước bằng trọng số `0` và chạy ablation:

- `w_spine_twist`
- `w_hip_split`
- `w_leg_crossing`
- `w_hip_adduction`
- `w_shoulder_hyper`
- `w_wrist_bend`
- `w_penetration`
- `w_head_collision`

Giữ lại reprojection, source anchor và temporal loss. Chỉ bật lại một prior khi có test chứng minh nó giảm pose lỗi mà không phạt pose hợp lệ.

Nếu cần joint limit thật, tính relative rotation trong local joint frame rồi decomposite swing–twist theo trục giải phẫu của SMPL. Không giới hạn trực tiếp từng phần tử axis-angle.

**Test bắt buộc**

- Khoanh tay, bắt chéo chân, tay chạm đầu và ngồi xổm không bị mặc định coi là pose lỗi.
- Pose gối bẻ ngược và elbow hyperextension vẫn phải bị phạt.
- Một rotation vật lý tương đương nhưng có biểu diễn axis-angle khác không được tạo loss khác đáng kể.

### Phase 7 — Sửa hoặc bỏ skeleton penetration proxy

**Root cause**

`utils/penetration.py::_segment_segment_distance` xử lý đoạn song song bằng cách đo hai đầu mút đầu tiên và clamp hai tham số độc lập. Đây không phải khoảng cách ngắn nhất giữa hai đoạn.

**Khuyến nghị**

- Bỏ penetration proxy khỏi objective trước; nó đang phạt cả tiếp xúc hợp lệ.
- Chỉ sửa hàm segment distance nếu diagnostics chứng minh loss này còn cần thiết.
- Nếu cần collision thật, dùng body-part capsule hoặc mesh self-intersection với danh sách cặp bộ phận được phép tiếp xúc.

**Test bắt buộc nếu giữ lại**

- Hai đoạn song song chồng nhau phải có distance bằng `0`.
- Hai đoạn có closest point tại endpoint phải trả đúng khoảng cách.
- Gradient phải hữu hạn quanh trường hợp gần song song.

### Phase 8 — Sửa evaluator và visualization

1. Trong `render_video.py`, biến `mirror_2d_np` hiện chứa `kp2d_real`. Đổi tên thành `detected_real_2d` và sửa label; hoặc thực sự lưu/project mirror data nếu muốn panel dual-view.
2. `audit_fusion.py` phải báo cả:
   - real-view reprojection;
   - mirror-view reprojection;
   - rotation velocity/acceleration;
   - pose change so với hai source;
   - tỷ lệ joint/frame bị low-confidence.
3. Khi có GT 3D, luôn báo cả MPJPE và PA-MPJPE. Không dùng riêng PA-MPJPE vì Procrustes có thể che lỗi scale/orientation.
4. Kiểm tra `image_width` từ video/K thay vì tin giá trị `1920` trong config. Video hiện tại rộng `720` pixel.

## 3. Bộ metric nghiệm thu

Một phiên bản chỉ được coi là tốt hơn khi đồng thời đạt các điều kiện sau trên cùng clip:

| Metric | Điều kiện |
|---|---|
| Real reprojection | Không tệ hơn baseline HMR4D |
| Mirror reprojection | Không tệ hơn baseline mirror sau cùng transform |
| Rotation velocity mean/p95 | Không cao hơn HMR4D quá 5% |
| Rotation acceleration mean/p95 | Giảm hoặc không tăng |
| Pose change so với source đáng tin | Không có spike vô căn cứ > 45° |
| Invalid depth | Không tăng |
| MPJPE/PA-MPJPE | Phải cải thiện nếu có GT 3D |

Không dùng “video nhìn có vẻ đẹp hơn” hoặc reprojection 2D đơn lẻ làm tiêu chí nghiệm thu.

## 4. Ma trận ablation tối thiểu

Chạy cùng seed và cùng input:

| Run | Thay đổi |
|---|---|
| A | HMR4D gốc |
| B | Pipeline hiện tại |
| C | Bỏ ray-cast khỏi belief |
| D | C + sequence-level acceptance |
| E | D + fixed inter-view transform |
| F | E + reliability-weighted reprojection |
| G | F + tắt heuristic priors |

Chỉ giữ thay đổi nếu nó cải thiện bộ metric nghiệm thu. Không tiếp tục cộng loss để chữa triệu chứng của phase trước.

## 5. Test suite cần bổ sung

Tạo một file mới `tests/test_pose_quality_regressions.py` với các nhóm test sau:

1. Surface visibility không gán mọi joint nội bộ thành occluded.
2. Acceptance giữ nguyên sequence continuity.
3. Fixed real→mirror transform trên synthetic camera pair.
4. Belief/reliability thực sự thay đổi gradient của từng view.
5. Cùng beta được dùng xuyên suốt pipeline.
6. End-to-end mini-sequence: fusion không làm rotation jitter cao hơn input.
7. Nếu có GT fixture nhỏ: MPJPE fusion không tệ hơn HMR4D.

Lệnh kiểm tra:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) '.runtime-deps')
& 'D:/miniconda/envs/gvhmr/python.exe' -m unittest discover -s tests -v
```

## 6. Definition of Done

- Không còn joint quan trọng bị ray-cast occluded 100% chỉ vì nằm trong mesh.
- Không còn ghép best pose theo từng frame từ các optimizer checkpoint khác nhau.
- Inter-view transform là clip-level/calibrated transform, không phụ thuộc prediction từng frame.
- Reliability của view tham gia trực tiếp và cố định trong reprojection objective.
- Evidence, optimization, render và evaluate dùng cùng body shape policy.
- Jitter của fusion không cao hơn HMR4D trong metric mean và p95.
- Có ít nhất một test end-to-end bắt được lỗi “2D tốt hơn nhưng temporal/3D xấu hơn”.
- Báo cáo ablation cho thấy đóng góp riêng của từng thay đổi.

## 7. Phạm vi chưa cần làm

- Không thêm neural network hoặc training loop mới trước khi pipeline hình học hiện tại đúng.
- Không thêm nhiều pose prior mới để bù cho transform/visibility sai.
- Không tối ưu betas/translation tự do nếu chưa có ràng buộc camera và scale đáng tin.
- Không refactor API; lỗi chất lượng hiện nằm trong fusion/refit, không nằm ở FastAPI.
