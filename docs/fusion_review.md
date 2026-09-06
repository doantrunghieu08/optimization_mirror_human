# Kiểm tra fusion — 2026-09-05

> **Snapshot lịch sử.** Các đường dẫn `runs/...`, `output/fusion_fix_backup/...` và số
> frame bên dưới chỉ thuộc lần chạy đã kiểm tra; không dùng tài liệu này làm lệnh vận hành
> hiện tại. Hãy dùng `configs/default.yaml` và `audit_fusion.py` để tái tạo báo cáo mới.

Đã kiểm tra hai file GVHMR và ViTPose trong `inputs/`, cùng kết quả fused cũ trong
`runs/9818710789f4/output/inference_results.pkl` (đã sao lưu vào
`output/fusion_fix_backup/pre_fix_inference_results.pkl`). Clip có 362 frame. Không có ground truth 3D trong
config; các số đo dưới đây là khoảng cách chiếu 2D tới ViTPose, tính bằng pixel.

## Nguyên nhân đã xác nhận

1. **Sai body model và điểm mốc.** GVHMR xuất tham số SMPL-X. Pipeline cũ đưa
   `body_pose`, `betas`, `global_orient`, `transl` vào SMPL, rồi chọn tâm khớp SMPL
   để so với COCO. Cùng pose real gốc, sai số 2D có trọng số confidence là
   **61,98 px** theo cách cũ và **21,78 px** khi dùng đúng SMPL-X cùng regressor
   COCO của GVHMR. Kết quả fused cũ, dựng đúng theo model SMPL đã dùng lúc tạo,
   có sai số **78,03 px**. Riêng gối trái/phải với confidence ≥ 0,5 là
   **122,45 / 133,31 px**.
2. **Trộn hệ tọa độ camera.** `global_orient` real và mirror thuộc hai góc nhìn
   khác nhau. Trong diagnostics cũ, `go_quality` trung bình khoảng 0,001 nhưng
   `bp_quality` khoảng 0,99. Code vẫn dùng root mirror đã lật trục X để blend,
   thay root và phạt toàn bộ tín hiệu mirror khi hai root bất đồng. Hướng nhìn
   khác nhau không phải bằng chứng các khớp cục bộ sai.
3. **Thiếu hoán đổi trái/phải của evidence.** Pose mirror đã hoán đổi các khớp,
   nhưng belief và confidence mirror vẫn dùng nhãn gương gốc để fuse. Độ tin cậy
   cổ tay trái của ảnh gương vì thế bị gán vào cổ tay trái của người thật.
4. **Blend gương hai lần.** `fuse_pose_precision_weighted` đã blend pose, sau đó
   `apply_mirror_correction` lại blend tiếp. Khi evidence hai phía bằng nhau,
   điều này đẩy tỷ lệ về gần 25% real / 75% mirror.
5. **Loss có thể thắng tín hiệu ảnh.** Đo trên 24 frame đầu, proxy xuyên thân
   đóng góp khoảng `5 × 0,09369 = 0,46845`, trong khi reprojection real chỉ
   khoảng `0,13555`. Đây là proxy khoảng cách giữa xương, chưa phải kiểm tra
   xuyên mesh chính xác. Nó có thể ép sai động tác tay/chân hợp lệ.
6. **Kết quả sau tối ưu không được kiểm tra.** Pipeline cũ nhận vòng Adam cuối
   rồi tiếp tục làm mượt root và body mà không so lại sai số quan sát. Đồng thời
   root camera được ghi thẳng vào trường root world, làm hai hệ tọa độ lẫn nhau.

## Thay đổi

- Dùng `SMPLX_NEUTRAL.npz` và regressor COCO (17 × 10475). Regresor được tạo từ
  `smpl_coco17_J_regressor @ smplx2smpl_sparse.to_dense()` trong bản GVHMR có sẵn
  tại `D:/GVHMR-Demo`. Model được sao chép từ tài nguyên đã có trên máy.
- Giữ shape real theo mặc định; `--bypass_refit` giữ chính xác pose và shape gốc.
- Local body pose chỉ dùng phản xạ theo trục đối xứng của body và hoán đổi L/R.
  Với helper gương tổng quát, root dùng `S_normal @ R @ S_body_X`, thay vì phản
  xạ mọi góc cục bộ bằng pháp tuyến trong hệ camera.
- Trong refit, suy ra phép đổi hướng camera từ cặp root gốc và giữ cố định phép
  đổi này khi truyền cập nhật root. Root real không được blend với root thuộc
  view khác. Phép đổi camera-to-world gốc cũng được giữ khi xuất root world.
- Belief nguồn lấy từ mesh/pose gốc riêng của từng view. Evidence mirror được
  đổi L/R trước khi làm pose anchor; mỗi khớp chỉ blend một lần.
- Loss 2D dùng confidence detector độc lập. Mesh nhìn thấy một khớp không đủ
  để coi detector hay pose đó là đúng; ray-casting là evidence có bất định vì
  các khớp xương còn nằm dưới bề mặt da.
- Giảm `w_penetration` từ 5,0 xuống 0,25. Temporal loss vẫn nằm trong refit;
  tắt bước smooth bổ sung ngoài objective theo mặc định.
- Lưu ứng viên tốt nhất mỗi 25 bước. Chỉ nhận frame nếu sai số có trọng số
  confidence của **cả hai view** không tăng và tổng sai số giảm. Nếu không có
  ứng viên đạt, giữ pose real gốc. Đây là bảo vệ theo phép đo 2D, không phải
  cam kết sai số 3D giảm ở từng khớp.
- Chia skinning theo lô 32 frame và checkpoint gradient để chạy cả clip trên
  GPU 4 GB, vẫn tính các temporal loss trên toàn bộ chuỗi.
- PKL mới có `body_model_type`; render và evaluate đọc đúng model. PKL cũ thiếu
  trường này vẫn được dựng bằng SMPL. Màu vàng ở panel 2 được ghi là evidence
  từ gương; nó không khẳng định khớp đã được sửa đúng.

## Tái lập

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) '.runtime-deps')
& 'D:/miniconda/envs/gvhmr/python.exe' inference.py --config output/fusion_fixed/config.yaml
& 'D:/miniconda/envs/gvhmr/python.exe' audit_fusion.py --results output/fusion_fix_backup/pre_fix_inference_results.pkl output/fusion_fixed/inference_results.pkl --output output/fusion_fixed/audit.json
& 'D:/miniconda/envs/gvhmr/python.exe' render_video.py --config output/fusion_fixed/config.yaml
& 'D:/miniconda/envs/gvhmr/python.exe' -m unittest discover -s tests -v
```

Kết quả kiểm chứng mới nằm riêng ở `output/fusion_fixed/`; bản fused cũ được giữ
trong job và bản sao lưu nêu trên. Các bản sao mã trước sửa nằm trong
`output/fusion_fix_backup/`. Các thư viện thiếu được cài riêng vào
`.runtime-deps`, không đổi các package trong môi trường GVHMR có sẵn.

## Giới hạn và nguồn

Sai số 2D giảm không chứng minh chiều sâu 3D đã đúng. Hai view vẫn phụ thuộc độ
chính xác của detector, đồng bộ frame và root/camera GVHMR. Phép đổi hướng camera
ước lượng từ root gốc chưa phải hiệu chuẩn mặt phẳng gương có ground truth.
Betas và translation được giữ cố định; các chuyển động bị khuất ở cả hai view
vẫn có thể mơ hồ. Cần GT 3D hoặc camera/gương đã hiệu chuẩn để kiểm chứng MPJPE.

- [GVHMR demo: dựng SMPL-X rồi chuyển mesh để render](https://github.com/zju3dv/GVHMR/blob/main/tools/demo/demo.py).
- [GVHMR setup: model đầu ra là SMPL-X](https://github.com/zju3dv/GVHMR/blob/main/docs/INSTALL.md).
- [GVHMR body-model configuration](https://github.com/zju3dv/GVHMR/blob/main/hmr4d/utils/smplx_utils.py).
