# 3D Pose Fusion Project

Fusion pose 3D từ view thật + view gương bằng ray-casting occlusion detection
trên mesh SMPL, belief theory (Dempster-Shafer) và tối ưu pose kiểu SMPLify.
Xem [báo cáo kiểm tra fusion](docs/fusion_review.md) về lỗi hình học và cách kiểm chứng.

## Cài đặt môi trường
```bash
pip install -r requirements.txt
```

GVHMR xuất tham số **SMPL-X**, nên config mặc định dùng `models/SMPLX_NEUTRAL.npz`.
Model này cần đi cùng `models/smplx_coco17_J_regressor.pt` (17 × 10475), tính bằng
`smpl_coco17_J_regressor @ smplx2smpl_sparse.to_dense()` từ các tài nguyên GVHMR.
Các model đã có trong workspace này. Khi chuyển máy, sử dụng model SMPL-X đã tải
theo giấy phép của bạn; không đưa tham số GVHMR trực tiếp vào `SMPL_NEUTRAL.pkl`.

## Chạy suy luận (Inference)
Không còn bước huấn luyện — pipeline tối ưu trực tiếp trên tham số SMPL cho
từng video (self-supervised, không cần checkpoint):
```bash
python inference.py --config configs/default.yaml
python inference.py --config configs/default.yaml --bypass_refit   # debug: dùng thẳng pose real gốc
```
Kết quả: `output/inference_results.pkl` (SMPL params đã fuse, dùng cho
`render_video.py` / `evaluate.py`) và `output/belief_diagnostics.pkl`
(belief theo khớp/frame/nguồn evidence, phục vụ debug).

Kiểm tra sai số chiếu 2D so với ViTPose, không cần ground truth 3D:
```bash
python audit_fusion.py --results output/inference_results.pkl
python render_video.py --config configs/default.yaml
python -m unittest discover -s tests -v
```

`--bypass_refit` giữ nguyên pose và shape real. Refit dùng detector confidence cho
loss 2D và belief của hai nguồn cho pose anchor; mặc định chỉ nhận frame làm giảm
sai số 2D của cả hai view. `body_model_type` trong PKL giữ đúng loại model khi render
và đánh giá. Các PKL cũ không có trường này được coi là SMPL.

Trong lần kiểm chứng trên máy này, các thư viện thiếu được đặt riêng ở `.runtime-deps`.
Nếu dùng môi trường GVHMR hiện có, chạy PowerShell với:
```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) '.runtime-deps')
& 'D:/miniconda/envs/gvhmr/python.exe' inference.py --config configs/default.yaml
```
