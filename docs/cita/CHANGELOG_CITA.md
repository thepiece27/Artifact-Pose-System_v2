# CHANGELOG CITA 2027

Mỗi lần sửa sau `cita-freezeC` thì tăng tag `cita-exp-v1.N` và thêm một mục vào đây. Mỗi mục ghi: tag, ngày, mô tả lỗi, file đã sửa, ảnh hưởng tới kết quả đã có, cần chạy lại replay nào.

| Tag | Ngày | Nội dung |
|---|---|---|
| `cita-baseline` | 18/11/2026 | Commit `eaed45e` (trước mọi thay đổi CITA) |
| `cita-freezeA` | 18/11/2026 | Phase 0–1: fixture + `expected_baseline`, profile, logger, registry, session, GT bảng ChArUco, I2 trên thiết bị, `capture_session.py`, `e0_noise`/`e0_settle`/`e0_repeat` (kèm lõi pose Phase 2 mà `e0_noise` cần) |
| `cita-freezeB` | 18/11/2026 | Phase 2: M1–M5, hiệp phương sai, `replay_pose`/`eval_pose`/`make_tables`, PoseService chọn phương pháp theo profile |
| `cita-freezeC` | 18/11/2026 | Phase 3–4: bộ điều khiển (C-A…C-D), `AlignmentSession`, rig ảo, `run_reloc_trials`/`eval_reloc`, phát hiện thay đổi P0–P4, `AIInputPackage`, công cụ E3 |
| `cita-exp-v1.0` | 18/11/2026 | Phase 5–6: lock file, `CITA_PROFILE` trong compose, smoke server, `grid --with-stereo` (thiếu ảnh stereo trong phiên E1), cảnh báo cỡ mẫu khi chọn ngưỡng, tài liệu, REPORT_AGENT |
| `cita-exp-v1.1` | 18/11/2026 | Sửa lỗi khi rà soát lại đặc tả: (1) server lưu J ở `DATA_DIR/control`, trước đây ghi vào cây mã nguồn `server/data/control`; (2) `eval_pose`/`eval_reloc`/`eval_change`/`make_tables`/`define_roi` ghi bản ghi JSONL `cita.trial.v1`; (3) `eval_change` thêm đường độ nhạy theo `offset_deg`; (4) test chế độ `render` của rig ảo. Không ảnh hưởng kết quả pose/kiểm tra đã có; chỉ cần chạy lại `eval_change` nếu muốn có đường theo độ |
| `cita-exp-v1.2` | 18/11/2026 | Đợt sửa theo `AGENT_PROMPT_CITA2027_v1.2.md`. **Chưa có dữ liệu thật nào bị ảnh hưởng**: chưa có phiên thật nào được chụp hay chạy. Chi tiết ở mục "Đợt v1.2" bên dưới và trong REPORT_AGENT. |

## `cita-exp-v1.2` (18/11/2026)

**Ảnh hưởng tới dữ liệu.** Chưa có dữ liệu thật nào bị ảnh hưởng, vì đợt này diễn ra trước mọi phiên thật.
* Log dạng cũ vẫn đọc được, và `noise_model` v1 vẫn được nạp.
* File J dạng cũ (v1, tên không chứa nhánh) không còn được đọc. Lần đầu, mỗi nhánh sẽ thăm dò J lại.
* Sau khi có dữ liệu thật, chạy theo thứ tự trong EXPERIMENT_CODE_GUIDE:
  1. `e0_noise` (VCE) và `e0_stereo_check`;
  2. `replay_pose`, rồi `eval_pose`;
  3. `eval_reloc` với τ ghi rõ;
  4. `replay_change`, rồi `eval_change`.

| Mục | Commit | Nội dung | File chính |
|---|---|---|---|
| R1 | `c237219` | CI đỏ trên Linux. So sánh P0 với `expected_baseline` dùng dung sai đo được giữa hai nền: ssim 1e-5, `valid_mask_ratio` 1e-3, `region_ssim` 5e-3. Khóa, lớp, hộp và đếm vẫn so khớp tuyệt đối. CI cài `requirements-lock.txt` và in phiên bản thư viện. `expected_baseline` **không** bị ghi lại | `tests/cita/test_legacy_regression.py`, `.github/workflows/ci.yml` |
| R2 | `413ef9f` | σ_d và σ_o ước lượng **cùng một định nghĩa** (phần dư chiếu lại), bằng VCE Helmert/Förstner trên tranh hiệu chuẩn P000. Mặc định là `--estimator vce`. `noise_model` v2 có `golden_stereo_source`. Thêm ablation `M4-jitter` | `artifact_pose/noise_vce.py`, `methods.py`, `tools/experiments/e0_noise.py`, `configs/profiles/M4-jitter.yaml` |
| R3 | `bad872c` | Mỗi nhánh có file J riêng; J của nhánh khác bị từ chối. `probe.reuse` có ba chế độ `ttl` / `per_block` / `never`, và `block_id`. Workflow nhận `cita_probe_reuse` / `cita_block_id` | `control.py`, `alignment_loop.py`, `profiles.py`, `inspection_service.py`, `schemas/workflow.py` |
| R5 | `22467b5` | Thống kê E1: đơn vị là vị trí `(session, pose_index)`, chỉ số chính `e_t_first`, M5 có trong kiểm định cặp. Các so sánh định trước (M3−M1, M4−M3, M5−M4, M4−M2, M4−M4-jitter) hiệu chỉnh Holm. Bootstrap cụm hai tầng, `fallback_rate`. `make_tables` viết lại cho E1/E2/E3 | `stats.py`, `eval_pose.py`, `replay_pose.py`, `make_tables.py` |
| R4 | `c309d4b` | `verified_regions` = vùng SSIM có phát hiện YOLO trùng (IoU ≥ 0,25 hoặc tâm nằm trong vùng). Hàng P3v được chấm riêng với P3. YOLO không đổi | `pipelines.py`, `replay_change.py`, `eval_change.py` |
| R6 | `e65687c` | Thêm `golden.stereo_source = commanded_epipolar`: hướng baseline lấy từ ràng buộc epipolar tịnh tiến thuần, độ dài lấy theo lệnh. Thêm `e0_stereo_check.py` cùng quy tắc chọn nguồn | `stereo_direction.py`, `methods.py`, `tools/experiments/e0_stereo_check.py` |
| R7 | `defd0d8` | E2 có hai điều kiện lắp, `aligned` và `yawed25`, ở cả chế độ ảo lẫn thật (`--mount-condition` bắt buộc khi chạy thật). Chỉ số chính là `total_moves`. τ bắt buộc trong `eval_reloc`. Thêm `--pwm-step-deg`, `--noise-from`, và dọn J cũ (`*.stale`) | `sim/virtual_rig.py`, `run_reloc_trials.py`, `eval_reloc.py` |
| R8 | `8ecc00b` | `capture_session.py`: in vùng di chuyển, thời gian và dung lượng rồi đòi xác nhận `yes`. Kiểm tra đĩa (1,2×, đo lại sau khung đầu). Thêm lệnh con `awb` và `--awb-from` (thiếu AWB khi chạy thật thì báo lỗi), `--pulse-delay-us`. `release_servos` **chỉ nhả pan**, không bao giờ nhả tilt (cả `main_app`) | `embed/tools/capture_session.py`, `adapters/hardware_controller.py`, `adapters/camera_manager.py` |
| R9 | `5d0338c` | Tài liệu: dấu của `rotvec(ΔR)` (FRAMES_AND_UNITS §4); đính chính REPORT §5.5 (giới hạn là rơ và vùng chết, không phải lượng tử hóa); mục "Trước mỗi phiên trên rig"; mục changelog này | `docs/cita/*` |
| R10 | `bb7b9a2` | Test snapshot cho bảng E1/E2/E3 (`.md` + `.tex`, vi/en): đúng cột, thứ tự hàng P0, P1, P2, P3, P3v, P4, và ô "—". Sửa lỗi `--out` có dấu chấm (ví dụ `eval_v1.2` bị `with_suffix` cắt thành `eval_v1.json` và có thể ghi đè kết quả khác) ở `eval_pose`/`eval_reloc`/`eval_change`/`make_tables` | `tests/cita/test_tables_snapshot.py`, `tools/experiments/_common.py` |
| R5 (bổ sung) | `c27503c` | Phát hiện khi smoke: `fallback_rate` chỉ áp dụng cho phương pháp lai; M1/M2 hiện "—" thay vì "100%" | `eval_pose.py` |
| R7 (bổ sung) | `93ef8d3` | Phát hiện khi smoke: `eval_reloc` cảnh báo khi τ của bộ điều khiển lỏng hơn τ chấm (giá trị giữ chỗ 10 mm trong `C-*`) | `eval_reloc.py` |
