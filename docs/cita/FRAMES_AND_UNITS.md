# Hệ tọa độ và đơn vị (CITA 2027)

> Tài liệu tham chiếu cho mọi code mới trong `server/app/experiments`, `modules/{ground_truth,change_detection,sim}`, `modules/artifact_pose/{methods,motion,covariance,markerless,control,alignment_loop}.py` và công cụ `server/tools/experiments`, `embed/tools`.
> Đã đối chiếu với code tại tag `cita-baseline` (eaed45e). Chỗ nào chưa xác minh được thì ghi rõ **CHƯA XÁC MINH**.

## 1. Đơn vị

| Ở đâu | Tịnh tiến | Góc | Ảnh |
|---|---|---|---|
| Bên trong thuật toán (code mới) | mét (`_m`) | radian (`_rad`) | pixel (`_px`) |
| Log JSONL, CLI, báo cáo, profile YAML | milimét (`_mm`) | độ (`_deg`) | pixel |
| Golden YAML (`golden_pose.yaml`) | mét | rvec (radian) | pixel có méo |
| Lệnh MQTT `move` | bước (`x_steps`, `z_steps`) + chiều (`x_dir`, `z_dir`) | độ (`yaw_delta`, `pitch_delta`) | — |

Quy đổi slider: `STEPS_PER_MM = 800` (vít me + TB6600). Baseline stereo cố định: `80000` bước = 100 mm (`embed/device_agent/runtime/main_app.py:89-90`).

Tên trường luôn mang hậu tố đơn vị (`c_mm`, `sigma_rot_deg`, `settle_s`, `backlash_steps`…).

## 2. Hệ Diamond = hệ world

`server/app/modules/artifact_pose/common.py:21-25`:

```
DIAMOND_OBJ_PTS = [[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]]   (h = 20 mm)
```

* Gốc: tâm ô vuông giữa của Diamond (ô 40 mm, marker 25 mm, DICT_4X4_50, ID 10/20/30/40).
* **X** sang phải, **Y** lên trên (góc trên-trái có Y = +h), **Z = X × Y** hướng ra khỏi mặt marker, về phía camera. Đây đúng quy ước ArUco/IPPE_SQUARE của OpenCV.
* Pose lưu và ước lượng là `T_cw = [R | t]`: biến tọa độ world → camera (`geometry.py:1`).

## 3. Hệ camera (OpenCV)

* **x** sang phải ảnh, **y** xuống dưới ảnh, **z** nhìn tới (trục quang).
* Khi camera nhìn chính diện vào tranh: `x_cam ≈ X_world`, `y_cam ≈ −Y_world`, `z_cam ≈ −Z_world`.
* Pixel quan sát là pixel **có méo**; `geometry.undistort` khử méo lặp (50 vòng) trước khi tối ưu.

## 4. Chuyển động tương đối chuẩn (Phụ lục A.1). Mọi code mới đều dùng

Cho pose camera←world của golden (g) và hiện tại (c):

```
ΔT = T_cw(c) · T_cw(g)⁻¹          # ánh xạ tọa độ camera golden → camera hiện tại
X_c = ΔR · X_g + Δt
c   = −ΔRᵀ · Δt                    # tâm camera hiện tại trong hệ camera golden (m)
θ   = rotvec(ΔR)                   # (rad; log bằng độ)
```

* Cài đặt: `app/modules/artifact_pose/motion.py` (`relative_motion`, `camera_center_in_golden`, `pose_errors`).
* M2 (không marker) cho ΔT trực tiếp vì bản đồ 3D nằm ở hệ camera golden (trái).
* GT (bảng ChArUco tham chiếu) dùng **đúng công thức trên** với pose bảng: `ΔT_gt = T_cb(c) · T_cb(g)⁻¹`. ΔT không phụ thuộc gốc/hướng của hệ bảng, nên bảng có thể đặt ở bất cứ đâu trên mặt phẳng tranh miễn là đứng yên.
* Sai số (A.2): `e_t = ‖c_est − c_gt‖` (mm), `e_R = arccos((tr(ΔR_est ΔR_gtᵀ) − 1)/2)` (độ), bias theo trục = trung bình `(c_est − c_gt)` trong hệ camera golden.

Quan hệ với trường cũ của `geometry.camera_deviation` (giữ nguyên cho app/MQTT):

| Trường cũ | Ý nghĩa chính xác |
|---|---|
| `translation_world_m` | `C_world(c) − C_world(g)`, tâm camera trong **hệ Diamond** |
| `translation_correction_world_m` = `delta_x/y/z` | `−translation_world_m` (hệ Diamond, **không** phải hệ slider) |
| `translation_magnitude` | `‖c‖` (bất biến với phép quay hệ) |
| `T_current_camera_desired_camera` | đúng bằng `ΔT` ở trên |
| `delta_pan` | góc "pitch" giữa của phân tích Euler ZYX của `ΔR` = quay quanh **trục y camera** |
| `delta_tilt` | góc "roll" = quay quanh **trục x camera** |
| `delta_roll` | góc "yaw" = quay quanh **trục quang z** |
| `rotation_magnitude` | `‖θ‖` (độ) |

## 5. Hệ rig (slider + servo)

| Trục | Phần cứng | Biến lệnh | Quy ước dấu trong code |
|---|---|---|---|
| X | slider vít me, stepper TB6600, 800 bước/mm | `x_steps ≥ 0`, `x_dir ∈ {+1,−1}` | `x_dir=+1` → GPIO DIR = HIGH (`hardware_controller.py:240`) |
| Z | slider vít me, stepper | `z_steps`, `z_dir` | `z_dir=+1` → DIR = HIGH |
| Pan (yaw) | servo MG90S, kênh PCA9685 0, 0–180° | `yaw_delta` (độ, cộng vào góc hiện tại) | góc servo tăng khi `yaw_delta > 0` |
| Tilt (pitch) | servo MG90S, kênh 1, 0–120° | `pitch_delta` | góc servo tăng khi `pitch_delta > 0` |

Home servo: Yaw = 110°, Pitch = 35° (`hardware_controller.py:39-40`). Odometry slider = bước tích lũy kể từ khi khởi động (chỉ ở RAM).

Ánh xạ cố định của PBL (`inspection_service.py:977-980`, nay là `control.FixedMappingController`):

```
x_steps_raw = delta_x[m] · 1000 · STEPS_PER_MM · SIGN_MOVE_X
z_steps_raw = delta_z[m] · 1000 · STEPS_PER_MM · SIGN_MOVE_Z
yaw_delta   = delta_pan[deg]  · SIGN_ROTATE_PAN
pitch_delta = delta_tilt[deg] · SIGN_ROTATE_TILT
```

Mặc định code: `SIGN_* = −1` (`core/config.py:290-293`). Lưu ý `docs/04` ghi mặc định là `1`, mâu thuẫn với code (code thắng).

Hệ quả **nếu** dấu mặc định đúng với rig thật (chưa có phép đo nào xác nhận):
* `x_dir = +1` dời camera theo **−X_world** (sang trái ảnh).
* `z_dir = +1` dời camera theo **−Z_world**, tức là **lại gần tranh**, nếu trục Z là trục chiều sâu.

### Trục Z của rig: **CHƯA XÁC MINH**

Hai nguồn trong repo mâu thuẫn nhau:
1. Phía server ánh xạ `delta_z` (thành phần **pháp tuyến** Diamond = chiều sâu) vào slider Z, tức là coi Z là trục **tiến–lùi**.
2. Phía thiết bị in nhãn `"UP(+)"/"DOWN(-)"` cho slider Z (`main_app.py:755`) và `_handle_slider_z` nhận `direction ∈ {"forward","up","positive"}` (`main_app.py:666-673`), tức là coi Z là trục **lên–xuống**.

Tài liệu `docs/01–07` không mô tả hướng lắp. **Nhóm cần đo trên rig** (E0.4: ra lệnh `+5 mm` trục Z, đọc ΔT từ bảng tham chiếu, xem thành phần lớn nhất của `c` là `y` hay `z`) rồi ghi kết quả vào đây. Các phần code mới không giả định điều này:
* `JacobianController` tự học ánh xạ (cột J), không cần biết trục.
* Rig ảo có tham số `z_axis: depth | vertical`.
* `FixedMappingController` giữ nguyên giả định cũ (Z = pháp tuyến Diamond) vì đó chính là baseline C-A.

### Tâm quay servo

Servo quay quanh trục của chính nó, **không** đi qua tâm quang. Ánh xạ cố định bỏ qua độ lệch này (một lỗi mô hình mà I3 xử lý). Rig ảo mô phỏng độ lệch 20–40 mm.

## 6. Hệ ảnh tham chiếu (kiểm tra hư hại)

* ROI tranh (`roi.json`) và mọi hộp GT/kết quả (`bbox_ref_xyxy`) nằm trong **hệ pixel của ảnh golden (tham chiếu)**.
* `H_ref_from_cur` ánh xạ pixel ảnh hiện tại → ảnh tham chiếu; ảnh "đã căn chỉnh" = `warpPerspective(cur, H_ref_from_cur)` nên cùng hệ với ảnh tham chiếu.
* Crop YOLO mang `bbox_src_xyxy` + `src ∈ {aligned, original}` để ánh xạ hộp kết quả về hệ tham chiếu (`change_detection/ai_input.py::map_detections_to_ref`).
* `px_per_mm` lấy từ `size_mm` của ROI, nếu không có thì từ cạnh Diamond 40 mm.
