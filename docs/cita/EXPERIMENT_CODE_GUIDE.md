# Hướng dẫn chạy thí nghiệm CITA 2027 bằng code

> Áp dụng cho nhánh `cita/experiment-ready` (tag `cita-freezeC` trở đi). Mọi lệnh phía server chạy trong thư mục `server/`; mọi lệnh phía Pi chạy trong thư mục `embed/`.
> Định nghĩa hệ tọa độ/đơn vị: [FRAMES_AND_UNITS.md](FRAMES_AND_UNITS.md). Sơ đồ code: [CODE_MAP.md](CODE_MAP.md).

## 0. Chuẩn bị chung

### 0.1 Môi trường server (máy phân tích)

```bash
cd server
python3.11 -m venv .venv311 && . .venv311/bin/activate          # Windows: .venv311\Scripts\activate
pip install -r requirements-dev.txt                             # đã gồm pyyaml, scikit-image, opencv<5
# Tái lập chính xác: pip install -r requirements-lock.txt
pytest -q                                                       # phải xanh trước khi chạy thí nghiệm
```

* **OpenCV phải < 5.** Code pose gốc gọi `cv2.undistortPointsIter`, hàm này đã bị bỏ ở OpenCV 5.
* YOLO thật: `pip install -r requirements-ai.txt`, rồi truyền đường dẫn file `.pt` của nhóm bằng `--model`.

### 0.2 Profile

| Mục đích | Profile |
|---|---|
| Hành vi cũ (mặc định server) | `pbl_legacy` |
| Pose E1 | `M1`, `M2`, `M2-full`, `M3`, `M4`, `M5` |
| Điều khiển E2 | `C-A`, `C-B`, `C-C`, `C-D` |
| Kiểm tra E3 | `P0` … `P4` |
| Đề xuất đầy đủ | `proposed` |

* Server: `CITA_PROFILE=<tên>` (mặc định `pbl_legacy`). Docker: biến `CITA_PROFILE` trong `.env.docker`.
* Mọi CLI: `--profile <tên hoặc đường dẫn .yaml>`, `--seed`, `--log-root`, `--run-id`.
* Profile tùy biến (ablation) đặt ở bất kỳ đâu, ví dụ `my.yaml` gồm `extends: M4` và các khóa cần ghi đè. Mọi khóa sai tên hay sai kiểu đều bị từ chối.
* `profile_hash` (SHA-256 của cấu hình đã hợp nhất, gồm cả giá trị mặc định) được ghi vào **mọi** bản ghi log.

### 0.3 Log

* `logs/experiments/<exp>/<run_id>.jsonl`, schema `cita.trial.v1`: commit, cờ dirty, profile/hash, seed, phiên bản thư viện, input (session, frame, sha256), output, thời gian.
* File log không bao giờ bị ghi đè: chạy lại thì dùng `--run-id` khác.
* Các công cụ đánh giá/bảng/ROI (`eval_*`, `make_tables`, `define_roi`) cũng ghi một bản ghi `record_type: "tool"`, gồm file đầu vào, sha256, argv và đường dẫn kết quả.
* `logs/`, `results/` và `data/control/` đã nằm trong `.gitignore`.

### 0.4 Bảng tham chiếu ChArUco (GT độc lập)

* Mặc định: 5×7 ô, ô 30 mm, marker 22 mm, `DICT_5X5_100`, ID bắt đầu từ 0 (không trùng ID Diamond 10/20/30/40). Sửa trong profile (`ground_truth.board`) nếu in khác.
* In bảng: dùng `cv2.aruco.CharucoBoard((5,7),0.03,0.022,cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)).generateImage(...)`. **Đo lại cạnh ô thật bằng thước kẹp** rồi ghi vào profile.
* Dán bảng phẳng, cố định trên cùng mặt phẳng với tranh (formex), trong khung hình ở mọi vị trí của lưới E1/E2. Code tự loại vùng bảng khỏi đặc trưng của mọi phương pháp (`board_mask`).

### 0.5 Công cụ chụp trên Pi

```bash
cd embed
python tools/capture_session.py --exp E0.2 --dry-run burst --n 5      # thử, không động phần cứng
```

* Khi không có `--dry-run`, phải thêm `--i-know-hardware-moves` thì tool mới cho chạy rig thật.
* Ảnh được chụp với camera **thủ công**: `LensPosition=1.5`, AE/AWB tắt, phơi sáng cố định (`--exposure-us`, `--gain`, `--awb r,b`), không chạy dò lấy nét tự động.
* Giới hạn an toàn: `--max-travel-mm 120`, `--max-servo-offset-deg 15`, cộng với giới hạn sẵn có của `HardwareController`.
* Tham số I2 (không bật = hành vi cũ): `--float-angles --settle-s 1.0 --unidirectional --backlash-deg 2 --backlash-steps 0 [--servo-release]`.
* Giá trị âm trong danh sách được hỗ trợ: `--x -40,-20,0,20,40`.
* Chép phiên sang server (không ghi đè):

```bash
rsync -av --ignore-existing embed/device_agent/data/sessions/ user@server:<repo>/server/data/sessions/
```

---

## 1. E0 – Đặc trưng hệ thống

### E0.1 Calibration

* Không đổi `camera_params_lens_1.5.yaml`. Mỗi manifest và mỗi bản ghi log đều mang `calibration_sha256`, nên nếu calibration bị đổi thì sẽ phát hiện được.
* Golden cũ bị từ chối khi fingerprint calibration khác (kiểm tra này đã có sẵn).

### E0.2 / E0.5 – Nhiễu góc Diamond σ_d và chất lượng GT (bảng)

```bash
# Pi: mỗi khoảng cách một phiên tĩnh 60 khung
python tools/capture_session.py --exp E0.2 --painting-id P001 --distance-cm 65 --lighting L0 \
    --settle-s 2 --i-know-hardware-moves burst --n 60 --first-is-golden
```

### E0.3 – Mô hình nhiễu σ_d, σ_o (VCE)

Chụp trên **tranh hiệu chuẩn riêng `P000`**. Tranh này **không** dùng trong E1. Chụp một lưới nhỏ để phần dư có đa dạng:

```bash
# Pi
python tools/capture_session.py --exp E0.3 --painting-id P000 --distance-cm 65 --settle-s <t_s> \
    --i-know-hardware-moves grid --x -20,0,20 --z 0 --pan -2,0,2 --frames 3 --golden-frames 3 --with-stereo --return-home
```

```bash
# Server: noise model chính (VCE, mặc định). M4/M5 bắt buộc có file này.
python tools/experiments/e0_noise.py --golden-session data/sessions/E0.3/E0.3_P000_* \
    [--burst data/sessions/E0.2/*] --out configs/noise_model.yaml
# Ablation M4-jitter (σ đo bằng độ rung khung tĩnh, như v1.1):
python tools/experiments/e0_noise.py --estimator jitter --golden-session data/sessions/E0.3/E0.3_P000_* \
    --out configs/noise_model_jitter.yaml
```

* **VCE** (ước lượng thành phần phương sai, Helmert/Förstner rút gọn):
  * σ_d và σ_o là thang của phần dư sau khi fit **cùng một** bài toán pose lai. Vì vậy cả hai đều gồm sai số hệ thống: lỗi in, méo còn sót, sai số landmark tam giác hóa.
  * Bắt đầu từ 0,5 / 1,5 px, dừng khi cả hai σ đổi < 1% (tối đa 20 vòng).
  * Có cảnh báo nếu tổng độ dư của Diamond < 20, khi đó cần chụp thêm khung.
* **Độ rung khung tĩnh** chỉ đo nhiễu tạm thời. Giá trị này được giữ trong file làm thông tin phụ (`diamond_jitter_px`, `orb_keypoint_jitter_px`), **không** dùng làm trọng số.
  * Trên dữ liệu mô phỏng: lệch hệ thống 0,3 px + nhiễu 0,03 px. VCE ra σ_d ≈ 0,34 px, σ_o ≈ 0,93 px (tỉ lệ trọng số khoảng 7:1), còn độ rung ra σ_d ≈ 0,026 px (tỉ lệ khoảng 1300:1, gần mức PBL 10⁴:1).
* **Nguồn golden.**
  * σ_o phụ thuộc cách dựng golden. File ghi `golden_stereo_source`, và M4/M5 **từ chối** file có nguồn khác `golden.stereo_source` của profile.
  * Khi chạy ablation với nguồn golden khác, tạo thêm noise model: `e0_noise.py --stereo-source commanded_epipolar --out configs/noise_model_epipolar.yaml`, rồi truyền `--noise-model` tương ứng cho `replay_pose`.
* File có thêm `pose_sigma_center_mm_median` / `pose_sigma_rot_deg_median` (σ tâm camera của một khung tại nghiệm VCE). Các giá trị này dùng cho `run_reloc_trials --noise-from`.

Kết quả:
* `configs/noise_model.yaml` (schema `cita.noise_model.v2`): σ_d, σ_o, lịch sử VCE (σ theo vòng, độ dư, số khung, số phần dư), độ rung, jitter của bảng (E0.5), nguồn session kèm sha256, hash nội dung.
* `configs/noise_model.report.json`: chi tiết theo từng phiên.

Muốn ghi đè file `noise_model.yaml` đã có thì phải thêm `--force`; bản cũ được giữ lại thành `.bak`.

### E0.3b – Chọn nguồn baseline cho golden (R6)

Chạy trên chính các phiên E0.3 (có stereo), **trước khi đánh giá E1**:

```bash
python tools/experiments/e0_stereo_check.py --sessions data/sessions/E0.3/E0.3_P000_*
# -> results/E0/stereo_check.json + quy tắc + nguồn được khuyến nghị
```

Báo cáo theo từng phiên:
* baseline ước lượng từ Diamond so với 100 mm ra lệnh (mm, %);
* góc giữa ba hướng: Diamond, epipolar (ORB, thuần tịnh tiến), trục x camera;
* số inlier epipolar.

**Quy tắc quyết định:**
* Nếu trung vị |baseline Diamond − 100 mm| > 2 mm **hoặc** trung vị góc (Diamond, epipolar) > 1° thì `proposed`, `M4`, `M5` dùng `golden.stereo_source: commanded_epipolar`.
* Ngược lại giữ `diamond`.
* Ghi quyết định (kèm số liệu) vào `CHANGELOG_CITA.md` **trước** khi chạy eval E1. Sau đó tạo noise model với đúng nguồn đó: `e0_noise.py --stereo-source <nguồn>`.

`commanded_epipolar`:
* R = I, vì giữa hai ảnh golden servo đứng yên.
* Hướng t̂ ước lượng bằng RANSAC 2 điểm trên ràng buộc t·(x₁ × x₂) = 0, tinh chỉnh bằng SVD trên inlier; dấu chọn theo cheirality.
* Độ dài = baseline ra lệnh.

Trên dữ liệu tổng hợp với slider lệch 5° so với trục x camera:

| Nguồn | Sai hướng | M4 e_t trung vị |
|---|---|---|
| `commanded_epipolar` | 0,27° | 0,81 mm |
| `commanded` | 5° | 2,39 mm |
| `diamond` | 3,45° (baseline 92,3 mm) | 2,07 mm |

### E0.4 – Thời gian chờ hết rung t_s, độ lặp lại, rơ, độ phân giải servo

```bash
# Pi: settle (đo t_s) cho từng trục
python tools/capture_session.py --exp E0.4 --i-know-hardware-moves settle --axis pan --delta 2 --n-frames 30 --trials 20 --alternate
python tools/capture_session.py --exp E0.4 --i-know-hardware-moves settle --axis x --delta 10 --n-frames 30 --trials 20 --alternate
```

```bash
# Pi: lặp lại / rơ. Chạy TRƯỚC (không cờ) và SAU I2 (--float-angles --unidirectional --backlash-deg 2 --settle-s <t_s>)
python tools/capture_session.py --exp E0.4 --session-id E04_rep_pan_before --i-know-hardware-moves --settle-s 1.5 \
    repeat --axis pan --levels 1,3 --reps 10 --both-sides
python tools/capture_session.py --exp E0.4 --session-id E04_rep_pan_after --i-know-hardware-moves --settle-s 1.5 \
    --float-angles --unidirectional --backlash-deg 2 repeat --axis pan --levels 1,3 --reps 10 --both-sides
```

```bash
# Pi: độ phân giải servo (đáp ứng theo lệnh nhỏ)
python tools/capture_session.py --exp E0.4 --session-id E04_res_pan --i-know-hardware-moves --settle-s 1.5 --float-angles \
    grid --x 0 --z 0 --pan -1,-0.5,-0.25,0,0.25,0.5,1 --frames 3 --golden-frames 5
```

```bash
# Server
python tools/experiments/e0_settle.py --session data/sessions/E0.4/<settle> --out results/E0.4/settle_pan.json
python tools/experiments/e0_repeat.py --session data/sessions/E0.4/E04_rep_pan_before --out results/E0.4/rep_before.json
python tools/experiments/e0_repeat.py --session data/sessions/E0.4/E04_res_pan --mode response --axis pan --out results/E0.4/res_pan.json
```

* Lúc đo E0.4, **xác định luôn trục Z** (xem FRAMES_AND_UNITS §5): ra lệnh `settle --axis z --delta 5`. Trong `e0_repeat --mode response --axis z`, nếu thành phần `cy_mm` lớn nhất thì Z là trục lên–xuống; nếu `cz_mm` lớn nhất thì Z là trục chiều sâu.
* `--measure board` (mặc định) dùng bảng tham chiếu. Chỉ dùng `--measure image_shift` cho dữ liệu dry-run.
* Sau E0.4: điền `settle_s`, `backlash_deg` vào `C-B.yaml`, `C-D.yaml`, `proposed.yaml`. Chốt `tol_translation_mm` / `tol_rotation_deg` (giống nhau cho cả 4 file C-*), và `control.probe.delta` lớn hơn vùng chết servo.

---

## 2. E1 – Độ chính xác pose (offline)

### Chụp (mỗi tranh × ánh sáng × khoảng cách một phiên)

```bash
python tools/capture_session.py --exp E1 --painting-id P003 --lighting L0 --distance-cm 65 --seed 2027 \
    --settle-s <t_s> --float-angles --unidirectional --backlash-deg 2 --i-know-hardware-moves \
    grid --x -40,-20,-10,-5,0,5,10,20,40 --z -20,0,20 --pan -4,0,4 --frames 3 --golden-frames 3 --with-stereo --return-home
```

Thứ tự duyệt lưới được xáo theo seed và ghi trong `manifest.grid.order`.

### Replay và đánh giá

```bash
python tools/experiments/replay_pose.py --session data/sessions/E1/E1_P003_L0_<...> \
    --methods M1,M2,M2-full,M3,M4,M5 --run-id E1_P003_L0
# ablation nguồn golden (cả ba nguồn; noise model phải được ước lượng với CÙNG nguồn):
python tools/experiments/replay_pose.py --session ... --methods M3,M4,M5 --override golden.stereo_source=diamond \
    --noise-model configs/noise_model_diamond.yaml --run-id E1_P003_L0_gdiamond
python tools/experiments/replay_pose.py --session ... --methods M3,M4,M5 --override golden.stereo_source=commanded \
    --noise-model configs/noise_model_commanded.yaml --run-id E1_P003_L0_gcmd
python tools/experiments/replay_pose.py --session ... --methods M3,M4,M5 --override golden.stereo_source=commanded_epipolar \
    --noise-model configs/noise_model_epipolar.yaml --run-id E1_P003_L0_gepi
# ablation trọng số (σ đo bằng độ rung) và tắt Huber:
python tools/experiments/replay_pose.py --session ... --methods M4-jitter --run-id E1_P003_L0_jitter
python tools/experiments/replay_pose.py --session ... --methods M4 --override pose.huber=false --run-id E1_P003_L0_nohuber

python tools/experiments/eval_pose.py --logs "logs/experiments/E1/E1_*.jsonl" --out results/E1/eval
python tools/experiments/make_tables.py --eval results/E1/eval.json --out results/tables/E1
```

* Arm `GT` trong log là pose của bảng cho từng khung. Một khung không có GT bị loại khỏi phần độ chính xác (số khung bị loại được báo riêng).
* `eval.json` gồm:
  * `failure_rate`;
  * `e_t_mm` / `e_R_deg`: trung vị, RMSE, p95, CI 95% bootstrap;
  * `bias_mm` theo trục của hệ camera golden;
  * `coverage_2sigma` (M4/M5: kiểm tra hiệp phương sai);
  * `paired_tests` (Friedman + Wilcoxon-Holm);
  * `odometry_scale_board_over_command`.
* `eval_units.csv`: một dòng cho mỗi (khung, phương pháp).

---

## 3. E2 – Định vị lại (điều khiển)

### Rig ảo trước (chỉnh λ, δ, τ)

```bash
python tools/experiments/run_reloc_trials.py --virtual --paintings P001,P002 --n-perturb 15 --arms C-A,C-B,C-C,C-D --seed 2027
python tools/experiments/run_reloc_trials.py --virtual ... --rig-params '{"slider_misalign_deg": 15, "pivot_offset_mm": [0, 10, -40]}'
python tools/experiments/eval_reloc.py --logs logs/experiments/E2/<run>.jsonl --out results/E2/virtual_eval
```

C-A/C-B trên rig ảo được cấp dấu `SIGN_*` đúng của rig lý tưởng (`--fixed-signs nominal`), để so sánh công bằng.

### Rig thật (qua server đang chạy)

1. Server: `CITA_PROFILE=pbl_legacy`. Arm được chọn theo từng lượt bằng `cita_arm` trong `start-alignment`.
2. Khởi tạo golden cho tranh, qua app hoặc `POST /workflows/{device}/start-initialization`.
3. ROI: `python tools/experiments/define_roi.py --image <golden_left.png> --auto --size-mm 297,210 --out uploads/golden_poses/<artifact_id>/roi.json`.
4. Thiết bị chạy firmware mới (`embed/`, hiểu `motion_profile` và `keep_pose`).
5. Chạy:

```bash
export CITA_API_PASSWORD=...            # mật khẩu tài khoản operator; không bao giờ ghi vào lệnh/log
python tools/experiments/run_reloc_trials.py --device <device_code> --artifact-id <artifact_id> --paintings P001 \
    --api http://127.0.0.1:8000 --username <operator> --password-env CITA_API_PASSWORD \
    --n-perturb 15 --arms C-A,C-B,C-C,C-D --seed 2027 --i-know-hardware-moves
python tools/experiments/eval_reloc.py --logs logs/experiments/E2/<run>.jsonl \
    --golden-image <uploads/artifacts/<id>/golden_left_*.png> --calib data/camera_params/camera_params_lens_1.5.yaml \
    --roi uploads/golden_poses/<id>/roi.json --out results/E2/eval
python tools/experiments/make_tables.py --eval results/E2/eval.json --out results/tables/E2
```

* Mỗi lượt gồm: `reset` → nhiễu (`move` với `float_angles`) → `start-alignment` (`cita_arm`, `cita_trial_id`, `keep_pose=true`) → chờ `alignment_status` về `complete` hoặc `failed`.
* Server ghi từng vòng điều khiển vào `logs/experiments/E2/server_<ngày>.jsonl`. Trong Docker, mount `/app/logs` nếu cần giữ log.
* J (Jacobian) được lưu ở `<DATA_DIR>/control/jacobian_<device>_<artifact>.json` (trong Docker: volume dữ liệu), hết hạn sau `probe.ttl_s`. Muốn đo lại J thì xóa file này.
* Thành công theo GT = khoảng cách giữa bảng ở ảnh cuối và bảng ở ảnh golden không vượt τ. Thành công theo bộ ước lượng được báo riêng.

---

## 4. E3 – Kiểm tra hư hại (tới SSIM + đầu vào AI)

### Định dạng cặp ảnh

Mỗi thư mục là một bộ cặp ảnh, với `split` là `val` hoặc `test`:

```json
{"schema": "cita.pairs.v1", "session_id": "E3_P101_test", "split": "test",
 "roi": {"polygon_ref_px": [[x,y],...], "size_mm": [297, 210]}, "px_per_mm": null,
 "pairs": [{"pair_id": "P101_S2_L0_r1", "golden_image": "golden.png", "current_image": "cur_001.png",
            "gt": "gt_001.json", "offset_mm": 3.0, "offset_deg": 0.5}]}
```

* GT của từng cặp theo Phụ lục B.4 (`cita.change_gt.v1`). Hộp nằm trong **hệ ảnh golden**; cặp không đổi thì `changes: []`.
* Ảnh chụp bằng `capture_session.py` (`burst`/`grid` với `--split val|test`); người gán nhãn vẽ hộp trên ảnh golden.
* Ví dụ đầy đủ: `python tools/experiments/make_synthetic_session.py change --root /tmp/e3 --split val`.

### Ngưỡng (chỉ dùng validation), replay, đánh giá

```bash
python tools/experiments/calibrate_thresholds.py --pairs data/E3/val_* --split val --target-far 0.05 \
    --out configs/thresholds.yaml [--min-area-grid 2,4,8]          # từ chối mọi cặp có split khác val
python tools/experiments/replay_change.py --pairs data/E3/test_* --pipelines P0,P1,P2,P3,P4 \
    --model /path/to/team_model.pt --run-id E3_test
python tools/experiments/eval_change.py --logs logs/experiments/E3/E3_test.jsonl --pairs data/E3/test_* --out results/E3/eval
python tools/experiments/make_tables.py --eval results/E3/eval.json --out results/tables/E3
```

* `AIInputPackage` của từng cặp: `results/E3/<run>/<P>/<pair>/ai_input/ai_input_package.json`, kèm các crop PNG.
* `eval.json` gồm:
  * ROC-AUC, TPR tại FPR 5%;
  * tỉ lệ báo động giả trên cặp không đổi (kèm CI);
  * P/R/F1 mức vùng (IoU ≥ 0,25 hoặc tâm nằm trong hộp);
  * recall theo cỡ (< 5 mm / 5–15 mm / > 15 mm);
  * macro-F1 lớp;
  * đường độ nhạy theo `offset_mm`.
* P0 giữ nguyên lỗi tọa độ crop của PBL. Từ P1 trở đi, crop lấy từ ảnh đã căn chỉnh.
* YOLO (model, ngưỡng tin cậy, hai pha, NMS) **không đổi**.

---

## 5. Kiểm tra trước khi chạy dữ liệu thật

```bash
pytest -q                                    # ~2 phút trên CPU
python tools/experiments/smoke_server.py --profile pbl_legacy
python tools/experiments/smoke_server.py --profile proposed
python tools/experiments/make_synthetic_session.py pose --root /tmp/cita
python tools/experiments/replay_pose.py --session /tmp/cita/E1_SYNTH --methods M1,M2,M3 --log-root /tmp/cita/logs
python tools/experiments/run_reloc_trials.py --virtual --n-perturb 5 --log-root /tmp/cita/logs
```

## 6. Sửa lỗi sau `cita-freezeC`

* Chỉ sửa lỗi. Mỗi lần sửa thì tăng tag `cita-exp-v1.N` và ghi vào [CHANGELOG_CITA.md](CHANGELOG_CITA.md).
* Vì E1/E3 đánh giá offline, sau khi sửa chỉ cần chạy lại replay và eval trên ảnh đã chụp. Log mới mang commit mới, log cũ vẫn giữ nguyên.
