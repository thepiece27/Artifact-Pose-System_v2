# Báo cáo của agent: Artifact-Pose-System v2 → "experiment-ready" cho CITA 2027

* Ngày: 18/11/2026.
* Nhánh: `cita/experiment-ready`.
* Tag: `cita-baseline` (eaed45e), `cita-freezeA`, `cita-freezeB`, `cita-freezeC`.
* Tài liệu đi kèm: [EXPERIMENT_CODE_GUIDE.md](EXPERIMENT_CODE_GUIDE.md) (cách chạy), [CODE_MAP.md](CODE_MAP.md), [FRAMES_AND_UNITS.md](FRAMES_AND_UNITS.md), [CHANGELOG_CITA.md](CHANGELOG_CITA.md).

## 1. Tóm tắt

* Mọi nhánh M1–M5, C-A…C-D và P0–P4 chạy từ cùng một code base, được chọn bằng profile YAML (`server/configs/profiles`). Cả server và CLI đều đi qua registry.
* `pbl_legacy` (và trường hợp không đặt profile) cho kết quả **giống hệt** `cita-baseline` trên fixture ở 3 điểm: pose (khởi tạo golden + 3 lần hiệu chỉnh), kiểm tra P0 (2 cặp), và chuỗi lệnh MQTT của vòng căn chỉnh (6 bước). Sai lệch bằng 0 trong cùng môi trường. Kiểm chứng: `tests/cita/test_legacy_regression.py`.
* Pose và kiểm tra hư hại chạy **offline** trên phiên đã chụp (`replay_pose.py`, `replay_change.py`). Bộ điều khiển chạy được trên **rig ảo** (`run_reloc_trials.py --virtual`) và trên rig thật qua API của server.
* Mọi CLI ghi JSONL `cita.trial.v1`: commit, cờ dirty, profile_hash, seed, phiên bản thư viện, sha256 của ảnh đầu vào, đầu ra, thời gian.
* Kết quả test: **125 passed, 1 skipped** (test parity với g2o bị bỏ qua vì không build được module native), chạy 3 lần liên tiếp đều xanh, mỗi lần khoảng 75–100 s trên CPU. `compileall` sạch. Smoke server ở `pbl_legacy` và `proposed` đều chạy không lỗi.

## 2. Đã làm theo từng phase

| Phase | Nội dung | Commit/tag |
|---|---|---|
| 0 | Đọc toàn bộ repo (§5.1). Dựng venv Python 3.11. Viết `tests/synth.py` (cảnh phẳng tổng hợp có seed). Tạo fixture `tests/fixtures/synthetic`. Ghi `expected_baseline/{pose,inspection,alignment}.json` từ **code chưa sửa**. Viết `FRAMES_AND_UNITS.md`. Thêm bước CI cho `tests/cita` | `910907a`, `745bfcd` |
| 1 | `profiles.py` (`extends`, kiểm tra kiểu/miền, hash), `trial_log.py`, `registry.py`, `session.py`, GT bảng ChArUco + `board_mask`, `motion.py` (A.1/A.2). I2 trên thiết bị: `motion_profile.py`, góc float, chờ ổn định, tiếp cận một chiều, ngắt PWM, ack mở rộng, `keep_pose`. `ManualCameraSession`. `embed/tools/capture_session.py` (5 lệnh, `--dry-run`). `e0_noise`/`e0_settle`/`e0_repeat` | `834005b`…`88e569e`, **cita-freezeA** |
| 2 | `methods.py` (`estimate_pose`, `PoseEstimate`, golden lai có mask và nguồn stereo). `solver.py` (tối ưu nhiều khối). `markerless.py` (M2). `covariance.py` (I1). PoseService chọn phương pháp theo profile. `replay_pose`/`eval_pose`/`make_tables`/`stats` | `576075c`, `eb34141`, `83b7e34`, **cita-freezeB** |
| 3 | `control.py` (FixedMapping nguyên văn, Jacobian I3). `alignment_loop.py` (`AlignmentSession`). `sim/virtual_rig.py`. `run_reloc_trials.py`. Phiên CITA trong `inspection_service` (`cita_arm`, gom khung M5, `keep_pose`, log E2). Trường tùy chọn cho route | `d575a39`, `4bd314c` |
| 4 | `change_detection/` (legacy P0 chuyển nguyên văn, roi, registration+ECC, photometric, ssim_map, regions, ai_input, pipelines). Công cụ E3 (`define_roi`, `calibrate_thresholds`, `replay_change`, `eval_change`). `eval_reloc` | `1c7f99c`, `322a72c`, **cita-freezeC** |
| 5–6 | `requirements-lock.txt`. `CITA_PROFILE` trong compose/env example. `smoke_server.py`. `grid --with-stereo`. Cảnh báo cỡ mẫu khi chọn ngưỡng. Tài liệu | `a5c1302` và các commit sau |

Thống kê: 118 file đổi so với `cita-baseline` (+13 553 / −543 dòng). Trong đó 41 file nằm trong `server/app` và `embed` (+5 933 / −541). Phần còn lại là test, fixture, công cụ, profile và tài liệu.

File code cũ bị sửa (mọi thay đổi đều giữ mặc định cũ):
* `common.py`: thêm `mask=None` cho `extract_orb`.
* `pose_service.py`: thêm nhánh cho profile khác `legacy`.
* `inspection_service.py`: P0 chuyển sang `legacy.py` và để lại hàm ủy quyền; lệnh vòng cũ tính qua `FixedMappingController`; thêm phiên CITA.
* `state.py`: nạp profile.
* Schema/route: chỉ thêm trường tùy chọn.
* `hardware_controller.py`, `main_app.py`, `camera_manager.py`: thêm tính năng sau cờ.
* `requirements*.txt`, `docker-compose.yml`, `env.docker.example`.

## 3. Quyết định và giả định

1. **`pbl_legacy` = code hiện tại, không phải trọng số của báo cáo PBL.**
   * Mâu thuẫn trong đặc tả: §3.3 yêu cầu `pbl_legacy` khớp `cita-baseline`, nhưng Phụ lục C ghi `diamond_sigma_px: 0.01, orb_sigma_px: 1.0`. Code tại `cita-baseline` dùng `QualityConfig` với 0.5/1.5 px.
   * Đã chọn: `pbl_legacy` dùng 0.5/1.5 (giữ hành vi cũ). Trọng số PBL (Ω_d = 10⁴·I, Ω_o = I) là nhánh **M3**.
   * Tương tự: tolerance, số vòng tối đa và `SIGN_*` của `pbl_legacy` để `null`, tức là lấy từ Settings như cũ (25 mm / 2° / 20 vòng). Phụ lục C ghi 10 mm / 1° / 10 vòng; các giá trị này nằm trong các file C-*.
2. **Golden lai của M3/M4/M5 mặc định giữ cách cũ (`golden.stereo_source: diamond`)** để so sánh M3–M5 chỉ khác ở trọng số. Phương án `commanded` (baseline slider 100 mm, dấu chọn theo cheirality) có sẵn qua cờ hoặc `--override`. Xem phát hiện §5.2.
3. **M2** dùng `stereo_source: commanded`, T_right_left = [I | (−s·0,1, 0, 0)], dấu s chọn theo cheirality. Đây là giả định của bài gốc: slider X song song với trục x camera.
4. **Đếm vòng E2.** `max_iterations` (N_max = 10) chỉ đếm **vòng điều khiển**. Các lần di chuyển thăm dò J được đếm riêng (`n_probe_moves`), và `eval_reloc` báo cả hai.
   * J được lưu và tái dùng theo TTL.
   * Trong `run_reloc_trials --virtual`, mỗi (tranh, nhánh) có file J riêng, để C-C và C-D đều phải trả chi phí thăm dò như nhau.
5. **Quy tắc "không cải thiện"**: dừng khi 2 quan sát liên tiếp (sau ít nhất 1 lệnh điều khiển) không giảm sai số chuẩn hóa ‖c‖/τ_t + ‖θ‖/τ_R. Áp dụng chung cho mọi nhánh.
6. **Rig ảo**: C-A/C-B được cấp `SIGN_*` đúng cho rig lý tưởng (`nominal_fixed_signs`), nếu không thì so sánh sẽ không công bằng. Lệch trục mặc định 10°, tâm quay lệch (0, 10, −30) mm, rơ 0,4°, vùng chết 0,2°, nhiễu 0,3 mm / 0,03°.
7. **YOLO**: P0/P1 giữ cơ chế dự phòng "YOLO toàn ảnh khi `damage_pct` < 0,5%". P1 chạy dự phòng này trên ảnh đã căn chỉnh. P3 chỉ chạy YOLO trên vùng SSIM (đúng định nghĩa "trên AIInputPackage"). Pha A/B, early-exit 0,40, NMS 0,45 và ngưỡng tin cậy 0,15 giữ nguyên. P2 không chạy YOLO.
8. **P4** chỉ dùng ảnh hiện tại. Để chấm điểm trong hệ ảnh golden, hộp được ánh xạ bằng một homography ROI **chỉ dùng cho đánh giá**, ghi rõ ở khóa `registration_eval_only`.
9. **ECC**: chạy trên ảnh co về cạnh dài ≤ `ecc.max_side_px` (2000, khóa profile mới). Nghiệm ECC bị loại nếu lệch quá 20 px so với H của RANSAC (chống phân kỳ). Cả hai giá trị đều được log.
10. **Diện tích tối thiểu, `image_score`**:
    * Diện tích tối thiểu tính theo mm²; `px_per_mm` lấy từ `size_mm` của ROI, nếu không có thì từ cạnh Diamond.
    * `image_score` dùng cho ROC = phân vị 99,9 của bản đồ d đã làm mờ trong vùng hợp lệ. Với P0 là 1 − SSIM; với P4 là độ tin cậy lớn nhất.
11. **Chọn ngưỡng**: τ nhỏ nhất đạt FAR ≤ mục tiêu trên các cặp không đổi của tập val. Nếu số cặp không đổi < 1/FAR mục tiêu thì tool ghi `warning` vào file và in cảnh báo.
12. **Thiết bị**: góc mặc định vẫn làm tròn (theo §6.8.5). `current_yaw/pitch` đổi sang kiểu float (giá trị vẫn nguyên khi ở chế độ cũ). Log in 2 chữ số thập phân.
13. **Trục Z của rig: chưa xác minh được** từ code/tài liệu (bằng chứng mâu thuẫn, xem FRAMES_AND_UNITS §5). Không có chỗ nào trong code mới phải giả định chiều trục Z; E0.4 có hướng dẫn cách đo.
14. **Bảng tham chiếu mặc định**: 5×7, 30/22 mm, `DICT_5X5_100` (khác họ với Diamond nên không thể trùng ID). Ước lượng pose bằng IPPE rồi tinh chỉnh LM.

## 4. Sai khác so với đặc tả

| Mục | Sai khác | Lý do |
|---|---|---|
| §5.1 | `docs/cita/plan/CITA2027_Ke_hoach_Phan{1,2}.md` **không có trong repo**, nên không đọc được | Nhóm chưa chép vào. Mẫu Bảng E1 trong `make_tables.py` được thiết kế theo §7.5, cần đối chiếu lại với kế hoạch |
| §5.3.1 | Không build được module native g2o (máy không có cmake hay trình biên dịch C++). Toàn bộ dùng backend scipy, test parity bị bỏ qua | §7.4 cho phép dùng scipy làm chuẩn |
| §6.2 | Ngoài các trường của B.2, bản ghi có thêm `record_type`, `profile_name`, `profile_hash_arm` (profile riêng của từng nhánh) | Trường hợp một run chạy nhiều nhánh |
| §6.6 | Thêm `grid --with-stereo` và `--split` | E1 cần golden + stereo trong cùng một phiên |
| §8.1 | Vòng căn chỉnh trên server vẫn theo kiểu sự kiện (mỗi lần upload là một bước). `AlignmentSession` dùng chung cho server, vòng đồng bộ và rig ảo, thay cho một vòng `move()/capture()` chặn | Kiến trúc hiện có (thiết bị tự chụp rồi upload sau mỗi move) |
| §8.5 rig thật | Điều phối qua REST (reset → move có nhiễu → `start-alignment` có `cita_arm`) | Không cần truy cập MQTT trực tiếp từ công cụ |
| §9.1 | Thêm khóa `cita_regions` vào kết quả của P0 | Để chấm P0 ở mức vùng. Các khóa cũ không đổi |
| §10.5.4 | Smoke server dùng uvicorn trong tiến trình + urllib, không dùng `TestClient` | `TestClient` cần `httpx`, mà đặc tả cấm thêm thư viện |
| Phụ thuộc | Ghim `opencv-contrib-python<5`. Thêm `scikit-image` vào `requirements-dev.txt` (đã có sẵn trong `requirements-ai.txt`) | Xem §5.1. Test kiểm tra hư hại trong CI cần skimage |

## 5. Phát hiện quan trọng (cần nhóm biết)

1. **OpenCV 5 làm hỏng code gốc.**
   * `requirements.txt` cũ (`>=4.8.0`) cài OpenCV 5.1, mà ở bản này `cv2.undistortPointsIter` không còn. Hệ quả: `refine_pose` sập ngay khi gọi.
   * CI và Docker build mới cũng sẽ dính lỗi này. Đã ghim `<5`.
2. **Baseline stereo ước lượng từ Diamond sai khoảng 15% trên dữ liệu tổng hợp**: 84,5 mm so với 100 mm ra lệnh. Do đó bản đồ ORB của golden lai bị lệch tỉ lệ, kéo theo M3/M4/M5.
   * Trên fixture: M4 dùng golden `commanded` có e_t = 0,07 / 1,8 mm, so với 4,4 / 6,6 mm khi dùng golden `diamond`.
   * Khuyến nghị: chạy thêm ablation `--override golden.stereo_source=commanded` trong E1. Sau khi có số liệu thật thì cân nhắc đặt làm mặc định cho `proposed`.
3. **P0 (và vòng cũ) không tất định giữa các lần gọi.** Bộ so khớp FLANN của SIFT lấy số ngẫu nhiên từ `cv::theRNG()`, nên kết quả phụ thuộc thứ tự các lần gọi trong một tiến trình.
   * Các công cụ replay đặt `cv2.setRNGSeed(seed)` trước mỗi cặp, test hồi quy đặt về seed 0.
   * Server sản xuất vẫn giữ hành vi cũ.
4. **Dấu `SIGN_*` mặc định mâu thuẫn**: `config.py` mặc định −1, `docker-compose.yml` mặc định +1, `docs/04` ghi +1. Nghĩa là "baseline" của ánh xạ cố định khác nhau tùy cách triển khai. Cần chốt bằng đo đạc (E0.4) trước E2.
5. **Lượng tử hóa servo 1° chặn hội tụ góc nhỏ.** Trên rig ảo với τ = 2 mm / 0,2°, C-A và C-C (không I2) không đạt, còn C-B và C-D (có góc float) đạt 99–100%. Phần tịnh tiến của C-C vẫn tốt (0,3 mm). Đây là lý lẽ định lượng cho I2.
6. **Bài toán nhìn chính diện bị điều kiện kém.**
   * Diamond 40 mm và bảng tham chiếu ở 65 cm cho sai số tâm camera cỡ mm tới cm khi độ phân giải thấp. Trên ảnh tổng hợp 960 px có siêu lấy mẫu: Diamond 7–12 mm, bảng 0,3–2 mm.
   * Ở 4K thì tốt hơn khoảng 4 lần, nhưng vẫn cần E0.5 định lượng chất lượng GT trước khi kết luận về các sai số dưới mm.
7. **P0/P1 bỏ sót hư hại nhỏ** vì `min_area = max(500, 1e-4·h·w)` px². Ở 4K ngưỡng này là 829 px², tương đương khoảng 40 mm² ở 4,5 px/mm. P2/P3 dùng `min_area_mm2` (mặc định 4 mm²).
8. **Retry khi không thấy Diamond ở vòng cũ sẽ đưa servo về home**: lệnh capture không có `keep_pose`. Phiên CITA luôn gửi `keep_pose=true`.

## 6. Rủi ro còn lại

* **Chưa chạy trên phần cứng thật.**
  * `ManualCameraSession`, I2 trên servo/stepper thật và `run_reloc_trials` ở chế độ thật mới chỉ được kiểm bằng fake/mock. Lần chạy đầu: dùng `--dry-run`, rồi chạy rig thật với biên độ nhỏ.
  * Picamera2 cần `AeEnable/AwbEnable=False` được driver chấp nhận trong `create_still_configuration`; nếu báo lỗi thì bỏ khóa `controls` ở bước configure (khi đó điều khiển vẫn được đặt bằng `set_controls`).
* **Hiệp phương sai dưới ước lượng khi có sai số hệ thống.** Đúng trong Monte Carlo có nhiễu khớp mô hình (độ phủ 2σ ≈ 95%). Trên dữ liệu tổng hợp có thiên lệch hệ thống (golden lệch tỉ lệ), độ phủ chỉ khoảng 11–20%. Cần kiểm lại trên E1 thật (`coverage_2sigma` trong `eval_pose`).
* **Ngưỡng E3** chỉ tốt khi tập val đủ lớn: ít nhất 20 cặp không đổi cho mục tiêu FAR 5%.
* **CI chưa chạy thử trên GitHub**, mới kiểm trên Windows/Python 3.11. Test hồi quy so số thực với dung sai 1e-6…1e-9. Nếu nền Linux cho khác biệt dấu phẩy động lớn hơn mức đó thì cần nới dung sai, **không** phải ghi lại baseline.
* **Log E2 phía server** ghi trong container tại `/app/logs` (không có volume): cần mount nếu chạy E2 bằng Docker.

## 7. Kết quả test và vòng ổn định

| Bước §10.5 | Kết quả |
|---|---|
| `pytest -q` (toàn bộ, 3 lần) | 125 passed, 1 skipped × 3, không có test chập chờn |
| `python -m compileall -q app ../embed ../ai_module` | sạch |
| Smoke offline: E0 noise → `replay_pose` M1–M5 → `eval_pose`; calibrate (val) → `replay_change` P0–P4 → `eval_change`; `run_reloc_trials --virtual` 4 nhánh × 5 lượt → `eval_reloc`; `make_tables` E1/E2/E3 | chạy hết, không lỗi |
| Smoke server `pbl_legacy` / `proposed` | health 200, khởi tạo golden 200, `/pose/correct` 400 sạch (không có golden toàn cục, đúng như cũ), kiểm tra 200 (`analysis_status=ok`, pipeline P0 / P3) |
| Nghiệm thu Phase 2 | M1–M5 dưới ngưỡng trên ảnh tổng hợp; Monte Carlo độ phủ 2σ trong [0,90; 1,0] với Huber δ = 3 và δ = 2; hồi quy `pbl_legacy` khớp |
| Nghiệm thu Phase 3 | Test khẳng định C-D ≥ 95% hội tụ, trung vị ≤ 6 vòng trên 100 lượt có seed ở τ = 2 mm / 0,3° (rig lệch trục 10°, tâm quay lệch 30 mm). Lần chạy thăm dò ở τ = 2 mm / 0,2°: 100% theo bộ ước lượng, 98% theo GT, trung vị 3 vòng. FixedMapping tái hiện đúng lệnh cũ (6 bước, 500 đầu vào ngẫu nhiên) |
| Nghiệm thu Phase 4 | P2/P3 có FAR trên cặp không đổi thấp hơn P0; recall ≥ 0,9; khứ hồi tọa độ crop ≤ 1 px; P0 khớp baseline; `calibrate_thresholds` từ chối dữ liệu test |
| `git status` | sạch sau commit cuối |
| Secrets trong diff | `git diff cita-baseline` chỉ chứa tên biến (`--password-env`, `AUTH_SECRET_KEY` được sinh ngẫu nhiên khi chạy smoke). Không file `.env*`, `ai_module/**`, `client/**` nào bị sửa (đã kiểm tra bằng `git diff --stat`) |

## 8. ⚠️ Cảnh báo secrets đang bị track trong git

Các file sau đang được **track** trong git và (theo đặc tả) chứa giá trị thật:
* `server/.env`
* `server/.env.docker`
* `embed/device_agent/.env`

Agent **không mở để đọc giá trị**, không sửa và không chép chúng vào đâu cả. `.gitignore` đã liệt kê các file này nhưng không có tác dụng với file đã được track từ trước.

Khuyến nghị (nhóm tự làm, có người duyệt):
1. **Đổi ngay** mọi khóa và mật khẩu trong các file đó: `AUTH_SECRET_KEY`, `ADMIN_PASSWORD`, `POSTGRES_PASSWORD`, `DEVICE_API_KEY`, `DEVICE_ENROLLMENT_KEY`, `MQTT_PASSWORD`… Coi như các giá trị cũ đã lộ, vì chúng có trong lịch sử git.
2. `git rm --cached server/.env server/.env.docker embed/device_agent/.env` rồi commit; giữ file thật ở máy triển khai.
3. Nếu repo đã public hoặc đã chia sẻ: xóa khỏi lịch sử bằng `git filter-repo` (hoặc BFG), sau đó force-push có phối hợp với cả nhóm. Agent **không** viết lại lịch sử.

## 9. Việc nhóm cần làm tiếp

1. Chép `docs/cita/plan/CITA2027_Ke_hoach_Phan1.md` và `…Phan2.md` vào repo, rồi đối chiếu cột của Bảng E1/E2/E3 trong `make_tables.py`.
2. In và dán bảng ChArUco tham chiếu. Đo cạnh ô thật rồi cập nhật `ground_truth.board` nếu khác 30/22 mm.
3. Tuần 23–29/11:
   * chạy E0 theo `EXPERIMENT_CODE_GUIDE.md` §1;
   * tạo `configs/noise_model.yaml`;
   * xác định trục Z và dấu `SIGN_*`;
   * chốt `settle_s`, `backlash_deg`, τ_t, τ_R, δ thăm dò trong các file C-*/`proposed`.
4. Chụp E1 bằng `grid --with-stereo`, replay và eval (kèm ablation `golden.stereo_source=commanded`).
5. Trước E2: thử `run_reloc_trials --virtual` với tham số rig đo được, rồi mới lên rig thật (biên độ nhỏ trước).
6. E3: chụp các tập val/test riêng, gán nhãn GT theo B.4, chọn ngưỡng **chỉ** trên val.
7. Sau `cita-freezeC` chỉ sửa lỗi, tăng tag `cita-exp-v1.N` và ghi vào `CHANGELOG_CITA.md`.
8. Xử lý secrets theo §8.
