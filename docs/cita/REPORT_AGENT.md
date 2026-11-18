# Báo cáo của agent: Artifact-Pose-System v2 → "experiment-ready" cho CITA 2027

* Ngày: 18/11/2026.
* Nhánh: `cita/experiment-ready`.
* Tag: `cita-baseline` (eaed45e), `cita-freezeA`, `cita-freezeB`, `cita-freezeC`, `cita-exp-v1.0`, `cita-exp-v1.1`, `cita-exp-v1.2`. Đợt sửa v1.2 xem §10.
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

Thống kê (tại tag `cita-exp-v1.1`): xem `git diff --stat cita-baseline cita-exp-v1.1`. Khoảng 120 file, trong đó khoảng 41 file nằm trong `server/app` và `embed`. Phần còn lại là test, fixture, công cụ, profile và tài liệu.

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
| §6.2 | Ngoài các trường của B.2, bản ghi có thêm `record_type`, `profile_name`, `profile_hash_arm` (profile riêng của từng nhánh). Công cụ đánh giá ghi `record_type: "tool"` với `profile` = null | Trường hợp một run chạy nhiều nhánh; công cụ đánh giá không chạy theo profile phương pháp |
| §7.1 | `run_correction_step` được giữ **nguyên văn** (chính là nhánh `legacy`); việc chọn phương pháp theo profile đặt ở `PoseService.correct_image`, không đặt bên trong `run_correction_step` | Giữ chữ ký, đầu ra và CLI `correction.py` cũ mà không phải truyền profile vào hàm cũ. `estimate_pose(method="legacy")` tái hiện nó chính xác (test hồi quy) |
| §7.2 (M2) | Golden M2 dùng `match_descriptors` (mutual + ratio 0,6) và `geometry.triangulate` với T = baseline ra lệnh, thay cho `match_stereo`/`triangulate_stereo` native | Module native không build được ở đây. Cách làm tương đương về hình học, có kiểm tra cheirality và các cổng chất lượng sẵn có |
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
5. ~~**Lượng tử hóa servo 1° chặn hội tụ góc nhỏ.** Trên rig ảo với τ = 2 mm / 0,2°, C-A và C-C (không I2) không đạt, còn C-B và C-D (có góc float) đạt 99–100%. Phần tịnh tiến của C-C vẫn tốt (0,3 mm). Đây là lý lẽ định lượng cho I2.~~

   **Đính chính (v1.2): nhận định trên quy sai nguyên nhân.**
   * Lượng tử hóa servo **không** cản việc quay về golden, vì golden nằm đúng góc home (số nguyên, và đúng một tick PWM).
   * Thứ thực sự giới hạn góc nhỏ là **rơ và vùng chết**:
     * Khi quay về, rơ để lại phần dư ±rơ/2.
     * Phần dư này nhỏ hơn bước lệnh nhỏ nhất (1° với góc nguyên, 0,586° với PWM), nên không bù được.
     * Vùng chết thì nuốt luôn các lệnh sửa nhỏ.
   * Rơ và vùng chết phải đo ở E0.4 trước khi chốt τ_R.

   Mô phỏng: 2 tranh × 25 nhiễu × 4 nhánh, seed 2027, τ_t = 2 mm. Mỗi ô ghi tỉ lệ thành công theo GT của C-A / C-B / C-C / C-D:

   | Rig ảo | τ_R | PWM 0 | PWM 0,586° |
   |---|---|---|---|
   | aligned, rơ 0,4°, vùng chết 0,2° (mặc định) | 1° | 94 / 90 / 94 / 92% | 92 / 92 / 98 / 94% |
   | yawed25, rơ 0,4°, vùng chết 0,2° | 1° | 100 / 100 / 96 / 96% | 100 / 100 / 98 / 98% |
   | aligned, **không rơ, không vùng chết** | 0,2° | 90 / 90 / 96 / 98% | 92 / 92 / 58 / 96% |
   | aligned, rơ 0,4°, vùng chết 0,2° | 0,2° | 0 / 94 / 0 / 92% | 0 / 0 / 0 / 0% |

   Đọc bảng:
   * Ở τ_R = 1°, bước PWM 0,586° không làm giảm thành công.
   * Khi không có rơ và vùng chết, C-A (góc nguyên) vẫn đạt 90% ở τ_R = 0,2°. Vậy lượng tử hóa không phải nguyên nhân.
   * Khi có rơ, góc float (I2) chỉ giúp được nếu bước lệnh thực sự nhỏ hơn phần dư. Với PCA9685 ở 50 Hz thì điều này không đúng, nên τ_R = 0,2° là không thực tế.
   * Ngoại lệ C-C 58% (PWM, không rơ): ghi nhận, chưa phân tích. Có khả năng do lệnh thăm dò J bị lượng tử hóa.
   * I2 vẫn có ích vì tiếp cận một chiều triệt được rơ. Nhưng lý lẽ cho I2 phải dựa vào rơ đo ở E0.4, không dựa vào lượng tử hóa.
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
  * *Cập nhật v1.2 (R1):* đã chạy trên Linux (WSL Ubuntu 22.04, Python 3.11, file lock) và nới dung sai theo số đo được, xem §10.2. CI trên GitHub vẫn chưa chạy thử.
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
7. Sau `cita-freezeC` chỉ sửa lỗi, tăng tag `cita-exp-v1.N` và ghi vào `CHANGELOG_CITA.md`. Bản bàn giao hiện tại là `cita-exp-v1.2` (§10).
8. Xử lý secrets theo §8.

## 10. Đợt v1.2 (`cita-exp-v1.2`)

* Ngày 18/11/2026. Bắt đầu từ `cita-exp-v1.1` (7016514) và làm theo `AGENT_PROMPT_CITA2027_v1.2.md`.
* Ràng buộc của đặc tả gốc (§3, §12) vẫn giữ nguyên.
* Đã làm đủ **R1–R10**, không mục nào dời sang v1.3.
* **Chưa có dữ liệu thật nào bị ảnh hưởng**, vì chưa có phiên thật nào được chụp hay chạy.

### 10.1 Đã làm

| Mục | Commit | Nội dung | Nghiệm thu |
|---|---|---|---|
| R1 | `c237219` | So sánh P0 với `expected_baseline` theo dung sai đo được giữa Windows và Linux; CI cài `requirements-lock.txt` và in phiên bản thư viện | Test hồi quy xanh trên cả hai nền; `expected_baseline` không đổi |
| R2 | `413ef9f` | VCE (Helmert/Förstner) cho σ_d, σ_o trên phần dư chiếu lại; `noise_model` v2 có `golden_stereo_source`; thêm `M4-jitter` | `test_noise_vce.py`: σ_d ∈ [0,2; 0,45], σ_o ∈ [0,8; 1,25] khi σ thật là 0,3 / 1,0 px |
| R3 | `bad872c` | Mỗi nhánh một file J; `probe.reuse` = `ttl` / `per_block` / `never`; `block_id`; workflow nhận `cita_probe_reuse` / `cita_block_id` | `test_jacobian_reuse.py` |
| R5 | `22467b5`, `c27503c` | Đơn vị là vị trí; `e_t_first` là chỉ số chính; M5 có trong kiểm định cặp; so sánh định trước + Holm; bootstrap cụm hai tầng; `fallback_rate`, `n_orb_inliers`, `e_t` tách lai/fallback; độ phủ a priori và a posteriori | `test_eval_pose_r5.py`: 12 đơn vị; CI cụm rộng hơn CI iid khi có thiên lệch theo phiên |
| R4 | `c309d4b` | `verified_regions` và `image_score_verified`; hàng ảo P3v; YOLO không đổi | `test_p3v.py`: cặp không đổi → P3 là báo động giả, P3v thì không; cặp có hư hại → P3v có TP |
| R6 | `e65687c` | `golden.stereo_source = commanded_epipolar`; `e0_stereo_check.py` | `test_stereo_epipolar.py` |
| R7 | `defd0d8`, `93ef8d3` | Hai điều kiện lắp `aligned` / `yawed25` (ảo và thật); chỉ số chính `total_moves`; τ bắt buộc; `pwm_step_deg`; `--noise-from`; cảnh báo khi τ bộ điều khiển lỏng hơn τ chấm | `test_e2_mount.py` |
| R8 | `8ecc00b` | `capture_session.py`: in vùng di chuyển, thời gian và dung lượng rồi đòi xác nhận; kiểm tra đĩa; lệnh con `awb` + `--awb-from`; `--pulse-delay-us`; `release_servos` chỉ nhả pan | `test_capture_session_r8.py` (dry-run và fake) |
| R9 | `5d0338c` | Tài liệu: dấu của θ; đính chính §5.5; mục "Trước mỗi phiên trên rig"; CHANGELOG | — |
| R10 | `bb7b9a2` | Test snapshot bảng E1/E2/E3 (`.md` + `.tex`, vi/en); sửa lỗi `--out` có dấu chấm | `test_tables_snapshot.py` |

### 10.2 Quyết định (chỗ đặc tả chưa nói rõ; đều chọn hướng thận trọng)

1. **R1, dung sai.**
   * Chênh lệch đo được giữa hai nền: `ssim` 4,3e-6 / 3,0e-6; `valid_mask_ratio` 1,3e-5 / 4,03e-4; `region_ssim` tối đa 1,7e-3.
   * Dung sai đặt khoảng 2–3 lần mức đó: 1e-5 / 1e-3 / 5e-3.
   * Lớp, độ tin cậy, hộp, crop, số vùng và mọi khóa nguyên/phân loại vẫn phải khớp **tuyệt đối**.
   * CI cài file lock để hai nền dùng cùng phiên bản thư viện.
2. **R2, cấu hình VCE.**
   * Huber δ = 2. Điểm ORB chỉ được tính khi phần dư chuẩn hóa ≤ 3.
   * Dừng khi thay đổi tương đối < 1%, tối đa 20 vòng.
   * Cảnh báo khi độ dư Diamond < 20.
   * Mô phỏng (σ thật 0,3 / 1,0 px): σ_d = 0,342, σ_o = 0,931 px, sau 4 vòng. Cách đo jitter cho σ_d = 0,026 px, tức tỉ trọng Diamond:ORB là 1314:1, so với 7,4:1 khi dùng VCE.
   * `noise_model` v1 vẫn nạp được nhưng không kiểm nguồn golden. v2 thì bắt buộc khớp nguồn, sai là từ chối.
3. **R3 và R7 dùng chung commit.** Phần nền cho R7 nằm trong commit R3 vì cùng file: `RigParams.preset`, PWM, `noise_from_file`, viết lại `run_reloc_trials`/`eval_reloc`. Commit R7 bổ sung test và hướng dẫn.
4. **R3, mặc định.**
   * `run_reloc_trials` dùng `--probe-reuse per_block`; server vẫn `ttl`.
   * Khi chạy thật, tool đổi tên J cũ thành `*.stale` (không xóa) và ghi việc này vào log.
   * File J dạng cũ (v1) có tên không chứa nhánh nên không còn được đọc. Lần đầu, mỗi nhánh sẽ thăm dò lại.
5. **R5.**
   * `make_tables.py` được viết lại trong commit R5, dùng chung cho E1/E2/E3. Ô không áp dụng ghi "—" (LaTeX `---`).
   * Phép so sánh thiếu phương pháp bị bỏ qua, không báo lỗi.
   * `fallback_rate` chỉ áp dụng cho phương pháp lai (M3/M4/M5). Với M1 (chỉ Diamond) và M2 (không marker) thì là "—".
6. **R4.** P1 khi có YOLO cũng có `verified_regions`, nhưng không thêm hàng "P1v" vì đặc tả chỉ yêu cầu P3v.
7. **R6.**
   * Giả định R = I giữa hai ảnh golden, vì chỉ slider trượt còn servo đứng yên.
   * Hướng tịnh tiến: RANSAC 2 điểm trên ràng buộc t·(x₁×x₂) = 0, rồi tinh chỉnh SVD trên inlier. Dấu chọn theo cheirality. Độ dài lấy theo baseline lệnh.
   * Quy tắc chọn nguồn: `commanded_epipolar` nếu trung vị |b_Diamond − b_lệnh| > 2 mm hoặc góc(Diamond, epipolar) > 1°.
   * Mô phỏng slider lệch 5° so với trục x camera:

     | Nguồn golden | Sai số hướng | e_t trung vị của M4 |
     |---|---|---|
     | `diamond` | 3,45° | 2,07 mm |
     | `commanded` | 5° | 2,39 mm |
     | `commanded_epipolar` | 0,27° | 0,81 mm |

8. **R7.**
   * Ở chế độ thật, `--mount-condition` là bắt buộc.
   * `eval_reloc` từ chối chạy nếu thiếu τ và τ trong log không thống nhất.
   * Thêm (từ smoke): nếu τ của bộ điều khiển lỏng hơn τ chấm thì có cảnh báo trong `warnings`. Các file `C-*` đang giữ chỗ 10 mm, nên khi chấm ở 2 mm thì tỉ lệ thành công chỉ đo quy tắc dừng.
9. **R8.**
   * Bảng kế hoạch được in ở mọi chế độ, nhưng chỉ ngoài `--dry-run` mới phải gõ `yes`. Gặp EOF (không có stdin) thì coi là từ chối.
   * Dry-run ước 0,3 MB/khung.
   * Lệnh `awb` không cần `--i-know-hardware-moves`, vì không di chuyển gì.
   * Mã thoát: 2 (thiếu cờ phần cứng), 3 (không xác nhận), 4 (thiếu AWB), 5 (thiếu đĩa).
10. **R10.** Snapshot ở `server/tests/cita/snapshots/`. Khi cố ý đổi định dạng bảng: chạy `CITA_UPDATE_SNAPSHOTS=1 pytest tests/cita/test_tables_snapshot.py` rồi duyệt diff.

### 10.3 Sai khác và phát hiện thêm

* **Lỗi tìm thấy khi chạy smoke, đã sửa kèm test:**
  * M1 hiện "Rơi về Diamond 100%" (`c27503c`).
  * τ bộ điều khiển và τ chấm không khớp mà không có cảnh báo (`93ef8d3`).
  * `--out` có dấu chấm, ví dụ `eval_v1.2`, bị `Path.with_suffix` cắt thành `eval_v1.json`, nên có thể ghi đè kết quả khác mà không báo. Lỗi này có ở `eval_pose`, `eval_reloc`, `eval_change` và `make_tables` (`bb7b9a2`).
* **R9.2, giới hạn của τ_R.** Mô phỏng ở §5.5 cho thấy: với rơ 0,4° và vùng chết 0,2°, τ_R = 0,2° **không nhánh nào đạt** khi có bước PWM 0,586°. Vậy τ_R dưới khoảng 0,3° là không thực tế với PCA9685 nếu rơ cỡ này. Chốt τ_R sau E0.4.
* **Bảng "hiện trạng" của R7 trong đặc tả** (lệch 10°/25°/45°) dùng tham số rig cũ. Hướng dẫn §3 dùng số đo lại với các preset mới.
* **P4 trên dữ liệu tổng hợp** có FAR 100% ở smoke, giống hệt v1.1. Đây không phải hồi quy của v1.2; dữ liệu tổng hợp chỉ có 3 cặp không đổi và YOLO giả.

### 10.4 Kết quả test và vòng ổn định

Hai vòng liên tiếp (A tại `93ef8d3`, B tại `edafb1c`; giữa hai vòng chỉ có thay đổi tài liệu):

| Bước | Vòng A | Vòng B |
|---|---|---|
| `pytest -q` Windows (Python 3.11.9), 3 lần | 170 passed, 1 skipped × 3 | 170 passed, 1 skipped × 3 |
| `pytest -q` Linux (WSL Ubuntu 22.04, Python 3.11.18, file lock), 3 lần | 170 passed, 1 skipped × 3 | 170 passed, 1 skipped × 3 |
| `compileall -q app ../embed ../ai_module` | sạch | sạch |
| Smoke: synthetic → `e0_stereo_check` → `e0_noise` (vce + jitter) → `replay_pose` M1–M5 + M4-jitter (cả ba nguồn golden) → `eval_pose` → `make_tables` E1; `run_reloc_trials --virtual` aligned + yawed25 × 4 nhánh × 5 lượt → `eval_reloc` → `make_tables` E2; `calibrate_thresholds` (val) → `replay_change` P0–P4 → `eval_change` (có P3v) → `make_tables` E3 | chạy hết | chạy hết |
| `smoke_server.py` với `pbl_legacy` / `proposed` | ok | ok |
| `pbl_legacy` / P0 / C-A so với `expected_baseline` | khớp (dung sai R1) | khớp |
| `git status`; secrets; `.env*`, `ai_module/**`, `client/**` | sạch; không có secret; không file nào bị sửa | như vòng A |

Test bị bỏ qua là test parity với g2o, vì không build được module native. Không có test chập chờn.

### 10.5 Việc nhóm cần làm tiếp (bổ sung §9)

1. Push nhánh và tag: `git push origin cita/experiment-ready --tags`. Các tag `cita-freezeA/B/C` và `cita-exp-v1.0/1.1/1.2` hiện **chưa có trên GitHub**.
2. Chạy E0.3 trên tranh hiệu chuẩn P000:
   * `e0_noise` (VCE) → `configs/noise_model.yaml`;
   * `e0_noise --estimator jitter` → `configs/noise_model_jitter.yaml`;
   * `e0_stereo_check` → chốt `golden.stereo_source` và ghi vào CHANGELOG **trước khi** đánh giá E1. Noise model phải ước lượng lại với đúng nguồn đã chốt.
3. Chạy E0.4: đo rơ và vùng chết của servo, rồi chốt τ_t, τ_R trong các file `C-*` (thay giá trị giữ chỗ 10 mm) trước E2. Lưu ý giới hạn τ_R ở §10.3.
4. E2: mỗi điều kiện lắp (`aligned`, `yawed25`) khởi tạo golden riêng.
5. Trước mỗi phiên trên rig: theo EXPERIMENT_CODE_GUIDE §0.6 (khoảng trống X, `--return-home`, thẻ nhớ, chạy `awb` khi đổi đèn).
