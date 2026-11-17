# Sơ đồ code (CITA 2027)

## 1. Luồng chạy trên hệ thống thật

```
App/REST ── POST /workflows/{dev}/start-alignment (cita_arm?, cita_trial_id?, keep_pose?)
   │            └─ MQTT cmd/{dev}: {"action":"capture", keep_pose?, workflow{auto_alignment_loop, cita_arm?}}
   ▼
Pi  runtime/main_app.py ── _handle_capture (về home trừ khi keep_pose / _internal_after_move)
   │   └─ CameraManager.capture_simple ─ api_client.upload_inspection (metadata.workflow quay lại server)
   ▼
Server  InspectionService.handle_upload
   ├─ workflow.cita_arm hoặc profile điều khiển không-legacy ─► _handle_cita_alignment
   │      ├─ gom k ảnh tĩnh (M5) bằng capture keep_pose
   │      ├─ PoseService.correct_image(profile=arm) ─► methods.estimate_pose (M1–M5) ─► PoseEstimate
   │      ├─ alignment_loop.AlignmentSession.on_observation ─► control.{FixedMapping,Jacobian}Controller
   │      └─ MQTT move {x_steps,…, motion_profile} | capture retry | alignment_complete/failed
   │         + log E2 (logs/experiments/E2/server_<ngày>.jsonl)
   └─ mặc định (pbl_legacy) ─► vòng cũ: PoseService (run_correction_step)
          ─► control.FixedMappingController.legacy_step ─► MQTT move (y hệt cita-baseline)
   ▼
Pi  _handle_compound_move ─ MotionProfile.resolve(motion_profile > env > mặc định)
   ├─ HardwareController.approach_pan_tilt (float / tiếp cận một chiều) + slider một chiều
   ├─ chờ settle_s, ngắt PWM (tùy chọn), ack {commanded, motion_profile}
   └─ capture sau move (nội bộ, không về home) ─► upload ─► vòng lặp tiếp
   ▼
Kiểm tra: POST /api/v1/artifacts/{id}/inspect ─► InspectionService.run_artifact_inspection
   └─ _analyze_against_reference ─ P0: change_detection.legacy.analyze_legacy (code cũ, giữ nguyên)
                                  └ P1–P4: change_detection.pipelines.run_server_pipeline
                                      registration ─ photometric ─ ssim_map ─ regions ─ ai_input ─ YOLO (ModelService, không đổi)
```

## 2. Module

| Đường dẫn | Vai trò |
|---|---|
| `server/app/experiments/profiles.py` | Profile YAML (`extends`, kiểm tra kiểu và miền giá trị, `profile_hash`, `CITA_PROFILE`) |
| `server/app/experiments/trial_log.py` | Log JSONL `cita.trial.v1`, `git_info`, `env_info`, `seed_everything` |
| `server/app/experiments/registry.py` | Mã nhánh → hàm cài đặt + profile mặc định |
| `server/app/experiments/session.py` | Đọc/kiểm tra phiên `cita.session.v1` |
| `server/app/experiments/stats.py` | Bootstrap, Friedman, Wilcoxon-Holm, McNemar, ROC |
| `server/app/modules/artifact_pose/{common,geometry,initialize,correction}.py` | Code gốc. Chỉ thêm `mask=None` cho `extract_orb` |
| `server/app/modules/artifact_pose/motion.py` | Chuyển động tương đối A.1, sai số A.2 |
| `server/app/modules/artifact_pose/methods.py` | `estimate_pose`, `PoseEstimate`, `GoldenModel`, dựng golden lai (có mask, nguồn stereo) |
| `server/app/modules/artifact_pose/solver.py` | Tối ưu pose-only gồm nhiều khối (M5 xếp chồng, M2 chỉ ORB, ablation tắt Huber) |
| `server/app/modules/artifact_pose/covariance.py` | Hiệp phương sai I1 (tiên nghiệm/hậu nghiệm, lan truyền sang tâm và góc) |
| `server/app/modules/artifact_pose/markerless.py` | M2 kiểu Hung et al. (golden + hiệu chỉnh, lưu/đọc `.npz`) |
| `server/app/modules/artifact_pose/control.py` | `FixedMappingController` (C-A/C-B), `JacobianController` (I3) |
| `server/app/modules/artifact_pose/alignment_loop.py` | `AlignmentSession` (máy trạng thái), `run_alignment` |
| `server/app/modules/sim/virtual_rig.py` | Rig ảo (lệch trục, tâm quay, rơ, vùng chết, lượng tử hóa, nhiễu) |
| `server/app/modules/ground_truth/reference_board.py` | GT từ bảng ChArUco, `board_mask` |
| `server/app/modules/change_detection/legacy.py` | P0 = code kiểm tra cũ (chuyển nguyên văn) |
| `server/app/modules/change_detection/{roi,registration,photometric,ssim_map,regions,ai_input,pipelines}.py` | I4, các pipeline P1–P4 |
| `server/app/services/pose_service.py` | Chọn phương pháp theo profile (`legacy` = đường cũ) |
| `server/app/services/inspection_service.py` | Vòng cũ + phiên CITA, bộ điều phối kiểm tra |
| `embed/device_agent/adapters/motion_profile.py` | I2: `MotionProfile`, `unidirectional_waypoints` |
| `embed/device_agent/adapters/hardware_controller.py` | Góc float (theo cờ), tiếp cận một chiều, ngắt PWM, driver tiêm được |
| `embed/device_agent/adapters/camera_manager.py` | `ManualCameraSession` (AE/AWB tắt, lấy nét cố định) |
| `embed/tools/capture_session.py` | Chụp E0/E1/E3 (burst/settle/repeat/grid/stereo, `--dry-run`) |
| `server/tools/experiments/*.py` | `e0_noise`, `e0_settle`, `e0_repeat`, `replay_pose`, `eval_pose`, `run_reloc_trials`, `eval_reloc`, `define_roi`, `calibrate_thresholds`, `replay_change`, `eval_change`, `make_tables`, `make_synthetic_session`, `smoke_server` |
| `server/configs/profiles/*.yaml` | Profile các nhánh |
| `server/tests/synth.py` | Cảnh phẳng tổng hợp có seed (Diamond, bảng, tranh, hư hại, vật gây nhiễu) |
| `server/tests/fixtures/` | Fixture tổng hợp + `expected_baseline` (đầu ra của code gốc) |
| `server/tests/cita/` | Test CITA (CI: `pytest tests/cita -q`) |

## 3. Bất biến quan trọng

* `CITA_PROFILE` không đặt (hoặc `pbl_legacy`) ⇒ pose, vòng căn chỉnh, kiểm tra và lệnh MQTT **giống `cita-baseline`**. Kiểm chứng bởi `tests/cita/test_legacy_regression.py`.
* Thiết bị nhận lệnh không có `motion_profile` hay `keep_pose` ⇒ hành vi cũ (góc nguyên, không chờ, về home khi capture).
* YOLO: model, suy luận, ngưỡng tin cậy, hai pha, NMS, luật trạng thái không đổi. Chỉ thay crop đầu vào và ánh xạ hộp kết quả.
