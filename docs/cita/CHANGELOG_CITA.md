# CHANGELOG CITA 2027

Mỗi lần sửa sau `cita-freezeC` thì tăng tag `cita-exp-v1.N` và thêm một mục vào đây. Mỗi mục ghi: tag, ngày, mô tả lỗi, file đã sửa, ảnh hưởng tới kết quả đã có, cần chạy lại replay nào.

| Tag | Ngày | Nội dung |
|---|---|---|
| `cita-baseline` | 18/11/2026 | Commit `eaed45e` (trước mọi thay đổi CITA) |
| `cita-freezeA` | 18/11/2026 | Phase 0–1: fixture + `expected_baseline`, profile, logger, registry, session, GT bảng ChArUco, I2 trên thiết bị, `capture_session.py`, `e0_noise`/`e0_settle`/`e0_repeat` (kèm lõi pose Phase 2 mà `e0_noise` cần) |
| `cita-freezeB` | 18/11/2026 | Phase 2: M1–M5, hiệp phương sai, `replay_pose`/`eval_pose`/`make_tables`, PoseService chọn phương pháp theo profile |
| `cita-freezeC` | 18/11/2026 | Phase 3–4: bộ điều khiển (C-A…C-D), `AlignmentSession`, rig ảo, `run_reloc_trials`/`eval_reloc`, phát hiện thay đổi P0–P4, `AIInputPackage`, công cụ E3 |
| (sau freezeC) | 18/11/2026 | Phase 5–6 (trước bản bàn giao): lock file, `CITA_PROFILE` trong compose, smoke server, `grid --with-stereo`, tài liệu |
