# 05 — Báo cáo rà soát hiện trạng và kế hoạch nâng cấp Artifact Pose System

**Ngày rà soát:** 05-09-2026  
**Phạm vi:** đọc tài liệu trước, sau đó đối chiếu source, schema, dữ liệu runtime, Docker/WSL và client.  
**Trạng thái:** chỉ phân tích và đề xuất; **chưa sửa mã nguồn, cấu hình, dữ liệu hay lịch sử Git**.

> Đây là tài liệu audit kỹ thuật, không phải cam kết rằng mọi chức năng đã được chạy end-to-end. Các kết luận được đánh dấu theo mức độ: **P0 — phải xử lý trước khi mở mạng/đưa vào vận hành**, **P1 — cần xử lý trước khi scale hoặc chạy liên tục**, **P2 — cải thiện chất lượng và chi phí bảo trì**.

## 1. Kết luận điều hành

Hệ thống có nền tảng tốt cho một prototype tích hợp phần cứng: Flutter client, FastAPI, PostgreSQL, MQTT/Mosquitto, Raspberry Pi agent, pose solver C++/pybind11 và pipeline SSIM/SIFT/YOLO đã được nối thành một luồng nghiệp vụ tương đối đầy đủ. Luồng ý tưởng là hợp lý:

```text
Flutter operator
      │ REST/JWT
      ▼
FastAPI ───── PostgreSQL (metadata)
  │  │
  │  └──── file uploads/model (filesystem volume)
  │
  ├──── MQTT broker ─── Raspberry Pi ─── camera + stepper/servo
  └──── pose C++/OpenCV + SSIM/YOLO (đang chạy trong process API)
```

Tuy nhiên, trạng thái hiện tại phù hợp với **demo nội bộ có kiểm soát**, chưa phù hợp với mạng không tin cậy, nhiều worker hoặc dữ liệu lớn. Ba nhóm rủi ro lớn nhất là:

1. **An toàn điều khiển và bí mật (P0):** broker anonymous, nhiều endpoint điều khiển/upload không xác thực, device tự đăng ký bằng `machine_hash` tùy ý, dữ liệu ảnh được mount public, secret mặc định và file `.env` đang tracked.
2. **Tính nhất quán và độ tin cậy (P0/P1):** queue/ACK/status/alignment state nằm trong RAM; HTTP fallback được báo là đã queue nhưng Pi không hề poll; timeout cấu hình nhưng không được thực thi; MQTT publish chưa chờ PUBACK; restart hoặc scale sẽ làm mất/nhân đôi tác vụ.
3. **Khả năng tái lập và tính đúng của AI/CV (P1):** model runtime không có trong workspace, compose ghi đè quy tắc auto-scan; tác vụ CPU-bound chạy trong request; fallback pose trả motor command bằng 0; native binding có nguy cơ trả NumPy view trỏ vào bộ nhớ đã hết vòng đời; pipeline training có nguy cơ contamination/evaluation bias.

Nếu chỉ có thời gian cho một đợt nâng cấp, thứ tự nên là: **khóa ingress và quyền điều khiển → làm state/job bền vững → xác định rõ pipeline/model và thêm test hồi quy → sau đó mới scale/đẹp hóa UI**.

## 2. Phạm vi và phương pháp kiểm tra

### 2.1. Tài liệu đã đọc trước

Đã đọc theo thứ tự tài liệu thiết kế rồi mới kiểm tra code:

- [`docs/01_System_Architecture.md`](01_System_Architecture.md): topology, API, database và Docker.
- [`docs/02_Pose_Estimation_Module.md`](02_Pose_Estimation_Module.md): ChArUco, ORB, triangulation, G2O, fallback.
- [`docs/03_AI_Inspection_Module.md`](03_AI_Inspection_Module.md): SIFT/SSIM, heatmap, YOLO crop/NMS, model service.
- [`docs/04_Configuration_and_Scalability.md`](04_Configuration_and_Scalability.md): biến môi trường, mở rộng, MQTT và deployment.
- [`README.md`](../README.md), [`SYSTEM_INSTRUCTION.md`](../SYSTEM_INSTRUCTION.md), [`DEPLOY_INSTRUCTION.md`](../DEPLOY_INSTRUCTION.md): luồng vận hành, lệnh chạy, các giả định phần cứng.

Sau đó đối chiếu với các nhóm thực thi: `server/app`, `server/native`, `server/tools`, `embed/device_agent`, `client/artifact_app`, `ai_module`, `artifact_db.sql`, Docker/compose và file môi trường. Toàn bộ file text có tác động đến business logic/giao thức đã được đọc; các file generated/platform runner, asset nhị phân được kiểm tra ở mức build, permission và release.

### 2.2. Inventory tĩnh

Tại thời điểm rà soát, Git có **260 file tracked**. Một số số liệu để định lượng phạm vi:

`docs/` hiện xuất hiện là thư mục untracked trong working tree (gồm các tài liệu 01–05); đây là trạng thái có sẵn của workspace, không phải thay đổi source/config do lần audit này tạo ra.

| Khu vực | Số file | Quy mô đo được | Vai trò |
|---|---:|---:|---|
| `server/` | 125 | 55 Python (~5.799 dòng); 11 C/C++ header/source (~1.399 dòng) | API, ORM, workflow, pose, AI, MQTT, Docker |
| `embed/device_agent/` | 18 | 8 Python (~1.433 dòng) | agent, camera, servo/stepper, MQTT, upload |
| `client/artifact_app/lib/` | 44 | 44 Dart (~8.208 dòng) | màn hình, provider, API client, workflow |
| `ai_module/` | 8 | 3 Python (~1.148 dòng); 2 notebook | preprocessing, thử nghiệm/training |
| Tài liệu/schema | 5 docs + README/hướng dẫn/SQL | Markdown và SQL | hợp đồng kiến trúc và dữ liệu |

Các file scaffold nhị phân/ảnh và project runner của Flutter được lập inventory; phân tích sâu tập trung vào phần có tác động đến giao thức, bảo mật, dữ liệu và runtime.

### 2.3. Kiểm tra môi trường và tính tái lập

- Python host: **3.13.15**; đã parse AST toàn bộ **66 file Python (~9.986 dòng)**, không phát hiện lỗi cú pháp.
- Docker client: **29.7.2**; Docker Compose: **v5.3.1**; `docker compose --env-file server/.env.docker -f server/docker-compose.yml config --quiet` hợp lệ.
- WSL mặc định được xác định là Ubuntu 22.04/WSL2 theo trạng thái đã kiểm tra; một số lệnh enumerate bị `E_ACCESSDENIED` trong phiên Windows nên không chạy build trong distro.
- Host không có Flutter, Dart, CMake, GCC/G++; không thể xác nhận build native hoặc `flutter analyze/test/build`.
- Không có container `artifact_*` đang chạy trong lần kiểm tra daemon trước đó (chỉ thấy container Supabase không thuộc dự án); ở phiên hiện tại `docker ps` đôi lúc bị Windows named-pipe `Access denied`. Vì vậy chưa có benchmark/live smoke test.
- Không có thư mục `model/`; chỉ thấy file calibration `server/data/camera_params/camera_params_lens_1.5.yaml`. Model `.pt` cần cho AI không thể xác nhận.
- Không có `.github/`, pytest config hay test suite tự động; các script `server/tools/test_inspect_pipeline*.py` là CLI/manual và có đường dẫn mẫu hardcode.

### 2.4. Snapshot cấu hình Compose đã resolve

`docker compose config` được chạy ở chế độ chỉ đọc với `server/.env.docker`. Các giá trị đáng chú ý (secret được cố ý không ghi lại) là:

| Hạng mục | Giá trị runtime quan sát | Nhận xét |
|---|---|---|
| API bind | `0.0.0.0:8000` | cần reverse proxy/TLS và network policy khi ra khỏi localhost |
| CORS | `*` | quá rộng cho production |
| PostgreSQL | service `postgres`, host port `5432` | không cần publish nếu chỉ server dùng |
| MQTT | service `mosquitto:1883`, host port `1883` | broker config anonymous |
| AI model | `DEFAULT_AI_MODEL_PATH=best_Quyen.pt` | khác quy tắc auto-scan; file không có trong workspace |
| Alignment | `MAX_ALIGNMENT_ITERATIONS=7`, `ALIGNMENT_TIMEOUT_SEC=300` | iteration có hiệu lực; timeout chưa được thực thi |
| Pose/hardware | `TRANS_TOLERANCE_MM=25.0`, `ROT_TOLERANCE_DEG=2.0`, `STEPS_PER_MM=800.0`; sign compose là `1` | các lớp khác còn default/hardcode khác |
| Volume | `./data:/app/data`, `../model:/app/model` | dữ liệu/media và model nằm ngoài image, cần quyền/retention |
| Bootstrap | admin/DB/JWT lấy từ env hoặc default compose | phải fail-fast nếu còn giá trị mẫu |

## 3. Kiến trúc hiện tại và luồng dữ liệu

### 3.1. Thành phần

| Thành phần | Entry point/đường dẫn | Trách nhiệm hiện tại | State/điểm phụ thuộc |
|---|---|---|---|
| Flutter | `client/artifact_app/lib/` | đăng nhập, artifact, device workflow, schedule, kết quả | JWT trong secure storage; URL mặc định hardcode |
| API | `server/app/main.py`, `server/run.py` | REST, auth, ORM, upload, orchestration | một process/container; state service trong RAM |
| DB | `server/app/models/`, `artifact_db.sql` | user, artifact, image, comparison, alert, schedule, device | startup `Base.metadata.create_all()` |
| MQTT | `server/app/services/mqtt_bridge.py`, Mosquitto | command, ACK, heartbeat/status | broker anonymous; JSONL log |
| Edge | `embed/device_agent/runtime/main_app.py` | nhận lệnh, camera, servo/stepper, upload | trạng thái cơ cấu/task journal trong RAM |
| Pose | `server/app/modules/artifact_pose/`, `server/native/pose_solver_cpp/` | ChArUco, ORB, stereo, G2O, deviation | calibration/golden pose trên filesystem |
| AI | `server/app/services/inspection_service.py`, `model_service.py`, `ai_module/` | SIFT/SSIM/heatmap/YOLO và preprocessing | model load trong process; không có registry bền vững |

### 3.2. Luồng khởi tạo golden pose

1. Operator gọi workflow có JWT.
2. Server publish `capture_stereo_pair` qua MQTT với baseline cố định 100 mm/80.000 steps.
3. Pi chụp trái, chạy slider, chụp phải, quay về; upload cặp ảnh.
4. Server đọc calibration, marker, ORB, triangulation, PnP/hybrid; ghi `uploads/golden_poses/{artifact_id}/golden_pose.yaml` và ảnh trái làm baseline DB.

### 3.3. Luồng alignment/inspection

1. Server publish lệnh capture/move.
2. Pi chụp và POST `/inspections/upload`.
3. `InspectionService.handle_upload()` lưu file, chạy pose correction đồng bộ, cập nhật metadata RAM, rồi có thể publish lệnh di chuyển tiếp.
4. Route inspection đọc ảnh, chạy SIFT/SSIM/heatmap và YOLO; kết quả ghi `images`, `image_comparisons`, `alerts`.
5. Flutter polling để xem status/ảnh/kết quả.

Luồng này không có durable job ID/state machine: response HTTP phụ thuộc thời gian chạy CV và trạng thái process hiện tại.

### 3.4. Ma trận lệch giữa tài liệu, cấu hình và runtime

| Chủ đề | Tài liệu/ý định | Runtime quan sát được | Việc cần chốt |
|---|---|---|---|
| Auto-load model | `docs/03` và `env.docker.example` nói để trống path sẽ quét `model/*.pt` | `docker-compose.yml:54-55` luôn ghi `best_Quyen.pt`; thư mục `model/` hiện vắng | chọn một model contract, kiểm tra tồn tại và checksum khi readiness |
| AI lúc upload | `RUN_AI_ON_UPLOAD`, `RUN_AI_ON_ALIGNED_IMAGE` được mô tả là feature flag | `handle_upload()` lưu ảnh/pose rồi trả `ai_result=None`; flag không điều khiển nhánh này | tách ingest/job hoặc triển khai flag thật, có test matrix |
| Pose lúc upload | `RUN_POSE_ON_UPLOAD` được khai báo | pose correction hiện bị gọi trực tiếp trong `handle_upload()`, chưa thấy guard theo flag | bỏ flag chết hoặc dùng nó như policy rõ ràng |
| Timeout | docs/config nói alignment có hard timeout | `_alignment_start_ts` không được khởi tạo/so sánh; chỉ giới hạn iteration | deadline phải là state của session, có terminal timeout |
| Golden pose | docs cũ có đường dẫn global/per-artifact khác nhau | code ưu tiên `uploads/golden_poses/{artifact_id}/...`, có migration fallback | công bố một canonical path và backup contract |
| Baseline phần cứng | docs mô tả 100 mm/800 steps/mm/80.000 steps | route, Pi và C++ cùng hardcode ở các lớp khác nhau | hardware profile versioned, không lặp constant |
| Identity | docs dùng lẫn `device_id`/`device_code` | DB PK 6 ký tự khác MQTT/API code; upload có chỗ gửi code làm FK | DTO/DB contract riêng cho PK, code và topic identity |
| HTTP fallback | API response có `http_queue_fallback` | Pi runtime chỉ chạy MQTT; method `receive_move_command()` không được gọi | triển khai poll có auth hoặc loại bỏ mode |
| Scheduler | CRUD schedule đã có trong docs/API | chưa có worker thực thi; route inspection còn tự tạo lịch | scheduler service + idempotent open schedule |
| WSL/ADB | docs tham chiếu `scripts/wsl_adb_setup.sh` | file không tồn tại trong inventory | thêm script được kiểm thử hoặc sửa tài liệu |

## 4. Bảng phát hiện ưu tiên

Các đường dẫn rút gọn như `devices.py`, `artifacts.py`, `pose.py`, `users.py` trong bảng dưới đây lần lượt là file cùng tên dưới `server/app/api/routes/`; `inspection_service.py`, `command_service.py`... là dưới `server/app/services/`, trừ khi có tiền tố khác ghi rõ.

| ID | Mức | Phát hiện | Bằng chứng | Tác động trực tiếp |
|---|---|---|---|---|
| SEC-01 | **P0** | MQTT cho anonymous và bind toàn bộ interface | `server/docker/mosquitto/mosquitto.conf:1-4` | bất kỳ máy trong mạng có thể publish command/giả ACK/status |
| SEC-02 | **P0** | endpoint move/status/acks/queue và upload không yêu cầu auth | `server/app/api/routes/devices.py:133-195`, `server/app/api/routes/inspections.py:12-24` | điều khiển cơ cấu, đọc trạng thái, đẩy CPU/disk bằng client không tin cậy |
| SEC-03 | **P0** | enrollment công khai, tin `machine_hash`/preferred ID tùy ý | `devices.py:106-130`, `device_registry.py:62-86`; golden pose cho registered device tại `pose.py:71-87` | attacker tự đăng ký rồi upload baseline/được coi là device hợp lệ |
| SEC-04 | **P0** | secret/credential mặc định và file môi trường tracked | `server/docker-compose.yml:7-11,47,83-95`; `server/tools/create_admin.py:35-48`; `git ls-files` gồm `server/.env`, `.env.docker`, `embed/device_agent/.env` | lộ DB/JWT/admin; restart có thể reset admin; cần rotate và xử lý lịch sử Git |
| SEC-04b | **P0** | Postgres, MQTT và API đều publish trực tiếp ra host; không có network segmentation | `server/docker-compose.yml:10-11,25-26,86-90` | mở rộng bề mặt tấn công và cho phép truy cập DB/broker ngoài ý muốn |
| SEC-05 | **P0** | upload không giới hạn kích thước/format/dimension; `artifact_id` đi thẳng vào path | `inspection_service.py:45-58,64-78,659-672`; `pose.py:28-33` | memory/disk exhaustion, decompression bomb, path traversal, file rác |
| SEC-06 | **P0** | `/uploads` và `/mqtt/events` công khai | `server/app/main.py:47-52`; `health.py:31-63` | lộ baseline/ảnh tổn hại/heatmap và dữ liệu vận hành |
| REL-01 | **P0/P1** | queue/ACK/status/latest capture/alignment chỉ ở RAM | `state.py:14-28`; `command_service.py:11-85`; `inspection_service.py:38-43` | restart/scale mất state; kết quả không nhất quán giữa worker |
| REL-02 | **P1** | HTTP fallback được queue nhưng Pi chỉ chạy MQTT | `devices.py:143-155`; `api_client.py:37-59`; `runtime/main_app.py:958-974` | API trả `http_queue_fallback` nhưng lệnh không bao giờ tới phần cứng |
| REL-03 | **P1** | `ALIGNMENT_TIMEOUT_SEC` được đọc nhưng deadline không được set/check | `config.py:115-116,220`; `inspection_service.py:694-706`; `workflows.py:150-171` | alignment sparse upload có thể chạy vô hạn, treo phần cứng |
| REL-04 | **P1** | publish chỉ kiểm tra `rc`, không chờ PUBACK/retry/outbox; initial connect failure không có backoff rõ ràng | `mqtt_bridge.py:91-120,31-60` | API báo thành công nhưng message có thể mất; broker khởi động chậm không tự hồi phục |
| REL-05 | **P1** | heartbeat chỉ cập nhật state RAM, không cập nhật `IotDevice.status/last_active_at` | `devices.py:75-96`; `mqtt_bridge.py:155-160` | restart làm thiết bị offline và timestamp DB stale; dashboard không có lịch sử tin cậy |
| PERF-01 | **P1** | SIFT/SSIM/YOLO/pose CPU-bound chạy trong request/async | `artifacts.py:175-220`; `inspections.py:12-24`; `pose.py:54-68` | block event loop, timeout client 30 s, duplicate retry |
| DATA-01 | **P1** | ID random 6 hex và contract ID/FK mâu thuẫn | `models/artifact.py:19-22,44`; `models/iot_device.py:13-24`; `artifact_db.sql:174-197`; workflow default `artifact_demo_001` | collision, `String(6)` không chứa giá trị workflow, device code không phải DB FK |
| DATA-02 | **P1** | schema SQL snapshot và ORM không có migration version làm nguồn sự thật | `database.py:45-48`; `models/user.py`; `artifact_db.sql:220-231` | deploy/rollback không kiểm soát; drift tương lai khó phát hiện |
| CV-01 | **P1** | native binding trả NumPy array alias bộ nhớ `cv::Mat` local | `server/native/pose_solver_cpp/bindings.cpp:50-69` | use-after-free/dữ liệu ngẫu nhiên sau khi hàm return |
| CV-02 | **P1** | fallback OpenCV tính deviation nhưng motor command luôn zero | `correction.py:112-169` | mất native extension thì UI vẫn báo deviation nhưng không thể tự căn chỉnh |
| CFG-01 | **P1** | `RUN_POSE_ON_UPLOAD`, `RUN_AI_ON_UPLOAD`, `RUN_AI_ON_ALIGNED_IMAGE`, `AUTO_DISPATCH_POSE_COMMAND` có khai báo nhưng usage không nhất quán | `server/app/core/config.py:93-126,198-227`; `inspection_service.py:659-747` | operator tin vào flag nhưng runtime hành xử khác; khó debug/safety review |
| AI-01 | **P1** | broad exception giữ kết quả rỗng, classifier có thể coi là `good` | `inspection_service.py:464-467`, `_classify_damage_status():203-208` | lỗi model/ảnh bị che thành false negative |
| AI-02 | **P1** | training bridge test→train và chuyển train→test; split multilabel theo primary class | `ai_module/preprocess_2stage.py:309-400,207-234` | leakage/evaluation bias, metric không đại diện deployment |
| AI-03 | **P1** | server chỉ load một detector YOLO; chưa orchestration stage-2 classifier | `model_service.py:58-84`; `ai_module/preprocess_2stage.py:15-20` | mô hình được train khác pipeline runtime |
| EDGE-01 | **P1** | command thread mới cho mỗi message; task journal TTL chỉ RAM | `runtime/main_app.py:419-436,92-130` | flood tạo thread; QoS1 redelivery sau restart có thể chạy motor hai lần |
| OPS-01 | **P1** | không có CI, dependency lock/hash, health/readiness/resource/log policy | `server/requirements*.txt`, `Dockerfile:17-67`, không có `.github/` | khó tái lập, khó phát hiện regression, disk/RAM tăng không kiểm soát |
| OPS-02 | **P1** | README nói fallback SQLite nhưng `server/.env` tự động ép PostgreSQL localhost | `server/app/core/database.py:26-33`; `README.md` phần Database | chạy local không có Postgres có thể fail thay vì fallback; dev/prod profile bị lẫn |
| APP-01 | **P2/P1** | URL HTTP hardcode, cleartext cho Android/iOS, release ký debug | `api_config.dart:4-14`; Android manifest:6-10; iOS `Info.plist:52-56`; `build.gradle.kts:33-38` | JWT/ảnh trên plaintext; artifact release không an toàn |

## 5. Phân tích chi tiết

### 5.1. Bảo mật và biên điều khiển (P0)

#### a) Chuỗi tin cậy của device chưa tồn tại

`POST /api/v1/devices/get_device_id` không có JWT, bootstrap secret, chữ ký hay chứng thực phần cứng. `DeviceRegistry.allocate_device_id()` chỉ chuẩn hóa chuỗi và ghi mapping JSON. Sau đó `pose.initialize_golden` chấp nhận request không JWT nếu `device_code` chỉ cần xuất hiện trong registry. Đây là trust escalation: self-registration → được coi là registered device → upload golden pose và ảnh baseline.

Thiết kế cần thay thế bằng quy trình enrollment một lần: mã bootstrap dùng một lần hoặc mTLS/device certificate; lưu device trong DB với trạng thái `pending/active/revoked`; cấp token riêng theo device; kiểm tra device–artifact authorization cho từng command/upload. Không dùng machine hash do client tự khai báo làm credential.

#### b) Các route nhạy cảm đang mở

Các route `queue_move`, `move` (pop queue), `status`, `acks`, `inspections/upload`, `mqtt/health`, `mqtt/events` không có `get_current_user` hoặc device credential. Đặc biệt `move` có thể làm mất lệnh fallback của thiết bị khác; `queue_move` nhận step/angle mà chưa có giới hạn an toàn ở schema/protocol. Cần tách:

- API operator: JWT + role/scope + audit log.
- API device: token/mTLS, ràng buộc token với `device_code`, nonce/sequence.
- API nội bộ: chỉ network nội bộ hoặc service identity.

#### c) Broker và transport

Mosquitto `allow_anonymous true`, listener `0.0.0.0:1883`; compose publish `5432`, `1883`, `8000` ra host. Payload ACK/status được tin theo topic; server không kiểm tra `payload.device_id` khớp topic và không có chữ ký/chống replay (`mqtt_bridge.py:142-166`). `device_code` còn được chèn vào topic từ input mà chưa có allowlist/wildcard validation. Cần bật username/password tối thiểu, ACL từng device; production dùng TLS/mTLS, topic allowlist, payload signature/nonce/sequence và reject message cũ.

#### d) Upload và media

`await file.read()` đọc toàn bộ body vào RAM; filename chỉ thay `/` và `\\`, còn `artifact_id` được dùng làm thư mục. Không kiểm tra Content-Length, MIME/magic bytes, kích thước pixel, quota, decode an toàn hay cleanup khi DB commit thất bại. `/uploads` mount `StaticFiles` khiến ảnh nhạy cảm trở thành URL public. Cần giới hạn body ở reverse proxy/app, streaming vào file tạm, decode bằng OpenCV để kiểm tra dimension, whitelist extension/magic, quota theo user/device, tên UUID, virus/decompression-bomb guard, private object storage và signed URL.

#### e) Auth người dùng và secret

Login/register không rate limit/lockout; password change không có min/max policy (`server/app/api/routes/users.py:17-20`), reset admin đặt password cố định (`users.py:182-194`). JWT chỉ có `sub/role/exp`, không có `iat/jti` hoặc token version/revocation (`server/app/core/security.py`). Compose có CORS `*`, default JWT secret và admin password trong command shell. File `.env` đã tracked dù `.gitignore` có rule; ignore không xóa dữ liệu đã nằm trong lịch sử. Việc cần làm là rotate ngay, purge lịch sử nếu repository đã chia sẻ, chuyển secret manager, fail-fast khi secret còn giá trị mẫu, reset password bằng token một lần và thêm session revocation.

### 5.2. State, giao thức và độ tin cậy (P0/P1)

`CommandService` giữ queue, ACK history, heartbeat/status, latest capture trong dict/list; `InspectionService` giữ counter/phase/deadline trong dict. `DeviceRegistry` ghi JSON trực tiếp, không atomic rename/fsync và `_load()` bỏ qua lỗi parse. Hậu quả là mất trạng thái khi restart, split-brain khi có nhiều worker và file registry có thể hỏng.

Ngay cả trong một process, các alignment dict không có transaction/per-session lock; hai upload gần đồng thời có thể tranh iteration/phase và phát lệnh ngược thứ tự. Durable state cần optimistic version hoặc lock theo `device_id + artifact_id`.

Khuyến nghị kiến trúc:

```text
API -> command_outbox (PostgreSQL) -> publisher worker -> MQTT
                                      └-> delivery_attempt/PUBACK
Pi ACK/status -> ingest + idempotency table -> device_state
alignment session -> DB/Redis (deadline, phase, iteration, version)
inspection upload -> job table -> bounded worker -> result/event
```

Mỗi command cần `command_id`/idempotency key, state `queued|sent|acked|failed|expired`, retry có backoff và giới hạn. ACK phải được deduplicate theo `command_id`, không chỉ append list.

HTTP fallback có hai lựa chọn rõ ràng: (1) implement authenticated long-poll/replay trong Pi và durable queue; hoặc (2) bỏ mode này, trả lỗi `503` khi MQTT không sẵn sàng. Không nên tiếp tục trả `ok=true` cho đường dẫn mà agent không sử dụng.

`MqttBridge.start()` chỉ connect một lần; compose chỉ đợi Mosquitto `service_started`, không có healthcheck broker. Cần startup readiness, reconnect/backoff, chờ PUBACK và kiểm thử broker restart. `/health` hiện luôn trả `status=ok` dù MQTT/model/DB có thể chưa sẵn sàng (`health.py:22-28`); tách liveness (process sống) và readiness (DB, broker, model, camera params hợp lệ).

Heartbeat MQTT hiện chỉ được ghi vào `CommandService` trong RAM. `list_devices()` dùng giá trị RAM để suy ra online nhưng không persist `last_active_at`/status vào `iot_devices`; sau restart dashboard mất lịch sử và mọi thiết bị trở về trạng thái DB cũ. Nên có bảng/device-state transaction, timestamp server-side, cửa sổ offline và sweeper định kỳ.

`ALIGNMENT_TIMEOUT_SEC` và `_alignment_start_ts` là cấu hình/biến có tên đúng nhưng không có assignment hoặc phép so sánh thời gian. Chỉ có `current_iter > max_iter`, nên một workflow có thể kéo dài vô hạn nếu ảnh đến thưa. Deadline phải được tạo lúc `start-alignment`, lưu bền vững, kiểm tra trước mỗi transition và phát terminal event/stop command khi hết hạn.

### 5.3. Database, contract và lifecycle dữ liệu

#### a) ID và foreign key

ORM sinh ID bằng `secrets.token_hex(3)` (24 bit). Birthday collision sẽ trở nên đáng kể khi số bản ghi tăng; code cũng không retry khi collision. Nhiều workflow dùng `artifact_demo_001`, dài hơn `String(6)`. `Image.device_id` trỏ `iot_devices.device_id` nội bộ 6 ký tự, trong khi upload thường truyền `device_code`; `inspect-from-device` không truyền `device_id` vào `run_artifact_inspection`, nên liên kết thiết bị có thể bị null.

Nên chuyển sang UUID/ULID (hoặc ID lớn hơn), giữ `device_code` là unique business key và luôn resolve sang DB PK trước khi ghi `Image`. Thêm contract test kiểm tra độ dài/foreign key giữa ORM, migration và SQL dump.

#### b) Schema và transaction

Startup gọi `Base.metadata.create_all()` thay vì migration versioned. ORM biểu diễn `UserRole` bằng `Enum(..., native_enum=False)` nên hiện tại có thể lưu dưới dạng string giống SQL dump, nhưng không có migration/constraint/version nào bảo đảm các thay đổi sau này đồng bộ. Dùng Alembic, migration forward/backward, schema diff trong CI; không coi SQL dump tĩnh là nguồn sự thật duy nhất.

#### c) Media orphan và quyền riêng tư

Xóa artifact chỉ xóa DB (`artifacts.py:251-257`), không xóa `uploads/artifacts/{id}`/golden pose. Upload reference ghi file trước khi commit DB (`artifacts.py:260-281`), lỗi transaction để lại file orphan. Cần media manifest, transaction/outbox cleanup, retention/legal hold, checksum và job garbage collector; signed URL thay cho static public mount.

#### d) Query và scheduler

List artifacts/users/devices/schedules dùng `.all()` không pagination; `inspections` nhận `limit` không có `Query(ge/le)` và chạy thêm `count()` full scan (`artifacts.py:285-302`). Thêm cursor pagination, limit cứng, index `(artifact_id, created_at)`, foreign-key indexes và query plan. CRUD schedule hiện chỉ lưu lịch; chưa có worker thực thi. Route inspection còn tự tạo schedule mới mỗi lần nếu interval > 0 (`artifacts.py:205-216`), có nguy cơ nhân lịch; cần unique open schedule/idempotency và scheduler worker riêng.

#### e) Kiểu dữ liệu/kết quả

`ssim_score` lưu `String(16)` thay vì numeric; `damage_score` là Float nhưng API có chỗ ép int. Khi không có baseline, `previous_image_id` bị gán bằng current image (`inspection_service.py:131-137`), làm sai nghĩa audit. Chuẩn hóa metric thành numeric, nullable rõ ràng, `baseline_missing` là trạng thái riêng và không tự tạo self-comparison.

### 5.4. Hiệu năng và khả năng mở rộng

- `inspect_artifact` là `async def` nhưng gọi sync DB + `run_artifact_inspection()`; `pose_correct` gọi C++/OpenCV trực tiếp. Một request 4K có thể chiếm event loop nhiều giây.
- SIFT full-resolution, SSIM ba kênh, heatmap, nhiều tight/wide crop và YOLO `imgsz=960` tạo peak CPU/RAM; upload đọc cả body vào RAM.
- `ModelService` chỉ lock registry, không có semaphore giới hạn inference; admin có thể load nhiều model lớn/đường dẫn tùy ý (`models.py:38-63`, `model_service.py:227-260`).
- Queue list dùng `pop(0)` O(n); JSONL MQTT/inspection không rotation/retention, `/mqtt/events` đọc toàn bộ file vào RAM.
- Filename dựa timestamp millisecond có nguy cơ collision khi request song song.

Kiến trúc tạm thời có thể dùng `run_in_threadpool` + semaphore, nhưng giải pháp bền vững là job queue bounded (Redis/RQ/Celery hoặc worker PostgreSQL), response `202 + job_id`, polling/event stream và backpressure. Cần đo p50/p95, queue depth, CPU/RAM, kích thước ảnh trước khi chọn worker count.

### 5.5. Pose solver, native ABI và phần cứng

#### a) Binding lifetime

Trong `bindings.cpp:64-69`, `py::array_t` được tạo với con trỏ `qr.descriptors.data` của `cv::Mat` nằm trong `QuadTreeResult` local. Sau khi hàm return, NumPy có thể trỏ vào vùng đã bị giải phóng/reused. Phải copy sang buffer sở hữu bởi Python (hoặc capsule giữ owner), thêm test lặp/ASAN/UBSan và kiểm tra shape rỗng.

`common.py:17-65` còn quét nhiều thư mục hệ thống, `ctypes.CDLL()` mọi thư viện có tên `libg2o*` rồi nuốt `OSError`. Cách này giúp chạy được trên nhiều máy nhưng có thể nạp nhầm ABI/version và làm lỗi native trở nên khó chẩn đoán. Nên dùng một đường dẫn/manifest thư viện đã kiểm tra, log rõ lý do load fail và chạy `ldd`/ABI smoke test trong CI.

#### b) Fallback không hoàn chỉnh

`correction.py:129-169` fallback OpenCV tính deviation nhưng đặt toàn bộ `motor_command` bằng 0. Tài liệu mô tả fallback có thể tiếp tục correction, nhưng thực tế chỉ cung cấp read-only deviation. Cần hoặc nói rõ degraded mode (không tự điều khiển), hoặc triển khai conversion command dùng chung với native và test sai số/giới hạn servo.

ORB matcher hiện chủ yếu dùng Lowe ratio/Hamming; chưa thấy mutual-consistency và geometric RANSAC đủ chặt trước khi đưa điểm vào solver. Stereo triangulation cũng dựa baseline ngang cố định, nên cần epipolar/reprojection/depth validation và reject ratio thấp để tránh pose hợp lệ giả.

#### c) Constants/config drift

Baseline stereo 100 mm, 800 steps/mm, 80.000 steps xuất hiện ở `workflows.py:18-20`, Pi runtime và C++ default. Sign motor trong Settings không được áp dụng thống nhất; tolerance được đọc lúc import trong `common.py:83-89`, nên thay env sau import không có tác dụng. Gom calibration thành một versioned hardware profile, truyền cùng command và ghi profile version vào từng result.

#### d) An toàn actuator

Schema `MoveCommand` cho phép step/angle rộng; cần clamp travel, servo range, tốc độ, thời gian chạy, emergency stop và watchdog. Vị trí slider chỉ ở RAM (`hardware_controller.py:51-52`), restart có thể không biết vị trí vật lý; cần homing/limit switch hoặc journal vị trí có xác nhận. Mỗi MQTT message hiện tạo thread mới (`runtime/main_app.py:429-432`); thay bằng bounded executor một hàng đợi command và reject khi đầy.

### 5.6. AI, dữ liệu và tính đúng của kết quả

#### a) Runtime AI không khớp tài liệu/config

`RUN_AI_ON_UPLOAD` và `RUN_AI_ON_ALIGNED_IMAGE` được khai báo ở `config.py:93-126`/compose nhưng `handle_upload()` trả `ai_result=None` và không dùng các flag. Compose hardcode `DEFAULT_AI_MODEL_PATH=best_Quyen.pt` (`docker-compose.yml:54-55`) trong khi docs mô tả để trống thì auto-scan; thư mục model hiện không tồn tại. `AUTO_DISPATCH_POSE_COMMAND` cũng cần kiểm tra usage thực tế. Cần chọn một contract rõ: upload chỉ ingest, alignment/inspection là job riêng; hoặc thực sự thực thi flag và có test matrix.

#### b) False-good khi pipeline lỗi

`_analyze_against_reference()` bắt rộng exception (`inspection_service.py:464-467`), để detections/SSIM rỗng; classifier không có candidate trả `good` (`:203-208`). Lỗi model, thiếu baseline, decode lỗi phải trả `analysis_failed`/`unknown` với error code và quality flag, không được hạ xuống `good`.

#### c) Chất lượng CV

SIFT thiếu inlier/homography sẽ fallback resize nhưng không ghi cờ `alignment_quality`; SSIM khi đó có thể bị diễn giải như so sánh chính xác. NMS hiện merge tight/wide theo IoU nhưng không tách class, có thể loại nhầm hai loại damage chồng lấp. Cần lưu `inlier_count`, reprojection error, valid mask ratio, source (`aligned|resize`), model checksum, threshold set và NMS class-aware; xây bộ ảnh golden với nhãn expected.

#### d) Training/evaluation

`preprocess_2stage.py` chủ động bridge một phần ảnh từ test vào train/val (`:309-356`), rồi chuyển ảnh train sang test để đủ class (`:369-394`). Split multilabel dùng “primary class” (`:207-234`) chứ không group theo artifact/source/session. Đây có thể là quyết định nghiên cứu có chủ đích để bridge domain, nhưng không được gọi là test độc lập. Tách `train`, `validation`, `public test`, `locked holdout`; manifest hash/source/group; multilabel iterative stratification; leakage check theo ảnh gốc/augmentation; log experiment và freeze dataset version.

`ENABLE_ENHANCEMENT=True` được bật trực tiếp trong preprocessing nhưng chưa có manifest/ablation report để biết CLAHE, sharpen và HSV boost làm thay đổi metric ra sao; parser cũng sửa/clamp một phần tọa độ label trước khi ghi output. Cần lưu raw label, normalized label, transform và lý do loại/move từng mẫu để tái lập và audit.

Stage 1 detection + Stage 2 classification được tạo ra ở preprocessing nhưng server runtime hiện load một YOLO detector. Cần pipeline orchestration rõ (detector → crop/normalizer → classifier), hoặc đánh lại tên/metric để không tuyên bố chạy two-stage khi production chỉ chạy stage 1.

Runtime đặt `imgsz=960` trong `model_service.py`, trong khi kích thước/augmentation/training arguments nằm rải ở notebook và preprocessing; chưa có model card ghi input size, normalization, confidence/NMS threshold và calibration. Đây là contract cần đóng băng cùng model checksum.

#### e) Model supply chain

Admin có thể truyền path tuyệt đối vào `/models/load`; loader `.pt`/TorchScript có khả năng deserialize nội dung thực thi và không có checksum/size/registry. Chỉ cho phép path dưới model mount read-only, model ID đã đăng ký, SHA-256/signature, giới hạn số model/size và semaphore inference.

### 5.7. Raspberry Pi agent

- `embed/device_agent/` không có `requirements.txt`/`pyproject`; docs hướng dẫn cài dependency nhưng không khóa version/system packages (`picamera2`, `RPi.GPIO`, `adafruit-circuitpython-servokit`, `requests`, `paho-mqtt`).
- Main loop có reconnect tốt hơn server, nhưng `_refresh_device_id_from_server()` rebuild MQTT client chưa thể hiện cleanup client cũ rõ ràng.
- Task dedup `ExpiringTaskIdStore` chỉ RAM, TTL 180 s; QoS1 redelivery sau restart có thể lặp motor/capture.
- ACK/status publish không có local durable spool hoặc xác nhận delivery; mất mạng đúng lúc sau khi actuator đã chạy có thể khiến server không biết kết quả.
- Camera mở/đóng Picamera2 cho từng capture, tạo latency; cần benchmark trước khi giữ pool camera mở.
- Agent dùng `print` chủ yếu, chưa có systemd unit/watchdog/log rotation/health reporting.

### 5.8. Flutter/mobile

- `api_config.dart:4-14` hardcode `192.168.1.169` và HTTP; cần build flavor/dev override, không nhúng host LAN vào release.
- Android bật `usesCleartextTraffic=true`; iOS bật `NSAllowsArbitraryLoads=true`. Production phải HTTPS, certificate policy/pinning nếu phù hợp.
- `ApiClient.timeout` mặc định 30 s (`api_client.dart:17-29`), ngắn hơn pipeline pose/inspection; retry mù có thể tạo inspection trùng. Dùng job ID/idempotency và retry có điều kiện.
- `ArtifactService.alerts()` gọi `/api/v1/artifacts/alerts` (`artifact_service.dart:25-31`) nhưng backend không có route tương ứng.
- Android release đang ký debug key (`build.gradle.kts:33-38`); chưa đủ điều kiện phân phối production.
- Pubspec dùng caret ranges, chưa có Flutter/FVM pin và không có CI analyze/test/build dù đã có `pubspec.lock`.

### 5.9. Build, Docker và vận hành

`Dockerfile` clone g2o từ Git tag và `pip install` dependency range; tag không phải immutable commit/digest. Runtime cài apt OpenCV đồng thời pip `opencv-contrib-python`, có nguy cơ ABI/duplicate library; container không có `USER` non-root. `CMakeLists.txt` không đặt rõ `CMAKE_BUILD_TYPE` cho module pybind (chỉ bước build g2o có `Release`), nên build native local có thể không tối ưu. Compose thiếu healthcheck server/MQTT, resource limits và log rotation. `server/.dockerignore` loại `.env` và `env.docker.example` nhưng không loại rõ `.env.docker`; kết hợp với `COPY . .`, file môi trường tracked có nguy cơ đi vào image build context/layer. Chỉ Postgres có healthcheck; server phụ thuộc `service_started` của Mosquitto.

Không có CI cho Python lint/type/test, native CMake/import/ABI, Docker smoke, security scan/SBOM hay Flutter. Đây là nút thắt lớn hơn việc tối ưu vài dòng CV vì regression hiện chỉ được phát hiện thủ công.

## 6. Các phần dư thừa và technical debt nên gom lại

| Nhóm | Hiện trạng | Cách tinh gọn đề xuất |
|---|---|---|
| Config flags | nhiều `RUN_*`, `AUTO_DISPATCH_*`, sign/tolerance không có test hoặc usage nhất quán | tạo config contract machine-readable; mỗi flag có owner, default, test; xóa flag chết |
| Pose constants | baseline/tolerance/steps/sign nằm ở docs, route, common, Pi, C++ | một hardware profile versioned, inject vào mọi layer |
| DTO/schema | schema auth/user và tên `device_id`/`device_code` chồng lấn | API DTO versioned; phân biệt PK, business code, MQTT identity |
| Inspection pipeline | logic trong service, script manual và `ai_module/analyze_damage.py` trùng/khác | package pipeline dùng chung + adapter CLI; một nguồn threshold |
| Training notebooks | hai notebook training gần cùng mục đích | lưu experiment config/manifest; archive notebook cũ |
| Storage | ảnh, heatmap, aligned/final và JSONL tăng vô hạn | media manifest/object storage + retention/GC + log rotation |
| Deployment | env tracked, docs tham chiếu `scripts/wsl_adb_setup.sh` nhưng file không có | profile dev/staging/prod, script tồn tại hoặc bỏ khỏi docs, secret injection |
| Scheduler | CRUD lịch nhưng chưa có executor | worker scheduler và trạng thái job rõ ràng; không auto-create trùng |

## 7. Kiến trúc mục tiêu đề xuất

Không cần đổi toàn bộ ngay. Mục tiêu hợp lý là tách các boundary sau:

```text
                 ┌──────────────────────────┐
Flutter ─HTTPS──►│ API + Auth/RBAC + DTO     │
                 └──────────┬───────────────┘
                            │ job_id / outbox
              ┌─────────────┼────────────────┐
              ▼             ▼                ▼
        PostgreSQL       Redis/queue       Object storage
        metadata +       bounded jobs      private media
        command state          │
                               ▼
                    Pose/Inspection workers
                               │
                        MQTT publisher
                               │ TLS/ACL
                               ▼
                           Pi agent
```

Nguyên tắc:

1. API nhận/validate nhanh, trả `202` + `job_id`; worker CPU-bound xử lý.
2. PostgreSQL là nguồn sự thật cho command/job/result; Redis chỉ làm queue/cache có TTL.
3. MQTT là transport, không phải database; mọi command có idempotency/delivery state.
4. Ảnh private, URL có hạn; metadata lưu checksum, dimensions, model/calibration version.
5. Mọi transition alignment là state machine có deadline, cancellation, actor và audit event.
6. Model và calibration là artifact bất biến có version/hash; runtime chỉ chạy artifact đã approve.

### 7.1. Mô hình dữ liệu đích (đề xuất)

Không cần tạo tất cả bảng trong một migration. Có thể đi theo các nhóm sau:

| Nhóm | Trường cốt lõi | Bất biến/quan hệ cần bảo đảm |
|---|---|---|
| `users`, `sessions` | UUID, username, role, password hash, token version, revoked_at | username unique; đổi mật khẩu tăng token version |
| `devices`, `device_credentials` | UUID PK, `device_code`, status, hardware profile, credential hash, last_seen_at | code unique; credential chỉ hash; revoke không xóa audit |
| `artifacts` | UUID, business code, status, baseline media/version | code không dùng làm filesystem path trực tiếp |
| `media_objects` | object key, SHA-256, MIME, bytes, width/height, owner, retention | object private; DB commit và cleanup có outbox |
| `commands`, `command_attempts` | command UUID, device UUID, idempotency key, payload schema/version, state, deadline | unique `(device_id, idempotency_key)`; ACK dedupe |
| `jobs`, `alignment_sessions` | job type, status, progress, phase, iteration, deadline, cancellation | transition hợp lệ; terminal state không quay lại pending |
| `inspections`, `detections` | input media, baseline version, model version, quality flags, numeric metrics | không self-compare; lưu `unknown/failed` riêng `good` |
| `model_versions`, `calibrations` | URI, checksum/signature, labels, input contract, approved_at | kết quả tham chiếu version bất biến |
| `audit_events` | actor/device, action, resource, request id, timestamp, outcome | append-only, retention/audit policy |

Định dạng API nên dùng `device_pk`, `device_code`, `mqtt_identity` với tên khác nhau; không tiếp tục truyền một trường `device_id` cho ba nghĩa. Mọi ảnh và kết quả nên mang `request_id`, `job_id`, `artifact_id`, `device_pk`, `model_version_id`, `calibration_version_id` để truy nguyên toàn bộ một lần kiểm tra.

### 7.2. Chiến lược di chuyển dữ liệu an toàn

1. Tạo migration bổ sung UUID/ULID và cột business code, giữ cột 6 ký tự trong giai đoạn tương thích.
2. Backfill mapping device/artifact/image bằng transaction và checksum; chạy foreign-key audit trước khi bật constraint mới.
3. Dual-write command/media metadata trong một phiên bản ngắn; đọc từ schema mới sau khi đối chiếu count/hash.
4. Chuyển file sang object key bất biến, giữ redirect/read-only window cho URL cũ; chạy orphan scanner trước khi xóa.
5. Chỉ sau khi replay/rollback test đạt mới bỏ cột/path cũ. Backup DB và media manifest trước mỗi bước.

## 8. Roadmap nâng cấp theo pha

### P0 — khóa rủi ro trước khi mở mạng hoặc demo với dữ liệu thật

1. Rotate/remove credential trong `.env*`, purge lịch sử Git nếu cần; fail startup với secret mẫu/weak admin password.
2. Tắt anonymous MQTT; ACL theo device, TLS/mTLS; không publish Postgres/MQTT ra mạng ngoài nếu không cần.
3. Bắt buộc auth cho device command/status/acks/upload; enrollment bootstrap một lần; token/certificate gắn device.
4. Validate/normalize `device_code`, `artifact_id`, filename; chặn traversal; giới hạn body, MIME, pixel, quota và rate.
5. Bỏ static public `/uploads`; dùng private storage/signed URL; thêm audit log cho command và baseline.
6. Giới hạn step/angle/travel/servo, emergency stop và reject command thiếu `command_id`/sequence.
7. Chọn contract rõ cho AI flags/model bootstrap; khi thiếu model, readiness phải `not_ready`, không silently chạy như đã sẵn sàng.

**Tiêu chí ra P0:** request không credential không thể move/upload; broker chỉ nhận client hợp lệ; file 50 MB/ảnh giả bị từ chối; secret mẫu làm container fail; ảnh không truy cập bằng URL public.

### P1 — độ tin cậy, hiệu năng và tính đúng

1. Alembic + UUID/ULID + schema contract; sửa FK device/artifact và metric types.
2. Durable command outbox/ACK/status, PUBACK/retry/dedup; bỏ hoặc triển khai HTTP fallback thật.
3. Job queue bounded cho pose/inspection/model; semaphore inference; response `202/job_id` và idempotency.
4. Persist alignment session/deadline/phase/iteration; timeout/cancel terminal event.
5. Sửa native descriptor ownership; ASAN/UBSan; synthetic pose/triangulation regression và fallback semantics.
6. Model registry (version/hash/labels/input/checkpoint), stage-1/2 orchestration; quality flags khi SIFT fallback/model lỗi.
7. Pi bounded executor, persistent task journal/sequence, homing/limit switch, watchdog và systemd.
8. Pagination/index/query bounds, media lifecycle, log rotation, heartbeat DB persistence và offline sweeper.

**Tiêu chí ra P1:** restart không mất command/ACK; broker restart giao mỗi command tối đa một lần theo `command_id`; alignment timeout đúng hạn; p95 request ingest không phụ thuộc thời gian inference; native test không có invalid memory; kết quả lỗi là `unknown/failed`, không phải `good`.

### P2 — tái lập, quan sát và trải nghiệm

1. Pin Python/Pi/Flutter dependencies, image digest, g2o commit; build SBOM và vulnerability scan.
2. CI/CD: ruff/mypy/pytest, CMake build/import/ldd, compose smoke, Flutter analyze/test/build, Trivy/SAST.
3. Metrics/tracing/log structured: queue depth, inference latency, model load, MQTT delivery, storage size, alignment convergence.
4. Tách dev HTTP/LAN và production HTTPS bằng flavor; ký release bằng CI secret, versioning app.
5. Dataset manifest/hash/provenance/group split, locked holdout, experiment tracking và calibration report.
6. Sửa route alerts/mobile contract, loại deprecation/warnings, hoàn thiện scheduler.

## 9. Kế hoạch kiểm thử và tiêu chí nghiệm thu

### 9.1. Contract và security

- Test matrix cho mọi route: unauthenticated, operator, admin, device token sai/đúng, device khác artifact.
- Fuzz `artifact_id`, `device_code`, filename, JSON metadata; assert không thể thoát khỏi upload root.
- Upload file rỗng, MIME giả, ảnh dimension cực lớn, body vượt quota; xác nhận 4xx và cleanup temp.
- MQTT ACL test: device A không publish/subscribe topic device B; replay nonce/sequence bị từ chối.
- Secret scan Git/image; kiểm tra container command line không chứa password.

### 9.2. Reliability/distributed state

- Restart API giữa các trạng thái queued/sent/acked; xác nhận command vẫn tồn tại và không chạy hai lần.
- Restart broker/network flap; đo retry/PUBACK và eventual failure rõ ràng.
- Hai API worker đồng thời cùng idempotency key; chỉ có một job/result.
- Alignment upload thưa hơn deadline; xác nhận stop command và trạng thái terminal.
- Heartbeat mất; DB chuyển offline sau cửa sổ xác định, không phụ thuộc RAM process.

### 9.3. CV/AI

- Bộ synthetic pose với transform biết trước; sai số translation/rotation có ngưỡng và test native/fallback.
- ASAN/UBSan + stress gọi `extract_with_quadtree` nhiều vòng để bắt dangling descriptor.
- Golden set có: marker thiếu, SIFT ít inlier, baseline lệch kích thước, ảnh hỏng, model thiếu, hai class chồng nhau.
- Snapshot expected cho `alignment_quality`, SSIM, damage status, detections và model/calibration hash.
- Dataset CI: duplicate perceptual hash, group leakage, class distribution, label bounds, locked holdout không bị bridge.

### 9.4. Performance/operations

- Benchmark ảnh 1K/4K, concurrency 1/2/4/8: p50/p95 ingest, pose, inspection, RSS/CPU.
- Load test upload/command với giới hạn queue; xác nhận backpressure 429/503 thay vì OOM.
- Disk quota/retention test cho media và JSONL; log rotation không mất event cần audit.
- Compose smoke từ database/broker cold start; readiness chỉ xanh khi dependency sẵn sàng.

### 9.5. Mobile/edge release

- Flutter analyze/test/build trên version pin; kiểm tra API contract (đặc biệt alerts, timeout, polling).
- Android/iOS production build bắt buộc HTTPS và signing key không phải debug.
- Pi install từ artifact requirements/systemd; test reboot, MQTT reconnect, duplicate QoS1 và emergency stop.

## 10. Giới hạn và giả định của lần rà soát

- Chưa chạy full pipeline vì workspace thiếu model `.pt`, dependency AI/native, camera thật và container dự án đang chạy.
- Chưa đo latency/throughput thực nghiệm; các bottleneck CPU/RAM là suy luận từ call graph và kích thước ảnh, cần benchmark ở P1.
- WSL/Docker có thể được kiểm tra ở mức client/config; không xác nhận behavior của kernel GPIO, Picamera2, PCA9685 hay broker production.
- Notebook và file scaffold platform được xem xét về pipeline/cấu hình liên quan; không coi ảnh/asset nhị phân là bằng chứng chất lượng model.
- Báo cáo không thay thế security penetration test, calibration metrology hoặc đánh giá model độc lập trên locked holdout.

## 11. Kết luận

Ý tưởng và phân rã module của dự án là đúng hướng, nhưng cần chuyển từ prototype “mọi thứ trong một API process + state RAM + filesystem public” sang hệ thống có boundary và hợp đồng rõ: **identity/device trust, durable job/state, private media, worker bounded, model/calibration versioning và test tự động**. Những nâng cấp này sẽ giảm rủi ro an toàn cơ khí, tránh mất lệnh khi restart, làm metric AI đáng tin hơn và giúp nhóm có thể tái lập build/deploy.

Trong lúc chưa thực hiện các mục P0, nên giới hạn hệ thống ở localhost/LAN tin cậy, dùng dữ liệu thử nghiệm, không dùng credential hiện tại cho môi trường thật và không cho phép client không kiểm soát tiếp cận broker hoặc endpoint điều khiển.
