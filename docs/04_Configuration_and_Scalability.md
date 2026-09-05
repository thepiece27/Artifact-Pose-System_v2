# 04 — Cấu hình & Khả năng Mở rộng

> **Đường dẫn nguồn:**  
> `server/app/core/config.py` — dataclass Settings tập trung  
> `server/docker-compose.yml` — injection biến môi trường Docker  
> `embed/device_agent/runtime/main_app.py` — cấu hình Pi agent

---

## 1. Kiến trúc Cấu hình

Server tuân theo pattern cấu hình **12-factor app**: mọi tham số runtime được inject dưới dạng biến môi trường. Dataclass `Settings` trong `config.py` là bất biến (`frozen=True`) và được khởi tạo một lần lúc startup từ sự kết hợp của:

1. Biến môi trường OS (ưu tiên cao nhất — dùng bởi Docker).
2. File `.env` trong thư mục gốc server (cho phát triển cục bộ).
3. Giá trị mặc định hardcoded (fallback cho tham số tuỳ chọn).

```python
@dataclass(frozen=True)
class Settings:
    app_name: str
    app_version: str
    data_dir: Path
    model_dir: Path
    run_pose_on_upload: bool
    run_ai_on_upload: bool
    auto_dispatch_pose_command: bool
    ...
```

Dataclass frozen đảm bảo tính bất biến cấu hình tại runtime — không có đường code nào có thể âm thầm thay đổi một setting sau khi khởi động.

> **Lưu ý quan trọng:** `docker-compose.yml` hardcode `AUTH_DATABASE_URL` và `MQTT_HOST` trực tiếp thay vì đi qua substitution `${VAR}`. Điều này ngăn file `.env` cục bộ server (chứa `127.0.0.1` cho phát triển local) vô ình ghi đè hostname nội bộ container (`postgres`, `mosquitto`).

---

## 2. Tham chiếu Đầy đủ Biến Môi trường

### 2.1 Ứng dụng

| Biến | Mặc định | Mô tả |
|------|---------|-------|
| `APP_NAME` | `IoT Artifact Server` | Tên hiển thị trong API response |
| `APP_VERSION` | `1.0.0` | Chuỗi semantic version |
| `APP_HOST` | `0.0.0.0` | Địa chỉ bind Uvicorn |
| `APP_PORT` | `8000` | Cổng bind Uvicorn |
| `CORS_ALLOW_ORIGINS` | `*` | Danh sách origin CORS phân cách bằng dấu phẩy |

### 2.2 Lưu trữ

| Biến | Mặc định | Mô tả |
|------|---------|-------|
| `DATA_DIR` | `/app/data` | Thư mục gốc chứa uploads, logs, thông số camera |
| `MODEL_DIR` | `/app/model` | Thư mục được quét tìm file mô hình `.pt` |

### 2.3 Cờ Tính năng

| Biến | Mặc định | Mô tả |
|------|---------|-------|
| `RUN_POSE_ON_UPLOAD` | `true` | Tự động chạy hiệu chỉnh tư thế trên mỗi ảnh từ thiết bị |
| `RUN_AI_ON_UPLOAD` | `false` | Tự động chạy kiểm tra SSIM+YOLO trên mỗi upload |
| `AUTO_DISPATCH_POSE_COMMAND` | `true` | Tự động publish lệnh MQTT sau khi giải tư thế |
| `RUN_AI_ON_ALIGNED_IMAGE` | `true` | Chạy AI trên ảnh đã căn chỉnh tư thế (thay vì ảnh thô) |

### 2.4 Cơ sở Dữ liệu

| Biến | Mặc định | Mô tả |
|------|---------|-------|
| `AUTH_DATABASE_URL` | — | URL SQLAlchemy đầy đủ (`postgresql+psycopg://...`) |
| `POSTGRES_DB` | `artifact_auth` | Tên database PostgreSQL |
| `POSTGRES_USER` | `artifact` | Người dùng PostgreSQL |
| `POSTGRES_PASSWORD` | `artifact123` | **Thay đổi trong môi trường production** |

### 2.5 Xác thực

| Biến | Mặc định | Mô tả |
|------|---------|-------|
| `AUTH_SECRET_KEY` | `CHANGE_ME_AUTH_SECRET` | Khoá ký HMAC-SHA256 — **bắt buộc phải thay đổi** |
| `AUTH_ALGORITHM` | `HS256` | Thuật toán ký JWT |
| `AUTH_ACCESS_TOKEN_EXPIRE_MINUTES` | `60` | Thời hạn JWT |
| `ADMIN_USERNAME` | `admin` | Tên đăng nhập admin bootstrap |
| `ADMIN_PASSWORD` | `123456` | Mật khẩu admin bootstrap — **thay trước khi deploy** |

### 2.6 MQTT

| Biến | Mặc định | Mô tả |
|------|---------|-------|
| `MQTT_HOST` | `mosquitto` | Hostname broker (tên container trong Docker) |
| `MQTT_PORT` | `1883` | Cổng broker |
| `MQTT_KEEPALIVE_SEC` | `60` | Khoảng thời gian keepalive MQTT (giây) |
| `MQTT_USERNAME` | _(trống)_ | Tên đăng nhập xác thực broker |
| `MQTT_PASSWORD` | _(trống)_ | Mật khẩu xác thực broker |
| `MQTT_QOS` | `1` | Cấp QoS mặc định (0=nhiều nhất một lần, 1=ít nhất một lần, 2=đúng một lần) |
| `MQTT_CMD_TOPIC_TEMPLATE` | `cmd/{device_id}` | Mẫu topic lệnh |
| `MQTT_ACK_TOPIC_TEMPLATE` | `ack/{device_id}` | Mẫu topic ACK |
| `MQTT_STATUS_TOPIC_TEMPLATE` | `status/{device_id}` | Mẫu topic heartbeat |
| `ACK_HISTORY_LIMIT` | `200` | Số bản ghi ACK tối đa giữ trong bộ nhớ mỗi thiết bị |

### 2.7 Tư thế & Căn chỉnh

| Biến | Mặc định | Mô tả |
|------|---------|-------|
| `ARTIFACT_POSE_ROOT` | `/app/app/modules/artifact_pose` | Đường dẫn chứa `.so` và golden pose |
| `ARTIFACT_CAMERA_PARAMS_DIR` | `/app/data/camera_params` | Thư mục chứa file YAML hiệu chỉnh theo từng lens |
| `ARTIFACT_LENS_POSITION` | `1.5` | Vị trí lens đang hoạt động (chọn `camera_params_lens_{pos}.yaml`) |
| `TRANS_TOLERANCE_MM` | `30.0` | Ngưỡng hội tụ tịnh tiến (mm) |
| `ROT_TOLERANCE_DEG` | `2.0` | Ngưỡng hội tụ quay (độ) |
| `STEPS_PER_MM` | `800.0` | Độ phân giải stepper (bước/mm) |
| `MAX_ALIGNMENT_ITERATIONS` | `20` | Số lần lặp hiệu chỉnh tối đa trước khi timeout |
| `ALIGNMENT_TIMEOUT_SEC` | `300` | Hard timeout cho toàn bộ workflow căn chỉnh (giây) |

### 2.8 Hiệu chỉnh Chiều Phần cứng

Cụm stepper và servo vật lý có thể bị đảo cực tuỳ thuộc vào hướng lắp đặt động cơ. Bốn biến này bù trừ mà không cần thay đổi code:

| Biến | Mặc định | Trục phần cứng |
|------|---------|--------------|
| `SIGN_MOVE_X` | `1` | Chiều stepper trục X (+1 hoặc -1) |
| `SIGN_MOVE_Z` | `1` | Chiều stepper trục Z (+1 hoặc -1) |
| `SIGN_ROTATE_PAN` | `1` | Chiều servo pan (+1 hoặc -1) |
| `SIGN_ROTATE_TILT` | `1` | Chiều servo tilt (+1 hoặc -1) |

### 2.9 Mô hình AI

| Biến | Mặc định | Mô tả |
|------|---------|-------|
| `DEFAULT_AI_MODEL_NAME` | `default` | Khoá registry cho mô hình được auto-load |
| `DEFAULT_AI_MODEL_PATH` | _(trống)_ | Đường dẫn tường minh; nếu trống, dùng `*.pt` đầu tiên trong `MODEL_DIR` |

---

## 3. Hệ thống Hiệu chỉnh Camera

### 3.1 File Hiệu chỉnh Theo Lens

Thông số nội tại camera thay đổi theo khoảng cách lấy nét (vị trí lens). Hệ thống duy trì bộ file YAML hiệu chỉnh được đặt tên theo vị trí lens:

```
data/camera_params/
├── camera_params_lens_1.5.yaml   ← active at ARTIFACT_LENS_POSITION=1.5
├── camera_params_lens_2.0.yaml
└── camera_params_lens_0.0.yaml   ← infinity focus
```

Mỗi file lưu theo định dạng hiệu chỉnh OpenCV:

```yaml
camera_matrix:
  rows: 3
  cols: 3
  data: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
dist_coeffs:
  rows: 1
  cols: 5
  data: [k1, k2, p1, p2, k3]
```

### 3.2 Thêm Vị trí Lens Mới

1. Chụp chuỗi hiệu chỉnh checkerboard tại vị trí lens mới.
2. Chạy `cv2.calibrateCamera()` hoặc dùng tiện ích `tools/organize_camera_params.py`.
3. Lưu kết quả dưới dạng `camera_params_lens_{position}.yaml` trong `data/camera_params/`.
4. Cập nhật `ARTIFACT_LENS_POSITION` trong biến môi trường (không cần rebuild — đây là tham số runtime).

---

## 4. Khả năng Mở rộng Đa thiết bị

### 4.1 Kiến trúc Hiện tại

Hệ thống hiện tại hỗ trợ **nhiều thiết bị IoT đồng thời**, mỗi thiết bị được định danh bởi `device_id` duy nhất (hex 6 ký tự). Cách ly thiết bị được thực hiện ở các cấp sau:

| Lớp | Cơ chế cách ly |
|-----|---------------|
| MQTT topic | `cmd/{device_id}`, `ack/{device_id}`, `status/{device_id}` |
| Golden pose | Lưu theo cổ vật, không theo thiết bị (`golden_pose_{artifact_id}.yaml`) |
| Lịch sử ACK trong bộ nhớ | `CommandService._acks: dict[device_id → list]` |
| Bộ đếm căn chỉnh | `InspectionService._alignment_counters: dict[device_id → int]` |

`MqttBridge` subscribe topic wildcard và điều phối payload đến đúng handler theo `device_id` trích xuất từ đường dẫn topic.

### 4.2 Thêm Thiết bị Mới

Không cần thay đổi cấu hình phía server. Pi agent tự đăng ký khi kết nối lần đầu:

```python
# Pi: runtime/main_app.py
POST /api/v1/devices/get_device_id
Body: {"device_code": "<machine_hash>"}
Response: {"device_id": "a3f1b2", "device_code": "..."}
```

Server tạo bản ghi `IotDevice` trong PostgreSQL và bắt đầu theo dõi trạng thái online/offline của thiết bị qua heartbeat MQTT.

### 4.3 Cân nhắc Mở rộng Ngang

Triển khai đơn server hiện tại có các phụ thuộc trạng thái sau ngăn việc mở rộng ngang không trạng thái:

| Trạng thái trong bộ nhớ | Vị trí | Đường hướng di chuyển |
|------------------------|--------|---------------------|
| Lịch sử ACK | `CommandService._acks` | Chuyển sang Redis hoặc bảng PostgreSQL |
| Mô hình AI đã nạp | `ModelService._models` | Chấp nhận chi phí reload mỗi worker, hoặc dùng model server chia sẻ |
| Bộ đếm căn chỉnh | `InspectionService._alignment_counters` | Chuyển sang atomic counter Redis |
| Subscription MQTT | `MqttBridge` (single subscriber) | MQTT shared subscription (MQTT 5.0) hoặc message queue |
| Cache golden pose | File system (`data/`) | Đã volume-mount — tương thích với NFS/S3 |

---

## 5. Hot-Swap Mô hình AI

Mô hình AI được nạp từ volume Docker-mount (`../model:/app/model`), có nghĩa là trọng số mô hình có thể được cập nhật **mà không cần rebuild Docker image**:

```bash
# 1. Đưa trọng số mới vào thư mục model
cp my_new_model.pt /path/to/Artifact-Pose-System/model/

# 2. (Tuỳ chọn) Cập nhật đường dẫn tường minh trong .env.docker
DEFAULT_AI_MODEL_PATH=my_new_model.pt

# 3. Khởi động lại chỉ container server (không rebuild)
docker compose restart server
```

Nếu `DEFAULT_AI_MODEL_PATH` để trống, server sẽ nạp **file `.pt` đầu tiên theo thứ tự bảng chữ cái** trong `MODEL_DIR`. Duy trì một file `.pt` duy nhất trong thư mục để tránh nhập nhằng.

---

## 6. Bảo mật MQTT Broker

Cấu hình Mosquitto mặc định (`docker/mosquitto/mosquitto.conf`) chạy không cần xác thực. Với môi trường production:

1. Bật `allow_anonymous false` trong `mosquitto.conf`.
2. Tạo file mật khẩu: `mosquitto_passwd -c /mosquitto/config/passwd <username>`.
3. Đặt biến môi trường `MQTT_USERNAME` và `MQTT_PASSWORD` trên cả server lẫn Pi agent.
4. Cân nhắc bật TLS (`listener 8883`, `certfile`, `keyfile`) cho triển khai công khai trên mạng.

---

## 7. Cấu hình Raspberry Pi Agent

Pi agent đọc cấu hình từ file `.env` trong thư mục `embed/device_agent/`:

| Biến | Ví dụ | Mô tả |
|------|-------|-------|
| `DEVICE_ID` | _(trống)_ | Để trống để dùng ID do server cấp |
| `USE_SERVER_DEVICE_ID` | `true` | Tự đăng ký với server khi khởi động |
| `SERVER_BASE_URL` | `http://192.168.137.1:8000` | IP server qua mạng USB tethered |
| `MQTT_HOST` | `192.168.137.1` | IP broker (cùng với server) |
| `IMAGE_DIR` | `./data/pictures` | Lưu ảnh cục bộ để debug |
| `LENS_POSITION` | `1.5` | Phải khớp với `ARTIFACT_LENS_POSITION` trên server |
| `SLIDER_FAST_RATIO` | `0.85` | Tỉ lệ bước chạy tốc độ cao (profiling tăng tốc) |
| `AUTO_CAPTURE_AFTER_MOVE` | `true` | Tự chụp sau mỗi lần di chuyển động cơ hiệu chỉnh |

**Topo mạng:** Pi kết nối đến PC chạy WSL qua IP USB tethered (thường là `192.168.137.x`). Script `scripts/wsl_adb_setup.sh` cấu hình WSL bridge adapter cho use case này.

---

## 8. Tham chiếu Lệnh Build

| Mục tiêu | Lệnh | Ghi chú |
|---------|------|--------|
| Toàn bộ stack (build lần đầu) | `cd server && docker compose up --build -d` | ~15–20 phút (compile g2o từ source) |
| Chỉ rebuild server | `docker compose build server` | Tái dùng layer cache cho dependency |
| Khởi động không rebuild | `docker compose up -d` | Chỉ khi image đã tồn tại |
| Dừng tất cả container | `docker compose down` | Volume được giữ nguyên |
| Dừng + xóa DB volume | `docker compose down -v` | **Phá huỷ — xóa toàn bộ dữ liệu** |
| Flutter APK | `flutter build apk --dart-define=API_BASE_URL=http://<server-ip>:8000` | Chạy từ `client/artifact_app/` |
| Pi agent | `pip install -r requirements.txt && python runtime/main_app.py` | Chạy từ `embed/device_agent/` |

---

## 9. Checklist Triển khai

Trước khi deploy lên môi trường production hoặc chia sẻ, kiểm tra các mục sau:

- [ ] `AUTH_SECRET_KEY` được đặt thành chuỗi ngẫu nhiên mật mã (≥ 32 byte).
- [ ] `ADMIN_PASSWORD` được thay từ mặc định `123456`.
- [ ] `POSTGRES_PASSWORD` được thay từ mặc định `artifact123`.
- [ ] `CORS_ALLOW_ORIGINS` được giới hạn vào các origin client đã biết (không dùng `*`).
- [ ] Xác thực broker MQTT được bật (`allow_anonymous false`).
- [ ] File `.env` được loại khỏi version control (mục `.gitignore`: `.env`).
- [ ] Thư mục `model/` chứa file trọng số `.pt` đúng.
- [ ] `ARTIFACT_LENS_POSITION` khớp với vị trí lens dùng trong hiệu chỉnh camera.
- [ ] `TRANS_TOLERANCE_MM` và `ROT_TOLERANCE_DEG` được tinh chỉnh cho cấu hình vật lý.
- [ ] `SIGN_MOVE_X`, `SIGN_MOVE_Z`, `SIGN_ROTATE_PAN`, `SIGN_ROTATE_TILT` được xác minh bằng một lần chạy test căn chỉnh.
