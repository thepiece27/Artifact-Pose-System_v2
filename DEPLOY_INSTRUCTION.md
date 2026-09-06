# Hướng Dẫn Deploy & Test

> Cập nhật: 28/05/2026
> Phạm vi: chạy FastAPI server bằng Docker trên PC/WSL2, cài Flutter app lên Android qua USB hoặc WiFi LAN, và kết nối Raspberry Pi device agent qua LAN/hotspot.

---

## 1. Tóm Tắt Kiến Trúc Deploy

```
Artifact-Pose-System/
├── server/                 FastAPI + MQTT bridge + PostgreSQL + pose/AI
├── client/artifact_app/    Flutter Android app
├── embed/device_agent/     Agent chạy trên Raspberry Pi
├── model/                  File model YOLO *.pt được mount vào container
└── artifact_db.sql         Schema PostgreSQL tham chiếu
```

Luồng chạy thực tế:

1. PC chạy Docker Compose: `artifact_server`, `artifact_postgres`, `artifact_mosquitto`.
2. App Android gọi REST API tới FastAPI server.
3. Server gửi lệnh điều khiển tới Raspberry Pi qua MQTT topic `cmd/{device_code}`.
4. Raspberry Pi chụp ảnh, di chuyển servo/slider, gửi ACK/status qua MQTT và upload ảnh về server.
5. Server chạy pose correction, golden pose initialization, SSIM/AI inspection và trả kết quả cho app.

---

## 2. Các Quy Ước Quan Trọng Hiện Tại

### `device_id` và `device_code`

- `device_id`: ID nội bộ trong bảng `iot_devices`, dạng chuỗi 6 ký tự.
- `device_code`: mã thiết bị Raspberry Pi dùng cho MQTT/API vận hành, ví dụ `dev-bbb742d369`.
- App và workflow phải dùng `device_code` khi gọi status, ACK, queue command, start workflow.
- Pi vẫn nhận field `device_id` trong response đăng ký để tương thích code agent hiện tại, nhưng giá trị đó chính là `device_code`.
- API tạo/sửa/xóa thiết bị thủ công đã bị loại bỏ. Thiết bị được quản lý qua registry và tự upsert vào DB khi Pi register.

Thiết bị hiện có trong registry:

```json
{
  "machine_to_device": {
    "md5-bbb742d369f860d8e4ed1069b715f6fd": "dev-bbb742d369"
  }
}
```

### Baseline stereo

Baseline stereo được khóa cố định:

- `100.0 mm`
- `80000 steps`
- `800 steps/mm`
- Pose solver dùng `0.10 m`

Baseline không còn là tham số nhập từ giao diện và không được phép override qua API `start-initialization`. Nếu client cố gửi `baseline_mm` hoặc `steps_per_mm`, server sẽ reject request.

---

## 3. Yêu Cầu Cài Đặt

### Trên PC/WSL2

- Docker Desktop + Docker Compose v2
- WSL2 Ubuntu nếu chạy trên Windows
- Flutter SDK phù hợp với `client/artifact_app/pubspec.yaml` (`sdk: ^3.7.2`)
- Android SDK / ADB
- `usbipd-win` nếu muốn dùng USB từ Windows vào WSL
- Git, curl

Kiểm tra nhanh:

```bash
docker --version
docker compose version
flutter --version
adb --version
```

### Trên Android

- Bật Developer Options.
- Bật USB Debugging nếu deploy qua USB.
- Cho phép cài APK từ nguồn ngoài nếu cài file release thủ công.

### Trên Raspberry Pi

- Python 3.10+
- Camera hỗ trợ `picamera2`
- I2C bật cho PCA9685
- GPIO hoạt động cho stepper driver
- Các package Python chính: `requests`, `paho-mqtt`, `picamera2`, `adafruit-circuitpython-servokit`, `RPi.GPIO`

---

## 4. Khởi Động Server Bằng Docker

```bash
cd /home/thepiece/System/Artifact-Pose-System/server
```

Nếu chưa có file env Docker:

```bash
cp env.docker.example .env.docker
```

Các biến quan trọng trong `server/.env.docker`:

```env
POSTGRES_DB=artifact_auth
POSTGRES_USER=artifact
POSTGRES_PASSWORD=<generate-a-unique-password-at-least-16-chars>

ADMIN_USERNAME=admin
ADMIN_PASSWORD=<generate-a-unique-password-at-least-12-chars>
AUTH_SECRET_KEY=<generate-a-unique-secret-at-least-32-chars>

MQTT_HOST=mosquitto
MQTT_PORT=1883

ARTIFACT_LENS_POSITION=1.5
MAX_ALIGNMENT_ITERATIONS=7
ALIGNMENT_TIMEOUT_SEC=300

SIGN_MOVE_X=1
SIGN_MOVE_Z=1
SIGN_ROTATE_PAN=-1
SIGN_ROTATE_TILT=-1
```

Với deploy thật, đổi `POSTGRES_PASSWORD`, `ADMIN_PASSWORD`, `AUTH_SECRET_KEY` trước khi đưa server ra mạng rộng.

Build và chạy lần đầu:

```bash
docker compose --env-file .env.docker up -d --build
```

Các lần sau:

```bash
docker compose --env-file .env.docker up -d
```

Kiểm tra container:

```bash
docker compose --env-file .env.docker ps
```

Kiểm tra server:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/mqtt/health
```

`/health` hợp lệ sẽ có dạng:

```json
{"status":"ok","mqtt_connected":"true"}
```

Đăng nhập lấy token:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"<ADMIN_PASSWORD>"}'
```

Xem log server:

```bash
docker compose --env-file .env.docker logs -f server
```

Dừng server:

```bash
docker compose --env-file .env.docker down
```

---

## 5. Model AI

Docker Compose mount thư mục:

```text
../model -> /app/model
```

File model hiện có:

```text
model/best_Quyen.pt
```

Server tự scan file `*.pt` đầu tiên trong `/app/model` và load với tên mặc định `default` khi khởi động.

Thay model:

```bash
cp /path/to/new-model.pt /home/thepiece/System/Artifact-Pose-System/model/
cd /home/thepiece/System/Artifact-Pose-System/server
docker compose --env-file .env.docker restart server
```

Hoặc load model qua API admin:

```bash
TOKEN="<admin_access_token>"

curl -X POST http://127.0.0.1:8000/models/load \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name":"default","path":"/app/model/best_Quyen.pt","backend":"auto","labels":[]}'
```

---

## 6. Deploy App Android Qua USB `adb reverse`

Dùng cách này khi điện thoại cắm USB vào PC và muốn app gọi server bằng `http://127.0.0.1:8000`.

### 6.1. Bật USB Debugging

1. Android Settings → About phone.
2. Nhấn Build number 7 lần để bật Developer Options.
3. Developer Options → bật USB Debugging.
4. Cắm USB, chọn Allow USB debugging trên điện thoại.

Kiểm tra:

```bash
adb devices
```

Thiết bị phải ở trạng thái `device`, không phải `unauthorized`.

### 6.2. Thiết lập tunnel thủ công

Nếu dùng WSL2 và Android cắm USB ở Windows, attach thiết bị vào WSL từ Windows PowerShell:

```powershell
usbipd list
usbipd bind --busid <BUSID>
usbipd attach --wsl --busid <BUSID>
```

Trong WSL, kiểm tra thiết bị và tạo tunnel:

```bash
adb devices
adb reverse tcp:8000 tcp:8000
```

Kiểm tra tunnel:

```bash
adb reverse --list
```

Kết quả cần có `tcp:8000 tcp:8000`.

Lưu ý: phải chạy lại `adb reverse` sau khi rút cáp, restart điện thoại, hoặc restart adb server.

### 6.3. Build/run app với USB

```bash
cd /home/thepiece/System/Artifact-Pose-System/client/artifact_app
flutter pub get
flutter run --dart-define=API_BASE_URL=http://127.0.0.1:8000
```

Build APK release:

```bash
flutter clean
flutter pub get
flutter build apk --release --dart-define=API_BASE_URL=http://127.0.0.1:8000
adb install -r build/app/outputs/flutter-apk/app-release.apk
```

Cài lại sạch:

```bash
adb uninstall com.pbl5.artifactapp
adb install build/app/outputs/flutter-apk/app-release.apk
```

---

## 7. Deploy App Android Qua WiFi LAN

Dùng cách này khi điện thoại và PC cùng mạng LAN/WiFi.

Tìm IP PC:

```bash
ip -4 addr show | grep -oP '(?<=inet )192\.[0-9.]+'
```

Hoặc trên Windows:

```bash
cmd.exe /c ipconfig
```

Mở firewall port 8000 nếu cần:

```bash
sudo ufw allow 8000/tcp
```

Build/run app với IP thật của PC:

```bash
cd /home/thepiece/System/Artifact-Pose-System/client/artifact_app
flutter pub get
flutter run --dart-define=API_BASE_URL=http://192.168.x.y:8000
```

Build APK:

```bash
flutter build apk --release --dart-define=API_BASE_URL=http://192.168.x.y:8000
adb install -r build/app/outputs/flutter-apk/app-release.apk
```

Nếu không truyền `--dart-define`, app sẽ dùng giá trị mặc định trong:

```text
client/artifact_app/lib/services/api_config.dart
```

Hiện tại mặc định đang là:

```dart
Pass `--dart-define=API_BASE_URL=https://<server-host>` when building the app; do not hardcode a LAN IP in release sources.
```

Khuyến nghị vẫn dùng `--dart-define=API_BASE_URL=...` khi build để tránh quên sửa hardcode.

---

## 8. Cấu Hình Raspberry Pi Device Agent

File cấu hình:

```text
embed/device_agent/.env
```

Ví dụ khi Pi nối vào hotspot của PC, PC có IP hotspot `192.168.137.1`:

```env
DEVICE_ID=
USE_SERVER_DEVICE_ID=true
SERVER_BASE_URL=http://192.168.137.1:8000

MQTT_HOST=192.168.137.1
MQTT_PORT=1883
MQTT_KEEPALIVE_SEC=60
MQTT_USERNAME=<mqtt-username>
MQTT_PASSWORD=<mqtt-password>
DEVICE_API_KEY=<device-api-key>
DEVICE_ENROLLMENT_KEY=<enrollment-key>
MQTT_QOS=1

MQTT_CMD_TOPIC_TEMPLATE=cmd/{device_id}
MQTT_ACK_TOPIC_TEMPLATE=ack/{device_id}
MQTT_STATUS_TOPIC_TEMPLATE=status/{device_id}

IMAGE_DIR=./data/pictures
DEFAULT_ARTIFACT_ID=artifact_demo_001
LENS_POSITION=1.5

SLIDER_FAST_RATIO=0.85
SLIDER_FAST_PULSE_DELAY_SEC=0.00035
SLIDER_SLOW_PULSE_DELAY_SEC=0.0008
AUTO_CAPTURE_AFTER_MOVE=true
```

Nếu Pi và PC cùng WiFi LAN, đổi:

```env
SERVER_BASE_URL=http://<IP_PC_LAN>:8000
MQTT_HOST=<IP_PC_LAN>
```

Chạy agent:

```bash
cd /home/pi/Artifact-Pose-System/embed/device_agent
PYTHONPATH=. python3 runtime/main_app.py
```

Launcher tương thích cũ cũng còn hoạt động:

```bash
PYTHONPATH=. python3 main_app.py
```

Khi Pi khởi động đúng, log cần có các ý chính:

```text
[APP] Nhận device_id từ server: dev-bbb742d369
[MQTT] Connected ...
[APP] Listening for commands on: cmd/dev-bbb742d369
```

Pi đăng ký bằng:

```text
POST /api/v1/devices/get_device_id
```

Server sẽ:

- đọc/cập nhật `server/data/device_registry.json`;
- upsert thiết bị vào bảng `iot_devices`;
- dùng `device_code` đó cho MQTT topic.

---

## 9. Kiểm Tra Thiết Bị Và MQTT

Xem danh sách device qua API cần token:

```bash
TOKEN="<access_token>"

curl http://127.0.0.1:8000/api/v1/devices \
  -H "Authorization: Bearer $TOKEN"
```

Kiểm tra status realtime của Pi:

```bash
curl http://127.0.0.1:8000/api/v1/devices/dev-bbb742d369/status
```

Xem ACK gần nhất:

```bash
curl "http://127.0.0.1:8000/api/v1/devices/dev-bbb742d369/acks?limit=20"
```

Theo dõi MQTT status trực tiếp:

```bash
docker exec -it artifact_mosquitto mosquitto_sub -h localhost -t "status/#" -v
```

Gửi một lệnh test nhỏ:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/devices/dev-bbb742d369/queue_move \
  -H "Content-Type: application/json" \
  -d '{"action":"move","yaw_delta":1.0,"pitch_delta":0.0,"x_steps":0,"z_steps":0,"x_dir":1,"z_dir":1}'
```

Cẩn thận: lệnh có `x_steps` hoặc `z_steps` khác 0 sẽ làm phần cứng di chuyển thật.

---

## 10. Smoke Test API + MQTT + ACK

Script:

```text
server/tools/smoke_test_api_mqtt_ack.py
```

Nếu Pi agent đang chạy sẵn:

```bash
cd /home/thepiece/System/Artifact-Pose-System
python3 server/tools/smoke_test_api_mqtt_ack.py \
  --base-url http://127.0.0.1:8000 \
  --no-remote-agent \
  --device-id dev-bbb742d369 \
  --allow-ack-non-ok
```

Script kiểm tra:

1. `/health`
2. `/mqtt/health`
3. device online
4. publish command qua `/api/v1/devices/{device_code}/queue_move`
5. đợi ACK từ Pi
6. đọc `/mqtt/events`

---

## 11. Quy Trình Deploy Nhanh Từ Đầu

```bash
# 1. Server
cd /home/thepiece/System/Artifact-Pose-System/server
docker compose --env-file .env.docker up -d --build
curl http://127.0.0.1:8000/health

# 2. USB tunnel cho Android
adb reverse tcp:8000 tcp:8000

# 3. Flutter app qua USB
cd client/artifact_app
flutter pub get
flutter build apk --release --dart-define=API_BASE_URL=http://127.0.0.1:8000
adb install -r build/app/outputs/flutter-apk/app-release.apk

# 4. Pi agent
cd /home/pi/Artifact-Pose-System/embed/device_agent
PYTHONPATH=. python3 runtime/main_app.py
```

---

## 12. Các Lỗi Thường Gặp

### App không đăng nhập được

Kiểm tra server:

```bash
curl http://127.0.0.1:8000/health
```

Nếu dùng USB, kiểm tra:

```bash
adb devices
adb reverse --list
```

Nếu dùng WiFi, mở browser trên điện thoại:

```text
http://<IP_PC>:8000/health
```

### Login báo inactive

Server hiện đã chặn user `is_active=false` ở cả login và JWT dependency. Dùng admin bật lại user trong màn hình Users hoặc cập nhật DB.

### Pi không xuất hiện trong app

Kiểm tra registry:

```bash
cat /home/thepiece/System/Artifact-Pose-System/server/data/device_registry.json
```

Kiểm tra DB:

```bash
docker exec -it artifact_postgres psql -U artifact -d artifact_auth \
  -c "SELECT device_id, device_code, status, last_active_at FROM iot_devices;"
```

Nếu Pi chưa register, restart Pi agent và kiểm tra `SERVER_BASE_URL`.

### MQTT disconnected

```bash
curl http://127.0.0.1:8000/mqtt/health
docker compose --env-file server/.env.docker -f server/docker-compose.yml logs --tail=80 server
docker compose --env-file server/.env.docker -f server/docker-compose.yml restart mosquitto server
```

### Golden initialization lỗi 400

Nguyên nhân thường gặp:

- camera params thiếu hoặc sai lens position;
- marker ChArUco không nằm rõ trong ảnh left;
- ảnh stereo không đủ feature ORB;
- Pi upload thiếu `device_code` hoặc device chưa có trong registry;
- file ảnh left/right không đọc được.

Kiểm tra ảnh upload:

```bash
ls -lh server/data/uploads/pose_init/
docker compose --env-file server/.env.docker -f server/docker-compose.yml logs --tail=100 server
```

### Model không load

```bash
ls -lh model/*.pt
docker compose --env-file server/.env.docker -f server/docker-compose.yml logs --tail=100 server | grep STARTUP
```

Nếu thay model, restart server.

---

## 13. Ghi Chú Bảo Mật

- Không dùng placeholder trong production; server sẽ fail-fast với secret yếu/thiếu.
- Mosquitto hiện yêu cầu username/password (`allow_anonymous false`); không publish broker/PostgreSQL ra mạng ngoài nếu không cần. Device command/upload và media đều yêu cầu credential.
- Các route app như artifacts, schedules, workflows, models, pose manual cần JWT.
- `POST /pose/initialize_golden` cho phép JWT hoặc device đã có trong registry, để Pi upload stereo pair mà không cần tài khoản app.
- `POST /inspections/upload` là endpoint Pi upload ảnh alignment/inspection, hiện vẫn dành cho device agent trong mạng nội bộ.

---

## 14. Dọn Database An Toàn

Chạy từ thư mục project hoặc bất kỳ đâu cũng được, miễn container `artifact_postgres` đang chạy.

### 14.1. Backup database trước

```bash
mkdir -p backups

docker exec artifact_postgres pg_dump \
  -U artifact \
  -d artifact_auth \
  -Fc \
  -f /tmp/artifact_auth_$(date +%Y%m%d_%H%M%S).dump

docker cp artifact_postgres:/tmp/$(docker exec artifact_postgres sh -c "ls -t /tmp/artifact_auth_*.dump | head -1 | xargs basename") ./backups/
```

### 14.2. Kiểm tra nhanh dữ liệu hiện có

```bash
docker exec -it artifact_postgres psql -U artifact -d artifact_auth -c "
SELECT 'artifacts' AS table, COUNT(*) FROM artifacts
UNION ALL SELECT 'images', COUNT(*) FROM images
UNION ALL SELECT 'image_comparisons', COUNT(*) FROM image_comparisons
UNION ALL SELECT 'alerts', COUNT(*) FROM alerts
UNION ALL SELECT 'schedules', COUNT(*) FROM schedules;
"
```

### 14.3. Dọn lịch sử kiểm tra nhưng giữ cổ vật, user, ảnh baseline

Lệnh này xóa:

- alerts;
- lịch sử comparison;
- ảnh inspection trong DB.

Lệnh này không xóa artifact, user, baseline image.

```bash
docker exec -it artifact_postgres psql -U artifact -d artifact_auth -c "
BEGIN;

DELETE FROM alerts;
DELETE FROM image_comparisons;
DELETE FROM images WHERE image_type = 'inspection';

COMMIT;
"
```

### 14.4. Reset trạng thái artifact về `good` nếu cần

```bash
docker exec -it artifact_postgres psql -U artifact -d artifact_auth -c "
UPDATE artifacts SET status = 'good';
"
```

### 14.5. Dọn file ảnh inspection/heatmap/detect trên volume nếu muốn

> Cẩn thận: lệnh này xóa file runtime trong uploads, nên chỉ chạy sau khi đã backup hoặc chắc chắn muốn dọn.

```bash
docker exec artifact_server sh -c "
find /app/data/uploads/artifacts -type f \
  \( -name 'inspection_*' -o -name 'heatmap_*' -o -name 'detect_*' -o -name 'aligned_*' -o -name 'final_aligned_*' \) \
  -delete
"
```

---

## 15. Xóa Sạch Dữ Liệu Nghiệp Vụ Nhưng Giữ User/Admin

Các lệnh dưới đây xóa dữ liệu nghiệp vụ nhưng giữ bảng `users`.

Sẽ xóa:

- `artifacts`
- `images`
- `image_comparisons`
- `alerts`
- `schedules`
- `iot_devices` nếu chạy bước reset thiết bị

Không xóa:

- `users`

### 15.1. Backup trước

```bash
mkdir -p backups

docker exec artifact_postgres pg_dump \
  -U artifact \
  -d artifact_auth \
  -Fc \
  -f /tmp/artifact_auth_clean_before_$(date +%Y%m%d_%H%M%S).dump

docker cp artifact_postgres:/tmp/$(docker exec artifact_postgres sh -c "ls -t /tmp/artifact_auth_clean_before_*.dump | head -1 | xargs basename") ./backups/
```

### 15.2. Xóa sạch dữ liệu nghiệp vụ, giữ `users`

```bash
docker exec -it artifact_postgres psql -U artifact -d artifact_auth -c "
BEGIN;

TRUNCATE TABLE
  alerts,
  image_comparisons,
  schedules,
  images,
  artifacts
RESTART IDENTITY CASCADE;

COMMIT;
"
```

### 15.3. Reset luôn thiết bị IoT nếu cần

Chỉ chạy nếu bạn muốn app/server quên danh sách thiết bị cũ.

```bash
docker exec -it artifact_postgres psql -U artifact -d artifact_auth -c "
TRUNCATE TABLE iot_devices RESTART IDENTITY CASCADE;
"
```

### 15.4. Xóa file uploads/model output liên quan artifact

Lệnh này xóa toàn bộ ảnh nghiệp vụ đã upload/chụp. Không đụng DB user.

```bash
docker exec artifact_server sh -c "
rm -rf /app/data/uploads/artifacts
mkdir -p /app/data/uploads/artifacts
"
```

### 15.5. Kiểm tra lại

```bash
docker exec -it artifact_postgres psql -U artifact -d artifact_auth -c "
SELECT 'users' AS table, COUNT(*) FROM users
UNION ALL SELECT 'artifacts', COUNT(*) FROM artifacts
UNION ALL SELECT 'images', COUNT(*) FROM images
UNION ALL SELECT 'image_comparisons', COUNT(*) FROM image_comparisons
UNION ALL SELECT 'alerts', COUNT(*) FROM alerts
UNION ALL SELECT 'schedules', COUNT(*) FROM schedules;
"
```
