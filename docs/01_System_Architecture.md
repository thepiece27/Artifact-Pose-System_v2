# 01 — Tổng quan Kiến trúc Hệ thống & Công nghệ

> **Dự án:** Hệ thống Tự động Căn chỉnh Tư thế Camera và Giám định Tổn hại Cổ vật bằng AI  
> **Phiên bản:** 1.0 | **Cập nhật lần cuối:** Tháng 6/2026

---

## 1. Tóm tắt Tổng quan

Hệ thống là một nền tảng đa thành phần phục vụ mục tiêu **tự động căn chỉnh tư thế 6-DoF (6 bậc tự do) của camera** so với một cổ vật văn hoá, đồng thời tích hợp **kiểm tra tổn hại cấu trúc bằng AI**. Kiến trúc được xây dựng theo mô hình phân tán Client–Server: một Raspberry Pi đặt cạnh cổ vật đảm nhiệm điều khiển phần cứng và thu thập ảnh, trong khi server FastAPI chạy trong container Docker trên WSL2 thực hiện toàn bộ tác vụ tính toán nặng — bao gồm tối ưu hoá tư thế C++/G2O và phân tích tổn hại bằng deep learning.

---

## 2. Kiến trúc Tổng thể

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        FLUTTER MOBILE CLIENT                                 │
│                    (Android / iOS — HTTP + Bearer JWT)                       │
└────────────────────────────────┬─────────────────────────────────────────────┘
                                 │ REST API (HTTP/8000)
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                     FASTAPI APPLICATION SERVER  (Docker / WSL2)             │
│                                                                              │
│  ┌─────────────┐  ┌──────────────────┐  ┌──────────────────────────────┐    │
│  │  Auth /     │  │  Pose Service    │  │   Inspection Service         │    │
│  │  User CRUD  │  │  (Python Wrapper │  │   (SIFT · SSIM · YOLO)       │    │
│  └─────────────┘  │   → C++ G2O)    │  └──────────────────────────────┘    │
│                   └──────────────────┘                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │             pose_solver_cpp.so  (pybind11 native extension)          │    │
│  │   QuadTree-ORB · Stereo DLT · Hybrid G2O · Deviation Calculator     │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────┐        ┌──────────────────────────────────┐         │
│  │  PostgreSQL 16      │        │  Mosquitto MQTT Broker           │         │
│  │  (Docker volume)    │        │  (eclipse-mosquitto:2)           │         │
│  └─────────────────────┘        └──────────────────────────────────┘         │
└─────────────────────────────────────────┬────────────────────────────────────┘
                                          │ MQTT (TCP/1883)
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                   RASPBERRY PI  —  EMBEDDED DEVICE AGENT                     │
│                                                                              │
│   Picamera2 (4K)    PCA9685 Servo      GPIO Stepper (X / Z)                 │
│   Pan / Tilt head   (CH0=Pan, CH1=Tilt) DRV8825 + 32× microstep            │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Phân tích các Thành phần

### 3.1 Flutter Mobile Client (`client/artifact_app/`)

| Khía cạnh | Triển khai |
|-----------|----------|
| Giao tiếp HTTP | `http 1.2.0` với Bearer token injection |
| Quản lý trạng thái | `provider 6.1.2` (ChangeNotifier) |
| Lưu trữ bảo mật | `flutter_secure_storage 9.2.2` |
| Chọn ảnh | `image_picker 1.1.2` |
| Bản địa hoá | `intl 0.19.0` |

Client giao tiếp hoàn toàn qua REST. Toàn bộ xác thực được xử lý bằng JWT ngắn hạn (`AUTH_ACCESS_TOKEN_EXPIRE_MINUTES`, mặc định 60 phút). Khi nhận lỗi 401, class `ApiClient` tự động điều hướng về màn hình đăng nhập.

### 3.2 FastAPI Server (`server/`)

Server là nhân tính toán của hệ thống, được cấu trúc theo bốn lớp ngang:

```
server/
├── app/api/routes/        # HTTP route handler (thin controller)
├── app/services/          # Business logic (InspectionService, PoseService, …)
├── app/modules/           # Module thuật toán (artifact_pose C++ wrapper)
└── app/models/            # SQLAlchemy ORM model
```

**Danh sách route:**

| Prefix | Module | Trách nhiệm |
|--------|--------|------------|
| `/api/v1/artifacts` | `artifacts.py` | CRUD cổ vật, upload ảnh baseline & kiểm tra |
| `/api/v1/devices` | `devices.py` | Đăng ký thiết bị IoT, trạng thái, lịch sử ACK |
| `/workflows/{device_id}` | `workflows.py` | Chuỗi capture đa bước điều phối qua MQTT |
| `/pose` | `pose.py` | Khởi tạo golden pose, hiệu chỉnh tư thế từng ảnh |
| `/api/v1/users`, `/auth` | `users.py`, `auth.py` | JWT auth, quản lý người dùng |
| `/api/v1/schedules` | `schedules.py` | Lịch kiểm tra định kỳ |
| `/api/v1/alerts` | (qua artifacts) | Cảnh báo leo thang khi có tổn hại |
| `/health` | `health.py` | Kiểm tra trạng thái hoạt động |

### 3.3 C++ Native Extension (`server/native/pose_solver_cpp/`)

Được biên dịch thành shared object đặc thù theo nền tảng (`pose_solver_cpp.cpython-3xx-linux-gnu.so`) bằng **CMake + pybind11**. Module này đóng gói năm đơn vị tính toán độc lập:

| Lớp C++ | Header | Vai trò |
|---------|--------|--------|
| `QuadTree` | `quadtree.h` | Phân phối đặc trưng ORB đồng đều theo không gian |
| `ORBMatcher` | `orb_matcher.h` | Ghép đặc trưng BF-Hamming với kiểm tra tỉ lệ Lowe |
| `StereoTriangulator` | `stereo_triangulation.h` | Phục hồi điểm 3D bằng phương pháp DLT |
| `HybridPoseSolver` | `hybrid_pose_solver.h` | Tối ưu hoá tư thế phi tuyến bằng G2O |
| `DeviationCalculator` | `deviation_calculator.h` | Delta SE3 → lệnh điều khiển động cơ |

Package Python `common.py` nạp shared object tại thời điểm import và chuyển sang đường dẫn dự phòng chỉ dùng OpenCV khi `.so` không khả dụng:

```python
try:
    from . import pose_solver_cpp   # compiled extension
    HAS_CPP = True
except Exception:
    pose_solver_cpp = None
    HAS_CPP = False
```

### 3.4 Embedded Device Agent (`embed/device_agent/`)

Chạy trên Raspberry Pi 4. Điểm khởi chạy chính là `runtime/main_app.py`, với stub launcher tương thích ngược tại `device_agent/main_app.py`. Agent thực hiện:

1. Đăng ký với server qua `POST /api/v1/devices/get_device_id`.
2. Subscribe MQTT topic `cmd/{device_id}`.
3. Publish xác nhận tới `ack/{device_id}` và heartbeat tới `status/{device_id}` mỗi 30 giây.
4. Thực thi **trùng lặp Task ID** với TTL 180 giây để từ chối lệnh MQTT bị phát lại.

Phần cứng ngoại vi được trừu tượng hoá thành hai lớp adapter:

| Adapter | Phần cứng |
|---------|----------|
| `CameraManager` | Picamera2 độ phân giải 3840 × 2160, lấy nét thủ công, vị trí lens có thể cấu hình |
| `HardwareController` | Servo PCA9685 (I²C) + driver stepper DRV8825 (GPIO) |

---

## 4. Giao thức Truyền thông

### 4.1 REST (HTTP/8000)

Mọi tương tác giữa client và server đều dùng JSON qua HTTP. Upload file dùng `multipart/form-data`. Xác thực theo RFC 6750 (Bearer token).

### 4.2 MQTT

Server và Pi giao tiếp bất đồng bộ qua MQTT thông qua broker Mosquitto dùng chung. Quy ước topic:

| Mẫu Topic | Chiều | Payload |
|-----------|-------|--------|
| `cmd/{device_id}` | Server → Pi | Lệnh JSON (`action`, `task_id`, tham số) |
| `ack/{device_id}` | Pi → Server | Xác nhận JSON với `task_id` và trạng thái |
| `status/{device_id}` | Pi → Server | Heartbeat định kỳ (`online`, số liệu cảm biến) |

Server duy trì **hàng đợi dự phòng HTTP** trong bộ nhớ (`CommandService`) cho trường hợp broker MQTT không khả dụng.

---

## 5. Sơ đồ Cơ sở Dữ liệu

PostgreSQL 16 được sử dụng làm lớp lưu trữ. Quan hệ giữa các bảng:

```
users
  │
  └──── artifacts ──────┬──── images (baseline | inspection)
           │            │         │
           │            └──── image_comparisons ──── alerts
           │
           └──── schedules
           └──── iot_devices
```

Các quyết định thiết kế quan trọng:
- Khoá chính dùng chuỗi hex 6 ký tự (`secrets.token_hex(3)`) thay vì số nguyên tự tăng, cung cấp ID mờ không thể đoán tuần tự.
- `ImageComparison` lưu báo cáo tổn hại định lượng (`damage_score`, `ssim_score`, `heatmap_path`, `detections_json`).
- Bản ghi `Alert` được `InspectionService` tự động tạo khi `ComparisonStatus` là `warning` hoặc `damaged`.

---

## 6. Docker Deployment

The server stack is defined in `docker-compose.yml` with three services:

```
services:
  postgres:  postgres:16-alpine
  mosquitto: eclipse-mosquitto:2
  server:    (multi-stage build)
```

The server `Dockerfile` uses a **two-stage build** to keep the runtime image lean:

| Stage | Base Image | Purpose |
|-------|-----------|---------|
| `cpp-builder` | `python:3.10-slim` | Build g2o from source, compile `pose_solver_cpp.so` |
| Runtime | `python:3.10-slim` | Install Python deps, copy `.so` from builder |

Critical runtime dependency: `libg2o*.so` and `libsuitesparse` are copied from the builder stage and loaded via `ldconfig`.

The `model/` directory and `server/data/` are mounted as Docker volumes, allowing model hot-swapping and persistent data storage without requiring an image rebuild.

---

## 7. Tổng hợp Công nghệ Sử dụng

| Lớp | Công nghệ | Phiên bản |
|-----|----------|-----------|
| Framework backend | FastAPI + Uvicorn | latest |
| ORM | SQLAlchemy | latest |
| Computer Vision | OpenCV | system (qua apt) |
| Tối ưu hoá tư thế | G2O | `20230806_git` (đã ghim) |
| Đại số tuyến tính | Eigen3 | system |
| Python binding | pybind11 | latest |
| Phát hiện AI | Ultralytics YOLO | latest |
| Tính toán SSIM | scikit-image | latest |
| MQTT client | Paho-MQTT | latest |
| Cơ sở dữ liệu | PostgreSQL | 16-alpine |
| Message broker | Eclipse Mosquitto | 2 |
| Mobile client | Flutter | 3.x (Dart) |
| Container hoá | Docker / Docker Compose | latest |
