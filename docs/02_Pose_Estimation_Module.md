# 02 — Module Căn chỉnh Tư thế: Nhân C++ & Lớp Python Wrapper

> **Đường dẫn nguồn:**  
> `server/native/pose_solver_cpp/` — Thư viện C++  
> `server/app/modules/artifact_pose/` — Lớp điều phối Python  
> `server/app/services/pose_service.py` — Giao diện dịch vụ

---

## 1. Tổng quan Module

Module Ước lượng Tư thế là nhân toán học của toàn hệ thống. Nhiệm vụ của nó là xác định **phép biến đổi cứng 6-DoF** (quay + tịnh tiến) của camera so với cảnh tham chiếu, sau đó tính toán **lệnh hiệu chỉnh động cơ** cần thiết để đưa camera trở về tư thế “vàng” (golden pose) đã ghi trước.

Module được triển khai dưới dạng thư viện shared object C++ hiệu năng cao, được expose ra Python qua **pybind11**. Gồm năm đơn vị biên dịch độc lập:

```
pose_solver_cpp/
├── src/
│   ├── quadtree.cpp            # Phân phối đặc trưng ORB đồng đều theo không gian
│   ├── orb_matcher.cpp         # Ghép đặc trưng BF-Hamming với kiểm tra tỉ lệ Lowe
│   ├── stereo_triangulation.cpp # Tam giác hoá DLT các cặp điểm stereo
│   ├── hybrid_pose_solver.cpp  # Tối ưu hoá tư thế phi tuyến G2O
│   └── deviation_calculator.cpp # Delta SE3 → Δx, Δz, Δpan, Δtilt
└── bindings.cpp                # Định nghĩa module pybind11
```

---

## 2. Công nghệ Marker: Diamond ChArUco

Hệ thống dựa trên **marker Diamond ChArUco** làm điểm tham chiếu 3D đã biết hình học. Marker được định nghĩa trong `common.py`:

```python
SQUARE_LENGTH = 0.040   # cạnh hình vuông ngoài: 40 mm
MARKER_LENGTH = 0.025   # cạnh marker ArUco bên trong: 25 mm
DICT_ID = aruco.DICT_4X4_50

HALF_SQ = SQUARE_LENGTH / 2.0
DIAMOND_OBJ_PTS = np.array([
    [-HALF_SQ,  HALF_SQ, 0],   # góc trên-trái
    [ HALF_SQ,  HALF_SQ, 0],   # góc trên-phải
    [ HALF_SQ, -HALF_SQ, 0],   # góc dưới-phải
    [-HALF_SQ, -HALF_SQ, 0],   # góc dưới-trái
], dtype=np.float32)
```

Phát hiện được thực hiện qua pipeline ArUco của OpenCV (`cv2.aruco.DetectorsParameters`) và `detectCharucoDiamond`. Bốn góc ngoài được dùng làm quan sát 2D. Toạ độ đã biết trong không gian vật thể `DIAMOND_OBJ_PTS` cung cấp bốn cặp tương ứng điểm-điểm cho bài toán PnP khởi đầu.

---

## 3. Quy trình A — Khởi tạo Golden Pose

Golden pose được tính một lần cho mỗi cổ vật và lưu dưới dạng file YAML. Nó thiết lập hệ toạ độ tham chiếu dùng để đánh giá toàn bộ các lần hiệu chỉnh tiếp theo.

### 3.1 Giao thức Chụp Stereo

Pi thực hiện **thao tác tịnh tiến lấy ảnh stereo** (`capture_stereo_pair`):
1. Chụp **ảnh trái** tại vị trí home.
2. Di chuyển stepper trục X **+80.000 bước** (≈ 100 mm tiến).
3. Chụp **ảnh phải**.
4. Trả slider về vị trí home.

Cặp ảnh stereo thu được được gửi lên server để tái tạo 3D.

### 3.2 Trích xuất Đặc trưng ORB — Phân phối QuadTree

Phát hiện ORB thô trả về tập keypoint phân bổ không đều trong không gian — các vùng giàu góc cạnh thống trị trong khi vùng phẳng hầu như không cho ra đặc trưng nào. Lớp `QuadTree` phân phối lại đặc trưng đồng đều trên mặt phẳng ảnh:

```
Mặt phẳng ảnh (W × H)
┌─────────┬─────────┐
│  TL     │  TR     │  ← Chia cấp 1
│  [best] │  [best] │
├─────────┼─────────┤
│  BL     │  BR     │
│  [best] │  [best] │
└─────────┴─────────┘
  Đệ quy đến khi đạt minNodeSize hoặc maxDepth
```

Thuật toán (`quadtree.cpp`):

1. Nạp toàn bộ keypoint phát hiện được vào nút gốc bao phủ toàn bộ ảnh.
2. Chia đệ quy mỗi nút chứa nhiều hơn một keypoint thành bốn ô con bằng nhau, dừng khi `node.width/2 < minNodeSize` hoặc `depth > maxDepth`.
3. Từ mỗi **nút lá**, chọn keypoint duy nhất có **response ORB cao nhất** (độ mạnh góc).

Điều này đảm bảo tập keypoint cuối cùng có tối đa một đặc trưng trên mỗi ô không gian — thuộc tính thiết yếu cho quá trình tam giác hoá ổn định.

### 3.3 Ghép Đặc trưng Stereo

Các cặp tương ứng trái–phải được thiết lập bởi `ORBMatcher` sử dụng ghép **Brute-Force khoảng cách Hamming** kết hợp **kiểm tra tỉ lệ Lowe** (`orb_matcher.cpp`):

```cpp
// Kiểm tra tỉ lệ Lowe
if (m[0].distance < config_.ratioThreshold * m[1].distance) {
    if (m[0].distance <= config_.maxHammingDistance) {
        goodMatches.push_back(m[0]);
    }
}
```

Mặc định `ratioThreshold = 0.75`. Kiểm tra này loại bỏ các cặp ghép mơ hồ khi hai descriptor tham chiếu gần như tương đồng như nhau với truy vấn, một kỹ thuật kinh điển được Lowe mô tả lần đầu năm 2004.

### 3.4 Tam giác hoá Stereo — Phương pháp DLT

Với mỗi cặp tương ứng đã xác minh $(x_L, y_L) \leftrightarrow (x_R, y_R)$, `StereoTriangulator` phục hồi điểm 3D $\mathbf{X}$ bằng cách giải hệ **Direct Linear Transform** (`stereo_triangulation.cpp`):

$$\mathbf{A} \cdot \mathbf{X} = 0 \quad \text{trong đó} \quad \mathbf{A} \in \mathbb{R}^{4 \times 4}$$

Các hàng của $\mathbf{A}$ mã hoá ràng buộc tích chéo từ mỗi camera:

```cpp
A.row(0) = left.x  * P1.row(2) - P1.row(0);   // x_L × (P1 hàng 2) - P1 hàng 0
A.row(1) = left.y  * P1.row(2) - P1.row(1);   // y_L × (P1 hàng 2) - P1 hàng 1
A.row(2) = right.x * P2.row(2) - P2.row(0);   // x_R × (P2 hàng 2) - P2 hàng 0
A.row(3) = right.y * P2.row(2) - P2.row(1);   // y_R × (P2 hàng 2) - P2 hàng 1
```

trong đó $P_1 = K [I | \mathbf{0}]$ và $P_2 = K [I | -b\hat{\mathbf{x}}]$, với $b = 0.10$ m (baseline stereo). Nghiệm là vector kỳ dị phải tương ứng với giá trị kỳ dị nhỏ nhất (hàng cuối của $V^T$ sau SVD).

Một điểm tam giác hoá được đánh dấu **hợp lệ** chỉ khi:
- $Z > 0$ (trước camera).
- Sai số chiếu ngược $\leq$ `maxReprojError` (chiếu lại lên cả hai ảnh).
- $|x_L - x_R| \geq$ `minDisparity` (thị sai đủ lớn).

### 3.5 PnP Khởi đầu — Ràng buộc Diamond

Sau tam giác hoá, ước lượng tư thế ban đầu $(\mathbf{r}_0, \mathbf{t}_0)$ được tính bằng cách giải:

$$\min_{\mathbf{r}, \mathbf{t}} \sum_{i=1}^{4} \left\| \mathbf{p}_i - \pi\!\left(K, \mathbf{r}, \mathbf{t}, \mathbf{P}_i \right) \right\|^2$$

dùng `cv2.solvePnP` trên bốn cặp tương ứng góc Diamond. Kết quả này cung cấp điểm khởi đầu hình học vững chắc cho bộ tối ưu G2O.

---

## 4. Quy trình B — Hiệu chỉnh Tư thế Trực tuyến

Trong vòng lặp căn chỉnh, mỗi ảnh mới từ Pi được xử lý bởi `run_correction_step()` trong `correction.py`.

### 4.1 Sơ đồ Pipeline

```
Ảnh mới
    │
    ├─► detect_diamond()          → rvec₀, tvec₀ (PnP seed)
    │
    ├─► extract_orb() + QuadTree  → descriptors_current
    │
    ├─► match_with_3d_reference() → {points_3d, points_2d} từ golden pose
    │
    ├─► undistortPoints()         → khử méo lens
    │
    └─► hybrid_optimize()         → rvec_final, tvec_final (tinh chỉnh G2O)
             │
             └─► calculate_deviation() → Δx, Δz, Δpan, Δtilt → lệnh động cơ
```

### 4.2 Tối ưu hoá Hybrid G2O

Đây là cốt lõi của thuật toán hiệu chỉnh. `HybridPoseSolver` phát biểu bài toán ước lượng tư thế như một **bài toán tối ưu đồ thị bình phương tối thiểu phi tuyến** trên nhóm Lie SE(3):

**Cấu trúc đồ thị:**

| Phần tử | Loại | Số lượng | Ghi chú |
|---------|------|----------|---------|
| Đỉnh tư thế camera | `VertexSE3Expmap` | 1 | Biến tự do duy nhất |
| Đỉnh 3D Diamond | `VertexPointXYZ` (cố định) | 4 | Đã biết từ mô hình vật thể |
| Đỉnh 3D ORB | `VertexPointXYZ` (cố định) | N | Từ tam giác hoá stereo |
| Cạnh chiếu Diamond | `EdgeSE3ProjectXYZ` | 4 | Trọng số cao, không có robust kernel |
| Cạnh chiếu ORB | `EdgeSE3ProjectXYZ` | N | Trọng số thấp, Huber kernel |

**Trọng số ma trận thông tin:**

```cpp
// Cạnh Diamond — độ tin cậy cao, mục tiêu là vật thể cứng
Matrix2d info_diamond = Matrix2d::Identity() * config_.diamondWeight;
// mặc định: diamondWeight = 1e4

// Cạnh ORB — điểm mốc tam giác hoá, có nhiễu
Matrix2d info_orb = Matrix2d::Identity() * config_.orbWeight;
// mặc định: orbWeight = 1.0
```

Tỉ lệ trọng số $10^4$ mã hoá tri thức miền: hình học marker Diamond là chính xác (độ chính xác chế tạo), trong khi điểm mốc ORB tam giác hoá tích luũ nhiễu đo lường từ cả bước ghép stereo lẫn bài giải DLT.

**Huber robust kernel trên cạnh ORB:**

```cpp
RobustKernelHuber* huber = new RobustKernelHuber();
huber->setDelta(config_.huberDelta);   // mặc định: δ = 2.0 px
edge->setRobustKernel(huber);
```

Huber kernel chuyển hàm chi phí ORB từ bậc hai (chế độ inlier, $\|e\| < \delta$) sang tuyến tính (chế độ outlier, $\|e\| \geq \delta$):

$$\rho(e) = \begin{cases} \frac{1}{2} e^2 & \text{nếu } |e| \leq \delta \\ \delta(|e| - \frac{\delta}{2}) & \text{nếu } |e| > \delta \end{cases}$$

Điều này triệt tiêu ảnh hưởng của các ORB outlier — cặp ghép sai do sự mơ hồ của texture cảnh — mà không loại bỏ hoàn toàn chúng, do đó vẫn giữ được ràng buộc hình học mà chúng cung cấp.

**Backend tối ưu hoá:**

```cpp
typedef BlockSolver<BlockSolverTraits<6, 3>> BlockSolverType;
typedef LinearSolverPCG<BlockSolverType::PoseMatrixType> LinearSolverType;
auto solver = new OptimizationAlgorithmLevenberg(
    std::make_unique<BlockSolverType>(
        std::make_unique<LinearSolverType>()
    )
);
```

- **Levenberg–Marquardt** cho các vòng lặp phi tuyến ngoài (hội tụ mạnh mẽ).
- **PCG (Preconditioned Conjugate Gradient)** cho bài giải hệ tuyến tính bên trong (tiết kiệm bộ nhớ, phù hợp với hệ thưa lớn).
- Tư thế được tham số hoá bằng $\mathfrak{se}(3)$ qua `SE3Quat` (ánh xạ mũ đại số Lie đảm bảo ràng buộc đa tạp quay được thoả mãn).

### 4.3 Tính toán Sai lệch Tư thế

Sau tối ưu hoá, `DeviationCalculator::calculate()` tính phép biến đổi tương đối giữa tư thế tối ưu hiện tại và golden pose đã lưu (`deviation_calculator.cpp`):

$$\Delta \mathbf{T} = \mathbf{t}_\text{hiện tại} - \mathbf{t}_\text{golden}$$

$$R_\Delta = R_\text{hiện tại} \cdot R_\text{golden}^{T}$$

Các góc Euler $(\Delta\text{tilt}, \Delta\text{pan}, \Delta\text{roll})$ được trích xuất theo quy ước ZYX chuẩn với kiểm tra khóa gimbal:

```cpp
bool singular = (sy = sqrt(R(0,0)² + R(1,0)²)) < 1e-6;
if (!singular) {
    roll  = atan2( R(2,1),  R(2,2));   // Δtilt
    pitch = atan2(-R(2,0),  sy);       // Δpan
    yaw   = atan2( R(1,0),  R(0,0));   // Δroll
} else {
    // Khóa gimbal: pitch ≈ ±90°, trích xuất suy biến
}
```

Sai lệch được so sánh với ngưỡng cấu hình:

```python
TRANS_TOLERANCE = float(os.environ.get("TRANS_TOLERANCE_MM", "30.0")) / 1000.0  # mét
ROT_TOLERANCE   = float(os.environ.get("ROT_TOLERANCE_DEG",  "3.0"))             # độ
```

Nếu `withinTolerance = True`, vòng lặp căn chỉnh kết thúc và server publish thông điệp MQTT `alignment_complete`.

---

## 5. Giao diện Binding pybind11

Năm module C++ được expose ra Python qua `bindings.cpp`. Binding áp dụng giao diện native NumPy: mọi đối số ảnh và ma trận đều là `py::array_t<T>`, được chuyển đổi sang `cv::Mat` qua buffer protocol:

```cpp
static Mat numpyToMat(py::array_t<uint8_t> input) {
    py::buffer_info buf = input.request();
    if (buf.ndim == 2)
        return Mat(buf.shape[0], buf.shape[1], CV_8UC1, (uint8_t*)buf.ptr).clone();
    else if (buf.ndim == 3 && buf.shape[2] == 3)
        return Mat(buf.shape[0], buf.shape[1], CV_8UC3, (uint8_t*)buf.ptr).clone();
}
```

Các hàm Python được expose:

| Symbol Python | Cài đặt C++ | Kiểu trả về |
|---------------|--------------|-------------|
| `extract_with_quadtree(img, maxFeatures, minNodeSize, maxDepth)` | `QuadTree::distribute` | `dict{keypoints, descriptors, grid_cells}` |
| `match_stereo(desc_L, desc_R)` | `ORBMatcher::matchBruteForce` | `dict{matches[]}` |
| `match_with_3d_reference(kp_xy, desc, pts3d, ref_desc)` | `ORBMatcher::matchWith3DReference` | `dict{points_3d, points_2d, num_matches}` |
| `triangulate_stereo(pts_L, pts_R, K, D, baseline)` | `StereoTriangulator::triangulate` | `dict{points_3d, valid_mask, stats}` |
| `hybrid_optimize(rvec, tvec, d3d, d2d, o3d, o2d, K, D)` | `HybridPoseSolver::optimize` | `dict{rvec, tvec, chi2, inliers}` |
| `calculate_deviation(rvec_g, tvec_g, rvec_c, tvec_c, …)` | `DeviationCalculator::calculate` | `dict{deltaX, deltaZ, deltaPan, deltaTilt, withinTolerance}` |

---

## 6. Đường Dự phòng (Không có C++ Extension)

Lớp Python được thiết kế để xuống cấp nhẹ nhàng. Khi `.so` không khả dụng (`HAS_CPP = False`), `run_correction_step()` chuyển sang:

1. **PnP chỉ dùng Diamond** (không ghép ORB, không G2O).
2. **Trích xuất góc Euler thủ công** từ ma trận quay Rodrigues của OpenCV.
3. **Kiểm tra ngưỡng xấp xỉ** bằng chuẩn Euclidean.

Điều này đảm bảo hệ thống vẫn hoạt động — với độ chính xác giảm — trong quá trình phát triển hoặc môi trường thiếu chuỗi dependency đã biên dịch.

---

## 7. Định dạng Lưu trữ Golden Pose

Golden pose được lưu dưới dạng file YAML (`golden_pose_{artifact_id}.yaml`) trong thư mục dữ liệu của từng cổ vật:

```yaml
rvec:       [rx, ry, rz]          # Vector quay Rodrigues (rad)
tvec:       [tx, ty, tz]          # Vector tịnh tiến (m)
points_3d:  [[x, y, z], ...]      # Điểm mốc ORB tam giác hoá (hệ toạ độ thế giới)
points_2d:  [[u, v], ...]         # Chiếu 2D tương ứng trên ảnh trái
descriptors: [...]                # Descriptor ORB nhị phân (N × 32 byte)
baseline:   0.10                  # Baseline stereo dùng để tam giác hoá (m)
lens_position: 1.5                # Vị trí lens camera lúc chụp
```

Cơ chế tự động di chuyển dữ liệu trong `PoseService` quét tìm định dạng cũ (file `golden_pose.yaml` duy nhất) và chuyển về đường dẫn per-artifact một cách trong suốt, đảm bảo khả năng tương thích ngược.
