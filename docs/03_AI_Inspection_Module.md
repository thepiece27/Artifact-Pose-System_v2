# 03 — Module Kiểm tra AI: Phân tích Cấu trúc SSIM & Phát hiện Tổn hại YOLO

> **Đường dẫn nguồn:** `server/app/services/inspection_service.py`  
> **Backend mô hình AI:** `server/app/services/model_service.py`

---

## 1. Tổng quan Module

Module Kiểm tra có trách nhiệm so sánh một ảnh cổ vật mới chụp với ảnh baseline (tham chiếu) đã lưu và tạo ra báo cáo tổn hại định lượng. Module triển khai một **pipeline lai đa giai đoạn** kết hợp hai tín hiệu tổn hại bổ trợ nhau:

1. **SSIM (Chỉ số Tương đồng Cấu trúc):** Chỉ số thống kê cấp pixel đo lường sự suy giảm ảnh theo nhận thức trên các chiều cấu trúc, độ sáng và độ tương phản.
2. **YOLOv8 Object Detection:** Bộ phát hiện deep learning được huấn luyện để định vị các loại tổn hại cụ thể (vết nứt, vết xước, ăn mòn, v.v.) ở mức độ ngữ nghĩa.

Hai tín hiệu được kết hợp bởi **quy tắc phân loại max-rank thận trọng**: trạng thái cuối cùng là nhãn nghiêm trọng nhất từ một trong hai tín hiệu.

---

## 2. Pipeline Kiểm tra Đầu cuối

```
 Ảnh kiểm tra (bytes)
         │
         ▼
 [1] Căn chỉnh ảnh dựa trên SIFT
     cv2.SIFT → BFMatcher (L2, tỉ lệ 0.75) → findHomography (RANSAC)
     → warpPerspective → aligned_img
         │
         ▼
 [2] Resize về kích thước ảnh tham chiếu (nếu khác nhau)
         │
         ▼
 [3] Tính toán SSIM đa kênh
     SSIM thang xám (trọng số 0.6) + SSIM từng kênh màu (trọng số 0.4)
     → ssim_score  ∈ [0, 1]
     → diff_map    ∈ [0, 255] (độ lệch cấu trúc theo từng pixel)
         │
         ▼
 [4] Tạo heatmap JET
     diff_map → COLORMAP_JET → pha trộn với ảnh gốc (α=0.6 ảnh gốc, β=0.4 heatmap)
         │
         ▼
 [5] Trích xuất mặt nạ tổn hại
     GaussianBlur(5×5) → ngưỡng Otsu (fallback=60 nếu Otsu<30)
     → MORPH_OPEN(3×3 ellipse, ×2) → MORPH_CLOSE(7×7 ellipse, ×2)
     → findContours → lọc diện tích > max(500, H×W×0.0001) px²
         │
         ▼
 [6] damage_pct = Σ contour_area / valid_area × 100
         │
         ▼
 [7] Pipeline YOLO crop lai
     nếu damage_pct < 0.5%: YOLO toàn ảnh
     ngược lại:  Pha A (tight crop) → Pha B (wide crop, early-exit)
         │
         ▼
 [8] NMS (IoU ≥ 0.45) để khử trùng tight vs wide crop
         │
         ▼
 [9] Tính SSIM theo từng vùng trên mỗi bounding box YOLO
         │
         ▼
[10] classify_damage_status(ssim, damage_pct, detections)
     → ComparisonStatus: good | warning | damaged
         │
         ▼
[11] Lưu ImageComparison → tự động tạo Alert nếu warning/damaged
```

---

## 3. Giai đoạn 1: Căn chỉnh Ảnh dựa trên SIFT

Trước bất kỳ phép so sánh nào, ảnh kiểm tra phải được đăng ký hình học về hệ toạ độ tham chiếu. Dù hệ thống căn chỉnh tư thế đang hoạt động, các lệch nhỏ còn lại (biến dạng phối cảnh dưới pixel, cong vênh độc lập với chiếu sáng) vẫn có thể làm tăng sai số SSIM. Phép căn chỉnh dựa trên homography được áp dụng:

```python
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray_ref,   None)
kp2, des2 = sift.detectAndCompute(gray_current, None)

matcher = cv2.BFMatcher(cv2.NORM_L2)
knn = matcher.knnMatch(des1, des2, k=2)
good = [m for m, n in knn if m.distance < 0.75 * n.distance]

src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])

H, inlier_mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
aligned_img = cv2.warpPerspective(current_img, H, (w_ref, h_ref))
```

Mặt nạ `inlier_mask` từ RANSAC còn đóng vai trò **mặt nạ pixel hợp lệ** — vùng nằm ngoài vùng chồng lấp căn chỉnh thành công được loại khỏi phép tính SSIM và heatmap.

---

## 4. Giai đoạn 2: SSIM Đa kênh

SSIM thang xám cổ điển được mở rộng để tích hợp thông tin màu sắc. Điểm kết hợp cho trọng số 60% cho suy giảm cấu trúc (độ sáng/tương phản/cấu trúc) và 40% cho độ trung thực màu:

$$S_\text{combined} = 0.6 \cdot S_\text{gray} + 0.4 \cdot \frac{1}{3} \sum_{c \in \{B,G,R\}} S_c$$

```python
score_gray, diff_gray = structural_similarity(gray_ref, gray_src,
                                              full=True, win_size=7)
score_channels = []
diff_color = np.zeros_like(gray_src, dtype=np.float64)
for c in range(3):
    sc, dc = structural_similarity(reference_img[:,:,c],
                                   ssim_source[:,:,c],
                                   full=True, win_size=7)
    score_channels.append(sc)
    diff_color += (1.0 - dc)

diff_color /= 3.0
diff_combined = np.maximum(1.0 - diff_gray, diff_color)
ssim_score = 0.6 * float(score_gray) + 0.4 * score_color
```

Bản đồ `diff_combined` chọn, cho mỗi pixel, **tín hiệu lệch tệ nhất** trên thang xám và các kênh màu qua phép maximum theo phần tử. Điều này ngăn suy giảm không gian màu (ví dụ: đổi màu bề mặt, oxy hoá) bị che khuất bởi điểm cấu trúc cao.

Kích thước cửa sổ `win_size=7` phản ánh quy mô không gian mà tại đó độ tương đồng cấu trúc được đánh giá — phù hợp với kiểm tra cổ vật ở thang centimet đến decimet.

---

## 5. Giai đoạn 3: Trích xuất Mặt nạ Tổn hại

### 5.1 Ngưỡng Thích ứng

Bản đồ sai số unsigned 8-bit trải qua nhị phân hoá thích ứng:

```python
blurred = cv2.GaussianBlur(diff_uint8, (5, 5), 0)
otsu_thresh, damage_mask = cv2.threshold(blurred, 0, 255,
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
if otsu_thresh < 30:
    # Otsu failed: image pair is nearly identical, use hard fallback
    _, damage_mask = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
```

Phương pháp Otsu tự động phân hoạch histogram sai số thành lớp nền (không tổn hại) và lớp tiền cảnh (tổn hại) bằng cách tối đa hoá phương sai giữa các lớp. Ngưỡng fallback 60 kích hoạt khi cảnh có sai số tổng thể rất thấp, ngăn Otsu chọn ngưỡng phân đoạn quá mức nhiễu.

### 5.2 Hậu xử lý Hình thái

```python
kernel_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_b = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
damage_mask = cv2.morphologyEx(damage_mask, cv2.MORPH_OPEN,  kernel_s, iterations=2)
damage_mask = cv2.morphologyEx(damage_mask, cv2.MORPH_CLOSE, kernel_b, iterations=2)
```

| Phép toán | Kernel | Tác dụng |
|-----------|--------|---------|
| `MORPH_OPEN` (erosion → dilation) | ellipse 3×3 × 2 | Loại nhiễu muối tiêu, thu gọn false positive |
| `MORPH_CLOSE` (dilation → erosion) | ellipse 7×7 × 2 | Nối các khe hở nhỏ trong vùng tổn hại thực |

Phần tử cấu trúc hình ellipse được ưu tiên hơn kernel hình chữ nhật để tránh đưa vào độ lệch nhân tạo theo chiều ngang/dọc trong quá trình phân đoạn tổn hại.

### 5.3 Phần trăm Tổn hại

Sau khi lọc contour theo diện tích tối thiểu:

```python
min_area = max(500, (h * w) * 0.0001)   # at least 500 px², or 0.01% of image
big_contours = sorted([c for c in contours if cv2.contourArea(c) > min_area],
                      key=cv2.contourArea, reverse=True)[:50]  # cap at 50

damage_pct = (Σ contourArea(c) / valid_area) * 100.0
```

`valid_area` là số pixel khác không trong mặt nạ căn chỉnh — bỏ qua vùng đệm ngoài vùng chồng lấp inlier của RANSAC.

---

## 6. Giai đoạn 4: Pipeline YOLO Crop Lai

Suy luận YOLO toàn ảnh trực tiếp trên ảnh cổ vật độ phân giải cao vừa tốn kém tính toán vừa thiếu chính xác: các đặc điểm tổn hại nhỏ (vết nứt mao phát, vi chip) rơi xuống dưới độ phân giải phát hiện hiệu dụng. Hệ thống triển khai **chiến lược crop hai pha** được dẫn hướng bởi mặt nạ tổn hại SSIM.

### 6.1 Quyết định Định tuyến

```python
if damage_pct < self._DAMAGE_PCT_GATE or not big_contours:
    # Route A: Full-image YOLO (low damage, no candidate regions)
    _yolo_raw = self._model_service.detect_image(model_name, image_bytes)
else:
    # Route B: Crop-based YOLO (Phase A tight → Phase B wide)
    ...
```

`_DAMAGE_PCT_GATE = 0.5%`: nếu mặt nạ SSIM bao phủ dưới 0.5% ảnh, tổn hại có thể vắng mặt hoặc dưới ngưỡng hình thái, và một lần pass toàn ảnh được thực hiện như mạng an toàn.

### 6.2 Pha A — Tight Crop

Với mỗi contour tổn hại, một tight crop được tạo ra với padding tỉ lệ theo bounding box contour:

```python
_TIGHT_PAD_FACTOR = 0.30    # padding = 30% of max(width, height)
_TIGHT_MIN_PAD    = 40      # minimum 40 px padding

side = max(bw, bh)
pad  = max(min_pad, int(side * pad_factor))
half = (side // 2) + pad
cx, cy = bx + bw // 2, by + bh // 2
crop = image[cy-half : cy+half, cx-half : cx+half]
```

Toàn bộ tight crop được xử lý theo batch (kích thước sub-batch = 4, có thể cấu hình qua `_YOLO_SUB_BATCH`) để phòng OOM trên thiết bị không có GPU.

### 6.3 Pha B — Wide Crop (Ngữ cảnh)

Wide crop cung cấp ngữ cảnh không gian rộng hơn quanh mỗi contour, cải thiện phát hiện tổn hại có ranh giới vượt ra ngoài bounding box tight:

```python
_WIDE_PAD_FACTOR = 0.70     # 70% of contour side
_WIDE_MIN_PAD    = 120      # minimum 120 px padding
```

**Tối ưu hoá early-exit:** Nếu tight crop đã cho kết quả phát hiện với confidence $\geq$ `_EARLY_EXIT_CONF` (0.40), wide crop cho contour đó bị **bỏ qua**. Điều này giảm suy luận thừa trên vùng đã được xác định tự tin.

```python
if max_conf >= self._EARLY_EXIT_CONF:
    continue   # skip wide crop for this contour
```

### 6.4 Non-Maximum Suppression

Sau khi gộp các phát hiện tight và wide, NMS được áp dụng toàn cục:

```python
_NMS_IOU_THRESH = 0.45
```

Các bounding box có IoU $> 0.45$ được khử trùng bằng cách giữ lại phát hiện có confidence cao nhất. Điều này loại bỏ vùng chồng lấp có hệ thống do chiến lược dual-crop tạo ra, khi cùng một vùng tổn hại vật lý có thể sinh ra phát hiện ở cả hai pha.

### 6.5 Tính SSIM Theo Vùng

Mỗi bounding box YOLO còn sót sau NMS được chú thích bằng **điểm SSIM cấp vùng** tính riêng cho vùng phát hiện:

```python
def _compute_region_ssim(self, src, ref, x1, y1, x2, y2, padding=0):
    # Crop both images to the bounding box region
    # Compute SSIM on the cropped patch
    ...
```

Điểm per-region này làm giàu bản ghi phát hiện lưu trong `detections_json`, cho phép Flutter client hiển thị chỉ số suy giảm cục bộ cạnh mỗi bounding box.

---

## 7. Logic Phân loại

### 7.1 Mức độ dựa trên SSIM

| Điều kiện | Trạng thái |
|-----------|-----------|
| `ssim ≥ 0.95` VÀ `damage_pct < 2.0%` | `good` |
| `ssim ≥ 0.85` VÀ `damage_pct < 10.0%` | `warning` |
| Trường hợp khác | `damaged` |

### 7.2 Mức độ dựa trên YOLO

| Confidence phát hiện tối đa | Trạng thái |
|-----------------------------|-----------|
| `conf ≥ 0.65` | `damaged` |
| `0.40 ≤ conf < 0.65` | `warning` |
| `conf < 0.40` | `good` |

### 7.3 Quy tắc Kết hợp

```python
_rank = {ComparisonStatus.good: 0,
         ComparisonStatus.warning: 1,
         ComparisonStatus.damaged: 2}
status = max([ssim_status, yolo_status], key=lambda s: _rank[s])
```

Quy tắc **maximum thận trọng** là lựa chọn thiết kế chủ ý trong bảo tồn di sản: tốt hơn là phát cảnh báo false-positive (cảnh báo trên cổ vật không tổn hại) còn hơn bỏ lỡ một sự kiện tổn hại thực. Tín hiệu SSIM cung cấp phạm vi bao phủ rộng; tín hiệu YOLO cung cấp tính đặc thù ngữ nghĩa.

### 7.4 Lan truyền Trạng thái Cổ vật

Trường `status` toàn cục của cổ vật được cập nhật theo chính sách leo thang đơn điệu:

```python
priority = {"good": 0, "archived": 0, "need_check": 1,
            "maintenance": 1, "warning": 2, "damaged": 3}
# Status can only increase, never decrease
artifact.status = new_status if priority[new_status] > priority[current] else current
```

Điều này ngăn một lần kiểm tra “good” tiếp theo che khuất trạng thái damaged đã ghi trước — yêu cầu kiểm toán thiết yếu trong hồ sơ bảo tồn.

---

## 8. Tạo Cảnh báo

```python
if status in [ComparisonStatus.warning, ComparisonStatus.damaged]:
    alert_level = AlertLevel.high if status == ComparisonStatus.damaged \
                  else AlertLevel.medium
    alert = Alert(
        artifact_id=artifact.artifact_id,
        comparison_id=comparison.comparison_id,
        alert_level=alert_level,
        is_handled=False,
    )
    db.add(alert)
```

Cảnh báo luôn được liên kết với một bản ghi `ImageComparison` cụ thể, cung cấp dấu vết kiểm toán hoàn chỉnh từ cảnh báo ngược về ảnh thô, heatmap và JSON phát hiện.

---

## 9. Kiến trúc Model Service

Lớp `ModelService` cung cấp một **registry mô hình không phụ thuộc backend** hỗ trợ ba backend suy luận:

| Khoá backend | Lớp | Trường hợp sử dụng |
|-------------|-----|------------------|
| `yolo` / `ultralytics` | `UltralyticsYoloRuntimeModel` | Trọng số YOLOv8 `.pt` tiêu chuẩn |
| `onnx` | `OnnxRuntimeModel` | ONNX Runtime, chỉ CPU |
| `torchscript` | `TorchScriptRuntimeModel` | TorchScript `.pt` (không phải YOLO) |

Phát hiện backend tự động từ phần mở rộng file:

```python
if suffix == ".onnx":   return "onnx"
if suffix in {".pt", ".pth"}: return "yolo"
# .pt is assumed to be Ultralytics YOLO
# TorchScript requires explicit backend="torchscript"
```

Lúc khởi động, `state.py` gọi `_auto_load_default_model()`:

```python
# If DEFAULT_AI_MODEL_PATH is empty → scan model_dir for first *.pt
pt_files = sorted(self.settings.model_dir.glob("*.pt"))
```

Mô hình được nạp một lần vào bộ nhớ và dùng chung cho mọi request. Thread safety được đảm bảo bởi `threading.Lock` bao quanh mọi thao tác với dictionary `_models`.

---

## 10. Tham chiếu Điều chỉnh Pipeline

Toàn bộ siêu tham số pipeline được tập trung dưới dạng hằng số cấp lớp trong `InspectionService`, được chú thích inline để dễ điều chỉnh:

| Hằng số | Mặc định | Tác dụng |
|---------|---------|--------|
| `_YOLO_CONF` | `0.15` | Ngưỡng confidence phát hiện YOLO |
| `_YOLO_SUB_BATCH` | `4` | Số crop mỗi lần gọi `predict()` (bảo vệ OOM) |
| `_MIN_AREA_FACTOR` | `0.0001` | Diện tích contour tối thiểu theo tỉ lệ ảnh |
| `_DAMAGE_PCT_GATE` | `0.5` | Chuyển từ crop-YOLO sang YOLO toàn ảnh |
| `_OTSU_FALLBACK` | `60` | Ngưỡng cứng khi Otsu < 30 |
| `_CONTOUR_CAP` | `50` | Số contour tối đa để xử lý |
| `_TIGHT_PAD_FACTOR` | `0.30` | Tỉ lệ padding tight crop |
| `_TIGHT_MIN_PAD` | `40 px` | Padding tối thiểu tight crop |
| `_WIDE_PAD_FACTOR` | `0.70` | Tỉ lệ padding wide crop |
| `_WIDE_MIN_PAD` | `120 px` | Padding tối thiểu wide crop |
| `_NMS_IOU_THRESH` | `0.45` | Ngưỡng IoU khử trùng NMS |
| `_EARLY_EXIT_CONF` | `0.40` | Bỏ wide crop nếu tight det conf ≥ giá trị này |
