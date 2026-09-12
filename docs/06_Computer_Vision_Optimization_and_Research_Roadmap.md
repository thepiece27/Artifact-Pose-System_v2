# Định hướng tối ưu Computer Vision và phát triển hệ thống Artifact Pose

> Ngày rà soát: 2026-09-12  
> Snapshot mã nguồn: commit 83aeaf1  
> Phạm vi: pose estimation, thu nhận và xử lý ảnh, nhận diện hư hại, tài nguyên tính toán, phần cứng, dữ liệu–thực nghiệm, triển khai và quản lý hệ thống.

**Trạng thái bằng chứng**

- PASS — đọc và đối chiếu tài liệu với mã nguồn pose, camera, inspection, model runtime, Docker và tiền xử lý dataset tại snapshot trên.
- PASS — nghiên cứu web từ tài liệu chính thức, trang nhà sản xuất và paper gốc; các nguồn được liệt kê cuối báo cáo.
- NOT RUN — chưa có camera, rig, actuator, model và dataset runtime trong phiên rà soát, nên không tuyên bố độ chính xác vật lý hay model mới chắc chắn thắng.
- NOT RUN — test tự động không chạy được trong môi trường hiện tại vì Python thiếu pytest; đây là giới hạn môi trường, không phải kết luận test của ứng dụng bị lỗi.

## 1. Kết luận ngắn

Không có một công nghệ nào “tốt hơn ChArUco trong mọi trường hợp”. Với đề tài này, phương án tốt nhất không phải xóa ChArUco rồi thay bằng AI, mà là:

1. **Giữ fiducial làm mốc hình học tuyệt đối**, nhưng thay Diamond đơn lẻ bằng một bảng cứng, lớn và dư thừa hơn; A/B test ChArUco board với AprilGrid.
2. **Hiệu chuẩn độc lập** camera, chuyển động stereo/slider và hand–eye. Không dùng chính Diamond đang đo làm bằng chứng duy nhất để suy ra baseline rồi gọi kết quả là metrology.
3. **Tách hai lần chụp**:
   - ảnh pose: phơi sáng ngắn, ưu tiên chống nhòe và nhận marker;
   - ảnh kiểm tra: camera đứng yên, ánh sáng/màu cố định, độ phân giải cao, ưu tiên RAW hoặc pipeline tuyến tính.
4. **Thay chuỗi “SIFT homography + ngưỡng SSIM + YOLO” bằng tổ hợp có kiểm chứng**:
   - đăng ký ảnh bằng pose/3D và feature matcher;
   - anomaly detection học từ ảnh bình thường;
   - detector/segmenter cho lỗi đã biết;
   - hợp nhất kết quả bằng ngưỡng được khóa trên validation và trạng thái UNKNOWN khi thiếu bằng chứng.
5. **Đặt vòng điều khiển pose ở edge/local**, chỉ đưa kiểm tra nặng, lưu trữ, huấn luyện và quản lý đội thiết bị lên máy chủ khác hoặc cloud.
6. **Đo trước khi tuyên bố tốt hơn**: sai số pose với ground truth độc lập, false-negative theo mức độ lỗi, độ lặp lại theo ngày/camera/ánh sáng, latency p95, RAM, điện năng và tỷ lệ từ chối.

Kiến trúc mục tiêu phù hợp nhất trong 6–12 tháng là một **edge server tại chỗ**: Raspberry Pi chỉ điều khiển camera/cơ cấu; một mini-PC hoặc Jetson chạy pose và inference; PostgreSQL giữ metadata; object storage giữ ảnh; job queue tách tác vụ CV khỏi API; cloud chỉ quản lý fleet, sao lưu và retraining. Đây là điểm cân bằng tốt giữa chi phí, riêng tư, độ trễ và khả năng phát triển.

## 2. Những gì hệ thống hiện tại đã làm đúng

Báo cáo cũ không nên được dùng nguyên trạng vì nhánh hiện tại đã cải thiện nhiều điểm:

- MQTT đã yêu cầu tài khoản; cấu hình production kiểm tra secret, CORS và khóa thiết bị.
- API có JWT/device authentication, giới hạn dung lượng và số pixel upload.
- Trạng thái command/alignment đã được ghi xuống tệp thay vì chỉ tồn tại trong RAM.
- Lỗi phân tích không còn tự động biến thành “good”; NMS hiện phân theo class.
- Inference có semaphore chống cạn RAM; container server chạy non-root.
- Pose dùng IPPE_SQUARE, giữ hai nghiệm để phát hiện nhập nhằng phẳng; matching ORB hai chiều; kiểm tra epipolar, parallax, độ sâu và reprojection; golden schema gắn fingerprint calibration.
- Python/native đã có hợp đồng API rõ hơn và descriptor từ native được copy sang bộ nhớ sở hữu.

Các điểm trên là nền tốt. Những hạn chế dưới đây là khoảng cách còn lại tới một hệ thống đo và kiểm tra có bằng chứng, không phải phủ nhận phần đã hoàn thành.

## 3. Điểm yếu hiện tại

### 3.1 Pose và hình học

| Mức | Quan sát từ mã nguồn | Hệ quả |
|---|---|---|
| Rất cao | Golden hybrid được server tạo mà không truyền stereo transform; khi thiếu transform, mã suy ra T_right_left từ pose của cùng Diamond và tự ghi nhãn diamond_estimate_not_independent_metrology ([pose_service.py](../server/app/services/pose_service.py#L150), [initialize.py](../server/app/modules/artifact_pose/initialize.py#L32)). | Baseline/extrinsic không độc lập với mốc đang đánh giá; sai số marker, in, mặt phẳng và calibration cùng đi vào cả landmark 3D lẫn pose. |
| Rất cao | Motor command trong pose core luôn disabled vì chưa có trục, dấu, pivot, hand–eye và driver đã hiệu chuẩn ([correction.py](../server/app/modules/artifact_pose/correction.py#L31)). | Có thể chứng minh ước lượng pose, chưa thể tuyên bố vòng kín định vị cơ khí chính xác. |
| Cao | Target cố định là một ChArUco Diamond 40 mm, marker 25 mm, dictionary 4x4_50; chỉ chấp nhận đúng một bộ ID ([common.py](../server/app/modules/artifact_pose/common.py#L20), [common.py](../server/app/modules/artifact_pose/common.py#L88)). | Ít dư thừa, dễ mất hoàn toàn khi che khuất, nhòe, phản xạ hoặc ra khỏi khung; pose phẳng vẫn có thể nhập nhằng. |
| Cao | Pose refinement là pose-only với landmark cố định, không phải joint bundle adjustment và không có covariance vật lý ([geometry.py](../server/app/modules/artifact_pose/geometry.py#L293)). | Không thể diễn giải condition number hay ngưỡng pixel thành độ không đảm bảo mm/degree. |
| Cao | Các sigma/ngưỡng là engineering defaults, không phải noise model đo được ([geometry.py](../server/app/modules/artifact_pose/geometry.py#L1)). | “Trong tolerance” chưa đồng nghĩa “đúng với xác suất đã biết”. |
| Trung bình | ORB WTA_K=2 và quadtree là lựa chọn cố định; chưa có benchmark với XFeat, LightGlue, EfficientLoFTR hoặc dense matching ([common.py](../server/app/modules/artifact_pose/common.py#L115)). | Có thể bỏ lỡ độ bền dưới thay đổi góc nhìn, texture lặp, ánh sáng hoặc chi tiết yếu. |
| Trung bình | Corner refinement bị đặt CORNER_REFINE_NONE và cửa sổ threshold cố định; chưa có profile theo kích thước marker trên ảnh. | Không nhất thiết sai, nhưng chưa có thực nghiệm chứng minh cấu hình tối ưu cho camera/lens/giấy in thực tế. |
| Cao | Không thấy test pose/CV trong server/tests; chủ yếu là upload và MQTT. | Regression hình học có thể lọt qua CI; chưa tái lập được độ chính xác từ repository. |

### 3.2 Thu nhận và xử lý điểm ảnh

- Chế độ capture_simple dùng lens manual nhưng AE/AWB vẫn tự động ([camera_manager.py](../embed/device_agent/adapters/camera_manager.py#L29)). Hai ảnh cùng hiện vật có thể khác sáng/màu dù hiện vật không đổi.
- Chế độ high_quality có stream RAW nhưng chỉ lưu main RGB888 thành PNG ([camera_manager.py](../embed/device_agent/adapters/camera_manager.py#L126)). Dữ liệu Bayer tuyến tính chưa được lưu để radiometric calibration.
- Camera được mở, warm-up rồi đóng cho mỗi lần chụp. Điều này tăng latency và làm trạng thái AE/AF/thermal khó lặp lại.
- Chưa có dark-frame, flat-field, lens shading, color target, white reference, kiểm tra flare, MTF/focus, chiếu sáng chuẩn hoặc polarization.
- Một camera/resolution/lens position được ràng buộc bằng fingerprint là tốt, nhưng chưa có registry calibration theo từng serial thiết bị, nhiệt độ, focus và phiên bản ISP.
- Pipeline enhancement huấn luyện áp CLAHE, unsharp và HSV boost cố định ([preprocess_2stage.py](../ai_module/preprocess_2stage.py#L59)). Các phép này có thể làm lỗi nổi hơn, nhưng cũng thay đổi màu/texture và sinh halo. Không nên dùng ảnh đã enhancement làm bằng chứng đo màu.

### 3.3 Đăng ký ảnh và phát hiện hư hại

- Trạng thái cuối dùng ngưỡng cố định SSIM 0.95/0.85, diện tích 2%/10% và confidence YOLO 0.65/0.40 ([inspection_service.py](../server/app/services/inspection_service.py#L233)). Đây không phải xác suất đã calibration và chưa gắn với mức độ nghiêm trọng nghiệp vụ.
- Đăng ký dùng một homography SIFT ([inspection_service.py](../server/app/services/inspection_service.py#L707)). Phù hợp vật thể gần phẳng; relief, parallax hoặc thay đổi viewpoint 3D sẽ để lại “hư hại giả”.
- SSIM màu hiện là trung bình từng kênh BGR, không phải sai khác màu cảm nhận như Lab/ΔE00; SSIM cũng nhạy với exposure và registration.
- Otsu, morphology, diện tích contour, padding và cap 50 đều hard-code ([inspection_service.py](../server/app/services/inspection_service.py#L291)). Không có tuning theo kích thước lỗi vật lý hoặc độ phân giải.
- Region SSIM trả 1.0 khi crop quá nhỏ hoặc có exception ([inspection_service.py](../server/app/services/inspection_service.py#L753)). Đây là đường false-normal cục bộ; phải trả UNKNOWN/NaN kèm reason.
- Crop được encode JPEG rồi decode ngay trước inference ([model_service.py](../server/app/services/model_service.py#L136)). Vừa tốn CPU, vừa thêm artefact nén có thể che vết xước nhỏ.
- Adapter ONNX chỉ dùng CPUExecutionProvider và nhận tensor generic ([model_service.py](../server/app/services/model_service.py#L37]); chưa phải đường deploy YOLO/segmentation ONNX có preprocess, postprocess và accelerator hoàn chỉnh.
- Một baseline đơn không biểu diễn được nhiễu bình thường do ánh sáng, camera, mùa, thao tác hoặc nhiều góc nhìn.
- Heatmap/aligned image lưu JPEG; phù hợp hiển thị, không nên là dữ liệu đo gốc.

### 3.4 Dữ liệu và tính khoa học

- BRIDGE_N=80 lấy ảnh từ miền test ban đầu đưa vào train/validation ([preprocess_2stage.py](../ai_module/preprocess_2stage.py#L53), [preprocess_2stage.py](../ai_module/preprocess_2stage.py#L309)). Đây có thể là domain adaptation hợp lệ, nhưng tập datatest còn lại không còn là external test hoàn toàn chưa nhìn thấy.
- Split hiện theo primary class của từng ảnh, không theo artifact, phiên chụp, camera, người gán nhãn hoặc site ([preprocess_2stage.py](../ai_module/preprocess_2stage.py#L207)). Các frame gần nhau có thể rơi vào train và validation.
- Repository không chứa model/dataset/version manifest đủ để người khác tái lập metric. Requirements dùng lower-bound thay vì lockfile; Docker tải g2o theo tag thay vì digest/commit.
- Chưa có protocol ground truth, inter-annotator agreement, confidence interval, calibration curve, kiểm tra drift hoặc ma trận chi phí false-negative/false-positive.
- MVTec AD 2 cho thấy benchmark anomaly hiện đại cần xét ảnh độ phân giải cao, lỗi rất nhỏ và thay đổi ánh sáng chưa thấy trong train; đồng thời đo runtime và memory, không chỉ AUROC.[^S14]

### 3.5 Kiến trúc và tài nguyên

- FastAPI, pose, SSIM/SIFT, YOLO, quản lý model, file I/O và orchestration cùng một process/container; tác vụ CPU nặng có thể kéo latency API.
- Docker hiện cài PyTorch CPU; chưa có image/profile riêng CPU, CUDA, TensorRT, OpenVINO/Hailo.
- Ảnh, golden pose và state nằm trên filesystem node. PostgreSQL chỉ giữ metadata/path; scale ngang hoặc failover sẽ khó.
- Compose chưa khai báo CPU/RAM limit, log rotation, queue worker, object storage, backup policy và observability metric cho CV.
- Uvicorn chạy một worker. Tăng worker tùy ý sẽ nhân bản model trong RAM; phải tách inference server/worker trước.

## 4. ChArUco có nên thay không?

### 4.1 So sánh theo mục tiêu

| Giải pháp | Điểm mạnh | Điểm yếu | Vai trò phù hợp |
|---|---|---|---|
| ChArUco board lớn | Corner nội suy chính xác, dùng được khi thấy một phần board; OpenCV hỗ trợ trực tiếp calibration và pose.[^S1] | Cần in/phẳng/đo kích thước tốt; quá nhỏ hoặc bóng gây mất corner; phải quản lý legacy pattern giữa phiên bản OpenCV. | **Khuyến nghị chính cho absolute anchor và calibration chi phí thấp.** |
| ChArUco Diamond hiện tại | Gọn, ID rõ, pipeline đã có IPPE và kiểm tra nhập nhằng. | Chỉ bốn góc ngoài, ít redundancy; mất một cụm marker có thể mất pose. | Giữ làm baseline tương thích, không nên là cấu hình cuối duy nhất. |
| AprilGrid | Nhiều AprilTag, nhận được một phần bảng; Kalibr khuyến nghị vì pose bảng được phân giải đầy đủ và tránh flip trong bài toán calibration.[^S4] | Corner tag không mặc nhiên chính xác hơn ChArUco; cần benchmark detector và kích thước tag. | **Ứng viên A/B mạnh cho calibration stereo/hand–eye.** |
| AprilTag board | Robust ID, layout linh hoạt, họ tagStandard41h12 được dự án AprilTag khuyến nghị cho đa số trường hợp.[^S3] | Vẫn là planar fiducial; độ chính xác phụ thuộc pixel footprint, blur, lens và print. | Tracking xa/che khuất hoặc nhiều camera. |
| ArUco Nano | Paper SoftwareX 2026 báo cáo tốc độ/F1 tốt hơn OpenCV trong bộ thử của tác giả.[^S5] | Công bố của chính tác giả; không tương đương ngay với ChArUco board và chưa được kiểm tra trong miền này. | Candidate tối ưu CPU, chỉ dùng sau benchmark. |
| Coded circular target/active LED–IR | Tâm ellipse/subpixel tốt, chịu blur hoặc thiếu texture; active marker tách nền rõ. | Phần cứng/in ấn/đồng bộ phức tạp; có thể ảnh hưởng ảnh bảo tồn. | Rig công nghiệp hoặc motion nhanh. |
| Natural features ORB/SIFT | Không cần marker trên hiện vật, rẻ, đã tích hợp. | Texture/ánh sáng/viewpoint phụ thuộc; khó làm absolute truth. | Refinement sau fiducial, không làm chuẩn duy nhất. |
| XFeat + LightGlue | XFeat nhắm tới matching nhẹ/CPU; LightGlue thích nghi số layer và pruning theo độ khó.[^S6][^S7] | Cần model, runtime và dữ liệu kiểm tra; latency phụ thuộc ảnh/accelerator. | Thay ORB ở edge/server nếu A/B thắng. |
| EfficientLoFTR/RoMa | Semi-dense/dense, mạnh hơn trong vùng ít keypoint và thay đổi viewpoint; RoMa khai thác feature foundation model.[^S8][^S9] | Nặng, cần GPU, có thể tạo correspondence sai có cấu trúc. | Recovery/offline QA hoặc server GPU. |
| FoundationPose/SAM-6D | Pose 6D vật thể mới từ CAD/reference và RGB-D; giảm phụ thuộc marker.[^S10][^S11] | Đắt tính toán, phụ thuộc depth/CAD/segmentation; không tự biến thành metrology. | Nhánh nghiên cứu markerless và fallback. |
| VGGT/FoundationStereo | Xu hướng geometry foundation model đa ảnh/stereo 2025–2026.[^S12][^S13] | Mới, nặng và domain-dependent; không nên điều khiển actuator trực tiếp khi chưa uncertainty/gate. | Bootstrap 3D, tạo pseudo-label, QA offline. |

### 4.2 Quyết định đề xuất

**Không bỏ ChArUco ở phiên bản kế tiếp.** Thực hiện ba nhánh song song:

- Baseline A: Diamond hiện tại.
- Candidate B: ChArUco board 5×7 hoặc lớn hơn, đặt trên plate cứng, nhiều corner nhìn thấy.
- Candidate C: AprilGrid cùng kích thước vật lý và cùng điều kiện.

Board nên gắn vào cradle/rig, không dán lên bề mặt hiện vật. Đo kích thước thực sau khi in; kiểm tra độ phẳng; lưu board definition, printer, vật liệu, nhiệt độ và SHA. OpenCV đã thay đổi quy ước ChArUco sau 4.6 nên phải khóa version và setLegacyPattern đúng với bảng cũ.[^S1]

Chọn thắng theo median/p95 translation–rotation error, failure rate, pose flip, vùng nhìn thấy, blur, glare, khoảng cách và latency trên chính phần cứng đích. Có thể kết quả cuối là **AprilGrid cho calibration, ChArUco cho pose vận hành và learned features cho refinement**.

## 5. Pipeline ảnh và Computer Vision đề xuất

### 5.1 Tầng quang học và cảm biến: ưu tiên trước model

1. Rig kín sáng hoặc hood; nền và khoảng cách cố định.
2. LED có nguồn dòng ổn định; đo độ lặp lại theo thời gian warm-up.
3. Hai chế độ ánh sáng: diffuse để đo màu/texture tổng quát; raking light để lộ xước, nứt, bong.
4. Cross-polarization giữa đèn và lens để giảm specular; lưu cả ảnh polarized và unpolarized nếu độ bóng là tín hiệu.
5. ColorChecker/gray card ở vùng không che hiện vật; không nhất thiết xuất hiện trong ảnh cuối nếu đã có phiên calibration.
6. Dark frame, flat field và lens-shading correction theo camera/lens/exposure.
7. Khóa exposure, gain, white balance, focus sau warm-up; ghi metadata vào mỗi ảnh.
8. Lưu Bayer RAW/DNG hoặc ít nhất PNG/TIFF lossless 16-bit cho master; JPEG chỉ cho thumbnail.
9. Chụp burst 3–5 frame và chọn frame theo blur/focus/glare; median chỉ dùng nếu không làm mất tín hiệu nhỏ.
10. Đo MTF/focus, noise, dynamic range và non-uniformity theo tinh thần EMVA 1288; không tuyên bố tuân thủ nếu chưa làm đủ chuẩn.[^S23]
11. Dùng FADGI/ISO 19264-1 làm khung tham chiếu cho màu, độ phân giải và tính nhất quán ảnh di sản; cần mua/đọc đầy đủ tiêu chuẩn trước khi tuyên bố compliance.[^S21][^S22]
12. Với chuyển động: global shutter và trigger/sync. Raspberry Pi Global Shutter Camera giảm méo rolling shutter nhưng chỉ 1.58 MP, nên hợp pose hơn kiểm tra lỗi siêu nhỏ.[^S24]
13. Dùng hai camera: global-shutter cho pose và camera high-resolution cho inspection, nếu một camera không thỏa cả hai.
14. Với bề mặt nổi: multi-light/Reflectance Transformation Imaging; Historic England mô tả đây là kỹ thuật tương đối rẻ để làm rõ chi tiết bề mặt.[^S20]
15. Phương án cao cấp: photometric stereo, structured-light/laser profiler, RGB-D, multispectral/hyperspectral, UV fluorescence hoặc thermal. Chỉ thêm modality khi có câu hỏi vật lý rõ ràng và ground truth tương ứng.

### 5.2 Tách pipeline đo và pipeline hiển thị

**Measurement branch** phải giữ ảnh tuyến tính và calibration:

- black-level → bad-pixel → flat-field → demosaic cố định;
- radiometric/color calibration;
- undistort bằng calibration đúng serial/focus/resolution;
- valid-mask và uncertainty mask;
- đăng ký 2D/3D;
- ΔE00, gradient/texture residual, anomaly/segmentation;
- lưu provenance và hash.

**Visualization/AI branch** có thể tone-map, CLAHE, sharpen hoặc tăng saturation. Mọi enhancement phải giống nhau giữa train và inference và có ablation. Không dùng super-resolution/generative restoration để đo hư hại; chỉ cho trình bày, vì mô hình có thể tạo hoặc xóa chi tiết.

### 5.3 Đăng ký ảnh tốt hơn

Thứ tự ưu tiên:

1. Dùng pose camera và mô hình mặt phẳng/mesh để warp có cơ sở hình học.
2. Coarse alignment bằng fiducial; fine alignment trong ROI hiện vật.
3. Với mặt phẳng: homography + mutual matching + geometric verification.
4. Với relief: depth/mesh và render correspondence, hoặc piecewise affine/TPS có regularization.
5. A/B ORB, SIFT, XFeat+LightGlue, EfficientLoFTR và RoMa; khóa budget latency.
6. Kiểm tra cycle consistency, spatial coverage, reprojection residual, Jacobian/condition, overlap và valid-mask.
7. Nếu alignment không đạt gate: **UNKNOWN**, không resize fallback rồi tiếp tục kết luận “good”.

### 5.4 Phát hiện hư hại

Một ensemble có thể triển khai theo bốn tầng:

1. **Rule-based đã calibration**: ΔE00, high-frequency/gradient, raking-light residual, depth change. Dễ giải thích.
2. **Normal-only anomaly detection**: PatchCore cho baseline mạnh, EfficientAD cho latency thấp, AnomalyDINO cho few-shot feature DINOv2.[^S15][^S16][^S18]
3. **Supervised detector/segmenter**: YOLO-seg nhẹ ở edge; Mask2Former/U-Net/SegFormer ở server cho biên lỗi. Mask2Former là kiến trúc segmentation đa nhiệm đáng làm baseline, không mặc nhiên là lựa chọn nhanh nhất.[^S19]
4. **Foundation model hỗ trợ dữ liệu**: SAM 3/SAM 2 để gợi ý mask khi annotation, human duyệt; không dùng mask tự động chưa kiểm tra làm ground truth.[^S30]

Với lỗi nhỏ trên ảnh 4K, thêm tiled inference/SAHI thay vì chỉ crop theo contour SSIM; SAHI được thiết kế cho small-object slicing.[^S31] Tuy nhiên phải de-duplicate đúng class và đo false-negative ở biên tile.

WinCLIP/zero-shot hữu ích để thăm dò class mới, nhưng không nên triển khai chỉ vì pixel AUROC cao; paper/supplement cho thấy metric thresholded có thể kém hơn nhiều.[^S17] Với bảo tồn, false-negative theo severity quan trọng hơn leaderboard tổng quát.

## 6. Các phương án phần cứng

Thông số dưới đây là thông số nhà sản xuất tại ngày rà soát, không phải throughput đo trên model của đề tài.

| Mức | Cấu hình | Phù hợp | Lưu ý |
|---|---|---|---|
| Tận dụng hiện có | Raspberry Pi + Camera Module 3; inference nhẹ trên server CPU | Demo, thu dữ liệu, pose chậm khi vật đứng yên | Cần khóa AE/AWB và lưu RAW; không cố chạy foundation model. |
| Chi phí thấp | Pi 5 + AI HAT+ 2 | Detector/segmenter đã convert, edge offline | HAT+ 2 công bố 40 TOPS INT4 và 8 GB RAM riêng; phải đo operator support và accuracy sau quantization.[^S25] |
| Cân bằng | Jetson Orin Nano Super + camera hiện có/camera GS | XFeat, YOLO-seg, EfficientAD, local service | NVIDIA công bố tới 67 TOPS, 7–25 W; dev kit niêm yết USD 249 khi ra mắt, chưa gồm camera/lưu trữ/thuế.[^S26] |
| Nhiều camera | Jetson Orin NX hoặc mini-PC Intel + GPU rời | Multi-stream, segmentation, local server | Orin NX công bố tới 157 TOPS; cần benchmark thermal và memory.[^S27] |
| Camera tích hợp | OAK-D Pro W hoặc RealSense D555 | Stereo/depth, PoE, edge feature | Depth có thể yếu trên vật phản xạ/tối/ít texture; vẫn phải calibration với rig. OAK-D Pro W công bố 4 TOPS và baseline 75 mm; D555 dùng global shutter, baseline 95 mm và PoE.[^S28][^S29] |
| Công nghiệp | Basler ace 2 5–25 MP, global shutter, lens C-mount, trigger | Độ lặp lại, sync, optics chọn được | Chi phí lens/đèn/frame grabber/PoE và calibration cao hơn camera maker.[^S32] |
| Rất cao | AGX Orin/RTX workstation hoặc Jetson Thor | RoMa, FoundationPose, multi-view/foundation models | Chỉ hợp lab/server; Jetson Thor công bố 128 GB và 40–130 W, quá mức cần thiết cho vòng pose đơn.[^S33] |

Nguyên tắc mua: tạo benchmark replay từ 100–500 ảnh trước, đo p50/p95 latency, peak RAM/VRAM, nhiệt độ, điện năng, accuracy FP32/FP16/INT8 và chi phí chuyển đổi. TOPS không dự đoán trực tiếp hiệu năng OpenCV, SIFT hay model chưa được compiler hỗ trợ.

## 7. Đưa hệ thống ra nơi khác để quản lý

### Phương án A — All-edge, ít phụ thuộc mạng

- Pi/Jetson chạy capture, pose và model nhẹ.
- SQLite/PostgreSQL nhỏ + ổ SSD; đồng bộ khi có mạng.
- Ưu: riêng tư, latency ổn định, chạy offline.
- Nhược: cập nhật model/fleet khó, tài nguyên hạn chế.
- Dùng Mender cho OTA A/B rollback hoặc balena cho container fleet; cả hai có nền tảng quản lý tập trung thiết bị.[^S38][^S39]

### Phương án B — Edge server tại bảo tàng/phòng lab (**khuyến nghị**)

- Pi là device agent; mini-PC/Jetson là gateway.
- Pose/control service tách khỏi inspection worker.
- PostgreSQL lưu metadata; S3-compatible object storage/NAS lưu master image; Redis/NATS/RabbitMQ làm queue.
- Docker Compose cho 1 node; chỉ dùng K3s khi có nhiều node, HA hoặc đội vận hành đủ năng lực. K3s là Kubernetes nhẹ cho edge/ARM/air-gapped, nhưng vẫn mang chi phí vận hành Kubernetes.[^S37]
- Ưu: không đưa ảnh nhạy cảm ra Internet, dùng GPU chung cho nhiều camera, dễ sao lưu.

### Phương án C — Hybrid edge/cloud

- Pose và safety gate ở edge.
- Edge tạo thumbnail/ROI/feature và đẩy job; master upload theo chính sách.
- Cloud dùng managed PostgreSQL, object storage versioning, GPU worker, model registry và dashboard.
- Thiết bị buffer khi mất mạng; command có TTL/idempotency; tuyệt đối không đóng vòng motor qua WAN.
- AWS IoT Greengrass là một lựa chọn có local compute, messaging, data sync, ML inference và fleet deployment; IoT Core dùng certificate/policy và rules để chuyển dữ liệu sang dịch vụ khác.[^S34][^S35] Có thể thay bằng Azure/GCP hoặc open-source tùy ngân sách.

### Phương án D — Cloud-managed hoàn toàn

Chỉ phù hợp nếu mạng, chính sách dữ liệu và chi phí cho phép. API, DB, object storage, queue và GPU inference chạy cloud; site chỉ capture/upload. Dễ scale và cộng tác, nhưng độ trễ, egress, downtime và quyền sở hữu dữ liệu là rủi ro lớn.

### Kiến trúc dịch vụ nên tách

- **capture-agent**: camera, metadata, checksum, local spool.
- **pose-service**: deterministic, real-time-ish, không phụ thuộc cloud.
- **motion-safety-service**: limit switch, homing, emergency stop, command envelope.
- **inspection-worker**: SIFT/learned matching, anomaly, segmentation.
- **inference-server**: model version và accelerator. ONNX Runtime cho phép chọn CPU/CUDA/TensorRT/OpenVINO/QNN Execution Provider; quantization phải kiểm tra accuracy trên calibration set.[^S40][^S41]
- **API/control-plane**: người dùng, artifact, job, audit.
- **PostgreSQL**: metadata và transactional state.
- **object-store**: RAW/master, derived images, masks, model artefacts, checksum/version/lifecycle.
- **queue**: retry, priority, backpressure, dead-letter.
- **observability**: OpenTelemetry/Prometheus/Grafana, log có correlation ID.
- **fleet/OTA**: signed artefact, staged rollout, rollback, inventory.

Triton hữu ích khi có GPU dùng chung và nhiều request vì hỗ trợ dynamic batching/multiple model instances; với một thiết bị, request thưa và latency thấp, batching có thể không đem lợi ích.[^S36]

## 8. Thiết kế thực nghiệm để chứng minh “tốt hơn”

### 8.1 Pose

Tạo rig ground truth độc lập: bàn trượt/rotary stage đã kiểm định, robot có encoder đã hiệu chuẩn, motion capture hoặc laser tracker tùy ngân sách. Không dùng chính pose fiducial làm ground truth.

Ma trận thử tối thiểu:

- khoảng cách, yaw/pitch/roll và dịch chuyển XYZ;
- 5 mức ánh sáng, glare, blur, defocus;
- che khuất 0/10/25/40%;
- 3 kích thước marker và 3 vật liệu in;
- ít nhất 3 ngày, 2 camera hoặc 2 lần tháo-lắp;
- Diamond, ChArUco board, AprilGrid;
- ORB, XFeat+LightGlue, một dense matcher.

Metric:

- translation error từng trục và magnitude; rotation geodesic error;
- median, p95, worst-case, bootstrap 95% CI;
- failure/reject/flip rate;
- reprojection chỉ là diagnostic, không thay thế pose error;
- BOP VSD/MSSD/MSPD hoặc ADD-S khi đánh giá markerless/symmetry.[^S42]
- latency p50/p95/p99, peak RAM/VRAM, W và J/inference.

Đánh giá uncertainty theo GUM: liệt kê nguồn sai số camera, target, kích thước in, extrinsic, actuator, nhiệt, repeatability; propagation bằng Jacobian hoặc Monte Carlo và kiểm tra coverage thực nghiệm.[^S43]

### 8.2 Inspection

- Split theo **artifact × session × site/camera**, không random frame.
- Giữ external test bất biến; BRIDGE data trở thành tập adaptation riêng.
- Ground truth mask/bbox/severity do tối thiểu hai người gán; đo agreement và adjudication.
- Negative set phải có thay đổi ánh sáng/focus/view nhưng không hư hại.
- Positive set phải bao phủ kích thước vật lý và severity; ghi kích thước mm nếu đo được.
- Threshold chỉ chọn trên validation; test chạy một lần cuối.
- Báo cáo object mAP50-95, mask IoU/Dice, image-level AUROC/AUPRC, F1 tại threshold khóa, false-negative theo severity, false alarms/artifact-day và calibration ECE/Brier.
- Báo cáo theo từng domain, không chỉ trung bình.
- Ablation: RAW vs RGB ISP; ánh sáng thường vs cross-pol/raking; SSIM vs ΔE/gradient; baseline đơn vs normal bank; YOLO vs anomaly vs ensemble; full image vs tile.
- Chạy ít nhất 3–5 seed cho training; bootstrap CI theo artifact, không theo frame.

### 8.3 Điều kiện chấp nhận đề xuất

Không đặt con số “đẹp” tùy ý. Trước pilot, chủ nhiệm đề tài và chuyên gia bảo tồn xác định:

- mức lỗi nhỏ nhất cần phát hiện;
- false-negative tối đa theo severity;
- false alarms có thể xử lý mỗi ngày;
- sai số pose cho phép dựa trên vùng an toàn cơ khí;
- latency và offline duration;
- retention, quyền truy cập và vị trí lưu ảnh.

## 9. Lộ trình ưu tiên

### 0–4 tuần: sửa nền đo và loại false-normal

1. Đổi region SSIM exception/small crop từ 1.0 thành UNKNOWN + reason.
2. Không resize fallback để kết luận; alignment gate thất bại thì UNKNOWN.
3. Bỏ JPEG round-trip cho crop inference.
4. Khóa camera exposure/AWB/focus trong cả pose và inspection; lưu metadata.
5. Tách master lossless và preview JPEG.
6. Đặt BRIDGE_N=0 cho external test mới; lập manifest theo artifact/session.
7. Thêm benchmark replay + test synthetic geometry vào CI.
8. Pin Python dependencies, model SHA, dataset manifest và g2o commit.

### 1–3 tháng: cải thiện lớn, chi phí vừa

1. In plate ChArUco lớn và AprilGrid; A/B trên rig.
2. Calibrate stereo/slider độc lập; thêm API nhận extrinsic và version registry.
3. Hiệu chuẩn actuator/hand–eye; safety gate và hardware-in-loop trước auto-dispatch.
4. Rig ánh sáng diffuse + cross-polarized/raking.
5. Xây normal bank đa phiên cho mỗi artifact.
6. Baseline PatchCore/EfficientAD/AnomalyDINO; thêm tiled inference.
7. A/B ORB với XFeat+LightGlue.
8. Tách inspection worker + queue + object storage trên edge server.

### 3–6 tháng: pilot có bằng chứng

1. Dataset khóa theo site/camera/time, double annotation.
2. Segmentation cho class lỗi quan trọng; SAM chỉ hỗ trợ annotation.
3. Calibration probability và threshold theo chi phí nghiệp vụ.
4. Dashboard drift: exposure, focus, marker size, reject rate, score distribution.
5. ONNX/TensorRT/OpenVINO/Hailo profile; chỉ quantize model thắng.
6. OTA staged rollout + rollback; backup/restore drill.
7. Pilot silent mode: AI không tự tác động, chuyên gia đối chiếu.

### 6–12 tháng: mở rộng nghiên cứu

- RGB-D/photometric stereo/RTI cho relief.
- FoundationPose/SAM-6D làm markerless fallback.
- Dense matching/VGGT/FoundationStereo cho offline 3D bootstrap.
- Active learning và hard-negative mining.
- Multi-site external validation; model card, datasheet và uncertainty budget.

## 10. Danh mục thêm nhiều giải pháp

Các ý sau không phải cùng làm một lúc:

1. Marker nhiều kích thước trên cùng rig để giữ pixel footprint theo khoảng cách.
2. Board 3D không đồng phẳng để loại ambiguity tốt hơn planar target.
3. Hai target ở hai mặt cradle để tăng vùng nhìn.
4. AprilTag ngoài ROI và ChArUco gần ROI để kết hợp robust ID + subpixel corner.
5. Temporal EKF/UKF hoặc factor graph, nhưng chỉ sau khi có covariance đo được.
6. Rolling-shutter model nếu không đổi camera và vẫn chụp khi chuyển động.
7. Trigger hardware giữa camera và actuator.
8. Focus metric và blur gate trước khi upload.
9. Glare/saturation mask trước SSIM/anomaly.
10. Per-pixel noise map từ flat/dark frame.
11. Exposure bracketing/HDR cho vật có dải sáng lớn.
12. Focus stacking cho relief; kiểm soát artefact ở biên.
13. Lab ΔE00 map với color calibration.
14. MS-SSIM/CW-SSIM/DISTS/LPIPS làm candidate, không dùng vô điều kiện.
15. Gabor/LBP/wavelet cho baseline texture giải thích được.
16. Optical flow/scene flow để phát hiện biến dạng theo thời gian.
17. Image pyramid và tile overlap theo kích thước lỗi vật lý.
18. Class-specific threshold và cost-sensitive learning.
19. Focal loss/oversampling nhưng giữ split group.
20. Self-supervised pretraining trên ảnh hiện vật chưa gán nhãn.
21. Synthetic defects chỉ bổ sung, không thay test thật.
22. Domain randomization ánh sáng có giới hạn vật lý.
23. Test-time adaptation chỉ trong nhánh nghiên cứu, tránh tự làm trôi chuẩn.
24. Conformal prediction hoặc abstention để tạo trạng thái UNKNOWN.
25. Model ensemble teacher–student và distillation xuống edge.
26. Quantization-aware training khi INT8 post-training làm giảm recall.
27. Structured pruning và input-size sweep.
28. Cache undistort maps, model session và camera process.
29. Zero-copy ndarray/tensor; pinned memory nếu GPU.
30. ROI từ pose để giảm pixel vô ích trước inference.
31. Micro-batch chỉ khi có tải đồng thời; không hy sinh latency vô ích.
32. Worker priority: pose > inspection > archival.
33. Thermal/power governor và watchdog trên edge.
34. Local spool có checksum, quota và chính sách xóa an toàn.
35. Content-addressed object key để chống trùng và kiểm tra toàn vẹn.
36. Immutable raw; derived asset liên kết parent hash + pipeline version.
37. DVC/lakeFS cho dataset; MLflow/W&B cho experiment; registry ký model.
38. Canary model chạy song song, không thay production ngay.
39. Shadow inference để đo drift không tác động người dùng.
40. Human-in-the-loop cho severity cao/uncertain.
41. Audit trail ai duyệt mask/kết luận.
42. RBAC theo site/artifact; mTLS/certificate riêng cho thiết bị.
43. Secret rotation và device revocation.
44. Signed OTA, A/B rollback, staged cohort.
45. Backup object store + PostgreSQL và thử restore định kỳ.
46. Data retention theo loại master/derived/telemetry.
47. OpenTelemetry trace từ capture → pose → inference → quyết định.
48. Golden dashboard cho calibration expiry và drift.
49. Digital twin của rig/camera để quản lý transform version.
50. Gắn tất cả kết quả với camera serial, calibration SHA, model SHA, code commit và config SHA.

## 11. Những điều không nên làm

- Không tuyên bố “AI tốt hơn ChArUco” nếu ground truth vẫn sinh từ ChArUco.
- Không coi tolerance cấu hình là độ chính xác đo được.
- Không dùng ảnh test để tune rồi vẫn gọi đó là independent test.
- Không coi pixel AUROC cao là đủ an toàn cho lỗi nhỏ.
- Không biến lỗi/exception thành similarity 1.0 hoặc trạng thái good.
- Không điều khiển motor qua cloud/WAN và không auto-dispatch trước hand–eye, homing, limit switch, emergency stop và hardware-in-loop.
- Không mua GPU lớn trước khi profile camera, I/O và model.
- Không triển khai Kubernetes chỉ để chạy một node.
- Không lưu chỉ ảnh đã warp, JPEG hoặc enhancement; phải giữ master bất biến.
- Không tuyên bố chuẩn ISO/FADGI/EMVA hoặc độ chính xác vật lý khi chưa thực hiện đầy đủ phép thử.

## 12. Đề xuất cấu hình cuối cùng

**Bản chi phí hợp lý**

- Giữ Pi làm capture/control.
- Plate ChArUco lớn và một AprilGrid để benchmark.
- Rig đèn diffuse + cross-polarization, camera khóa thông số, master PNG/TIFF/RAW.
- Mini-PC x86 hoặc Jetson Orin Nano Super làm edge server.
- XFeat+LightGlue chỉ thay ORB nếu benchmark thắng.
- EfficientAD/PatchCore + YOLO-seg/tiled inference.
- PostgreSQL + object storage + queue + Docker Compose.
- Mender/balena cho OTA khi có nhiều thiết bị.

**Bản hiệu năng/chất lượng cao**

- Camera industrial high-resolution cho inspection + global-shutter camera cho pose.
- Hardware trigger, lens C-mount chất lượng, lighting dome/raking/RTI.
- AprilGrid/3D target cho calibration; hand–eye metrology độc lập.
- Depth/structured-light hoặc photometric stereo cho relief.
- Orin NX/AGX hoặc RTX edge workstation; Triton/TensorRT.
- RoMa/FoundationPose/SAM-6D chỉ là nhánh fallback/QA, có gate uncertainty.
- Object store versioned, model registry, fleet OTA, multi-site validation.

Nếu chỉ chọn **ba việc đem lại giá trị cao nhất**, hãy chọn: (1) hiệu chuẩn extrinsic/hand–eye độc lập; (2) kiểm soát ánh sáng–RAW–màu và loại mọi đường false-normal; (3) xây external test theo artifact/session cùng benchmark ChArUco board–AprilGrid–learned features. Ba việc này có khả năng cải thiện độ tin cậy khoa học nhiều hơn việc đổi sang một model “SOTA” đơn lẻ.

## Nguồn web

Các con số hiệu năng trong paper hoặc trang hãng chỉ mô tả điều kiện của nguồn; báo cáo không coi chúng là kết quả của hệ thống này.

[^S1]: OpenCV, [Detection of ChArUco Boards](https://docs.opencv.org/4.10.0/df/d4a/tutorial_charuco_detection.html).
[^S3]: AprilRobotics, [AprilTag 3 README](https://github.com/AprilRobotics/apriltag/blob/master/README.md).
[^S4]: ETH Zürich ASL, [Kalibr calibration targets](https://github.com/ethz-asl/kalibr/wiki/calibration-targets).
[^S5]: Romero-Ramirez et al., [ArUco Nano, SoftwareX 2026](https://doi.org/10.1016/j.softx.2026.102690).
[^S6]: Potje et al., [XFeat, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Potje_XFeat_Accelerated_Features_for_Lightweight_Image_Matching_CVPR_2024_paper.html).
[^S7]: Lindenberger et al., [LightGlue, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.html).
[^S8]: Wang et al., [Efficient LoFTR, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Efficient_LoFTR_Semi-Dense_Local_Feature_Matching_with_Sparse-Like_Speed_CVPR_2024_paper.html).
[^S9]: Edstedt et al., [RoMa, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Edstedt_RoMa_Robust_Dense_Feature_Matching_CVPR_2024_paper.html).
[^S10]: Wen et al., [FoundationPose, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Wen_FoundationPose_Unified_6D_Pose_Estimation_and_Tracking_of_Novel_Objects_CVPR_2024_paper.html).
[^S11]: Lin et al., [SAM-6D, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Lin_SAM-6D_Segment_Anything_Model_Meets_Zero-Shot_6D_Object_Pose_Estimation_CVPR_2024_paper.html).
[^S12]: Wang et al., [VGGT, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.html).
[^S13]: Wen et al., [FoundationStereo, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Wen_FoundationStereo_Zero-Shot_Stereo_Matching_CVPR_2025_paper.html).
[^S14]: MVTec, [MVTec AD 2](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2).
[^S15]: Roth et al., [PatchCore, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html).
[^S16]: Batzner et al., [EfficientAD, WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.html).
[^S17]: Jeong et al., [WinCLIP, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.html).
[^S18]: Damm et al., [AnomalyDINO, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.html).
[^S19]: Cheng et al., [Mask2Former, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.html).
[^S20]: Historic England, [Multi-light imaging for heritage applications](https://historicengland.org.uk/images-books/publications/multi-light-imaging-heritage-applications/).
[^S21]: ISO, [ISO 19264-1:2021](https://www.iso.org/standard/79172.html).
[^S22]: FADGI, [Technical Guidelines for Digitizing Cultural Heritage Materials](https://www.digitizationguidelines.gov/guidelines/digitize-technical.html).
[^S23]: EMVA, [EMVA Standard 1288 downloads](https://www.emva.org/standards-technology/emva-1288/emva-standard-1288-downloads-2/).
[^S24]: Raspberry Pi, [Global Shutter Camera product brief](https://pip-assets.raspberrypi.com/categories/810-raspberry-pi-global-shutter-camera/documents/RP-008196-DS/gs-camera-product-brief).
[^S25]: Raspberry Pi, [AI HAT+ 2](https://www.raspberrypi.com/products/ai-hat-plus-2/).
[^S26]: NVIDIA, [Jetson Orin Nano Super Developer Kit](https://developer.nvidia.com/blog/?p=93942).
[^S27]: NVIDIA, [Jetson modules](https://developer.nvidia.com/embedded/jetson-modules).
[^S28]: Luxonis, [OAK-D Pro W](https://shop.luxonis.com/products/oak-d-pro-w).
[^S29]: RealSense, [D555 PoE](https://www.realsenseai.com/products/d555-poe/).
[^S30]: Meta AI, [SAM 3](https://ai.meta.com/research/sam3/).
[^S31]: Akyon et al., [SAHI for small object detection](https://arxiv.org/abs/2202.06934).
[^S32]: Basler, [ace 2 cameras](https://www.baslerweb.com/en/cameras/ace2/).
[^S33]: NVIDIA, [Jetson Thor](https://developer.nvidia.com/blog/?p=104879).
[^S34]: AWS, [AWS IoT Greengrass](https://docs.aws.amazon.com/greengrass/v2/APIReference/Welcome.html).
[^S35]: AWS, [How AWS IoT works](https://docs.aws.amazon.com/iot/latest/developerguide/aws-iot-how-it-works.html).
[^S36]: NVIDIA, [Triton dynamic batching and concurrent execution](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Conceptual_Guide/Part_2-improving_resource_utilization/README.html).
[^S37]: K3s, [Lightweight Kubernetes](https://docs.k3s.io/).
[^S38]: Mender, [Introduction and OTA architecture](https://docs.mender.io/overview/introduction).
[^S39]: balena, [Fleet deployment documentation](https://docs.balena.io/learn/deploy/deployment).
[^S40]: ONNX Runtime, [Execution Providers](https://onnxruntime.ai/docs/execution-providers/).
[^S41]: ONNX Runtime, [Model quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).
[^S42]: BOP, [6D object pose estimation tasks and metrics](https://bop.felk.cvut.cz/tasks/).
[^S43]: BIPM/JCGM, [Guide to the Expression of Uncertainty in Measurement](https://www.bipm.org/en/doi/10.59161/jcgm100-2008e).
