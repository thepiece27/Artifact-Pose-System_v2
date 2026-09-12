# Khả thi và thiết kế căn chỉnh thị giác thời gian thực

> Ngày đánh giá: 2026-09-12  
> Snapshot mã nguồn: commit 83aeaf1  
> Thiết bị giả định: Raspberry Pi, Raspberry Pi Camera Module 3, pan–tilt dùng servo, slider dùng động cơ bước.

**Trạng thái bằng chứng**

- PASS — đã đối chiếu camera, device runtime, motion controller, pose service và alignment orchestration trong mã nguồn hiện tại.
- PASS — đã kiểm tra tài liệu Raspberry Pi/Picamera2, OpenCV và nghiên cứu visual servoing/marker gốc.
- NOT RUN — chưa benchmark trên đúng Raspberry Pi, Camera Module 3 và cơ cấu thật; mọi kết luận về tải là đánh giá khả thi có điều kiện, không phải FPS đã đo.
- NOT RUN — chưa đo backlash, độ lặp lại servo, mất bước, rung hoặc sai số pose bằng ground truth độc lập; chưa thể đặt tolerance vật lý đáng tin cậy.

## 1. Kết luận

Ý tưởng **căn chỉnh bằng video nhưng chỉ chụp ảnh độ phân giải cao sau khi cơ cấu đã dừng** là khả thi và hợp lý hơn vòng lặp chụp–upload–tính toán hiện tại.

Tuy nhiên, kiến trúc nên được gọi là **soft real-time visual servoing kết hợp stop-and-settle**, không phải hard real-time và cũng không phải xử lý liên tục ảnh 12 MP:

- Video độ phân giải thấp/trung bình được xử lý **ngay trên Raspberry Pi** để tạo phản hồi điều khiển.
- Pan–tilt servo chỉ căn thô, giữ marker trong trường nhìn và giảm sai số góc lớn.
- Slider stepper thực hiện dịch chuyển thô rồi tinh, với profile tăng/giảm tốc và kiểm tra hết rung.
- Khi sai số ổn định trong nhiều frame liên tiếp, dừng toàn bộ cơ cấu, chờ ổn định theo tín hiệu ảnh/IMU rồi chụp ảnh full-resolution.
- Chỉ ảnh cuối và telemetry cần upload. Server/cloud không nằm trong vòng điều khiển nhanh.

Phương án này giảm số lần mở camera, thời gian warm-up, encode PNG, upload và round-trip MQTT. Nó cũng cho phép nhìn thấy overshoot, backlash, rung và sai số thực sau mỗi chuyển động. Nhưng video feedback **không làm servo rẻ tiền trở thành cơ cấu chính xác**; nó chỉ giúp bù sai số đầu ra trong giới hạn deadband, backlash, độ rơ và độ phân giải cơ khí.

Đề tài cần thay đổi mức **trung bình đến lớn ở device agent và điều khiển**, nhưng phần pose toán học hiện tại có thể tái sử dụng. Server chỉ cần thay đổi nhỏ nếu vòng nhanh được đưa xuống Pi.

## 2. “Realtime” nào thực sự cần thiết?

Ba khái niệm cần tách rõ:

| Khái niệm | Mục tiêu | Có phù hợp không? |
|---|---|---|
| Hard real-time | Mỗi chu kỳ phải hoàn thành trước deadline tuyệt đối | Không phù hợp với Python + Raspberry Pi OS + Picamera2 + MQTT hiện tại. |
| Soft real-time | Ưu tiên frame mới, latency thấp và ổn định; đôi lúc bỏ frame được | **Phù hợp cho căn chỉnh thị giác.** |
| High-resolution continuous processing | Tính pose/AI trên mọi frame 4K/12 MP | Không cần thiết và gây lãng phí RAM, CPU, băng thông. |

Visual servoing là dùng đặc trưng ảnh trong vòng phản hồi điều khiển. Nền tảng học thuật phân biệt Image-Based Visual Servoing (IBVS), Position-Based Visual Servoing (PBVS) và phương án hybrid; latency, trường nhìn, singularity và sai số mô hình đều ảnh hưởng ổn định.[^1]

Với kết cấu hiện tại, lựa chọn thực dụng là hybrid:

- **IBVS ở vòng nhanh**: so sánh tâm, kích thước, góc và bốn corner marker hiện tại với corner tham chiếu. Điều khiển theo sai số pixel/normalized image giúp giữ marker trong ảnh và ít phụ thuộc hơn vào pose 3D tuyệt đối.
- **PBVS ở vòng chậm**: định kỳ chạy PnP/ChArUco pose để kiểm tra translation/rotation và điều kiện kết thúc.
- **Still verification**: ảnh full-resolution sau settle dùng để xác nhận pose cuối và kiểm tra hư hại.

Không cần 30 hoặc 60 lệnh cơ khí mỗi giây. Camera có thể tạo frame nhanh hơn, nhưng controller chỉ lấy frame mới nhất và phát lệnh ở tốc độ mà servo/slider có thể đáp ứng ổn định.

## 3. Camera Module 3 có đủ không?

### 3.1 Khả năng cảm biến

Camera Module 3 dùng IMX708, ảnh tĩnh 4608×2592 (11.9 MP), có autofocus. Các video mode chính thức gồm 2304×1296 ở tối đa 56 fps, HDR 2304×1296 ở 30 fps và 1536×864 ở 120 fps.[^2] Full sensor 4608×2592 có mode khoảng 14.35 fps theo output liệt kê của Raspberry Pi.[^3]

Độ phân giải đó đủ tốt để:

- phát hiện fiducial ở preview nếu marker chiếm đủ pixel;
- chụp ảnh cuối có chi tiết cao khi hệ thống đứng yên;
- tạo đồng thời main stream và lores stream nhờ ISP.

Picamera2 cho phép ISP sinh tối đa hai ảnh đã xử lý cho mỗi frame: main và lores; raw cũng có thể được giao cho ứng dụng. Trên Pi 4 trở xuống, lores phải là YUV; Pi 5 cho phép RGB hoặc YUV.[^4] Vì pose fiducial chỉ cần luminance, có thể dùng trực tiếp mặt phẳng Y của YUV để tránh chuyển BGR và giảm copy.

### 3.2 Giới hạn quan trọng

Camera Module 3 là **rolling shutter**. Raspberry Pi xác nhận phần lớn camera module, trừ Global Shutter Camera, quét ảnh theo từng dòng và có thể méo đối tượng/camera đang chuyển động.[^2] Vì camera đặt trên pan–tilt/slider, frame trong lúc rung hoặc quay không nên dùng làm pose đo cuối.

Autofocus dùng voice-coil và có thể chạy liên tục trong video.[^5] Điều này tốt cho preview phổ thông nhưng không tốt cho metrology: focus thay đổi có thể làm thay đổi nhẹ intrinsics, distortion và độ sắc nét. Quy trình nên autofocus một lần, ghi LensPosition, chuyển sang manual và dùng calibration đúng lens position.

Camera độ phân giải cao không đồng nghĩa pose realtime tốt hơn. Chi phí detection tăng theo pixel; marker quá nhỏ mới cần tăng resolution. Với marker đủ lớn, ảnh 640×360, 640×480 hoặc 960×540 thường là điểm bắt đầu hợp lý; khi mất marker mới tăng tạm lên 1280×720 hoặc main stream.

### 3.3 Ngân sách dữ liệu

Các con số sau là phép tính dữ liệu thô, không phải benchmark CPU:

- 3840×2160 RGB888: khoảng 24.9 MB/frame.
- 10 frame/s ở kích thước trên: khoảng 249 MB/s chỉ cho một lần biểu diễn RGB.
- 4608×2592 RGB888: khoảng 35.8 MB/frame.
- 640×360 grayscale: khoảng 0.23 MB/frame; 20 frame/s khoảng 4.6 MB/s.

ISP và CSI có đường xử lý tối ưu riêng, nhưng Python/OpenCV vẫn phải map/copy hoặc đọc buffer. Vì vậy xử lý lores tại Pi hợp lý hơn nhiều so với encode rồi stream full-resolution.

### 3.4 Kết luận tài nguyên

- **Pi 5 có tản nhiệt chủ động**: khả thi cao cho ChArUco/AprilTag lores, optical-flow ngắn và PnP định kỳ.
- **Pi 4**: vẫn khả thi nếu dùng grayscale lores, ROI, bỏ frame cũ và không chạy YOLO/ORB full-frame trong vòng nhanh.
- **Pi 3 hoặc thấp hơn**: cần benchmark nghiêm ngặt; có thể giảm control rate hoặc chuyển detector native.
- Không đủ dữ liệu để cam kết fps cụ thể vì chưa biết model Pi, OS, cooling, kích thước marker, ánh sáng và backend OpenCV. Một nghiên cứu 2026 trên Pi 4 cho thấy OpenCV ArUco nhanh và nhẹ hơn các detector YOLO nhỏ trong bài toán marker của họ, nhưng số đo của miền nông nghiệp không được chuyển thẳng sang rig này.[^6]

## 4. Vòng hiện tại và nút thắt

Mã hiện tại là vòng ảnh tĩnh qua mạng:

1. Server gửi lệnh capture/MQTT.
2. Device đưa servo về home hoặc thực hiện move.
3. CameraManager tạo một Picamera2 mới, configure, start và chờ 1.5 giây.
4. Lens được đặt manual rồi chờ thêm 0.5 giây.
5. Chụp RGB888 3840×2160 thành PNG.
6. Upload ảnh lên server.
7. Server đọc file, chạy một lần pose.
8. Server gửi lệnh move; device chạy cơ cấu rồi lại capture.

Các bằng chứng mã nguồn:

- [camera_manager.py](../embed/device_agent/adapters/camera_manager.py#L29) mở/đóng camera cho từng ảnh và có hai khoảng chờ.
- [main_app.py](../embed/device_agent/runtime/main_app.py#L986) upload mỗi alignment image để server tính.
- [pose_service.py](../server/app/services/pose_service.py#L121) nhận đường dẫn một ảnh rồi chạy correction một lần.
- [correction.py](../server/app/modules/artifact_pose/correction.py#L1) tự mô tả là one-image correction và chưa cho phép motor actuation.

Do đó vòng hiện tại có thể mất vài giây mỗi iteration ngay cả khi pose chỉ cần vài chục hoặc vài trăm millisecond. Realtime video chủ yếu loại bỏ lifecycle camera, encode, disk và network khỏi vòng điều khiển.

Một mâu thuẫn khác cần xử lý trước: pose core hiện trả motor_command disabled vì chưa có hand–eye, pivot, axis/sign và driver đã hiệu chuẩn ([correction.py](../server/app/modules/artifact_pose/correction.py#L31)). Trong khi server orchestration đã có logic luân phiên translation/rotation và auto-dispatch nếu command enabled ([inspection_service.py](../server/app/services/inspection_service.py#L971)). Realtime không nên nối tắt fail-closed này.

## 5. Kiến trúc đề xuất

### 5.1 Camera luôn mở

Tạo một CameraService sống suốt phiên:

- Sensor mode preview: ưu tiên 2304×1296 full FoV; ISP tạo lores 640×360 hoặc 960×540.
- Lores YUV420 dùng cho detector; chỉ đọc Y plane.
- Main stream dùng preview/UI hoặc snapshot kiểm chứng trung gian.
- Camera start một lần; khóa exposure/gain/AWB/focus sau warm-up.
- Buffer nhỏ và queue policy ưu tiên frame mới nhất.
- Không ghi ảnh preview xuống disk.
- Frame mang SensorTimestamp, sequence number và camera metadata.

Picamera2 video configuration mặc định dùng nhiều buffer để giảm frame drop; tài liệu cũng lưu ý queue thường giữ frame hoàn tất gần nhất. Khi cần bảo đảm ảnh được chụp sau lệnh, đặt queue=False hoặc dùng request mới sau settle.[^4]

### 5.2 Vòng điều khiển local

State machine khuyến nghị:

1. **ACQUIRE**  
   Phát hiện ChArUco/AprilTag trên lores. Nếu mất marker, không di chuyển mù; thử resolution/exposure khác hoặc yêu cầu can thiệp.

2. **TRACK**  
   Detect đầy đủ mỗi N frame; giữa các lần detect có thể dùng pyramidal Lucas–Kanade optical flow để theo dõi corner, nhưng phải detect lại khi residual/forward-backward error vượt gate.[^7]

3. **COARSE_PAN_TILT**  
   Chỉ đưa tâm marker vào vùng giữa và sửa góc lớn. Lệnh nhỏ, có giới hạn tốc độ/gia tốc; không tìm độ chính xác cuối bằng hobby servo.

4. **SERVO_SETTLE**  
   Không chỉ sleep cố định. Yêu cầu corner velocity, pose variation và blur metric dưới ngưỡng trong N frame liên tiếp; có timeout.

5. **SLIDER_COARSE**  
   Di chuyển phần lớn khoảng cách bằng stepper với acceleration/deceleration.

6. **SLIDER_FINE**  
   Lệnh nhỏ hơn, gain giảm dần theo sai số. Luôn tiếp cận vị trí cuối từ cùng một hướng nếu backlash đáng kể.

7. **GLOBAL_VERIFY**  
   Chạy PnP/pose đầy đủ; yêu cầu marker không nhập nhằng, reprojection đạt gate và translation/rotation ổn định nhiều frame.

8. **FINAL_SETTLE**  
   Không có pulse stepper/servo command; kiểm tra rung bằng corner motion và có thể bằng IMU/accelerometer gắn trên camera bracket.

9. **CAPTURE_FINAL**  
   Chuyển sang still mode full-resolution hoặc chụp main high-resolution, queue=False, khóa controls. Chụp burst 2–3 ảnh; loại frame blur/rung.

10. **UPLOAD_AND_AUDIT**  
    Upload ảnh cuối, pose, covariance/quality proxy, actuator state, timestamps, calibration SHA và thống kê vòng lặp.

### 5.3 Server ngoài vòng nhanh

MQTT/server chỉ:

- start/stop/abort alignment;
- gửi artifact ID và cấu hình phiên;
- nhận telemetry giảm tần số;
- lưu kết quả và ảnh cuối;
- chạy inspection AI nặng;
- cung cấp dashboard.

Không stream video qua FastAPI/MQTT để tính pose điều khiển. Nếu cần xem từ xa, tạo H.264/MJPEG preview riêng, rate-limit và không dùng frame decoded từ UI làm feedback. Picamera2 có encoder H.264/MJPEG; tài liệu Raspberry Pi lưu ý Pi 5 dùng software video encoder và có tùy chọn low-latency nhưng vẫn phát sinh độ trễ mã hóa.[^8]

## 6. Chiến lược riêng cho pan–tilt servo

### 6.1 Vấn đề hiện hữu

Mã hiện tại:

- lưu current_yaw/current_pitch là **góc đã ra lệnh**, không phải góc đo;
- làm tròn target thành số nguyên độ trước khi gửi ServoKit ([hardware_controller.py](../embed/device_agent/adapters/hardware_controller.py#L98));
- không có encoder, calibration pulse-width, deadband, backlash map, load/sag model hoặc tốc độ thực;
- reset về home bằng command rồi chờ cố định 0.5 giây.

Do đó độ phân giải phần mềm hiện đã bị giới hạn 1°, trước cả sai số servo cơ khí. Camera feedback có thể quan sát residual nhưng dễ tạo hunting: +1°, −1°, +1° quanh điểm đích.

### 6.2 Cách sử dụng servo hợp lý

- Cho pan–tilt làm **coarse actuator**, không làm metrology actuator.
- Bỏ integer rounding ở tầng software, nhưng không coi float command là độ chính xác vật lý.
- Hiệu chuẩn map pulse-width → góc thực bằng fiducial qua nhiều lần tiến/lùi.
- Đo repeatability, backlash, deadband, overshoot và sag ở các góc/tải khác nhau.
- Dùng hysteresis: vào tolerance nhỏ hơn ngưỡng thoát tolerance.
- Dùng proportional gain nhỏ và giới hạn delta; giảm gain khi gần đích.
- Không đảo hướng liên tục; nếu cần, overshoot có kiểm soát rồi tiếp cận từ một hướng.
- Yêu cầu K frame ổn định, không chỉ một frame.
- Tách pan và tilt nếu coupling mạnh.
- Nếu tolerance góc nhỏ hơn repeatability đo được, thay actuator: smart servo có encoder/telemetry, stepper + gearbox/worm drive, hoặc pan–tilt có absolute encoder.
- Phương án đơn giản nhất cho ảnh cuối: pan–tilt căn thô rồi khóa cơ khí; slider làm tinh.

Visual feedback cải thiện **độ chính xác kết quả cuối**, nhưng latency quá lớn, gain quá cao hoặc measurement nhiễu có thể làm vòng kín dao động. Đây là vấn đề control-system, không thể giải chỉ bằng tăng fps.[^1]

## 7. Chiến lược riêng cho slider stepper và rung

Microstep 1/32 trong cấu hình ([hardware_controller.py](../embed/device_agent/adapters/hardware_controller.py#L23)) không chứng minh độ chính xác tuyệt đối 1/32 full step. Độ chính xác còn phụ thuộc torque, driver, belt/lead-screw, backlash, rail, tải, mất bước và cộng hưởng.

Mã phát pulse bằng vòng Python + time.sleep ([hardware_controller.py](../embed/device_agent/adapters/hardware_controller.py#L219)). Cách này đủ cho prototype tốc độ thấp, nhưng scheduling jitter của Linux/Python không phù hợp motion profile chính xác cao.

Nâng cấp theo thứ tự:

1. Đo mm/step theo nhiều vị trí và hai hướng.
2. Thêm homing/limit switch thật; current_x_steps trong RAM không phải absolute position.
3. Thêm acceleration/deceleration S-curve hoặc trapezoid; profile hiện chỉ chia fast/slow, chưa ramp liên tục.
4. Chuyển pulse generation sang microcontroller hoặc motion controller chuyên dụng; Pi gửi target, controller chịu timing.
5. Dùng driver có current control và cấu hình decay/microstep phù hợp.
6. Cố định ribbon cable và dây điện để chúng không kéo camera.
7. Tăng độ cứng bracket, giảm moment arm của camera/pan–tilt.
8. Tách nguồn servo/motor khỏi Pi/camera nhưng nối ground đúng thiết kế.
9. Sau mỗi move, phát hiện settle bằng ảnh: median corner displacement/pose delta trong cửa sổ thời gian.
10. Nếu rung khó quan sát từ marker, gắn accelerometer/IMU trên camera bracket; điều kiện chụp là RMS vibration dưới ngưỡng đã đo.
11. Chỉ phát lệnh tiếp theo sau khi frame timestamp mới hơn thời điểm motor stop.
12. Nếu mất bước vẫn xảy ra, thêm encoder tuyến tính hoặc magnetic scale.

## 8. Xử lý frame để đủ tải

### 8.1 Pipeline vòng nhanh

- Lấy frame mới nhất; bỏ frame cũ nếu worker bận.
- Grayscale/Y plane, không RGB.
- Crop ROI dự đoán quanh marker lần trước.
- Downscale detector; refine corner trên ROI độ phân giải cao hơn khi cần.
- Detect ChArUco/AprilTag mỗi N frame.
- Optical flow corner giữa các lần detect.
- PnP đầy đủ ở tần số thấp hơn tracker.
- Không ORB 5000 feature, g2o, SSIM hoặc YOLO trong mọi frame.
- Không PNG/JPEG encode trong loop.
- Dùng một worker process/thread có bounded queue kích thước 1.
- Ghi metric latency từng stage và frame age, không chỉ fps.

### 8.2 Adaptive quality

| Trạng thái | Resolution/thuật toán |
|---|---|
| Marker chưa thấy | 1280×720 hoặc 2304×1296, detect toàn ảnh nhưng rate thấp |
| Marker đã thấy | 640×360/960×540 ROI, detect/tracking nhanh |
| Gần hội tụ | Corner refinement + PnP nhiều frame |
| Verify cuối | Main/high-resolution pose |
| Inspection | Full-resolution still trên server/edge worker |

### 8.3 Không stream nếu không cần

Nếu stream full RGB888 4K qua mạng, băng thông thô vượt xa nhu cầu điều khiển. H.264 giảm băng thông nhưng thêm encode, packet buffering, jitter và decode; frame nhận được có thể đã cũ. Đối với control, frame age quan trọng hơn frame rate.

Stream chỉ dành cho operator:

- 720p/1080p, bitrate giới hạn;
- kênh riêng;
- overlay pose lấy từ local telemetry;
- mất stream không được làm dừng vòng local nếu safety vẫn tốt;
- command start/stop vẫn qua kênh authenticated.

## 9. Calibration phải thay đổi

Realtime preview và final still có thể dùng sensor mode, crop và resolution khác nhau. Intrinsics không được lấy một ma trận 4K rồi chia tỷ lệ một cách mù quáng nếu mode crop/binning khác.

Cần registry:

- camera serial/module ID;
- Pi/model và libcamera/Picamera2 version;
- sensor mode, output size, crop/scaler crop;
- lens position;
- K, D và reprojection report;
- board definition và kích thước vật lý;
- hand–eye transform;
- actuator map và phiên bản;
- ngày calibration, nhiệt độ/rig state và SHA.

Với Camera Module 3:

- chọn một lens position cố định;
- calibration riêng cho preview pose mode và final still mode;
- nếu chỉ dùng IBVS normalized corners ở preview, vẫn cần kiểm tra mapping desired/current cùng mode;
- final pose phải dùng calibration của final mode;
- không chạy continuous AF trong lúc so pose với golden.

ChArUco vẫn phù hợp làm target vì corner có độ chính xác cao và board có thể dùng khi chỉ thấy một phần; OpenCV cũng cảnh báo thay đổi quy ước board sau phiên bản 4.6, nên phải pin version/legacyPattern.[^9]

## 10. Rủi ro và biện pháp

| Rủi ro | Hậu quả | Gate/biện pháp |
|---|---|---|
| Latency biến thiên, frame cũ | Overshoot/dao động | Queue 1, timestamp, drop stale frame, controller local |
| Servo backlash/deadband | Hunting hoặc không vào tolerance | Coarse-only, hysteresis, approach một hướng, feedback actuator tốt hơn |
| Slider rung/cộng hưởng | Rolling-shutter distortion, blur | S-curve, bracket cứng, visual/IMU settle |
| Stepper mất bước | Vị trí RAM khác vị trí thật | Home/limit, encoder/scale, pose verification |
| Rolling shutter | Pose sai trong lúc chuyển động | Không dùng moving frame làm final; exposure ngắn; global-shutter nếu cần |
| Autofocus thay đổi | Calibration/độ nét trôi | AF một lần → manual, calibration theo LensPosition |
| Preview/final khác crop | Pose nhảy khi switch mode | Calibration theo mode; giữ full FoV; final verify |
| Marker mất/che/phản sáng | Controller di chuyển mù | Fail-stop; board lớn/dư thừa; ánh sáng diffuse |
| Pose phẳng nhập nhằng | Lệnh sai hướng | Giữ kiểm tra hai nghiệm IPPE; prior chỉ hỗ trợ, không che ambiguity.[^10] |
| Gain quá cao | Dao động cơ khí | Gain scheduling, cap delta, minimum command interval |
| Gain quá thấp/deadband | Không hội tụ | Dither/approach strategy hoặc thay actuator |
| Nguồn servo làm nhiễu Pi | Reset/camera lỗi | Nguồn riêng đủ dòng, grounding, decoupling |
| Thermal throttling | Latency tăng theo thời gian | Tản nhiệt chủ động, log clock/temp, soak test |
| Frame drop/buffer stall | Controller dùng dữ liệu gián đoạn | Release request đúng hạn, buffer đủ, watchdog |
| Race với MQTT command | Hai controller cùng điều khiển | Single owner lease, session ID, command arbitration |
| Network mất | UI không thấy nhưng rig còn chạy | Local timeout/estop; store-and-forward telemetry |
| Sai hand–eye/axis map | Correction có vẻ hợp lý nhưng motor sai | Calibration độc lập, HIL, giới hạn nhỏ, staged enable |
| Một frame “đẹp” giả | Kết thúc sớm do outlier | K frame liên tiếp + robust median + quality gate |
| Rung nhỏ dưới marker resolution | Ảnh inspection vẫn mờ | Blur/MTF gate trên full-res burst hoặc accelerometer |

## 11. Cần sửa gì trong repository?

### 11.1 Device agent — thay đổi lớn nhất

1. Refactor [camera_manager.py](../embed/device_agent/adapters/camera_manager.py) thành persistent CameraService.
2. Thêm preview/main/lores configurations và frame timestamp.
3. Thêm local PoseLite: detect fiducial, corner quality, PnP định kỳ.
4. Thêm LatestFrameBuffer thay vì file queue.
5. Thêm AlignmentController state machine.
6. Thêm settle detector và blur gate.
7. Thêm calibration loader theo sensor mode/lens position.
8. Thêm local telemetry/event log.
9. MQTT action mới: alignment_start, alignment_abort, alignment_status; không gửi từng frame.
10. Tách motion pulse sang microcontroller hoặc driver daemon nếu có thể.

### 11.2 Pose core — tái sử dụng nhưng cần adapter

- Tái sử dụng detect_diamond, IPPE ambiguity, validation và camera_deviation.
- Thêm API nhận ndarray lores, scaled/per-mode K,D và không đọc file.
- Tách fast fiducial-only path khỏi hybrid ORB/g2o.
- Thêm desired image features cho IBVS.
- Thêm temporal quality state; không tự chọn nghiệm chỉ từ previous pose khi hai nghiệm thực sự gần nhau.
- Giữ motor command fail-closed đến khi hand–eye/actuator calibration hoàn tất.

### 11.3 Server — thay đổi nhỏ đến vừa

- Server khởi tạo/cancel session, không orchestration từng ảnh.
- Lưu alignment session summary và telemetry sampled.
- Endpoint upload final still và diagnostic burst.
- Schema gồm frame timestamp, latency, dropped frames, settle duration, controller state, calibration/model SHA.
- Giữ đường ảnh tĩnh hiện tại làm fallback/debug.
- Không chạy inspection AI trong vòng local.

### 11.4 Test

- Replay video đã ghi: marker tốt, mất marker, blur, glare, vibration.
- Synthetic delayed frame và variable latency.
- Fake actuator có backlash/deadband/overshoot.
- Property test sign/frame transform.
- Hardware-in-loop với giới hạn chuyển động nhỏ.
- Estop, timeout, mất camera, mất mạng, queue đầy, Pi reboot.

## 12. Lộ trình triển khai

### P0 — đo baseline, chưa điều khiển

- Giữ camera mở, capture lores vào RAM.
- Chạy detector/PnP và log latency, CPU, temperature, frame age.
- Không phát motor command.
- So pose lores với pose ảnh tĩnh hiện tại.

Điều kiện qua: không backlog; frame age bị chặn; marker detection/rejection ổn định trong ma trận ánh sáng/góc nhìn.

### P1 — shadow realtime

- Controller tính lệnh nhưng chỉ log.
- Operator hoặc vòng ảnh tĩnh hiện tại vẫn điều khiển.
- So predicted command với residual sau move.

Điều kiện qua: đúng dấu/trục, không có lệnh lớn bất thường, quality gate fail-closed.

### P2 — pan–tilt coarse

- Cho phép delta rất nhỏ, tốc độ thấp.
- Hysteresis, timeout, boundary, estop.
- Không yêu cầu servo đạt tolerance nhỏ hơn repeatability đã đo.

### P3 — slider closed-loop

- Homing/limit switch.
- Move–settle–measure, không đo khi chạy.
- Thêm acceleration profile và đo backlash/missed steps.

### P4 — final capture

- K frame stable → final settle → full-res burst.
- Final pose verify bằng calibration final mode.
- Upload một ảnh master và metadata.

### P5 — tối ưu

- ROI, optical flow, native detector, adaptive resolution.
- Tùy benchmark, thử AprilGrid/AprilTag/ArUco Nano.[^11][^12]
- Thêm IMU hoặc global-shutter camera chỉ cho pose nếu rolling shutter vẫn là giới hạn.

## 13. Benchmark bắt buộc trên Raspberry Pi

Chạy trên đúng Pi, camera, cooling và nguồn điện:

| Bài thử | Đầu ra |
|---|---|
| Camera lores không CV | camera fps, dropped frames, frame age |
| Detect full-frame 640/960/1280 | median/p95 latency, detection rate |
| ROI detect | latency và recovery khi marker rời ROI |
| PnP mỗi frame / mỗi 3–5 frame | accuracy–load tradeoff |
| Optical flow giữa detect | drift và re-detection rate |
| Servo step response | delay, overshoot, settle, repeatability, backlash |
| Slider step response | mm error, missed steps, vibration settle |
| 30 phút và 2 giờ | thermal throttle, memory growth |
| Network/UI stream bật/tắt | ảnh hưởng latency local |
| Preview → still switch | delay, FoV/pose jump, focus/exposure consistency |

Metric tối thiểu:

- capture-to-pose latency p50/p95/p99;
- frame age tại thời điểm command;
- effective control update rate;
- CPU từng core, RAM, temperature, clock;
- marker reject/loss/ambiguity rate;
- pose jitter khi đứng yên;
- overshoot, iterations, settle time;
- final pose repeatability;
- full-resolution blur/focus score.

Không chọn cấu hình chỉ theo fps trung bình. Cấu hình tốt là cấu hình có p95 latency và frame age bị chặn, không dao động và cho ảnh cuối lặp lại.

## 14. Khuyến nghị cuối cùng cho đề tài

### Kiến trúc nên chọn

**Local hybrid visual alignment**

- 640×360 hoặc 960×540 lores Y-plane trên Pi.
- ChArUco/AprilTag detect + IBVS nhanh; PnP chậm hơn.
- Pan–tilt coarse, slider fine.
- Move → visual settle → verify nhiều frame.
- Still 4608×2592 sau khi dừng.
- Upload ảnh cuối; server chạy inspection.

### Không nên chọn

- Full-resolution RGB processing ở mọi frame.
- Stream video lên server để server điều khiển motor.
- Continuous motor motion trong khi lấy pose final bằng Camera Module 3.
- Dùng continuous autofocus trong vòng metrology.
- Kết luận hội tụ từ một frame.
- Coi commanded servo angle/current step counter là ground truth.

### Khi nào cần đổi camera hoặc cơ cấu?

- Đổi sang global-shutter pose camera nếu exposure ngắn + settle vẫn không loại rolling-shutter/blur.
- Giữ Camera Module 3 làm inspection camera vì độ phân giải cao.
- Đổi servo nếu tolerance yêu cầu nhỏ hơn repeatability thực nghiệm hoặc controller vẫn hunting sau khi tune.
- Thêm encoder/linear scale nếu stepper có missed-step/backlash không quan sát đủ bằng pose.
- Dùng hai camera nếu một camera không thể vừa chống chuyển động vừa cung cấp chi tiết inspection.

Tóm lại: **video realtime tốt hơn cho quá trình căn chỉnh; ảnh tĩnh tốt hơn cho quyết định cuối**. Hai cách không loại trừ nhau. Thiết kế đúng là dùng video như cảm biến phản hồi và ảnh tĩnh như phép đo/bằng chứng sau khi cơ cấu đã ổn định.

## Sources

[^1]: Chaumette, F. và Hutchinson, S. “[Visual Servo Control, Part I: Basic Approaches](https://web.mit.edu/amcp/OldFiles/drg/Chaumette_Part_I.pdf).” IEEE Robotics & Automation Magazine, 2006.
[^2]: Raspberry Pi. “[Camera hardware documentation](https://www.raspberrypi.com/documentation/accessories/camera.html).” Truy cập 2026-09-12.
[^3]: Raspberry Pi. “[Compute Module camera mode listing](https://www.raspberrypi.com/documentation/hardware/computemodule/).” Truy cập 2026-09-12.
[^4]: Raspberry Pi. “[The Picamera2 Library](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf).” Bản cập nhật 2026.
[^5]: Raspberry Pi. “[New autofocus camera modules](https://www.raspberrypi.com/news/new-autofocus-camera-modules/).” 2023.
[^6]: Smart Agricultural Technology. “[Development of a low-cost ground robot for real-time ArUco marker localization using lightweight deep learning models on an edge device](https://doi.org/10.1016/j.atech.2026.101898).” 2026.
[^7]: OpenCV. “[Optical Flow](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html).” Truy cập 2026-09-12.
[^8]: Raspberry Pi. “[Camera software and video encoding](https://www.raspberrypi.com/documentation/computers/camera_software.html).” Truy cập 2026-09-12.
[^9]: OpenCV. “[Detection of ChArUco Boards](https://docs.opencv.org/4.10.0/df/d4a/tutorial_charuco_detection.html).” Truy cập 2026-09-12.
[^10]: OpenCV. “[Pose estimation methods and IPPE_SQUARE](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html).” Truy cập 2026-09-12.
[^11]: AprilRobotics. “[AprilTag 3](https://github.com/AprilRobotics/apriltag).” Truy cập 2026-09-12.
[^12]: Romero-Ramirez et al. “[ArUco Nano: a simpler, faster, and more reliable fiducial marker detector](https://doi.org/10.1016/j.softx.2026.102690).” SoftwareX, 2026.
