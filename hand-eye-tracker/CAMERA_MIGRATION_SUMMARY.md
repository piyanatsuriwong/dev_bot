# 📊 สรุปการเปลี่ยนแปลง Camera Configuration

## ✅ การอัปเดตล่าสุด (2026-01-22 19:58)

### 🔄 เปลี่ยนจาก:
- **HAND Mode:** USB Webcam (`cv2.VideoCapture`)
- **AI Mode:** IMX500 + IMX708 (dual camera)

### ➡️ เปลี่ยนเป็น:
- **HAND Mode:** IMX708 HQ Camera (CSI-2, `Picamera2`)
- **AI Mode:** IMX500 AI Camera (CSI-2, `modlib`)

---

## 🎯 Camera Configuration ปัจจุบัน

| Mode | Camera | Interface | Library | Resolution | Purpose |
|------|--------|-----------|---------|-----------|---------|
| **HAND** | IMX708 | CSI-2 (/dev/video0) | Picamera2 | 1280x720 | Hand Tracking |
| **AI** | IMX500 | CSI-2 (/dev/video1) | modlib.AiCamera | 4056x3040 | YOLO Detection |

### ข้อดีของการใช้ CSI Cameras:

1. ✅ **ความเร็วสูง:** CSI-2 interface เร็วกว่า USB
2. ✅ **Latency ต่ำ:** เหมาะกับ real-time tracking
3. ✅ **คุณภาพสูง:** IMX708 = 12MP sensor
4. ✅ **Native Support:** ใช้ Picamera2 library ที่ optimize สำหรับ RPi
5. ✅ **ไม่ใช้ USB bandwidth:** เหลือให้อุปกรณ์อื่น

---

## 🔌 Hardware Connection

### IMX708 (HAND Mode)
```
Raspberry Pi 5
   CAM0/CAM1 port (CSI-2)
        ↓
   IMX708 HQ Camera
        ↓
   /dev/video0
        ↓
   Picamera2 → MediaPipe → Hand Tracking
```

### IMX500 (AI Mode)
```
Raspberry Pi 5
   CAM0/CAM1 port (CSI-2)
        ↓
   IMX500 AI Camera
        ↓
   /dev/video1
        ↓
   modlib.AiCamera → YOLO11n/YOLOv8n → Object Detection
```

---

## 📝 Code Changes Required

### 1. HandTrackerMediaPipe (ต้องแก้)

**เดิม (USB Webcam):**
```python
class HandTrackerMediaPipe:
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)  # USB webcam
```

**ใหม่ (IMX708 CSI):**
```python
class HandTrackerMediaPipe:
    def __init__(self, camera_id=0):
        from picamera2 import Picamera2
        self.picam = Picamera2(camera_num=camera_id)
        config = self.picam.create_preview_configuration(
            main={"size": (1280, 720), "format": "RGB888"}
        )
        self.picam.configure(config)
        self.picam.start()
```

### 2. HandTrackerOpenCV (ต้องแก้)

**เดิม:**
```python
class HandTrackerOpenCV:
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
```

**ใหม่:**
```python
class HandTrackerOpenCV:
    def __init__(self, camera_id=0):
        from picamera2 import Picamera2
        self.picam = Picamera2(camera_num=camera_id)
        # Same as above
```

### 3. Frame Capture Changes

**เดิม:**
```python
ret, frame = self.cap.read()
```

**ใหม่:**
```python
frame = self.picam.capture_array("main")  # Already RGB888
# No need for BGR to RGB conversion
```

---

## 🎯 Camera Index Assignment

| Camera | Index | Path | Notes |
|--------|-------|------|-------|
| **IMX708** | 0 | /dev/video0 | Primary CSI port |
| **IMX500** | 1 | /dev/video1 | Secondary CSI port |

**Verification:**
```bash
libcamera-hello --list-cameras
# Expected output:
# 0 : imx708 [4608x2592] (/base/soc/i2c0mux/i2c@1/imx708@1a)
# 1 : imx500 [4056x3040] (/base/soc/i2c0mux/i2c@0/imx500@1f)
```

---

## 🚀 Benefits Summary

### Performance:
- **CSI-2 bandwidth:** Up to 2.5 Gbps per lane (4 lanes = 10 Gbps)
- **USB 3.0:** Only 5 Gbps (shared with other devices)
- **Result:** Lower latency, higher frame rates

### Quality:
- **IMX708:** 12MP, better low-light performance
- **USB Webcam:** Usually 2-5MP, generic sensors
- **Result:** Better hand detection accuracy

### System Load:
- **CSI:** Direct to GPU/ISP hardware acceleration
- **USB:** Software decoding on CPU
- **Result:** Lower CPU usage

---

## ⚠️ Migration Notes

### ต้องติดตั้งเพิ่ม:
```bash
# Picamera2 library (should be pre-installed on RPi OS)
sudo apt install python3-picamera2

# If not available:
pip3 install picamera2
```

### ต้องแก้ไขโค้ด:
1. ✏️ `HandTrackerMediaPipe.__init__()` - เปลี่ยนจาก cv2 เป็น Picamera2
2. ✏️ `HandTrackerOpenCV.__init__()` - เปลี่ยนจาก cv2 เป็น Picamera2  
3. ✏️ `update()` method - ใช้ `capture_array()` แทน `read()`
4. ✏️ Color conversion - ลบ BGR→RGB (Picamera2 output RGB directly)

### ไม่ต้องแก้:
- ✅ YOLO tracker (ใช้ modlib อยู่แล้ว)
- ✅ RoboEyes rendering
- ✅ Servo controller
- ✅ Display drivers

---

## 🎬 Next Steps

1. **✏️ แก้ไข `main_roboeyes.py`:**
   - Update HandTrackerMediaPipe class
   - Update HandTrackerOpenCV class
   - Test IMX708 initialization

2. **🧪 ทดสอบ:**
   ```bash
   # Test IMX708
   python3 main_roboeyes.py --mode hand
   
   # Test IMX500
   python3 main_roboeyes.py --mode ai
   ```

3. **📝 อัปเดตเอกสาร:**
   - ✅ DEVICE_CONNECTIONS.md (เสร็จแล้ว)
   - ⏳ README.md (รออัปเดต)
   - ⏳ TROUBLESHOOTING.md (เพิ่มส่วน CSI camera)

---

**Last Updated:** 2026-01-22 19:58  
**Status:** Documentation updated ✅ | Code changes needed ⏳
