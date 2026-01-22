# IMX500 AI Camera - YOLO Model Selection Guide

## Overview

โปรเจกต์นี้รองรับการใช้ **IMX500 AI Camera** สำหรับ object detection ด้วย YOLO models โดยมีการเลือก model อัตโนมัติเพื่อให้ได้ประสิทธิภาพที่ดีที่สุด

## Supported Models

### 1. YOLO11n (Recommended - Default)

**ข้อดี:**
- ✅ **ลด Complexity ได้ 37%** เมื่อเทียบกับ YOLOv8n
- ✅ **ความแม่นยำสูงกว่า** (Better mAP - mean Average Precision)
- ✅ **Stable Bounding Boxes** - กรอบที่ detect ได้มีความเสถียรกว่า
- ✅ **Optimized สำหรับ Embedded Devices** - ออกแบบมาเพื่อ Raspberry Pi โดยเฉพาะ
- ✅ **Faster Inference** - ประมวลผลเร็วกว่าในบาง scenarios
- ✅ **Better Detection Count** - detect objects ได้มากกว่าและแม่นกว่า

**ข้อมูล Technical:**
- Model Architecture: C3k2 และ C2PSA blocks (ปรับปรุงจาก YOLOv8)
- Parameters: น้อยกว่า YOLOv8n ~37%
- Inference Speed: ~13 FPS (Raspberry Pi 5, INT8 quantization, CPU-only)
- Latency: ~77ms average end-to-end
- Suitable for: Real-time object detection, tracking, complex environments

### 2. YOLOv8n (Fallback)

**ใช้เมื่อ:**
- YOLO11n ไม่มีใน system
- มี modlib version เก่าที่ยังไม่รองรับ YOLO11n

**ข้อดี:**
- ✅ เร็วกว่าเล็กน้อยใน raw FPS (เฉพาะบาง scenarios)
- ✅ Mature และ stable (มีการใช้งานแพร่หลาย)
- ✅ Compatible กับ modlib versions ทั้งหมด

**ข้อมูล Technical:**
- Inference Speed: ~13 FPS (Raspberry Pi 5, INT8 quantization, CPU-only)
- Suitable for: General object detection

## How Model Selection Works

โปรแกรมจะทำการเลือก model ดังนี้:

```python
try:
    from modlib.models.zoo import YOLO11n
    YOLO_MODEL_CLASS = YOLO11n
    model_name = "YOLO11n"
    print("✓ Using YOLO11n (Optimized for IMX500)")
except ImportError:
    from modlib.models.zoo import YOLOv8n
    YOLO_MODEL_CLASS = YOLOv8n
    model_name = "YOLOv8n"
    print("✓ Using YOLOv8n (Fallback)")
```

### Startup Messages

เมื่อเริ่มโปรแกรม คุณจะเห็น message แจ้งว่าใช้ model อะไร:

```
YOLO (IMX500): Available - Using YOLO11n model
Loading YOLO11n model...
Model loaded! Classes: 80
```

หรือ

```
YOLO (IMX500): Available - Using YOLOv8n model
Loading YOLOv8n model...
Model loaded! Classes: 80
```

## Installation Requirements

### ติดตั้ง modlib สำหรับ IMX500

```bash
# Install Sony modlib (รองรับทั้ง YOLO11n และ YOLOv8n)
pip3 install modlib --upgrade

# หรือ build จาก source สำหรับ version ล่าสุด
git clone https://github.com/raspberrypi/picamera2
cd picamera2
pip3 install .
```

### ตรวจสอบว่า YOLO11n มีใน system หรือไม่

```bash
python3 -c "from modlib.models.zoo import YOLO11n; print('YOLO11n is available')"
```

ถ้า import สำเร็จ แสดงว่า YOLO11n พร้อมใช้งาน

## Performance Comparison

| Metric | YOLO11n | YOLOv8n |
|--------|---------|---------|
| Model Complexity | 37% less | Baseline |
| Accuracy (mAP) | Higher ⬆️ | Standard |
| FPS (RPi5, CPU) | ~13 FPS | ~13 FPS |
| Latency | ~77ms | ~77ms |
| Detection Stability | Better ⭐ | Good |
| Objects Detected | More ⬆️ | Standard |
| Embedded Optimization | Yes ✅ | Partial |

*Based on Raspberry Pi 5, INT8 quantization, CPU-only benchmarks*

## Usage Examples

### โหมด AI Camera (YOLO Detection)

```bash
# เริ่มโปรแกรมด้วย AI Camera mode (จะใช้ YOLO11n automatically)
python3 main_roboeyes.py --ai-camera

# ปรับ confidence threshold
python3 main_roboeyes.py --ai-camera --yolo-confidence 0.6
```

### Keyboard Controls (ขณะรันโปรแกรม)

- **D**: สลับไปโหมด DETECT (แสดงผล objects ที่ detect ได้)
- **T**: สลับไปโหมด TRACK (ติดตาม object เฉพาะ เช่น person)
- **ESC**: ออกจากโปรแกรม

## Troubleshooting

### YOLO11n ไม่พบใน system

**อาการ:**
```
YOLO (IMX500): Available - Using YOLOv8n model
```

**วิธีแก้:**
1. Update modlib เป็น version ล่าสุด:
   ```bash
   pip3 install modlib --upgrade
   ```

2. ตรวจสอบ version:
   ```bash
   pip3 show modlib
   ```

3. ถ้า modlib version เก่า ให้ install จาก source:
   ```bash
   pip3 install git+https://github.com/raspberrypi/picamera2.git#subdirectory=modlib
   ```

### Performance ไม่ดีตามที่คาดหวัง

**การแก้ไข:**
1. ลด confidence threshold:
   ```bash
   python3 main_roboeyes.py --ai-camera --yolo-confidence 0.4
   ```

2. ตรวจสอบว่าใช้ IMX500 จริง (ไม่ใช่ USB webcam):
   ```bash
   libcamera-hello --list-cameras
   # ต้องเห็น IMX500 ใน list
   ```

3. Overclock Raspberry Pi 5 (เพิ่ม performance):
   ```bash
   # Edit /boot/firmware/config.txt
   sudo nano /boot/firmware/config.txt
   ```
   เพิ่ม:
   ```
   arm_freq=2800
   gpu_freq=900
   ```

### IMX500 "Device or resource busy"

ดูได้ที่: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

## References

- [YOLO11 Official Documentation](https://docs.ultralytics.com/models/yolo11/)
- [Raspberry Pi AI Camera Guide](https://www.raspberrypi.com/documentation/accessories/ai-camera.html)
- [modlib GitHub Repository](https://github.com/raspberrypi/picamera2)
- [Performance Benchmarks](https://learnopencv.com/yolov11-on-raspberry-pi/)

## Summary

| Scenario | Recommended Model |
|----------|------------------|
| **ต้องการประสิทธิภาพสูงสุด** | YOLO11n ✅ |
| **ต้องการ accuracy สูง** | YOLO11n ✅ |
| **ต้องการ stable detection** | YOLO11n ✅ |
| **modlib version เก่า** | YOLOv8n (auto fallback) |
| **ไม่สามารถ update modlib ได้** | YOLOv8n (auto fallback) |

**คำแนะนำ:** ใช้ **YOLO11n** เสมอถ้าเป็นไปได้ เพราะให้ประสิทธิภาพและความแม่นยำที่ดีกว่าใน embedded environment
