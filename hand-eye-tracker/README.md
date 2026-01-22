# Hand-Eye Tracker for Raspberry Pi 5

ระบบติดตามมือด้วย Webcam และให้ตา (Sentient Eye) ขยับตามมือ
ออกแบบมาสำหรับ **Raspberry Pi 5 ARM64**

## Features

- ติดตามตำแหน่งมือแบบ real-time
- ตา (Sentient Eye Engine) ขยับตามตำแหน่งมือ
- รองรับ 2 โหมด:
  - **MediaPipe** (แม่นยำ, ต้องติดตั้ง MediaPipe สำหรับ ARM64)
  - **OpenCV Skin Detection** (fallback, ไม่ต้องติดตั้ง MediaPipe)
- **AI Camera Mode (IMX500)**:
  - รองรับ YOLO object detection ด้วย IMX500 AI Camera
  - ใช้ **YOLO11n** model (ตัวเลือกแรก) - ประสิทธิภาพดีกว่า YOLOv8n 37%
  - Fallback เป็น **YOLOv8n** ถ้า YOLO11n ไม่มีใน system
  - เหมาะสำหรับ embedded devices (ลด complexity, เพิ่ม accuracy)


## การติดตั้ง

### 1. ติดตั้ง System Dependencies

```bash
sudo apt update
sudo apt install -y python3-pygame python3-numpy python3-opencv
```

### 2. ติดตั้ง Python Packages

```bash
# OpenCV สำหรับ Raspberry Pi 5 (ใช้ version 4.9.0 ที่ stable)
pip3 install opencv-python-headless==4.9.0.80 --break-system-packages
```

### 3. ติดตั้ง MediaPipe สำหรับ ARM64 (Optional)

MediaPipe ไม่มี official wheel สำหรับ ARM64 ต้องใช้วิธีใดวิธีหนึ่ง:

#### วิธี A: ใช้ Pre-built Wheel

```bash
# ดาวน์โหลด pre-built wheel สำหรับ Raspberry Pi 5
# จาก: https://github.com/Melvinsajith/How-to-Install-Mediapipe-on-Raspberry-Pi-5

pip3 install mediapipe-0.10.14-cp311-cp311-linux_aarch64.whl --break-system-packages
```

#### วิธี B: Build จาก Source

```bash
# ดู: https://github.com/jiuqiant/mediapipe-python-aarch64
git clone https://github.com/google/mediapipe.git
cd mediapipe
pip3 install -r requirements.txt --break-system-packages
python3 setup.py bdist_wheel
pip3 install dist/*.whl --break-system-packages
```

**หมายเหตุ:** ถ้าไม่ติดตั้ง MediaPipe โปรแกรมจะใช้ OpenCV skin detection โดยอัตโนมัติ

## การใช้งาน

### โหมดปกติ (Hand Tracking)

```bash
cd ~/hand-eye-tracker
./run.sh

# หรือ
python3 main_roboeyes.py

# บังคับใช้ OpenCV mode (ไม่ใช้ MediaPipe)
python3 main_roboeyes.py --no-mediapipe

# ระบุ camera
python3 main_roboeyes.py --camera 0
```

### โหมด AI Camera (IMX500 YOLO Detection)

```bash
# เปิดใช้งาน AI Camera mode
python3 main_roboeyes.py --ai-camera

# ปรับค่า confidence threshold
python3 main_roboeyes.py --ai-camera --yolo-confidence 0.6
```

**หมายเหตุ**: โปรแกรมจะเลือกใช้ **YOLO11n** โดยอัตโนมัติ (ถ้ามี) เพราะมีประสิทธิภาพดีกว่า YOLOv8n

## Controls


| ปุ่ม | การทำงาน |
|------|----------|
| ขยับมือ | ตาจะขยับตามตำแหน่งมือ |
| SPACE | เปลี่ยน emotion แบบสุ่ม |
| W | เปิด/ปิดหน้าต่าง webcam |
| ESC | ออกจากโปรแกรม |
| **D** | **(AI Camera)** สลับไปโหมด DETECT (YOLO object detection) |
| **T** | **(AI Camera)** สลับไปโหมด TRACK (Hand tracking) |


## โครงสร้างไฟล์

```
hand-eye-tracker/
├── main.py           # โปรแกรมหลัก
├── face_renderer.py  # ตัว render ตา (Sentient-Eye-Engine)
├── config.py         # การตั้งค่า
├── assets/           # ข้อมูล emotion และภาพ
│   ├── data/         # JSON files สำหรับ emotions
│   └── eyes/         # ภาพตา
├── requirements.txt  # dependencies
├── run.sh           # script รัน
└── README.md        # คู่มือนี้
```

## การทำงาน

1. โปรแกรมอ่านภาพจาก webcam (Logitech C920)
2. ตรวจจับตำแหน่งมือ:
   - **MediaPipe**: ใช้ AI model ตรวจจับ hand landmarks
   - **OpenCV**: ใช้ skin color detection + contour analysis
3. แปลงตำแหน่งมือเป็นทิศทางการมองของตา
4. แสดงผลตาด้วย Sentient Eye Engine

## Troubleshooting

### Webcam ไม่ทำงาน
```bash
# ตรวจสอบ webcam
ls -la /dev/video*
v4l2-ctl --list-devices
```

### MediaPipe ติดตั้งไม่ได้
- ใช้ `--no-mediapipe` flag เพื่อใช้ OpenCV mode แทน
- OpenCV mode ทำงานได้โดยไม่ต้องติดตั้ง MediaPipe

### Performance ช้า
- ลด resolution ใน config.py
- ใช้ `model_complexity=0` ใน MediaPipe
- ปิดหน้าต่าง webcam ด้วยปุ่ม W

## Credits

- Eye rendering: [Sentient-Eye-Engine](../Sentient-Eye-Engine)
- Hand tracking: MediaPipe / OpenCV
