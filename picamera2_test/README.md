# Picamera2 Test Project

โปรเจกต์สำหรับทดสอบ Picamera2 กับ IMX500 AI Camera บน Raspberry Pi 5

## วัตถุประสงค์

1. ทดสอบการใช้งาน Picamera2 API
2. ทดสอบการอ่าน metadata จาก IMX500
3. ทดสอบ AI inference บน IMX500 chip
4. ทดสอบ configuration ต่างๆ ของกล้อง

## โครงสร้างไฟล์

```
picamera2_test/
├── README.md
├── requirements.txt
│
├── 01_basic_capture.py      # ทดสอบ capture ภาพพื้นฐาน
├── 02_preview_display.py    # ทดสอบ preview หน้าจอ (Qt/OpenCV)
├── 03_camera_info.py        # แสดงข้อมูลกล้องทั้งหมด
├── 04_metadata_test.py      # อ่าน metadata จากกล้อง
├── 05_video_record.py       # ทดสอบบันทึกวีดีโอ
├── 06_dual_camera.py        # ทดสอบกล้อง 2 ตัวพร้อมกัน
│
└── imx500/                  # ตัวอย่างสำหรับ IMX500 AI Camera
    ├── imx500_get_device_id.py   # อ่าน Device ID
    ├── imx500_simple_test.py     # ทดสอบแบบ headless
    ├── imx500_classification.py  # Image Classification
    ├── imx500_object_detection.py # Object Detection
    └── assets/
        └── coco_labels.txt       # COCO dataset labels
```

## การใช้งาน

### ติดตั้ง Dependencies บน Raspberry Pi
```bash
pip install -r requirements.txt
```

### Deploy ไปยัง Raspberry Pi
```powershell
# จาก Windows
scp -i "C:\Users\piyanat\.ssh\id_ed25519_pi5" -r D:/mobile/pi5/picamera2_test pi@192.168.1.43:/home/pi/
```

### รันบน Raspberry Pi

#### ทดสอบพื้นฐาน
```bash
cd ~/picamera2_test

# ดูข้อมูลกล้องทั้งหมด
python3 03_camera_info.py

# อ่าน metadata
python3 04_metadata_test.py

# Capture ภาพ
python3 01_basic_capture.py

# Preview ผ่าน OpenCV
python3 02_preview_display.py --opencv
```

#### ทดสอบ IMX500 AI Camera
```bash
cd ~/picamera2_test/imx500

# ทดสอบแบบ headless (SSH)
python3 imx500_simple_test.py

# อ่าน Device ID
python3 imx500_get_device_id.py

# Object Detection (ต้องมี display)
python3 imx500_object_detection.py

# Image Classification
python3 imx500_classification.py
```

## Hardware Requirements

- Raspberry Pi 5
- IMX500 AI Camera (CSI) - สำหรับตัวอย่าง imx500/
- กล้อง CSI อื่นๆ เช่น IMX219, IMX708 - สำหรับตัวอย่างทั่วไป

## AI Models ที่รองรับ

IMX500 มี models ติดตั้งมาให้ใน `/usr/share/imx500-models/`:

| Model | Task | Description |
|-------|------|-------------|
| `imx500_network_mobilenet_v2.rpk` | Classification | MobileNet V2 ImageNet |
| `imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk` | Object Detection | SSD MobileNet COCO |
| `imx500_network_efficientnet_lite0.rpk` | Classification | EfficientNet Lite |
| `imx500_network_efficientdet_lite0_pp.rpk` | Object Detection | EfficientDet |

## เอกสารอ้างอิง

- [Picamera2 Manual (PDF)](../testimax500/RP-008156-DS-2-picamera2-manual.pdf)
- [Picamera2 GitHub Examples](https://github.com/raspberrypi/picamera2/tree/main/examples)
- [IMX500 AI Camera Docs](https://www.raspberrypi.com/documentation/accessories/ai-camera.html)

## Source

ตัวอย่างส่วนใหญ่ดัดแปลงจาก Official Picamera2 Examples:
- https://github.com/raspberrypi/picamera2/tree/main/examples
- https://github.com/raspberrypi/picamera2/tree/main/examples/imx500
