# Raspberry Pi 5 Remote Development

## SSH Connection Details

- **Host:** 192.168.1.43
- **Username:** pi
- **SSH Key:** `C:\Users\piyanat\.ssh\id_ed25519_pi5`

---

# คำสั่งทั้งหมดที่ต้องใช้

## 1. SSH Commands (Windows Terminal)

```bash
# เชื่อมต่อ SSH แบบ Interactive
ssh -i "C:\Users\piyanat\.ssh\id_ed25519_pi5" pi@192.168.1.43

# รันคำสั่งเดี่ยว
ssh -i "C:\Users\piyanat\.ssh\id_ed25519_pi5" pi@192.168.1.43 "command_here"

# Copy ไฟล์ไป Pi (scp)
scp -i "C:\Users\piyanat\.ssh\id_ed25519_pi5" "D:/mobile/pi5/numbot/filename.py" pi@192.168.1.43:/home/pi/numbot/

# Copy หลายไฟล์
scp -i "C:\Users\piyanat\.ssh\id_ed25519_pi5" D:/mobile/pi5/numbot/*.py pi@192.168.1.43:/home/pi/numbot/
```

## 2. Fresh Install (หลัง Reinstall OS)

```bash
# สร้างโฟลเดอร์โปรเจค
ssh -i "C:\Users\piyanat\.ssh\id_ed25519_pi5" pi@192.168.1.43 "mkdir -p /home/pi/numbot/assets/eyes /home/pi/numbot/assets/data /home/pi/numbot/assets/sounds"

# สร้าง Virtual Environment
ssh -i "C:\Users\piyanat\.ssh\id_ed25519_pi5" pi@192.168.1.43 "cd /home/pi/numbot && python3 -m venv env"

# ติดตั้ง Dependencies ทั้งหมด
ssh -i "C:\Users\piyanat\.ssh\id_ed25519_pi5" pi@192.168.1.43 "cd /home/pi/numbot && source env/bin/activate && pip install --upgrade pip && pip install mediapipe opencv-python pygame SpeechRecognition gTTS numpy pillow spidev lgpio"

# เปิด SPI
ssh -i "C:\Users\piyanat\.ssh\id_ed25519_pi5" pi@192.168.1.43 "sudo raspi-config nonint do_spi 0"

# ติดตั้ง flac (สำหรับ Speech Recognition)
ssh -i "C:\Users\piyanat\.ssh\id_ed25519_pi5" pi@192.168.1.43 "sudo apt-get update && sudo apt-get install -y flac"
```

## 3. Copy ไฟล์โปรเจค (Windows Terminal)

```bash
# Copy Python files
scp -i "C:\Users\piyanat\.ssh\id_ed25519_pi5" D:/mobile/pi5/numbot/*.py pi@192.168.1.43:/home/pi/numbot/

# Copy assets
scp -i "C:\Users\piyanat\.ssh\id_ed25519_pi5" D:/mobile/pi5/numbot/assets/eyes/*.png pi@192.168.1.43:/home/pi/numbot/assets/eyes/
scp -i "C:\Users\piyanat\.ssh\id_ed25519_pi5" D:/mobile/pi5/numbot/assets/data/*.json pi@192.168.1.43:/home/pi/numbot/assets/data/
scp -i "C:\Users\piyanat\.ssh\id_ed25519_pi5" D:/mobile/pi5/numbot/assets/sounds/*.wav pi@192.168.1.43:/home/pi/numbot/assets/sounds/
scp -i "C:\Users\piyanat\.ssh\id_ed25519_pi5" D:/mobile/pi5/numbot/assets/solver.png pi@192.168.1.43:/home/pi/numbot/assets/
```

## 4. Run โปรแกรม

```bash
# Run main_roboeyes.py (จาก Windows)
ssh -i "C:\Users\piyanat\.ssh\id_ed25519_pi5" pi@192.168.1.43 "cd /home/pi/numbot && source env/bin/activate && python3 main_roboeyes.py"

# Run ใน Demo mode (ไม่ต้องกล้อง)
ssh -i "C:\Users\piyanat\.ssh\id_ed25519_pi5" pi@192.168.1.43 "cd /home/pi/numbot && source env/bin/activate && python3 main_roboeyes.py --demo"

# Run test hand landmarks
ssh -i "C:\Users\piyanat\.ssh\id_ed25519_pi5" pi@192.168.1.43 "cd /home/pi/numbot && source env/bin/activate && python3 test_hand_landmarks.py"

# Run test microphone
ssh -i "C:\Users\piyanat\.ssh\id_ed25519_pi5" pi@192.168.1.43 "cd /home/pi/numbot && source env/bin/activate && python3 test_mic.py"
```

## 5. Run โปรแกรม (SSH Interactive บน Pi)

```bash
# หลังจาก SSH เข้าไปแล้ว
cd /home/pi/numbot
source env/bin/activate

# Run programs
python3 main_roboeyes.py
python3 main_roboeyes.py --demo
python3 test_hand_landmarks.py
python3 test_mic.py
```

## 6. ตรวจสอบ Hardware

```bash
# ตรวจสอบ Webcam
ssh -i "C:\Users\piyanat\.ssh\id_ed25519_pi5" pi@192.168.1.43 "lsusb && v4l2-ctl --list-devices"

# ตรวจสอบ SPI
ssh -i "C:\Users\piyanat\.ssh\id_ed25519_pi5" pi@192.168.1.43 "ls -la /dev/spi*"

# ตรวจสอบ Microphone
ssh -i "C:\Users\piyanat\.ssh\id_ed25519_pi5" pi@192.168.1.43 "arecord -l"
```

## 7. Kill Process

```bash
# Kill all Python processes
ssh -i "C:\Users\piyanat\.ssh\id_ed25519_pi5" pi@192.168.1.43 "pkill -9 python3"

# Kill specific script
ssh -i "C:\Users\piyanat\.ssh\id_ed25519_pi5" pi@192.168.1.43 "pkill -9 -f main_roboeyes.py"

# ดู process ที่รันอยู่
ssh -i "C:\Users\piyanat\.ssh\id_ed25519_pi5" pi@192.168.1.43 "ps aux | grep python"

# Kill ทุกอย่างที่เกี่ยวกับ numbot
ssh -i "C:\Users\piyanat\.ssh\id_ed25519_pi5" pi@192.168.1.43 "pkill -9 -f numbot"
```

---

## Project Paths

- **Local (Windows):** `D:\mobile\pi5\numbot\`
- **Remote (Pi):** `/home/pi/numbot/`

---

# Hand-Eye Tracker Commands

## Quick Start

```bash
# SSH to Pi
ssh -i "C:\Users\piyanat\.ssh\id_ed25519_pi5" pi@192.168.1.43

# Run on Pi
cd /home/pi/numbot
python3 main_roboeyes.py
```

## main_roboeyes.py - RoboEyes Hand Tracker

แอปหลักสำหรับแสดงตาหุ่นยนต์บนจอ GC9A01A พร้อม hand tracking

### Usage

```bash
python3 main_roboeyes.py [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--demo` | โหมด Demo (ไม่ใช้กล้อง) - ตาจะเคลื่อนไหวอัตโนมัติ |
| `--no-mediapipe` | ใช้ OpenCV แทน MediaPipe สำหรับ hand tracking |
| `--camera N` | เลือกกล้อง (default: 0) เช่น `--camera 2` สำหรับ webcam |
| `--no-camera-view` | ซ่อนหน้าต่างกล้องบน HDMI |
| `--no-tracking` | ปิด hand tracking (ตาจะไม่ตามมือ) |

### Examples

```bash
# โหมดปกติ (กล้อง 0 + MediaPipe + แสดงกล้องบน HDMI)
python3 main_roboeyes.py

# โหมด Demo (ไม่ต้องกล้อง)
python3 main_roboeyes.py --demo

# ใช้กล้อง webcam (device 2)
python3 main_roboeyes.py --camera 2

# ใช้ OpenCV แทน MediaPipe
python3 main_roboeyes.py --no-mediapipe

# Headless mode (ไม่แสดงหน้าต่างกล้อง)
python3 main_roboeyes.py --no-camera-view

# รวมหลาย options
python3 main_roboeyes.py --camera 2 --no-mediapipe --no-camera-view
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `ESC` | ออกจากโปรแกรม |
| `SPACE` | สุ่มเปลี่ยนอารมณ์ตา |

### Voice Commands (ถ้าเปิดใช้งาน)

| Command | Mood |
|---------|------|
| "happy", "smile" | HAPPY |
| "angry", "mad" | ANGRY |
| "tired", "sleep" | TIRED |
| "scary", "fear" | SCARY |
| "frozen", "cold" | FROZEN |
| "curious", "what" | CURIOUS |
| "normal", "reset" | DEFAULT |

### Gesture Controls (OpenCV mode)

| Gesture | Mood |
|---------|------|
| Fist (กำปั้น) | ANGRY |
| 2 fingers | DEFAULT |
| 3 fingers | TIRED |
| 4 fingers | SCARY |
| Open hand (กางมือ) | HAPPY |

---

## test_mic.py - Microphone Test

ทดสอบไมโครโฟนและ Speech Recognition

### Usage

```bash
python3 test_mic.py
```

### Output

- **Step 1:** ทดสอบ hardware recording (arecord)
- **Step 2:** ทดสอบ speech_recognition library
- **Step 3:** ทดสอบ Google Speech API

---

## config.py - Configuration

| Setting | Value | Description |
|---------|-------|-------------|
| `DISPLAY_MODE` | `"gc9a01a"` | โหมดจอ (gc9a01a หรือ pygame) |
| `GC9A01A_WIDTH` | `240` | ความกว้างจอ |
| `GC9A01A_HEIGHT` | `240` | ความสูงจอ |
| `GC9A01A_SPI_PORT` | `0` | SPI port |
| `GC9A01A_DC_PIN` | `25` | Data/Command pin |
| `GC9A01A_RST_PIN` | `27` | Reset pin |
| `GC9A01A_BL_PIN` | `18` | Backlight pin |
| `FPS` | `60` | Frame rate |

---

## Hardware Setup

### Camera Devices

```bash
# List available cameras
ls -la /dev/video*

# Check camera info
v4l2-ctl --list-devices
```

### Microphone Devices

```bash
# List recording devices
arecord -l

# Test recording (3 seconds)
arecord -D plughw:2,0 -d 3 -f cd -t wav /tmp/test.wav

# Play back
aplay /tmp/test.wav
```

### GC9A01A Display Pins

| Pin | GPIO | Description |
|-----|------|-------------|
| VCC | 3.3V | Power |
| GND | GND | Ground |
| DIN | GPIO 10 (MOSI) | SPI Data |
| CLK | GPIO 11 (SCLK) | SPI Clock |
| CS | GPIO 8 (CE0) | Chip Select |
| DC | GPIO 25 | Data/Command |
| RST | GPIO 27 | Reset |
| BL | GPIO 18 | Backlight |

---

## Troubleshooting

### Camera not found

```bash
# Check if camera exists
ls /dev/video*

# Try different camera index
python3 main_roboeyes.py --camera 2
```

### No display output

```bash
# Use demo mode first
python3 main_roboeyes.py --demo

# Check SPI is enabled
ls /dev/spidev*
```

### Speech recognition error

```bash
# Install dependencies
pip3 install SpeechRecognition --break-system-packages

# Install flac
sudo apt-get install flac
```

### dpkg/apt broken

```bash
# Restore from backup
sudo zcat /var/backups/dpkg.status.1.gz > /tmp/status_good
sudo cp /tmp/status_good /var/lib/dpkg/status
```

---

# Raspberry Pi AI Camera (IMX500)

## Overview

กล้อง AI ที่มี Neural Network Accelerator ในตัว (Sony IMX500)

| Component | Specification |
|-----------|---------------|
| Sensor | Sony IMX500 |
| Resolution | 4056x3040 (12.3MP) |
| AI Accelerator | Built-in NN Processor |
| Interface | CSI |

---

## AI Camera Installation

```bash
# Update system
sudo apt update && sudo apt full-upgrade -y

# Install IMX500 packages
sudo apt install -y imx500-all

# Reboot
sudo reboot

# Verify
rpicam-hello --list-cameras
ls /usr/share/imx500-models/
```

---

## Quick Test Commands

### Object Detection (HDMI Display)
```bash
# MobileNet SSD
rpicam-hello -t 0 --post-process-file /usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json --viewfinder-width 1920 --viewfinder-height 1080

# Pose Estimation
rpicam-hello -t 0 --post-process-file /usr/share/rpi-camera-assets/imx500_posenet.json
```

### Check Camera
```bash
rpicam-hello --list-cameras
```

---

## YOLO on IMX500

### Install Dependencies
```bash
cd /home/pi/numbot
source env/bin/activate
pip install git+https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library.git
```

### Run YOLO (HTTP Streaming)
```bash
# Start YOLO
ssh -i "C:\Users\piyanat\.ssh\id_ed25519_pi5" pi@192.168.1.43 "cd /home/pi/numbot && source env/bin/activate && nohup python3 -u test_yolo_imx500.py > /tmp/yolo.log 2>&1 &"

# View in browser
http://192.168.1.43:8080

# Check log
ssh -i "C:\Users\piyanat\.ssh\id_ed25519_pi5" pi@192.168.1.43 "tail -f /tmp/yolo.log"

# Stop
ssh -i "C:\Users\piyanat\.ssh\id_ed25519_pi5" pi@192.168.1.43 "pkill -f test_yolo"
```

### Available Models (modlib)
```python
from modlib.models.zoo import YOLOv8n, YOLO11n, NanoDetPlus416x416, SSDMobileNetV2FPNLite320x320, Posenet
```

| Model | Task | mAP |
|-------|------|-----|
| YOLOv8n | Object Detection | 0.279 |
| YOLO11n | Object Detection | 0.374 |
| NanoDet | Object Detection | - |
| PoseNet | Pose Estimation | - |

---

## AI Camera Files

| File | Description |
|------|-------------|
| `test_ai_camera.py` | NanoDet Object Detection + HTTP Stream |
| `test_yolo_imx500.py` | YOLOv8n Object Detection + HTTP Stream |
| `AI_CAMERA.md` | Full documentation |

---

## Hardware Pins (Current Setup)

### AI Camera
- CSI Connector (ไม่ต้องใช้ GPIO)

### GC9A01A Display
| Pin | GPIO | Description |
|-----|------|-------------|
| DC | GPIO 24 | Pin 18 |
| RST | GPIO 25 | Pin 22 |
| BL | GPIO 23 | Pin 16 |
| CS | GPIO 8 | Pin 24 |
| MOSI | GPIO 10 | Pin 19 |
| SCLK | GPIO 11 | Pin 23 |

### Servo (Camera Pan)
| Pin | GPIO | Description |
|-----|------|-------------|
| Signal | GPIO 18 | Pin 12 |
| VCC | 5V | Pin 2/4 |
| GND | GND | Pin 9 |

---

## Troubleshooting AI Camera

### Camera Not Detected
```bash
rpicam-hello --list-cameras
dmesg | grep imx500
```

### Slow First Start
- Firmware upload ครั้งแรกใช้เวลา 1-2 นาที
- รอจนเห็น "Network Firmware Upload: 100%"

### Kill Camera Process
```bash
pkill rpicam
pkill -f test_yolo
pkill -f test_ai_camera
pkill -f main_roboeyes
```

---

# AI Camera Mode (NEW)

## Overview

โหมดใช้ AI Camera (IMX500) ตัวเดียวสำหรับทั้ง Object Detection และ Eye Tracking
ควบคุมด้วยเสียงภาษาไทย/อังกฤษ

## Quick Start

```bash
# รัน AI Camera mode
ssh -i "C:\Users\piyanat\.ssh\id_ed25519_pi5" pi@192.168.1.43 "cd /home/pi/numbot && source env/bin/activate && python3 main_roboeyes.py --ai-camera"
```

## Modes

| Mode | คำอธิบาย |
|------|----------|
| **DETECT** | แสดงสิ่งที่ตรวจจับได้บน GC9A01A |
| **TRACK** | ตาตามวัตถุที่เลือก (default: person) |

## Voice Commands (คำสั่งเสียง)

### เปลี่ยน Mode

| คำสั่ง (ไทย) | คำสั่ง (English) | ผลลัพธ์ |
|-------------|-----------------|---------|
| ตรวจจับ | detect | เปลี่ยนเป็น DETECT mode |
| ติดตาม | track, follow | เปลี่ยนเป็น TRACK mode (ตามคน) |
| ติดตามแมว | track cat | ตาตามแมว |
| ติดตามหมา | track dog | ตาตามหมา |
| คน | person | เปลี่ยนเป้าหมายเป็นคน |
| แมว | cat | เปลี่ยนเป้าหมายเป็นแมว |
| หมา / สุนัข | dog | เปลี่ยนเป้าหมายเป็นหมา |

### ควบคุมการแสดงผล

| คำสั่ง (ไทย) | คำสั่ง (English) | ผลลัพธ์ |
|-------------|-----------------|---------|
| ซ่อน | hide | ซ่อนข้อความบน GC9A01A |
| แสดง | show | แสดงข้อความบน GC9A01A |

### เปลี่ยนอารมณ์ตา

| คำสั่ง (ไทย) | คำสั่ง (English) | Mood |
|-------------|-----------------|------|
| มีความสุข / ยิ้ม | happy, smile | HAPPY |
| โกรธ / โมโห | angry, mad | ANGRY |
| เหนื่อย / ง่วง | tired, sleep | TIRED |
| กลัว / หลอน | scary, fear | SCARY |
| หนาว / เย็น | frozen, cold | FROZEN |
| สงสัย / อะไร | curious, what | CURIOUS |
| ปกติ / รีเซ็ต | normal, reset | DEFAULT |

## Keyboard Controls

| Key | Action |
|-----|--------|
| ESC | ออกจากโปรแกรม |
| SPACE | สุ่ม mood |
| D | เปลี่ยนเป็น DETECT mode |
| T | เปลี่ยนเป็น TRACK mode |

## Available Track Targets (80 Classes)

สามารถติดตามวัตถุได้ 80 ชนิด (COCO Dataset):

**คน/สัตว์:** person, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

**ยานพาหนะ:** bicycle, car, motorcycle, airplane, bus, train, truck, boat

**ของใช้:** bottle, cup, fork, knife, spoon, bowl, chair, couch, bed, toilet, tv, laptop, mouse, remote, keyboard, cell phone, book, clock, scissors, teddy bear

## Command Line Options

```bash
python3 main_roboeyes.py --ai-camera [OPTIONS]

Options:
  --yolo-confidence FLOAT   Confidence threshold (0.0-1.0, default: 0.5)
```

## Examples

```bash
# รัน AI Camera mode แบบปกติ
python3 main_roboeyes.py --ai-camera

# ปรับ confidence สูงขึ้น (แม่นยำกว่า)
python3 main_roboeyes.py --ai-camera --yolo-confidence 0.7

# รัน background + ดู log
nohup python3 main_roboeyes.py --ai-camera > /tmp/ai_camera.log 2>&1 &
tail -f /tmp/ai_camera.log
```

## Display Output

### GC9A01A (Round LCD 240x240)

```
    ┌─────────────────┐
    │                 │
    │   [Robot Eyes]  │
    │                 │
    ├─────────────────┤
    │ TRACK: person   │
    │ cat             │
    └─────────────────┘
```

### HDMI (Camera View 640x480)

แสดง:
- ภาพจากกล้องพร้อม bounding boxes
- Mode ปัจจุบัน (DETECT/TRACK)
- FPS
- รายการวัตถุที่ตรวจจับได้

## Troubleshooting AI Camera Mode

### Camera Timeout Error
```
ERROR Camera frontend has timed out!
Please check that your camera sensor connector is attached securely.
```

**แก้ไข:**
1. ปิด Pi และเช็คสาย CSI
2. ตรวจสอบว่า AI Camera ติดตั้งถูกต้อง
3. ลองรีบูต

### YOLO Not Available
```bash
# ติดตั้ง modlib
cd /home/pi/numbot
source env/bin/activate
pip install git+https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library.git
```

### Voice Not Working
```bash
# ตรวจสอบ microphone
arecord -l

# ทดสอบบันทึกเสียง
arecord -D plughw:2,0 -d 3 -f S16_LE -c 1 -r 16000 test.wav
aplay test.wav
```
