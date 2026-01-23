# 📊 สรุป Device และการเชื่อมต่อทั้งหมด - main_roboeyes.py

## 🎯 ภาพรวมระบบ

**main_roboeyes.py** เป็น Application หลักที่รองรับ **3 โหมด**:
1. **Hand Mode** - ติดตามมือด้วย USB Webcam
2. **AI Mode** - Object Detection ด้วย IMX500 AI Camera
3. **Demo Mode** - แสดงตาแบบ idle (ไม่ใช้กล้อง)

---

## 🔌 รายการ Hardware/Devices ทั้งหมด

### 1️⃣ **Display (จอแสดงผล)**
**ตำแหน่ง:** `RoboEyesApp.__init__` (บรรทัด 387-395)

| รายการ | Interface | Pin/Connection | ขนาด | สถานะ |
|--------|-----------|----------------|------|--------|
| **ST7735S** | SPI1 | CS0, DC=6, RST=13, BLK=5 | 160x128 | ✅ Active (Default) |
| **GC9A01A** | SPI1 | CS0, DC=GPIO6, RST=GPIO13 | 240x240 (Round) | 🔄 Alternative |
| **Pygame Window** | HDMI | - | 800x480 | 🔄 Fallback |

**การตั้งค่า:**
```python
# config.py
DISPLAY_MODE = "st7735s"  # or "gc9a01a" or "pygame"
```

**การเชื่อมต่อ:**
```python
if config.DISPLAY_MODE in ["gc9a01a", "st7735s"]:
    self.display = create_display()  # Hardware SPI Display
    self.screen = pygame.Surface(...)
else:
    self.screen = pygame.display.set_mode(...)  # HDMI/Software
```

---

### 2️⃣ **Primary Camera (กล้องหลัก)**
**ตำแหน่ง:** Depends on mode

#### Mode: HAND (Hand Tracking)
| รายการ | Interface | Library | Resolution | สถานะ |
|--------|-----------|---------|-----------|--------|
| **IMX708 HQ Camera** | CSI-2 | Picamera2 | 1280x720 | ✅ Primary |
| **Hand Tracker** | - | MediaPipe (preferred) | - | ✅ Active |
| **Hand Tracker** | - | OpenCV Skin Detection | - | 🔄 Fallback |

**การเชื่อมต่อ:**
```python
# บรรทัด 436-440
# Uses Picamera2 for IMX708
picam = Picamera2(camera_num=0)  # IMX708
if use_mediapipe and MEDIAPIPE_AVAILABLE:
    self.hand_tracker = HandTrackerMediaPipe(camera_id=0)
else:
    self.hand_tracker = HandTrackerOpenCV(camera_id=0)
```

**Device Path:** `/dev/video0` (IMX708 CSI camera)


#### Mode: AI (Object Detection)
| รายการ | Interface | Library | Model | สถานะ |
|--------|-----------|---------|-------|--------|
| **IMX500 AI Camera** | CSI-2 | modlib `AiCamera` | YOLO11n / YOLOv8n | ✅ Primary |
| **YOLO Tracker** | - | yolo_tracker | - | ✅ Active |

**การเชื่อมต่อ:**
```python
# บรรทัด 444-448
if YOLO_AVAILABLE:
    self.yolo_tracker = create_yolo_tracker(confidence_threshold=0.5)
    self.yolo_tracker.start()
```

**Device Path:** `/dev/video1` (usually IMX500)
**Camera Index:** Auto-detected (tries [1, 0, 2, 3])

---

### 3️⃣ **Secondary Camera (กล้องรอง - สำหรับ HDMI View)**
**ตำแหน่ง:** `RoboEyesApp.__init__` (บรรทัด 459-497)

| Mode | Camera | Interface | Library | Resolution | สถานะ |
|------|--------|-----------|---------|-----------|--------|
| **HAND** | None | - | - | - | ❌ Disabled (ตามที่ User ขอ) |
| **AI** | None | - | - | - | ❌ Disabled (ใช้ IMX500 เดียว) |

**หมายเหตุ:**
- **HAND Mode:** ไม่มี secondary camera
- **AI Mode:** ใช้เฉพาะ IMX500 สำหรับทั้ง YOLO detection และ preview
- IMX708 **ไม่ได้ถูกใช้** ในโหมดนี้

**Device Path:** N/A (ไม่มี secondary camera)

---

### 4️⃣ **HDMI Display (Camera View)**
**ตำแหน่ง:** `RoboEyesApp.__init__` (บรรทัด 457-497)

| รายการ | Interface | Resolution | สถานะ |
|--------|-----------|-----------|--------|
| **Dual Camera Screen** | HDMI | 1280x480 (2 panels) | ✅ Active (if `show_camera=True`) |

**Layout:**
```
┌─────────────┬─────────────┐
│  Left (640) │ Right (640) │
│             │             │
│   IMX708    │   IMX500    │
└─────────────┴─────────────┘
     HAND Mode      AI Mode
```

**หมายเหตุ:**
- **HAND Mode:** Left panel = IMX708 (Hand tracking), Right panel = Empty
- **AI Mode:** Left panel = Empty, Right panel = IMX500 (YOLO + Preview)



**การเชื่อมต่อ:**
```python
# บรรทัด 459-461
if self.show_camera and self.mode != "demo":
    self.camera_screen = pygame.display.set_mode((1280, 480))
    pygame.display.set_caption("Dual Camera View")
```

---

### 5️⃣ **Servo Controller (Pan-Tilt)**
**ตำแหน่ง:** `RoboEyesApp.__init__` (บรรทัด 452)

| รายการ | Interface | Chip | Channels | สถานะ |
|--------|-----------|------|----------|--------|
| **PCA9685** | I2C | PCA9685 PWM Driver | 16 channels | ✅ Active |

**การเชื่อมต่อ:**
```python
# บรรทัด 452
self.servo = ServoController()
```

**Servo Mapping:**
- **Channel 0:** Pan (horizontal rotation)
- **Channel 1:** Tilt (vertical rotation)

**I2C Address:** `0x40` (default)

---

### 6️⃣ **Audio (เสียง)**
**ตำแหน่ง:** Global functions (บรรทัด 65-90)

#### Sound Output
| รายการ | Interface | Library | สถานะ |
|--------|-----------|---------|--------|
| **Audio Output** | 3.5mm / HDMI | `ffplay` subprocess | ✅ Enabled |

**การเชื่อมต่อ:**
```python
# บรรทัด 87
subprocess.call(f'ffplay -nodisp -autoexit -loglevel quiet "{sound_file}"', shell=True)
```

**Sound Files Location:** `/home/piyanat/hand-eye-tracker/assets/sounds/`

#### Voice Input (Optional - Currently DISABLED)
| รายการ | Interface | Library | สถานะ |
|--------|-----------|---------|--------|
| **USB Microphone** | USB (plughw:2,0) | `arecord` + SpeechRecognition | ❌ Disabled |

**การเชื่อมต่อ (ถ้าเปิดใช้):**
```python
# บรรทัด 59
VOICE_AVAILABLE = False  # Force disabled
```

---

## 📊 Device Connection Summary Table

| # | Device | Interface | GPIO/I2C/USB | Library | Mode | Status |
|---|--------|-----------|--------------|---------|------|--------|
| 1 | ST7735S Display | SPI1 | DC=6, RST=13, CS0, BLK=5 | st7735_display | All | ✅ Active |
| 2 | GC9A01A Display (Alt) | SPI1 | DC=6, RST=13, CS0 | gc9a01a_display | All | 🔄 Optional |
| 3 | IMX708 HQ Camera | CSI-2 | /dev/video0 | Picamera2 | HAND | ✅ Active |
| 4 | IMX500 AI Camera | CSI-2 | /dev/video1 | modlib.AiCamera | AI | ✅ Active |
| 5 | PCA9685 Servo Driver | I2C | 0x40 | servo_controller | All | ✅ Active |
| 6 | HDMI Monitor (Eyes) | HDMI | - | Pygame | All | ✅ Primary |
| 7 | HDMI Monitor (Cameras) | HDMI | - | Pygame | HAND/AI | ✅ Secondary |
| 8 | Audio Output | 3.5mm/HDMI | - | ffplay | All | ✅ Enabled |
| 9 | USB Microphone | USB (hw:2,0) | - | arecord | All | ❌ Disabled |

**หมายเหตุ:** ระบบใช้เฉพาะ CSI cameras (IMX708 + IMX500) ไม่ใช้ USB Webcam



---

## 🔄 Data Flow Diagram

### HAND Mode:
```
┌─────────────┐
│ IMX708 (HQ) │ /dev/video0
└─────┬───────┘
      │ Picamera2 → MediaPipe
      ↓
┌─────────────────┐
│ Hand Tracker    │ Detect hand position
└─────┬───────────┘
      │ (x, y) normalized
      ├──→ RoboEyes (ตาขยับตาม)
      ├──→ Servo (กล้องหมุนตาม)
      └──→ ST7735S/GC9A01A Display
```


### AI Mode:
```
┌──────────────┐
│ IMX500 (AI)  │
│ /dev/video1  │
└──────┬───────┘
       │ YOLO Detection + Preview
       │ (Single camera for both)
       ↓
┌────────────────┐
│ YOLO Tracker   │ Objects detected
└──────┬─────────┘
       │ Object positions
       ├──→ RoboEyes (ตามองวัตถุ)
       ├──→ Servo (กล้องหมุนตาม)
       ├──→ ST7735S Display + Text Overlay
       └──→ HDMI View (Right Panel)
```


---

## ⚙️ Configuration Files

### config.py
```python
DISPLAY_MODE = "st7735s"  # "gc9a01a" | "st7735s" | "pygame"
SCREEN_WIDTH = 160  # or 240 for GC9A01A
SCREEN_HEIGHT = 128  # or 240 for GC9A01A
FPS = 60
```

### SPI Pin Configuration (ST7735S)
```python
ST7735_SPI_PORT = 1      # SPI1
ST7735_SPI_CS = 0        # CS0 (GPIO 8)
ST7735_DC_PIN = 6        # Data/Command
ST7735_RST_PIN = 13      # Reset
ST7735_BL_PIN = 5        # Backlight (GPIO 5)
ST7735_SPI_SPEED = 24000000  # 24 MHz
```

### I2C Configuration (Servo)
```python
# Default PCA9685 I2C address
I2C_ADDRESS = 0x40
I2C_BUS = 1  # /dev/i2c-1
```

---

## 🖥️ Command Line Usage

```bash
# HAND Mode (IMX708 CSI camera tracking)
python3 main_roboeyes.py --mode hand

# AI Mode (IMX500 YOLO detection)
python3 main_roboeyes.py --mode ai

# Demo Mode (no camera)
python3 main_roboeyes.py --mode demo

# Hide camera view
python3 main_roboeyes.py --mode hand --no-view
```

**หมายเหตุ:** ระบบใช้ CSI cameras เท่านั้น (IMX708 สำหรับ HAND, IMX500 สำหรับ AI)


---

## 🔧 Troubleshooting

### Camera Index Detection
```bash
# List all cameras
libcamera-hello --list-cameras

# Expected output:
# 0 : imx708 [4608x2592]  # HQ Camera
# 1 : imx500 [4056x3040]  # AI Camera

# Check USB cameras
v4l2-ctl --list-devices
```

### I2C Devices
```bash
# List I2C devices
sudo i2cdetect -y 1

# Expected: PCA9685 at 0x40
```

### SPI Devices
```bash
# Check SPI is enabled
ls -la /dev/spidev*

# Expected: /dev/spidev1.0 (SPI1, CS0)
```

---

## 📝 Notes

1. **HAND Mode:** Uses IMX708 (CSI-2) for hand tracking
2. **AI Mode:** Uses IMX500 (CSI-2) for YOLO detection + preview
3. **Camera Setup:** Both cameras active, no secondary camera needed
4. **USB Webcam:** **ไม่ได้ใช้งาน** - ใช้เฉพาะ CSI cameras
5. **YOLO Model:** Auto-selects YOLO11n (preferred) or YOLOv8n (fallback)
6. **Voice Control:** Currently **DISABLED** (line 59)
7. **Display:** ST7735S is current default (160x128 landscape)



---

**Last Updated:** 2026-01-22  
**File:** main_roboeyes.py (749 lines)
