# 📋 Project Requirements - Robot Eye Tracker

## 🎯 Project Overview

**Project Name:** Hand-Eye Tracker with RoboEyes  
**Platform:** Raspberry Pi 5  
**Purpose:** Interactive robot eyes that track hand movements and detect objects in real-time  
**Version:** 3.0  
**Last Updated:** 2026-01-22

---

## 1. System Requirements

### 1.1 Hardware Platform

| Component | Specification | Required | Notes |
|-----------|--------------|----------|-------|
| **Main Board** | Raspberry Pi 5 (4GB/8GB RAM) | ✅ Required | ARM64 architecture |
| **Operating System** | Raspberry Pi OS (64-bit) | ✅ Required | Bookworm or newer |
| **Storage** | microSD 32GB+ (Class 10) | ✅ Required | For OS and data |
| **Power Supply** | 5V 3A USB-C | ✅ Required | Official RPi PSU recommended |

### 1.2 Display Hardware

| Display Type | Resolution | Interface | Status | Use Case |
|--------------|-----------|-----------|--------|----------|
| **ST7735S** | 160x128 | SPI1 | ✅ Primary | Robot eyes (landscape) |
| **GC9A01A** | 240x240 | SPI1 | 🔄 Alternative | Robot eyes (round) |
| **HDMI Monitor** | 1280x720+ | HDMI | ✅ Required | Camera preview |

**Display Pinout (SPI1):**
- **DC Pin:** GPIO 6
- **RST Pin:** GPIO 13
- **CS:** CS0 (SPI1)
- **SPI Speed:** 24 MHz (ST7735S) / 40 MHz (GC9A01A)

### 1.3 Camera Hardware

| Camera | Sensor | Interface | Resolution | Frame Rate | Purpose |
|--------|--------|-----------|-----------|------------|---------|
| **IMX708** | 12MP HQ | CSI-2 | 1280x720 | 30 FPS | Hand tracking (HAND mode) |
| **IMX500** | 12MP AI | CSI-2 | 4056x3040 | 10-30 FPS | Object detection (AI mode) |

**Camera Ports:**
- **CAM0/CAM1:** 15-pin CSI-2 connectors on Raspberry Pi 5
- **IMX708:** /dev/video0 (Primary CSI port)
- **IMX500:** /dev/video1 (Secondary CSI port)

**Important:** USB webcams are NOT supported in this project

### 1.4 Servo Controller

| Component | Specification | Interface | Purpose |
|-----------|--------------|-----------|---------|
| **PCA9685** | 16-channel PWM | I2C (0x40) | Pan-Tilt camera control |

**Servo Channels:**
- Channel 0: Pan (horizontal rotation)
- Channel 1: Tilt (vertical rotation)

**I2C Configuration:**
- Bus: /dev/i2c-1
- Address: 0x40
- Frequency: 100 kHz

### 1.5 Audio (Optional)

| Component | Interface | Status | Purpose |
|-----------|-----------|--------|---------|
| **Audio Output** | 3.5mm / HDMI | ✅ Enabled | Robot sound effects |
| **USB Microphone** | USB (hw:2,0) | ❌ Disabled | Voice commands (future) |

---

## 2. Software Requirements

### 2.1 Operating System

```bash
# Raspberry Pi OS (64-bit) - Bookworm or newer
# Enable SPI, I2C, Camera
sudo raspi-config
# → Interface Options → Enable SPI, I2C, Camera
```

### 2.2 System Dependencies

#### Core Libraries
```bash
sudo apt update
sudo apt install -y \
    python3-pygame \
    python3-numpy \
    python3-opencv \
    python3-picamera2 \
    python3-pip \
    git \
    libcamera-apps
```

#### SPI Display Drivers
```bash
# For ST7735S / GC9A01A
sudo apt install -y \
    python3-rpi.gpio \
    python3-spidev
```

#### I2C Tools
```bash
# For PCA9685 servo controller
sudo apt install -y \
    python3-smbus \
    i2c-tools
```

#### Audio Tools (Optional)
```bash
# For sound effects
sudo apt install -y \
    ffmpeg \
    alsa-utils
```

### 2.3 Python Dependencies

#### Hand Tracking
```bash
# MediaPipe for hand detection (preferred)
pip3 install mediapipe==0.10.14

# Note: MediaPipe requires ARM64 pre-built wheel
# Download from: https://github.com/Melvinsajith/How-to-Install-Mediapipe-on-Raspberry-Pi-5
```

#### Camera Libraries
```bash
# Picamera2 for IMX708 (should be pre-installed)
sudo apt install python3-picamera2

# modlib for IMX500 AI Camera
pip3 install modlib --upgrade
```

#### Display & Graphics
```bash
# Pygame (should be pre-installed)
sudo apt install python3-pygame

# PIL for image processing
pip3 install Pillow
```

#### Servo Control
```bash
# Adafruit PCA9685 library
pip3 install adafruit-circuitpython-pca9685
```

#### Optional: Voice Control
```bash
# Currently disabled, for future use
pip3 install SpeechRecognition
sudo apt install flac
```

### 2.4 Configuration Files

#### Enable SPI & I2C
```bash
# /boot/firmware/config.txt
dtparam=spi=on
dtparam=i2c_arm=on
```

#### Camera Configuration
```bash
# Check cameras are detected
libcamera-hello --list-cameras

# Expected output:
# 0 : imx708 [4608x2592] (/base/soc/i2c0mux/i2c@1/imx708@1a)
# 1 : imx500 [4056x3040] (/base/soc/i2c0mux/i2c@0/imx500@1f)
```

---

## 3. Functional Requirements

### 3.1 Operating Modes

The system MUST support 3 distinct operating modes:

#### Mode 1: HAND (Hand Tracking)
- **Input:** IMX708 camera feed (1280x720 @ 30 FPS)
- **Processing:** 
  - MediaPipe hand detection (preferred)
  - OpenCV skin detection (fallback)
- **Output:**
  - Robot eyes track hand position
  - Pan-Tilt servo actively tracks and centers the hand in the frame
  - Display hand tracking info on HDMI (visible MediaPipe landmarks)
  - Eye mood changes based on number of extended fingers
- **Performance:** < 50ms latency from hand movement to eye response

#### Mode 2: AI (Object Detection)
- **Input:** IMX500 camera feed (configurable resolution)
- **Processing:**
  - YOLO11n object detection (preferred)
  - YOLOv8n (fallback)
  - 80 COCO classes
- **Output:**
  - Robot eyes track detected objects
  - Pan-Tilt servo follows primary object
  - Display detection results on HDMI
  - Overlay object names on robot display
- **Performance:** ~13 FPS inference, ~77ms latency

#### Mode 3: DEMO
- **Input:** None (idle mode)
- **Processing:** Idle eye animations
- **Output:** Robot eyes with random movements
- **Purpose:** Testing display without cameras

### 3.2 Display Requirements

#### Robot Eyes Display (ST7735S/GC9A01A)
- **Resolution:** 160x128 (ST7735S) or 240x240 (GC9A01A)
- **Refresh Rate:** 20 FPS (throttled for performance)
- **Content:**
  - Animated robot eyes (RoboEyes engine)
  - Eye movements based on tracking
  - Mood/emotion display
  - Text overlay for status info (AI mode)
- **Color Scheme:** Customizable (default: purple/cyan for Uzi Doorman style)

#### HDMI Preview Display
- **Resolution:** 1280x480 (dual panel layout)
- **Layout:**
  ```
  ┌─────────────┬─────────────┐
  │  Left (640) │ Right (640) │
  │             │             │
  │   IMX708    │   IMX500    │
  └─────────────┴─────────────┘
       HAND Mode      AI Mode
  ```
- **Refresh Rate:** 30 FPS
- **Content:**
  - Camera feed with overlays
  - MediaPipe Hand Landmarks overlay (skeleton/joints) (HAND mode)
  - Detection bounding boxes (AI mode)
  - FPS counter
  - Mode indicator

### 3.3 Control Inputs

#### Keyboard Controls
| Key | Function | Mode |
|-----|----------|------|
| **ESC** | Exit application | All |
| **SPACE** | Random mood/emotion | All |
| **D** | Switch to DETECT mode | AI only |
| **T** | Switch to TRACK mode | AI only |

#### Voice Controls (Optional - Currently Disabled)
- "ติดตามมือ" / "track hand" → HAND mode
- "ตรวจจับ" / "detect" → AI DETECT mode
- "มีความสุข" / "happy" → Change mood to HAPPY
- "โกรธ" / "angry" → Change mood to ANGRY

### 3.4 Hand Gesture Controls (Finger Count)

| Fingers | Mood | description |
|---------|------|-------------|
| 0 (Fist) | ANGRY | Aggressive/Closed |
| 1 | CURIOUS | Looking/Pointing |
| 2 | HAPPY | Peace sign/Happy |
| 3 | TIRED | Low energy |
| 4 | SCARY | Intimidating |
| 5 (Open) | DEFAULT | Normal/Neutral |

### 3.4 Performance Requirements

| Metric | Target | Mode |
|--------|--------|------|
| **Hand Tracking Latency** | < 50ms | HAND |
| **YOLO Inference Time** | ~77ms | AI |
| **Display Refresh** | 20-30 FPS | All |
| **Camera Frame Rate** | 30 FPS | HAND |
| **YOLO Frame Rate** | 10-13 FPS | AI |
| **CPU Usage** | < 60% | All |
| **Memory Usage** | < 2GB | All |

---

## 4. Technical Specifications

### 4.1 Software Architecture

```
┌─────────────────────────────────────────┐
│         main_roboeyes.py                │
│         (Main Application)              │
└──────┬──────────────────────────────────┘
       │
       ├─→ config.py (Configuration)
       │
       ├─→ Display Layer
       │   ├─→ st7735_display.py (ST7735S driver)
       │   ├─→ gc9a01a_display.py (GC9A01A driver)
       │   └─→ roboeyes.py (Eye animation engine)
       │
       ├─→ Camera Layer
       │   ├─→ hand_tracker.py (IMX708 Hand Tracking)
       │   └─→ yolo_tracker.py (IMX500 YOLO Detection)
       │
       ├─→ Servo Layer
       │   └─→ servo_controller.py (PCA9685)
       │
       └─→ UI Layer
           └─→ TextOverlay (Status display)
```

### 4.2 Data Flow

#### HAND Mode Flow
```
IMX708 (CSI-2)
    ↓
hand_tracker.py (Picamera2)
    ↓
MediaPipe Hand Detection
    ↓
Normalize Coordinates (-1 to 1)
    ↓
    ├─→ RoboEyes (Eye position)
    ├─→ ServoController (Pan-Tilt)
    └─→ HDMI Display (Preview with Landmarks)
```

#### AI Mode Flow
```
IMX500 (CSI-2)
    ↓
modlib.AiCamera
    ↓
YOLO11n/YOLOv8n (80 classes)
    ↓
Filter by confidence (>0.5)
    ↓
Track Primary Object
    ↓
    ├─→ RoboEyes (Eye position)
    ├─→ ServoController (Pan-Tilt)
    ├─→ TextOverlay (Object names)
    └─→ HDMI Display (Bounding boxes)
```

### 4.3 Camera Configuration

#### IMX708 (HAND Mode)
```python
# hand_tracker.py
from picamera2 import Picamera2

picam = Picamera2(camera_num=0)
config = picam.create_preview_configuration(
    main={"size": (1280, 720), "format": "RGB888"}
)
picam.configure(config)
picam.start()

# Frame capture
frame = picam.capture_array("main")  # Already RGB888
```

#### IMX500 (AI Mode)
```python
from modlib.devices import AiCamera
from modlib.models.zoo import YOLO11n  # or YOLOv8n

model = YOLO11n()  # Auto-select best available
device = AiCamera(frame_rate=30, num=1)  # /dev/video1
device.deploy(model)

# Frame capture with detection
with device as stream:
    for frame in stream:
        detections = frame.detections[
            frame.detections.confidence > 0.5
        ]
```

### 4.4 Display Configuration

#### ST7735S (SPI1)
```python
# config.py
DISPLAY_MODE = "st7735s"
ST7735_WIDTH = 160
ST7735_HEIGHT = 128
ST7735_SPI_PORT = 1
ST7735_SPI_CS = 0
ST7735_DC_PIN = 6
ST7735_RST_PIN = 13
ST7735_SPI_SPEED = 24000000  # 24 MHz
```

#### GC9A01A (SPI1)
```python
# config.py
DISPLAY_MODE = "gc9a01a"
GC9A01A_WIDTH = 240
GC9A01A_HEIGHT = 240
GC9A01A_SPI_PORT = 1
GC9A01A_SPI_CS = 0
GC9A01A_DC_PIN = 6
GC9A01A_RST_PIN = 13
GC9A01A_SPI_SPEED = 40000000  # 40 MHz
```

### 4.5 Servo Configuration

```python
from adafruit_pca9685 import PCA9685

# I2C setup
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50  # 50 Hz for servos

# Pan servo (Channel 0)
pan_channel = 0
pan_min = 150  # Min pulse width
pan_max = 600  # Max pulse width

# Tilt servo (Channel 1)
tilt_channel = 1
tilt_min = 150
tilt_max = 600
```

---

## 5. Implementation Guidelines

### 5.1 Project Structure

```
hand-eye-tracker/
├── main_roboeyes.py          # Main application
├── config.py                  # Configuration settings
├── roboeyes.py               # Eye animation engine
│
├── # Display Drivers
├── st7735_display.py         # ST7735S SPI display
├── gc9a01a_display.py        # GC9A01A SPI display
│
├── # Camera & Tracking
├── hand_tracker.py           # IMX708 Hand Tracking
├── yolo_tracker.py           # IMX500 YOLO detection
│
├── # Hardware Control
├── servo_controller.py       # PCA9685 servo control
│
├── # Documentation
├── README.md                 # User guide
├── DEVICE_CONNECTIONS.md     # Hardware wiring
├── CAMERA_MIGRATION_SUMMARY.md  # Camera setup
├── YOLO_MODEL_GUIDE.md       # YOLO model info
├── TROUBLESHOOTING.md        # Common issues
└── PROJECT_REQUIREMENTS.md   # This file
```

### 5.2 Initialization Sequence

```python
1. Load config.py
2. Initialize Pygame
3. Create Display (ST7735S/GC9A01A/HDMI)
4. Initialize RoboEyes engine
5. Based on mode:
   - HAND: Initialize IMX708 + HandTracker
   - AI: Initialize IMX500 + YOLO
   - DEMO: Skip camera init
6. Initialize ServoController (PCA9685)
7. Start main loop
```

### 5.3 Main Loop

```python
while running:
    # Handle events (keyboard, quit)
    for event in pygame.event.get():
        handle_event(event)
    
    # Update tracking
    if mode == "HAND":
        hand_tracker.update()
        position = hand_tracker.get_normalized_position()
    elif mode == "AI":
        position = yolo_tracker.get_normalized_position()
    
    # Update robot eyes
    robo.eyeLxNext = position.x
    robo.eyeLyNext = position.y
    robo.update()
    
    # Update servo
    servo.track_hand(position.x, position.y)
    
    # Update displays
    update_robot_display()    # ST7735S/GC9A01A @ 20 FPS
    update_camera_display()   # HDMI @ 30 FPS
    
    clock.tick(30)  # Main loop at 30 FPS
```

### 5.4 Error Handling

#### Camera Errors
- **Device Busy:** Cleanup zombie processes, retry with different indices
- **No Camera:** Fallback to DEMO mode
- **Low FPS:** Reduce resolution, lower refresh rate

#### Display Errors
- **SPI Failure:** Check wiring, verify SPI enabled
- **No HDMI:** Disable camera preview, use robot display only

#### Servo Errors
- **I2C Error:** Check PCA9685 address (0x40), verify I2C enabled
- **No Response:** Check power supply, servo connections

---

## 6. Testing Requirements

### 6.1 Unit Tests

- [ ] Display initialization (ST7735S, GC9A01A)
- [ ] Camera initialization (IMX708, IMX500)
- [ ] Hand detection accuracy
- [ ] YOLO detection accuracy
- [ ] Servo response time
- [ ] Text overlay rendering

### 6.2 Integration Tests

- [ ] HAND mode end-to-end
- [ ] AI mode end-to-end
- [ ] Mode switching
- [ ] Graceful shutdown
- [ ] Resource cleanup

### 6.3 Performance Tests

- [ ] FPS measurement (target: 30 FPS camera, 20 FPS display)
- [ ] Latency measurement (target: < 50ms HAND, ~77ms AI)
- [ ] CPU usage (target: < 60%)
- [ ] Memory usage (target: < 2GB)
- [ ] Thermal testing (target: < 80°C sustained)

### 6.4 Hardware Tests

```bash
# Test SPI display
python3 -c "from st7735_display import create_display; d=create_display(); print('OK')"

# Test IMX708
libcamera-hello --camera 0 -t 5000

# Test IMX500
python3 -c "from modlib.devices import AiCamera; AiCamera(num=1)"

# Test PCA9685
sudo i2cdetect -y 1  # Should show 0x40

# Test cameras list
libcamera-hello --list-cameras
```

---

## 7. Deployment

### 7.1 Installation Steps

```bash
# 1. Clone repository
git clone <repository-url>
cd hand-eye-tracker

# 2. Install system dependencies
sudo apt update
sudo apt install -y python3-pygame python3-numpy python3-opencv \
    python3-picamera2 libcamera-apps

# 3. Install Python packages
pip3 install mediapipe==0.10.14 modlib --upgrade

# 4. Enable SPI, I2C
sudo raspi-config
# Interface Options → SPI → Yes
# Interface Options → I2C → Yes

# 5. Verify cameras
libcamera-hello --list-cameras

# 6. Test run
python3 main_roboeyes.py --mode demo
```

### 7.2 Command Line Usage

```bash
# HAND Mode (IMX708 hand tracking)
python3 main_roboeyes.py --mode hand

# AI Mode (IMX500 YOLO detection)
python3 main_roboeyes.py --mode ai

# Demo Mode (no camera)
python3 main_roboeyes.py --mode demo

# Hide camera preview
python3 main_roboeyes.py --mode hand --no-view
```

---

## 8. Maintenance & Support

### 8.1 Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed guides on:
- Camera detection issues
- SPI display problems
- YOLO model loading errors
- Servo control failures

### 8.2 Updates

- **Raspberry Pi OS:** Keep updated via `sudo apt update && sudo apt upgrade`
- **Python packages:** Update via `pip3 install --upgrade <package>`
- **modlib:** Check for YOLO model updates regularly

### 8.3 Known Limitations

- USB webcams not supported (CSI cameras only)
- MediaPipe requires ARM64 pre-built wheel
- YOLO inference limited to ~13 FPS on CPU
- Voice control currently disabled

---

## 9. Future Enhancements

### 9.1 Planned Features

- [ ] Voice command support (Thai + English)
- [ ] Multiple object tracking (AI mode)
- [ ] Gesture recognition (hand mode)
- [ ] Face recognition integration
- [ ] Wi-Fi remote control
- [ ] Mobile app integration
- [ ] Cloud logging/analytics

### 9.2 Hardware Upgrades

- [ ] Neural accelerator (Coral TPU) for faster YOLO
- [ ] Higher resolution displays (320x240+)
- [ ] Additional servos for full head movement
- [ ] LED ring for visual feedback

---

## 10. References

- [Raspberry Pi Documentation](https://www.raspberrypi.com/documentation/)
- [Picamera2 Guide](https://github.com/raspberrypi/picamera2)
- [MediaPipe Documentation](https://google.github.io/mediapipe/)
- [YOLO11 Guide](https://docs.ultralytics.com/models/yolo11/)
- [modlib Repository](https://github.com/raspberrypi/picamera2)

---

**Document Version:** 1.0  
**Created:** 2026-01-22  
**Status:** ✅ Complete  
**Approved For:** New project implementation
