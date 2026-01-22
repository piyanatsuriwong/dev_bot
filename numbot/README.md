# NumBot - Robot Eye Tracker

Robot eye tracking system for Raspberry Pi 5 with ST7735S/GC9A01A display.

## Features

- **Hand Tracking Mode**: Track hands using MediaPipe or OpenCV
- **AI Mode**: Object detection using IMX500 AI Camera with YOLO
- **Demo Mode**: Animated eyes without camera
- **Pan-Tilt Servo Control**: Track objects or hands with camera movement
- **Multiple Display Support**: ST7735S (160x128) or GC9A01A (240x240)

## Hardware Requirements

- Raspberry Pi 5
- ST7735S LCD (160x128) or GC9A01A LCD (240x240)
- IMX708 Camera (for hand tracking)
- IMX500 AI Camera (for object detection)
- PCA9685 PWM Driver (for servos)
- Pan-Tilt Servo Kit

## Pin Connections

### ST7735S Display (SPI1)
| Pin | GPIO | Description |
|-----|------|-------------|
| DC | GPIO 6 | Data/Command |
| RST | GPIO 13 | Reset |
| CS | GPIO 8 | Chip Select |
| MOSI | GPIO 10 | SPI Data |
| SCLK | GPIO 11 | SPI Clock |

### PCA9685 Servo Driver (I2C)
| Pin | Description |
|-----|-------------|
| SDA | GPIO 2 (Pin 3) |
| SCL | GPIO 3 (Pin 5) |
| VCC | 3.3V or 5V |
| GND | Ground |

Servos connected to PCA9685:
- Channel 8: Pan servo
- Channel 9: Tilt servo

## Installation

### 1. Create Virtual Environment
```bash
cd /home/pi/numbot
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install pygame opencv-python numpy pillow spidev lgpio
pip install mediapipe  # For hand tracking
```

### 3. Install AI Camera Support (Optional)
```bash
pip install git+https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library.git
```

### 4. Enable SPI
```bash
sudo raspi-config nonint do_spi 0
```

## Usage

### Hand Tracking Mode (Default)
```bash
python3 main_roboeyes.py --mode hand
```

### AI Object Detection Mode
```bash
python3 main_roboeyes.py --mode ai
```

### Demo Mode (No Camera)
```bash
python3 main_roboeyes.py --mode demo
```

### Options
| Option | Description |
|--------|-------------|
| `--mode hand` | Hand tracking mode (default) |
| `--mode ai` | AI object detection mode |
| `--mode demo` | Demo mode without camera |
| `--camera N` | Camera index (default: 0) |
| `--no-view` | Hide camera view on HDMI |

## Keyboard Controls

| Key | Action |
|-----|--------|
| ESC | Exit program |
| SPACE | Random mood change |
| D | Switch to DETECT mode (AI mode) |
| T | Switch to TRACK mode (AI mode) |

## Gesture Controls (Hand Mode)

| Fingers | Eye Mood |
|---------|----------|
| 0 (Fist) | ANGRY |
| 1 | CURIOUS |
| 2 (Peace) | HAPPY |
| 3 | TIRED |
| 4 | SCARY |
| 5 (Open) | DEFAULT |

## Configuration

Edit `config.py` to customize:

```python
# Display mode: "st7735s" or "gc9a01a"
DISPLAY_MODE = "st7735s"

# ST7735S settings
ST7735_WIDTH = 160
ST7735_HEIGHT = 128
ST7735_SPI_PORT = 1
ST7735_DC_PIN = 6
ST7735_RST_PIN = 13

# Servo settings
SERVO_I2C_ADDRESS = 0x40
SERVO_PAN_CHANNEL = 0
SERVO_TILT_CHANNEL = 1
```

## Project Structure

```
numbot/
├── main_roboeyes.py      # Main application
├── config.py             # Configuration
├── roboeyes.py           # Eye animation engine
├── fbutil.py             # FrameBuffer utilities
├── st7735_display.py     # ST7735S display driver
├── gc9a01a_display.py    # GC9A01A display driver
├── pi_camera.py          # Camera wrapper
├── yolo_tracker.py       # YOLO object detection
├── servo_controller.py   # Servo control
├── PCA9685.py            # PCA9685 PWM driver
├── assets/
│   ├── eyes/             # Eye images
│   ├── data/             # Animation data
│   └── sounds/           # Sound effects
└── README.md
```

## Troubleshooting

### Display not working
1. Check SPI is enabled: `ls /dev/spidev*`
2. Verify pin connections
3. Try demo mode first: `python3 main_roboeyes.py --mode demo`

### Camera not found
1. List cameras: `ls /dev/video*`
2. Check camera: `rpicam-hello --list-cameras`
3. Try different camera index: `--camera 1`

### Servo not moving
1. Check I2C is enabled: `sudo i2cdetect -y 1`
2. Verify PCA9685 address (default 0x40)
3. Check power supply to servo

### MediaPipe not working
```bash
pip install mediapipe --upgrade
```
If still failing, use OpenCV fallback (automatic).

## License

This project uses code from:
- FluxGarage RoboEyes (GPL)
- MicroPython RoboEyes port by Meurisse Dominique
