#!/bin/bash
# Fix corrupted config.py on Raspberry Pi

echo "Fixing config.py on Raspberry Pi..."

# Remove corrupted file and create new one
cat > ~/hand-eye-tracker/config.py << 'EOF'
# config.py

# =============================================================================
# Display Mode: "pygame" (HDMI/Monitor) or "gc9a01a" (Round LCD SPI)
# =============================================================================
DISPLAY_MODE = "gc9a01a"  # เปลี่ยนเป็น "pygame" สำหรับจอปกติ

# =============================================================================
# GC9A01A SPI Configuration (Round LCD 240x240)
# =============================================================================
GC9A01A_WIDTH = 240
GC9A01A_HEIGHT = 240
GC9A01A_SPI_PORT = 0      # SPI0 (GPIO 10=MOSI, 11=SCLK, 8=CE0)
GC9A01A_SPI_CS = 0        # CE0 (GPIO 8)
GC9A01A_DC_PIN = 25       # Data/Command pin
GC9A01A_RST_PIN = 27      # Reset pin
GC9A01A_BL_PIN = 18       # Backlight pin (PWM)
GC9A01A_SPI_SPEED = 62500000  # 62.5 MHz

# =============================================================================
# Screen Settings (based on display mode)
# =============================================================================
if DISPLAY_MODE == "gc9a01a":
    SCREEN_WIDTH = GC9A01A_WIDTH
    SCREEN_HEIGHT = GC9A01A_HEIGHT
    # ปรับขนาดตาให้พอดี 2 ตาบนจอกลม 240x240 (ลด 50% อีก)
    EYE_WIDTH = 17
    EYE_HEIGHT = 23
    SINGLE_EYE_MODE = False  # แสดงสองตาบนจอกลม
else:
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 480
    EYE_WIDTH = 130
    EYE_HEIGHT = 190
    SINGLE_EYE_MODE = False  # แสดงสองตาสำหรับจอปกติ

FPS = 60
CAPTION = "Uzi Doorman Visor OS v3.0"

# =============================================================================
# Colors
# =============================================================================
COLOR_BLACK = (0, 0, 0)        # สีดำสำหรับ GC9A01A
COLOR_UZI_PURPLE = (170, 50, 255)
COLOR_GLOW = (100, 30, 150)    # สีเรืองแสง

# =============================================================================
# Eye Animation Settings
# =============================================================================
BLINK_INTERVAL_MIN = 2.0  # วินาที
BLINK_INTERVAL_MAX = 6.0
GAZE_SMOOTHING = 0.15
GAZE_RANGE = 1.5
EOF

echo "config.py has been fixed!"
echo "Testing import..."
python3 -c "import config; print('✓ config.py is valid')"
