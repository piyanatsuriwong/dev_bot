# config.py
# Configuration for NumBot - Robot Eye Tracker

# =============================================================================
# Display Mode: "pygame", "gc9a01a", "st7735s"
# =============================================================================
DISPLAY_MODE = "st7735s"  # ST7735S as default display

# =============================================================================
# ST7735S SPI Configuration (Rectangular LCD 160x128 Landscape)
# =============================================================================
ST7735_WIDTH = 160
ST7735_HEIGHT = 128
ST7735_OFFSET_X = 0
ST7735_OFFSET_Y = 0

ST7735_SPI_PORT = 1      # SPI1
ST7735_SPI_CS = 0        # CS0
ST7735_DC_PIN = 6        # GPIO 6
ST7735_RST_PIN = 13      # GPIO 13
ST7735_BL_PIN = None     # Not connected
ST7735_SPI_SPEED = 24000000  # 24 MHz

# =============================================================================
# GC9A01A SPI Configuration (Round LCD 240x240)
# =============================================================================
GC9A01A_WIDTH = 240
GC9A01A_HEIGHT = 240
GC9A01A_SPI_PORT = 1
GC9A01A_SPI_CS = 0
GC9A01A_DC_PIN = 6
GC9A01A_RST_PIN = 13
GC9A01A_BL_PIN = None
GC9A01A_SPI_SPEED = 40000000  # 40 MHz

# =============================================================================
# Screen Settings (based on display mode)
# =============================================================================
if DISPLAY_MODE == "gc9a01a":
    SCREEN_WIDTH = GC9A01A_WIDTH
    SCREEN_HEIGHT = GC9A01A_HEIGHT
    EYE_WIDTH = 17
    EYE_HEIGHT = 23
    SINGLE_EYE_MODE = False
elif DISPLAY_MODE == "st7735s":
    SCREEN_WIDTH = ST7735_WIDTH
    SCREEN_HEIGHT = ST7735_HEIGHT
    EYE_WIDTH = 40
    EYE_HEIGHT = 50
    SINGLE_EYE_MODE = False
else:  # pygame
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 480
    EYE_WIDTH = 130
    EYE_HEIGHT = 190
    SINGLE_EYE_MODE = False

FPS = 60
CAPTION = "NumBot Eye Tracker v1.0"

# =============================================================================
# Colors
# =============================================================================
COLOR_BLACK = (0, 0, 0)
COLOR_UZI_PURPLE = (170, 50, 255)
COLOR_GLOW = (100, 30, 150)

# =============================================================================
# Eye Animation Settings
# =============================================================================
BLINK_INTERVAL_MIN = 2.0
BLINK_INTERVAL_MAX = 6.0
GAZE_SMOOTHING = 0.15
GAZE_RANGE = 1.5

# =============================================================================
# Camera Configuration
# =============================================================================
# IMX708 (HAND Mode) - /dev/video0
CAMERA_IMX708_NUM = 0
CAMERA_IMX708_WIDTH = 1280
CAMERA_IMX708_HEIGHT = 720
CAMERA_IMX708_FPS = 30

# IMX500 (AI Mode) - /dev/video1
CAMERA_IMX500_NUM = 1
CAMERA_IMX500_WIDTH = 640
CAMERA_IMX500_HEIGHT = 480
CAMERA_IMX500_FPS = 30

# =============================================================================
# YOLO Configuration
# =============================================================================
YOLO_CONFIDENCE_THRESHOLD = 0.5
YOLO_DEFAULT_TARGET = "person"

# =============================================================================
# Servo Configuration (PCA9685)
# =============================================================================
SERVO_I2C_ADDRESS = 0x40
SERVO_I2C_BUS = 1
SERVO_FREQUENCY = 50  # 50 Hz for servos

# Pan servo (Channel 0)
SERVO_PAN_CHANNEL = 0
SERVO_PAN_MIN = 150
SERVO_PAN_MAX = 600
SERVO_PAN_CENTER = 375

# Tilt servo (Channel 1)
SERVO_TILT_CHANNEL = 1
SERVO_TILT_MIN = 150
SERVO_TILT_MAX = 600
SERVO_TILT_CENTER = 375

# =============================================================================
# HDMI Display Configuration
# =============================================================================
HDMI_WIDTH = 1280
HDMI_HEIGHT = 480
HDMI_PANEL_WIDTH = 640

# =============================================================================
# Audio Configuration
# =============================================================================
AUDIO_ENABLED = True
SOUND_PATH = "assets/sounds/"

# =============================================================================
# Voice Control (Disabled by default)
# =============================================================================
VOICE_ENABLED = False
