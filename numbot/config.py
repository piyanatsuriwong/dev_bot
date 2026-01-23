#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NumBot Configuration - Robot Eye Tracker
Based on PROJECT_REQUIREMENTS.md v3.0
"""

# =============================================================================
# Display Mode: "pygame", "gc9a01a", "st7735s"
# =============================================================================
DISPLAY_MODE = "st7735s"  # ST7735S as default display

# =============================================================================
# ST7735S SPI Configuration (Rectangular LCD 160x128 Landscape)
# =============================================================================
ST7735_WIDTH = 160
ST7735_HEIGHT = 128
# ST7735S has 132x162 memory but 128x160 display - need offset to avoid garbage pixels
# For landscape mode with MADCTL 0x68: X maps to rows, Y maps to columns
ST7735_OFFSET_X = 1   # Column offset
ST7735_OFFSET_Y = 2   # Row offset (smaller value for landscape)

ST7735_SPI_PORT = 1      # SPI1
ST7735_SPI_CS = 0        # CS0
ST7735_DC_PIN = 6        # GPIO 6
ST7735_RST_PIN = 13      # GPIO 13
ST7735_BL_PIN = 5        # GPIO 5 (Pin 29)
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
    EYE_WIDTH = 90
    EYE_HEIGHT = 90
    EYE_SPACING = 20
    EYE_RADIUS = 20
elif DISPLAY_MODE == "st7735s":
    SCREEN_WIDTH = ST7735_WIDTH
    SCREEN_HEIGHT = ST7735_HEIGHT
    EYE_WIDTH = 50
    EYE_HEIGHT = 60
    EYE_SPACING = 10
    EYE_RADIUS = 10
else:  # pygame
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 480
    EYE_WIDTH = 130
    EYE_HEIGHT = 190
    EYE_SPACING = 30
    EYE_RADIUS = 20

FPS = 30
DISPLAY_FPS = 30  # Robot display refresh rate (increased for smoother animation)
CAPTION = "NumBot Eye Tracker v3.0"

# Eye movement smoothing (0.0 = no movement, 1.0 = instant)
# Lower = smoother but slower, Higher = faster but more jerky
EYE_SMOOTHING = 0.15  # Smooth eye tracking

# =============================================================================
# Colors
# =============================================================================
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_UZI_PURPLE = (170, 50, 255)
COLOR_GLOW = (100, 30, 150)

# Eye colors (Uzi Doorman style)
EYE_BG_COLOR = COLOR_BLACK
EYE_FG_COLOR = COLOR_WHITE

# =============================================================================
# Operating Modes
# =============================================================================
MODE_HAND = "hand"      # Hand tracking with IMX708 + MediaPipe
MODE_AI = "ai"          # YOLO detection with IMX500
MODE_DEMO = "demo"      # Demo mode (no camera)

DEFAULT_MODE = MODE_DEMO

# =============================================================================
# Camera Configuration
# =============================================================================
# IMX708 (HAND Mode) - CSI port 1
# Using 1920x1080 for maximum FOV (Field of View)
CAMERA_IMX708_NUM = 1
CAMERA_IMX708_WIDTH = 1920
CAMERA_IMX708_HEIGHT = 1080
CAMERA_IMX708_FPS = 30

# IMX500 (AI Mode) - CSI port 0
CAMERA_IMX500_NUM = 0
CAMERA_IMX500_WIDTH = 640
CAMERA_IMX500_HEIGHT = 480
CAMERA_IMX500_FPS = 30

# =============================================================================
# Hand Tracking Configuration (MediaPipe)
# =============================================================================
MEDIAPIPE_MAX_HANDS = 1
MEDIAPIPE_DETECTION_CONFIDENCE = 0.7    # High accuracy detection (0.5 -> 0.7)
MEDIAPIPE_TRACKING_CONFIDENCE = 0.7     # High accuracy tracking (0.5 -> 0.7)
MEDIAPIPE_MODEL_COMPLEXITY = 0          # Lite model (0=lite, 1=full, 2=heavy) - fastest & stable for Pi5

# Finger count to mood mapping
# 0 (Fist) -> ANGRY, 1 -> CURIOUS, 2 -> HAPPY, 3 -> TIRED, 4 -> SCARY, 5 (Open) -> DEFAULT
FINGER_MOOD_MAP = {
    0: "ANGRY",
    1: "CURIOUS",
    2: "HAPPY",
    3: "TIRED",
    4: "SCARY",
    5: "DEFAULT"
}

# =============================================================================
# YOLO Configuration
# =============================================================================
YOLO_CONFIDENCE_THRESHOLD = 0.5
YOLO_DEFAULT_TARGET = "person"
YOLO_MODEL = "YOLO11n"  # Options: "YOLO11n", "YOLOv8n"

# =============================================================================
# Servo Configuration (PCA9685)
# =============================================================================
SERVO_ENABLED = True
SERVO_I2C_ADDRESS = 0x40
SERVO_I2C_BUS = 1
SERVO_FREQUENCY = 50  # 50 Hz for servos

# Pan servo (Channel 8 on PCA9685)
SERVO_PAN_CHANNEL = 8
SERVO_PAN_MIN = 0       # Min angle degrees
SERVO_PAN_MAX = 180     # Max angle degrees
SERVO_PAN_CENTER = 90   # Center position

# Tilt servo (Channel 9 on PCA9685)
SERVO_TILT_CHANNEL = 9
SERVO_TILT_MIN = 30     # Min angle (look down)
SERVO_TILT_MAX = 150    # Max angle (look up)
SERVO_TILT_CENTER = 90  # Center position

# Servo smoothing (0.0 = instant, 1.0 = very smooth)
SERVO_SMOOTHING = 0.85

# =============================================================================
# HDMI Display Configuration
# =============================================================================
HDMI_WIDTH = 1280
HDMI_HEIGHT = 480
HDMI_PANEL_WIDTH = 640
HDMI_ENABLED = True

# =============================================================================
# Audio Configuration
# =============================================================================
AUDIO_ENABLED = False
SOUND_PATH = "assets/sounds/"

# =============================================================================
# Voice Control (Disabled by default)
# =============================================================================
VOICE_ENABLED = False

# =============================================================================
# Performance Targets
# =============================================================================
TARGET_HAND_LATENCY_MS = 50     # < 50ms for hand tracking
TARGET_YOLO_LATENCY_MS = 77     # ~77ms for YOLO inference
TARGET_CPU_USAGE = 60           # < 60%
TARGET_MEMORY_MB = 2048         # < 2GB

# =============================================================================
# New Display Features (v4.0)
# =============================================================================
USE_NEW_DISPLAY = False          # Use new DisplayRenderer for ST7735S
DUAL_CAMERA_MODE = True          # Run both cameras simultaneously

# Text rendering settings
TEXT_FONT_PATH = None            # Path to custom font (None = system font)
TEXT_SHOW_ICONS = True           # Show emoji-like icons in detection labels
TEXT_SHOW_CONFIDENCE = True      # Show confidence bars in detection labels
TEXT_MAX_DETECTIONS = 2          # Max detections to show on ST7735S info panel

# Mode manager settings
MODE_AUTO_HAND_PRIORITY = True   # Hand detection takes priority in auto mode
MODE_SWITCH_ANIMATION = False    # Show animation during mode switch

# Default track targets for cycling
TRACK_TARGETS = ['person', 'cat', 'dog', 'bird', 'car', 'bottle', 'cup']
