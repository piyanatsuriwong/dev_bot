# config.py

# =============================================================================
# Display Mode: "pygame" (HDMI/Monitor) or "gc9a01a" (Round LCD SPI)
# =============================================================================
DISPLAY_MODE = "gc9a01a"  # à¹€à¸›à¸¥à¸µà¹ˆà¸¢à¸™à¹€à¸›à¹‡à¸™ "pygame" à¸ªà¸³à¸«à¸£à¸±à¸šà¸ˆà¸­à¸›à¸à¸•à¸´

# =============================================================================
# GC9A01A SPI Configuration (Round LCD 240x240)
# =============================================================================
GC9A01A_WIDTH = 240
GC9A01A_HEIGHT = 240
GC9A01A_SPI_PORT = 1      # SPI1 (GPIO 20=MOSI, 21=SCLK, 16=CS)
GC9A01A_SPI_CS = 0        # CS0 remapped to GPIO 16 / Pin 36
GC9A01A_DC_PIN = 6        # Data/Command pin (GPIO 6 / Pin 31)
GC9A01A_RST_PIN = 13      # Reset pin (GPIO 13 / Pin 33)
GC9A01A_BL_PIN = None     # Backlight pin (not connected)
GC9A01A_SPI_SPEED = 40000000  # 40 MHz (stable for SPI1)

# =============================================================================
# Screen Settings (based on display mode)
# =============================================================================
if DISPLAY_MODE == "gc9a01a":
    SCREEN_WIDTH = GC9A01A_WIDTH
    SCREEN_HEIGHT = GC9A01A_HEIGHT
    # à¸›à¸£à¸±à¸šà¸‚à¸™à¸²à¸”à¸•à¸²à¹ƒà¸«à¹‰à¸žà¸­à¸”à¸µ 2 à¸•à¸²à¸šà¸™à¸ˆà¸­à¸à¸¥à¸¡ 240x240 (à¸¥à¸” 50% à¸­à¸µà¸)
    EYE_WIDTH = 17
    EYE_HEIGHT = 23
    SINGLE_EYE_MODE = False  # à¹à¸ªà¸”à¸‡à¸ªà¸­à¸‡à¸•à¸²à¸šà¸™à¸ˆà¸­à¸à¸¥à¸¡
else:
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 480
    EYE_WIDTH = 130
    EYE_HEIGHT = 190
    SINGLE_EYE_MODE = False  # à¹à¸ªà¸”à¸‡à¸ªà¸­à¸‡à¸•à¸²à¸ªà¸³à¸«à¸£à¸±à¸šà¸ˆà¸­à¸›à¸à¸•à¸´

FPS = 60
CAPTION = "Uzi Doorman Visor OS v3.0"

# =============================================================================
# Colors
# =============================================================================
COLOR_BLACK = (0, 0, 0)        # à¸ªà¸µà¸”à¸³à¸ªà¸³à¸«à¸£à¸±à¸š GC9A01A
COLOR_UZI_PURPLE = (170, 50, 255)
COLOR_GLOW = (100, 30, 150)    # à¸ªà¸µà¹€à¸£à¸·à¸­à¸‡à¹à¸ªà¸‡

# =============================================================================
# Eye Animation Settings
# =============================================================================
BLINK_INTERVAL_MIN = 2.0  # à¸§à¸´à¸™à¸²à¸—à¸µ
BLINK_INTERVAL_MAX = 6.0
GAZE_SMOOTHING = 0.15
GAZE_RANGE = 1.5
