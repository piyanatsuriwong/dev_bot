# gc9a01a_display.py
# GC9A01A Round LCD Display Driver for Raspberry Pi 5
# 240x240 pixel, SPI interface
# Uses lgpio for Pi 5 compatibility

import config

# Try lgpio first (Pi 5 compatible), then fall back to RPi.GPIO
GPIO_LIB = None
try:
    import spidev
    try:
        import lgpio
        GPIO_LIB = "lgpio"
        RPI_AVAILABLE = True
    except ImportError:
        try:
            import RPi.GPIO as GPIO
            GPIO_LIB = "rpigpio"
            RPI_AVAILABLE = True
        except ImportError:
            RPI_AVAILABLE = False
except ImportError:
    RPI_AVAILABLE = False

if not RPI_AVAILABLE:
    print("Warning: GPIO/spidev not available. Running in simulation mode.")
else:
    print(f"GPIO library: {GPIO_LIB}")

import time
import numpy as np
from PIL import Image

# GC9A01A Commands
CMD_NOP = 0x00
CMD_SWRESET = 0x01
CMD_SLPIN = 0x10
CMD_SLPOUT = 0x11
CMD_INVOFF = 0x20
CMD_INVON = 0x21
CMD_DISPOFF = 0x28
CMD_DISPON = 0x29
CMD_CASET = 0x2A
CMD_RASET = 0x2B
CMD_RAMWR = 0x2C
CMD_COLMOD = 0x3A
CMD_MADCTL = 0x36
CMD_TEON = 0x35
CMD_WRDISBV = 0x51


class GC9A01ADisplay:
    """
    GC9A01A Round LCD Display Driver
    - 240x240 pixels
    - 16-bit color (RGB565)
    - SPI interface
    """

    def __init__(self):
        self.width = config.GC9A01A_WIDTH
        self.height = config.GC9A01A_HEIGHT
        self.spi_port = config.GC9A01A_SPI_PORT
        self.spi_cs = config.GC9A01A_SPI_CS
        self.dc_pin = config.GC9A01A_DC_PIN
        self.rst_pin = config.GC9A01A_RST_PIN
        self.bl_pin = config.GC9A01A_BL_PIN
        self.spi_speed = config.GC9A01A_SPI_SPEED

        self.spi = None
        self.gpio_handle = None  # For lgpio
        self._buffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        if RPI_AVAILABLE:
            self._init_gpio()
            self._init_spi()
            self._init_display()
        else:
            print("GC9A01A: Simulation mode - no hardware")

    def _init_gpio(self):
        """Initialize GPIO pins"""
        if GPIO_LIB == "lgpio":
            # Use lgpio for Pi 5
            self.gpio_handle = lgpio.gpiochip_open(0)
            lgpio.gpio_claim_output(self.gpio_handle, self.dc_pin)
            lgpio.gpio_claim_output(self.gpio_handle, self.rst_pin)
            # Backlight pin is optional
            if self.bl_pin is not None:
                lgpio.gpio_claim_output(self.gpio_handle, self.bl_pin)
                lgpio.gpio_write(self.gpio_handle, self.bl_pin, 1)
        else:
            # Use RPi.GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(self.dc_pin, GPIO.OUT)
            GPIO.setup(self.rst_pin, GPIO.OUT)
            # Backlight pin is optional
            if self.bl_pin is not None:
                GPIO.setup(self.bl_pin, GPIO.OUT)
                GPIO.output(self.bl_pin, GPIO.HIGH)

    def _init_spi(self):
        """Initialize SPI interface"""
        self.spi = spidev.SpiDev()
        self.spi.open(self.spi_port, self.spi_cs)
        self.spi.max_speed_hz = self.spi_speed
        self.spi.mode = 0

    def _init_display(self):
        """Initialize GC9A01A display"""
        # Hardware reset
        self._reset()

        # Software reset
        self._write_cmd(CMD_SWRESET)
        time.sleep(0.15)

        # Sleep out
        self._write_cmd(CMD_SLPOUT)
        time.sleep(0.15)

        # Memory Data Access Control
        # MY=0, MX=1, MV=0, ML=0, BGR=1, MH=0
        # MX=1 (bit 6) mirrors horizontally to fix text display
        self._write_cmd(CMD_MADCTL)
        self._write_data([0x48])

        # Pixel Format: 16bit/pixel (RGB565)
        self._write_cmd(CMD_COLMOD)
        self._write_data([0x55])

        # Tearing Effect Line ON
        self._write_cmd(CMD_TEON)
        self._write_data([0x00])

        # GC9A01A specific initialization
        self._gc9a01a_init_sequence()

        # Display Inversion ON (often needed for correct colors)
        self._write_cmd(CMD_INVON)

        # Display ON
        self._write_cmd(CMD_DISPON)
        time.sleep(0.02)

        # Clear screen
        self.clear()

    def _gc9a01a_init_sequence(self):
        """GC9A01A specific initialization sequence"""
        init_cmds = [
            (0xEF, []),
            (0xEB, [0x14]),
            (0xFE, []),
            (0xEF, []),
            (0xEB, [0x14]),
            (0x84, [0x40]),
            (0x85, [0xFF]),
            (0x86, [0xFF]),
            (0x87, [0xFF]),
            (0x88, [0x0A]),
            (0x89, [0x21]),
            (0x8A, [0x00]),
            (0x8B, [0x80]),
            (0x8C, [0x01]),
            (0x8D, [0x01]),
            (0x8E, [0xFF]),
            (0x8F, [0xFF]),
            (0xB6, [0x00, 0x00]),
            (0x90, [0x08, 0x08, 0x08, 0x08]),
            (0xBD, [0x06]),
            (0xBC, [0x00]),
            (0xFF, [0x60, 0x01, 0x04]),
            (0xC3, [0x13]),
            (0xC4, [0x13]),
            (0xC9, [0x22]),
            (0xBE, [0x11]),
            (0xE1, [0x10, 0x0E]),
            (0xDF, [0x21, 0x0C, 0x02]),
            (0xF0, [0x45, 0x09, 0x08, 0x08, 0x26, 0x2A]),
            (0xF1, [0x43, 0x70, 0x72, 0x36, 0x37, 0x6F]),
            (0xF2, [0x45, 0x09, 0x08, 0x08, 0x26, 0x2A]),
            (0xF3, [0x43, 0x70, 0x72, 0x36, 0x37, 0x6F]),
            (0xED, [0x1B, 0x0B]),
            (0xAE, [0x77]),
            (0xCD, [0x63]),
            (0x70, [0x07, 0x07, 0x04, 0x0E, 0x0F, 0x09, 0x07, 0x08, 0x03]),
            (0xE8, [0x34]),
            (0x62, [0x18, 0x0D, 0x71, 0xED, 0x70, 0x70, 0x18, 0x0F, 0x71, 0xEF, 0x70, 0x70]),
            (0x63, [0x18, 0x11, 0x71, 0xF1, 0x70, 0x70, 0x18, 0x13, 0x71, 0xF3, 0x70, 0x70]),
            (0x64, [0x28, 0x29, 0xF1, 0x01, 0xF1, 0x00, 0x07]),
            (0x66, [0x3C, 0x00, 0xCD, 0x67, 0x45, 0x45, 0x10, 0x00, 0x00, 0x00]),
            (0x67, [0x00, 0x3C, 0x00, 0x00, 0x00, 0x01, 0x54, 0x10, 0x32, 0x98]),
            (0x74, [0x10, 0x85, 0x80, 0x00, 0x00, 0x4E, 0x00]),
            (0x98, [0x3E, 0x07]),
            (0x35, []),
            (0x21, []),
        ]

        for cmd, data in init_cmds:
            self._write_cmd(cmd)
            if data:
                self._write_data(data)

    def _reset(self):
        """Hardware reset"""
        if not RPI_AVAILABLE:
            return
        if GPIO_LIB == "lgpio":
            lgpio.gpio_write(self.gpio_handle, self.rst_pin, 1)
            time.sleep(0.01)
            lgpio.gpio_write(self.gpio_handle, self.rst_pin, 0)
            time.sleep(0.01)
            lgpio.gpio_write(self.gpio_handle, self.rst_pin, 1)
            time.sleep(0.12)
        else:
            GPIO.output(self.rst_pin, GPIO.HIGH)
            time.sleep(0.01)
            GPIO.output(self.rst_pin, GPIO.LOW)
            time.sleep(0.01)
            GPIO.output(self.rst_pin, GPIO.HIGH)
            time.sleep(0.12)

    def _write_cmd(self, cmd):
        """Write command to display"""
        if not RPI_AVAILABLE:
            return
        if GPIO_LIB == "lgpio":
            lgpio.gpio_write(self.gpio_handle, self.dc_pin, 0)
        else:
            GPIO.output(self.dc_pin, GPIO.LOW)
        self.spi.writebytes([cmd])

    def _write_data(self, data):
        """Write data to display"""
        if not RPI_AVAILABLE:
            return
        if GPIO_LIB == "lgpio":
            lgpio.gpio_write(self.gpio_handle, self.dc_pin, 1)
        else:
            GPIO.output(self.dc_pin, GPIO.HIGH)
        if isinstance(data, list):
            self.spi.writebytes(data)
        else:
            self.spi.writebytes2(data)

    def _set_window(self, x0, y0, x1, y1):
        """Set drawing window"""
        self._write_cmd(CMD_CASET)
        self._write_data([x0 >> 8, x0 & 0xFF, x1 >> 8, x1 & 0xFF])

        self._write_cmd(CMD_RASET)
        self._write_data([y0 >> 8, y0 & 0xFF, y1 >> 8, y1 & 0xFF])

        self._write_cmd(CMD_RAMWR)

    def clear(self, color=(0, 0, 0)):
        """Clear display with color"""
        try:
            self._buffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            self._buffer[:, :] = color
            self.update()
        except Exception:
            pass  # Ignore errors during cleanup

    def set_pixel(self, x, y, color):
        """Set single pixel"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self._buffer[y, x] = color

    def draw_from_surface(self, pygame_surface):
        """
        Draw from Pygame surface to display
        Converts Pygame surface to RGB565 and sends to display
        """
        # Convert Pygame surface to numpy array
        try:
            import pygame
            # Get raw pixel data
            raw_str = pygame.image.tostring(pygame_surface, 'RGB')
            img_array = np.frombuffer(raw_str, dtype=np.uint8)
            img_array = img_array.reshape((self.height, self.width, 3))
            self._buffer = img_array
            self.update()
        except Exception as e:
            print(f"Error drawing surface: {e}")

    def draw_from_pil(self, pil_image):
        """Draw from PIL Image"""
        if pil_image.size != (self.width, self.height):
            pil_image = pil_image.resize((self.width, self.height))
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        self._buffer = np.array(pil_image)
        self.update()

    def update(self):
        """Send buffer to display"""
        if not RPI_AVAILABLE:
            return

        # Convert RGB888 to RGB565
        rgb565 = self._rgb888_to_rgb565(self._buffer)

        # Set full screen window
        self._set_window(0, 0, self.width - 1, self.height - 1)

        # Send pixel data
        self._write_data(rgb565.tobytes())

    def _rgb888_to_rgb565(self, rgb888):
        """Convert RGB888 to RGB565"""
        r = (rgb888[:, :, 0] >> 3).astype(np.uint16)
        g = (rgb888[:, :, 1] >> 2).astype(np.uint16)
        b = (rgb888[:, :, 2] >> 3).astype(np.uint16)

        rgb565 = (r << 11) | (g << 5) | b

        # Convert to big-endian bytes
        rgb565_bytes = np.empty((self.height, self.width, 2), dtype=np.uint8)
        rgb565_bytes[:, :, 0] = (rgb565 >> 8).astype(np.uint8)
        rgb565_bytes[:, :, 1] = (rgb565 & 0xFF).astype(np.uint8)

        return rgb565_bytes

    def set_backlight(self, brightness):
        """
        Set backlight brightness (0-100)
        Uses PWM if available
        """
        if not RPI_AVAILABLE or self.bl_pin is None:
            return

        if GPIO_LIB == "lgpio":
            lgpio.gpio_write(self.gpio_handle, self.bl_pin, 1 if brightness > 0 else 0)
        else:
            if brightness > 0:
                GPIO.output(self.bl_pin, GPIO.HIGH)
            else:
                GPIO.output(self.bl_pin, GPIO.LOW)

    def sleep(self):
        """Put display to sleep"""
        if RPI_AVAILABLE:
            self._write_cmd(CMD_SLPIN)

    def wake(self):
        """Wake display from sleep"""
        if RPI_AVAILABLE:
            self._write_cmd(CMD_SLPOUT)
            time.sleep(0.12)

    def cleanup(self):
        """Cleanup GPIO and SPI"""
        if RPI_AVAILABLE:
            self.clear()
            self.set_backlight(0)
            if self.spi:
                self.spi.close()
            if GPIO_LIB == "lgpio":
                if self.gpio_handle is not None:
                    lgpio.gpiochip_close(self.gpio_handle)
            else:
                GPIO.cleanup()


class GC9A01ASimulator:
    """
    Simulator for GC9A01A when running on non-Pi systems
    Uses Pygame to simulate the round display
    """

    def __init__(self):
        import pygame
        self.pygame = pygame
        self.width = config.GC9A01A_WIDTH
        self.height = config.GC9A01A_HEIGHT

        pygame.init()
        # Create a window slightly larger to show the round mask
        self.screen = pygame.display.set_mode((self.width + 40, self.height + 40))
        pygame.display.set_caption("GC9A01A Simulator (240x240 Round)")

        # Create circular mask
        self.mask = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.circle(self.mask, (255, 255, 255, 255),
                          (self.width // 2, self.height // 2),
                          self.width // 2)

        self._buffer = pygame.Surface((self.width, self.height))

    def clear(self, color=(0, 0, 0)):
        """Clear display"""
        self._buffer.fill(color)
        self._update_screen()

    def draw_from_surface(self, pygame_surface):
        """Draw from Pygame surface"""
        self._buffer.blit(pygame_surface, (0, 0))
        self._update_screen()

    def _update_screen(self):
        """Update simulator screen with round mask"""
        self.screen.fill((30, 30, 30))  # Dark gray background

        # Apply circular mask
        masked = self._buffer.copy()
        masked.blit(self.mask, (0, 0), special_flags=self.pygame.BLEND_RGBA_MULT)

        # Draw round bezel
        self.pygame.draw.circle(self.screen, (50, 50, 50),
                               (self.width // 2 + 20, self.height // 2 + 20),
                               self.width // 2 + 5, 5)

        # Draw display content
        self.screen.blit(masked, (20, 20))

        self.pygame.display.flip()

    def update(self):
        """Update display"""
        self._update_screen()

    def set_backlight(self, brightness):
        pass

    def cleanup(self):
        self.pygame.quit()


def create_display():
    """
    Factory function to create appropriate display
    Returns GC9A01ADisplay on Pi, GC9A01ASimulator otherwise
    """
    if RPI_AVAILABLE:
        return GC9A01ADisplay()
    else:
        return GC9A01ASimulator()
