# st7735_display.py
# ST7735S LCD Display Driver for Raspberry Pi 5
# 128x160 pixel (typical), SPI interface
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

# ST7735 Commands
CMD_NOP     = 0x00
CMD_SWRESET = 0x01
CMD_RDDID   = 0x04
CMD_RDDST   = 0x09

CMD_SLPIN   = 0x10
CMD_SLPOUT  = 0x11
CMD_PTLON   = 0x12
CMD_NORON   = 0x13

CMD_INVOFF  = 0x20
CMD_INVON   = 0x21
CMD_DISPOFF = 0x28
CMD_DISPON  = 0x29
CMD_CASET   = 0x2A
CMD_RASET   = 0x2B
CMD_RAMWR   = 0x2C
CMD_RAMRD   = 0x2E

CMD_PTLAR   = 0x30
CMD_COLMOD  = 0x3A
CMD_MADCTL  = 0x36

CMD_FRMCTR1 = 0xB1
CMD_FRMCTR2 = 0xB2
CMD_FRMCTR3 = 0xB3
CMD_INVCTR  = 0xB4
CMD_DISSET5 = 0xB6

CMD_PWCTR1  = 0xC0
CMD_PWCTR2  = 0xC1
CMD_PWCTR3  = 0xC2
CMD_PWCTR4  = 0xC3
CMD_PWCTR5  = 0xC4
CMD_VMCTR1  = 0xC5

CMD_RDID1   = 0xDA
CMD_RDID2   = 0xDB
CMD_RDID3   = 0xDC
CMD_RDID4   = 0xDD

CMD_GMCTRP1 = 0xE0
CMD_GMCTRN1 = 0xE1

class ST7735Display:
    """
    ST7735S LCD Display Driver
    - 128x160 pixels (Configurable via config.py)
    - 16-bit color (RGB565)
    - SPI interface
    """

    def __init__(self):
        # Default to 128x160 if not in config
        self.width = getattr(config, 'ST7735_WIDTH', 128)
        self.height = getattr(config, 'ST7735_HEIGHT', 160)
        self.offset_x = getattr(config, 'ST7735_OFFSET_X', 0)
        self.offset_y = getattr(config, 'ST7735_OFFSET_Y', 0)
        
        # Reuse GC9A01A pins from config if specific ones aren't defined
        self.spi_port = getattr(config, 'ST7735_SPI_PORT', getattr(config, 'GC9A01A_SPI_PORT', 0))
        self.spi_cs = getattr(config, 'ST7735_SPI_CS', getattr(config, 'GC9A01A_SPI_CS', 0))
        self.dc_pin = getattr(config, 'ST7735_DC_PIN', getattr(config, 'GC9A01A_DC_PIN', 25))
        self.rst_pin = getattr(config, 'ST7735_RST_PIN', getattr(config, 'GC9A01A_RST_PIN', 27))
        self.bl_pin = getattr(config, 'ST7735_BL_PIN', getattr(config, 'GC9A01A_BL_PIN', 18))
        self.spi_speed = getattr(config, 'ST7735_SPI_SPEED', 24000000) # 24MHz default for ST7735

        self.spi = None
        self.gpio_handle = None  # For lgpio
        self._buffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        if RPI_AVAILABLE:
            self._init_gpio()
            self._init_spi()
            self._init_display()
        else:
            print("ST7735: Simulation mode - no hardware")

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
        """Initialize ST7735 display"""
        # Hardware reset
        self._reset()

        # Software reset
        self._write_cmd(CMD_SWRESET)
        time.sleep(0.15)

        # Sleep out
        self._write_cmd(CMD_SLPOUT)
        time.sleep(0.2)

        # Frame rate control
        self._write_cmd(CMD_FRMCTR1)
        self._write_data([0x01, 0x2C, 0x2D])
        self._write_cmd(CMD_FRMCTR2)
        self._write_data([0x01, 0x2C, 0x2D])
        self._write_cmd(CMD_FRMCTR3)
        self._write_data([0x01, 0x2C, 0x2D, 0x01, 0x2C, 0x2D])

        # Display inversion
        self._write_cmd(CMD_INVCTR)
        self._write_data([0x07])

        # Power control
        self._write_cmd(CMD_PWCTR1)
        self._write_data([0xA2, 0x02, 0x84])
        self._write_cmd(CMD_PWCTR2)
        self._write_data([0xC5])
        self._write_cmd(CMD_PWCTR3)
        self._write_data([0x0A, 0x00])
        self._write_cmd(CMD_PWCTR4)
        self._write_data([0x8A, 0x2A])
        self._write_cmd(CMD_PWCTR5)
        self._write_data([0x8A, 0xEE])

        # VCOM control
        self._write_cmd(CMD_VMCTR1)
        self._write_data([0x0E])

        # Display Inversion On/Off (Try OFF first, some panels need ON)
        # self._write_cmd(CMD_INVON) 
        self._write_cmd(CMD_INVOFF) 

        # Memory Data Access Control (Orientation)
        # RGB565 (16-bit)
        self._write_cmd(CMD_COLMOD)
        self._write_data([0x05])

        self._write_cmd(CMD_MADCTL)
        # Inverted Landscape Mode (MV=1)
        # 0x68 = MY=0, MX=1, MV=1, ML=0, BGR=1 (Inverted Landscape with BGR)
        # Previous was 0xA8 (Standard Landscape)
        self._write_data([0x68]) 

        # Gamma sequence (Standard ST7735S gamma)
        self._write_cmd(CMD_GMCTRP1)
        self._write_data([0x02, 0x1c, 0x07, 0x12, 0x37, 0x32, 0x29, 0x2d, 0x29, 0x25, 0x2B, 0x39, 0x00, 0x01, 0x03, 0x10])
        self._write_cmd(CMD_GMCTRN1)
        self._write_data([0x03, 0x1d, 0x07, 0x06, 0x2E, 0x2C, 0x29, 0x2D, 0x2E, 0x2E, 0x37, 0x3F, 0x00, 0x00, 0x02, 0x10])

        # Display ON
        self._write_cmd(CMD_DISPON)
        time.sleep(0.1)

        # Clear screen
        self.clear()

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
        # Apply offsets for ST7735 (some panels have offsets)
        x0 += self.offset_x
        x1 += self.offset_x
        y0 += self.offset_y
        y1 += self.offset_y
        
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
            # print(f"Error drawing surface: {e}")
            pass

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
        """Set backlight brightness (0-100)"""
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

class ST7735Simulator:
    """Simulator for ST7735 on non-Pi systems"""
    def __init__(self):
        import pygame
        self.pygame = pygame
        self.width = getattr(config, 'ST7735_WIDTH', 128)
        self.height = getattr(config, 'ST7735_HEIGHT', 160)

        pygame.init()
        self.screen = pygame.display.set_mode((self.width + 40, self.height + 40))
        pygame.display.set_caption("ST7735 Simulator")
        self._buffer = pygame.Surface((self.width, self.height))

    def clear(self, color=(0, 0, 0)):
        self._buffer.fill(color)
        self._update_screen()

    def draw_from_surface(self, pygame_surface):
        self._buffer.blit(pygame_surface, (0, 0))
        self._update_screen()

    def _update_screen(self):
        self.screen.fill((30, 30, 30))
        # Draw bezel
        self.pygame.draw.rect(self.screen, (50, 50, 50), 
                             (15, 15, self.width + 10, self.height + 10))
        # Draw display
        self.screen.blit(self._buffer, (20, 20))
        self.pygame.display.flip()

    def update(self):
        self._update_screen()

    def set_backlight(self, brightness):
        pass

    def cleanup(self):
        self.pygame.quit()

def create_display():
    """Factory function"""
    if RPI_AVAILABLE:
        return ST7735Display()
    else:
        return ST7735Simulator()
