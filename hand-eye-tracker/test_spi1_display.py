#!/usr/bin/env python3
"""
Test GC9A01 Display with SPI1 Configuration

Pin Mapping (Physical Pin -> GPIO):
- VCC: External power or Pin 1/17 (3.3V)
- GND: Pin 39
- SCL (SCLK): Pin 40 (GPIO 21) - SPI1 SCLK
- SDA (MOSI): Pin 38 (GPIO 20) - SPI1 MOSI
- CS: Pin 36 (GPIO 16) - SPI1 CE2
- DC: Pin 31 (GPIO 6)
- RST: Pin 33 (GPIO 13)
"""

import time
import sys

# SPI1 Configuration
SPI_PORT = 1          # SPI1
SPI_DEVICE = 0        # CS0 remapped to GPIO 16
DC_PIN = 6            # GPIO 6 (Pin 31)
RST_PIN = 13          # GPIO 13 (Pin 33)
BL_PIN = None         # No backlight pin specified
SPI_SPEED = 40000000  # 40 MHz (start slower for testing)

WIDTH = 240
HEIGHT = 240

def test_gpio():
    """Test GPIO access"""
    print("\n=== Step 1: Testing GPIO Access ===")
    try:
        import lgpio
        h = lgpio.gpiochip_open(0)
        print(f"✓ GPIO chip opened successfully")

        # Test DC pin
        lgpio.gpio_claim_output(h, DC_PIN)
        lgpio.gpio_write(h, DC_PIN, 1)
        print(f"✓ DC pin (GPIO {DC_PIN}) configured")

        # Test RST pin
        lgpio.gpio_claim_output(h, RST_PIN)
        lgpio.gpio_write(h, RST_PIN, 1)
        print(f"✓ RST pin (GPIO {RST_PIN}) configured")

        lgpio.gpiochip_close(h)
        return True
    except Exception as e:
        print(f"✗ GPIO Error: {e}")
        return False


def test_spi():
    """Test SPI1 access"""
    print("\n=== Step 2: Testing SPI1 Access ===")
    try:
        import spidev
        spi = spidev.SpiDev()
        spi.open(SPI_PORT, SPI_DEVICE)
        spi.max_speed_hz = SPI_SPEED
        spi.mode = 0
        print(f"✓ SPI{SPI_PORT}.{SPI_DEVICE} opened successfully")
        print(f"  Speed: {SPI_SPEED/1000000:.1f} MHz")
        spi.close()
        return True
    except FileNotFoundError:
        print(f"✗ SPI device /dev/spidev{SPI_PORT}.{SPI_DEVICE} not found!")
        print("  Try enabling SPI1:")
        print("  1. Edit /boot/firmware/config.txt")
        print("  2. Add: dtoverlay=spi1-3cs")
        print("  3. Reboot")
        return False
    except Exception as e:
        print(f"✗ SPI Error: {e}")
        return False


def test_display():
    """Test display initialization and colors"""
    print("\n=== Step 3: Testing Display ===")

    try:
        import lgpio
        import spidev

        # Open GPIO
        h = lgpio.gpiochip_open(0)
        lgpio.gpio_claim_output(h, DC_PIN)
        lgpio.gpio_claim_output(h, RST_PIN)

        # Open SPI
        spi = spidev.SpiDev()
        spi.open(SPI_PORT, SPI_DEVICE)
        spi.max_speed_hz = SPI_SPEED
        spi.mode = 0

        def write_cmd(cmd):
            lgpio.gpio_write(h, DC_PIN, 0)
            spi.writebytes([cmd])

        def write_data(data):
            lgpio.gpio_write(h, DC_PIN, 1)
            if isinstance(data, int):
                spi.writebytes([data])
            else:
                spi.writebytes(list(data))

        def write_data_bulk(data):
            lgpio.gpio_write(h, DC_PIN, 1)
            # Send in chunks
            chunk_size = 4096
            for i in range(0, len(data), chunk_size):
                spi.writebytes(data[i:i+chunk_size])

        # Hardware reset
        print("  Resetting display...")
        lgpio.gpio_write(h, RST_PIN, 0)
        time.sleep(0.1)
        lgpio.gpio_write(h, RST_PIN, 1)
        time.sleep(0.12)

        # GC9A01A Initialization sequence
        print("  Initializing GC9A01A...")

        # Software reset
        write_cmd(0x01)
        time.sleep(0.12)

        # Display off
        write_cmd(0x28)

        # Memory access control
        write_cmd(0x36)
        write_data(0x48)  # Row/Col order

        # Pixel format: 16-bit RGB565
        write_cmd(0x3A)
        write_data(0x55)

        # Sleep out
        write_cmd(0x11)
        time.sleep(0.12)

        # Display on
        write_cmd(0x29)
        time.sleep(0.02)

        print("✓ Display initialized")

        # Test colors
        colors = [
            ("RED", 0xF800),
            ("GREEN", 0x07E0),
            ("BLUE", 0x001F),
            ("WHITE", 0xFFFF),
            ("PURPLE", 0xA81F),
            ("BLACK", 0x0000),
        ]

        for name, color in colors:
            print(f"  Filling {name}...")

            # Set column address
            write_cmd(0x2A)
            write_data(0x00)
            write_data(0x00)
            write_data((WIDTH - 1) >> 8)
            write_data((WIDTH - 1) & 0xFF)

            # Set row address
            write_cmd(0x2B)
            write_data(0x00)
            write_data(0x00)
            write_data((HEIGHT - 1) >> 8)
            write_data((HEIGHT - 1) & 0xFF)

            # Memory write
            write_cmd(0x2C)

            # Fill with color
            hi = (color >> 8) & 0xFF
            lo = color & 0xFF
            data = [hi, lo] * (WIDTH * HEIGHT)
            write_data_bulk(data)

            time.sleep(0.5)

        print("✓ Color test completed")

        # Cleanup
        spi.close()
        lgpio.gpiochip_close(h)

        return True

    except Exception as e:
        print(f"✗ Display Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 50)
    print("GC9A01A Display Test - SPI1 Configuration")
    print("=" * 50)
    print(f"\nConfiguration:")
    print(f"  SPI Port: {SPI_PORT} (SPI1)")
    print(f"  SPI Device: {SPI_DEVICE} (CE2)")
    print(f"  DC Pin: GPIO {DC_PIN} (Pin 31)")
    print(f"  RST Pin: GPIO {RST_PIN} (Pin 33)")
    print(f"  Speed: {SPI_SPEED/1000000:.1f} MHz")

    # Run tests
    gpio_ok = test_gpio()
    if not gpio_ok:
        print("\n❌ GPIO test failed. Check permissions.")
        sys.exit(1)

    spi_ok = test_spi()
    if not spi_ok:
        print("\n❌ SPI test failed. Enable SPI1 first:")
        print("   sudo nano /boot/firmware/config.txt")
        print("   Add: dtoverlay=spi1-3cs")
        print("   Then reboot")
        sys.exit(1)

    display_ok = test_display()
    if display_ok:
        print("\n" + "=" * 50)
        print("✅ All tests passed! Display is working.")
        print("=" * 50)
    else:
        print("\n❌ Display test failed. Check wiring.")
        sys.exit(1)


if __name__ == "__main__":
    main()
