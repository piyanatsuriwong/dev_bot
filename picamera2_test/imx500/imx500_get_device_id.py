#!/usr/bin/env python3
"""
imx500_get_device_id.py - รับ Device ID ของ IMX500 AI Camera

ทดสอบ:
1. โหลด model บน IMX500
2. เริ่ม stream
3. อ่าน Device ID

Usage:
    python3 imx500_get_device_id.py

Source: Official Picamera2 IMX500 example
https://github.com/raspberrypi/picamera2/blob/main/examples/imx500/imx500_get_device_id.py
"""

from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500

# Default model path
model = "/usr/share/imx500-models/imx500_network_mobilenet_v2.rpk"

print("=" * 60)
print("IMX500 Device ID Test")
print("=" * 60)

print(f"\n[1] Loading model: {model}")

# Startup IMX500 / Picamera2
# Note: IMX500() must be called before Picamera2()
imx500 = IMX500(model)

print("[2] Initializing Picamera2...")
picam2 = Picamera2()
config = picam2.create_preview_configuration()

print("[3] Starting camera (no preview)...")
picam2.start(config, show_preview=False)

# Wait for the device to be streaming
print("[4] Waiting for stream...")
picam2.capture_metadata()

# Get device_id
print("[5] Getting device ID...")
device_id = imx500.get_device_id()

print(f"\n✓ IMX500 Device ID = {device_id}")

print("\n[6] Stopping camera...")
picam2.stop()
picam2.close()

print("\n" + "=" * 60)
print("Device ID test complete!")
print("=" * 60)
