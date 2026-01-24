#!/usr/bin/env python3
"""
01_basic_capture.py - ทดสอบ capture ภาพพื้นฐานด้วย Picamera2

ทดสอบ:
1. สร้าง Picamera2 instance
2. Configure กล้อง
3. Capture ภาพนิ่ง
4. บันทึกเป็นไฟล์

Usage:
    python3 01_basic_capture.py

Source: Based on Picamera2 examples - capture_headless.py
"""

from picamera2 import Picamera2
import time
from datetime import datetime

def main():
    print("=" * 50)
    print("Picamera2 Basic Capture Test")
    print("=" * 50)
    
    # สร้าง Picamera2 instance
    print("\n[1] Creating Picamera2 instance...")
    picam2 = Picamera2()
    
    # แสดงข้อมูลกล้อง
    print(f"    Camera Model: {picam2.camera_properties.get('Model', 'Unknown')}")
    
    # สร้าง configuration สำหรับ capture still image
    print("\n[2] Creating still configuration...")
    config = picam2.create_still_configuration(
        main={"size": (1920, 1080)},
        display=None
    )
    
    # Apply configuration
    print("\n[3] Configuring camera...")
    picam2.configure(config)
    
    # Start camera
    print("\n[4] Starting camera...")
    picam2.start()
    
    # รอให้กล้อง warm up
    print("    Waiting for camera to warm up...")
    time.sleep(2)
    
    # Capture image
    print("\n[5] Capturing image...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"capture_{timestamp}.jpg"
    
    picam2.capture_file(filename)
    print(f"    ✓ Saved to: {filename}")
    
    # Also capture as numpy array
    print("\n[6] Capturing as numpy array...")
    np_array = picam2.capture_array()
    print(f"    Array shape: {np_array.shape}")
    print(f"    Array dtype: {np_array.dtype}")
    
    # Stop camera
    print("\n[7] Stopping camera...")
    picam2.stop()
    picam2.close()
    
    print("\n" + "=" * 50)
    print("Test completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main()
