#!/usr/bin/env python3
"""
04_metadata_test.py - ทดสอบการอ่าน Metadata จากกล้อง

ทดสอบ:
1. อ่าน metadata พื้นฐานจาก Picamera2
2. แสดงข้อมูล exposure, gain, temperature
3. สำหรับ IMX500 จะแสดง AI inference metadata

Usage:
    python3 04_metadata_test.py

Source: Based on Picamera2 examples - metadata.py
"""

import time
from picamera2 import Picamera2

def print_metadata(metadata, title="Metadata"):
    """Pretty print metadata dictionary"""
    print(f"\n[{title}]")
    print("-" * 50)
    
    # Group important keys
    important_keys = [
        'ExposureTime', 'AnalogueGain', 'DigitalGain',
        'Lux', 'ColourTemperature', 'FocusFoM',
        'SensorTimestamp', 'FrameDuration'
    ]
    
    # Print important keys first
    for key in important_keys:
        if key in metadata:
            value = metadata[key]
            if key == 'SensorTimestamp':
                # Convert nanoseconds to seconds
                value = f"{value / 1e9:.3f}s"
            elif key == 'ExposureTime':
                value = f"{value}μs ({value/1000:.2f}ms)"
            elif key == 'FrameDuration':
                fps = 1000000 / value if value > 0 else 0
                value = f"{value}μs (≈{fps:.1f} FPS)"
            print(f"   {key}: {value}")
    
    # Print other keys
    print("\n   [Other Metadata]")
    for key, value in sorted(metadata.items()):
        if key not in important_keys:
            # Truncate long values
            str_value = str(value)
            if len(str_value) > 60:
                str_value = str_value[:60] + "..."
            print(f"   {key}: {str_value}")

def main():
    print("=" * 60)
    print("Picamera2 Metadata Test")
    print("=" * 60)
    
    print("\n[1] Creating Picamera2 instance...")
    picam2 = Picamera2()
    
    model = picam2.camera_properties.get('Model', 'Unknown')
    print(f"    Camera Model: {model}")
    
    # Create configuration
    print("\n[2] Creating preview configuration...")
    config = picam2.create_preview_configuration()
    picam2.configure(config)
    
    # Start camera
    print("\n[3] Starting camera...")
    picam2.start()
    
    # Wait for camera to stabilize
    print("    Waiting for camera to stabilize...")
    time.sleep(2)
    
    # Capture metadata several times
    print("\n[4] Capturing metadata (5 samples, 1 second apart)...")
    
    for i in range(5):
        metadata = picam2.capture_metadata()
        print_metadata(metadata, f"Sample {i+1}")
        time.sleep(1)
    
    # Stop camera
    print("\n[5] Stopping camera...")
    picam2.stop()
    picam2.close()
    
    print("\n" + "=" * 60)
    print("Metadata test complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
