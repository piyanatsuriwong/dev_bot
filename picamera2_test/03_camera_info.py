#!/usr/bin/env python3
"""
03_camera_info.py - แสดงข้อมูลกล้องทั้งหมดที่ต่ออยู่

ทดสอบ:
1. ตรวจจับกล้องที่ต่ออยู่ทั้งหมด
2. แสดง properties ของแต่ละกล้อง
3. แสดง sensor modes ที่รองรับ

Usage:
    python3 03_camera_info.py

Source: Based on Picamera2 examples
"""

from picamera2 import Picamera2
import json

def print_camera_info():
    print("=" * 60)
    print("Picamera2 Camera Information")
    print("=" * 60)
    
    # Get list of cameras
    cameras = Picamera2.global_camera_info()
    
    print(f"\n[Detected Cameras: {len(cameras)}]")
    print("-" * 60)
    
    for idx, cam_info in enumerate(cameras):
        print(f"\n📷 Camera {idx}:")
        print(f"   Model: {cam_info.get('Model', 'Unknown')}")
        print(f"   Location: {cam_info.get('Location', 'Unknown')}")
        print(f"   Rotation: {cam_info.get('Rotation', 'Unknown')}")
        print(f"   ID: {cam_info.get('Id', 'Unknown')}")
        
    # Detailed info for each camera
    for idx in range(len(cameras)):
        print(f"\n{'='*60}")
        print(f"📸 Detailed Info for Camera {idx}")
        print("=" * 60)
        
        try:
            picam2 = Picamera2(idx)
            
            # Camera properties
            props = picam2.camera_properties
            print("\n[Camera Properties]")
            for key, value in props.items():
                if isinstance(value, (list, tuple)) and len(value) > 5:
                    print(f"   {key}: [{len(value)} items]")
                else:
                    print(f"   {key}: {value}")
            
            # Sensor modes
            print("\n[Available Sensor Modes]")
            sensor_modes = picam2.sensor_modes
            for i, mode in enumerate(sensor_modes):
                print(f"\n   Mode {i}:")
                for key, value in mode.items():
                    print(f"      {key}: {value}")
            
            # Available controls
            print("\n[Available Controls]")
            controls = picam2.camera_controls
            for name, (min_val, max_val, default) in controls.items():
                print(f"   {name}: min={min_val}, max={max_val}, default={default}")
            
            picam2.close()
            
        except Exception as e:
            print(f"   Error accessing camera {idx}: {e}")
    
    print("\n" + "=" * 60)
    print("Camera info complete!")
    print("=" * 60)

if __name__ == "__main__":
    print_camera_info()
