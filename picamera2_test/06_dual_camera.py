#!/usr/bin/env python3
"""
06_dual_camera.py - ทดสอบกล้อง 2 ตัวพร้อมกัน

ทดสอบ:
1. เปิดกล้อง 2 ตัวพร้อมกัน
2. Preview ทั้ง 2 กล้อง
3. Capture จากทั้ง 2 กล้อง

Requirements:
- ต้องมีกล้องต่ออยู่ 2 ตัว (เช่น IMX500 + IMX708)

Usage:
    python3 06_dual_camera.py

Source: Based on Picamera2 examples - multicamera_preview.py
"""

import time
import numpy as np
import cv2
from datetime import datetime
from picamera2 import Picamera2

def main():
    print("=" * 60)
    print("Picamera2 Dual Camera Test")
    print("=" * 60)
    
    # Check available cameras
    cameras = Picamera2.global_camera_info()
    print(f"\n[Detected Cameras: {len(cameras)}]")
    
    if len(cameras) < 2:
        print("⚠️  Warning: Less than 2 cameras detected!")
        print("    This test requires 2 cameras connected.")
        print(f"    Found: {len(cameras)} camera(s)")
        
        if len(cameras) == 1:
            print(f"\n    Camera 0: {cameras[0].get('Model', 'Unknown')}")
        return
    
    # Print camera info
    for idx, cam_info in enumerate(cameras):
        print(f"   Camera {idx}: {cam_info.get('Model', 'Unknown')}")
    
    print("\n[1] Initializing Camera 0...")
    picam2_a = Picamera2(0)
    config_a = picam2_a.create_preview_configuration(
        main={"size": (1280, 720), "format": "RGB888"}
    )
    picam2_a.configure(config_a)
    
    print("[2] Initializing Camera 1...")
    picam2_b = Picamera2(1)
    config_b = picam2_b.create_preview_configuration(
        main={"size": (1280, 720), "format": "RGB888"}
    )
    picam2_b.configure(config_b)
    
    print("[3] Starting both cameras...")
    picam2_a.start()
    picam2_b.start()
    
    print("[4] Warming up cameras...")
    time.sleep(2)
    
    print("\n[5] Capturing preview (OpenCV window)")
    print("    Press 'c' to capture images from both cameras")
    print("    Press 'q' to quit")
    
    capture_count = 0
    
    while True:
        # Capture frames from both cameras
        frame_a = picam2_a.capture_array()
        frame_b = picam2_b.capture_array()
        
        # Convert RGB to BGR for OpenCV
        frame_a_bgr = cv2.cvtColor(frame_a, cv2.COLOR_RGB2BGR)
        frame_b_bgr = cv2.cvtColor(frame_b, cv2.COLOR_RGB2BGR)
        
        # Add labels
        label_a = f"Camera 0: {cameras[0].get('Model', 'Unknown')}"
        label_b = f"Camera 1: {cameras[1].get('Model', 'Unknown')}"
        
        cv2.putText(frame_a_bgr, label_a, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame_b_bgr, label_b, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Resize for side-by-side display
        h, w = frame_a_bgr.shape[:2]
        scale = 0.5  # Scale down for display
        frame_a_small = cv2.resize(frame_a_bgr, (int(w*scale), int(h*scale)))
        frame_b_small = cv2.resize(frame_b_bgr, (int(w*scale), int(h*scale)))
        
        # Combine side by side
        combined = np.hstack([frame_a_small, frame_b_small])
        
        cv2.imshow("Dual Camera Preview", combined)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            # Capture images
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            capture_count += 1
            
            filename_a = f"cam0_{timestamp}.jpg"
            filename_b = f"cam1_{timestamp}.jpg"
            
            cv2.imwrite(filename_a, frame_a_bgr)
            cv2.imwrite(filename_b, frame_b_bgr)
            
            print(f"    ✓ Captured: {filename_a}, {filename_b}")
            
        elif key == ord('q'):
            print("\n[6] Quitting...")
            break
    
    cv2.destroyAllWindows()
    
    print("[7] Stopping cameras...")
    picam2_a.stop()
    picam2_b.stop()
    picam2_a.close()
    picam2_b.close()
    
    print(f"\n✓ Total captures: {capture_count} pairs")
    print("\n" + "=" * 60)
    print("Dual camera test complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
