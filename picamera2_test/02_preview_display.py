#!/usr/bin/env python3
"""
02_preview_display.py - ทดสอบ Preview หน้าจอด้วย Picamera2

ทดสอบ:
1. Preview ผ่าน Qt (ถ้ามี display)
2. Preview ผ่าน OpenCV
3. แสดง FPS

Usage:
    python3 02_preview_display.py
    python3 02_preview_display.py --opencv  # ใช้ OpenCV แทน Qt
    python3 02_preview_display.py --duration 30

Source: Based on Picamera2 examples
"""

import argparse
import time
import cv2
import numpy as np
from picamera2 import Picamera2

def preview_opencv(picam2, duration=10):
    """Preview using OpenCV window"""
    print("\n[OpenCV Preview Mode]")
    print(f"Duration: {duration} seconds")
    print("Press 'q' to quit early, 'c' to capture")
    
    # Create preview configuration
    config = picam2.create_preview_configuration(
        main={"size": (1280, 720), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    
    start_time = time.time()
    frame_count = 0
    fps = 0
    fps_update_time = start_time
    capture_count = 0
    
    while True:
        # Capture frame
        frame = picam2.capture_array()
        frame_count += 1
        
        # Calculate FPS every second
        current_time = time.time()
        if current_time - fps_update_time >= 1.0:
            fps = frame_count / (current_time - fps_update_time)
            frame_count = 0
            fps_update_time = current_time
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Add info overlay
        model = picam2.camera_properties.get('Model', 'Unknown')
        cv2.putText(frame_bgr, f"Camera: {model}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_bgr, f"Resolution: {frame.shape[1]}x{frame.shape[0]}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        elapsed = current_time - start_time
        remaining = max(0, duration - elapsed)
        cv2.putText(frame_bgr, f"Time: {elapsed:.1f}s / {duration}s", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display
        cv2.imshow("Picamera2 Preview", frame_bgr)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('c'):
            # Capture current frame
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"preview_capture_{timestamp}.jpg"
            cv2.imwrite(filename, frame_bgr)
            capture_count += 1
            print(f"    ✓ Captured: {filename}")
        
        # Check duration
        if current_time - start_time >= duration:
            print(f"\n{duration} seconds elapsed, stopping...")
            break
    
    cv2.destroyAllWindows()
    picam2.stop()
    
    if capture_count > 0:
        print(f"\n✓ Total captures: {capture_count}")

def preview_qt(picam2, duration=10):
    """Preview using Qt window (requires display)"""
    print("\n[Qt Preview Mode]")
    print(f"Duration: {duration} seconds")
    
    try:
        from picamera2 import Preview
        
        config = picam2.create_preview_configuration(
            main={"size": (1280, 720)}
        )
        picam2.configure(config)
        picam2.start_preview(Preview.QTGL)
        picam2.start()
        
        print("Preview running... (Ctrl+C to stop early)")
        time.sleep(duration)
        
        picam2.stop_preview()
        picam2.stop()
        
    except Exception as e:
        print(f"Qt preview failed: {e}")
        print("Falling back to OpenCV...")
        preview_opencv(picam2, duration)

def main():
    parser = argparse.ArgumentParser(description="Picamera2 Preview Test")
    parser.add_argument("--opencv", action="store_true", 
                        help="Use OpenCV for preview instead of Qt")
    parser.add_argument("--duration", type=int, default=10,
                        help="Preview duration in seconds (default: 10)")
    args = parser.parse_args()
    
    print("=" * 50)
    print("Picamera2 Preview Display Test")
    print("=" * 50)
    
    print("\n[1] Creating Picamera2 instance...")
    picam2 = Picamera2()
    
    model = picam2.camera_properties.get('Model', 'Unknown')
    print(f"    Camera Model: {model}")
    
    if args.opencv:
        preview_opencv(picam2, args.duration)
    else:
        preview_qt(picam2, args.duration)
    
    picam2.close()
    
    print("\n" + "=" * 50)
    print("Test completed!")
    print("=" * 50)

if __name__ == "__main__":
    main()
