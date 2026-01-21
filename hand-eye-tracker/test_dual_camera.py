#!/usr/bin/env python3
"""
Dual Camera Preview - Display both cameras on HDMI
Camera 0: ov5647 (CSI)
Camera 1: imx500 (AI Camera)
"""

import cv2
import numpy as np
from picamera2 import Picamera2
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description='Dual Camera Preview')
    parser.add_argument('--camera', type=int, nargs='+', default=[0, 1],
                       help='Camera numbers to use (0, 1, or both). Example: --camera 0 or --camera 1 or --camera 0 1')
    args = parser.parse_args()
    
    # Validate camera numbers
    cameras_to_use = sorted(set(args.camera))
    if not all(cam in [0, 1] for cam in cameras_to_use):
        print("Error: Camera numbers must be 0 or 1")
        return
    
    print(f"Initializing cameras: {cameras_to_use}...")

    # Initialize cameras based on selection
    cameras = {}
    if 0 in cameras_to_use:
        cameras[0] = Picamera2(camera_num=0)
    if 1 in cameras_to_use:
        cameras[1] = Picamera2(camera_num=1)

    # Configure cameras for preview
    for cam_num in cameras_to_use:
        picam = cameras[cam_num]
        config = picam.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        picam.configure(config)
        
        # Fix pink tint on Camera 0 (ov5647) - set AWB and color gains
        if cam_num == 0:
            picam.set_controls({
                "AwbEnable": True,
                "AwbMode": 0,  # Auto
                "ColourGains": (1.5, 1.5),  # Adjust if still pink
            })

    print("Starting cameras...")
    for cam_num in cameras_to_use:
        cameras[cam_num].start()

    # Wait for cameras to warm up (longer for AWB to stabilize)
    print("Waiting for AWB to stabilize...")
    time.sleep(3)

    camera_names = {
        0: "ov5647",
        1: "imx500 (AI)"
    }
    
    window_title = "Camera Preview"
    if len(cameras_to_use) == 2:
        window_title = "Dual Camera"
        print("Cameras started! Press 'q' to quit, ESC to exit")
    else:
        cam_name = camera_names[cameras_to_use[0]]
        print(f"Camera {cameras_to_use[0]} ({cam_name}) started! Press 'q' to quit, ESC to exit")

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    if len(cameras_to_use) == 2:
        cv2.resizeWindow(window_title, 1280, 480)
    else:
        cv2.resizeWindow(window_title, 640, 480)

    fps_time = time.time()
    frame_count = 0
    fps = 0

    try:
        while True:
            # Capture from selected cameras
            frames = {}
            for cam_num in cameras_to_use:
                picam = cameras[cam_num]
                frame = picam.capture_array()
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frames[cam_num] = frame_bgr

            # Calculate FPS
            frame_count += 1
            if time.time() - fps_time >= 1.0:
                fps = frame_count
                frame_count = 0
                fps_time = time.time()

            # Add labels and FPS to frames
            for cam_num in cameras_to_use:
                frame = frames[cam_num]
                cam_name = camera_names[cam_num]
                cv2.putText(frame, f"Camera {cam_num}: {cam_name}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"FPS: {fps}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Display frames
            if len(cameras_to_use) == 2:
                # Combine side by side
                combined = np.hstack((frames[0], frames[1]))
                cv2.imshow(window_title, combined)
            else:
                # Show single camera
                cv2.imshow(window_title, frames[cameras_to_use[0]])

            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        print("Stopping cameras...")
        for cam_num in cameras_to_use:
            cameras[cam_num].stop()
        cv2.destroyAllWindows()
        print("Done!")

if __name__ == "__main__":
    main()
