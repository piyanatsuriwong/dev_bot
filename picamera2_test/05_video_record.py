#!/usr/bin/env python3
"""
05_video_record.py - ทดสอบบันทึกวีดีโอด้วย Picamera2

ทดสอบ:
1. บันทึกวีดีโอ H.264
2. บันทึกเป็นไฟล์ MP4
3. Preview ขณะบันทึก

Usage:
    python3 05_video_record.py
    python3 05_video_record.py --duration 10
    python3 05_video_record.py --output myvideo.mp4

Source: Based on Picamera2 examples
"""

import argparse
import time
from datetime import datetime
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput

def main():
    parser = argparse.ArgumentParser(description="Video Recording Test")
    parser.add_argument("--duration", type=int, default=5, 
                        help="Recording duration in seconds (default: 5)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename (default: timestamped)")
    parser.add_argument("--resolution", type=str, default="1920x1080",
                        help="Video resolution WxH (default: 1920x1080)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second (default: 30)")
    args = parser.parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    
    # Generate output filename if not specified
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"video_{timestamp}.mp4"
    else:
        output_file = args.output
    
    print("=" * 60)
    print("Picamera2 Video Recording Test")
    print("=" * 60)
    print(f"\nSettings:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {args.fps}")
    print(f"   Duration: {args.duration} seconds")
    print(f"   Output: {output_file}")
    
    print("\n[1] Creating Picamera2 instance...")
    picam2 = Picamera2()
    
    model = picam2.camera_properties.get('Model', 'Unknown')
    print(f"    Camera Model: {model}")
    
    # Create video configuration
    print("\n[2] Creating video configuration...")
    video_config = picam2.create_video_configuration(
        main={"size": (width, height), "format": "RGB888"},
        controls={"FrameRate": args.fps}
    )
    picam2.configure(video_config)
    
    # Create encoder and output
    print("\n[3] Setting up H.264 encoder...")
    encoder = H264Encoder(10000000)  # 10 Mbps bitrate
    output = FfmpegOutput(output_file)
    
    # Start recording
    print("\n[4] Starting recording...")
    picam2.start_recording(encoder, output)
    
    # Recording progress
    for i in range(args.duration):
        print(f"    Recording... {i+1}/{args.duration}s", end="\r")
        time.sleep(1)
    print(f"    Recording... {args.duration}/{args.duration}s - DONE")
    
    # Stop recording
    print("\n[5] Stopping recording...")
    picam2.stop_recording()
    picam2.close()
    
    print(f"\n✓ Video saved to: {output_file}")
    print("\n" + "=" * 60)
    print("Video recording test complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
