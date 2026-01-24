#!/usr/bin/env python3
"""
imx500_detect_test.py - ทดสอบ Object Detection บน IMX500 แบบ Headless

ทดสอบ:
1. โหลด SSD MobileNet model
2. รัน detection บน IMX500 chip
3. แสดงผลลัพธ์ detection ใน console

Usage:
    python3 imx500_detect_test.py
    python3 imx500_detect_test.py --duration 30
"""

import time
import argparse
import numpy as np
from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500, NetworkIntrinsics
from picamera2.devices.imx500 import postprocess_nanodet_detection

# COCO labels
COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


def parse_detections(imx500, metadata, threshold=0.55):
    """Parse detection outputs from IMX500"""
    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    
    if np_outputs is None:
        return []
    
    # Get model input size
    input_w, input_h = imx500.get_input_size()
    
    # Parse outputs (format depends on model)
    try:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        
        detections = []
        for box, score, cls in zip(boxes, scores, classes):
            if score > threshold:
                cls_idx = int(cls)
                label = COCO_LABELS[cls_idx] if cls_idx < len(COCO_LABELS) else f"class_{cls_idx}"
                detections.append({
                    'label': label,
                    'score': float(score),
                    'class': cls_idx
                })
        
        return detections
    except Exception as e:
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=20, help="Test duration in seconds")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    parser.add_argument("--model", type=str, 
                        default="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk",
                        help="Path to model")
    args = parser.parse_args()

    print("=" * 60)
    print("IMX500 Object Detection Test (Headless)")
    print("=" * 60)
    print(f"Model: {args.model.split('/')[-1]}")
    print(f"Duration: {args.duration}s")
    print(f"Threshold: {args.threshold}")
    print("=" * 60)

    # Initialize IMX500
    print("\n[1] Loading model on IMX500...")
    imx500 = IMX500(args.model)
    print("    Model loaded!")

    # Initialize Picamera2
    print("\n[2] Initializing Picamera2...")
    picam2 = Picamera2(imx500.camera_num)
    
    config = picam2.create_preview_configuration(buffer_count=12)
    picam2.configure(config)

    # Start camera
    print("\n[3] Starting camera...")
    picam2.start(show_preview=False)
    
    print("\n[4] Waiting for camera to stabilize...")
    time.sleep(2)

    print("\n[5] Running object detection...")
    print("-" * 60)
    
    start_time = time.time()
    frame_count = 0
    total_detections = 0
    detected_objects = {}
    
    try:
        while time.time() - start_time < args.duration:
            # Capture metadata with inference results
            metadata = picam2.capture_metadata()
            frame_count += 1
            
            # Parse detections
            detections = parse_detections(imx500, metadata, args.threshold)
            
            if detections:
                total_detections += len(detections)
                elapsed = time.time() - start_time
                
                print(f"\n[{elapsed:.1f}s] Frame {frame_count} - Found {len(detections)} object(s):")
                
                for det in detections:
                    label = det['label']
                    score = det['score']
                    print(f"    • {label}: {score:.1%}")
                    
                    # Track detected objects
                    if label not in detected_objects:
                        detected_objects[label] = 0
                    detected_objects[label] += 1
            
            time.sleep(0.1)  # Small delay between frames
            
    except KeyboardInterrupt:
        print("\n\nStopped by user")

    # Stop camera
    print("\n" + "-" * 60)
    print("[6] Stopping camera...")
    picam2.stop()
    picam2.close()

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("DETECTION SUMMARY")
    print("=" * 60)
    print(f"Duration: {elapsed:.1f}s")
    print(f"Frames processed: {frame_count}")
    print(f"FPS: {frame_count/elapsed:.1f}")
    print(f"Total detections: {total_detections}")
    
    if detected_objects:
        print(f"\nObjects detected ({len(detected_objects)} types):")
        for label, count in sorted(detected_objects.items(), key=lambda x: -x[1]):
            print(f"    • {label}: {count} times")
    else:
        print("\nNo objects detected above threshold")
        print("Tip: Try lowering --threshold or pointing camera at objects")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
