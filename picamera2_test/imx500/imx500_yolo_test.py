#!/usr/bin/env python3
"""
imx500_yolo_test.py - ทดสอบ YOLO Object Detection บน IMX500

ใช้ YOLO models ที่มี post-processing บน chip (_pp)
Output format: boxes, scores, classes (เหมือน SSD)

Usage:
    python3 imx500_yolo_test.py
    python3 imx500_yolo_test.py --model yolov8n
    python3 imx500_yolo_test.py --duration 30
"""

import time
import argparse
import numpy as np
from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500, NetworkIntrinsics

# COCO labels (80 classes)
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


def parse_yolo_pp_detections(imx500, metadata, threshold=0.5):
    """Parse YOLO _pp (post-processed) output tensors
    
    For _pp models, output is already processed:
    - Output 0: boxes [N, 4] normalized (y1, x1, y2, x2)
    - Output 1: scores [N]
    - Output 2: classes [N]
    """
    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    
    if np_outputs is None:
        return []
    
    try:
        # Debug: print shapes
        shapes = [o.shape for o in np_outputs]
        
        # Check format
        if len(np_outputs) >= 3:
            # Standard _pp format: boxes, scores, classes
            boxes = np_outputs[0][0]   # [N, 4]
            scores = np_outputs[1][0]  # [N]
            classes = np_outputs[2][0] # [N]
        else:
            return []
        
        detections = []
        for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
            if score > threshold:
                cls_idx = int(cls)
                label = COCO_LABELS[cls_idx] if cls_idx < len(COCO_LABELS) else f"class_{cls_idx}"
                detections.append({
                    'label': label,
                    'score': float(score),
                    'class': cls_idx,
                    'bbox': box.tolist() if hasattr(box, 'tolist') else list(box)
                })
        
        return detections
        
    except Exception as e:
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolo11n", 
                        choices=["yolo11n", "yolov8n"],
                        help="YOLO model to use")
    parser.add_argument("--duration", type=int, default=20, help="Test duration in seconds")
    parser.add_argument("--threshold", type=float, default=0.4, help="Detection threshold")
    args = parser.parse_args()
    
    # Model path
    model_paths = {
        "yolo11n": "/usr/share/imx500-models/imx500_network_yolo11n_pp.rpk",
        "yolov8n": "/usr/share/imx500-models/imx500_network_yolov8n_pp.rpk"
    }
    model_path = model_paths[args.model]

    print("=" * 60)
    print(f"IMX500 {args.model.upper()} Object Detection Test")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Path: {model_path}")
    print(f"Duration: {args.duration}s")
    print(f"Threshold: {args.threshold}")
    print("=" * 60)

    # Initialize IMX500 with YOLO model
    print("\n[1] Loading YOLO model on IMX500...")
    print("    NOTE: First load may take 1-2 minutes for firmware upload")
    imx500 = IMX500(model_path)
    print("    Model loaded!")
    
    # Get input size
    try:
        input_size = imx500.get_input_size()
        print(f"    Input size: {input_size}")
    except:
        pass

    # Initialize Picamera2
    print("\n[2] Initializing Picamera2...")
    picam2 = Picamera2(imx500.camera_num)
    
    config = picam2.create_preview_configuration(buffer_count=12)
    picam2.configure(config)

    # Start camera
    print("\n[3] Starting camera...")
    imx500.show_network_fw_progress_bar()
    picam2.start(show_preview=False)
    
    print("\n[4] Waiting for inference to start...")
    # Wait for first valid inference
    for _ in range(50):
        metadata = picam2.capture_metadata()
        outputs = imx500.get_outputs(metadata)
        if outputs is not None:
            print(f"    Inference ready! Output shapes: {[o.shape for o in outputs]}")
            break
        time.sleep(0.1)
    else:
        print("    Warning: No inference output received")

    print("\n[5] Running YOLO object detection...")
    print("-" * 60)
    
    start_time = time.time()
    frame_count = 0
    total_detections = 0
    detected_objects = {}
    last_print_time = 0
    
    try:
        while time.time() - start_time < args.duration:
            # Capture metadata with inference results
            metadata = picam2.capture_metadata()
            frame_count += 1
            
            # Parse YOLO detections
            detections = parse_yolo_pp_detections(imx500, metadata, args.threshold)
            
            if detections:
                total_detections += len(detections)
                elapsed = time.time() - start_time
                
                # Print every detection
                print(f"\n[{elapsed:.1f}s] Frame {frame_count} - Found {len(detections)} object(s):")
                
                for det in detections:
                    label = det['label']
                    score = det['score']
                    print(f"    • {label}: {score:.1%}")
                    
                    # Track detected objects
                    if label not in detected_objects:
                        detected_objects[label] = 0
                    detected_objects[label] += 1
            
            time.sleep(0.05)  # ~20 FPS
            
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
    print(f"{args.model.upper()} DETECTION SUMMARY")
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
        print("Tips:")
        print("  - Try lowering --threshold (e.g., 0.3)")
        print("  - Point camera at common objects (person, chair, bottle)")
        print("  - Ensure good lighting")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
