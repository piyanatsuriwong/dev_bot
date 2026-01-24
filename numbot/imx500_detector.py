#!/usr/bin/env python3
"""
IMX500 YOLO Detector - Background YOLO detection using IMX500

Based on: picamera2_test/imx500/imx500_yolo_test.py
Uses IMX500 class directly with Picamera2 for YOLO inference.

Usage:
    detector = IMX500Detector()
    if detector.start():
        while running:
            detections = detector.get_detections()
            # detections = [{'label': 'person', 'score': 0.85}, ...]
    detector.stop()
"""

import threading
import time
import numpy as np

# Try to import IMX500
try:
    from picamera2 import Picamera2
    from picamera2.devices.imx500 import IMX500
    IMX500_AVAILABLE = True
except ImportError:
    IMX500_AVAILABLE = False
    print("Warning: IMX500 not available")

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


class IMX500Detector:
    """
    IMX500 YOLO Detector - runs in background thread
    
    Provides continuous YOLO object detection using IMX500 AI Camera.
    Detection results are stored and can be retrieved anytime.
    """
    
    # Model paths
    MODEL_PATHS = {
        "yolo11n": "/usr/share/imx500-models/imx500_network_yolo11n_pp.rpk",
        "yolov8n": "/usr/share/imx500-models/imx500_network_yolov8n_pp.rpk"
    }
    
    def __init__(self, model="yolo11n", threshold=0.4):
        """
        Initialize IMX500 Detector
        
        Args:
            model: Model name ("yolo11n" or "yolov8n")
            threshold: Detection confidence threshold (0.0-1.0)
        """
        self.model_name = model
        self.model_path = self.MODEL_PATHS.get(model, self.MODEL_PATHS["yolo11n"])
        self.threshold = threshold
        
        # Hardware
        self.imx500 = None
        self.picam2 = None
        
        # Thread control
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        
        # Detection results
        self._detections = []
        self._detection_text = ""
        self._fps = 0.0
        
    def start(self):
        """Start detection in background thread"""
        if not IMX500_AVAILABLE:
            print("IMX500Detector: IMX500 not available")
            return False
            
        if self._running:
            return True
            
        try:
            # Initialize IMX500 with YOLO model
            print(f"IMX500Detector: Loading {self.model_name}...")
            print(f"    Model: {self.model_path}")
            print("    NOTE: First load may take 1-2 minutes")
            
            self.imx500 = IMX500(self.model_path)
            print("    Model loaded!")
            
            # Initialize Picamera2
            self.picam2 = Picamera2(self.imx500.camera_num)
            config = self.picam2.create_preview_configuration(buffer_count=12)
            self.picam2.configure(config)
            
            # Start camera
            self.imx500.show_network_fw_progress_bar()
            self.picam2.start(show_preview=False)
            
            # Wait for first inference
            print("IMX500Detector: Waiting for inference...")
            for _ in range(50):
                metadata = self.picam2.capture_metadata()
                outputs = self.imx500.get_outputs(metadata)
                if outputs is not None:
                    print(f"    Inference ready! Shapes: {[o.shape for o in outputs]}")
                    break
                time.sleep(0.1)
            
            # Start detection thread
            self._running = True
            self._thread = threading.Thread(target=self._detection_loop, daemon=True)
            self._thread.start()
            
            print("IMX500Detector: Started")
            return True
            
        except Exception as e:
            print(f"IMX500Detector: Failed to start - {e}")
            self.stop()
            return False
    
    def stop(self):
        """Stop detection thread"""
        self._running = False
        
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
            
        if self.picam2:
            try:
                self.picam2.stop()
                self.picam2.close()
            except:
                pass
            self.picam2 = None
            
        self.imx500 = None
        print("IMX500Detector: Stopped")
    
    def _detection_loop(self):
        """Background detection loop"""
        frame_count = 0
        fps_time = time.time()
        
        while self._running:
            try:
                # Capture metadata
                metadata = self.picam2.capture_metadata()
                
                # Parse detections
                detections = self._parse_detections(metadata)
                
                # Update results (thread-safe)
                with self._lock:
                    self._detections = detections
                    if detections:
                        labels = [f"{d['label']}:{d['score']:.0%}" for d in detections[:3]]
                        self._detection_text = ", ".join(labels)
                    else:
                        self._detection_text = ""
                
                # FPS calculation
                frame_count += 1
                now = time.time()
                if now - fps_time >= 1.0:
                    with self._lock:
                        self._fps = frame_count / (now - fps_time)
                    frame_count = 0
                    fps_time = now
                
                time.sleep(0.05)  # ~20 FPS
                
            except Exception as e:
                print(f"IMX500Detector: Error - {e}")
                time.sleep(0.1)
    
    def _parse_detections(self, metadata):
        """
        Parse YOLO _pp (post-processed) output tensors
        
        For _pp models, output is already processed:
        - Output 0: boxes [N, 4] normalized (y1, x1, y2, x2)
        - Output 1: scores [N]
        - Output 2: classes [N]
        """
        np_outputs = self.imx500.get_outputs(metadata, add_batch=True)
        
        if np_outputs is None:
            return []
        
        try:
            # Check format
            if len(np_outputs) >= 3:
                boxes = np_outputs[0][0]   # [N, 4]
                scores = np_outputs[1][0]  # [N]
                classes = np_outputs[2][0] # [N]
            else:
                return []
            
            detections = []
            for box, score, cls in zip(boxes, scores, classes):
                if score > self.threshold:
                    cls_idx = int(cls)
                    label = COCO_LABELS[cls_idx] if cls_idx < len(COCO_LABELS) else f"class_{cls_idx}"
                    detections.append({
                        'label': label,
                        'score': float(score),
                        'class': cls_idx,
                        'bbox': box.tolist() if hasattr(box, 'tolist') else list(box)
                    })
            
            return detections
            
        except Exception:
            return []
    
    def get_detections(self):
        """Get current detections (thread-safe)"""
        with self._lock:
            return list(self._detections)
    
    def get_detection_text(self):
        """Get formatted detection text for display"""
        with self._lock:
            return self._detection_text
    
    def get_fps(self):
        """Get current FPS"""
        with self._lock:
            return self._fps
    
    @property
    def running(self):
        """Check if detector is running"""
        return self._running


# Test
if __name__ == "__main__":
    print("Testing IMX500Detector...")
    
    detector = IMX500Detector(model="yolo11n", threshold=0.4)
    
    if detector.start():
        print("\nRunning for 15 seconds...")
        start = time.time()
        
        try:
            while time.time() - start < 15:
                detections = detector.get_detections()
                text = detector.get_detection_text()
                fps = detector.get_fps()
                
                if text:
                    print(f"[{time.time()-start:.1f}s] FPS:{fps:.1f} | {text}")
                
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\nStopped")
        
        detector.stop()
    else:
        print("Failed to start detector")
