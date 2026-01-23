#!/usr/bin/env python3
"""
IMX500 Object Detection using Picamera2 (Direct API)
ใช้ Picamera2 โดยตรงแทน modlib เพื่อหลีกเลี่ยงการ deploy โมเดลที่ค้าง

Benefits:
- ใช้โมเดล .rpk ที่มีอยู่แล้ว (pre-installed)
- ไม่ต้อง deploy โมเดลใหม่ทุกครั้ง
- ดึง metadata จาก inference โดยตรง
"""

import threading
import time
import gc
from enum import Enum
from functools import lru_cache
from typing import Optional, List, Dict, Tuple

# Try to import Picamera2 IMX500 support
IMX500_AVAILABLE = False
try:
    from picamera2 import Picamera2
    from picamera2.devices import IMX500
    from picamera2.devices.imx500 import (
        NetworkIntrinsics,
        postprocess_nanodet_detection
    )
    import numpy as np
    IMX500_AVAILABLE = True
    print("IMX500 (Picamera2): Available")
except ImportError as e:
    print(f"IMX500 (Picamera2): Not available - {e}")


class DetectionMode(Enum):
    """Detection operation modes"""
    DETECT = "detect"
    TRACK = "track"
    TRACK_ALL = "track_all"


class Detection:
    """Single detection result"""
    def __init__(self, coords, category: int, confidence: float, metadata, imx500, picam2):
        self.category = category
        self.confidence = confidence
        # Convert inference coords to ISP output coords
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)

    @property
    def x(self) -> int:
        return self.box[0]

    @property
    def y(self) -> int:
        return self.box[1]

    @property
    def width(self) -> int:
        return self.box[2]

    @property
    def height(self) -> int:
        return self.box[3]

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)


class IMX500Detector:
    """
    IMX500 Object Detector using Picamera2 Direct API
    
    ใช้โมเดลที่มีอยู่แล้วบน IMX500 โดยไม่ต้อง deploy ใหม่
    """

    # Available pre-installed models
    AVAILABLE_MODELS = {
        "ssd_mobilenet": "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk",
        "yolov8n": "/usr/share/imx500-models/imx500_network_yolov8n_pp.rpk",
        "nanodet": "/usr/share/imx500-models/imx500_network_nanodet_plus_416x416_pp.rpk",
        "efficientdet": "/usr/share/imx500-models/imx500_network_efficientdet_lite0_pp.rpk",
    }

    # COCO class names
    COCO_LABELS = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    ]

    THAI_CLASS_MAP = {
        "คน": "person", "แมว": "cat", "หมา": "dog", "สุนัข": "dog",
        "นก": "bird", "รถ": "car", "รถยนต์": "car", "มอเตอร์ไซค์": "motorcycle",
        "จักรยาน": "bicycle", "ขวด": "bottle", "แก้ว": "cup",
        "โทรศัพท์": "cell phone", "มือถือ": "cell phone",
    }

    def __init__(
        self,
        model: str = "ssd_mobilenet",
        threshold: float = 0.55,
        iou_threshold: float = 0.65,
        max_detections: int = 10,
        fps: int = 10,
        camera_num: Optional[int] = None
    ):
        """
        Initialize IMX500 Detector
        
        Args:
            model: Model name or path to .rpk file
            threshold: Detection confidence threshold
            iou_threshold: IoU threshold for NMS
            max_detections: Maximum detections per frame
            fps: Target frame rate
            camera_num: Camera index (None = auto-detect)
        """
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.fps = fps

        # State
        self._mode = DetectionMode.DETECT
        self._track_target = "person"
        self._show_text = True
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Results cache
        self._last_detections: List[Detection] = []
        self._cached_labels: List[str] = []
        self._cached_position: Tuple[float, float] = (0, 0)

        # FPS tracking
        self._fps_actual = 0
        self._frame_count = 0
        self._last_fps_time = time.time()

        # IMX500 and Picamera2 objects
        self.imx500: Optional[IMX500] = None
        self.picam2: Optional[Picamera2] = None
        self.intrinsics: Optional[NetworkIntrinsics] = None
        self._labels: List[str] = []

        # Determine model path
        if model in self.AVAILABLE_MODELS:
            self.model_path = self.AVAILABLE_MODELS[model]
            self.model_name = model
        else:
            self.model_path = model
            self.model_name = model.split("/")[-1].replace(".rpk", "")

        # Initialize
        if IMX500_AVAILABLE:
            self._init_imx500(camera_num)

    def _init_imx500(self, camera_num: Optional[int] = None):
        """Initialize IMX500 with the specified model"""
        try:
            print(f"[IMX500] Initializing with model: {self.model_name}")
            print(f"[IMX500] Model path: {self.model_path}")

            # Create IMX500 device - this loads the model firmware
            self.imx500 = IMX500(self.model_path)

            # Get network intrinsics (model metadata)
            self.intrinsics = self.imx500.network_intrinsics
            if not self.intrinsics:
                self.intrinsics = NetworkIntrinsics()
                self.intrinsics.task = "object detection"
            elif self.intrinsics.task != "object detection":
                print(f"[IMX500] Warning: Model task is '{self.intrinsics.task}', not object detection")

            # Set up defaults
            self.intrinsics.update_with_defaults()

            # Get labels
            if self.intrinsics.labels:
                self._labels = self.intrinsics.labels
            else:
                self._labels = self.COCO_LABELS
            print(f"[IMX500] Labels loaded: {len(self._labels)} classes")

            # Determine camera number
            if camera_num is None:
                camera_num = self.imx500.camera_num
            print(f"[IMX500] Using camera index: {camera_num}")

            # Create Picamera2 instance
            self.picam2 = Picamera2(camera_num)

            # Create configuration with desired FPS
            inference_rate = self.intrinsics.inference_rate or self.fps
            config = self.picam2.create_preview_configuration(
                controls={"FrameRate": inference_rate},
                buffer_count=12
            )

            print(f"[IMX500] Configuration: FPS={inference_rate}")
            print("[IMX500] Initialization complete!")

        except FileNotFoundError as e:
            print(f"[IMX500] Model file not found: {e}")
            self._cleanup()
        except Exception as e:
            print(f"[IMX500] Initialization error: {e}")
            self._cleanup()

    def start(self) -> bool:
        """Start the detection thread"""
        if not IMX500_AVAILABLE or self.picam2 is None:
            print("[IMX500] Cannot start - not initialized")
            return False

        try:
            # Show firmware upload progress
            self.imx500.show_network_fw_progress_bar()

            # Start camera
            config = self.picam2.create_preview_configuration(
                controls={"FrameRate": self.fps},
                buffer_count=12
            )
            self.picam2.start(config, show_preview=False)

            if self.intrinsics and self.intrinsics.preserve_aspect_ratio:
                self.imx500.set_auto_aspect_ratio()

            print("[IMX500] Camera started")

            # Start detection thread
            self._running = True
            self._thread = threading.Thread(target=self._detection_loop, daemon=True)
            self._thread.start()
            print("[IMX500] Detection thread started")
            return True

        except Exception as e:
            print(f"[IMX500] Start error: {e}")
            return False

    def stop(self):
        """Stop detection thread"""
        if self._running:
            print("[IMX500] Stopping...")
            self._running = False
            if self._thread:
                self._thread.join(timeout=3.0)
            self._cleanup()
            print("[IMX500] Stopped")

    def _cleanup(self):
        """Clean up resources"""
        if self.picam2:
            try:
                self.picam2.stop()
                self.picam2.close()
            except:
                pass
            self.picam2 = None
        self.imx500 = None
        gc.collect()

    def _detection_loop(self):
        """Main detection loop - runs in separate thread"""
        print("[IMX500] Detection loop started")

        while self._running:
            try:
                # Capture metadata containing inference results
                metadata = self.picam2.capture_metadata()

                # Parse detections from metadata
                detections = self._parse_detections(metadata)

                # Process detections
                with self._lock:
                    current_mode = self._mode
                    track_target = self._track_target

                # Calculate tracked position
                tracked_pos = None
                for det in detections:
                    label = self._get_label(det.category)

                    if tracked_pos is None:
                        should_track = False
                        if current_mode == DetectionMode.TRACK:
                            should_track = (label == track_target)
                        elif current_mode == DetectionMode.TRACK_ALL:
                            should_track = True

                        if should_track:
                            # Normalize to -1 to 1 range
                            # Assume 640x480 or get from config
                            frame_w, frame_h = 640, 480
                            cx = det.center[0] / frame_w
                            cy = det.center[1] / frame_h
                            tracked_pos = ((cx - 0.5) * 2, (cy - 0.5) * 2)

                # Update cache
                with self._lock:
                    self._last_detections = detections
                    self._cached_labels = [
                        self._get_label(d.category)
                        for d in sorted(detections, key=lambda x: x.confidence, reverse=True)[:3]
                    ]
                    if tracked_pos:
                        self._cached_position = tracked_pos

                # FPS tracking
                self._frame_count += 1
                now = time.time()
                if now - self._last_fps_time >= 1.0:
                    self._fps_actual = self._frame_count
                    self._frame_count = 0
                    self._last_fps_time = now

                    # Log detections
                    if detections:
                        det_str = ", ".join([
                            f"{self._get_label(d.category)} ({d.confidence:.2f})"
                            for d in detections[:3]
                        ])
                        print(f"[IMX500] Detected: {det_str} | FPS: {self._fps_actual}")

            except Exception as e:
                if self._running:
                    print(f"[IMX500] Detection error: {e}")
                    time.sleep(0.1)

        print("[IMX500] Detection loop ended")

    def _parse_detections(self, metadata: dict) -> List[Detection]:
        """Parse detections from metadata"""
        try:
            # Get inference outputs from metadata
            np_outputs = self.imx500.get_outputs(metadata, add_batch=True)
            if np_outputs is None:
                return self._last_detections

            input_w, input_h = self.imx500.get_input_size()

            # Handle different postprocess methods
            if self.intrinsics.postprocess == "nanodet":
                boxes, scores, classes = postprocess_nanodet_detection(
                    outputs=np_outputs[0],
                    conf=self.threshold,
                    iou_thres=self.iou_threshold,
                    max_out_dets=self.max_detections
                )[0]
                from picamera2.devices.imx500.postprocess import scale_boxes
                boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
            else:
                # Standard output format
                boxes = np_outputs[0][0]
                scores = np_outputs[1][0]
                classes = np_outputs[2][0]

                if self.intrinsics.bbox_normalization:
                    boxes = boxes / input_h

                if self.intrinsics.bbox_order == "xy":
                    boxes = boxes[:, [1, 0, 3, 2]]

                boxes = np.array_split(boxes, 4, axis=1)
                boxes = list(zip(*boxes))

            # Create Detection objects
            detections = []
            for box, score, category in zip(boxes, scores, classes):
                if score > self.threshold:
                    det = Detection(
                        coords=box,
                        category=int(category),
                        confidence=float(score),
                        metadata=metadata,
                        imx500=self.imx500,
                        picam2=self.picam2
                    )
                    detections.append(det)

            return detections[:self.max_detections]

        except Exception as e:
            print(f"[IMX500] Parse error: {e}")
            return []

    @lru_cache(maxsize=100)
    def _get_label(self, category: int) -> str:
        """Get label for category index"""
        try:
            # Filter out dash labels if configured
            labels = self._labels
            if self.intrinsics and self.intrinsics.ignore_dash_labels:
                labels = [l for l in labels if l and l != "-"]
            return labels[category] if category < len(labels) else f"class_{category}"
        except:
            return f"class_{category}"

    # --- Public Properties ---

    @property
    def detections(self) -> List[Detection]:
        """Get current detections"""
        with self._lock:
            return self._last_detections.copy()

    @property
    def fps_actual(self) -> int:
        """Get actual FPS"""
        return self._fps_actual

    @property
    def running(self) -> bool:
        """Check if running"""
        return self._running

    @property
    def mode(self) -> DetectionMode:
        """Get current mode"""
        with self._lock:
            return self._mode

    @property
    def track_target(self) -> str:
        """Get current track target"""
        with self._lock:
            return self._track_target

    # --- Public Methods ---

    def set_mode(self, mode: DetectionMode):
        """Set detection mode"""
        with self._lock:
            old_mode = self._mode
            self._mode = mode
        if old_mode != mode:
            print(f"[IMX500] Mode: {mode.value}")

    def set_track_target(self, target: str) -> bool:
        """Set tracking target"""
        # Handle Thai class names
        if target in self.THAI_CLASS_MAP:
            target = self.THAI_CLASS_MAP[target]

        # Validate
        if target not in self.COCO_LABELS:
            print(f"[IMX500] Warning: '{target}' is not a valid class")
            return False

        with self._lock:
            old = self._track_target
            self._track_target = target
        if old != target:
            print(f"[IMX500] Track target: {target}")
        return True

    def get_detection_text(self, max_items: int = 3) -> List[str]:
        """Get detection labels as text"""
        with self._lock:
            return self._cached_labels[:max_items]

    def get_normalized_position(self) -> Tuple[float, float]:
        """Get normalized tracked position (-1 to 1)"""
        with self._lock:
            return self._cached_position

    def get_mode_text(self) -> str:
        """Get mode description text"""
        with self._lock:
            mode = self._mode
            target = self._track_target
        if mode == DetectionMode.DETECT:
            return "DETECT"
        elif mode == DetectionMode.TRACK:
            return f"TRACK: {target}"
        elif mode == DetectionMode.TRACK_ALL:
            return "TRACK ALL"
        return ""

    def cleanup(self):
        """Clean up resources"""
        self.stop()


class DummyIMX500Detector:
    """Dummy detector when IMX500 is not available"""

    def __init__(self, *args, **kwargs):
        print("[IMX500] Using dummy detector")
        self._mode = DetectionMode.DETECT
        self._track_target = "person"

    def start(self) -> bool:
        return False

    def stop(self):
        pass

    @property
    def detections(self):
        return []

    @property
    def fps_actual(self):
        return 0

    @property
    def running(self):
        return False

    @property
    def mode(self):
        return self._mode

    @property
    def track_target(self):
        return self._track_target

    def set_mode(self, mode):
        self._mode = mode

    def set_track_target(self, target):
        self._track_target = target
        return True

    def get_detection_text(self, max_items=3):
        return []

    def get_normalized_position(self):
        return (0, 0)

    def get_mode_text(self):
        return ""

    def cleanup(self):
        pass


def create_imx500_detector(**kwargs) -> "IMX500Detector":
    """Factory function to create detector"""
    if IMX500_AVAILABLE:
        return IMX500Detector(**kwargs)
    else:
        return DummyIMX500Detector(**kwargs)


# --- CLI Test ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IMX500 Object Detection Test")
    parser.add_argument("--model", default="ssd_mobilenet", help="Model name or path")
    parser.add_argument("--threshold", type=float, default=0.55, help="Confidence threshold")
    parser.add_argument("--fps", type=int, default=10, help="Target FPS")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable pre-installed models:")
        for name, path in IMX500Detector.AVAILABLE_MODELS.items():
            print(f"  - {name}: {path}")
        exit(0)

    print("=" * 60)
    print("IMX500 Object Detection Test (Picamera2 API)")
    print("=" * 60)

    detector = create_imx500_detector(
        model=args.model,
        threshold=args.threshold,
        fps=args.fps
    )

    if detector.start():
        print("\nPress Ctrl+C to stop\n")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nInterrupted")
    else:
        print("Failed to start detector")

    detector.cleanup()
    print("Done")
