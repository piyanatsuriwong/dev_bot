#!/usr/bin/env python3
"""
YOLO Object Detection Tracker v2 for IMX500 AI Camera
Uses modlib as PRIMARY backend (best compatibility with YOLO11n/YOLOv8n)

Features:
- Uses modlib AiCamera for reliable detection
- Picamera2 as fallback option
- Same API as original yolo_tracker.py for compatibility

Modes:
- DETECT: Show detected objects as text
- TRACK: Track specific object (person by default) with eyes
"""

import threading
import time
import gc
import os
from enum import Enum
from functools import lru_cache
from typing import Optional, List, Dict, Tuple, Any

# Backend detection
YOLO_AVAILABLE = False
BACKEND = "none"
model_name = "None"
YOLO_MODEL_CLASS = None

# Try modlib FIRST (primary backend - works best!)
try:
    from modlib.devices import AiCamera
    from modlib.apps import Annotator
    import numpy as np
    import cv2

    # Try YOLO11n first, fallback to YOLOv8n
    try:
        from modlib.models.zoo import YOLO11n
        YOLO_MODEL_CLASS = YOLO11n
        model_name = "YOLO11n"
    except ImportError:
        try:
            from modlib.models.zoo import YOLOv8n
            YOLO_MODEL_CLASS = YOLOv8n
            model_name = "YOLOv8n"
        except ImportError:
            pass

    if YOLO_MODEL_CLASS is not None:
        YOLO_AVAILABLE = True
        BACKEND = "modlib"
        print(f"YOLO (IMX500): Available - Using modlib ({model_name})")
except ImportError as e:
    print(f"YOLO (IMX500): modlib not available - {e}")

# Fallback to Picamera2 if modlib not available
if not YOLO_AVAILABLE:
    try:
        from picamera2 import Picamera2
        from picamera2.devices import IMX500
        from picamera2.devices.imx500 import (
            NetworkIntrinsics,
            postprocess_nanodet_detection
        )
        import numpy as np
        import cv2
        YOLO_AVAILABLE = True
        BACKEND = "picamera2"
        print("YOLO (IMX500): Available - Using Picamera2 Direct API (fallback)")
    except ImportError as e:
        print(f"YOLO (IMX500): Picamera2 IMX500 not available - {e}")


class YoloMode(Enum):
    """YOLO operation modes"""
    DETECT = "detect"
    TRACK = "track"
    TRACK_ALL = "track_all"


class YoloTracker:
    """
    YOLO Object Detection using IMX500 AI Camera
    Supports both Picamera2 Direct API and modlib backends
    """

    # Available pre-installed models for Picamera2 backend
    AVAILABLE_MODELS = {
        "ssd_mobilenet": "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk",
        "yolov8n": "/home/pi/.modlib/zoo/imx500_network_yolov8n_pp.rpk",
        "yolo11n": "/home/pi/.modlib/zoo/imx500_network_yolo11n_pp.rpk",
        "nanodet": "/usr/share/imx500-models/imx500_network_nanodet_plus_416x416_pp.rpk",
        "efficientdet": "/usr/share/imx500-models/imx500_network_efficientdet_lite0_pp.rpk",
    }

    COCO_CLASSES = [
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
        "แล็ปท็อป": "laptop", "คอมพิวเตอร์": "laptop",
        "ทีวี": "tv", "โทรทัศน์": "tv", "เก้าอี้": "chair",
        "โซฟา": "couch", "หนังสือ": "book", "นาฬิกา": "clock",
        "กรรไกร": "scissors", "ตุ๊กตาหมี": "teddy bear",
    }

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        frame_rate: int = 10,
        model: str = "yolov8n",
        use_modlib: bool = False
    ):
        """
        Initialize YOLO Tracker

        Args:
            confidence_threshold: Detection confidence threshold
            frame_rate: Target frame rate
            model: Model name or path (for Picamera2 backend)
            use_modlib: Force use of modlib backend
        """
        self.confidence_threshold = confidence_threshold
        self.frame_rate = frame_rate
        self.model_name = model

        # Determine which model path to use
        if model in self.AVAILABLE_MODELS:
            self.model_path = self.AVAILABLE_MODELS[model]
        else:
            self.model_path = model

        # Backend selection (modlib is preferred)
        self._use_modlib = use_modlib
        self._backend = BACKEND  # Will be "modlib" if available (primary)

        # Backend-specific objects
        # For Picamera2 (fallback)
        self.imx500 = None
        self.picam2 = None
        self.intrinsics = None

        # For modlib (primary)
        self.device = None
        self.model = None
        self.annotator = None
        self.stream = None

        # State
        self._mode = YoloMode.DETECT
        self._track_target = "person"
        self._show_text = True
        self._labels: List[str] = self.COCO_CLASSES

        # Thread control
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Results cache
        self._detections: List[Dict] = []
        self._latest_frame = None
        self._tracked_position = None
        self._cached_position = (0, 0)
        self._cached_labels: List[str] = []

        # FPS tracking
        self._fps = 0
        self._last_time = time.time()
        self._frame_count = 0

        # Callbacks
        self._on_mode_change = None
        self._on_detection = None

        # Initialize based on backend
        if YOLO_AVAILABLE:
            if self._backend == "modlib":
                self._init_modlib()
            elif self._backend == "picamera2":
                self._init_picamera2()

    def _init_picamera2(self):
        """Initialize using Picamera2 Direct API"""
        try:
            print(f"[YOLO] Initializing with Picamera2 API")
            print(f"[YOLO] Model: {self.model_name}")

            # Check if model exists
            if not os.path.exists(self.model_path):
                # Fallback to SSD MobileNet which is usually pre-installed
                print(f"[YOLO] Model not found at {self.model_path}")
                self.model_path = self.AVAILABLE_MODELS.get(
                    "ssd_mobilenet",
                    "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
                )
                print(f"[YOLO] Falling back to: {self.model_path}")

            # Create IMX500 device
            self.imx500 = IMX500(self.model_path)

            # Get network intrinsics
            self.intrinsics = self.imx500.network_intrinsics
            if not self.intrinsics:
                self.intrinsics = NetworkIntrinsics()
                self.intrinsics.task = "object detection"

            self.intrinsics.update_with_defaults()

            # Get labels
            if self.intrinsics.labels:
                self._labels = list(self.intrinsics.labels)
            else:
                self._labels = self.COCO_CLASSES

            print(f"[YOLO] Labels: {len(self._labels)} classes")

            # Get camera index
            try:
                import config
                cam_idx = getattr(config, 'CAMERA_IMX500_NUM', self.imx500.camera_num)
            except ImportError:
                cam_idx = self.imx500.camera_num

            print(f"[YOLO] Camera index: {cam_idx}")

            # Create Picamera2 instance
            self.picam2 = Picamera2(cam_idx)
            print("[YOLO] Picamera2 initialized!")

        except Exception as e:
            print(f"[YOLO] Picamera2 init error: {e}")
            self.imx500 = None
            self.picam2 = None

    def _init_modlib(self):
        """Initialize using modlib (PRIMARY backend)"""
        try:
            from modlib.devices import AiCamera

            # Get camera index from config
            try:
                import config
                cam_idx = getattr(config, 'CAMERA_IMX500_NUM', 0)
            except ImportError:
                cam_idx = 0

            print(f"[YOLO] Loading {model_name} model...")
            self.model = YOLO_MODEL_CLASS()
            self._labels = list(self.model.labels) if hasattr(self.model, 'labels') else self.COCO_CLASSES
            print(f"[YOLO] Model loaded! Classes: {len(self._labels)}")

            print(f"[YOLO] Initializing AiCamera on index {cam_idx}...")
            self.device = AiCamera(frame_rate=self.frame_rate, num=cam_idx)

            print(f"[YOLO] Deploying model to IMX500...")
            self.device.deploy(self.model)
            print(f"[SUCCESS] IMX500 initialized on camera {cam_idx}")
            print("[YOLO] Ready!")

        except Exception as e:
            print(f"[YOLO] modlib init error: {e}")
            self.device = None

    def start(self) -> bool:
        """Start the detection thread"""
        if not YOLO_AVAILABLE:
            print("[YOLO] Not available, cannot start")
            return False

        if self._backend == "picamera2":
            return self._start_picamera2()
        else:
            return self._start_modlib()

    def _start_picamera2(self) -> bool:
        """Start using Picamera2 backend"""
        if self.picam2 is None or self.imx500 is None:
            print("[YOLO] Picamera2 not initialized")
            return False

        try:
            # Show firmware upload progress
            print("[YOLO] Uploading firmware to IMX500...")
            self.imx500.show_network_fw_progress_bar()

            # Configure and start camera
            inference_rate = self.frame_rate
            if self.intrinsics and self.intrinsics.inference_rate:
                inference_rate = min(self.intrinsics.inference_rate, self.frame_rate)

            config = self.picam2.create_preview_configuration(
                controls={"FrameRate": inference_rate},
                buffer_count=12
            )
            self.picam2.start(config, show_preview=False)

            if self.intrinsics and self.intrinsics.preserve_aspect_ratio:
                self.imx500.set_auto_aspect_ratio()

            print("[YOLO] Camera started")

            # Start detection thread
            self._running = True
            self._thread = threading.Thread(target=self._detection_loop_picamera2, daemon=True)
            self._thread.start()
            print("[YOLO] Detection thread started")
            return True

        except Exception as e:
            print(f"[YOLO] Start error: {e}")
            return False

    def _start_modlib(self) -> bool:
        """Start using modlib backend"""
        if self.device is None:
            print("[YOLO] modlib device not initialized")
            return False

        # Reset FPS counters
        self._fps = 0
        self._frame_count = 0
        self._last_time = time.time()

        self._running = True
        self._thread = threading.Thread(target=self._detection_loop_modlib, daemon=True)
        self._thread.start()
        print("[YOLO] Detection thread started (modlib)")
        return True

    def stop(self):
        """Stop the detection thread"""
        if self._running:
            print("[YOLO] Stopping detection thread...")
            self._running = False
            if self._thread:
                self._thread.join(timeout=3.0)
                if self._thread.is_alive():
                    print("[YOLO] Warning - thread did not stop in time")
                else:
                    print("[YOLO] Thread stopped")

    def _detection_loop_picamera2(self):
        """Detection loop for Picamera2 backend"""
        print("[YOLO] Picamera2 detection loop started")

        while self._running:
            try:
                # Capture metadata
                metadata = self.picam2.capture_metadata()

                # Parse detections
                detections = self._parse_detections_picamera2(metadata)

                # Process detections
                with self._lock:
                    current_mode = self._mode
                    track_target = self._track_target

                # Get frame dimensions
                frame_w, frame_h = 640, 480
                if self.imx500:
                    try:
                        input_w, input_h = self.imx500.get_input_size()
                        frame_w, frame_h = input_w, input_h
                    except:
                        pass

                # Calculate tracking position
                processed = []
                tracked_pos = None

                for det in detections:
                    label = self._get_label(det.get("category", 0))
                    processed.append({
                        "label": label,
                        "confidence": det.get("confidence", 0),
                        "bbox": det.get("bbox", (0, 0, 0, 0))
                    })

                    if tracked_pos is None:
                        should_track = False
                        if current_mode == YoloMode.TRACK:
                            should_track = (label == track_target)
                        elif current_mode == YoloMode.TRACK_ALL:
                            should_track = True

                        if should_track:
                            bbox = det.get("bbox", (0, 0, 0, 0))
                            x, y, w, h = bbox
                            cx = (x + w / 2) / frame_w
                            cy = (y + h / 2) / frame_h
                            tracked_pos = (cx, cy)

                # Update cached position
                if tracked_pos is not None:
                    cached_pos = ((tracked_pos[0] - 0.5) * 2, (tracked_pos[1] - 0.5) * 2)
                else:
                    cached_pos = (0, 0)

                sorted_labels = [d["label"] for d in sorted(processed, key=lambda x: x["confidence"], reverse=True)[:3]]

                with self._lock:
                    self._detections = processed
                    self._tracked_position = tracked_pos

                self._cached_position = cached_pos
                self._cached_labels = sorted_labels

                # FPS tracking
                self._frame_count += 1
                now = time.time()
                if now - self._last_time >= 1.0:
                    self._fps = self._frame_count
                    self._frame_count = 0
                    self._last_time = now

                    if processed:
                        det_str = ", ".join([
                            f"{d['label']} ({d['confidence']:.2f})"
                            for d in processed[:3]
                        ])
                        print(f"[YOLO] Detected: {det_str} | FPS: {self._fps}")
                    else:
                        print(f"[YOLO] No detections | FPS: {self._fps}")

            except Exception as e:
                if self._running:
                    print(f"[YOLO] Detection error: {e}")
                    time.sleep(0.1)

        print("[YOLO] Detection loop ended")

    def _parse_detections_picamera2(self, metadata: dict) -> List[Dict]:
        """Parse detections from Picamera2 metadata with robust error handling"""
        try:
            # Try to get outputs - may fail if tensor size mismatch
            try:
                np_outputs = self.imx500.get_outputs(metadata, add_batch=True)
            except Exception as output_err:
                # Tensor size mismatch - try without add_batch
                try:
                    np_outputs = self.imx500.get_outputs(metadata, add_batch=False)
                except:
                    # Still failing - skip this frame
                    return []

            if np_outputs is None:
                return []

            input_w, input_h = self.imx500.get_input_size()

            # Handle different postprocess methods
            boxes = None
            scores = None
            classes = None

            if self.intrinsics and self.intrinsics.postprocess == "nanodet":
                try:
                    boxes, scores, classes = postprocess_nanodet_detection(
                        outputs=np_outputs[0],
                        conf=self.confidence_threshold,
                        iou_thres=0.65,
                        max_out_dets=10
                    )[0]
                    from picamera2.devices.imx500.postprocess import scale_boxes
                    boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
                except Exception as e:
                    # Fallback to standard parsing
                    pass

            # Standard parsing for most models
            if boxes is None:
                try:
                    # Try standard 3-output format [boxes, scores, classes]
                    if len(np_outputs) >= 3:
                        boxes_raw = np_outputs[0]
                        scores_raw = np_outputs[1]
                        classes_raw = np_outputs[2]

                        # Handle batch dimension
                        if len(boxes_raw.shape) == 3:
                            boxes_raw = boxes_raw[0]
                        if len(scores_raw.shape) == 2:
                            scores_raw = scores_raw[0]
                        if len(classes_raw.shape) == 2:
                            classes_raw = classes_raw[0]

                        boxes = boxes_raw
                        scores = scores_raw
                        classes = classes_raw

                    # Try combined output format [N, 6] where 6 = [x1,y1,x2,y2,score,class]
                    elif len(np_outputs) == 1:
                        output = np_outputs[0]
                        if len(output.shape) == 3:
                            output = output[0]
                        if output.shape[-1] >= 6:
                            boxes = output[:, :4]
                            scores = output[:, 4]
                            classes = output[:, 5]
                        elif output.shape[-1] == 5:
                            # [x1,y1,x2,y2,score] format - default class 0 (person)
                            boxes = output[:, :4]
                            scores = output[:, 4]
                            classes = np.zeros(len(scores))

                except Exception as parse_err:
                    # Last resort - return empty
                    return []

            if boxes is None or scores is None or classes is None:
                return []

            # Apply normalization if needed
            if self.intrinsics and self.intrinsics.bbox_normalization:
                boxes = boxes / input_h

            # Apply bbox order swap if needed
            if self.intrinsics and self.intrinsics.bbox_order == "xy":
                if hasattr(boxes, 'shape') and len(boxes.shape) == 2:
                    boxes = boxes[:, [1, 0, 3, 2]]

            # Convert boxes to list format if needed
            if hasattr(boxes, 'shape') and len(boxes.shape) == 2:
                boxes = [tuple(box) for box in boxes]
            elif hasattr(boxes, '__iter__') and not isinstance(boxes, (list, tuple)):
                boxes = list(boxes)

            # Build detection list
            detections = []
            for i, (box, score, category) in enumerate(zip(boxes, scores, classes)):
                try:
                    score_val = float(score)
                    if score_val > self.confidence_threshold:
                        # Convert box format to (x, y, w, h)
                        if isinstance(box, (list, tuple, np.ndarray)) and len(box) >= 4:
                            x1 = float(box[0])
                            y1 = float(box[1])
                            x2 = float(box[2])
                            y2 = float(box[3])
                            w = x2 - x1
                            h = y2 - y1
                            bbox = (x1, y1, w, h)
                        else:
                            continue

                        detections.append({
                            "category": int(category),
                            "confidence": score_val,
                            "bbox": bbox
                        })
                except (ValueError, TypeError, IndexError):
                    continue

            return detections[:10]

        except Exception as e:
            # Only print error occasionally to avoid spam
            if not hasattr(self, '_last_parse_error_time'):
                self._last_parse_error_time = 0
            now = time.time()
            if now - self._last_parse_error_time > 5.0:
                print(f"[YOLO] Parse error: {e}")
                self._last_parse_error_time = now
            return []

    def _detection_loop_modlib(self):
        """Detection loop for modlib backend (PRIMARY)"""
        print("[YOLO] modlib detection loop starting...")

        while self._running:
            try:
                if self.device is None:
                    print("[YOLO] Device is None, waiting...")
                    time.sleep(2.0)
                    continue

                print("[YOLO] Acquiring camera stream...")

                try:
                    with self.device as stream:
                        print("[YOLO] Stream started!")
                        # Reset FPS counters for accurate measurement
                        self._frame_count = 0
                        self._last_time = time.time()
                        last_frame_id = None
                        for frame in stream:
                            if not self._running:
                                break

                            frame_image = frame.image
                            if frame_image is None:
                                continue

                            # Check if this is a new frame using frame ID
                            frame_id = id(frame)
                            if frame_id == last_frame_id:
                                time.sleep(0.001)  # Small delay to prevent busy loop
                                continue
                            last_frame_id = frame_id

                            with self._lock:
                                current_mode = self._mode
                                track_target = self._track_target

                            # Get detections from modlib
                            try:
                                detections = frame.detections[
                                    frame.detections.confidence > self.confidence_threshold
                                ]
                            except (TypeError, AttributeError):
                                detections = []

                            frame_h, frame_w = frame_image.shape[:2] if frame_image is not None else (480, 640)

                            processed = []
                            tracked_pos = None

                            for det in detections:
                                _, confidence, class_id, bbox = det
                                class_id = int(class_id)
                                try:
                                    label = self._labels[class_id] if class_id < len(self._labels) else f"class_{class_id}"
                                except:
                                    label = f"class_{class_id}"

                                processed.append({
                                    "label": label,
                                    "confidence": float(confidence),
                                    "bbox": bbox
                                })

                                if tracked_pos is None:
                                    should_track = False
                                    if current_mode == YoloMode.TRACK:
                                        should_track = (label == track_target)
                                    elif current_mode == YoloMode.TRACK_ALL:
                                        should_track = True

                                    if should_track:
                                        x1, y1, x2, y2 = bbox
                                        cx = ((x1 + x2) / 2) / frame_w
                                        cy = ((y1 + y2) / 2) / frame_h
                                        tracked_pos = (cx, cy)

                            if tracked_pos is not None:
                                cached_pos = ((tracked_pos[0] - 0.5) * 2, (tracked_pos[1] - 0.5) * 2)
                            else:
                                cached_pos = (0, 0)

                            sorted_labels = [d["label"] for d in sorted(processed, key=lambda x: x["confidence"], reverse=True)[:3]]

                            with self._lock:
                                self._detections = processed
                                self._latest_frame = frame_image
                                self._tracked_position = tracked_pos

                            self._cached_position = cached_pos
                            self._cached_labels = sorted_labels

                            # FPS tracking - count every unique frame
                            self._frame_count += 1
                            now = time.time()
                            if now - self._last_time >= 1.0:
                                self._fps = self._frame_count
                                self._frame_count = 0
                                self._last_time = now

                                # Log detections
                                if processed:
                                    det_str = ", ".join([
                                        f"{d['label']} ({d['confidence']:.2f})"
                                        for d in processed[:3]
                                    ])
                                    print(f"[YOLO] {det_str} | FPS: {self._fps}")
                                else:
                                    print(f"[YOLO] No detections | FPS: {self._fps}")

                except Exception as stream_error:
                    print(f"[YOLO] Stream error: {stream_error}")
                    time.sleep(3.0)

            except Exception as e:
                if not self._running:
                    break
                print(f"[YOLO] Error: {e}")
                time.sleep(3.0)

        print("[YOLO] Detection loop ended")

    @lru_cache(maxsize=100)
    def _get_label(self, category: int) -> str:
        """Get label for category index"""
        try:
            if self.intrinsics and self.intrinsics.ignore_dash_labels:
                labels = [l for l in self._labels if l and l != "-"]
            else:
                labels = self._labels
            return labels[category] if category < len(labels) else f"class_{category}"
        except:
            return f"class_{category}"

    def cleanup(self):
        """Clean up resources"""
        print("[YOLO] Cleaning up...")
        self.stop()

        if self._backend == "picamera2":
            if self.picam2:
                try:
                    self.picam2.stop()
                    self.picam2.close()
                except:
                    pass
            self.picam2 = None
            self.imx500 = None
        else:
            if self.device:
                try:
                    if hasattr(self.device, 'close'):
                        self.device.close()
                    elif hasattr(self.device, '__exit__'):
                        self.device.__exit__(None, None, None)
                except:
                    pass
            self.device = None

        self.model = None
        gc.collect()
        print("[YOLO] Cleanup complete")

    # --- Properties ---

    @property
    def detections(self) -> List[Dict]:
        with self._lock:
            return self._detections.copy()

    @property
    def latest_frame(self):
        return self._latest_frame

    @property
    def tracked_position(self):
        with self._lock:
            return self._tracked_position

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def running(self) -> bool:
        return self._running

    @property
    def mode(self) -> YoloMode:
        with self._lock:
            return self._mode

    @property
    def track_target(self) -> str:
        with self._lock:
            return self._track_target

    @property
    def show_text(self) -> bool:
        with self._lock:
            return self._show_text

    # --- Public Methods ---

    def set_mode(self, mode: YoloMode):
        with self._lock:
            old_mode = self._mode
            self._mode = mode
        if old_mode != mode:
            print(f"[YOLO] Mode: {mode.value}")
            if self._on_mode_change:
                self._on_mode_change(mode)

    def set_track_target(self, target: str) -> bool:
        if target in self.THAI_CLASS_MAP:
            target = self.THAI_CLASS_MAP[target]

        if target not in self.COCO_CLASSES:
            print(f"[YOLO] Warning: '{target}' is not a valid class")
            return False

        with self._lock:
            old = self._track_target
            self._track_target = target

        if old != target:
            print(f"[YOLO] Track target: {target}")
        return True

    def set_show_text(self, show: bool):
        with self._lock:
            self._show_text = show

    def set_on_mode_change(self, callback):
        self._on_mode_change = callback

    def get_detection_text(self, max_items: int = 3) -> List[str]:
        if not self._show_text:
            return []
        return self._cached_labels[:max_items]

    def get_mode_text(self) -> str:
        with self._lock:
            mode = self._mode
            target = self._track_target

        if mode == YoloMode.DETECT:
            return "DETECT"
        elif mode == YoloMode.TRACK:
            return f"TRACK: {target}"
        elif mode == YoloMode.TRACK_ALL:
            return "TRACK ALL"
        return ""

    def get_normalized_position(self) -> Tuple[float, float]:
        return self._cached_position

    def process_voice_command(self, text: str) -> bool:
        text_lower = text.lower()

        if "ตรวจจับ" in text or "detect" in text_lower:
            self.set_mode(YoloMode.DETECT)
            return True

        if "ติดตาม" in text or "track" in text_lower or "follow" in text_lower:
            for thai, english in self.THAI_CLASS_MAP.items():
                if thai in text:
                    self.set_track_target(english)
                    self.set_mode(YoloMode.TRACK)
                    return True

            for cls in self.COCO_CLASSES:
                if cls in text_lower:
                    self.set_track_target(cls)
                    self.set_mode(YoloMode.TRACK)
                    return True

            self.set_mode(YoloMode.TRACK)
            return True

        for thai, english in self.THAI_CLASS_MAP.items():
            if thai in text:
                self.set_track_target(english)
                self.set_mode(YoloMode.TRACK)
                return True

        if "ซ่อน" in text or "hide" in text_lower:
            self.set_show_text(False)
            return True

        if "แสดง" in text or "show" in text_lower:
            self.set_show_text(True)
            return True

        return False


class DummyYoloTracker:
    """Dummy tracker when YOLO is not available"""

    def __init__(self, *args, **kwargs):
        print("[YOLO] Using dummy tracker (no AI Camera)")
        self._mode = YoloMode.DETECT
        self._track_target = "person"
        self._show_text = True

    def start(self) -> bool:
        return False

    def stop(self):
        pass

    @property
    def detections(self):
        return []

    @property
    def latest_frame(self):
        return None

    @property
    def tracked_position(self):
        return None

    @property
    def fps(self):
        return 0

    @property
    def mode(self):
        return self._mode

    @property
    def track_target(self):
        return self._track_target

    @property
    def show_text(self):
        return self._show_text

    def set_mode(self, mode):
        self._mode = mode

    def set_track_target(self, target):
        self._track_target = target
        return True

    def set_show_text(self, show):
        self._show_text = show

    def set_on_mode_change(self, callback):
        pass

    def get_detection_text(self, max_items=3):
        return []

    def get_mode_text(self):
        return ""

    def get_normalized_position(self):
        return (0, 0)

    def process_voice_command(self, text):
        return False

    def cleanup(self):
        pass


def create_yolo_tracker(
    confidence_threshold: float = None,
    frame_rate: int = None,
    model: str = "yolov8n",
    use_modlib: bool = False
) -> "YoloTracker":
    """
    Factory function to create YOLO tracker

    Args:
        confidence_threshold: Detection threshold (default from config)
        frame_rate: Target FPS (default from config)
        model: Model name or path (yolov8n, yolo11n, ssd_mobilenet, etc.)
        use_modlib: Force use of modlib backend instead of Picamera2
    """
    try:
        import config
        if confidence_threshold is None:
            confidence_threshold = getattr(config, 'YOLO_CONFIDENCE_THRESHOLD', 0.5)
        if frame_rate is None:
            frame_rate = getattr(config, 'YOLO_FRAME_RATE', 5)
    except ImportError:
        if confidence_threshold is None:
            confidence_threshold = 0.5
        if frame_rate is None:
            frame_rate = 5

    if YOLO_AVAILABLE:
        return YoloTracker(
            confidence_threshold=confidence_threshold,
            frame_rate=frame_rate,
            model=model,
            use_modlib=use_modlib
        )
    else:
        return DummyYoloTracker()


__all__ = ['YoloTracker', 'DummyYoloTracker', 'YoloMode', 'create_yolo_tracker', 'YOLO_AVAILABLE', 'BACKEND']


# --- CLI Test ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLO Tracker v2 Test")
    parser.add_argument("--model", default="yolov8n", help="Model name (yolov8n, ssd_mobilenet, etc.)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--fps", type=int, default=10, help="Target FPS")
    parser.add_argument("--modlib", action="store_true", help="Force use modlib backend")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable models:")
        for name, path in YoloTracker.AVAILABLE_MODELS.items():
            exists = "✅" if os.path.exists(path) else "❌"
            print(f"  {exists} {name}: {path}")
        exit(0)

    print("=" * 60)
    print(f"YOLO Tracker v2 - Backend: {BACKEND}")
    print("=" * 60)

    tracker = create_yolo_tracker(
        confidence_threshold=args.threshold,
        frame_rate=args.fps,
        model=args.model,
        use_modlib=args.modlib
    )

    if tracker.start():
        print("\nPress Ctrl+C to stop\n")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nInterrupted")
    else:
        print("Failed to start tracker")

    tracker.cleanup()
    print("Done")
