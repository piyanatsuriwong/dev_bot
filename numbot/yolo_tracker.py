#!/usr/bin/env python3
"""
YOLO Object Detection Tracker for IMX500 AI Camera
Combines with RoboEyes hand tracking system

Modes:
- DETECT: Show detected objects as text
- TRACK: Track specific object (person by default) with eyes
"""

import threading
import time
import gc
import weakref
from enum import Enum

# Try to import modlib (AI Camera library)
YOLO_AVAILABLE = False
YOLO_MODEL_CLASS = None
model_name = "None"

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
        from modlib.models.zoo import YOLOv8n
        YOLO_MODEL_CLASS = YOLOv8n
        model_name = "YOLOv8n"

    YOLO_AVAILABLE = True
    print(f"YOLO (IMX500): Available - Using {model_name} model")
except ImportError as e:
    print(f"YOLO (IMX500): Not available - {e}")


class YoloMode(Enum):
    """YOLO operation modes"""
    DETECT = "detect"
    TRACK = "track"
    TRACK_ALL = "track_all"


class YoloTracker:
    """
    YOLO Object Detection using IMX500 AI Camera
    Runs in a separate thread for non-blocking detection
    """

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
        "คน": "person",
        "แมว": "cat",
        "หมา": "dog",
        "สุนัข": "dog",
        "นก": "bird",
        "รถ": "car",
        "รถยนต์": "car",
        "มอเตอร์ไซค์": "motorcycle",
        "จักรยาน": "bicycle",
        "ขวด": "bottle",
        "แก้ว": "cup",
        "โทรศัพท์": "cell phone",
        "มือถือ": "cell phone",
        "แล็ปท็อป": "laptop",
        "คอมพิวเตอร์": "laptop",
        "ทีวี": "tv",
        "โทรทัศน์": "tv",
        "เก้าอี้": "chair",
        "โซฟา": "couch",
        "หนังสือ": "book",
        "นาฬิกา": "clock",
        "กรรไกร": "scissors",
        "ตุ๊กตาหมี": "teddy bear",
    }

    def __init__(self, confidence_threshold=0.5, frame_rate=10):
        self.confidence_threshold = confidence_threshold
        self.frame_rate = frame_rate

        self.device = None
        self.model = None
        self.annotator = None
        self.stream = None

        self._mode = YoloMode.DETECT
        self._track_target = "person"
        self._show_text = True

        self._lock = threading.Lock()
        self._detections = []
        self._latest_frame = None
        self._tracked_position = None

        self._cached_position = (0, 0)
        self._cached_labels = []

        self._running = False
        self._thread = None

        self._fps = 0
        self._last_time = time.time()
        self._frame_count = 0

        self._on_mode_change = None
        self._on_detection = None

        self._finalizer = weakref.finalize(self, self._finalize_cleanup, weakref.ref(self))

        if YOLO_AVAILABLE:
            self._init_camera()

    @staticmethod
    def _finalize_cleanup(self_ref):
        try:
            self = self_ref() if self_ref else None
            if self is not None and hasattr(self, 'device'):
                device = self.device
                if device is not None:
                    try:
                        if hasattr(device, 'close'):
                            device.close()
                        elif hasattr(device, '__exit__'):
                            device.__exit__(None, None, None)
                    except Exception:
                        pass
        except Exception:
            pass

    def _init_camera(self):
        try:
            import config
            cam_idx = getattr(config, 'CAMERA_IMX500_NUM', 1)

            print(f"Loading {model_name} model...")
            self.model = YOLO_MODEL_CLASS()
            print(f"Model loaded! Classes: {len(self.model.labels)}")

            print(f"Initializing AI Camera (IMX500) on index {cam_idx}...")
            time.sleep(0.5)

            try:
                self.device = AiCamera(frame_rate=self.frame_rate, num=cam_idx)
                print(f"Deploying model to IMX500...")
                self.device.deploy(self.model)
                self.annotator = Annotator()
                print(f"[SUCCESS] IMX500 initialized on camera {cam_idx}")

            except Exception as e:
                print(f"[ERROR] Failed to initialize IMX500: {e}")
                self.device = None
                return

            print("YOLO ready!")
            time.sleep(1.0)
            print("YOLO: Device ready for acquisition")

        except Exception as e:
            print(f"YOLO init error: {e}")

    def start(self):
        if not YOLO_AVAILABLE or self.device is None:
            print("YOLO not available, cannot start")
            return False

        if self.device is not None:
            time.sleep(0.5)

        self._running = True
        self._thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._thread.start()
        print("YOLO: Detection thread started")
        return True

    def stop(self):
        if self._running:
            print("YOLO: Stopping detection thread...")
            self._running = False
            if self._thread:
                self._thread.join(timeout=3.0)
                if self._thread.is_alive():
                    print("YOLO: Warning - thread did not stop in time")
                else:
                    print("YOLO: Thread stopped")

    def _cleanup_device(self):
        try:
            if self.device is not None:
                device_ref = self.device
                try:
                    if hasattr(device_ref, 'close'):
                        device_ref.close()
                        print("YOLO: Device closed via close()")
                    elif hasattr(device_ref, '__exit__'):
                        device_ref.__exit__(None, None, None)
                        print("YOLO: Device closed via __exit__()")
                except Exception as close_error:
                    print(f"YOLO: Error closing device: {close_error}")

                self.device = None
                collected = gc.collect()
                if collected > 0:
                    print(f"YOLO: GC collected {collected} objects")
                time.sleep(1.0)
                print("YOLO: Device cleaned up")
        except Exception as e:
            print(f"YOLO cleanup warning: {e}")

    def _detection_loop(self):
        while self._running:
            try:
                if self.device is None:
                    print("YOLO: Device is None, attempting to initialize...")
                    try:
                        self._init_camera()
                    except Exception as e:
                        print(f"YOLO: Init failed ({e}), waiting 2s...")
                        time.sleep(2.0)
                        continue

                print("YOLO: Acquiring camera stream...")

                try:
                    time.sleep(1.0)

                    with self.device as stream:
                        print("YOLO: Stream acquired! Starting process...")

                        for frame in stream:
                            if not self._running:
                                break

                            start_time = time.time()
                            frame_image = frame.image

                            if frame_image is None:
                                print("YOLO: Warning - Received empty frame")
                                time.sleep(0.1)
                                continue

                            with self._lock:
                                current_mode = self._mode
                                track_target = self._track_target

                            try:
                                detections = frame.detections[
                                    frame.detections.confidence > self.confidence_threshold
                                ]
                            except (TypeError, AttributeError):
                                detections = []

                            if frame_image is not None:
                                frame_h, frame_w = frame_image.shape[:2]
                            else:
                                frame_h, frame_w = 480, 640

                            processed = []
                            tracked_pos = None

                            for det in detections:
                                _, confidence, class_id, bbox = det
                                class_id = int(class_id)
                                try:
                                    label = self.model.labels[class_id]
                                except:
                                    label = "unknown"

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

                            self._frame_count += 1
                            now = time.time()
                            if now - self._last_time >= 1.0:
                                self._fps = self._frame_count
                                if self._fps > 100:
                                    print(f"YOLO: [WARNING] FPS anomaly ({self._fps})")
                                else:
                                    if len(processed) > 0:
                                        top_detections = sorted(processed, key=lambda x: x["confidence"], reverse=True)[:3]
                                        detection_str = ", ".join([
                                            f"{d['label']} ({d['confidence']:.2f})"
                                            for d in top_detections
                                        ])
                                        print(f"[YOLO] Detected: {detection_str} | FPS: {self._fps}")
                                    else:
                                        print(f"[YOLO] No detections | FPS: {self._fps}")

                                self._frame_count = 0
                                self._last_time = now

                except Exception as stream_error:
                    error_str = str(stream_error)
                    if "timeout" in error_str.lower() or "timed out" in error_str.lower():
                        print(f"YOLO: Stream timeout detected! Error: {stream_error}")
                    else:
                        print(f"YOLO: Stream crashed! Error: {stream_error}")
                    print("YOLO: Initiating Auto-Recovery...")
                    raise stream_error

            except Exception as e:
                if not self._running:
                    break

                print(f"YOLO: Critical Error detected: {e}")
                print("YOLO: Performing Hard Reset...")

                self._cleanup_device()
                self.device = None

                print("YOLO: Cooling down (3s)...")
                time.sleep(3.0)
                continue

        print("YOLO: Thread exiting...")

    @property
    def detections(self):
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
    def fps(self):
        return self._fps

    @property
    def running(self):
        return self._running and self.device is not None

    @property
    def mode(self):
        with self._lock:
            return self._mode

    @property
    def track_target(self):
        with self._lock:
            return self._track_target

    @property
    def show_text(self):
        with self._lock:
            return self._show_text

    def set_mode(self, mode):
        with self._lock:
            old_mode = self._mode
            self._mode = mode
        if old_mode != mode:
            print(f"YOLO Mode: {mode.value}")
            if self._on_mode_change:
                self._on_mode_change(mode)

    def set_track_target(self, target):
        if target in self.THAI_CLASS_MAP:
            target = self.THAI_CLASS_MAP[target]

        if target not in self.COCO_CLASSES:
            print(f"Warning: '{target}' is not a valid COCO class")
            return False

        with self._lock:
            old_target = self._track_target
            self._track_target = target

        if old_target != target:
            print(f"YOLO Track Target: {target}")

        return True

    def set_show_text(self, show):
        with self._lock:
            self._show_text = show

    def set_on_mode_change(self, callback):
        self._on_mode_change = callback

    def get_detection_text(self, max_items=3):
        if not self._show_text:
            return []
        return self._cached_labels[:max_items]

    def get_mode_text(self):
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

    def get_normalized_position(self):
        return self._cached_position

    def process_voice_command(self, text):
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

    def cleanup(self):
        print("YOLO: Cleaning up...")
        self.stop()
        time.sleep(0.5)
        self._cleanup_device()
        self.model = None
        self.annotator = None
        self.stream = None

        with self._lock:
            self._detections = []
            self._latest_frame = None
            self._tracked_position = None
        self._cached_position = (0, 0)
        self._cached_labels = []

        collected = gc.collect()
        if collected > 0:
            print(f"YOLO: Final GC collected {collected} objects")

        print("YOLO: Cleanup complete")


class DummyYoloTracker:
    """Dummy tracker for when YOLO is not available"""

    def __init__(self, *args, **kwargs):
        print("YOLO: Using dummy tracker (no AI Camera)")
        self._mode = YoloMode.DETECT
        self._track_target = "person"
        self._show_text = True

    def start(self):
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

    def get_detection_text(self, max_items=3):
        return []

    def get_mode_text(self):
        return ""

    def get_normalized_position(self):
        return 0, 0

    def process_voice_command(self, text):
        return False

    def cleanup(self):
        pass


def create_yolo_tracker(confidence_threshold=None, frame_rate=None):
    """Factory function to create appropriate YOLO tracker"""
    # Use config values if not specified
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
        return YoloTracker(confidence_threshold, frame_rate)
    else:
        return DummyYoloTracker()


__all__ = ['YoloTracker', 'DummyYoloTracker', 'YoloMode', 'create_yolo_tracker', 'YOLO_AVAILABLE']
