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
try:
    from modlib.devices import AiCamera
    from modlib.apps import Annotator
    import numpy as np
    import cv2
    
    # Try to import YOLO11n first (newer, more efficient for IMX500)
    # YOLO11n advantages:
    # - 37% less complex than YOLOv8
    # - Better accuracy and stability
    # - Optimized for embedded devices
    try:
        from modlib.models.zoo import YOLO11n
        YOLO_MODEL_CLASS = YOLO11n
        model_name = "YOLO11n"
    except ImportError:
        # Fallback to YOLOv8n if YOLO11n not available
        from modlib.models.zoo import YOLOv8n
        YOLO_MODEL_CLASS = YOLOv8n
        model_name = "YOLOv8n"
    
    YOLO_AVAILABLE = True
    print(f"YOLO (IMX500): Available - Using {model_name} model")
except ImportError as e:
    print(f"YOLO (IMX500): Not available - {e}")


class YoloMode(Enum):
    """YOLO operation modes"""
    DETECT = "detect"      # Detect and display all objects
    TRACK = "track"        # Track specific object with eyes
    TRACK_ALL = "track_all"  # Track any detected object


class YoloTracker:
    """
    YOLO Object Detection using IMX500 AI Camera
    Runs in a separate thread for non-blocking detection
    Supports multiple modes: DETECT, TRACK, TRACK_ALL
    """

    # COCO class names (80 classes)
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

    # Thai to English class mapping for voice commands
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
        """
        Initialize YOLO tracker

        Args:
            confidence_threshold: Minimum confidence for detections (0.0-1.0)
            frame_rate: Camera frame rate
        """
        self.confidence_threshold = confidence_threshold
        self.frame_rate = frame_rate

        self.device = None
        self.model = None
        self.annotator = None
        self.stream = None

        # Mode control
        self._mode = YoloMode.DETECT
        self._track_target = "person"  # What to track in TRACK mode
        self._show_text = True  # Show detection text on display

        # Detection results (thread-safe)
        self._lock = threading.Lock()
        self._detections = []  # List of (label, confidence, bbox)
        self._latest_frame = None
        self._tracked_position = None  # (x, y) normalized position of tracked object

        # Cached values (updated atomically, read without lock for performance)
        self._cached_position = (0, 0)  # Pre-calculated normalized position (-1 to 1)
        self._cached_labels = []  # Pre-sorted labels for display

        # Thread control
        self._running = False
        self._thread = None

        # Stats
        self._fps = 0
        self._last_time = time.time()
        self._frame_count = 0

        # Callbacks
        self._on_mode_change = None
        self._on_detection = None

        # Register finalizer as fallback cleanup (runs when object is garbage collected)
        # This is a safety net, but explicit cleanup() is preferred
        # Note: We pass 'self' so finalizer can access self.device when it runs
        self._finalizer = weakref.finalize(self, self._finalize_cleanup, weakref.ref(self))

        if YOLO_AVAILABLE:
            self._init_camera()
    
    @staticmethod
    def _finalize_cleanup(self_ref):
        """
        Finalizer callback - fallback cleanup when object is garbage collected
        This is a safety net, but explicit cleanup() is always preferred
        """
        try:
            # Try to get the object reference (may be None if already collected)
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
                        pass  # Ignore errors in finalizer
        except Exception:
            pass  # Ignore all errors in finalizer

    def _init_camera(self):
        """Initialize AI Camera and YOLO model"""
        try:
            print(f"Loading {model_name} model...")
            self.model = YOLO_MODEL_CLASS()
            print(f"Model loaded! Classes: {len(self.model.labels)}")

            print("Initializing AI Camera and Deploying Model...")

            # Wait a bit to ensure previous device is fully released
            time.sleep(1.0)
            
            # Try to find the correct camera index by iterating
            # We must try to DEPLOY to confirm it's the IMX500
            device_initialized = False
            last_error = None
            
            # Try indices in this order: [1, 0, 2, 3]
            # Start with 1 to avoid conflicts with picamera2's default camera (often 0)
            for cam_idx in [1, 0, 2, 3]:
                print(f"   - Testing Camera Index {cam_idx}...")
                temp_device = None
                try:
                    # 1. Initialize Device
                    temp_device = AiCamera(frame_rate=self.frame_rate, num=cam_idx)
                    
                    # 2. Try to Deploy (The real test)
                    print(f"     [Index {cam_idx}] Attempting deploy...")
                    temp_device.deploy(self.model)
                    
                    # If we get here, it worked!
                    self.device = temp_device
                    self.annotator = Annotator()
                    device_initialized = True
                    print(f"   [SUCCESS] Camera Index {cam_idx} is IMX500!")
                    break
                    
                except KeyError as e:
                    # Specific handling for picamera2 threading conflict
                    print(f"     [Index {cam_idx}] KeyError (picamera2 conflict): {e}")
                    last_error = e
                    # Cleanup and try next index
                    if temp_device:
                        try:
                            if hasattr(temp_device, 'close'): temp_device.close()
                            elif hasattr(temp_device, '__exit__'): temp_device.__exit__(None, None, None)
                        except: pass
                    gc.collect()
                    time.sleep(1.0)  # Wait longer for picamera2 to release
                    
                except Exception as e:
                    print(f"     [Index {cam_idx}] Failed: {e}")
                    last_error = e
                    # Cleanup failed device
                    if temp_device:
                        try:
                            if hasattr(temp_device, 'close'): temp_device.close()
                            elif hasattr(temp_device, '__exit__'): temp_device.__exit__(None, None, None)
                        except: pass
                    gc.collect()
                    time.sleep(0.5)

            if not device_initialized:
                print(f"   [ERROR] Could not find IMX500 on any index. Last error: {last_error}")
                self.device = None
                return

            print("YOLO ready!")
            
            # Wait a bit for device to be fully ready before allowing acquisition
            time.sleep(1.0)
            print("YOLO: Device ready for acquisition")

        except Exception as e:
            print(f"YOLO init error: {e}")

    def start(self):
        """Start detection thread"""
        if not YOLO_AVAILABLE or self.device is None:
            print("YOLO not available, cannot start")
            return False

        # Ensure device is ready before starting thread
        # This prevents race condition where thread tries to acquire before device is ready
        if self.device is not None:
            # Give device a moment to be fully ready
            time.sleep(0.5)

        self._running = True
        self._thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._thread.start()
        print("YOLO: Detection thread started")
        return True

    def stop(self):
        """Stop detection thread"""
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
        """Cleanup AI Camera device resources - Best practices for resource cleanup"""
        try:
            if self.device is not None:
                device_ref = self.device
                
                # Step 1: Try explicit cleanup methods (context manager pattern)
                try:
                    if hasattr(device_ref, 'close'):
                        device_ref.close()
                        print("YOLO: Device closed via close()")
                    elif hasattr(device_ref, '__exit__'):
                        # Use context manager exit (most reliable)
                        device_ref.__exit__(None, None, None)
                        print("YOLO: Device closed via __exit__()")
                except Exception as close_error:
                    print(f"YOLO: Error closing device: {close_error}")
                
                # Step 2: Clear reference to break reference cycles
                self.device = None
                
                # Step 3: Force garbage collection to break cycles and finalize objects
                # This helps ensure file handles and device resources are released
                collected = gc.collect()
                if collected > 0:
                    print(f"YOLO: GC collected {collected} objects")
                
                # Step 4: Give OS time to release device resources
                # Camera devices need time for kernel to release file descriptors
                time.sleep(1.0)
                print("YOLO: Device cleaned up")
        except Exception as e:
            print(f"YOLO cleanup warning: {e}")

    def _detection_loop(self):
        """Main detection loop with Auto-Recovery"""
        
        # Loop หลัก: จะทำงานตลอดจนกว่าจะสั่ง stop()
        while self._running:
            try:
                # 1. ตรวจสอบสถานะ Device ก่อนเริ่ม
                if self.device is None:
                    print("YOLO: Device is None, attempting to initialize...")
                    try:
                        self._init_camera()
                    except Exception as e:
                        print(f"YOLO: Init failed ({e}), waiting 2s...")
                        time.sleep(2.0)
                        continue

                # 2. เริ่ม Acquire Stream
                print("YOLO: Acquiring camera stream...")
                
                # ใช้ try-except ครอบการ Stream ทั้งหมด
                try:
                    # รอสักนิดเพื่อให้ Device พร้อม (สำคัญมากสำหรับ IMX500)
                    time.sleep(1.0)
                    
                    with self.device as stream:
                        print("YOLO: Stream acquired! Starting process...")
                        
                        # Loop ย่อย: อ่านภาพทีละเฟรม
                        for frame in stream:
                            # เช็คว่ายังต้องรันอยู่ไหม
                            if not self._running:
                                break
                                
                            start_time = time.time()

                            # --- ส่วนประมวลผล ---
                            frame_image = frame.image
                            
                            # ป้องกัน FPS พุ่ง (ถ้าภาพมา null)
                            if frame_image is None:
                                print("YOLO: Warning - Received empty frame")
                                time.sleep(0.1)
                                continue

                            # Get current mode and target (thread-safe)
                            with self._lock:
                                current_mode = self._mode
                                track_target = self._track_target

                            # Filter detections by confidence
                            try:
                                detections = frame.detections[
                                    frame.detections.confidence > self.confidence_threshold
                                ]
                            except (TypeError, AttributeError):
                                detections = []

                            # Get frame dimensions
                            if frame_image is not None:
                                frame_h, frame_w = frame_image.shape[:2]
                            else:
                                frame_h, frame_w = 480, 640

                            # Process detections
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

                                # Tracking logic
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

                            # Update State
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

                            # --- FPS Calculation (ปรับปรุง) ---
                            self._frame_count += 1
                            now = time.time()
                            if now - self._last_time >= 1.0:
                                self._fps = self._frame_count
                                # ถ้า FPS สูงผิดปกติ แสดงว่า Loop หมุนฟรี
                                if self._fps > 100: 
                                    print(f"YOLO: [WARNING] FPS anomaly ({self._fps}). Possible stream sync issue.")
                                else:
                                    # Log detection results to console (once per second)
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
                            
                            # --- End Processing ---

                except Exception as stream_error:
                    # นี่คือจุดที่จะจับ Error "Camera frontend has timed out"
                    error_str = str(stream_error)
                    if "timeout" in error_str.lower() or "timed out" in error_str.lower():
                        print(f"YOLO: Stream timeout detected! Error: {stream_error}")
                    else:
                        print(f"YOLO: Stream crashed! Error: {stream_error}")
                    print("YOLO: Initiating Auto-Recovery...")
                    raise stream_error  # ส่งต่อไปให้วงนอกจัดการ Cleanup

            except Exception as e:
                # Catch-all สำหรับ Error ทั้งหมด (Acquire fail, Stream crash)
                if not self._running:
                    break
                    
                print(f"YOLO: Critical Error detected: {e}")
                print("YOLO: Performing Hard Reset...")
                
                # 1. Cleanup ของเก่า
                self._cleanup_device()
                self.device = None
                
                # 2. รอให้ Hardware เย็นลง/Reset
                print("YOLO: Cooling down (3s)...")
                time.sleep(3.0)
                
                # Loop จะวนกลับไปบรรทัดแรก เพื่อ _init_camera ใหม่เอง
                continue

        print("YOLO: Thread exiting...")

    @property
    def detections(self):
        """Get current detections (thread-safe)"""
        with self._lock:
            return self._detections.copy()

    @property
    def latest_frame(self):
        """Get latest annotated frame (RGB format, no copy for performance)"""
        # No lock needed - we just read a reference that's atomically assigned
        # No copy needed - YOLO thread assigns new array each frame, doesn't modify in-place
        return self._latest_frame

    @property
    def tracked_position(self):
        """Get tracked object position for eye tracking (thread-safe)"""
        with self._lock:
            return self._tracked_position

    @property
    def fps(self):
        """Get current FPS"""
        return self._fps

    @property
    def running(self):
        """Check if tracker is running"""
        return self._running and self.device is not None

    @property
    def mode(self):
        """Get current mode"""
        with self._lock:
            return self._mode

    @property
    def track_target(self):
        """Get current track target"""
        with self._lock:
            return self._track_target

    @property
    def show_text(self):
        """Get show_text setting"""
        with self._lock:
            return self._show_text

    def set_mode(self, mode):
        """
        Set operation mode

        Args:
            mode: YoloMode enum value
        """
        with self._lock:
            old_mode = self._mode
            self._mode = mode
        if old_mode != mode:
            print(f"YOLO Mode: {mode.value}")
            if self._on_mode_change:
                self._on_mode_change(mode)

    def set_track_target(self, target):
        """
        Set what object to track in TRACK mode

        Args:
            target: Class name (e.g., "person", "cat", "dog")
                   Can be Thai or English
        """
        # Convert Thai to English if needed
        if target in self.THAI_CLASS_MAP:
            target = self.THAI_CLASS_MAP[target]

        # Validate target
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
        """Set whether to show detection text on display"""
        with self._lock:
            self._show_text = show

    def set_on_mode_change(self, callback):
        """Set callback for mode changes"""
        self._on_mode_change = callback

    def get_detection_text(self, max_items=3):
        """
        Get detection results as text for display (lock-free for performance)

        Args:
            max_items: Maximum number of items to show

        Returns:
            List of strings like ["person", "cat", "dog"]
        """
        # Use cached pre-sorted labels (no lock needed)
        if not self._show_text:
            return []
        return self._cached_labels[:max_items]

    def get_mode_text(self):
        """Get current mode as display text"""
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
        """
        Get normalized tracked position for eye tracking (lock-free for performance)
        Returns (x, y) where x, y are in range -1 to 1
        Returns (0, 0) if no object tracked
        """
        # Use pre-calculated cached position (no lock needed)
        return self._cached_position

    def process_voice_command(self, text):
        """
        Process voice command for mode switching

        Args:
            text: Voice command text (Thai or English)

        Returns:
            True if command was recognized, False otherwise
        """
        text_lower = text.lower()

        # Mode commands
        if "ตรวจจับ" in text or "detect" in text_lower:
            self.set_mode(YoloMode.DETECT)
            return True

        if "ติดตาม" in text or "track" in text_lower or "follow" in text_lower:
            # Check if there's a target specified
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

            # Default to track person
            self.set_mode(YoloMode.TRACK)
            return True

        # Target-only commands (just say the object name)
        for thai, english in self.THAI_CLASS_MAP.items():
            if thai in text:
                self.set_track_target(english)
                self.set_mode(YoloMode.TRACK)
                return True

        # Show/hide text
        if "ซ่อน" in text or "hide" in text_lower:
            self.set_show_text(False)
            return True

        if "แสดง" in text or "show" in text_lower:
            self.set_show_text(True)
            return True

        return False

    def cleanup(self):
        """
        Cleanup resources - Comprehensive cleanup following best practices
        
        Steps:
        1. Stop detection thread (exits context manager)
        2. Wait for thread to fully exit
        3. Cleanup device explicitly
        4. Clear all references
        5. Force garbage collection to break cycles
        6. Clear cached data
        """
        print("YOLO: Cleaning up...")
        
        # Step 1: Stop thread first (this will exit the context manager in _detection_loop)
        self.stop()
        
        # Step 2: Wait for thread to fully exit and context manager to close
        time.sleep(0.5)
        
        # Step 3: Cleanup device explicitly
        self._cleanup_device()
        
        # Step 4: Clear all object references to break reference cycles
        self.model = None
        self.annotator = None
        self.stream = None
        
        # Step 5: Clear cached data (thread-safe)
        with self._lock:
            self._detections = []
            self._latest_frame = None
            self._tracked_position = None
        self._cached_position = (0, 0)
        self._cached_labels = []
        
        # Step 6: Final garbage collection to ensure all cycles are broken
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


def create_yolo_tracker(confidence_threshold=0.5, frame_rate=10):
    """Factory function to create appropriate YOLO tracker"""
    if YOLO_AVAILABLE:
        return YoloTracker(confidence_threshold, frame_rate)
    else:
        return DummyYoloTracker()


# Export YoloMode for external use
__all__ = ['YoloTracker', 'DummyYoloTracker', 'YoloMode', 'create_yolo_tracker', 'YOLO_AVAILABLE']
