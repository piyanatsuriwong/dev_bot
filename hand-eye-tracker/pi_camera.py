#!/usr/bin/env python3
"""
Pi Camera wrapper for Raspberry Pi 5
Supports both picamera2 and OpenCV fallback
"""
import cv2
import numpy as np

# Try to import picamera2
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("Warning: picamera2 not available, using OpenCV fallback")


class PiCamera:
    """Camera wrapper that uses picamera2 on Pi, OpenCV elsewhere"""

    def __init__(self, camera_id=0, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.camera_id = camera_id
        self.use_picamera2 = PICAMERA2_AVAILABLE
        self.cap = None
        self.picam2 = None

        if self.use_picamera2:
            self._init_picamera2()
        else:
            self._init_opencv()

    def _init_picamera2(self):
        """Initialize picamera2"""
        try:
            self.picam2 = Picamera2(camera_num=self.camera_id)
            config = self.picam2.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            self.picam2.configure(config)
            self.picam2.start()
            print(f"PiCamera2 initialized: camera {self.camera_id} @ {self.width}x{self.height}")
        except Exception as e:
            print(f"PiCamera2 init failed: {e}, falling back to OpenCV")
            self.use_picamera2 = False
            self._init_opencv()

    def _init_opencv(self):
        """Initialize OpenCV VideoCapture"""
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        print(f"OpenCV camera initialized: {self.width}x{self.height}")

    def read(self):
        """Read a frame from camera
        Returns: (success, frame) tuple like OpenCV
        """
        if self.use_picamera2 and self.picam2:
            try:
                frame = self.picam2.capture_array()
                # Convert RGB to BGR for OpenCV compatibility
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return True, frame
            except Exception as e:
                print(f"PiCamera2 read error: {e}")
                return False, None
        elif self.cap:
            return self.cap.read()
        return False, None

    def release(self):
        """Release camera resources - Force close to return camera to Available state"""
        if self.picam2:
            try:
                # Stop camera first
                self.picam2.stop()
                # Then close to release resources and return to Available state
                # This is critical for libcamera state management
                if hasattr(self.picam2, 'close'):
                    self.picam2.close()
                    print("   [PiCamera] Picamera2 closed (returned to Available state)")
                else:
                    # Fallback: set to None to release reference
                    self.picam2 = None
            except Exception as e:
                print(f"   [PiCamera] Error closing picamera2: {e}")
                self.picam2 = None
        if self.cap:
            try:
                self.cap.release()
            except Exception as e:
                print(f"   [PiCamera] Error releasing OpenCV camera: {e}")
            self.cap = None

    def isOpened(self):
        """Check if camera is opened"""
        if self.use_picamera2 and self.picam2:
            return True
        elif self.cap:
            return self.cap.isOpened()
        return False
