#!/usr/bin/env python3
"""
IMX500 Face Detector - Uses AI on-chip for fast face detection
Falls back to OpenCV Haar Cascade if IMX500 face model not available
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class FaceDetection:
    """Detected face with bounding box"""
    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0
    
    @property
    def box(self) -> Tuple[int, int, int, int]:
        """Return (top, right, bottom, left) for face_recognition"""
        return (self.y, self.x + self.width, self.y + self.height, self.x)
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

class IMX500FaceDetector:
    """
    Face detector using IMX500 AI chip when available,
    falls back to OpenCV for compatibility
    """
    
    def __init__(self, camera_id: int = 0, use_imx500: bool = True):
        self.camera_id = camera_id
        self.use_imx500 = use_imx500
        self.picam = None
        self.haar_cascade = None
        self._setup()
    
    def _setup(self):
        """Setup camera and detector"""
        # Try IMX500 with AI model
        if self.use_imx500:
            try:
                from picamera2 import Picamera2
                from picamera2.devices.imx500 import IMX500
                
                # Check for face detection model
                model_path = Path("/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_face.rpk")
                
                if model_path.exists():
                    self.imx500 = IMX500(str(model_path))
                    self.picam = Picamera2(self.camera_id)
                    config = self.picam.create_preview_configuration(
                        main={"size": (640, 480)},
                        controls={"FrameRate": 30}
                    )
                    self.picam.configure(config)
                    print("✅ IMX500 Face Detection initialized")
                    return
                else:
                    print("⚠️ IMX500 face model not found, using OpenCV fallback")
            except Exception as e:
                print(f"⚠️ IMX500 setup failed: {e}, using OpenCV fallback")
        
        # Fallback to OpenCV Haar Cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.haar_cascade = cv2.CascadeClassifier(cascade_path)
        self.use_imx500 = False
        print("✅ OpenCV Face Detection initialized")
    
    def start(self):
        """Start camera capture"""
        if self.picam:
            self.picam.start()
    
    def stop(self):
        """Stop camera capture"""
        if self.picam:
            self.picam.stop()
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame"""
        if self.picam:
            frame = self.picam.capture_array()
            # Convert RGB to BGR for OpenCV
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return None
    
    def detect_faces(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in image
        Returns list of FaceDetection objects
        """
        if self.use_imx500 and hasattr(self, 'imx500'):
            return self._detect_imx500(image)
        else:
            return self._detect_opencv(image)
    
    def _detect_imx500(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect using IMX500 AI chip"""
        try:
            # Get inference results from IMX500
            metadata = self.picam.capture_metadata()
            detections = []
            
            if 'object_detect' in metadata:
                for det in metadata['object_detect']:
                    # det format: (class_id, confidence, x, y, w, h)
                    if det[1] > 0.5:  # confidence threshold
                        detections.append(FaceDetection(
                            x=int(det[2] * image.shape[1]),
                            y=int(det[3] * image.shape[0]),
                            width=int(det[4] * image.shape[1]),
                            height=int(det[5] * image.shape[0]),
                            confidence=det[1]
                        ))
            return detections
        except Exception as e:
            print(f"IMX500 detection error: {e}")
            return self._detect_opencv(image)
    
    def _detect_opencv(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect using OpenCV Haar Cascade"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.haar_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return [
            FaceDetection(x=x, y=y, width=w, height=h)
            for (x, y, w, h) in faces
        ]
    
    def detect_and_crop(self, image: np.ndarray, margin: int = 20) -> List[Tuple[np.ndarray, FaceDetection]]:
        """Detect faces and return cropped face images"""
        detections = self.detect_faces(image)
        crops = []
        
        h, w = image.shape[:2]
        for det in detections:
            x1 = max(0, det.x - margin)
            y1 = max(0, det.y - margin)
            x2 = min(w, det.x + det.width + margin)
            y2 = min(h, det.y + det.height + margin)
            
            crop = image[y1:y2, x1:x2]
            crops.append((crop, det))
        
        return crops

# Test
if __name__ == "__main__":
    detector = IMX500FaceDetector()
    detector.start()
    
    import time
    time.sleep(1)
    
    frame = detector.capture_frame()
    if frame is not None:
        faces = detector.detect_faces(frame)
        print(f"Detected {len(faces)} face(s)")
        for i, face in enumerate(faces):
            print(f"  Face {i+1}: {face.width}x{face.height} @ ({face.x},{face.y})")
    
    detector.stop()
