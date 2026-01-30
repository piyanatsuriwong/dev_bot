#!/usr/bin/env python3
"""CCTV Person Detector - HOG + Optional YOLO
ตรวจจับคนด้วย OpenCV HOG (เบา) หรือ YOLO (แม่นยำ)
"""

import cv2
import numpy as np
import time
import logging

logger = logging.getLogger("cctv.detector")


class PersonDetector:
    """Person detector using OpenCV HOG or YOLO"""
    
    def __init__(self, method="hog", threshold=0.5, min_area=3000):
        self.method = method
        self.threshold = threshold
        self.min_area = min_area
        self._last_detections = []
        
        if method == "hog":
            self._init_hog()
        elif method == "yolo":
            self._init_yolo()
        else:
            raise ValueError(f"Unknown method: {method}. Use 'hog' or 'yolo'")
    
    def _init_hog(self):
        """Initialize HOG Person Detector (built-in OpenCV, no extra model needed)"""
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        logger.info("HOG Person Detector initialized")
    
    def _init_yolo(self):
        """Initialize YOLO detector using OpenCV DNN"""
        import os
        # Try to find YOLO model
        model_paths = [
            "yolo26n.pt",
            "yolov8n.onnx",
            "yolov8n.pt",
        ]
        self.yolo_net = None
        for path in model_paths:
            if os.path.exists(path):
                try:
                    self.yolo_net = cv2.dnn.readNet(path)
                    logger.info(f"YOLO loaded from {path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")
        
        if self.yolo_net is None:
            logger.warning("No YOLO model found, falling back to HOG")
            self.method = "hog"
            self._init_hog()
    
    def detect(self, frame):
        """Detect persons in frame
        
        Returns:
            list of (x, y, w, h, confidence) tuples
        """
        if self.method == "hog":
            return self._detect_hog(frame)
        elif self.method == "yolo":
            return self._detect_yolo(frame)
        return []
    
    def _detect_hog(self, frame):
        """HOG-based person detection"""
        # Resize for faster processing
        h, w = frame.shape[:2]
        scale = min(1.0, 400.0 / max(h, w))
        if scale < 1.0:
            small = cv2.resize(frame, None, fx=scale, fy=scale)
        else:
            small = frame
            scale = 1.0
        
        # Detect people
        boxes, weights = self.hog.detectMultiScale(
            small,
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05
        )
        
        detections = []
        for (x, y, bw, bh), weight in zip(boxes, weights):
            conf = float(weight)
            if conf >= self.threshold:
                # Scale back to original size
                ox = int(x / scale)
                oy = int(y / scale)
                obw = int(bw / scale)
                obh = int(bh / scale)
                area = obw * obh
                if area >= self.min_area:
                    detections.append((ox, oy, obw, obh, conf))
        
        # Non-maximum suppression
        if len(detections) > 0:
            detections = self._nms(detections)
        
        self._last_detections = detections
        return detections
    
    def _detect_yolo(self, frame):
        """YOLO-based person detection (class 0 = person in COCO)"""
        if self.yolo_net is None:
            return self._detect_hog(frame)
        
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
        self.yolo_net.setInput(blob)
        
        try:
            outputs = self.yolo_net.forward(self.yolo_net.getUnconnectedOutLayersNames())
        except Exception:
            return self._detect_hog(frame)
        
        detections = []
        for output in outputs:
            for det in output:
                scores = det[5:]
                class_id = np.argmax(scores)
                conf = float(scores[class_id])
                # class 0 = person in COCO
                if class_id == 0 and conf >= self.threshold:
                    cx, cy, bw, bh = det[0:4]
                    x = int((cx - bw/2) * w)
                    y = int((cy - bh/2) * h)
                    bw = int(bw * w)
                    bh = int(bh * h)
                    if bw * bh >= self.min_area:
                        detections.append((x, y, bw, bh, conf))
        
        if len(detections) > 0:
            detections = self._nms(detections)
        
        self._last_detections = detections
        return detections
    
    def _nms(self, detections, overlap_thresh=0.4):
        """Non-maximum suppression"""
        if len(detections) == 0:
            return []
        
        boxes = np.array([[d[0], d[1], d[0]+d[2], d[1]+d[3]] for d in detections])
        scores = np.array([d[4] for d in detections])
        
        indices = cv2.dnn.NMSBoxes(
            [[d[0], d[1], d[2], d[3]] for d in detections],
            scores.tolist(),
            self.threshold,
            overlap_thresh
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        return detections
    
    def draw_detections(self, frame, detections=None):
        """Draw bounding boxes on frame"""
        if detections is None:
            detections = self._last_detections
        
        annotated = frame.copy()
        for (x, y, w, h, conf) in detections:
            # Green box
            cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Label
            label = f"Person {conf:.1%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x, y-th-10), (x+tw, y), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Timestamp
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated, ts, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, ts, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        return annotated
