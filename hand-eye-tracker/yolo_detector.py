#!/usr/bin/env python3
"""
YOLO Object Detection for Raspberry Pi 5
Uses Ultralytics YOLOv8 for real-time object detection
"""

import cv2
import time
import numpy as np

# Try to import ultralytics YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. Install with: pip install ultralytics")

# Try to import PiCamera
try:
    from pi_camera import PiCamera
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False

# Try to import servo controller
try:
    from servo_controller import ServoController
    SERVO_AVAILABLE = True
except ImportError:
    SERVO_AVAILABLE = False


class YOLODetector:
    """
    YOLO Object Detector with optional servo tracking
    """

    # Common COCO classes
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]

    def __init__(self, model_path="yolov8n.pt", camera_id=0, conf_threshold=0.5,
                 track_class=None, enable_servo=True):
        """
        Initialize YOLO detector

        Args:
            model_path: Path to YOLO model (default: yolov8n.pt - nano model)
            camera_id: Camera device ID
            conf_threshold: Confidence threshold for detection
            track_class: Class name to track with servo (e.g., 'person', 'cat')
            enable_servo: Enable servo tracking
        """
        self.conf_threshold = conf_threshold
        self.track_class = track_class
        self.enable_servo = enable_servo and SERVO_AVAILABLE

        # Load YOLO model
        if YOLO_AVAILABLE:
            print(f"Loading YOLO model: {model_path}")
            self.model = YOLO(model_path)
            print("YOLO model loaded successfully")
        else:
            self.model = None
            print("YOLO not available!")

        # Initialize camera
        if PICAMERA_AVAILABLE:
            self.camera = PiCamera(camera_id, width=640, height=480, fps=30)
        else:
            self.camera = cv2.VideoCapture(camera_id)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Initialize servo
        self.servo = None
        if self.enable_servo:
            self.servo = ServoController()
            if self.servo.enabled:
                print(f"Servo tracking enabled for class: {track_class or 'all'}")
            else:
                self.enable_servo = False

        # Detection results
        self.detections = []
        self.tracked_object = None
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()

    def detect(self, frame):
        """
        Run YOLO detection on frame

        Args:
            frame: BGR image

        Returns:
            List of detections: [(class_name, confidence, x1, y1, x2, y2), ...]
        """
        if self.model is None:
            return []

        # Run inference
        results = self.model(frame, conf=self.conf_threshold, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]

                    detections.append({
                        'class': cls_name,
                        'confidence': conf,
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'center': (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    })

        self.detections = detections
        return detections

    def draw_detections(self, frame, detections=None):
        """
        Draw detection boxes on frame

        Args:
            frame: BGR image
            detections: List of detections (uses self.detections if None)

        Returns:
            Frame with drawn detections
        """
        if detections is None:
            detections = self.detections

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls_name = det['class']
            conf = det['confidence']

            # Choose color based on class
            if cls_name == self.track_class:
                color = (0, 255, 0)  # Green for tracked class
            elif cls_name == 'person':
                color = (255, 0, 0)  # Blue for person
            else:
                color = (0, 255, 255)  # Yellow for others

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{cls_name}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Draw center point
            cx, cy = det['center']
            cv2.circle(frame, (cx, cy), 5, color, -1)

        # Draw FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw detection count
        cv2.putText(frame, f"Objects: {len(detections)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

    def track_object(self, frame_width, frame_height):
        """
        Track specified object class with servo

        Args:
            frame_width: Frame width
            frame_height: Frame height
        """
        if not self.enable_servo or self.servo is None:
            return

        # Find object to track
        target = None
        for det in self.detections:
            if self.track_class is None or det['class'] == self.track_class:
                # Track the largest object of the target class
                if target is None:
                    target = det
                else:
                    # Compare area
                    x1, y1, x2, y2 = det['bbox']
                    area = (x2 - x1) * (y2 - y1)
                    tx1, ty1, tx2, ty2 = target['bbox']
                    target_area = (tx2 - tx1) * (ty2 - ty1)
                    if area > target_area:
                        target = det

        if target:
            self.tracked_object = target
            cx, cy = target['center']

            # Convert to normalized coordinates (-1 to 1)
            norm_x = (cx / frame_width - 0.5) * 2
            norm_y = (cy / frame_height - 0.5) * 2

            # Track with servo
            self.servo.track_hand(norm_x, norm_y, smoothing=0.2)
        else:
            self.tracked_object = None

    def update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_fps_time

        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_time = current_time

    def run(self, show_window=True):
        """
        Run detection loop

        Args:
            show_window: Show OpenCV window
        """
        print("\n" + "=" * 50)
        print("  YOLO Object Detection")
        print("=" * 50)
        print(f"Track class: {self.track_class or 'all'}")
        print(f"Confidence threshold: {self.conf_threshold}")
        print(f"Servo tracking: {'enabled' if self.enable_servo else 'disabled'}")
        print("Press 'q' to quit, 's' to save screenshot")
        print("=" * 50 + "\n")

        try:
            while True:
                # Read frame
                if PICAMERA_AVAILABLE:
                    ret, frame = self.camera.read()
                else:
                    ret, frame = self.camera.read()

                if not ret:
                    print("Failed to read frame")
                    break

                frame = cv2.flip(frame, 1)  # Mirror

                # Run detection
                detections = self.detect(frame)

                # Track object with servo
                h, w = frame.shape[:2]
                self.track_object(w, h)

                # Draw detections
                frame = self.draw_detections(frame)

                # Draw tracked object info
                if self.tracked_object:
                    info = f"Tracking: {self.tracked_object['class']}"
                    cv2.putText(frame, info, (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Update FPS
                self.update_fps()

                # Show frame
                if show_window:
                    cv2.imshow("YOLO Detection", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        filename = f"screenshot_{int(time.time())}.jpg"
                        cv2.imwrite(filename, frame)
                        print(f"Saved: {filename}")

        except KeyboardInterrupt:
            print("\nStopped by user")

        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        if PICAMERA_AVAILABLE:
            self.camera.release()
        else:
            self.camera.release()

        if self.servo:
            self.servo.cleanup()

        cv2.destroyAllWindows()
        print("Cleanup complete")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='YOLO Object Detection')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='YOLO model path (default: yolov8n.pt)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--track', type=str, default=None,
                       help='Class to track (e.g., person, cat, dog)')
    parser.add_argument('--no-servo', action='store_true',
                       help='Disable servo tracking')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable display window')
    args = parser.parse_args()

    detector = YOLODetector(
        model_path=args.model,
        camera_id=args.camera,
        conf_threshold=args.conf,
        track_class=args.track,
        enable_servo=not args.no_servo
    )

    detector.run(show_window=not args.no_display)


if __name__ == "__main__":
    main()
