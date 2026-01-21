#!/usr/bin/env python3
"""
IMX500 AI Camera Object Detection
Uses the built-in NPU on Sony IMX500 sensor for fast inference
"""

import time
import cv2
import numpy as np

try:
    from picamera2 import Picamera2
    from picamera2.devices.imx500 import IMX500, NetworkIntrinsics
    from picamera2.devices.imx500.postprocess import COCODrawer
    IMX500_AVAILABLE = True
except ImportError as e:
    IMX500_AVAILABLE = False
    print(f"IMX500 not available: {e}")

try:
    from servo_controller import ServoController
    SERVO_AVAILABLE = True
except ImportError:
    SERVO_AVAILABLE = False


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


class IMX500Detector:
    """
    Object detection using IMX500's built-in AI accelerator
    Much faster than CPU-based YOLO (30+ FPS vs 2-3 FPS)
    """

    # Default model path
    DEFAULT_MODEL = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"

    def __init__(self, model_path=None, conf_threshold=0.5, track_class=None, enable_servo=True):
        """
        Initialize IMX500 detector

        Args:
            model_path: Path to .rpk model file
            conf_threshold: Confidence threshold
            track_class: Class name to track (e.g., 'person')
            enable_servo: Enable servo tracking
        """
        self.conf_threshold = conf_threshold
        self.track_class = track_class
        self.enable_servo = enable_servo and SERVO_AVAILABLE

        if not IMX500_AVAILABLE:
            raise RuntimeError("IMX500 not available")

        # Load model
        model_path = model_path or self.DEFAULT_MODEL
        print(f"Loading IMX500 model: {model_path}")

        self.imx500 = IMX500(model_path)
        self.intrinsics = self.imx500.network_intrinsics or NetworkIntrinsics()
        self.intrinsics.task = "object detection"

        # Set confidence threshold
        if hasattr(self.intrinsics, 'confidence_threshold'):
            self.intrinsics.confidence_threshold = conf_threshold

        # Initialize camera
        self.picam2 = Picamera2(self.imx500.camera_num)
        config = self.picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            controls={"FrameRate": 30}
        )
        self.picam2.configure(config)

        # Initialize servo
        self.servo = None
        if self.enable_servo:
            self.servo = ServoController()
            if self.servo.enabled:
                print(f"Servo tracking enabled for: {track_class or 'all'}")
            else:
                self.enable_servo = False

        # FPS counter
        self.fps = 0
        self.frame_count = 0
        self.fps_time = time.time()

        # Detection results
        self.detections = []
        self.frame_width = 640
        self.frame_height = 480

        print("IMX500 initialized!")

    def start(self):
        """Start camera"""
        self.picam2.start()
        print("Camera started")

    def get_detections(self):
        """
        Get detections from IMX500

        Returns:
            List of detections: [(class_name, confidence, x1, y1, x2, y2), ...]
        """
        metadata = self.picam2.capture_metadata()

        # Parse detections from IMX500
        detections = []

        try:
            # Get output tensors from IMX500
            outputs = self.imx500.get_outputs(metadata)

            if outputs is not None:
                # Parse SSD MobileNet output format
                # Format depends on the model, typically: boxes, classes, scores, num_detections
                if len(outputs) >= 4:
                    boxes = outputs[0]      # [num_det, 4] - ymin, xmin, ymax, xmax
                    classes = outputs[1]    # [num_det]
                    scores = outputs[2]     # [num_det]
                    num_det = int(outputs[3][0]) if len(outputs[3]) > 0 else 0

                    for i in range(min(num_det, len(scores))):
                        score = float(scores[i])
                        if score >= self.conf_threshold:
                            class_id = int(classes[i])
                            class_name = COCO_LABELS[class_id] if class_id < len(COCO_LABELS) else f"class_{class_id}"

                            # Convert normalized coordinates to pixels
                            ymin, xmin, ymax, xmax = boxes[i]
                            x1 = int(xmin * self.frame_width)
                            y1 = int(ymin * self.frame_height)
                            x2 = int(xmax * self.frame_width)
                            y2 = int(ymax * self.frame_height)

                            detections.append({
                                'class': class_name,
                                'confidence': score,
                                'bbox': (x1, y1, x2, y2),
                                'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                            })
        except Exception as e:
            print(f"Detection error: {e}")

        self.detections = detections
        return detections

    def capture_frame(self):
        """Capture frame from camera"""
        frame = self.picam2.capture_array()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def draw_detections(self, frame):
        """Draw detection boxes on frame"""
        for det in self.detections:
            x1, y1, x2, y2 = det['bbox']
            cls_name = det['class']
            conf = det['confidence']

            # Color based on class
            if cls_name == self.track_class:
                color = (0, 255, 0)  # Green for tracked
            elif cls_name == 'person':
                color = (255, 0, 0)  # Blue for person
            else:
                color = (0, 255, 255)  # Yellow for others

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

    def track_object(self):
        """Track detected object with servo"""
        if not self.enable_servo or not self.servo:
            return

        target = None
        for det in self.detections:
            if self.track_class is None or det['class'] == self.track_class:
                if target is None:
                    target = det
                else:
                    # Track largest object
                    x1, y1, x2, y2 = det['bbox']
                    area = (x2 - x1) * (y2 - y1)
                    tx1, ty1, tx2, ty2 = target['bbox']
                    if area > (tx2 - tx1) * (ty2 - ty1):
                        target = det

        if target:
            cx, cy = target['center']
            norm_x = (cx / self.frame_width - 0.5) * 2
            norm_y = (cy / self.frame_height - 0.5) * 2
            self.servo.track_hand(norm_x, norm_y, smoothing=0.25)

    def update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        elapsed = time.time() - self.fps_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_time = time.time()

    def run(self, show_window=True):
        """Main detection loop"""
        print("\n" + "=" * 50)
        print("  IMX500 AI Camera - Object Detection")
        print("=" * 50)
        print(f"Track class: {self.track_class or 'all'}")
        print(f"Confidence: {self.conf_threshold}")
        print(f"Servo: {'enabled' if self.enable_servo else 'disabled'}")
        print("Press 'q' to quit")
        print("=" * 50 + "\n")

        self.start()

        try:
            while True:
                # Get detections from IMX500 NPU
                detections = self.get_detections()

                # Track object
                self.track_object()

                # Update FPS
                self.update_fps()

                # Display
                if show_window:
                    frame = self.capture_frame()
                    frame = self.draw_detections(frame)
                    cv2.imshow("IMX500 Detection", frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    # Print detections
                    if detections:
                        for det in detections[:3]:
                            print(f"{det['class']}: {det['confidence']:.2f}")
                    time.sleep(0.03)

        except KeyboardInterrupt:
            print("\nStopped")

        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        if self.servo:
            self.servo.cleanup()
        self.picam2.stop()
        cv2.destroyAllWindows()
        print("Cleanup complete")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='IMX500 Object Detection')
    parser.add_argument('--model', type=str, default=None,
                       help='Model path (.rpk file)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--track', type=str, default=None,
                       help='Class to track (e.g., person)')
    parser.add_argument('--no-servo', action='store_true',
                       help='Disable servo')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable display')
    args = parser.parse_args()

    detector = IMX500Detector(
        model_path=args.model,
        conf_threshold=args.conf,
        track_class=args.track,
        enable_servo=not args.no_servo
    )

    detector.run(show_window=not args.no_display)


if __name__ == "__main__":
    main()
