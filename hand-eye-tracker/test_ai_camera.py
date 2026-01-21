#!/usr/bin/env python3
"""
Raspberry Pi AI Camera (IMX500) Test
Object Detection with HTTP Streaming - Simplified Version
"""

import sys
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

import cv2
import numpy as np
from picamera2 import Picamera2, MappedArray
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics, postprocess_nanodet_detection

# COCO Labels
LABELS = [
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

# Global
latest_frame = None
frame_lock = threading.Lock()
last_detections = []
last_boxes = []
last_scores = []
last_classes = []


class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = '''<!DOCTYPE html>
<html>
<head>
    <title>Raspberry Pi AI Camera</title>
    <style>
        body { font-family: Arial; background: #1a1a2e; color: white; margin: 0; padding: 20px; text-align: center; }
        h1 { color: #00ff88; }
        img { max-width: 100%; border: 3px solid #00ff88; border-radius: 10px; }
        .info { background: #16213e; padding: 15px; border-radius: 10px; margin: 20px auto; max-width: 640px; }
    </style>
</head>
<body>
    <h1>Raspberry Pi AI Camera (IMX500)</h1>
    <div class="info">Object Detection - NanoDet</div>
    <img src="/stream">
</body>
</html>'''
            self.wfile.write(html.encode())
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            while True:
                try:
                    with frame_lock:
                        if latest_frame is not None:
                            frame = latest_frame.copy()
                        else:
                            time.sleep(0.05)
                            continue
                    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                    self.wfile.write(jpeg.tobytes())
                    self.wfile.write(b'\r\n')
                    time.sleep(0.033)
                except:
                    break


def draw_detections(request):
    """Callback to draw detections on each frame"""
    global last_boxes, last_scores, last_classes

    with MappedArray(request, "main") as m:
        frame = m.array
        h, w = frame.shape[:2]

        # Draw each detection
        for i in range(len(last_boxes)):
            try:
                box = last_boxes[i]
                score = last_scores[i]
                cls = int(last_classes[i])

                if score < 0.5:
                    continue

                # Box format: [x_center, y_center, width, height] normalized
                x_c, y_c, bw, bh = box

                # Convert to pixel coordinates
                x1 = int((x_c - bw/2) * w / 416)
                y1 = int((y_c - bh/2) * h / 416)
                x2 = int((x_c + bw/2) * w / 416)
                y2 = int((y_c + bh/2) * h / 416)

                # Clamp
                x1 = max(0, min(w-1, x1))
                y1 = max(0, min(h-1, y1))
                x2 = max(0, min(w-1, x2))
                y2 = max(0, min(h-1, y2))

                # Draw
                color = (0, 255, 136)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = LABELS[cls] if cls < len(LABELS) else f"class_{cls}"
                text = f"{label}: {score:.2f}"
                cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            except Exception as e:
                pass

        # Status
        cv2.putText(frame, f"AI Camera - Detections: {len(last_boxes)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 136), 2)


def parse_detections(metadata, imx500):
    """Parse detections from IMX500 output"""
    global last_boxes, last_scores, last_classes

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    if np_outputs is None:
        return

    try:
        boxes, scores, classes = postprocess_nanodet_detection(
            outputs=np_outputs[0],
            conf=0.5,
            iou_thres=0.5,
            max_out_dets=10
        )

        last_boxes = boxes[0] if len(boxes) > 0 else []
        last_scores = scores[0] if len(scores) > 0 else []
        last_classes = classes[0] if len(classes) > 0 else []

        if len(last_boxes) > 0:
            print(f"Detected {len(last_boxes)} objects")

    except Exception as e:
        print(f"Detection error: {e}")


def camera_loop():
    """Main camera capture loop"""
    global latest_frame

    print("Initializing AI Camera...")
    print("=" * 60)

    # Load model
    model = "/usr/share/imx500-models/imx500_network_nanodet_plus_416x416_pp.rpk"

    print(f"Loading model: {model}")
    imx500 = IMX500(model)

    # Get intrinsics
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"

    print("Creating camera configuration...")
    picam2 = Picamera2(imx500.camera_num)

    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"},
        controls={"FrameRate": 30}
    )

    picam2.configure(config)
    picam2.pre_callback = draw_detections

    print("Starting camera (firmware loading may take a few minutes)...")
    picam2.start()

    print("=" * 60)
    print("AI Camera Ready!")
    print("Open browser: http://192.168.1.44:8080")
    print("=" * 60)
    sys.stdout.flush()

    while True:
        try:
            # Capture frame
            frame = picam2.capture_array("main")

            # Get metadata and parse detections
            metadata = picam2.capture_metadata()
            parse_detections(metadata, imx500)

            # Convert RGB to BGR and update global
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            with frame_lock:
                latest_frame = frame_bgr

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(0.1)


def main():
    print("Starting Raspberry Pi AI Camera Test")
    print("-" * 60)

    # Start camera thread
    cam_thread = threading.Thread(target=camera_loop, daemon=True)
    cam_thread.start()

    # Wait for camera to start
    time.sleep(3)

    # Start HTTP server
    server = HTTPServer(('0.0.0.0', 8080), StreamHandler)
    print("HTTP Server started on port 8080")
    sys.stdout.flush()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
