#!/usr/bin/env python3
"""
YOLO Object Detection on Raspberry Pi AI Camera (IMX500)
Using Sony Application Module Library (modlib)
"""

import sys
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

import cv2
import numpy as np

# Import modlib
from modlib.devices import AiCamera
from modlib.models.zoo import YOLOv8n  # or YOLO11n
from modlib.apps import Annotator

# Global
latest_frame = None
frame_lock = threading.Lock()
running = True


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
    <title>YOLO on IMX500</title>
    <style>
        body { background: #1a1a2e; color: white; text-align: center; font-family: Arial; padding: 20px; }
        h1 { color: #ff6b6b; }
        img { max-width: 100%; border: 3px solid #ff6b6b; border-radius: 10px; }
        .info { background: #16213e; padding: 15px; border-radius: 10px; margin: 20px auto; max-width: 800px; }
    </style>
</head>
<body>
    <h1>YOLOv8n on Raspberry Pi AI Camera (IMX500)</h1>
    <div class="info">Real-time Object Detection - 80 COCO Classes</div>
    <img src="/stream">
    <div class="info">Press Ctrl+C to stop</div>
</body>
</html>'''
            self.wfile.write(html.encode())

        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()

            while running:
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


def main():
    global latest_frame, running

    print("=" * 60)
    print("YOLO Object Detection on IMX500")
    print("=" * 60)

    # Initialize model
    print("Loading YOLOv8n model...")
    model = YOLOv8n()
    print(f"Model loaded! Classes: {len(model.labels)}")

    # Initialize camera
    print("Initializing AI Camera...")
    device = AiCamera(frame_rate=30)

    # Deploy model
    print("Deploying model to IMX500 (this may take a few minutes)...")
    device.deploy(model)

    # Annotator for drawing
    annotator = Annotator()

    # Start HTTP server
    server = HTTPServer(('0.0.0.0', 8080), StreamHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print("=" * 60)
    print("YOLO Ready!")
    print("Open browser: http://192.168.1.47:8080")
    print("=" * 60)
    sys.stdout.flush()

    try:
        with device as stream:
            for frame in stream:
                # Filter detections by confidence
                detections = frame.detections[frame.detections.confidence > 0.5]

                # Create labels
                labels = [
                    f"{model.labels[int(class_id)]}: {score:.2f}"
                    for _, score, class_id, _ in detections
                ]

                # Draw boxes
                annotator.annotate_boxes(frame, detections, labels=labels)

                # Get frame image (numpy array)
                frame_image = frame.image

                # Convert to BGR for OpenCV
                if frame_image is not None and len(frame_image.shape) == 3:
                    frame_bgr = cv2.cvtColor(frame_image, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame_image if frame_image is not None else np.zeros((480, 640, 3), dtype=np.uint8)

                # Add status text
                cv2.putText(frame_bgr, f"YOLO - Detections: {len(detections)}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # Update global frame
                with frame_lock:
                    latest_frame = frame_bgr

                # Also display locally if HDMI connected
                # frame.display()

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        running = False
        server.shutdown()


if __name__ == "__main__":
    main()
