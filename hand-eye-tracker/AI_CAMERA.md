# Raspberry Pi AI Camera (IMX500) Setup Guide

## Overview

Raspberry Pi AI Camera ใช้ Sony IMX500 Intelligent Vision Sensor ที่มี Neural Network Accelerator ในตัว สามารถทำ AI Inference ได้โดยตรงบนกล้อง

## Hardware

| Component | Specification |
|-----------|---------------|
| Sensor | Sony IMX500 |
| Resolution | 4056x3040 (12.3MP) |
| AI Accelerator | Built-in NN Processor |
| Interface | CSI (Camera Serial Interface) |
| Supported Boards | Raspberry Pi 4, Pi 5 |

---

## Installation

### 1. Update System
```bash
sudo apt update && sudo apt full-upgrade -y
```

### 2. Install IMX500 Package
```bash
sudo apt install -y imx500-all
```

### 3. Reboot
```bash
sudo reboot
```

### 4. Verify Installation
```bash
# Check firmware files
ls /lib/firmware/imx500*

# Check AI models
ls /usr/share/imx500-models/

# Check camera detection
rpicam-hello --list-cameras
```

---

## Available AI Models

| Model | Task | File |
|-------|------|------|
| **YOLO11n** | Object Detection | `imx500_network_yolo11n_pp.rpk` |
| **YOLOv8n** | Object Detection | `imx500_network_yolov8n_pp.rpk` |
| NanoDet | Object Detection | `imx500_network_nanodet_plus_416x416_pp.rpk` |
| SSD MobileNet | Object Detection | `imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk` |
| EfficientDet | Object Detection | `imx500_network_efficientdet_lite0_pp.rpk` |
| PoseNet | Pose Estimation | `imx500_network_posenet.rpk` |
| HigherHRNet | Pose Estimation | `imx500_network_higherhrnet_coco.rpk` |
| DeepLabV3+ | Segmentation | `imx500_network_deeplabv3plus.rpk` |
| EfficientNet | Classification | `imx500_network_efficientnet_*.rpk` |
| MobileNet V2 | Classification | `imx500_network_mobilenet_v2.rpk` |
| ResNet18 | Classification | `imx500_network_resnet18.rpk` |

### YOLO Performance (on IMX500)

| Model | mAP@640 | Inference Time |
|-------|---------|----------------|
| YOLO11n | 0.374 | ~58ms |
| YOLOv8n | 0.279 | ~58ms |

---

## Quick Start Commands

### Object Detection (MobileNet SSD)
```bash
rpicam-hello -t 0 --post-process-file /usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json --viewfinder-width 1920 --viewfinder-height 1080
```

### Pose Estimation (PoseNet)
```bash
rpicam-hello -t 0 --post-process-file /usr/share/rpi-camera-assets/imx500_posenet.json
```

### Capture Image with Detection
```bash
rpicam-still -o test.jpg --post-process-file /usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json
```

### Record Video with Detection
```bash
rpicam-vid -t 10000 -o test.h264 --post-process-file /usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json
```

---

## Python Code Example

### Object Detection with HTTP Streaming

```python
#!/usr/bin/env python3
"""
Raspberry Pi AI Camera - Object Detection with HTTP Streaming
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

# COCO Labels (80 classes)
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

# Global variables
latest_frame = None
frame_lock = threading.Lock()
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
    <title>AI Camera</title>
    <style>
        body { background: #1a1a2e; color: white; text-align: center; font-family: Arial; }
        h1 { color: #00ff88; }
        img { max-width: 100%; border: 3px solid #00ff88; border-radius: 10px; }
    </style>
</head>
<body>
    <h1>Raspberry Pi AI Camera</h1>
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
    """Callback to draw detections on frame"""
    global last_boxes, last_scores, last_classes

    with MappedArray(request, "main") as m:
        frame = m.array
        h, w = frame.shape[:2]

        for i in range(len(last_boxes)):
            box = last_boxes[i]
            score = last_scores[i]
            cls = int(last_classes[i])

            if score < 0.5:
                continue

            x_c, y_c, bw, bh = box
            x1 = int((x_c - bw/2) * w / 416)
            y1 = int((y_c - bh/2) * h / 416)
            x2 = int((x_c + bw/2) * w / 416)
            y2 = int((y_c + bh/2) * h / 416)

            color = (0, 255, 136)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = LABELS[cls] if cls < len(LABELS) else f"class_{cls}"
            cv2.putText(frame, f"{label}: {score:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, f"Detections: {len(last_boxes)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 136), 2)


def parse_detections(metadata, imx500):
    """Parse detections from IMX500"""
    global last_boxes, last_scores, last_classes

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    if np_outputs is None:
        return

    boxes, scores, classes = postprocess_nanodet_detection(
        outputs=np_outputs[0], conf=0.5, iou_thres=0.5, max_out_dets=10
    )

    last_boxes = boxes[0] if len(boxes) > 0 else []
    last_scores = scores[0] if len(scores) > 0 else []
    last_classes = classes[0] if len(classes) > 0 else []


def main():
    global latest_frame

    # Initialize IMX500
    model = "/usr/share/imx500-models/imx500_network_nanodet_plus_416x416_pp.rpk"
    imx500 = IMX500(model)

    # Setup camera
    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"},
        controls={"FrameRate": 30}
    )
    picam2.configure(config)
    picam2.pre_callback = draw_detections
    picam2.start()

    print("AI Camera Ready! Open: http://<PI_IP>:8080")

    # Start HTTP server in background
    server = HTTPServer(('0.0.0.0', 8080), StreamHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    # Main loop
    while True:
        frame = picam2.capture_array("main")
        metadata = picam2.capture_metadata()
        parse_detections(metadata, imx500)

        with frame_lock:
            latest_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    main()
```

---

## Post-Process Configuration Files

Location: `/usr/share/rpi-camera-assets/`

| File | Description |
|------|-------------|
| `imx500_mobilenet_ssd.json` | MobileNet SSD Object Detection |
| `imx500_posenet.json` | PoseNet Pose Estimation |

---

## Important Notes

1. **First Run**: Loading AI model firmware takes 2-5 minutes on first run
2. **Firmware Cache**: After first load, subsequent starts are faster
3. **Display Required**: `rpicam-hello` requires HDMI display or X server
4. **HTTP Streaming**: Use Python script for headless/remote viewing
5. **Multiple Cameras**: IMX500 can work alongside regular Pi Camera

---

## Troubleshooting

### Camera Not Detected
```bash
# Check camera connection
rpicam-hello --list-cameras

# Check kernel modules
lsmod | grep imx500
```

### Slow Startup
- Normal on first run (firmware upload)
- Wait for "Network Firmware Upload: 100%"

### No Detection Output
- Ensure object is within camera view
- Check confidence threshold (default 0.5)
- Verify correct model file path

---

## SSH Commands Reference

```bash
# Run object detection (HDMI)
ssh pi@192.168.1.44 "DISPLAY=:0 rpicam-hello -t 0 --post-process-file /usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json"

# Run Python script (HTTP streaming)
ssh pi@192.168.1.44 "cd /home/pi/hand-eye-tracker && source env/bin/activate && python3 test_ai_camera.py"

# Kill camera process
ssh pi@192.168.1.44 "pkill rpicam; pkill python3"

# Check running processes
ssh pi@192.168.1.44 "ps aux | grep -E 'rpicam|python3'"
```

---

## Hardware Pin Configuration (Pi 5)

AI Camera connects via CSI port (no GPIO required)

### If using with GC9A01A Display:
| Component | GPIO | Pin |
|-----------|------|-----|
| DC | GPIO 24 | Pin 18 |
| RST | GPIO 25 | Pin 22 |
| BL | GPIO 23 | Pin 16 |
| CS | GPIO 8 (CE0) | Pin 24 |
| MOSI | GPIO 10 | Pin 19 |
| SCLK | GPIO 11 | Pin 23 |

---

---

## YOLO on IMX500

### Install YOLO Models

```bash
# Install all IMX500 models including YOLO
sudo apt install imx500-models

# Check YOLO models
ls /usr/share/imx500-models/ | grep yolo
```

### Download Picamera2 Examples

```bash
git clone https://github.com/raspberrypi/picamera2.git
cd picamera2/examples/imx500
```

### Run YOLO11n Object Detection

```bash
python imx500_object_detection_demo.py \
  --model /usr/share/imx500-models/imx500_network_yolo11n_pp.rpk \
  --bbox-normalization --bbox-order xy
```

### Run YOLOv8n Object Detection

```bash
python imx500_object_detection_demo.py \
  --model /usr/share/imx500-models/imx500_network_yolov8n_pp.rpk \
  --bbox-normalization --bbox-order xy
```

---

## Custom YOLO Model Export (Ultralytics)

### Install Ultralytics

```bash
pip install ultralytics
```

### Export YOLOv8/YOLO11 to IMX500 Format

**Python:**
```python
from ultralytics import YOLO

# Load model
model = YOLO("yolo11n.pt")  # or yolov8n.pt

# Export to IMX500 format
model.export(format="imx", data="coco8.yaml")
```

**CLI:**
```bash
yolo export model=yolo11n.pt format="imx" data=coco8.yaml
```

Output: `yolo11n_imx_model/packerOut.zip` (สำหรับ deploy)

### Deploy Custom Model with Application Module Library

```bash
pip install git+https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library.git
```

**Python Code:**
```python
import numpy as np
from modlib.apps import Annotator
from modlib.devices import AiCamera
from modlib.models import COLOR_FORMAT, MODEL_TYPE, Model
from modlib.models.post_processors import pp_od_yolo_ultralytics

class YOLO(Model):
    def __init__(self):
        super().__init__(
            model_file="yolo11n_imx_model/packerOut.zip",
            model_type=MODEL_TYPE.CONVERTED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )
        self.labels = np.genfromtxt(
            "yolo11n_imx_model/labels.txt",
            dtype=str,
            delimiter="\n",
        )

    def post_process(self, output_tensors):
        return pp_od_yolo_ultralytics(output_tensors)

# Initialize
device = AiCamera(frame_rate=16)
model = YOLO()
device.deploy(model)
annotator = Annotator()

# Run detection
with device as stream:
    for frame in stream:
        detections = frame.detections[frame.detections.confidence > 0.55]
        labels = [f"{model.labels[class_id]}: {score:0.2f}"
                  for _, score, class_id, _ in detections]
        annotator.annotate_boxes(frame, detections, labels=labels)
        frame.display()
```

### Export Arguments

| Parameter | Default | Description |
|-----------|---------|-------------|
| `format` | `"imx"` | Export format for IMX500 |
| `imgsz` | 640 | Input image size |
| `int8` | True | INT8 quantization (required) |
| `data` | `"coco8.yaml"` | Dataset for quantization calibration |

---

## YOLO with Picamera2 (Alternative Method)

```python
#!/usr/bin/env python3
"""YOLO Object Detection with IMX500 and Picamera2"""

from picamera2 import Picamera2, MappedArray
from picamera2.devices import IMX500
from picamera2.devices.imx500 import postprocess_yolov8
import cv2

# COCO Labels
LABELS = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", ...]

def draw_detections(request, boxes, scores, classes):
    with MappedArray(request, "main") as m:
        for box, score, cls in zip(boxes, scores, classes):
            if score > 0.5:
                x1, y1, x2, y2 = box
                label = LABELS[int(cls)]
                cv2.rectangle(m.array, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(m.array, f"{label}: {score:.2f}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Initialize
model = "/usr/share/imx500-models/imx500_network_yolov8n_pp.rpk"
imx500 = IMX500(model)
picam2 = Picamera2(imx500.camera_num)

config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

while True:
    metadata = picam2.capture_metadata()
    outputs = imx500.get_outputs(metadata, add_batch=True)

    if outputs:
        boxes, scores, classes = postprocess_yolov8(outputs[0])
        # Draw and display...
```

---

## Resources

- [Raspberry Pi AI Camera Documentation](https://www.raspberrypi.com/documentation/accessories/ai-camera.html)
- [Picamera2 GitHub](https://github.com/raspberrypi/picamera2)
- [IMX500 Model Zoo](https://github.com/raspberrypi/imx500-models)
- [Ultralytics IMX500 Export Guide](https://docs.ultralytics.com/integrations/sony-imx500/)
- [Sony Application Module Library](https://github.com/SonySemiconductorSolutions/aitrios-rpi-application-module-library)
- [IMX500 Getting Started](https://www.raspberrypi.com/news/how-to-get-started-with-your-raspberry-pi-ai-camera/)
