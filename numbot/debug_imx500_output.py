#!/usr/bin/env python3
"""
Debug script to check IMX500 inference output format
"""
import time
import sys

try:
    from picamera2 import Picamera2
    from picamera2.devices import IMX500
    from picamera2.devices.imx500 import NetworkIntrinsics
    import numpy as np
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Model path - use YOLOv8n
MODEL = "/home/pi/.modlib/zoo/imx500_network_yolov8n_pp.rpk"

print(f"Model: {MODEL}")
print("Initializing IMX500...")

imx500 = IMX500(MODEL)
intrinsics = imx500.network_intrinsics

print(f"\n=== Network Intrinsics ===")
if intrinsics:
    print(f"  Task: {intrinsics.task}")
    print(f"  Postprocess: {intrinsics.postprocess}")
    print(f"  Inference Rate: {intrinsics.inference_rate}")
    print(f"  BBox Normalization: {intrinsics.bbox_normalization}")
    print(f"  BBox Order: {intrinsics.bbox_order}")
    print(f"  Labels: {len(intrinsics.labels) if intrinsics.labels else 'None'}")
else:
    print("  (No intrinsics)")

print(f"\nInput Size: {imx500.get_input_size()}")
print(f"Camera Num: {imx500.camera_num}")

print("\nStarting camera...")
picam2 = Picamera2(imx500.camera_num)
config = picam2.create_preview_configuration(
    controls={"FrameRate": 10},
    buffer_count=12
)
imx500.show_network_fw_progress_bar()
picam2.start(config, show_preview=False)
print("Camera started!")

print("\n=== Checking inference outputs ===")
for i in range(10):
    metadata = picam2.capture_metadata()
    
    try:
        np_outputs = imx500.get_outputs(metadata, add_batch=True)
    except Exception as e:
        print(f"Frame {i+1}: Error getting outputs: {e}")
        try:
            np_outputs = imx500.get_outputs(metadata, add_batch=False)
        except:
            np_outputs = None
    
    if np_outputs is None:
        print(f"Frame {i+1}: No outputs yet...")
    else:
        print(f"\nFrame {i+1}: Got {len(np_outputs)} output tensors")
        for j, out in enumerate(np_outputs):
            print(f"  Output[{j}]: shape={out.shape}, dtype={out.dtype}")
            if len(out.shape) <= 2:
                # Show first few values
                flat = out.flatten()
                if len(flat) > 0:
                    non_zero = flat[flat != 0]
                    if len(non_zero) > 0:
                        print(f"    Non-zero values: {len(non_zero)}, min={non_zero.min():.3f}, max={non_zero.max():.3f}")
                    else:
                        print(f"    All zeros")
        break
    
    time.sleep(0.5)

print("\nCleaning up...")
picam2.stop()
picam2.close()
print("Done!")
