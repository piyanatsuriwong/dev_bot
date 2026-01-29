#!/usr/bin/env python3
"""
YOLO26n for Pi5 1GB RAM - Ultra Optimized
Usage: python3 yolo26n_lowram.py [--camera N] [--conf 0.5]
"""

import argparse
import numpy as np
import cv2
from ultralytics import YOLO
import time
import signal
import sys
import psutil
import os
import gc  # Garbage collector

# Parse arguments
parser = argparse.ArgumentParser(description='YOLO26n Low-RAM Detection')
parser.add_argument('--camera', type=int, default=1, help='Camera index (0=IMX500, 1=IMX708)')
parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
args = parser.parse_args()

def signal_handler(sig, frame):
    print("\n👋 Exiting...")
    gc.collect()  # ล้าง RAM
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

print("🚀 YOLO26n Low-RAM Mode (416x416)")
print("=" * 60)

# 1. เช็ค RAM
mem = psutil.virtual_memory()
print(f"💾 RAM: {mem.used/1024**2:.0f}MB used / {mem.total/1024**2:.0f}MB total")
print(f"   Available: {mem.available/1024**2:.0f}MB")
if mem.available < 400 * 1024**2:
    print("⚠️  Available RAM < 400MB - อาจช้า!")

# 2. โหลดโมเดล (416x416)
model_path = "yolo26n_ncnn_model"  # หรือ yolo26n_416_ncnn_model
if not os.path.exists(model_path):
    print(f"❌ Model not found: {model_path}")
    print("   Export ด้วย: yolo export model=yolo26n.pt format=ncnn imgsz=416")
    sys.exit(1)

print(f"📦 Loading {model_path}...")
model = YOLO(model_path, task='detect')
gc.collect()  # ล้าง RAM หลังโหลด
print("✅ Model loaded")

# Warmup inference (โหลด NCNN ก่อน)
print("🔥 Warming up NCNN...", flush=True)
warmup_img = np.zeros((416, 416, 3), dtype=np.uint8)
warmup_start = time.time()
_ = model.predict(warmup_img, conf=0.5, verbose=False, device='cpu')
print(f"✅ NCNN ready ({time.time()-warmup_start:.2f}s)", flush=True)
gc.collect()

# 3. ตั้งค่ากล้อง (ใช้ resolution เดียวกับโมเดล)
print("📷 Starting camera (416x416)...")
camera = None
camera_type = None

# ลำดับความสำคัญ: Picamera2 > modlib (ถ้า IMX500) > rpicam
camera_initialized = False

# Method 1: ลองใช้ Picamera2 ก่อน (รองรับทั้ง IMX500 และ IMX708)
if not camera_initialized:
    try:
        from picamera2 import Picamera2
        cam_idx = args.camera
        print(f"   Trying Picamera2 for camera {cam_idx}...")
        picam2 = Picamera2(cam_idx)
        config = picam2.create_preview_configuration(
            main={"size": (416, 416), "format": "RGB888"},
            buffer_count=2  # ลด buffer
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(1)
        camera = picam2
        camera_type = "picamera2"
        camera_initialized = True
        print(f"✅ Camera ready (Picamera2 cam={cam_idx})")
    except (ImportError, ValueError, Exception) as e:
        error_msg = str(e)
        if "numpy.dtype" in error_msg:
            print(f"⚠️  Picamera2 numpy incompatibility detected")
            print("   Trying alternative methods...")
        else:
            print(f"⚠️  Picamera2 failed: {error_msg}")
            print("   Trying alternative methods...")

# Method 2: ไม่ใช้ modlib.AiCamera (เพราะต้อง deploy model)
# เน้นให้แก้ปัญหา numpy เพื่อใช้ Picamera2 แทน

# Method 3: ใช้ rpicam-vid (ถ้ามี)
if not camera_initialized:
    try:
        import subprocess
        # ตรวจสอบว่ามี rpicam-vid หรือไม่
        result = subprocess.run(['which', 'rpicam-vid'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            print("   rpicam-vid found, but requires more setup...")
            print("   (Skipping for now)")
    except:
        pass

# ถ้ายังไม่ได้ camera
if not camera_initialized:
    print("\n❌ Cannot initialize camera")
    print("\n💡 Solutions (ต้องแก้เพื่อใช้ YOLO26n อย่างเดียว):")
    print("   1. แก้ปัญหา numpy incompatibility (แนะนำ):")
    print("      pip uninstall simplejpeg -y && pip install simplejpeg --no-cache-dir")
    print("      หรือ: pip uninstall numpy -y && pip install numpy --no-cache-dir")
    print("   2. ใช้ IMX708 แทน (camera 1):")
    print("      python3 yolo26n_lowram.py --camera 1")
    print("   3. ตรวจสอบกล้อง:")
    print("      v4l2-ctl --list-devices")
    print("      rpicam-hello --list-cameras")
    print("\n⚠️  หมายเหตุ: ต้องแก้ปัญหา numpy เพื่อใช้ Picamera2")
    print("   เพื่อให้ใช้ YOLO26n อย่างเดียว (ไม่ต้องใช้ YOLO11n dummy model)")
    sys.exit(1)

print("\n🎥 Detection started (Ctrl+C to stop)")
print("-" * 60)

stats = {
    "frames": 0,
    "detections": 0,
    "start": time.time(),
    "last_print": time.time()
}

try:
    first_frame = True
    while True:
        # จับภาพ
        try:
            if camera_type == "picamera2":
                frame = camera.capture_array()
            else:
                raise ValueError(f"Unknown camera type: {camera_type}")
            
            if first_frame:
                print(f"📸 First frame captured: {frame.shape}")
                first_frame = False
                
        except Exception as e:
            print(f"⚠️  Failed to capture frame: {e}")
            time.sleep(0.1)
            continue
        
        stats["frames"] += 1
        
        # ตรวจจับ
        results = model.predict(
            frame,
            conf=args.conf,
            verbose=False,
            device='cpu',
            max_det=50  # จำกัด detections
        )
        
        # ประมวลผล
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            stats["detections"] += len(boxes)
            
            # แสดงเฉพาะ class ที่สำคัญ
            for box in boxes[:3]:  # แสดงแค่ 3 ตัวแรก
                cls = int(box.cls)
                conf = float(box.conf)
                name = model.names[cls]
                print(f"🎯 {name}: {conf:.1%}")
        
        # Stats ทุก 5 วินาที (ลดจาก 3)
        if time.time() - stats["last_print"] >= 5.0:
            duration = time.time() - stats["start"]
            fps = stats["frames"] / duration
            mem = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=0.1)
            
            print("-" * 60)
            print(f"📊 FPS: {fps:.1f} | Frames: {stats['frames']}")
            print(f"   CPU: {cpu:.1f}% | RAM: {mem.used/1024**2:.0f}MB ({mem.percent:.1f}%)")
            print(f"   Available: {mem.available/1024**2:.0f}MB")
            print("-" * 60)
            
            stats["last_print"] = time.time()
            
            # ล้าง RAM ทุก 5 วินาที
            gc.collect()
            
            # เตือนถ้า RAM ต่ำ
            if mem.available < 200 * 1024**2:
                print("⚠️  WARNING: RAM < 200MB!")

except KeyboardInterrupt:
    pass
finally:
    # ปิดกล้อง
    try:
        if camera_type == "picamera2":
            camera.stop()
    except Exception as e:
        print(f"⚠️  Error closing camera: {e}")
    gc.collect()
    
    duration = time.time() - stats["start"]
    fps = stats["frames"] / duration if duration > 0 else 0
    
    print("\n" + "=" * 60)
    print(f"📈 FPS: {fps:.1f} | Frames: {stats['frames']}")
    print(f"   Detections: {stats['detections']}")
    print("=" * 60)
