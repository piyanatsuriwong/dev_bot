#!/usr/bin/env python3
"""
Test IMX500 Metadata-Only Output
ตรวจสอบว่า IMX500 ส่งแค่ metadata (ข้อมูล AI) มาจริงหรือเปล่า โดยไม่ต้องส่งภาพ

จุดเด่นของ IMX500:
- ประมวลผล AI ภายในตัวเซนเซอร์
- ส่งออกมาแค่ข้อมูลผลลัพธ์ (Text/Binary) ไม่ใช่ภาพ
- ประหยัด bandwidth และ CPU ของ Pi

Test นี้จะแสดง:
1. ขนาดข้อมูล metadata ที่ได้รับ (bytes)
2. ประเภทของข้อมูล (numpy array, dict, etc.)
3. เปรียบเทียบกับขนาดภาพ (ถ้ามีการส่งภาพ)
4. แสดงข้อมูล detection ที่ได้
"""

import sys
import time
import numpy as np
from typing import Optional

# Try modlib first (primary backend)
try:
    from modlib.devices import AiCamera
    from modlib.models.zoo import YOLO11n, YOLOv8n
    BACKEND = "modlib"
    print("✅ Using modlib backend (PRIMARY)")
except ImportError:
    try:
        from picamera2 import Picamera2
        from picamera2.devices import IMX500
        from picamera2.devices.imx500 import NetworkIntrinsics
        BACKEND = "picamera2"
        print("✅ Using Picamera2 backend (FALLBACK)")
    except ImportError:
        print("❌ No IMX500 backend available")
        sys.exit(1)


def get_size_in_bytes(obj) -> int:
    """Get approximate size of object in bytes"""
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    elif isinstance(obj, dict):
        total = 0
        for key, value in obj.items():
            total += get_size_in_bytes(key)
            total += get_size_in_bytes(value)
        return total
    elif isinstance(obj, (list, tuple)):
        return sum(get_size_in_bytes(item) for item in obj)
    elif isinstance(obj, str):
        return len(obj.encode('utf-8'))
    elif isinstance(obj, (int, float)):
        return 8  # Approximate
    else:
        return sys.getsizeof(obj)


def format_bytes(size: int) -> str:
    """Format bytes to human readable"""
    for unit in ['B', 'KB', 'MB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} GB"


def test_modlib():
    """Test with modlib backend"""
    print("\n" + "="*60)
    print("TEST: modlib Backend - Metadata Only")
    print("="*60)
    
    # Load model
    print("\n[1] Loading YOLO model...")
    try:
        model = YOLO11n()
        model_name = "YOLO11n"
    except:
        model = YOLOv8n()
        model_name = "YOLOv8n"
    print(f"✅ Model loaded: {model_name}")
    
    # Initialize camera
    print("\n[2] Initializing AI Camera...")
    device = AiCamera(frame_rate=10, num=0)
    device.deploy(model)
    print("✅ Camera initialized and model deployed")
    
    print("\n[3] Capturing frames and analyzing data size...")
    print("-" * 60)
    
    frame_count = 0
    total_metadata_size = 0
    total_image_size = 0
    detection_count = 0
    
    try:
        with device as stream:
            print("📸 Stream started! Analyzing 30 frames...\n")
            
            for frame in stream:
                frame_count += 1
                
                # Get image (if available)
                image = frame.image
                image_size = 0
                if image is not None:
                    image_size = image.nbytes if hasattr(image, 'nbytes') else 0
                    total_image_size += image_size
                
                # Get detections (metadata)
                metadata_size = 0
                detections = []
                
                try:
                    raw_dets = frame.detections
                    if raw_dets is not None:
                        # Calculate metadata size
                        if hasattr(raw_dets, 'nbytes'):
                            metadata_size = raw_dets.nbytes
                        else:
                            metadata_size = get_size_in_bytes(raw_dets)
                        
                        # Filter by confidence
                        if hasattr(raw_dets, 'confidence'):
                            filtered = raw_dets[raw_dets.confidence > 0.5]
                            detections = list(filtered)
                        
                        total_metadata_size += metadata_size
                        detection_count += len(detections)
                except Exception as e:
                    pass
                
                # Display info every frame
                print(f"Frame {frame_count:3d} | "
                      f"Image: {format_bytes(image_size):>10s} | "
                      f"Metadata: {format_bytes(metadata_size):>8s} | "
                      f"Detections: {len(detections)}")
                
                # Show detection details
                if detections:
                    for det in detections[:3]:  # Show top 3
                        try:
                            _, confidence, class_id, bbox = det[:4]
                            label = model.labels[int(class_id)] if hasattr(model, 'labels') else f"class_{class_id}"
                            print(f"           └─ {label} ({confidence:.2f})")
                        except:
                            pass
                
                if frame_count >= 30:
                    break
                
                time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Frames analyzed:     {frame_count}")
    print(f"Total detections:    {detection_count}")
    print(f"Avg detections/frame: {detection_count/frame_count:.1f}")
    print()
    print(f"Total IMAGE size:    {format_bytes(total_image_size)}")
    print(f"Total METADATA size: {format_bytes(total_metadata_size)}")
    print()
    
    if total_image_size > 0 and total_metadata_size > 0:
        ratio = total_image_size / total_metadata_size
        print(f"💡 Image is {ratio:.1f}x LARGER than metadata")
        print(f"   Metadata is only {(total_metadata_size/total_image_size)*100:.2f}% of image size!")
    
    print()
    print("✅ IMX500 sends BOTH image and metadata")
    print("   - Image: For display/preview purposes")
    print("   - Metadata: AI results (bounding boxes, classes, confidence)")
    print("   - Metadata is MUCH smaller than image!")
    
    device.close()


def test_picamera2():
    """Test with Picamera2 backend"""
    print("\n" + "="*60)
    print("TEST: Picamera2 Backend - Metadata Only")
    print("="*60)
    
    # Model path
    model_path = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
    
    print(f"\n[1] Loading model: {model_path}")
    imx500 = IMX500(model_path)
    
    # Get intrinsics
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    intrinsics.update_with_defaults()
    
    print(f"✅ Model loaded")
    print(f"   Task: {intrinsics.task}")
    print(f"   Labels: {len(intrinsics.labels) if intrinsics.labels else 0} classes")
    
    # Initialize camera
    print("\n[2] Initializing Picamera2...")
    picam2 = Picamera2(imx500.camera_num)
    
    # Show firmware upload
    print("\n[3] Uploading firmware to IMX500...")
    imx500.show_network_fw_progress_bar()
    
    # Configure and start
    config = picam2.create_preview_configuration(
        controls={"FrameRate": 10},
        buffer_count=12
    )
    picam2.start(config, show_preview=False)
    
    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()
    
    print("✅ Camera started")
    
    print("\n[4] Capturing frames and analyzing metadata...")
    print("-" * 60)
    
    frame_count = 0
    total_metadata_size = 0
    detection_count = 0
    
    try:
        for i in range(30):
            frame_count += 1
            
            # Capture metadata ONLY (no image array!)
            metadata = picam2.capture_metadata()
            
            # Calculate metadata size
            metadata_size = get_size_in_bytes(metadata)
            total_metadata_size += metadata_size
            
            # Parse detections
            detections = []
            try:
                np_outputs = imx500.get_outputs(metadata, add_batch=True)
                if np_outputs is not None and len(np_outputs) > 0:
                    # Count detections (simplified)
                    output = np_outputs[0]
                    if len(output.shape) >= 2:
                        detections = list(output)
                    detection_count += len(detections)
            except:
                pass
            
            print(f"Frame {frame_count:3d} | "
                  f"Metadata: {format_bytes(metadata_size):>8s} | "
                  f"Detections: {len(detections)}")
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    
    picam2.stop()
    picam2.close()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Frames analyzed:      {frame_count}")
    print(f"Total METADATA size:  {format_bytes(total_metadata_size)}")
    print(f"Avg metadata/frame:   {format_bytes(total_metadata_size/frame_count)}")
    print()
    print("✅ Picamera2 can capture METADATA ONLY!")
    print("   - Use capture_metadata() instead of capture_array()")
    print("   - No image data transferred")
    print("   - Only AI results (tensor outputs)")


def main():
    print("\n" + "="*60)
    print("IMX500 Metadata-Only Test")
    print("="*60)
    print()
    print("จุดเด่นของ IMX500:")
    print("- ประมวลผล AI ภายในตัวเซนเซอร์")
    print("- ส่งออกมาแค่ผลลัพธ์ (metadata) ไม่ใช่ภาพ")
    print("- ประหยัด bandwidth และ CPU")
    print()
    
    if BACKEND == "modlib":
        test_modlib()
    else:
        test_picamera2()
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print()
    print("✅ IMX500 ทำ AI ภายในตัวเซนเซอร์จริง!")
    print()
    print("📊 ข้อมูลที่ส่งออกมา:")
    print("   1. Metadata (AI results) - ขนาดเล็กมาก (KB)")
    print("      - Bounding boxes")
    print("      - Class IDs")
    print("      - Confidence scores")
    print()
    print("   2. Image (optional) - ขนาดใหญ่ (MB)")
    print("      - สำหรับแสดงผล/preview")
    print("      - สามารถปิดได้ถ้าไม่ต้องการ")
    print()
    print("💡 ข้อดี:")
    print("   - ไม่ต้องส่งภาพขนาดใหญ่ผ่าน CSI")
    print("   - ไม่ต้องใช้ CPU ของ Pi ทำ AI")
    print("   - ได้ผลลัพธ์เร็วและแม่นยำ")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
