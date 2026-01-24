#!/usr/bin/env python3
"""
IMX500 Metadata vs Image Size Comparison
เปรียบเทียบขนาดข้อมูลระหว่าง Metadata (AI results) กับ Image (ภาพจริง)

Test 2 แบบ:
1. Metadata ONLY (capture_metadata) - ไม่ส่งภาพ
2. Metadata + Image (capture_array) - ส่งทั้งภาพและ metadata
"""

import sys
import time
import numpy as np
from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics


def format_bytes(size: int) -> str:
    """Format bytes to human readable"""
    for unit in ['B', 'KB', 'MB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} GB"


def get_metadata_size(metadata: dict) -> int:
    """Calculate metadata size"""
    total = 0
    for key, value in metadata.items():
        if isinstance(value, np.ndarray):
            total += value.nbytes
        elif isinstance(value, (int, float)):
            total += 8
        elif isinstance(value, str):
            total += len(value.encode('utf-8'))
        elif isinstance(value, dict):
            total += get_metadata_size(value)
    return total


def test_metadata_only(picam2, imx500, num_frames=20):
    """Test 1: Metadata ONLY (no image)"""
    print("\n" + "="*60)
    print("TEST 1: Metadata ONLY (capture_metadata)")
    print("="*60)
    print("📊 ส่งแค่ผลลัพธ์ AI ไม่ส่งภาพ\n")
    
    total_size = 0
    detection_count = 0
    
    for i in range(num_frames):
        # Capture metadata ONLY
        metadata = picam2.capture_metadata()
        
        # Calculate size
        metadata_size = get_metadata_size(metadata)
        total_size += metadata_size
        
        # Parse detections
        detections = 0
        try:
            np_outputs = imx500.get_outputs(metadata, add_batch=True)
            if np_outputs is not None and len(np_outputs) > 0:
                output = np_outputs[0]
                if len(output.shape) >= 2:
                    detections = len(output)
                detection_count += detections
        except:
            pass
        
        print(f"Frame {i+1:2d} | Metadata: {format_bytes(metadata_size):>10s} | Detections: {detections}")
        time.sleep(0.05)
    
    avg_size = total_size / num_frames
    print(f"\n📊 Summary:")
    print(f"   Total:   {format_bytes(total_size)}")
    print(f"   Average: {format_bytes(avg_size)}/frame")
    print(f"   Detections: {detection_count} total")
    
    return total_size, avg_size


def test_metadata_and_image(picam2, imx500, num_frames=20):
    """Test 2: Metadata + Image"""
    print("\n" + "="*60)
    print("TEST 2: Metadata + Image (capture_array)")
    print("="*60)
    print("📊 ส่งทั้งภาพและผลลัพธ์ AI\n")
    
    total_metadata_size = 0
    total_image_size = 0
    detection_count = 0
    
    for i in range(num_frames):
        # Capture metadata first
        metadata = picam2.capture_metadata()
        
        # Capture image array
        image = picam2.capture_array()
        
        # Calculate sizes
        metadata_size = get_metadata_size(metadata)
        image_size = image.nbytes if hasattr(image, 'nbytes') else 0
        
        total_metadata_size += metadata_size
        total_image_size += image_size
        
        # Parse detections
        detections = 0
        try:
            np_outputs = imx500.get_outputs(metadata, add_batch=True)
            if np_outputs is not None and len(np_outputs) > 0:
                output = np_outputs[0]
                if len(output.shape) >= 2:
                    detections = len(output)
                detection_count += detections
        except:
            pass
        
        print(f"Frame {i+1:2d} | "
              f"Image: {format_bytes(image_size):>10s} | "
              f"Metadata: {format_bytes(metadata_size):>10s} | "
              f"Detections: {detections}")
        time.sleep(0.05)
    
    avg_metadata = total_metadata_size / num_frames
    avg_image = total_image_size / num_frames
    
    print(f"\n📊 Summary:")
    print(f"   Image total:    {format_bytes(total_image_size)}")
    print(f"   Image avg:      {format_bytes(avg_image)}/frame")
    print(f"   Metadata total: {format_bytes(total_metadata_size)}")
    print(f"   Metadata avg:   {format_bytes(avg_metadata)}/frame")
    print(f"   Detections:     {detection_count} total")
    
    return total_metadata_size, total_image_size, avg_metadata, avg_image


def main():
    print("\n" + "="*60)
    print("IMX500 Metadata vs Image Comparison")
    print("="*60)
    
    # Model path
    model_path = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
    
    print(f"\n[1] Loading model...")
    imx500 = IMX500(model_path)
    
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    intrinsics.update_with_defaults()
    
    print(f"✅ Model loaded: {intrinsics.task}")
    
    # Initialize camera
    print("\n[2] Initializing camera...")
    picam2 = Picamera2(imx500.camera_num)
    
    # Upload firmware
    print("\n[3] Uploading firmware (this takes ~30 seconds)...")
    imx500.show_network_fw_progress_bar()
    
    # Configure
    config = picam2.create_preview_configuration(
        controls={"FrameRate": 10},
        buffer_count=12
    )
    picam2.start(config, show_preview=False)
    
    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()
    
    print("✅ Camera started\n")
    time.sleep(1)
    
    # Run tests
    num_frames = 20
    
    # Test 1: Metadata only
    metadata_total, metadata_avg = test_metadata_only(picam2, imx500, num_frames)
    
    time.sleep(1)
    
    # Test 2: Metadata + Image
    meta_total, img_total, meta_avg, img_avg = test_metadata_and_image(picam2, imx500, num_frames)
    
    # Cleanup
    picam2.stop()
    picam2.close()
    
    # Final comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print()
    print("📊 ขนาดข้อมูลต่อ frame:")
    print(f"   Metadata only:  {format_bytes(metadata_avg):>10s}")
    print(f"   Image only:     {format_bytes(img_avg):>10s}")
    print()
    
    if img_avg > 0 and metadata_avg > 0:
        ratio = img_avg / metadata_avg
        percent = (metadata_avg / img_avg) * 100
        print(f"💡 ภาพใหญ่กว่า metadata ถึง {ratio:.1f}x")
        print(f"   Metadata เป็นเพียง {percent:.2f}% ของภาพ!")
        print()
    
    print("✅ สรุป:")
    print("   - IMX500 ประมวลผล AI ภายในตัวเซนเซอร์")
    print("   - ส่งออกมาเป็น metadata (ขนาดเล็ก ~11 KB)")
    print("   - ภาพเป็น optional (ขนาดใหญ่ ~1.2 MB)")
    print("   - ถ้าไม่ต้องการแสดงภาพ ใช้ capture_metadata() ก็พอ")
    print()
    print("🚀 ข้อดี:")
    print("   - ประหยัด bandwidth มหาศาล")
    print("   - ไม่ต้องใช้ CPU ของ Pi ทำ AI")
    print("   - ได้ผลลัพธ์เร็วและแม่นยำ")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
