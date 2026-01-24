#!/usr/bin/env python3
"""
imx500_simple_test.py - ทดสอบ IMX500 AI Camera แบบง่าย

ทดสอบ:
1. ตรวจจับ IMX500
2. โหลด model
3. อ่าน metadata และ inference output

Usage:
    python3 imx500_simple_test.py

Note: 
    This is a simplified test without GUI preview.
    Good for headless testing via SSH.
"""

import time
from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500

def main():
    print("=" * 60)
    print("IMX500 Simple Test (Headless)")
    print("=" * 60)
    
    # Check available cameras first
    cameras = Picamera2.global_camera_info()
    print(f"\n[Available Cameras: {len(cameras)}]")
    
    imx500_found = False
    imx500_idx = None
    
    for idx, cam in enumerate(cameras):
        model = cam.get('Model', 'Unknown')
        print(f"   Camera {idx}: {model}")
        if 'imx500' in model.lower():
            imx500_found = True
            imx500_idx = idx
    
    if not imx500_found:
        print("\n⚠️  IMX500 not detected!")
        print("   Make sure IMX500 AI Camera is connected properly.")
        return
    
    print(f"\n✓ IMX500 found at index {imx500_idx}")
    
    # Model to load
    model_path = "/usr/share/imx500-models/imx500_network_mobilenet_v2.rpk"
    
    print(f"\n[1] Loading model: {model_path}")
    try:
        imx500 = IMX500(model_path)
        print("    ✓ Model loaded successfully")
    except Exception as e:
        print(f"    ✗ Failed to load model: {e}")
        return
    
    print("\n[2] Initializing Picamera2...")
    picam2 = Picamera2(imx500.camera_num)
    
    print("\n[3] Creating preview configuration...")
    config = picam2.create_preview_configuration()
    picam2.configure(config)
    
    print("\n[4] Starting camera (no preview)...")
    picam2.start(show_preview=False)
    
    print("\n[5] Waiting for camera to stabilize...")
    time.sleep(2)
    
    print("\n[6] Capturing metadata samples...")
    print("-" * 60)
    
    for i in range(5):
        metadata = picam2.capture_metadata()
        
        # Get inference outputs
        np_outputs = imx500.get_outputs(metadata)
        
        print(f"\nSample {i+1}:")
        print(f"   Timestamp: {metadata.get('SensorTimestamp', 'N/A')}")
        print(f"   Exposure: {metadata.get('ExposureTime', 'N/A')}μs")
        print(f"   Gain: {metadata.get('AnalogueGain', 'N/A')}")
        
        if np_outputs is not None:
            print(f"   Inference Output Shapes: {[o.shape for o in np_outputs]}")
            # For MobileNet V2, show top prediction
            if len(np_outputs) > 0 and len(np_outputs[0]) > 0:
                top_idx = np_outputs[0].argmax()
                top_score = np_outputs[0][top_idx]
                print(f"   Top Prediction: Class {top_idx} (score: {top_score:.4f})")
        else:
            print("   Inference Output: None (model may still be loading)")
        
        time.sleep(1)
    
    print("\n[7] Getting IMX500 device info...")
    try:
        device_id = imx500.get_device_id()
        print(f"   Device ID: {device_id}")
    except Exception as e:
        print(f"   Could not get device ID: {e}")
    
    print("\n[8] Getting input size...")
    try:
        input_size = imx500.get_input_size()
        print(f"   Model Input Size: {input_size}")
    except Exception as e:
        print(f"   Could not get input size: {e}")
    
    print("\n[9] Stopping camera...")
    picam2.stop()
    picam2.close()
    
    print("\n" + "=" * 60)
    print("✓ IMX500 simple test complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
