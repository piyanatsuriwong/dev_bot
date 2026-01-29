#!/usr/bin/env python3
"""Test Picamera2 import"""
import sys
try:
    from picamera2 import Picamera2
    print("✅ Picamera2 import OK")
    sys.exit(0)
except Exception as e:
    print(f"❌ Picamera2 import failed: {e}")
    sys.exit(1)
