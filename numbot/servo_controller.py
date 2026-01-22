#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Servo Controller - PCA9685 Pan-Tilt Control
For NumBot Robot Eye Tracker
Based on Server-pi5/servo.py
"""

import time
import config

# Try to import PCA9685
try:
    from PCA9685 import PCA9685
    PCA9685_AVAILABLE = True
except ImportError:
    PCA9685_AVAILABLE = False
    print("Warning: PCA9685 not available")


class ServoController:
    """
    Pan-Tilt Servo Controller using PCA9685

    Channels (from Server-pi5):
    - Channel 8: Pan (horizontal)
    - Channel 9: Tilt (vertical)
    """

    def __init__(self):
        self.enabled = config.SERVO_ENABLED and PCA9685_AVAILABLE
        self.pca = None

        # Servo configuration from config
        self.pan_channel = config.SERVO_PAN_CHANNEL      # Channel 8
        self.tilt_channel = config.SERVO_TILT_CHANNEL    # Channel 9

        self.pan_min = config.SERVO_PAN_MIN
        self.pan_max = config.SERVO_PAN_MAX
        self.pan_center = config.SERVO_PAN_CENTER

        self.tilt_min = config.SERVO_TILT_MIN
        self.tilt_max = config.SERVO_TILT_MAX
        self.tilt_center = config.SERVO_TILT_CENTER

        self.smoothing = config.SERVO_SMOOTHING

        # Current positions
        self.current_pan = self.pan_center
        self.current_tilt = self.tilt_center
        self.target_pan = self.pan_center
        self.target_tilt = self.tilt_center

        if self.enabled:
            self._init_pca9685()

    def _init_pca9685(self):
        """Initialize PCA9685"""
        try:
            self.pca = PCA9685(address=config.SERVO_I2C_ADDRESS, debug=True)
            self.pca.setPWMFreq(config.SERVO_FREQUENCY)
            time.sleep(0.1)

            # Move to center position
            self.center()

            print(f"ServoController: PCA9685 @ 0x{config.SERVO_I2C_ADDRESS:02X}")
            print(f"  Pan (Ch{self.pan_channel}): {self.pan_min}-{self.pan_max} deg")
            print(f"  Tilt (Ch{self.tilt_channel}): {self.tilt_min}-{self.tilt_max} deg")

        except Exception as e:
            print(f"ServoController: Failed to initialize PCA9685: {e}")
            self.enabled = False
            self.pca = None

    def _angle_to_pulse(self, angle, error=10):
        """
        Convert angle (0-180) to pulse width
        Formula from Server-pi5/servo.py: 500 + (angle + error) / 0.09
        """
        pulse = 500 + int((angle + error) / 0.09)
        return pulse

    def set_pan(self, angle):
        """Set pan angle (0-180 degrees)"""
        if not self.enabled or self.pca is None:
            return

        angle = max(self.pan_min, min(self.pan_max, angle))
        pulse = self._angle_to_pulse(angle)
        self.pca.setServoPulse(self.pan_channel, pulse)
        self.current_pan = angle

    def set_tilt(self, angle):
        """Set tilt angle (0-180 degrees)"""
        if not self.enabled or self.pca is None:
            return

        angle = max(self.tilt_min, min(self.tilt_max, angle))
        pulse = self._angle_to_pulse(angle)
        self.pca.setServoPulse(self.tilt_channel, pulse)
        self.current_tilt = angle

    def set_position(self, pan, tilt):
        """Set both pan and tilt angles"""
        self.set_pan(pan)
        self.set_tilt(tilt)

    def center(self):
        """Move servos to center position"""
        self.set_position(self.pan_center, self.tilt_center)
        self.target_pan = self.pan_center
        self.target_tilt = self.tilt_center

    def track_normalized(self, x, y):
        """
        Track target using normalized coordinates (-1 to 1)
        x: -1 (left) to 1 (right)
        y: -1 (up) to 1 (down)
        
        Natural tracking (servo follows hand direction):
        - Hand left (x=-1) → servo pans left (pan_min)
        - Hand right (x=1) → servo pans right (pan_max)
        - Hand up (y=-1) → servo tilts up (tilt_min)
        - Hand down (y=1) → servo tilts down (tilt_max)
        """
        # Convert normalized coordinates to angles
        pan_range = self.pan_max - self.pan_min
        tilt_range = self.tilt_max - self.tilt_min

        # Map -1..1 to angle range with natural tracking
        # x: -1→pan_min, 0→pan_center, 1→pan_max
        # y: -1→tilt_min, 0→tilt_center, 1→tilt_max
        self.target_pan = self.pan_center + (x * pan_range / 2)
        self.target_tilt = self.tilt_center + (y * tilt_range / 2)

        # Clamp to limits
        self.target_pan = max(self.pan_min, min(self.pan_max, self.target_pan))
        self.target_tilt = max(self.tilt_min, min(self.tilt_max, self.target_tilt))

    def update(self):
        """
        Update servo positions with smoothing
        Call this in the main loop for smooth motion
        """
        if not self.enabled:
            return

        # Apply smoothing
        if self.smoothing > 0:
            self.current_pan += (self.target_pan - self.current_pan) * (1 - self.smoothing)
            self.current_tilt += (self.target_tilt - self.current_tilt) * (1 - self.smoothing)
        else:
            self.current_pan = self.target_pan
            self.current_tilt = self.target_tilt

        # Set servo positions
        self.set_pan(self.current_pan)
        self.set_tilt(self.current_tilt)

    def cleanup(self):
        """Cleanup servo controller"""
        if self.enabled and self.pca is not None:
            self.center()
            time.sleep(0.3)
            print("ServoController: Cleaned up")


# Test function
if __name__ == "__main__":
    print("Testing ServoController (Channels 8 & 9)...")

    servo = ServoController()

    if servo.enabled:
        print("\nMoving to center...")
        servo.center()
        time.sleep(1)

        print("Testing pan sweep...")
        for angle in [45, 90, 135, 90]:
            print(f"  Pan: {angle} degrees")
            servo.set_pan(angle)
            time.sleep(0.5)

        print("Testing tilt sweep...")
        for angle in [60, 90, 120, 90]:
            print(f"  Tilt: {angle} degrees")
            servo.set_tilt(angle)
            time.sleep(0.5)

        servo.cleanup()
        print("\nTest complete!")
    else:
        print("ServoController not available")
