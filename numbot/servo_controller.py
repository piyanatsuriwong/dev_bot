#!/usr/bin/env python3
"""
Servo Controller using PCA9685 for Pan-Tilt tracking
- Servo 0 (Channel 0): Pan (left-right, X axis)
- Servo 1 (Channel 1): Tilt (up-down, Y axis)
"""

import time

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

    Servo mapping:
    - Channel 0 -> PCA9685 channel 8 (Pan: left-right)
    - Channel 1 -> PCA9685 channel 9 (Tilt: up-down)
    """

    # Servo angle limits
    PAN_MIN = 0
    PAN_MAX = 180
    PAN_CENTER = 90

    TILT_MIN = 0       # Maximum downward tilt (can look straight down)
    TILT_MAX = 180     # Maximum upward tilt (full range)
    TILT_CENTER = 90
    
    # Deadzone (ignore small movements to reduce jitter)
    DEADZONE_DEGREES = 1.5  # Don't move if change is less than this

    def __init__(self, i2c_address=0x40):
        self.enabled = False
        self.pwm = None

        # Current positions
        self.pan_angle = self.PAN_CENTER
        self.tilt_angle = self.TILT_CENTER

        # Smoothing targets
        self.pan_target = self.PAN_CENTER
        self.tilt_target = self.TILT_CENTER
        
        # First detection flag (for smooth initial movement)
        self.first_detection = True
        self.detection_count = 0

        if PCA9685_AVAILABLE:
            try:
                self.pwm = PCA9685(i2c_address, debug=False)
                self.pwm.setPWMFreq(50)  # 50Hz for servos

                # Center both servos on startup
                self._set_servo_angle(0, self.PAN_CENTER)
                self._set_servo_angle(1, self.TILT_CENTER)

                self.enabled = True
                print(f"ServoController: PCA9685 @ 0x{i2c_address:02X}")
                print(f"  Pan (Ch0): {self.PAN_MIN}-{self.PAN_MAX} deg")
                print(f"  Tilt (Ch1): {self.TILT_MIN}-{self.TILT_MAX} deg")
            except Exception as e:
                print(f"ServoController: Failed - {e}")
                self.enabled = False
        else:
            print("ServoController: PCA9685 not available")

    def _set_servo_angle(self, channel, angle, error=10):
        """
        Set servo angle (0-180 degrees)

        Args:
            channel: 0 for Pan, 1 for Tilt
            angle: Angle in degrees (0-180)
            error: Calibration offset
        """
        if not self.enabled or self.pwm is None:
            return

        angle = int(angle)

        # Map channel to PCA9685 channel (0->8, 1->9)
        pca_channel = 8 + channel

        # Convert angle to pulse width
        # 500us = 0 deg, 2500us = 180 deg
        pulse = 500 + int((angle + error) / 0.09)

        self.pwm.setServoPulse(pca_channel, pulse)

    def set_pan(self, angle):
        """Set pan angle (left-right)"""
        angle = max(self.PAN_MIN, min(self.PAN_MAX, angle))
        self.pan_angle = angle
        self._set_servo_angle(0, angle)

    def set_tilt(self, angle):
        """Set tilt angle (up-down)"""
        angle = max(self.TILT_MIN, min(self.TILT_MAX, angle))
        self.tilt_angle = angle
        self._set_servo_angle(1, angle)

    def set_position(self, pan, tilt):
        """Set both pan and tilt angles"""
        self.set_pan(pan)
        self.set_tilt(tilt)

    def set_angle(self, angle):
        """Compatibility: Set pan angle only"""
        self.set_pan(angle)

    def center(self):
        """Center both servos"""
        self.set_position(self.PAN_CENTER, self.TILT_CENTER)
    
    def reset_smoothing(self):
        """Reset smoothing for new detection"""
        self.first_detection = True
        self.detection_count = 0

    def track_hand(self, hand_x, hand_y=0, smoothing=0.15, debug=False):
        """
        Track hand position with servos

        Args:
            hand_x: Normalized X position (-1 to 1, left to right)
            hand_y: Normalized Y position (-1 to 1, top to bottom)
            smoothing: Smoothing factor (0-1, lower = smoother)
            debug: Print debug info
        """
        if not self.enabled:
            return

        # Convert normalized position to servo angles
        # hand_x: -1 (left) to 1 (right)
        # Pan: direct mapping (no inversion here)
        target_pan = self.PAN_CENTER + (hand_x * (self.PAN_MAX - self.PAN_MIN) / 2)

        # hand_y: -1 (top) to 1 (bottom)
        # Tilt: follow hand direction
        target_tilt = self.TILT_CENTER + (hand_y * (self.TILT_MAX - self.TILT_MIN) / 2)

        # Apply deadzone - ignore small movements to reduce jitter
        # Use smaller deadzone for tilt to make it more responsive
        pan_diff = abs(target_pan - self.pan_target)
        tilt_diff = abs(target_tilt - self.tilt_target)
        
        if pan_diff > self.DEADZONE_DEGREES:
            self.pan_target = target_pan
        if tilt_diff > (self.DEADZONE_DEGREES * 0.5):  # Tilt more sensitive (half deadzone)
            self.tilt_target = target_tilt
        
        # Extra smooth for first detection (first 20 frames for smoother start)
        effective_smoothing_pan = smoothing
        effective_smoothing_tilt = smoothing * 1.3  # Tilt faster (1.5x smoothing factor)
        
        if self.first_detection or self.detection_count < 20:
            effective_smoothing_pan = max(0.03, smoothing * 0.2)
            effective_smoothing_tilt = max(0.05, smoothing * 0.3)  # Tilt starts faster
            self.detection_count += 1
            if self.detection_count >= 20:
                self.first_detection = False

        # Smooth movement with exponential moving average
        # Tilt uses higher smoothing factor = more responsive
        new_pan = self.pan_angle + (self.pan_target - self.pan_angle) * effective_smoothing_pan
        new_tilt = self.tilt_angle + (self.tilt_target - self.tilt_angle) * effective_smoothing_tilt
        
        # Only update servo if change is significant (reduce micro-jitter)
        # Tilt uses smaller threshold for more responsiveness
        if abs(new_pan - self.pan_angle) > 0.3:
            self.pan_angle = new_pan
            self.set_pan(self.pan_angle)
        if abs(new_tilt - self.tilt_angle) > 0.15:  # Tilt: lower threshold = more responsive
            self.tilt_angle = new_tilt
            self.set_tilt(self.tilt_angle)

        if debug:
            print(f"Servo: Pan={self.pan_angle:.1f} Tilt={self.tilt_angle:.1f}")

    def cleanup(self):
        """Return servos to center position"""
        if self.enabled:
            self.center()
            print("ServoController: Cleaned up")


# Test code
if __name__ == '__main__':
    print("Testing ServoController (PCA9685)...")

    servo = ServoController()

    if servo.enabled:
        print("\nCentering servos...")
        servo.center()
        time.sleep(1)

        print("\nTesting Pan (left-right)...")
        for angle in [45, 90, 135, 90]:
            print(f"  Pan: {angle}")
            servo.set_pan(angle)
            time.sleep(0.5)

        print("\nTesting Tilt (up-down)...")
        for angle in [60, 90, 120, 90]:
            print(f"  Tilt: {angle}")
            servo.set_tilt(angle)
            time.sleep(0.5)

        print("\nTesting tracking simulation...")
        import math
        for i in range(100):
            # Simulate circular motion
            x = math.sin(i * 0.1)
            y = math.cos(i * 0.1) * 0.5
            servo.track_hand(x, y, smoothing=0.2)
            time.sleep(0.05)

        servo.cleanup()
        print("\nTest complete!")
    else:
        print("Servo not available for testing")
