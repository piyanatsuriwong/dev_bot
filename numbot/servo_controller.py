#!/usr/bin/env python3
"""
Servo Controller using GPIO PWM for Pan-Tilt tracking
- GPIO 18 (Pin 12): Pan (left-right, X axis) - Hardware PWM
- GPIO 13 (Pin 33): Tilt (up-down, Y axis) - Hardware PWM (optional)
"""

import time

# Try lgpio first (Raspberry Pi 5)
try:
    import lgpio
    LGPIO_AVAILABLE = True
    print("ServoController: Using lgpio (Raspberry Pi 5)")
except ImportError:
    LGPIO_AVAILABLE = False
    print("Warning: lgpio not available")


class ServoController:
    """
    Pan-Tilt Servo Controller using GPIO Hardware PWM

    GPIO mapping:
    - GPIO 18 (Pin 12): Pan servo (left-right, X axis)
    - GPIO 13 (Pin 33): Tilt servo (up-down, Y axis) - Optional

    PWM Specs:
    - Frequency: 50Hz (20ms period)
    - Pulse width: 500-2500us (0.5-2.5ms)
    - 500us = 0 deg, 1500us = 90 deg, 2500us = 180 deg
    """

    # GPIO Pins
    PAN_PIN = 18   # Hardware PWM0
    TILT_PIN = 13  # Hardware PWM1 (optional)

    # PWM Settings
    PWM_FREQ = 50  # 50Hz for servos

    # Servo angle limits
    PAN_MIN = 0
    PAN_MAX = 180
    PAN_CENTER = 90

    TILT_MIN = 30      # Limit to prevent looking too far down
    TILT_MAX = 150     # Limit to prevent looking too far up
    TILT_CENTER = 90

    def __init__(self, pan_only=False):
        self.enabled = False
        self.chip = None
        self.pan_only = pan_only

        # Current positions
        self.pan_angle = self.PAN_CENTER
        self.tilt_angle = self.TILT_CENTER

        # Smoothing targets
        self.pan_target = self.PAN_CENTER
        self.tilt_target = self.TILT_CENTER

        if LGPIO_AVAILABLE:
            try:
                # Open GPIO chip
                self.chip = lgpio.gpiochip_open(4)  # /dev/gpiochip4 for Pi 5

                # Setup Pan servo (GPIO 18)
                lgpio.gpio_claim_output(self.chip, self.PAN_PIN)
                lgpio.tx_pwm(self.chip, self.PAN_PIN, self.PWM_FREQ, 50)  # Start with 50% duty

                # Setup Tilt servo (GPIO 13) if not pan_only
                if not pan_only:
                    lgpio.gpio_claim_output(self.chip, self.TILT_PIN)
                    lgpio.tx_pwm(self.chip, self.TILT_PIN, self.PWM_FREQ, 50)

                # Center servos on startup
                self._set_servo_angle(self.PAN_PIN, self.PAN_CENTER)
                if not pan_only:
                    self._set_servo_angle(self.TILT_PIN, self.TILT_CENTER)

                self.enabled = True
                print(f"ServoController: GPIO PWM initialized")
                print(f"  Pan (GPIO {self.PAN_PIN}): {self.PAN_MIN}-{self.PAN_MAX} deg")
                if not pan_only:
                    print(f"  Tilt (GPIO {self.TILT_PIN}): {self.TILT_MIN}-{self.TILT_MAX} deg")
                else:
                    print(f"  Tilt: Disabled (pan_only mode)")
            except Exception as e:
                print(f"ServoController: Failed - {e}")
                self.enabled = False
                if self.chip is not None:
                    try:
                        lgpio.gpiochip_close(self.chip)
                    except:
                        pass
                    self.chip = None
        else:
            print("ServoController: lgpio not available (Raspberry Pi 5 only)")

    def _set_servo_angle(self, gpio_pin, angle):
        """
        Set servo angle (0-180 degrees) using PWM duty cycle

        Args:
            gpio_pin: GPIO pin number (18 for Pan, 13 for Tilt)
            angle: Angle in degrees (0-180)
        """
        if not self.enabled or self.chip is None:
            return

        angle = int(max(0, min(180, angle)))

        # Convert angle to pulse width (us)
        # 0 deg = 500us, 90 deg = 1500us, 180 deg = 2500us
        pulse_us = 500 + (angle * 2000 / 180)

        # Convert pulse width to duty cycle percentage
        # Period = 20ms = 20000us at 50Hz
        duty_cycle = (pulse_us / 20000.0) * 100.0

        # Set PWM
        lgpio.tx_pwm(self.chip, gpio_pin, self.PWM_FREQ, duty_cycle)

    def set_pan(self, angle):
        """Set pan angle (left-right)"""
        angle = max(self.PAN_MIN, min(self.PAN_MAX, angle))
        self.pan_angle = angle
        self._set_servo_angle(self.PAN_PIN, angle)

    def set_tilt(self, angle):
        """Set tilt angle (up-down)"""
        if self.pan_only:
            return  # Tilt disabled in pan_only mode
        angle = max(self.TILT_MIN, min(self.TILT_MAX, angle))
        self.tilt_angle = angle
        self._set_servo_angle(self.TILT_PIN, angle)

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
        target_pan = self.PAN_CENTER + (hand_x * (self.PAN_MAX - self.PAN_MIN) / 2)
        target_tilt = self.TILT_CENTER + (hand_y * (self.TILT_MAX - self.TILT_MIN) / 2)

        # Apply smoothing
        self.pan_target = target_pan
        self.tilt_target = target_tilt

        # Smooth movement
        self.pan_angle += (self.pan_target - self.pan_angle) * smoothing
        self.tilt_angle += (self.tilt_target - self.tilt_angle) * smoothing

        # Clamp and set
        self.set_pan(self.pan_angle)
        self.set_tilt(self.tilt_angle)

        if debug:
            print(f"Servo: Pan={self.pan_angle:.1f} Tilt={self.tilt_angle:.1f}")

    def cleanup(self):
        """Return servos to center position and cleanup GPIO"""
        if self.enabled:
            try:
                # Center servos first
                self.center()
                time.sleep(0.2)

                # Stop PWM
                if self.chip is not None:
                    lgpio.tx_pwm(self.chip, self.PAN_PIN, 0, 0)  # Stop PWM
                    if not self.pan_only:
                        lgpio.tx_pwm(self.chip, self.TILT_PIN, 0, 0)

                    # Close chip
                    lgpio.gpiochip_close(self.chip)
                    self.chip = None

                print("ServoController: Cleaned up")
            except Exception as e:
                print(f"ServoController cleanup error: {e}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Servo Controller")
    parser.add_argument('--pan-only', action='store_true', help='Test pan servo only')
    args = parser.parse_args()

    print("Testing ServoController (GPIO PWM)...")
    print(f"Mode: {'Pan only' if args.pan_only else 'Pan + Tilt'}")

    servo = ServoController(pan_only=args.pan_only)

    if servo.enabled:
        print("\nCentering servos...")
        servo.center()
        time.sleep(1)

        print("\nTesting Pan (left-right)...")
        for angle in [0, 45, 90, 135, 180, 90]:
            print(f"  Pan: {angle} deg")
            servo.set_pan(angle)
            time.sleep(0.8)

        if not args.pan_only:
            print("\nTesting Tilt (up-down)...")
            for angle in [30, 60, 90, 120, 150, 90]:
                print(f"  Tilt: {angle} deg")
                servo.set_tilt(angle)
                time.sleep(0.8)

            print("\nTesting tracking simulation...")
            import math
            for i in range(100):
                x = math.sin(i * 0.1)
                y = math.cos(i * 0.1) * 0.5
                servo.track_hand(x, y, smoothing=0.2)
                time.sleep(0.05)
        else:
            print("\nTesting pan tracking only...")
            import math
            for i in range(100):
                x = math.sin(i * 0.1)
                servo.track_hand(x, 0, smoothing=0.2)
                time.sleep(0.05)

        servo.cleanup()
        print("\nTest complete!")
    else:
        print("Servo not available for testing")
