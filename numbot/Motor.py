import time
import math
from PCA9685 import PCA9685
from ADC import *

class Motor:
    """
    Motor controller for 4-wheel Mecanum drive robot.

    Mecanum wheels allow omnidirectional movement:
    - Forward/Backward
    - Strafe Left/Right
    - Diagonal movement
    - Rotation in place
    - Combined translation + rotation

    Wheel layout (top view):
        FL \\  // FR
            []
        RL //  \\ RR

    Wheel mapping:
        - FL (Front Left)  = left_Upper_Wheel  (duty1)
        - RL (Rear Left)   = left_Lower_Wheel  (duty2)
        - FR (Front Right) = right_Upper_Wheel (duty3)
        - RR (Rear Right)  = right_Lower_Wheel (duty4)
    """

    # Speed constants
    MAX_SPEED = 4095
    DEFAULT_SPEED = 2000

    def __init__(self):
        self.pwm = PCA9685(0x40, debug=True)
        self.pwm.setPWMFreq(50)
        self.time_proportion = 3     #Depend on your own car,If you want to get the best out of the rotation mode, change the value by experimenting.
        self.adc = Adc()
        self.current_speed = self.DEFAULT_SPEED
    def duty_range(self,duty1,duty2,duty3,duty4):
        if duty1>4095:
            duty1=4095
        elif duty1<-4095:
            duty1=-4095        
        
        if duty2>4095:
            duty2=4095
        elif duty2<-4095:
            duty2=-4095
            
        if duty3>4095:
            duty3=4095
        elif duty3<-4095:
            duty3=-4095
            
        if duty4>4095:
            duty4=4095
        elif duty4<-4095:
            duty4=-4095
        return duty1,duty2,duty3,duty4
        
    def left_Upper_Wheel(self,duty):
        if duty>0:
            self.pwm.setMotorPwm(0,0)
            self.pwm.setMotorPwm(1,duty)
        elif duty<0:
            self.pwm.setMotorPwm(1,0)
            self.pwm.setMotorPwm(0,abs(duty))
        else:
            self.pwm.setMotorPwm(0,4095)
            self.pwm.setMotorPwm(1,4095)
    def left_Lower_Wheel(self,duty):
        if duty>0:
            self.pwm.setMotorPwm(3,0)
            self.pwm.setMotorPwm(2,duty)
        elif duty<0:
            self.pwm.setMotorPwm(2,0)
            self.pwm.setMotorPwm(3,abs(duty))
        else:
            self.pwm.setMotorPwm(2,4095)
            self.pwm.setMotorPwm(3,4095)
    def right_Upper_Wheel(self,duty):
        if duty>0:
            self.pwm.setMotorPwm(7,0)
            self.pwm.setMotorPwm(6,duty)
        elif duty<0:
            self.pwm.setMotorPwm(6,0)
            self.pwm.setMotorPwm(7,abs(duty))
        else:
            self.pwm.setMotorPwm(6,4095)
            self.pwm.setMotorPwm(7,4095)
    def right_Lower_Wheel(self,duty):
        if duty>0:
            self.pwm.setMotorPwm(4,0)
            self.pwm.setMotorPwm(5,duty)
        elif duty<0:
            self.pwm.setMotorPwm(5,0)
            self.pwm.setMotorPwm(4,abs(duty))
        else:
            self.pwm.setMotorPwm(4,4095)
            self.pwm.setMotorPwm(5,4095)
            
 
    def setMotorModel(self,duty1,duty2,duty3,duty4):
        duty1,duty2,duty3,duty4=self.duty_range(duty1,duty2,duty3,duty4)
        self.left_Upper_Wheel(duty1)
        self.left_Lower_Wheel(duty2)
        self.right_Upper_Wheel(duty3)
        self.right_Lower_Wheel(duty4)

    # ==================== Mecanum Wheel Methods ====================

    def stop(self):
        """Stop all motors."""
        self.setMotorModel(0, 0, 0, 0)

    def forward(self, speed=None):
        """Move forward. All wheels rotate forward."""
        speed = speed or self.current_speed
        self.setMotorModel(speed, speed, speed, speed)

    def backward(self, speed=None):
        """Move backward. All wheels rotate backward."""
        speed = speed or self.current_speed
        self.setMotorModel(-speed, -speed, -speed, -speed)

    def strafe_left(self, speed=None):
        """
        Strafe left (move sideways to the left).
        Standard Mecanum: FL-, FR+, RL+, RR-
        """
        speed = speed or self.current_speed
        self.setMotorModel(-speed, speed, speed, -speed)

    def strafe_right(self, speed=None):
        """
        Strafe right (move sideways to the right).
        Standard Mecanum: FL+, FR-, RL-, RR+
        """
        speed = speed or self.current_speed
        self.setMotorModel(speed, -speed, -speed, speed)

    def rotate_left(self, speed=None):
        """Rotate counter-clockwise in place. Left wheels back, right wheels forward."""
        speed = speed or self.current_speed
        self.setMotorModel(-speed, -speed, speed, speed)

    def rotate_right(self, speed=None):
        """Rotate clockwise in place. Left wheels forward, right wheels back."""
        speed = speed or self.current_speed
        self.setMotorModel(speed, speed, -speed, -speed)

    def diagonal_forward_left(self, speed=None):
        """Move diagonally forward-left. FL and RR stop, FR and RL move forward."""
        speed = speed or self.current_speed
        # Standard: FL=0, RL=+, FR=+, RR=0
        self.setMotorModel(0, speed, speed, 0)

    def diagonal_forward_right(self, speed=None):
        """Move diagonally forward-right. FR and RL stop, FL and RR move forward."""
        speed = speed or self.current_speed
        # Standard: FL=+, RL=0, FR=0, RR=+
        self.setMotorModel(speed, 0, 0, speed)

    def diagonal_backward_left(self, speed=None):
        """Move diagonally backward-left. FR and RL stop, FL and RR move backward."""
        speed = speed or self.current_speed
        # Standard: FL=-, RL=0, FR=0, RR=-
        self.setMotorModel(-speed, 0, 0, -speed)

    def diagonal_backward_right(self, speed=None):
        """Move diagonally backward-right. FL and RR stop, FR and RL move backward."""
        speed = speed or self.current_speed
        # Standard: FL=0, RL=-, FR=-, RR=0
        self.setMotorModel(0, -speed, -speed, 0)

    def move(self, vx, vy, omega=0):
        """
        Omnidirectional movement using velocity components.

        Args:
            vx: Sideways velocity (-1.0 to 1.0, positive = right)
            vy: Forward velocity (-1.0 to 1.0, positive = forward)
            omega: Rotation velocity (-1.0 to 1.0, positive = clockwise)

        Standard Mecanum kinematic equations (X-configuration):
            FL = vy + vx - omega
            FR = vy - vx + omega
            RL = vy - vx - omega
            RR = vy + vx + omega
        """
        # Calculate wheel speeds using Mecanum kinematics
        fl = vy + vx - omega
        fr = vy - vx + omega
        rl = vy - vx - omega
        rr = vy + vx + omega

        # Normalize if any value exceeds 1.0
        max_val = max(abs(fl), abs(rl), abs(fr), abs(rr))
        if max_val > 1.0:
            fl /= max_val
            rl /= max_val
            fr /= max_val
            rr /= max_val

        # Scale to motor PWM range
        scale = self.current_speed
        self.setMotorModel(
            int(fl * scale),
            int(rl * scale),
            int(fr * scale),
            int(rr * scale)
        )

    def move_angle(self, angle, speed=None, omega=0):
        """
        Move in a specific direction angle.

        Args:
            angle: Direction in degrees (0=forward, 90=right, 180=back, 270=left)
            speed: Movement speed (0.0 to 1.0)
            omega: Rotation velocity (-1.0 to 1.0, positive = clockwise)
        """
        speed = speed if speed is not None else 1.0
        rad = math.radians(angle)
        vx = speed * math.sin(rad)
        vy = speed * math.cos(rad)
        self.move(vx, vy, omega)

    def set_speed(self, speed):
        """Set the default speed for movement commands."""
        self.current_speed = min(max(speed, 0), self.MAX_SPEED)

    # ==================== End Mecanum Methods ====================

    def Rotate(self,n):
        angle = n
        bat_compensate =7.5/(self.adc.recvADC(2)*3)
        while True:
            W = 2000

            VY = int(2000 * math.cos(math.radians(angle)))
            VX = -int(2000 * math.sin(math.radians(angle)))

            FR = VY - VX + W
            FL = VY + VX - W
            BL = VY - VX - W
            BR = VY + VX + W

            PWM.setMotorModel(FL, BL, FR, BR)
            print("rotating")
            time.sleep(5*self.time_proportion*bat_compensate/1000)
            angle -= 5

PWM=Motor()
def loop():
    PWM.setMotorModel(2000,2000,2000,2000)       #Forward
    time.sleep(3)
    PWM.setMotorModel(-2000,-2000,-2000,-2000)   #Back
    time.sleep(3)
    PWM.setMotorModel(-500,-500,2000,2000)       #Left
    time.sleep(3)
    PWM.setMotorModel(2000,2000,-500,-500)       #Right
    time.sleep(3)
    PWM.setMotorModel(0,0,0,0)                   #Stop

def mecanum_demo():
    """Demonstrate Mecanum wheel omnidirectional movement."""
    print("=== Mecanum Wheel Demo (Slow) ===")

    # ความเร็วปานกลาง และระยะเวลาสั้น
    slow_speed = 1500  # ปานกลาง (max 4095)
    move_time = 1.0    # 1 วินาที

    print("Forward...")
    PWM.forward(slow_speed)
    time.sleep(move_time)
    PWM.stop()
    time.sleep(0.3)

    print("Backward...")
    PWM.backward(slow_speed)
    time.sleep(move_time)
    PWM.stop()
    time.sleep(0.3)

    print("Strafe Left...")
    PWM.strafe_left(slow_speed)
    time.sleep(move_time)
    PWM.stop()
    time.sleep(0.3)

    print("Strafe Right...")
    PWM.strafe_right(slow_speed)
    time.sleep(move_time)
    PWM.stop()
    time.sleep(0.3)

    print("Diagonal Forward-Left...")
    PWM.diagonal_forward_left(slow_speed)
    time.sleep(move_time)
    PWM.stop()
    time.sleep(0.3)

    print("Diagonal Forward-Right...")
    PWM.diagonal_forward_right(slow_speed)
    time.sleep(move_time)
    PWM.stop()
    time.sleep(0.3)

    print("Diagonal Backward-Left...")
    PWM.diagonal_backward_left(slow_speed)
    time.sleep(move_time)
    PWM.stop()
    time.sleep(0.3)

    print("Diagonal Backward-Right...")
    PWM.diagonal_backward_right(slow_speed)
    time.sleep(move_time)
    PWM.stop()
    time.sleep(0.3)

    print("Rotate Left (CCW)...")
    PWM.rotate_left(slow_speed)
    time.sleep(move_time)
    PWM.stop()
    time.sleep(0.3)

    print("Rotate Right (CW)...")
    PWM.rotate_right(slow_speed)
    time.sleep(move_time)
    PWM.stop()
    time.sleep(0.3)

    print("Circle movement using move_angle()...")
    PWM.set_speed(slow_speed)
    for angle in range(0, 360, 45):
        print(f"  Angle: {angle}°")
        PWM.move_angle(angle, speed=0.5)
        time.sleep(0.3)
        PWM.stop()
        time.sleep(0.2)

    print("Done!")
    PWM.stop()
    
def destroy():
    PWM.setMotorModel(0,0,0,0)

if __name__=='__main__':
    import sys
    try:
        if len(sys.argv) > 1 and sys.argv[1] == '--mecanum':
            mecanum_demo()
        else:
            print("Usage: python3 Motor.py [--mecanum]")
            print("  --mecanum  Run Mecanum wheel demo")
            print("  (default)  Run original 4-wheel test")
            loop()
    except KeyboardInterrupt:  # When 'Ctrl+C' is pressed, the child program destroy() will be executed.
        destroy()
