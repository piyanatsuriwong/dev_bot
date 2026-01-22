#!/usr/bin/python
# PCA9685 16-Channel PWM Servo Driver for Raspberry Pi

import time
import math

try:
    import smbus
    SMBUS_AVAILABLE = True
except ImportError:
    SMBUS_AVAILABLE = False
    print("Warning: smbus not available. PCA9685 will not work.")


class PCA9685:
    """PCA9685 16-Channel PWM Servo Driver"""

    # Registers
    __SUBADR1 = 0x02
    __SUBADR2 = 0x03
    __SUBADR3 = 0x04
    __MODE1 = 0x00
    __PRESCALE = 0xFE
    __LED0_ON_L = 0x06
    __LED0_ON_H = 0x07
    __LED0_OFF_L = 0x08
    __LED0_OFF_H = 0x09
    __ALLLED_ON_L = 0xFA
    __ALLLED_ON_H = 0xFB
    __ALLLED_OFF_L = 0xFC
    __ALLLED_OFF_H = 0xFD

    def __init__(self, address=0x40, debug=False):
        self.address = address
        self.debug = debug
        self.bus = None

        if SMBUS_AVAILABLE:
            try:
                self.bus = smbus.SMBus(1)
                self.write(self.__MODE1, 0x00)
            except Exception as e:
                if debug:
                    print(f"PCA9685 init error: {e}")
                self.bus = None

    def write(self, reg, value):
        """Writes an 8-bit value to the specified register/address"""
        if self.bus:
            self.bus.write_byte_data(self.address, reg, value)

    def read(self, reg):
        """Read an unsigned byte from the I2C device"""
        if self.bus:
            result = self.bus.read_byte_data(self.address, reg)
            return result
        return 0

    def setPWMFreq(self, freq):
        """Sets the PWM frequency"""
        if not self.bus:
            return

        prescaleval = 25000000.0    # 25MHz
        prescaleval /= 4096.0       # 12-bit
        prescaleval /= float(freq)
        prescaleval -= 1.0
        prescale = math.floor(prescaleval + 0.5)

        oldmode = self.read(self.__MODE1)
        newmode = (oldmode & 0x7F) | 0x10        # sleep
        self.write(self.__MODE1, newmode)        # go to sleep
        self.write(self.__PRESCALE, int(math.floor(prescale)))
        self.write(self.__MODE1, oldmode)
        time.sleep(0.005)
        self.write(self.__MODE1, oldmode | 0x80)

    def setPWM(self, channel, on, off):
        """Sets a single PWM channel"""
        if not self.bus:
            return

        self.write(self.__LED0_ON_L + 4 * channel, on & 0xFF)
        self.write(self.__LED0_ON_H + 4 * channel, on >> 8)
        self.write(self.__LED0_OFF_L + 4 * channel, off & 0xFF)
        self.write(self.__LED0_OFF_H + 4 * channel, off >> 8)

    def setMotorPwm(self, channel, duty):
        """Set motor PWM duty cycle"""
        self.setPWM(channel, 0, duty)

    def setServoPulse(self, channel, pulse):
        """Sets the Servo Pulse. The PWM frequency must be 50HZ"""
        if not self.bus:
            return
        # PWM frequency is 50HZ, the period is 20000us
        pulse = pulse * 4096 / 20000
        self.setPWM(channel, 0, int(pulse))


if __name__ == '__main__':
    print("PCA9685 test")
    pwm = PCA9685(0x40, debug=True)
    if pwm.bus:
        pwm.setPWMFreq(50)
        print("PCA9685 initialized")
    else:
        print("PCA9685 not available")
