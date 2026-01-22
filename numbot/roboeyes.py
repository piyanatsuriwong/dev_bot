# RoboEyes library
# Ported from Arduino FluxGarage RoboEyes for OLED Displays
# https://github.com/FluxGarage/RoboEyes
#
# Copyright (C) 2024 Dennis Hoelscher - www.fluxgarage.com (Arduino Version)
# Copyright (C) 2025 Meurisse Dominique - shop.mchobby.be (MicroPython Version)
#
# GNU General Public License <https://www.gnu.org/licenses/>.

from fbutil import FBUtil
from random import randint
import time

# CPython compatibility
def const(x):
    return x

# CPython compatibility for MicroPython time functions
if not hasattr(time, 'ticks_ms'):
    def ticks_ms():
        return int(time.time() * 1000)
    time.ticks_ms = ticks_ms

if not hasattr(time, 'ticks_diff'):
    def ticks_diff(a, b):
        return a - b
    time.ticks_diff = ticks_diff

if not hasattr(time, 'ticks_add'):
    def ticks_add(a, b):
        return a + b
    time.ticks_add = ticks_add

# Colors
BGCOLOR = 0
FGCOLOR = 1

# Mood types
DEFAULT = 0
TIRED = 1
ANGRY = 2
HAPPY = 3
FROZEN = 4
SCARY = 5
CURIOUS = 6

# On/Off
ON = 1
OFF = 0

# Predefined positions
N = 1   # north
NE = 2  # north-east
E = 3   # east
SE = 4  # south-east
S = 5   # south
SW = 6  # south-west
W = 7   # west
NW = 8  # north-west


class StepData:
    __slots__ = ["done", "ms_timing", "_lambda", "owner_seq"]

    def __init__(self, owner_seq, ms_timing, _lambda):
        self.done = False
        self.ms_timing = ms_timing
        self._lambda = _lambda
        self.owner_seq = owner_seq

    def update(self, ticks_ms):
        if self.done:
            return
        if time.ticks_diff(ticks_ms, self.owner_seq._start) < self.ms_timing:
            return
        self._lambda(self.owner_seq.owner)
        self.done = True


class Sequence(list):
    """A Sequence is a collection of Steps"""

    def __init__(self, owner, name):
        super().__init__()
        self.owner = owner
        self.name = name
        self._start = None

    def step(self, ms_timing, _lambda):
        _r = StepData(self, ms_timing, _lambda)
        self.append(_r)

    def start(self):
        self._start = time.ticks_ms()

    def reset(self):
        self._start = None
        for _step in self:
            _step.done = False

    @property
    def done(self):
        if self._start is None:
            return True
        return all([_step.done for _step in self])

    def update(self, ticks_ms):
        if self._start is None:
            return
        [_step.update(ticks_ms) for _step in self if not _step.done]


class Sequences(list):
    """Collection of Sequence"""

    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def add(self, name):
        _r = Sequence(self.owner, name)
        self.append(_r)
        return _r

    @property
    def done(self):
        return all([_seq.done for _seq in self])

    def update(self):
        _ms_ticks = time.ticks_ms()
        [_seq.update(_ms_ticks) for _seq in self]


class RoboEyes:
    def __init__(self, fb, width, height, frame_rate=20, on_show=None, bgcolor=BGCOLOR, fgcolor=FGCOLOR):
        assert on_show is not None, "on_show event not defined"
        self.fb = fb
        self.gfx = FBUtil(fb)
        self.on_show = on_show
        self.screenWidth = width
        self.screenHeight = height
        self.bgcolor = bgcolor
        self.fgcolor = fgcolor

        self.sequences = Sequences(self)
        self.fpsTimer = 0
        self._position = 0

        # Mood control
        self._mood = DEFAULT
        self.tired = False
        self.angry = False
        self.happy = False

        self._curious = False
        self._cyclops = False
        self.eyeL_open = False
        self.eyeR_open = False

        # Eye geometry
        self.spaceBetweenDefault = 10

        # Left eye
        self.eyeLwidthDefault = 36
        self.eyeLheightDefault = 36
        self.eyeLwidthCurrent = self.eyeLwidthDefault
        self.eyeLheightCurrent = 1
        self.eyeLwidthNext = self.eyeLwidthDefault
        self.eyeLheightNext = self.eyeLheightDefault
        self.eyeLheightOffset = 0
        self.eyeLborderRadiusDefault = 8
        self.eyeLborderRadiusCurrent = self.eyeLborderRadiusDefault
        self.eyeLborderRadiusNext = self.eyeLborderRadiusDefault

        # Right eye
        self.eyeRwidthDefault = self.eyeLwidthDefault
        self.eyeRheightDefault = self.eyeLheightDefault
        self.eyeRwidthCurrent = self.eyeRwidthDefault
        self.eyeRheightCurrent = 1
        self.eyeRwidthNext = self.eyeRwidthDefault
        self.eyeRheightNext = self.eyeRheightDefault
        self.eyeRheightOffset = 0
        self.eyeRborderRadiusDefault = 8
        self.eyeRborderRadiusCurrent = self.eyeRborderRadiusDefault
        self.eyeRborderRadiusNext = self.eyeRborderRadiusDefault

        # Eye coordinates
        self.eyeLxDefault = int(((self.screenWidth) - (self.eyeLwidthDefault + self.spaceBetweenDefault + self.eyeRwidthDefault)) / 2)
        self.eyeLyDefault = int((self.screenHeight - self.eyeLheightDefault) / 2)
        self.eyeLx = self.eyeLxDefault
        self.eyeLy = self.eyeLyDefault
        self.eyeLxNext = self.eyeLx
        self.eyeLyNext = self.eyeLy

        self.eyeRxDefault = self.eyeLx + self.eyeLwidthCurrent + self.spaceBetweenDefault
        self.eyeRyDefault = self.eyeLy
        self.eyeRx = self.eyeRxDefault
        self.eyeRy = self.eyeRyDefault
        self.eyeRxNext = self.eyeRx
        self.eyeRyNext = self.eyeRy

        # Eyelids
        self.eyelidsHeightMax = int(self.eyeLheightDefault / 2)
        self.eyelidsTiredHeight = 0
        self.eyelidsTiredHeightNext = self.eyelidsTiredHeight
        self.eyelidsAngryHeight = 0
        self.eyelidsAngryHeightNext = self.eyelidsAngryHeight
        self.eyelidsHappyBottomOffsetMax = int(self.eyeLheightDefault / 2) + 3
        self.eyelidsHappyBottomOffset = 0
        self.eyelidsHappyBottomOffsetNext = 0
        self.spaceBetweenCurrent = self.spaceBetweenDefault
        self.spaceBetweenNext = 10

        # Animations
        self.hFlicker = False
        self.hFlickerAlternate = False
        self.hFlickerAmplitude = 2

        self.vFlicker = False
        self.vFlickerAlternate = False
        self.vFlickerAmplitude = 10

        self.autoblinker = False
        self.blinkInterval = 1
        self.blinkIntervalVariation = 4
        self.blinktimer = 0

        self.idle = False
        self.idleInterval = 1
        self.idleIntervalVariation = 3
        self.idleAnimationTimer = 0

        self._confused = False
        self.confusedAnimationTimer = 0
        self.confusedAnimationDuration = 500
        self.confusedToggle = True

        self._laugh = False
        self.laughAnimationTimer = 0
        self.laughAnimationDuration = 500
        self.laughToggle = True

        self.clear_display()
        self.on_show(self)
        self.eyeLheightCurrent = 1
        self.eyeRheightCurrent = 1
        self.set_framerate(frame_rate)

    def update(self):
        self.sequences.update()
        if time.ticks_diff(time.ticks_ms(), self.fpsTimer) >= self.frameInterval:
            self.draw_eyes()
            self.fpsTimer = time.ticks_ms()

    def clear_display(self):
        self.fb.fill(self.bgcolor)

    def set_framerate(self, fps):
        self.frameInterval = 1000 // fps

    def eyes_width(self, leftEye=None, rightEye=None):
        if leftEye is not None:
            self.eyeLwidthNext = leftEye
            self.eyeLwidthDefault = leftEye
        if rightEye is not None:
            self.eyeRwidthNext = rightEye
            self.eyeRwidthDefault = rightEye

    def eyes_height(self, leftEye=None, rightEye=None):
        if leftEye is not None:
            self.eyeLheightNext = leftEye
            self.eyeLheightDefault = leftEye
        if rightEye is not None:
            self.eyeRheightNext = rightEye
            self.eyeRheightDefault = rightEye

    def eyes_radius(self, leftEye=None, rightEye=None):
        if leftEye is not None:
            self.eyeLborderRadiusNext = leftEye
            self.eyeLborderRadiusDefault = leftEye
        if rightEye is not None:
            self.eyeRborderRadiusNext = rightEye
            self.eyeRborderRadiusDefault = rightEye

    def eyes_spacing(self, space):
        self.spaceBetweenNext = space
        self.spaceBetweenDefault = space

    @property
    def mood(self):
        return self._mood

    @mood.setter
    def mood(self, mood):
        if (self._mood in (SCARY, FROZEN)) and not (mood in (SCARY, FROZEN)):
            self.horiz_flicker(False)
            self.vert_flicker(False)

        if self._curious and (mood != CURIOUS):
            self._curious = False

        if mood == TIRED:
            self.tired = True
            self.angry = False
            self.happy = False
        elif mood == ANGRY:
            self.tired = False
            self.angry = True
            self.happy = False
        elif mood == HAPPY:
            self.tired = False
            self.angry = False
            self.happy = True
        elif mood == FROZEN:
            self.tired = False
            self.angry = False
            self.happy = False
            self.horiz_flicker(True, 2)
            self.vert_flicker(False)
        elif mood == SCARY:
            self.tired = True
            self.angry = False
            self.happy = False
            self.horiz_flicker(False)
            self.vert_flicker(True, 2)
        elif mood == CURIOUS:
            self.tired = False
            self.angry = False
            self.happy = False
            self._curious = True
        else:
            self.tired = False
            self.angry = False
            self.happy = False
        self._mood = mood

    def set_mood(self, value):
        self.mood = value

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, direction):
        if direction == N:
            self.eyeLxNext = self.get_screen_constraint_X() // 2
            self.eyeLyNext = 0
        elif direction == NE:
            self.eyeLxNext = self.get_screen_constraint_X()
            self.eyeLyNext = 0
        elif direction == E:
            self.eyeLxNext = self.get_screen_constraint_X()
            self.eyeLyNext = self.get_screen_constraint_Y() // 2
        elif direction == SE:
            self.eyeLxNext = self.get_screen_constraint_X()
            self.eyeLyNext = self.get_screen_constraint_Y()
        elif direction == S:
            self.eyeLxNext = self.get_screen_constraint_X() // 2
            self.eyeLyNext = self.get_screen_constraint_X()
        elif direction == SW:
            self.eyeLxNext = 0
            self.eyeLyNext = self.get_screen_constraint_Y()
        elif direction == W:
            self.eyeLxNext = 0
            self.eyeLyNext = self.get_screen_constraint_Y() // 2
        elif direction == NW:
            self.eyeLxNext = 0
            self.eyeLyNext = 0
        else:
            self.eyeLxNext = self.get_screen_constraint_X() // 2
            self.eyeLyNext = self.get_screen_constraint_Y() // 2
        self._position = direction

    def set_position(self, value):
        self.position = value

    def set_auto_blinker(self, active, interval=None, variation=None):
        self.autoblinker = active
        if interval is not None:
            self.blinkInterval = interval
        if variation is not None:
            self.blinkIntervalVariation = variation

    def set_idle_mode(self, active, interval=None, variation=None):
        self.idle = active
        if interval is not None:
            self.idleInterval = interval
        if variation is not None:
            self.idleIntervalVariation = variation

    @property
    def curious(self):
        return self._curious

    @curious.setter
    def curious(self, enable):
        self._curious = enable

    def set_curious(self, value):
        self.curious = value

    @property
    def cyclops(self):
        return self._cyclops

    @cyclops.setter
    def cyclops(self, enabled):
        self._cyclops = enabled

    def set_cyclops(self, value):
        self.cyclops = value

    def horiz_flicker(self, enable, amplitude=None):
        self.hFlicker = enable
        if amplitude is not None:
            self.hFlickerAmplitude = amplitude

    def vert_flicker(self, enable, amplitude=None):
        self.vFlicker = enable
        if amplitude is not None:
            self.vFlickerAmplitude = amplitude

    def get_screen_constraint_X(self):
        return self.screenWidth - self.eyeLwidthCurrent - self.spaceBetweenCurrent - self.eyeRwidthCurrent

    def get_screen_constraint_Y(self):
        return self.screenHeight - self.eyeLheightDefault

    def close(self, left=None, right=None):
        if (left is None) and (right is None):
            self.eyeLheightNext = 1
            self.eyeRheightNext = 1
            self.eyeL_open = False
            self.eyeR_open = False
        else:
            if left is not None:
                self.eyeLheightNext = 1
                self.eyeL_open = False
            if right is not None:
                self.eyeRheightNext = 1
                self.eyeR_open = False

    def open(self, left=None, right=None):
        if (left is None) and (right is None):
            self.eyeL_open = True
            self.eyeR_open = True
        else:
            if left is not None:
                self.eyeL_open = True
            if right is not None:
                self.eyeR_open = True

    def blink(self, left=None, right=None):
        if (left is None) and (right is None):
            self.close()
            self.open()
        else:
            self.close(left, right)
            self.open(left, right)

    def confuse(self):
        self._confused = True

    def laugh(self):
        self._laugh = True

    def wink(self, left=None, right=None):
        assert left or right, "Wink must be activated on right or left"
        self.autoblinker = False
        self.idle = False
        self.blink(left=left, right=right)

    def draw_eyes(self):
        # Curious gaze offsets
        if self._curious:
            if self.eyeLxNext <= 10:
                self.eyeLheightOffset = 8
            elif (self.eyeLxNext >= self.get_screen_constraint_X() - 10) and self._cyclops:
                self.eyeLheightOffset = 8
            else:
                self.eyeLheightOffset = 0

            if self.eyeRxNext >= (self.screenWidth - self.eyeRwidthCurrent - 10):
                self.eyeRheightOffset = 8
            else:
                self.eyeRheightOffset = 0
        else:
            self.eyeLheightOffset = 0
            self.eyeRheightOffset = 0

        # Eye heights
        self.eyeLheightCurrent = (self.eyeLheightCurrent + self.eyeLheightNext + self.eyeLheightOffset) // 2
        self.eyeLy += (self.eyeLheightDefault - self.eyeLheightCurrent) // 2
        self.eyeLy -= self.eyeLheightOffset // 2

        self.eyeRheightCurrent = (self.eyeRheightCurrent + self.eyeRheightNext + self.eyeRheightOffset) // 2
        self.eyeRy += (self.eyeRheightDefault - self.eyeRheightCurrent) // 2
        self.eyeRy -= self.eyeRheightOffset // 2

        # Open eyes after closing
        if self.eyeL_open:
            if self.eyeLheightCurrent <= (1 + self.eyeLheightOffset):
                self.eyeLheightNext = self.eyeLheightDefault

        if self.eyeR_open:
            if self.eyeRheightCurrent <= (1 + self.eyeRheightOffset):
                self.eyeRheightNext = self.eyeRheightDefault

        # Eye widths
        self.eyeLwidthCurrent = (self.eyeLwidthCurrent + self.eyeLwidthNext) // 2
        self.eyeRwidthCurrent = (self.eyeRwidthCurrent + self.eyeRwidthNext) // 2

        # Space between
        self.spaceBetweenCurrent = (self.spaceBetweenCurrent + self.spaceBetweenNext) // 2

        # Eye coordinates
        self.eyeLx = (self.eyeLx + self.eyeLxNext) // 2
        self.eyeLy = (self.eyeLy + self.eyeLyNext) // 2

        self.eyeRxNext = self.eyeLxNext + self.eyeLwidthCurrent + self.spaceBetweenCurrent
        self.eyeRyNext = self.eyeLyNext
        self.eyeRx = (self.eyeRx + self.eyeRxNext) // 2
        self.eyeRy = (self.eyeRy + self.eyeRyNext) // 2

        # Border radius
        self.eyeLborderRadiusCurrent = (self.eyeLborderRadiusCurrent + self.eyeLborderRadiusNext) // 2
        self.eyeRborderRadiusCurrent = (self.eyeRborderRadiusCurrent + self.eyeRborderRadiusNext) // 2

        # Autoblinker
        if self.autoblinker:
            if time.ticks_diff(time.ticks_ms(), self.blinktimer) >= 0:
                self.blink()
                self.blinktimer = time.ticks_add(time.ticks_ms(),
                    (self.blinkInterval * 1000) + (randint(0, self.blinkIntervalVariation) * 1000))

        # Laughing animation
        if self._laugh:
            if self.laughToggle:
                self.vert_flicker(1, 5)
                self.laughAnimationTimer = time.ticks_ms()
                self.laughToggle = False
            elif time.ticks_diff(time.ticks_ms(), self.laughAnimationTimer) >= self.laughAnimationDuration:
                self.vert_flicker(0, 0)
                self.laughToggle = True
                self._laugh = False

        # Confused animation
        if self._confused:
            if self.confusedToggle:
                self.horiz_flicker(1, 20)
                self.confusedAnimationTimer = time.ticks_ms()
                self.confusedToggle = False
            elif time.ticks_diff(time.ticks_ms(), self.confusedAnimationTimer) >= self.confusedAnimationDuration:
                self.horiz_flicker(0, 0)
                self.confusedToggle = True
                self._confused = False

        # Idle mode
        if self.idle:
            if time.ticks_diff(time.ticks_ms(), self.idleAnimationTimer) >= 0:
                self.eyeLxNext = randint(0, self.get_screen_constraint_X())
                self.eyeLyNext = randint(0, self.get_screen_constraint_Y())
                self.idleAnimationTimer = time.ticks_add(time.ticks_ms(),
                    (self.idleInterval * 1000) + (randint(0, self.idleIntervalVariation) * 1000))

        # Horizontal flicker
        if self.hFlicker:
            if self.hFlickerAlternate:
                self.eyeLx += self.hFlickerAmplitude
                self.eyeRx += self.hFlickerAmplitude
            else:
                self.eyeLx -= self.hFlickerAmplitude
                self.eyeRx -= self.hFlickerAmplitude
            self.hFlickerAlternate = not self.hFlickerAlternate

        # Vertical flicker
        if self.vFlicker:
            if self.vFlickerAlternate:
                self.eyeLy += self.vFlickerAmplitude
                self.eyeRy += self.vFlickerAmplitude
            else:
                self.eyeLy -= self.vFlickerAmplitude
                self.eyeRy -= self.vFlickerAmplitude
            self.vFlickerAlternate = not self.vFlickerAlternate

        # Cyclops mode
        if self._cyclops:
            self.eyeRwidthCurrent = 0
            self.eyeRheightCurrent = 0
            self.spaceBetweenCurrent = 0

        # Draw
        self.clear_display()

        # Left eye
        self.gfx.fill_rrect(self.eyeLx, self.eyeLy, self.eyeLwidthCurrent,
                           self.eyeLheightCurrent, self.eyeLborderRadiusCurrent, self.fgcolor)

        # Right eye
        if not self._cyclops:
            self.gfx.fill_rrect(self.eyeRx, self.eyeRy, self.eyeRwidthCurrent,
                               self.eyeRheightCurrent, self.eyeRborderRadiusCurrent, self.fgcolor)

        # Mood eyelids
        if self.tired:
            self.eyelidsTiredHeightNext = self.eyeLheightCurrent // 2
            self.eyelidsAngryHeightNext = 0
        else:
            self.eyelidsTiredHeightNext = 0
        if self.angry:
            self.eyelidsAngryHeightNext = self.eyeLheightCurrent // 2
            self.eyelidsTiredHeightNext = 0
        else:
            self.eyelidsAngryHeightNext = 0
        if self.happy:
            self.eyelidsHappyBottomOffsetNext = self.eyeLheightCurrent // 2
        else:
            self.eyelidsHappyBottomOffsetNext = 0

        # Tired eyelids
        self.eyelidsTiredHeight = (self.eyelidsTiredHeight + self.eyelidsTiredHeightNext) // 2
        if not self._cyclops:
            self.gfx.fill_triangle(self.eyeLx, self.eyeLy - 1,
                                  self.eyeLx + self.eyeLwidthCurrent, self.eyeLy - 1,
                                  self.eyeLx, self.eyeLy + self.eyelidsTiredHeight - 1, self.bgcolor)
            self.gfx.fill_triangle(self.eyeRx, self.eyeRy - 1,
                                  self.eyeRx + self.eyeRwidthCurrent, self.eyeRy - 1,
                                  self.eyeRx + self.eyeRwidthCurrent, self.eyeRy + self.eyelidsTiredHeight - 1, self.bgcolor)
        else:
            self.gfx.fill_triangle(self.eyeLx, self.eyeLy - 1,
                                  self.eyeLx + (self.eyeLwidthCurrent // 2), self.eyeLy - 1,
                                  self.eyeLx, self.eyeLy + self.eyelidsTiredHeight - 1, self.bgcolor)
            self.gfx.fill_triangle(self.eyeLx + (self.eyeLwidthCurrent // 2), self.eyeLy - 1,
                                  self.eyeLx + self.eyeLwidthCurrent, self.eyeLy - 1,
                                  self.eyeLx + self.eyeLwidthCurrent, self.eyeLy + self.eyelidsTiredHeight - 1, self.bgcolor)

        # Angry eyelids
        self.eyelidsAngryHeight = (self.eyelidsAngryHeight + self.eyelidsAngryHeightNext) // 2
        if not self._cyclops:
            self.gfx.fill_triangle(self.eyeLx, self.eyeLy - 1,
                                  self.eyeLx + self.eyeLwidthCurrent, self.eyeLy - 1,
                                  self.eyeLx + self.eyeLwidthCurrent, self.eyeLy + self.eyelidsAngryHeight - 1, self.bgcolor)
            self.gfx.fill_triangle(self.eyeRx, self.eyeRy - 1,
                                  self.eyeRx + self.eyeRwidthCurrent, self.eyeRy - 1,
                                  self.eyeRx, self.eyeRy + self.eyelidsAngryHeight - 1, self.bgcolor)
        else:
            self.gfx.fill_triangle(self.eyeLx, self.eyeLy - 1,
                                  self.eyeLx + (self.eyeLwidthCurrent // 2), self.eyeLy - 1,
                                  self.eyeLx + (self.eyeLwidthCurrent // 2), self.eyeLy + self.eyelidsAngryHeight - 1, self.bgcolor)
            self.gfx.fill_triangle(self.eyeLx + (self.eyeLwidthCurrent // 2), self.eyeLy - 1,
                                  self.eyeLx + self.eyeLwidthCurrent, self.eyeLy - 1,
                                  self.eyeLx + (self.eyeLwidthCurrent // 2), self.eyeLy + self.eyelidsAngryHeight - 1, self.bgcolor)

        # Happy eyelids
        self.eyelidsHappyBottomOffset = (self.eyelidsHappyBottomOffset + self.eyelidsHappyBottomOffsetNext) // 2
        self.gfx.fill_rrect(self.eyeLx - 1, (self.eyeLy + self.eyeLheightCurrent) - self.eyelidsHappyBottomOffset + 1,
                          self.eyeLwidthCurrent + 2, self.eyeLheightDefault, self.eyeLborderRadiusCurrent, self.bgcolor)
        if not self._cyclops:
            self.gfx.fill_rrect(self.eyeRx - 1, (self.eyeRy + self.eyeRheightCurrent) - self.eyelidsHappyBottomOffset + 1,
                              self.eyeRwidthCurrent + 2, self.eyeRheightDefault, self.eyeRborderRadiusCurrent, self.bgcolor)

        self.on_show(self)
