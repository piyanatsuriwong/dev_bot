#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NumBot - Robot Eye Tracker
Main Application with HAND/AI/DEMO Modes

Based on PROJECT_REQUIREMENTS.md v3.0

Modes:
- HAND: Hand tracking with IMX708 + MediaPipe
- AI: YOLO object detection with IMX500
- DEMO: Demo mode (no camera)
"""

import pygame
import cv2
import numpy as np
import sys
import random
import time
import argparse
import os

# Set libcamera environment before importing anything that uses it
# This prevents libcamera from trying to load IMX500 when using HAND mode
os.environ.setdefault('LIBCAMERA_LOG_LEVELS', 'ERROR')

import config

# Sound configuration
SOUND_ENABLED = True
SOUND_HAPPY = "assets/sounds/Voicy_WALL-E 4.mp3"
from roboeyes import *
from servo_controller import ServoController

# Import display driver
if config.DISPLAY_MODE == "gc9a01a":
    from gc9a01a_display import create_display, RPI_AVAILABLE
elif config.DISPLAY_MODE == "st7735s":
    from st7735_display import create_display, RPI_AVAILABLE
else:
    RPI_AVAILABLE = False

# Try to import trackers
try:
    from hand_tracker import HandTracker, PICAMERA2_AVAILABLE, MEDIAPIPE_AVAILABLE
    HAND_AVAILABLE = PICAMERA2_AVAILABLE and MEDIAPIPE_AVAILABLE
except ImportError:
    HAND_AVAILABLE = False
    print("Warning: HandTracker not available")

try:
    from yolo_tracker import create_yolo_tracker, YOLO_AVAILABLE, YoloMode
except ImportError:
    YOLO_AVAILABLE = False
    YoloMode = None
    print("Warning: YOLO Tracker not available")

# Import IMX500 Detector for HAND+AI mode
try:
    from imx500_detector import IMX500Detector, IMX500_AVAILABLE
except ImportError:
    IMX500_AVAILABLE = False
    IMX500Detector = None
    print("Warning: IMX500 Detector not available")


# Mood name mapping
MOOD_NAMES = {
    DEFAULT: "DEFAULT",
    TIRED: "TIRED",
    ANGRY: "ANGRY",
    HAPPY: "HAPPY",
    FROZEN: "FROZEN",
    SCARY: "SCARY",
    CURIOUS: "CURIOUS"
}

MOOD_FROM_NAME = {v: k for k, v in MOOD_NAMES.items()}


class NumBotApp:
    """
    NumBot Main Application

    Supports three operating modes:
    - HAND: Hand tracking with MediaPipe
    - AI: YOLO object detection
    - DEMO: Demo mode with random eye movements
    """

    def __init__(self, mode=None, show_hdmi=True, yolo_confidence=0.5):
        self.mode = mode or config.DEFAULT_MODE
        self.show_hdmi = show_hdmi
        self.yolo_confidence = yolo_confidence

        # Components
        self.display = None
        self.screen = None
        self.robo = None
        self.servo = None
        self.hand_tracker = None
        self.yolo_tracker = None
        self.imx500_detector = None  # IMX500 detector for HAND+AI mode

        # Sound
        self.sound_happy = None
        self.sound_cooldown = 0  # Cooldown timer to prevent sound spam

        # IMX500 detection text (shown on LCD in HAND mode)
        self.imx500_text = ""
        self.imx500_text_display = ""  # What's actually shown (with hold)
        self.imx500_last_update = 0  # Time of last detection change
        self.imx500_hold_duration = 2.0  # Hold detection for 2 seconds
        self.imx500_is_new = False  # Show "NEW" indicator
        self.imx500_priority_objects = ["person", "cat", "dog", "bird"]  # Hold these longer

        # HDMI display
        self.hdmi_screen = None
        self.clock = None
        self.font = None
        self.small_font = None

        # State
        self.running = False
        self.current_mood = DEFAULT
        self.target_x = 0.0
        self.target_y = 0.0
        self.has_target = False
        self.prev_has_target = False  # Track previous state for smooth re-detection
        self.finger_count = -1  # Current finger count for display (-1 = no hand)

        # LCD overlay text (displayed at bottom center)
        self.overlay_text = ""  # Text to show on LCD (finger count, detection, etc.)

        # Demo mode
        self.demo_timer = 0
        self.demo_interval = 2.0

        # FPS tracking
        self.frame_count = 0
        self.fps_time = time.time()
        self.actual_fps = 0

        # LCD overlay font (for finger count display)
        self.lcd_font = None

    def init_display(self):
        """Initialize SPI display for robot eyes"""
        if config.DISPLAY_MODE in ["gc9a01a", "st7735s"]:
            self.display = create_display()
            print(f"Display: {config.DISPLAY_MODE.upper()} {config.SCREEN_WIDTH}x{config.SCREEN_HEIGHT}")
        else:
            print("Display: Pygame window")

        # Create pygame surface for RoboEyes
        self.screen = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))

    def draw_mouth(self, finger_count):
        """
        Draw mouth shape based on finger count
        0 = Angry mouth (frown)
        1 = Curious mouth (small O)
        2 = Happy mouth (small smile)
        3 = Tired mouth (horizontal line)
        4 = Scary mouth (zigzag)
        5 = Default mouth (wide smile)
        """
        # Mouth position (centered at bottom)
        center_x = config.SCREEN_WIDTH // 2
        bottom_y = config.SCREEN_HEIGHT - 8
        
        # Scale mouth size based on screen width
        mouth_width = int(config.SCREEN_WIDTH * 0.3)  # 30% of screen width
        mouth_height = int(config.SCREEN_WIDTH * 0.08)  # 8% of screen width
        
        color = (255, 255, 255)  # White
        
        if finger_count == 0:  # ANGRY - Inverted smile (frown)
            # Arc facing down
            rect = pygame.Rect(center_x - mouth_width//2, bottom_y - mouth_height, 
                              mouth_width, mouth_height * 2)
            pygame.draw.arc(self.screen, color, rect, 0, 3.14159, 3)
            
        elif finger_count == 1:  # CURIOUS - Small circle (O)
            radius = mouth_height
            pygame.draw.circle(self.screen, color, (center_x, bottom_y - radius), radius, 2)
            
        elif finger_count == 2:  # HAPPY - Small smile
            rect = pygame.Rect(center_x - mouth_width//2, bottom_y - mouth_height*2, 
                              mouth_width, mouth_height * 2)
            pygame.draw.arc(self.screen, color, rect, 3.14159, 6.28318, 3)
            
        elif finger_count == 3:  # TIRED - Horizontal line
            pygame.draw.line(self.screen, color, 
                           (center_x - mouth_width//2, bottom_y - mouth_height),
                           (center_x + mouth_width//2, bottom_y - mouth_height), 3)
            
        elif finger_count == 4:  # SCARY - Zigzag mouth
            points = [
                (center_x - mouth_width//2, bottom_y - mouth_height),
                (center_x - mouth_width//4, bottom_y - mouth_height*2),
                (center_x, bottom_y - mouth_height),
                (center_x + mouth_width//4, bottom_y - mouth_height*2),
                (center_x + mouth_width//2, bottom_y - mouth_height)
            ]
            pygame.draw.lines(self.screen, color, False, points, 3)
            
        else:  # 5 = DEFAULT - Wide smile
            rect = pygame.Rect(center_x - mouth_width//2, bottom_y - mouth_height*3, 
                              mouth_width, mouth_height * 3)
            pygame.draw.arc(self.screen, color, rect, 3.14159, 6.28318, 3)

    def draw_detection_overlay(self, text):
        """
        Draw IMX500 detection text at top of LCD screen
        Shows what objects are detected by YOLO
        Beautiful compact display for small screens
        """
        if not text:
            return

        # Extract just the object names (remove scores for cleaner look)
        parts = text.split(", ")
        labels = []
        for part in parts[:2]:  # Max 2 items for small screen
            if ":" in part:
                labels.append(part.split(":")[0])
            else:
                labels.append(part)
        
        display_text = ", ".join(labels) if labels else ""
        
        if not display_text:
            return
        
        # Truncate if too long
        max_chars = (config.SCREEN_WIDTH // 7) - 2  # Leave space for icon
        if len(display_text) > max_chars:
            display_text = display_text[:max_chars-1] + ".."

        if not self.lcd_font:
            return
        
        # Colors
        text_color = (0, 255, 180)  # Bright cyan-green
        bg_color = (0, 0, 0)  # Solid black background
        border_color = (0, 120, 100)  # Teal border
        icon_color = (255, 200, 0)  # Yellow/gold for eye icon
        
        text_surf = self.lcd_font.render(display_text, True, text_color)
        
        # Icon size
        icon_size = 10
        icon_padding = 4
        
        # Total width = icon + padding + text
        total_width = icon_size + icon_padding + text_surf.get_width()
        
        # Position at top center
        start_x = (config.SCREEN_WIDTH - total_width) // 2
        y = 3
        
        # Draw background pill
        padding_x = 5
        padding_y = 2
        bg_rect = pygame.Rect(
            start_x - padding_x, 
            y - padding_y, 
            total_width + padding_x * 2, 
            max(text_surf.get_height(), icon_size) + padding_y * 2
        )
        
        # Solid background
        pygame.draw.rect(self.screen, bg_color, bg_rect, border_radius=7)
        
        # Draw border
        pygame.draw.rect(self.screen, border_color, bg_rect, 1, border_radius=7)
        
        # Draw eye icon (simple eye shape)
        icon_x = start_x + icon_size // 2
        icon_y = y + text_surf.get_height() // 2
        
        # Eye outer (yellow ellipse) - or green if NEW
        if self.imx500_is_new and (time.time() - self.imx500_last_update) < 0.5:
            # Flash green for new detection
            eye_color = (0, 255, 100)
        else:
            eye_color = icon_color
            self.imx500_is_new = False  # Reset after flash
        
        eye_rect = pygame.Rect(icon_x - 5, icon_y - 3, 10, 6)
        pygame.draw.ellipse(self.screen, eye_color, eye_rect)
        
        # Eye pupil (black circle)
        pygame.draw.circle(self.screen, (0, 0, 0), (icon_x, icon_y), 2)
        
        # Eye highlight (white dot)
        pygame.draw.circle(self.screen, (255, 255, 255), (icon_x + 1, icon_y - 1), 1)
        
        # Draw text after icon
        text_x = start_x + icon_size + icon_padding
        self.screen.blit(text_surf, (text_x, y))
        
        # Draw "NEW!" badge when new detection (for 1 second)
        time_since_new = time.time() - self.imx500_last_update
        if self.imx500_is_new and time_since_new < 1.0:
            # Blinking effect (on/off every 0.2s)
            if int(time_since_new * 5) % 2 == 0:
                new_font = pygame.font.Font(None, 14)
                new_surf = new_font.render("NEW!", True, (255, 255, 255))
                
                # Position to the right of the pill
                new_x = bg_rect.right + 2
                new_y = y
                
                # Red background pill for NEW
                new_bg = pygame.Rect(new_x, new_y - 1, new_surf.get_width() + 4, new_surf.get_height() + 2)
                pygame.draw.rect(self.screen, (255, 50, 50), new_bg, border_radius=4)
                pygame.draw.rect(self.screen, (255, 100, 100), new_bg, 1, border_radius=4)
                
                self.screen.blit(new_surf, (new_x + 2, new_y))
        elif time_since_new >= 1.0:
            self.imx500_is_new = False  # Reset after 1 second

    def init_roboeyes(self):
        """Initialize RoboEyes animation engine"""
        # Initialize LCD font FIRST (needed in callback)
        pygame.font.init()
        font_size = 20 if config.SCREEN_WIDTH >= 160 else 16
        self.lcd_font = pygame.font.Font(None, font_size)

        def robo_show(roboeyes):
            # Draw mouth based on finger count (instead of text)
            if self.finger_count >= 0 and self.mode == config.MODE_HAND:
                self.draw_mouth(self.finger_count)

            # Draw IMX500 detection text on LCD (HAND mode with YOLO)
            if self.imx500_text_display and self.mode == config.MODE_HAND:
                self.draw_detection_overlay(self.imx500_text_display)

            if self.display:
                self.display.draw_from_surface(self.screen)

        # Get smoothing from config (default 0.15 for smooth movement)
        smoothing = getattr(config, 'EYE_SMOOTHING', 0.15)

        self.robo = RoboEyes(
            self.screen, config.SCREEN_WIDTH, config.SCREEN_HEIGHT,
            frame_rate=config.DISPLAY_FPS, on_show=robo_show,
            bgcolor=config.EYE_BG_COLOR, fgcolor=config.EYE_FG_COLOR,
            smoothing=smoothing
        )

        # Configure eye dimensions
        self.robo.eyes_width(config.EYE_WIDTH, config.EYE_WIDTH)
        self.robo.eyes_height(config.EYE_HEIGHT, config.EYE_HEIGHT)
        self.robo.eyes_spacing(config.EYE_SPACING)
        self.robo.eyes_radius(config.EYE_RADIUS, config.EYE_RADIUS)
        self.robo.set_auto_blinker(ON, 3, 2)

        print("RoboEyes: Initialized")

    def init_hdmi(self):
        """Initialize HDMI display for camera preview"""
        if not self.show_hdmi:
            self.hdmi_screen = pygame.display.set_mode((320, 240))
            pygame.display.set_caption("NumBot Control")
            return

        self.hdmi_screen = pygame.display.set_mode((config.HDMI_WIDTH, config.HDMI_HEIGHT))
        pygame.display.set_caption(f"NumBot - {self.mode.upper()} Mode")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        print(f"HDMI: {config.HDMI_WIDTH}x{config.HDMI_HEIGHT}")

    def init_servo(self):
        """Initialize servo controller"""
        self.servo = ServoController()
        if self.servo.enabled:
            print("Servo: PCA9685 initialized")
        else:
            print("Servo: Not available")

    def init_hand_tracker(self):
        """Initialize hand tracker for HAND mode"""
        if not HAND_AVAILABLE:
            print("HandTracker: Not available")
            return False

        self.hand_tracker = HandTracker()
        if self.hand_tracker.start():
            print("HandTracker: Started")
            return True
        else:
            print("HandTracker: Failed to start")
            self.hand_tracker = None
            return False

    def init_yolo_tracker(self):
        """Initialize YOLO tracker for AI mode"""
        if not YOLO_AVAILABLE:
            print("YOLO: Not available")
            return False

        self.yolo_tracker = create_yolo_tracker(
            confidence_threshold=self.yolo_confidence,
            frame_rate=10
        )

        if self.yolo_tracker and self.yolo_tracker.start():
            self.yolo_tracker.set_mode(YoloMode.TRACK)
            self.yolo_tracker.set_track_target(config.YOLO_DEFAULT_TARGET)
            print("YOLO: Started")
            return True
        else:
            print("YOLO: Failed to start")
            self.yolo_tracker = None
            return False

    def init_imx500_detector(self):
        """Initialize IMX500 detector for background YOLO detection in HAND mode"""
        if not IMX500_AVAILABLE or IMX500Detector is None:
            print("IMX500 Detector: Not available")
            return False

        self.imx500_detector = IMX500Detector(
            model="yolov8n",
            threshold=self.yolo_confidence
        )

        if self.imx500_detector.start():
            print("IMX500 Detector: Started (running with hand tracking)")
            return True
        else:
            print("IMX500 Detector: Failed to start")
            self.imx500_detector = None
            return False

    def run(self):
        """Main application loop"""
        print("=" * 50)
        print(f"NumBot - Robot Eye Tracker v3.0")
        print(f"Mode: {self.mode.upper()}")
        print("=" * 50)

        # Initialize Pygame
        pygame.init()

        # Initialize sound
        if SOUND_ENABLED:
            try:
                pygame.mixer.init()
                if os.path.exists(SOUND_HAPPY):
                    self.sound_happy = pygame.mixer.Sound(SOUND_HAPPY)
                    print(f"Sound: Loaded {SOUND_HAPPY}")
                else:
                    print(f"Sound: File not found {SOUND_HAPPY}")
            except Exception as e:
                print(f"Sound: Init failed - {e}")

        # Initialize components
        self.init_display()
        self.init_roboeyes()
        self.init_hdmi()
        self.init_servo()

        # Initialize mode-specific components
        if self.mode == config.MODE_HAND:
            if not self.init_hand_tracker():
                print("Falling back to DEMO mode")
                self.mode = config.MODE_DEMO
            else:
                # Also start IMX500 detector for simultaneous YOLO detection
                self.init_imx500_detector()
        elif self.mode == config.MODE_AI:
            if not self.init_yolo_tracker():
                print("Falling back to DEMO mode")
                self.mode = config.MODE_DEMO

        print("\nKeyboard Controls:")
        print("  ESC   - Exit")
        print("  SPACE - Random mood")
        print("  1-6   - Set mood (1=Default, 2=Happy, 3=Angry, 4=Tired, 5=Scary, 6=Curious)")
        if self.mode == config.MODE_AI:
            print("  D     - DETECT mode")
            print("  T     - TRACK mode")
        print()

        self.running = True

        try:
            while self.running:
                # Handle events
                self.handle_events()

                # Update tracking based on mode
                if self.mode == config.MODE_HAND:
                    self.update_hand_mode()
                elif self.mode == config.MODE_AI:
                    self.update_ai_mode()
                else:
                    self.update_demo_mode()

                # Update robot eyes
                self.update_roboeyes()

                # Update servo
                if self.servo and self.servo.enabled:
                    if self.has_target:
                        # Reset smoothing when hand is detected for the first time
                        if not self.prev_has_target and hasattr(self.servo, 'reset_smoothing'):
                            self.servo.reset_smoothing()
                        
                        # Support both old (track_normalized + update) and new (track_hand) API
                        if hasattr(self.servo, 'track_hand'):
                            # New API: track_hand with built-in smoothing (lower = smoother)
                            self.servo.track_hand(self.target_x, self.target_y, smoothing=0.1)
                        else:
                            # Old API: track_normalized + update
                            self.servo.track_normalized(self.target_x, self.target_y)
                            self.servo.update()
                    
                    # Update previous state
                    self.prev_has_target = self.has_target

                # Update HDMI display
                self.update_hdmi()

                # FPS tracking
                self.frame_count += 1
                now = time.time()
                if now - self.fps_time >= 1.0:
                    self.actual_fps = self.frame_count / (now - self.fps_time)
                    self.frame_count = 0
                    self.fps_time = now

                pygame.display.flip()
                if self.clock:
                    self.clock.tick(config.FPS)

        except KeyboardInterrupt:
            print("\nInterrupted")

        self.cleanup()

    def handle_events(self):
        """Handle keyboard and window events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

                elif event.key == pygame.K_SPACE:
                    # Random mood
                    new_mood = random.choice([DEFAULT, HAPPY, ANGRY, TIRED, SCARY, CURIOUS])
                    self.set_mood(new_mood)

                elif event.key == pygame.K_1:
                    self.set_mood(DEFAULT)
                elif event.key == pygame.K_2:
                    self.set_mood(HAPPY)
                elif event.key == pygame.K_3:
                    self.set_mood(ANGRY)
                elif event.key == pygame.K_4:
                    self.set_mood(TIRED)
                elif event.key == pygame.K_5:
                    self.set_mood(SCARY)
                elif event.key == pygame.K_6:
                    self.set_mood(CURIOUS)

                elif event.key == pygame.K_d and self.mode == config.MODE_AI:
                    if self.yolo_tracker and YoloMode:
                        self.yolo_tracker.set_mode(YoloMode.DETECT)

                elif event.key == pygame.K_t and self.mode == config.MODE_AI:
                    if self.yolo_tracker and YoloMode:
                        self.yolo_tracker.set_mode(YoloMode.TRACK)

    def set_mood(self, mood):
        """Set robot mood"""
        self.current_mood = mood
        self.robo.mood = mood
        print(f"Mood: {MOOD_NAMES.get(mood, 'UNKNOWN')}")

        # Play sound for HAPPY mood (with 3 second cooldown)
        now = time.time()
        if mood == HAPPY and self.sound_happy and now > self.sound_cooldown:
            try:
                self.sound_happy.play()
                self.sound_cooldown = now + 3.0  # 3 seconds cooldown
            except Exception:
                pass

    def update_hand_mode(self):
        """Update hand tracking mode"""
        if not self.hand_tracker:
            return

        if self.hand_tracker.update():
            if self.hand_tracker.has_hand:
                x, y = self.hand_tracker.get_normalized_position()
                self.target_x = x
                self.target_y = y
                self.has_target = True

                # Store finger count for mouth display
                self.finger_count = self.hand_tracker.finger_count

                # Update mood based on finger count
                mood_name = self.hand_tracker.get_mood_from_fingers()
                mood = MOOD_FROM_NAME.get(mood_name, DEFAULT)
                if mood != self.current_mood:
                    self.set_mood(mood)
            else:
                self.has_target = False
                self.finger_count = -1  # No mouth when no hand

        # Update IMX500 detection text (runs in parallel)
        if self.imx500_detector and self.imx500_detector.running:
            new_text = self.imx500_detector.get_detection_text()
            now = time.time()
            
            # Check if we should update display
            should_update = False
            time_since_update = now - self.imx500_last_update
            
            if new_text != self.imx500_text:
                # New detection found
                if new_text:
                    # Check if it's a priority object (hold current less time)
                    is_priority = any(obj in new_text.lower() for obj in self.imx500_priority_objects)
                    
                    if is_priority:
                        # Priority object - update immediately
                        should_update = True
                        self.imx500_hold_duration = 3.0  # Hold priority objects longer
                    elif time_since_update >= 1.0:
                        # Non-priority - wait at least 1 second
                        should_update = True
                        self.imx500_hold_duration = 2.0
                elif time_since_update >= self.imx500_hold_duration:
                    # No detection and hold time expired
                    should_update = True
                
                if should_update:
                    self.imx500_text = new_text
                    self.imx500_text_display = new_text
                    self.imx500_last_update = now
                    self.imx500_is_new = bool(new_text)  # Show NEW indicator
                    # Print detection to console
                    if new_text:
                        print(f"Detected: {new_text}")

    def update_ai_mode(self):
        """Update AI/YOLO mode"""
        if not self.yolo_tracker:
            return

        x, y = self.yolo_tracker.get_normalized_position()
        if x != 0 or y != 0:
            self.target_x = x
            self.target_y = y
            self.has_target = True
            # Set overlay text with detection labels
            labels = self.yolo_tracker.get_detection_text(max_items=2)
            if labels:
                self.overlay_text = ", ".join(labels)
            else:
                self.overlay_text = ""
        else:
            self.has_target = False
            self.overlay_text = ""

    def update_demo_mode(self):
        """Update demo mode with random movements"""
        now = time.time()
        if now - self.demo_timer >= self.demo_interval:
            self.target_x = random.uniform(-0.8, 0.8)
            self.target_y = random.uniform(-0.5, 0.5)
            self.has_target = True
            self.demo_timer = now
            self.demo_interval = random.uniform(1.0, 3.0)

    def update_roboeyes(self):
        """Update robot eye animation"""
        if self.has_target:
            max_x = self.robo.get_screen_constraint_X()
            max_y = self.robo.get_screen_constraint_Y()

            # Convert normalized coords to eye position
            # X: -1 (left) to +1 (right) -> 0 to max_x
            # Y: -1 (top) to +1 (bottom) -> 0 to max_y
            eye_x = int(((self.target_x + 1) / 2) * max_x)
            eye_y = int(((self.target_y + 1) / 2) * max_y)

            self.robo.eyeLxNext = max(0, min(max_x, eye_x))
            self.robo.eyeLyNext = max(0, min(max_y, eye_y))

        self.robo.update()
        # Note: overlay_text is drawn in robo_show callback (fixes flickering)

    def update_hdmi(self):
        """Update HDMI display"""
        if not self.show_hdmi:
            return

        self.hdmi_screen.fill((20, 20, 20))

        # Draw camera frames based on mode
        if self.mode == config.MODE_HAND:
            self.draw_hand_view()
        elif self.mode == config.MODE_AI:
            self.draw_ai_view()
        else:
            self.draw_demo_view()

        # Draw status bar
        self.draw_status_bar()

    def draw_hand_view(self):
        """Draw hand tracking camera view"""
        if self.hand_tracker:
            frame = self.hand_tracker.get_frame()
            if frame is not None:
                # Resize to fit panel
                frame = cv2.resize(frame, (config.HDMI_PANEL_WIDTH, config.HDMI_HEIGHT - 40))
                # Draw centered
                x_offset = (config.HDMI_WIDTH - config.HDMI_PANEL_WIDTH) // 2
                surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                self.hdmi_screen.blit(surf, (x_offset, 0))

                # Draw info
                status = self.hand_tracker.get_status_text()
                self.draw_text(status, x_offset + 10, 10, (0, 255, 0))
        else:
            self.draw_placeholder("No Hand Tracker")

    def draw_ai_view(self):
        """Draw AI/YOLO camera view"""
        if self.yolo_tracker:
            frame = self.yolo_tracker.latest_frame
            if frame is not None:
                # Resize to fit panel
                frame = cv2.resize(frame, (config.HDMI_PANEL_WIDTH, config.HDMI_HEIGHT - 40))
                # Convert BGR to RGB
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Draw centered
                x_offset = (config.HDMI_WIDTH - config.HDMI_PANEL_WIDTH) // 2
                surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                self.hdmi_screen.blit(surf, (x_offset, 0))

                # Draw detection info
                labels = self.yolo_tracker.get_detection_text(max_items=3)
                if labels:
                    self.draw_text(f"Detected: {', '.join(labels)}", x_offset + 10, 10, (0, 255, 0))

                # Draw mode
                mode_text = self.yolo_tracker.get_mode_text()
                self.draw_text(f"Mode: {mode_text}", x_offset + 10, 40, (0, 255, 255))
            else:
                self.draw_placeholder("Waiting for YOLO...")
        else:
            self.draw_placeholder("No YOLO Tracker")

    def draw_demo_view(self):
        """Draw demo mode view"""
        # Draw eye preview in center
        x_offset = (config.HDMI_WIDTH - config.SCREEN_WIDTH * 2) // 2
        y_offset = (config.HDMI_HEIGHT - config.SCREEN_HEIGHT * 2 - 40) // 2

        # Scale up the robot eye display
        scaled = pygame.transform.scale(self.screen, (config.SCREEN_WIDTH * 2, config.SCREEN_HEIGHT * 2))
        self.hdmi_screen.blit(scaled, (x_offset, y_offset))

        # Draw info
        self.draw_text("DEMO MODE - Random Eye Movements", 10, 10, (255, 255, 0))
        self.draw_text("Press 1-6 to change mood, SPACE for random", 10, 40, (150, 150, 150))

    def draw_placeholder(self, text):
        """Draw placeholder when no camera"""
        x_offset = (config.HDMI_WIDTH - config.HDMI_PANEL_WIDTH) // 2
        pygame.draw.rect(self.hdmi_screen, (30, 30, 30),
                         (x_offset, 0, config.HDMI_PANEL_WIDTH, config.HDMI_HEIGHT - 40))
        self.draw_text(text, config.HDMI_WIDTH // 2 - 80, config.HDMI_HEIGHT // 2 - 20, (100, 100, 100))

    def draw_status_bar(self):
        """Draw status bar at bottom"""
        y = config.HDMI_HEIGHT - 35

        # Mode
        self.draw_text(f"Mode: {self.mode.upper()}", 10, y, (0, 255, 255))

        # Mood
        mood_name = MOOD_NAMES.get(self.current_mood, "UNKNOWN")
        self.draw_text(f"Mood: {mood_name}", 200, y, (255, 255, 0))

        # FPS
        self.draw_text(f"FPS: {self.actual_fps:.1f}", 400, y, (200, 200, 200))

        # Controls hint
        self.draw_text("ESC=Exit  SPACE=Random  1-6=Moods", 600, y, (100, 100, 100), small=True)

    def draw_text(self, text, x, y, color, small=False):
        """Draw text on HDMI display"""
        font = self.small_font if small else self.font
        if font:
            surf = font.render(text, True, color)
            self.hdmi_screen.blit(surf, (x, y))

    def cleanup(self):
        """Cleanup all resources"""
        print("\nCleaning up...")
        self.running = False

        if self.hand_tracker:
            self.hand_tracker.cleanup()

        if self.yolo_tracker:
            self.yolo_tracker.cleanup()

        if self.imx500_detector:
            self.imx500_detector.stop()

        if self.servo:
            self.servo.cleanup()

        if self.display:
            self.display.cleanup()

        pygame.quit()
        print("NumBot stopped.")


def main():
    parser = argparse.ArgumentParser(description="NumBot - Robot Eye Tracker")
    parser.add_argument('--mode', choices=['hand', 'ai', 'demo'], default='demo',
                        help='Operating mode (hand/ai/demo)')
    parser.add_argument('--no-hdmi', action='store_true',
                        help='Disable HDMI camera preview')
    parser.add_argument('--confidence', type=float, default=0.7,
                        help='YOLO confidence threshold (0.0-1.0)')
    args = parser.parse_args()

    app = NumBotApp(
        mode=args.mode,
        show_hdmi=not args.no_hdmi,
        yolo_confidence=args.confidence
    )
    app.run()


if __name__ == "__main__":
    main()
