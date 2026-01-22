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

import config
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

        # Demo mode
        self.demo_timer = 0
        self.demo_interval = 2.0

        # FPS tracking
        self.frame_count = 0
        self.fps_time = time.time()
        self.actual_fps = 0

    def init_display(self):
        """Initialize SPI display for robot eyes"""
        if config.DISPLAY_MODE in ["gc9a01a", "st7735s"]:
            self.display = create_display()
            print(f"Display: {config.DISPLAY_MODE.upper()} {config.SCREEN_WIDTH}x{config.SCREEN_HEIGHT}")
        else:
            print("Display: Pygame window")

        # Create pygame surface for RoboEyes
        self.screen = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))

    def init_roboeyes(self):
        """Initialize RoboEyes animation engine"""
        def robo_show(roboeyes):
            if self.display:
                self.display.draw_from_surface(self.screen)

        self.robo = RoboEyes(
            self.screen, config.SCREEN_WIDTH, config.SCREEN_HEIGHT,
            frame_rate=config.DISPLAY_FPS, on_show=robo_show,
            bgcolor=config.EYE_BG_COLOR, fgcolor=config.EYE_FG_COLOR
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

    def run(self):
        """Main application loop"""
        print("=" * 50)
        print(f"NumBot - Robot Eye Tracker v3.0")
        print(f"Mode: {self.mode.upper()}")
        print("=" * 50)

        # Initialize Pygame
        pygame.init()

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

                # Update mood based on finger count
                mood_name = self.hand_tracker.get_mood_from_fingers()
                mood = MOOD_FROM_NAME.get(mood_name, DEFAULT)
                if mood != self.current_mood:
                    self.set_mood(mood)
            else:
                self.has_target = False

    def update_ai_mode(self):
        """Update AI/YOLO mode"""
        if not self.yolo_tracker:
            return

        x, y = self.yolo_tracker.get_normalized_position()
        if x != 0 or y != 0:
            self.target_x = x
            self.target_y = y
            self.has_target = True
        else:
            self.has_target = False

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
            # Note: X is inverted for natural tracking
            eye_x = int((1 - (self.target_x + 1) / 2) * max_x)
            eye_y = int(((self.target_y + 1) / 2) * max_y)

            self.robo.eyeLxNext = max(0, min(max_x, eye_x))
            self.robo.eyeLyNext = max(0, min(max_y, eye_y))

        self.robo.update()

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
    parser.add_argument('--confidence', type=float, default=0.5,
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
