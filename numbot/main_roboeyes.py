#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NumBot - Robot Eye Tracker with RoboEyes
Raspberry Pi 5 with ST7735S / GC9A01A Display

Architecture:
- Main Thread: HDMI Display + Keyboard Events
- Thread 1: Eye Animation (ST7735S SPI Display)
- Thread 2: YOLO Detection (IMX500 AI Camera)
"""

import pygame
import cv2
import numpy as np
import sys
import random
import time
import gc
import threading
import subprocess
import os
import signal
import shutil
from PIL import Image, ImageDraw, ImageFont

import config
from roboeyes import *
from servo_controller import ServoController

# Try to import Picamera2
try:
    from picamera2 import Picamera2
    PICAM2_AVAILABLE = True
except ImportError:
    PICAM2_AVAILABLE = False
    print("Warning: Picamera2 not available")

# Import display driver based on config
if config.DISPLAY_MODE == "gc9a01a":
    from gc9a01a_display import create_display, RPI_AVAILABLE
elif config.DISPLAY_MODE == "st7735s":
    from st7735_display import create_display, RPI_AVAILABLE
else:
    RPI_AVAILABLE = False

try:
    from yolo_tracker import create_yolo_tracker, YOLO_AVAILABLE, YoloMode
except ImportError:
    YOLO_AVAILABLE = False
    YoloMode = None
    print("YOLO Tracker: Not available")


class SharedState:
    """Thread-safe shared state between threads"""

    def __init__(self):
        self._lock = threading.Lock()
        self._target_x = 0.0
        self._target_y = 0.0
        self._has_target = False
        self._mood = DEFAULT
        self._detected_labels = []
        self._yolo_fps = 0
        self._yolo_frame = None
        self._running = True

    @property
    def running(self):
        with self._lock:
            return self._running

    @running.setter
    def running(self, value):
        with self._lock:
            self._running = value

    def set_target(self, x, y, has_target=True):
        with self._lock:
            self._target_x = x
            self._target_y = y
            self._has_target = has_target

    def get_target(self):
        with self._lock:
            return self._target_x, self._target_y, self._has_target

    @property
    def mood(self):
        with self._lock:
            return self._mood

    @mood.setter
    def mood(self, value):
        with self._lock:
            self._mood = value

    def set_detections(self, labels, fps, frame=None):
        with self._lock:
            self._detected_labels = labels
            self._yolo_fps = fps
            if frame is not None:
                self._yolo_frame = frame

    def get_detections(self):
        with self._lock:
            return self._detected_labels.copy(), self._yolo_fps

    def get_yolo_frame(self):
        with self._lock:
            return self._yolo_frame


class EyeAnimationThread(threading.Thread):
    """
    Thread 1: Eye Animation on ST7735S/GC9A01A SPI Display
    Uses its own pygame instance for Surface drawing (not display)
    """

    def __init__(self, shared_state, servo=None):
        super().__init__(name="EyeAnimationThread", daemon=True)
        self.shared = shared_state
        self.servo = servo
        self.display = None
        self.screen = None
        self.robo = None
        self.fps = config.FPS

    def run(self):
        print("[Thread 1] Eye Animation: Starting...")

        # Create SPI Display
        if config.DISPLAY_MODE in ["gc9a01a", "st7735s"]:
            self.display = create_display()
            print(f"[Thread 1] Display: {config.DISPLAY_MODE.upper()} {config.SCREEN_WIDTH}x{config.SCREEN_HEIGHT}")
        else:
            print("[Thread 1] No SPI display configured")
            return

        # Create pygame Surface (not display window)
        self.screen = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))

        # RoboEyes Setup
        def robo_show(roboeyes):
            if self.display:
                self.display.draw_from_surface(self.screen)

        self.robo = RoboEyes(
            self.screen, config.SCREEN_WIDTH, config.SCREEN_HEIGHT,
            frame_rate=self.fps, on_show=robo_show,
            bgcolor=0, fgcolor=(255, 255, 255)
        )

        # Config sizes based on display
        if config.DISPLAY_MODE == "gc9a01a":
            self.robo.eyes_width(90, 90)
            self.robo.eyes_height(90, 90)
            self.robo.eyes_spacing(20)
            self.robo.eyes_radius(20, 20)
        elif config.DISPLAY_MODE == "st7735s":
            self.robo.eyes_width(50, 50)
            self.robo.eyes_height(60, 60)
            self.robo.eyes_spacing(10)
            self.robo.eyes_radius(10, 10)

        self.robo.set_auto_blinker(ON, 3, 2)

        print("[Thread 1] Eye Animation: Running...")

        frame_interval = 1.0 / self.fps
        last_time = time.time()
        frame_count = 0
        last_fps_time = time.time()

        while self.shared.running:
            now = time.time()

            # Get target from shared state
            target_x, target_y, has_target = self.shared.get_target()

            # Update mood from shared state
            new_mood = self.shared.mood
            if self.robo.mood != new_mood:
                self.robo.mood = new_mood
                print(f"[Thread 1] Mood changed to: {new_mood}")

            # Update Eye Position
            if has_target:
                max_x = self.robo.get_screen_constraint_X()
                max_y = self.robo.get_screen_constraint_Y()
                eye_x = int((1 - (target_x + 1) / 2) * max_x)
                eye_y = int(((target_y + 1) / 2) * max_y)
                self.robo.eyeLxNext = max(0, min(max_x, eye_x))
                self.robo.eyeLyNext = max(0, min(max_y, eye_y))

                # Update servo if available
                if self.servo and self.servo.enabled:
                    self.servo.track_hand(target_x, target_y)

            # Update RoboEyes animation
            self.robo.update()

            # FPS tracking
            frame_count += 1
            if now - last_fps_time >= 5.0:
                actual_fps = frame_count / (now - last_fps_time)
                print(f"[Thread 1] Eye FPS: {actual_fps:.1f}")
                frame_count = 0
                last_fps_time = now

            # Frame rate control
            elapsed = time.time() - last_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_time = time.time()

        # Cleanup
        print("[Thread 1] Eye Animation: Stopping...")
        if self.display:
            self.display.cleanup()
        print("[Thread 1] Eye Animation: Stopped")


class YoloDetectionThread(threading.Thread):
    """
    Thread 2: YOLO Object Detection on IMX500
    """

    def __init__(self, shared_state, confidence=0.5):
        super().__init__(name="YoloDetectionThread", daemon=True)
        self.shared = shared_state
        self.confidence = confidence
        self.yolo_tracker = None
        self.track_target = "person"

    def run(self):
        print("[Thread 2] YOLO Detection: Starting...")

        if not YOLO_AVAILABLE:
            print("[Thread 2] YOLO not available, thread will idle")
            while self.shared.running:
                time.sleep(1)
            return

        # Initialize YOLO Tracker
        self.yolo_tracker = create_yolo_tracker(
            confidence_threshold=self.confidence,
            frame_rate=10
        )

        if not self.yolo_tracker:
            print("[Thread 2] Failed to create YOLO tracker")
            return

        # Set initial mode
        if YoloMode:
            self.yolo_tracker.set_mode(YoloMode.TRACK)
        self.yolo_tracker.set_track_target(self.track_target)

        # Start YOLO detection
        if not self.yolo_tracker.start():
            print("[Thread 2] Failed to start YOLO tracker")
            return

        print("[Thread 2] YOLO Detection: Running...")

        last_log_time = time.time()

        while self.shared.running:
            # Get detection results
            tx, ty = self.yolo_tracker.get_normalized_position()

            if tx != 0 or ty != 0:
                self.shared.set_target(tx, ty, has_target=True)
            else:
                self.shared.set_target(0, 0, has_target=False)

            # Update detection labels and frame
            labels = self.yolo_tracker.get_detection_text(max_items=3)
            fps = self.yolo_tracker.fps
            frame = self.yolo_tracker.latest_frame
            self.shared.set_detections(labels, fps, frame)

            # Log every 5 seconds
            now = time.time()
            if now - last_log_time >= 5.0:
                if labels:
                    print(f"[Thread 2] YOLO: {', '.join(labels)} | FPS: {fps}")
                else:
                    print(f"[Thread 2] YOLO: No detections | FPS: {fps}")
                last_log_time = now

            time.sleep(0.01)

        # Cleanup
        print("[Thread 2] YOLO Detection: Stopping...")
        if self.yolo_tracker:
            self.yolo_tracker.cleanup()
        print("[Thread 2] YOLO Detection: Stopped")

    def set_mode(self, mode):
        """Set YOLO mode from main thread"""
        if self.yolo_tracker and YoloMode:
            self.yolo_tracker.set_mode(mode)
            print(f"[Thread 2] Mode changed to: {mode.value}")

    def set_track_target(self, target):
        """Set track target from main thread"""
        if self.yolo_tracker:
            self.yolo_tracker.set_track_target(target)


class NumBotApp:
    """
    Main Application
    - Main thread handles HDMI display and keyboard
    - Thread 1: Eye Animation (SPI)
    - Thread 2: YOLO Detection
    """

    def __init__(self, show_hdmi=True, yolo_confidence=0.5):
        self.show_hdmi = show_hdmi
        self.yolo_confidence = yolo_confidence

        # Shared state
        self.shared = SharedState()

        # Servo controller
        self.servo = ServoController()

        # Threads
        self.eye_thread = None
        self.yolo_thread = None

        # Secondary camera (IMX708)
        self.secondary_cam = None

        # Pygame (HDMI)
        self.screen = None
        self.clock = None
        self.font = None

    def init_secondary_camera(self):
        """Initialize IMX708 secondary camera"""
        if not PICAM2_AVAILABLE:
            print("[Main] Picamera2 not available")
            return False

        for cam_idx in [0, 1]:
            try:
                print(f"[Main] Trying camera index {cam_idx}...")
                self.secondary_cam = Picamera2(camera_num=cam_idx)
                cam_config = self.secondary_cam.create_preview_configuration(
                    main={"size": (640, 480), "format": "RGB888"}
                )
                self.secondary_cam.configure(cam_config)
                self.secondary_cam.start()
                print(f"[Main] Secondary camera (IMX708) started on index {cam_idx}")
                return True
            except Exception as e:
                print(f"[Main] Camera {cam_idx} failed: {e}")
                if self.secondary_cam:
                    try:
                        self.secondary_cam.close()
                    except:
                        pass
                self.secondary_cam = None

        print("[Main] Could not initialize secondary camera")
        return False

    def run(self):
        print("=" * 50)
        print("NumBot - Multi-Threaded Robot Eye Tracker")
        print("=" * 50)

        # Initialize Pygame (Main Thread)
        pygame.init()

        if self.show_hdmi:
            self.screen = pygame.display.set_mode((1280, 480))
            pygame.display.set_caption("NumBot - Dual Camera View")
        else:
            # Minimal display for event handling
            self.screen = pygame.display.set_mode((320, 240))
            pygame.display.set_caption("NumBot Control")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 30)
        self.small_font = pygame.font.Font(None, 24)

        # Initialize secondary camera
        if self.show_hdmi:
            self.init_secondary_camera()

        # Create and start threads
        print("\nStarting threads...")

        self.eye_thread = EyeAnimationThread(self.shared, self.servo)
        self.eye_thread.start()
        print("  [OK] Thread 1: Eye Animation (SPI Display)")

        time.sleep(0.5)

        self.yolo_thread = YoloDetectionThread(self.shared, self.yolo_confidence)
        self.yolo_thread.start()
        print("  [OK] Thread 2: YOLO Detection (IMX500)")

        print("\nAll threads running!")
        print("\nKeyboard Controls:")
        print("  ESC   - Exit")
        print("  SPACE - Random mood")
        print("  D     - DETECT mode")
        print("  T     - TRACK mode")
        print("  1-5   - Set mood (1=Happy, 2=Angry, 3=Tired, 4=Scared, 5=Default)")
        print()

        # Main loop (HDMI display + keyboard)
        try:
            while self.shared.running:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.shared.running = False

                    if event.type == pygame.KEYDOWN:
                        self.handle_key(event.key)

                # Draw HDMI view
                if self.show_hdmi:
                    self.draw_hdmi_view()

                pygame.display.flip()
                self.clock.tick(30)

        except KeyboardInterrupt:
            print("\nCtrl+C received")

        self.cleanup()

    def handle_key(self, key):
        """Handle keyboard input"""
        if key == pygame.K_ESCAPE:
            print("[Main] ESC pressed - exiting")
            self.shared.running = False

        elif key == pygame.K_SPACE:
            new_mood = random.choice([HAPPY, ANGRY, TIRED, SCARY, DEFAULT])
            self.shared.mood = new_mood
            mood_names = {HAPPY: "HAPPY", ANGRY: "ANGRY", TIRED: "TIRED", SCARY: "SCARY", DEFAULT: "DEFAULT"}
            print(f"[Main] Mood: {mood_names.get(new_mood, 'UNKNOWN')}")

        elif key == pygame.K_d:
            if self.yolo_thread and YoloMode:
                self.yolo_thread.set_mode(YoloMode.DETECT)

        elif key == pygame.K_t:
            if self.yolo_thread and YoloMode:
                self.yolo_thread.set_mode(YoloMode.TRACK)

        elif key == pygame.K_1:
            self.shared.mood = HAPPY
            print("[Main] Mood: HAPPY")
        elif key == pygame.K_2:
            self.shared.mood = ANGRY
            print("[Main] Mood: ANGRY")
        elif key == pygame.K_3:
            self.shared.mood = TIRED
            print("[Main] Mood: TIRED")
        elif key == pygame.K_4:
            self.shared.mood = SCARY
            print("[Main] Mood: SCARY")
        elif key == pygame.K_5:
            self.shared.mood = DEFAULT
            print("[Main] Mood: DEFAULT")

    def draw_hdmi_view(self):
        """Draw dual camera view on HDMI"""
        self.screen.fill((20, 20, 20))

        # Left Panel: Secondary Camera (IMX708)
        if self.secondary_cam:
            try:
                frame = self.secondary_cam.capture_array()
                frame = cv2.resize(frame, (640, 480))
                # RGB format from Picamera2
                surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                self.screen.blit(surf, (0, 0))
            except Exception as e:
                self.draw_placeholder(0, f"Camera Error: {e}")
        else:
            self.draw_placeholder(0, "No Secondary Camera")

        # Right Panel: YOLO Frame (IMX500)
        yolo_frame = self.shared.get_yolo_frame()
        if yolo_frame is not None:
            try:
                frame = cv2.resize(yolo_frame, (640, 480))
                # Convert BGR to RGB if needed
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                self.screen.blit(surf, (640, 0))
            except Exception as e:
                self.draw_placeholder(640, f"YOLO Error: {e}")
        else:
            self.draw_placeholder(640, "Waiting for YOLO...")

        # Draw Labels
        self.screen.blit(self.font.render("IMX708 (Secondary)", True, (0, 255, 0)), (10, 10))
        self.screen.blit(self.font.render("IMX500 (YOLO)", True, (0, 255, 0)), (650, 10))

        # Draw detection info
        labels, fps = self.shared.get_detections()
        if labels:
            info_text = f"Detected: {', '.join(labels)}"
            self.screen.blit(self.font.render(info_text, True, (255, 255, 0)), (650, 40))

        fps_text = f"YOLO FPS: {fps}"
        self.screen.blit(self.small_font.render(fps_text, True, (200, 200, 200)), (650, 460))

        # Draw mode info
        if self.yolo_thread and self.yolo_thread.yolo_tracker:
            mode_text = self.yolo_thread.yolo_tracker.get_mode_text()
            self.screen.blit(self.font.render(f"Mode: {mode_text}", True, (0, 255, 255)), (10, 460))

        # Draw controls hint
        hint = "Keys: ESC=Exit  SPACE=Mood  D=Detect  T=Track  1-5=Moods"
        self.screen.blit(self.small_font.render(hint, True, (150, 150, 150)), (300, 460))

    def draw_placeholder(self, x_offset, text):
        """Draw placeholder for missing camera"""
        pygame.draw.rect(self.screen, (30, 30, 30), (x_offset, 0, 640, 480))
        text_surf = self.font.render(text, True, (100, 100, 100))
        text_rect = text_surf.get_rect(center=(x_offset + 320, 240))
        self.screen.blit(text_surf, text_rect)

    def cleanup(self):
        print("\nCleaning up...")

        # Signal threads to stop
        self.shared.running = False

        # Wait for threads
        if self.eye_thread and self.eye_thread.is_alive():
            self.eye_thread.join(timeout=3)
        if self.yolo_thread and self.yolo_thread.is_alive():
            self.yolo_thread.join(timeout=3)

        # Cleanup secondary camera
        if self.secondary_cam:
            try:
                self.secondary_cam.stop()
                self.secondary_cam.close()
                print("[Main] Secondary camera closed")
            except:
                pass

        # Cleanup servo
        if self.servo:
            self.servo.cleanup()

        pygame.quit()
        print("NumBot stopped.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="NumBot - Multi-Threaded Robot Eye Tracker")
    parser.add_argument('--no-hdmi', action='store_true', help='Disable HDMI camera view')
    parser.add_argument('--confidence', type=float, default=0.5, help='YOLO confidence threshold')
    args = parser.parse_args()

    app = NumBotApp(
        show_hdmi=not args.no_hdmi,
        yolo_confidence=args.confidence
    )
    app.run()


if __name__ == "__main__":
    main()
