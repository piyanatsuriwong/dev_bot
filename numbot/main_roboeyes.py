#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NumBot - Robot Eye Tracker with RoboEyes
Raspberry Pi 5 with ST7735S / GC9A01A Display

Architecture:
- Main Thread: HDMI Display + Keyboard Events
- Thread 1: Eye Animation (ST7735S SPI Display)
- Thread 2: Hand Tracking (Regular Camera with MediaPipe/OpenCV)
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

# Try to import MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available, will use OpenCV fallback")

# Try to import Picamera2
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("Warning: Picamera2 not available, will use OpenCV")

# Import display driver based on config
if config.DISPLAY_MODE == "gc9a01a":
    from gc9a01a_display import create_display, RPI_AVAILABLE
elif config.DISPLAY_MODE == "st7735s":
    from st7735_display import create_display, RPI_AVAILABLE
else:
    RPI_AVAILABLE = False


class SharedState:
    """Thread-safe shared state between threads"""

    def __init__(self):
        self._lock = threading.Lock()
        self._target_x = 0.0
        self._target_y = 0.0
        self._has_target = False
        self._mood = DEFAULT
        self._tracking_fps = 0
        self._camera_frame = None
        self._running = True
        self._tracking_info = ""

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

    def set_tracking_data(self, fps, frame=None, info=""):
        with self._lock:
            self._tracking_fps = fps
            self._tracking_info = info
            if frame is not None:
                self._camera_frame = frame

    def get_tracking_data(self):
        with self._lock:
            return self._tracking_fps, self._tracking_info

    def get_camera_frame(self):
        with self._lock:
            return self._camera_frame


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


class HandTrackingThread(threading.Thread):
    """
    Thread 2: Hand Tracking with Camera 0 (IMX708) only
    """

    def __init__(self, shared_state, use_mediapipe=True):
        super().__init__(name="HandTrackingThread", daemon=True)
        self.shared = shared_state
        self.camera_index = 0  # LOCKED to Camera 0 (IMX708)
        self.use_mediapipe = use_mediapipe and MEDIAPIPE_AVAILABLE
        self.use_picamera2 = PICAMERA2_AVAILABLE
        self.cap = None
        self.picam2 = None
        self.mp_hands = None
        self.hands = None

    def run(self):
        print("[Thread 2] Hand Tracking: Starting...")

        # Initialize camera
        if not self._init_camera():
            return

        # Initialize MediaPipe if available
        if self.use_mediapipe:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("[Thread 2] Using MediaPipe hand tracking")
        else:
            print("[Thread 2] Using OpenCV hand tracking")

        print("[Thread 2] Hand Tracking: Running...")

        frame_count = 0
        last_fps_time = time.time()
        last_log_time = time.time()
        fps = 0

        while self.shared.running:
            # Read frame from camera
            if self.use_picamera2:
                frame = self.picam2.capture_array()
                # Picamera2 returns RGB, convert to BGR for OpenCV/MediaPipe
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                ret, frame = self.cap.read()
                if not ret:
                    print("[Thread 2] Failed to read frame")
                    time.sleep(0.1)
                    continue

            frame = cv2.flip(frame, 1)

            # Track hand
            if self.use_mediapipe:
                tx, ty, has_hand = self._track_mediapipe(frame)
            else:
                tx, ty, has_hand = self._track_opencv(frame)

            # Update shared state
            if has_hand:
                self.shared.set_target(tx, ty, has_target=True)
            else:
                self.shared.set_target(0, 0, has_target=False)

            # Calculate FPS
            frame_count += 1
            now = time.time()
            if now - last_fps_time >= 1.0:
                fps = frame_count / (now - last_fps_time)
                frame_count = 0
                last_fps_time = now

            # Update tracking data
            info = "Hand detected" if has_hand else "No hand"
            self.shared.set_tracking_data(fps, frame, info)

            # Log every 5 seconds
            if now - last_log_time >= 5.0:
                print(f"[Thread 2] IMX708: {info} | FPS: {fps:.1f}")
                last_log_time = now

        # Cleanup
        print("[Thread 2] Hand Tracking: Stopping...")
        if self.hands:
            self.hands.close()
        if self.use_picamera2 and self.picam2:
            self.picam2.stop()
            self.picam2.close()
        if self.cap:
            self.cap.release()
        print("[Thread 2] Hand Tracking: Stopped")

    def _init_camera(self):
        """Initialize Camera 0 (IMX708) only"""
        if self.use_picamera2:
            try:
                self.picam2 = Picamera2(0)
                config = self.picam2.create_preview_configuration(
                    main={"size": (640, 480), "format": "RGB888"}
                )
                self.picam2.configure(config)
                self.picam2.start()
                print("[Thread 2] Camera 0 (IMX708) opened with Picamera2")
                return True
            except Exception as e:
                print(f"[Thread 2] Picamera2 failed: {e}")
                self.use_picamera2 = False
                # Fall through to OpenCV
        
        # Fallback to OpenCV
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("[Thread 2] Failed to open Camera 0 (IMX708)")
            return False

        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("[Thread 2] Camera 0 (IMX708) opened with OpenCV")
        return True

    def _track_mediapipe(self, frame):
        """Track hand using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            # Use index finger tip (landmark 8)
            index_tip = hand_landmarks.landmark[8]
            
            # Convert to normalized coordinates (-1 to 1)
            # MediaPipe: x=0 (left), x=1 (right)
            # Our system: x=-1 (left), x=1 (right)
            tx = (index_tip.x - 0.5) * 2
            ty = (index_tip.y - 0.5) * 2
            
            # Draw landmarks on frame
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )
            
            return tx, ty, True
        
        return 0, 0, False

    def _track_opencv(self, frame):
        """Track hand using OpenCV (simple skin detection)"""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define skin color range (adjust as needed)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour
            max_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(max_contour)
            
            if area > 1000:  # Minimum area threshold
                # Get center of contour
                M = cv2.moments(max_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Draw contour and center
                    cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)
                    
                    # Convert to normalized coordinates
                    h, w = frame.shape[:2]
                    tx = (cx / w - 0.5) * 2
                    ty = (cy / h - 0.5) * 2
                    
                    return tx, ty, True
        
        return 0, 0, False


class NumBotApp:
    """
    Main Application
    - Main thread handles HDMI display and keyboard
    - Thread 1: Eye Animation (SPI)
    - Thread 2: Hand Tracking (Camera 0: IMX708 only)
    """

    def __init__(self, show_hdmi=True, use_mediapipe=True):
        self.show_hdmi = show_hdmi
        self.use_mediapipe = use_mediapipe

        # Shared state
        self.shared = SharedState()

        # Servo controller
        self.servo = ServoController()

        # Threads
        self.eye_thread = None
        self.tracking_thread = None

        # Pygame (HDMI)
        self.screen = None
        self.clock = None
        self.font = None

    def run(self):
        print("=" * 50)
        print("NumBot - Robot Eye Tracker with Hand Tracking")
        print("=" * 50)

        # Initialize Pygame only if HDMI display is needed
        if self.show_hdmi:
            pygame.init()
            self.screen = pygame.display.set_mode((640, 480))
            pygame.display.set_caption("NumBot - Hand Tracking")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 30)
            self.small_font = pygame.font.Font(None, 24)

        # Create and start threads
        print("\nStarting threads...")

        self.eye_thread = EyeAnimationThread(self.shared, self.servo)
        self.eye_thread.start()
        print("  [OK] Thread 1: Eye Animation (SPI Display)")

        time.sleep(0.5)

        self.tracking_thread = HandTrackingThread(
            self.shared,
            use_mediapipe=self.use_mediapipe
        )
        self.tracking_thread.start()
        print("  [OK] Thread 2: Hand Tracking (Camera 0: IMX708)")

        print("\nAll threads running!")
        print("\nKeyboard Controls:")
        print("  ESC   - Exit")
        print("  SPACE - Random mood")
        print("  1-5   - Set mood (1=Happy, 2=Angry, 3=Tired, 4=Scared, 5=Default)")
        print()

        # Main loop (HDMI display + keyboard)
        try:
            if self.show_hdmi:
                # HDMI mode with pygame display
                while self.shared.running:
                    # Handle events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.shared.running = False

                        if event.type == pygame.KEYDOWN:
                            self.handle_key(event.key)

                    # Draw HDMI view
                    self.draw_hdmi_view()
                    pygame.display.flip()
                    self.clock.tick(30)
            else:
                # Headless mode - just wait for threads
                while self.shared.running:
                    time.sleep(0.1)

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
        """Draw camera view with hand tracking on HDMI"""
        self.screen.fill((20, 20, 20))

        # Get camera frame from tracking thread
        camera_frame = self.shared.get_camera_frame()
        
        if camera_frame is not None:
            try:
                frame = cv2.resize(camera_frame, (640, 480))
                # Convert BGR to RGB
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                self.screen.blit(surf, (0, 0))
            except Exception as e:
                self.draw_placeholder(f"Camera Error: {e}")
        else:
            self.draw_placeholder("Waiting for camera...")

        # Draw title
        tracking_type = "MediaPipe" if self.use_mediapipe else "OpenCV"
        title = f"IMX708 Hand Tracking ({tracking_type})"
        self.screen.blit(self.font.render(title, True, (0, 255, 0)), (10, 10))

        # Draw tracking info
        fps, info = self.shared.get_tracking_data()
        info_text = f"{info} | FPS: {fps:.1f}"
        self.screen.blit(self.font.render(info_text, True, (255, 255, 0)), (10, 40))

        # Draw mood
        mood_names = {
            HAPPY: "HAPPY", 
            ANGRY: "ANGRY", 
            TIRED: "TIRED", 
            SCARY: "SCARY", 
            DEFAULT: "DEFAULT"
        }
        mood_text = f"Mood: {mood_names.get(self.shared.mood, 'UNKNOWN')}"
        self.screen.blit(self.font.render(mood_text, True, (0, 255, 255)), (10, 440))

        # Draw controls hint
        hint = "Keys: ESC=Exit  SPACE=Random Mood  1-5=Set Mood"
        self.screen.blit(self.small_font.render(hint, True, (150, 150, 150)), (150, 460))

    def draw_placeholder(self, text):
        """Draw placeholder for missing camera"""
        pygame.draw.rect(self.screen, (30, 30, 30), (0, 0, 640, 480))
        text_surf = self.font.render(text, True, (100, 100, 100))
        text_rect = text_surf.get_rect(center=(320, 240))
        self.screen.blit(text_surf, text_rect)

    def cleanup(self):
        print("\nCleaning up...")

        # Signal threads to stop
        self.shared.running = False

        # Wait for threads
        if self.eye_thread and self.eye_thread.is_alive():
            self.eye_thread.join(timeout=3)
        if self.tracking_thread and self.tracking_thread.is_alive():
            self.tracking_thread.join(timeout=3)

        # Cleanup servo
        if self.servo:
            self.servo.cleanup()

        # Cleanup pygame if initialized
        if self.show_hdmi:
            pygame.quit()
        
        print("NumBot stopped.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="NumBot - Robot Eye Tracker (Camera 0: IMX708 only)")
    parser.add_argument('--no-hdmi', action='store_true', help='Disable HDMI camera view')
    parser.add_argument('--no-mediapipe', action='store_true', help='Use OpenCV instead of MediaPipe')
    args = parser.parse_args()

    print("=" * 60)
    print("NumBot - Locked to Camera 0 (IMX708)")
    print("=" * 60)

    app = NumBotApp(
        show_hdmi=not args.no_hdmi,
        use_mediapipe=not args.no_mediapipe
    )
    app.run()


if __name__ == "__main__":
    main()
