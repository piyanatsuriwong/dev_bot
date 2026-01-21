#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hand-Eye Tracker for GC9A01A Round LCD Display
Raspberry Pi 5 with 240x240 round display

Supports:
- GC9A01A SPI display (real hardware)
- Pygame simulator (for development)
- Hand tracking with MediaPipe or OpenCV
"""

import pygame
import cv2
import numpy as np
import sys
import random
import time
import config
from face_renderer import FaceRenderer

# Import display driver
if config.DISPLAY_MODE == "gc9a01a":
    from gc9a01a_display import create_display, RPI_AVAILABLE
else:
    RPI_AVAILABLE = False

# Try import MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    print("MediaPipe: Available")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe: Not available - using OpenCV fallback")


class HandTrackerMediaPipe:
    """Hand tracking with MediaPipe"""

    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.hand_x = 0.5
        self.hand_y = 0.5
        self.hand_detected = False
        self.latest_frame = None
        self.gesture = "unknown"
        self.gesture_changed = False

    def update(self):
        ret, frame = self.cap.read()
        if not ret:
            return False

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        self.hand_detected = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                palm = hand_landmarks.landmark[9]
                self.hand_x = palm.x
                self.hand_y = palm.y
                self.hand_detected = True

        self.latest_frame = frame
        return True

    def get_normalized_position(self):
        if not self.hand_detected:
            return 0, 0
        norm_x = (self.hand_x - 0.5) * 2
        norm_y = (self.hand_y - 0.5) * 2
        return norm_x, norm_y

    def release(self):
        self.cap.release()
        self.hands.close()


class HandTrackerOpenCV:
    """Hand tracking with OpenCV skin detection"""

    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.hand_x = 0.5
        self.hand_y = 0.5
        self.hand_detected = False
        self.latest_frame = None

        self.finger_count = 0
        self.gesture = "unknown"
        self.gesture_changed = False
        self.last_gesture = "unknown"

        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    def _count_fingers(self, contour):
        try:
            hull = cv2.convexHull(contour, returnPoints=False)
            if len(hull) < 3:
                return 0

            defects = cv2.convexityDefects(contour, hull)
            if defects is None:
                return 0

            finger_count = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])

                a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

                if b * c == 0:
                    continue
                angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))

                if angle <= np.pi / 2 and d > 10000:
                    finger_count += 1

            return min(finger_count + 1, 5) if finger_count > 0 else 0

        except Exception:
            return 0

    def update(self):
        ret, frame = self.cap.read()
        if not ret:
            return False

        frame = cv2.flip(frame, 1)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        self.hand_detected = False
        self.gesture_changed = False

        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(max_contour)

            if area > 5000:
                M = cv2.moments(max_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    h, w = frame.shape[:2]
                    self.hand_x = cx / w
                    self.hand_y = cy / h
                    self.hand_detected = True

                    self.finger_count = self._count_fingers(max_contour)

                    old_gesture = self.gesture
                    if self.finger_count == 0 or self.finger_count == 1:
                        self.gesture = "fist"
                    elif self.finger_count == 2:
                        self.gesture = "two"
                    elif self.finger_count == 3:
                        self.gesture = "three"
                    elif self.finger_count == 4:
                        self.gesture = "four"
                    elif self.finger_count >= 5:
                        self.gesture = "open"
                    else:
                        self.gesture = "unknown"

                    if self.gesture != old_gesture and self.gesture != "unknown":
                        self.gesture_changed = True
                        self.last_gesture = self.gesture

        self.latest_frame = frame
        return True

    def get_normalized_position(self):
        if not self.hand_detected:
            return 0, 0
        norm_x = (self.hand_x - 0.5) * 2
        norm_y = (self.hand_y - 0.5) * 2
        return norm_x, norm_y

    def release(self):
        self.cap.release()


class GC9A01AHandEyeApp:
    """Main application for GC9A01A round display"""

    def __init__(self, use_mediapipe=True, camera_id=0, headless=False, show_camera=True):
        self.headless = headless
        self.show_camera = show_camera

        # Initialize Pygame
        pygame.init()

        if config.DISPLAY_MODE == "gc9a01a":
            # Create GC9A01A display (or simulator)
            self.gc9a01a = create_display()
            # Create off-screen surface for rendering eyes
            self.screen = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
            print(f"Eye Display: GC9A01A {config.SCREEN_WIDTH}x{config.SCREEN_HEIGHT}")

            # Create main window for camera view
            if show_camera:
                self.camera_screen = pygame.display.set_mode((640, 480))
                pygame.display.set_caption("Hand Tracking - Camera View")
                print("Camera Display: 640x480 on HDMI")
            else:
                self.camera_screen = None
        else:
            # Normal Pygame display
            self.screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
            pygame.display.set_caption(config.CAPTION)
            self.gc9a01a = None
            self.camera_screen = None
            print(f"Display: Pygame {config.SCREEN_WIDTH}x{config.SCREEN_HEIGHT}")

        self.clock = pygame.time.Clock()

        # Create Face Renderer
        self.face_renderer = FaceRenderer(self.screen)
        self.face_renderer.external_control = True

        # Select Hand Tracker
        if use_mediapipe and MEDIAPIPE_AVAILABLE:
            self.hand_tracker = HandTrackerMediaPipe(camera_id)
            print("Tracking: MediaPipe")
        else:
            self.hand_tracker = HandTrackerOpenCV(camera_id)
            print("Tracking: OpenCV")

        self.smoothing = config.GAZE_SMOOTHING
        self.current_look_x = 0
        self.current_look_y = 0
        self.running = True

        # Load default emotion
        if "normal" in self.face_renderer.emotions_data:
            self.face_renderer.set_emotion("normal")

    def run(self):
        print("=" * 40)
        print("  GC9A01A Hand-Eye Tracker")
        print("=" * 40)
        print("Controls:")
        print("  ESC = Quit")
        print("  SPACE = Random emotion")
        print("=" * 40)

        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_SPACE:
                        emotions = list(self.face_renderer.emotions_data.keys())
                        if emotions:
                            emo = random.choice(emotions)
                            self.face_renderer.set_emotion(emo)
                            print(f"Emotion: {emo}")

            # Update hand tracking
            self.hand_tracker.update()
            hand_x, hand_y = self.hand_tracker.get_normalized_position()

            # Handle gesture-based emotion changes
            if hasattr(self.hand_tracker, 'gesture_changed') and self.hand_tracker.gesture_changed:
                gesture = self.hand_tracker.gesture
                emotion_map = {
                    "fist": "angry",
                    "two": "normal",
                    "three": "sad",
                    "four": "scared",
                    "open": "happy",
                }
                if gesture in emotion_map:
                    emotion = emotion_map[gesture]
                    self.face_renderer.set_emotion(emotion)
                    print(f"Gesture: {gesture} -> Emotion: {emotion}")

            # Update gaze position
            if self.hand_tracker.hand_detected:
                self.current_look_x += (hand_x - self.current_look_x) * self.smoothing
                self.current_look_y += (hand_y - self.current_look_y) * self.smoothing

                look_x = -self.current_look_x * config.GAZE_RANGE
                look_y = self.current_look_y * config.GAZE_RANGE

                self.face_renderer.target_look_x = max(-1.0, min(1.0, look_x))
                self.face_renderer.target_look_y = max(-1.0, min(1.0, look_y))
                self.face_renderer.is_tracking = True
            else:
                self.current_look_x *= 0.95
                self.current_look_y *= 0.95
                self.face_renderer.target_look_x = self.current_look_x
                self.face_renderer.target_look_y = self.current_look_y

            # Render eye
            self.face_renderer.draw()

            # Send to GC9A01A display
            if self.gc9a01a:
                self.gc9a01a.draw_from_surface(self.screen)

            # Draw camera view on HDMI screen
            if self.camera_screen and self.hand_tracker.latest_frame is not None:
                frame = self.hand_tracker.latest_frame
                # Convert BGR to RGB and create pygame surface
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                self.camera_screen.blit(frame_surface, (0, 0))

                # Draw tracking info
                font = pygame.font.Font(None, 36)
                if self.hand_tracker.hand_detected:
                    status_text = f"TRACKING: X={hand_x:.2f} Y={hand_y:.2f}"
                    color = (0, 255, 0)
                else:
                    status_text = "NO HAND DETECTED"
                    color = (255, 0, 0)
                text_surface = font.render(status_text, True, color)
                self.camera_screen.blit(text_surface, (10, 10))

                # Show current emotion
                if hasattr(self.hand_tracker, 'gesture'):
                    gesture_text = f"Gesture: {self.hand_tracker.gesture}"
                    gesture_surface = font.render(gesture_text, True, (255, 255, 0))
                    self.camera_screen.blit(gesture_surface, (10, 50))

                pygame.display.flip()
            elif not self.gc9a01a:
                pygame.display.flip()

            self.clock.tick(config.FPS)

        self.cleanup()

    def cleanup(self):
        self.hand_tracker.release()
        if self.gc9a01a:
            self.gc9a01a.cleanup()
        pygame.quit()


class DemoMode:
    """Demo mode without camera - auto animation"""

    def __init__(self):
        pygame.init()

        if config.DISPLAY_MODE == "gc9a01a":
            self.gc9a01a = create_display()
            self.screen = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        else:
            self.screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
            pygame.display.set_caption("GC9A01A Demo Mode")
            self.gc9a01a = None

        self.clock = pygame.time.Clock()
        self.face_renderer = FaceRenderer(self.screen)
        self.running = True

        self.demo_time = 0
        self.last_emotion_change = 0
        self.emotion_interval = 3.0

    def run(self):
        print("=" * 40)
        print("  GC9A01A Demo Mode (No Camera)")
        print("=" * 40)

        emotions = list(self.face_renderer.emotions_data.keys())
        if emotions:
            self.face_renderer.set_emotion(emotions[0])

        while self.running:
            current_time = time.time()
            self.demo_time = current_time

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False

            # Auto change emotion
            if current_time - self.last_emotion_change > self.emotion_interval:
                if emotions:
                    emo = random.choice(emotions)
                    self.face_renderer.set_emotion(emo)
                    print(f"Demo emotion: {emo}")
                self.last_emotion_change = current_time

            # Auto look around
            look_x = np.sin(current_time * 0.5) * 0.5
            look_y = np.cos(current_time * 0.7) * 0.3
            self.face_renderer.target_look_x = look_x
            self.face_renderer.target_look_y = look_y

            # Render
            self.face_renderer.draw()

            if self.gc9a01a:
                self.gc9a01a.draw_from_surface(self.screen)
            else:
                pygame.display.flip()

            self.clock.tick(config.FPS)

        self.cleanup()

    def cleanup(self):
        if self.gc9a01a:
            self.gc9a01a.cleanup()
        pygame.quit()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='GC9A01A Hand-Eye Tracker')
    parser.add_argument('--demo', action='store_true',
                        help='Run in demo mode (no camera)')
    parser.add_argument('--no-mediapipe', action='store_true',
                        help='Use OpenCV instead of MediaPipe')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('--no-camera-view', action='store_true',
                        help='Hide camera view window on HDMI')
    args = parser.parse_args()

    if args.demo:
        app = DemoMode()
    else:
        app = GC9A01AHandEyeApp(
            use_mediapipe=not args.no_mediapipe,
            camera_id=args.camera,
            show_camera=not args.no_camera_view
        )

    try:
        app.run()
    except KeyboardInterrupt:
        print("\nStopped by user")
        app.cleanup()


if __name__ == "__main__":
    main()
