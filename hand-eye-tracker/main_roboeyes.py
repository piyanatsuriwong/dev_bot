#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hand-Eye Tracker with RoboEyes for GC9A01A Round LCD Display
Raspberry Pi 5 with 240x240 round display

Uses the RoboEyes library for animated robot eyes
Supports YOLO object detection with IMX500 AI Camera
"""

# -*- coding: utf-8 -*-
import pygame
import cv2
import numpy as np
import sys
import random
import time
import threading
import subprocess
import os
import shutil
from PIL import Image, ImageDraw, ImageFont

# Monkey patch shutil.which to force SpeechRecognition to use WAV
# This prevents "FLAC conversion utility not available" error
original_which = shutil.which
def mock_which(cmd, mode=os.F_OK | os.X_OK, path=None):
    # Aggressively block flac
    if "flac" in str(cmd).lower():
        print(f"DEBUG: Blocked access to FLAC ({cmd})")
        return None
    return original_which(cmd, mode, path)
shutil.which = mock_which

import config
from roboeyes import *
from pi_camera import PiCamera
from servo_controller import ServoController

try:
    import speech_recognition as sr
    # Extra safety: Force FLAC converter to be None within the library if possible
    # (Though patching shutil.which usually suffices)
    
    VOICE_AVAILABLE = True
    print("Voice Control: Available")
except ImportError:
    VOICE_AVAILABLE = False
    print("Voice Control: Not available (install SpeechRecognition)")

SOUND_ENABLED = True  # Set to True to enable robot voice

# Wall-E style sound files path
SOUND_PATH = "/home/piyanat/hand-eye-tracker/assets/sounds/"

# Mood-specific Wall-E sounds (empty = no sound)
MOOD_SOUNDS = {
    "HAPPY": "Voicy_WALL-E 4.mp3",
    "ANGRY": "",
    "TIRED": "",
    "SCARY": "",
    "FROZEN": "",
    "CURIOUS": "",
    "DEFAULT": ""
}

def play_robot_sound(mood=None, speed=1.0):
    """Play Wall-E style sound based on mood"""
    if not SOUND_ENABLED:
        return
    def _play():
        try:
            # Get sound file for mood
            sound_file = MOOD_SOUNDS.get(mood, "")
            if not sound_file:
                return  # No sound for this mood
            full_path = SOUND_PATH + sound_file

            if os.path.exists(full_path):
                subprocess.call(f'ffplay -nodisp -autoexit -loglevel quiet "{full_path}"', shell=True)
        except Exception as e:
            print(f"Sound Error: {e}")
    # Run in background thread to not block
    threading.Thread(target=_play, daemon=True).start()

if VOICE_AVAILABLE:
    class LinuxMicrophone(sr.AudioSource):
        def __init__(self, device_index=None, sample_rate=16000, chunk_size=1024):
            self.device_index = device_index
            self.SAMPLE_RATE = sample_rate
            self.CHUNK = chunk_size
            self.WIDTH = 2
        
        def __enter__(self):
            return self
            
        def __exit__(self, exc_type, exc_value, traceback):
            pass

    class VoiceThread(threading.Thread):
        def __init__(self, callback):
            threading.Thread.__init__(self)
            self.callback = callback
            self.running = True
            self.recognizer = sr.Recognizer()

        def run(self):
            while self.running:
                try:
                    print("Voice: Recording via arecord...")
                    # Record 3 seconds of audio to a WAV file using system 'arecord'
                    # Using Mono (c 1) 16kHz (r 16000) S16_LE
                    filename = "/tmp/voice_cmd.wav"
                    # -D plughw:2,0 = Webcam
                    cmd = f"arecord -D plughw:2,0 -d 3 -f S16_LE -c 1 -r 16000 -t wav -q {filename}"
                    subprocess.call(cmd, shell=True)
                    
                    if os.path.exists(filename) and os.path.getsize(filename) > 0:
                        with sr.AudioFile(filename) as source:
                            audio = self.recognizer.record(source)
                            try:
                                # Use Thai language (th-TH)
                                text = self.recognizer.recognize_google(audio, language="th-TH")
                                print("=" * 50)
                                print(f"  ได้ยิน: '{text}'")
                                print("=" * 50)
                                self.callback(text)
                            except sr.UnknownValueError:
                                print("  (ไม่ได้ยินเสียงพูด)")
                            except sr.RequestError as e:
                                print(f"API Error: {e}")
                            except Exception as e:
                                # Catch flac error specifically if it still leaks
                                if "flac" in str(e).lower():
                                    print("Voice Warn: FLAC error still occurred (ignoring)")
                                else:
                                    print(f"Voice Error: {e}")
                    else:
                         print("Voice Warn: Recording failed or empty")
                except Exception as e:
                    print(f"Voice Thread Error: {e}")
                    time.sleep(1)

        def stop(self):
            self.running = False


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

# Import YOLO tracker
try:
    from yolo_tracker import create_yolo_tracker, YOLO_AVAILABLE, YoloMode
    print(f"YOLO Tracker: {'Available' if YOLO_AVAILABLE else 'Not available'}")
except ImportError:
    YOLO_AVAILABLE = False
    YoloMode = None
    print("YOLO Tracker: Not available")


class TextOverlay:
    """Helper class to render text overlay on GC9A01A display using pygame"""

    def __init__(self, width=240, height=240):
        self.width = width
        self.height = height
        self.font_size = 16
        self.font = None
        self._init_font()

    def _init_font(self):
        """Initialize pygame font"""
        pygame.font.init()
        # Use default pygame font (fast and reliable)
        self.font = pygame.font.Font(None, self.font_size)

    def create_text_surface(self, texts, bg_color=(0, 0, 0, 180), text_color=(0, 255, 255)):
        """
        Create a pygame surface with text overlay

        Args:
            texts: List of strings to display
            bg_color: Background color (RGBA) - alpha used for transparency effect
            text_color: Text color (RGB)

        Returns:
            pygame.Surface with text
        """
        if not texts:
            return None

        # Create transparent surface
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        # Calculate text area at bottom of screen
        padding = 6
        line_height = self.font_size + 2
        total_height = len(texts) * line_height + padding * 2

        # Draw semi-transparent background at bottom
        y_start = self.height - total_height
        bg_rect = pygame.Rect(0, y_start, self.width, total_height)
        pygame.draw.rect(surface, bg_color, bg_rect)

        # Draw each line of text
        y = y_start + padding
        for text in texts:
            text_surface = self.font.render(text, True, text_color)
            text_rect = text_surface.get_rect(centerx=self.width // 2, top=y)
            surface.blit(text_surface, text_rect)
            y += line_height

        return surface

    def blend_with_eyes(self, eye_surface, text_surface):
        """
        Blend text overlay with eye surface

        Args:
            eye_surface: pygame.Surface with robot eyes
            text_surface: pygame.Surface with text overlay

        Returns:
            Combined pygame.Surface
        """
        if text_surface is None:
            return eye_surface

        # Create a copy
        result = eye_surface.copy()

        # Blit text surface with alpha
        result.blit(text_surface, (0, 0))

        return result


class HandTrackerMediaPipe:
    """Hand tracking with MediaPipe - Enhanced with finger counting"""

    # MediaPipe hand landmark indices
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20

    # Finger tip IDs
    TIP_IDS = [4, 8, 12, 16, 20]

    def __init__(self, camera_id=0):
        self.cap = PiCamera(camera_id, width=640, height=480, fps=30)

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        self.hand_x = 0.5
        self.hand_y = 0.5
        self.hand_detected = False
        self.latest_frame = None
        self.gesture = "unknown"
        self.gesture_changed = False
        self.finger_count = 0
        self.landmarks_list = []

    def _get_landmarks_list(self, hand_landmarks, frame_shape):
        """Convert landmarks to pixel coordinates"""
        h, w, _ = frame_shape
        landmarks = []
        for id, lm in enumerate(hand_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            landmarks.append((id, cx, cy))
        return landmarks

    def _count_fingers(self, landmarks, handedness="Right"):
        """Count raised fingers using MediaPipe landmarks"""
        if len(landmarks) < 21:
            return 0

        fingers = []

        # Thumb - check x position (different for left/right hand)
        if handedness == "Right":
            # Right hand: thumb tip should be left of thumb IP
            if landmarks[self.THUMB_TIP][1] < landmarks[self.THUMB_TIP - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            # Left hand: thumb tip should be right of thumb IP
            if landmarks[self.THUMB_TIP][1] > landmarks[self.THUMB_TIP - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        # Other 4 fingers - check if tip is above PIP joint (y is smaller = higher)
        for tip_id in [8, 12, 16, 20]:
            if landmarks[tip_id][2] < landmarks[tip_id - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return sum(fingers)

    def _detect_gesture(self, finger_count):
        """Detect gesture from finger count"""
        if finger_count == 0:
            return "fist"
        elif finger_count == 1:
            return "one"
        elif finger_count == 2:
            return "two"
        elif finger_count == 3:
            return "three"
        elif finger_count == 4:
            return "four"
        elif finger_count == 5:
            return "open"
        return "unknown"

    def update(self):
        ret, frame = self.cap.read()
        if not ret:
            return False

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        self.hand_detected = False
        self.gesture_changed = False
        old_gesture = self.gesture

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get palm position for eye tracking
                palm = hand_landmarks.landmark[9]
                self.hand_x = palm.x
                self.hand_y = palm.y
                self.hand_detected = True

                # Get landmarks list
                self.landmarks_list = self._get_landmarks_list(hand_landmarks, frame.shape)

                # Determine handedness
                handedness = "Right"
                if results.multi_handedness:
                    handedness = results.multi_handedness[idx].classification[0].label

                # Count fingers
                self.finger_count = self._count_fingers(self.landmarks_list, handedness)

                # Detect gesture
                self.gesture = self._detect_gesture(self.finger_count)

                # Draw hand landmarks on frame
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
                )

                # Draw landmark IDs on frame
                for id, cx, cy in self.landmarks_list:
                    # Draw landmark ID number
                    cv2.putText(frame, str(id), (cx - 10, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # Draw finger tips with different colors
                tip_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
                tip_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
                for i, tip_id in enumerate(self.TIP_IDS):
                    cx, cy = self.landmarks_list[tip_id][1], self.landmarks_list[tip_id][2]
                    cv2.circle(frame, (cx, cy), 12, tip_colors[i], cv2.FILLED)
                    cv2.circle(frame, (cx, cy), 14, (255, 255, 255), 2)

                # Draw finger count and gesture info
                h, w, _ = frame.shape
                cv2.rectangle(frame, (0, 0), (250, 100), (0, 0, 0), -1)
                cv2.putText(frame, f"Fingers: {self.finger_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Gesture: {self.gesture}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"Hand: {handedness}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if self.gesture != old_gesture and self.gesture != "unknown":
            self.gesture_changed = True

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
        self.cap = PiCamera(camera_id, width=640, height=480, fps=30)

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


class RoboEyesHandTracker:
    """Main application using RoboEyes for GC9A01A round display"""

    def __init__(self, use_mediapipe=True, camera_id=0, show_camera=True, enable_tracking=True,
                 enable_yolo=False, yolo_confidence=0.5):
        self.show_camera = show_camera
        self.enable_tracking = enable_tracking
        self.enable_yolo = enable_yolo
        self.yolo_confidence = yolo_confidence

        # YOLO tracker
        self.yolo_tracker = None
        self.yolo_detections = []

        # Text overlay for GC9A01A
        self.text_overlay = None

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

        # Create RoboEyes callback
        def robo_show(roboeyes):
            if self.gc9a01a:
                self.gc9a01a.draw_from_surface(self.screen)
            else:
                pygame.display.flip()

        # Initialize RoboEyes
        # Use white color for eyes (255, 255, 255)
        self.robo = RoboEyes(
            self.screen, 
            config.SCREEN_WIDTH, 
            config.SCREEN_HEIGHT, 
            frame_rate=config.FPS,
            on_show=robo_show,
            bgcolor=0,  # Black background
            fgcolor=(255, 255, 255)  # White color
        )

        # Configure RoboEyes for round display
        if config.DISPLAY_MODE == "gc9a01a":
            # Larger eyes for 240x240 display
            # Increased size to fill the round screen better
            self.robo.eyes_width(90, 90)
            self.robo.eyes_height(90, 90)
            self.robo.eyes_spacing(20)
            self.robo.eyes_radius(20, 20)  # More rounded corners
        
        # Enable auto-blinker and idle mode
        self.robo.set_auto_blinker(ON, 3, 2)
        if not self.enable_tracking:
            self.robo.set_idle_mode(ON, 2, 2)  # Enable idle mode if not tracking
        else:
            self.robo.set_idle_mode(OFF)  # Disable idle, we'll control position manually

        # Select Hand Tracker
        if use_mediapipe and MEDIAPIPE_AVAILABLE:
            self.hand_tracker = HandTrackerMediaPipe(camera_id)
            print("Tracking: MediaPipe")
        else:
            self.hand_tracker = HandTrackerOpenCV(camera_id)
            print("Tracking: OpenCV")

        # Initialize YOLO tracker if enabled
        if self.enable_yolo:
            if YOLO_AVAILABLE:
                self.yolo_tracker = create_yolo_tracker(
                    confidence_threshold=self.yolo_confidence,
                    frame_rate=30
                )
                self.yolo_tracker.start()
                print("YOLO: Started (IMX500 AI Camera)")
            else:
                print("YOLO: Not available (modlib not installed)")

        # Initialize text overlay for GC9A01A
        if config.DISPLAY_MODE == "gc9a01a":
            self.text_overlay = TextOverlay(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
            print("Text Overlay: Enabled")

        # Initialize Servo Controller for camera pan-tilt
        self.servo = ServoController()
        if self.servo.enabled:
            print("Servo: Pan-Tilt enabled (PCA9685)")
        else:
            print("Servo: Disabled")

        # Initialize Voice Control
        self.voice_thread = None
        if VOICE_AVAILABLE:
            self.voice_thread = VoiceThread(self.on_voice_command)
            self.voice_thread.start()

        self.running = True

        # Set default mood
        self.robo.mood = DEFAULT

    def on_voice_command(self, text):
        mood_changed = True
        # Thai commands
        if "มีความสุข" in text or "ยิ้ม" in text or "แฮปปี้" in text or "happy" in text or "smile" in text:
            self.robo.mood = HAPPY
            mood_name = "HAPPY (มีความสุข)"
        elif "โกรธ" in text or "โมโห" in text or "angry" in text or "mad" in text:
            self.robo.mood = ANGRY
            mood_name = "ANGRY (โกรธ)"
        elif "เหนื่อย" in text or "ง่วง" in text or "นอน" in text or "tired" in text or "sleep" in text:
            self.robo.mood = TIRED
            mood_name = "TIRED (เหนื่อย)"
        elif "กลัว" in text or "หลอน" in text or "scary" in text or "fear" in text:
            self.robo.mood = SCARY
            mood_name = "SCARY (กลัว)"
        elif "หนาว" in text or "เย็น" in text or "frozen" in text or "cold" in text:
            self.robo.mood = FROZEN
            mood_name = "FROZEN (หนาว)"
        elif "สงสัย" in text or "อะไร" in text or "curious" in text or "what" in text:
            self.robo.mood = CURIOUS
            mood_name = "CURIOUS (สงสัย)"
        elif "ปกติ" in text or "รีเซ็ต" in text or "normal" in text or "reset" in text:
            self.robo.mood = DEFAULT
            mood_name = "DEFAULT (ปกติ)"
        else:
            mood_changed = False
            mood_name = "ไม่เปลี่ยน"

        if mood_changed:
            print(f"  >> เปลี่ยนอารมณ์เป็น: {mood_name}")
            # Play robot sound
            sound_key = mood_name.split()[0]  # Get "HAPPY", "ANGRY", etc.
            if sound_key in MOOD_SOUNDS:
                play_robot_sound(sound_key)
        else:
            print(f"  >> ไม่พบคำสั่ง (พูดว่า: มีความสุข/โกรธ/เหนื่อย/กลัว/หนาว/สงสัย/ปกติ)")

    def run(self):

        print("=" * 40)
        print("  RoboEyes Hand-Eye Tracker")
        if self.enable_yolo:
            print("  + YOLO Object Detection")
        print("=" * 40)
        print("Controls:")
        print("  ESC = Quit")
        print("  SPACE = Random emotion")
        print("=" * 40)

        # Track detected objects for display
        detected_objects = []
        yolo_update_interval = 0.1  # Update YOLO display every 100ms
        last_yolo_update = 0

        while self.running:
            current_time = time.time()

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_SPACE:
                        moods = [DEFAULT, TIRED, ANGRY, HAPPY, FROZEN, SCARY, CURIOUS]
                        mood = random.choice(moods)
                        self.robo.mood = mood
                        print(f"Mood: {mood}")

            # Update YOLO detections
            if self.enable_yolo and self.yolo_tracker:
                if current_time - last_yolo_update > yolo_update_interval:
                    detected_objects = self.yolo_tracker.get_detection_text(max_items=3)
                    last_yolo_update = current_time

                    # Print detections periodically
                    if detected_objects:
                        print(f"  YOLO: {', '.join(detected_objects)} (FPS: {self.yolo_tracker.fps})")

            # Update hand tracking
            hand_x, hand_y = 0, 0
            if self.enable_tracking:
                self.hand_tracker.update()
                hand_x, hand_y = self.hand_tracker.get_normalized_position()

                # Handle gesture-based emotion changes
                if hasattr(self.hand_tracker, 'gesture_changed') and self.hand_tracker.gesture_changed:
                    gesture = self.hand_tracker.gesture
                    finger_count = getattr(self.hand_tracker, 'finger_count', 0)
                    emotion_map = {
                        "fist": (ANGRY, "ANGRY"),      # 0 fingers - กำปั้น
                        "one": (CURIOUS, "CURIOUS"),   # 1 finger - ชี้
                        "two": (DEFAULT, "DEFAULT"),   # 2 fingers - ปกติ
                        "three": (TIRED, "TIRED"),     # 3 fingers - เหนื่อย
                        "four": (SCARY, "SCARY"),      # 4 fingers - กลัว
                        "open": (HAPPY, "HAPPY"),      # 5 fingers - มีความสุข
                    }
                    if gesture in emotion_map:
                        mood, mood_name = emotion_map[gesture]
                        self.robo.mood = mood
                        print(f"  Gesture: {gesture} ({finger_count} fingers) -> {mood_name}")
                        # Play robot sound
                        if mood_name in MOOD_SOUNDS:
                            play_robot_sound(mood_name)

                # Update eye position based on hand tracking
                if self.hand_tracker.hand_detected:
                    # Map hand position to screen coordinates
                    # hand_x, hand_y range from -1 to 1
                    # Map to eye position (0 to max constraint)
                    max_x = self.robo.get_screen_constraint_X()
                    max_y = self.robo.get_screen_constraint_Y()

                    # Invert X for natural tracking
                    eye_x = int((1 - (hand_x + 1) / 2) * max_x)
                    eye_y = int(((hand_y + 1) / 2) * max_y)

                    # Move servo to track hand (pan-tilt camera)
                    if self.servo.enabled:
                        # Track hand position with both servos
                        # hand_x, hand_y are already in range -1 to 1
                        self.servo.track_hand(hand_x, hand_y, smoothing=0.15, debug=True)

                    # Clamp values
                    eye_x = max(0, min(max_x, eye_x))
                    eye_y = max(0, min(max_y, eye_y))

                    # Set eye position
                    self.robo.eyeLxNext = eye_x
                    self.robo.eyeLyNext = eye_y

                # Fallback: Use YOLO person detection for eye tracking if no hand detected
                elif self.enable_yolo and self.yolo_tracker:
                    yolo_x, yolo_y = self.yolo_tracker.get_normalized_position()
                    if yolo_x != 0 or yolo_y != 0:
                        max_x = self.robo.get_screen_constraint_X()
                        max_y = self.robo.get_screen_constraint_Y()

                        eye_x = int((1 - (yolo_x + 1) / 2) * max_x)
                        eye_y = int(((yolo_y + 1) / 2) * max_y)

                        eye_x = max(0, min(max_x, eye_x))
                        eye_y = max(0, min(max_y, eye_y))

                        self.robo.eyeLxNext = eye_x
                        self.robo.eyeLyNext = eye_y

            # Update RoboEyes (this handles animation and rendering)
            self.robo.update()

            # Draw text overlay on GC9A01A if YOLO detected something
            if self.gc9a01a and self.text_overlay and self.enable_yolo and detected_objects:
                # Create text surface with detected objects
                text_surface = self.text_overlay.create_text_surface(
                    detected_objects,
                    bg_color=(0, 0, 0, 200),
                    text_color=(0, 255, 255)  # Cyan text
                )
                if text_surface:
                    # Blend with current eye surface and send to display
                    combined = self.text_overlay.blend_with_eyes(self.screen, text_surface)
                    self.gc9a01a.draw_from_surface(combined)

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

                # Show current gesture
                if hasattr(self.hand_tracker, 'gesture'):
                    gesture_text = f"Gesture: {self.hand_tracker.gesture}"
                    gesture_surface = font.render(gesture_text, True, (255, 255, 0))
                    self.camera_screen.blit(gesture_surface, (10, 50))

                # Show YOLO detections on HDMI too
                if self.enable_yolo and detected_objects:
                    yolo_text = f"YOLO: {', '.join(detected_objects)}"
                    yolo_surface = font.render(yolo_text, True, (0, 255, 255))
                    self.camera_screen.blit(yolo_surface, (10, 90))

                pygame.display.flip()

            self.clock.tick(config.FPS)

        self.cleanup()

    def cleanup(self):
        if self.voice_thread:
            self.voice_thread.stop()
        self.hand_tracker.release()
        if self.yolo_tracker:
            self.yolo_tracker.cleanup()
        if self.servo:
            self.servo.cleanup()
        if self.gc9a01a:
            self.gc9a01a.cleanup()
        pygame.quit()


class DemoMode:
    """Demo mode without camera - auto animation with RoboEyes"""

    def __init__(self):
        pygame.init()

        if config.DISPLAY_MODE == "gc9a01a":
            self.gc9a01a = create_display()
            self.screen = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        else:
            self.screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
            pygame.display.set_caption("RoboEyes Demo Mode")
            self.gc9a01a = None

        self.clock = pygame.time.Clock()

        # Create RoboEyes callback
        def robo_show(roboeyes):
            if self.gc9a01a:
                self.gc9a01a.draw_from_surface(self.screen)
            else:
                pygame.display.flip()

        # Initialize RoboEyes
        self.robo = RoboEyes(
            self.screen,
            config.SCREEN_WIDTH,
            config.SCREEN_HEIGHT,
            frame_rate=config.FPS,
            on_show=robo_show,
            bgcolor=0,
            fgcolor=(255, 255, 255)
        )

        # Configure for round display
        if config.DISPLAY_MODE == "gc9a01a":
            self.robo.eyes_width(90, 90)
            self.robo.eyes_height(90, 90)
            self.robo.eyes_spacing(20)
            self.robo.eyes_radius(20, 20)

        # Enable auto features
        self.robo.set_auto_blinker(ON, 3, 2)
        self.robo.set_idle_mode(ON, 2, 2)

        self.running = True
        self.last_mood_change = 0
        self.mood_interval = 5.0

        # Initialize Voice Control
        self.voice_thread = None
        if VOICE_AVAILABLE:
            self.voice_thread = VoiceThread(self.on_voice_command)
            self.voice_thread.start()
            print("Voice Control: Started (Thai)")

    def on_voice_command(self, text):
        mood_changed = True
        # Thai commands
        if "มีความสุข" in text or "ยิ้ม" in text or "แฮปปี้" in text or "happy" in text or "smile" in text:
            self.robo.mood = HAPPY
            mood_name = "HAPPY (มีความสุข)"
        elif "โกรธ" in text or "โมโห" in text or "angry" in text or "mad" in text:
            self.robo.mood = ANGRY
            mood_name = "ANGRY (โกรธ)"
        elif "เหนื่อย" in text or "ง่วง" in text or "นอน" in text or "tired" in text or "sleep" in text:
            self.robo.mood = TIRED
            mood_name = "TIRED (เหนื่อย)"
        elif "กลัว" in text or "หลอน" in text or "scary" in text or "fear" in text:
            self.robo.mood = SCARY
            mood_name = "SCARY (กลัว)"
        elif "หนาว" in text or "เย็น" in text or "frozen" in text or "cold" in text:
            self.robo.mood = FROZEN
            mood_name = "FROZEN (หนาว)"
        elif "สงสัย" in text or "อะไร" in text or "curious" in text or "what" in text:
            self.robo.mood = CURIOUS
            mood_name = "CURIOUS (สงสัย)"
        elif "ปกติ" in text or "รีเซ็ต" in text or "normal" in text or "reset" in text:
            self.robo.mood = DEFAULT
            mood_name = "DEFAULT (ปกติ)"
        else:
            mood_changed = False
            mood_name = "ไม่เปลี่ยน"

        if mood_changed:
            print(f"  >> เปลี่ยนอารมณ์เป็น: {mood_name}")
            self.last_mood_change = time.time()  # Reset auto mood timer
            # Play robot sound
            sound_key = mood_name.split()[0]  # Get "HAPPY", "ANGRY", etc.
            if sound_key in MOOD_SOUNDS:
                play_robot_sound(sound_key)
        else:
            print(f"  >> ไม่พบคำสั่ง (พูดว่า: มีความสุข/โกรธ/เหนื่อย/กลัว/หนาว/สงสัย/ปกติ)")

    def run(self):
        print("=" * 40)
        print("  RoboEyes Demo Mode (Voice Control)")
        print("=" * 40)
        print("คำสั่งเสียง: มีความสุข/โกรธ/เหนื่อย/กลัว/หนาว/สงสัย/ปกติ")
        print("=" * 40)
        # Say startup message
        play_robot_sound("DEFAULT")

        moods = [DEFAULT, TIRED, ANGRY, HAPPY, FROZEN, SCARY, CURIOUS]
        self.robo.mood = DEFAULT

        while self.running:
            current_time = time.time()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False

            # Auto change mood
            if current_time - self.last_mood_change > self.mood_interval:
                mood = random.choice(moods)
                self.robo.mood = mood
                print(f"Demo mood: {mood}")
                self.last_mood_change = current_time

            # Update RoboEyes
            self.robo.update()

            self.clock.tick(config.FPS)

        self.cleanup()

    def cleanup(self):
        if self.voice_thread:
            self.voice_thread.stop()
        if self.gc9a01a:
            self.gc9a01a.cleanup()
        pygame.quit()


class AICameraMode:
    """
    AI Camera Only Mode - Uses IMX500 for both object detection and tracking
    No USB webcam needed. Voice commands switch between modes.

    Voice Commands:
    - "ตรวจจับ" / "detect" - Detect and display objects
    - "ติดตาม" / "track" / "ติดตามคน" - Track person with eyes
    - "ติดตามแมว" / "track cat" - Track cat with eyes
    - "ซ่อน" / "hide" - Hide detection text
    - "แสดง" / "show" - Show detection text
    - Plus all mood commands (มีความสุข, โกรธ, etc.)
    """

    def __init__(self, yolo_confidence=0.5):
        self.yolo_confidence = yolo_confidence

        # Initialize Pygame
        pygame.init()

        if config.DISPLAY_MODE == "gc9a01a":
            self.gc9a01a = create_display()
            self.screen = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
            print(f"Eye Display: GC9A01A {config.SCREEN_WIDTH}x{config.SCREEN_HEIGHT}")

            # Create HDMI window for camera view
            self.camera_screen = pygame.display.set_mode((640, 480))
            pygame.display.set_caption("AI Camera Mode - YOLO Detection")
            print("Camera Display: 640x480 on HDMI")
        else:
            self.screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
            pygame.display.set_caption("AI Camera Mode")
            self.gc9a01a = None
            self.camera_screen = None

        self.clock = pygame.time.Clock()

        # Create RoboEyes callback
        def robo_show(roboeyes):
            if self.gc9a01a:
                # Don't auto-update, we'll handle it with text overlay
                pass
            else:
                pygame.display.flip()

        # Initialize RoboEyes
        self.robo = RoboEyes(
            self.screen,
            config.SCREEN_WIDTH,
            config.SCREEN_HEIGHT,
            frame_rate=config.FPS,
            on_show=robo_show,
            bgcolor=0,
            fgcolor=(255, 255, 255)
        )

        # Configure for round display
        if config.DISPLAY_MODE == "gc9a01a":
            self.robo.eyes_width(90, 90)
            self.robo.eyes_height(90, 90)
            self.robo.eyes_spacing(20)
            self.robo.eyes_radius(20, 20)

        # Enable auto-blinker, disable idle mode (we control position)
        self.robo.set_auto_blinker(ON, 3, 2)
        self.robo.set_idle_mode(OFF)

        # Initialize YOLO tracker
        self.yolo_tracker = None
        if YOLO_AVAILABLE:
            self.yolo_tracker = create_yolo_tracker(
                confidence_threshold=self.yolo_confidence,
                frame_rate=30
            )
            self.yolo_tracker.start()
            # Start in TRACK mode by default
            self.yolo_tracker.set_mode(YoloMode.TRACK)
            print("YOLO: Started in TRACK mode")
        else:
            print("YOLO: Not available!")

        # Initialize text overlay
        self.text_overlay = TextOverlay(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)

        # Initialize Voice Control
        self.voice_thread = None
        if VOICE_AVAILABLE:
            self.voice_thread = VoiceThread(self.on_voice_command)
            self.voice_thread.start()
            print("Voice Control: Started")

        self.running = True
        self.robo.mood = DEFAULT

    def on_voice_command(self, text):
        """Handle voice commands for both YOLO mode and mood"""
        # First check YOLO commands
        if self.yolo_tracker and self.yolo_tracker.process_voice_command(text):
            mode_text = self.yolo_tracker.get_mode_text()
            print(f"  >> YOLO: {mode_text}")
            return

        # Then check mood commands
        mood_changed = True
        if "มีความสุข" in text or "ยิ้ม" in text or "แฮปปี้" in text or "happy" in text or "smile" in text:
            self.robo.mood = HAPPY
            mood_name = "HAPPY"
        elif "โกรธ" in text or "โมโห" in text or "angry" in text or "mad" in text:
            self.robo.mood = ANGRY
            mood_name = "ANGRY"
        elif "เหนื่อย" in text or "ง่วง" in text or "นอน" in text or "tired" in text or "sleep" in text:
            self.robo.mood = TIRED
            mood_name = "TIRED"
        elif "กลัว" in text or "หลอน" in text or "scary" in text or "fear" in text:
            self.robo.mood = SCARY
            mood_name = "SCARY"
        elif "หนาว" in text or "เย็น" in text or "frozen" in text or "cold" in text:
            self.robo.mood = FROZEN
            mood_name = "FROZEN"
        elif "สงสัย" in text or "อะไร" in text or "curious" in text or "what" in text:
            self.robo.mood = CURIOUS
            mood_name = "CURIOUS"
        elif "ปกติ" in text or "รีเซ็ต" in text or "normal" in text or "reset" in text:
            self.robo.mood = DEFAULT
            mood_name = "DEFAULT"
        else:
            mood_changed = False
            mood_name = ""

        if mood_changed:
            print(f"  >> Mood: {mood_name}")
            if mood_name in MOOD_SOUNDS:
                play_robot_sound(mood_name)
        else:
            print(f"  >> ไม่เข้าใจคำสั่ง: {text}")

    def run(self):
        print("=" * 50)
        print("  AI Camera Mode (YOLO + Voice Control)")
        print("=" * 50)
        print("Voice Commands:")
        print("  Mode: detect / track / track cat / track dog")
        print("  Display: hide / show")
        print("  Mood: happy / angry / tired / scared / normal")
        print("=" * 50)
        print("Keyboard: ESC=Quit, SPACE=Random mood")
        print("          D=Detect mode, T=Track mode")
        print("=" * 50)

        detected_objects = []
        yolo_update_interval = 0.2  # Update YOLO text every 200ms
        last_yolo_update = 0
        display_update_interval = 0.05  # Update GC9A01A at 20 FPS max
        last_display_update = 0
        hdmi_update_interval = 0.033  # Update HDMI at 30 FPS
        last_hdmi_update = 0

        # Cache for text overlay
        cached_text_surface = None
        cached_texts = []

        # Pre-create font for HDMI
        hdmi_font = pygame.font.Font(None, 36)

        while self.running:
            current_time = time.time()

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_SPACE:
                        moods = [DEFAULT, TIRED, ANGRY, HAPPY, FROZEN, SCARY, CURIOUS]
                        mood = random.choice(moods)
                        self.robo.mood = mood
                        print(f"Mood: {mood}")
                    elif event.key == pygame.K_d:
                        if self.yolo_tracker:
                            self.yolo_tracker.set_mode(YoloMode.DETECT)
                    elif event.key == pygame.K_t:
                        if self.yolo_tracker:
                            self.yolo_tracker.set_mode(YoloMode.TRACK)

            # Update YOLO detections (throttled)
            if self.yolo_tracker:
                if current_time - last_yolo_update > yolo_update_interval:
                    detected_objects = self.yolo_tracker.get_detection_text(max_items=3)
                    last_yolo_update = current_time

                # Update eye position based on YOLO tracking
                yolo_x, yolo_y = self.yolo_tracker.get_normalized_position()
                if yolo_x != 0 or yolo_y != 0:
                    max_x = self.robo.get_screen_constraint_X()
                    max_y = self.robo.get_screen_constraint_Y()

                    eye_x = int((1 - (yolo_x + 1) / 2) * max_x)
                    eye_y = int(((yolo_y + 1) / 2) * max_y)

                    eye_x = max(0, min(max_x, eye_x))
                    eye_y = max(0, min(max_y, eye_y))

                    self.robo.eyeLxNext = eye_x
                    self.robo.eyeLyNext = eye_y

            # Update RoboEyes
            self.robo.update()

            # Update GC9A01A display (throttled to 20 FPS)
            if self.gc9a01a and current_time - last_display_update > display_update_interval:
                last_display_update = current_time

                # Get mode text
                mode_text = self.yolo_tracker.get_mode_text() if self.yolo_tracker else ""

                # Prepare text lines
                display_texts = []
                if mode_text:
                    display_texts.append(mode_text)
                if detected_objects and self.yolo_tracker and self.yolo_tracker.show_text:
                    display_texts.extend(detected_objects[:2])

                # Only recreate text surface if content changed
                if display_texts != cached_texts:
                    cached_texts = display_texts.copy()
                    if display_texts:
                        cached_text_surface = self.text_overlay.create_text_surface(
                            display_texts,
                            bg_color=(0, 0, 0, 180),
                            text_color=(0, 255, 255)
                        )
                    else:
                        cached_text_surface = None

                # Draw to GC9A01A
                if cached_text_surface:
                    combined = self.text_overlay.blend_with_eyes(self.screen, cached_text_surface)
                    self.gc9a01a.draw_from_surface(combined)
                else:
                    self.gc9a01a.draw_from_surface(self.screen)

            # Draw camera view on HDMI (throttled to 30 FPS)
            if self.camera_screen and self.yolo_tracker:
                if current_time - last_hdmi_update > hdmi_update_interval:
                    last_hdmi_update = current_time

                    frame = self.yolo_tracker.latest_frame
                    if frame is not None:
                        # Frame is already RGB from yolo_tracker (no conversion needed)
                        if frame.shape[0] != 480 or frame.shape[1] != 640:
                            frame = cv2.resize(frame, (640, 480))
                        frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                        self.camera_screen.blit(frame_surface, (0, 0))

                        # Mode
                        mode_text = self.yolo_tracker.get_mode_text()
                        mode_surface = hdmi_font.render(f"Mode: {mode_text}", True, (0, 255, 255))
                        self.camera_screen.blit(mode_surface, (10, 10))

                        # FPS
                        fps_surface = hdmi_font.render(f"FPS: {self.yolo_tracker.fps}", True, (0, 255, 0))
                        self.camera_screen.blit(fps_surface, (10, 50))

                        # Detections
                        if detected_objects:
                            det_text = f"Detected: {', '.join(detected_objects)}"
                            det_surface = hdmi_font.render(det_text, True, (255, 255, 0))
                            self.camera_screen.blit(det_surface, (10, 90))

                        pygame.display.flip()

            # Limit main loop to 30 FPS to reduce CPU
            self.clock.tick(30)

        self.cleanup()

    def cleanup(self):
        if self.voice_thread:
            self.voice_thread.stop()
        if self.yolo_tracker:
            self.yolo_tracker.cleanup()
        if self.gc9a01a:
            self.gc9a01a.cleanup()
        pygame.quit()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='RoboEyes Hand-Eye Tracker')
    parser.add_argument('--demo', action='store_true',
                        help='Run in demo mode (no camera)')
    parser.add_argument('--ai-camera', action='store_true',
                        help='AI Camera only mode (IMX500 YOLO + Voice control)')
    parser.add_argument('--no-mediapipe', action='store_true',
                        help='Use OpenCV instead of MediaPipe')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('--no-camera-view', action='store_true',
                        help='Hide camera view window on HDMI')
    parser.add_argument('--no-tracking', action='store_true',
                        help='Disable hand tracking (eyes wont follow hand)')
    parser.add_argument('--yolo', action='store_true',
                        help='Enable YOLO object detection (requires IMX500 AI Camera)')
    parser.add_argument('--yolo-confidence', type=float, default=0.5,
                        help='YOLO detection confidence threshold (0.0-1.0, default: 0.5)')
    args = parser.parse_args()

    if args.demo:
        app = DemoMode()
    elif args.ai_camera:
        # AI Camera only mode - use IMX500 for everything
        app = AICameraMode(yolo_confidence=args.yolo_confidence)
    else:
        app = RoboEyesHandTracker(
            use_mediapipe=not args.no_mediapipe,
            camera_id=args.camera,
            show_camera=not args.no_camera_view,
            enable_tracking=not args.no_tracking,
            enable_yolo=args.yolo,
            yolo_confidence=args.yolo_confidence
        )

    try:
        app.run()
    except KeyboardInterrupt:
        print("\nStopped by user")
        app.cleanup()


if __name__ == "__main__":
    main()
