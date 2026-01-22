#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hand-Eye Tracker with RoboEyes for ST7735S / GC9A01A Display
Raspberry Pi 5

Dual Camera Logic:
- Mode 1: Hand Tracking (Webcam)
- Mode 2: Object Detection (AI Camera)
"""

# -*- coding: utf-8 -*-
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

# Import psutil for better process management
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available - using fallback process management")

# Monkey patch shutil.which to force SpeechRecognition to use WAV
original_which = shutil.which
def mock_which(cmd, mode=os.F_OK | os.X_OK, path=None):
    if "flac" in str(cmd).lower():
        # print(f"DEBUG: Blocked access to FLAC ({cmd})")
        return None
    return original_which(cmd, mode, path)
shutil.which = mock_which

import config
from roboeyes import *
from pi_camera import PiCamera
from servo_controller import ServoController

# Try to import Picamera2 for secondary view
try:
    from picamera2 import Picamera2
    PICAM2_AVAILABLE = True
except ImportError:
    PICAM2_AVAILABLE = False
    print("Warning: Picamera2 not available")

try:
    import speech_recognition as sr
    VOICE_AVAILABLE = False # Force disabled as requested
    # print("Voice Control: Available")
except ImportError:
    VOICE_AVAILABLE = False
    print("Voice Control: Not available (install SpeechRecognition)")

SOUND_ENABLED = True
SOUND_PATH = "/home/piyanat/hand-eye-tracker/assets/sounds/"
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
    if not SOUND_ENABLED:
        return
    def _play():
        try:
            sound_file = MOOD_SOUNDS.get(mood, "")
            if not sound_file:
                return
            full_path = SOUND_PATH + sound_file
            if os.path.exists(full_path):
                subprocess.call(f'ffplay -nodisp -autoexit -loglevel quiet "{full_path}"', shell=True)
        except Exception as e:
            pass
    threading.Thread(target=_play, daemon=True).start()

if VOICE_AVAILABLE:
    class VoiceThread(threading.Thread):
        def __init__(self, callback):
            threading.Thread.__init__(self)
            self.callback = callback
            self.running = True
            self.recognizer = sr.Recognizer()

        def run(self):
            while self.running:
                try:
                    # print("Voice: Recording via arecord...")
                    filename = "/tmp/voice_cmd.wav"
                    cmd = f"arecord -D plughw:2,0 -d 3 -f S16_LE -c 1 -r 16000 -t wav -q {filename}"
                    subprocess.call(cmd, shell=True)
                    
                    if os.path.exists(filename) and os.path.getsize(filename) > 0:
                        with sr.AudioFile(filename) as source:
                            audio = self.recognizer.record(source)
                            try:
                                text = self.recognizer.recognize_google(audio, language="th-TH")
                                print(f"  > Voice: '{text}'")
                                self.callback(text)
                            except sr.UnknownValueError:
                                pass
                            except Exception as e:
                                pass
                except Exception as e:
                    time.sleep(1)

        def stop(self):
            self.running = False

# Import display driver based on config
if config.DISPLAY_MODE == "gc9a01a":
    from gc9a01a_display import create_display, RPI_AVAILABLE
elif config.DISPLAY_MODE == "st7735s":
    from st7735_display import create_display, RPI_AVAILABLE
else:
    RPI_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe: Not available - using OpenCV fallback")

try:
    from yolo_tracker import create_yolo_tracker, YOLO_AVAILABLE, YoloMode
except ImportError:
    YOLO_AVAILABLE = False
    YoloMode = None
    print("YOLO Tracker: Not available")


class TextOverlay:
    """Helper class to render text overlay on display using pygame"""

    def __init__(self, width=160, height=128):
        self.width = width
        self.height = height
        # Adaptive font size based on display type
        if width == 128:  # ST7735 (128x160 portrait)
            self.font_size = 12  # Smaller font for ST7735
        elif width == 160 and height == 128:  # ST7735 (160x128 landscape)
            self.font_size = 12
        else:  # GC9A01A (240x240)
            self.font_size = 14
        self.font = None
        self._init_font()

    def _init_font(self):
        """Initialize pygame font"""
        pygame.font.init()
        # Use default pygame font
        self.font = pygame.font.Font(None, self.font_size)

    def create_text_surface(self, texts, bg_color=(0, 0, 0, 150), text_color=(0, 255, 255)):
        """Create a pygame surface with text overlay"""
        if not texts:
            return None

        # Create transparent surface
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        # Calculate text area at bottom of screen
        padding = 2 if self.width == 128 else 4  # Smaller padding for ST7735
        margin_bottom = 1 if self.width == 128 else 2
        line_height = self.font_size + 2 if self.width == 128 else self.font_size + 4
        total_height = len(texts) * line_height + padding * 2

        # Draw semi-transparent background at bottom
        y_start = self.height - total_height - margin_bottom
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
        """Blend text overlay with eye surface"""
        if text_surface is None:
            return eye_surface
        result = eye_surface.copy()
        result.blit(text_surface, (0, 0))
        return result


class HandTrackerMediaPipe:
    """Hand tracking with MediaPipe"""
    WRIST = 0
    THUMB_TIP = 4
    INDEX_FINGER_TIP = 8  # Track index finger tip instead of palm
    MIDDLE_FINGER_TIP = 12
    TIP_IDS = [4, 8, 12, 16, 20]

    def __init__(self, camera_id=0):
        print("   [INFO] Initializing MediaPipe Hand Tracker...")
        # Use 1280x720 to capture full FOV of IMX708 (16:9 sensor)
        self.cap = PiCamera(camera_id, width=1280, height=720, fps=30)
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
        h, w, _ = frame_shape
        landmarks = []
        for id, lm in enumerate(hand_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            landmarks.append((id, cx, cy))
        return landmarks

    def _count_fingers(self, landmarks, handedness="Right"):
        if len(landmarks) < 21: return 0
        fingers = []
        if handedness == "Right":
            if landmarks[self.THUMB_TIP][1] < landmarks[self.THUMB_TIP - 1][1]: fingers.append(1)
            else: fingers.append(0)
        else:
            if landmarks[self.THUMB_TIP][1] > landmarks[self.THUMB_TIP - 1][1]: fingers.append(1)
            else: fingers.append(0)
        for tip_id in [8, 12, 16, 20]:
            if landmarks[tip_id][2] < landmarks[tip_id - 2][2]: fingers.append(1)
            else: fingers.append(0)
        return sum(fingers)

    def _detect_gesture(self, finger_count):
        if finger_count == 0: return "fist"
        elif finger_count == 1: return "one"
        elif finger_count == 2: return "two"
        elif finger_count == 3: return "three"
        elif finger_count == 4: return "four"
        elif finger_count == 5: return "open"
        return "unknown"

    def update(self):
        ret, frame = self.cap.read()
        if not ret: return False
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        self.hand_detected = False
        self.gesture_changed = False
        old_gesture = self.gesture

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Track INDEX FINGER TIP (landmark #8) instead of palm
                index_tip = hand_landmarks.landmark[self.INDEX_FINGER_TIP]
                self.hand_x = index_tip.x
                self.hand_y = index_tip.y
                self.hand_detected = True
                self.landmarks_list = self._get_landmarks_list(hand_landmarks, frame.shape)
                handedness = "Right"
                if results.multi_handedness:
                    handedness = results.multi_handedness[idx].classification[0].label
                self.finger_count = self._count_fingers(self.landmarks_list, handedness)
                self.gesture = self._detect_gesture(self.finger_count)

                # Draw hand landmarks on frame
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
                )

                # Draw landmark IDs on frame and highlight index finger tip
                for id, cx, cy in self.landmarks_list:
                    cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    # Highlight index finger tip with a circle
                    if id == self.INDEX_FINGER_TIP:
                        cv2.circle(frame, (cx, cy), 10, (0, 255, 255), 2)

        if self.gesture != old_gesture and self.gesture != "unknown":
            self.gesture_changed = True
        self.latest_frame = frame
        return True

    def get_normalized_position(self):
        if not self.hand_detected: return 0, 0
        norm_x = (self.hand_x - 0.5) * 2
        norm_y = (self.hand_y - 0.5) * 2
        return norm_x, norm_y

    def release(self):
        if self.cap is not None:
            if hasattr(self.cap, 'close'): self.cap.close()
            elif hasattr(self.cap, 'release'): self.cap.release()
            self.cap = None
        if hasattr(self, 'hands') and self.hands is not None:
            self.hands.close()
            self.hands = None

class HandTrackerOpenCV:
    """Hand tracking with OpenCV skin detection (Fallback)"""
    def __init__(self, camera_id=0):
        print("   [WARNING] Initializing OpenCV Hand Tracker (FALLBACK)...")
        # Use 1280x720 to capture full FOV of IMX708
        self.cap = PiCamera(camera_id, width=1280, height=720, fps=30)
        self.hand_x = 0.5
        self.hand_y = 0.5
        self.hand_detected = False
        self.latest_frame = None
        self.finger_count = 0
        self.gesture = "unknown"
        self.gesture_changed = False
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    def update(self):
        ret, frame = self.cap.read()
        if not ret: return False
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
            if cv2.contourArea(max_contour) > 5000:
                M = cv2.moments(max_contour)
                if M["m00"] != 0:
                    self.hand_x = int(M["m10"] / M["m00"]) / frame.shape[1]
                    self.hand_y = int(M["m01"] / M["m00"]) / frame.shape[0]
                    self.hand_detected = True
        self.latest_frame = frame
        return True

    def get_normalized_position(self):
        if not self.hand_detected: return 0, 0
        return (self.hand_x - 0.5) * 2, (self.hand_y - 0.5) * 2

    def release(self):
        if self.cap is not None:
            if hasattr(self.cap, 'close'): self.cap.close()
            elif hasattr(self.cap, 'release'): self.cap.release()
            self.cap = None

class RoboEyesApp:
    """Main application supporting Single (Hand) and AI (Object) modes"""

    def __init__(self, mode="hand", use_mediapipe=True, camera_id=0, show_camera=True):
        self.mode = mode  # "hand" or "ai" or "demo"
        self.show_camera = show_camera
        
        # Initialize Pygame
        pygame.init()

        # Create Display
        if config.DISPLAY_MODE in ["gc9a01a", "st7735s"]:
            self.display = create_display()
            self.screen = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
            print(f"Display: {config.DISPLAY_MODE.upper()} {config.SCREEN_WIDTH}x{config.SCREEN_HEIGHT}")
        else:
            self.display = None
            self.screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
            print(f"Display: Pygame Window {config.SCREEN_WIDTH}x{config.SCREEN_HEIGHT}")

        self.clock = pygame.time.Clock()

        # RoboEyes Setup
        def robo_show(roboeyes):
            if self.display:
                self.display.draw_from_surface(self.screen)
            else:
                pygame.display.flip()

        self.robo = RoboEyes(
            self.screen, config.SCREEN_WIDTH, config.SCREEN_HEIGHT,
            frame_rate=config.FPS, on_show=robo_show,
            bgcolor=0, fgcolor=(255, 255, 255)
        )

        # Config sizes
        if config.DISPLAY_MODE == "gc9a01a":
            self.robo.eyes_width(90, 90)
            self.robo.eyes_height(90, 90)
            self.robo.eyes_spacing(20)
            self.robo.eyes_radius(20, 20)
        elif config.DISPLAY_MODE == "st7735s":
             # 128x160 settings - adjusted for robot eyes
            self.robo.eyes_width(50, 50)
            self.robo.eyes_height(60, 60)
            self.robo.eyes_spacing(10)
            self.robo.eyes_radius(10, 10)
        
        self.robo.set_auto_blinker(ON, 3, 2)

        # Text Overlay
        self.text_overlay = TextOverlay(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
        
        # Initialize Trackers
        self.hand_tracker = None
        self.yolo_tracker = None
        
        if self.mode == "hand":
            print(f"Mode: HAND TRACKING (Camera {camera_id})")
            if use_mediapipe and MEDIAPIPE_AVAILABLE:
                self.hand_tracker = HandTrackerMediaPipe(camera_id)
            else:
                self.hand_tracker = HandTrackerOpenCV(camera_id)
            self.robo.set_idle_mode(OFF)
            
        elif self.mode == "ai":
            print("Mode: AI OBJECT DETECTION (IMX500)")
            if YOLO_AVAILABLE:
                self.yolo_tracker = create_yolo_tracker(confidence_threshold=0.5)
                if self.yolo_tracker:
                    self.yolo_tracker.start()
            self.robo.set_idle_mode(OFF)
            
        else: # demo
            print("Mode: DEMO")
            self.robo.set_idle_mode(ON)

        # Servo
        self.servo = ServoController()
        
        # HDMI Camera View
        self.camera_screen = None
        self.secondary_cam = None
        
        if self.show_camera and self.mode != "demo":
            # Dual View Window
            self.camera_screen = pygame.display.set_mode((1280, 480))
            pygame.display.set_caption("Dual Camera View (Left: USB, Right: IMX500)")
            
            # Setup Secondary Camera
            if self.mode == "hand":
                # User requested to Close IMX500 in Hand Mode
                print("Hand Mode: IMX500 (Secondary) is DISABLED.")
                self.secondary_cam = None
            
            elif self.mode == "ai":
                # Primary is IMX500 (AI). Secondary is IMX708 (CSI).
                # Since both are CSI, use Picamera2 for secondary too.
                if PICAM2_AVAILABLE:
                    print("Init Secondary: Scanning for free CSI camera...")
                    for cam_idx in [0, 1]:
                        try:
                            # Try to open camera
                            temp_cam = Picamera2(camera_num=cam_idx)
                            # Use 1280x720 for wider FOV
                            cam_config = temp_cam.create_preview_configuration(
                                main={"size": (1280, 720), "format": "RGB888"}
                            )
                            temp_cam.configure(cam_config)
                            temp_cam.start()
                            
                            self.secondary_cam = temp_cam
                            print(f"   [SUCCESS] Secondary Camera (IMX708) started on Index {cam_idx}")
                            break
                        except Exception as e:
                            print(f"   [INFO] Camera {cam_idx} busy or failed: {e}")
                            if 'temp_cam' in locals():
                                try: temp_cam.close()
                                except: pass
                            self.secondary_cam = None
                
                if self.secondary_cam is None:
                    print("Error: Could not initialize secondary camera (IMX708)")

        # Voice
        self.voice_thread = None
        if VOICE_AVAILABLE:
            self.voice_thread = VoiceThread(self.on_voice_command)
            self.voice_thread.start()

        self.running = True

    def on_voice_command(self, text):
        # Mood commands
        mood_map = {
            "happy": HAPPY, "ยิ้ม": HAPPY, "มีความสุข": HAPPY,
            "angry": ANGRY, "โกรธ": ANGRY, "โมโห": ANGRY,
            "tired": TIRED, "เหนื่อย": TIRED,
            "scary": SCARY, "กลัว": SCARY,
            "frozen": FROZEN, "หนาว": FROZEN,
            "curious": CURIOUS, "สงสัย": CURIOUS,
            "normal": DEFAULT, "ปกติ": DEFAULT
        }
        
        for kw, mood in mood_map.items():
            if kw in text.lower():
                self.robo.mood = mood
                print(f"  > Mood changed: {kw}")
                return

    def run(self):
        print("Running...")
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: self.running = False
                    if event.key == pygame.K_SPACE:
                        self.robo.mood = random.choice([HAPPY, ANGRY, TIRED, DEFAULT])
                    
                    # Camera Swap (Hand Mode Only)
                    if event.key == pygame.K_c and self.mode == "hand":
                        print("Swapping cameras...")
                        # 1. Determine new IDs
                        current_id = self.hand_tracker.cap.camera_id if self.hand_tracker else 0
                        new_id = 1 if current_id == 0 else 0
                        
                        # 2. Release resources
                        if self.hand_tracker: self.hand_tracker.release()
                        if self.secondary_cam:
                            try:
                                if hasattr(self.secondary_cam, 'stop'): self.secondary_cam.stop()
                                if hasattr(self.secondary_cam, 'close'): self.secondary_cam.close()
                            except: pass
                            self.secondary_cam = None
                        
                        # 3. Re-init Hand Tracker
                        print(f"Re-init Hand Tracker on Camera {new_id}...")
                        if MEDIAPIPE_AVAILABLE:
                            self.hand_tracker = HandTrackerMediaPipe(new_id)
                        else:
                            self.hand_tracker = HandTrackerOpenCV(new_id)
                            
                        # 4. Re-init Secondary Camera (on the other ID)
                        sec_id = 1 if new_id == 0 else 0
                        if PICAM2_AVAILABLE:
                            try:
                                print(f"Re-init Secondary (View) on Camera {sec_id}...")
                                self.secondary_cam = Picamera2(camera_num=sec_id)
                                cfg = self.secondary_cam.create_preview_configuration(
                                    main={"size": (640, 480), "format": "RGB888"}
                                )
                                self.secondary_cam.configure(cfg)
                                self.secondary_cam.start()
                            except Exception as e:
                                print(f"Secondary init failed: {e}")

            target_x, target_y = 0, 0
            has_target = False

            if self.mode == "hand" and self.hand_tracker:
                self.hand_tracker.update()
                if self.hand_tracker.hand_detected:
                    target_x, target_y = self.hand_tracker.get_normalized_position()
                    has_target = True
                    # Gesture emotions
                    if self.hand_tracker.gesture == "fist": self.robo.mood = ANGRY
                    elif self.hand_tracker.gesture == "open": self.robo.mood = HAPPY

            elif self.mode == "ai" and self.yolo_tracker:
                tx, ty = self.yolo_tracker.get_normalized_position()
                if tx != 0 or ty != 0:
                    target_x, target_y = tx, ty
                    has_target = True

            # Update Eye Position
            if has_target:
                max_x = self.robo.get_screen_constraint_X()
                max_y = self.robo.get_screen_constraint_Y()
                eye_x = int((1 - (target_x + 1) / 2) * max_x)
                eye_y = int(((target_y + 1) / 2) * max_y)
                self.robo.eyeLxNext = max(0, min(max_x, eye_x))
                self.robo.eyeLyNext = max(0, min(max_y, eye_y))
                
                if self.servo.enabled:
                    self.servo.track_hand(target_x, target_y)

            self.robo.update()

            # Render Text Overlay (Hand Mode)
            if self.mode == "hand" and self.hand_tracker and self.display:
                fingers = getattr(self.hand_tracker, 'finger_count', 0)
                gesture = getattr(self.hand_tracker, 'gesture', '')
                
                # Prepare text - shorter format for small screens
                texts = []
                if config.DISPLAY_MODE == "st7735s":
                    # Compact format for ST7735
                    texts = [f"F:{fingers}"]  # "F:3" instead of "Fingers: 3"
                    if gesture and gesture != "unknown":
                        texts.append(f"{gesture.upper()}")  # "OPEN" instead of "Gesture: open"
                else:
                    # Full format for GC9A01A
                    texts = [f"Fingers: {fingers}"]
                    if gesture and gesture != "unknown":
                        texts.append(f"Gesture: {gesture}")
                
                text_surface = self.text_overlay.create_text_surface(texts)
                if text_surface:
                    self.screen.blit(text_surface, (0, 0))

            # Render Text Overlay (AI Mode)
            if self.mode == "ai" and self.yolo_tracker and self.display:
                detected_texts = self.yolo_tracker.get_detection_text(max_items=2)
                # print(f"DEBUG: detected_texts={detected_texts}") # DEBUG
                if detected_texts:
                    text_surface = self.text_overlay.create_text_surface(detected_texts)
                    if text_surface:
                        # Blend text onto the main screen surface *before* sending to display
                        self.screen.blit(text_surface, (0, 0))
            
            # Camera View
            if self.camera_screen:
                frame_left = None  # USB
                frame_right = None # IMX500

                # --- DEBUG: Print status every 3 seconds ---
                if int(time.time()) % 3 == 0 and int(time.time() * 10) % 10 == 0:
                    print(f"DEBUG: Mode={self.mode}")
                    if self.mode == "ai":
                        if self.yolo_tracker:
                            print(f"  YOLO Running: {self.yolo_tracker.running}")
                            print(f"  YOLO Frame: {'Yes' if self.yolo_tracker.latest_frame is not None else 'No'}")
                        if self.secondary_cam:
                            print(f"  USB Cam Open: {self.secondary_cam.isOpened()}")
                # -------------------------------------------

                # Get Primary Frame
                if self.mode == "hand" and self.hand_tracker:
                    frame_left = self.hand_tracker.latest_frame # Already BGR
                elif self.mode == "ai" and self.yolo_tracker:
                    frame_right = self.yolo_tracker.latest_frame # Already RGB usually?

                # Get Secondary Frame
                if self.secondary_cam:
                    try:
                        # Secondary is always Picamera2 now (for both CSI cameras)
                        frame_sec = self.secondary_cam.capture_array()
                        
                        if self.mode == "hand": 
                            # Hand Mode: Primary=USB/CSI(0), Sec=IMX500(1) -> Right
                            frame_right = frame_sec
                        elif self.mode == "ai": 
                            # AI Mode: Primary=IMX500(1?), Sec=IMX708(0?) -> Left
                            # Picamera2 returns RGB, convert to BGR for OpenCV consistency if needed
                            # But here we render to Pygame which wants RGB.
                            # YOLO frame (frame_right) logic below assumes BGR conversion?
                            # Let's keep RGB for display.
                            frame_left = cv2.cvtColor(frame_sec, cv2.COLOR_RGB2BGR) # Store as BGR for consistency
                    except Exception as e:
                        # print(f"Sec Cam Error: {e}")
                        pass

                # Draw Left (USB)
                if frame_left is not None:
                    try:
                        frame_left = cv2.resize(frame_left, (640, 480))
                        # Convert BGR to RGB for Pygame
                        frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
                        surf = pygame.surfarray.make_surface(frame_left.swapaxes(0, 1))
                        self.camera_screen.blit(surf, (0, 0))
                    except: pass
                else:
                    # Draw Placeholder for Left
                    pygame.draw.rect(self.camera_screen, (20, 20, 20), (0, 0, 640, 480))
                    font = pygame.font.Font(None, 36)
                    self.camera_screen.blit(font.render("No USB Cam", True, (100, 100, 100)), (250, 220))
                
                # Draw Right (IMX500)
                if frame_right is not None:
                    try:
                        frame_right = cv2.resize(frame_right, (640, 480))
                        # Check if it needs conversion. 
                        if self.mode == "hand": # Picamera2 (RGB)
                            pass # Already RGB
                        elif self.mode == "ai": # YOLO (BGR from cv2/modlib?)
                             # Let's assume YOLO returns BGR like opencv
                             frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)

                        surf = pygame.surfarray.make_surface(frame_right.swapaxes(0, 1))
                        self.camera_screen.blit(surf, (640, 0))
                    except: pass
                else:
                    # Draw Placeholder for Right
                    pygame.draw.rect(self.camera_screen, (20, 20, 20), (640, 0, 640, 480))
                    font = pygame.font.Font(None, 36)
                    self.camera_screen.blit(font.render("No IMX500", True, (100, 100, 100)), (890, 220))

                # Draw Labels
                font = pygame.font.Font(None, 30)
                self.camera_screen.blit(font.render("USB Camera", True, (0, 255, 0)), (10, 10))
                self.camera_screen.blit(font.render("IMX500 AI", True, (0, 255, 0)), (650, 10))
                
                pygame.display.flip()

            self.clock.tick(config.FPS)
        self.cleanup()

    def cleanup(self):
        if self.voice_thread: self.voice_thread.stop()
        if self.hand_tracker: self.hand_tracker.release()
        if self.yolo_tracker: self.yolo_tracker.cleanup()
        if self.secondary_cam:
            if hasattr(self.secondary_cam, 'stop'): self.secondary_cam.stop()
            elif hasattr(self.secondary_cam, 'release'): self.secondary_cam.release()
        if self.display: self.display.cleanup()
        pygame.quit()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['hand', 'ai', 'demo'], default='hand', help='Operation mode')
    parser.add_argument('--camera', type=int, default=0, help='Webcam ID (for hand mode)')
    parser.add_argument('--no-view', action='store_true', help='Hide camera view')
    args = parser.parse_args()

    app = RoboEyesApp(mode=args.mode, camera_id=args.camera, show_camera=not args.no_view)
    app.run()

if __name__ == "__main__":
    main()