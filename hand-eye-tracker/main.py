#!/usr/bin/env python3
"""
Hand-Eye Tracker for Raspberry Pi 5
ระบบติดตามมือด้วย webcam และให้ตาขยับตาม
ใช้ Sentient-Eye-Engine สำหรับแสดงผลตา

สำหรับ Raspberry Pi 5 ARM64:
- ใช้ opencv-python-headless
- รองรับทั้ง MediaPipe (ถ้าติดตั้งได้) และ OpenCV-only fallback
"""

import pygame
import cv2
import numpy as np
import sys
import random
import time
import config
from face_renderer import FaceRenderer

# ลอง import MediaPipe (อาจไม่มีบน ARM64)
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    print("MediaPipe: Available")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe: Not available - using OpenCV skin detection fallback")


class HandTrackerMediaPipe:
    """Hand tracking ด้วย MediaPipe (ถ้ามี)"""
    
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,  # ใช้โมเดลเบาสำหรับ Pi
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.hand_x = 0.5
        self.hand_y = 0.5
        self.hand_detected = False
        self.latest_frame = None
        
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
                # ใช้ตำแหน่งกลางฝ่ามือ
                palm = hand_landmarks.landmark[9]
                self.hand_x = palm.x
                self.hand_y = palm.y
                self.hand_detected = True
                
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1)
                )
        
        self._draw_status(frame)
        self.latest_frame = frame
        return True
        
    def _draw_status(self, frame):
        status = "HAND DETECTED" if self.hand_detected else "NO HAND"
        color = (0, 255, 0) if self.hand_detected else (0, 0, 255)
        cv2.putText(frame, f"[MediaPipe] {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if self.hand_detected:
            cv2.putText(frame, f"X:{self.hand_x:.2f} Y:{self.hand_y:.2f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
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
    """Hand tracking ด้วย OpenCV skin detection + gesture detection"""
    
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.hand_x = 0.5
        self.hand_y = 0.5
        self.hand_detected = False
        self.latest_frame = None
        
        # Gesture detection
        self.finger_count = 0
        self.gesture = "unknown"  # "fist", "open", "unknown"
        self.last_gesture = "unknown"
        self.gesture_changed = False
        
        # HSV range สำหรับ skin detection
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
    def _count_fingers(self, contour):
        """นับนิ้วมือโดยใช้ Convex Hull และ Convexity Defects"""
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
                
                # คำนวณความยาวด้าน
                a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                
                # คำนวณมุมโดยใช้กฎโคไซน์
                if b * c == 0:
                    continue
                angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))
                
                # ถ้ามุม < 90 องศา และความลึกมากพอ = นิ้ว
                if angle <= np.pi / 2 and d > 10000:
                    finger_count += 1
            
            # finger_count คือจำนวนช่องว่างระหว่างนิ้ว
            # ดังนั้นนิ้ว = finger_count + 1 (ถ้า > 0)
            return min(finger_count + 1, 5) if finger_count > 0 else 0
            
        except Exception:
            return 0
        
    def update(self):
        ret, frame = self.cap.read()
        if not ret:
            return False
            
        frame = cv2.flip(frame, 1)
        
        # Skin detection ด้วย HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        
        # หา contour ที่ใหญ่ที่สุด
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        self.hand_detected = False
        self.gesture_changed = False
        
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(max_contour)
            
            # กรองขนาดขั้นต่ำ
            if area > 5000:
                # หาจุดศูนย์กลาง
                M = cv2.moments(max_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    h, w = frame.shape[:2]
                    self.hand_x = cx / w
                    self.hand_y = cy / h
                    self.hand_detected = True
                    
                    # นับนิ้วและตรวจจับ gesture
                    self.finger_count = self._count_fingers(max_contour)
                    
                    # กำหนด gesture ตามจำนวนนิ้ว
                    old_gesture = self.gesture
                    if self.finger_count == 0 or self.finger_count == 1:
                        self.gesture = "fist"      # กำมือ
                    elif self.finger_count == 2:
                        self.gesture = "two"       # 2 นิ้ว (peace)
                    elif self.finger_count == 3:
                        self.gesture = "three"     # 3 นิ้ว
                    elif self.finger_count == 4:
                        self.gesture = "four"      # 4 นิ้ว
                    elif self.finger_count >= 5:
                        self.gesture = "open"      # แบมือ
                    else:
                        self.gesture = "unknown"
                    
                    # ตรวจจับว่า gesture เปลี่ยนหรือไม่
                    if self.gesture != old_gesture and self.gesture != "unknown":
                        self.gesture_changed = True
                        self.last_gesture = self.gesture
                    
                    # วาด convex hull
                    hull_points = cv2.convexHull(max_contour)
                    cv2.drawContours(frame, [hull_points], -1, (255, 255, 0), 2)
                    
                    # วาด contour และจุดศูนย์กลาง
                    cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 10, (255, 0, 255), -1)
                    
                    # วาด bounding box
                    x, y, w_box, h_box = cv2.boundingRect(max_contour)
                    cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 255), 2)
        
        self._draw_status(frame)
        self.latest_frame = frame
        return True
        
    def _draw_status(self, frame):
        status = "HAND DETECTED" if self.hand_detected else "NO HAND"
        color = (0, 255, 0) if self.hand_detected else (0, 0, 255)
        cv2.putText(frame, f"[OpenCV] {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if self.hand_detected:
            cv2.putText(frame, f"X:{self.hand_x:.2f} Y:{self.hand_y:.2f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # แสดง gesture และจำนวนนิ้ว
            gesture_color = (0, 255, 255) if self.gesture == "open" else (0, 165, 255) if self.gesture == "fist" else (128, 128, 128)
            cv2.putText(frame, f"Fingers: {self.finger_count} | {self.gesture.upper()}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gesture_color, 2)
    
    def get_normalized_position(self):
        if not self.hand_detected:
            return 0, 0
        norm_x = (self.hand_x - 0.5) * 2
        norm_y = (self.hand_y - 0.5) * 2
        return norm_x, norm_y
        
    def release(self):
        self.cap.release()


class HandEyeApp:
    """แอพหลักที่รวม Hand Tracking กับ Eye Renderer"""
    
    def __init__(self, use_mediapipe=True, camera_id=0):
        pygame.init()
        pygame.mixer.init()
        
        self.screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        pygame.display.set_caption("Hand-Eye Tracker - Raspberry Pi 5")
        self.clock = pygame.time.Clock()
        
        # สร้าง Face Renderer
        self.face_renderer = FaceRenderer(self.screen)
        self.face_renderer.external_control = True  # ใช้ hand tracking แทน mouse
        
        # เลือก Hand Tracker
        if use_mediapipe and MEDIAPIPE_AVAILABLE:
            self.hand_tracker = HandTrackerMediaPipe(camera_id)
            print("Using: MediaPipe Hand Tracking")
        else:
            self.hand_tracker = HandTrackerOpenCV(camera_id)
            print("Using: OpenCV Skin Detection")
        
        self.show_webcam = True
        self.smoothing = 0.15
        self.current_look_x = 0
        self.current_look_y = 0
        self.running = True
        
    def run(self):
        print("="*50)
        print("  HAND-EYE TRACKER for Raspberry Pi 5")
        print("="*50)
        print("Gesture Controls:")
        print("  0-1 นิ้ว (กำมือ) = angry")
        print("  2 นิ้ว (peace)   = normal")
        print("  3 นิ้ว           = sad")
        print("  4 นิ้ว           = scared")
        print("  5 นิ้ว (แบมือ)   = happy")
        print("-"*50)
        print("Keyboard:")
        print("  SPACE = เปลี่ยน emotion สุ่ม")
        print("  W     = เปิด/ปิด webcam overlay")
        print("  ESC   = ออก")
        print("="*50)
        
        while self.running:
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
                    elif event.key == pygame.K_w:
                        self.show_webcam = not self.show_webcam
                        print(f"Webcam overlay: {'ON' if self.show_webcam else 'OFF'}")
            
            # Update hand tracking
            self.hand_tracker.update()
            hand_x, hand_y = self.hand_tracker.get_normalized_position()
            
            # ตรวจจับ gesture และเปลี่ยน emotion
            if hasattr(self.hand_tracker, 'gesture_changed') and self.hand_tracker.gesture_changed:
                gesture = self.hand_tracker.gesture
                emotion_map = {
                    "fist": "angry",      # กำมือ = โกรธ
                    "two": "normal",      # 2 นิ้ว = ปกติ
                    "three": "sad",       # 3 นิ้ว = เศร้า
                    "four": "scared",     # 4 นิ้ว = กลัว
                    "open": "happy",      # แบมือ = มีความสุข
                }
                if gesture in emotion_map:
                    emotion = emotion_map[gesture]
                    self.face_renderer.set_emotion(emotion)
                    print(f"Gesture: {gesture.upper()} ({self.hand_tracker.finger_count} fingers) -> Emotion: {emotion}")
            
            if self.hand_tracker.hand_detected:
                self.current_look_x += (hand_x - self.current_look_x) * self.smoothing
                self.current_look_y += (hand_y - self.current_look_y) * self.smoothing
                
                # เพิ่มความไวของการเคลื่อนที่ (multiplier 1.5x)
                look_x = -self.current_look_x * 1.5
                look_y = self.current_look_y * 1.5
                
                self.face_renderer.target_look_x = max(-1.0, min(1.0, look_x))
                self.face_renderer.target_look_y = max(-1.0, min(1.0, look_y))
                self.face_renderer.is_tracking = True
            else:
                self.current_look_x *= 0.95
                self.current_look_y *= 0.95
                self.face_renderer.target_look_x = self.current_look_x
                self.face_renderer.target_look_y = self.current_look_y
            
            # Draw eyes
            self.face_renderer.draw()
            
            # Show webcam as overlay (using pygame, not cv2.imshow)
            if self.show_webcam and self.hand_tracker.latest_frame is not None:
                self._draw_webcam_overlay()
            
            pygame.display.flip()
            self.clock.tick(config.FPS)
        
        self.hand_tracker.release()
        pygame.quit()
    
    def _draw_webcam_overlay(self):
        """แสดงภาพ webcam เป็น overlay มุมขวาล่าง"""
        frame = self.hand_tracker.latest_frame
        if frame is None:
            return
        
        # Resize และแปลงเป็น pygame surface
        h, w = frame.shape[:2]
        scale = 0.25  # ขนาด 25% ของ original
        new_w, new_h = int(w * scale), int(h * scale)
        
        small_frame = cv2.resize(frame, (new_w, new_h))
        small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # แปลงเป็น pygame surface
        surf = pygame.surfarray.make_surface(small_frame.swapaxes(0, 1))
        
        # วาดที่มุมขวาล่าง
        x = config.SCREEN_WIDTH - new_w - 10
        y = config.SCREEN_HEIGHT - new_h - 10
        
        # วาดกรอบ
        pygame.draw.rect(self.screen, (255, 0, 255), (x-2, y-2, new_w+4, new_h+4), 2)
        self.screen.blit(surf, (x, y))


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Hand-Eye Tracker for Raspberry Pi 5')
    parser.add_argument('--no-mediapipe', action='store_true', 
                       help='Force use OpenCV skin detection instead of MediaPipe')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    args = parser.parse_args()
    
    app = HandEyeApp(
        use_mediapipe=not args.no_mediapipe,
        camera_id=args.camera
    )
    app.run()


if __name__ == "__main__":
    main()
