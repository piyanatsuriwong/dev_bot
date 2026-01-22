#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hand Tracker - MediaPipe Hand Tracking with IMX708 Camera
For NumBot Robot Eye Tracker
"""

import cv2
import numpy as np
import time
import config

# Try to import Picamera2
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("Warning: Picamera2 not available")

# Try to import MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available")


class HandTracker:
    """
    Hand Tracker using IMX708 camera and MediaPipe

    Features:
    - Hand position tracking (palm center)
    - Finger count detection
    - Hand landmarks visualization
    - Normalized coordinates output (-1 to 1)
    """

    def __init__(self, camera_num=None, width=None, height=None, fps=None):
        # Camera settings
        self.camera_num = camera_num or config.CAMERA_IMX708_NUM
        self.width = width or config.CAMERA_IMX708_WIDTH
        self.height = height or config.CAMERA_IMX708_HEIGHT
        self.fps = fps or config.CAMERA_IMX708_FPS

        # Camera instance
        self.camera = None
        self.running = False

        # MediaPipe Hands
        self.mp_hands = None
        self.hands = None
        self.mp_drawing = None

        # Tracking state
        self.hand_position = (0.0, 0.0)  # Normalized (-1 to 1)
        self.has_hand = False
        self.finger_count = 0
        self.landmarks = None
        self.latest_frame = None
        self.latest_frame_rgb = None

        # FPS tracking
        self.frame_count = 0
        self.fps_time = time.time()
        self.actual_fps = 0

        # Initialize components
        self._init_mediapipe()

    def _init_mediapipe(self):
        """Initialize MediaPipe Hands"""
        if not MEDIAPIPE_AVAILABLE:
            print("HandTracker: MediaPipe not available")
            return

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.MEDIAPIPE_MAX_HANDS,
            min_detection_confidence=config.MEDIAPIPE_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MEDIAPIPE_TRACKING_CONFIDENCE
        )
        print("HandTracker: MediaPipe initialized")

    def start(self):
        """Start camera and tracking"""
        if not PICAMERA2_AVAILABLE:
            print("HandTracker: Picamera2 not available")
            return False

        try:
            # Initialize camera
            print(f"HandTracker: Opening camera {self.camera_num}...")
            self.camera = Picamera2(camera_num=self.camera_num)

            # Configure camera
            cam_config = self.camera.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            self.camera.configure(cam_config)
            self.camera.start()

            self.running = True
            print(f"HandTracker: Camera started ({self.width}x{self.height} @ {self.fps}fps)")
            return True

        except Exception as e:
            print(f"HandTracker: Failed to start camera: {e}")
            self.cleanup()
            return False

    def update(self):
        """
        Capture frame and process hand detection
        Returns: True if frame was processed, False otherwise
        """
        if not self.running or self.camera is None:
            return False

        try:
            # Capture frame (already RGB888 from Picamera2)
            frame = self.camera.capture_array("main")
            self.latest_frame_rgb = frame.copy()

            # Process with MediaPipe
            if self.hands is not None:
                results = self.hands.process(frame)

                if results.multi_hand_landmarks:
                    # Get first hand
                    hand_landmarks = results.multi_hand_landmarks[0]
                    self.landmarks = hand_landmarks
                    self.has_hand = True

                    # Calculate palm center (average of wrist and middle finger base)
                    wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                    middle_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

                    palm_x = (wrist.x + middle_mcp.x) / 2
                    palm_y = (wrist.y + middle_mcp.y) / 2

                    # Convert to normalized coordinates (-1 to 1)
                    # Note: X is inverted (mirror effect)
                    self.hand_position = (
                        (palm_x * 2 - 1),  # 0-1 -> -1 to 1
                        (palm_y * 2 - 1)   # 0-1 -> -1 to 1
                    )

                    # Count fingers
                    self.finger_count = self._count_fingers(hand_landmarks)

                    # Draw landmarks on frame
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                else:
                    self.has_hand = False
                    self.landmarks = None
                    self.finger_count = 0

            # Store frame with landmarks drawn
            self.latest_frame = frame

            # FPS tracking
            self.frame_count += 1
            now = time.time()
            if now - self.fps_time >= 1.0:
                self.actual_fps = self.frame_count / (now - self.fps_time)
                self.frame_count = 0
                self.fps_time = now

            return True

        except Exception as e:
            print(f"HandTracker: Error processing frame: {e}")
            return False

    def _count_fingers(self, landmarks):
        """
        Count extended fingers
        Returns: 0-5 finger count
        """
        if landmarks is None:
            return 0

        finger_tips = [
            self.mp_hands.HandLandmark.THUMB_TIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]

        finger_pips = [
            self.mp_hands.HandLandmark.THUMB_IP,
            self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            self.mp_hands.HandLandmark.RING_FINGER_PIP,
            self.mp_hands.HandLandmark.PINKY_PIP
        ]

        count = 0

        # Check thumb (compare x position)
        thumb_tip = landmarks.landmark[finger_tips[0]]
        thumb_ip = landmarks.landmark[finger_pips[0]]
        wrist = landmarks.landmark[self.mp_hands.HandLandmark.WRIST]

        # Determine hand orientation
        if wrist.x < landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP].x:
            # Right hand
            if thumb_tip.x < thumb_ip.x:
                count += 1
        else:
            # Left hand
            if thumb_tip.x > thumb_ip.x:
                count += 1

        # Check other fingers (compare y position - tip should be above pip)
        for i in range(1, 5):
            tip = landmarks.landmark[finger_tips[i]]
            pip = landmarks.landmark[finger_pips[i]]
            if tip.y < pip.y:  # In image coordinates, y increases downward
                count += 1

        return count

    def get_normalized_position(self):
        """
        Get hand position in normalized coordinates (-1 to 1)
        Returns: (x, y) tuple, (0, 0) if no hand detected
        """
        if self.has_hand:
            return self.hand_position
        return (0.0, 0.0)

    def get_finger_count(self):
        """Get current finger count (0-5)"""
        return self.finger_count

    def get_mood_from_fingers(self):
        """
        Get mood based on finger count
        Based on config.FINGER_MOOD_MAP
        """
        return config.FINGER_MOOD_MAP.get(self.finger_count, "DEFAULT")

    def get_frame(self):
        """Get latest frame with landmarks drawn"""
        return self.latest_frame

    def get_frame_rgb(self):
        """Get latest raw frame (RGB)"""
        return self.latest_frame_rgb

    def get_status_text(self):
        """Get status text for display"""
        if self.has_hand:
            x, y = self.hand_position
            return f"Hand: ({x:.2f}, {y:.2f}) | Fingers: {self.finger_count}"
        return "No hand detected"

    def cleanup(self):
        """Cleanup resources"""
        self.running = False

        if self.hands:
            self.hands.close()
            self.hands = None

        if self.camera:
            try:
                self.camera.stop()
                self.camera.close()
            except:
                pass
            self.camera = None

        print("HandTracker: Cleaned up")


def create_hand_tracker(**kwargs):
    """Factory function to create HandTracker"""
    if not PICAMERA2_AVAILABLE:
        print("HandTracker: Cannot create - Picamera2 not available")
        return None
    if not MEDIAPIPE_AVAILABLE:
        print("HandTracker: Cannot create - MediaPipe not available")
        return None
    return HandTracker(**kwargs)


# Test function
if __name__ == "__main__":
    import cv2

    print("Testing HandTracker...")
    tracker = create_hand_tracker()

    if tracker and tracker.start():
        print("Press 'q' to quit")

        try:
            while True:
                if tracker.update():
                    frame = tracker.get_frame()
                    if frame is not None:
                        # Add status text
                        status = tracker.get_status_text()
                        mood = tracker.get_mood_from_fingers()
                        cv2.putText(frame, status, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"Mood: {mood}", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(frame, f"FPS: {tracker.actual_fps:.1f}", (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                        # Show frame
                        cv2.imshow("Hand Tracker", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            tracker.cleanup()
            cv2.destroyAllWindows()
    else:
        print("Failed to start HandTracker")
