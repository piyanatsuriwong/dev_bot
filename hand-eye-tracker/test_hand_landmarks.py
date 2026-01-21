#!/usr/bin/env python3
"""
Test Hand Landmarks with MediaPipe
แสดง Hand Landmarks บน webcam UI
"""
import cv2
import mediapipe as mp
import time

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Hand landmark indices
TIP_IDS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
TIP_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
TIP_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

def count_fingers(landmarks, handedness="Right"):
    """Count raised fingers"""
    if len(landmarks) < 21:
        return 0, []

    fingers = []

    # Thumb
    if handedness == "Right":
        if landmarks[4][1] < landmarks[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    else:
        if landmarks[4][1] > landmarks[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)

    # Other 4 fingers
    for tip_id in [8, 12, 16, 20]:
        if landmarks[tip_id][2] < landmarks[tip_id - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers), fingers

def get_gesture(finger_count):
    """Get gesture name from finger count"""
    gestures = {
        0: "Fist (กำปั้น)",
        1: "One (ชี้)",
        2: "Two (สอง)",
        3: "Three (สาม)",
        4: "Four (สี่)",
        5: "Open (กางมือ)"
    }
    return gestures.get(finger_count, "Unknown")

def main():
    # Try different camera indices
    camera_id = 0
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"Cannot open camera {camera_id}, trying others...")
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                camera_id = i
                print(f"Using camera {i}")
                break

    if not cap.isOpened():
        print("No camera found!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    prev_time = time.time()

    print("=" * 50)
    print("  MediaPipe Hand Landmarks Test")
    print("=" * 50)
    print("Press 'q' to quit")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get handedness
                handedness = "Right"
                if results.multi_handedness:
                    handedness = results.multi_handedness[idx].classification[0].label

                # Convert to pixel coordinates
                h, w, _ = frame.shape
                landmarks = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append((id, cx, cy))

                # Draw MediaPipe landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
                )

                # Draw landmark IDs
                for id, cx, cy in landmarks:
                    cv2.putText(frame, str(id), (cx + 5, cy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # Draw finger tips with colors
                for i, tip_id in enumerate(TIP_IDS):
                    cx, cy = landmarks[tip_id][1], landmarks[tip_id][2]
                    cv2.circle(frame, (cx, cy), 15, TIP_COLORS[i], cv2.FILLED)
                    cv2.circle(frame, (cx, cy), 17, (255, 255, 255), 2)
                    # Draw tip name
                    cv2.putText(frame, TIP_NAMES[i], (cx - 20, cy + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, TIP_COLORS[i], 1)

                # Count fingers
                finger_count, fingers_state = count_fingers(landmarks, handedness)
                gesture = get_gesture(finger_count)

                # Draw info box
                box_x = 10 if idx == 0 else w - 260
                cv2.rectangle(frame, (box_x, 10), (box_x + 250, 140), (0, 0, 0), -1)
                cv2.rectangle(frame, (box_x, 10), (box_x + 250, 140), (255, 255, 255), 2)

                cv2.putText(frame, f"Hand: {handedness}", (box_x + 10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Fingers: {finger_count}", (box_x + 10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Gesture: {gesture}", (box_x + 10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Draw finger states
                finger_names = ["T", "I", "M", "R", "P"]
                for i, (name, state) in enumerate(zip(finger_names, fingers_state)):
                    color = (0, 255, 0) if state else (0, 0, 255)
                    cv2.putText(frame, name, (box_x + 10 + i * 40, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.circle(frame, (box_x + 20 + i * 40, 130), 5, color, -1)

        # Draw FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (w - 100, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw instructions
        cv2.putText(frame, "Press 'q' to quit", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Hand Landmarks Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()
