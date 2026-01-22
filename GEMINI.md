# Raspberry Pi 5 Robot Project

## Project Overview
This project is a dual-component robotics system running on a Raspberry Pi 5. It combines a **Hand-Eye Tracker** (for interactive, expressive robot eyes) and a **Robot Server** (for controlling the mobile robot chassis and hardware).

## System Architecture

The system consists of two main independent components:

1.  **Hand-Eye Tracker (`hand-eye-tracker/`)**:
    *   **Function:** Visualizes animated "eyes" on a GC9A01A round LCD.
    *   **Input:** Webcam or AI Camera (IMX500) for tracking hands, faces, or objects. Microphone for voice commands.
    *   **Logic:** Uses MediaPipe or OpenCV for computer vision to update the eye gaze and "mood".
    *   **Output:** Renders graphics to SPI display.

2.  **Robot Server (`Server-pi5/`)**:
    *   **Function:** Low-level hardware control server.
    *   **Input:** TCP commands from a client (mobile app or controller).
    *   **Logic:** Processes commands to drive motors, move servos, and read sensors.
    *   **Output:** PWM signals to motors/servos, video stream (MJPEG) back to client.

---

## Component 1: Hand-Eye Tracker

Located in `D:\mobile\pi5\hand-eye-tracker\` (Local) or `/home/pi/hand-eye-tracker/` (Remote).

### Key Files
*   `main_roboeyes.py`: **Main Entry Point.** Runs the eye animation loop and tracking logic.
*   `config.py`: Configuration for display pins (`GC9A01A`), camera settings, and AI models.
*   `face_renderer.py`: Handles the drawing of the eyes and emotions.
*   `roboeyes.py`: Core logic for eye movement dynamics.
*   `test_*.py`: Various test scripts (`test_mic.py`, `test_ai_camera.py`, etc.) for hardware verification.

### Running the Tracker
**Standard Mode (Webcam + Hand Tracking):**
```bash
cd /home/pi/hand-eye-tracker
source env/bin/activate
python3 main_roboeyes.py
```

**AI Camera Mode (IMX500 - Object Tracking):**
```bash
python3 main_roboeyes.py --ai-camera
```
*   **Voice Commands:** "track person", "track cat", "happy", "angry".
*   **Keyboard:** `Space` (random mood), `ESC` (quit).

**Demo Mode (No Camera):**
```bash
python3 main_roboeyes.py --demo
```

---

## Component 2: Robot Server

Located in `D:\mobile\pi5\Server-pi5\` (Local).

### Key Files
*   `mainv3.py`: **Main Entry Point.** Initializes the `ServerController` and starts the TCP server.
*   `server.py`: Implements the TCP socket logic.
    *   **Port 5000:** Command channel (Motor, Servo, LED, Ultrasonic).
    *   **Port 8000:** Video streaming channel (MJPEG).
*   `Motor.py`, `Servo.py`, `Led.py`: Hardware driver classes.

### Hardware Control Protocol
The server accepts text-based commands over TCP (Port 5000).
*   **Format:** `COMMAND#ARG1#ARG2#...`
*   **Examples:**
    *   `CMD_MOTOR#100#100#100#100`: Drive motors.
    *   `CMD_SERVO#1#90`: Move servo ID 1 to 90 degrees.
    *   `CMD_LED_MOD#1`: Change LED mode.

---

## Development Workflow

### Remote Development (SSH)
*   **Host:** `192.168.1.47` (User: `pi`)
*   **Key:** `C:\Users\piyanat\.ssh\id_ed25519_pi5`

### Common Commands
**Deploy Code (Windows -> Pi):**
```powershell
scp -i "C:\Users\piyanat\.ssh\id_ed25519_pi5" D:/mobile/pi5/hand-eye-tracker/*.py pi@192.168.1.47:/home/pi/hand-eye-tracker/
```

**Install Dependencies (Pi):**
```bash
# In virtual environment
pip install -r requirements.txt
# System dependencies
sudo apt install python3-pygame python3-opencv flac
```

### Hardware setup
*   **Display:** GC9A01A (SPI0)
*   **Camera:** USB Webcam or CSI AI Camera (IMX500)
*   **Audio:** USB Microphone (for voice commands)
*   **Robot HAT:** Waveshare or similar generic driver board (I2C for PWM/Servos).
