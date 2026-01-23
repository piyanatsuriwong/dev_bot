# Feature Specification: IMX500 Camera Object Detection

## Overview

เอกสารนี้อธิบาย feature การใช้กล้อง IMX500 สำหรับตรวจจับวัตถุ พร้อมแสดงผลบนจอ ST7735S โดยยังคงรักษา feature ใบหน้าหุ่นยนต์ (Robot Eyes) ไว้

### Implementation Status ✅

**Current Implementation:**
- **Module:** `yolo_tracker_v2.py`
- **Backend:** `modlib` (Primary) using **YOLO11n**
- **Performance:** ~10-11 FPS (Real-time)
- **Status:** Object Detection Working

---

## 1. Feature Requirements

### 1.1 Core Features

| Feature | Description | Priority |
|---------|-------------|----------|
| **Object Detection** | IMX500 ตรวจจับวัตถุ 80 ประเภท (COCO) | High |
| **Display Output** | แสดงผลวัตถุที่พบบน ST7735S (160x128) | High |
| **Robot Face** | คงไว้ซึ่ง Robot Eyes animation | High |
| **Beautiful Text** | ข้อความสวยงาม อ่านง่าย บนจอเล็ก | Medium |
| **Mode Switching** | สลับระหว่าง Hand/Object detection | High |
| **File Separation** | แยก code จาก main_roboeyes.py | Medium |

### 1.2 Camera Usage

```
┌─────────────────────────────────────────────────┐
│              Camera Assignment                   │
├─────────────────────────────────────────────────┤
│  IMX500 (CSI Port 0)  →  Object Detection ONLY  │
│  IMX708 (CSI Port 1)  →  Hand Tracking          │
└─────────────────────────────────────────────────┘
```

---

## 2. Architecture Design

### 2.1 Current File Structure

```
numbot/
├── main_roboeyes.py          # Main entry point
├── config.py                 # Configuration
│
├── yolo_tracker_v2.py        # ⭐ YOLO Object Detection (modlib/Picamera2)
├── yolo_tracker.py           # Legacy tracker (deprecated)
├── hand_tracker.py           # Hand tracking logic
│
├── imx500_detector.py        # Standalone detector utility
│
├── docs/
    └── FEATURE_IMX500_DETECTION.md
```

### 2.2 Key Module: `yolo_tracker_v2.py`

This module is a drop-in replacement for the original `yolo_tracker.py` but enhanced for stability:

```python
# Usage in main_roboeyes.py
from yolo_tracker_v2 import create_yolo_tracker, YoloMode

tracker = create_yolo_tracker(model="yolov8n", use_modlib=True)
tracker.start()
```

### 2.3 Module Dependency Diagram

```
                    ┌─────────────────────┐
                    │  main_roboeyes.py   │
                    │  (Entry Point)      │
                    └──────────┬──────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
┌────────▼────────┐  ┌─────────▼─────────┐  ┌───────▼───────┐
│  HandTracker    │  │   ST7735Display   │  │  RoboEyes     │
│  (IMX708)       │  │   - Display       │  │  - Animation  │
│  - MediaPipe    │  │   - Text render   │  │  - Moods      │
│                 │  │                   │  │               │
├─────────────────┤  └───────────────────┘  └───────────────┘
│ YoloTrackerV2   │
│  (IMX500)       │
│  - modlib       │
│  - YOLO11n      │
└─────────────────┘
```

---

## 3. Display Layout Design (ST7735S 160x128)

### 3.1 Screen Zones

```
┌─────────────────────────────────────┐
│            STATUS BAR (12px)        │  Zone 1: Mode indicator
├─────────────────────────────────────┤
│                                     │
│                                     │
│           ROBOT EYES                │  Zone 2: Face area (80px)
│         (Main Content)              │
│                                     │
│                                     │
├─────────────────────────────────────┤
│         DETECTION INFO              │  Zone 3: Detection text (36px)
│    [icon] Object Name               │
│    Confidence: 95%                  │
└─────────────────────────────────────┘
```

### 3.2 Layout Modes

#### Mode A: Face Only (Hand Tracking)
```
┌─────────────────────────────────────┐
│ HAND 👋  [====]  FPS:30             │  Status bar
├─────────────────────────────────────┤
│                                     │
│        ┌───────┐   ┌───────┐        │
│        │  👁️   │   │  👁️   │        │  Robot Eyes
│        └───────┘   └───────┘        │
│             ╲___😊___╱              │  Mouth (finger count)
│                                     │
├─────────────────────────────────────┤
│ ✋ Fingers: 3    Mood: HAPPY        │  Hand info
└─────────────────────────────────────┘
```

#### Mode B: Detection Display (IMX500)
```
┌─────────────────────────────────────┐
│ DETECT 🔍  [====]  FPS:25           │  Status bar
├─────────────────────────────────────┤
│                                     │
│        ┌───────┐   ┌───────┐        │
│        │  👁️   │   │  👁️   │        │  Eyes looking at object
│        └───────┘   └───────┘        │
│                                     │
│                                     │
├─────────────────────────────────────┤
│ 🧑 person    ██████████ 95%         │  Primary detection
│ 🐱 cat       ████████░░ 82%         │  Secondary detection
└─────────────────────────────────────┘
```

#### Mode C: Tracking Mode (IMX500)
```
┌─────────────────────────────────────┐
│ TRACK 🎯  Target: person            │  Status bar with target
├─────────────────────────────────────┤
│                                     │
│        ┌───────┐   ┌───────┐        │
│        │ →👁️   │   │ →👁️   │        │  Eyes following target
│        └───────┘   └───────┘        │
│                                     │
│                                     │
├─────────────────────────────────────┤
│ 🎯 Tracking: person                 │  Target info
│    Position: ← CENTER →             │  Position indicator
└─────────────────────────────────────┘
```

---

## 4. Beautiful Text Rendering Solution

### 4.1 Font Strategy

```python
# text_renderer.py

class TextRenderer:
    """Beautiful text rendering for small displays"""

    # Font sizes optimized for 160x128
    FONT_SIZES = {
        'tiny': 8,      # Status indicators
        'small': 10,    # Secondary text
        'medium': 12,   # Primary text
        'large': 16,    # Headers
    }

    # Thai-compatible fonts
    FONTS = {
        'default': 'assets/fonts/NotoSansThai-Regular.ttf',
        'bold': 'assets/fonts/NotoSansThai-Bold.ttf',
        'mono': 'assets/fonts/NotoSansMono-Regular.ttf',
    }
```

### 4.2 Text Styling Components

```python
class TextStyle:
    """Text styling options"""

    def __init__(self,
                 font_size: str = 'small',
                 color: tuple = (255, 255, 255),
                 bg_color: tuple = None,
                 shadow: bool = False,
                 outline: bool = False,
                 align: str = 'left',  # left, center, right
                 padding: int = 2):
        ...

# Pre-defined styles
STYLES = {
    'header': TextStyle(font_size='medium', color=WHITE, shadow=True),
    'detection': TextStyle(font_size='small', color=CYAN, outline=True),
    'confidence': TextStyle(font_size='tiny', color=GREEN),
    'warning': TextStyle(font_size='small', color=YELLOW, bg_color=DARK_RED),
}
```

### 4.3 UI Components

```python
# ui_components.py

class DetectionLabel(UIComponent):
    """Beautiful detection label with icon and progress bar"""

    def __init__(self, class_name: str, confidence: float, icon: str = None):
        self.class_name = class_name
        self.confidence = confidence
        self.icon = icon or self._get_icon(class_name)

    def render(self, surface, x, y, width):
        # ┌────────────────────────────────┐
        # │ 🧑 person    ██████████ 95%    │
        # └────────────────────────────────┘

        # Draw icon
        self._draw_icon(surface, x, y)

        # Draw class name
        self._draw_text(surface, self.class_name, x + 16, y)

        # Draw confidence bar
        bar_width = int((width - 80) * self.confidence)
        self._draw_progress_bar(surface, x + 70, y, bar_width)

        # Draw percentage
        self._draw_text(surface, f"{int(self.confidence*100)}%", x + width - 25, y)


class StatusBar(UIComponent):
    """Status bar with mode, FPS, and indicators"""

    def render(self, surface, mode, fps, target=None):
        # Mode icon
        icons = {'HAND': '👋', 'DETECT': '🔍', 'TRACK': '🎯', 'DEMO': '🎮'}
        ...


class PositionIndicator(UIComponent):
    """Visual position indicator"""

    def render(self, surface, x, y, position_x, position_y):
        # Shows where the tracked object is
        # ← [===●===] →
        # ↑ [===●===] ↓
        ...
```

### 4.4 Icon Mapping

```python
# Object class to emoji/icon mapping
DETECTION_ICONS = {
    # People
    'person': '🧑',

    # Animals
    'cat': '🐱',
    'dog': '🐕',
    'bird': '🐦',
    'horse': '🐴',
    'cow': '🐄',

    # Vehicles
    'car': '🚗',
    'bicycle': '🚲',
    'motorcycle': '🏍️',
    'bus': '🚌',
    'truck': '🚚',

    # Objects
    'bottle': '🍼',
    'cup': '☕',
    'chair': '🪑',
    'laptop': '💻',
    'phone': '📱',
    'book': '📚',

    # Default
    'default': '📦',
}
```

---

## 5. Mode Switching Solution

### 5.1 Problem Analysis

**Current Issue:**
- Hand tracking ใช้ MediaPipe บน IMX708
- Object detection ใช้ YOLO บน IMX500
- ไม่สามารถ hot-swap ได้เพราะต้อง reinitialize camera/model

**Constraints:**
- IMX500 ต้องใช้สำหรับ Object Detection เท่านั้น (hardware acceleration)
- IMX708 ใช้สำหรับ Hand Tracking (MediaPipe optimized)
- Memory limited บน Pi 5

### 5.2 Proposed Solutions

#### Solution A: Dual Camera Mode (Recommended)

```
┌─────────────────────────────────────────────────────────┐
│                    DUAL CAMERA MODE                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  IMX708 (CSI-0)              IMX500 (CSI-1)            │
│  ┌─────────────┐             ┌─────────────┐           │
│  │ Hand Track  │             │ Object Det  │           │
│  │ MediaPipe   │             │ YOLO11n     │           │
│  │ Always On   │             │ Always On   │           │
│  └──────┬──────┘             └──────┬──────┘           │
│         │                           │                   │
│         └──────────┬────────────────┘                   │
│                    │                                    │
│         ┌──────────▼──────────┐                         │
│         │    Mode Manager     │                         │
│         │  - Priority logic   │                         │
│         │  - Data fusion      │                         │
│         └──────────┬──────────┘                         │
│                    │                                    │
│         ┌──────────▼──────────┐                         │
│         │   Display Output    │                         │
│         │  ST7735S / HDMI     │                         │
│         └─────────────────────┘                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Implementation:**

```python
# mode_manager.py

class ModeManager:
    """Manages mode switching between Hand and Detection"""

    # Modes
    MODE_HAND = 'hand'           # Hand tracking priority
    MODE_DETECT = 'detect'       # Object detection (show all)
    MODE_TRACK = 'track'         # Track specific object
    MODE_AUTO = 'auto'           # Auto-switch based on input

    def __init__(self, hand_tracker, yolo_tracker):
        self.hand_tracker = hand_tracker   # IMX708
        self.yolo_tracker = yolo_tracker   # IMX500
        self.current_mode = self.MODE_AUTO
        self.track_target = 'person'

        # Both trackers run continuously
        self._start_trackers()

    def _start_trackers(self):
        """Start both trackers in parallel"""
        # Hand tracker runs in main thread (MediaPipe)
        # YOLO tracker runs in daemon thread (existing)
        self.hand_tracker.start()
        self.yolo_tracker.start()

    def update(self) -> TrackerResult:
        """Get tracking result based on current mode"""

        hand_result = self.hand_tracker.get_result()
        yolo_result = self.yolo_tracker.get_result()

        if self.current_mode == self.MODE_HAND:
            return hand_result if hand_result.detected else None

        elif self.current_mode == self.MODE_DETECT:
            return yolo_result

        elif self.current_mode == self.MODE_TRACK:
            return self._filter_target(yolo_result, self.track_target)

        elif self.current_mode == self.MODE_AUTO:
            # Priority: Hand > Specific Target > Any Detection
            if hand_result.detected:
                return hand_result
            elif yolo_result.has_target(self.track_target):
                return yolo_result.get_target(self.track_target)
            else:
                return yolo_result

    def switch_mode(self, new_mode: str):
        """Switch mode without restarting cameras"""
        self.current_mode = new_mode
        # No camera restart needed!
```

**Pros:**
- Instant mode switching (no delay)
- Both cameras always ready
- Can combine data from both sources

**Cons:**
- Higher power consumption
- Uses both CSI ports

---

#### Solution B: Sequential Mode Switch (Alternative)

```
┌─────────────────────────────────────────────────────────┐
│                 SEQUENTIAL MODE SWITCH                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Active Camera                    Inactive Camera       │
│  ┌─────────────┐                 ┌─────────────┐        │
│  │   IMX708    │     Switch      │   IMX500    │        │
│  │ Hand Track  │ ◄───────────► │ Object Det  │        │
│  │  Running    │                 │  Stopped    │        │
│  └─────────────┘                 └─────────────┘        │
│                                                         │
│  Switch Process:                                        │
│  1. Stop current tracker (0.5s)                         │
│  2. Release camera resources                            │
│  3. Initialize new camera (1-2s for IMX500)             │
│  4. Load model if needed (2-3s for YOLO)                │
│  5. Start new tracker                                   │
│                                                         │
│  Total switch time: 3-6 seconds                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Implementation:**

```python
# mode_manager.py (Sequential version)

class SequentialModeManager:
    """Sequential mode switching (one camera at a time)"""

    def __init__(self):
        self.active_tracker = None
        self.current_mode = None
        self.switching = False

    async def switch_mode(self, new_mode: str):
        """Switch mode with loading animation"""
        if self.switching:
            return

        self.switching = True

        # Show loading animation on display
        self._show_loading_animation(f"Switching to {new_mode}...")

        # Stop current tracker
        if self.active_tracker:
            await self._stop_tracker(self.active_tracker)

        # Initialize new tracker
        if new_mode in ['hand']:
            self.active_tracker = await self._init_hand_tracker()
        elif new_mode in ['detect', 'track']:
            self.active_tracker = await self._init_yolo_tracker()

        self.current_mode = new_mode
        self.switching = False

    def _show_loading_animation(self, message: str):
        """Show loading spinner on ST7735S"""
        # Robot eyes show "loading" expression
        # Progress bar or spinner
        # Message text
        ...
```

**Pros:**
- Lower memory usage
- One camera at a time

**Cons:**
- 3-6 second delay on switch
- IMX500 YOLO model load is slow

---

#### Solution C: Hybrid Mode (Best of Both)

```
┌─────────────────────────────────────────────────────────┐
│                      HYBRID MODE                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │              YOLO Tracker (Always On)             │  │
│  │                    IMX500                          │  │
│  │  - Object detection runs continuously             │  │
│  │  - Low power mode when not primary                │  │
│  │  - Model stays loaded in memory                   │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │              Hand Tracker (On Demand)             │  │
│  │                    IMX708                          │  │
│  │  - Start when MODE_HAND activated                 │  │
│  │  - Stop when MODE_DETECT activated                │  │
│  │  - Quick start (MediaPipe loads fast)             │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  Switch Times:                                          │
│  - HAND → DETECT: Instant (YOLO always ready)          │
│  - DETECT → HAND: ~1 second (MediaPipe init)           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Rationale:**
- IMX500 YOLO takes 2-3 seconds to load model → keep running
- MediaPipe on IMX708 loads fast (~1 second) → can stop/start

---

### 5.3 Recommended Solution: Dual Camera Mode (A)

**Why:**
1. **Zero delay switching** - Both cameras ready
2. **Auto mode possible** - Can intelligently combine inputs
3. **Best user experience** - No loading screens
4. **Hardware utilization** - Both CSI ports used effectively

**Resource Management:**

```python
# Resource usage estimation
# IMX500 YOLO: ~300MB RAM (model + inference)
# IMX708 MediaPipe: ~150MB RAM (model + buffers)
# Display: ~50MB RAM (surfaces, fonts)
# Total: ~500MB (Pi 5 has 4-8GB)
```

---

## 6. Implementation Plan

### Phase 1: File Reorganization
```
Week 1:
[ ] Create directory structure
[ ] Move existing files to new locations
[ ] Create __init__.py files
[ ] Update import statements
[ ] Test existing functionality
```

### Phase 2: Mode Manager
```
Week 2:
[ ] Implement ModeManager class
[ ] Implement TrackerInterface
[ ] Modify HandTracker to implement interface
[ ] Modify YoloTracker to implement interface
[ ] Add mode switching logic
```

### Phase 3: Display Improvements
```
Week 3:
[ ] Create TextRenderer class
[ ] Create UIComponents
[ ] Design detection labels
[ ] Implement beautiful text styles
[ ] Add Thai font support
```

### Phase 4: Integration & Testing
```
Week 4:
[ ] Integrate all components
[ ] Test mode switching
[ ] Performance optimization
[ ] Documentation
[ ] User testing
```

---

## 7. API Reference

### 7.1 ModeManager

```python
class ModeManager:
    """Central mode management for NumBot"""

    # Constants
    MODE_HAND = 'hand'
    MODE_DETECT = 'detect'
    MODE_TRACK = 'track'
    MODE_AUTO = 'auto'

    # Methods
    def switch_mode(mode: str) -> None
    def get_mode() -> str
    def set_track_target(target: str) -> None
    def update() -> TrackerResult
    def is_hand_detected() -> bool
    def is_object_detected() -> bool
    def get_detections() -> List[Detection]
```

### 7.2 TextRenderer

```python
class TextRenderer:
    """Beautiful text rendering for ST7735S"""

    def draw_text(surface, text, x, y, style: TextStyle) -> None
    def draw_label(surface, icon, text, x, y, width) -> None
    def draw_detection(surface, detection: Detection, x, y) -> None
    def draw_progress_bar(surface, x, y, width, value, max_value) -> None
```

### 7.3 DisplayRenderer

```python
class DisplayRenderer:
    """Main display rendering controller"""

    def render_hand_mode(hand_result: TrackerResult) -> None
    def render_detect_mode(detections: List[Detection]) -> None
    def render_track_mode(target: Detection) -> None
    def render_loading(message: str) -> None
    def render_error(message: str) -> None
```

---

## 8. Configuration

### 8.1 New Config Options

```python
# config.py additions

# Mode Settings
DEFAULT_MODE = 'auto'           # hand, detect, track, auto
DUAL_CAMERA_MODE = True         # Use both cameras simultaneously

# Display Text Settings
TEXT_FONT_PATH = 'assets/fonts/NotoSansThai-Regular.ttf'
TEXT_SHOW_ICONS = True
TEXT_SHOW_CONFIDENCE = True
TEXT_MAX_DETECTIONS = 3         # Max detections to show on ST7735S

# Detection Settings
DETECTION_CONFIDENCE_THRESHOLD = 0.5
TRACK_TARGET_DEFAULT = 'person'
DETECTION_REFRESH_RATE = 10     # Hz

# Mode Switch Settings
AUTO_MODE_HAND_PRIORITY = True  # Hand detection takes priority in auto mode
MODE_SWITCH_ANIMATION = True    # Show animation during mode switch
```

---

## 9. Example Usage

### 9.1 Basic Usage

```bash
# Run with default settings (auto mode)
python3 main_roboeyes.py

# Run in detection mode only
python3 main_roboeyes.py --mode detect

# Run in hand tracking mode only
python3 main_roboeyes.py --mode hand

# Track specific object
python3 main_roboeyes.py --mode track --target cat
```

### 9.2 Runtime Commands

| Key | Action |
|-----|--------|
| `H` | Switch to HAND mode |
| `D` | Switch to DETECT mode |
| `T` | Switch to TRACK mode |
| `A` | Switch to AUTO mode |
| `1-9` | Select track target (person, cat, dog, etc.) |
| `SPACE` | Random mood |
| `ESC` | Exit |

### 9.3 Voice Commands

| Thai | English | Action |
|------|---------|--------|
| "มือ" | "hand" | Switch to HAND mode |
| "ตรวจจับ" | "detect" | Switch to DETECT mode |
| "ติดตาม" | "track" | Switch to TRACK mode |
| "คน" | "person" | Track person |
| "แมว" | "cat" | Track cat |

---

## 10. Summary

### Key Decisions

1. **Dual Camera Architecture** - ใช้ทั้ง IMX500 และ IMX708 พร้อมกัน
2. **IMX500 for Objects Only** - Object detection ด้วย YOLO
3. **File Separation** - แยก code เป็น modules ตาม function
4. **Beautiful Text** - ใช้ Thai font + icons + progress bars
5. **Instant Mode Switch** - ไม่ต้องรอ load camera/model

### Benefits

- ✅ Zero-delay mode switching
- ✅ Clean code architecture
- ✅ Beautiful display output
- ✅ Thai language support
- ✅ Extensible design

---

*Document Version: 1.1*
*Last Updated: 2026-01-24*
*Status: IMX500 Object Detection Implemented with yolo_tracker_v2*
