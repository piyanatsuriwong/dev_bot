# Implementation Guide: IMX500 Detection Feature

## Quick Reference

### ไฟล์ที่ต้องสร้างใหม่

```
numbot/
├── core/
│   ├── __init__.py
│   ├── mode_manager.py      # Mode switching logic
│   └── tracker_result.py    # Unified result class
│
├── display/
│   ├── __init__.py
│   ├── text_renderer.py     # Beautiful text
│   └── ui_components.py     # Detection labels, status bar
│
└── assets/
    └── fonts/
        └── NotoSansThai-Regular.ttf
```

---

## 1. TrackerResult (Unified Result Class)

```python
# core/tracker_result.py

from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class Detection:
    """Single detection result"""
    class_name: str           # e.g., "person", "cat"
    confidence: float         # 0.0 - 1.0
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    center: Tuple[float, float]      # normalized -1 to 1

    @property
    def icon(self) -> str:
        """Get emoji icon for class"""
        ICONS = {
            'person': '🧑', 'cat': '🐱', 'dog': '🐕',
            'bird': '🐦', 'car': '🚗', 'bicycle': '🚲',
            'bottle': '🍼', 'cup': '☕', 'laptop': '💻',
        }
        return ICONS.get(self.class_name, '📦')


@dataclass
class TrackerResult:
    """Unified result from any tracker"""
    source: str               # 'hand' or 'yolo'
    detected: bool
    position: Tuple[float, float]  # normalized -1 to 1
    detections: List[Detection]
    finger_count: Optional[int] = None  # For hand tracker
    frame: Optional[any] = None         # Camera frame

    @classmethod
    def empty(cls, source: str = 'none'):
        return cls(source=source, detected=False,
                   position=(0, 0), detections=[])
```

---

## 2. ModeManager Implementation

```python
# core/mode_manager.py

import threading
from enum import Enum
from typing import Optional
from .tracker_result import TrackerResult, Detection


class Mode(Enum):
    HAND = 'hand'
    DETECT = 'detect'
    TRACK = 'track'
    AUTO = 'auto'


class ModeManager:
    """
    Manages mode switching between Hand and Object detection.
    Both trackers run continuously for instant switching.
    """

    def __init__(self, hand_tracker, yolo_tracker, config):
        self.hand_tracker = hand_tracker
        self.yolo_tracker = yolo_tracker
        self.config = config

        self._mode = Mode.AUTO
        self._track_target = config.TRACK_TARGET_DEFAULT
        self._lock = threading.Lock()

        # State
        self._last_hand_result: Optional[TrackerResult] = None
        self._last_yolo_result: Optional[TrackerResult] = None

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def track_target(self) -> str:
        return self._track_target

    def switch_mode(self, new_mode: Mode) -> None:
        """Switch mode instantly (no camera restart needed)"""
        with self._lock:
            self._mode = new_mode
            print(f"[ModeManager] Switched to {new_mode.value}")

    def set_track_target(self, target: str) -> None:
        """Set target class for TRACK mode"""
        with self._lock:
            self._track_target = target
            print(f"[ModeManager] Track target: {target}")

    def update(self) -> TrackerResult:
        """
        Get tracking result based on current mode.
        Called every frame from main loop.
        """
        # Get results from both trackers
        hand_result = self._get_hand_result()
        yolo_result = self._get_yolo_result()

        with self._lock:
            mode = self._mode

        if mode == Mode.HAND:
            return hand_result if hand_result.detected else TrackerResult.empty('hand')

        elif mode == Mode.DETECT:
            return yolo_result

        elif mode == Mode.TRACK:
            return self._filter_by_target(yolo_result)

        elif mode == Mode.AUTO:
            return self._auto_select(hand_result, yolo_result)

        return TrackerResult.empty()

    def _get_hand_result(self) -> TrackerResult:
        """Get result from hand tracker"""
        if self.hand_tracker is None:
            return TrackerResult.empty('hand')

        result = self.hand_tracker.update()
        if result:
            self._last_hand_result = result
        return self._last_hand_result or TrackerResult.empty('hand')

    def _get_yolo_result(self) -> TrackerResult:
        """Get result from YOLO tracker"""
        if self.yolo_tracker is None:
            return TrackerResult.empty('yolo')

        position = self.yolo_tracker.get_normalized_position()
        detections = self.yolo_tracker.get_detections()

        result = TrackerResult(
            source='yolo',
            detected=len(detections) > 0,
            position=position or (0, 0),
            detections=[
                Detection(
                    class_name=d['class'],
                    confidence=d['confidence'],
                    bbox=d['bbox'],
                    center=d['center']
                ) for d in detections
            ]
        )
        self._last_yolo_result = result
        return result

    def _filter_by_target(self, yolo_result: TrackerResult) -> TrackerResult:
        """Filter detections to only include target class"""
        target_detections = [
            d for d in yolo_result.detections
            if d.class_name == self._track_target
        ]

        if not target_detections:
            return TrackerResult.empty('yolo')

        # Use first matching detection as position
        primary = target_detections[0]
        return TrackerResult(
            source='yolo',
            detected=True,
            position=primary.center,
            detections=target_detections
        )

    def _auto_select(self, hand_result: TrackerResult,
                     yolo_result: TrackerResult) -> TrackerResult:
        """
        Auto mode: Hand has priority, then track target, then any detection.
        """
        # Priority 1: Hand detected
        if hand_result.detected:
            return hand_result

        # Priority 2: Track target detected
        target_result = self._filter_by_target(yolo_result)
        if target_result.detected:
            return target_result

        # Priority 3: Any detection
        return yolo_result
```

---

## 3. TextRenderer Implementation

```python
# display/text_renderer.py

import pygame
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class TextStyle:
    """Text styling options"""
    font_size: int = 10
    color: Tuple[int, int, int] = (255, 255, 255)
    bg_color: Optional[Tuple[int, int, int]] = None
    shadow: bool = False
    shadow_color: Tuple[int, int, int] = (0, 0, 0)
    shadow_offset: int = 1
    outline: bool = False
    outline_color: Tuple[int, int, int] = (0, 0, 0)


# Pre-defined styles
class Styles:
    HEADER = TextStyle(font_size=12, color=(255, 255, 255), shadow=True)
    DETECTION = TextStyle(font_size=10, color=(0, 255, 255), outline=True)
    CONFIDENCE = TextStyle(font_size=8, color=(0, 255, 0))
    WARNING = TextStyle(font_size=10, color=(255, 255, 0), bg_color=(100, 0, 0))
    MODE_ACTIVE = TextStyle(font_size=10, color=(0, 255, 0), shadow=True)
    MODE_INACTIVE = TextStyle(font_size=10, color=(128, 128, 128))


class TextRenderer:
    """Beautiful text rendering for small displays"""

    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    CYAN = (0, 255, 255)
    GREEN = (0, 255, 0)
    YELLOW = (255, 255, 0)
    RED = (255, 0, 0)
    GRAY = (128, 128, 128)

    def __init__(self, font_path: str = None):
        """
        Initialize text renderer.

        Args:
            font_path: Path to Thai-compatible font (optional)
        """
        pygame.font.init()
        self.font_path = font_path
        self._fonts = {}

    def get_font(self, size: int) -> pygame.font.Font:
        """Get cached font of specified size"""
        if size not in self._fonts:
            if self.font_path:
                try:
                    self._fonts[size] = pygame.font.Font(self.font_path, size)
                except:
                    self._fonts[size] = pygame.font.Font(None, size)
            else:
                self._fonts[size] = pygame.font.Font(None, size)
        return self._fonts[size]

    def draw_text(self, surface: pygame.Surface, text: str,
                  x: int, y: int, style: TextStyle = None) -> pygame.Rect:
        """
        Draw styled text on surface.

        Returns:
            Bounding rect of drawn text
        """
        if style is None:
            style = TextStyle()

        font = self.get_font(style.font_size)

        # Draw background if specified
        if style.bg_color:
            text_surface = font.render(text, True, style.color)
            padding = 2
            bg_rect = pygame.Rect(
                x - padding, y - padding,
                text_surface.get_width() + padding * 2,
                text_surface.get_height() + padding * 2
            )
            pygame.draw.rect(surface, style.bg_color, bg_rect)

        # Draw shadow
        if style.shadow:
            shadow_surface = font.render(text, True, style.shadow_color)
            surface.blit(shadow_surface,
                        (x + style.shadow_offset, y + style.shadow_offset))

        # Draw outline
        if style.outline:
            outline_surface = font.render(text, True, style.outline_color)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                surface.blit(outline_surface, (x + dx, y + dy))

        # Draw main text
        text_surface = font.render(text, True, style.color)
        surface.blit(text_surface, (x, y))

        return text_surface.get_rect(topleft=(x, y))

    def draw_text_centered(self, surface: pygame.Surface, text: str,
                           y: int, style: TextStyle = None) -> pygame.Rect:
        """Draw text centered horizontally"""
        font = self.get_font((style or TextStyle()).font_size)
        text_surface = font.render(text, True, (255, 255, 255))
        x = (surface.get_width() - text_surface.get_width()) // 2
        return self.draw_text(surface, text, x, y, style)

    def draw_progress_bar(self, surface: pygame.Surface,
                          x: int, y: int, width: int, height: int,
                          value: float, max_value: float = 1.0,
                          fg_color: Tuple[int, int, int] = GREEN,
                          bg_color: Tuple[int, int, int] = GRAY) -> None:
        """
        Draw a progress bar.

        Args:
            value: Current value (0 to max_value)
            max_value: Maximum value (default 1.0)
        """
        # Background
        pygame.draw.rect(surface, bg_color, (x, y, width, height))

        # Foreground (filled portion)
        fill_width = int(width * (value / max_value))
        if fill_width > 0:
            pygame.draw.rect(surface, fg_color, (x, y, fill_width, height))

        # Border
        pygame.draw.rect(surface, self.WHITE, (x, y, width, height), 1)
```

---

## 4. UI Components Implementation

```python
# display/ui_components.py

import pygame
from typing import List, Tuple, Optional
from .text_renderer import TextRenderer, TextStyle, Styles
from core.tracker_result import Detection


class StatusBar:
    """Status bar component for top of screen"""

    HEIGHT = 12

    def __init__(self, text_renderer: TextRenderer, width: int):
        self.renderer = text_renderer
        self.width = width

    def draw(self, surface: pygame.Surface, mode: str, fps: int,
             target: str = None) -> None:
        """Draw status bar"""
        y = 0

        # Mode icons
        mode_icons = {
            'hand': '👋 HAND',
            'detect': '🔍 DETECT',
            'track': '🎯 TRACK',
            'auto': '🔄 AUTO',
        }

        # Draw mode
        mode_text = mode_icons.get(mode, mode.upper())
        self.renderer.draw_text(surface, mode_text, 2, y, Styles.MODE_ACTIVE)

        # Draw target if tracking
        if mode == 'track' and target:
            self.renderer.draw_text(surface, f"→{target}", 70, y, Styles.DETECTION)

        # Draw FPS
        fps_text = f"FPS:{fps}"
        self.renderer.draw_text(surface, fps_text, self.width - 40, y, Styles.CONFIDENCE)


class DetectionLabel:
    """Single detection label with icon and confidence bar"""

    HEIGHT = 14

    def __init__(self, text_renderer: TextRenderer):
        self.renderer = text_renderer

    def draw(self, surface: pygame.Surface, detection: Detection,
             x: int, y: int, width: int) -> int:
        """
        Draw detection label.

        Returns:
            Height of drawn label
        """
        # Icon and class name
        label = f"{detection.icon} {detection.class_name}"
        self.renderer.draw_text(surface, label, x, y, Styles.DETECTION)

        # Confidence bar
        bar_x = x + 60
        bar_width = width - 95
        bar_height = 6
        bar_y = y + 4

        self.renderer.draw_progress_bar(
            surface, bar_x, bar_y, bar_width, bar_height,
            detection.confidence,
            fg_color=(0, 255, 0) if detection.confidence > 0.7 else (255, 255, 0)
        )

        # Confidence percentage
        conf_text = f"{int(detection.confidence * 100)}%"
        self.renderer.draw_text(surface, conf_text, x + width - 28, y, Styles.CONFIDENCE)

        return self.HEIGHT


class DetectionPanel:
    """Panel showing multiple detections"""

    def __init__(self, text_renderer: TextRenderer, max_detections: int = 3):
        self.renderer = text_renderer
        self.label = DetectionLabel(text_renderer)
        self.max_detections = max_detections

    def draw(self, surface: pygame.Surface, detections: List[Detection],
             x: int, y: int, width: int) -> int:
        """
        Draw detection panel.

        Returns:
            Total height drawn
        """
        if not detections:
            # No detections
            self.renderer.draw_text(surface, "No objects detected",
                                   x, y, Styles.MODE_INACTIVE)
            return 14

        total_height = 0
        for i, detection in enumerate(detections[:self.max_detections]):
            h = self.label.draw(surface, detection, x, y + total_height, width)
            total_height += h

        return total_height


class PositionIndicator:
    """Visual indicator showing tracked position"""

    def __init__(self, text_renderer: TextRenderer):
        self.renderer = text_renderer

    def draw(self, surface: pygame.Surface, x: int, y: int,
             width: int, position: Tuple[float, float]) -> None:
        """
        Draw position indicator.

        Args:
            position: Normalized position (-1 to 1)
        """
        px, py = position

        # Horizontal indicator: ← [===●===] →
        bar_width = width - 20
        bar_x = x + 10
        bar_y = y

        # Draw track
        pygame.draw.line(surface, (128, 128, 128),
                        (bar_x, bar_y + 4), (bar_x + bar_width, bar_y + 4), 2)

        # Draw position marker
        marker_x = bar_x + int((px + 1) / 2 * bar_width)
        pygame.draw.circle(surface, (0, 255, 255), (marker_x, bar_y + 4), 4)

        # Draw arrows
        self.renderer.draw_text(surface, "←", x, y, Styles.MODE_INACTIVE)
        self.renderer.draw_text(surface, "→", x + width - 10, y, Styles.MODE_INACTIVE)
```

---

## 5. Display Renderer Integration

```python
# display/display_renderer.py

import pygame
from typing import List, Optional
from .text_renderer import TextRenderer, Styles
from .ui_components import StatusBar, DetectionPanel, PositionIndicator
from core.tracker_result import TrackerResult, Detection


class DisplayRenderer:
    """Main display rendering controller for ST7735S"""

    def __init__(self, width: int, height: int, font_path: str = None):
        self.width = width
        self.height = height

        # Initialize components
        self.text = TextRenderer(font_path)
        self.status_bar = StatusBar(self.text, width)
        self.detection_panel = DetectionPanel(self.text, max_detections=2)
        self.position_indicator = PositionIndicator(self.text)

        # Layout zones
        self.STATUS_HEIGHT = 12
        self.INFO_HEIGHT = 36
        self.FACE_HEIGHT = height - self.STATUS_HEIGHT - self.INFO_HEIGHT

    def render_frame(self, surface: pygame.Surface, mode: str, fps: int,
                     result: TrackerResult, robo_surface: pygame.Surface = None,
                     track_target: str = None) -> None:
        """
        Render complete frame to surface.

        Args:
            surface: Target pygame surface (ST7735S buffer)
            mode: Current mode ('hand', 'detect', 'track', 'auto')
            fps: Current FPS
            result: Tracking result
            robo_surface: Pre-rendered robot eyes surface
            track_target: Target class for track mode
        """
        # Clear surface
        surface.fill((0, 0, 0))

        # Zone 1: Status bar
        self.status_bar.draw(surface, mode, fps, track_target)

        # Zone 2: Robot eyes (face area)
        if robo_surface:
            face_y = self.STATUS_HEIGHT
            # Center the robot face
            face_x = (self.width - robo_surface.get_width()) // 2
            surface.blit(robo_surface, (face_x, face_y))

        # Zone 3: Info panel
        info_y = self.height - self.INFO_HEIGHT

        if mode == 'hand' and result.source == 'hand':
            self._render_hand_info(surface, info_y, result)
        elif mode in ['detect', 'track', 'auto']:
            self._render_detection_info(surface, info_y, result, mode)

    def _render_hand_info(self, surface: pygame.Surface, y: int,
                          result: TrackerResult) -> None:
        """Render hand tracking info"""
        x = 2
        width = self.width - 4

        if result.detected:
            # Finger count
            finger_text = f"✋ Fingers: {result.finger_count}"
            self.text.draw_text(surface, finger_text, x, y, Styles.DETECTION)

            # Position indicator
            self.position_indicator.draw(surface, x, y + 16, width, result.position)
        else:
            self.text.draw_text(surface, "No hand detected", x, y, Styles.MODE_INACTIVE)

    def _render_detection_info(self, surface: pygame.Surface, y: int,
                               result: TrackerResult, mode: str) -> None:
        """Render object detection info"""
        x = 2
        width = self.width - 4

        if result.detected and result.detections:
            self.detection_panel.draw(surface, result.detections, x, y, width)
        else:
            msg = "Searching..." if mode == 'track' else "No objects"
            self.text.draw_text(surface, msg, x, y, Styles.MODE_INACTIVE)

    def render_loading(self, surface: pygame.Surface, message: str) -> None:
        """Render loading screen"""
        surface.fill((0, 0, 0))
        self.text.draw_text_centered(surface, "Loading...",
                                     self.height // 2 - 10, Styles.HEADER)
        self.text.draw_text_centered(surface, message,
                                     self.height // 2 + 5, Styles.MODE_INACTIVE)

    def render_error(self, surface: pygame.Surface, message: str) -> None:
        """Render error screen"""
        surface.fill((50, 0, 0))
        self.text.draw_text_centered(surface, "Error",
                                     self.height // 2 - 10, Styles.WARNING)
        self.text.draw_text_centered(surface, message,
                                     self.height // 2 + 5, Styles.MODE_INACTIVE)
```

---

## 6. Main Integration Example

```python
# main_roboeyes.py (simplified example)

import pygame
from core.mode_manager import ModeManager, Mode
from display.display_renderer import DisplayRenderer
from tracking.hand_tracker import HandTracker
from tracking.yolo_tracker import YoloTracker
from animation.roboeyes import RoboEyes
import config


class NumBotApp:
    def __init__(self):
        pygame.init()

        # Initialize display
        self.display = ST7735Display()
        self.renderer = DisplayRenderer(
            config.SCREEN_WIDTH,
            config.SCREEN_HEIGHT,
            config.TEXT_FONT_PATH
        )

        # Initialize trackers
        self.hand_tracker = HandTracker() if config.DUAL_CAMERA_MODE else None
        self.yolo_tracker = YoloTracker()

        # Initialize mode manager
        self.mode_manager = ModeManager(
            self.hand_tracker,
            self.yolo_tracker,
            config
        )

        # Initialize robot eyes
        self.robo = RoboEyes(config.SCREEN_WIDTH, config.FACE_HEIGHT)

        # State
        self.running = True
        self.fps = 0

    def run(self):
        clock = pygame.time.Clock()

        while self.running:
            # Handle events
            self._handle_events()

            # Update tracking
            result = self.mode_manager.update()

            # Update robot eyes
            if result.detected:
                self.robo.set_position(result.position[0], result.position[1])
            self.robo.update()

            # Render to display buffer
            self.renderer.render_frame(
                self.display.surface,
                self.mode_manager.mode.value,
                self.fps,
                result,
                self.robo.get_surface(),
                self.mode_manager.track_target
            )

            # Send to ST7735S
            self.display.update()

            # FPS control
            clock.tick(30)
            self.fps = int(clock.get_fps())

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_h:
                    self.mode_manager.switch_mode(Mode.HAND)
                elif event.key == pygame.K_d:
                    self.mode_manager.switch_mode(Mode.DETECT)
                elif event.key == pygame.K_t:
                    self.mode_manager.switch_mode(Mode.TRACK)
                elif event.key == pygame.K_a:
                    self.mode_manager.switch_mode(Mode.AUTO)


if __name__ == '__main__':
    app = NumBotApp()
    app.run()
```

---

## 7. Quick Start Checklist

### Step 1: Create directories
```bash
mkdir -p numbot/core numbot/display numbot/tracking numbot/animation numbot/hardware
mkdir -p numbot/assets/fonts
```

### Step 2: Download Thai font
```bash
# Download NotoSansThai
wget -O numbot/assets/fonts/NotoSansThai-Regular.ttf \
  "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansThai/NotoSansThai-Regular.ttf"
```

### Step 3: Create `__init__.py` files
```bash
touch numbot/core/__init__.py
touch numbot/display/__init__.py
touch numbot/tracking/__init__.py
touch numbot/animation/__init__.py
touch numbot/hardware/__init__.py
```

### Step 4: Implement core classes
1. `core/tracker_result.py` - Data classes
2. `core/mode_manager.py` - Mode switching
3. `display/text_renderer.py` - Text rendering
4. `display/ui_components.py` - UI components
5. `display/display_renderer.py` - Main renderer

### Step 5: Update main_roboeyes.py
- Import new modules
- Use ModeManager for mode switching
- Use DisplayRenderer for rendering

### Step 6: Test
```bash
# Test in demo mode first
python3 main_roboeyes.py --mode demo

# Test with hand tracking
python3 main_roboeyes.py --mode hand

# Test with detection
python3 main_roboeyes.py --mode detect
```

---

*Implementation Guide Version: 1.0*
