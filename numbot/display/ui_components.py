"""
Reusable UI components for ST7735S display (160x128).
Components: StatusBar, DetectionLabel, DetectionPanel, PositionIndicator, etc.
"""

import pygame
from typing import List, Tuple, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .text_renderer import TextRenderer, TextStyle, Styles, Colors

# Import Detection from core (handle both import styles)
try:
    from core.tracker_result import Detection
except ImportError:
    from ..core.tracker_result import Detection


class StatusBar:
    """
    Status bar component for top of screen.
    Shows mode, target, and FPS.

    Layout (160px width):
    [MODE] [TARGET/INFO] [FPS]
    """

    HEIGHT = 12
    BG_COLOR = Colors.BG_STATUS

    # Mode display info
    MODE_INFO = {
        'hand': {'text': 'HAND', 'icon': 'H', 'color': Colors.HAND_COLOR},
        'detect': {'text': 'DETECT', 'icon': 'D', 'color': Colors.DETECT_COLOR},
        'track': {'text': 'TRACK', 'icon': 'T', 'color': Colors.TRACK_COLOR},
        'auto': {'text': 'AUTO', 'icon': 'A', 'color': Colors.AUTO_COLOR},
        'demo': {'text': 'DEMO', 'icon': '>', 'color': Colors.GRAY},
    }

    def __init__(self, renderer: TextRenderer, width: int):
        """
        Initialize status bar.

        Args:
            renderer: TextRenderer instance
            width: Screen width
        """
        self.renderer = renderer
        self.width = width

    def draw(self, surface: pygame.Surface, mode: str, fps: int,
             target: str = None, extra_info: str = None) -> None:
        """
        Draw status bar.

        Args:
            surface: Target pygame surface
            mode: Current mode ('hand', 'detect', 'track', 'auto', 'demo')
            fps: Current FPS
            target: Track target (for track mode)
            extra_info: Additional info to display
        """
        y = 1

        # Draw background
        pygame.draw.rect(surface, self.BG_COLOR, (0, 0, self.width, self.HEIGHT))

        # Get mode info
        mode_info = self.MODE_INFO.get(mode, self.MODE_INFO['demo'])

        # Draw mode indicator
        mode_style = TextStyle(font_size=9, color=mode_info['color'], shadow=True)
        mode_text = f"[{mode_info['icon']}]{mode_info['text']}"
        self.renderer.draw_text(surface, mode_text, 2, y, mode_style)

        # Draw target or extra info
        if mode == 'track' and target:
            info_text = f">{target}"
            info_style = TextStyle(font_size=9, color=Colors.CYAN)
            self.renderer.draw_text(surface, info_text, 55, y, info_style)
        elif extra_info:
            info_style = TextStyle(font_size=9, color=Colors.LIGHT_GRAY)
            self.renderer.draw_text(surface, extra_info, 55, y, info_style)

        # Draw FPS (right aligned)
        fps_text = f"{fps:2d}fps"
        fps_style = TextStyle(font_size=8, color=Colors.CONF_HIGH if fps >= 25 else Colors.CONF_MED)
        self.renderer.draw_text_right(surface, fps_text, y + 1, 2, fps_style)

        # Draw separator line
        pygame.draw.line(surface, Colors.DARK_GRAY, (0, self.HEIGHT - 1),
                        (self.width, self.HEIGHT - 1), 1)


class DetectionLabel:
    """
    Single detection label with class name and confidence bar.

    Layout:
    [icon] classname  [=====] 95%
    """

    HEIGHT = 14
    ICON_WIDTH = 14
    CONF_BAR_WIDTH = 30
    CONF_TEXT_WIDTH = 25

    def __init__(self, renderer: TextRenderer):
        """
        Initialize detection label.

        Args:
            renderer: TextRenderer instance
        """
        self.renderer = renderer

    def draw(self, surface: pygame.Surface, detection: Detection,
             x: int, y: int, width: int, show_confidence: bool = True) -> int:
        """
        Draw detection label.

        Args:
            surface: Target pygame surface
            detection: Detection object
            x: X position
            y: Y position
            width: Available width
            show_confidence: Whether to show confidence bar

        Returns:
            Height of drawn label
        """
        # Draw icon/indicator
        icon_style = TextStyle(font_size=10, color=Colors.CYAN, bold=True)
        self.renderer.draw_text(surface, f"[{detection.icon}]", x, y, icon_style)

        # Calculate text width
        if show_confidence:
            text_width = width - self.ICON_WIDTH - self.CONF_BAR_WIDTH - self.CONF_TEXT_WIDTH - 8
        else:
            text_width = width - self.ICON_WIDTH - 4

        # Draw class name (truncated if needed)
        name_style = TextStyle(font_size=9, color=Colors.WHITE)
        class_name = self.renderer.truncate_text(detection.class_name, text_width, name_style)
        self.renderer.draw_text(surface, class_name, x + self.ICON_WIDTH, y + 1, name_style)

        if show_confidence:
            # Draw confidence bar
            bar_x = x + width - self.CONF_BAR_WIDTH - self.CONF_TEXT_WIDTH - 4
            bar_y = y + 4
            self.renderer.draw_confidence_bar(surface, bar_x, bar_y,
                                             self.CONF_BAR_WIDTH, detection.confidence)

            # Draw confidence percentage
            conf_color = Colors.CONF_HIGH if detection.confidence >= 0.7 else Colors.CONF_MED
            conf_style = TextStyle(font_size=8, color=conf_color)
            conf_text = f"{detection.confidence_percent}%"
            self.renderer.draw_text(surface, conf_text,
                                   x + width - self.CONF_TEXT_WIDTH, y + 1, conf_style)

        return self.HEIGHT


class DetectionPanel:
    """
    Panel showing multiple detections.
    Displays up to max_detections with labels and confidence.
    """

    def __init__(self, renderer: TextRenderer, max_detections: int = 2):
        """
        Initialize detection panel.

        Args:
            renderer: TextRenderer instance
            max_detections: Maximum detections to show
        """
        self.renderer = renderer
        self.label = DetectionLabel(renderer)
        self.max_detections = max_detections

    def draw(self, surface: pygame.Surface, detections: List[Detection],
             x: int, y: int, width: int) -> int:
        """
        Draw detection panel.

        Args:
            surface: Target pygame surface
            detections: List of Detection objects
            x: X position
            y: Y position
            width: Available width

        Returns:
            Total height drawn
        """
        if not detections:
            # No detections message
            empty_style = TextStyle(font_size=9, color=Colors.GRAY)
            self.renderer.draw_text(surface, "No objects detected", x, y, empty_style)
            return 14

        total_height = 0
        for i, detection in enumerate(detections[:self.max_detections]):
            h = self.label.draw(surface, detection, x, y + total_height, width)
            total_height += h

        # Show count if more detections
        remaining = len(detections) - self.max_detections
        if remaining > 0:
            more_style = TextStyle(font_size=8, color=Colors.GRAY)
            self.renderer.draw_text(surface, f"+{remaining} more", x, y + total_height, more_style)
            total_height += 10

        return total_height


class PositionIndicator:
    """
    Visual indicator showing tracked position.
    Displays horizontal and/or vertical position.

    Layout:
    < [====O====] >
    """

    HEIGHT = 12

    def __init__(self, renderer: TextRenderer):
        """
        Initialize position indicator.

        Args:
            renderer: TextRenderer instance
        """
        self.renderer = renderer

    def draw(self, surface: pygame.Surface, x: int, y: int,
             width: int, position: Tuple[float, float],
             show_vertical: bool = False) -> int:
        """
        Draw position indicator.

        Args:
            surface: Target pygame surface
            x: X position
            y: Y position
            width: Available width
            position: Normalized position (-1 to 1)
            show_vertical: Whether to show vertical indicator

        Returns:
            Total height drawn
        """
        px, py = position
        height = 0

        # Horizontal indicator
        self._draw_horizontal(surface, x, y, width, px)
        height += self.HEIGHT

        # Vertical indicator (if enabled)
        if show_vertical:
            self._draw_vertical(surface, x, y + height, width, py)
            height += self.HEIGHT

        return height

    def _draw_horizontal(self, surface: pygame.Surface, x: int, y: int,
                         width: int, value: float) -> None:
        """Draw horizontal position bar."""
        # Left arrow
        arrow_style = TextStyle(font_size=8, color=Colors.GRAY)
        self.renderer.draw_text(surface, "<", x, y, arrow_style)

        # Track bar
        bar_x = x + 10
        bar_width = width - 20
        bar_y = y + 4

        # Draw track
        pygame.draw.line(surface, Colors.DARK_GRAY,
                        (bar_x, bar_y), (bar_x + bar_width, bar_y), 2)

        # Draw center marker
        center_x = bar_x + bar_width // 2
        pygame.draw.line(surface, Colors.GRAY,
                        (center_x, bar_y - 2), (center_x, bar_y + 2), 1)

        # Draw position marker
        # Convert -1 to 1 range to pixel position
        marker_x = bar_x + int((value + 1) / 2 * bar_width)
        marker_x = max(bar_x, min(bar_x + bar_width, marker_x))
        pygame.draw.circle(surface, Colors.CYAN, (marker_x, bar_y), 3)

        # Right arrow
        self.renderer.draw_text(surface, ">", x + width - 8, y, arrow_style)

    def _draw_vertical(self, surface: pygame.Surface, x: int, y: int,
                       width: int, value: float) -> None:
        """Draw vertical position indicator (simplified)."""
        # Up/Down text indicator
        if value < -0.3:
            text = "UP"
            color = Colors.CYAN
        elif value > 0.3:
            text = "DOWN"
            color = Colors.CYAN
        else:
            text = "CENTER"
            color = Colors.GRAY

        style = TextStyle(font_size=8, color=color)
        self.renderer.draw_text_centered(surface, f"Y:{text}", y, style)


class HandInfoPanel:
    """
    Panel showing hand tracking information.
    Displays finger count and position.
    """

    def __init__(self, renderer: TextRenderer):
        """
        Initialize hand info panel.

        Args:
            renderer: TextRenderer instance
        """
        self.renderer = renderer
        self.position_indicator = PositionIndicator(renderer)

    def draw(self, surface: pygame.Surface, x: int, y: int, width: int,
             finger_count: int = None, position: Tuple[float, float] = None,
             detected: bool = True) -> int:
        """
        Draw hand info panel.

        Args:
            surface: Target pygame surface
            x: X position
            y: Y position
            width: Available width
            finger_count: Number of fingers detected
            position: Normalized hand position
            detected: Whether hand is detected

        Returns:
            Total height drawn
        """
        if not detected:
            no_hand_style = TextStyle(font_size=9, color=Colors.GRAY)
            self.renderer.draw_text(surface, "No hand detected", x, y, no_hand_style)
            return 14

        total_height = 0

        # Finger count display
        if finger_count is not None:
            finger_style = TextStyle(font_size=10, color=Colors.HAND_COLOR, shadow=True)

            # Create finger visualization
            fingers = "I" * finger_count + "." * (5 - finger_count)
            finger_text = f"[{fingers}] {finger_count}"

            self.renderer.draw_text(surface, finger_text, x, y, finger_style)
            total_height += 14

        # Position indicator
        if position is not None:
            h = self.position_indicator.draw(surface, x, y + total_height, width, position)
            total_height += h

        return total_height


class LoadingIndicator:
    """
    Loading indicator with spinner and message.
    """

    def __init__(self, renderer: TextRenderer):
        """
        Initialize loading indicator.

        Args:
            renderer: TextRenderer instance
        """
        self.renderer = renderer
        self.frame = 0
        self.spinner_chars = ['|', '/', '-', '\\']

    def draw(self, surface: pygame.Surface, message: str = "Loading...") -> None:
        """
        Draw loading indicator.

        Args:
            surface: Target pygame surface
            message: Loading message
        """
        width = surface.get_width()
        height = surface.get_height()

        # Clear with dark background
        surface.fill(Colors.BG_DARK)

        # Draw spinner
        spinner = self.spinner_chars[self.frame % len(self.spinner_chars)]
        spinner_style = TextStyle(font_size=16, color=Colors.CYAN)
        self.renderer.draw_text_centered(surface, spinner, height // 2 - 15, spinner_style)

        # Draw message
        msg_style = TextStyle(font_size=10, color=Colors.LIGHT_GRAY)
        self.renderer.draw_text_centered(surface, message, height // 2 + 5, msg_style)

        self.frame += 1


class ErrorDisplay:
    """
    Error message display.
    """

    def __init__(self, renderer: TextRenderer):
        """
        Initialize error display.

        Args:
            renderer: TextRenderer instance
        """
        self.renderer = renderer

    def draw(self, surface: pygame.Surface, message: str, title: str = "Error") -> None:
        """
        Draw error display.

        Args:
            surface: Target pygame surface
            message: Error message
            title: Error title
        """
        width = surface.get_width()
        height = surface.get_height()

        # Clear with dark red background
        surface.fill((40, 0, 0))

        # Draw title
        title_style = TextStyle(font_size=12, color=Colors.RED, shadow=True)
        self.renderer.draw_text_centered(surface, title, height // 2 - 15, title_style)

        # Draw message (truncated if needed)
        msg_style = TextStyle(font_size=9, color=Colors.LIGHT_GRAY)
        truncated = self.renderer.truncate_text(message, width - 10, msg_style)
        self.renderer.draw_text_centered(surface, truncated, height // 2 + 5, msg_style)
