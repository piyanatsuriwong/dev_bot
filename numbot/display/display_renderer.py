"""
Main display renderer for NumBot ST7735S display (160x128).
Coordinates all UI components and renders complete frames.
"""

import pygame
from typing import Optional, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .text_renderer import TextRenderer, TextStyle, Styles, Colors
from .ui_components import (
    StatusBar, DetectionPanel, PositionIndicator,
    HandInfoPanel, LoadingIndicator, ErrorDisplay
)

# Import TrackerResult from core
try:
    from core.tracker_result import TrackerResult
except ImportError:
    from ..core.tracker_result import TrackerResult


class DisplayRenderer:
    """
    Main display rendering controller for ST7735S.

    Manages screen layout with three zones:
    1. Status Bar (12px) - Mode, target, FPS
    2. Face Area (variable) - Robot eyes
    3. Info Panel (36px) - Detection/hand info

    Usage:
        renderer = DisplayRenderer(160, 128)
        renderer.render_frame(surface, mode, fps, result, robo_surface)
    """

    # Layout constants
    STATUS_HEIGHT = 12
    INFO_HEIGHT = 36

    def __init__(self, width: int, height: int, font_path: str = None):
        """
        Initialize display renderer.

        Args:
            width: Screen width (typically 160 for ST7735S)
            height: Screen height (typically 128 for ST7735S)
            font_path: Path to custom font file
        """
        self.width = width
        self.height = height
        self.font_path = font_path

        # Calculate face area height
        self.face_height = height - self.STATUS_HEIGHT - self.INFO_HEIGHT

        # Initialize text renderer
        self.text = TextRenderer(font_path)

        # Initialize UI components
        self.status_bar = StatusBar(self.text, width)
        self.detection_panel = DetectionPanel(self.text, max_detections=2)
        self.position_indicator = PositionIndicator(self.text)
        self.hand_info = HandInfoPanel(self.text)
        self.loading = LoadingIndicator(self.text)
        self.error_display = ErrorDisplay(self.text)

        # State
        self._last_mode = None
        self._frame_count = 0

    def render_frame(self, surface: pygame.Surface, mode: str, fps: int,
                     result: TrackerResult, robo_surface: pygame.Surface = None,
                     track_target: str = None) -> None:
        """
        Render complete frame to surface.

        This is the main rendering method that should be called each frame.

        Args:
            surface: Target pygame surface (ST7735S buffer)
            mode: Current mode ('hand', 'detect', 'track', 'auto', 'demo')
            fps: Current FPS
            result: TrackerResult with current tracking data
            robo_surface: Pre-rendered robot eyes surface
            track_target: Target class for track mode
        """
        # Clear surface
        surface.fill(Colors.BG_DARK)

        # Zone 1: Status bar (top)
        self._render_status_bar(surface, mode, fps, result, track_target)

        # Zone 2: Robot eyes (middle)
        self._render_face_area(surface, robo_surface)

        # Zone 3: Info panel (bottom)
        self._render_info_panel(surface, mode, result)

        self._frame_count += 1
        self._last_mode = mode

    def _render_status_bar(self, surface: pygame.Surface, mode: str, fps: int,
                           result: TrackerResult, track_target: str) -> None:
        """Render status bar at top of screen."""
        extra_info = None

        # Add extra info based on mode
        if mode == 'auto':
            if result.source == 'hand':
                extra_info = "hand"
            elif result.detected:
                extra_info = f"{result.detection_count}obj"

        self.status_bar.draw(surface, mode, fps, track_target, extra_info)

    def _render_face_area(self, surface: pygame.Surface,
                          robo_surface: pygame.Surface) -> None:
        """Render robot eyes in middle area."""
        if robo_surface is None:
            return

        face_y = self.STATUS_HEIGHT

        # Center the robot face horizontally
        face_x = (self.width - robo_surface.get_width()) // 2

        # Ensure we don't exceed face area
        if robo_surface.get_height() > self.face_height:
            # Scale down if needed
            scale = self.face_height / robo_surface.get_height()
            new_width = int(robo_surface.get_width() * scale)
            new_height = int(robo_surface.get_height() * scale)
            robo_surface = pygame.transform.scale(robo_surface, (new_width, new_height))
            face_x = (self.width - new_width) // 2

        surface.blit(robo_surface, (face_x, face_y))

    def _render_info_panel(self, surface: pygame.Surface, mode: str,
                           result: TrackerResult) -> None:
        """Render info panel at bottom of screen."""
        info_y = self.height - self.INFO_HEIGHT
        x = 2
        width = self.width - 4

        # Draw separator line
        pygame.draw.line(surface, Colors.DARK_GRAY,
                        (0, info_y), (self.width, info_y), 1)

        info_y += 2  # Padding after separator

        # Render based on mode and result source
        if mode == 'hand' or result.source == 'hand':
            self._render_hand_info(surface, x, info_y, width, result)
        elif mode in ['detect', 'track'] or result.source == 'yolo':
            self._render_detection_info(surface, x, info_y, width, result, mode)
        elif mode == 'auto':
            self._render_auto_info(surface, x, info_y, width, result)
        else:
            self._render_demo_info(surface, x, info_y, width)

    def _render_hand_info(self, surface: pygame.Surface, x: int, y: int,
                          width: int, result: TrackerResult) -> None:
        """Render hand tracking info panel."""
        self.hand_info.draw(
            surface, x, y, width,
            finger_count=result.finger_count,
            position=result.position if result.detected else None,
            detected=result.detected
        )

    def _render_detection_info(self, surface: pygame.Surface, x: int, y: int,
                               width: int, result: TrackerResult, mode: str) -> None:
        """Render object detection info panel."""
        if result.detected and result.detections:
            self.detection_panel.draw(surface, result.detections, x, y, width)
        else:
            # Show searching message
            if mode == 'track':
                msg = "Searching..."
            else:
                msg = "No objects"
            style = TextStyle(font_size=9, color=Colors.GRAY)
            self.text.draw_text(surface, msg, x, y, style)

    def _render_auto_info(self, surface: pygame.Surface, x: int, y: int,
                          width: int, result: TrackerResult) -> None:
        """Render auto mode info panel."""
        if result.source == 'hand' and result.detected:
            self._render_hand_info(surface, x, y, width, result)
        elif result.detected:
            self._render_detection_info(surface, x, y, width, result, 'detect')
        else:
            style = TextStyle(font_size=9, color=Colors.GRAY)
            self.text.draw_text(surface, "Waiting for input...", x, y, style)

    def _render_demo_info(self, surface: pygame.Surface, x: int, y: int,
                          width: int) -> None:
        """Render demo mode info panel."""
        style = TextStyle(font_size=9, color=Colors.GRAY)
        self.text.draw_text(surface, "Demo mode", x, y, style)

        # Show animation frame indicator
        dots = "." * (self._frame_count // 15 % 4)
        self.text.draw_text(surface, dots, x + 70, y, style)

    def render_loading(self, surface: pygame.Surface, message: str) -> None:
        """
        Render loading screen.

        Args:
            surface: Target pygame surface
            message: Loading message to display
        """
        self.loading.draw(surface, message)

    def render_error(self, surface: pygame.Surface, message: str,
                     title: str = "Error") -> None:
        """
        Render error screen.

        Args:
            surface: Target pygame surface
            message: Error message
            title: Error title
        """
        self.error_display.draw(surface, message, title)

    def render_mode_switch(self, surface: pygame.Surface, old_mode: str,
                           new_mode: str) -> None:
        """
        Render mode switch transition.

        Args:
            surface: Target pygame surface
            old_mode: Previous mode
            new_mode: New mode
        """
        surface.fill(Colors.BG_DARK)

        # Show transition message
        msg = f"{old_mode} > {new_mode}"
        style = TextStyle(font_size=12, color=Colors.CYAN, shadow=True)
        self.text.draw_text_centered(surface, msg, self.height // 2 - 5, style)

    def render_startup(self, surface: pygame.Surface, progress: float = 0,
                       message: str = "Starting...") -> None:
        """
        Render startup screen with progress.

        Args:
            surface: Target pygame surface
            progress: Progress value (0.0 to 1.0)
            message: Status message
        """
        surface.fill(Colors.BG_DARK)

        # Title
        title_style = TextStyle(font_size=14, color=Colors.WHITE, shadow=True)
        self.text.draw_text_centered(surface, "NumBot", 30, title_style)

        # Progress bar
        bar_width = self.width - 40
        bar_x = 20
        bar_y = self.height // 2
        self.text.draw_progress_bar(surface, bar_x, bar_y, bar_width, 8,
                                    progress, 1.0, Colors.CYAN, Colors.DARK_GRAY)

        # Progress percentage
        pct_style = TextStyle(font_size=10, color=Colors.CYAN)
        self.text.draw_text_centered(surface, f"{int(progress * 100)}%",
                                     bar_y + 15, pct_style)

        # Message
        msg_style = TextStyle(font_size=9, color=Colors.GRAY)
        self.text.draw_text_centered(surface, message, self.height - 20, msg_style)

    def get_face_area_rect(self) -> Tuple[int, int, int, int]:
        """
        Get the rectangle for the face area.

        Returns:
            Tuple of (x, y, width, height) for face area
        """
        return (0, self.STATUS_HEIGHT, self.width, self.face_height)

    def get_info_area_rect(self) -> Tuple[int, int, int, int]:
        """
        Get the rectangle for the info area.

        Returns:
            Tuple of (x, y, width, height) for info area
        """
        return (0, self.height - self.INFO_HEIGHT, self.width, self.INFO_HEIGHT)
