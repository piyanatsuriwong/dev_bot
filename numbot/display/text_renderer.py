"""
Beautiful text rendering for small displays (ST7735S 160x128).
Supports styled text, shadows, outlines, and progress bars.
"""

import pygame
from dataclasses import dataclass
from typing import Tuple, Optional, Dict


@dataclass
class TextStyle:
    """Text styling options for display rendering."""

    font_size: int = 10
    color: Tuple[int, int, int] = (255, 255, 255)
    bg_color: Optional[Tuple[int, int, int]] = None
    shadow: bool = False
    shadow_color: Tuple[int, int, int] = (0, 0, 0)
    shadow_offset: int = 1
    outline: bool = False
    outline_color: Tuple[int, int, int] = (0, 0, 0)
    bold: bool = False
    padding: int = 2


class Styles:
    """Pre-defined text styles for common use cases."""

    # Headers and titles
    HEADER = TextStyle(font_size=12, color=(255, 255, 255), shadow=True)
    TITLE = TextStyle(font_size=14, color=(255, 255, 255), shadow=True, bold=True)

    # Mode indicators
    MODE_ACTIVE = TextStyle(font_size=10, color=(0, 255, 0), shadow=True)
    MODE_INACTIVE = TextStyle(font_size=10, color=(128, 128, 128))
    MODE_HAND = TextStyle(font_size=10, color=(255, 200, 0), shadow=True)
    MODE_DETECT = TextStyle(font_size=10, color=(0, 200, 255), shadow=True)
    MODE_TRACK = TextStyle(font_size=10, color=(255, 100, 100), shadow=True)

    # Detection labels
    DETECTION = TextStyle(font_size=10, color=(0, 255, 255), outline=True)
    DETECTION_NAME = TextStyle(font_size=9, color=(255, 255, 255))
    CONFIDENCE = TextStyle(font_size=8, color=(0, 255, 0))
    CONFIDENCE_LOW = TextStyle(font_size=8, color=(255, 255, 0))

    # Status and info
    INFO = TextStyle(font_size=9, color=(200, 200, 200))
    WARNING = TextStyle(font_size=10, color=(255, 255, 0), bg_color=(100, 50, 0))
    ERROR = TextStyle(font_size=10, color=(255, 100, 100), bg_color=(100, 0, 0))
    SUCCESS = TextStyle(font_size=10, color=(100, 255, 100), bg_color=(0, 100, 0))

    # Small indicators
    TINY = TextStyle(font_size=8, color=(180, 180, 180))
    FPS = TextStyle(font_size=8, color=(100, 255, 100))


class Colors:
    """Common colors for display rendering."""

    # Basic colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (128, 128, 128)
    DARK_GRAY = (64, 64, 64)
    LIGHT_GRAY = (192, 192, 192)

    # Primary colors
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)

    # Secondary colors
    CYAN = (0, 255, 255)
    MAGENTA = (255, 0, 255)
    YELLOW = (255, 255, 0)

    # UI colors
    HAND_COLOR = (255, 200, 0)      # Orange-yellow for hand mode
    DETECT_COLOR = (0, 200, 255)    # Cyan for detect mode
    TRACK_COLOR = (255, 100, 100)   # Red for track mode
    AUTO_COLOR = (150, 150, 255)    # Purple for auto mode

    # Confidence colors
    CONF_HIGH = (0, 255, 0)         # Green for high confidence
    CONF_MED = (255, 255, 0)        # Yellow for medium confidence
    CONF_LOW = (255, 100, 0)        # Orange for low confidence

    # Background colors
    BG_DARK = (20, 20, 30)
    BG_STATUS = (30, 30, 40)
    BG_INFO = (40, 40, 50)


class TextRenderer:
    """
    Beautiful text rendering for small displays.

    Provides styled text rendering with support for:
    - Multiple font sizes
    - Shadows and outlines
    - Background colors
    - Progress bars
    - Centered text
    """

    def __init__(self, font_path: str = None):
        """
        Initialize text renderer.

        Args:
            font_path: Path to custom font file (optional)
        """
        pygame.font.init()
        self.font_path = font_path
        self._fonts: Dict[Tuple[int, bool], pygame.font.Font] = {}

    def get_font(self, size: int, bold: bool = False) -> pygame.font.Font:
        """
        Get cached font of specified size.

        Args:
            size: Font size in pixels
            bold: Whether to use bold font

        Returns:
            Pygame font object
        """
        key = (size, bold)
        if key not in self._fonts:
            if self.font_path:
                try:
                    self._fonts[key] = pygame.font.Font(self.font_path, size)
                except Exception:
                    self._fonts[key] = pygame.font.SysFont('arial', size, bold=bold)
            else:
                self._fonts[key] = pygame.font.SysFont('arial', size, bold=bold)
        return self._fonts[key]

    def draw_text(self, surface: pygame.Surface, text: str,
                  x: int, y: int, style: TextStyle = None) -> pygame.Rect:
        """
        Draw styled text on surface.

        Args:
            surface: Target pygame surface
            text: Text to draw
            x: X position
            y: Y position
            style: TextStyle object (optional, uses default if None)

        Returns:
            Bounding rect of drawn text
        """
        if style is None:
            style = TextStyle()

        font = self.get_font(style.font_size, style.bold)

        # Get text surface for size calculation
        text_surface = font.render(text, True, style.color)
        text_rect = text_surface.get_rect(topleft=(x, y))

        # Draw background if specified
        if style.bg_color:
            bg_rect = text_rect.inflate(style.padding * 2, style.padding * 2)
            pygame.draw.rect(surface, style.bg_color, bg_rect)
            # Adjust text position for padding
            x += style.padding
            y += style.padding

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

        return text_rect

    def draw_text_centered(self, surface: pygame.Surface, text: str,
                           y: int, style: TextStyle = None) -> pygame.Rect:
        """
        Draw text centered horizontally.

        Args:
            surface: Target pygame surface
            text: Text to draw
            y: Y position
            style: TextStyle object

        Returns:
            Bounding rect of drawn text
        """
        if style is None:
            style = TextStyle()

        font = self.get_font(style.font_size, style.bold)
        text_surface = font.render(text, True, style.color)
        x = (surface.get_width() - text_surface.get_width()) // 2

        return self.draw_text(surface, text, x, y, style)

    def draw_text_right(self, surface: pygame.Surface, text: str,
                        y: int, margin: int = 2, style: TextStyle = None) -> pygame.Rect:
        """
        Draw text aligned to right edge.

        Args:
            surface: Target pygame surface
            text: Text to draw
            y: Y position
            margin: Right margin
            style: TextStyle object

        Returns:
            Bounding rect of drawn text
        """
        if style is None:
            style = TextStyle()

        font = self.get_font(style.font_size, style.bold)
        text_surface = font.render(text, True, style.color)
        x = surface.get_width() - text_surface.get_width() - margin

        return self.draw_text(surface, text, x, y, style)

    def draw_progress_bar(self, surface: pygame.Surface,
                          x: int, y: int, width: int, height: int,
                          value: float, max_value: float = 1.0,
                          fg_color: Tuple[int, int, int] = None,
                          bg_color: Tuple[int, int, int] = None,
                          border: bool = True,
                          border_color: Tuple[int, int, int] = None) -> None:
        """
        Draw a progress bar.

        Args:
            surface: Target pygame surface
            x: X position
            y: Y position
            width: Bar width
            height: Bar height
            value: Current value (0 to max_value)
            max_value: Maximum value (default 1.0)
            fg_color: Foreground (fill) color
            bg_color: Background color
            border: Whether to draw border
            border_color: Border color
        """
        # Default colors based on value
        if fg_color is None:
            ratio = value / max_value if max_value > 0 else 0
            if ratio >= 0.7:
                fg_color = Colors.CONF_HIGH
            elif ratio >= 0.4:
                fg_color = Colors.CONF_MED
            else:
                fg_color = Colors.CONF_LOW

        if bg_color is None:
            bg_color = Colors.DARK_GRAY

        if border_color is None:
            border_color = Colors.GRAY

        # Draw background
        pygame.draw.rect(surface, bg_color, (x, y, width, height))

        # Draw filled portion
        fill_width = int(width * (value / max_value)) if max_value > 0 else 0
        if fill_width > 0:
            pygame.draw.rect(surface, fg_color, (x, y, fill_width, height))

        # Draw border
        if border:
            pygame.draw.rect(surface, border_color, (x, y, width, height), 1)

    def draw_confidence_bar(self, surface: pygame.Surface,
                            x: int, y: int, width: int,
                            confidence: float) -> None:
        """
        Draw a confidence indicator bar.

        Args:
            surface: Target pygame surface
            x: X position
            y: Y position
            width: Bar width
            confidence: Confidence value (0.0 to 1.0)
        """
        self.draw_progress_bar(surface, x, y, width, 6, confidence, 1.0)

    def get_text_size(self, text: str, style: TextStyle = None) -> Tuple[int, int]:
        """
        Get size of text without drawing.

        Args:
            text: Text to measure
            style: TextStyle object

        Returns:
            Tuple of (width, height)
        """
        if style is None:
            style = TextStyle()

        font = self.get_font(style.font_size, style.bold)
        text_surface = font.render(text, True, (255, 255, 255))
        return text_surface.get_size()

    def truncate_text(self, text: str, max_width: int,
                      style: TextStyle = None, suffix: str = '..') -> str:
        """
        Truncate text to fit within max_width.

        Args:
            text: Text to truncate
            max_width: Maximum width in pixels
            style: TextStyle object
            suffix: Suffix to add when truncated

        Returns:
            Truncated text
        """
        if style is None:
            style = TextStyle()

        font = self.get_font(style.font_size, style.bold)

        # Check if text fits
        if font.size(text)[0] <= max_width:
            return text

        # Truncate until it fits
        for i in range(len(text), 0, -1):
            truncated = text[:i] + suffix
            if font.size(truncated)[0] <= max_width:
                return truncated

        return suffix
