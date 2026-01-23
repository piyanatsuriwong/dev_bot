"""
Unified tracker result classes for NumBot.
Provides common data structures for hand tracking and object detection.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any


# Object class to emoji/icon mapping
DETECTION_ICONS = {
    # People
    'person': 'P',

    # Animals
    'cat': 'C',
    'dog': 'D',
    'bird': 'B',
    'horse': 'H',
    'sheep': 'S',
    'cow': 'W',
    'elephant': 'E',
    'bear': 'R',
    'zebra': 'Z',
    'giraffe': 'G',

    # Vehicles
    'car': 'c',
    'bicycle': 'b',
    'motorcycle': 'm',
    'airplane': 'a',
    'bus': 'u',
    'train': 't',
    'truck': 'T',
    'boat': 'o',

    # Objects
    'bottle': '!',
    'cup': 'U',
    'fork': 'f',
    'knife': 'k',
    'spoon': 's',
    'bowl': 'w',
    'chair': 'h',
    'couch': 'O',
    'bed': 'e',
    'toilet': 'i',
    'tv': 'v',
    'laptop': 'L',
    'mouse': 'M',
    'remote': 'r',
    'keyboard': 'K',
    'cell phone': 'p',
    'book': 'x',
    'clock': 'l',
    'scissors': 'X',
    'teddy bear': 'y',

    # Food
    'banana': 'n',
    'apple': 'A',
    'sandwich': 'N',
    'orange': 'O',
    'broccoli': 'I',
    'carrot': 'J',
    'pizza': 'z',
    'donut': 'q',
    'cake': 'Q',

    # Default
    'default': '?',
}


@dataclass
class Detection:
    """Single detection result from YOLO or other detector."""

    class_name: str           # e.g., "person", "cat"
    confidence: float         # 0.0 - 1.0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, width, height
    center: Tuple[float, float] = (0.0, 0.0)        # normalized -1 to 1

    @property
    def icon(self) -> str:
        """Get icon character for class."""
        return DETECTION_ICONS.get(self.class_name, DETECTION_ICONS['default'])

    @property
    def label(self) -> str:
        """Get formatted label with icon."""
        return f"[{self.icon}] {self.class_name}"

    @property
    def confidence_percent(self) -> int:
        """Get confidence as percentage."""
        return int(self.confidence * 100)

    def __str__(self) -> str:
        return f"{self.class_name} ({self.confidence_percent}%)"


@dataclass
class TrackerResult:
    """
    Unified result from any tracker (hand or object detection).

    Attributes:
        source: Source of tracking ('hand', 'yolo', 'none')
        detected: Whether something was detected
        position: Normalized position (-1 to 1) for eye tracking
        detections: List of Detection objects (for object detection)
        finger_count: Number of fingers detected (for hand tracking)
        mood_suggestion: Suggested mood based on detection
        frame: Camera frame (optional, for display)
    """

    source: str = 'none'
    detected: bool = False
    position: Tuple[float, float] = (0.0, 0.0)
    detections: List[Detection] = field(default_factory=list)
    finger_count: Optional[int] = None
    mood_suggestion: Optional[int] = None
    frame: Optional[Any] = None

    @classmethod
    def empty(cls, source: str = 'none') -> 'TrackerResult':
        """Create an empty result."""
        return cls(source=source, detected=False, position=(0.0, 0.0), detections=[])

    @classmethod
    def from_hand(cls, position: Tuple[float, float], finger_count: int,
                  mood: int = None, frame: Any = None) -> 'TrackerResult':
        """Create result from hand tracker."""
        return cls(
            source='hand',
            detected=True,
            position=position,
            detections=[],
            finger_count=finger_count,
            mood_suggestion=mood,
            frame=frame
        )

    @classmethod
    def from_yolo(cls, detections: List[Detection],
                  primary_position: Tuple[float, float] = None,
                  frame: Any = None) -> 'TrackerResult':
        """Create result from YOLO tracker."""
        detected = len(detections) > 0

        # Use first detection's center as primary position if not specified
        if primary_position is None and detected:
            primary_position = detections[0].center
        elif primary_position is None:
            primary_position = (0.0, 0.0)

        return cls(
            source='yolo',
            detected=detected,
            position=primary_position,
            detections=detections,
            frame=frame
        )

    @property
    def primary_detection(self) -> Optional[Detection]:
        """Get the primary (first) detection."""
        if self.detections:
            return self.detections[0]
        return None

    @property
    def detection_count(self) -> int:
        """Get number of detections."""
        return len(self.detections)

    def has_class(self, class_name: str) -> bool:
        """Check if a specific class was detected."""
        return any(d.class_name == class_name for d in self.detections)

    def get_class(self, class_name: str) -> Optional[Detection]:
        """Get detection for a specific class."""
        for d in self.detections:
            if d.class_name == class_name:
                return d
        return None

    def filter_by_class(self, class_name: str) -> List[Detection]:
        """Get all detections of a specific class."""
        return [d for d in self.detections if d.class_name == class_name]

    def filter_by_confidence(self, min_confidence: float) -> List[Detection]:
        """Get detections above confidence threshold."""
        return [d for d in self.detections if d.confidence >= min_confidence]

    def __str__(self) -> str:
        if not self.detected:
            return f"TrackerResult({self.source}: no detection)"
        if self.source == 'hand':
            return f"TrackerResult(hand: fingers={self.finger_count}, pos={self.position})"
        return f"TrackerResult({self.source}: {self.detection_count} objects)"
