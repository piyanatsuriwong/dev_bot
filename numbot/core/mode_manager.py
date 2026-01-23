"""
Mode Manager for NumBot.
Manages mode switching between Hand tracking and Object detection.
Supports dual camera mode for instant switching.
"""

import threading
from enum import Enum
from typing import Optional, Callable, Any
from .tracker_result import TrackerResult, Detection


class Mode(Enum):
    """Operating modes for NumBot."""
    HAND = 'hand'       # Hand tracking with IMX708
    DETECT = 'detect'   # Object detection (show all) with IMX500
    TRACK = 'track'     # Track specific object with IMX500
    AUTO = 'auto'       # Auto-switch based on input
    DEMO = 'demo'       # Demo mode (no camera)


class ModeManager:
    """
    Manages mode switching between Hand and Object detection.

    Supports two operational strategies:
    1. Dual Camera Mode: Both trackers run continuously (instant switching)
    2. Single Camera Mode: One tracker at a time (requires restart)

    Usage:
        manager = ModeManager(hand_tracker, yolo_tracker)
        manager.switch_mode(Mode.DETECT)
        result = manager.update()
    """

    # Default track targets
    DEFAULT_TARGETS = ['person', 'cat', 'dog', 'bird', 'car', 'bottle', 'cup']

    def __init__(self, hand_tracker=None, yolo_tracker=None,
                 dual_camera_mode: bool = True,
                 default_mode: Mode = Mode.AUTO,
                 default_target: str = 'person'):
        """
        Initialize ModeManager.

        Args:
            hand_tracker: Hand tracker instance (IMX708 + MediaPipe)
            yolo_tracker: YOLO tracker instance (IMX500)
            dual_camera_mode: If True, both trackers run continuously
            default_mode: Initial operating mode
            default_target: Default target class for TRACK mode
        """
        self.hand_tracker = hand_tracker
        self.yolo_tracker = yolo_tracker
        self.dual_camera_mode = dual_camera_mode

        self._mode = default_mode
        self._track_target = default_target
        self._lock = threading.Lock()

        # State caching
        self._last_hand_result: Optional[TrackerResult] = None
        self._last_yolo_result: Optional[TrackerResult] = None

        # Callbacks
        self._on_mode_change: Optional[Callable[[Mode], None]] = None
        self._on_target_change: Optional[Callable[[str], None]] = None

        # Auto mode settings
        self.auto_hand_priority = True  # Hand takes priority in auto mode
        self.auto_switch_delay = 0.5    # Seconds before auto-switching

    @property
    def mode(self) -> Mode:
        """Get current mode."""
        with self._lock:
            return self._mode

    @property
    def mode_name(self) -> str:
        """Get current mode name as string."""
        return self.mode.value

    @property
    def track_target(self) -> str:
        """Get current track target class."""
        with self._lock:
            return self._track_target

    @property
    def available_modes(self) -> list:
        """Get list of available modes based on initialized trackers."""
        modes = [Mode.DEMO]
        if self.hand_tracker is not None:
            modes.append(Mode.HAND)
        if self.yolo_tracker is not None:
            modes.extend([Mode.DETECT, Mode.TRACK])
        if self.hand_tracker is not None or self.yolo_tracker is not None:
            modes.append(Mode.AUTO)
        return modes

    def set_mode_change_callback(self, callback: Callable[[Mode], None]) -> None:
        """Set callback for mode changes."""
        self._on_mode_change = callback

    def set_target_change_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for target changes."""
        self._on_target_change = callback

    def switch_mode(self, new_mode: Mode) -> bool:
        """
        Switch to a new mode.

        Args:
            new_mode: The mode to switch to

        Returns:
            True if switch was successful, False otherwise
        """
        # Validate mode is available
        if new_mode not in self.available_modes:
            print(f"[ModeManager] Mode {new_mode.value} not available")
            return False

        with self._lock:
            old_mode = self._mode
            self._mode = new_mode

        print(f"[ModeManager] Switched: {old_mode.value} -> {new_mode.value}")

        # Call callback if set
        if self._on_mode_change:
            self._on_mode_change(new_mode)

        return True

    def set_track_target(self, target: str) -> None:
        """
        Set target class for TRACK mode.

        Args:
            target: Class name to track (e.g., 'person', 'cat')
        """
        with self._lock:
            old_target = self._track_target
            self._track_target = target

        print(f"[ModeManager] Track target: {old_target} -> {target}")

        # Call callback if set
        if self._on_target_change:
            self._on_target_change(target)

    def cycle_mode(self) -> Mode:
        """
        Cycle to next available mode.

        Returns:
            The new mode after cycling
        """
        modes = self.available_modes
        current_idx = modes.index(self.mode) if self.mode in modes else 0
        next_idx = (current_idx + 1) % len(modes)
        new_mode = modes[next_idx]
        self.switch_mode(new_mode)
        return new_mode

    def cycle_target(self) -> str:
        """
        Cycle to next track target.

        Returns:
            The new target after cycling
        """
        targets = self.DEFAULT_TARGETS
        current_idx = targets.index(self._track_target) if self._track_target in targets else 0
        next_idx = (current_idx + 1) % len(targets)
        new_target = targets[next_idx]
        self.set_track_target(new_target)
        return new_target

    def update(self) -> TrackerResult:
        """
        Get tracking result based on current mode.
        Called every frame from main loop.

        Returns:
            TrackerResult with current tracking data
        """
        with self._lock:
            mode = self._mode

        if mode == Mode.DEMO:
            return TrackerResult.empty('demo')

        elif mode == Mode.HAND:
            return self._update_hand_mode()

        elif mode == Mode.DETECT:
            return self._update_detect_mode()

        elif mode == Mode.TRACK:
            return self._update_track_mode()

        elif mode == Mode.AUTO:
            return self._update_auto_mode()

        return TrackerResult.empty()

    def _update_hand_mode(self) -> TrackerResult:
        """Update for HAND mode."""
        result = self._get_hand_result()
        if result and result.detected:
            return result
        return TrackerResult.empty('hand')

    def _update_detect_mode(self) -> TrackerResult:
        """Update for DETECT mode (show all detections)."""
        return self._get_yolo_result()

    def _update_track_mode(self) -> TrackerResult:
        """Update for TRACK mode (follow specific target)."""
        yolo_result = self._get_yolo_result()
        return self._filter_by_target(yolo_result)

    def _update_auto_mode(self) -> TrackerResult:
        """
        Update for AUTO mode.
        Priority: Hand > Target Object > Any Object
        """
        # Get results from both trackers
        hand_result = self._get_hand_result()
        yolo_result = self._get_yolo_result()

        # Priority 1: Hand detected (if hand priority enabled)
        if self.auto_hand_priority and hand_result and hand_result.detected:
            return hand_result

        # Priority 2: Track target detected
        target_result = self._filter_by_target(yolo_result)
        if target_result.detected:
            return target_result

        # Priority 3: Any detection
        if yolo_result.detected:
            return yolo_result

        # Fallback to hand if available but YOLO has nothing
        if hand_result and hand_result.detected:
            return hand_result

        return TrackerResult.empty('auto')

    def _get_hand_result(self) -> Optional[TrackerResult]:
        """Get result from hand tracker."""
        if self.hand_tracker is None:
            return None

        try:
            # Call hand tracker update
            if hasattr(self.hand_tracker, 'get_result'):
                result = self.hand_tracker.get_result()
            elif hasattr(self.hand_tracker, 'update'):
                # Legacy interface
                pos, finger_count, frame, _ = self.hand_tracker.update()
                if pos is not None:
                    result = TrackerResult.from_hand(pos, finger_count, frame=frame)
                else:
                    result = TrackerResult.empty('hand')
            else:
                return None

            if result:
                self._last_hand_result = result
            return self._last_hand_result

        except Exception as e:
            print(f"[ModeManager] Hand tracker error: {e}")
            return None

    def _get_yolo_result(self) -> TrackerResult:
        """Get result from YOLO tracker."""
        if self.yolo_tracker is None:
            return TrackerResult.empty('yolo')

        try:
            # Get position and detections from YOLO tracker
            position = None
            detections = []

            if hasattr(self.yolo_tracker, 'get_normalized_position'):
                position = self.yolo_tracker.get_normalized_position()

            if hasattr(self.yolo_tracker, 'get_detections'):
                raw_detections = self.yolo_tracker.get_detections()
                if raw_detections:
                    detections = [
                        Detection(
                            class_name=d.get('class', d.get('label', 'unknown')),
                            confidence=d.get('confidence', d.get('score', 0.0)),
                            bbox=d.get('bbox', (0, 0, 0, 0)),
                            center=d.get('center', (0.0, 0.0))
                        ) for d in raw_detections
                    ]

            # Get frame if available
            frame = None
            if hasattr(self.yolo_tracker, 'get_frame'):
                frame = self.yolo_tracker.get_frame()

            result = TrackerResult.from_yolo(
                detections=detections,
                primary_position=position,
                frame=frame
            )

            self._last_yolo_result = result
            return result

        except Exception as e:
            print(f"[ModeManager] YOLO tracker error: {e}")
            return TrackerResult.empty('yolo')

    def _filter_by_target(self, yolo_result: TrackerResult) -> TrackerResult:
        """Filter YOLO result to only include target class."""
        if not yolo_result.detected:
            return TrackerResult.empty('yolo')

        target = self._track_target
        target_detections = yolo_result.filter_by_class(target)

        if not target_detections:
            return TrackerResult.empty('yolo')

        # Use first matching detection as primary position
        primary = target_detections[0]
        return TrackerResult(
            source='yolo',
            detected=True,
            position=primary.center,
            detections=target_detections,
            frame=yolo_result.frame
        )

    # Helper methods for checking state

    def is_hand_mode(self) -> bool:
        """Check if currently in hand mode."""
        return self.mode == Mode.HAND

    def is_detect_mode(self) -> bool:
        """Check if currently in detect mode."""
        return self.mode == Mode.DETECT

    def is_track_mode(self) -> bool:
        """Check if currently in track mode."""
        return self.mode == Mode.TRACK

    def is_auto_mode(self) -> bool:
        """Check if currently in auto mode."""
        return self.mode == Mode.AUTO

    def is_demo_mode(self) -> bool:
        """Check if currently in demo mode."""
        return self.mode == Mode.DEMO

    def has_hand_tracker(self) -> bool:
        """Check if hand tracker is available."""
        return self.hand_tracker is not None

    def has_yolo_tracker(self) -> bool:
        """Check if YOLO tracker is available."""
        return self.yolo_tracker is not None

    def get_status(self) -> dict:
        """Get current status as dictionary."""
        return {
            'mode': self.mode.value,
            'track_target': self._track_target,
            'dual_camera': self.dual_camera_mode,
            'hand_available': self.has_hand_tracker(),
            'yolo_available': self.has_yolo_tracker(),
            'available_modes': [m.value for m in self.available_modes]
        }

    def __str__(self) -> str:
        return f"ModeManager(mode={self.mode.value}, target={self._track_target})"
