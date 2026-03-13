import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass, field


# --- LANDMARK INDICES (MediaPipe Hand) ---
WRIST           = 0
MIDDLE_BASE     = 9   # Used as hand center proxy
FINGER_TIPS     = [8, 12, 16, 20]   # Index, Middle, Ring, Pinky tips
FINGER_PIPS     = [6, 10, 14, 18]   # The joint directly below each tip
THUMB_TIP       = 4
THUMB_IP        = 3                  # Thumb's inner joint

# --- TUNING CONSTANTS ---
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE  = 0.7
OPEN_FINGER_THRESHOLD    = 3        # How many fingers must be extended to count as "open"


@dataclass
class HandState:
    """
    Carries all hand tracking data for a single frame.
    Using a dataclass prevents silent tuple-unpacking bugs in main.py.
    """
    frame:   object           # The annotated BGR frame for display
    cx:      float  = 0.5    # Normalized X position [0, 1]
    cy:      float  = 0.5    # Normalized Y position [0, 1]
    is_open: bool   = False  # True = open hand / vortex, False = fist / drop


class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        self.mp_draw = mp.solutions.drawing_utils

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_hand_state(self, frame) -> HandState:
        """
        Process a single BGR frame and return a HandState.
        Falls back to (center, closed) when no hand is detected.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = self.hands.process(rgb_frame)

        state = HandState(frame=frame)   # defaults: cx=0.5, cy=0.5, is_open=False

        if results.multi_hand_landmarks:
            # Only track the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
            )

            state.cx      = hand_landmarks.landmark[MIDDLE_BASE].x
            state.cy      = hand_landmarks.landmark[MIDDLE_BASE].y
            state.is_open = self._is_hand_open(hand_landmarks)

        state.frame = frame
        return state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_hand_open(self, hand_landmarks) -> bool:
        """
        Count how many fingers are extended by comparing each fingertip Y
        to the PIP joint below it. Works regardless of hand orientation
        because it checks per-finger curl rather than global wrist distance.

        MediaPipe Y is 0 at the top of frame, so a lower Y value means
        the landmark is higher on screen — i.e. the finger is extended.
        """
        extended = 0

        # Four fingers (skip thumb — handled separately below)
        for tip_idx, pip_idx in zip(FINGER_TIPS, FINGER_PIPS):
            tip_y = hand_landmarks.landmark[tip_idx].y
            pip_y = hand_landmarks.landmark[pip_idx].y
            if tip_y < pip_y:   # tip is above the pip joint → finger is out
                extended += 1

        # Thumb: compare tip X to IP joint X (thumb extends sideways)
        thumb_tip_x = hand_landmarks.landmark[THUMB_TIP].x
        thumb_ip_x  = hand_landmarks.landmark[THUMB_IP].x
        if abs(thumb_tip_x - thumb_ip_x) > 0.04:
            extended += 1

        return extended >= OPEN_FINGER_THRESHOLD

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def __del__(self):
        """Release MediaPipe resources when the tracker is garbage collected."""
        try:
            self.hands.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.__del__()