"""Landmark processor for inference.

Replicates normalization and feature engineering logic from SignCapture-ADA."""

from dataclasses import dataclass

import numpy as np
from src.preprocessing import build_feature_vector, normalize_landmarks_array


@dataclass
class LandmarkPoint:
    """Landmark point with 3D coordinates."""

    x: float
    y: float
    z: float


class LandmarkProcessor:
    """Landmark processor for normalization and feature engineering.

    Exactly replicates the logic from SignCapture-ADA Gold pipeline.
    """

    NUM_LANDMARKS = 21

    def normalize_landmarks(self, landmarks: list[LandmarkPoint] | np.ndarray) -> np.ndarray:
        """Normalizes landmarks to the range [-1, 1].

        Args:
            landmarks: List of 21 LandmarkPoints or array [21, 3].

        Returns:
            Normalized array [63,].
        """
        landmark_array = self._to_landmark_array(landmarks)
        normalized_landmarks = normalize_landmarks_array(landmark_array)
        return normalized_landmarks.reshape(-1)

    def build_features(self, landmarks: list[LandmarkPoint] | np.ndarray) -> np.ndarray:
        """Builds full feature vector [63 landmarks + 14 angles]."""
        landmark_array = self._to_landmark_array(landmarks)
        return build_feature_vector(landmark_array)

    def process_landmarks(self, landmarks: list[LandmarkPoint]) -> np.ndarray | None:
        """Processes a list of LandmarkPoint objects.

        Args:
            landmarks: List of LandmarkPoint from LandmarkDetector.

        Returns:
            Feature vector [77,] or None if invalid.
        """
        if landmarks is None or len(landmarks) != self.NUM_LANDMARKS:
            return None
        return self.build_features(landmarks)

    def process_mediapipe_landmarks(self, hand_landmarks) -> np.ndarray | None:
        """Processes MediaPipe landmarks (legacy support).

        Args:
            hand_landmarks: Result from MediaPipe Hands (deprecated API).

        Returns:
            Feature vector [77,] or None if no landmarks are present.
        """
        if hand_landmarks is None:
            return None

        landmarks = [
            LandmarkPoint(x=lm.x, y=lm.y, z=lm.z)
            for lm in hand_landmarks.landmark
        ]

        if len(landmarks) != self.NUM_LANDMARKS:
            return None

        return self.build_features(landmarks)

    def _to_landmark_array(self, landmarks: list[LandmarkPoint] | np.ndarray) -> np.ndarray:
        """Convert input landmarks to a [21, 3] NumPy array."""
        if isinstance(landmarks, np.ndarray):
            if landmarks.shape != (self.NUM_LANDMARKS, 3):
                raise ValueError(f"Unsupported landmarks shape: {landmarks.shape}")
            return landmarks.astype(np.float64, copy=False)

        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)


from src.inference.landmark_detector import LandmarkDetector

__all__ = ["LandmarkPoint", "LandmarkProcessor", "LandmarkDetector"]
