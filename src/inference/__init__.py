"""Landmark processor for inference.

Replicates the normalization logic from SignCapture-ADA for consistency."""

from dataclasses import dataclass

import numpy as np


@dataclass
class LandmarkPoint:
    """Landmark point with 3D coordinates."""

    x: float
    y: float
    z: float


class LandmarkProcessor:
    """Landmark processor for normalization.

    Exactly replicates the logic from SignCapture-ADA/src/gold/normalizer.py
    """

    NUM_LANDMARKS = 21

    def normalize_landmarks(self, landmarks: list[LandmarkPoint] | np.ndarray) -> np.ndarray:
        """Normalizes landmarks to the range [-1, 1].

        Args:
            landmarks: List of 21 LandmarkPoints or array [21, 3].

        Returns:
            Normalized array [63,].
        """
        if isinstance(landmarks, np.ndarray):
            if landmarks.shape == (21, 3):
                x_coords = landmarks[:, 0]
                y_coords = landmarks[:, 1]
                z_coords = landmarks[:, 2]
            else:
                raise ValueError(f"Unsupported landmarks shape: {landmarks.shape}")
        else:
            x_coords = np.array([lm.x for lm in landmarks])
            y_coords = np.array([lm.y for lm in landmarks])
            z_coords = np.array([lm.z for lm in landmarks])

        min_x, max_x = x_coords.min(), x_coords.max()
        min_y, max_y = y_coords.min(), y_coords.max()
        min_z, max_z = z_coords.min(), z_coords.max()

        normalized = np.zeros(63, dtype=np.float32)

        for i in range(self.NUM_LANDMARKS):
            normalized[i * 3] = (x_coords[i] - min_x) / (max_x - min_x) * 2 - 1 if max_x > min_x else 0
            normalized[i * 3 + 1] = (y_coords[i] - min_y) / (max_y - min_y) * 2 - 1 if max_y > min_y else 0
            normalized[i * 3 + 2] = (z_coords[i] - min_z) / (max_z - min_z) * 2 - 1 if max_z > min_z else 0

        return normalized

    def process_landmarks(self, landmarks: list[LandmarkPoint]) -> np.ndarray | None:
        """Processes a list of LandmarkPoint objects.

        Args:
            landmarks: List of LandmarkPoint from LandmarkDetector.

        Returns:
            Normalized array [63,] or None if invalid.
        """
        if landmarks is None or len(landmarks) != self.NUM_LANDMARKS:
            return None
        return self.normalize_landmarks(landmarks)

    def process_mediapipe_landmarks(self, hand_landmarks) -> np.ndarray | None:
        """Processes MediaPipe landmarks (legacy support).

        Args:
            hand_landmarks: Result from MediaPipe Hands (deprecated API).

        Returns:
            Normalized array [63,] or None if no landmarks are present.
        """
        if hand_landmarks is None:
            return None

        landmarks = [
            LandmarkPoint(x=lm.x, y=lm.y, z=lm.z)
            for lm in hand_landmarks.landmark
        ]

        if len(landmarks) != self.NUM_LANDMARKS:
            return None

        return self.normalize_landmarks(landmarks)


from src.inference.landmark_detector import LandmarkDetector

__all__ = ["LandmarkPoint", "LandmarkProcessor", "LandmarkDetector"]
