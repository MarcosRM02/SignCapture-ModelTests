"""Landmark preprocessing helpers aligned with ADA Gold pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd


LANDMARK_FEATURE_COLUMNS: list[str] = [
    f"landmark{i}_{coord}"
    for i in range(21)
    for coord in ("x", "y", "z")
]

ANGLE_FEATURE_COLUMNS: list[str] = [
    "angle_thumb_1_2_3",
    "angle_thumb_2_3_4",
    "angle_index_5_6_7",
    "angle_index_6_7_8",
    "angle_middle_9_10_11",
    "angle_middle_10_11_12",
    "angle_ring_13_14_15",
    "angle_ring_14_15_16",
    "angle_pinky_17_18_19",
    "angle_pinky_18_19_20",
    "angle_thumb_index_1_0_5",
    "angle_index_middle_6_m59_10",
    "angle_middle_ring_10_m913_14",
    "angle_ring_pinky_14_m1317_18",
]

ALL_FEATURE_COLUMNS: list[str] = LANDMARK_FEATURE_COLUMNS + ANGLE_FEATURE_COLUMNS

_EPSILON = 1e-12


def normalize_landmarks_array(landmarks: np.ndarray) -> np.ndarray:
    """Normalize landmarks to [-1, 1] using ADA normalization logic."""
    if landmarks.shape != (21, 3):
        raise ValueError(f"Unsupported landmarks shape: {landmarks.shape}")

    x_coords = landmarks[:, 0]
    y_coords = landmarks[:, 1]
    z_coords = landmarks[:, 2]

    min_x, max_x = x_coords.min(), x_coords.max()
    min_y, max_y = y_coords.min(), y_coords.max()
    min_z, max_z = z_coords.min(), z_coords.max()

    normalized = np.zeros((21, 3), dtype=np.float32)
    normalized[:, 0] = (x_coords - min_x) / (max_x - min_x) * 2 - 1 if max_x > min_x else 0
    normalized[:, 1] = (y_coords - min_y) / (max_y - min_y) * 2 - 1 if max_y > min_y else 0
    normalized[:, 2] = (z_coords - min_z) / (max_z - min_z) * 2 - 1 if max_z > min_z else 0

    return normalized


def build_feature_vector(landmarks: np.ndarray) -> np.ndarray:
    """Build model feature vector as normalized landmarks + engineered angles."""
    normalized_landmarks = normalize_landmarks_array(landmarks)
    angle_values = _compute_angles_single(normalized_landmarks)

    flat_landmarks = normalized_landmarks.reshape(-1)
    ordered_angles = np.array([angle_values[column] for column in ANGLE_FEATURE_COLUMNS], dtype=np.float32)
    return np.concatenate([flat_landmarks, ordered_angles]).astype(np.float32)


def add_angle_features_to_dataframe(input_df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered angle features if they are not already present."""
    output_df = input_df.copy()

    missing_angles = [column for column in ANGLE_FEATURE_COLUMNS if column not in output_df.columns]
    if not missing_angles:
        return output_df

    missing_landmarks = [column for column in LANDMARK_FEATURE_COLUMNS if column not in output_df.columns]
    if missing_landmarks:
        raise ValueError(
            "Cannot compute angle features because landmark columns are missing: "
            f"{missing_landmarks[:5]}{'...' if len(missing_landmarks) > 5 else ''}"
        )

    landmark_matrix = output_df[LANDMARK_FEATURE_COLUMNS].to_numpy(dtype=np.float64, copy=False)
    landmark_tensor = landmark_matrix.reshape(-1, 21, 3)
    angle_df = _compute_angles_batch(landmark_tensor)

    for column in missing_angles:
        output_df[column] = angle_df[column]

    return output_df


def _compute_angles_single(landmarks: np.ndarray) -> dict[str, float]:
    """Compute angle features for one sample [21, 3]."""
    midpoint_59 = (landmarks[5] + landmarks[9]) / 2.0
    midpoint_913 = (landmarks[9] + landmarks[13]) / 2.0
    midpoint_1317 = (landmarks[13] + landmarks[17]) / 2.0

    return {
        "angle_thumb_1_2_3": _compute_angle_degrees(landmarks[1], landmarks[2], landmarks[3]),
        "angle_thumb_2_3_4": _compute_angle_degrees(landmarks[2], landmarks[3], landmarks[4]),
        "angle_index_5_6_7": _compute_angle_degrees(landmarks[5], landmarks[6], landmarks[7]),
        "angle_index_6_7_8": _compute_angle_degrees(landmarks[6], landmarks[7], landmarks[8]),
        "angle_middle_9_10_11": _compute_angle_degrees(landmarks[9], landmarks[10], landmarks[11]),
        "angle_middle_10_11_12": _compute_angle_degrees(landmarks[10], landmarks[11], landmarks[12]),
        "angle_ring_13_14_15": _compute_angle_degrees(landmarks[13], landmarks[14], landmarks[15]),
        "angle_ring_14_15_16": _compute_angle_degrees(landmarks[14], landmarks[15], landmarks[16]),
        "angle_pinky_17_18_19": _compute_angle_degrees(landmarks[17], landmarks[18], landmarks[19]),
        "angle_pinky_18_19_20": _compute_angle_degrees(landmarks[18], landmarks[19], landmarks[20]),
        "angle_thumb_index_1_0_5": _compute_angle_degrees(landmarks[1], landmarks[0], landmarks[5]),
        "angle_index_middle_6_m59_10": _compute_angle_degrees(landmarks[6], midpoint_59, landmarks[10]),
        "angle_middle_ring_10_m913_14": _compute_angle_degrees(landmarks[10], midpoint_913, landmarks[14]),
        "angle_ring_pinky_14_m1317_18": _compute_angle_degrees(landmarks[14], midpoint_1317, landmarks[18]),
    }


def _compute_angles_batch(landmarks: np.ndarray) -> pd.DataFrame:
    """Compute angle features for a batch [n_samples, 21, 3]."""
    midpoint_59 = (landmarks[:, 5, :] + landmarks[:, 9, :]) / 2.0
    midpoint_913 = (landmarks[:, 9, :] + landmarks[:, 13, :]) / 2.0
    midpoint_1317 = (landmarks[:, 13, :] + landmarks[:, 17, :]) / 2.0

    result = {
        "angle_thumb_1_2_3": _compute_batch_angle_degrees(landmarks[:, 1, :], landmarks[:, 2, :], landmarks[:, 3, :]),
        "angle_thumb_2_3_4": _compute_batch_angle_degrees(landmarks[:, 2, :], landmarks[:, 3, :], landmarks[:, 4, :]),
        "angle_index_5_6_7": _compute_batch_angle_degrees(landmarks[:, 5, :], landmarks[:, 6, :], landmarks[:, 7, :]),
        "angle_index_6_7_8": _compute_batch_angle_degrees(landmarks[:, 6, :], landmarks[:, 7, :], landmarks[:, 8, :]),
        "angle_middle_9_10_11": _compute_batch_angle_degrees(landmarks[:, 9, :], landmarks[:, 10, :], landmarks[:, 11, :]),
        "angle_middle_10_11_12": _compute_batch_angle_degrees(landmarks[:, 10, :], landmarks[:, 11, :], landmarks[:, 12, :]),
        "angle_ring_13_14_15": _compute_batch_angle_degrees(landmarks[:, 13, :], landmarks[:, 14, :], landmarks[:, 15, :]),
        "angle_ring_14_15_16": _compute_batch_angle_degrees(landmarks[:, 14, :], landmarks[:, 15, :], landmarks[:, 16, :]),
        "angle_pinky_17_18_19": _compute_batch_angle_degrees(landmarks[:, 17, :], landmarks[:, 18, :], landmarks[:, 19, :]),
        "angle_pinky_18_19_20": _compute_batch_angle_degrees(landmarks[:, 18, :], landmarks[:, 19, :], landmarks[:, 20, :]),
        "angle_thumb_index_1_0_5": _compute_batch_angle_degrees(landmarks[:, 1, :], landmarks[:, 0, :], landmarks[:, 5, :]),
        "angle_index_middle_6_m59_10": _compute_batch_angle_degrees(landmarks[:, 6, :], midpoint_59, landmarks[:, 10, :]),
        "angle_middle_ring_10_m913_14": _compute_batch_angle_degrees(landmarks[:, 10, :], midpoint_913, landmarks[:, 14, :]),
        "angle_ring_pinky_14_m1317_18": _compute_batch_angle_degrees(landmarks[:, 14, :], midpoint_1317, landmarks[:, 18, :]),
    }

    return pd.DataFrame(result)


def _compute_angle_degrees(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray) -> float:
    """Compute angle ABC in degrees for one sample."""
    ba = point_a - point_b
    bc = point_c - point_b

    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom <= _EPSILON:
        return 0.0

    cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _compute_batch_angle_degrees(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray) -> np.ndarray:
    """Compute angle ABC in degrees for a batch of samples."""
    ba = point_a - point_b
    bc = point_c - point_b

    dot = np.einsum("ij,ij->i", ba, bc)
    norm_ba = np.linalg.norm(ba, axis=1)
    norm_bc = np.linalg.norm(bc, axis=1)
    denom = norm_ba * norm_bc

    cosine = np.ones_like(dot)
    valid = denom > _EPSILON
    cosine[valid] = np.clip(dot[valid] / denom[valid], -1.0, 1.0)

    return np.degrees(np.arccos(cosine)).astype(np.float32)