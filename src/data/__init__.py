"""Data loader for Gold dataset training."""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import config
from src.preprocessing import (
    ALL_FEATURE_COLUMNS,
    ANGLE_FEATURE_COLUMNS,
    add_angle_features_to_dataframe,
)


class DataLoader:
    """Data loader for the Gold dataset.

    Loads the train/val/test splits from the Gold directory.
    """

    def __init__(self, gold_dir: Path | None = None) -> None:
        self.gold_dir = gold_dir or config.paths.gold_dir
        self.label_encoder = LabelEncoder()
        self.label_column: str = "letter"
        self.feature_columns: list[str] = []

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load all data splits.

        Returns:
            Tuple with (X_train, y_train, X_val, y_val, X_test, y_test).
        """
        train_df = self._prepare_split(self._load_split("train"))
        val_df = self._prepare_split(self._load_split("val"))
        test_df = self._prepare_split(self._load_split("test"))

        self.feature_columns = [column for column in ALL_FEATURE_COLUMNS if column in train_df.columns]

        if not self.feature_columns:
            raise ValueError("No valid feature columns were found in Gold dataset.")

        for split_df in (val_df, test_df):
            missing_features = [column for column in self.feature_columns if column not in split_df.columns]
            if missing_features:
                raise ValueError(
                    "Feature column mismatch between train and other splits. "
                    f"Missing columns: {missing_features[:5]}"
                )

        X_train = train_df[self.feature_columns].values.astype(np.float32)
        X_val = val_df[self.feature_columns].values.astype(np.float32)
        X_test = test_df[self.feature_columns].values.astype(np.float32)

        y_train = self.label_encoder.fit_transform(train_df[self.label_column].values)
        y_val = self.label_encoder.transform(val_df[self.label_column].values)
        y_test = self.label_encoder.transform(test_df[self.label_column].values)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def _load_split(self, split_name: str) -> pd.DataFrame:
        """Load a specific data split."""
        csv_path = self.gold_dir / f"{split_name}.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path)
        raise FileNotFoundError(f"File not found: {split_name}.csv in {self.gold_dir}")

    def _prepare_split(self, split_df: pd.DataFrame) -> pd.DataFrame:
        """Ensure required engineered features are present in the split."""
        if any(column not in split_df.columns for column in ANGLE_FEATURE_COLUMNS):
            return add_angle_features_to_dataframe(split_df)
        return split_df

    def get_class_names(self) -> list[str]:
        """Return the names of the classes."""
        return list(self.label_encoder.classes_)

    def decode_labels(self, encoded_labels: np.ndarray) -> np.ndarray:
        """Decode numeric labels to strings."""
        return self.label_encoder.inverse_transform(encoded_labels)
