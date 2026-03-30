"""Random Forest classifier for ASL."""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier

from src.models.base import BaseModel
from src.config import config


class RandomForestClassifier(BaseModel):
    """Random Forest classifier for ASL."""

    def __init__(self, model_config: dict[str, Any] | None = None) -> None:
        """Initializes the classifier.

        Args:
            model_config: Optional hyperparameters.
        """
        rf_config = config.random_forest
        default_config = {
            "n_estimators": rf_config.n_estimators,
            "max_depth": rf_config.max_depth,
            "min_samples_split": rf_config.min_samples_split,
            "min_samples_leaf": rf_config.min_samples_leaf,
            "max_features": rf_config.max_features,
            "class_weight": rf_config.class_weight,
            "n_jobs": -1,
            "random_state": config.training.seed,
        }
        final_config = {**default_config, **(model_config or {})}
        super().__init__(name="random_forest", config=final_config)
        self.model = SklearnRandomForestClassifier(**final_config)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Trains the Random Forest model."""
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        metrics = {"train_accuracy": self.model.score(X_train, y_train)}
        if X_val is not None and y_val is not None:
            metrics["val_accuracy"] = self.model.score(X_val, y_val)

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the classes."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns the probabilities for each class."""
        return self.model.predict_proba(X)

    def save(self, path: Path) -> None:
        """Saves the model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "config": self.config, "model_name": self.name}, f)

    @classmethod
    def load(cls, path: Path) -> "RandomForestClassifier":
        """Loads a model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls(model_config=data["config"])
        instance.model = data["model"]
        instance.is_fitted = True
        return instance
