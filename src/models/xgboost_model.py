"""XGBoost classifier for ASL."""

import importlib
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from xgboost import XGBClassifier

from src.config import config
from src.models.base import BaseModel


class XGBoostClassifier(BaseModel):
    """XGBoost classifier for ASL."""

    def __init__(self, model_config: dict[str, Any] | None = None) -> None:
        """Initializes the classifier.

        Args:
            model_config: Optional hyperparameters.
        """
        xgb_config = config.xgboost
        default_config = {
            "n_estimators": xgb_config.n_estimators,
            "max_depth": xgb_config.max_depth,
            "learning_rate": xgb_config.learning_rate,
            "min_child_weight": xgb_config.min_child_weight,
            "subsample": xgb_config.subsample,
            "colsample_bytree": xgb_config.colsample_bytree,
            "objective": xgb_config.objective,
            "tree_method": xgb_config.tree_method,
            "n_jobs": -1,
            "random_state": config.training.seed,
        }
        final_config = {**default_config, **(model_config or {})}
        super().__init__(name="xgboost", config=final_config)
        self.model = XGBClassifier(**final_config)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Trains the XGBoost model."""
        fit_kwargs: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = False

        self.model.fit(X_train, y_train, **fit_kwargs)
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
    def load(cls, path: Path) -> "XGBoostClassifier":
        """Loads a model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls(model_config=data["config"])
        instance.model = data["model"]
        instance.is_fitted = True
        return instance
