"""Module for classification models."""

from src.models.base import BaseModel
from src.models.registry import available_models, create_model, load_model

__all__ = [
    "BaseModel",
    "NeuralNetworkClassifier",
    "RandomForestClassifier",
    "XGBoostClassifier",
    "create_model",
    "available_models",
    "load_model",
]


def __getattr__(name: str):
    """Resolve heavy model classes lazily."""
    if name == "NeuralNetworkClassifier":
        from src.models.neural_network import NeuralNetworkClassifier

        return NeuralNetworkClassifier
    if name == "RandomForestClassifier":
        from src.models.random_forest import RandomForestClassifier

        return RandomForestClassifier
    if name == "XGBoostClassifier":
        from src.models.xgboost import XGBoostClassifier

        return XGBoostClassifier
    raise AttributeError(f"module 'src.models' has no attribute {name!r}")
