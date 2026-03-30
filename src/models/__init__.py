"""Module for classification models."""

from src.models.base import BaseModel
from src.models.registry import available_models, create_model
from src.models.random_forest import RandomForestClassifier
from src.models.xgboost_model import XGBoostClassifier

__all__ = [
    "BaseModel",
    "RandomForestClassifier",
    "XGBoostClassifier",
    "create_model",
    "available_models",
]
