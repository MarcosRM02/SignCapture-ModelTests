"""Module for classification models."""

from src.models.base import BaseModel
from src.models.random_forest import RandomForestClassifier

__all__ = ["BaseModel", "RandomForestClassifier"]
