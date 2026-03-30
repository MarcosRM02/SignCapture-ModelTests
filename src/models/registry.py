"""Model factory for ASL classifiers."""

from typing import Any

from src.models.base import BaseModel
from src.models.random_forest import RandomForestClassifier
from src.models.xgboost_model import XGBoostClassifier

MODEL_REGISTRY = {
    "random_forest": RandomForestClassifier,
    "xgboost": XGBoostClassifier,
}


def create_model(model_name: str, model_config: dict[str, Any] | None = None) -> BaseModel:
    """Create a model instance from its registered name."""
    normalized_name = model_name.strip().lower()
    model_cls = MODEL_REGISTRY.get(normalized_name)
    if model_cls is None:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported model '{model_name}'. Available models: {available}")

    return model_cls(model_config=model_config)


def available_models() -> list[str]:
    """Return the list of available model names."""
    return sorted(MODEL_REGISTRY)
