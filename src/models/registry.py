"""Model factory for ASL classifiers."""

import pickle
from pathlib import Path
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


def _infer_model_name(model_data: dict[str, Any], model_path: Path) -> str:
    """Infer the registered model name from serialized payload or filename."""
    configured_name = model_data.get("model_name")
    if isinstance(configured_name, str):
        normalized_name = configured_name.strip().lower()
        if normalized_name in MODEL_REGISTRY:
            return normalized_name

    model_object = model_data.get("model")
    if model_object is not None:
        class_name = model_object.__class__.__name__.lower()
        module_name = model_object.__class__.__module__.lower()

        if "xgbclassifier" in class_name or "xgboost" in module_name:
            return "xgboost"
        if "randomforest" in class_name and "sklearn" in module_name:
            return "random_forest"

    filename = model_path.stem.lower()
    for model_name in MODEL_REGISTRY:
        if model_name in filename:
            return model_name

    available = ", ".join(sorted(MODEL_REGISTRY))
    raise ValueError(
        f"Unable to infer model type for '{model_path}'. "
        f"Supported model types: {available}"
    )


def load_model(model_path: Path | str) -> BaseModel:
    """Load a serialized model and return the correct model wrapper instance."""
    path = Path(model_path)
    with open(path, "rb") as f:
        model_data = pickle.load(f)

    if not isinstance(model_data, dict):
        raise ValueError(f"Invalid model format in '{path}'. Expected a dictionary payload.")

    model_name = _infer_model_name(model_data, path)
    model_cls = MODEL_REGISTRY[model_name]
    return model_cls.load(path)
