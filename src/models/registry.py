"""Model factory for ASL classifiers."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from src.models.base import BaseModel

MODEL_NAMES = ("neural_network", "random_forest", "xgboost")


def _get_model_cls(model_name: str) -> type[BaseModel]:
    """Resolve the concrete model class lazily to avoid importing unused stacks."""
    normalized_name = model_name.strip().lower()
    if normalized_name == "neural_network":
        from src.models.neural_network import NeuralNetworkClassifier

        return NeuralNetworkClassifier
    if normalized_name == "random_forest":
        from src.models.random_forest import RandomForestClassifier

        return RandomForestClassifier
    if normalized_name == "xgboost":
        from src.models.xgboost import XGBoostClassifier

        return XGBoostClassifier

    available = ", ".join(sorted(MODEL_NAMES))
    raise ValueError(f"Unsupported model '{model_name}'. Available models: {available}")


def create_model(model_name: str, model_config: dict[str, Any] | None = None) -> BaseModel:
    """Create a model instance from its registered name."""
    model_cls = _get_model_cls(model_name)
    return model_cls(model_config=model_config)


def available_models() -> list[str]:
    """Return the list of available model names."""
    return sorted(MODEL_NAMES)


def _infer_model_name(model_data: dict[str, Any], model_path: Path) -> str:
    """Infer the registered model name from serialized payload or filename."""
    configured_name = model_data.get("model_name")
    if isinstance(configured_name, str):
        normalized_name = configured_name.strip().lower()
        if normalized_name in MODEL_NAMES:
            return normalized_name

    model_object = model_data.get("model")
    if model_object is not None:
        class_name = model_object.__class__.__name__.lower()
        module_name = model_object.__class__.__module__.lower()

        if "xgbclassifier" in class_name or "xgboost" in module_name:
            return "xgboost"
        if "randomforest" in class_name and "sklearn" in module_name:
            return "random_forest"
        if "neuralnetworkclassifier" in class_name:
            return "neural_network"

    filename = model_path.stem.lower()
    for model_name in MODEL_NAMES:
        if model_name in filename:
            return model_name

    available = ", ".join(sorted(MODEL_NAMES))
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
    model_cls = _get_model_cls(model_name)
    return model_cls.load(path)
