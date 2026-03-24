"""Centralized configuration for SignCapture-ModelTests."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class PathsConfig:
    """Project paths configuration."""

    root_dir: Path
    data_dir: Path
    gold_dir: Path
    models_dir: Path

    def __init__(self) -> None:
        env_path = Path(__file__).resolve().parents[1] / ".env"
        load_dotenv(dotenv_path=env_path)

        env_root = os.getenv("SIGNCAPTURE_ROOT")
        if env_root:
            self.root_dir = Path(env_root)
        else:
            self.root_dir = Path(__file__).resolve().parents[2]

        self.data_dir = self.root_dir / "data"
        self.gold_dir = self.data_dir / "gold"
        self.models_dir = self.root_dir / "models"


@dataclass
class TrainingConfig:
    """Training configuration."""

    seed: int = 42

    def __init__(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
        if config_path.exists():
            settings = load_yaml(config_path)
            self.seed = settings.get("general", {}).get("seed", 42)


@dataclass
class RandomForestConfig:
    """Configuration for the Random Forest model."""

    n_estimators: int = 200
    max_depth: int = 20
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    max_features: str = "sqrt"
    class_weight: str = "balanced"

    def __init__(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
        if config_path.exists():
            settings = load_yaml(config_path)
            rf = settings.get("random_forest", {})
            self.n_estimators = rf.get("n_estimators", 200)
            self.max_depth = rf.get("max_depth", 20)
            self.min_samples_split = rf.get("min_samples_split", 5)
            self.min_samples_leaf = rf.get("min_samples_leaf", 2)
            self.max_features = rf.get("max_features", "sqrt")
            self.class_weight = rf.get("class_weight", "balanced")


@dataclass
class Config:
    """Main configuration."""

    paths: PathsConfig
    training: TrainingConfig
    random_forest: RandomForestConfig

    def __init__(self) -> None:
        self.paths = PathsConfig()
        self.training = TrainingConfig()
        self.random_forest = RandomForestConfig()


config = Config()
