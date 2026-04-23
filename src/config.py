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
    model: str = "random_forest"

    def __init__(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
        if config_path.exists():
            settings = load_yaml(config_path)
            self.seed = settings.get("general", {}).get("seed", 42)
            self.model = settings.get("training", {}).get("model", "random_forest")


@dataclass
class MediaPipeConfig:
    """Configuration for MediaPipe hand landmark detection."""

    max_num_hands: int = 1
    min_detection_confidence: float = 0.3
    model_path: str = "../models/hand_landmarker.task"

    def __init__(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
        if config_path.exists():
            settings = load_yaml(config_path)
            mp_config = settings.get("mediapipe", {})
            self.max_num_hands = mp_config.get("max_num_hands", 1)
            self.min_detection_confidence = mp_config.get("min_detection_confidence", 0.3)
            self.model_path = mp_config.get("model_path", "../models/hand_landmarker.task")


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
class XGBoostConfig:
    """Configuration for the XGBoost model."""

    n_estimators: int = 300
    max_depth: int = 8
    learning_rate: float = 0.05
    min_child_weight: int = 1
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    objective: str = "multi:softprob"

    def __init__(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
        if config_path.exists():
            settings = load_yaml(config_path)
            xgb = settings.get("xgboost", {})
            self.n_estimators = xgb.get("n_estimators", 300)
            self.max_depth = xgb.get("max_depth", 8)
            self.learning_rate = xgb.get("learning_rate", 0.05)
            self.min_child_weight = xgb.get("min_child_weight", 1)
            self.subsample = xgb.get("subsample", 0.9)
            self.colsample_bytree = xgb.get("colsample_bytree", 0.9)
            self.objective = xgb.get("objective", "multi:softprob")


@dataclass
class NeuralNetworkConfig:
    """Configuration for the Neural Network model."""

    hidden_dim_1: int = 256
    hidden_dim_2: int = 128
    epochs: int = 500
    batch_size: int = 256
    learning_rate: float = 0.001

    def __init__(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
        if config_path.exists():
            settings = load_yaml(config_path)
            nn_cfg = settings.get("neural_network", {})
            self.hidden_dim_1 = nn_cfg.get("hidden_dim_1", 256)
            self.hidden_dim_2 = nn_cfg.get("hidden_dim_2", 128)
            self.epochs = nn_cfg.get("epochs", 500)
            self.batch_size = nn_cfg.get("batch_size", 64)
            self.learning_rate = nn_cfg.get("learning_rate", 0.001)


@dataclass
class Config:
    """Main configuration."""

    paths: PathsConfig
    training: TrainingConfig
    mediapipe: MediaPipeConfig
    random_forest: RandomForestConfig
    xgboost: XGBoostConfig
    neural_network: NeuralNetworkConfig

    def __init__(self) -> None:
        self.paths = PathsConfig()
        self.training = TrainingConfig()
        self.mediapipe = MediaPipeConfig()
        self.random_forest = RandomForestConfig()
        self.xgboost = XGBoostConfig()
        self.neural_network = NeuralNetworkConfig()


config = Config()
