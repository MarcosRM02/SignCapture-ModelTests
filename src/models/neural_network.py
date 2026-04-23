"""Neural Network classifier for ASL."""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import config
from src.models.base import BaseModel


class _Complex2MLP(nn.Module):
    """MLP architecture with two hidden layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int,
        hidden_dim_2: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for input samples."""
        return self.network(x)


class NeuralNetworkClassifier(BaseModel):
    """PyTorch Neural Network classifier for ASL."""

    def __init__(self, model_config: dict[str, Any] | None = None) -> None:
        """Initialize the classifier with configurable hyperparameters.

        Args:
            model_config: Optional override for default hyperparameters.
        """
        nn_config = config.neural_network
        default_config = {
            "hidden_dim_1": nn_config.hidden_dim_1,
            "hidden_dim_2": nn_config.hidden_dim_2,
            "epochs": nn_config.epochs,
            "batch_size": nn_config.batch_size,
            "learning_rate": nn_config.learning_rate,
            "seed": config.training.seed,
        }

        final_config = {**default_config, **(model_config or {})}
        super().__init__(name="neural_network", config=final_config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: _Complex2MLP | None = None
        self.input_dim: int | None = None
        self.num_classes: int | None = None

    def _build_model(self, input_dim: int, num_classes: int) -> None:
        """Build the neural network for current dataset dimensions."""
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model = _Complex2MLP(
            input_dim=input_dim,
            hidden_dim_1=int(self.config["hidden_dim_1"]),
            hidden_dim_2=int(self.config["hidden_dim_2"]),
            num_classes=num_classes,
        ).to(self.device)

    def _check_is_fitted(self) -> None:
        """Validate that the model has been trained or loaded."""
        if not self.is_fitted or self.model is None:
            raise ValueError("Model is not fitted. Train or load the model first.")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Train the neural network model."""
        torch.manual_seed(int(self.config["seed"]))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.config["seed"]))

        input_dim = int(X_train.shape[1])
        observed_labels = y_train if y_val is None else np.concatenate((y_train, y_val))
        num_classes = int(np.max(observed_labels)) + 1
        self._build_model(input_dim=input_dim, num_classes=num_classes)

        assert self.model is not None
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.config["learning_rate"]),
        )

        dataset = TensorDataset(
            torch.from_numpy(X_train.astype(np.float32)),
            torch.from_numpy(y_train.astype(np.int64)),
        )
        data_loader = DataLoader(
            dataset,
            batch_size=max(1, int(self.config["batch_size"])),
            shuffle=True,
        )

        for _ in range(int(self.config["epochs"])):
            for batch_features, batch_labels in data_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)

                logits = self.model(batch_features)
                loss = criterion(logits, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.is_fitted = True
        metrics = {
            "train_accuracy": float(np.mean(self.predict(X_train) == y_train)),
        }

        if X_val is not None and y_val is not None:
            metrics["val_accuracy"] = float(np.mean(self.predict(X_val) == y_val))

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes for each sample."""
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities for each sample."""
        self._check_is_fitted()
        assert self.model is not None

        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities.detach().cpu().numpy()

    def save(self, path: Path) -> None:
        """Save model state and metadata to disk."""
        self._check_is_fitted()
        assert self.model is not None

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model_state_dict": self.model.state_dict(),
                    "config": self.config,
                    "model_name": self.name,
                    "input_dim": self.input_dim,
                    "num_classes": self.num_classes,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "NeuralNetworkClassifier":
        """Load model state and metadata from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls(model_config=data["config"])
        instance._build_model(
            input_dim=int(data["input_dim"]),
            num_classes=int(data["num_classes"]),
        )

        assert instance.model is not None
        instance.model.load_state_dict(data["model_state_dict"])
        instance.model.eval()
        instance.is_fitted = True
        return instance