"""Script for training the Random Forest model.

Usage:
    python train.py
"""

from pathlib import Path

from sklearn.metrics import classification_report, accuracy_score

from src.config import config
from src.data import DataLoader
from src.models import RandomForestClassifier
from src.utils import set_seed


def main() -> None:
    """Trains the Random Forest model and saves the result."""
    set_seed(config.training.seed)

    print("=" * 60)
    print("Training Random Forest for ASL classification")
    print("=" * 60)

    # Load data
    print("\n Loading data from:", config.paths.gold_dir)
    loader = DataLoader()
    X_train, y_train, X_val, y_val, X_test, y_test = loader.load_data()

    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Val:   {X_val.shape[0]} samples")
    print(f"   Test:  {X_test.shape[0]} samples")
    print(f"   Classes: {loader.get_class_names()}")

    # Train model
    print("\n Training model...")
    model = RandomForestClassifier()
    metrics = model.train(X_train, y_train, X_val, y_val)

    print(f"   Train accuracy: {metrics['train_accuracy']:.4f}")
    print(f"   Val accuracy:   {metrics['val_accuracy']:.4f}")

    # Evaluate on test set
    print("\n Evaluating on test set:")
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"   Test accuracy: {test_accuracy:.4f}")

    print("\n" + classification_report(
        y_test, y_pred, target_names=loader.get_class_names()
    ))

    # Save model
    model_path = config.paths.models_dir / "random_forest_asl.pkl"
    model.save(model_path)
    print(f"\n Model saved to: {model_path}")


if __name__ == "__main__":
    main()
