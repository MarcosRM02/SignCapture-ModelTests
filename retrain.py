"""Automated retraining entrypoint for Hito 5 feedback loop.

This script augments the Gold train split with production feedback samples,
trains a fresh model, compares it against the currently deployed model, and
promotes the candidate only when it does not regress beyond the accepted F1
threshold.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from src.preprocessing import (
    ALL_FEATURE_COLUMNS,
    LANDMARK_FEATURE_COLUMNS,
    add_angle_features_to_dataframe,
)
from src.utils import set_seed

TRUTHY_VALUES = {"1", "true", "t", "yes", "y", "si", "s"}
FEATURE_VECTOR_COLUMNS = ("feature_vector", "feature_vector_json", "landmarks_json", "landmarks")
SUPPORTED_MODELS = ("random_forest", "xgboost", "neural_network")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Retrain ASL model with feedback samples")
    parser.add_argument("--model", default="xgboost", choices=SUPPORTED_MODELS, help="Model name to train")
    parser.add_argument("--gold-dir", type=Path, required=True, help="Directory containing train.csv, val.csv and test.csv")
    parser.add_argument("--existing-model-path", type=Path, required=True, help="Currently deployed model path")
    parser.add_argument("--candidate-model-path", type=Path, required=True, help="Where to save the retrained model before promotion")
    parser.add_argument("--promoted-model-path", type=Path, help="Destination path for the promoted model")
    parser.add_argument("--summary-path", type=Path, help="Where to save a JSON summary of the retraining run")
    parser.add_argument("--feedback-csv", type=Path, help="Optional CSV with feedback samples")
    parser.add_argument("--db-dsn", type=str, help="Optional PostgreSQL/Supabase connection string")
    parser.add_argument("--feedback-table", default="prediction_feedback", help="Feedback table name")
    parser.add_argument("--feedback-limit", type=int, default=1000, help="Maximum number of feedback rows to pull")
    parser.add_argument("--max-f1-regression", type=float, default=0.0, help="Maximum allowed macro-F1 drop versus the deployed model")
    parser.add_argument("--min-feedback-samples", type=int, default=1, help="Minimum usable feedback samples required to retrain")
    return parser.parse_args()


def main() -> None:
    """Run the retraining workflow."""
    args = parse_args()
    set_seed(42)

    gold_dir = args.gold_dir.resolve()
    train_df = _load_split(gold_dir / "train.csv")
    val_df = _load_split(gold_dir / "val.csv")
    test_df = _load_split(gold_dir / "test.csv")

    feedback_df = _load_feedback_samples(args)
    feedback_count = len(feedback_df)

    if feedback_count < args.min_feedback_samples:
        summary = {
            "status": "skipped",
            "reason": "not_enough_feedback_samples",
            "feedback_samples": feedback_count,
            "min_feedback_samples": args.min_feedback_samples,
        }
        _write_summary(args.summary_path, summary)
        print(json.dumps(summary, indent=2))
        return

    augmented_train_df = pd.concat([train_df, feedback_df], ignore_index=True)
    feature_columns = [column for column in ALL_FEATURE_COLUMNS if column in augmented_train_df.columns]
    if not feature_columns:
        raise ValueError("No compatible feature columns were found after merging feedback samples.")

    label_encoder = LabelEncoder()
    X_train, y_train = _split_features_and_labels(augmented_train_df, feature_columns, label_encoder, fit=True)
    X_val, y_val = _split_features_and_labels(val_df, feature_columns, label_encoder)
    X_test, y_test = _split_features_and_labels(test_df, feature_columns, label_encoder)

    candidate_model = _create_model(args.model)
    training_metrics = candidate_model.train(X_train, y_train, X_val, y_val)
    candidate_predictions = candidate_model.predict(X_test)
    candidate_metrics = _build_metrics(y_test, candidate_predictions)

    args.candidate_model_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_model.save(args.candidate_model_path)

    baseline_metrics: dict[str, float] | None = None
    promoted = True
    promotion_reason = "no_existing_model"

    if args.existing_model_path.exists():
        deployed_model = _load_model(args.model, args.existing_model_path)
        baseline_predictions = deployed_model.predict(X_test)
        baseline_metrics = _build_metrics(y_test, baseline_predictions)
        promoted = candidate_metrics["macro_f1"] >= (baseline_metrics["macro_f1"] - args.max_f1_regression)
        promotion_reason = "candidate_meets_threshold" if promoted else "candidate_regressed"

    if promoted and args.promoted_model_path is not None:
        args.promoted_model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.candidate_model_path, args.promoted_model_path)

    summary = {
        "status": "completed",
        "model": args.model,
        "feedback_samples": feedback_count,
        "train_samples_after_merge": int(len(augmented_train_df)),
        "promoted": promoted,
        "promotion_reason": promotion_reason,
        "max_f1_regression": args.max_f1_regression,
        "training_metrics": _to_builtin(training_metrics),
        "candidate_metrics": _to_builtin(candidate_metrics),
        "baseline_metrics": _to_builtin(baseline_metrics),
        "candidate_model_path": str(args.candidate_model_path),
        "promoted_model_path": str(args.promoted_model_path) if args.promoted_model_path else None,
    }

    _write_summary(args.summary_path, summary)
    print(json.dumps(summary, indent=2))


def _load_split(csv_path: Path) -> pd.DataFrame:
    """Load one Gold split and ensure angle features exist."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Gold split not found: {csv_path}")
    return add_angle_features_to_dataframe(pd.read_csv(csv_path))


def _load_feedback_samples(args: argparse.Namespace) -> pd.DataFrame:
    """Load labeled feedback samples from CSV or PostgreSQL."""
    raw_feedback_df = pd.DataFrame()

    if args.feedback_csv is not None:
        raw_feedback_df = pd.read_csv(args.feedback_csv)
    elif args.db_dsn:
        raw_feedback_df = _query_feedback_dataframe(
            dsn=args.db_dsn,
            table_name=args.feedback_table,
            limit=args.feedback_limit,
        )

    if raw_feedback_df.empty:
        return raw_feedback_df

    normalized_feedback_df = _normalize_feedback_dataframe(raw_feedback_df)
    return add_angle_features_to_dataframe(normalized_feedback_df)


def _query_feedback_dataframe(dsn: str, table_name: str, limit: int) -> pd.DataFrame:
    """Query feedback rows from PostgreSQL/Supabase."""
    import psycopg
    from psycopg import sql

    query = sql.SQL("SELECT * FROM {} ORDER BY created_at DESC NULLS LAST LIMIT %s").format(
        sql.Identifier(table_name)
    )

    with psycopg.connect(dsn) as connection:
        with connection.cursor(row_factory=psycopg.rows.dict_row) as cursor:
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()

    return pd.DataFrame(rows)


def _normalize_feedback_dataframe(raw_feedback_df: pd.DataFrame) -> pd.DataFrame:
    """Map flexible feedback schemas into the training dataframe schema."""
    feedback_df = raw_feedback_df.copy()

    if "letter" not in feedback_df.columns:
        if {"predicted_label", "is_correct"}.issubset(feedback_df.columns):
            feedback_df = feedback_df[_truthy_mask(feedback_df["is_correct"])].copy()
            feedback_df["letter"] = feedback_df["predicted_label"]
        elif "expected_label" in feedback_df.columns:
            feedback_df["letter"] = feedback_df["expected_label"]
        else:
            raise ValueError(
                "Feedback data must contain either 'letter', "
                "'expected_label', or ('predicted_label' + 'is_correct')."
            )

    feedback_df["letter"] = feedback_df["letter"].astype(str).str.strip().str.upper()
    feedback_df = feedback_df[feedback_df["letter"] != ""].copy()

    missing_direct_features = [column for column in LANDMARK_FEATURE_COLUMNS if column not in feedback_df.columns]
    if missing_direct_features:
        feature_vector_column = next((column for column in FEATURE_VECTOR_COLUMNS if column in feedback_df.columns), None)
        if feature_vector_column is None:
            raise ValueError(
                "Feedback data must contain raw landmark columns or one feature vector column: "
                f"{', '.join(FEATURE_VECTOR_COLUMNS)}"
            )
        expanded_features = feedback_df[feature_vector_column].apply(_parse_feature_vector).apply(pd.Series)
        expanded_features.columns = LANDMARK_FEATURE_COLUMNS
        feedback_df = pd.concat([feedback_df.drop(columns=[feature_vector_column]), expanded_features], axis=1)

    required_columns = ["letter", *LANDMARK_FEATURE_COLUMNS]
    return feedback_df[required_columns]


def _parse_feature_vector(raw_value: Any) -> list[float]:
    """Parse a serialized feature vector into the 63 raw landmark values."""
    if isinstance(raw_value, str):
        parsed_value = json.loads(raw_value)
    elif isinstance(raw_value, (list, tuple, np.ndarray)):
        parsed_value = list(raw_value)
    else:
        raise ValueError("Unsupported feature vector format in feedback sample.")

    if len(parsed_value) == len(ALL_FEATURE_COLUMNS):
        parsed_value = parsed_value[: len(LANDMARK_FEATURE_COLUMNS)]

    if len(parsed_value) != len(LANDMARK_FEATURE_COLUMNS):
        raise ValueError(
            "Feedback feature vectors must contain 63 landmark values "
            f"or 77 full features, got {len(parsed_value)}."
        )

    return [float(value) for value in parsed_value]


def _truthy_mask(series: pd.Series) -> pd.Series:
    """Convert a heterogeneous boolean-like column into a boolean mask."""
    return series.astype(str).str.strip().str.lower().isin(TRUTHY_VALUES)


def _split_features_and_labels(
    split_df: pd.DataFrame,
    feature_columns: list[str],
    label_encoder: LabelEncoder,
    *,
    fit: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix and encoded labels for one split."""
    prepared_df = add_angle_features_to_dataframe(split_df)
    X = prepared_df[feature_columns].to_numpy(dtype=np.float32, copy=False)
    labels = prepared_df["letter"].astype(str).str.strip().str.upper().to_numpy()
    y = label_encoder.fit_transform(labels) if fit else label_encoder.transform(labels)
    return X, y


def _build_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute evaluation metrics for one model."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def _create_model(model_name: str) -> Any:
    """Lazily instantiate the requested model without importing unused stacks."""
    normalized_name = model_name.strip().lower()
    if normalized_name == "xgboost":
        from src.models.xgboost import XGBoostClassifier

        return XGBoostClassifier()
    if normalized_name == "random_forest":
        from src.models.random_forest import RandomForestClassifier

        return RandomForestClassifier()
    if normalized_name == "neural_network":
        from src.models.neural_network import NeuralNetworkClassifier

        return NeuralNetworkClassifier()
    raise ValueError(f"Unsupported model '{model_name}'. Available models: {', '.join(SUPPORTED_MODELS)}")


def _load_model(model_name: str, model_path: Path) -> Any:
    """Lazily load the deployed model using the requested wrapper class."""
    normalized_name = model_name.strip().lower()
    if normalized_name == "xgboost":
        from src.models.xgboost import XGBoostClassifier

        return XGBoostClassifier.load(model_path)
    if normalized_name == "random_forest":
        from src.models.random_forest import RandomForestClassifier

        return RandomForestClassifier.load(model_path)
    if normalized_name == "neural_network":
        from src.models.neural_network import NeuralNetworkClassifier

        return NeuralNetworkClassifier.load(model_path)
    raise ValueError(f"Unsupported model '{model_name}'. Available models: {', '.join(SUPPORTED_MODELS)}")


def _write_summary(summary_path: Path | None, payload: dict[str, Any]) -> None:
    """Persist retraining summary when requested."""
    if summary_path is None:
        return
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(_to_builtin(payload), indent=2), encoding="utf-8")


def _to_builtin(value: Any) -> Any:
    """Recursively convert NumPy and pandas values into JSON-safe builtins."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


if __name__ == "__main__":
    main()
