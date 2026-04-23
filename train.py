"""Script for training ASL classification models.

Usage:
    python train.py
    python train.py --model random_forest
    python train.py --model xgboost
    python train.py --model neural_network
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from src.config import config
from src.data import DataLoader
from src.models import available_models, create_model
from src.utils import set_seed


def save_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Guarda la matriz de confusión como imagen.
    
    Args:
        y_test: Labels reales del conjunto de test.
        y_pred: Predicciones del modelo.
        class_names: Nombres de las clases.
        output_path: Ruta donde guardar la imagen.
    """
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Predicciones'})
    plt.title('Matriz de Confusión', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicción', fontsize=12)
    plt.ylabel('Valor Real', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_classification_report(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Guarda el classification report como tabla.
    
    Args:
        y_test: Labels reales del conjunto de test.
        y_pred: Predicciones del modelo.
        class_names: Nombres de las clases.
        output_path: Ruta donde guardar la imagen.
    """
    # Generar reporte como diccionario
    report_dict = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True
    )
    
    # Convertir a DataFrame
    df = pd.DataFrame(report_dict).transpose()
    
    # Formatear columnas
    df = df.round(3)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(10, len(df) * 0.4 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    # Crear tabla
    table = ax.table(
        cellText=df.values,
        rowLabels=df.index,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.15] * len(df.columns)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Estilo de encabezados
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Estilo de índices de fila
    for i in range(len(df)):
        table[(i + 1, -1)].set_facecolor('#E7E6E6')
        table[(i + 1, -1)].set_text_props(weight='bold')
    
    plt.title('Classification Report', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_metrics_summary(
    metrics: dict,
    test_accuracy: float,
    output_path: Path,
) -> None:
    """Guarda tabla con resumen de métricas.
    
    Args:
        metrics: Diccionario con métricas de entrenamiento.
        test_accuracy: Accuracy en conjunto de test.
        output_path: Ruta donde guardar la imagen.
    """
    # Crear datos de la tabla
    data = {
        'Split': ['Train', 'Validation', 'Test'],
        'Accuracy': [
            metrics.get('train_accuracy', 0),
            metrics.get('val_accuracy', 0),
            test_accuracy
        ]
    }
    
    df = pd.DataFrame(data)
    df['Accuracy'] = df['Accuracy'].apply(lambda x: f'{x:.4f}')
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('tight')
    ax.axis('off')
    
    # Crear tabla
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.5, 0.5]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Estilo de encabezados
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternar colores de filas
    for i in range(len(df)):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i + 1, j)].set_facecolor('#F2F2F2')
    
    plt.title('Resumen de Métricas', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main() -> None:
    """Trains a configured model and saves the result."""
    parser = argparse.ArgumentParser(description="Train ASL classifier")
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=available_models(),
        help="Model name to train",
    )
    args = parser.parse_args()

    model_name = args.model
    set_seed(config.training.seed)

    print("=" * 60)
    print(f"Training {model_name} for ASL classification")
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
    model = create_model(model_name)
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
    model_path = config.paths.models_dir / f"{model.name}_asl.pkl"
    model.save(model_path)
    print(f"\n Model saved to: {model_path}")
    
    # Save visualizations
    print("\n Generating visualizations...")
    report_dir = config.paths.models_dir / model.name
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion matrix
    cm_path = report_dir / "confusion_matrix.png"
    save_confusion_matrix(y_test, y_pred, loader.get_class_names(), cm_path)
    print(f"   - Confusion matrix: {cm_path}")
    
    # 2. Classification report
    cr_path = report_dir / "classification_report.png"
    save_classification_report(y_test, y_pred, loader.get_class_names(), cr_path)
    print(f"   - Classification report: {cr_path}")
    
    # 3. Metrics summary
    metrics_path = report_dir / "metrics.png"
    save_metrics_summary(metrics, test_accuracy, metrics_path)
    print(f"   - Metrics summary: {metrics_path}")


if __name__ == "__main__":
    main()

