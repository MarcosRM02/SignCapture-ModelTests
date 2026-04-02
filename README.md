# SignCapture ModelTests

Módulo de entrenamiento, evaluación y comparación de modelos para la clasificación de lenguaje de signos americano (ASL). El sistema utiliza datos preprocesados de la capa Gold, entrena modelos de scikit-learn, los exporta en formato serializado (pickle) y proporciona herramientas de inferencia en tiempo real.

## Objetivo

El proyecto proporciona un framework extensible para experimentar con modelos de clasificación mediante un patrón Factory que facilita la incorporación de nuevos algoritmos. Actualmente soporta:

- **Random Forest**: Clasificador basado en ensamble de árboles de decisión.
- **XGBoost**: Algoritmo de gradient boosting optimizado para velocidad y rendimiento.

Documentación ampliada:

- [docs/architecture.md](./docs/architecture.md)
- [docs/traceability.md](./docs/traceability.md)

## Puesta en marcha

1. Crear y activar el entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate 
```

2. Instalar dependencias:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Configurar variables de entorno:

Crea un archivo `.env` en la raíz del proyecto con:

```env
SIGNCAPTURE_ROOT=/ruta/absoluta/a/SignCapture
```

4. Verificar configuración:

Revisa y ajusta los parámetros en `config/settings.yaml` según tus necesidades:

```yaml
general:
  seed: 42

mediapipe:
  min_detection_confidence: 0.3
  max_num_hands: 1
  model_path: "../models/hand_landmarker.task"

random_forest:
  n_estimators: 200
  max_depth: 20
  min_samples_split: 5
  min_samples_leaf: 2
  max_features: "sqrt"
  class_weight: "balanced"

xgboost:
  n_estimators: 300
  max_depth: 8
  learning_rate: 0.05
  min_child_weight: 1
  subsample: 0.9
  colsample_bytree: 0.9
  objective: "multi:softprob"
  tree_method: "hist"
```

## Entrenamiento de modelos

### Entrenar Random Forest (por defecto)

```bash
python train.py
```

### Entrenar XGBoost

```bash
python train.py --model xgboost
```

Resultado:
- Modelo serializado en `{SIGNCAPTURE_ROOT}/models/{model_name}_asl.pkl`
- Reporte de métricas: accuracy, precision, recall, F1-score por clase

## Inferencia en tiempo real

### Demo con webcam

Ejecutar clasificación en tiempo real usando la webcam:

```bash
python infer.py
```

### Opciones avanzadas

```bash
# Usar un modelo específico
python infer.py --model ../models/xgboost_asl.pkl

# Seleccionar otra cámara
python infer.py --camera 1

# Ajustar umbral de confianza
python infer.py --confidence 0.7
```

Controles durante la ejecución:
- **q**: Salir de la aplicación
- La aplicación muestra en tiempo real:
  - Landmarks de la mano detectada
  - Letra predicha con mayor confianza

## Flujo funcional

1. **Carga de datos**: El `DataLoader` lee los CSVs de `data/gold/` (train, val, test).
2. **Preprocesamiento**: Si faltan features angulares, se calculan automáticamente.
3. **Entrenamiento**: El modelo seleccionado se entrena usando los hiperparámetros de configuración.
4. **Evaluación**: Se evalúa el modelo en validación y test, reportando métricas completas.
5. **Serialización**: El modelo entrenado se guarda en pickle con metadatos (nombre, clase, label_encoder).
6. **Inferencia**: El script `infer.py` carga el modelo y clasifica frames de video en tiempo real.

## Configuración

Las variables de entorno disponibles se documentan en `.env.example`:

- `SIGNCAPTURE_ROOT`: ruta raíz del proyecto que sigue la siguiente estructura:

```
SignCapture/
├── SignCapture-ADA/
├── SignCapture-ModelTests/
├── SignCapture-API/
├── SignCapture-Front/
├── data/
│   ├── bronze/
│   ├── silver/
│   └── gold/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
└── models/
    ├── hand_landmarker.task
    ├── random_forest_asl.pkl
    └── xgboost_asl.pkl
```

Los parámetros de modelos se configuran en `config/settings.yaml`:

- `general.seed`: semilla para reproducibilidad
- `random_forest.*`: hiperparámetros de Random Forest (n_estimators, max_depth, etc.)
- `xgboost.*`: hiperparámetros de XGBoost (learning_rate, subsample, etc.)
- `mediapipe.*`: configuración de detección de landmarks (solo para inferencia)

## Estructura de código

### Módulos principales

```
SignCapture-ModelTests/
├── src/
│   ├── config.py                      # Configuración centralizada
│   ├── data/
│   │   └── __init__.py                # DataLoader para Gold datasets
│   ├── models/
│   │   ├── base.py                    # Clase abstracta BaseModel
│   │   ├── registry.py                # Factory Pattern + MODEL_REGISTRY
│   │   ├── random_forest.py           # Implementación de Random Forest
│   │   └── xgboost_model.py           # Implementación de XGBoost
│   ├── preprocessing/
│   │   └── landmark_features.py       # Normalización y features angulares
│   ├── inference/
│   │   ├── landmark_detector.py       # Wrapper de MediaPipe HandLandmarker
│   │   └── webcam_demo.py             # Demo interactiva con webcam
│   └── utils/
│       └── __init__.py                # Utilidades (set_seed)
├── config/
│   └── settings.yaml                  # Hiperparámetros de modelos
├── train.py                           # Script de entrenamiento
├── infer.py                           # Script de inferencia webcam
└── requirements.txt                   # Dependencias Python
```

## Añadir un nuevo modelo

1. **Crear implementación**: Añade un archivo en `src/models/` que herede de `BaseModel`.

```python
from src.models.base import BaseModel

class MiNuevoModelo(BaseModel):
    def train(self, X_train, y_train, X_val=None, y_val=None):
        # Implementar lógica de entrenamiento
        pass
    
    def predict(self, X):
        # Implementar predicción
        pass
    
    # ... otros métodos abstractos
```

2. **Registrar en Factory**: Edita `src/models/registry.py`:

```python
from src.models.mi_nuevo_modelo import MiNuevoModelo

MODEL_REGISTRY = {
    "random_forest": RandomForestClassifier,
    "xgboost": XGBoostClassifier,
    "mi_modelo": MiNuevoModelo,  # <-- añadir aquí
}
```

3. **Configurar hiperparámetros**: Añade sección en `config/settings.yaml`:

```yaml
mi_modelo:
  param1: valor1
  param2: valor2
```

4. **Actualizar config.py**: Añade dataclass de configuración:

```python
@dataclass
class MiModeloConfig:
    param1: int = valor1
    param2: str = "valor2"
    # ...
```

5. **Entrenar**: `python train.py --model mi_modelo`

## Características técnicas

### Patrón Factory

El sistema utiliza un `MODEL_REGISTRY` dict que mapea nombres de modelos a clases:

```python
MODEL_REGISTRY = {
    "random_forest": RandomForestClassifier,
    "xgboost": XGBoostClassifier,
}
```

Esto permite crear modelos dinámicamente:

```python
model = create_model("random_forest")
```

### Clase base abstracta

Todos los modelos heredan de `BaseModel` y deben implementar:

- `train()`: Entrena el modelo con datos de entrenamiento
- `predict()`: Predice clases para nuevas muestras
- `predict_proba()`: Retorna probabilidades por clase
- `save()`: Serializa el modelo a disco
- `load()`: Deserializa el modelo desde disco

### Normalización consistente

El módulo `preprocessing/landmark_features.py` replica **exactamente** la lógica de normalización de SignCapture-ADA:

- Normalización min-max por imagen a rango [-1, 1]
- Cálculo de 14 ángulos entre falanges y dedos
- Orden de features idéntico al pipeline Gold

**Importante**: Cualquier cambio en features de ADA debe sincronizarse aquí.

## Integración con otros módulos

Este módulo depende de:

- **SignCapture-ADA**: Lee datos de `data/gold/` generados por el pipeline ADA.

Este módulo es consumido por:

- **SignCapture-API**: Carga modelos desde `models/` para servir predicciones REST.

**Crítico**: La normalización en `src/preprocessing/landmark_features.py` debe mantenerse sincronizada con `SignCapture-ADA/src/gold/normalizer.py`.

## Evaluación de modelos

Los modelos se evalúan automáticamente durante el entrenamiento:

1. **Validación**: Métricas en conjunto de validación durante entrenamiento
2. **Test**: Métricas finales en conjunto de test (sin data leakage)

Métricas reportadas:
- **Accuracy**: Porcentaje de predicciones correctas
- **Precision**: Por clase (positivos verdaderos / positivos predichos)
- **Recall**: Por clase (positivos verdaderos / positivos reales)
- **F1-score**: Media armónica de precision y recall
- **Support**: Número de muestras por clase en test

## Notas

- El módulo es independiente de SignCapture-ADA pero requiere sus datos de salida.
- Los modelos se entrenan con CPU por defecto (scikit-learn, xgboost optimizados para CPU).
- La inferencia en webcam requiere cámara disponible y modelo entrenado.
- Los landmarks se normalizan por imagen para invarianza a escala y posición.
- El seed fijo (42) garantiza reproducibilidad en splits y entrenamiento.