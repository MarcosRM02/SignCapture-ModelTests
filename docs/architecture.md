# Arquitectura

## Visión general

El módulo SignCapture-ModelTests implementa un sistema de entrenamiento y evaluación de modelos de clasificación basado en el patrón Factory. El diseño prioriza la extensibilidad, permitiendo incorporar nuevos algoritmos sin modificar el código existente, y mantiene separación clara entre datos, modelos, preprocesamiento e inferencia.

## Objetivos de diseño

- Facilitar la experimentación con múltiples algoritmos de clasificación.
- Mantener consistencia con el pipeline de preprocesamiento de SignCapture-ADA.
- Encapsular la complejidad de cada modelo detrás de una interfaz común.
- Garantizar reproducibilidad mediante configuración centralizada y seeds fijos.
- Separar entrenamiento de inferencia para facilitar despliegue.

## Arquitectura de componentes

### Capa de Datos (Data Layer)

**Responsabilidad**: Cargar datasets Gold y preparar features para entrenamiento.

**Módulo**: `src/data/__init__.py` (clase `DataLoader`)

**Flujo**:
1. Lee archivos CSV de `data/gold/` (train.csv, val.csv, test.csv).
2. Extrae columnas de features (63 landmarks + 14 ángulos = 77 features).
3. Codifica labels (letras A-Y) a índices numéricos mediante `LabelEncoder`.
4. Retorna arrays NumPy listos para entrenamiento.

**Salida**: 
- `X_train, y_train, X_val, y_val, X_test, y_test`: Arrays NumPy con features y labels codificados.
- `label_encoder`: Mapeo bidireccional letra ↔ índice numérico.

### Capa de Modelos (Model Layer)

**Responsabilidad**: Definir interfaz común y encapsular algoritmos de clasificación.

**Módulos**:
- `src/models/base.py`: Clase abstracta `BaseModel`.
- `src/models/registry.py`: Factory pattern y registro de modelos.
- `src/models/random_forest.py`: Implementación de Random Forest.
- `src/models/xgboost_model.py`: Implementación de XGBoost.

**Flujo**:
1. **Factory Pattern**: `create_model(name)` instancia el modelo correcto desde `MODEL_REGISTRY`.
2. **Entrenamiento**: Cada modelo implementa su lógica de `.train()` con hiperparámetros de configuración.
3. **Predicción**: `.predict()` retorna clases, `.predict_proba()` retorna probabilidades.
4. **Serialización**: `.save()` guarda modelo + metadatos en pickle, `.load()` los restaura.

**Decisiones técnicas**:

#### Clase abstracta BaseModel

Define el contrato que todos los modelos deben cumplir:

```python
class BaseModel(ABC):
    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None) -> dict[str, float]
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray
    
    @abstractmethod
    def save(self, path: Path) -> None
    
    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel"
```

Ventajas:
- Polimorfismo: todo modelo puede usarse de forma intercambiable.
- Testing simplificado: se pueden crear mocks de `BaseModel` fácilmente.

#### Factory Pattern

El diccionario `MODEL_REGISTRY` mapea strings a clases:

```python
MODEL_REGISTRY = {
    "random_forest": RandomForestClassifier,
    "xgboost": XGBoostClassifier,
}
```

Ventajas:
- Añadir un modelo nuevo requiere solo registrarlo en el diccionario.
- El código cliente no necesita conocer las clases concretas.

#### Serialización con contexto

Los modelos se guardan como diccionarios pickle que incluyen:

```python
{
    "model": trained_sklearn_model,
    "model_name": "random_forest",
    "label_encoder": fitted_LabelEncoder,
    "feature_columns": list_of_column_names,
}
```

Ventajas:
- El modelo es autocontenido: incluye todo lo necesario para inferencia.
- No se requiere configuración externa al deserializar.
- El `label_encoder` permite traducir predicciones a letras directamente.

### Capa de Preprocesamiento (Preprocessing Layer)

**Responsabilidad**: Replicar la normalización y feature engineering de SignCapture-ADA.

**Módulo**: `src/preprocessing/landmark_features.py`

**Flujo**:
1. **Normalización**: `normalize_landmarks_array()` escala landmarks a [-1, 1] por imagen.
2. **Feature engineering**: Calcula 14 ángulos entre falanges y dedos adyacentes.
3. **Vector de features**: `build_feature_vector()` concatena landmarks + ángulos en orden correcto.

**Funciones clave**:

```python
def normalize_landmarks_array(landmarks: np.ndarray) -> np.ndarray:
    """Normaliza landmarks [21, 3] a [-1, 1] usando lógica de ADA."""
    # Escalado min-max independiente por coordenada

def build_feature_vector(landmarks: np.ndarray) -> np.ndarray:
    """Construye vector completo: 63 landmarks + 14 ángulos = 77 features."""
    normalized = normalize_landmarks_array(landmarks)
    angles = _compute_angles_single(normalized)
    return np.concatenate([normalized.flatten(), angles])
```

**Decisiones técnicas**:

#### Normalización idéntica a ADA

El módulo **debe** replicar exactamente `SignCapture-ADA/src/gold/normalizer.py`:

```python
# Para cada coordenada (x, y, z) independientemente:
normalized = (coord - min_coord) / (max_coord - min_coord) * 2 - 1
```

Riesgo: Si la normalización diverge entre ADA y ModelTests, los modelos se entrenan con datos diferentes a los de producción, causando degradación de accuracy.

Estrategia de mitigación:
- Documentar explícitamente que ambos módulos deben sincronizarse.
- Considerar extraer la lógica a un paquete compartido.

#### Cálculo de ángulos

Se calculan 14 ángulos geométricos:

- **10 ángulos intra-dedo**: Entre falanges consecutivas (p.ej., thumb_1_2_3).
- **4 ángulos inter-dedos**: Entre dedos adyacentes usando puntos medios (p.ej., thumb_index_1_0_5).

Fórmula: Ángulo ABC = arccos((BA · BC) / (|BA| × |BC|))

**Por qué usar ángulos**:
- Invarianza a escala adicional (los ángulos no dependen de distancias absolutas).
- Representación geométrica interpretable (facilita debugging).
- Reduce correlaciones entre landmarks (features más independientes).

### Capa de Inferencia (Inference Layer)

**Responsabilidad**: Clasificar landmarks en tiempo real desde fuentes de video.

**Módulos**:
- `src/inference/landmark_detector.py`: Wrapper de MediaPipe HandLandmarker.
- `src/inference/webcam_demo.py`: Aplicación interactiva de clasificación.

**Flujo**:
1. **Captura**: Lee frames de webcam con OpenCV.
2. **Detección**: MediaPipe extrae landmarks de la mano.
3. **Normalización**: Se normalizan landmarks y calculan ángulos.
4. **Clasificación**: El modelo cargado predice la letra.
5. **Visualización**: Se dibuja la mano y la predicción sobre el frame.

**Decisiones técnicas**:

#### Detección con MediaPipe

Se usa `MediaPipe HandLandmarker` en modo VIDEO para optimizar rendimiento:

```python
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,  # Optimizado para video
    max_num_hands=1,
    min_detection_confidence=0.3,
)
```

Ventajas del modo VIDEO:
- Tracking temporal: reduce jitter entre frames.
- Mayor velocidad: optimizaciones para procesamiento secuencial.

#### Umbral de confianza

La demo permite ajustar `min_confidence` (default 0.4):

```python
if max_proba >= self.min_confidence:
    predicted_letter = self.class_names[predicted_class]
else:
    predicted_letter = "?"
```

Esto filtra predicciones poco confiables (p.ej., cuando no hay mano visible o está parcialmente ocluida).

## Estructura de carpetas

```
SignCapture-ModelTests/
├── src/
│   ├── config.py                      # Configuración centralizada
│   ├── data/
│   │   └── __init__.py                # DataLoader para Gold datasets
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                    # Clase abstracta BaseModel
│   │   ├── registry.py                # Factory + MODEL_REGISTRY
│   │   ├── random_forest.py           # RandomForestClassifier
│   │   └── xgboost_model.py           # XGBoostClassifier
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── landmark_features.py       # Normalización y ángulos
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── landmark_detector.py       # Wrapper MediaPipe
│   │   └── webcam_demo.py             # Demo interactiva
│   └── utils/
│       └── __init__.py                # Utilidades (set_seed)
├── config/
│   └── settings.yaml                  # Hiperparámetros
├── docs/                              # Documentación
│   ├── architecture.md
│   └── traceability.md
├── train.py                           # Script de entrenamiento
├── infer.py                           # Script de inferencia
└── requirements.txt
```

## Flujo de datos completo

```
┌─────────────────────────────────────────────────────────────┐
│ DATA LOADING                                                │
│                                                             │
│ data/gold/{train,val,test}.csv                              │
│     ↓                                                       │
│ DataLoader.load_data()                                      │
│     ├─> Verifica features angulares                         │
│     ├─> Calcula ángulos si faltan                           │
│     ├─> Extrae columnas de features (77 dims)               │
│     └─> Codifica labels con LabelEncoder                    │
│     ↓                                                       │
│ Output: X_train, y_train, X_val, y_val, X_test, y_test      │
│         (NumPy arrays float32)                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ MODEL TRAINING                                              │
│                                                             │
│ create_model(name) → Factory instantiation                  │
│     ↓                                                       │
│ model.train(X_train, y_train, X_val, y_val)                 │
│     ├─> Configura hiperparámetros desde settings.yaml       │
│     ├─> Entrena algoritmo (RandomForest o XGBoost)          │
│     └─> Evalúa en validación                                │
│     ↓                                                       │
│ model.save(path)                                            │
│     └─> Pickle: {model, model_name, label_encoder, ...}     │
│     ↓                                                       │
│ Output: models/{model_name}_asl.pkl                         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ EVALUATION                                                  │
│                                                             │
│ model.predict(X_test)                                       │
│     ↓                                                       │
│ classification_report(y_test, y_pred)                       │
│     ├─> Accuracy                                            │
│     └─> Precision / Recall / F1 por clase                   │
│     ↓                                                       │
│ Output: Métricas en consola                                 │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ INFERENCE                                                   │
│                                                             │
│ Webcam frame                                                │
│     ↓                                                       │
│ MediaPipe HandLandmarker.detect()                           │
│     ↓                                                       │
│ normalize_landmarks_array() + angle features                │
│     ↓                                                       │
│ build_feature_vector() → [77 dims]                          │
│     ↓                                                       │
│ model.predict_proba(features)                               │
│     ├─> Filter by min_confidence                            │
│     └─> Decode with label_encoder                           │
│     ↓                                                       │
│ Output: Predicted letter + probabilities                    │
└─────────────────────────────────────────────────────────────┘
```

## Configuración centralizada

### `src/config.py`

Define 6 dataclasses para configuración:

- **PathsConfig**: Rutas de datos (root, data_dir, gold_dir, models_dir).
- **TrainingConfig**: Seed de reproducibilidad, modelo por defecto.
- **MediaPipeConfig**: Parámetros de detección (solo para inferencia).
- **RandomForestConfig**: Hiperparámetros de Random Forest.
- **XGBoostConfig**: Hiperparámetros de XGBoost.
- **Config**: Instancia global que agrega todas las configuraciones.

La instancia global `config = Config()` carga desde `config/settings.yaml` con fallback a defaults.

### `config/settings.yaml`

Parámetros clave:

```yaml
general:
  seed: 42                       # Reproducibilidad

random_forest:
  n_estimators: 200              # Número de árboles
  max_depth: 20                  # Profundidad máxima
  class_weight: "balanced"       # Balance de clases

xgboost:
  n_estimators: 300
  learning_rate: 0.05
  objective: "multi:softprob"    # Clasificación multiclase
  tree_method: "hist"            # Algoritmo rápido
```

## Decisiones técnicas clave

### Pickle vs ONNX

- **Elección**: Serialización con pickle (formato nativo Python).
- **Alternativa**: Exportar a ONNX para interoperabilidad.
- **Justificación**: 
  - Pickle permite guardar objetos Python complejos (LabelEncoder, metadatos).
  - El despliegue está pensado para entorno Python (FastAPI en SignCapture-API).
  - ONNX sería necesario solo para despliegue en C++/JavaScript (fuera del alcance actual).

## Mantenibilidad

- **Modularidad**: Cada componente (data, models, preprocessing, inference) es independiente.
- **Configuración externa**: Hiperparámetros en YAML, no hardcodeados.
- **Interfaz común**: `BaseModel` garantiza consistencia entre algoritmos.
- **Factory Pattern**: Añadir modelos sin modificar código existente.
- **Reproducibilidad**: Seed fijo garantiza resultados deterministas.
- **Testing**: Estructura preparada para unit tests (aunque no implementados aún).
