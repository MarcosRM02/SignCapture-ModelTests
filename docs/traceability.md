# Trazabilidad y mantenimiento

## Responsabilidades por módulo

### Data (Carga de datos)

Responsable de cargar datasets Gold y preparar features para entrenamiento. No debe contener lógica de modelado ni preprocesamiento complejo.

### Models (Algoritmos de clasificación)

Responsable de encapsular algoritmos de ML con interfaz común. No debe contener lógica de carga de datos ni preprocesamiento de landmarks.

### Preprocessing (Normalización y features)

Responsable de replicar exactamente la normalización y feature engineering de SignCapture-ADA. No debe contener lógica de entrenamiento ni evaluación.

### Inference (Clasificación en tiempo real)

Responsable de integrar detección de landmarks con modelos entrenados para clasificación interactiva. No debe contener lógica de entrenamiento.

### Config (Configuración centralizada)

Responsable de cargar y exponer parámetros desde archivos YAML y variables de entorno. Debe ser el único punto de acceso a configuración.

### Utils (Utilidades)

Responsables de funciones auxiliares transversales (set_seed, logging, etc.). No debe contener lógica de negocio específica de clasificación.

## Matriz de trazabilidad

| Necesidad funcional | Módulo principal | Elementos implicados |
| --- | --- | --- |
| Cargar datos de Gold | `src/data/__init__.py` | `DataLoader.load_data`, `_load_split`, `_prepare_split` |
| Codificar labels | `src/data/__init__.py` | `DataLoader.label_encoder`, `LabelEncoder` (sklearn) |
| Normalizar landmarks | `src/preprocessing/landmark_features.py` | `normalize_landmarks_array`, escalado min-max |
| Calcular features angulares | `src/preprocessing/landmark_features.py` | `_compute_angles_single`, `_compute_angles_batch`, `_compute_angle_degrees`, `_compute_batch_angle_degrees` |
| Añadir features angulares a DataFrame | `src/preprocessing/landmark_features.py` | `add_angle_features_to_dataframe` |
| Construir vector de features | `src/preprocessing/landmark_features.py` | `build_feature_vector`, concatenación landmarks + ángulos |
| Procesar landmarks para inferencia | `src/inference/__init__.py` | `LandmarkProcessor.process_landmarks`, `build_features`, `normalize_landmarks` |
| Crear modelo desde Factory | `src/models/registry.py` | `create_model`, `MODEL_REGISTRY`, `available_models` |
| Entrenar Random Forest | `src/models/random_forest.py` | `RandomForestClassifier.train`, `sklearn.ensemble.RandomForestClassifier` |
| Entrenar XGBoost | `src/models/xgboost_model.py` | `XGBoostClassifier.train`, `xgboost.XGBClassifier` |
| Predecir clases | `src/models/base.py` | `BaseModel.predict` (abstracto), implementaciones concretas |
| Predecir probabilidades | `src/models/base.py` | `BaseModel.predict_proba` (abstracto), implementaciones concretas |
| Serializar modelo | `src/models/base.py` | `BaseModel.save`, `pickle.dump` |
| Deserializar modelo | `src/models/registry.py` | `load_model`, `_infer_model_name`, `pickle.load` |
| Detectar landmarks en webcam | `src/inference/landmark_detector.py` | `LandmarkDetector.detect_landmarks`, `annotate_image`, `MediaPipe HandLandmarker` |
| Clasificar desde webcam | `src/inference/webcam_demo.py` | `WebcamDemo.run`, `_draw_prediction`, `cv2.VideoCapture` |
| Cargar configuración | `src/config.py` | `Config`, `PathsConfig`, `TrainingConfig`, `RandomForestConfig`, `XGBoostConfig`, `MediaPipeConfig`, `load_yaml` |
| Fijar seed global | `src/utils/__init__.py` | `set_seed`, `np.random.seed`, `random.seed` |

## Reglas de mantenimiento

- Si cambia el formato de features en SignCapture-ADA Gold, actualizar `src/preprocessing/landmark_features.py` (constantes `LANDMARK_FEATURE_COLUMNS`, `ANGLE_FEATURE_COLUMNS`).
- Si cambia la normalización en SignCapture-ADA, actualizar `normalize_landmarks_array` en `src/preprocessing/landmark_features.py`.
- Si cambian las features angulares en SignCapture-ADA, actualizar `_compute_angles_single` y `_compute_angles_batch`.
- Si se añade un nuevo modelo, registrarlo en `MODEL_REGISTRY` y crear dataclass de configuración en `src/config.py`.
- Si cambian hiperparámetros de modelos, actualizar `config/settings.yaml` sin modificar código.
- Si cambia la estructura de carpetas (rutas de datos/modelos), actualizar `PathsConfig` en `src/config.py`.
- Cualquier cambio en la lógica de preprocesamiento debe sincronizarse con SignCapture-API para mantener consistencia entre entrenamiento e inferencia.

## Convenciones de documentación

- Cada módulo debe describir su responsabilidad en el docstring principal.
- Cada clase debe explicar su propósito, atributos principales y ejemplo de uso.
- Cada función pública debe documentar entradas, salidas y comportamiento (siguiendo formato español).
- Las decisiones de arquitectura deben reflejarse en `docs/architecture.md`.
- Los parámetros configurables deben reflejarse en `README.md` y `config/settings.yaml`.
- Los cambios que afectan a otros módulos (ADA, API) deben documentarse explícitamente en commits y PRs.

## Dependencias críticas

### SignCapture-ADA → ModelTests

ModelTests **depende** de SignCapture-ADA para:

1. **Datos Gold**: Los CSVs `train.csv`, `val.csv`, `test.csv` deben existir en `data/gold/`.
2. **Formato de features**: Las columnas de landmarks y ángulos deben seguir la convención de ADA.
3. **Normalización**: La lógica de normalización debe ser idéntica.

**Riesgo**: Si ADA cambia su pipeline Gold sin actualizar ModelTests, los modelos se entrenan con features inconsistentes.

**Estrategia de mitigación**:
- Documentar explícitamente que `src/preprocessing/landmark_features.py` debe sincronizarse con `SignCapture-ADA/src/gold/normalizer.py`.
- Considerar crear tests de integración que validen consistencia de features.
- Versionar datos Gold con DVC o similar para rastrear cambios.

### ModelTests → SignCapture-API

SignCapture-API **consume** modelos de ModelTests:

1. **Modelos serializados**: La API carga archivos `.pkl` desde `models/`.
2. **Formato de serialización**: La API espera diccionario pickle con claves específicas (`model`, `model_name`, `label_encoder`).
3. **Preprocesamiento**: La API debe replicar exactamente `build_feature_vector`.

**Riesgo**: Si ModelTests cambia el formato de serialización o preprocesamiento sin actualizar API, la inferencia falla.

**Estrategia de mitigación**:
- Mantener formato de serialización estable (contrato de interfaz).
- Documentar explícitamente qué módulos de API deben sincronizarse.
- Considerar crear una librería compartida de preprocesamiento.

## Convenciones de código

### Nombres de archivos y módulos

- Archivos de módulo: `snake_case.py` (p.ej., `landmark_features.py`).
- Clases: `PascalCase` (p.ej., `DataLoader`, `BaseModel`).
- Funciones y variables: `snake_case` (p.ej., `load_data`, `feature_columns`).
- Constantes: `UPPER_SNAKE_CASE` (p.ej., `MODEL_REGISTRY`, `ANGLE_FEATURE_COLUMNS`).


### Docstrings

- Usar formato de docstring para funciones principales.
- Describir propósito, parámetros, retorno y excepciones.

Ejemplo:

```python
def normalize_landmarks_array(landmarks: np.ndarray) -> np.ndarray:
    """Normaliza landmarks a [-1, 1] usando lógica de ADA.
    
    Args:
        landmarks: Array de landmarks [21, 3] con coordenadas (x, y, z).
    
    Returns:
        Array normalizado [21, 3] con valores en rango [-1, 1].
    
    Raises:
        ValueError: Si la forma del array no es (21, 3).
    """
```

### Imports

- Ordenar imports en tres bloques:
  1. Librería estándar (pathlib, pickle, abc, etc.)
  2. Librerías de terceros (numpy, pandas, sklearn, etc.)
  3. Imports locales (src.config, src.models, etc.)
- Usar imports absolutos desde la raíz del proyecto (`from src.models import ...`).
- Evitar imports circulares.

Ejemplo:

```python
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import config
from src.preprocessing import build_feature_vector
```

## Checklist de cambios

Al modificar el código, verificar:

- [ ] **Type hints**: Todas las funciones públicas tienen anotaciones de tipo.
- [ ] **Docstrings**: Funciones principales tienen documentación en español.
- [ ] **Tests**: Si se modifica lógica de negocio, añadir/actualizar tests.
- [ ] **Config sync**: Si se añaden parámetros, actualizar `config/settings.yaml` y `src/config.py`.
- [ ] **README sync**: Si se modifica funcionalidad, actualizar README.
- [ ] **ADA sync**: Si se modifica preprocesamiento, verificar consistencia con ADA.
- [ ] **API sync**: Si se modifica serialización o features, notificar al equipo de API.
- [ ] **Reproducibilidad**: Si se usa aleatoriedad, usar seed de configuración.

## Patrones de diseño aplicados

### Factory Pattern

**Ubicación**: `src/models/registry.py`

**Propósito**: Crear instancias de modelos sin conocer clases concretas.

**Ventajas**:
- Añadir nuevos modelos sin modificar código cliente.
- Centralizar lógica de creación e inferencia de tipos.
- Facilitar testing con mocks.

**Ejemplo de uso**:

```python
# En lugar de:
if model_name == "random_forest":
    model = RandomForestClassifier()
elif model_name == "xgboost":
    model = XGBoostClassifier()

# Se usa:
model = create_model(model_name)
```

### Strategy Pattern (implícito)

**Ubicación**: `src/models/base.py` + implementaciones concretas

**Propósito**: Encapsular algoritmos de clasificación detrás de interfaz común.

**Ventajas**:
- Intercambiar modelos en runtime sin modificar código.
- Polimorfismo: todo modelo implementa `train`, `predict`, `save`, `load`.

**Ejemplo de uso**:

```python
# El código cliente no necesita conocer el tipo concreto:
model: BaseModel = create_model("random_forest")
model.train(X_train, y_train)
predictions = model.predict(X_test)
```

### Singleton Pattern (configuración)

**Ubicación**: `src/config.py`

**Propósito**: Instancia global de configuración compartida.

**Implementación**:

```python
config = Config()  # Instancia única compartida en todo el proyecto
```

**Ventajas**:
- Evita cargar configuración múltiples veces.
- Punto único de acceso a parámetros.

**Desventajas**:
- Estado global (dificulta testing en algunos casos).
- No thread-safe (pero no es necesario para este proyecto).

## Testing (pendiente de implementar)

### Unit tests

Crear tests para:

- `src/preprocessing/landmark_features.py`: Verificar normalización y cálculo de ángulos.
- `src/data/__init__.py`: Verificar carga de datos y encoding de labels.
- `src/models/registry.py`: Verificar factory pattern y carga de modelos.

### Integration tests

Crear tests para:

- Pipeline completo: `load_data` → `train` → `predict` → `save` → `load`.
- Consistencia con ADA: Verificar que features calculadas coinciden con Gold CSVs.

### Estructura sugerida

```
tests/
├── __init__.py
├── test_data.py
├── test_preprocessing.py
├── test_models.py
├── test_registry.py
└── fixtures/
    └── sample_gold.csv
```

## Versionado de modelos (pendiente de implementar)

### Estrategia recomendada

1. **DVC (Data Version Control)**: Versionar datasets Gold y modelos entrenados.
2. **MLflow**: Trackear experimentos (hiperparámetros, métricas, artefactos).
3. **Naming convention**: Incluir timestamp y hash de configuración en nombres de modelos.

### Ejemplo de naming

```
models/
├── random_forest_asl_20260331_a3f2b1.pkl
├── xgboost_asl_20260331_c8d4e9.pkl
└── random_forest_asl_latest.pkl  # Symlink al mejor modelo
```

## Troubleshooting común

### Error: "Feature column mismatch between train and other splits"

**Causa**: Los CSVs Gold de train, val, test tienen columnas diferentes.

**Solución**: Reejecutar pipeline Gold de SignCapture-ADA para regenerar splits consistentes.

### Error: "Unable to infer model type for X"

**Causa**: El modelo pickle no contiene metadatos reconocibles.

**Solución**: Asegurarse de que el modelo fue guardado con `.save()` de `BaseModel` (no con `pickle.dump` directo).

### Warning: sklearn version mismatch

**Causa**: El modelo fue entrenado con una versión de scikit-learn diferente a la actual.

**Solución**: 
- Usar entorno virtual con versiones fijas en `requirements.txt`.
- Reentrenar el modelo con la versión actual.

### Accuracy muy baja en inferencia pero alta en test

**Causa**: Inconsistencia en normalización entre entrenamiento e inferencia.

**Solución**: 
- Verificar que `src/preprocessing/landmark_features.py` y `src/inference/webcam_demo.py` usan la misma normalización.
- Comparar con `SignCapture-ADA/src/gold/normalizer.py`.

## Contribución de nuevos modelos

### Pasos para añadir un modelo

1. **Crear implementación**:

```python
# src/models/mi_modelo.py
from src.models.base import BaseModel

class MiModelo(BaseModel):
    def __init__(self, model_config: dict | None = None):
        config = model_config or config.mi_modelo.__dict__
        super().__init__(name="mi_modelo", config=config)
        # Inicializar modelo interno
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        # Implementar entrenamiento
        pass
    
    # ... implementar otros métodos abstractos
```

2. **Registrar en Factory**:

```python
# src/models/registry.py
from src.models.mi_modelo import MiModelo

MODEL_REGISTRY = {
    "random_forest": RandomForestClassifier,
    "xgboost": XGBoostClassifier,
    "mi_modelo": MiModelo,  # <-- añadir aquí
}
```

3. **Añadir configuración**:

```python
# src/config.py
@dataclass
class MiModeloConfig:
    param1: int = 100
    param2: float = 0.01
    # ...

    def __init__(self):
        config_path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
        if config_path.exists():
            settings = load_yaml(config_path)
            mi_config = settings.get("mi_modelo", {})
            self.param1 = mi_config.get("param1", 100)
            # ...

@dataclass
class Config:
    # ...
    mi_modelo: MiModeloConfig
    
    def __init__(self):
        # ...
        self.mi_modelo = MiModeloConfig()
```

4. **Añadir hiperparámetros**:

```yaml
# config/settings.yaml
mi_modelo:
  param1: 100
  param2: 0.01
```

5. **Entrenar**: `python train.py --model mi_modelo`

## Conclusión

Este documento establece las responsabilidades de cada módulo, las reglas de mantenimiento y las convenciones de código para garantizar consistencia y trazabilidad en el proyecto. Cualquier cambio debe reflejarse en esta documentación para mantener el sistema predecible y mantenible.
