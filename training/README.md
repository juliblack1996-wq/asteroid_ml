# Asteroid Size Predictor - Training Pipeline

Este directorio contiene el pipeline de entrenamiento para el modelo de predicción de tamaño de asteroides.

## Requisitos

### Instalación de Dependencias

```bash
pip install -r requirements.txt
```

**Nota:** La instalación de TensorFlow puede tardar varios minutos debido a las dependencias.

## Dataset

El modelo se entrena con el dataset de asteroides de Kaggle:
- **Dataset:** [sakhawat18/asteroid-dataset](https://www.kaggle.com/datasets/sakhawat18/asteroid-dataset)
- **Características utilizadas:**
  - `H`: Magnitud absoluta del asteroide
  - `diameter`: Diámetro del asteroide en kilómetros

El dataset se descarga automáticamente usando `kagglehub` cuando ejecutas el script de entrenamiento.

## Uso

### Entrenar el Modelo

Para entrenar el modelo completo:

```bash
python training/train_model.py
```

Este script ejecutará:
1. **Carga de datos:** Descarga y carga el dataset de Kaggle
2. **Exploración:** Analiza las distribuciones y rangos de datos
3. **Limpieza:** Maneja valores faltantes y outliers
4. **Preprocesamiento:** Divide los datos (80/20) y normaliza
5. **Entrenamiento:** Entrena una red neuronal con early stopping
6. **Evaluación:** Calcula MAE, RMSE y R²
7. **Exportación:** Convierte el modelo a formato TensorFlow.js

### Archivos Generados

Después del entrenamiento, se generarán los siguientes archivos en el directorio `model/`:

- `model.json` - Arquitectura del modelo en formato TensorFlow.js
- `group*.bin` - Pesos del modelo
- `normalization.json` - Parámetros de normalización (min, max, mean, std)
- `metrics.json` - Métricas de evaluación (MAE, RMSE, R²)
- `keras_model.h5` - Modelo en formato Keras (para referencia)

## Arquitectura del Modelo

El modelo es una red neuronal simple para regresión:

```
Input (1) → Dense(64, relu) → Dropout(0.2) → Dense(32, relu) → Dropout(0.2) → Dense(16, relu) → Dense(1)
```

- **Entrada:** Magnitud absoluta normalizada
- **Salida:** Diámetro del asteroide normalizado
- **Optimizador:** Adam (learning rate = 0.001)
- **Función de pérdida:** MSE (Mean Squared Error)
- **Early Stopping:** Paciencia de 15 épocas

## Tests

### Ejecutar Tests de Propiedades

```bash
python -m pytest training/test_train_model.py -v
```

### Tests Implementados

**Property-Based Tests:**
- ✅ Property 5: Completitud de división de datos (Requirements 2.3)
- ✅ Property 6: Validez de métricas de evaluación (Requirements 2.5)

**Unit Tests:**
- ✅ Manejo de columnas faltantes
- ✅ Formas correctas de preprocesamiento
- ✅ Cálculo de parámetros de normalización
- ✅ Creación de archivos de exportación
- ✅ Manejo de valores faltantes

## Parámetros de Normalización

El modelo utiliza normalización estándar (StandardScaler de scikit-learn):

```
normalized_value = (value - mean) / std
```

Los parámetros se guardan en `model/normalization.json` para su uso en la aplicación web.

## Notas

- El dataset contiene ~958,000 asteroides
- Solo ~136,000 tienen valores de diámetro (los demás se filtran)
- El rango típico de magnitud absoluta es de 3 a 35
- El rango de diámetros es de ~0.001 km a ~1000 km

## Troubleshooting

### Error: "No module named 'kagglehub'"
```bash
pip install kagglehub
```

### Error: "No module named 'tensorflow'"
```bash
pip install tensorflow
```

### Dataset no se descarga
Verifica tu conexión a internet. El dataset es de ~182 MB.
