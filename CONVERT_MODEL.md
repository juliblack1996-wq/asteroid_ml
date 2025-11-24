# Conversión del Modelo a TensorFlow.js

El modelo ha sido entrenado exitosamente y guardado en formato Keras (`model/keras_model.h5`).

## Métricas del Modelo

- **MAE (Error Absoluto Medio)**: 1.29 km
- **RMSE (Error Cuadrático Medio)**: 1.61 km  
- **R² (Coeficiente de Determinación)**: 0.56
- **Muestras de Prueba**: 25,441

## Archivos Generados

✅ `model/keras_model.h5` - Modelo entrenado en formato Keras
✅ `model/normalization.json` - Parámetros de normalización
✅ `model/metrics.json` - Métricas de evaluación

## ⚠️ Problema de Dependencias

Hay conflictos de versiones entre TensorFlow 2.15 y TensorFlowJS. 

## Solución: Usar el Modelo Directamente

**Opción 1: Crear entorno separado para conversión**
```bash
# Crear nuevo entorno solo para conversión
python -m venv tfjs_env
tfjs_env\Scripts\activate
pip install tensorflow==2.13.0 tensorflowjs==3.18.0
python training/convert_model_simple.py
tensorflowjs_converter --input_format=keras model/keras_model.h5 model/
```

**Opción 2: Usar Google Colab (Recomendado)**
1. Sube `model/keras_model.h5` a Google Colab
2. Ejecuta:
```python
!pip install tensorflowjs
import tensorflowjs as tfjs
from tensorflow import keras
model = keras.models.load_model('keras_model.h5')
tfjs.converters.save_keras_model(model, './')
```
3. Descarga los archivos `model.json` y `*.bin` generados

**Opción 3: Continuar sin conversión (para desarrollo)**
Por ahora podemos continuar con el desarrollo de la interfaz web y hacer la conversión después.
