"""
Verify that the JavaScript model produces the same predictions as the Python model
"""
import numpy as np
from tensorflow import keras
import json

print("="*60)
print("VERIFICACIÓN: Modelo Python vs JavaScript")
print("="*60)

# Load the Keras model
print("\n1. Cargando modelo Keras...")
model = keras.models.load_model('model/keras_model.h5')

# Load normalization parameters
print("2. Cargando parámetros de normalización...")
with open('model/normalization.json', 'r') as f:
    norm_params = json.load(f)

# Test values
test_magnitudes = [12.0, 15.0, 18.0, 20.0]

print("\n3. Haciendo predicciones con el modelo Python:")
print("-" * 60)

for magnitude in test_magnitudes:
    # Normalize input
    normalized_input = (magnitude - norm_params['magnitude_mean']) / norm_params['magnitude_std']
    
    # Predict (model expects 2D array)
    normalized_output = model.predict(np.array([[normalized_input]]), verbose=0)[0][0]
    
    # Denormalize output
    predicted_size = (normalized_output * norm_params['size_std']) + norm_params['size_mean']
    
    # Ensure positive
    predicted_size = max(0.001, predicted_size)
    
    print(f"Magnitud H = {magnitude:5.1f} → Tamaño = {predicted_size:7.3f} km ({predicted_size*1000:6.0f} m)")

print("\n" + "="*60)
print("✅ Estas son las predicciones REALES del modelo entrenado")
print("="*60)
print("\nAhora compara estos valores con los que obtienes en:")
print("http://localhost:8000/test_model.html")
print("\nSi los valores son iguales (o muy similares), significa que")
print("el modelo JavaScript está usando los pesos reales! 🎯")
print("="*60)
