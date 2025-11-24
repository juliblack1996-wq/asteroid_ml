"""
Simple script to convert Keras model to TensorFlow.js format
This uses the model's save method to create a format that can be converted
"""
import tensorflow as tf
from tensorflow import keras
import json
import os

print("Loading Keras model...")
model = keras.models.load_model('model/keras_model.h5')

print("\nModel architecture:")
model.summary()

# Save in TensorFlow SavedModel format (compatible with tfjs converter)
print("\nSaving model in SavedModel format...")
model.save('model/saved_model', save_format='tf')

print("\n✅ Model saved successfully!")
print("\nTo convert to TensorFlow.js, run:")
print("tensorflowjs_converter --input_format=tf_saved_model model/saved_model model/")
print("\nOr install tensorflowjs in a separate environment and convert there.")
