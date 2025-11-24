"""
Convert Keras model to TensorFlow.js format
"""
import tensorflowjs as tfjs

print("Converting model to TensorFlow.js format...")
tfjs.converters.convert_tf_saved_model(
    'model/keras_model.h5',
    'model',
    'tfjs_layers_model'
)
print("Conversion complete!")
