"""
Export model weights to JSON format for use in JavaScript
This creates a lightweight model that can be used in the browser without TensorFlow.js
"""
import json
import numpy as np
from tensorflow import keras

print("Loading Keras model...")
model = keras.models.load_model('model/keras_model.h5')

print("\nModel architecture:")
model.summary()

# Extract model architecture and weights
model_config = {
    'architecture': [],
    'weights': []
}

print("\nExtracting weights from each layer...")
for i, layer in enumerate(model.layers):
    layer_info = {
        'name': layer.name,
        'type': layer.__class__.__name__,
        'config': {}
    }
    
    # Get layer configuration
    if hasattr(layer, 'units'):
        layer_info['config']['units'] = int(layer.units)
    if hasattr(layer, 'activation'):
        layer_info['config']['activation'] = layer.activation.__name__
    if hasattr(layer, 'rate'):
        layer_info['config']['rate'] = float(layer.rate)
    
    model_config['architecture'].append(layer_info)
    
    # Get weights for Dense layers
    if layer.__class__.__name__ == 'Dense':
        weights = layer.get_weights()
        if len(weights) > 0:
            # weights[0] is the kernel (weights matrix)
            # weights[1] is the bias vector
            layer_weights = {
                'layer_index': i,
                'layer_name': layer.name,
                'kernel': weights[0].tolist(),  # Convert numpy array to list
                'bias': weights[1].tolist()
            }
            model_config['weights'].append(layer_weights)
            print(f"  Layer {i} ({layer.name}): kernel shape {weights[0].shape}, bias shape {weights[1].shape}")

# Save to JSON
output_file = 'model/model_weights.json'
print(f"\nSaving model to {output_file}...")

with open(output_file, 'w') as f:
    json.dump(model_config, f, indent=2)

print(f"✅ Model exported successfully!")
print(f"\nFile size: {len(json.dumps(model_config)) / 1024:.2f} KB")
print(f"\nThe model can now be used in JavaScript without TensorFlow.js")
print(f"You'll need to implement the forward pass (prediction) in JavaScript")
