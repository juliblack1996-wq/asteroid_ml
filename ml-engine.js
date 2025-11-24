/**
 * ML Engine for Asteroid Size Prediction
 * Pure JavaScript implementation - no TensorFlow.js required
 */

console.log('✅ ml-engine.js cargado correctamente');

class MLEngine {
    constructor() {
        this.model = null;
        this.normalizationParams = null;
        this.isReady = false;
    }

    /**
     * Load model weights and normalization parameters
     */
    async loadModel() {
        try {
            console.log('Loading model weights...');
            
            // Load model weights
            const modelResponse = await fetch('model/model_weights.json');
            this.model = await modelResponse.json();
            
            // Load normalization parameters
            const normResponse = await fetch('model/normalization.json');
            this.normalizationParams = await normResponse.json();
            
            this.isReady = true;
            console.log('✅ Model loaded successfully!');
            console.log('Model architecture:', this.model.architecture.length, 'layers');
            
            return true;
        } catch (error) {
            console.error('Error loading model:', error);
            throw new Error('Failed to load model: ' + error.message);
        }
    }

    /**
     * Normalize input magnitude value
     */
    normalizeInput(magnitude) {
        const { magnitude_mean, magnitude_std } = this.normalizationParams;
        return (magnitude - magnitude_mean) / magnitude_std;
    }

    /**
     * Denormalize output size value
     */
    denormalizeOutput(normalizedSize) {
        const { size_mean, size_std } = this.normalizationParams;
        return (normalizedSize * size_std) + size_mean;
    }

    /**
     * ReLU activation function
     */
    relu(x) {
        return Math.max(0, x);
    }

    /**
     * Matrix-vector multiplication
     */
    matmul(matrix, vector) {
        const result = [];
        for (let i = 0; i < matrix[0].length; i++) {
            let sum = 0;
            for (let j = 0; j < vector.length; j++) {
                sum += matrix[j][i] * vector[j];
            }
            result.push(sum);
        }
        return result;
    }

    /**
     * Add bias to vector
     */
    addBias(vector, bias) {
        return vector.map((v, i) => v + bias[i]);
    }

    /**
     * Apply activation function to vector
     */
    applyActivation(vector, activation) {
        if (activation === 'relu') {
            return vector.map(v => this.relu(v));
        }
        // Linear activation (no change)
        return vector;
    }

    /**
     * Forward pass through the neural network
     */
    forward(input) {
        let activation = [input]; // Start with input as 1D array
        
        // Process each Dense layer
        for (const layerWeights of this.model.weights) {
            const layerInfo = this.model.architecture[layerWeights.layer_index];
            
            // Matrix multiplication: activation * kernel + bias
            activation = this.matmul(layerWeights.kernel, activation);
            activation = this.addBias(activation, layerWeights.bias);
            
            // Apply activation function
            const activationFunc = layerInfo.config.activation || 'linear';
            activation = this.applyActivation(activation, activationFunc);
        }
        
        // Return the output (should be a single value)
        return activation[0];
    }

    /**
     * Predict asteroid size from absolute magnitude
     * @param {number} magnitude - Absolute magnitude (H) of the asteroid
     * @returns {Object} Prediction result with size in km
     */
    async predict(magnitude) {
        if (!this.isReady) {
            throw new Error('Model not loaded. Call loadModel() first.');
        }

        // Validate input
        if (typeof magnitude !== 'number' || isNaN(magnitude)) {
            throw new Error('Invalid input: magnitude must be a number');
        }

        // Check if magnitude is in valid range
        const { magnitude_min, magnitude_max } = this.normalizationParams;
        if (magnitude < magnitude_min || magnitude > magnitude_max) {
            console.warn(`Warning: magnitude ${magnitude} is outside training range [${magnitude_min}, ${magnitude_max}]`);
        }

        try {
            // Normalize input
            const normalizedInput = this.normalizeInput(magnitude);
            
            // Run forward pass
            const normalizedOutput = this.forward(normalizedInput);
            
            // Denormalize output
            const predictedSize = this.denormalizeOutput(normalizedOutput);
            
            // Ensure positive size
            const size = Math.max(0.001, predictedSize);
            
            return {
                size_km: size,
                magnitude: magnitude,
                timestamp: new Date()
            };
        } catch (error) {
            console.error('Prediction error:', error);
            throw new Error('Prediction failed: ' + error.message);
        }
    }

    /**
     * Check if model is ready for predictions
     */
    isModelReady() {
        return this.isReady;
    }

    /**
     * Get model information
     */
    getModelInfo() {
        if (!this.isReady) {
            return null;
        }

        return {
            layers: this.model.architecture.length,
            parameters: this.model.weights.reduce((sum, layer) => {
                const kernelSize = layer.kernel.length * layer.kernel[0].length;
                const biasSize = layer.bias.length;
                return sum + kernelSize + biasSize;
            }, 0),
            magnitudeRange: [
                this.normalizationParams.magnitude_min,
                this.normalizationParams.magnitude_max
            ],
            sizeRange: [
                this.normalizationParams.size_min,
                this.normalizationParams.size_max
            ]
        };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MLEngine;
}
