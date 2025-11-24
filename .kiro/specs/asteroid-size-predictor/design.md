# Design Document

## Overview

El sistema de predicción de tamaño de asteroides es una aplicación web estática de una sola página (SPA) que utiliza TensorFlow.js para ejecutar inferencias de Machine Learning directamente en el navegador. La aplicación se alojará en GitHub Pages y consistirá en tres componentes principales: un script de entrenamiento en Python para generar el modelo, archivos estáticos HTML/CSS/JavaScript para la interfaz de usuario, y el modelo ML convertido a formato TensorFlow.js.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     GitHub Pages (Static)                    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                    Frontend (Browser)                   │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │ │
│  │  │   UI Layer   │  │  ML Engine   │  │ Visualization│ │ │
│  │  │  (HTML/CSS)  │  │(TensorFlow.js)│  │   (Chart.js) │ │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘ │ │
│  │         │                  │                  │         │ │
│  │         └──────────────────┴──────────────────┘         │ │
│  │                            │                            │ │
│  │                   ┌────────▼────────┐                   │ │
│  │                   │  Model Files    │                   │ │
│  │                   │  (model.json)   │                   │ │
│  │                   └─────────────────┘                   │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              Training Pipeline (Local/Offline)               │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐│
│  │   Kaggle   │─▶│  Training  │─▶│  Model Conversion      ││
│  │  Dataset   │  │  (Python)  │  │  (TensorFlow.js)       ││
│  └────────────┘  └────────────┘  └────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

**Frontend:**
- HTML5 para estructura
- CSS3 con diseño responsivo (sin frameworks para mantenerlo ligero)
- JavaScript vanilla para lógica de UI
- TensorFlow.js para inferencia del modelo ML
- Chart.js para visualizaciones

**Training Pipeline:**
- Python 3.8+
- pandas para manipulación de datos
- scikit-learn para preprocesamiento y modelo
- TensorFlow/Keras para entrenamiento
- tensorflowjs_converter para conversión del modelo

**Hosting:**
- GitHub Pages (servicio gratuito de hosting estático)

## Components and Interfaces

### 1. Training Pipeline (train_model.py)

**Responsabilidades:**
- Descargar y cargar el dataset de Kaggle
- Explorar y limpiar los datos
- Entrenar un modelo de regresión
- Evaluar el modelo
- Convertir el modelo a formato TensorFlow.js
- Guardar estadísticas de normalización

**Interfaces:**
```python
class AsteroidModelTrainer:
    def load_data(self, path: str) -> pd.DataFrame
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]
    def train_model(self, X_train, y_train) -> keras.Model
    def evaluate_model(self, model, X_test, y_test) -> Dict[str, float]
    def export_model(self, model, output_path: str) -> None
    def save_normalization_params(self, params: Dict, output_path: str) -> None
```

### 2. ML Engine (ml-engine.js)

**Responsabilidades:**
- Cargar el modelo TensorFlow.js
- Normalizar inputs del usuario
- Ejecutar predicciones
- Desnormalizar outputs

**Interfaces:**
```javascript
class MLEngine {
    async loadModel(modelPath)
    async predict(absoluteMagnitude)
    normalizeInput(value)
    denormalizeOutput(value)
    isModelReady()
}
```

### 3. UI Controller (app.js)

**Responsabilidades:**
- Manejar eventos de usuario
- Validar inputs
- Coordinar entre UI y ML Engine
- Gestionar historial de predicciones
- Actualizar visualizaciones

**Interfaces:**
```javascript
class AppController {
    initialize()
    handlePredictionRequest(magnitude)
    validateInput(value)
    displayPrediction(result)
    displayError(message)
    updateHistory(prediction)
    clearHistory()
}
```

### 4. Visualization Component (visualization.js)

**Responsabilidades:**
- Crear gráficos comparativos
- Mostrar historial de predicciones
- Renderizar indicadores de confianza

**Interfaces:**
```javascript
class Visualization {
    createSizeComparison(predictedSize)
    updateHistoryChart(history)
    showConfidenceInterval(prediction, confidence)
}
```

## Data Models

### Input Data (Training)

```python
{
    "absolute_magnitude": float,  # Brillo absoluto (H)
    "estimated_diameter_min": float,  # Diámetro mínimo en km
    "estimated_diameter_max": float,  # Diámetro máximo en km
    # Otros campos del dataset que puedan ser relevantes
}
```

### Normalized Model Input

```javascript
{
    "magnitude": float  // Valor normalizado entre 0 y 1
}
```

### Prediction Result

```javascript
{
    "size_km": float,           // Tamaño predicho en kilómetros
    "confidence": float,        // Nivel de confianza (0-1)
    "input_magnitude": float,   // Valor de entrada original
    "timestamp": Date           // Momento de la predicción
}
```

### History Entry

```javascript
{
    "id": string,
    "magnitude": float,
    "predicted_size": float,
    "timestamp": Date
}
```

### Normalization Parameters

```json
{
    "magnitude_min": float,
    "magnitude_max": float,
    "size_min": float,
    "size_max": float,
    "mean": float,
    "std": float
}
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*


### Property 1: Input validation range checking
*For any* numeric input value, the validation function should correctly identify whether it falls within the valid range of absolute magnitude values (typically 0-35 for asteroids)
**Validates: Requirements 1.1**

### Property 2: Valid input produces prediction
*For any* valid absolute magnitude value, the ML model should return a numeric prediction representing the asteroid size in kilometers
**Validates: Requirements 1.2**

### Property 3: Invalid input error handling
*For any* invalid or out-of-range input value, the system should display an error message and the application state should remain unchanged from before the invalid input
**Validates: Requirements 1.3**

### Property 4: Prediction output format
*For any* prediction result, the displayed output should include the unit "km" or "kilómetros" to clearly indicate the measurement unit
**Validates: Requirements 1.4**

### Property 5: Data split completeness
*For any* dataset split into training and validation sets, the sum of the sizes of both sets should equal the original dataset size, and there should be no overlap between the sets
**Validates: Requirements 2.3**

### Property 6: Model evaluation metrics validity
*For any* trained model evaluation, the calculated metrics (MAE, RMSE, R²) should be valid numbers (not NaN or Infinity) and within reasonable ranges for the problem domain
**Validates: Requirements 2.5**

### Property 7: Loading indicator presence
*For any* prediction request initiated by the user, a loading indicator should be visible in the DOM while the prediction is being processed
**Validates: Requirements 3.3**

### Property 8: Prediction result completeness
*For any* generated prediction, the displayed result should include: the predicted size value, contextual information, a visual comparison with known objects, and a confidence level or error margin
**Validates: Requirements 4.1, 4.2, 4.3**

### Property 9: Client-side execution
*For any* prediction operation after initial model load, the system should not make any HTTP requests to external servers (all computation happens in the browser)
**Validates: Requirements 5.1**

### Property 10: Prediction error resilience
*For any* error that occurs during prediction generation, the system should catch the error, display an appropriate message to the user, and maintain the UI in a functional state
**Validates: Requirements 6.2, 6.4**

### Property 11: Continuous input availability
*For any* completed prediction, the input field should remain enabled and ready to accept a new value without requiring a page reload
**Validates: Requirements 7.1**

### Property 12: History accumulation
*For any* sequence of N predictions, all N predictions should be stored in the history (subject to the maximum limit)
**Validates: Requirements 7.2**

### Property 13: History size limit
*For any* history state with more than 5 predictions, only the 5 most recent predictions should be displayed, each showing both input magnitude and predicted size
**Validates: Requirements 7.3**

## Error Handling

### Input Validation Errors
- **Invalid numeric format**: Display "Por favor ingresa un número válido"
- **Out of range**: Display "El valor debe estar entre X y Y" (ranges determined from training data)
- **Empty input**: Display "Por favor ingresa un valor de brillo absoluto"

### Model Loading Errors
- **Network failure**: Display "Error al cargar el modelo. Por favor verifica tu conexión."
- **Corrupted model**: Display "Error al inicializar el modelo. Por favor recarga la página."
- **Browser incompatibility**: Display "Tu navegador no soporta las características necesarias. Por favor usa Chrome, Firefox o Edge actualizado."

### Prediction Errors
- **Model inference failure**: Display "Error al generar la predicción. Por favor intenta con otro valor."
- **Unexpected output**: Log error to console, display generic error message to user

### Error Recovery Strategy
1. Catch all errors at the highest appropriate level
2. Log detailed error information to console for debugging
3. Display user-friendly messages in Spanish
4. Maintain application state - never leave UI in broken state
5. Provide actionable guidance when possible

## Testing Strategy

### Unit Testing

**Framework**: Jest para JavaScript, pytest para Python

**Unit Test Coverage:**
- Input validation functions (valid/invalid ranges, format checking)
- Normalization/denormalization functions
- History management (add, limit, clear)
- Error message generation
- Data preprocessing functions (Python training pipeline)
- Model export functions

**Example Unit Tests:**
- Test that validation rejects values < 0 and > 35
- Test that validation accepts values within range
- Test that history never exceeds 5 items
- Test that normalization parameters are saved correctly
- Test that dataset loading handles missing columns

### Property-Based Testing

**Framework**: fast-check para JavaScript, Hypothesis para Python

**Configuration**: Each property test should run a minimum of 100 iterations to ensure thorough coverage of the input space.

**Property Test Requirements:**
- Each property-based test MUST be tagged with a comment explicitly referencing the correctness property from this design document
- Tag format: `// Feature: asteroid-size-predictor, Property {number}: {property_text}`
- Each correctness property MUST be implemented by a SINGLE property-based test

**Property Test Coverage:**
- Property 1: Generate random numeric values (valid and invalid ranges) and verify validation behavior
- Property 2: Generate random valid magnitudes and verify predictions are returned
- Property 3: Generate random invalid inputs and verify error handling
- Property 4: Generate random predictions and verify output format includes units
- Property 5: Generate random dataset splits and verify completeness
- Property 6: Generate random model evaluations and verify metric validity
- Property 9: Monitor network activity during random prediction sequences
- Property 10: Inject random errors and verify UI remains functional
- Property 11: Generate random prediction sequences and verify input availability
- Property 12: Generate random prediction sequences and verify history accumulation
- Property 13: Generate sequences of >5 predictions and verify only 5 are shown

### Integration Testing

**Scope:**
- End-to-end flow: input → validation → prediction → display
- Model loading and initialization
- History persistence across predictions
- Error scenarios with full UI interaction

**Test Scenarios:**
- User enters valid magnitude → sees prediction with all required information
- User enters invalid magnitude → sees error, can retry
- User makes multiple predictions → history updates correctly
- Model fails to load → user sees appropriate error message

### Manual Testing Checklist

- [ ] Test on Chrome, Firefox, Safari, Edge
- [ ] Test on mobile devices (iOS, Android)
- [ ] Test with slow network (model loading)
- [ ] Test with network disconnected after load
- [ ] Verify responsive design at different screen sizes
- [ ] Verify accessibility (keyboard navigation, screen readers)
- [ ] Test with various magnitude values from dataset
- [ ] Verify visual comparisons make sense

## Performance Considerations

### Model Size
- Target model size: < 1MB for fast loading
- Use model quantization if necessary to reduce size
- Consider using a simpler model architecture (linear regression or small neural network)

### Loading Strategy
- Load model asynchronously on page load
- Show loading indicator during model initialization
- Cache model in browser if possible (service worker)

### Prediction Speed
- Target prediction time: < 100ms
- TensorFlow.js should handle this easily for simple models
- No optimization needed unless model is complex

### Browser Compatibility
- Minimum browser versions:
  - Chrome 57+
  - Firefox 52+
  - Safari 11+
  - Edge 79+
- Feature detection for WebGL (required by TensorFlow.js)

## Deployment Strategy

### GitHub Pages Setup
1. Create repository: `username/asteroid-size-predictor`
2. Enable GitHub Pages from main branch
3. Place all static files in root or `/docs` folder
4. Model files in `/model` directory
5. Access via: `https://username.github.io/asteroid-size-predictor`

### File Structure
```
/
├── index.html              # Main page
├── styles.css              # Styling
├── app.js                  # Main application controller
├── ml-engine.js            # ML inference engine
├── visualization.js        # Charts and visualizations
├── model/
│   ├── model.json          # TensorFlow.js model
│   ├── group1-shard1of1.bin # Model weights
│   └── normalization.json  # Normalization parameters
├── training/
│   ├── train_model.py      # Training script
│   ├── requirements.txt    # Python dependencies
│   └── README.md           # Training instructions
└── README.md               # Project documentation
```

### Continuous Integration
- GitHub Actions workflow to validate HTML/CSS/JS
- Automated testing on push
- Optional: Retrain model periodically with updated data

## Security Considerations

### Input Sanitization
- Validate all user inputs before processing
- Prevent XSS through proper escaping
- No eval() or similar dangerous functions

### Data Privacy
- No user data is collected or transmitted
- All computation happens locally
- No cookies or tracking

### Content Security Policy
- Restrict script sources
- Prevent inline scripts where possible
- Use nonce or hash for inline scripts if needed

## Accessibility

### WCAG 2.1 Compliance
- Semantic HTML structure
- Proper ARIA labels for interactive elements
- Keyboard navigation support
- Sufficient color contrast (4.5:1 minimum)
- Focus indicators on all interactive elements
- Screen reader friendly error messages

### Internationalization
- All text in Spanish (primary language)
- Use proper lang attribute in HTML
- Consider adding English translation in future

## Future Enhancements

### Potential Features
- Multiple input features (not just magnitude)
- Confidence intervals visualization
- Comparison with real asteroid database
- Export predictions as CSV
- Dark mode
- Multiple language support
- Progressive Web App (PWA) capabilities
- Offline support with service workers

### Model Improvements
- Ensemble models for better accuracy
- Uncertainty quantification
- Feature importance visualization
- Model versioning and A/B testing
