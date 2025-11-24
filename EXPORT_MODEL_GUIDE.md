# Guía para Exportar y Usar el Modelo en Otra Página Web

## Archivos Necesarios

Para usar el modelo de predicción de asteroides en cualquier otra página web, necesitas copiar estos 3 archivos:

### 1. Archivos del Modelo (carpeta `model/`)
```
model/
├── model_weights.json      # Pesos de la red neuronal (2,753 parámetros)
└── normalization.json      # Parámetros de normalización
```

### 2. Motor de ML (JavaScript puro)
```
ml-engine.js               # Implementación de la red neuronal (sin dependencias)
```

## Pasos para Integrar en Otra Página Web

### Opción 1: Integración Completa (Recomendada)

1. **Copia los archivos a tu proyecto:**
   ```
   tu-proyecto/
   ├── model/
   │   ├── model_weights.json
   │   └── normalization.json
   ├── ml-engine.js
   └── tu-pagina.html
   ```

2. **Incluye el script en tu HTML:**
   ```html
   <!DOCTYPE html>
   <html>
   <head>
       <title>Mi Predictor de Asteroides</title>
   </head>
   <body>
       <input type="number" id="magnitude" placeholder="Magnitud (H)">
       <button onclick="predecir()">Predecir</button>
       <div id="resultado"></div>

       <!-- Incluir el motor ML -->
       <script src="ml-engine.js"></script>
       
       <!-- Tu código -->
       <script>
           let mlEngine = null;

           // Inicializar al cargar la página
           async function init() {
               mlEngine = new MLEngine();
               await mlEngine.loadModel();
               console.log('Modelo cargado:', mlEngine.getModelInfo());
           }

           // Hacer predicción
           async function predecir() {
               const magnitude = parseFloat(document.getElementById('magnitude').value);
               const result = await mlEngine.predict(magnitude);
               document.getElementById('resultado').innerHTML = 
                   `Diámetro estimado: ${result.size_km.toFixed(2)} km`;
           }

           // Inicializar
           init();
       </script>
   </body>
   </html>
   ```

### Opción 2: Integración Mínima (Solo Predicción)

Si solo necesitas la funcionalidad de predicción sin la interfaz completa:

```javascript
// 1. Cargar el modelo
const mlEngine = new MLEngine();
await mlEngine.loadModel();

// 2. Hacer predicción
const result = await mlEngine.predict(15.5);
console.log(`Diámetro: ${result.size_km} km`);
```

## API del Motor ML

### Crear instancia
```javascript
const mlEngine = new MLEngine();
```

### Cargar modelo
```javascript
await mlEngine.loadModel();
// Carga automáticamente desde model/model_weights.json y model/normalization.json
```

### Hacer predicción
```javascript
const result = await mlEngine.predict(magnitude);
// Retorna: { size_km: number, magnitude: number, timestamp: Date }
```

### Verificar si está listo
```javascript
if (mlEngine.isModelReady()) {
    // El modelo está listo para predicciones
}
```

### Obtener información del modelo
```javascript
const info = mlEngine.getModelInfo();
// Retorna:
// {
//   layers: 6,
//   parameters: 2753,
//   magnitudeRange: [10.4, 20.7],
//   sizeRange: [0.148, 14.707]
// }
```

## Ejemplo Completo con Manejo de Errores

```javascript
async function setupPredictor() {
    try {
        // Crear y cargar modelo
        const mlEngine = new MLEngine();
        await mlEngine.loadModel();
        
        console.log('✅ Modelo cargado exitosamente');
        console.log('Información:', mlEngine.getModelInfo());
        
        // Hacer predicción
        const magnitude = 15.0;
        const result = await mlEngine.predict(magnitude);
        
        console.log(`Magnitud: ${result.magnitude} H`);
        console.log(`Diámetro: ${result.size_km.toFixed(2)} km`);
        console.log(`Radio: ${(result.size_km / 2).toFixed(2)} km`);
        
    } catch (error) {
        console.error('❌ Error:', error.message);
    }
}

setupPredictor();
```

## Características del Motor ML

✅ **Sin dependencias externas** - No requiere TensorFlow.js ni otras librerías
✅ **Ligero** - Solo ~6 KB de código JavaScript
✅ **Rápido** - Predicciones instantáneas en el navegador
✅ **Compatible** - Funciona en todos los navegadores modernos
✅ **Portable** - Fácil de integrar en cualquier proyecto web

## Validación de Entrada

El modelo acepta magnitudes absolutas (H) en el rango:
- **Mínimo:** 10.4 H (asteroides grandes ~15 km)
- **Máximo:** 20.7 H (asteroides pequeños ~0.15 km)

Si proporcionas valores fuera de este rango, el modelo generará una advertencia pero seguirá haciendo la predicción.

## Estructura de los Archivos JSON

### model_weights.json
```json
{
  "architecture": [...],  // Configuración de capas
  "weights": [            // Pesos de cada capa
    {
      "layer_index": 0,
      "kernel": [[...]],  // Matriz de pesos
      "bias": [...]       // Vector de bias
    },
    ...
  ]
}
```

### normalization.json
```json
{
  "magnitude_mean": 16.57,
  "magnitude_std": 1.89,
  "magnitude_min": 10.4,
  "magnitude_max": 20.7,
  "size_mean": 1.45,
  "size_std": 1.29,
  "size_min": 0.148,
  "size_max": 14.707
}
```

## Hosting y Despliegue

Puedes hospedar estos archivos en:
- **GitHub Pages** (gratis)
- **Netlify** (gratis)
- **Vercel** (gratis)
- **Tu propio servidor web**

Solo asegúrate de que los archivos JSON estén en la carpeta `model/` relativa al HTML.

## Ejemplo de Integración en React

```jsx
import { useEffect, useState } from 'react';

function AsteroidPredictor() {
    const [mlEngine, setMlEngine] = useState(null);
    const [magnitude, setMagnitude] = useState(15);
    const [result, setResult] = useState(null);

    useEffect(() => {
        // Cargar modelo al montar componente
        const loadModel = async () => {
            const engine = new MLEngine();
            await engine.loadModel();
            setMlEngine(engine);
        };
        loadModel();
    }, []);

    const predict = async () => {
        if (mlEngine) {
            const prediction = await mlEngine.predict(magnitude);
            setResult(prediction);
        }
    };

    return (
        <div>
            <input 
                type="number" 
                value={magnitude}
                onChange={(e) => setMagnitude(parseFloat(e.target.value))}
            />
            <button onClick={predict}>Predecir</button>
            {result && <p>Diámetro: {result.size_km.toFixed(2)} km</p>}
        </div>
    );
}
```

## Soporte

El modelo fue entrenado con 127,203 asteroides del dataset de Kaggle y tiene:
- **MAE:** 1.29 km
- **R²:** 0.56

Para más información, consulta los archivos de entrenamiento en la carpeta `training/`.
