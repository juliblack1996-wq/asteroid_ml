# Requirements Document

## Introduction

Este documento describe los requisitos para un sistema de predicción del tamaño de asteroides basado en su brillo absoluto. El sistema será una aplicación web estática alojada en GitHub Pages que utiliza un modelo de Machine Learning entrenado con datos de asteroides para proporcionar predicciones en tiempo real a través de una interfaz de usuario interactiva.

## Glossary

- **Sistema**: La aplicación web de predicción de tamaño de asteroides
- **Usuario**: Cualquier persona que accede a la página web para obtener predicciones
- **Modelo ML**: El modelo de Machine Learning entrenado con el dataset de asteroides de Kaggle
- **Brillo Absoluto**: Magnitud absoluta (H) de un asteroide, medida de su brillo intrínseco
- **Predicción**: El tamaño estimado del asteroide en kilómetros basado en el brillo absoluto
- **GitHub Pages**: Servicio de hosting estático gratuito de GitHub
- **Dataset**: Conjunto de datos de asteroides de Kaggle (sakhawat18/asteroid-dataset)

## Requirements

### Requirement 1

**User Story:** Como usuario, quiero ingresar el brillo absoluto de un asteroide, para que el sistema me proporcione una predicción de su tamaño

#### Acceptance Criteria

1. WHEN el Usuario ingresa un valor numérico de brillo absoluto, THE Sistema SHALL validar que el valor esté dentro de un rango razonable
2. WHEN el Usuario envía un valor válido de brillo absoluto, THE Sistema SHALL ejecutar el Modelo ML y retornar una Predicción del tamaño en kilómetros
3. WHEN el Usuario ingresa un valor inválido o fuera de rango, THE Sistema SHALL mostrar un mensaje de error descriptivo y mantener el estado actual
4. WHEN el Sistema genera una Predicción, THE Sistema SHALL mostrar el resultado de forma clara con unidades apropiadas

### Requirement 2

**User Story:** Como desarrollador, quiero entrenar un modelo de Machine Learning con el dataset de Kaggle, para que pueda predecir tamaños de asteroides con precisión

#### Acceptance Criteria

1. WHEN el sistema de entrenamiento descarga el Dataset, THE sistema de entrenamiento SHALL cargar los datos de asteroides correctamente
2. WHEN el sistema de entrenamiento procesa el Dataset, THE sistema de entrenamiento SHALL extraer las características relevantes incluyendo brillo absoluto y tamaño
3. WHEN el sistema de entrenamiento prepara los datos, THE sistema de entrenamiento SHALL dividir el Dataset en conjuntos de entrenamiento y validación
4. WHEN el sistema de entrenamiento completa el entrenamiento, THE sistema de entrenamiento SHALL exportar el Modelo ML en un formato compatible con navegadores web
5. WHEN el sistema de entrenamiento evalúa el Modelo ML, THE sistema de entrenamiento SHALL calcular métricas de precisión y error

### Requirement 3

**User Story:** Como usuario, quiero una interfaz visual atractiva y fácil de usar, para que pueda interactuar con el sistema de predicción de forma intuitiva

#### Acceptance Criteria

1. WHEN el Usuario accede a la página, THE Sistema SHALL mostrar una interfaz limpia con un campo de entrada claramente etiquetado
2. WHEN el Usuario interactúa con los controles, THE Sistema SHALL proporcionar retroalimentación visual inmediata
3. WHEN el Sistema procesa una solicitud, THE Sistema SHALL mostrar un indicador de carga mientras se genera la Predicción
4. WHEN el Sistema muestra resultados, THE Sistema SHALL presentar la información de forma visualmente atractiva con gráficos o visualizaciones relevantes
5. WHILE el Usuario navega por la interfaz, THE Sistema SHALL mantener un diseño responsivo que funcione en dispositivos móviles y de escritorio

### Requirement 4

**User Story:** Como usuario, quiero ver información adicional sobre el asteroide predicho, para que pueda entender mejor el contexto de la predicción

#### Acceptance Criteria

1. WHEN el Sistema genera una Predicción, THE Sistema SHALL mostrar información contextual sobre el tamaño predicho
2. WHEN el Sistema muestra resultados, THE Sistema SHALL incluir una comparación visual del tamaño con objetos conocidos
3. WHEN el Usuario visualiza una Predicción, THE Sistema SHALL mostrar el nivel de confianza o margen de error de la predicción

### Requirement 5

**User Story:** Como desarrollador, quiero que la aplicación sea completamente estática y funcione sin servidor, para que pueda ser alojada gratuitamente en GitHub Pages

#### Acceptance Criteria

1. THE Sistema SHALL ejecutar todas las predicciones del Modelo ML en el navegador del Usuario sin requerir llamadas a servidor
2. THE Sistema SHALL cargar el Modelo ML como un archivo estático desde GitHub Pages
3. WHEN el Usuario accede a la aplicación, THE Sistema SHALL funcionar completamente sin dependencias de backend
4. THE Sistema SHALL utilizar únicamente tecnologías web estáticas compatibles con GitHub Pages

### Requirement 6

**User Story:** Como usuario, quiero que el sistema maneje errores de forma elegante, para que tenga una experiencia fluida incluso cuando algo falla

#### Acceptance Criteria

1. WHEN ocurre un error al cargar el Modelo ML, THE Sistema SHALL mostrar un mensaje de error informativo al Usuario
2. WHEN el Modelo ML no puede generar una Predicción, THE Sistema SHALL informar al Usuario del problema sin romper la interfaz
3. WHEN el navegador del Usuario no soporta las características necesarias, THE Sistema SHALL mostrar un mensaje de compatibilidad
4. IF el Usuario ingresa datos que causan un error en el Modelo ML, THEN THE Sistema SHALL capturar el error y mostrar orientación al Usuario

### Requirement 7

**User Story:** Como usuario, quiero poder probar múltiples valores de brillo, para que pueda explorar diferentes escenarios de predicción

#### Acceptance Criteria

1. WHEN el Usuario completa una Predicción, THE Sistema SHALL permitir ingresar un nuevo valor sin recargar la página
2. WHEN el Usuario genera múltiples predicciones, THE Sistema SHALL mantener un historial visual de las predicciones recientes
3. WHEN el Usuario visualiza el historial, THE Sistema SHALL mostrar hasta las últimas 5 predicciones con sus valores de entrada y salida

### Requirement 8

**User Story:** Como desarrollador, quiero documentación clara del modelo y su uso, para que otros puedan entender y contribuir al proyecto

#### Acceptance Criteria

1. THE Sistema SHALL incluir un archivo README con instrucciones de uso y despliegue
2. THE Sistema SHALL documentar el proceso de entrenamiento del Modelo ML con scripts reproducibles
3. THE Sistema SHALL incluir información sobre las fuentes de datos y referencias científicas
4. THE Sistema SHALL proporcionar ejemplos de valores de brillo absoluto típicos para diferentes tipos de asteroides
