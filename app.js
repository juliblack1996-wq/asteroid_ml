// Variables globales
let mlEngine = null;
let history = [];

// Debug: Verificar que el script se cargó
console.log('✅ app.js cargado correctamente');

// Inicializar la aplicación
async function initApp() {
    try {
        updateModelStatus('loading', 'Cargando modelo...');
        mlEngine = new MLEngine();
        await mlEngine.loadModel();
        updateModelStatus('ready', 'Modelo listo');
        console.log('Aplicación inicializada correctamente');
        console.log('Información del modelo:', mlEngine.getModelInfo());
    } catch (error) {
        console.error('Error al inicializar:', error);
        updateModelStatus('error', 'Error al cargar modelo');
    }
}

// Actualizar estado del modelo
function updateModelStatus(status, message) {
    const statusElement = document.getElementById('modelStatus');
    const indicator = statusElement.querySelector('.status-indicator');
    const text = statusElement.querySelector('span');
    
    indicator.className = 'status-indicator ' + status;
    text.textContent = message;
}

// Establecer valor de magnitud desde ejemplos
function setMagnitude(value) {
    document.getElementById('magnitude').value = value;
    document.getElementById('magnitude').focus();
}

// Realizar predicción
async function makePrediction() {
    const magnitudeInput = document.getElementById('magnitude');
    const magnitude = parseFloat(magnitudeInput.value);
    
    // Validar entrada
    if (isNaN(magnitude)) {
        alert('Por favor ingresa un valor válido para la magnitud');
        return;
    }
    
    if (magnitude < 10 || magnitude > 21) {
        alert('La magnitud debe estar entre 10 y 21');
        return;
    }
    
    // Verificar que el modelo esté cargado
    if (!mlEngine || !mlEngine.isModelReady()) {
        alert('El modelo aún no está cargado. Por favor espera un momento.');
        return;
    }
    
    // Mostrar estado de carga
    const btn = document.getElementById('predictBtn');
    btn.classList.add('loading');
    btn.disabled = true;
    
    try {
        // Realizar predicción
        const result = await mlEngine.predict(magnitude);
        const diameter = result.size_km;
        
        // Mostrar resultados
        displayResults(magnitude, diameter);
        
        // Actualizar visualización 3D
        updateAsteroidVisualization(diameter);
        
        // Agregar al historial
        addToHistory(magnitude, diameter);
        
    } catch (error) {
        console.error('Error en predicción:', error);
        alert('Error al realizar la predicción: ' + error.message);
    } finally {
        btn.classList.remove('loading');
        btn.disabled = false;
    }
}

// Mostrar resultados
function displayResults(magnitude, diameter) {
    const resultsPanel = document.getElementById('resultsPanel');
    const radius = diameter / 2;
    
    // Mostrar panel de resultados
    resultsPanel.style.display = 'block';
    resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    
    // Actualizar valores básicos
    document.getElementById('resultMagnitude').textContent = magnitude.toFixed(2) + ' H';
    document.getElementById('resultSize').textContent = diameter.toFixed(2) + ' km';
    document.getElementById('resultRadius').textContent = radius.toFixed(2) + ' km';
    
    // Evaluar riesgo
    displayRiskAssessment(diameter);
    
    // Mostrar comparación
    displayComparison(diameter);
}

// Evaluar y mostrar riesgo
function displayRiskAssessment(diameter) {
    const riskCard = document.getElementById('riskCard');
    const riskLevel = document.getElementById('riskLevel');
    const riskTitle = document.getElementById('riskTitle');
    const riskDescription = document.getElementById('riskDescription');
    const riskDetails = document.getElementById('riskDetails');
    
    let risk, color, icon, title, description;
    
    if (diameter < 0.025) {
        risk = 'low';
        color = 'var(--success)';
        icon = '✅';
        title = 'Riesgo Muy Bajo';
        description = 'Asteroide muy pequeño. Se desintegraría completamente en la atmósfera.';
    } else if (diameter < 0.14) {
        risk = 'low';
        color = 'var(--success)';
        icon = '🟢';
        title = 'Riesgo Bajo';
        description = 'Asteroide pequeño. Causaría daño local limitado en caso de impacto.';
    } else if (diameter < 1) {
        risk = 'medium';
        color = 'var(--warning)';
        icon = '🟡';
        title = 'Riesgo Moderado';
        description = 'Asteroide mediano. Podría causar destrucción regional significativa.';
    } else if (diameter < 10) {
        risk = 'high';
        color = '#ff6b35';
        icon = '🟠';
        title = 'Riesgo Alto';
        description = 'Asteroide grande. Causaría devastación a escala continental.';
    } else {
        risk = 'critical';
        color = 'var(--danger)';
        icon = '🔴';
        title = 'Riesgo Crítico';
        description = 'Asteroide masivo. Evento de extinción global potencial.';
    }
    
    // Actualizar estilos
    riskCard.style.borderLeftColor = color;
    const indicator = riskLevel.querySelector('.risk-indicator');
    indicator.style.background = color;
    indicator.textContent = icon;
    
    riskTitle.textContent = title;
    riskDescription.textContent = description;
    
    // Calcular detalles adicionales
    const energy = calculateImpactEnergy(diameter);
    const craterSize = diameter * 20; // Aproximación simple
    const tsunamiRisk = diameter > 0.3 ? 'Sí (si impacta en océano)' : 'No';
    
    riskDetails.innerHTML = `
        <div class="risk-detail-item">
            <h5>Energía de Impacto</h5>
            <p>${energy.toFixed(1)} Megatones TNT</p>
        </div>
        <div class="risk-detail-item">
            <h5>Tamaño de Cráter</h5>
            <p>~${craterSize.toFixed(1)} km</p>
        </div>
        <div class="risk-detail-item">
            <h5>Riesgo de Tsunami</h5>
            <p>${tsunamiRisk}</p>
        </div>
        <div class="risk-detail-item">
            <h5>Clasificación</h5>
            <p>${getAsteroidClass(diameter)}</p>
        </div>
    `;
}

// Calcular energía de impacto (aproximación)
function calculateImpactEnergy(diameter) {
    // Fórmula simplificada: E ≈ 0.5 * m * v²
    // Asumiendo densidad de 2.6 g/cm³ y velocidad de 20 km/s
    const radius = diameter / 2;
    const volume = (4/3) * Math.PI * Math.pow(radius * 1000, 3); // en m³
    const mass = volume * 2600; // kg (densidad 2.6 g/cm³)
    const velocity = 20000; // m/s
    const energy = 0.5 * mass * velocity * velocity; // Joules
    const megatons = energy / 4.184e15; // Convertir a megatones TNT
    return megatons;
}

// Clasificar asteroide por tamaño
function getAsteroidClass(diameter) {
    if (diameter < 0.001) return 'Meteoroide';
    if (diameter < 0.025) return 'Bólido pequeño';
    if (diameter < 0.14) return 'Asteroide pequeño';
    if (diameter < 1) return 'Asteroide mediano';
    if (diameter < 10) return 'Asteroide grande';
    return 'Asteroide masivo';
}

// Mostrar comparación de tamaño
function displayComparison(diameter) {
    const comparison = document.getElementById('comparison');
    
    const comparisons = [
        { name: 'Persona', size: 0.0017, icon: '🧍', unit: '1.7m' },
        { name: 'Casa', size: 0.01, icon: '🏠', unit: '10m' },
        { name: 'Estadio', size: 0.1, icon: '🏟️', unit: '100m' },
        { name: 'Ciudad', size: 10, icon: '🏙️', unit: '10km' },
        { name: 'Asteroide', size: diameter, icon: '☄️', unit: diameter.toFixed(2) + 'km' }
    ];
    
    // Encontrar el tamaño máximo para escalar
    const maxSize = Math.max(...comparisons.map(c => c.size));
    const maxHeight = 200; // píxeles
    
    comparison.innerHTML = comparisons.map(item => {
        const height = (item.size / maxSize) * maxHeight;
        const isAsteroid = item.name === 'Asteroide';
        
        return `
            <div class="comparison-item ${isAsteroid ? 'highlight' : ''}">
                <div class="comparison-icon">${item.icon}</div>
                <div class="comparison-bar" style="height: ${height}px; ${isAsteroid ? 'box-shadow: 0 0 20px rgba(99, 102, 241, 0.5);' : ''}"></div>
                <div class="comparison-label">${item.name}</div>
                <div class="comparison-value">${item.unit}</div>
            </div>
        `;
    }).join('');
}

// Actualizar visualización 3D del asteroide
function updateAsteroidVisualization(diameter) {
    const viewer = document.getElementById('asteroidViewer');
    
    // Calcular tamaño visual (escala logarítmica para mejor visualización)
    const minSize = 50;
    const maxSize = 400;
    const logDiameter = Math.log10(diameter + 1);
    const logMax = Math.log10(100);
    const size = minSize + (logDiameter / logMax) * (maxSize - minSize);
    
    viewer.innerHTML = `
        <div class="asteroid-3d">
            <div class="asteroid-sphere" style="width: ${size}px; height: ${size}px;"></div>
        </div>
        <div class="asteroid-info">
            <h3>${diameter.toFixed(2)} km</h3>
            <p>Diámetro estimado</p>
        </div>
    `;
}

// Agregar al historial
function addToHistory(magnitude, diameter) {
    const timestamp = new Date();
    history.unshift({
        magnitude,
        diameter,
        timestamp
    });
    
    // Limitar historial a 10 elementos
    if (history.length > 10) {
        history = history.slice(0, 10);
    }
    
    displayHistory();
}

// Mostrar historial
function displayHistory() {
    const historyPanel = document.getElementById('historyPanel');
    const historyList = document.getElementById('historyList');
    
    if (history.length === 0) {
        historyPanel.style.display = 'none';
        return;
    }
    
    historyPanel.style.display = 'block';
    
    historyList.innerHTML = history.map((item, index) => `
        <div class="history-item" onclick="loadFromHistory(${index})">
            <div class="history-item-data">
                <span>
                    <small>Magnitud</small>
                    <strong>${item.magnitude.toFixed(2)} H</strong>
                </span>
                <span>
                    <small>Diámetro</small>
                    <strong>${item.diameter.toFixed(2)} km</strong>
                </span>
                <span>
                    <small>Fecha</small>
                    <strong>${item.timestamp.toLocaleString('es-ES', { 
                        hour: '2-digit', 
                        minute: '2-digit',
                        day: '2-digit',
                        month: '2-digit'
                    })}</strong>
                </span>
            </div>
            <div>🔄</div>
        </div>
    `).join('');
}

// Cargar desde historial
function loadFromHistory(index) {
    const item = history[index];
    document.getElementById('magnitude').value = item.magnitude;
    makePrediction();
}

// Inicializar cuando se carga la página
window.addEventListener('DOMContentLoaded', initApp);

// Permitir predicción con Enter
document.addEventListener('DOMContentLoaded', () => {
    const magnitudeInput = document.getElementById('magnitude');
    if (magnitudeInput) {
        magnitudeInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                makePrediction();
            }
        });
    }
});
