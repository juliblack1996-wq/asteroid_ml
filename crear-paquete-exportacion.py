#!/usr/bin/env python3
"""
Script para crear un paquete exportable del modelo de predicción de asteroides.
Este paquete puede ser usado en cualquier página web.
"""

import os
import shutil
import json
from datetime import datetime

def crear_paquete():
    """Crea un paquete con todos los archivos necesarios para usar el modelo."""
    
    # Nombre del paquete
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_name = f"asteroid-predictor-package_{timestamp}"
    
    print(f"📦 Creando paquete: {package_name}")
    print("-" * 50)
    
    # Crear directorio del paquete
    if os.path.exists(package_name):
        shutil.rmtree(package_name)
    os.makedirs(package_name)
    
    # Crear subdirectorio model
    os.makedirs(os.path.join(package_name, "model"))
    
    # Lista de archivos a copiar
    archivos = [
        ("ml-engine.js", "ml-engine.js"),
        ("model/model_weights.json", "model/model_weights.json"),
        ("model/normalization.json", "model/normalization.json"),
        ("ejemplo-minimo.html", "ejemplo-minimo.html"),
        ("EXPORT_MODEL_GUIDE.md", "README.md"),
    ]
    
    # Copiar archivos
    archivos_copiados = []
    for origen, destino in archivos:
        if os.path.exists(origen):
            destino_path = os.path.join(package_name, destino)
            shutil.copy2(origen, destino_path)
            size = os.path.getsize(destino_path)
            archivos_copiados.append((destino, size))
            print(f"✅ Copiado: {destino} ({size:,} bytes)")
        else:
            print(f"⚠️  No encontrado: {origen}")
    
    # Crear archivo de información
    info = {
        "package_name": package_name,
        "created_at": datetime.now().isoformat(),
        "version": "1.0.0",
        "model_info": {
            "training_samples": 127203,
            "mae_km": 1.29,
            "r2_score": 0.56,
            "magnitude_range": [10.4, 20.7],
            "size_range_km": [0.148, 14.707]
        },
        "files": [
            {
                "name": name,
                "size_bytes": size,
                "description": get_file_description(name)
            }
            for name, size in archivos_copiados
        ]
    }
    
    info_path = os.path.join(package_name, "package-info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"✅ Creado: package-info.json")
    
    # Crear archivo ZIP
    print("\n📦 Creando archivo ZIP...")
    shutil.make_archive(package_name, 'zip', package_name)
    zip_size = os.path.getsize(f"{package_name}.zip")
    print(f"✅ Creado: {package_name}.zip ({zip_size:,} bytes)")
    
    # Resumen
    print("\n" + "=" * 50)
    print("✅ PAQUETE CREADO EXITOSAMENTE")
    print("=" * 50)
    print(f"\n📁 Carpeta: {package_name}/")
    print(f"📦 Archivo ZIP: {package_name}.zip")
    print(f"\n📊 Tamaño total: {zip_size:,} bytes ({zip_size/1024:.1f} KB)")
    print(f"📄 Archivos incluidos: {len(archivos_copiados)}")
    
    print("\n📋 CONTENIDO DEL PAQUETE:")
    for name, size in archivos_copiados:
        print(f"  • {name} ({size:,} bytes)")
    
    print("\n🚀 CÓMO USAR:")
    print("  1. Descomprime el archivo ZIP")
    print("  2. Abre ejemplo-minimo.html en un navegador")
    print("  3. O integra los archivos en tu proyecto web")
    print("  4. Lee README.md para más detalles")
    
    print("\n💡 ARCHIVOS NECESARIOS PARA INTEGRACIÓN:")
    print("  • ml-engine.js (motor de predicción)")
    print("  • model/model_weights.json (pesos del modelo)")
    print("  • model/normalization.json (parámetros de normalización)")
    
    return package_name

def get_file_description(filename):
    """Retorna una descripción del archivo."""
    descriptions = {
        "ml-engine.js": "Motor de predicción ML en JavaScript puro (sin dependencias)",
        "model/model_weights.json": "Pesos de la red neuronal (2,753 parámetros)",
        "model/normalization.json": "Parámetros de normalización de entrada/salida",
        "ejemplo-minimo.html": "Ejemplo mínimo de integración",
        "README.md": "Guía completa de uso e integración"
    }
    return descriptions.get(filename, "Archivo del paquete")

if __name__ == "__main__":
    try:
        package_name = crear_paquete()
        print(f"\n✨ Listo para compartir: {package_name}.zip")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
