import os
import json
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from joblib import Parallel, delayed

# =========================
# CONFIG - RUTAS
# =========================
# Carpeta donde están tus archivos normalizados (los 3 tipos de nombres)
INPUT_DIR = "/media/mrsmile/IA/tesis/data/processed/normalized"

# Rutas de salida para los binarios
OUTPUT_DIR = "/media/mrsmile/IA/tesis/data/processed/memmap"
VOLUME_OUT = os.path.join(OUTPUT_DIR, "volumes")
META_DIR = os.path.join(OUTPUT_DIR, "meta")

# Crear carpetas si no existen
os.makedirs(VOLUME_OUT, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

def process_to_memmap(fname):
    """
    Lee un archivo .nii.gz y lo convierte en un binario .dat (float32)
    """
    try:
        # El nombre base será el mismo del archivo (ej: asoca_normal_20)
        base = fname.replace('.nii.gz', '').replace('.nii', '')
        
        in_path = os.path.join(INPUT_DIR, fname)
        mmap_path = os.path.join(VOLUME_OUT, base + ".dat")
        meta_path = os.path.join(META_DIR, base + ".json")

        # Saltar si ya existe el binario para ahorrar tiempo
        if os.path.exists(mmap_path) and os.path.exists(meta_path):
            return True

        # 1. Cargar volumen médico
        img = sitk.ReadImage(in_path)
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        
        # 2. Crear y llenar el Memmap
        # w+ crea el archivo o lo sobreescribe
        shape = arr.shape
        mmap = np.memmap(
            mmap_path, 
            dtype="float32", 
            mode="w+", 
            shape=shape
        )
        mmap[:] = arr[:]
        mmap.flush() # Asegura que los datos se escriban físicamente en el disco

        # 3. Guardar Metadata técnica (Vital para reconstruir el mmap después)
        meta = {
            "filename": fname,
            "shape": shape,
            "dtype": "float32",
            "spacing": img.GetSpacing(),
            "origin": img.GetOrigin(),
            "direction": img.GetDirection()
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)

        return True
    except Exception as e:
        return f"Error en {fname}: {str(e)}"

if __name__ == "__main__":
    # Listar todos los archivos .nii.gz en la carpeta normalized
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".nii.gz")]
    
    if not files:
        print(f"No se encontraron archivos .nii.gz en {INPUT_DIR}")
    else:
        print(f"Iniciando construcción de Memmaps para {len(files)} volúmenes...")
        print("Tipos detectados: imagecas_*, asoca_normal_*, asoca_diseased_*")
        
        # Paralelismo con Joblib para máxima velocidad
        results = Parallel(n_jobs=-1)(
            delayed(process_to_memmap)(f) for f in tqdm(files)
        )
        
        # Conteo de resultados
        exitosos = sum(1 for r in results if r is True)
        errores = [r for r in results if isinstance(r, str)]
        
        print(f"\n>>> Proceso finalizado.")
        print(f"Éxitos: {exitosos}")
        if errores:
            print(f"Errores: {len(errores)}")
            for err in errores[:5]: # Mostrar solo los primeros 5 errores
                print(f" - {err}")