import os
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
import json
import numpy as np
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

# --- CONFIGURACION ---
BASE_DIR = "/media/mrsmile/IA/tesis"
METADATA_PATH = os.path.join(BASE_DIR, "data/metadata/metadata.pkl")
FILTERED_DIR = os.path.join(BASE_DIR, "data/processed/filtered")
NORMALIZED_DIR = os.path.join(BASE_DIR, "data/processed/normalized")
JSON_PATH = os.path.join(BASE_DIR, "data/metadata/data_split.json")

os.makedirs(NORMALIZED_DIR, exist_ok=True)
df = pd.read_pickle(METADATA_PATH)

def normalize_hu(image, min_hu=-100, max_hu=700):
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    arr = np.clip(arr, min_hu, max_hu)
    mean = arr.mean()
    std = arr.std() + 1e-8
    arr = (arr - mean) / std
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(image)
    return out

# --- CLASIFICACION Y SPLIT ---
df_asoca = df[df['Path'].str.contains('diseased|normal', case=False, na=False)].copy()
df_imagecas = df[~df.index.isin(df_asoca.index)].copy()

tr_cas, temp_cas = train_test_split(df_imagecas.index, test_size=0.30, random_state=42)
vl_cas, ts_cas = train_test_split(temp_cas, test_size=0.50, random_state=42)
tr_asoca, ts_asoca = train_test_split(df_asoca.index, test_size=0.25, random_state=42)

split_map = {
    **{idx: "train" for idx in tr_cas},
    **{idx: "val" for idx in vl_cas},
    **{idx: "test" for idx in ts_cas},
    **{idx: "fine_tuning" for idx in tr_asoca},
    **{idx: "test_blind" for idx in ts_asoca}
}

def process_volume(i, row):
    image_id_raw = str(row["ImageId"])
    path_str = str(row["Path"]).lower()
    num = "".join(filter(str.isdigit, image_id_raw))
    
    # Logica de busqueda por nombre de archivo
    if "diseased" in path_str:
        prefix = "asoca_diseased"
        in_file = f"Diseased_{num}.nii.gz"
    elif "normal" in path_str:
        prefix = "asoca_normal"
        in_file = f"Normal_{num}.nii.gz"
    else:
        prefix = "imagecas"
        in_file = f"{num}.img.nii.gz"

    in_path = os.path.join(FILTERED_DIR, in_file)
    
    # Busqueda alternativa si el nombre exacto falla
    if not os.path.exists(in_path):
        alt_names = [image_id_raw, f"{num}.nii.gz", f"{i+1}.img.nii.gz"]
        for alt in alt_names:
            if os.path.exists(os.path.join(FILTERED_DIR, alt)):
                in_path = os.path.join(FILTERED_DIR, alt)
                break
    
    if not os.path.exists(in_path):
        return None

    out_name = f"{prefix}_{num}.nii.gz"
    out_path = os.path.join(NORMALIZED_DIR, out_name)

    try:
        if not os.path.exists(out_path):
            img = sitk.ReadImage(in_path)
            img_norm = normalize_hu(img)
            sitk.WriteImage(img_norm, out_path)
        
        fase = split_map.get(i)
        source = "asoca" if "asoca" in prefix else "imagecas"
        return (source, fase, out_name)
    except Exception:
        return None

print(f"Iniciando procesamiento en paralelo: {len(df)} archivos...")

results = Parallel(n_jobs=-1)(
    delayed(process_volume)(i, row) for i, row in tqdm(df.iterrows(), total=len(df))
)

# --- ORGANIZACION DE RESULTADOS ---
final_json = {
    "imagecas": {"train": [], "val": [], "test": []},
    "asoca": {"fine_tuning": [], "test_blind": []}
}

for res in results:
    if res:
        source, fase, name = res
        final_json[source][fase].append(name)

with open(JSON_PATH, 'w') as f:
    json.dump(final_json, f, indent=4)

print("\nPROCESO COMPLETADO")
print(f"ImageCAS -> Train: {len(final_json['imagecas']['train'])}, Val: {len(final_json['imagecas']['val'])}, Test: {len(final_json['imagecas']['test'])}")
print(f"ASOCA    -> Fine-Tuning: {len(final_json['asoca']['fine_tuning'])}, Test Blind: {len(final_json['asoca']['test_blind'])}")