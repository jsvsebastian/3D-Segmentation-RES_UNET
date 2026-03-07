import os
import gc
import SimpleITK as sitk
from tqdm import tqdm
from skimage.filters import frangi

INPUT_DIR = "/media/mrsmile/IA/tesis/data/processed/normalized"
OUTPUT_DIR = "/media/mrsmile/IA/tesis/data/processed/vesselness"

os.makedirs(OUTPUT_DIR, exist_ok=True)

paths = sorted([p for p in os.listdir(INPUT_DIR) if p.endswith(".nii.gz")])

for name in tqdm(paths):
    try:
        in_path = os.path.join(INPUT_DIR, name)
        out_path = os.path.join(OUTPUT_DIR, name)

        if os.path.exists(out_path):
            continue


        img = sitk.ReadImage(in_path)
        arr = sitk.GetArrayFromImage(img)

   
        vessel = frangi(
            arr,
            sigmas=(0.5, 1.0, 1.5, 2.0),
            alpha=0.5,
            beta=0.5,
            gamma=15,
            black_ridges=False
        )

        vessel_img = sitk.GetImageFromArray(vessel)
        vessel_img.CopyInformation(img)
        sitk.WriteImage(vessel_img, out_path)

        del img, arr, vessel, vessel_img
        gc.collect()

    except Exception as e:
        print(f"Error en {name}: {e}")
