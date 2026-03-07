import os
import sys
sys.path.append(".")

import SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from tqdm import tqdm

from src.filtering import anisotropic_filter


INPUT_DIR = "data/processed/resampled"
OUTPUT_DIR = "data/processed/filtered"

ITERATIONS = 5
TIMESTEP = 0.02
CONDUCTANCE = 1.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_one(path):
    try:
        base = os.path.splitext(os.path.basename(path))[0]
        name = base + ".nii.gz"
        out_path = os.path.join(OUTPUT_DIR, name)

        if os.path.exists(out_path):
            return

        img = sitk.ReadImage(path)

        img_f = anisotropic_filter(
            img,
            iterations=ITERATIONS,
            timestep=TIMESTEP,
            conductance=CONDUCTANCE
        )

        sitk.WriteImage(img_f, out_path)

    except Exception as e:
        print("Error:", path)
        print(e)

# ===============================
# MAIN
# ===============================
def main():
    paths = sorted([os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR)])

    workers = max(1, mp.cpu_count() - 2)
    print(f"Using {workers} workers")

    with ProcessPoolExecutor(max_workers=workers) as exe:
        list(tqdm(exe.map(process_one, paths), total=len(paths)))

if __name__ == "__main__":
    main()
