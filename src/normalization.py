import SimpleITK as sitk
import numpy as np

def normalize_hu(image, min_hu=-100, max_hu=700):
    arr = sitk.GetArrayFromImage(image).astype(np.float32)

    arr = np.clip(arr, min_hu, max_hu)

    mean = arr.mean()
    std = arr.std() + 1e-8
    arr = (arr - mean) / std

    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(image)

    return out
