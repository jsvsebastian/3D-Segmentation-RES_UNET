import numpy as np
import SimpleITK as sitk
from skimage.filters import frangi


def frangi_3d(
    image,
    sigmas=(0.5, 1.0, 1.5, 2.0),
    alpha=0.5,
    beta=0.5,
    gamma=15,
    bright_object=True
):
    """
    Frangi vesselness 3D using scikit-image (stable) + SimpleITK IO
    """

    
    arr = sitk.GetArrayFromImage(image) 


    vesselness = frangi(
        arr,
        sigmas=sigmas,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        black_ridges=not bright_object
    )


    vesselness = (vesselness - vesselness.min()) / (vesselness.max() - vesselness.min() + 1e-8)


    out = sitk.GetImageFromArray(vesselness.astype(np.float32))
    out.CopyInformation(image)

    return out
