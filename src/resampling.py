import SimpleITK as sitk

def resamplingImage(image_path, target_spacing):
    img = sitk.ReadImage(image_path)

    size = img.GetSize()
    spacing = img.GetSpacing()

    new_size = [
        int(round(old_size * (old_spacing / target)))
        for old_size, old_spacing, target in zip(size, spacing, target_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkLinear)

    resampled_img = resampler.Execute(img)
    return resampled_img
