import SimpleITK as sitk

def anisotropic_filter(image, iterations=3, timestep=0.02, conductance=1.5):
    image = sitk.Cast(image, sitk.sitkFloat32)

    f = sitk.CurvatureAnisotropicDiffusionImageFilter()
    f.SetNumberOfIterations(iterations)
    f.SetTimeStep(timestep)
    f.SetConductanceParameter(conductance)

    return f.Execute(image)
