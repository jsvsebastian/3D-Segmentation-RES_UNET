import SimpleITK as sitk
import os

def extraer_info_tecnica(ruta_imagen):
    idImage= os.path.basename(ruta_imagen)
    img = sitk.ReadImage(ruta_imagen)
    size = img.GetSize() 
    spacing = img.GetSpacing() 
    dimension = img.GetDimension()
    return idImage, size, spacing, dimension