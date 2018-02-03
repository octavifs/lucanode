import SimpleITK as sitk


__LAPLACIAN_RECURSIVE_GAUSSIAN_FILTER = sitk.LaplacianRecursiveGaussianImageFilter()

def laplacian_recursive_gaussian(img: sitk.Image) -> sitk.Image:
    return __LAPLACIAN_RECURSIVE_GAUSSIAN_FILTER.Execute(img)
