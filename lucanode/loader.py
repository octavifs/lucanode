from typing import Tuple, Callable, Generator
from glob import glob
import SimpleITK as sitk
import numpy as np

def slices_with_nodules(
    ct_scans: Tuple[str, Callable[[str], str]],
    lung_masks: Tuple[str, Callable[[str], str]],
    nodule_masks: Tuple[str, Callable[[str], str]],
) -> Generator[ Tuple[np.array, np.array], None, None ]:
    """ Return a generator you can pass to a keras model.fit_generator
    The tuple has the format (input, output)
    """
    # Load paths
    ct_scans = { ct_scans[1](path): path for path in glob(ct_scans[0]) }
    lung_masks = { lung_masks[1](path): path for path in glob(lung_masks[0]) }
    nodule_masks = { nodule_masks[1](path): path for path in glob(nodule_masks[0]) }

    ids = set(ct_scans.keys()) & set(lung_masks.keys()) & set(nodule_masks.keys())

    # We are only interested in ids for which we have both the scan and the masks
    iterator = [ (ct_scans[id], lung_masks[id], nodule_masks[id]) for id in ids ]
    for (ct_scan_path, lung_mask_path, nodule_mask_path) in iterator:
        # Load images (not ArrayView, since img goes out of scope and then you have dead pointer)
        ct_scan_img = sitk.GetArrayFromImage(sitk.ReadImage(ct_scan_path))
        lung_mask_img = sitk.GetArrayFromImage(sitk.ReadImage(lung_mask_path))
        nodule_mask_img = sitk.GetArrayFromImage(sitk.ReadImage(nodule_mask_path))

        # Make sure the masks only contain boolean values
        lung_mask_img = lung_mask_img.astype(np.bool)
        nodule_mask_img = nodule_mask_img.astype(np.bool)

        # Apply lung mask to the ct scan
        ct_scan_img *= lung_mask_img

        # Get slice idxs on which a nodule appears
        z_idxs_with_nodules = np.argwhere(np.any(nodule_mask_img, axis=(2,1))).ravel()

        # yield slices
        for z_idx in z_idxs_with_nodules:
            input_arr = ct_scan_img[z_idx, :, :]
            output_arr =nodule_mask_img[z_idx, :, :]
            yield (input_arr, output_arr)
