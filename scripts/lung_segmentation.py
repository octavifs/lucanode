"""
Perform lung segmentation onto a scan, then return a file with the lung segmentation mask
"""

import argparse
from pathlib import Path
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import zoom
from lucanode.training import DEFAULT_UNET_SIZE
from lucanode.models.unet import Unet
from lucanode import augmentation


VALID_FILE_FORMATS = ['.nii.gz', '.nii', '.mhd']
PREDICT_SPACING = np.array([1.0, 1.0, 1.0])  # (z, y, x). They HAVE to be floats
WEIGHTS_PATH = Path(__file__).parent / "../weights" / "lung_segmentation_e5b2112.h5"


def main(args):
    img = sitk.ReadImage(args.ct_scan)
    img_arr = sitk.GetArrayFromImage(img)
    img_spacing = img.GetSpacing()[::-1]  # since sitk is (x,y,z) but np.array (z,y,x)

    # Resize CT scan
    resize_factor = img_spacing / PREDICT_SPACING
    img_resized_arr = zoom(img_arr, resize_factor)

    # Prepare scan for the prediction
    ct_scan = augmentation.crop_to_shape(img_resized_arr, [img_resized_arr.shape[0], *DEFAULT_UNET_SIZE], cval=-3000)
    ct_scan = ct_scan[:, :, :, np.newaxis]

    # Perform prediction
    network_shape = [*DEFAULT_UNET_SIZE, 1]
    model = Unet(*network_shape)
    model.load_weights(WEIGHTS_PATH, by_name=True)
    mask_arr = model.predict(ct_scan, batch_size=5) > 0.5

    # Resize mask
    mask_arr_crop = augmentation.crop_to_shape(np.squeeze(mask_arr), img_resized_arr.shape)
    mask_resized_arr = zoom(mask_arr_crop, 1 / resize_factor).astype(img_resized_arr.dtype)

    # Set resized scan back as a SimpleITK image object
    mask_resized = sitk.GetImageFromArray(mask_resized_arr)
    mask_resized.SetSpacing(img.GetSpacing())
    mask_resized.SetOrigin(img.GetOrigin())

    # Write image to disk
    sitk.WriteImage(mask_resized, args.mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict lung segmentation mask for a ct scan')
    parser.add_argument('ct_scan', type=str, help="CT scan file. Valid formats: " + repr(VALID_FILE_FORMATS))
    parser.add_argument('mask', type=str, help="Lung segmentation mask. Valid formats: " + repr(VALID_FILE_FORMATS))
    arguments = parser.parse_args()
    main(arguments)
