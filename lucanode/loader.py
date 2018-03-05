from pathlib import Path
from typing import Tuple, Callable, Iterator
from glob import glob
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from keras.utils import Sequence
from math import ceil
from lucanode import augmentation


def slices_with_nodules(
    ct_scans: Tuple[str, Callable[[str], str]],
    lung_masks: Tuple[str, Callable[[str], str]],
    nodule_masks: Tuple[str, Callable[[str], str]],
    img_filters: Iterator[Callable[[sitk.Image], sitk.Image]] = [],
) -> Iterator[ Tuple[np.array, np.array] ]:
    """ Return a generator that yields slices in the format (input, output)
    """

    ct_scans = { ct_scans[1](path): path for path in glob(ct_scans[0]) }
    lung_masks = { lung_masks[1](path): path for path in glob(lung_masks[0]) }
    nodule_masks = { nodule_masks[1](path): path for path in glob(nodule_masks[0]) }

    ids = set(ct_scans.keys()) & set(lung_masks.keys()) & set(nodule_masks.keys())

    # We are only interested in ids for which we have both the scan and the masks
    images_to_load = [ (ct_scans[id], lung_masks[id], nodule_masks[id]) for id in ids ]
    for (ct_scan_path, lung_mask_path, nodule_mask_path) in \
        tqdm(images_to_load, desc="Loading slices from images:"):
        # Apply image filters to ct_scan, if any
        ct_scan_sitk_img = sitk.ReadImage(ct_scan_path)
        for img_filter in img_filters:
            ct_scan_sitk_img = img_filter(ct_scan_sitk_img)
        # Load images (not ArrayView, since img goes out of scope and then you have dead pointer)
        ct_scan_img = sitk.GetArrayFromImage(ct_scan_sitk_img)
        lung_mask_img = sitk.GetArrayFromImage(sitk.ReadImage(lung_mask_path))
        nodule_mask_img = sitk.GetArrayFromImage(sitk.ReadImage(nodule_mask_path))

        # Make sure the masks only contain boolean values [0, 1] ints
        lung_mask_img = lung_mask_img.astype(np.bool).astype(np.int8)
        nodule_mask_img = nodule_mask_img.astype(np.bool).astype(np.int8)

        # Apply lung mask to the ct scan
        ct_scan_img *= lung_mask_img

        # Get slice idxs on which a nodule appears
        z_idxs_with_nodules = np.argwhere(np.any(nodule_mask_img, axis=(2,1))).ravel()

        # yield slices
        for z_idx in z_idxs_with_nodules:
            # Specifically copy the slice. That way the rest of the image can be garbage collected
            input_arr = ct_scan_img[z_idx, :, :].copy()
            output_arr = nodule_mask_img[z_idx, :, :].copy()
            yield (input_arr, output_arr)


class LunaSequence(Sequence):
    def __init__(self, df, data_array, batch_size, do_augmentation=True):
        """
        Dataset loader to use when called via model.fit_generator
        :param df: Pandas dataframe containing data_array indexes
        :param data_array: mmapped numpy arrat referencing to the slices
        :param batch_size: integer (how many items will __getitem__ return
        """
        # Augment dataframe with transformations and shuffle it in random order
        if do_augmentation:
            df = augmentation.augment_dataframe(df)
        self.df = df.sample(frac=1)
        self.data_array = data_array
        self.batch_size = batch_size

    def __len__(self):
        return ceil(len(self.df) / self.batch_size)

    def _get_batch(self, idx):
        sliced_df = self.df.iloc[idx * self.batch_size: (idx + 1) * self.batch_size]
        rows = [r for _, r in sliced_df.iterrows()]
        arr_idxs = [r.original_idx for r in rows]
        return rows, self.data_array[arr_idxs]

    @staticmethod
    def _apply_augmentation(rows, slices):
        for row, slc in zip(rows, slices):
            row_dict = row.to_dict()
            yield augmentation.apply_chained_transformations(slc, row_dict)

    @staticmethod
    def _split_scan_from_mask(batch):
        for slc in batch:
            slc = slc[:, :, :, np.newaxis]  # Necessary for conv layer
            ct = slc[0, :, :, :]
            lung_mask = slc[0, :, :, :].astype(np.bool).astype(np.int32)
            nodule_mask = slc[2, :, :, :].astype(np.bool).astype(np.float32)
            masked_ct = ct * lung_mask + (lung_mask-2)*4000

            yield (masked_ct, nodule_mask)

    def __getitem__(self, idx):
        batch = self._apply_augmentation(*self._get_batch(idx))

        batch_x = []
        batch_y = []
        for slice_ct, slice_nodule in self._split_scan_from_mask(batch):
            batch_x.append(slice_ct)
            batch_y.append(slice_nodule)

        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        return batch_x, batch_y
