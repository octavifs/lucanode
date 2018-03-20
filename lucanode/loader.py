from pathlib import Path
from typing import Tuple, Callable, Iterator
from glob import glob
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from keras.utils import Sequence
from math import ceil
from lucanode import augmentation


class LunaSequence(Sequence):
    def __init__(self, df, data_array, batch_size, do_augmentation=True):
        """
        Dataset loader to use when called via model.fit_generator
        :param df: Pandas dataframe containing data_array indexes
        :param data_array: mmapped numpy arrat referencing to the slices
        :param batch_size: integer (how many items will __getitem__ return
        """
        # Since we use RGB expansion to add pre and posterior slices, we make sure
        # nodule at the edges are not included, since edge+-1 wouldn't exist and then crash
        df = self._filter_nodule_boundary_slices(df)
        # Augment dataframe with transformations and shuffle it in random order
        if do_augmentation:
            df = augmentation.augment_dataframe(df)
            df = df.sample(frac=1)
        self.df = df
        self.data_array = data_array
        self.batch_size = batch_size

    def __len__(self):
        return ceil(len(self.df) / self.batch_size)

    @staticmethod
    def _filter_nodule_boundary_slices(df):
        sorted_df = df.sort_values(by=["seriesuid", "original_idx"])
        slice_nodule_limits = []
        min_export_idx = None
        for idx in range(len(sorted_df) - 1):
            has_index_jumped = sorted_df.iloc[idx + 1].original_idx > (sorted_df.iloc[idx].original_idx + 1)
            has_scan_changed = sorted_df.iloc[idx + 1].seriesuid != sorted_df.iloc[idx].seriesuid
            if min_export_idx is None:
                min_export_idx = sorted_df.iloc[idx].export_idx
            elif has_index_jumped or has_scan_changed:
                max_export_idx = sorted_df.iloc[idx].export_idx
                slice_nodule_limits.append((min_export_idx, max_export_idx))
                min_export_idx = None
        slice_nodule_limits.append((min_export_idx, sorted_df.iloc[-1].export_idx))
        edge_slices_export_idxs = [idx for limits in slice_nodule_limits for idx in limits]
        filtered_df = df[~df.export_idx.isin(edge_slices_export_idxs)]
        return filtered_df

    def _get_batch(self, idx):
        sliced_df = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        rows = [r for _, r in sliced_df.iterrows()]
        arr_idxs = np.array([r.export_idx for r in rows])
        return rows, self.data_array[arr_idxs - 1], self.data_array[arr_idxs], self.data_array[arr_idxs + 1]

    @staticmethod
    def _apply_augmentation(rows, pre_slices, slices, post_slices):
        for row, pre_slc, slc, post_slc in zip(rows, pre_slices, slices, post_slices):
            row_dict = row.to_dict()
            yield (
                augmentation.apply_chained_transformations(pre_slc, row_dict),
                augmentation.apply_chained_transformations(slc, row_dict),
                augmentation.apply_chained_transformations(post_slc, row_dict),
            )

    @staticmethod
    def _split_scan_from_mask(batches):
        for pre_slc, slc, post_slc in batches:
            masked_ct_arr = []
            nodule_mask_arr = []
            for s in [pre_slc, slc, post_slc]:
                ct = s[0, :, :]
                lung_mask = s[1, :, :].astype(np.bool).astype(np.int32)
                nodule_mask = s[2, :, :].astype(np.bool).astype(np.float32)
                masked_ct = ct * lung_mask + (lung_mask-1)*4000
                # Add them as a channel
                masked_ct_arr.append(masked_ct)
                nodule_mask_arr.append(nodule_mask)
            # Stack pre, slice and post as RGB channels
            masked_ct_arr = np.stack(masked_ct_arr, axis=-1)
            nodule_mask_arr = np.stack(nodule_mask_arr, axis=-1)
            yield masked_ct_arr, nodule_mask_arr

    def __getitem__(self, idx):
        augmented_batches_gen = self._apply_augmentation(*self._get_batch(idx))

        batch_x = []
        batch_y = []
        for slice_ct, slice_nodule in self._split_scan_from_mask(augmented_batches_gen):
            batch_x.append(slice_ct)
            batch_y.append(slice_nodule)

        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        return batch_x, batch_y
