import SimpleITK as sitk
import numpy as np
import pandas as pd
from keras.utils import Sequence
from math import ceil
from lucanode import augmentation
from skimage import measure
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
import random
import h5py

from lucanode.training import DEFAULT_UNET_SIZE


class LunaSequence(Sequence):
    def __init__(self, df, data_array, batch_size, do_augmentation=True, do_nodule_segmentation=True):
        """
        Dataset loader to use when called via model.fit_generator
        :param df: Pandas dataframe containing data_array indexes
        :param data_array: mmapped numpy array referencing to the slices
        :param batch_size: integer. Indicates how many items will __getitem__ return
        :param do_augmentation: bool. Indicates whether or not to apply data augmentation to the dataset
        :param do_nodule_segmentation: bool. Prepare the sequence to do nodule (default) or lung segmentation
        """
        # Since we use RGB expansion to add pre and posterior slices, we make sure
        # nodule at the edges are not included, since edge+-1 wouldn't exist and then crash
        df = self._filter_nodule_boundary_slices(df)
        # Augment dataframe with transformations and shuffle it in random order
        if do_augmentation:
            df = df.sample(frac=1)
        self.df = df
        self.data_array = data_array
        self.batch_size = batch_size
        self.do_nodule_segmentation = do_nodule_segmentation

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

    def _split_scan_from_mask(self, batches):
        for pre_slc, slc, post_slc in batches:
            masked_ct_arr = []
            nodule_mask_arr = []
            for s in [pre_slc, slc, post_slc]:
                ct = s[0, :, :]
                lung_mask = s[1, :, :].astype(np.bool).astype(np.int32)
                nodule_mask = s[2, :, :].astype(np.bool).astype(np.float32)
                if self.do_nodule_segmentation:
                    masked_ct = ct * lung_mask + (lung_mask-1)*4000
                    mask = nodule_mask
                else:
                    masked_ct = ct
                    mask = lung_mask
                # Add them as a channel
                masked_ct_arr.append(masked_ct)
                nodule_mask_arr.append(mask)
            # Stack pre, slice and post as RGB channels
            masked_ct_arr = np.stack(masked_ct_arr, axis=-1)
            nodule_mask_arr = nodule_mask_arr[1][:, :, np.newaxis]  # Just return mask for the mid slice (not pre/post)
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


def load_scan_in_training_format(
        seriesuid,
        ct_scan_path,
        lung_mask_path,
        nodule_mask_path,
        output_slice_shape=DEFAULT_UNET_SIZE):
    """Load a ct scan (normalized spacing 1x1x1mm) in the same format used by the network while training
    Returns a tuple with a dataframe (rows contain: export_idx, original_idx, plane, seriesuid) and a np.array
    of shape (3, 400, 400) containing 3 channels with the scan, lung mask and nodule mask.
    This output can be directly used by the LunaSequence to evaluate the results.

    The columns physical_origin and spacing of the resulting dataframe have the axes in the order (Z, Y, X)
    """
    # Load scans to np.array
    img = sitk.ReadImage(str(ct_scan_path))
    img_mask = sitk.ReadImage(str(lung_mask_path))
    img_nodule = sitk.ReadImage(str(nodule_mask_path))
    img_arr = sitk.GetArrayFromImage(img)
    img_mask_arr = sitk.GetArrayFromImage(img_mask) > 0
    img_nodule_mask_arr = sitk.GetArrayFromImage(img_nodule) > 0

    # Get spacing
    spacing = img.GetSpacing()

    # Find ct scan bounding box
    label = measure.label(img_mask_arr)
    z_min, y_min, x_min, z_max, y_max, x_max = measure.regionprops(label)[0].bbox

    # Get scan shape
    z_len, y_len, x_len = img_arr.shape
    total_export_slices = z_len + y_len + x_len

    # Prepare variables to generate output
    row_columns = ["export_idx", "original_idx", "plane", "seriesuid", "physical_origin", "spacing"]
    rows_arr = []
    export_slices = np.zeros((total_export_slices, 3, 400, 400), dtype=np.int32)
    export_idx_ct = 0

    with tqdm(total=total_export_slices) as progress_bar:
        progress_bar.set_description("Retrieving individual slices from scan (all planes)")
        # Iterate over the axial slices
        for z_idx in range(z_len):
            axial_plane_ct = img_arr[z_idx, y_min:y_max, x_min:x_max]
            axial_plane_lung_mask = img_mask_arr[z_idx, y_min:y_max, x_min:x_max]
            axial_plane_nodule_mask = img_nodule_mask_arr[z_idx, y_min:y_max, x_min:x_max]
            export_slice, offset = build_slice(axial_plane_ct, axial_plane_lung_mask, axial_plane_nodule_mask,
                                               *output_slice_shape)
            export_slices[export_idx_ct] = export_slice
            new_origin = np.array([z_idx, y_min, x_min]) - np.array([0, offset[0], offset[1]])
            new_origin_sitk = tuple(int(i) for i in new_origin[::-1])  # sitk does not like np.arrays nor np.ints
            physical_origin = img.TransformIndexToPhysicalPoint(new_origin_sitk)[::-1]
            row = [export_idx_ct, z_idx, "axial", seriesuid, physical_origin, spacing]
            rows_arr.append(row)
            export_idx_ct += 1
            progress_bar.update()

        # Iterate over the coronal slices
        for y_idx in range(y_len):
            coronal_plane_ct = img_arr[z_min:z_max, y_idx, x_min:x_max]
            coronal_plane_lung_mask = img_mask_arr[z_min:z_max, y_idx, x_min:x_max]
            coronal_plane_nodule_mask = img_nodule_mask_arr[z_min:z_max, y_idx, x_min:x_max]
            export_slice, offset = build_slice(coronal_plane_ct, coronal_plane_lung_mask, coronal_plane_nodule_mask,
                                               *output_slice_shape)
            export_slices[export_idx_ct] = export_slice
            new_origin = np.array([z_min, y_idx, x_min]) - np.array([offset[0], 0, offset[1]])
            new_origin_sitk = tuple(int(i) for i in new_origin[::-1])  # sitk does not like np.arrays nor np.ints
            physical_origin = img.TransformIndexToPhysicalPoint(new_origin_sitk)[::-1]
            row = [export_idx_ct, y_idx, "coronal", seriesuid, physical_origin, spacing]
            rows_arr.append(row)
            export_idx_ct += 1
            progress_bar.update()

        # Iterate over the sagittal slices
        for x_idx in range(x_len):
            sagittal_plane_ct = img_arr[z_min:z_max, y_min:y_max, x_idx]
            sagittal_plane_lung_mask = img_mask_arr[z_min:z_max, y_min:y_max, x_idx]
            sagittal_plane_nodule_mask = img_nodule_mask_arr[z_min:z_max, y_min:y_max, x_idx]
            export_slice, offset = build_slice(sagittal_plane_ct, sagittal_plane_lung_mask, sagittal_plane_nodule_mask,
                                               *output_slice_shape)
            export_slices[export_idx_ct] = export_slice
            new_origin = np.array([z_min, y_min, x_idx]) - np.array([offset[0], offset[1], 0])
            new_origin_sitk = tuple(int(i) for i in new_origin[::-1])  # sitk does not like np.arrays nor np.ints
            physical_origin = img.TransformIndexToPhysicalPoint(new_origin_sitk)[::-1]
            row = [export_idx_ct, x_idx, "sagittal", seriesuid, physical_origin, spacing]
            rows_arr.append(row)
            export_idx_ct += 1
            progress_bar.update()

    # Create the dataframe and return the slices
    metadata_df = pd.DataFrame(rows_arr, columns=row_columns)
    return metadata_df, export_slices


def build_slice(ct, lung_mask, nodule_mask, height, width):
    slc = np.zeros((3, height, width), dtype=ct.dtype)
    offset = (np.array(slc.shape[1:]) - np.array(ct.shape)) // 2  # Assuming ct is smaller than slc
    slc[0, offset[0]:offset[0] + ct.shape[0], offset[1]:offset[1] + ct.shape[1]] = ct
    slc[1, offset[0]:offset[0] + ct.shape[0], offset[1]:offset[1] + ct.shape[1]] = lung_mask
    slc[2, offset[0]:offset[0] + ct.shape[0], offset[1]:offset[1] + ct.shape[1]] = nodule_mask
    return slc, offset


def dataset_metadata_as_dataframe(dataset, key="lung_masks"):
    scans = set(dataset[key].keys())
    available_ids = {sid for sid in scans}
    metadata = []
    for sid in available_ids:
        num_slices = dataset[key][sid].shape[0]
        attrs = {
            "origin": dataset[key][sid].attrs["origin"],
            "spacing": dataset[key][sid].attrs["spacing"],
            "subset": dataset[key][sid].attrs["subset"],
            "has_mask": dataset[key][sid].attrs["slices_with_mask"]
        }
        for idx in range(num_slices):
            e = {
                "seriesuid": sid,
                "slice_idx": idx,
                "origin": attrs["origin"],
                "spacing": attrs["spacing"],
                "subset": attrs["subset"],
                "has_mask": attrs["has_mask"][idx],
            }
            metadata.append(e)
    return pd.DataFrame(metadata)


class LungSegmentationSequence(Sequence):
    def __init__(self, dataset_path, batch_size, subsets={0, 1, 2, 3, 4, 5, 6, 7}, dataframe=None,
                 epoch_len=None, epoch_frac=1.0, epoch_shuffle=True, laplacian=False, augment_factor=1,
                 mislabel=0.0, nodule_mask_key="nodule_masks_spherical"):
        self._dataset_path = dataset_path
        self._dataset = None
        if dataframe is not None:
            self.df = dataframe
        else:
            df = dataset_metadata_as_dataframe(self.dataset)
            self.df = df[df.subset.isin(subsets)]
            # Doing this 'cause we want to ensure lazy loading when fit_generator starts on a different process
            self._dataset.close()
            self._dataset = None
        self.batch_size = batch_size
        self.augment_factor = augment_factor
        self.img_gen = ImageDataGenerator(
            rotation_range=10,
            shear_range=0.2,
            zoom_range=0.2,
            vertical_flip=True,
            horizontal_flip=True,
            data_format="channels_last",
            fill_mode="nearest",
        )
        self.mislabel = mislabel
        self.nodule_mask_key = nodule_mask_key
        self.epoch_len = epoch_len
        self.epoch_frac = epoch_frac
        self.epoch_shuffle = epoch_shuffle
        if epoch_shuffle:
            self.epoch_df = self.df.sample(n=epoch_len, frac=epoch_frac)
        else:
            self.epoch_df = self.df
        if augment_factor > 1:
            self.epoch_df = pd.concat([self.epoch_df] * self.augment_factor, ignore_index=True).sample(frac=1.0)
        self.laplacian = laplacian

    @property
    def dataset(self):
        """lazy loading of the HDF5 dataset so that it can work well with multiprocessing when training the model"""
        if not self._dataset:
            self._dataset = h5py.File(self._dataset_path, "r")
        return self._dataset

    def __len__(self):
        return ceil(len(self.epoch_df) / self.batch_size)

    def on_epoch_end(self):
        if self.epoch_shuffle:
            self.epoch_df = self.df.sample(n=self.epoch_len, frac=self.epoch_frac)
            self.epoch_df = pd.concat([self.epoch_df] * self.augment_factor, ignore_index=True).sample(frac=1.0)

    def _apply_preprocessing(self, slices):
        for scan, mask in slices:
            scan = augmentation.LaplacianTransform(scan).apply(self.laplacian)
            yield (
                augmentation.crop_to_shape(scan, [*DEFAULT_UNET_SIZE, scan.shape[-1]], cval=-4000),
                augmentation.crop_to_shape(mask, [*DEFAULT_UNET_SIZE, mask.shape[-1]])
            )

    def _apply_augmentation(self, slices):
        for scan, mask in slices:
            if self.augment_factor > 1:
                comb_img = np.concatenate([scan, mask], axis=-1)
                transformed_img = self.img_gen.random_transform(comb_img)
                scan = transformed_img[:, :, :-1]
                mask = transformed_img[:, :, -1]
            if scan.ndim < 3:
                scan = scan[:, :, np.newaxis]
            if mask.ndim < 3:
                mask = mask[:, :, np.newaxis]
            yield scan, mask

    def _get_batch_metadata(self, idx):
        idx_df_min = idx * self.batch_size
        idx_df_max = (idx + 1) * self.batch_size
        for _, r in self.epoch_df.iloc[idx_df_min:idx_df_max].iterrows():
            yield r

    def _get_slices(self, metadata):
        for row in metadata:
            scan = self.dataset["ct_scans"][row.seriesuid][row.slice_idx, :, :]
            mask = self.dataset["lung_masks"][row.seriesuid][row.slice_idx, :, :] > 0
            yield scan[:, :, np.newaxis], mask[:, :, np.newaxis]

    def __getitem__(self, idx):
        metadata_gen = self._get_batch_metadata(idx)
        raw_slices_gen = self._get_slices(metadata_gen)
        slices_preprocessed_gen = self._apply_preprocessing(raw_slices_gen)
        slices_gen = self._apply_augmentation(slices_preprocessed_gen)
        batch_x = []
        batch_y = []
        for scan, mask in slices_gen:
            batch_x.append(scan)
            batch_y.append(mask.astype(np.int))
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        return batch_x, batch_y


class NoduleSegmentationSequence(LungSegmentationSequence):
    def _get_slices(self, metadata):
        for row in metadata:
            scan = self.dataset["ct_scans"][row.seriesuid][row.slice_idx, :, :]
            lung_mask = self.dataset["lung_masks"][row.seriesuid][row.slice_idx, :, :] > 0
            if row.seriesuid in self.dataset[self.nodule_mask_key] and random.random() >= self.mislabel:
                nodule_mask = self.dataset[self.nodule_mask_key][row.seriesuid][row.slice_idx, :, :] > 0
            else:
                nodule_mask = np.zeros(scan.shape, scan.dtype)
            masked_scan = scan * lung_mask + (lung_mask - 1) * 4000  # Apply lung segmentation to the scan
            nodule_mask *= lung_mask
            yield masked_scan[:, :, np.newaxis], nodule_mask[:, :, np.newaxis]


class NoduleSegmentation3CHSequence(LungSegmentationSequence):
    def _get_1ch_slice(self, seriesuid, slice_idx, scan_shape=None):
        if (scan_shape is not None) and (slice_idx < 0 or slice_idx >= scan_shape[0]):
            return np.ones((*scan_shape[1:], 1)) * -4000
        scan = self.dataset["ct_scans"][seriesuid][slice_idx, :, :]
        lung_mask = self.dataset["lung_masks"][seriesuid][slice_idx, :, :] > 0
        masked_scan = scan * lung_mask + (lung_mask - 1) * 4000  # Apply lung segmentation to the scan
        return masked_scan[:, :, np.newaxis]

    def _get_slices(self, metadata):
        for row in metadata:
            scan_shape = self.dataset["ct_scans"][row.seriesuid].shape
            masked_scan = np.concatenate([
                self._get_1ch_slice(row.seriesuid, row.slice_idx - 1, scan_shape),
                self._get_1ch_slice(row.seriesuid, row.slice_idx),
                self._get_1ch_slice(row.seriesuid, row.slice_idx + 1, scan_shape)
            ], axis=-1)
            if row.seriesuid in self.dataset[self.nodule_mask_key] and random.random() >= self.mislabel:
                nodule_mask = self.dataset[self.nodule_mask_key][row.seriesuid][row.slice_idx, :, :] > 0
            else:
                nodule_mask = np.zeros(masked_scan.shape[:-1], masked_scan.dtype)
            lung_mask = self.dataset["lung_masks"][row.seriesuid][row.slice_idx, :, :] > 0
            nodule_mask *= lung_mask
            yield masked_scan, nodule_mask[:, :, np.newaxis]


class NoduleClassificationSequence(Sequence):
    def __init__(self, dataset_path, batch_size, dataframe, do_augmentation=True, epoch_frac=0.5):
        self._dataset_path = dataset_path
        self._dataset = None
        self.df = dataframe
        self.batch_size = batch_size
        self.vol_gen = augmentation.VolumeDataGenerator(
            rotation_range=90,
            shear_range=0.2,
            zoom_range=0.1,
            vertical_flip=True,
            horizontal_flip=True,
            width_shift_range=0.05,
            height_shift_range=0.05,
            data_format="channels_last",
            fill_mode="nearest",
        )
        self.do_augmentation = do_augmentation
        self.cube_size = 32
        self.epoch_frac = epoch_frac
        self.epoch_df = self.df.sample(frac=self.epoch_frac)

    @property
    def dataset(self):
        """lazy loading of the HDF5 dataset so that it can work well with multiprocessing when training the model"""
        if not self._dataset:
            self._dataset = h5py.File(self._dataset_path, "r", driver='core')
        return self._dataset

    def __len__(self):
        return ceil(len(self.epoch_df) / self.batch_size)

    def on_epoch_end(self):
        self.epoch_df = self.df.sample(frac=self.epoch_frac)

    def _apply_augmentation(self, slices):
        for cube, klass in slices:
            if self.do_augmentation:
                cube = self.vol_gen.random_transform(cube)
            cube = augmentation.crop_to_shape(cube, (self.cube_size, self.cube_size, self.cube_size))
            yield cube, klass

    def _get_batch_metadata(self, idx):
        idx_df_min = idx * self.batch_size
        idx_df_max = (idx + 1) * self.batch_size
        for _, r in self.epoch_df.iloc[idx_df_min:idx_df_max].iterrows():
            yield r

    def _get_slices(self, metadata):
        for row in metadata:
            world_coords = np.array([row.coordX, row.coordY, row.coordZ])
            world_origin = np.array(self.dataset["ct_scans"][row.seriesuid].attrs["origin"])
            vol_coords = np.round(world_coords - world_origin).astype(np.int)[::-1] + self.cube_size
            # I'm getting cubes double the size I need so the AffineTransformation won't be lossy
            z_min = vol_coords[0] - self.cube_size
            z_max = vol_coords[0] + self.cube_size
            y_min = vol_coords[1] - self.cube_size
            y_max = vol_coords[1] + self.cube_size
            x_min = vol_coords[2] - self.cube_size
            x_max = vol_coords[2] + self.cube_size
            ct_scan_shape_aug = np.array(self.dataset["ct_scans"][row.seriesuid].shape) + self.cube_size
            ct_scan = augmentation.crop_to_shape(self.dataset["ct_scans"][row.seriesuid], ct_scan_shape_aug)
            cube = ct_scan[z_min:z_max, y_min:y_max, x_min:x_max]
            yield cube, row["class"]

    def __getitem__(self, idx):
        metadata_gen = self._get_batch_metadata(idx)
        raw_slices_gen = self._get_slices(metadata_gen)
        slices_gen = self._apply_augmentation(raw_slices_gen)
        batch_x = []
        batch_y = []
        for cube, klass in slices_gen:
            batch_x.append(cube)
            batch_y.append(klass)
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        return batch_x[:, :, :, :, np.newaxis], batch_y
