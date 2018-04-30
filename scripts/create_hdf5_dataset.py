"""
Script that converts the compressed equally spaced CT scans and masks into a single HDF5 dataset, which lucanode
will use to train the neural networks.
"""

import argparse
from pathlib import Path
from glob import glob

from tqdm import tqdm
import h5py
import SimpleITK as sitk

# Check if dataset exists. If it does not create in chunked mode
# Check what had been added into the dataset. Basically I am interested in which scans. I will use filename as id
# Read scan with sitk. Read as np array. Calculate (z-axis) which slices have something. Save as attribute

VALID_FILE_FORMATS = ['.nii.gz', '.nii', '.mhd']
GROUPS = ['ct_scans', 'lung_masks', 'nodule_masks_spherical', 'nodule_masks_bbox']


def which_file_format(file_path):
    for fmt in VALID_FILE_FORMATS:
        if file_path.endswith(fmt):
            return fmt
    return None


def which_filename(file_path):
    format = which_file_format(file_path)
    file = file_path.split("/")[-1]
    name = file.split(format)[0]
    return name


def slices_with_mask(img_arr):
    return (img_arr != 0).sum(axis=-1).sum(axis=-1) != 0


def store_unprocessed_scans_into_dataset(dataset, filesystem_group_scans, unprocessed_group_ids):
    for group, ids in tqdm(unprocessed_group_ids.items()):
        for scan_id in tqdm(ids):
            img = sitk.ReadImage(filesystem_group_scans[group][scan_id])
            img_arr = sitk.GetArrayFromImage(img)

            img_dataset = dataset[group].create_dataset(scan_id, compression="gzip", data=img_arr)
            img_dataset.attrs["slices_with_mask"] = slices_with_mask(img_arr)


def main(args):
    dataset_filename = Path(args.hdf5_file)
    args_dict = {
        'ct_scans': args.ct_scans,
        'lung_masks': args.lung_masks,
        'nodule_masks_spherical': args.nodule_masks_spherical,
        'nodule_masks_bbox': args.nodule_masks_bbox,
    }

    # Read or create file
    if dataset_filename.exists():
        dataset = h5py.File(dataset_filename, 'r+')
    else:
        dataset = h5py.File(dataset_filename, 'w')

    with dataset as dataset:
        # Create groups if they don't exist
        for g in GROUPS:
            if g not in dataset:
                dataset.create_group(g)

        # Get list of ids already contained in the dataset per group
        processed_group_ids = {g: set(dataset[g].keys()) for g in GROUPS}

        # Expand list of ids and files defined by the glob pattern
        filesystem_group_scans = {}
        for group in GROUPS:
            if args_dict[group]:
                files = {which_filename(f): f for f in glob(args_dict[group]) if which_file_format(f)}
            else:
                files = {}
            filesystem_group_scans[group] = files

        # Get list of pending ids to be added into the dataset
        unprocessed_group_ids = {
            group: set(filesystem_group_scans[group].keys()) - processed_group_ids[group] for group in GROUPS
        }

        store_unprocessed_scans_into_dataset(dataset, filesystem_group_scans, unprocessed_group_ids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Store CT scans and masks into a single HDF5 compressed dataset. ' +
                                     'Valid raw image formats are %s' % (repr(VALID_FILE_FORMATS),))
    parser.add_argument('hdf5_file', type=str, help="File where the dataset should be stored into. Can be an existing"
                        "dataset")
    parser.add_argument('--ct-scans-glob', dest='ct_scans', type=str, default=None,
                        help="glob compatible pattern where the CT scans are")
    parser.add_argument('--lung-masks-glob', dest='lung_masks', type=str, default=None,
                        help="glob compatible pattern where the lung masks are")
    parser.add_argument('--nodule-masks-spherical-glob', dest='nodule_masks_spherical', type=str, default=None,
                        help="glob compatible pattern where the spherical nodule masks are")
    parser.add_argument('--nodule-masks-bbox', dest='nodule_masks_bbox', type=str, default=None,
                        help="glob compatible pattern where the bounding boxed nodule masks are")

    arguments = parser.parse_args()
    main(arguments)
