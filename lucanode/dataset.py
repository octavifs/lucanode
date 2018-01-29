"Utility functions to deal with the different datasets"


LUNA_CT_SCAN_EXTENSION = ".mhd"
LUNA_LUNG_MASK_EXTENSION = ".mhd"
LUNA_NODULE_MASK_EXTENSION = ".nii.gz"


def _id_luna(path, extension):
    return path.split("/")[-1].split(extension)[0]

def id_ct_scan_luna(path):
    return _id_luna(path, LUNA_CT_SCAN_EXTENSION)

def id_lung_mask_luna(path):
    return _id_luna(path, LUNA_LUNG_MASK_EXTENSION)

def id_nodule_mask_luna(path):
    return _id_luna(path, LUNA_NODULE_MASK_EXTENSION)
