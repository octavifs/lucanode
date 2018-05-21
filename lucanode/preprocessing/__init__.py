from tqdm import tqdm
import numpy as np

from lucanode import augmentation


def load_cubes(df, dataset, cube_size=32):
    seriesuid = None
    ct_scan = None
    cubes = []
    cube_side = cube_size * 3 // 4
    for _, row in tqdm(df.iterrows(), desc="Loading cubes into memory", total=len(df)):
        if row.seriesuid != seriesuid:
            seriesuid = row.seriesuid
            ct_scan_shape_aug = np.array(dataset["ct_scans"][row.seriesuid].shape) + (cube_side * 2)
            ct_scan = augmentation.crop_to_shape(dataset["ct_scans"][row.seriesuid], ct_scan_shape_aug)
        world_coords = np.array([row.coordX, row.coordY, row.coordZ])
        world_origin = np.array(dataset["ct_scans"][row.seriesuid].attrs["origin"])
        vol_coords = np.round(world_coords - world_origin).astype(np.int)[::-1] + cube_side
        z_min = vol_coords[0] - cube_side
        z_max = vol_coords[0] + cube_side
        y_min = vol_coords[1] - cube_side
        y_max = vol_coords[1] + cube_side
        x_min = vol_coords[2] - cube_side
        x_max = vol_coords[2] + cube_side
        cube = ct_scan[z_min:z_max, y_min:y_max, x_min:x_max]
        cubes.append(cube)
    return cubes
