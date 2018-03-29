"""Extract nodule candidates + features from model segmentation masks
"""
import numpy as np
import pandas as pd
from skimage import measure


def retrieve_candidates(dataset_metadata_df, predictions, plane, threshold=0.5):
    """Extract nodule candidates from scan predictions"""
    metadata = dataset_metadata_df[(dataset_metadata_df.plane == plane) & (dataset_metadata_df.original_idx == 0)].iloc[0]
    nodule_mask = predictions[:, 0, :, :, 0] > threshold
    labels = measure.label(nodule_mask, connectivity=2)
    regionprops = measure.regionprops(labels)
    rows = []
    for props in regionprops:
        origin = np.array(metadata.physical_origin, dtype=np.float32)
        centroid_offset = plane_centroid(props.centroid, plane) * np.array(metadata.spacing, dtype=np.float32)
        centroid = origin + centroid_offset
        squareness = squareness_ratio(props)
        row = {
            "seriesuid": metadata.seriesuid,
            "coordX": centroid[2],
            "coordY": centroid[1],
            "coordZ": centroid[0],
            "probability": nodule_likelihood(props.equivalent_diameter, squareness),
            "diameter": props.equivalent_diameter,
            "squareness": squareness,
        }
        rows.append(row)
    columns = ["seriesuid", "coordX", "coordY", "coordZ", "probability", "diameter", "squareness"]
    return pd.DataFrame(rows, columns=columns)


def squareness_ratio(props):
    shape = np.array(props.bbox)
    shape = shape[3:] - shape[:3]
    plane_shape = sorted(shape[1:])
    squareness_plane = plane_shape[0] / plane_shape[1]
    orto_shape = sorted([shape[0], shape[1:].mean()])
    squareness_orto = orto_shape[0] / orto_shape[1]
    return abs(squareness_plane - squareness_orto)


def plane_centroid(point, plane):
    """Returns point [z, y, x]"""
    if plane == "axial":
        return np.array([point[0], point[1], point[2]])
    elif plane == "coronal":
        return np.array([point[1], point[0], point[2]])
    elif plane == "sagittal":
        return np.array([point[1], point[2], point[0]])
    else:
        raise ValueError("Invalid plane " + str(plane))


def nodule_likelihood(diameter, squareness):
    """For now, return a very coarse pdf based on the features passed as parameter"""
    diameter_likelihood = 1.0 if diameter >= 3 else 0.5
    squareness_likelihood = 1.0 if squareness < 0.3 else 0.5
    return diameter_likelihood * squareness_likelihood
