"""Extract nodule candidates + features from model segmentation masks
"""
import numpy as np
import pandas as pd
from skimage import measure


def retrieve_candidates_dataset(seriesuid, dataset_attrs, predictions, threshold=0.5):
    """Extract nodule candidates from scan predictions"""
    nodule_mask = predictions > threshold
    labels = measure.label(nodule_mask)
    regionprops = measure.regionprops(labels)
    rows = []
    for props in regionprops:
        origin = np.array(dataset_attrs["origin"], dtype=np.float32)[::-1]
        centroid_offset = plane_centroid(props.centroid, "axial") * np.array(dataset_attrs["spacing"], dtype=np.float32)[::-1]
        centroid = origin + centroid_offset
        relative_centroid = weighted_relative_centroid(props, "axial")
        squareness = squareness_ratio(props)
        row = {
            "seriesuid": seriesuid,
            "coordX": centroid[2],
            "coordY": centroid[1],
            "coordZ": centroid[0],
            "diameter_mm": props.equivalent_diameter,
            # "relative_coordX": relative_centroid[2],
            # "relative_coordY": relative_centroid[1],
            # "relative_coordZ": relative_centroid[0],
            "squareness": squareness,
            "extent": props.extent,
            "layers": num_layers(props),
            "eccentricity_top": eccentricity_axis(props, nodule_mask, 0),
            "eccentricity_side": eccentricity_axis(props, nodule_mask, 1),
        }
        rows.append(row)
    columns = ["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm",
               # "relative_coordX", "relative_coordY", "relative_coordZ",
               "squareness", "extent", "layers", "eccentricity_top", "eccentricity_side"]
    return pd.DataFrame(rows, columns=columns)


def retrieve_candidates(dataset_metadata_df, predictions, plane, threshold=0.5):
    """Extract nodule candidates from scan predictions"""
    metadata = dataset_metadata_df[(dataset_metadata_df.plane == plane) & (dataset_metadata_df.original_idx == 0)].iloc[0]
    nodule_mask = predictions[:, 0, :, :, 0] > threshold
    labels = measure.label(nodule_mask)
    regionprops = measure.regionprops(labels)
    rows = []
    for props in regionprops:
        origin = np.array(metadata.physical_origin, dtype=np.float32)
        centroid_offset = plane_centroid(props.centroid, plane) * np.array(metadata.spacing, dtype=np.float32)
        centroid = origin + centroid_offset
        relative_centroid = weighted_relative_centroid(props, plane)
        squareness = squareness_ratio(props)
        row = {
            "seriesuid": metadata.seriesuid,
            "coordX": centroid[2],
            "coordY": centroid[1],
            "coordZ": centroid[0],
            "diameter_mm": props.equivalent_diameter,
            # "relative_coordX": relative_centroid[2],
            # "relative_coordY": relative_centroid[1],
            # "relative_coordZ": relative_centroid[0],
            "squareness": squareness,
            "extent": props.extent,
            "layers": num_layers(props),
            "eccentricity_top": eccentricity_axis(props, nodule_mask, 0),
            "eccentricity_side": eccentricity_axis(props, nodule_mask, 1),
        }
        rows.append(row)
    columns = ["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm",
               # "relative_coordX", "relative_coordY", "relative_coordZ",
               "squareness", "extent", "layers", "eccentricity_top", "eccentricity_side"]
    return pd.DataFrame(rows, columns=columns)


def num_layers(props):
    return props.bbox[3] - props.bbox[0]


def eccentricity_axis(props, mask, axis):
    z_min, y_min, x_min, z_max, y_max, x_max = props.bbox
    cut = mask[z_min:z_max, y_min:y_max, x_min:x_max]
    projection_mask = cut.sum(axis=axis) > 0
    labeled_mask = measure.label(projection_mask)
    # expanding the mask by 1px 'cause regionprops can't calculate eccentricity if either of the sides is of length 1
    expanded_labeled_mask = np.zeros(np.array(labeled_mask.shape)+2, dtype=labeled_mask.dtype)
    expanded_labeled_mask[1:-1, 1:-1] = labeled_mask
    p = measure.regionprops(expanded_labeled_mask)[0]
    return p.eccentricity


def squareness_ratio(props):
    shape = np.array(props.bbox)
    shape = shape[3:] - shape[:3]
    plane_shape = sorted(shape[1:])
    squareness_plane = plane_shape[0] / plane_shape[1]
    orto_shape = sorted([shape[0], shape[1:].mean()])
    squareness_orto = orto_shape[0] / orto_shape[1]
    return abs(squareness_plane - squareness_orto)


def weighted_relative_centroid(props, plane):
    shape = np.array(props.bbox)
    shape = shape[3:] - shape[:3]
    relative_centroid = tuple(np.array(props.centroid) / shape)
    return plane_centroid(relative_centroid, plane)


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


#def nodule_likelihood(diameter, squareness):
#    """For now, return a very coarse pdf based on the features passed as parameter"""
#    diameter_likelihood = 1.0 if diameter >= 3 else 0.5
#    squareness_likelihood = 1.0 if squareness < 0.3 else 0.5
#    return diameter_likelihood * squareness_likelihood
