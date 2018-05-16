import argparse
import pandas as pd
import h5py
from tqdm import tqdm
import numpy as np
import os
from pathlib import Path

from lucanode import loader
from lucanode import augmentation
from lucanode.training import DEFAULT_UNET_SIZE
from lucanode.models.unet import Unet, UnetSansBN
from lucanode.metrics import np_dice_coef
from lucanode import nodule_candidates


def predict(seriesuid, model, dataset_gen, dataset_path, mask_type):
    scan_mask = np.squeeze(model.predict_generator(
        generator=dataset_gen,
        use_multiprocessing=True,
        workers=4,
        max_queue_size=20,
        verbose=0
    ))

    with h5py.File(dataset_path, "r") as dataset:
        scan_mask = augmentation.crop_to_shape(scan_mask, dataset["lung_masks"][seriesuid].shape)
        if seriesuid in dataset[mask_type]:
            nodule_mask = dataset[mask_type][seriesuid].value > 0
        else:
            nodule_mask = np.zeros(scan_mask.shape, scan_mask.dtype)
        lung_mask = dataset["lung_masks"][seriesuid].value > 0
        nodule_mask *= lung_mask
        scan_dice = np_dice_coef(scan_mask, nodule_mask)
    return scan_dice, scan_mask


def evaluate_candidates(pred_df, ann_df_view):
    cross_df = pd.merge(pred_df, ann_df_view, on="seriesuid", suffixes=("_pred", "_ann"))
    cross_df["distance"] = np.sqrt(
        (cross_df.coordX_pred - cross_df.coordX_ann) ** 2 +
        (cross_df.coordY_pred - cross_df.coordY_ann) ** 2 +
        (cross_df.coordZ_pred - cross_df.coordZ_ann) ** 2
    )
    cross_df["is_match"] = cross_df["distance"] <= (cross_df["diameter_mm_ann"] / 2)
    TP = len(cross_df[cross_df.is_match])
    FP = len(pred_df) - TP
    P = len(ann_df_view)
    sensitivity = TP / P if P else 0

    return sensitivity, TP, FP, P


def main():
    parser = argparse.ArgumentParser(description='Evaluate CT nodule scan segmentation for a subset')
    parser.add_argument('dataset', type=str, help="Path to the hdf5 with the equalized spaced data")
    parser.add_argument('csv_annotations', type=str, help="CSV with real annotations")
    parser.add_argument('model_weights', type=str, help="path where the model weights are stored")
    parser.add_argument('output', type=str, help="path where to store the detailed output")
    parser.add_argument('subset', type=int, help="subset for which you want evaluate the segmentation")
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=5, action="store",
                        help="evaluation batch size")
    parser.add_argument('--no-normalization', dest='batch_normalization', action='store_false')
    parser.add_argument('--loss-binary-crossentropy', dest='loss_binary_crossentropy', action='store_true')
    parser.add_argument('--laplacian', dest='use_laplacian', action='store_true')
    parser.add_argument('--mask-type', dest='mask_type', default="nodule_masks_spherical", action='store_true')
    parser.add_argument('--ch3', dest='ch3', action='store_true')
    args = parser.parse_args()

    print("""
    
############################################
######### lucanode scan evaluation #########
############################################
""")
    # Create directory for exports if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    if args.ch3:
        num_channels = 3
    else:
        num_channels = 1
    if args.loss_binary_crossentropy:
        network_shape = [*DEFAULT_UNET_SIZE, num_channels, 'binary_crossentropy']
    else:
        network_shape = [*DEFAULT_UNET_SIZE, num_channels]
    if args.batch_normalization:
        model = Unet(*network_shape)
    else:
        model = UnetSansBN(*network_shape)
    model.load_weights(args.model_weights, by_name=True)

    ann_df = pd.read_csv(args.csv_annotations)
    candidates = []

    with h5py.File(args.dataset, "r") as dataset:
        df = loader.dataset_metadata_as_dataframe(dataset, key='ct_scans')
    df = df[df.subset == args.subset]
    scan_ids = set(df.seriesuid)
    metrics = []
    for seriesuid in tqdm(scan_ids, desc="eval scans"):
        # Prepare data loader
        df_view = df[df.seriesuid == seriesuid]
        if args.ch3:
            loader_class = loader.NoduleSegmentation3CHSequence
        else:
            loader_class = loader.NoduleSegmentationSequence
        dataset_gen = loader_class(
            args.dataset,
            batch_size=args.batch_size,
            dataframe=df_view,
            epoch_frac=1.0,
            epoch_shuffle=False,
            laplacian=args.use_laplacian,
        )

        # Predict mask
        scan_dice, scan_mask = predict(seriesuid, model, dataset_gen, args.dataset, args.mask_type)

        # Retrieve candidates
        with h5py.File(args.dataset, "r") as dataset:
            pred_df = nodule_candidates.retrieve_candidates_dataset(seriesuid,
                                                                    dict(dataset["ct_scans"][seriesuid].attrs),
                                                                    scan_mask)
        candidates.append(pred_df)

        # Evaluate candidates
        pred_df = pred_df.reset_index()
        ann_df_view = ann_df[ann_df.seriesuid == seriesuid].reset_index()
        sensitivity, TP, FP, P = evaluate_candidates(pred_df, ann_df_view)

        # Save mask
        dataset_filename = Path(args.output) / "mask_predictions_subset%d.csv" % (args.subset, )
        mode = 'r+' if dataset_filename.exists() else 'w'
        with h5py.File(dataset_filename, mode) as export_ds:
            if seriesuid in export_ds.keys():
                del export_ds[seriesuid]
            export_ds.create_dataset(seriesuid, compression="gzip", data=(scan_mask > 0.5))

        # Save metrics
        scan_metrics = {
            "seriesuid": seriesuid,
            "dice": scan_dice,
            "sensitivity": sensitivity,
            "FP": FP,
            "TP": TP,
            "P": P
        }
        metrics.append(scan_metrics)

    # Export metrics
    columns=["seriesuid", "dice", "sensitivity", "FP", "TP", "P"]
    metrics_df = pd.DataFrame(metrics, columns=columns)
    metrics_df.to_csv(Path(args.output) / "evaluation_subset%d.h5" % (args.subset,))
    pd.concat(candidates, ignore_index=True).to_csv(Path(args.output) / "candidates_subset%d.csv" % (args.subset,))

    metrics = "Metrics mean for the subset: %s\n\nMetrics variance for the subset: %s" % (
        repr(metrics_df.mean()),
        repr(metrics_df.var())
    )
    with open(Path(args.output) / "metrics_subset%d.txt" % (args.subset,), "w") as fd:
        fd.write(metrics)
    print(metrics)


if __name__ == '__main__':
    main()
