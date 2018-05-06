import argparse
import pandas as pd
import h5py
from tqdm import tqdm
import numpy as np

from lucanode import loader
from lucanode import augmentation
from lucanode.training import DEFAULT_UNET_SIZE
from lucanode.models.unet import Unet, UnetSansBN
from lucanode.metrics import eval_dice_coef
from lucanode import nodule_candidates


def predict(seriesuid, model, dataset_gen, dataset):
    mask_batches = []
    dice_batches = []
    for (X, y), _ in tqdm(zip(dataset_gen, range(len(dataset_gen))), total=len(dataset_gen), desc="eval batches"):
        y_pred = model.predict_on_batch(X)
        if X.shape[0] == 1:
            y_pred = np.array(y_pred)
        dice = eval_dice_coef(y.astype(np.float), y_pred.astype(np.float))  # prediction returns floats
        dice_batches.append(dice)
        mask_batches.append(y_pred)
    scan_dice = np.array(dice_batches)
    scan_mask = np.squeeze(np.concatenate(mask_batches))
    scan_mask = augmentation.crop_to_shape(scan_mask, dataset["lung_masks"][seriesuid].shape)
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
    parser.add_argument('subset', type=int, help="subset for which you want evaluate the segmentation")
    parser.add_argument('csv_output', type=str, help="path where to store the detailed CSV output")
    parser.add_argument('--candidates', dest='csv_candidates', type=str, help="path where to store the candidates CSV output")
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=5, action="store",
                        help="evaluation batch size")
    parser.add_argument('--no-normalization', dest='batch_normalization', action='store_false')
    parser.add_argument('--loss-binary-crossentropy', dest='loss_binary_crossentropy', action='store_true')
    args = parser.parse_args()

    print("""
    
############################################
######### lucanode scan evaluation #########
############################################
""")

    if args.loss_binary_crossentropy:
        network_shape = [*DEFAULT_UNET_SIZE, 1, 'binary_crossentropy']
    else:
        network_shape = [*DEFAULT_UNET_SIZE, 1]

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
            dataset_gen = loader.NoduleSegmentationSequence(
                dataset,
                batch_size=args.batch_size,
                dataframe=df_view,
                do_augmentation=False,
                epoch_frac=1.0,
                epoch_shuffle=False
            )

            # Predict mask
            scan_dice, scan_mask = predict(seriesuid, model, dataset_gen, dataset)

            # Retrieve candidates
            pred_df = nodule_candidates.retrieve_candidates_dataset(seriesuid,
                                                                    dict(dataset["ct_scans"][seriesuid].attrs),
                                                                    scan_mask)
            #candidates.append(pred_df)

            # Evaluate candidates
            pred_df = pred_df.reset_index()
            ann_df_view = ann_df[ann_df.seriesuid == seriesuid].reset_index()
            sensitivity, TP, FP, P = evaluate_candidates(pred_df, ann_df_view)

            # Save metrics
            scan_metrics = {
                "seriesuid": seriesuid,
                "dice": scan_dice.mean(),
                "sensitivity": sensitivity,
                "FP": FP,
                "TP": TP,
                "P": P
            }
            metrics.append(scan_metrics)

    # Export metrics
    columns=["seriesuid", "dice", "sensitivity", "FP", "TP", "P"]
    metrics_df = pd.DataFrame(metrics, columns=columns)
    metrics_df.to_csv(args.csv_output)

    #if args.csv_candidates:
    #    pd.concat(candidates, ignore_index=True).to_csv(args.csv_candidates)

    print("Metrics mean for the subset:")
    print(metrics_df.mean())
    print("\nMetrics variance for the subset:")
    print(metrics_df.var())


if __name__ == '__main__':
    main()
