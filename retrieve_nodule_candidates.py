# Launch training
import argparse
import logging

from pathlib import Path
import pandas as pd
from tqdm import tqdm

from lucanode.training import evaluation
from lucanode import loader
from lucanode import nodule_candidates

# Configure tensorflow for memory growth (instead of preallocating upfront)
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Retrieve nodule candidates for the given annotations')
    parser.add_argument('annotations_csv', type=str, help="path to the annotations csv file, with ground truth nodules")
    parser.add_argument('dataset_path', type=str, help="path where equalized_spacing, equalized_spacing_lung_masks "
                                                       "and equalized_spacing_nodule_masks can be found")
    parser.add_argument('model_weights', type=str, help="path where the model weights are stored")
    parser.add_argument('candidates_csv', type=str, help="path where the csv with the candidates will be written")
    parser.add_argument('--plane', dest='plane', default="axial")
    args = parser.parse_args()

    print("""

#######################################################
######### lucanode nodule candidate retrieval #########
#######################################################
""")

    annotations_df = pd.read_csv(args.annotations_csv)
    seriesuid_list = set(annotations_df.seriesuid)

    candidates_list = []
    pbar = tqdm(seriesuid_list)
    pbar.set_description("Retrieving nodule candidates")
    for seriesuid in pbar:
        ct_scan_path = Path(args.dataset_path) / "equalized_spacing" / (seriesuid + ".nii.gz")
        lung_mask_path = Path(args.dataset_path) / "equalized_spacing_lung_masks" / (seriesuid + ".nii.gz")
        nodule_mask_path = Path(args.dataset_path) / "equalized_spacing_nodule_masks" / (seriesuid + ".nii.gz")

        if not (ct_scan_path.exists() and lung_mask_path.exists() and nodule_mask_path.exists()):
            logging.warning("Could not find scan for seriesuid " + seriesuid)
            continue

        dataset_metadata_df, dataset_array = loader.load_scan_in_training_format(
            seriesuid,
            ct_scan_path,
            lung_mask_path,
            nodule_mask_path
        )
        dataset_metadata_df = dataset_metadata_df[dataset_metadata_df["plane"] == args.plane]

        _, predictions = evaluation.evaluate_generator(
            dataset_metadata_df,
            dataset_array,
            args.model_weights,
            test_split_min=0.0,
            test_split_max=1.0,
            sort_by_loss=False,
            only_predictions=True
        )

        candidates_df = nodule_candidates.retrieve_candidates(dataset_metadata_df, predictions, args.plane)
        candidates_list.append(candidates_df)

    candidates_merged_df = pd.concat(candidates_list, ignore_index=True)
    candidates_merged_df.to_csv(args.candidates_csv, index=False)
