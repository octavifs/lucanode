# Launch training
import argparse
from pathlib import Path
import numpy as np

from lucanode.training import evaluation
from lucanode import loader

# Configure tensorflow for memory growth (instead of preallocating upfront)
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate CT scan segmentation')
    parser.add_argument('seriesuid', type=str, help="seriesuid of the scan")
    parser.add_argument('dataset_path', type=str, help="path where equalized_spacing, equalized_spacing_lung_masks "
                                                       "and equalized_spacing_nodule_masks can be found")
    parser.add_argument('model_weights', type=str, help="path where the model weights are stored")
    parser.add_argument('--plane', dest='plane', default="axial")
    parser.add_argument('--save-transformed-input', dest='save_transformed_input', action='store_true')
    parser.add_argument('--results-folder', dest='results_folder', type=str, default=None,
                        help="Folder to store detailed results")
    args = parser.parse_args()

    if not args.results_folder:
        args.results_folder = str(Path(".") / ("results_" + args.seriesuid))

    print("""

############################################
######### lucanode scan evaluation #########
############################################
""")

    ct_scan_path = Path(args.dataset_path) / "equalized_spacing" / (args.seriesuid + ".nii.gz")
    lung_mask_path = Path(args.dataset_path) / "equalized_spacing_lung_masks" / (args.seriesuid + ".nii.gz")
    nodule_mask_path = Path(args.dataset_path) / "equalized_spacing_nodule_masks" / (args.seriesuid + ".nii.gz")

    print("Retrieving individual slices from scan (all planes)...")
    dataset_metadata_df, dataset_array = loader.load_scan_in_training_format(
        args.seriesuid,
        ct_scan_path,
        lung_mask_path,
        nodule_mask_path
    )
    if args.save_transformed_input:
        results_folder_path = Path(args.results_folder)
        results_folder_path.mkdir(parents=True, exist_ok=True)
        dataset_metadata_df.to_csv((results_folder_path / "input_metadata.csv").open('w'))
        np.save(str(results_folder_path / "input_arr.npy"), dataset_array)

    dataset_metadata_df = dataset_metadata_df[dataset_metadata_df["plane"] == args.plane]
    evaluation.evaluate_generator(
        dataset_metadata_df,
        dataset_array,
        args.model_weights,
        test_split_min=0.0,
        test_split_max=1.0,
        export_results_folder=args.results_folder,
        sort_by_loss=False
    )
