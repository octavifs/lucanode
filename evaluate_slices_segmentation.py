# Launch training
import argparse
import pandas as pd
import numpy as np
from lucanode.training import evaluation

# Configure tensorflow for memory growth (instead of preallocating upfront)
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate nodule segmentation on individual CT slices')
    parser.add_argument('dataset_metadata', type=str, help="path where the csv metadata of the luna dataset is stored")
    parser.add_argument('dataset_array', type=str, help="path where the np mmaped array of the luna dataset is stored")
    parser.add_argument('model_weights', type=str, help="path where the model weights are stored")
    parser.add_argument('--plane', dest='plane', default="axial")
    parser.add_argument('--split-min', dest='test_split_min', type=float, default=0.9, help="min split % on dataset")
    parser.add_argument('--split-max', dest='test_split_max', type=float, default=1.0, help="max split % on dataset")
    parser.add_argument('--results-folder', dest='results_folder', type=str, default=None,
                        help="Folder to store detailed results")
    args = parser.parse_args()

    print("""

#############################################
######### lucanode slice evaluation #########
#############################################
""")

    dataset_metadata_df = pd.read_csv(args.dataset_metadata)
    dataset_metadata_df = dataset_metadata_df[dataset_metadata_df["plane"] == args.plane]
    dataset_array = np.load(args.dataset_array, mmap_mode='r')

    evaluation.evaluate_generator(
        dataset_metadata_df,
        dataset_array,
        args.model_weights,
        test_split_min=args.test_split_min,
        test_split_max=args.test_split_max,
        export_results_folder=args.results_folder
    )
