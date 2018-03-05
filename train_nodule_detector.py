# Launch training
import argparse
import pandas as pd
import numpy as np
from lucanode.training import detection

# Configure tensorflow for memory growth (instead of preallocating upfront)
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train nodule detection neural network')
    parser.add_argument('dataset_metadata', type=str, help="path where the csv metadata of the luna dataset is stored")
    parser.add_argument('dataset_array', type=str, help="path where the np mmaped array of the luna dataset is stored")
    parser.add_argument('weights_file_output', type=str, help="path where the network weights will be saved")
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=5, action="store", help="Training batch size")
    parser.add_argument('--num-epochs', dest='num_epochs', type=int, default=5, action="store",
                        help="Number of training epochs")
    args = parser.parse_args()

    dataset_metadata_df = pd.read_csv(args.dataset_metadata)
    dataset_metadata_df = dataset_metadata_df[dataset_metadata_df["plane"] == "axial"]
    dataset_array = np.load(args.dataset_array, mmap_mode='r')
    detection.train_generator(
        dataset_metadata_df,
        dataset_array,
        args.weights_file_output,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
    )
