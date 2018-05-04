# Launch training for lung segmentation
import argparse
from lucanode.training import detection

NETWORK_VARIATIONS = {
    "no_augmentation_no_normalization_binary_crossentropy":
        detection.train_nodule_segmentation_no_augmentation_no_normalization_binary_crossentropy,
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train nodule segmentation neural network')
    parser.add_argument('dataset_hdf5', type=str, help="path where the dataset hdf5 (1mm spacing) is stored")
    parser.add_argument('weights_file_output', type=str, help="path where the network weights will be saved")
    parser.add_argument('variation', type=str, help="Name of the function", choices=NETWORK_VARIATIONS.keys())
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=5, action="store",
                        help="Training batch size")
    parser.add_argument('--num-epochs', dest='num_epochs', type=int, default=5, action="store",
                        help="Number of training epochs")
    parser.add_argument('--last-epoch', dest='last_epoch', type=int, default=0, action='store',
                        help="Last finished epoch (picks up training from there). Useful if passing --initial-weights")
    parser.add_argument('--initial-weights', dest='initial_weights', type=str, default=None, action='store',
                        help="Initial weights to load into the network (.h5 file path)")
    args = parser.parse_args()

    NETWORK_VARIATIONS[args.variation](
        args.dataset_hdf5,
        args.weights_file_output,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        last_epoch=args.last_epoch,
        initial_weights=args.initial_weights,
    )
