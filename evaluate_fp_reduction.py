# Launch training for lung segmentation
import argparse
from lucanode.training import evaluation

NETWORK_VARIATIONS = {
    "resnet_50": evaluation.evaluate_fp_reduction_resnet
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate fp reduction neural network')
    parser.add_argument('dataset_hdf5', type=str, help="path where the dataset hdf5 (1mm spacing) is stored")
    parser.add_argument('variation', type=str, help="Name of the function", choices=NETWORK_VARIATIONS.keys())
    parser.add_argument('weights_file', type=str, help="path where the network weights will be saved")
    parser.add_argument('candidates_csv', type=str, help="path where the candidates csv is stored")
    parser.add_argument('probabilities_csv', type=str, help="path where the probabilities will be stored")
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=64, action="store",
                        help="Training batch size")
    args = parser.parse_args()

    NETWORK_VARIATIONS[args.variation](
        args.dataset_hdf5,
        args.weights_file,
        args.candidates_csv,
        args.probabilities_csv,
        batch_size=args.batch_size
    )
