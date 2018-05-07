import argparse
import pandas as pd
import h5py
from tqdm import tqdm

from lucanode import loader
from lucanode.training import DEFAULT_UNET_SIZE
from lucanode.models.unet import Unet


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate CT lung scan segmentation for a subset using DICE')
    parser.add_argument('dataset', type=str, help="Path to the hdf5 with the equalized spaced data")
    parser.add_argument('model_weights', type=str, help="path where the model weights are stored")
    parser.add_argument('subset', type=int, help="subset for which you want evaluate the segmentation")
    parser.add_argument('csv_output', type=str, help="path where to store the detailed CSV output")
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=5, action="store",
                        help="evaluation batch size")
    args = parser.parse_args()

    print("""
    
############################################
######### lucanode scan evaluation #########
############################################
""")

    network_shape = [*DEFAULT_UNET_SIZE, 1]
    model = Unet(*network_shape)
    model.load_weights(args.model_weights, by_name=True)

    with h5py.File(args.dataset, "r") as dataset:
        df = loader.dataset_metadata_as_dataframe(dataset)
        df = df[df.subset == args.subset]
        scan_ids = set(df.seriesuid)
        metrics = []
        for seriesuid in tqdm(scan_ids):
            df_view = df[df.seriesuid == seriesuid]
            dataset_gen = loader.LungSegmentationSequence(
                dataset,
                batch_size=args.batch_size,
                dataframe=df_view,
                epoch_frac=1.0,
                epoch_shuffle=False
            )
            scan_metrics = [seriesuid, *model.evaluate_generator(dataset_gen)]
            metrics.append(scan_metrics)
        metrics_df = pd.DataFrame(metrics, columns=["seriesuid", *model.metrics_names])
        metrics_df.to_csv(args.csv_output)

        print("Metrics mean for the subset:\n", metrics_df.mean())
        print("\nMetrics variance for the subset:\n", metrics_df.var())
