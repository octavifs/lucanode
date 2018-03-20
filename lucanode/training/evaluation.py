from pathlib import Path
from multiprocessing import Pool

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from skimage import measure
import pandas as pd

from lucanode import loader
from lucanode.training import split_dataset, DEFAULT_UNET_SIZE
from lucanode.models.unet import Unet


def evaluate_generator(
        metadata_df,
        slices_array,
        model_weights,
        test_split_min,
        test_split_max,
        export_results_folder=None,
):
    """Evaluate the network on a set of slices from the metadata data frame"""

    print("""

#############################################
######### lucanode slice evaluation #########
#############################################
""")
    test_df, _ = split_dataset(metadata_df, test_split_min, test_split_max, test_split_max)

    if not len(test_df):
        print("Dataset is empty, can't evaluate.")
        return

    model = Unet(*DEFAULT_UNET_SIZE)
    model.load_weights(model_weights, by_name=True)
    test_loader = loader.LunaSequence(test_df, slices_array, 1, False)

    print("Predicting segmentation on all test slices...")
    prediction_arr = []
    for x, _ in tqdm(test_loader):
        y_pred = model.predict(x, batch_size=1, verbose=0)
        prediction_arr.append(y_pred)
    prediction_arr = np.array(prediction_arr)

    print("Evaluating model on all test slices...")
    loss_arr = []
    for x, y in tqdm(test_loader):
        loss, _ = model.evaluate(x, y, verbose=0)
        loss_arr.append(loss)
    loss_arr = np.array(loss_arr)

    overall_results_str = "Overall model results:\nloss mean: %f; std: %f; max: %f; min: %f\n" % \
          (loss_arr.mean(), loss_arr.std(), loss_arr.max(), loss_arr.min())

    # Save predictions to a folder with pictures, a csv and shit
    if export_results_folder is not None:
        num_rows = len(test_df)
        results_folder_path = Path(export_results_folder)
        results_folder_path.mkdir(parents=True, exist_ok=True)
        rows_gen = (e[1] for e in test_df.iterrows())
        x_gen = (e[0] for e in test_loader)
        y_gen = (e[1] for e in test_loader)
        print("Exporting results on all test slices...")
        args_arr = [args for args in zip(loss_arr, rows_gen, x_gen, y_gen, prediction_arr)]
        args_arr = sorted(args_arr, key=lambda a: a[0], reverse=True)  # Sort in descending order, by loss (worse first)
        figure_paths = [str(results_folder_path / ("%06d" % (idx,))) + ".png" for idx in range(num_rows)]
        args_arr = [(*args, figure_path) for args, figure_path in zip(args_arr, figure_paths)]
        export_arr = []
        export_columns = ["loss", "plot_image", "export_idx", "original_idx", "plane", "seriesuid"]
        # Export slice results as csv
        for args in tqdm(args_arr):
            df_row = args[1]
            export_row = [
                args[0],
                Path(args[-1]).name,
                df_row.export_idx,
                df_row.original_idx,
                df_row.plane,
                df_row.seriesuid,
            ]
            export_arr.append(export_row)
        pd.DataFrame(export_arr, columns=export_columns).to_csv((results_folder_path / 'results_slices.csv').open('w'))

        # Export slice results as plots via multiprocessing
        with Pool() as p:
            with tqdm(total=num_rows) as progress_bar:
                for _ in tqdm(p.imap_unordered(calculate_results_per_slice, args_arr)):
                    progress_bar.update()

        # Plot loss histogram
        loss_hist, bins = np.histogram(np.array(loss_arr), bins=20, range=(0, 1), density=True)
        plt.figure(figsize=(7, 5))
        plt.hist(loss_hist, bins)
        plt.title(overall_results_str)
        plt.savefig(str(results_folder_path / 'results_overall.png'), bbox_inches='tight')
        plt.close()

    print(overall_results_str)


def calculate_results_per_slice(loss, df_row, x, y, pred, figure_path):
    truth_handle = mlines.Line2D([], [], color='b', label='ground truth')
    pred_handle = mlines.Line2D([], [], color='r', label='prediction')

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
    ax1.set_title("pre slice")
    ax1.imshow(x[0, :, :, 0], cmap='gray')
    draw_nodule_contour(ax1, y[0, :, :, 0], color='b')
    ax1.legend(handles=[truth_handle])

    ax2.set_title("slice")
    ax2.imshow(x[0, :, :, 1], cmap='gray')
    draw_nodule_contour(ax2, y[0, :, :, 1], color='b')
    draw_nodule_contour(ax2, pred[0, :, :, 0], color='r')
    ax2.legend(handles=[truth_handle, pred_handle])

    ax3.set_title("post slice")
    ax3.imshow(x[0, :, :, 2], cmap='gray')
    draw_nodule_contour(ax3, y[0, :, :, 2], color='b')
    ax3.legend(handles=[truth_handle])

    y_truth_hist, bins = np.histogram(y[0, :, :, 1], bins=20, range=(0, 1))
    y_pred_hist, _ = np.histogram(pred[0, :, :, 0], bins=20, range=(0, 1))
    hist = np.stack([y_truth_hist, y_pred_hist], axis=-1)
    colors = ['b', 'r']
    labels = ['ground truth', 'prediction']
    ax4.set_title("nodule mask value histogram")
    ax4.hist(hist, bins, color=colors, label=labels, histtype='bar')
    ax4.set_yscale("log")
    ax4.legend()

    plt.suptitle("loss: %f; export_idx: %d, original_idx: %d, plane: %s; seriesuid: %s" %
                 (loss, df_row.export_idx, df_row.original_idx, df_row.plane, df_row.seriesuid))
    plt.savefig(figure_path, bbox_inches='tight')
    plt.close()


def draw_nodule_contour(ax, nodule_mask, color):
    contours = measure.find_contours(nodule_mask, 0.8)
    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color=color, alpha=0.8)

