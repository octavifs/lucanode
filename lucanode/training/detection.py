from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm

from lucanode import loader
from lucanode import augmentation
from lucanode.training import split_dataset, DEFAULT_UNET_SIZE
from lucanode.metrics import dice_coef_loss
from lucanode.models.unet import Unet, UnetSansBN
from lucanode.models.resnet3d import Resnet3DBuilder
from lucanode.callbacks import HistoryLog


def train_generator(
        metadata_df,
        slices_array,
        output_weights_file,
        batch_size=5,
        num_epochs=5,
        last_epoch=0,
        initial_weights=None,
        do_nodule_segmentation=True
):
    """Train the network from scratch or from a preexisting set of weights on the dataset"""
    training_df, validation_df = split_dataset(metadata_df)
    training_loader = loader.LunaSequence(training_df, slices_array, batch_size, True, do_nodule_segmentation)
    validation_loader = loader.LunaSequence(validation_df, slices_array, batch_size, False, do_nodule_segmentation)

    model_checkpoint = ModelCheckpoint(
        output_weights_file,
        monitor='loss',
        verbose=1,
        save_best_only=True
    )

    model = Unet(*DEFAULT_UNET_SIZE)

    if initial_weights:
        model.load_weights(initial_weights)
    model.fit_generator(
        generator=training_loader,
        epochs=num_epochs,
        initial_epoch=last_epoch,
        verbose=1,
        validation_data=validation_loader,
        use_multiprocessing=True,
        shuffle=True,
        callbacks=[model_checkpoint]
    )
    return model


def train_lung_segmentation(
        dataset_file,
        output_weights_file,
        batch_size=5,
        num_epochs=10,
        last_epoch=0,
        initial_weights=None,
):
    """Train the network from scratch or from a preexisting set of weights on the dataset"""

    # Loaders
    training_loader = loader.LungSegmentationSequence(
        dataset_file,
        batch_size,
        epoch_frac=0.1
    )
    validation_loader = loader.LungSegmentationSequence(
        dataset_file,
        batch_size,
        subsets={8},
        epoch_frac=0.3,
        epoch_shuffle=False
    )

    # Callbacks
    model_checkpoint = ModelCheckpoint(
        output_weights_file,
        monitor='val_loss',
        verbose=1,
        save_best_only=True
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5
    )
    history_log = HistoryLog(output_weights_file + ".history")

    # Setup network
    network_size = [*DEFAULT_UNET_SIZE, 1]
    model = Unet(*network_size)

    if initial_weights:
        model.load_weights(initial_weights)

    # Train
    model.fit_generator(
        generator=training_loader,
        epochs=num_epochs,
        initial_epoch=last_epoch,
        verbose=1,
        validation_data=validation_loader,
        use_multiprocessing=True,
        workers=4,
        max_queue_size=20,
        shuffle=True,
        callbacks=[model_checkpoint, early_stopping, history_log]
    )


def train_nodule_segmentation_no_augmentation_no_normalization_binary_crossentropy(
        dataset_file,
        output_weights_file,
        batch_size=5,
        num_epochs=10,
        last_epoch=0,
        initial_weights=None,
):
    """Train the network from scratch or from a preexisting set of weights on the dataset"""

    # Loaders
    dataset = h5py.File(dataset_file, "r")
    df = loader.dataset_metadata_as_dataframe(dataset, key='nodule_masks_spherical')
    df_training = df[df.subset.isin([0, 1, 2, 3, 4, 5, 6, 7]) & df.has_mask]
    dataset.close()
    training_loader = loader.NoduleSegmentationSequence(
        dataset_file,
        batch_size,
        dataframe=df_training,
        epoch_frac=1.0,
        epoch_shuffle=False
    )
    df_validation = df[df.subset.isin([8]) & df.has_mask]
    validation_loader = loader.NoduleSegmentationSequence(
        dataset_file,
        batch_size,
        dataframe=df_validation,
        epoch_frac=1.0,
        epoch_shuffle=False
    )

    # Callbacks
    model_checkpoint = ModelCheckpoint(
        output_weights_file,
        monitor='val_loss',
        verbose=1,
        save_best_only=True
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5
    )
    history_log = HistoryLog(output_weights_file + ".history")

    # Setup network
    network_size = [*DEFAULT_UNET_SIZE, 1, 'binary_crossentropy']
    model = UnetSansBN(*network_size)

    if initial_weights:
        model.load_weights(initial_weights)

    # Train
    model.fit_generator(
        generator=training_loader,
        epochs=num_epochs,
        initial_epoch=last_epoch,
        verbose=1,
        validation_data=validation_loader,
        use_multiprocessing=True,
        workers=4,
        max_queue_size=20,
        shuffle=True,
        callbacks=[model_checkpoint, early_stopping, history_log]
    )


def train_nodule_segmentation_no_augmentation_normalization_binary_crossentropy(
        dataset_file,
        output_weights_file,
        batch_size=5,
        num_epochs=10,
        last_epoch=0,
        initial_weights=None,
):
    """Train the network from scratch or from a preexisting set of weights on the dataset"""

    # Loaders
    dataset = h5py.File(dataset_file, "r")
    df = loader.dataset_metadata_as_dataframe(dataset, key='nodule_masks_spherical')
    df_training = df[df.subset.isin([0, 1, 2, 3, 4, 5, 6, 7]) & df.has_mask]
    dataset.close()
    training_loader = loader.NoduleSegmentationSequence(
        dataset_file,
        batch_size,
        dataframe=df_training,
        epoch_frac=1.0,
        epoch_shuffle=False
    )
    df_validation = df[df.subset.isin([8]) & df.has_mask]
    validation_loader = loader.NoduleSegmentationSequence(
        dataset_file,
        batch_size,
        dataframe=df_validation,
        epoch_frac=1.0,
        epoch_shuffle=False
    )

    # Callbacks
    model_checkpoint = ModelCheckpoint(
        output_weights_file,
        monitor='val_loss',
        verbose=1,
        save_best_only=True
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5
    )
    history_log = HistoryLog(output_weights_file + ".history")

    # Setup network
    network_size = [*DEFAULT_UNET_SIZE, 1, 'binary_crossentropy']
    model = Unet(*network_size)

    if initial_weights:
        model.load_weights(initial_weights)

    # Train
    model.fit_generator(
        generator=training_loader,
        epochs=num_epochs,
        initial_epoch=last_epoch,
        verbose=1,
        validation_data=validation_loader,
        use_multiprocessing=True,
        workers=4,
        max_queue_size=20,
        shuffle=True,
        callbacks=[model_checkpoint, early_stopping, history_log]
    )


def train_nodule_segmentation_no_augmentation_normalization_dice(
        dataset_file,
        output_weights_file,
        batch_size=5,
        num_epochs=10,
        last_epoch=0,
        initial_weights=None,
):
    """Train the network from scratch or from a preexisting set of weights on the dataset"""

    # Loaders
    dataset = h5py.File(dataset_file, "r")
    df = loader.dataset_metadata_as_dataframe(dataset, key='nodule_masks_spherical')
    df_training = df[df.subset.isin([0, 1, 2, 3, 4, 5, 6, 7]) & df.has_mask]
    dataset.close()
    training_loader = loader.NoduleSegmentationSequence(
        dataset_file,
        batch_size,
        dataframe=df_training,
        epoch_frac=1.0,
        epoch_shuffle=False
    )
    df_validation = df[df.subset.isin([8]) & df.has_mask]
    validation_loader = loader.NoduleSegmentationSequence(
        dataset_file,
        batch_size,
        dataframe=df_validation,
        epoch_frac=1.0,
        epoch_shuffle=False
    )

    # Callbacks
    model_checkpoint = ModelCheckpoint(
        output_weights_file,
        monitor='val_loss',
        verbose=1,
        save_best_only=True
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5
    )
    history_log = HistoryLog(output_weights_file + ".history")

    # Setup network
    network_size = [*DEFAULT_UNET_SIZE, 1, dice_coef_loss]
    model = Unet(*network_size)

    if initial_weights:
        model.load_weights(initial_weights)

    # Train
    model.fit_generator(
        generator=training_loader,
        epochs=num_epochs,
        initial_epoch=last_epoch,
        verbose=1,
        validation_data=validation_loader,
        use_multiprocessing=True,
        workers=4,
        max_queue_size=20,
        shuffle=True,
        callbacks=[model_checkpoint, early_stopping, history_log]
    )


def train_nodule_segmentation_augmentation_normalization_bce(
        dataset_file,
        output_weights_file,
        batch_size=5,
        num_epochs=10,
        last_epoch=0,
        initial_weights=None,
):
    """Train the network from scratch or from a preexisting set of weights on the dataset"""

    # Loaders
    dataset = h5py.File(dataset_file, "r")
    df = loader.dataset_metadata_as_dataframe(dataset, key='nodule_masks_spherical')
    df_training = df[df.subset.isin([0, 1, 2, 3, 4, 5, 6, 7]) & df.has_mask]
    dataset.close()
    training_loader = loader.NoduleSegmentationSequence(
        dataset_file,
        batch_size,
        dataframe=df_training,
        epoch_frac=1.0,
        epoch_shuffle=True,
        laplacian=False,
        augment_factor=5,
        mislabel=0.0,
    )
    df_validation = df[df.subset.isin([8]) & df.has_mask]
    validation_loader = loader.NoduleSegmentationSequence(
        dataset_file,
        batch_size,
        dataframe=df_validation,
        epoch_frac=1.0,
        epoch_shuffle=True,
        laplacian=False,
        augment_factor=1,
        mislabel=0.0,
    )

    # Callbacks
    model_checkpoint = ModelCheckpoint(
        output_weights_file,
        monitor='val_loss',
        verbose=1,
        save_best_only=True
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5
    )
    history_log = HistoryLog(output_weights_file + ".history")

    # Setup network
    network_size = [*DEFAULT_UNET_SIZE, 1, 'binary_crossentropy']
    model = Unet(*network_size)

    if initial_weights:
        model.load_weights(initial_weights)

    # Train
    model.fit_generator(
        generator=training_loader,
        epochs=num_epochs,
        initial_epoch=last_epoch,
        verbose=1,
        validation_data=validation_loader,
        use_multiprocessing=True,
        workers=4,
        max_queue_size=20,
        shuffle=True,
        callbacks=[model_checkpoint, early_stopping, history_log]
    )


def train_nodule_segmentation_augmentation_normalization_dice_3ch(
        dataset_file,
        output_weights_file,
        batch_size=5,
        num_epochs=10,
        last_epoch=0,
        initial_weights=None,
):
    """Train the network from scratch or from a preexisting set of weights on the dataset"""

    # Loaders
    dataset = h5py.File(dataset_file, "r")
    df = loader.dataset_metadata_as_dataframe(dataset, key='nodule_masks_spherical')
    df_training = df[df.subset.isin([0, 1, 2, 3, 4, 5, 6, 7]) & df.has_mask]
    dataset.close()
    training_loader = loader.NoduleSegmentation3CHSequence(
        dataset_file,
        batch_size,
        dataframe=df_training,
        epoch_frac=1.0,
        epoch_shuffle=True,
        laplacian=False,
        augment_factor=5,
        mislabel=0.0,
    )
    df_validation = df[df.subset.isin([8]) & df.has_mask]
    validation_loader = loader.NoduleSegmentation3CHSequence(
        dataset_file,
        batch_size,
        dataframe=df_validation,
        epoch_frac=1.0,
        epoch_shuffle=True,
        laplacian=False,
        augment_factor=1,
        mislabel=0.0,
    )

    # Callbacks
    model_checkpoint = ModelCheckpoint(
        output_weights_file,
        monitor='val_loss',
        verbose=1,
        save_best_only=True
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5
    )
    history_log = HistoryLog(output_weights_file + ".history")

    # Setup network
    network_size = [*DEFAULT_UNET_SIZE, 3, dice_coef_loss]
    model = Unet(*network_size)

    if initial_weights:
        model.load_weights(initial_weights)

    # Train
    model.fit_generator(
        generator=training_loader,
        epochs=num_epochs,
        initial_epoch=last_epoch,
        verbose=1,
        validation_data=validation_loader,
        use_multiprocessing=True,
        workers=4,
        max_queue_size=20,
        shuffle=True,
        callbacks=[model_checkpoint, early_stopping, history_log]
    )


def train_nodule_segmentation_augmentation_normalization_dice_3ch_laplacian(
        dataset_file,
        output_weights_file,
        batch_size=5,
        num_epochs=10,
        last_epoch=0,
        initial_weights=None,
):
    """Train the network from scratch or from a preexisting set of weights on the dataset"""


    # Loaders
    dataset = h5py.File(dataset_file, "r")
    df = loader.dataset_metadata_as_dataframe(dataset, key='nodule_masks_spherical')
    df_training = df[df.subset.isin([0, 1, 2, 3, 4, 5, 6, 7]) & df.has_mask]
    dataset.close()
    training_loader = loader.NoduleSegmentation3CHSequence(
        dataset_file,
        batch_size,
        dataframe=df_training,
        epoch_frac=1.0,
        epoch_shuffle=True,
        laplacian=True,
        augment_factor=5,
        mislabel=0.0,
    )
    df_validation = df[df.subset.isin([8]) & df.has_mask]
    validation_loader = loader.NoduleSegmentation3CHSequence(
        dataset_file,
        batch_size,
        dataframe=df_validation,
        epoch_frac=1.0,
        epoch_shuffle=True,
        laplacian=True,
        augment_factor=1,
        mislabel=0.0,
    )

    # Callbacks
    model_checkpoint = ModelCheckpoint(
        output_weights_file,
        monitor='val_loss',
        verbose=1,
        save_best_only=True
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5
    )
    history_log = HistoryLog(output_weights_file + ".history")

    # Setup network
    network_size = [*DEFAULT_UNET_SIZE, 3, dice_coef_loss]
    model = Unet(*network_size)

    if initial_weights:
        model.load_weights(initial_weights)

    # Train
    model.fit_generator(
        generator=training_loader,
        epochs=num_epochs,
        initial_epoch=last_epoch,
        verbose=1,
        validation_data=validation_loader,
        use_multiprocessing=True,
        workers=4,
        max_queue_size=20,
        shuffle=True,
        callbacks=[model_checkpoint, early_stopping, history_log]
    )


def train_nodule_segmentation_augmentation_normalization_bce_3ch_laplacian(
        dataset_file,
        output_weights_file,
        batch_size=5,
        num_epochs=10,
        last_epoch=0,
        initial_weights=None,
):
    """Train the network from scratch or from a preexisting set of weights on the dataset"""


    # Loaders
    dataset = h5py.File(dataset_file, "r")
    df = loader.dataset_metadata_as_dataframe(dataset, key='nodule_masks_spherical')
    df_training = df[df.subset.isin([0, 1, 2, 3, 4, 5, 6, 7]) & df.has_mask]
    dataset.close()
    training_loader = loader.NoduleSegmentation3CHSequence(
        dataset_file,
        batch_size,
        dataframe=df_training,
        epoch_frac=1.0,
        epoch_shuffle=True,
        laplacian=True,
        augment_factor=5,
        mislabel=0.0,
    )
    df_validation = df[df.subset.isin([8]) & df.has_mask]
    validation_loader = loader.NoduleSegmentation3CHSequence(
        dataset_file,
        batch_size,
        dataframe=df_validation,
        epoch_frac=1.0,
        epoch_shuffle=True,
        laplacian=True,
        augment_factor=1,
        mislabel=0.0,
    )

    # Callbacks
    model_checkpoint = ModelCheckpoint(
        output_weights_file,
        monitor='val_loss',
        verbose=1,
        save_best_only=True
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5
    )
    history_log = HistoryLog(output_weights_file + ".history")

    # Setup network
    network_size = [*DEFAULT_UNET_SIZE, 3, 'binary_crossentropy']
    model = Unet(*network_size)

    if initial_weights:
        model.load_weights(initial_weights)

    # Train
    model.fit_generator(
        generator=training_loader,
        epochs=num_epochs,
        initial_epoch=last_epoch,
        verbose=1,
        validation_data=validation_loader,
        use_multiprocessing=True,
        workers=4,
        max_queue_size=20,
        shuffle=True,
        callbacks=[model_checkpoint, early_stopping, history_log]
    )


def train_nodule_segmentation_augmentation_normalization_dice_3ch_laplacian_mislabeling(
        dataset_file,
        output_weights_file,
        batch_size=5,
        num_epochs=10,
        last_epoch=0,
        initial_weights=None,
):
    """Train the network from scratch or from a preexisting set of weights on the dataset"""

    # Loaders
    dataset = h5py.File(dataset_file, "r")
    df = loader.dataset_metadata_as_dataframe(dataset, key='nodule_masks_spherical')
    df_training = df[df.subset.isin([0, 1, 2, 3, 4, 5, 6, 7]) & df.has_mask]
    dataset.close()
    training_loader = loader.NoduleSegmentation3CHSequence(
        dataset_file,
        batch_size,
        dataframe=df_training,
        epoch_frac=1.0,
        epoch_shuffle=True,
        laplacian=True,
        augment_factor=5,
        mislabel=0.1,
    )
    df_validation = df[df.subset.isin([8]) & df.has_mask]
    validation_loader = loader.NoduleSegmentation3CHSequence(
        dataset_file,
        batch_size,
        dataframe=df_validation,
        epoch_frac=1.0,
        epoch_shuffle=True,
        laplacian=True,
        augment_factor=1,
        mislabel=0.1,
    )

    # Callbacks
    model_checkpoint = ModelCheckpoint(
        output_weights_file,
        monitor='val_loss',
        verbose=1,
        save_best_only=True
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5
    )
    history_log = HistoryLog(output_weights_file + ".history")

    # Setup network
    network_size = [*DEFAULT_UNET_SIZE, 3, dice_coef_loss]
    model = Unet(*network_size)

    if initial_weights:
        model.load_weights(initial_weights)

    # Train
    model.fit_generator(
        generator=training_loader,
        epochs=num_epochs,
        initial_epoch=last_epoch,
        verbose=1,
        validation_data=validation_loader,
        use_multiprocessing=True,
        workers=4,
        max_queue_size=20,
        shuffle=True,
        callbacks=[model_checkpoint, early_stopping, history_log]
    )


def train_fp_reduction_resnet(
        dataset_file,
        candidates_file,
        output_weights_file,
        batch_size=64,
        num_epochs=30,
        last_epoch=0,
        initial_weights=None,
):
    df = pd.read_csv(candidates_file)
    df = df[~df.subset.isin([9])]
    df = df[df.seriesuid.isin(
        pd.Series(df.seriesuid.unique()).sample(frac=0.35)
    )].sort_values("seriesuid")
    with h5py.File(dataset_file, 'r') as dataset:
        cubes = load_cubes(df, dataset)
        df = pd.DataFrame({"cube": cubes, "class": df["class"].tolist(), "subset": df["subset"].tolist()})
    # Rebalance dataset to 2:1
    no_nodule_len = len(df[df["class"] == 0])
    df = pd.concat([
        df[df["class"] == 0],
        df[df["class"] == 1].sample(n=no_nodule_len // 2, replace=True)
    ], ignore_index=True)
    df_training = df[df.subset.isin([0, 1, 2, 3, 4, 5, 6, 7])]
    df_validation = df[df.subset.isin([8])]

    training_loader = loader.NoduleClassificationSequence(
        batch_size,
        dataframe=df_training,
        do_augmentation=True
    )
    validation_loader = loader.NoduleClassificationSequence(
        batch_size,
        dataframe=df_validation,
        do_augmentation=False
    )

    # Callbacks
    model_checkpoint = ModelCheckpoint(
        output_weights_file,
        monitor='val_loss',
        verbose=1,
        save_best_only=True
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5
    )
    history_log = HistoryLog(output_weights_file + ".history")

    # Setup network
    model = Resnet3DBuilder.build_resnet_50((32, 32, 32, 1), 1)
    model.compile(
        optimizer=Adam(lr=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    if initial_weights:
        model.load_weights(initial_weights)

    # Train
    model.fit_generator(
        generator=training_loader,
        epochs=num_epochs,
        initial_epoch=last_epoch,
        verbose=1,
        validation_data=validation_loader,
        use_multiprocessing=False,
        workers=8,
        max_queue_size=20,
        shuffle=True,
        callbacks=[model_checkpoint, early_stopping, history_log]
    )


def load_cubes(df, dataset, cube_size=32):
    seriesuid = None
    ct_scan = None
    cubes = []
    cube_side = cube_size * 3 // 4
    for _, row in tqdm(df.iterrows(), desc="Loading cubes into memory", total=len(df)):
        if row.seriesuid != seriesuid:
            seriesuid = row.seriesuid
            ct_scan_shape_aug = np.array(dataset["ct_scans"][row.seriesuid].shape) + (cube_side * 2)
            ct_scan = augmentation.crop_to_shape(dataset["ct_scans"][row.seriesuid], ct_scan_shape_aug)
        world_coords = np.array([row.coordX, row.coordY, row.coordZ])
        world_origin = np.array(dataset["ct_scans"][row.seriesuid].attrs["origin"])
        vol_coords = np.round(world_coords - world_origin).astype(np.int)[::-1] + cube_side
        z_min = vol_coords[0] - cube_side
        z_max = vol_coords[0] + cube_side
        y_min = vol_coords[1] - cube_side
        y_max = vol_coords[1] + cube_side
        x_min = vol_coords[2] - cube_side
        x_max = vol_coords[2] + cube_side
        cube = ct_scan[z_min:z_max, y_min:y_max, x_min:x_max]
        cubes.append(cube)
    return cubes
