from keras.callbacks import ModelCheckpoint
import h5py

from lucanode import loader
from lucanode.training import split_dataset, DEFAULT_UNET_SIZE
from lucanode.metrics import dice_coef_loss
from lucanode.models.unet import Unet, UnetSansBN
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

    with h5py.File(dataset_file, "r") as dataset:
        # Loaders
        training_loader = loader.LungSegmentationSequence(
            dataset,
            batch_size,
            epoch_frac=0.1
        )
        validation_loader = loader.LungSegmentationSequence(
            dataset,
            batch_size,
            subsets={8},
            epoch_frac=0.3,
            epoch_shuffle=False
        )

        # Callbacks
        model_checkpoint = ModelCheckpoint(
            output_weights_file,
            monitor='loss',
            verbose=1,
            save_best_only=True
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
            use_multiprocessing=False,
            shuffle=True,
            callbacks=[model_checkpoint, history_log]
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

    with h5py.File(dataset_file, "r") as dataset:
        # Loaders
        df = loader.dataset_metadata_as_dataframe(dataset, key='nodule_masks_spherical')
        df_training = df[df.subset.isin([0, 1, 2, 3, 4, 5, 6, 7]) & df.has_mask]
        training_loader = loader.NoduleSegmentationSequence(
            dataset,
            batch_size,
            dataframe=df_training,
            epoch_frac=1.0,
            epoch_shuffle=False
        )
        df_validation = df[df.subset.isin([8]) & df.has_mask]
        validation_loader = loader.NoduleSegmentationSequence(
            dataset,
            batch_size,
            dataframe=df_validation,
            epoch_frac=1.0,
            epoch_shuffle=False
        )

        # Callbacks
        model_checkpoint = ModelCheckpoint(
            output_weights_file,
            monitor='loss',
            verbose=1,
            save_best_only=True
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
            use_multiprocessing=False,
            shuffle=True,
            callbacks=[model_checkpoint, history_log]
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

    with h5py.File(dataset_file, "r") as dataset:
        # Loaders
        df = loader.dataset_metadata_as_dataframe(dataset, key='nodule_masks_spherical')
        df_training = df[df.subset.isin([0, 1, 2, 3, 4, 5, 6, 7]) & df.has_mask]
        training_loader = loader.NoduleSegmentationSequence(
            dataset,
            batch_size,
            dataframe=df_training,
            epoch_frac=1.0,
            epoch_shuffle=False
        )
        df_validation = df[df.subset.isin([8]) & df.has_mask]
        validation_loader = loader.NoduleSegmentationSequence(
            dataset,
            batch_size,
            dataframe=df_validation,
            epoch_frac=1.0,
            epoch_shuffle=False
        )

        # Callbacks
        model_checkpoint = ModelCheckpoint(
            output_weights_file,
            monitor='loss',
            verbose=1,
            save_best_only=True
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
            use_multiprocessing=False,
            shuffle=True,
            callbacks=[model_checkpoint, history_log]
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

    with h5py.File(dataset_file, "r") as dataset:
        # Loaders
        df = loader.dataset_metadata_as_dataframe(dataset, key='nodule_masks_spherical')
        df_training = df[df.subset.isin([0, 1, 2, 3, 4, 5, 6, 7]) & df.has_mask]
        training_loader = loader.NoduleSegmentationSequence(
            dataset,
            batch_size,
            dataframe=df_training,
            epoch_frac=1.0,
            epoch_shuffle=False
        )
        df_validation = df[df.subset.isin([8]) & df.has_mask]
        validation_loader = loader.NoduleSegmentationSequence(
            dataset,
            batch_size,
            dataframe=df_validation,
            epoch_frac=1.0,
            epoch_shuffle=False
        )

        # Callbacks
        model_checkpoint = ModelCheckpoint(
            output_weights_file,
            monitor='loss',
            verbose=1,
            save_best_only=True
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
            use_multiprocessing=False,
            shuffle=True,
            callbacks=[model_checkpoint, history_log]
        )


def train_nodule_segmentation_augmentation_normalization_dice(
        dataset_file,
        output_weights_file,
        batch_size=5,
        num_epochs=10,
        last_epoch=0,
        initial_weights=None,
):
    """Train the network from scratch or from a preexisting set of weights on the dataset"""

    with h5py.File(dataset_file, "r") as dataset:
        # Loaders
        df = loader.dataset_metadata_as_dataframe(dataset, key='nodule_masks_spherical')
        df_training = df[df.subset.isin([0, 1, 2, 3, 4, 5, 6, 7]) & df.has_mask]
        training_loader = loader.NoduleSegmentationSequence(
            dataset,
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
            dataset,
            batch_size,
            dataframe=df_validation,
            epoch_frac=1.0,
            epoch_shuffle=True,
            laplacian=False,
            augment_factor=5,
            mislabel=0.0,
        )

        # Callbacks
        model_checkpoint = ModelCheckpoint(
            output_weights_file,
            monitor='loss',
            verbose=1,
            save_best_only=True
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
            use_multiprocessing=False,
            shuffle=True,
            callbacks=[model_checkpoint, history_log]
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

    with h5py.File(dataset_file, "r") as dataset:
        # Loaders
        df = loader.dataset_metadata_as_dataframe(dataset, key='nodule_masks_spherical')
        df_training = df[df.subset.isin([0, 1, 2, 3, 4, 5, 6, 7]) & df.has_mask]
        training_loader = loader.NoduleSegmentation3CHSequence(
            dataset,
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
            dataset,
            batch_size,
            dataframe=df_validation,
            epoch_frac=1.0,
            epoch_shuffle=True,
            laplacian=False,
            augment_factor=5,
            mislabel=0.0,
        )

        # Callbacks
        model_checkpoint = ModelCheckpoint(
            output_weights_file,
            monitor='loss',
            verbose=1,
            save_best_only=True
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
            use_multiprocessing=False,
            shuffle=True,
            callbacks=[model_checkpoint, history_log]
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

    with h5py.File(dataset_file, "r") as dataset:
        # Loaders
        df = loader.dataset_metadata_as_dataframe(dataset, key='nodule_masks_spherical')
        df_training = df[df.subset.isin([0, 1, 2, 3, 4, 5, 6, 7]) & df.has_mask]
        training_loader = loader.NoduleSegmentationSequence(
            dataset,
            batch_size,
            dataframe=df_training,
            epoch_frac=1.0,
            epoch_shuffle=True,
            laplacian=True,
            augment_factor=5,
            mislabel=0.0,
        )
        df_validation = df[df.subset.isin([8]) & df.has_mask]
        validation_loader = loader.NoduleSegmentationSequence(
            dataset,
            batch_size,
            dataframe=df_validation,
            epoch_frac=1.0,
            epoch_shuffle=True,
            laplacian=True,
            augment_factor=5,
            mislabel=0.0,
        )

        # Callbacks
        model_checkpoint = ModelCheckpoint(
            output_weights_file,
            monitor='loss',
            verbose=1,
            save_best_only=True
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
            use_multiprocessing=False,
            shuffle=True,
            callbacks=[model_checkpoint, history_log]
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

    with h5py.File(dataset_file, "r") as dataset:
        # Loaders
        df = loader.dataset_metadata_as_dataframe(dataset, key='nodule_masks_spherical')
        df_training = df[df.subset.isin([0, 1, 2, 3, 4, 5, 6, 7]) & df.has_mask]
        training_loader = loader.NoduleSegmentation3CHSequence(
            dataset,
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
            dataset,
            batch_size,
            dataframe=df_validation,
            epoch_frac=1.0,
            epoch_shuffle=True,
            laplacian=True,
            augment_factor=5,
            mislabel=0.1,
        )

        # Callbacks
        model_checkpoint = ModelCheckpoint(
            output_weights_file,
            monitor='loss',
            verbose=1,
            save_best_only=True
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
            use_multiprocessing=False,
            shuffle=True,
            callbacks=[model_checkpoint, history_log]
        )
