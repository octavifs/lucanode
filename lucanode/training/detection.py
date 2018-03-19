from typing import Tuple, Callable

import numpy as np
from keras.callbacks import ModelCheckpoint

from lucanode import loader
from lucanode.models.unet import Unet

VALIDATION_SPLIT_PERCENT = 0.7
TEST_SPLIT_PERCENT = 0.9


def split_dataset(metadata_df):
    dataset_length = len(metadata_df)
    validation_split_idx = int(dataset_length * VALIDATION_SPLIT_PERCENT)
    test_split_idx = int(dataset_length * TEST_SPLIT_PERCENT)

    training_df = metadata_df.iloc[:validation_split_idx]
    validation_df = metadata_df[validation_split_idx:test_split_idx]
    return training_df, validation_df


def train_generator(
        metadata_df,
        slices_array,
        output_weights_file,
        batch_size=5,
        num_epochs=5,
        initial_epoch=0,
        initial_weights=None,
        use_small_network=False,
):
    """Train the network from scratch or from a preexisting set of weights on the dataset"""
    training_df, validation_df = split_dataset(metadata_df)
    training_loader = loader.LunaSequence(training_df, slices_array, batch_size, True)
    validation_loader = loader.LunaSequence(validation_df, slices_array, batch_size, False)

    model_checkpoint = ModelCheckpoint(
        output_weights_file,
        monitor='loss',
        verbose=1,
        save_best_only=True
    )

    if use_small_network:
        model = Unet(400, 400)
    else:
        model = Unet(512, 512)

    if initial_weights:
        model.load_weights(initial_weights)
    model.fit_generator(
        generator=training_loader,
        epochs=num_epochs,
        initial_epoch=initial_epoch,
        verbose=1,
        validation_data=validation_loader,
        use_multiprocessing=True,
        shuffle=True,
        callbacks=[model_checkpoint]
    )
    return model


def evaluate(
    input_weights_file: str,
    img_shape: Tuple[int, int],
    ct_scans: Tuple[str, Callable[[str], str]],
    lung_masks: Tuple[str, Callable[[str], str]],
    nodule_masks: Tuple[str, Callable[[str], str]],
):
    """Train the network from scratch or from a preexisting set of weights on the dataset"""
    imgs_train = []
    imgs_mask_train = []
    ct = 0
    for img_train, img_mask_train in loader.slices_with_nodules(ct_scans, lung_masks, nodule_masks):
        ct += 1
        if ct >= 10:
            break
        imgs_train.append(img_train[:, :, np.newaxis])
        imgs_mask_train.append(img_mask_train[:, :, np.newaxis])

    imgs_train = np.array(imgs_train)
    imgs_mask_train = np.array(imgs_mask_train)
    
    model = Unet(*img_shape)
    model.load_weights(input_weights_file, by_name=True)
    return model.evaluate(
        x=imgs_train,
        y=imgs_mask_train,
        batch_size=1,
        verbose=1
    )
