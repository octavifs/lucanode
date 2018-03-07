from typing import Tuple, Callable, Iterator
from keras.callbacks import ModelCheckpoint
from lucanode import loader
from lucanode.models.unet import Unet
import numpy as np
import SimpleITK as sitk
from collections import Counter

def _yield_until(iterator, max_iterations=None):
    "Yield until iterator is exhausted or max_iterations is reached"
    if max_iterations is None:
        yield from iterator
    else:
        for e, _ in zip(iterator, range(max_iterations)):
            yield e

def train(
    input_weights_file: str,
    output_weights_file: str,
    img_shape: Tuple[int, int],
    ct_scans: Tuple[str, Callable[[str], str]],
    lung_masks: Tuple[str, Callable[[str], str]],
    nodule_masks: Tuple[str, Callable[[str], str]],
    img_filters: Iterator[Callable[[sitk.Image], sitk.Image]] = [],
    max_samples: int = None
):
    "Train the network from scratch or from a preexisting set of weights on the dataset"
    imgs_train = []
    imgs_mask_train = []
    slices_iterator = loader.slices_with_nodules(ct_scans, lung_masks, nodule_masks, img_filters)
    for img_train, img_mask_train in _yield_until(slices_iterator, max_samples):
        imgs_train.append(img_train[:, :, np.newaxis])
        imgs_mask_train.append(img_mask_train[:, :, np.newaxis])
    imgs_train = np.array(imgs_train)
    imgs_mask_train = np.array(imgs_mask_train)

    # Reshape output as rows of vectors
    samples, height, width, _ = imgs_mask_train.shape
    imgs_mask_train = imgs_mask_train.reshape(samples, height * width, 1)

    # Now calculate average class weights
    imgs_mask_1d = imgs_mask_train.ravel()
    class_count = Counter(imgs_mask_1d)
    sample_weights = np.ones(imgs_mask_train.shape)
    sample_weights[imgs_mask_train == 1] = class_count[0] / class_count[1]
    sample_weights = np.squeeze(sample_weights)

    model_checkpoint = ModelCheckpoint(
        output_weights_file,
        monitor='loss',
        verbose=1,
        save_best_only=True
    )

    model = Unet(*img_shape)
    model.fit(
        imgs_train,
        imgs_mask_train,
        batch_size=4,
        epochs=50,
        verbose=1,
        sample_weight=sample_weights,
        validation_split=0.2,
        shuffle=True,
        callbacks=[model_checkpoint]
    )
    return model


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
    "Train the network from scratch or from a preexisting set of weights on the dataset"
    training_df = metadata_df[metadata_df["export_idx"] % 10 < 7]
    validation_df = metadata_df[(metadata_df["export_idx"] % 10 < 9) & (metadata_df["export_idx"] % 10 >= 7)]
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
    "Train the network from scratch or from a preexisting set of weights on the dataset"
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
