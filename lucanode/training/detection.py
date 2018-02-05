from typing import Tuple, Callable, Iterator
from keras.callbacks import ModelCheckpoint
from lucanode import loader
from lucanode.models.unet import Unet
import numpy as np
import SimpleITK as sitk
from collections import Counter


def train(
    input_weights_file: str,
    output_weights_file: str,
    img_shape: Tuple[int, int], 
    ct_scans: Tuple[str, Callable[[str], str]],
    lung_masks: Tuple[str, Callable[[str], str]],
    nodule_masks: Tuple[str, Callable[[str], str]],
    img_filters: Iterator[Callable[[sitk.Image], sitk.Image]] = [],
):
    "Train the network from scratch or from a preexisting set of weights on the dataset"
    imgs_train = []
    imgs_mask_train = []
    iterator = loader.slices_with_nodules(ct_scans, lung_masks, nodule_masks, img_filters)
    for img_train, img_mask_train in iterator:
        imgs_train.append(img_train[:, :, np.newaxis])
        imgs_mask_train.append(img_mask_train[:, :, np.newaxis])
    imgs_train = np.array(imgs_train)
    imgs_mask_train = np.array(imgs_mask_train)

    # Now calculate average class weights
    imgs_mask_1d = imgs_mask_train.ravel()
    total_pixels_masks = len(imgs_mask_1d)
    class_count = Counter(imgs_mask_1d)
    class_weights = {
        -1: 0,  # Essentially, I don't care for pixels outside the lung
         0: 1,  # Pixels inside the lung
         1: class_count[0] / class_count[1]  # Nodule pixels. Represent them at the same level that
                                             # those of the lung
    }

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
        class_weight=class_weights,
        validation_split=0.2,
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
