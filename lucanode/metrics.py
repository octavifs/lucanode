import keras.backend as K
import numpy as np


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    p = true_positives / (predicted_positives + K.epsilon())
    return p


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    r = true_positives / (possible_positives + K.epsilon())
    return r


def f1(y_true, y_pred):
    "From https://stackoverflow.com/a/45305384"
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2*((p*r)/(p+r))


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def np_dice_coef(y_true, y_pred, smooth=1):
    """Just use this to avoid evaluating after a prediction"""
    y_true_f = y_true.astype(np.float).ravel()
    y_pred_f = y_pred.astype(np.float).ravel()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def dice_coef_3ch(y_true, y_pred, smooth=1):
    y_true_f = (
       K.flatten(y_true[:, :, :, 0]) / 4 +  # pre-channel has 25% incidence
       K.flatten(y_true[:, :, :, 2]) / 4 +  # post-channel has 25% incidence
       K.flatten(y_true[:, :, :, 1]) / 2    # current channel has 50% incidence
    )
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss_3ch(y_true, y_pred):
    return 1 - dice_coef_3ch(y_true, y_pred)
