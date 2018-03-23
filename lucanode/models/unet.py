from keras.models import *
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, Activation
from keras.optimizers import *
from lucanode.metrics import dice_coef, dice_coef_loss


def ConvBN2D(*args, **kwargs):
    def layer_closure(input_layer):
        layer = Conv2D(*args, **kwargs)(input_layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        return layer
    return layer_closure


def Unet(num_rows: int, num_cols: int) -> Model:
    inputs = Input((num_rows, num_cols, 3))

    conv1 = ConvBN2D(64, 3, padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = ConvBN2D(64, 3, padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = ConvBN2D(128, 3, padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = ConvBN2D(128, 3, padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = ConvBN2D(256, 3, padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = ConvBN2D(256, 3, padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = ConvBN2D(512, 3, padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = ConvBN2D(512, 3, padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = ConvBN2D(1024, 3, padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = ConvBN2D(1024, 3, padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = ConvBN2D(512, 2, padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2,2))(drop5))
    concatenate6 = concatenate([drop4,up6], axis=3)
    conv6 = ConvBN2D(512, 3, padding='same', kernel_initializer='he_normal')(concatenate6)
    conv6 = ConvBN2D(512, 3, padding='same', kernel_initializer='he_normal')(conv6)

    up7 = ConvBN2D(256, 2, padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2,2))(conv6))
    concatenate7 = concatenate([conv3,up7], axis=3)
    conv7 = ConvBN2D(256, 3, padding='same', kernel_initializer='he_normal')(concatenate7)
    conv7 = ConvBN2D(256, 3, padding='same', kernel_initializer='he_normal')(conv7)

    up8 = ConvBN2D(128, 2, padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2,2))(conv7))
    concatenate8 = concatenate([conv2,up8], axis=3)
    conv8 = ConvBN2D(128, 3, padding='same', kernel_initializer='he_normal')(concatenate8)
    conv8 = ConvBN2D(128, 3, padding='same', kernel_initializer='he_normal')(conv8)

    up9 = ConvBN2D(64, 2, padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2,2))(conv8))
    concatenate9 = concatenate([conv1,up9], axis=3)
    conv9 = ConvBN2D(64, 3, padding='same', kernel_initializer='he_normal')(concatenate9)
    conv9 = ConvBN2D(64, 3, padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = ConvBN2D(2, 3, padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    #output = Reshape((num_rows * num_cols, 1))(conv10)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(
        optimizer=Adam(lr=1e-3),
        loss=dice_coef_loss,
        metrics=[dice_coef],
        sample_weight_mode='temporal',
    )

    return model
