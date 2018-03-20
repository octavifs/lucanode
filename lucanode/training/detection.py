from keras.callbacks import ModelCheckpoint

from lucanode import loader
from lucanode.training import split_dataset, DEFAULT_UNET_SIZE
from lucanode.models.unet import Unet


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
        model = Unet(*DEFAULT_UNET_SIZE)
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
