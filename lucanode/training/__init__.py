TRAINING_SPLIT_PERCENT = 0.0
VALIDATION_SPLIT_PERCENT = 0.8
TEST_SPLIT_PERCENT = 0.9
DEFAULT_UNET_SIZE = (400, 400)


def split_dataset(metadata_df,
                  min_split=TRAINING_SPLIT_PERCENT,
                  mid_split=VALIDATION_SPLIT_PERCENT,
                  max_split=TEST_SPLIT_PERCENT):
    dataset_length = len(metadata_df)
    training_split_idx = int(dataset_length * min_split)
    validation_split_idx = int(dataset_length * mid_split)
    test_split_idx = int(dataset_length * max_split)

    training_df = metadata_df.iloc[training_split_idx:validation_split_idx]
    validation_df = metadata_df[validation_split_idx:test_split_idx]
    return training_df, validation_df
