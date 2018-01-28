docker run \
    -p 8888:8888 \
    -v ~/lucanode/segmentation:/notebooks \
    -v ~/lucanode/.jupyter:/root/.jupyter \
    -v /mnt/DATASETS:/mnt/DATASETS \
    octavifs/lucanode