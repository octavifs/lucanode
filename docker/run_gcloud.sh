docker run \
    --runtime=nvidia \
    -p 8888:8888 \
    -v ~/lucanode:/notebooks \
    -v ~/lucanode/.jupyter:/root/.jupyter \
    -v /mnt/DATASETS:/mnt/DATASETS \
    lucanode
