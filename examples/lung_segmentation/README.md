# Lung segmentation with lucanode and docker

## Setup
**PLEASE DO THIS**: Download the lung segmentation weights file from
this [link](https://www.dropbox.com/s/c75fuur0yc6nkyw/lung_segmentation_e5b2112.h5?dl=0)
and save it into the weights folder of this repo (same name) before 
building the docker image. Otherwise it won't work.

## Intro
Lucanode has a script to segment any arbitrary CT scan supported by SimpleITK
(file formats .mhd, .nii or .nii.gz are safe to use) to a mask file.

This script can be executed from the lucanode docker image, so there is no
need to install dependencies in your system.

NOTE: GPU assisted prediction is **NOT** enabled in the Docker image, so
the network will use CPU instead. For a whole scan (around 300 slices) this
can take a long time (around 1h).

For testing purposes I've included shorter ct scans, so the execution of
the image is going to be faster.

## Building the docker image
Run docker build at the root of the repo:

    $ docker build -t lucanode .

## Performing a lung segmentation

Here's an example on how to call the lung segmentation script to produce
its segmentation mask:

    $ export INPUT_FOLDER=/path/to/this/repo/examples/lung_segmentation
    $ export OUTPUT_FOLDER=/real/path/to/destination
    $ docker run \
    -v $INPUT_FOLDER:/input \
    -v $OUTPUT_FOLDER:/output \
    lucanode \
    python scripts/lung_segmentation.py /input/ct_scan_1_short.nii.gz /output/ct_scan_1_short_mask.nii.gz


## Visualizing scan + segmentation

There is a jupyter notebook that can be used to visualize the outcome of
the segmentation along the scan itself. The following runs it and exports
the results as a self-contained HTML:

    $ export INPUT_FOLDER=/path/to/this/repo/examples/lung_segmentation
    $ export OUTPUT_FOLDER=/real/path/to/destination
    $ docker run \
    -v $INPUT_FOLDER:/input \
    -v $OUTPUT_FOLDER:/output \
    -e CT_SCAN_FILE=/input/ct_scan_1_short.nii.gz \
    -e PRED_MASK_FILE=/input/ct_scan_1_short_mask.nii.gz \
    lucanode \
    jupyter nbconvert --output-dir=/output --execute --ExecutePreprocessor.timeout=0 --to html notebooks/display_lung_segmentation_results.ipynb

[Here's](display_lung_segmentation_results.html) an example of the report this generates.