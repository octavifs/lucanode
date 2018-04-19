# lucanode (Lung Cancer Nodule Detection)

Lucanode is a CAD system to automatically detect lung cancer nodules in CT scans.

## Install
Use anaconda. `environment-gpu.yml` and `environment-cpu.yml` contain the dependencies (use one or another based on
whether you have hardware with GPU support or not).

## Current results
TODO. Basically I should put here some current results for the segmentation and the false positive reduction. Also,
better tag those results with the commit they are from, since they are going to become outdated almost certainly.

## The ML pipeline
Steps that need to be performed to make this whole thing work.

- train lung segmentation with cross validation (LUNA folds 0 to 8)
- evaluate lung segmentation (LUNA fold 9)
    - **metric**: DICE
    - **intent**: accuracy score for the automatic lung segmentation
- train nodule segmentation with LUNA lung masks and cross validation (LUNA folds 0 to 8)
- evaluate nodule segmentation (LUNA fold 9)
    - **metric**: DICE
    - **intent**: raw accuracy score for the 2D nodule segmentation NN
- evaluate nodule segmentation using lung segmentation NN with cross validation (LUNA folds 0 to 9)
- extract candidates (LUNA folds 0 to 9)
- evaluate candidates (LUNA fold 9)
    - **metric**: sensitivity & FPxScan
    - **intent**: Baseline score for the nodule segmentation
- train FP reduction classifier using candidates (LUNA folds 0 to 8)
- evaluate FP reduction classifier using candidates (LUNA fold 9)
    - **metric**: accuracy
    - **intent**: raw accuracy score of candidates based on the features extracted on the candidate detection step
- evaluate candidates with FP reduction classifier (LUNA folds 0 to 8)
- evaluate candidates with FP reduction classifier (LUNA fold 9)
    - **metric**: sensitivity & FPxScan
    - **intent**: Final score of the nodule detection system. It is useful to evaluate on the training data to detect
    signs of overfitting (if any).
