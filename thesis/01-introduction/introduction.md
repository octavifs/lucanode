# Introduction
\clearpage

Some brief intro as to how I am planning to organize this thing. Let's try to add a citation @Jacobs2015. And I've added something else, let's see if the auto build system picks it up. Does it now? Maybe it is working?

## Lung cancer
Lung cancer is the most deadly cancer in the world. Bring some figures and talk about trends

## Computed Tomography
Basically talk about the technique and how it has been changing diagnosis recently.

## Lung cancer screening with CT
Talk about the NLST study and NELSON. Reduction of 20% in mortality if screened, so early detection is important to improve the outcomes.

## Lung nodule CAD
Explain why it would be useful (reduce workload, reduce intra-variability for radiologists). Also cheaper. Explain why historically they haven't worked (mention main problems a system like this faces) and why I think now is a good time to create a system that improves upon the existing state of the art.

## Deep learning in medical imaging
A brief introduction on deep learning in computer vision, how it started and how it is currently being used to

## Lung cancer detection challenges
Talk about the ongoing competitions about lung cancer, based on the LIDC study (most of them). Talk about LUNA, Kaggle, ANODE09, the ISBI. Also the different variants, with just detection, inferring malignancy, etc. I wouldn't expand too much apart from this and just create a chapter to really drill into what they use and why it works.

## Metrics
Small section to introduce the metrics I'll use and what are they used for and what drawbacks they have:

        - DICE
        - FROC
        - Average FROC
        - AUC
        - TP, FP, sensitivity and F1

## Pipeline:
Describe the pipeline for a CAD system. The 3 problems we are facing and how they stack together. Also mention how we are scoring each of this problems individually.

## Outline
Talk about the chapters, and how the work is organized.

