# Nodule segmentation

## Introduction
I could say that based on the work I did in the LUNA challenge chapter, best approach right now seems UNET based. Explain again that for this part of the system what we are interested in is basically something with very high sensitivity. And finally I guess say that I went for a 2D network cause the images are big, it is a very deep network, and I wanted to avoid as much technical trouble as possible, especially since it was a first for me.

## Methods
Explain the basis of the unet network I am using. Explain also the batch normalization and relu layers I've introduced on the convolutional layers of the network. Also explain all the variations, both in terms of augmentation and preprocessing that I am applying.

Finally talk about the actual evaluation system, which in this case it ain't even visual. Mostly just sensitivity and average FP per scan, which is not nearly as important.

Also interesting and should be mentioned, we want to know if the variants are diverse. That is, if different variations of the network capture different nodules (to test if an ensamble would be a worthwhile approach).

## Results
Put the comparison tables. Also put the venn diagrams. There is no diversity. Finally plot examples of what it is that each variation captures exclusively, so that we can explain what it is we are adding on each step and also what is that we are missing most (how could we improve it). See Table \ref{nodule_detection_results}

\begin{table}
\caption{Nodule segmentation network} \label{nodule_detection_results} 
\begin{tabular}{rrll}
\toprule
                               &      & sensitivity &         FP \\
                               &      &        mean &       mean \\
network & set &             &            \\
\midrule
augmentation normalization bce & test &    0.859275 &   6.011364 \\
                               & train &    0.972629 &   5.703652 \\
                               & validation &    0.922778 &   5.488636 \\
\midrule
augmentation normalization bce 3ch laplacian & test &    0.915490 &   5.750000 \\
                               & train &    0.974303 &   5.515449 \\
                               & validation &    0.940417 &   5.181818 \\
\midrule
augmentation normalization dice & test &    0.339407 &   1.443182 \\
                               & train &    0.390669 &   2.252809 \\
                               & validation &    0.399306 &   1.750000 \\
\midrule
augmentation normalization dice 3ch & test &    0.803672 &  34.125000 \\
                               & train &    0.818526 &  34.228933 \\
                               & validation &    0.806389 &  33.715909 \\
\midrule
augmentation normalization dice 3ch laplacian & test &    0.930791 &  15.420455 \\
                               & train &    0.944604 &  17.234551 \\
                               & validation &    1.044861 &  14.193182 \\
\midrule
no augmentation normalization binary crossentropy & test &    0.783051 &   7.329545 \\
                               & train &    0.977351 &   6.707865 \\
                               & validation &    0.796944 &   6.840909 \\
\midrule
no augmentation normalization dice & test &    0.740254 &   7.125000 \\
                               & train &    0.828008 &   7.063202 \\
                               & validation &    0.795972 &   6.784091 \\
\midrule
unet 3ch axial 400x400 laplacian & test &    0.771234 &  85.488636 \\
                               & train &    0.801148 &  84.466292 \\
                               & validation &    0.861389 &  82.840909 \\
\bottomrule
\end{tabular}
\end{table}

## Discussion
Ok, so basically based on the results, speculate on what is missing. Also mention that we don't really have a conclusive view of wheter having a lesser FP rate (despite lower sensitivity) is a worthwhile tradeoff or not.

Also failings of the current system actually include airways and nodules which are too flat, so basically say that yes, 3D would be better, but the big question is if 3D would actually increase the sensitivity of the system or what. That really is the key. 'Cause if it does not, training a 3D network is VERY expensive (which by the way, should be mentioned in the methods part of this) and not only is it expensive to train, it is expensive to run and evaluate, so really, beware 3D fanboys.
