# General discussion
\newpage

## Lung segmentation
Mention that the initial idea was to perform a multiatlas lung segmentation, along the lines of @Rohlfing2004 and @VanRikxoort2009a, but that I could not justify the cost and complexity to do this, when the neural network performed very well with very little overhead, and it was also fast to run.

Also mention here that it is very interesting how the architecture translates so well, even though the data is different. It was supposed to, but it is quite fantastic that, just by writing a different data loader, you can train a network that works on an entirely different problem and still yield state of the art results.

For the problems exposed in Figure \ref{lung_segmentation}, my approach would be to rebalance the dataset, to prioritize slices from the inferior lobe. Also, applying a laplacian of gaussians as part of the preprocessing would help to increase the initial contrast of the image, which would make the task easier to start with. If anything, some augmentation could be used as well so that the artificial rebalance of the dataset does not cause overfitting. Finally, there are some obvious errors, such as holes inside the masks, which could be dealt with with some image processing, such as closing any masks returned by the algorithm. It would be safe and would increase the scores a little bit.

At the end of the day, all of this is pending work because it is just not the bottleneck of the system. Other areas are much more of a priority (FP reduction) if we want to increase the overall score of the system. Basically in here, even if the segmentation is not perfect, we can apply mask dilation over the automated segmentation and work with that.
