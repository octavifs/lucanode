# Lung segmentation

I might just put this after nodule segmentation and false positive reduction, since it basically just an addendum on nodule segmentation that needs to be done for the pipeline to work in an end to end fashin. Interestingly, this chapter could serve to demonstrate the transferability of deep learning techniques to other domains, which is not a bad thing. Essentially the network and everything is exactly the same thing as the nodule segmentation, but using the lung masks as ground truth, instead of nodule masks, so the problem is actually simpler.

## Introduction
Not much really. Basically the idea is that, if the previous network works well for something as complicated as segmenting nodules, segmenting the lungs themselves should be easier, but basically the same concepts should apply. 

## Methods
Pretty much the same as the nodule segmentation, but actually with less preprocessing. I have to check whether or not I did stuff like laplacians and augmentations (I don't think so, really).

Also, in terms of how I am going to evaluate this, 2 measures. One is the typical DICE score, which OK, is good. Problem is, the segmentation itself, although is based on state of the art methods (and here I should really ask with Mario how they were performed, and add this to the introduction of this section). So, since this is basically a preprocessing step, and what I am interested in is to actually detect nodules, what I am going to do is compare how much nodule mass the segmentation is cutting out, compared to the nodule mass lost in the original masks. And if it is close enough, basically I'm going to consider this good enough.

## Results
Put dice score of both. Put lost nodule mass in both, compare the numbers.
Then actually evaluate the cuts of some of the slices and basically show where it fails more.

## Discussion
Maybe discuss about the failings of the current system, such as the holes that sometimes appear inside a lung. Basically all of this stuff could be corrected quite easily by applying some transformations on the mask to fix this obviously wrong problems. Also worth noting is the fact that some expansion of the mask could be done, to diminish problems on the parenchyma, which has nodules that thend to be cancerous (moreso than other areas in the lungs). Also basically say that we could relax our requirements in this part as long as we don't miss nodules nor too much mass, which really would screw us on the overall performance of the system. 'Cause this is a big one on this system. Since it is actually multiple problems in one, you need to know where to put the effort, and this is not really one of the areas that would result in big wins for the system, so let's rule it out and worry about other stuff instead, like FP reduction.
