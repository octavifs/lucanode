# Conclusions
\newpage

## Lung segmentation
Edges need more training. Especially end of lung is hard for the segmentation. Still, this could be easily fixed by retraining, we just need to rebalance the dataset to oversample on the last 20% of the lung, which is otherwise underrepresented. Also, it would be easy to apply some expansion on the mask, and maybe a bit of smoothing, to correct out rough edges. Finally, we know that masks are not supposed to have holes inside them, so that could be automatically removed. Also, no filtering has been applied to the original image, nor image augmentation of any kind. Probably both ideas would help a lot.

Even though this part can be improved, in the grand scheme of the system, it wouldn't yield much benefit. Basically we are barely losing any significant mass of the segmented nodules with the new masks, so the possible gains are minimal. We are much better off focusing on a better FP reduction.

Maybe discuss about the failings of the current system, such as the holes that sometimes appear inside a lung. Basically all of this stuff could be corrected quite easily by applying some transformations on the mask to fix this obviously wrong problems. Also worth noting is the fact that some expansion of the mask could be done, to diminish problems on the parenchyma, which has nodules that thend to be cancerous (moreso than other areas in the lungs). Also basically say that we could relax our requirements in this part as long as we don't miss nodules nor too much mass, which really would screw us on the overall performance of the system. 'Cause this is a big one on this system. Since it is actually multiple problems in one, you need to know where to put the effort, and this is not really one of the areas that would result in big wins for the system, so let's rule it out and worry about other stuff instead, like FP reduction.


## Nodule detection
In here
Ok, so basically based on the results, speculate on what is missing. Also mention that we don't really have a conclusive view of wheter having a lesser FP rate (despite lower sensitivity) is a worthwhile tradeoff or not.

Also failings of the current system actually include airways and nodules which are too flat, so basically say that yes, 3D would be better, but the big question is if 3D would actually increase the sensitivity of the system or what. That really is the key. 'Cause if it does not, training a 3D network is VERY expensive (which by the way, should be mentioned in the methods part of this) and not only is it expensive to train, it is expensive to run and evaluate, so really, beware 3D fanboys.

## FP reduction

## Overall
We are not doing ensembling, which as the LUNA paper demonstrates, it improves the performances of those systems by a huge margin. As we've seen, variations on our approaches offer only incremental improvements, but they are not very good in terms of diversity. It would be interesting to actually investigate further if the different FP reduction approaches could be combined to offer better performance. So far it is a bit of an unknown quantity that angle.

Basically as we can see with the results, the nodule detection and segmentation part is very competitive. Basically it shows all the work I've performed, with the variations and filters I've applied. It is also quite novel the fact that the architecture takes whole slices, instead of using patches on the slice, which is more typical and also easier to train. Also very important, how crucial it is to introduce normalization and augmentation to train systems. It is absolutely necessary and it makes a world of difference. Also, performance is key. If you don't have a performing system, you're dead in the water, you can't train fast and if you don't, you just don't have the feedback to develop a system such as this. Lots of trial and error, which is a bit disheartening when trying to explain why the system works as it does.
