- Abstract:
    - A very rough outline of the topic, objectives of the thesis and outcomes.
- Introduction:
    - Lung cancer
    - Computed Tomography
    - Lung cancer screening with CT
        - Talk about the NLST study and NELSON. Reduction of 20% in mortality if screened, so early detection is important to improve the outcomes.
    - Lung nodule CAD
        - Explain why it would be useful (reduce workload, reduce intra-variability for radiologists). Also cheaper. Explain why historically they haven't worked (mention main problems a system like this faces) and why I think now is a good time to create a system that improves upon the existing state of the art.
    - Deep learning in medical imaging
        - A brief introduction on deep learning in computer vision, how it started and how it is currently being used to
    - Lung cancer detection challenges
        - Talk about the ongoing competitions about lung cancer, based on the LIDC study (most of them). Talk about LUNA, Kaggle, ANODE09, the ISBI. Also the different variants, with just detection, inferring malignancy, etc. I wouldn't expand too much apart from this and just create a chapter to really drill into what they use and why it works.
    - Metrics
        - Small section to introduce the metrics I'll use and what are they used for and what drawbacks they have:
            - DICE
            - FROC
            - Average FROC
            - AUC
            - TP, FP, sensitivity and F1
    - Pipeline:
        - Describe the pipeline for a CAD system. The 3 problems we are facing and how they stack together. Also mention how we are scoring each of this problems individually.
    - Outline
        - Talk about the chapters, and how the work is organized.
- A review of the state of the art:
    - The idea of this chapter is to work as a survey of methods. Take inspiration from other literature reviews, such as LUNA's, or some other generic literature reviews on ML.
    - Intro: Review methods and results of challengers in Kaggle and LUNA
    - Methodology: Go over LUNA paper, LUNA top scores, Kaggle and Kaggle publications. Break down the methods and how they performed
    - Results: List grid of methods for each part of the detection + FP reduction.
    - Conclusions: Basically, from here what I want is to choose a bunch of techniques which might help me determine what is going on with the nodule thing. I don't know, but maybe it would be worth to create a section to talk about the technique, network layers and architectures in this chapter. Maybe as another point.
- Lung segmentation:
    - Intro: Lung segmentation is the first of the challenges. The idea is we want to reduce the area where the neural network will have to focus on, and thus reduce the possibility of false positives by performing lung segmentation. Maybe mention as well different approaches, and hopefully search exactly how the LUNA ones were obtained (according to Mario it's a method that follows the state of the art).
    - Methodology: Taking the slices and the masks, I've trained a UNET for 40 (I think epochs). No augmentation whatsoever, no rebalancing of classes, no anything of the sort.
    - Results: Compare DICE scores per slice. Put some graphs with the mask overlays. Put the data about excess nodule mask on the original masks and the new ones. Hopefully they don't differ too much.
    - Conclusions: Hopefully the results already match what you can attain by using other (non deep learning) mehods, without too much work. Lung edges require more work, but this should be doable doing a few more epochs focusing solely on those slices. Really, this would be very cool to do. Collect the batches with the worst DICE scores and then train another 20 epochs or so, half good, half bad. And then see again how it compares with the original score.
- Nodule segmentation:
    - Intro: Second step to our pipeline. This is basically where we obtain a list of candidates that we will later use. Also to note, performing a good segmentation has the benefit of potentially being useful to do FP reduction. Why is that? Because we can obtain useful features from the segmentation that will help us classify between nodules and non-nodules. Mention maybe the segmentations and readings of the LIDC dataset. Still, this is an area which should focus on sensitivity.
    - Methodology: Using a UNET, trained only on slices with known nodules. Those are the only layers we've used for training. Then the preprocessing has been performed in multiple different ways, and the loss functions have also been different. Basically here expose all the variants of the networks I am going to evaluate. Talk also about the preprocessing, how the data is stored, the amount of data, the multiprocessing I've had to do, etc.
    - Results: Put the training loss over time for training / validation. Put a table ranking the different methods (score the 3 subsets of data separately). I'm interested in DICE, sensitivity and FP. Maybe mention that DICE for this, on the overall scan, is not really so good, since DICE is gonna be 1 on the scans with nothing, so most of the slices will have a good value, and since conflictive slices are few, you get a system that scores very high even though is shit. Maybe F1 score as an overall number, although it's tricky since how do you relly weight how important is your sensitivity vs FP
    - Conclusions: Probably talk about the overfitting and how some augmentation is not really that helpful. Interesting topics to cover as well: usage of different loss functions, batch normalization, how well do augmentation variants help to detect complicated nodules, such as subsolid ones, how well does it work with nodules on the edges of the pulmonary lobes. Also how do different masks compare? Is it worth it to go with spherical masks, or does it learn better by using cubes, since the features are automatically learnt by the network.
    - Discussion: I can talk a bit about the preprocessing here
- FP reduction:
    - Intro: 3rd step to the pipeline. Basically we have a bunch of ideas based on what we've learned from the state of the art and the segmentation.
    - Methodology: From nodule annotations, create blocks of 32x32x32 and predict them with the UNET. Once predicted and I have the mask + features, train a bunch of classifiers. Pich best and see how it goes. 2nd is to do the same, but features will be learned with radiomics. Then see how it goes. 3rd is to basically do this with a ResNet (so yet another network) and see how it goes. For the resnet and the others, a bit of mislabeling might be a good idea to reduce overfitting, if I find it happening.
    - Results: Check accuracy. Do confusion matrix. Check accuracy per class (nodule / non-nodule).
    - Discussion: I'll think more about this when I work on it more
- Integration of the system as a pipeline:
    - Intro: So far we have been working on individual parts of a CAD, but they need to be integrated together if we want this to be of use for clinicians. Deploying software that can be used and that will take care of the differences in scans, spacings and formats, plus creating something which is at the same time easy to deploy is not an easy task.
    - Methodology: System that takes care of the preprocessing of a CT file (making it isotropic), applies lung segmentation, then nodule segmentation, then FP reduction, then retrieve candidates as a csv with real world coordinates in 1 single step. All of this dockerized, so no dependencies are needed and is easy to deploy.
    - Results: Probably timing to setup and time to run, with and without GPU support (which is totally needed BTW)
    - Conclusions: It's more work, but at least makes all of these results way useful. Also easier to test out other datasets if you have a real end-to-end pipeline such as this.
- Evaluating CAD on LUNA:
    - Intro: Basically it's time to take our pipeline and evaluate it with FROC with LUNA
    - Methodology: Taking the best of lung segmentation, with and without automated lung masks and maybe a few different FP systems, test out the LUNA score and check what would be my ranking compared to the rest of the systems. On FP reduction I could already present a score comparing myself with LUNA, but not on the nodule detection track.
    - Results: Basically FROC, mean FROC and AUC.
- Evaluating CAD on EURECAT dataset:
    - Intro: So basically same thing we've done, but with another dataset completely different. This is basically so we can check how well the features transfer to another dataset of images. Does it really generalize?
    - Methodology: Same as before
    - Results: FROC, mean FROC and AUC. Compare it with the results on the previous section. Maybe these 2 chapters could actually be merged together. It would make enough sense actually.
- Summary:
    - So similar to the thesis outline but putting more emphasis on the conclusions of the research I've performed based on the results.
- General discussion:
    - Talk about the results, what went wrong and what are the next windows of opportunity. Probably talk about malignancy, followups and also other datasets (such as PET info) which could be used to expand 

Steps:
    - Nodule segmentation (UNET 2D, or maybe 3D if I have the time, but I have to change the pipeline)
    - Lung segmentation (registration and ATLAS)
    - FP reduction (with my feature model and also another 3D network)

I should probably go in depth with the few architectures that work really well. UNET is a must. ResNet, batch normalization, DCNN. Also image registration.

    - What about preprocessing and augmentation? What about ADAM and loss functions? What about how the loss function relates to the ultimate number we want to optimize (sensitivity vs FP, or F1 score).
    - Describe the pipeline I have developed. It would be interesting to sort the variants of the networks I have and then plot the results against each other. Like no augmentation, different batch size, with batch normalization, without batch normalization, see how much better it converges or it doesn't. See if I can apply some other normalization techniques, since the image is big and batches are quite small.
    - It may be also worth it to describe the hardware I'm running this on and the setup so that it can be executed in multiple PCs without too much trouble (cluster as well).

Do the same thing that I've done for the False Positive reduction track with whatever methods I come up with (basically, I have to explain the methodology I follow, on how I break down the features), explain the training mechanism. Then show different results for different algorithms (logistic, adaboost, other random forest or classifiers, etc.). Also try to graph which of the features are most useful.

Finally I would have to graph the results of both systems together. How does that perform in LUNA. How does that perform in Kaggle?

Last point is how do I transform this into a product? Talk about the dockerization, how the inputs are transformed on the fly, how lungs are segmented on the fly? (which by the way, It would be nice to see how much better or worse it is to segment or not the goddamn lungs)

