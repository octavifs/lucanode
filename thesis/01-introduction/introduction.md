# Introduction
\newpage

Some brief intro as to how I am planning to organize this thing. Let's try to add a citation @Jacobs2015. And I've added something else, let's see if the auto build system picks it up. Does it now? Maybe it is working?

## Clinical context

### Lung cancer
Lung cancer is the most deadly cancer in the world. Bring some figures and talk about trends. Explain why this was collected, essentially the whole reasoning behind it was to provide a good benchmark to easily compare CAD systems

### Computed Tomography
Basically talk about the technique and how it has been changing diagnosis recently.

### Lung cancer screening with CT
Talk about the NLST study and NELSON. Reduction of 20% in mortality if screened, so early detection is important to improve the outcomes.

### Lung nodules
Explain nodule types. Solid and subsolid.

## Lung nodule CAD
### Objectives
Explain why it would be useful (reduce workload, reduce intra-variability for radiologists). Also cheaper. Explain why historically they haven't worked (mention main problems a system like this faces) and why I think now is a good time to create a system that improves upon the existing state of the art.

### Shortcomings
Explain what are the main things that fail

### Pipeline
Even though there has been much effort in developing new techniques to improve the performance of CAD systems due to the availability of annotated datasets and challenges (NLST, ISBI, LUNA, DSB2017), the published systems tend to be brittle and very much focused on demonstrating good results on those specific challenges but useless as an integrated system. Also, what is not available tends to be proprietary systems, which might be good, but who knows really.

![the lucanode pipeline](lucanode_pipeline.png){ width=40% }

One of the improvements that I wanted to bring to the state of the art was to prepare a system which could be easily deployed in a real system. To achieve this I had to automate the scan preprocessing and prepare a full pipeline that could later on be integrated in a real system. In fact, this integration with a system has been performed by Albert <!--TODO put ref -->, that has a queue which picks up the scan and returns a CSV with the annotated nodules to check for.

What do we need to do:

- preprocessing: basically reading the ct scan in a SimpleITK compatible format and rescale it to 1x1x1mm
- lung segmentation: using the input from before, segment the lung and get a segmentation mask
- nodule segmentation: using the isotropic scan and the lung segmentation, compute the segmentation mask for the nodules, then measure the centroids of the labels and convert those coordinates to real world coordinates.
- fp reduction: Using the scan and the centroids in the previous step, apply the nodule classifier and retrieve a probability for each of the nodules. Once we have this probability per candidate, discard any that are below a required threshold. If instead of using a probability threshold what we are interested is in a false positive rate, use the numbers in the evaluation phase to basically determine how the probability maps to a specific FPR, and adjust the output candidates with that.

To run this basically I've packaed everything in a conda environment. This has allowed me to list all the necessary packages and provide an easy way to create environments with all the necessary dependencies, even stuff like CUDA libraries, which is not native python, can be easily installed using conda. This also makes it very easy to then create a Docker image that has all the necessary packages to run this stuff.

What else? Well, the docker image contains the weights of the different neural networks. I've basically just included the best network for each of the steps, based on the evaluation of the results. Both the code, dependencies and weights is included in a Docker image, which can also have GPU support (very much recommended) by using [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

Once that is built, we have a ready to go image, which only needs to mount 2 volumes (folders) for the input image and the output result. Then it's just a matter of running a command and all of this code can be easily run. Apart from the ease in reproducibility (not only the final script can be executed, but everything else, such as evaluation scripts and the like), we gain a very convenient way to distribute the results and an even better way to test our system in other datasets with minimum hassle, since the whole pipeline has been integrated.

Currently on an i7 7700, 32GB of RAM, GTX 1080Ti, evaluating a scan from start to finish requires around 2mins of processing time.

<!--TODO basically put some real numbers and also put an example of the comand and mention how big the image would be for all of this to work. Break down the results step by step as well (performance wise) -->

### Metrics
Small section to introduce the metrics I'll use and what are they used for and what drawbacks they have:

        - DICE
        - FROC
        - Average FROC
        - AUC
        - TP, FP, sensitivity and F1

### Lung segmentation
I might just put this after nodule segmentation and false positive reduction, since it basically just an addendum on nodule segmentation that needs to be done for the pipeline to work in an end to end fashin. Interestingly, this chapter could serve to demonstrate the transferability of deep learning techniques to other domains, which is not a bad thing. Essentially the network and everything is exactly the same thing as the nodule segmentation, but using the lung masks as ground truth, instead of nodule masks, so the problem is actually simpler.

Not much really. Basically the idea is that, if the previous network works well for something as complicated as segmenting nodules, segmenting the lungs themselves should be easier, but basically the same concepts should apply.

### Nodule detection
I could say that based on the work I did in the LUNA challenge chapter, best approach right now seems UNET based. Explain again that for this part of the system what we are interested in is basically something with very high sensitivity. And finally I guess say that I went for a 2D network cause the images are big, it is a very deep network, and I wanted to avoid as much technical trouble as possible, especially since it was a first for me.

### FP reduction
Similarly to an object detection problem (@Hosang2016), we've divided our pipeline in two phases: candidate proposal and false positive reduction. As we have seen in the previous chapter, our UNET-based proposal network primed sensitivity above all else, but now we need a classifier with high precision so that the signal-to-noise ratio of the system will be high enough to prove useful to a radiologist.

One of the main benefits of performing a previous step to detect candidates is the fact that the search space is reduced and that makes it computationally feasible to run image recognition algorithms with high computational costs within a reasonable timeframe.

In this chapter we'll cover two different approaches to false positive reduction. The first one will be a classifier trained on features manually extracted from the previous segmentation phase of the pipeline. The second one is based on a volumetric ResNet (@Chen2018). The original 2D version of this deep neural network (@Wu2017) achieved a deeper architecture bypassing the vanishing/exploding gradients problem (@Bengio1994, @Glorot) by using a combination of normalization techniques (@Ioffe2015, @LeCun2012, @He2014) and the use of residuals.

<!--TODO
    add some ref based on @Wu2017 that would properly explain the concept**)
-->

## State of the art
### The LUNA grand challenge
This chapter will serve as an introduction to what is the LUNA grand challenge, its dataset, competition tracks and metrics. After that is out of the way, I'll go over the current top 20 and do a survey of the different techniques that compound the state of the art for this kind of problem. This will serve as an introduction to what I amb about to do.

Basically talk about the technique and how it has been changing diagnosis recently. This could be a copy pasta of @Arindra2017 and explain a bit on how they've reworked on the LIDC dataset to prepare the data, what it does and what is missing (malignancy!), which is actually available in LIDC.

What is this dataset and why is it useful to evaluate CAD systems

Talk about the tracks and metrics. Again, this appears in @Arindra2017, so I don't know how much I want to add

Interesting to go over the top 20 of LUNA as it stands right now. Thankfully most of the systems are closed so I don't have to explain them, but for the open ones, it would be good to go over the methods they present, and basically argument why I chose what I did
Talk about the top 20. Basically put a table with the methods, describe them slightly. Then divide method by groups and expand more on that.

### Nodule detection track
Review of the top20. Cover here any deep learning content I might have to.

### FP reduction
Review of the top20 (again paper and all). Basically cover here any deep learning content I might have to.

## Techniques?
### Image registration
LUNA and other registration methods?
### Image segmentation
Talk about LUNA and radiomics?
### Image recognition
RESNET?

## Outline
Talk about the chapters, and how the work is organized.
