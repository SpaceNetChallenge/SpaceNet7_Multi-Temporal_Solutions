# Marathon Match - Solution Description

## Overview
Congrats on winning this marathon match. As part of your final submission and in order to receive payment for this marathon match, please complete the following document.


## 1. Introduction
Tell us a bit about yourself, and why you have decided to participate in the contest.

- Name: Motoki Kimura
- Handle: motokimura
- Placement you achieved in the MM:
- About you: A research engineer working on computer vision and machine learning. I'm especially interested in robotics and remote sensing imagery.
- Why you participated in the MM: The uniqueness of SpaceNet-7 challenge made me to participate this competition. SpaceNet-7 is the first competition that provides large amount of satellite image time-series. Also I've been curious how useful Planet's Dove constellation imagery (with 4 meter resolution) would be for urban development analysis.

## 2. Solution Development 
How did you solve the problem? What approaches did you try and what choices did you make, and why? Also, what alternative approaches did you consider?

- I started by merging [my solution for SpaceNet-6 challenge](https://github.com/SpaceNetChallenge/SpaceNet_SAR_Buildings_Solutions/tree/master/4-motokimura) with the building tracking algorithm used in [the baseline algorithm provided by CosmiQ Works](https://github.com/CosmiQ/CosmiQ_SN7_Baseline).
- For the first 2 weeks, I tried to make improvements in building detection algorithm I used for the SpaceNet-6 challenge that combined U-Net with EfficientNet-B7/B8 encoders and watershed postprocessing. Unlike the SpaceNet-6, I found that bigger encoders like EfficientNet-B7/B8 did not improve the detection score. I also found that enlarging images before input to the network greatly improved the detection result especially for small buildings. Considering GPU memory consumption and inference time, smaller encoders are more advantageous to input larger images. So I decided to use EfficientNet-B3 as the U-Net encoder and tried to find the best hyper parameters for this model.
- During the last 2 weeks before the submission deadline, I was working hard to improve the tracking algorithm. [The visualizer provided by the competition host](https://drive.google.com/file/d/1ejXfv-UDAf_vnQU8-LQk12oUyT4A3cay/view) was really helpful. Thanks to this tool, I was able to figure out what made the tracking score worse: falsely detected buildings (false positives) significantly worsened the change detection score. I have implemented several techniques to reduce these false positives, and this actually led to a big improvement in the tracking score.

Things I have tried but did not help:
- Bigger EfficientNet encoders, i.e., EfficientNet-B4, B5, etc.
- Data augmentations: random horizontal flipping, random vertical flipping, random rotation, random brightness, and random blur.
- Test time augmentation (TTA): horizontal flipping, vertical flipping, and resizing.
- Cosine annealing learning rate schedule.
- Use scSE attention module in the U-Net decoder.
- Multi frame input: Concatenate 3 consecutive frames and input it to the model as an image with 9 channels.

## 3. Final Approach
Please provide a bulleted description of your final approach. What ideas/decisions/features have been found to be the most important for your solution performance:

My solution can be divided into 2 parts: building detection step and building tracking step.
In the detection step, building polygons are detected from each of the SpaceNet-7 RGB images.
In the tracking step, unique id is assigned to each of the detected buildings.
Both steps process each AOI independently.

The details of each step are described below:

### 3.1 Building Detection Step

![](media/detection_pipeline_01.png)

- I trained U-Net whose input is RGB imagery and which outputs mask for 3 classes: building body, edge, and contact.
- The U-Net models were trained on 5-folds. I split 60 training AOIs into 5 groups randomly.
- On each fold, I trained 2 U-Net models with 2 different image scales: 3x and 4x.
- All U-Net models used EfficientNet-B3 (pre-trained on ImageNet) as the encoder.
- Loss function was defined as the sum of BCE and dice loss, and Adam was used as the optimizer.
- All U-Net models were trained for 300 epochs. The learning rate started from 1e-4 and was dropped to 1e-5 at the epoch of 270.
- Pixel IoU score for the building body class was evaluated every 3 epochs and was used to choose the best model.
- As for input data in training, I randomly cropped 160 pixel x 160 pixel region in the original scale, resized it by the factor of 3x or 4x by bilinear interpolation, and normalized it using ImageNet mean and std values.
- As for input data in inference, I used whole the image (1024 pixel x 1024 pixel), resized it by the factor of 3x or 4x by bilinear interpolation, and normalized it using ImageNet mean and std values.
- Batch size was set to 8 for training and 1 for inference. Each U-Net was trained with one Tesla V100 GPU.
- I did not use data augmentation except for random cropping, and no TTA was used.
- N_fold x N_scale = 5 x 2 = 10 U-Net models in total were used for average ensembling to get more accurate mask.
- After ensembling, for each AOI, I aggregated mean mask for all frames in the AOI. Then, the mask at t-th frame in the AOI was weighted averaged with this mean mask: `mask_refined_t = w * mask_mean + (1 - w) * mask_t`. The weight `w` depends on the class: 0.25 for building body class and 1.0 for building edge and contact classes. Thanks to this mask refinement using time-series, false positives were significantly reduced.
- Finally, building mask was computed from the refined mask as `mask_body * (1 - 0.5 * mask_edge) * (1 - mask_contact)` and was input to watershed algorithm in order to extract building footprints as polygons.

Especially important were:
- **Correct train-val split:** I split the dataset by AOI. Random splitting without considering AOIs causes leakage and makes it difficult to evaluate the model correctly.
- **Multi class prediction:** Predicting building edge and contact helped to separate neighboring buildings.
- **Image resizing (3x or 4x):** before input to the network, images were enlarged by the factor of 3x or 4x. This helps to detect small buildings and to separate neighboring ones.
- **Mask refinement using time-series:** I aggregated mean mask for all frames in the AOI, and used this to refine the mask for each frame in the AOI. This refinement significantly reduced false positives.
- **Watershed:** watershed algorithm worked much better than a simpler alternative used in [the baseline algorithm provided by CosmiQ Works](https://github.com/CosmiQ/CosmiQ_SN7_Baseline).

### 3.2 Building Tracking Step

- I started from the tracking algorithm used in [the baseline algorithm provided by CosmiQ Works](https://github.com/CosmiQ/CosmiQ_SN7_Baseline). Following 3 improvements were added to reduce false positives for new buildings. These significantly improved the change detection precision.
- When assigning a new id to a new building candidate, if there is no footprint with an IoU greater than 0.25 in any of the subsequent 3 frames, the candidate is removed as a false positive.
- When assigning a new id to a new building candidate, if there is any footprint that has been detected in the previous frames and has overlap with the candidate, the candidate is removed as a false positive.
- When assigning a new id to a new building candidate, if there are 5 or more footprints that have been detected in the previous frames within 12 pixels around with the candidate, the candidate is removed as a false positive.
- Additionally, I added an improvement to increase the tracking recall: for each id, extract the first frame and the last frame at which the id appears, and if the corresponding id does not exist in the intervening frames, insert the footprint for that id to those frames.

All those mentioned above were important to win. Each of them increased the public score by more than 1 point!

## 4. Open Source Resources, Frameworks and Libraries
Please specify the name of the open source resource along with a URL to where it’s housed and it’s license type:

See [licenses/README.md](licenses/README.md).

## 5. Potential Algorithm Improvements
Please specify any potential improvements that can be made to the algorithm:

- Pre-training U-Net models on larger datasets in satellite image domain (e.g., datasets used in previous SpaceNet challenges) may improve the result.
- Ensemble U-Net with instance segmentation models like Mask R-CNN.
- In tracking step, use more features to match buildings (e.g., appearance, shape, etc.) to reduce false matches.
- Learn building detection and tracking in an end-to-end manner.

## 6. Algorithm Limitations
Please specify any potential limitations with the algorithm:

- The algorithm does not work well in the scene where the small buildings are densely located. Some neighboring buildings tend to be detected as one learger building and this causes false tracking.
- Because of image resizing (3x or 4x), inference takes some time and consumes GPU memory a lot.

## 7. Deployment Guide
Please provide the exact steps required to build and deploy the code:

See [docs/FINAL_SCORING.md](docs/FINAL_SCORING.md).

## 8. Final Verification
Please provide instructions that explain how to train the algorithm and have it execute against sample data:

See [docs/FINAL_SCORING.md](docs/FINAL_SCORING.md).

## 9. Feedback 
Please provide feedback on the following - what worked, and what could have been done better or differently? 

- Problem Statement
    - I think it was well described and easy to understand.
    - The evaluation metric was also well studied and designed for the tracking task for remote sensing imagery.
- Data
    - Thank you for making this awesome dataset public!
    - It was well organized and documented so I had no difficulty to work around with it.
- Contest
    - Response from the organizers in the forum was a bit slow compared to the SpaceNet-6 challenge. It would be great if we could get the answer within 2 days at the most.
    - The baseline algorithm provided with Jupyter Notebook was easy to understand and I found it a good starting point (especially for the building tracking part).
- Scoring
    - Would it be possible to keep the evaluation server live after the end of the competition? SpaceNet-7 is a unique and great dataset and I’m sure benchmarking with this dataset is really helpful for many researchers in remote sensing or computer vision fields.
