# ReCowGnition: A Realistic Biometric Benchmark for Cow Face Recognition

This is the official repository of the "ReCowGnition" benchmark.

## Overview

The ReCowGnition benchmark is a realistic biometric benchmark of identity-labeled cow faces and aims to encourage future and comparable research in the field of Cow Face Recognition. The dataset is **free for non-commercial use** and can be requested here by **providing a full name and affiliation**: [Request Access](https://drive.google.com/drive/folders/1ywFj3_pew7uOG0FUnbDbkRiz6JPOpAaB?usp=sharing)

Along with the images of the dataset, we also provide additional tools, including *CowDetect*, a cow face and muzzle detection model which was used during the creation of the dataset, and *evaluation scripts* that performs the benchmark evaluation based on the defined protocols. The *CowDetect* model can be requested together with the dataset (see link above). 

The benchmark recognition models utilized in the paper are **not** provided, however, for questions or interest in cooperation in this regard, please email: marco.huber@igd.fraunhofer.de

The associated paper was accepted at the *Sustainable Pattern Recognition & Pattern Recognition for Environment* Workshop at the *International Conference on Pattern Recognition (ICPR) 2026*. 

The paper which provides more information can be found here: [To-be-added-after-final-publication].

The rest of this readme is structured as follows:
- [Installation](#installation)
- [Usage](#usage)
- [Dataset & Detection Details](#dataset-&-detection-details)
- [Evaluation Protocols](#evaluation-protocols)
- [Benchmark Results](#benchmark-results)
- [References](#references)
- [Citation](#citation)
- [License](#license)


## Installation

1. Clone this repository & prepare python and virtual environments (recommended)
2. Install dependencies
> pip install -r requirements.txt

## Usage

### Cow Face Detection
The pre-trained *CowDetect* model can be used for a single video with:
> python CowDetection.py --mode [mode] --input [input-path] --output [output-path] --model [model-path] 

e.g., 

> python CowDetection.py --mode "detect_video" --input "./input_vid/GX014028_47.mp4" --output "./output" --model "./model/CowDetect.pt"

This code extract for a single video, GX014028_47.mp4, each detected cow face (without alignment or scaling). Further available parameters are:
- \-\-thr_c: cow face confidence threshold (default: 0.5)
- \-\-thr_m: muzzle confidence threhsold (default: 0.2)
- \-\-frame_step: frame extraction interval (default: 1, for the dataset 5 was used)
- \-\-rotation: rotation mode for frames: '90_clockwise', '90_counterclockwise', '180', '45' or 'None'.

To extract cow faces in images (without alignment or scaling), the pre-trained *CowDetect* can be used with, e.g.,
> python CowDetection.py --mode "detect_images" --input "./input_img/" --output "./output" -- model "./model/CowDetect.pt"

The available additional parameter are: 
- \-\-thr_c: cow face confidence threshold (default: 0.5)
- \-\-thr_m: muzzle confidence threhsold (default: 0.2)

To generate a dataset based on a set of videos, the "create_dataset" mode can be used, e.g.,:
> python CowDetection.py --mode "create_dataset" --input "./input_db/" --output "./output" -- model "./model/CowDetect.pt"

This generates based on the videos a similar image dataset as the provided *ReCowGnition* dataset. The output consists of extracted, aligned and scaled cow faces. To maintain consists filenames that include the identity, the provided input videos filenames should follow the structure of the provided image filenames.

The available additional parameters are, similar to the video extraction: 
- \-\-thr_c: cow face confidence threshold (default: 0.5)
- \-\-thr_m: muzzle confidence threhsold (default: 0.2)
- \-\-frame_step: frame extraction interval (default: 1, for the dataset 5 was used)
- \-\-rotation: rotation mode for frames: '90_clockwise', '90_counterclockwise', '180', '45' or 'None'.


## Dataset & Detection Details

### Initial Video Acquisition
The original videos were recorded autonomously at a dairy farm in Germany directly after the milking process of the cows in five recording sessions on three different days using a GoPro Hero11 Black.  The camera was mounted at the exit of a walkway that the animals had to pass through after milking. The time of recording and the position of the camera varied for each session to increase variability and reflect different camera positions. In total, 287 videos were recorded of 116 different cows. Due to privacy reasons, as working farmers can be seen in the background, the videos will not be released to the public.

The five recording sessions are named: 
- GX024040
- GX024041
- GX014041
- GX014040
- GX014028

### Frame Extraction
To obtain a reasonable amount of cow images for testing and to ensure some variation between the images, every 5th frame has been extracted from the recorded video streams. After the frame extraction, some of the images have been rotated based on the recording angle to normalize the head position for the cow detection system.

### Cow Detection System
To detect and align the cow face images, we developed and provide *CowDetect*. *CowDetect* is based on a pre-trained YOLO-v11n [1] and has been trained to detect cow face images and cow muzzles. The detected muzzle is used to align the cow face images by ensuring that the detected muzzles always appear at the same position. This is done as alignment has been proven as a simple technique to increase face recognition performance. *CowDetect* was fine-tuned on a subset (6,375 cow images) of the dataset of Yao et al. [2] and the CSCE873CV dataset [3] images .

### Image Preprocessing & Alignment
After the detection and alignment of the cow faces, the images have been scaled and cropped to the size of 112x112 pixels, following the standard image size for human faces in face recognition. Afterwards, the images have been manually scanned and misdetected images have been removed. 

The final benchmark consists of 6,838 different images of 161 individual cows. The average number of images per individual cow is 42 images. 

The filenames follow the structure:

> **[recording\_session]\_[cowID]\_[a]\_[b].jpg**

where [recording\_session] referst to one of the five session IDs mentioned above, [cowID] refers to a unique numerical identity label of each individual cow, and [a] and [b] are arbitrary numbers added during the preprocessing pipeline to obtain unique filenames.

### The Cows
All of the recorded cows are Holstein Friesian cows and some of them are related by blood, as the majority of cows have been bred at the farm. The heritage lines are not tracked in the dataset and just mentioned as a potential additional challenge of the dataset. All the cows are used for milk production and are therefore (obviously) female. The age range of the cows covers 2 to 7 years. The recorded cows were identity labeled by an employee of the farm with an identity based on the records of the farm and identity markers not visible in the videos. An additional challenge present in this dataset is the greater degree of pitch freedom cows have enabled by their long necks. This allows the cows to tilt their head much further than humans typically do, resulting in facial views from steep top-down or bottom-up angles. 

## Evaluation Protocols
In contrast to most existing work, the benchmark assumes a biometric approach to cattle recognition, similar to human face recognition. This involves a feature extraction and comparison pipeline during evaluation rather than a classification approach. Approaching (human or cow) face recognition as a classification problem has the disadvantage that it requires several instances of the respective class (e.g., an individual cow) and the classes (e.g. all cows to be classified) have to be fixed in advance before the classification model is trained. If a new cow is added to the farm, the model has to be updated as unknown classes appear and not updating the model would lead to false classifications. This approach is impractical as, in reality, a cow face recognition system should be able to handle large numbers of changing individuals. Cows might, on a regular basis, be born, die, be bought or be sold. 

In the feature extraction approach a recognition model is trained and then used to extract mathematical feature representations (embeddings) that describe the identity features of the individual. These extracted embeddings are then utilized for comparison between different embeddings to conclude based on the similarity and some system threshold if the embedded identity in both images used for comparison is the same or not. In contrast to the classification approach, this approach does not need any data of the individual during training and can also be applied to other cattle populations without re-training.

For the benchmark, we differentiate between verification protocols (1:1) and identification protocols (1:n). 

###  Verification Protocols
Biometric verification refers to the 1:1 comparison of a reference and a probe image of an individual that answers the question: *Is this the same individual?* In cow face recognition, a possible practical scenario would be a veterinarian who verifies that the presented cow is the one that needs to be vaccinated.

The *verification protocols* cover two evaluation scenarios: $V_{all}$ and $V_{CS}$. Both consist of all possible impostor pairs (different cow image pairs, in total 23,105,619 pairs), but differ in size of genuine (same cow image pairs). In the $V_{all}$ scenario, all possible genuine pairs are utilized (in total 270,084 pairs), while in the $V_{CS}$ (cross-session) all pairs from the same recording sessions are removed, leaving only genuine pairs from different recording sessions (in total 90,063 pairs). $V_{CS}$ provides a more realistic and harder evaluation as it does not allow the model to benefit from possible same-session similarities. 

To evaluate the performance, the benchmark follows the standard [4] and requires reporting the **Equal Error Rate (EER)** and the **False Non-Match Rate (FNMR) at a certain False Match Rate (FMR)**. The EER is the system's performance at the decision threshold where FNMR=FMR. In the ReCowGnition verification protocol, the FNMR@FMR=10% and FNMR@FMR=1% are reported. Additionally, for a graphical, threshold-independent evaluation, the **Receiver Operating Characteristic (ROC) curve based on FMR and 1-FNMR** is provided. 


### Identification Protocols
Biometric identification refers to the 1:many comparisons of a probe image to a whole set of reference images with the goal of answering the question: *Which individual is it?* In cow face recognition, a possible practical scenario would be to identify a cow that has just been milked to connect her with the amount of milk she produced.

The *identification protocols* cover four evaluation scenarios: $I_{ALL}$, $I_{CS}$, $I_{EF}$, and $I_{SF}$. In the $I_{ALL}$ setup, each image is used as a probe (as long as it has a possible correct identification), and all other images are used as gallery images. In the $I_{CS}$ (cross-session) setup, same-session images are removed from the gallery based on the probe image to provide a harder, more realistic scenario. More complex, "video-based" evaluation setups are provided with $I_{EF}$ (embedding fusion), and $I_{SF}$ (score fusion). In the $I_{EF}$ scenario, all embeddings from one video are grouped and the mean embedding is calculated as a single representation of the individual shown in the video. This is done for the probe and the gallery. In the $I_{SF}$ scenario, all embeddings from one video are also grouped and the pairwise similarities between the probe group and all gallery groups are calculated, and the mean similarity score is used for identification.

To evaluate the identification performance, the **Top-1 accuracy** and **Top-5 accuracy** are reported. Top-1 accuracy measures the percentage of query images for which the highest-scoring identity prediction matches the correct identity. Top-5 accuracy measures the percentage of query images for which the correct identity appears among the five highest-scoring identity predictions. For a graphical and top-k independent evaluation, the **Cumulative Match Characteristic (CMC) curve** is reported. The CMC curve reports the probability that the correct identity appears within the top-k ranked matches, as a function of the rank k. 

## Benchmark Results

The associated research paper evaluated six baseline benchmark models, covering three different approaches. These are: 
- $ArcFace$ [5] (ResNet-100 trained from scratch with ArcFace loss only on cow images) 
- $ElasticFace-Arc$ [6] (ResNet-100 trained from scratch with ElasticFace-Arc loss only on cow images) 
- $ArcFace_{FT}$ (human-based ResNet-100 model fine-tuned with ArcFace loss on cow images) 
- $ElasticFace-Arc_{FT}$ (human-based ResNet-100 model fine-tuned with ElasticFace-Arc loss on cow images) 
- ViT-B/16 [7] (zero-shot foundation model)
- ViT-L/14 [7] (zero-shot foundation model)
 
The benchmark results regarding the **verification scenarios** of the baseline models on the ReCowGnition benchmark are provided in the table below (for a proper discussion see the associated paper). 

| Model | $V_{ALL}$ EER | $V_{ALL}$ FNMR@10% | $V_{ALL}$ FNMR@1% | $V_{CS}$ EER | $V_{CS}$ FNMR@10% | $V_{CS}$ FNMR@1% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ArcFace** | 0.254 | 0.518 | 0.818 | 0.305 | 0.711 | 0.957 |
| **ElasticFace-Arc** | 0.243 | 0.478 | 0.792 | 0.311 | 0.694 | 0.936 |
| **$ArcFace_{FT}$** | 0.137 | 0.194 | 0.585 | 0.163 | 0.284 | 0.742 |
| **$ElasticFace-Arc_{FT}$** | 0.129 | 0.172 | 0.533 | 0.154 | 0.260 | 0.713 |
| **ViT-B/16** | 0.364 | 0.626 | 0.842 | 0.479 | 0.873 | 0.985 |
| **ViT-L/14** | 0.367 | 0.650 | 0.850 | 0.465 | 0.867 | 0.985 |


The benchmark results regarding the **identification scenarios** of the baseline models on the ReCowGnition benchmark are provided in the table below (for a proper discussion see the associated paper). 
| Model | $I_{ALL}$ Top-1 (%) | $I_{ALL}$ Top-5 (%) | $I_{CS}$ Top-1 (%) | $I_{CS}$ Top-5 (%) | $I_{EF}$ Top-1 (%) | $I_{EF}$ Top-5 (%) | $I_{SF}$ Top-1 (%) | $I_{SF}$ Top-5 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ArcFace** | 82.76 | 94.09 | 9.30 | 18.15 | 8.10 | 20.95 | 5.71 | 19.05 |
| **ElasticFace-Arc** | 81.41 | 93.05 | 8.53 | 17.95 | 9.05 | 18.57 | 5.71 | 20.95 |
| **$ArcFace_{FT}$** | 95.58 | 98.87 | 30.28 | 52.58 | 34.29 | 62.38 | 30.48 | 55.71 |
| **$ElasticFace-Arc_{FT}$** | 96.39 | 99.20 | 30.30 | 53.54 | 32.86 | 61.43 | 32.86 | 60.95 |
| **ViT-B/16** | 69.23 | 86.18 | 2.29 | 8.05 | 3.33 | 8.10 | 1.43 | 6.67 |
| **ViT-L/14** | 65.82 | 83.53 | 2.15 | 7.24 | 1.90 | 8.57 | 0.48 | 4.76 |


## References

[1] Glenn Jocher, Jing Qiu, Ultralytics YOLO11, 2024: https://github.com/ultralytics/ultralytics

[2] Liyao Yao, Zexi Hu, Caixing Liu, Yingjie Kuang, Yuefang Gao: Cow Face Detection and Recognition Based on Automatic Feature Extraction Algorithm. In: ACM TURC '19: Proceedings of the ACM Turing Celebration Conference - China

[3] CSCE873CV Dataset - Open Source Dataset, Cattle Detection, 2025: https://universe.roboflow.com/cattledetection-dn9uy/csce873cv-pd9an

[4] ISO/IEC 19795-1:2021: Information technology - Biometric performance testing and reporting - Part1: Principles and framework (2021)

[5] Jiankang Deng, Jia Guo, Jing Yang, Niannan Xue, Irene Kotsia, Stefanos Zafeiriou: ArcFace: Additive Angular Margin Loss for Deep Face Recognition. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019

[6] Fadi Boutros, Naser Damer, Florian Kirchbuchner, Arjan Kuijper: ElasticFace: Elastic Margin Loss for Deep Face Recognition. In: IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) - Workshops, 2022

[7] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever: Learning Transferable Visual Models From Natural Language Supervision. In: Proceedings of the 38th International Conference on Machine Learning (ICML), 2021

## Citation

To be added upon final publication.

## License

> This project is licensed under the terms of the Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license. 
Copyright (c) 2026 Fraunhofer Institute for Computer Graphics Research IGD 
