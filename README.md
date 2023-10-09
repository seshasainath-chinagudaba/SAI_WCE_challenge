# SAI_WCE_challenge
## Introduction
This repository implements an ensemble learning approach for detecting the bleeding portion in Wireless Capsule Endoscopy (WCE) images. Leveraging deep learning models, especially the variants of ResNet architecture, our model aims to enhance bleeding detection accuracy in medical imaging.

## Overview
The codebase encompasses data preprocessing, model architecture definition, training, validation, and testing. By employing an ensemble of deep learning models and utilizing techniques such as majority voting, max voting, and mean voting, our approach aims to provide robust predictions for identifying bleeding portion in WCE images.
Unleash the power of our bleeding detection model designed for Wireless Capsule Endoscopy (WCE) images. Seamlessly integrated with an ensemble of neural networks, our model accurately identifies bleeding, providing critical insights for medical practitioners. Simply follow our easy-to-use steps and dive into a new era of efficient diagnosis and analysis.

# Results
## Classification
Evaluation Metric | Value
|------------------|------------|
Accuracy   |0.7745|
Recall     |0.6560|
F1-Score   |0.7354|
## Detection
Evaluation Metric | Value
|------------------|------------|
mAP@0.5   |0.723|
mAP@0.5-0.95 |0.482|
avg.precision  |0.7614|
avg.recall  |0.6562|
iou  |0.4505|

## 2.Top 10 Validation Dataset Images: Classification & Detection with Confidence-Marked Bounding Boxes

Following are the top 10 images from the validation dataset which showcase our model's classification and detection capabilities. Each image is annotated with confidence-marked bounding boxes, highlighting precise object localizations.


![img- (52)](https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/assets/61778966/dd9c3278-4e19-4b40-bc88-b574aa6c1180)
![img- (50)](https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/assets/61778966/0d6dfa20-74c1-4a23-be03-1fa694099537)
![img- (48)](https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/assets/61778966/73987d78-6fa8-4531-a676-f58e2ce0c7fa)
![img- (47)](https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/assets/61778966/56211687-8454-41a4-90c1-7bf9023c5e77)
![img- (43)](https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/assets/61778966/2785ef55-5ef5-421c-8b56-d0166559cbb4)
![img- (42)](https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/assets/61778966/91ed6b71-7e09-4b57-b453-ae7581da1476)
![img- (35)](https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/assets/61778966/c5077a91-8965-415c-85ad-7a550c293ce9)
![img- (34)](https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/assets/61778966/d653ec43-326e-4372-8937-81b14eb24eef)
![img- (24)](https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/assets/61778966/7a9054e1-c554-4e47-a5cf-250f64997d8e)
![img- (1)](https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/assets/61778966/db6d836e-7d6e-4908-9236-f4033c57706e)

 Following is the link provided for the folder containing the above images: 
https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/tree/ad20bf56223a40b200c4ea1096d0b06710482200/images/validation_best_10

## 3. Visualizing Interpretability: Insights from Top 10 Images in Validation Dataset
![img- (52)](https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/assets/61778966/7844fc52-1af9-4626-acc4-43f2b4cdfe83)
![img- (50)](https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/assets/61778966/df394dd8-7920-4cab-9a38-2dab282bade0)
![img- (48)](https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/assets/61778966/dbf51b23-23c8-484f-b76a-6ff603648a89)
![img- (43)](https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/assets/61778966/45987663-6b64-4b28-b570-d2c0874c8dd6)
![img- (47)](https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/assets/61778966/0fafd97e-fc73-4449-9e96-b12f5ffe7ea4)
![img- (42)](https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/assets/61778966/c86c3f5c-d08c-4ed0-aeb1-e696508f731b)
![img- (35)](https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/assets/61778966/d7b76e84-8183-40b1-a1bf-eb54a3690d75)
![img- (34)](https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/assets/61778966/1877b011-eed8-41fa-9b5c-41a38a54665b)
![img- (24)](https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/assets/61778966/8870bd47-49fb-476b-ba01-1aa514fc7a69)
![img- (1)](https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/assets/61778966/7086633f-ea4d-4970-a395-8ab08d7d5dd2)

Following is the link provided for the folder containing the above image:
https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/tree/a1c3972fd25423f7521a4716b025cdbdd33e1326/images/validation_interpretability
## 4. Top 5 images for each of the test datasets :

### Test_data_set_1
![A0046](https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/assets/61778966/2f9327c8-cc2d-482d-a228-1d31d223ff5f)
![A0038](https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/assets/61778966/9a5526b0-43e5-45dc-b4a5-c3872c79a53a)
![A0034](https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/assets/61778966/6604cea6-92d6-4e0b-90f7-3b18e76eaf50)
![A0027](https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/assets/61778966/cafce554-13ab-4950-8822-8e098a640fed)
![A0007](https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/assets/61778966/1f048cce-0d1b-42e7-a606-225119c30782)

 Following is the link provided for the folder containing the above images: 
https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/tree/ad20bf56223a40b200c4ea1096d0b06710482200/images/Test_data_set_1
 
### Test_data_set_2
![A0446](https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/assets/61778966/c6c81587-c1ee-4226-bb32-504fd77e0acf)
![A0244](https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/assets/61778966/29d2cd5d-8570-463b-b04a-c3e545329966)
![A0177](https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/assets/61778966/b382b559-242c-4408-9806-01eee2271b7b)
![A0161](https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/assets/61778966/0dd5d9d6-86ba-4ba8-a23f-bf8b4580ea8a)
![A0071](https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/assets/61778966/43a8899e-d89d-44d5-80c1-6a2ed595c51b)

 Following is the link provided for the folder containing the above images: 
https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/tree/ad20bf56223a40b200c4ea1096d0b06710482200/images/Test_data_set_2

## 5.Interpretability for top 5 images for each of the test datasets:

### Test_data_set_2

![A0446](https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/assets/61778966/76fb257f-74bf-4fbf-b753-6ba7e8e01cb1)
![A0244](https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/assets/61778966/fd5e99a5-94ab-48f0-9ad9-07da95f9be88)
![A0177](https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/assets/61778966/120381c5-0fec-4f4e-b326-1962c44231ab)
![A0161](https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/assets/61778966/fb70512a-394e-45f6-b385-3a534d2430f3)
![A0071](https://github.com/seshasainath-chinagudaba/SAI_WCE_challenge/assets/61778966/ee0cbe1e-75eb-46c7-8a77-692ba0994986)

Following is the link provided for the folder containing the above images:

