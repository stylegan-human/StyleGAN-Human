# SHHQ Dataset
<img src="../img/preview_samples1.png" width="96%" height="96%">

## Overview
SHHQ is a dataset with high-quality full-body human images in a resolution of 1024 × 512.
Since we need to follow a rigorous legal review in our institute, we can not release all of the data at once.

For now, SHHQ-1.0 with 40K images is released! More data will be released in the later versions.


## Data Sources
Images are collected in two main ways: 
1) From the Internet. 
We developed a crawler tool with an official API, mainly downloading images from Flickr, Unsplash, Pixabay and Pexels. Images were collected under the CC-BY-2.0 license.
2) From the data providers. 
We purchased images from databases of individual photographers, modeling agencies and other suppliers.
Images were reviewed by our legal team prior to purchase to ensure permission for use in research.

### Note: 
The composition of SHHQ-1.0: 

1) Images obtained from the above sources.
2) Processed 9991 DeepFashion images (retain only full body images).
3) 1940 African images from the InFashAI [[1]](#1) dataset to increase data diversity.

## Data License
We are aware of privacy concerns and seriously treat the license and privacy issues. All released data will be ensured under the license of Creative Commons Attribution 2.0 Generic (CC-BY-2.0) and free for research use. Also, persons in the dataset are anonymised without additional private or sensitive metadata.

## Agreement
The SHHQ is available for non-commercial research purposes only. 

You agree not to reproduce, duplicate, copy, sell, trade, resell or exploit any portion of the images and any portion of the derived data for commercial purposes. 

You agree NOT to further copy, publish or distribute any portion of SHHQ to any third party for any purpose. Except, for internal use at a single site within the same organization it is allowed to make copies of the dataset.

Shanghai AI Lab reserves the right to terminate your access to the SHHQ at any time.

## Dataset Preview
For those interested in our dataset, we provide a preview version with 100 images randomly sampled from SHHQ-1.0: [SHHQ-1.0_samples](https://drive.google.com/file/d/1tnNFfmFtzRbYL3qEnNXQ_ShaN9YV5tI5/view?usp=sharing).

If you want to access the full SHHQ-1.0, please read the following instructions.

## Download Instructions
Please download the SHHQ Dataset Release Agreement from [link](./SHHQ_Dataset_Release_Agreement.pdf).
Read it carefully, complete and sign it appropriately. 

Please send the completed form to Jianglin Fu (arlenefu@outlook.com) and Shikai Li (lishikai@pjlab.org.cn), and cc to Wayne Wu (wuwenyan0503@gmail.com) using institutional email address. The email Subject Title is "SHHQ Dataset Release Agreement". We will verify your request and contact you with the dataset link and password to unzip the image data.

Note:

We are currently facing large incoming applications, and we need to carefully verify all the applicants, please be patient, and we will reply to you as soon as possible.

## References
<a id="1">[1]</a> 
Hacheme, G., Sayouti, N.: Neural fashion image captioning: Accounting for data diversity. arXiv preprint arXiv:2106.12154 (2021)