# Descriptor Vector Exchange


This repo provides code for learning dense landmarks without supervision.  Our approach is described in the ICCV 2019 [paper](TODO-update-link) "Unsupervised learning of landmarks by exchanging descriptor vectors".


![DVE diagram](figs/DVE.png)

**High level Overview:** The goal of this work is to learn a dense embedding Φ<sub>u</sub>(x) ∈ R<sup>d</sup> of image pixels without annotation. Our starting point was the *Dense Equivariant Labelling* approach of [3] (references follow at the end of the README), which similarly tackles the same problem, but is restricted to learning low-dimensional embeddings to achieve the key objective of generalisation *across different identities*.  The key focus of *Descriptor Vector Exchange* (DVE) is to address this dimensionality issue to enable the learning of more powerful, higher dimensional embeddings while still preserving their generalisation ability. To do so, we take inspiration from methods which enforce transitive/cyclic consistency constraints [4, 5, 6].

The embedding is learned from pairs of images (x,x′) related by a known warp v = g(u). In the image above, on the left we show the approach used by [3], which directly matches embedding Φ<sub>u</sub>(x) from the left image to embeddings Φ<sub>v</sub>(x′) in the right image to generate a loss. On the right, *DVE* replaces Φ<sub>u</sub>(x) with its reconstruction Φˆ<sub>u</sub>(x|xα) obtained from the embeddings in a third auxiliary image xα (the correspondence with xα does not need to be known). This mechanism encourages the embeddings to act consistently across different instances, even when the dimensionality is increased (see the paper for more details).


**Requirements:** The code assumes PyTorch 1.1 and Python 3.7 (other versions may work, but have not been tested).  See the section on dependencies towards the end of this file for specific package requirements.


### Learned Embeddings

We provide pretrained models for each dataset to reproduce the results reported in the paper [1]. The training is performed with **CelebA**, a dataset of over 200k faces of celebrities that was originally described in [this paper](http://openaccess.thecvf.com/content_iccv_2015/papers/Liu_Deep_Learning_Face_ICCV_2015_paper.pdf).  We use this dataset to train our embedding function without annotations. 

Each model is accompanied by training and evaluation logs and its mean pixel error performance on the task of matching annotated landmarks across the MAFL test set (described in more detail below). We use two architectures: the *smallnet* model of [3] and the more powerful *hourglass* model used in [7].

The goal of these initial experiments is to demonstrate that DVE allows models to generalise across identities even when using higher dimensional embeddings (e.g. 64d rather than 3d).  By contrast, this does not occur when DVE is removed (see the ablation section below).


| Embed. Dim | Model | Same Identity | Different Identity | Params | Links |  
| :-----------: | :-: | :----: | :----: | :----: | :----: |
|  3 | smallnet | 1.36 | 3.03 | 334.9k | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-3d-dve/2019-08-08_17-54-21/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-3d-dve/2019-08-08_17-54-21/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/celeba-smallnet-3d-dve/2019-08-08_17-54-21/info.log) |
|  16 | smallnet | 1.28 | 2.79 | 338.2k | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-16d-dve/2019-08-02_06-20-13/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-16d-dve/2019-08-02_06-20-13/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/celeba-smallnet-16d-dve/2019-08-02_06-20-13/info.log) |
|  32 | smallnet | 1.29 | 2.79 | 342.3k | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-32d-dve/2019-08-02_06-19-59/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-32d-dve/2019-08-02_06-19-59/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/celeba-smallnet-32d-dve/2019-08-02_06-19-59/info.log) |
|  64 | smallnet | 1.28 | 2.77 | 350.6k | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-64d-dve/2019-08-02_06-20-28/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-64d-dve/2019-08-02_06-20-28/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/celeba-smallnet-64d-dve/2019-08-02_06-20-28/info.log) |
|  64 | hourglass | 0.93 | 2.37 | 12.6M | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-hourglass-64d-dve/0618_103501/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-hourglass-64d-dve/0618_103501/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/celeba-hourglass-64d-dve/0618_103501/info.log) |


**Notes**: The error metrics for the `hourglass` model, which are included for completeness, are approximately (but are not exactly) comparable to the metrics for the `smallnet` due to very slight differences in the cropping ratios used by the two architectures (0.3 for smallnet, 0.294 for Hourglass).

### Landmark Regression

**Protocol Description**: To transform the learned dense embeddings into landmark predictions, we use the same approach as [3].  For each target dataset, we freeze the dense embeddings and learn to peg onto them a collection of 50 "virtual" keypoints via a spatial softmax.  These virtual keypoints are then used to regress the target keypoints of the dataset.  We report the error as a percentage of inter-ocular distance (a metric defined by the landmarks of each dataset).

**MAFL landmark regression**

[MAFL](https://github.com/zhzhanp/TCDCN-face-alignment/tree/master/MAFL) is a dataset of 20k faces which includes landmark annotations.  The dataset is partitioned into 19k training images and 1k testing images.

| Embed. Dim | Model | Error (%IOD) | Links | 
| :-----------: | :-: | :----: | :----: |
|  3 | smallnet | 4.17 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-3d-dve/2019-08-11_08-33-22/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-3d-dve/2019-08-11_08-33-22/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/mafl-keypoints-celeba-smallnet-3d-dve/2019-08-11_08-33-22/info.log) |
|  16 | smallnet | 3.97 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-16d-dve/2019-08-11_08-29-31/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-16d-dve/2019-08-11_08-29-31/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/mafl-keypoints-celeba-smallnet-16d-dve/2019-08-11_08-29-31/info.log) |
|  32 | smallnet | 3.82 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-32d-dve/2019-08-11_08-29-53/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-32d-dve/2019-08-11_08-29-53/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/mafl-keypoints-celeba-smallnet-32d-dve/2019-08-11_08-29-53/info.log) |
|  64 | smallnet | 3.42 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-64d-dve/2019-08-11_08-40-48/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-64d-dve/2019-08-11_08-40-48/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/mafl-keypoints-celeba-smallnet-64d-dve/2019-08-11_08-40-48/info.log) |
|  64 | hourglass | 2.86 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-hourglass-64d-dve/2019-08-11_14-30-53/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-hourglass-64d-dve/2019-08-11_14-30-53/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/mafl-keypoints-celeba-hourglass-64d-dve/2019-08-11_14-30-53/info.log) |

**300-W landmark regression**

The 300-W This dataset contains 3,148 training images and 689 testing images with 68 facial landmark annotations for each face (with the split introduced [this CVPR 2014 paper](http://www.jiansun.org/papers/CVPR14_FaceAlignment.pdf)).  The dataset is described in this [2013 ICCV workshop paper](https://www.cv-foundation.org/openaccess/content_iccv_workshops_2013/W11/papers/Sagonas_300_Faces_in-the-Wild_2013_ICCV_paper.pdf). 


| Embed. Dim | Model | Error (%IOD) | Links | 
| :-----------: | :--: | :----: | :----: |
|  3 | smallnet | 7.66 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-3d-dve/2019-08-11_14-50-50/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-3d-dve/2019-08-11_14-50-50/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-keypoints-celeba-smallnet-3d-dve/2019-08-11_14-50-50/info.log) |
|  16 | smallnet | 6.29 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-16d-dve/2019-08-11_14-50-52/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-16d-dve/2019-08-11_14-50-52/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-keypoints-celeba-smallnet-16d-dve/2019-08-11_14-50-52/info.log) |
|  32 | smallnet | 6.13 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-32d-dve/2019-08-11_14-50-53/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-32d-dve/2019-08-11_14-50-53/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-keypoints-celeba-smallnet-32d-dve/2019-08-11_14-50-53/info.log) |
|  64 | smallnet | 5.75 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-64d-dve/2019-08-11_14-50-54/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-64d-dve/2019-08-11_14-50-54/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-keypoints-celeba-smallnet-64d-dve/2019-08-11_14-50-54/info.log) |
|  64 | hourglass | 4.65 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-hourglass-64d-dve/2019-08-11_18-50-33/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-hourglass-64d-dve/2019-08-11_18-50-33/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-keypoints-celeba-hourglass-64d-dve/2019-08-11_18-50-33/info.log) |


**AFLW landmark regression**

The [original AFLW](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/) contains around 25k images with up to 21 landmarks. For the purposes of evaluating five-landmark detectors, the authors of [TCDCN](http://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html) introduced a test subset of almost 3K faces, these pre-cropped images are available in the [MTFL download](http://mmlab.ie.cuhk.edu.hk/projects/TCDCN/data/MTFL.zip)

There are two slightly different partitions of AFLW that have been used in prior work (we report numbers on both to allow for comparison).  One is a set of recropped faces released by [7] (2991 test faces with 132 duplicates, 10122 train faces) (here we call this AFLW-recrop). The second is the train/test partition of AFLW used in the works of [2,3] which used the existing crops from MTFL (2995 faces) for testing and 10122 AFLW faces for training [(download)](http://www.robots.ox.ac.uk/~jdt/aflw_10122train_cropped.zip) (we call this dataset split AFLW-MTFL).

Additionally, in the tables immediately below, each embedding is further fine-tuned on the AFLW-recrop/AFLW-MTFL training sets (without annotations), as was done in [2], [3], [7], [8].  The rationale for this is that (i) it does not require any additional superviserion; (ii) it allows the model to adjust for the differences in the face crops provided by the detector.  To give an idea of how sensitive the method is to this step, we also report performance without finetuning in the ablation studies below.

*AFLW-recrop landmark regression*


| Embed. Dim | Model | Error (%IOD) | Links | 
| :-----------: | :--: | :----: | :----: |
|  3 | smallnet | 10.13 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-3d-dve/2019-08-12_05-50-54/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-3d-dve/2019-08-12_05-50-54/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-keypoints-celeba-smallnet-3d-dve/2019-08-12_05-50-54/info.log) |
|  16 | smallnet | 8.40 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-16d-dve/2019-08-11_07-57-00/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-16d-dve/2019-08-11_07-57-00/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-keypoints-celeba-smallnet-16d-dve/2019-08-11_07-57-00/info.log) |
|  32 | smallnet | 8.18 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-32d-dve/2019-08-11_07-56-57/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-32d-dve/2019-08-11_07-56-57/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-keypoints-celeba-smallnet-32d-dve/2019-08-11_07-56-57/info.log) |
|  64 | smallnet | 7.79 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-64d-dve/2019-08-11_18-53-30/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-64d-dve/2019-08-11_18-53-30/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-keypoints-celeba-smallnet-64d-dve/2019-08-11_18-53-30/info.log) |
|  64 | hourglass | 6.54 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-hourglass-64d-dve/2019-08-12_06-00-52/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-hourglass-64d-dve/2019-08-12_06-00-52/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-keypoints-celeba-hourglass-64d-dve/2019-08-12_06-00-52/info.log) |


*AFLW-MTFL landmark regression*

AFLW-MTFLis a dataset of faces which also includes landmark annotations. We use the P = 5 landmark test split (10,122 training images and 2,991 test images). The dataset can be obtained [here](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/) and is described in this [2011 ICCV workshop paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.384.2988&rep=rep1&type=pdf). 

| Embed. Dim | Model | Error (%IOD) | Links | 
| :-----------: | :--: | :----: | :----: |
|  3 | smallnet | 11.12 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-3d-dve/2019-08-11_18-43-20/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-3d-dve/2019-08-11_18-43-20/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-keypoints-celeba-smallnet-3d-dve/2019-08-11_18-43-20/info.log) |
|  16 | smallnet | 9.15 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-16d-dve/2019-08-11_18-43-24/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-16d-dve/2019-08-11_18-43-24/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-keypoints-celeba-smallnet-16d-dve/2019-08-11_18-43-24/info.log) |
|  32 | smallnet | 9.17 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-32d-dve/2019-08-11_18-43-30/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-32d-dve/2019-08-11_18-43-30/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-keypoints-celeba-smallnet-32d-dve/2019-08-11_18-43-30/info.log) |
|  64 | smallnet | 8.60 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-64d-dve/2019-08-11_18-43-35/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-64d-dve/2019-08-11_18-43-35/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-keypoints-celeba-smallnet-64d-dve/2019-08-11_18-43-35/info.log) |
|  64 | hourglass | 7.53 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-hourglass-64d-dve/2019-08-12_06-00-59/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-hourglass-64d-dve/2019-08-12_06-00-59/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-keypoints-celeba-hourglass-64d-dve/2019-08-12_06-00-59/info.log) |


## Ablation Studies

We can study the effect of the DVE method by removing it during training and assessing the resulting embeddings for landmark regression.  The ablations are performed on the lighter SmallNet model.

| Embed. Dim | Model | DVE | Same Identity | Different Identity | Links | 
| :-----------: | :--:  | :-: | :----: | :----: |
|  3 | smallnet | :heavy_multiplication_x: / :heavy_check_mark:  | 1.33 / 1.36| 2.89 / 3.03 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-3d/2019-08-04_17-55-48/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-3d/2019-08-04_17-55-48/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/celeba-smallnet-3d/2019-08-04_17-55-48/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-3d-dve/2019-08-08_17-54-21/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-3d-dve/2019-08-08_17-54-21/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/celeba-smallnet-3d-dve/2019-08-08_17-54-21/info.log)) |
|  16 | smallnet | :heavy_multiplication_x: / :heavy_check_mark:  | 1.25 / 1.28| 5.65 / 2.79 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-16d/2019-08-04_17-55-52/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-16d/2019-08-04_17-55-52/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/celeba-smallnet-16d/2019-08-04_17-55-52/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-16d-dve/2019-08-02_06-20-13/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-16d-dve/2019-08-02_06-20-13/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/celeba-smallnet-16d-dve/2019-08-02_06-20-13/info.log)) |
|  32 | smallnet | :heavy_multiplication_x: / :heavy_check_mark:  | 1.26 / 1.29| 5.81 / 2.79 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-32d/2019-08-04_17-55-57/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-32d/2019-08-04_17-55-57/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/celeba-smallnet-32d/2019-08-04_17-55-57/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-32d-dve/2019-08-02_06-19-59/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-32d-dve/2019-08-02_06-19-59/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/celeba-smallnet-32d-dve/2019-08-02_06-19-59/info.log)) |
|  64 | smallnet | :heavy_multiplication_x: / :heavy_check_mark:  | 1.25 / 1.28| 5.68 / 2.77 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-64d/2019-08-04_17-56-04/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-64d/2019-08-04_17-56-04/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/celeba-smallnet-64d/2019-08-04_17-56-04/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-64d-dve/2019-08-02_06-20-28/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-64d-dve/2019-08-02_06-20-28/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/celeba-smallnet-64d-dve/2019-08-02_06-20-28/info.log)) |

We see that without DVE, the learned embedding performs reasonably when the dimensionality is restricted to 3d.  However, when we seek to learn higher dimesionality embeddings without DVE, they lose their ability to match across different identities.  This inability to generalise at higher dimensions is similarly reflected when the embeddings are used to regress landmarks:

**DVE Ablation: MAFL landmark regression**

| Embed. Dim | Model | DVE | Error (%IOD) | Links | 
| :-----------: | :--:  | :-: | :----: | :----: |
|  3 | smallnet | :heavy_multiplication_x: / :heavy_check_mark: | 4.02/4.17 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-3d/2019-08-11_08-24-51/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-3d/2019-08-11_08-24-51/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/mafl-keypoints-celeba-smallnet-3d/2019-08-11_08-24-51/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-3d-dve/2019-08-11_08-33-22/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-3d-dve/2019-08-11_08-33-22/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/mafl-keypoints-celeba-smallnet-3d-dve/2019-08-11_08-33-22/info.log)) |
|  16 | smallnet | :heavy_multiplication_x: / :heavy_check_mark: | 5.31/3.97 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-16d/2019-08-11_08-29-08/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-16d/2019-08-11_08-29-08/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/mafl-keypoints-celeba-smallnet-16d/2019-08-11_08-29-08/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-16d-dve/2019-08-11_08-29-31/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-16d-dve/2019-08-11_08-29-31/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/mafl-keypoints-celeba-smallnet-16d-dve/2019-08-11_08-29-31/info.log)) |
|  32 | smallnet | :heavy_multiplication_x: / :heavy_check_mark: | 5.36/3.82 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-32d/2019-08-11_08-25-14/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-32d/2019-08-11_08-25-14/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/mafl-keypoints-celeba-smallnet-32d/2019-08-11_08-25-14/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-32d-dve/2019-08-11_08-29-53/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-32d-dve/2019-08-11_08-29-53/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/mafl-keypoints-celeba-smallnet-32d-dve/2019-08-11_08-29-53/info.log)) |
|  64 | smallnet | :heavy_multiplication_x: / :heavy_check_mark: | 4.99/3.42 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-64d/2019-08-08_17-56-47/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-64d/2019-08-08_17-56-47/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/mafl-keypoints-celeba-smallnet-64d/2019-08-08_17-56-47/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-64d-dve/2019-08-11_08-40-48/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-64d-dve/2019-08-11_08-40-48/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/mafl-keypoints-celeba-smallnet-64d-dve/2019-08-11_08-40-48/info.log)) |



**DVE Ablation: 300w landmark regression**

| Embed. Dim | Model | DVE | Error (%IOD) | Links | 
| :-----------: | :--:  | :-: | :----: | :----: |
|  3 | smallnet | :heavy_multiplication_x: / :heavy_check_mark: | 8.23/7.66 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-3d/2019-08-11_14-50-45/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-3d/2019-08-11_14-50-45/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-keypoints-celeba-smallnet-3d/2019-08-11_14-50-45/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-3d-dve/2019-08-11_14-50-50/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-3d-dve/2019-08-11_14-50-50/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-keypoints-celeba-smallnet-3d-dve/2019-08-11_14-50-50/info.log)) |
|  16 | smallnet | :heavy_multiplication_x: / :heavy_check_mark: | 10.66/6.29 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-16d/2019-08-11_14-50-46/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-16d/2019-08-11_14-50-46/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-keypoints-celeba-smallnet-16d/2019-08-11_14-50-46/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-16d-dve/2019-08-11_14-50-52/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-16d-dve/2019-08-11_14-50-52/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-keypoints-celeba-smallnet-16d-dve/2019-08-11_14-50-52/info.log)) |
|  32 | smallnet | :heavy_multiplication_x: / :heavy_check_mark: | 10.33/6.13 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-32d/2019-08-11_14-50-47/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-32d/2019-08-11_14-50-47/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-keypoints-celeba-smallnet-32d/2019-08-11_14-50-47/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-32d-dve/2019-08-11_14-50-53/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-32d-dve/2019-08-11_14-50-53/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-keypoints-celeba-smallnet-32d-dve/2019-08-11_14-50-53/info.log)) |
|  64 | smallnet | :heavy_multiplication_x: / :heavy_check_mark: | 9.33/5.75 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-64d/2019-08-11_14-50-48/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-64d/2019-08-11_14-50-48/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-keypoints-celeba-smallnet-64d/2019-08-11_14-50-48/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-64d-dve/2019-08-11_14-50-54/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-64d-dve/2019-08-11_14-50-54/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-keypoints-celeba-smallnet-64d-dve/2019-08-11_14-50-54/info.log)) |


**DVE Ablation: AFLW-mtfl landmark regression**

| Embed. Dim | Model | DVE | Error (%IOD) | Links | 
| :-----------: | :--:  | :-: | :----: | :----: |
|  3 | smallnet | :heavy_multiplication_x: / :heavy_check_mark: | 10.99/11.12 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-3d/2019-08-11_18-42-45/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-3d/2019-08-11_18-42-45/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-keypoints-celeba-smallnet-3d/2019-08-11_18-42-45/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-3d-dve/2019-08-11_18-43-20/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-3d-dve/2019-08-11_18-43-20/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-keypoints-celeba-smallnet-3d-dve/2019-08-11_18-43-20/info.log)) |
|  16 | smallnet | :heavy_multiplication_x: / :heavy_check_mark: | 12.22/9.15 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-16d/2019-08-11_18-43-03/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-16d/2019-08-11_18-43-03/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-keypoints-celeba-smallnet-16d/2019-08-11_18-43-03/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-16d-dve/2019-08-11_18-43-24/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-16d-dve/2019-08-11_18-43-24/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-keypoints-celeba-smallnet-16d-dve/2019-08-11_18-43-24/info.log)) |
|  32 | smallnet | :heavy_multiplication_x: / :heavy_check_mark: | 12.60/9.17 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-32d/2019-08-11_18-43-09/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-32d/2019-08-11_18-43-09/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-keypoints-celeba-smallnet-32d/2019-08-11_18-43-09/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-32d-dve/2019-08-11_18-43-30/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-32d-dve/2019-08-11_18-43-30/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-keypoints-celeba-smallnet-32d-dve/2019-08-11_18-43-30/info.log)) |
|  64 | smallnet | :heavy_multiplication_x: / :heavy_check_mark: | 12.92/8.60 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-64d/2019-08-11_18-43-14/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-64d/2019-08-11_18-43-14/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-keypoints-celeba-smallnet-64d/2019-08-11_18-43-14/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-64d-dve/2019-08-11_18-43-35/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-64d-dve/2019-08-11_18-43-35/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-keypoints-celeba-smallnet-64d-dve/2019-08-11_18-43-35/info.log)) |


**DVE Ablation: AFLW-recrop landmark regression**

| Embed. Dim | Model | DVE | Error (%IOD) | Links | 
| :-----------: | :--:  | :-: | :----: | :----: |
|  3 | smallnet | :heavy_multiplication_x: / :heavy_check_mark: | 10.14/10.13 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-3d/2019-08-11_07-56-34/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-3d/2019-08-11_07-56-34/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-keypoints-celeba-smallnet-3d/2019-08-11_07-56-34/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-3d-dve/2019-08-12_05-50-54/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-3d-dve/2019-08-12_05-50-54/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-keypoints-celeba-smallnet-3d-dve/2019-08-12_05-50-54/info.log)) |
|  16 | smallnet | :heavy_multiplication_x: / :heavy_check_mark: | 10.73/8.40 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-16d/2019-08-11_07-56-41/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-16d/2019-08-11_07-56-41/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-keypoints-celeba-smallnet-16d/2019-08-11_07-56-41/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-16d-dve/2019-08-11_07-57-00/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-16d-dve/2019-08-11_07-57-00/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-keypoints-celeba-smallnet-16d-dve/2019-08-11_07-57-00/info.log)) |
|  32 | smallnet | :heavy_multiplication_x: / :heavy_check_mark: | 11.05/8.18 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-32d/2019-08-11_07-56-46/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-32d/2019-08-11_07-56-46/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-keypoints-celeba-smallnet-32d/2019-08-11_07-56-46/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-32d-dve/2019-08-11_07-56-57/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-32d-dve/2019-08-11_07-56-57/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-keypoints-celeba-smallnet-32d-dve/2019-08-11_07-56-57/info.log)) |
|  64 | smallnet | :heavy_multiplication_x: / :heavy_check_mark: | 11.43/7.79 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-64d/2019-08-11_07-56-52/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-64d/2019-08-11_07-56-52/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-keypoints-celeba-smallnet-64d/2019-08-11_07-56-52/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-64d-dve/2019-08-11_18-53-30/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-64d-dve/2019-08-11_18-53-30/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-keypoints-celeba-smallnet-64d-dve/2019-08-11_18-53-30/info.log)) |


Next we investigate how sensitive our approach is to finetuning on the target dataset (this is done for the AFLW-recrop and AFLW-mtfl landmark regressions).   We do two sets of experiments.  First we, remove the finetuning for both the AFLW dataset variants and re-evaluate on the landmark regression tasks.  Second, we add in a finetuning step for a different dataset, 300w, to see how the method is affected on a different benchmark. Note that all models for these experiments use DVE, and the finetuning consists of training the embeddings for an additional 50 epochs without annotations.  We see that for the AFLW datasets, it makes a reasonable difference to performance.  However, for 300w, particularly for stronger models, it adds little benefit (for this reason we do not use finetuning on 300w for the results reported in the paper).

**Finetuning Ablation: AFLW-mtfl landmark regression**

| Embed. Dim | Model | Finetune | Error (%IOD) | Links | 
| :-----------: | :--:  | :-: | :----: | :----: |
|  3 | smallnet | :heavy_multiplication_x: / :heavy_check_mark: | 11.82/11.12 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-keypoints-celeba-smallnet-3d-dve/2019-08-11_14-02-25/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-keypoints-celeba-smallnet-3d-dve/2019-08-11_14-02-25/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-keypoints-celeba-smallnet-3d-dve/2019-08-11_14-02-25/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-3d-dve/2019-08-11_18-43-20/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-3d-dve/2019-08-11_18-43-20/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-keypoints-celeba-smallnet-3d-dve/2019-08-11_18-43-20/info.log)) |
|  16 | smallnet | :heavy_multiplication_x: / :heavy_check_mark: | 10.22/9.15 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-keypoints-celeba-smallnet-16d-dve/2019-08-11_14-02-40/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-keypoints-celeba-smallnet-16d-dve/2019-08-11_14-02-40/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-keypoints-celeba-smallnet-16d-dve/2019-08-11_14-02-40/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-16d-dve/2019-08-11_18-43-24/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-16d-dve/2019-08-11_18-43-24/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-keypoints-celeba-smallnet-16d-dve/2019-08-11_18-43-24/info.log)) |
|  32 | smallnet | :heavy_multiplication_x: / :heavy_check_mark: | 9.80/9.17 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-keypoints-celeba-smallnet-32d-dve/2019-08-11_14-02-56/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-keypoints-celeba-smallnet-32d-dve/2019-08-11_14-02-56/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-keypoints-celeba-smallnet-32d-dve/2019-08-11_14-02-56/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-32d-dve/2019-08-11_18-43-30/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-32d-dve/2019-08-11_18-43-30/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-keypoints-celeba-smallnet-32d-dve/2019-08-11_18-43-30/info.log)) |
|  64 | smallnet | :heavy_multiplication_x: / :heavy_check_mark: | 9.28/8.60 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-keypoints-celeba-smallnet-64d-dve/2019-08-11_14-03-14/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-keypoints-celeba-smallnet-64d-dve/2019-08-11_14-03-14/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-keypoints-celeba-smallnet-64d-dve/2019-08-11_14-03-14/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-64d-dve/2019-08-11_18-43-35/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-64d-dve/2019-08-11_18-43-35/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-keypoints-celeba-smallnet-64d-dve/2019-08-11_18-43-35/info.log)) |
|  64 | hourglass | :heavy_multiplication_x: / :heavy_check_mark: | 8.15/7.53 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-keypoints-celeba-hourglass-64d-dve/2019-08-11_18-49-34/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-keypoints-celeba-hourglass-64d-dve/2019-08-11_18-49-34/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-keypoints-celeba-hourglass-64d-dve/2019-08-11_18-49-34/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-hourglass-64d-dve/2019-08-12_06-00-59/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-hourglass-64d-dve/2019-08-12_06-00-59/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-keypoints-celeba-hourglass-64d-dve/2019-08-12_06-00-59/info.log)) |


**Finetuning Ablation: AFLW-recrop landmark regression**

| Embed. Dim | Model | Finetune | Error (%IOD) | Links | 
| :-----------: | :--:  | :-: | :----: | :----: |
|  3 | smallnet | :heavy_multiplication_x: / :heavy_check_mark: | 9.65/10.13 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-keypoints-celeba-smallnet-3d-dve/2019-08-11_08-42-33/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-keypoints-celeba-smallnet-3d-dve/2019-08-11_08-42-33/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-keypoints-celeba-smallnet-3d-dve/2019-08-11_08-42-33/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-3d-dve/2019-08-12_05-50-54/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-3d-dve/2019-08-12_05-50-54/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-keypoints-celeba-smallnet-3d-dve/2019-08-12_05-50-54/info.log)) |
|  16 | smallnet | :heavy_multiplication_x: / :heavy_check_mark: | 8.91/8.40 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-keypoints-celeba-smallnet-16d-dve/2019-08-10_12-53-44/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-keypoints-celeba-smallnet-16d-dve/2019-08-10_12-53-44/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-keypoints-celeba-smallnet-16d-dve/2019-08-10_12-53-44/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-16d-dve/2019-08-11_07-57-00/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-16d-dve/2019-08-11_07-57-00/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-keypoints-celeba-smallnet-16d-dve/2019-08-11_07-57-00/info.log)) |
|  32 | smallnet | :heavy_multiplication_x: / :heavy_check_mark: | 8.73/8.18 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-keypoints-celeba-smallnet-32d-dve/2019-08-10_09-20-02/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-keypoints-celeba-smallnet-32d-dve/2019-08-10_09-20-02/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-keypoints-celeba-smallnet-32d-dve/2019-08-10_09-20-02/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-32d-dve/2019-08-11_07-56-57/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-32d-dve/2019-08-11_07-56-57/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-keypoints-celeba-smallnet-32d-dve/2019-08-11_07-56-57/info.log)) |
|  64 | smallnet | :heavy_multiplication_x: / :heavy_check_mark: | 8.14/7.79 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-keypoints-celeba-smallnet-64d-dve/2019-08-10_12-55-24/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-keypoints-celeba-smallnet-64d-dve/2019-08-10_12-55-24/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-keypoints-celeba-smallnet-64d-dve/2019-08-10_12-55-24/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-64d-dve/2019-08-11_18-53-30/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-64d-dve/2019-08-11_18-53-30/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-keypoints-celeba-smallnet-64d-dve/2019-08-11_18-53-30/info.log)) |
|  64 | hourglass | :heavy_multiplication_x: / :heavy_check_mark: | 6.88/6.54 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-keypoints-celeba-hourglass-64d-dve/2019-08-09_14-29-19/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-keypoints-celeba-hourglass-64d-dve/2019-08-09_14-29-19/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-keypoints-celeba-hourglass-64d-dve/2019-08-09_14-29-19/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-hourglass-64d-dve/2019-08-12_06-00-52/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-hourglass-64d-dve/2019-08-12_06-00-52/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-keypoints-celeba-hourglass-64d-dve/2019-08-12_06-00-52/info.log)) |


**Finetuning Ablation: 300w landmark regression**

| Embed. Dim | Model | Finetune | Error (%IOD) | Links | 
| :-----------: | :--:  | :-: | :----: | :----: |
|  3 | smallnet | :heavy_multiplication_x: / :heavy_check_mark: | 7.66/7.20 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-3d-dve/2019-08-11_14-50-50/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-3d-dve/2019-08-11_14-50-50/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-keypoints-celeba-smallnet-3d-dve/2019-08-11_14-50-50/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-keypoints-celeba-smallnet-3d-dve/2019-08-12_05-21-09/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-keypoints-celeba-smallnet-3d-dve/2019-08-12_05-21-09/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-ft-keypoints-celeba-smallnet-3d-dve/2019-08-12_05-21-09/info.log)) |
|  16 | smallnet | :heavy_multiplication_x: / :heavy_check_mark: | 6.29/5.90 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-16d-dve/2019-08-11_14-50-52/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-16d-dve/2019-08-11_14-50-52/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-keypoints-celeba-smallnet-16d-dve/2019-08-11_14-50-52/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-keypoints-celeba-smallnet-16d-dve/2019-08-12_05-21-17/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-keypoints-celeba-smallnet-16d-dve/2019-08-12_05-21-17/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-ft-keypoints-celeba-smallnet-16d-dve/2019-08-12_05-21-17/info.log)) |
|  32 | smallnet | :heavy_multiplication_x: / :heavy_check_mark: | 6.13/5.75 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-32d-dve/2019-08-11_14-50-53/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-32d-dve/2019-08-11_14-50-53/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-keypoints-celeba-smallnet-32d-dve/2019-08-11_14-50-53/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-keypoints-celeba-smallnet-32d-dve/2019-08-12_05-21-27/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-keypoints-celeba-smallnet-32d-dve/2019-08-12_05-21-27/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-ft-keypoints-celeba-smallnet-32d-dve/2019-08-12_05-21-27/info.log)) |
|  64 | smallnet | :heavy_multiplication_x: / :heavy_check_mark: | 5.75/5.58 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-64d-dve/2019-08-11_14-50-54/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-64d-dve/2019-08-11_14-50-54/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-keypoints-celeba-smallnet-64d-dve/2019-08-11_14-50-54/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-keypoints-celeba-smallnet-64d-dve/2019-08-12_05-21-35/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-keypoints-celeba-smallnet-64d-dve/2019-08-12_05-21-35/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-ft-keypoints-celeba-smallnet-64d-dve/2019-08-12_05-21-35/info.log)) |
|  64 | hourglass | :heavy_multiplication_x: / :heavy_check_mark: | 4.65/4.65 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-hourglass-64d-dve/2019-08-11_18-50-33/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-hourglass-64d-dve/2019-08-11_18-50-33/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-keypoints-celeba-hourglass-64d-dve/2019-08-11_18-50-33/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-keypoints-celeba-hourglass-64d-dve/2019-08-11_18-42-44/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-keypoints-celeba-hourglass-64d-dve/2019-08-11_18-42-44/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-ft-keypoints-celeba-hourglass-64d-dve/2019-08-11_18-42-44/info.log)) |


To enable the finetuning experiments to be reproduced, the training logs for each of the three datasets are provided below, together with their performance on the matching task.

**Finetuning on AFLW-mtfl**

| Embed. Dim | Model | Same Identity | Different Identity | Links | 
| :-----------: | :-: | :----: | :----: | :----: |
|  3 | smallnet | 5.99 | 7.16 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-celeba-smallnet-3d-dve/2019-08-11_08-20-03/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-celeba-smallnet-3d-dve/2019-08-11_08-20-03/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-celeba-smallnet-3d-dve/2019-08-11_08-20-03/info.log) |
|  16 | smallnet | 4.72 | 7.11 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-celeba-smallnet-16d-dve/2019-08-11_08-19-58/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-celeba-smallnet-16d-dve/2019-08-11_08-19-58/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-celeba-smallnet-16d-dve/2019-08-11_08-19-58/info.log) |
|  32 | smallnet | 6.42 | 8.71 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-celeba-smallnet-32d-dve/2019-08-11_08-19-55/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-celeba-smallnet-32d-dve/2019-08-11_08-19-55/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-celeba-smallnet-32d-dve/2019-08-11_08-19-55/info.log) |
|  64 | smallnet | 8.07 | 10.09 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-celeba-smallnet-64d-dve/2019-08-11_08-19-54/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-celeba-smallnet-64d-dve/2019-08-11_08-19-54/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-celeba-smallnet-64d-dve/2019-08-11_08-19-54/info.log) |
|  64 | hourglass | 1.53 | 3.65 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-celeba-hourglass-64d-dve/2019-08-11_16-40-28/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-celeba-hourglass-64d-dve/2019-08-11_16-40-28/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-celeba-hourglass-64d-dve/2019-08-11_16-40-28/info.log) |

**Finetuning on AFLW-recrop**

| Embed. Dim | Model | Same Identity | Different Identity | Links | 
| :-----------: | :-: | :----: | :----: | :----: |
|  3 | smallnet | 6.36 | 7.69 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-celeba-smallnet-3d-dve/2019-08-11_18-51-40/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-celeba-smallnet-3d-dve/2019-08-11_18-51-40/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-celeba-smallnet-3d-dve/2019-08-11_18-51-40/info.log) |
|  16 | smallnet | 6.34 | 8.62 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-celeba-smallnet-16d-dve/2019-08-10_12-50-30/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-celeba-smallnet-16d-dve/2019-08-10_12-50-30/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-celeba-smallnet-16d-dve/2019-08-10_12-50-30/info.log) |
|  32 | smallnet | 8.10 | 10.11 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-celeba-smallnet-32d-dve/2019-08-10_12-50-31/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-celeba-smallnet-32d-dve/2019-08-10_12-50-31/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-celeba-smallnet-32d-dve/2019-08-10_12-50-31/info.log) |
|  64 | smallnet | 4.08 | 5.21 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-celeba-smallnet-64d-dve/2019-08-10_12-50-32/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-celeba-smallnet-64d-dve/2019-08-10_12-50-32/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-celeba-smallnet-64d-dve/2019-08-10_12-50-32/info.log) |
|  64 | hourglass | 1.17 | 4.04 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-celeba-hourglass-64d-dve/2019-08-11_14-43-34/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-celeba-hourglass-64d-dve/2019-08-11_14-43-34/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-celeba-hourglass-64d-dve/2019-08-11_14-43-34/info.log) |

**Finetuning on 300w**

| Embed. Dim | Model | Same Identity | Different Identity | Links | 
| :-----------: | :-: | :----: | :----: | :----: |
|  3 | smallnet | 5.21 | 6.51 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-celeba-smallnet-3d-dve/2019-08-11_18-11-57/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-celeba-smallnet-3d-dve/2019-08-11_18-11-57/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-ft-celeba-smallnet-3d-dve/2019-08-11_18-11-57/info.log) |
|  16 | smallnet | 5.55 | 7.30 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-celeba-smallnet-16d-dve/2019-08-11_18-12-04/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-celeba-smallnet-16d-dve/2019-08-11_18-12-04/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-ft-celeba-smallnet-16d-dve/2019-08-11_18-12-04/info.log) |
|  32 | smallnet | 5.85 | 7.47 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-celeba-smallnet-32d-dve/2019-08-11_18-12-15/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-celeba-smallnet-32d-dve/2019-08-11_18-12-15/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-ft-celeba-smallnet-32d-dve/2019-08-11_18-12-15/info.log) |
|  64 | smallnet | 6.58 | 8.19 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-celeba-smallnet-64d-dve/2019-08-11_18-12-24/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-celeba-smallnet-64d-dve/2019-08-11_18-12-24/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-ft-celeba-smallnet-64d-dve/2019-08-11_18-12-24/info.log) |
|  64 | hourglass | 1.63 | 3.82 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-celeba-hourglass-64d-dve/2019-08-11_12-57-08/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-celeba-hourglass-64d-dve/2019-08-11_12-57-08/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-ft-celeba-hourglass-64d-dve/2019-08-11_12-57-08/info.log) |


### Dataset mirrors

For each dataset used in the paper, we provide a preprocessed copy to allow the results described above to be reproduced directly.  These can be downloaded and unpacked with a [utility script]() (recommended -- see example usage below), which will store them in the locations expected by the training code. Each dataset has a brief README, which also provides the citations for use with each dataset, together with a link from which it can be downloaded directly.

| Dataset   | Details and links | Archive size | sha1sum |
|:-------------:|:-----:|:----:|:---:|
| CelebA (+ MAFL) | [README](misc/datasets/celeba/README.md)| TODO | <sup><sub>TODO</sub></sup> |
| 300w | [README](misc/datasets/300w/README.md)| 3.0 GiB | <sup><sub>885b09159c61fa29998437747d589c65cfc4ccd3</sub></sup> |
| AFLW-mtfl | [README](misc/datasets/aflw-mtfl/README.md)| 252 MiB | <sup><sub>1ff31c07cef4f2777b416d896a65f6c17d8ae2ee</sub></sup> |



### Notes


**Metrics**
TODO(Samuuel): Explain why some logs are v. slow compared to others, why some are generated. 
TODO(Samuuel): Explain the definition of Inter-ocular distance on each dataset.  
300w can be obtained [here](https://ibug.doc.ic.ac.uk/resources/300-W/) and   
For 300w, we compute the inter-ocular distance according to the definition given by the dataset organizers [here](https://ibug.doc.ic.ac.uk/resources/300-W/).
TODO(Samuuel): `AFLW-recrop` is simply referred to as `AFLW` in the codebase.  
The regressor is learned for a fixed number of epochs for all models with the Adam solver.
The AFLW-recrop dataset is implemented in the `AFLW` class in [data_loaders.py](data_loader/data_loaders.py).

**Explain logs:** Some of the logs are generated from existing logfiles that were created with a slightly older version of the codebase (these differences only affect the log format, rather than the training code itself - the log generator can be found [here](misc/update_deprecated_exps.py).) TODO(Samuel): Explain why IOD isn't used as a metric here.




### Evaluating a pretrained embedding

Evaluting a pretrained model for a given dataset requires:
1. The target dataset, which should be located in `<root>/data/<dataset-name>` (this will be done automatically by the [data fetching script](misc/fetch_datasets.py), or can be done manually).
2. A `config.json` file.
3. A `checkpoint.pth` file.

Evaluation is then performed with the following command:
```
python3 test_matching.py --config <path-to-config.json> --resume <path-to-trained_model.pth> --device <gpu-id>
```
where `<gpu-id>` is the index of the GPU to evaluate on.  This option can be ommitted to run the evaluation on the CPU.

For example, to reproduce the `smallnet-32d-dve` results described above, run the following sequence of commands:

```
# fetch the mafl dataset (contained with celeba) 
python misc/fetch_datasets.py --dataset celeba

# find the name of a pretrained model using the links in the tables above 
export MODEL=data/models/celeba-smallnet-32d-dve/2019-08-08_17-56-24/checkpoint-epoch100.pth

# create a local directory and download the model into it 
mkdir -p $(dirname "${MODEL}")
wget --output-document="${MODEL}" "http://www.robots.ox.ac.uk/~vgg/research/DVE/${MODEL}"

# Evaluate the model
python3 test.py --config configs/celeba/smallnet-32d-dve.json --resume ${MODEL} --device 0
```

### Regressing landmarks

Learning a landmark regressor for a given pretrained embedding requires:
1. The target dataset, which should be located in `<root>/data/<dataset-name>` (this will be done automatically by the [data fetching script](misc/fetch_datasets.py), or can be done manually).
2. A `config.json` file.
3. A `checkpoint.pth` file.

See the [regressor code](model/keypoint_prediction.py) for details of how the regressor is implemented (it consists of a conv, then a spatial softmax, then a group conv).

Landmark learning is then performed with the following command:
```
python3 train.py --config <path-to-config.json> --resume <path-to-trained_model.pth> --device <gpu-id>
```
where `<gpu-id>` is the index of the GPU to evaluate on.  This option can be ommitted to run the evaluation on the CPU.

For example, to reproduce the `smallnet-32d-dve` landmark regression results described above, run the following sequence of commands:

```
# fetch the mafl dataset (contained with celeba) 
python misc/fetch_datasets.py --dataset celeba

# find the name of a pretrained model using the links in the tables above 
export MODEL=data/models/celeba-smallnet-32d-dve/2019-08-08_17-56-24/checkpoint-epoch100.pth

# create a local directory and download the model into it 
mkdir -p $(dirname "${MODEL}")
wget --output-document="${MODEL}" "http://www.robots.ox.ac.uk/~vgg/research/DVE/${MODEL}"

# Evaluate the model
python3 test.py --config configs/celeba/smallnet-32d-dve.json --resume ${MODEL} --device 0
```

### Learning new embeddings

TODO (add automatic setup scripts for users to reproduce numbers)

Requires CelebA.
The dataset can be obtained [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).  This dataset is implemented in the `CelebABase` class in [data_loaders.py](data_loader/data_loaders.py).

Learning a new embedding requires:
1. The dataset used for training, which should be located in `<root>/data/<dataset-name>` (this will be done automatically by the [data fetching script](misc/fetch_datasets.py), or can be done manually).
2. A `config.json` file.  You can define your own, or use one of the provided configs in the [configs](configs) directory.

Training is then performed with the following command:
```
python3 train.py --config <path-to-config.json> --device <gpu-id>
```
where `<gpu-id>` is the index of the GPU to train on.  This option can be ommitted to run the training on the CPU.

For example, to train a `16d-dve` embedding on `celeba`, run the following sequence of commands:

```
# fetch the celeba dataset 
python misc/fetch_datasets.py --dataset celeba

# Train the model
python3 train.py --config configs/celeba/smallnet-16d-dve.json --device 0
```



### Dependencies

If you have enough disk space, the recommended approach to installing the dependencies for this project is to create a conda enviroment via the `requirements/conda-requirements.txt`:

TODO(Samuel)

```
conda env create -f requirements/conda-freeze.yml
```

Otherwise, if you'd prefer to take a leaner approach, you can either:
1. `pip/conda install` each missing package each time you hit an `ImportError`
2. manually inspect the slightly more readable `requirements/pip-requirements.txt`



### Citation

If you find this code useful, please consider citing:

```
@inproceedings{Thewlis2019a,
  author    = {Thewlis, J. and Albanie, S. and Bilen, H. and Vedaldi, A.},
  booktitle = {International Conference on Computer Vision},
  title     = {Unsupervised learning of landmarks by exchanging descriptor vectors},
  date      = {2019},
}
```

### Related useful codebases

Some other codebases you might like to check out if you are interested in self-supervised learning of object structure.

* [LMDIS-REP](https://github.com/YutingZhang/lmdis-rep) [7]
* [IMM](https://github.com/tomasjakab/imm) [8]
* [FabNet](https://github.com/oawiles/FAb-Net) [9]


### Acknowledgements


We would like to thank Almut Sophia Koepke for her help.  The project structure uses the [pytorch-template](https://github.com/victoresque/pytorch-template) by @victoresque.

### References

[1] James Thewlis, Samuel Albanie, Hakan Bilen, and Andrea Vedaldi. "Unsupervised learning of landmarks by exchanging descriptor vectors" ICCV 2019.

[2] James Thewlis, Hakan Bilen and Andrea Vedaldi, "Unsupervised learning of object landmarks by factorized spatial embeddings." ICCV 2017.

[3] James Thewlis, Hakan Bilen and Andrea Vedaldi, "Unsupervised learning of object frames by dense equivariant image labelling." NeurIPS 2017

[4] Sundaram, N., Brox, T., & Keutzer, K. "Dense point trajectories by GPU-accelerated large displacement optical flow", ECCV 2010

[5] C. Zach, M. Klopschitz, and M. Pollefeys. "Disambiguating visual relations using loop constraints", CVPR, 2010

[6] Zhou, T., Jae Lee, Y., Yu, S. X., & Efros, A. A. "Flowweb: Joint image set alignment by weaving consistent, pixel-wise correspondences". CVPR 2015.

[7] Zhang, Yuting, Yijie Guo, Yixin Jin, Yijun Luo, Zhiyuan He, and Honglak Lee. "Unsupervised discovery of object landmarks as structural representations.", CVPR 2018

[8] Jakab, T., Gupta, A., Bilen, H., & Vedaldi, A. Unsupervised learning of object landmarks through conditional image generation, NeurIPS 2018

[9] Olivia Wiles, A. Sophia Koepke and Andrew Zisserman. "Self-supervised learning of a facial attribute embedding from video" , BMVC 2018
