# Descriptor Vector Exchange


This repo provides code for learning dense landmarks without supervision.  Our approach is described in the ICCV 2019 paper "Unsupervised learning of landmarks by exchanging descriptor vectors" ([link]()).


![DVE diagram](figs/DVE.png)

**High level Overview:** The goal of this work is to learn a dense embedding Φ<sub>u</sub>(x) ∈ R<sup>d</sup> of image pixels without annotation. Our starting point was the *Dense Equivariant Labelling* approach of [3] (references follow at the end of the README), which similarly tackles the same problem, but is restricted to learning low-dimensional embeddings to achieve inter-instance generalisation.  The key focus of *Descriptor Vector Exchange (DVE)* is to address this dimensionality issue to enable the learning of more powerful, higher dimensional embeddings while still preserving generalisation. To do so, we take inspiration from methods which enforce transitive/cyclic consistency constraints [4, 5, 6].

The embedding is learned from pairs of images (x,x′) related by a known warp v = g(u). In the image above, on the left we show the approach used by [3], which directly matches embedding Φ<sub>u</sub>(x) from the left image to embeddings Φ<sub>v</sub>(x′) in the right image. On the right, *DVE* replaces Φ<sub>u</sub>(x) with its reconstruction Φˆ<sub>u</sub>(x|xα) obtained from the embeddings in a third auxiliary image xα. Crucially, the correspondence with xα does not need to be known, but the process of learning in this manner encourages the embeddings to act consistently across different instances, even when the dimensionality is increased (see the paper for more details).


**Requirements:** The code assumes PyTorch 1.1 and Python 3.7 (other versions may work, but have not been tested).  See the section on dependencies towards the end of this file for specific package requirements.

### Datasets

In this work we use the following datasets:

**CelebA** is a dataset of over 200k faces of celebrities.  We use this dataset to train our embedding function without annotations. The dataset can be obtained [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and was originally described in [this paper](http://openaccess.thecvf.com/content_iccv_2015/papers/Liu_Deep_Learning_Face_ICCV_2015_paper.pdf). This dataset is implemented in the `CelebABase` class in [data_loaders.py](data_loader/data_loaders.py).

**MAFL** is a dataset of over 20k faces which includes landmark annotations.  The dataset is partitioned into 19k training images and 1k testing images.  We follow the protocol used in previous work [2], [3] (and described in more detail below) to evaluate the quality of the embeddings learned on CelebA.

**AFLW** is a dataset of faces which also includes landmark annotations. We use the P = 5 landmark test split (10,122 training images and 2,991 test images). The dataset can be obtained [here](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/) and is described in this [2011 ICCV workshop paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.384.2988&rep=rep1&type=pdf). This dataset is implemented in the `AFLW` class in [data_loaders.py](data_loader/data_loaders.py).

**300-W** This dataset contains 3,148 training images and 689 testing images with 68 facial landmark annotations for each face.  The dataset can be obtained [here](https://ibug.doc.ic.ac.uk/resources/300-W/) and is described in this [2013 ICCV workshop paper](https://www.cv-foundation.org/openaccess/content_iccv_workshops_2013/W11/papers/Sagonas_300_Faces_in-the-Wild_2013_ICCV_paper.pdf). 


### Learned Embeddings

We provide pretrained models for each dataset to reproduce the results reported in the paper [1]. Each model is accompanied by training and evaluation logs and its mean pixel error performance on the task of matching annotated landmarks across the MAFL test set.  The goal of these experiments is to demonstrate that DVE allows models to generalise across identities even when using higher dimensional embeddings (e.g. 64d rather than 3d).  Most of the embeddings are learned with the architecture described by [3].  We also learn one larger model using the hourglass architecture described in [7] (their code is available [here](https://github.com/YutingZhang/lmdis-rep)).

| Embedding Dim | Model | DVE | Same Identity | Different Identity | Links | 
| ------------- | :--:  | :-: | :----: | :----: | :----: |
|  3 | smallnet | :heavy_multiplication_x: | 1.33 | 2.89 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-3d/2019-08-04_17-55-48/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-3d/2019-08-04_17-55-48/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/celeba-smallnet-3d/2019-08-04_17-55-48/info.log) |
|  16 | smallnet | :heavy_multiplication_x: | 1.25 | 5.65 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-16d/2019-08-04_17-55-52/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-16d/2019-08-04_17-55-52/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/celeba-smallnet-16d/2019-08-04_17-55-52/info.log) |
|  32 | smallnet | :heavy_multiplication_x: | 1.26 | 5.81 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-32d/2019-08-04_17-55-57/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-32d/2019-08-04_17-55-57/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/celeba-smallnet-32d/2019-08-04_17-55-57/info.log) |
|  64 | smallnet | :heavy_multiplication_x: | 1.25 | 5.68 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-64d/2019-08-04_17-56-04/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-64d/2019-08-04_17-56-04/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/celeba-smallnet-64d/2019-08-04_17-56-04/info.log) |
|  3 | smallnet | :heavy_check_mark: | 1.36 | 3.03 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-3d-dve/2019-08-08_17-54-21/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-3d-dve/2019-08-08_17-54-21/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/celeba-smallnet-3d-dve/2019-08-08_17-54-21/info.log) |
|  16 | smallnet | :heavy_check_mark: | 1.28 | 2.79 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-16d-dve/2019-08-02_06-20-13/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-16d-dve/2019-08-02_06-20-13/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/celeba-smallnet-16d-dve/2019-08-02_06-20-13/info.log) |
|  32 | smallnet | :heavy_check_mark: | 1.29 | 2.79 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-32d-dve/2019-08-02_06-19-59/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-32d-dve/2019-08-02_06-19-59/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/celeba-smallnet-32d-dve/2019-08-02_06-19-59/info.log) |
|  64 | smallnet | :heavy_check_mark: | 1.28 | 2.77 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-64d-dve/2019-08-02_06-20-28/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-smallnet-64d-dve/2019-08-02_06-20-28/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/celeba-smallnet-64d-dve/2019-08-02_06-20-28/info.log) |
|  64 | hourglass | :heavy_check_mark: | 0.93 | 2.37 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-hourglass-64d-dve/0618_103501/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/celeba-hourglass-64d-dve/0618_103501/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/celeba-hourglass-64d-dve/0618_103501/info.log) |


**Notes**: The error metrics for the `hourglass` model are included for completeness, but are not exactly comparable to the performance of the smallnet due to slight differences in the cropping ratios used by the two architectures (0.3 for smallnet, 0.294 for Hourglass).  The numbers are normalised to account for the difference in input size, so they are approximately comparable.  Some of the logs are generated from existing logfiles that were created with a slightly older version of the codebase (these differences only affect the log format, rather than the training code itself - the log generator can be found [here](misc/update_deprecated_exps.py).) TODO(Samuel): Explain why IOD isn't used as a metric here.


### Landmark Regression

**Protocol Description**: To transform the learned dense embeddings into landmark predictions, we use the same approach as [3].  For each target dataset, we freeze the dense embeddings and learn to peg onto them a collection of 50 "virtual" keypoints via a spatial softmax (see the [regressor code](model/keypoint_prediction.py) for details).  These virtual keypoints are then used to regress the target keypoints of the dataset.

**MAFL landmark regression**

| Embedding Dim | Model | DVE | Inter-ocular distance | Links | 
| ------------- | :--:  | :-: | :----: | :----: |
|  3 | smallnet | :heavy_multiplication_x: | 4.02 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-3d/2019-08-11_08-24-51/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-3d/2019-08-11_08-24-51/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/mafl-keypoints-celeba-smallnet-3d/2019-08-11_08-24-51/info.log) |
|  16 | smallnet | :heavy_multiplication_x: | 5.31  | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-16d/2019-08-11_08-29-08/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-16d/2019-08-11_08-29-08/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/mafl-keypoints-celeba-smallnet-16d/2019-08-11_08-29-08/info.log) |
|  32 | smallnet | :heavy_multiplication_x: | 5.36  | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-32d/2019-08-11_08-25-14/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-32d/2019-08-11_08-25-14/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/mafl-keypoints-celeba-smallnet-32d/2019-08-11_08-25-14/info.log) |
|  64 | smallnet | :heavy_multiplication_x: | 4.99 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-64d/2019-08-08_17-56-47/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-64d/2019-08-08_17-56-47/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/mafl-keypoints-celeba-smallnet-64d/2019-08-08_17-56-47/info.log) |
|  3 | smallnet | :heavy_check_mark: | 4.17 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-3d-dve/2019-08-11_08-33-22/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-3d-dve/2019-08-11_08-33-22/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/mafl-keypoints-celeba-smallnet-3d-dve/2019-08-11_08-33-22/info.log) |
|  16 | smallnet | :heavy_check_mark: | 3.97 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-16d-dve/2019-08-11_08-29-31/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-16d-dve/2019-08-11_08-29-31/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/mafl-keypoints-celeba-smallnet-16d-dve/2019-08-11_08-29-31/info.log) |
|  32 | smallnet | :heavy_check_mark: | 3.82 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-32d-dve/2019-08-11_08-29-53/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-32d-dve/2019-08-11_08-29-53/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/mafl-keypoints-celeba-smallnet-32d-dve/2019-08-11_08-29-53/info.log) |
|  64 | smallnet | :heavy_check_mark: | 3.42 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-64d-dve/2019-08-11_08-40-48/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-64d-dve/2019-08-11_08-40-48/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/mafl-keypoints-celeba-smallnet-64d-dve/2019-08-11_08-40-48/info.log) |
|  64 | hourglass | :heavy_check_mark: | 2.86 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-hourglass-64d-dve/2019-08-11_14-30-53/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-hourglass-64d-dve/2019-08-11_14-30-53/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/mafl-keypoints-celeba-hourglass-64d-dve/2019-08-11_14-30-53/info.log) |

**AFLW landmark regression**

To compare with prior work, we evaluate the learned embeddings on AFLW under two settings: with and without finetuning the embedding on the AFLW dataset (without annotations). There are two slightly different partitions of AFLW that have been used in prior work.  One is a set of recropped faces released by [7] (here we simply call this AFLW). The second is the MTFL split of AFLW used in the works of [2], [3] (we call this split AFLW-MTFL).  


*Without finetuning on AFLW*

| Embedding Dim | Model | DVE | Inter-ocular distance | Links | 
| ------------- | :--:  | :-: | :----: | :----: |
|  3 | smallnet | :heavy_multiplication_x: | 9.69 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-keypoints-celeba-smallnet-3d/2019-08-10_07-54-09/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-keypoints-celeba-smallnet-3d/2019-08-10_07-54-09/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-keypoints-celeba-smallnet-3d/2019-08-10_07-54-09/info.log) |
|  16 | smallnet | :heavy_multiplication_x: | 11.05  | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-keypoints-celeba-smallnet-16d/2019-08-10_07-52-46/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-keypoints-celeba-smallnet-16d/2019-08-10_07-52-46/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-keypoints-celeba-smallnet-16d/2019-08-10_07-52-46/info.log) |
|  32 | smallnet | :heavy_multiplication_x: | 11.50  | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-keypoints-celeba-smallnet-32d/2019-08-10_09-19-46/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-keypoints-celeba-smallnet-32d/2019-08-10_09-19-46/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-keypoints-celeba-smallnet-32d/2019-08-10_09-19-46/info.log) |
|  64 | smallnet | :heavy_multiplication_x: | 11.68 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-keypoints-celeba-smallnet-64d/2019-08-10_10-50-57/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-keypoints-celeba-smallnet-64d/2019-08-10_10-50-57/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-keypoints-celeba-smallnet-64d/2019-08-10_10-50-57/info.log) |
|  3 | smallnet | :heavy_check_mark: | 9.65 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-keypoints-celeba-smallnet-3d-dve/2019-08-11_08-42-33/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-keypoints-celeba-smallnet-3d-dve/2019-08-11_08-42-33/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-keypoints-celeba-smallnet-3d-dve/2019-08-11_08-42-33/info.log) |
|  16 | smallnet | :heavy_check_mark: | 8.91 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-keypoints-celeba-smallnet-16d-dve/2019-08-10_12-53-44/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-keypoints-celeba-smallnet-16d-dve/2019-08-10_12-53-44/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-keypoints-celeba-smallnet-16d-dve/2019-08-10_12-53-44/info.log) |
|  32 | smallnet | :heavy_check_mark: | 8.73 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-keypoints-celeba-smallnet-32d-dve/2019-08-10_09-20-02/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-keypoints-celeba-smallnet-32d-dve/2019-08-10_09-20-02/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-keypoints-celeba-smallnet-32d-dve/2019-08-10_09-20-02/info.log) |
|  64 | smallnet | :heavy_check_mark: | 8.14 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-keypoints-celeba-smallnet-64d-dve/2019-08-10_12-55-24/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-keypoints-celeba-smallnet-64d-dve/2019-08-10_12-55-24/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-keypoints-celeba-smallnet-64d-dve/2019-08-10_12-55-24/info.log) |
|  64 | hourglass | :heavy_check_mark: | 6.88 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-keypoints-celeba-hourglass-64d-dve/2019-08-09_14-29-19/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-keypoints-celeba-hourglass-64d-dve/2019-08-09_14-29-19/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-keypoints-celeba-hourglass-64d-dve/2019-08-09_14-29-19/info.log) |

*With finetuning on AFLW*

First we fine-tune the embeddings for a fixed number of epochs:

| Embedding Dim | Model | DVE | Same Identity | Different Identity | Links | 
| ------------- | :--:  | :-: | :----: | :----: | :----: |
|  3 | smallnet | :heavy_multiplication_x: | 3.80 | 4.43 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-celeba-smallnet-3d/2019-08-10_12-50-40/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-celeba-smallnet-3d/2019-08-10_12-50-40/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-celeba-smallnet-3d/2019-08-10_12-50-40/info.log) |
|  16 | smallnet | :heavy_multiplication_x: | 1.77 | 9.42 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-celeba-smallnet-16d/2019-08-10_12-50-41/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-celeba-smallnet-16d/2019-08-10_12-50-41/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-celeba-smallnet-16d/2019-08-10_12-50-41/info.log) |
|  32 | smallnet | :heavy_multiplication_x: | 1.57 | 9.80 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-celeba-smallnet-32d/2019-08-10_12-50-43/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-celeba-smallnet-32d/2019-08-10_12-50-43/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-celeba-smallnet-32d/2019-08-10_12-50-43/info.log) |
|  64 | smallnet | :heavy_multiplication_x: | 1.26 | 5.89 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-celeba-smallnet-64d/2019-08-10_12-50-44/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-celeba-smallnet-64d/2019-08-10_12-50-44/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-celeba-smallnet-64d/2019-08-10_12-50-44/info.log) |
|  3 | smallnet | :heavy_check_mark: | 6.36 | 7.69 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-celeba-smallnet-3d-dve/2019-08-11_18-51-40/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-celeba-smallnet-3d-dve/2019-08-11_18-51-40/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-celeba-smallnet-3d-dve/2019-08-11_18-51-40/info.log) |
|  16 | smallnet | :heavy_check_mark: | 6.34 | 8.62 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-celeba-smallnet-16d-dve/2019-08-10_12-50-30/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-celeba-smallnet-16d-dve/2019-08-10_12-50-30/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-celeba-smallnet-16d-dve/2019-08-10_12-50-30/info.log) |
|  32 | smallnet | :heavy_check_mark: | 8.10 | 10.11 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-celeba-smallnet-32d-dve/2019-08-10_12-50-31/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-celeba-smallnet-32d-dve/2019-08-10_12-50-31/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-celeba-smallnet-32d-dve/2019-08-10_12-50-31/info.log) |
|  64 | smallnet | :heavy_check_mark: | 4.08 | 5.21 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-celeba-smallnet-64d-dve/2019-08-10_12-50-32/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-celeba-smallnet-64d-dve/2019-08-10_12-50-32/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-celeba-smallnet-64d-dve/2019-08-10_12-50-32/info.log) |
|  64 | hourglass | :heavy_check_mark: | 1.17 | 4.04 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-celeba-hourglass-64d-dve/2019-08-11_14-43-34/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-celeba-hourglass-64d-dve/2019-08-11_14-43-34/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-celeba-hourglass-64d-dve/2019-08-11_14-43-34/info.log) |


Then re-evaluate the performance of a learned landmark regressor:

| Embedding Dim | Model | DVE | Inter-ocular distance | Links | 
| ------------- | :--:  | :-: | :----: | :----: |
|  3 | smallnet | :heavy_multiplication_x: | 10.14 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-3d/2019-08-11_07-56-34/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-3d/2019-08-11_07-56-34/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-keypoints-celeba-smallnet-3d/2019-08-11_07-56-34/info.log) |
|  16 | smallnet | :heavy_multiplication_x: | 10.73  | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-16d/2019-08-11_07-56-41/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-16d/2019-08-11_07-56-41/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-keypoints-celeba-smallnet-16d/2019-08-11_07-56-41/info.log) |
|  32 | smallnet | :heavy_multiplication_x: | 11.05  | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-32d/2019-08-11_07-56-46/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-32d/2019-08-11_07-56-46/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-keypoints-celeba-smallnet-32d/2019-08-11_07-56-46/info.log) |
|  64 | smallnet | :heavy_multiplication_x: | 11.43 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-64d/2019-08-11_07-56-52/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-64d/2019-08-11_07-56-52/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-keypoints-celeba-smallnet-64d/2019-08-11_07-56-52/info.log) |
|  3 | smallnet | :heavy_check_mark: | TODO | [config](TODO), [model](TODO), [log](TODO) |
|  16 | smallnet | :heavy_check_mark: | 8.40 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-16d-dve/2019-08-11_07-57-00/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-16d-dve/2019-08-11_07-57-00/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-keypoints-celeba-smallnet-16d-dve/2019-08-11_07-57-00/info.log) |
|  32 | smallnet | :heavy_check_mark: | 8.18 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-32d-dve/2019-08-11_07-56-57/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-32d-dve/2019-08-11_07-56-57/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-keypoints-celeba-smallnet-32d-dve/2019-08-11_07-56-57/info.log) |
|  64 | smallnet | :heavy_check_mark: | 7.79 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-64d-dve/2019-08-11_18-53-30/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-ft-keypoints-celeba-smallnet-64d-dve/2019-08-11_18-53-30/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-ft-keypoints-celeba-smallnet-64d-dve/2019-08-11_18-53-30/info.log) |
|  64 | hourglass | :heavy_check_mark: | TODO | [config](TODO), [model](TODO), [log](TODO) |

**AFLW-MTFL landmark regression**

*Without finetuning on AFLW-MTFL*

| Embedding Dim | Model | DVE | Inter-ocular distance | Links | 
| ------------- | :--:  | :-: | :----: | :----: |
|  3 | smallnet | :heavy_multiplication_x: | 11.71 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-keypoints-celeba-smallnet-3d/2019-08-11_13-59-41/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-keypoints-celeba-smallnet-3d/2019-08-11_13-59-41/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-keypoints-celeba-smallnet-3d/2019-08-11_13-59-41/info.log) |
|  16 | smallnet | :heavy_multiplication_x: | 13.86  | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-keypoints-celeba-smallnet-16d/2019-08-11_13-59-36/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-keypoints-celeba-smallnet-16d/2019-08-11_13-59-36/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-keypoints-celeba-smallnet-16d/2019-08-11_13-59-36/info.log) |
|  32 | smallnet | :heavy_multiplication_x: | 14.04  | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-keypoints-celeba-smallnet-32d/2019-08-11_14-00-21/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-keypoints-celeba-smallnet-32d/2019-08-11_14-00-21/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-keypoints-celeba-smallnet-32d/2019-08-11_14-00-21/info.log) |
|  64 | smallnet | :heavy_multiplication_x: | 14.05 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-keypoints-celeba-smallnet-64d/2019-08-11_18-49-10/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-keypoints-celeba-smallnet-64d/2019-08-11_18-49-10/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-keypoints-celeba-smallnet-64d/2019-08-11_18-49-10/info.log) |
|  3 | smallnet | :heavy_check_mark: | 11.82 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-keypoints-celeba-smallnet-3d-dve/2019-08-11_14-02-25/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-keypoints-celeba-smallnet-3d-dve/2019-08-11_14-02-25/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-keypoints-celeba-smallnet-3d-dve/2019-08-11_14-02-25/info.log) |
|  16 | smallnet | :heavy_check_mark: | 10.22 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-keypoints-celeba-smallnet-16d-dve/2019-08-11_14-02-40/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-keypoints-celeba-smallnet-16d-dve/2019-08-11_14-02-40/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-keypoints-celeba-smallnet-16d-dve/2019-08-11_14-02-40/info.log) |
|  32 | smallnet | :heavy_check_mark: | 9.80 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-keypoints-celeba-smallnet-32d-dve/2019-08-11_14-02-56/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-keypoints-celeba-smallnet-32d-dve/2019-08-11_14-02-56/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-keypoints-celeba-smallnet-32d-dve/2019-08-11_14-02-56/info.log) |
|  64 | smallnet | :heavy_check_mark: | 9.28 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-keypoints-celeba-smallnet-64d-dve/2019-08-11_14-03-14/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-keypoints-celeba-smallnet-64d-dve/2019-08-11_14-03-14/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-keypoints-celeba-smallnet-64d-dve/2019-08-11_14-03-14/info.log) |
|  64 | hourglass | :heavy_check_mark: | 8.15 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-keypoints-celeba-hourglass-64d-dve/2019-08-11_18-49-34/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-keypoints-celeba-hourglass-64d-dve/2019-08-11_18-49-34/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-keypoints-celeba-hourglass-64d-dve/2019-08-11_18-49-34/info.log) |

*With finetuning on AFLW-MTFL*

First we fine-tune the embeddings for a fixed number of epochs:

| Embedding Dim | Model | DVE | Same Identity | Different Identity | Links | 
| ------------- | :--:  | :-: | :----: | :----: | :----: |
|  3 | smallnet | :heavy_multiplication_x: | 5.91 | 6.97 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-celeba-smallnet-3d/2019-08-11_08-19-48/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-celeba-smallnet-3d/2019-08-11_08-19-48/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-celeba-smallnet-3d/2019-08-11_08-19-48/info.log) |
|  16 | smallnet | :heavy_multiplication_x: | 1.63 | 9.68 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-celeba-smallnet-16d/2019-08-11_08-19-50/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-celeba-smallnet-16d/2019-08-11_08-19-50/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-celeba-smallnet-16d/2019-08-11_08-19-50/info.log) |
|  32 | smallnet | :heavy_multiplication_x: | 1.40 | 9.94 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-celeba-smallnet-32d/2019-08-11_08-19-52/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-celeba-smallnet-32d/2019-08-11_08-19-52/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-celeba-smallnet-32d/2019-08-11_08-19-52/info.log) |
|  64 | smallnet | :heavy_multiplication_x: | 1.32 | 9.42 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-celeba-smallnet-64d/2019-08-11_08-19-53/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-celeba-smallnet-64d/2019-08-11_08-19-53/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-celeba-smallnet-64d/2019-08-11_08-19-53/info.log) |
|  3 | smallnet | :heavy_check_mark: | 5.99 | 7.16 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-celeba-smallnet-3d-dve/2019-08-11_08-20-03/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-celeba-smallnet-3d-dve/2019-08-11_08-20-03/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-celeba-smallnet-3d-dve/2019-08-11_08-20-03/info.log) |
|  16 | smallnet | :heavy_check_mark: | 4.72 | 7.11 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-celeba-smallnet-16d-dve/2019-08-11_08-19-58/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-celeba-smallnet-16d-dve/2019-08-11_08-19-58/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-celeba-smallnet-16d-dve/2019-08-11_08-19-58/info.log) |
|  32 | smallnet | :heavy_check_mark: | 6.42 | 8.71 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-celeba-smallnet-32d-dve/2019-08-11_08-19-55/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-celeba-smallnet-32d-dve/2019-08-11_08-19-55/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-celeba-smallnet-32d-dve/2019-08-11_08-19-55/info.log) |
|  64 | smallnet | :heavy_check_mark: | 8.07 | 10.09 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-celeba-smallnet-64d-dve/2019-08-11_08-19-54/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-celeba-smallnet-64d-dve/2019-08-11_08-19-54/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-celeba-smallnet-64d-dve/2019-08-11_08-19-54/info.log) |
|  64 | hourglass | :heavy_check_mark: | 1.53 | 3.65 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-celeba-hourglass-64d-dve/2019-08-11_16-40-28/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-celeba-hourglass-64d-dve/2019-08-11_16-40-28/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-celeba-hourglass-64d-dve/2019-08-11_16-40-28/info.log) |


Then re-evaluate the performance of a learned landmark regressor:

| Embedding Dim | Model | DVE | Inter-ocular distance | Links | 
| ------------- | :--:  | :-: | :----: | :----: |
|  3 | smallnet | :heavy_multiplication_x: | 10.99 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-3d/2019-08-11_18-42-45/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-3d/2019-08-11_18-42-45/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-keypoints-celeba-smallnet-3d/2019-08-11_18-42-45/info.log) |
|  16 | smallnet | :heavy_multiplication_x: | 12.22  | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-16d/2019-08-11_18-43-03/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-16d/2019-08-11_18-43-03/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-keypoints-celeba-smallnet-16d/2019-08-11_18-43-03/info.log) |
|  32 | smallnet | :heavy_multiplication_x: | 12.60  | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-32d/2019-08-11_18-43-09/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-32d/2019-08-11_18-43-09/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-keypoints-celeba-smallnet-32d/2019-08-11_18-43-09/info.log) |
|  64 | smallnet | :heavy_multiplication_x: | 12.92 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-64d/2019-08-11_18-43-14/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-64d/2019-08-11_18-43-14/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-keypoints-celeba-smallnet-64d/2019-08-11_18-43-14/info.log) |
|  3 | smallnet | :heavy_check_mark: | 11.12 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-3d-dve/2019-08-11_18-43-20/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-3d-dve/2019-08-11_18-43-20/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-keypoints-celeba-smallnet-3d-dve/2019-08-11_18-43-20/info.log) |
|  16 | smallnet | :heavy_check_mark: | 9.15 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-16d-dve/2019-08-11_18-43-24/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-16d-dve/2019-08-11_18-43-24/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-keypoints-celeba-smallnet-16d-dve/2019-08-11_18-43-24/info.log) |
|  32 | smallnet | :heavy_check_mark: | 9.17 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-32d-dve/2019-08-11_18-43-30/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-32d-dve/2019-08-11_18-43-30/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-keypoints-celeba-smallnet-32d-dve/2019-08-11_18-43-30/info.log) |
|  64 | smallnet | :heavy_check_mark: | 8.60 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-64d-dve/2019-08-11_18-43-35/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/aflw-mtfl-ft-keypoints-celeba-smallnet-64d-dve/2019-08-11_18-43-35/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/aflw-mtfl-ft-keypoints-celeba-smallnet-64d-dve/2019-08-11_18-43-35/info.log) |
|  64 | hourglass | :heavy_check_mark: | TODO | [config](TODO), [model](TODO), [log](TODO) |


**300-W landmark regression**

*Without finetuning*

| Embedding Dim | Model | DVE | Inter-ocular distance | Links | 
| ------------- | :--:  | :-: | :----: | :----: |
|  3 | smallnet | :heavy_multiplication_x: | 8.23 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-3d/2019-08-11_14-50-45/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-3d/2019-08-11_14-50-45/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-keypoints-celeba-smallnet-3d/2019-08-11_14-50-45/info.log) |
|  16 | smallnet | :heavy_multiplication_x: | 10.66  | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-16d/2019-08-11_14-50-46/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-16d/2019-08-11_14-50-46/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-keypoints-celeba-smallnet-16d/2019-08-11_14-50-46/info.log) |
|  32 | smallnet | :heavy_multiplication_x: | 10.33  | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-32d/2019-08-11_14-50-47/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-32d/2019-08-11_14-50-47/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-keypoints-celeba-smallnet-32d/2019-08-11_14-50-47/info.log) |
|  64 | smallnet | :heavy_multiplication_x: | 9.33 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-64d/2019-08-11_14-50-48/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-64d/2019-08-11_14-50-48/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-keypoints-celeba-smallnet-64d/2019-08-11_14-50-48/info.log) |
|  3 | smallnet | :heavy_check_mark: | 7.66 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-3d-dve/2019-08-11_14-50-50/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-3d-dve/2019-08-11_14-50-50/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-keypoints-celeba-smallnet-3d-dve/2019-08-11_14-50-50/info.log) |
|  16 | smallnet | :heavy_check_mark: | 6.29 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-16d-dve/2019-08-11_14-50-52/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-16d-dve/2019-08-11_14-50-52/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-keypoints-celeba-smallnet-16d-dve/2019-08-11_14-50-52/info.log) |
|  32 | smallnet | :heavy_check_mark: | 6.13 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-32d-dve/2019-08-11_14-50-53/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-32d-dve/2019-08-11_14-50-53/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-keypoints-celeba-smallnet-32d-dve/2019-08-11_14-50-53/info.log) |
|  64 | smallnet | :heavy_check_mark: | 5.75 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-64d-dve/2019-08-11_14-50-54/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-smallnet-64d-dve/2019-08-11_14-50-54/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-keypoints-celeba-smallnet-64d-dve/2019-08-11_14-50-54/info.log) |
|  64 | hourglass | :heavy_check_mark: | 4.65 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-hourglass-64d-dve/2019-08-11_18-50-33/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-keypoints-celeba-hourglass-64d-dve/2019-08-11_18-50-33/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-keypoints-celeba-hourglass-64d-dve/2019-08-11_18-50-33/info.log) |

*With finetuning*

First we fine-tune the embeddings for a fixed number of epochs:

| Embedding Dim | Model | DVE | Same Identity | Different Identity | Links | 
| ------------- | :--:  | :-: | :----: | :----: | :----: |
|  3 | smallnet | :heavy_multiplication_x: | 6.28 | 7.10 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-celeba-smallnet-3d/2019-08-11_18-11-26/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-celeba-smallnet-3d/2019-08-11_18-11-26/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-ft-celeba-smallnet-3d/2019-08-11_18-11-26/info.log) |
|  16 | smallnet | :heavy_multiplication_x: | 1.52 | 9.09 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-celeba-smallnet-16d/2019-08-11_18-11-34/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-celeba-smallnet-16d/2019-08-11_18-11-34/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-ft-celeba-smallnet-16d/2019-08-11_18-11-34/info.log) |
|  32 | smallnet | :heavy_multiplication_x: | 1.37 | 9.04 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-celeba-smallnet-32d/2019-08-11_18-11-42/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-celeba-smallnet-32d/2019-08-11_18-11-42/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-ft-celeba-smallnet-32d/2019-08-11_18-11-42/info.log) |
|  64 | smallnet | :heavy_multiplication_x: | TODO | TODO | [config](TODO), [model](TODO), [log](TODO) |
|  3 | smallnet | :heavy_check_mark: | 5.21 | 6.51 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-celeba-smallnet-3d-dve/2019-08-11_18-11-57/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-celeba-smallnet-3d-dve/2019-08-11_18-11-57/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-ft-celeba-smallnet-3d-dve/2019-08-11_18-11-57/info.log) |
|  16 | smallnet | :heavy_check_mark: | 5.55 | 7.30 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-celeba-smallnet-16d-dve/2019-08-11_18-12-04/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-celeba-smallnet-16d-dve/2019-08-11_18-12-04/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-ft-celeba-smallnet-16d-dve/2019-08-11_18-12-04/info.log) |
|  32 | smallnet | :heavy_check_mark: | 5.85 | 7.47 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-celeba-smallnet-32d-dve/2019-08-11_18-12-15/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-celeba-smallnet-32d-dve/2019-08-11_18-12-15/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-ft-celeba-smallnet-32d-dve/2019-08-11_18-12-15/info.log) |
|  64 | smallnet | :heavy_check_mark: | 6.58 | 8.19 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-celeba-smallnet-64d-dve/2019-08-11_18-12-24/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-celeba-smallnet-64d-dve/2019-08-11_18-12-24/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-ft-celeba-smallnet-64d-dve/2019-08-11_18-12-24/info.log) |
|  64 | hourglass | :heavy_check_mark: | 1.63 | 3.82 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-celeba-hourglass-64d-dve/2019-08-11_12-57-08/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-celeba-hourglass-64d-dve/2019-08-11_12-57-08/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-ft-celeba-hourglass-64d-dve/2019-08-11_12-57-08/info.log) |

Then re-evaluate the performance of a learned landmark regressor:

| Embedding Dim | Model | DVE | Inter-ocular distance | Links | 
| ------------- | :--:  | :-: | :----: | :----: |
|  3 | smallnet | :heavy_multiplication_x: | TODO | [config](TODO), [model](TODO), [log](TODO) |
|  16 | smallnet | :heavy_multiplication_x: | TODO  | [config](TODO), [model](TODO), [log](TODO) |
|  32 | smallnet | :heavy_multiplication_x: | TODO  | [config](TODO), [model](TODO), [log](TODO) |
|  64 | smallnet | :heavy_multiplication_x: | TODO | [config](TODO), [model](TODO), [log](TODO) |
|  3 | smallnet | :heavy_check_mark: | TODO | [config](TODO), [model](TODO), [log](TODO) |
|  16 | smallnet | :heavy_check_mark: | TODO | [config](TODO), [model](TODO), [log](TODO) |
|  32 | smallnet | :heavy_check_mark: | TODO | [config](TODO), [model](TODO), [log](TODO) |
|  64 | smallnet | :heavy_check_mark: | TODO | [config](TODO), [model](TODO), [log](TODO) |
|  64 | hourglass | :heavy_check_mark: | 4.65 | [config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-keypoints-celeba-hourglass-64d-dve/2019-08-11_18-42-44/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/300w-ft-keypoints-celeba-hourglass-64d-dve/2019-08-11_18-42-44/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/300w-ft-keypoints-celeba-hourglass-64d-dve/2019-08-11_18-42-44/info.log) |

### Ablation Studies

We can study the effect of the DVE method by removing it during training and assessing the resulting embeddings for landmark regression.  The ablations are performed on the SmallNet (because it's much faster to train).

| Embedding Dim | Model | DVE | Error (%IOD) | Links | 
| ------------- | :--:  | :-: | :----: | :----: |
|  3 | smallnet | :heavy_multiplication_x: / :heavy_check_mark: | 4.02/4.17 | ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-3d-dve/2019-08-11_08-33-22/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-3d-dve/2019-08-11_08-33-22/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/mafl-keypoints-celeba-smallnet-3d-dve/2019-08-11_08-33-22/info.log)) / ([config](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-3d-dve/2019-08-11_08-33-22/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/models/mafl-keypoints-celeba-smallnet-3d-dve/2019-08-11_08-33-22/model_best.pth), [log](http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/log/mafl-keypoints-celeba-smallnet-3d-dve/2019-08-11_08-33-22/info.log)) |

### Notes

TODO(Samuuel): Explain why some logs are v. slow compared to others, why some are generated.


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

### References

[1] James Thewlis, Samuel Albanie, Hakan Bilen, and Andrea Vedaldi. "Unsupervised learning of landmarks by exchanging descriptor vectors" ICCV 2019.

[2] James Thewlis, Hakan Bilen and Andrea Vedaldi, "Unsupervised learning of object landmarks by factorized spatial embeddings." ICCV 2017.

[3] James Thewlis, Hakan Bilen and Andrea Vedaldi, "Unsupervised learning of object frames by dense equivariant image labelling." NeurIPS 2017

[4] Sundaram, N., Brox, T., & Keutzer, K. "Dense point trajectories by GPU-accelerated large displacement optical flow", ECCV 2010

[5] C. Zach, M. Klopschitz, and M. Pollefeys. "Disambiguating visual relations using loop constraints", CVPR, 2010

[6] Zhou, T., Jae Lee, Y., Yu, S. X., & Efros, A. A. "Flowweb: Joint image set alignment by weaving consistent, pixel-wise correspondences". CVPR 2015.

[7] Zhang, Yuting, Yijie Guo, Yixin Jin, Yijun Luo, Zhiyuan He, and Honglak Lee. "Unsupervised discovery of object landmarks as structural representations.", CVPR 2018

### Acknowledgements


We would like to thank [Tom Jakab](http://www.robots.ox.ac.uk/~tomj/) for sharing code.  The project structure uses the [pytorch-template](https://github.com/victoresque/pytorch-template) by @victoresque.