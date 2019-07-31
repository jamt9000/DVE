# Descriptor Vector Exchange


This repo provides code for learning dense landmarks with supervision.  Our approach is described in the ICCV 2019 paper "Unsupervised learning of landmarks by exchanging descriptor vectors" ([link]()).


![CE diagram](figs/DVE.png)


**Requirements:** The code assumes PyTorch 1.1 and Python 3.7 (other versions may work, but have not been tested).  See the section on dependencies towards the end of this file for specific package requirements.

### Datasets

In this work we use the following datasets:

**CelebA** is a dataset of over 200k faces of celebrities.  We use this dataset to train our embedding function without annotations. The dataset can be obtained [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and was originally described in [this paper](http://openaccess.thecvf.com/content_iccv_2015/papers/Liu_Deep_Learning_Face_ICCV_2015_paper.pdf). This dataset is implemented in the `CelebABase` class in [data_loaders.py](data_loader/data_loaders.py).

**MAFL** is a dataset of over 20k faces which includes landmark annotations.  The dataset is partitioned into 19k training images and 1k testing images.  We follow the protocol used in previous work [2], [3] (and described in more detail below) to evaluate the quality of the embeddings learned on CelebA.

**AFLW** is a dataset. We use the P = 5 landmark test split. The dataset can be obtained [here]() and is described in this [2011 ICCV workshop paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.384.2988&rep=rep1&type=pdf). This dataset is implemented in the `AFLW` class in [data_loaders.py](data_loader/data_loaders.py).


**300-W** This dataset contains 3,158 training images and 689 testing images with 68 facial landmark annotations for each face.  The dataset can be obtained [here](https://ibug.doc.ic.ac.uk/resources/300-W/) and is described in this [2013 ICCV workshop paper](https://www.cv-foundation.org/openaccess/content_iccv_workshops_2013/W11/papers/Sagonas_300_Faces_in-the-Wild_2013_ICCV_paper.pdf). 

**CelebA Pixel Matching**

| Embedding Dim | DVE | Same Identity | Different Identity | Links | 
| ------------- | :-: | :----: | :----: | :----: |
|  3 | :heavy_multiplication_x: | {{celeba-smallnet-3d.same-identity}} | {{celeba-smallnet-3d.different-identity}} | [config]({{celeba-smallnet-3d.config}}), [model]({{celeba-smallnet-3d.model}}), [log]({{celeba-smallnet-3d.log}}) |
|  3 | :heavy_multiplication_x: | {{celeba-smallnet-3d-dve.same-identity}} | {{celeba-smallnet-3d-dve.different-identity}} | [config]({{celeba-smallnet-3d-dve.config}}), [model]({{celeba-smallnet-3d-dve.model}}), [log]({{celeba-smallnet-3d-dve.log}}) |
|  16 | :heavy_multiplication_x: | {{celeba-smallnet-16d.same-identity}} | {{celeba-smallnet-16d.different}} | [config]({{celeba-smallnet-16d.config}}), [model]({{celeba-smallnet-16d.model}}), [log]({{celeba-smallnet-16d.log}}) |
|  16 | :heavy_multiplication_x: | {{celeba-smallnet-16d-dve.same-identity}} | {{celeba-smallnet-16d-dve.different-identity}} | [config]({{celeba-smallnet-16d-dve.config}}), [model]({{celeba-smallnet-16d-dve.model}}), [log]({{celeba-smallnet-16d-dve.log}}) |
|  64 | :heavy_multiplication_x: | {{celeba-smallnet-64d.same-identity}} | {{celeba-smallnet-64d.different}} | [config]({{celeba-smallnet-64d.config}}), [model]({{celeba-smallnet-64d.model}}), [log]({{celeba-smallnet-64d.log}}) |
|  64 | :heavy_multiplication_x: | {{celeba-smallnet-64d-dve.same-identity}} | {{celeba-smallnet-64d-dve.different-identity}} | [config]({{celeba-smallnet-64d-dve.config}}), [model]({{celeba-smallnet-64d-dve.model}}), [log]({{celeba-smallnet-64d-dve.log}}) |

<!-- |  16 | :heavy_multiplication_x: | TODO| TODO |[config](), [model](), [log]()
|  64 | :heavy_multiplication_x: | TODO| TODO |[config](), [model](), [log]()
|  3 | :heavy_check_mark: | TODO| TODO |[config](), [model](), [log]()
|  16 | :heavy_check_mark: | TODO| TODO |[config](), [model](), [log]()
|  64 | :heavy_check_mark: | TODO| TODO |[config](), [model](), [log]() -->


### Landmark Regression


**Protocol Description**: TODO (Train on 19k test on 1k).


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

[1] James Thewlis, Samuel Albanie, Hakan Bilen, and Andrea Vedaldi. "Unsupervised learning of landmarks by exchanging descriptor vectors" Proceedings of the IEEE International Conference on Computer Vision. 2019.

[2] James Thewlis, Hakan Bilen and Andrea Vedaldi, "Unsupervised learning of object landmarks by factorized spatial embeddings." Proceedings of the IEEE International Conference on Computer Vision. 2017.

[3] James Thewlis, Hakan Bilen and Andrea Vedaldi, "Unsupervised learning of object frames by dense equivariant image labelling." Advances in Neural Information Processing Systems. 201


### Acknowledgements


We would like to thank [Tom Jakab](http://www.robots.ox.ac.uk/~tomj/) for sharing code.  The project structure uses the [pytorch-template](https://github.com/victoresque/pytorch-template) by @victoresque.
