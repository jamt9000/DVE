# Descriptor Vector Exchange


This repo provides code for learning dense landmarks with supervision.  Our approach is described in the ICCV 2019 paper "Unsupervised learning of landmarks by exchanging descriptor vectors" ([link]()).


![CE diagram](figs/DVE.png)


**Requirements:** The code assumes PyTorch 1.1 and Python 3.7 (other versions may work, but have not been tested).  See the section on dependencies towards the end of this file for specific package requirements.

### Datasets

In this work we use the following datasets:

**CelebA** is a dataset of over 200k faces of celebrities.  We use this dataset to train our embedding function without annotations. The dataset can be obtained [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and was originally described in [this paper](http://openaccess.thecvf.com/content_iccv_2015/papers/Liu_Deep_Learning_Face_ICCV_2015_paper.pdf).

**MAFL** is a dataset of over 20k faces which includes landmark annotations.  The dataset is partitioned into 19k training images and 1k testing images.  We follow the protocol used in previous work [2], [3] (and described in more detail below) to evaluate the quality of the embeddings learned on CelebA.

**AFLW** TODO. 

**CelebA Pixel Matching**

| Embedding Dim | DVE | Same Identity | Different Identity | Links | 
| ------------- | :-: | :----: | :----: | :----: |
|  3 | :heavy_multiplication_x: | TODO| TODO |[config](), [model](), [log]()
|  16 | :heavy_multiplication_x: | TODO| TODO |[config](), [model](), [log]()
|  64 | :heavy_multiplication_x: | TODO| TODO |[config](), [model](), [log]()
|  3 | :heavy_check_mark: | TODO| TODO |[config](), [model](), [log]()
|  16 | :heavy_check_mark: | TODO| TODO |[config](), [model](), [log]()
|  64 | :heavy_check_mark: | TODO| TODO |[config](), [model](), [log]()


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
