## CelebA

This folder contains a copy of the CelebA dataset [1], together with the MAFL subset splits [2]
to help reproduce the results of the paper:

*James Thewlis, Samuel Albanie, Hakan Bilen, Andrea Vedaldi. Unsupervised learning of landmarks via vector exchange. ICCV 2019*

**Tar contents**

The folder contains two forms of the CelebA images (one set of higher quality images, and
one copy at slightly lower quality).  In practice, the higher image qualities made no
discernible difference to performance for embedding learning, but they were used in one set of experiments so they are included here for completeness.

The compressed tar file (TODO GiB) can be downloaded from:

```
http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/datasets/celeba.tar.gz
sha1sum: TODO
```
A list of the contents of the tar file are given in [tar_include.txt](tar_include.txt).


The original dataset and the MAFL splits can be downloaded from the links below:

* **CelebA:** http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
* **MAFL:** https://github.com/zhzhanp/TCDCN-face-alignment/tree/master/MAFL

### References:

[1] If you use the CelebA dataset, please cite:
```
@inproceedings{liu2015faceattributes,
 title = {Deep Learning Face Attributes in the Wild},
 author = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
 booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
 month = {December},
 year = {2015} 
}
```

[2] If you use the MAFL subset of CelebA, please additionally cite:

```
@article{zhang2015learning,
  title={Learning deep representation for face alignment with auxiliary attributes},
  author={Zhang, Zhanpeng and Luo, Ping and Loy, Chen Change and Tang, Xiaoou},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={38},
  number={5},
  pages={918--930},
  year={2015},
  publisher={IEEE}
}
```