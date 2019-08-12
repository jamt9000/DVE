## 300w

This folder contains a copy of the 300w dataset [1] to help reproduce the results of the paper:

*James Thewlis, Samuel Albanie, Hakan Bilen, Andrea Vedaldi. Unsupervised learning of landmarks via vector exchange. ICCV 2019*

**Tar contents**

The folder contains two forms of the CelebA images (one set of higher quality images, and
one copy at slightly lower quality).  In practice, the higher image qualities made no
discernible difference to performance for embedding learning, but they were used in one set of experiments so they are included here for completeness.

The compressed tar file (3.0 GiB) can be downloaded from:

```
http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/datasets/300w.tar.gz
sha1sum: 885b09159c61fa29998437747d589c65cfc4ccd3
```
A list of the contents of the tar file are given in [tar_include.txt](tar_include.txt).


The original dataset can also be downloaded from the link below:

* **300-W:** https://ibug.doc.ic.ac.uk/resources/300-W/

### References:

[1] If you use the 300w dataset, please cite:
```
@inproceedings{sagonas2013300,
  title={300 faces in-the-wild challenge: The first facial landmark localization challenge},
  author={Sagonas, Christos and Tzimiropoulos, Georgios and Zafeiriou, Stefanos and Pantic, Maja},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision Workshops},
  pages={397--403},
  year={2013}
}
```