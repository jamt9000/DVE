## AFLW-recrop

This folder contains a copy of the AFLW dataset [1], recropped and released by [2] to help reproduce the results of the paper:

*James Thewlis, Samuel Albanie, Hakan Bilen, Andrea Vedaldi. Unsupervised learning of landmarks via vector exchange. ICCV 2019*

**Tar contents**

TODO explain this split.


The compressed tar file (1.1 GiB) can be downloaded from:

```
http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/datasets/aflw-recrop.tar.gz
sha1sum: 939fdce0e6262a14159832c71d4f84a9d516de5e
```
A list of the contents of the tar file are given in [tar_include.txt](tar_include.txt).


The original datasets can also be downloaded from the links below:

* **AFLW** https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/
* **preprocessed AFLW images** http://files.ytzhang.net/lmdis-rep/release-v1/aflw/aflw_data.tar.gz

### References:
If you use the AFLW dataset, in particular with MTFL split, please cite the following papers:

```
@inproceedings{koestinger2011annotated,
  title={Annotated facial landmarks in the wild: A large-scale, real-world database for facial landmark localization},
  author={Koestinger, Martin and Wohlhart, Paul and Roth, Peter M and Bischof, Horst},
  booktitle={2011 IEEE international conference on computer vision workshops (ICCV workshops)},
  pages={2144--2151},
  year={2011},
  organization={IEEE}
}
```
```
@inproceedings{zhang2018unsupervised,
  title={Unsupervised discovery of object landmarks as structural representations},
  author={Zhang, Yuting and Guo, Yijie and Jin, Yixin and Luo, Yijun and He, Zhiyuan and Lee, Honglak},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={2694--2703},
  year={2018}
}
```