## AFLW-MTFL

This folder contains a copy of the AFLW-mtfl dataset [1], [2] to help reproduce the results of the paper:

*James Thewlis, Samuel Albanie, Hakan Bilen, Andrea Vedaldi. Unsupervised learning of landmarks via vector exchange. ICCV 2019*

**Tar contents**

The original AFLW contains around 25k images with up to 21 landmarks. For the purposes of evaluating five-landmark detectors, the authors of [TCDCN](http://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html) introduced a test subset of almost 3K faces.


The compressed tar file (252 MiB) can be downloaded from:

```
http:/www.robots.ox.ac.uk/~vgg/research/DVE/data/datasets/aflw-mtfl.tar.gz
sha1sum: 885b09159c61fa29998437747d589c65cfc4ccd3
```
A list of the contents of the tar file are given in [tar_include.txt](tar_include.txt).


The original datasets can also be downloaded from the links below:

* **AFLW** https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/
* **MTFL cropped images** http://mmlab.ie.cuhk.edu.hk/projects/TCDCN/data/MTFL.zip

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
@inproceedings{zhang2014facial,
  title={Facial landmark detection by deep multi-task learning},
  author={Zhang, Zhanpeng and Luo, Ping and Loy, Chen Change and Tang, Xiaoou},
  booktitle={European conference on computer vision},
  pages={94--108},
  year={2014},
  organization={Springer}
}
```