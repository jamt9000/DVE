from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch
from utils import tps

import pandas as pd
import os
from PIL import Image


class PcaAug(object):
    _eigval = torch.Tensor([0.2175, 0.0188, 0.0045])
    _eigvec = torch.Tensor([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def __call__(self, im):
        alpha = torch.randn(3) * self.alpha
        rgb = (self._eigvec * alpha.expand(3, 3) * self._eigval.expand(3, 3)).sum(1)
        return im + rgb.reshape(3, 1, 1)


class CelebAPrunedAligned_MAFLVal(Dataset):
    def __init__(self, root, train=True, pair_warper=None):
        self.root = root
        self.imwidth = 100
        self.train = train
        self.warper = pair_warper

        anno = pd.read_csv(os.path.join(root, 'Anno', 'list_landmarks_align_celeba.txt'), header=1,
                           delim_whitespace=True)
        assert len(anno.index) == 202599
        split = pd.read_csv(os.path.join(root, 'Eval', 'list_eval_partition.txt'), header=None,
                            delim_whitespace=True, index_col=0)
        assert len(split.index) == 202599

        mafltest = pd.read_csv(os.path.join(root, 'MAFL', 'testing.txt'), header=None, delim_whitespace=True,
                               index_col=0)
        split.loc[mafltest.index] = 4
        assert (split[1] == 4).sum() == 1000

        if train:
            self.data = anno.loc[split[split[1] == 0].index]
        else:
            self.data = anno.loc[split[split[1] == 4].index]

        self.filenames = self.data.index.to_list()

        normalize = transforms.Normalize(mean=[0.5084, 0.4224, 0.3769], std=[0.2599, 0.2371, 0.2323])
        augmentations = [transforms.transforms.ColorJitter(.4, .4, .4),
                         transforms.ToTensor(), PcaAug()] if train else [transforms.ToTensor()]
        self.transforms = transforms.Compose(
            [transforms.Resize(self.imwidth),
             transforms.CenterCrop(self.imwidth)]
            + augmentations +
            [normalize])

    def __getitem__(self, index):
        im = Image.open(os.path.join(self.root, 'Img', 'img_align_celeba', self.filenames[index]))

        if self.warper is not None:
            im1 = self.transforms(im)
            im2 = self.transforms(im)

            res = self.warper(im1, im2)
        else:
            res = self.transforms(im),

        return res

    def __len__(self):
        return len(self.data.index)


if __name__ == '__main__':
    import pylab

    dataset = CelebAPrunedAligned_MAFLVal('/scratch/shared/nfs1/jdt/celeba', True, pair_warper=tps.Warper(100,100))

    x = dataset[123]
    print(x[0].shape)
    pylab.imshow(x[0].permute(1, 2, 0) + 0.5)
    pylab.figure()
    pylab.imshow(x[1].permute(1, 2, 0) + 0.5)
    pylab.show()
