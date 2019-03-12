from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch
from utils import tps

import numpy as np
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


def kp_normalize(H, W, kp):
    kp = kp.clone()
    kp[..., 0] = 2. * kp[..., 0] / (W - 1) - 1
    kp[..., 1] = 2. * kp[..., 1] / (H - 1) - 1
    return kp


class CelebAPrunedAligned_MAFLVal(Dataset):
    eye_kp_idxs = [0, 1]

    def __init__(self, root, train=True, pair_warper=None, imwidth=100):
        self.root = root
        self.imwidth = imwidth
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

        # lefteye_x lefteye_y ; righteye_x righteye_y ; nose_x nose_y ; leftmouth_x leftmouth_y ; rightmouth_x rightmouth_y
        self.keypoints = np.array(self.data, dtype=np.float32).reshape(-1, 5, 2)

        self.filenames = list(self.data.index)

        # Move head up a bit
        initial_crop = lambda im: transforms.functional.crop(im, 30, 0, 178, 178)
        self.keypoints[:, :, 1] -= 30
        self.keypoints *= self.imwidth / 178.

        normalize = transforms.Normalize(mean=[0.5084, 0.4224, 0.3769], std=[0.2599, 0.2371, 0.2323])
        augmentations = [transforms.transforms.ColorJitter(.4, .4, .4),
                         transforms.ToTensor(), PcaAug()] if train else [transforms.ToTensor()]
        self.transforms = transforms.Compose(
            [initial_crop,
             transforms.Resize(self.imwidth)]
            + augmentations +
            [normalize])

    def __getitem__(self, index):
        im = Image.open(os.path.join(self.root, 'Img', 'img_align_celeba', self.filenames[index]))
        kp = self.keypoints[index]

        if self.warper is not None:
            if self.warper.returns_pairs:
                im1 = self.transforms(im)
                im2 = self.transforms(im)

                im1, im2, flow, grid, kp1, kp2 = self.warper(im1, im2, keypts=kp)
                C, H, W = im1.shape
                data = torch.stack((im1, im2), 0)
                meta = {'flow': flow[0], 'grid': grid[0], 'kp1': kp1, 'kp2': kp2}
            else:
                im1 = self.transforms(im)
                im1, kp = self.warper(im1, keypts=kp)
                C, H, W = im1.shape
                data = im1
                meta = {'keypts': kp, 'keypts_normalized': kp_normalize(H, W, kp)}

        else:
            data = self.transforms(im)
            C, H, W = data.shape
            meta = {'keypts': kp, 'keypts_normalized': kp_normalize(H, W, kp)}

        return data, meta

    def __len__(self):
        return len(self.data.index)


if __name__ == '__main__':
    import pylab

    dataset = CelebAPrunedAligned_MAFLVal('data/celeba', True, pair_warper=tps.Warper(100, 100))

    x, meta = dataset[6]
    print(x[0].shape)
    pylab.imshow(x[0].permute(1, 2, 0) + 0.5)
    pylab.figure()
    pylab.imshow(x[1].permute(1, 2, 0) + 0.5)
    pylab.show()
