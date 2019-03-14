from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch
from utils import tps

import numpy as np
import pandas as pd
import os
from PIL import Image

from io import BytesIO

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


class CelebABase(Dataset):
    def __getitem__(self, index):
        im = Image.open(os.path.join(self.root, 'Img', 'img_align_celeba', self.filenames[index]))
        kp = self.keypoints[index].copy()

        if self.warper is not None:
            if self.warper.returns_pairs:
                im1 = self.initial_transforms(im)
                im1 = TF.to_tensor(im1)*255

                im1, im2, flow, grid, kp1, kp2 = self.warper(im1, keypts=kp, crop=self.crop)

                im1 = im1.to(torch.uint8)
                im2 = im2.to(torch.uint8)

                C,H,W = im1.shape

                im1 = TF.to_pil_image(im1)
                im2 = TF.to_pil_image(im2)

                rW = max(int(0.8*W),int(W * (1+0.5*torch.randn([]))))
                im1 = TF.resize(im1, (rW,rW))
                buf = BytesIO()
                im1.save(buf, format='JPEG', quality=torch.randint(50, 99, []).item())
                im1 = Image.open(buf)


                rW = max(int(0.8*W),int(W * (1+0.5*torch.randn([]))))
                im2 = TF.resize(im2, (rW,rW))
                buf = BytesIO()
                im2.save(buf, format='JPEG', quality=torch.randint(50, 99, []).item())
                im2 = Image.open(buf)

                im1 = TF.resize(im1, (H,W))
                im2 = TF.resize(im2, (H,W))

                im1 = self.transforms(im1)
                im2 = self.transforms(im2)

                C, H, W = im1.shape
                data = torch.stack((im1, im2), 0)
                meta = {'flow': flow[0], 'grid': grid[0], 'kp1': kp1, 'kp2': kp2}
            else:
                im1 = self.initial_transforms(im)
                im1 = TF.to_tensor(im1)*255

                im1, kp = self.warper(im1, keypts=kp, crop=self.crop)

                im1 = im1.to(torch.uint8)
                im1 = TF.to_pil_image(im1)
                im1 = self.transforms(im1)

                C, H, W = im1.shape
                data = im1
                meta = {'keypts': kp, 'keypts_normalized': kp_normalize(H, W, kp)}

        else:
            data = self.transforms(self.initial_transforms(im))

            if self.crop != 0:
                data = data[:, self.crop:-self.crop, self.crop:-self.crop]
                kp = kp - self.crop
                kp = torch.tensor(kp)

            C, H, W = data.shape
            meta = {'keypts': kp, 'keypts_normalized': kp_normalize(H, W, kp)}

        return data, meta



class CelebAPrunedAligned_MAFLVal(CelebABase):
    eye_kp_idxs = [0, 1]

    def __init__(self, root, train=True, pair_warper=None, imwidth=100, crop=18, do_augmentations=True):
        self.root = root
        self.imwidth = imwidth
        self.train = train
        self.warper = pair_warper
        self.crop = crop

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
                         transforms.ToTensor(), PcaAug()] if (train and do_augmentations) else [transforms.ToTensor()]

        self.initial_transforms = transforms.Compose([initial_crop, transforms.Resize(self.imwidth)])
        self.transforms = transforms.Compose( augmentations + [normalize])

    def __len__(self):
        return len(self.data.index)


class MAFLAligned(CelebABase):
    eye_kp_idxs = [0, 1]

    def __init__(self, root, train=True, pair_warper=None, imwidth=100, crop=18, do_augmentations=True):
        self.root = root
        self.imwidth = imwidth
        self.train = train
        self.warper = pair_warper
        self.crop = crop

        anno = pd.read_csv(os.path.join(root, 'Anno', 'list_landmarks_align_celeba.txt'), header=1,
                           delim_whitespace=True)
        assert len(anno.index) == 202599
        split = pd.read_csv(os.path.join(root, 'Eval', 'list_eval_partition.txt'), header=None,
                            delim_whitespace=True, index_col=0)
        assert len(split.index) == 202599

        mafltest = pd.read_csv(os.path.join(root, 'MAFL', 'testing.txt'), header=None, delim_whitespace=True,
                               index_col=0)
        split.loc[mafltest.index] = 4

        mafltrain = pd.read_csv(os.path.join(root, 'MAFL', 'training.txt'), header=None, delim_whitespace=True,
                               index_col=0)
        split.loc[mafltrain.index] = 5

        assert (split[1] == 4).sum() == 1000
        assert (split[1] == 5).sum() == 19000

        if train:
            self.data = anno.loc[split[split[1] == 5].index]
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
                         transforms.ToTensor(), PcaAug()] if (train and do_augmentations) else [transforms.ToTensor()]

        self.initial_transforms = transforms.Compose([initial_crop, transforms.Resize(self.imwidth)])
        self.transforms = transforms.Compose(augmentations + [normalize])

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
