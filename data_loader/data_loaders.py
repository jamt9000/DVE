import numpy as np
import pandas as pd
import time
import os
from PIL import Image
from utils import tps
import torch
from os.path import join as pjoin
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data.dataset import Dataset


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


class JPEGNoise(object):
    def __init__(self, low=30, high=99):
        self.low = low
        self.high = high

    def __call__(self, im):
        H = im.height
        W = im.width
        rW = max(int(0.8 * W), int(W * (1 + 0.5 * torch.randn([]))))
        im = TF.resize(im, (rW, rW))
        buf = BytesIO()
        im.save(buf, format='JPEG', quality=torch.randint(self.low, self.high,
                                                          []).item())
        im = Image.open(buf)
        im = TF.resize(im, (H, W))
        return im


def kp_normalize(H, W, kp):
    kp = kp.clone()
    kp[..., 0] = 2. * kp[..., 0] / (W - 1) - 1
    kp[..., 1] = 2. * kp[..., 1] / (H - 1) - 1
    return kp


class CachedDataset(Dataset):
    eye_kp_idxs = [0, 1]

    def __init__(self, root, imwidth, train=True, pair_warper=None):

        self.root = root
        self.train = train

        # sanity check on loaded feature path
        template = "imwidth{}".format(imwidth)
        msg = "expected {} to occur in path to cached features".format(template)
        assert template in root, msg

        if train:
            feat_path = pjoin(root, "train-feats.npy")
            kpts_path = pjoin(root, "train-kpts.npy")
        else:
            feat_path = pjoin(root, "val-feats.npy")
            kpts_path = pjoin(root, "val-kpts.npy")

        print("loading feature cache from disk....")
        tic = time.time()
        self.feats = np.load(feat_path)
        print("done in {:.3f}s".format(time.time() - tic))

        # flatten kpts store
        kpts_store = np.load(kpts_path)
        self.keypts = np.vstack([x["keypts"] for x in kpts_store])
        self.keypts_normalized = np.vstack([x["keypts_normalized"] for x in kpts_store])
        self.index = np.hstack([x["index"] for x in kpts_store])

        for meta_name in ("keypts", "keypts_normalized", "index"):
            msg = "expected dataset size of {}, found {}"
            msg = msg.format(self.feats.shape[0], getattr(self, meta_name).shape[0])
            assert self.feats.shape[0] == getattr(self, meta_name).shape[0], msg

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, index):
        im = self.feats[index]
        meta = {
            "keypts": self.keypts[index],
            "keypts_normalized": self.keypts_normalized[index],
            "index": self.index[index],
        }
        return torch.from_numpy(im), meta


class CelebABase(Dataset):
    def __getitem__(self, index):
        if self.use_hq_ims:
            subdir = "img_align_celeba_hq"
        else:
            subdir = "img_align_celeba"
        subdir = os.path.join(self.root, 'Img', subdir)
        im = Image.open(os.path.join(subdir, self.filenames[index]))
        kp = None
        if self.use_keypoints:
            kp = self.keypoints[index].copy()
        meta = {}

        if self.warper is not None:
            if self.warper.returns_pairs:
                im1 = self.initial_transforms(im)
                im1 = TF.to_tensor(im1) * 255

                im1, im2, flow, grid, kp1, kp2 = self.warper(im1, keypts=kp,
                                                             crop=self.crop)

                im1 = im1.to(torch.uint8)
                im2 = im2.to(torch.uint8)

                C, H, W = im1.shape

                im1 = TF.to_pil_image(im1)
                im2 = TF.to_pil_image(im2)

                im1 = self.transforms(im1)
                im2 = self.transforms(im2)

                C, H, W = im1.shape
                data = torch.stack((im1, im2), 0)
                meta = {
                    'flow': flow[0],
                    'grid': grid[0],
                    'im1': im1,
                    'im2': im2,
                    'index': index
                }
                if self.use_keypoints:
                    meta = {**meta, **{'kp1': kp1, 'kp2': kp2}}
            else:
                im1 = self.initial_transforms(im)
                im1 = TF.to_tensor(im1) * 255

                im1, kp = self.warper(im1, keypts=kp, crop=self.crop)

                im1 = im1.to(torch.uint8)
                im1 = TF.to_pil_image(im1)
                im1 = self.transforms(im1)

                C, H, W = im1.shape
                data = im1
                if self.use_keypoints:
                    meta = {
                        'keypts': kp,
                        'keypts_normalized': kp_normalize(H, W, kp),
                        'index': index
                    }

        else:
            data = self.transforms(self.initial_transforms(im))

            if self.crop != 0:
                data = data[:, self.crop:-self.crop, self.crop:-self.crop]
                kp = kp - self.crop
                kp = torch.tensor(kp)

            C, H, W = data.shape
            if self.use_keypoints:
                meta = {
                    'keypts': kp,
                    'keypts_normalized': kp_normalize(H, W, kp),
                    'index': index
                }

        return data, meta


class CelebAPrunedAligned_MAFLVal(CelebABase):
    eye_kp_idxs = [0, 1]

    def __init__(self, root, train=True, pair_warper=None, imwidth=100, crop=18,
                 do_augmentations=True, use_keypoints=False, use_hq_ims=False):
        self.root = root
        self.imwidth = imwidth
        self.train = train
        self.use_hq_ims = use_hq_ims
        self.warper = pair_warper
        self.crop = crop
        self.use_keypoints = use_keypoints

        anno = pd.read_csv(
            os.path.join(root, 'Anno', 'list_landmarks_align_celeba.txt'), header=1,
            delim_whitespace=True)
        assert len(anno.index) == 202599
        split = pd.read_csv(os.path.join(root, 'Eval', 'list_eval_partition.txt'),
                            header=None, delim_whitespace=True, index_col=0)
        assert len(split.index) == 202599

        mafltest = pd.read_csv(os.path.join(root, 'MAFL', 'testing.txt'), header=None,
                               delim_whitespace=True, index_col=0)
        split.loc[mafltest.index] = 4
        assert (split[1] == 4).sum() == 1000

        if train:
            self.data = anno.loc[split[split[1] == 0].index]
        else:
            self.data = anno.loc[split[split[1] == 4].index]

        # lefteye_x lefteye_y ; righteye_x righteye_y ; nose_x nose_y ;
        # leftmouth_x leftmouth_y ; rightmouth_x rightmouth_y
        self.keypoints = np.array(self.data, dtype=np.float32).reshape(-1, 5, 2)

        self.filenames = list(self.data.index)

        # Move head up a bit
        initial_crop = lambda im: transforms.functional.crop(im, 30, 0, 178, 178)
        self.keypoints[:, :, 1] -= 30
        self.keypoints *= self.imwidth / 178.

        normalize = transforms.Normalize(mean=[0.5084, 0.4224, 0.3769],
                                         std=[0.2599, 0.2371, 0.2323])
        augmentations = [
            JPEGNoise(),
            transforms.transforms.ColorJitter(.4, .4, .4),
            transforms.ToTensor(),
            PcaAug()
        ] if (train and do_augmentations) else [transforms.ToTensor()]

        self.initial_transforms = transforms.Compose(
            [initial_crop, transforms.Resize(self.imwidth)])
        self.transforms = transforms.Compose(augmentations + [normalize])

    def __len__(self):
        return len(self.data.index)


class MAFLAligned(CelebABase):
    eye_kp_idxs = [0, 1]

    def __init__(self, root, train=True, pair_warper=None, imwidth=100, crop=18,
                 do_augmentations=True, use_keypoints=False, use_hq_ims=False):
        self.root = root
        self.imwidth = imwidth
        self.use_hq_ims = use_hq_ims
        self.train = train
        self.warper = pair_warper
        self.crop = crop
        self.use_keypoints = use_keypoints

        anno = pd.read_csv(
            os.path.join(root, 'Anno', 'list_landmarks_align_celeba.txt'), header=1,
            delim_whitespace=True)
        assert len(anno.index) == 202599
        split = pd.read_csv(os.path.join(root, 'Eval', 'list_eval_partition.txt'),
                            header=None, delim_whitespace=True, index_col=0)
        assert len(split.index) == 202599

        mafltest = pd.read_csv(os.path.join(root, 'MAFL', 'testing.txt'), header=None,
                               delim_whitespace=True, index_col=0)
        split.loc[mafltest.index] = 4

        mafltrain = pd.read_csv(os.path.join(root, 'MAFL', 'training.txt'), header=None,
                                delim_whitespace=True, index_col=0)
        split.loc[mafltrain.index] = 5

        assert (split[1] == 4).sum() == 1000
        assert (split[1] == 5).sum() == 19000

        if train:
            self.data = anno.loc[split[split[1] == 5].index]
        else:
            self.data = anno.loc[split[split[1] == 4].index]

        # lefteye_x lefteye_y ; righteye_x righteye_y ; nose_x nose_y ;
        # leftmouth_x leftmouth_y ; rightmouth_x rightmouth_y
        self.keypoints = np.array(self.data, dtype=np.float32).reshape(-1, 5, 2)

        self.filenames = list(self.data.index)

        # Move head up a bit
        vertical_shift = 30
        crop_params = dict(i=vertical_shift, j=0, h=178, w=178)
        initial_crop = lambda im: transforms.functional.crop(im, **crop_params)
        self.keypoints[:, :, 1] -= vertical_shift
        self.keypoints *= self.imwidth / 178.

        normalize = transforms.Normalize(mean=[0.5084, 0.4224, 0.3769],
                                         std=[0.2599, 0.2371, 0.2323])
        augmentations = [
            JPEGNoise(),
            transforms.transforms.ColorJitter(.4, .4, .4),
            transforms.ToTensor(),
            PcaAug()
        ] if (train and do_augmentations) else [transforms.ToTensor()]

        self.initial_transforms = transforms.Compose(
            [initial_crop, transforms.Resize(self.imwidth)])
        self.transforms = transforms.Compose(augmentations + [normalize])

    def __len__(self):
        return len(self.data.index)


if __name__ == '__main__':
    import pylab

    dataset = CelebAPrunedAligned_MAFLVal('data/celeba', True,
                                          pair_warper=tps.Warper(100, 100))

    x, meta = dataset[6]
    print(x[0].shape)
    pylab.imshow(x[0].permute(1, 2, 0) + 0.5)
    pylab.figure()
    pylab.imshow(x[1].permute(1, 2, 0) + 0.5)
    pylab.show()
