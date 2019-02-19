from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import pandas as pd
import os
from PIL import Image


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super(MnistDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class CelebAPrunedAligned_MAFLVal(Dataset):
    def __init__(self, data_dir, train):
        self.data_dir = data_dir
        self.imwidth = 100

        anno = pd.read_csv(os.path.join(data_dir, 'Anno', 'list_landmarks_align_celeba.txt'), header=1,
                           delim_whitespace=True)
        assert len(anno.index) == 202599
        split = pd.read_csv(os.path.join(data_dir, 'Eval', 'list_eval_partition.txt'), header=None,
                            delim_whitespace=True, index_col=0)
        assert len(split.index) == 202599

        mafltest = pd.read_csv(os.path.join(data_dir, 'MAFL', 'testing.txt'), header=None, delim_whitespace=True,
                               index_col=0)
        split.loc[mafltest.index] = 4
        assert (split[1] == 4).sum() == 1000

        if train:
            self.data = anno.loc[split[split[1] == 0].index]
        else:
            self.data = anno.loc[split[split[1] == 4].index]

        self.filenames = self.data.index.to_list()

        normalize = transforms.Normalize(mean=[0.5084, 0.4224, 0.3769], std=[0.2599, 0.2371, 0.2323])
        augmentations = [transforms.transforms.ColorJitter(.4, .4, .4)] if train else []
        self.transforms = transforms.Compose(
            [transforms.Resize(self.imwidth),
             transforms.CenterCrop(self.imwidth)]
            + augmentations +
            [transforms.ToTensor(),
            normalize])

    def __getitem__(self, index):
        im = Image.open(os.path.join(self.data_dir, 'Img', 'img_align_celeba', self.filenames[index]))

        return self.transforms(im)

    def __len__(self):
        return len(self.data.index)


if __name__ == '__main__':
    import pylab
    dataset = CelebAPrunedAligned_MAFLVal('/scratch/shared/nfs1/jdt/celeba', True)

    x = dataset[123]
    print(x.shape)
    pylab.imshow(x.permute(1,2,0) + 0.5)
    pylab.show()