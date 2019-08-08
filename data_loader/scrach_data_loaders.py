import time
import torch
import numpy as np
from utils.global_caches import global_cache
from os.path import join as pjoin
from data_loader import MAFLAligned


# ---------------------------------------------------------
# debugging code
# ---------------------------------------------------------
def check_cache(key, fetcher, refresh):
    tic = time.time()
    if key not in global_cache or refresh:
        val = fetcher()
        global_cache[key] = val
    print("fetched {} in {:.3f}s".format(key, time.time() - tic))
    return global_cache[key]


def np_loader(np_path, l2norm=False):
    tic = time.time()
    print("loading features from {}".format(np_path))
    with open(np_path, "rb") as f:
        data = np.load(f)
    print("done in {:.3f}s".format(time.time() - tic))
    if isinstance(data, np.ndarray) and data.size == 1:
        data = data[()]  # handle numpy dict storage convnetion
    return data


class CachedDataset(MAFLAligned):

    def __init__(self, root, cache_root, imwidth, crop, train, use_hq_ims,
                 pair_warper=None, visualize=False):
        super().__init__(
            root=root,
            train=train,
            pair_warper=None,
            imwidth=imwidth,
            crop=crop,
            do_augmentations=False,
            use_keypoints=True,
            use_hq_ims=use_hq_ims,
        )
        self.train = train
        self.visualize = visualize

        # sanity check on loaded feature path
        template = "imwidth{}".format(imwidth)
        msg = "expected {} to occur in path to cached features".format(template)
        assert template in cache_root, msg

        if train:
            feat_path = pjoin(cache_root, "train-feats.npy")
            kpts_path = pjoin(cache_root, "train-kpts.npy")
        else:
            feat_path = pjoin(cache_root, "val-feats.npy")
            kpts_path = pjoin(cache_root, "val-kpts.npy")

        use_cache = True
        refresh = False

        if use_cache:
            fetcher = lambda: np_loader(feat_path)
            self.feats = check_cache(key=feat_path, fetcher=fetcher, refresh=refresh)
        else:
            self.feats = np.load(feat_path)

        if self.use_hq_ims:
            self.subdir = "img_align_celeba_hq"
        else:
            self.subdir = "img_align_celeba"

        if use_cache:  # flatten kpts store
            fetcher = lambda: np_loader(kpts_path)
            kpts_store = check_cache(key=kpts_path, fetcher=fetcher, refresh=refresh)
        else:
            kpts_store = np_loader(kpts_path)

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
        im = torch.from_numpy(self.feats[index])
        meta = {
            "keypts": self.keypts[index],
            "keypts_normalized": self.keypts_normalized[index],
            "index": self.index[index],
        }
        sample = {"data": im, "meta": meta}
        if self.visualize:
            subdir = os.path.join(self.root, 'Img', self.subdir)
            im = Image.open(os.path.join(subdir, self.filenames[index]))
            im_data = self.transforms(self.initial_transforms(im))
        else:
            im_data = []
        sample["im_data"] = im_data
        return sample

