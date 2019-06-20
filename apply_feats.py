"""
python apply_feats.py --action warp_ims --config configs/warp_ims_smallnet_mafl_64d_evc_128in_keypoints-ep57.json
"""
import os
import argparse
import torch
import json
import numpy as np
from torchvision.utils import make_grid
import time
from tqdm import tqdm
from pathlib import Path
import data_loader.data_loaders as module_data
from os.path import join as pjoin
from torch.utils.data import DataLoader
import model.model as module_arch
from pathlib import Path
from utils import tps, get_instance, clean_state_dict

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # NOQA

sys.path.insert(0, str(Path.home() / "coding/src/zsvision/python"))
from zsvision.zs_iterm import zs_dispFig # NOQA


def build_model_name(ckpt_name, use_hq_ims, crop, imwidth):
    template = "{}-hq{}-crop{}-imwidth{}"
    return template.format(ckpt_name, use_hq_ims, crop, imwidth)


def matcher(src_feats, dest_feats):
    import ipdb; ipdb.set_trace()


def dump_cache(feats, kpts, dest_dir):
    for key in feats.keys():
        feat_store = feats[key]
        kpts_store = kpts[key]
        feat_dest = str(Path(dest_dir) / "{}-feats.npy".format(key))
        kpts_dest = str(Path(dest_dir) / "{}-kpts.npy".format(key))
        tic = time.time()
        print("caching features to {}....".format(feat_dest))
        if config["reduce_prec"]:
            feat_store = feat_store.astype(np.float16)
        np.save(feat_dest, feat_store)
        print("caching ckpts to {}....".format(kpts_dest))
        np.save(kpts_dest, kpts_store)
        print("finished storing to disk in {:.3f}s".format(time.time() - tic))


def main(config):
    # setup data_loader instances
    # data_loader = getattr(module_data, config['data_loader']['type'])(
    #     config['data_loader']['args']['data_dir'], batch_size=512, shuffle=False,
    #     validation_split=0.0, training=False, num_workers=2)

    imwidth = config['dataset']['args']['imwidth']
    warper = get_instance(tps, 'warper', config, imwidth,
                          imwidth) if 'warper' in config.keys() else None
    train_dataset = get_instance(module_data, 'dataset', config, pair_warper=warper)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        num_workers=4,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
    )

    warp_val = config.get('warp_val', True)
    val_dataset = get_instance(
        module=module_data,
        name='dataset',
        config=config,
        train=False,
        pair_warper=warper if warp_val else None,
    )
    val_loader = DataLoader(val_dataset, batch_size=int(config["batch_size"]))

    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    # model.summary()

    # load state dict
    checkpoint = torch.load(config["weights"])
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(clean_state_dict(state_dict))

    model_name = build_model_name(
        ckpt_name=Path(config["weights"]).stem,
        use_hq_ims=config["dataset"]["args"]["use_hq_ims"],
        crop=config["dataset"]["args"]["crop"],
        imwidth=imwidth,
    )
    if config["action"] == "warp_ims":
        warp_dir = config["warp_dir"]
        dest_dir = pjoin(warp_dir, model_name)
    elif config["action"] == "cache_feats":
        feat_cache_dir = config["feat_cache_dest_dir"]
        dest_dir = pjoin(feat_cache_dir, model_name)

    if not Path(dest_dir).exists():
        Path(dest_dir).mkdir(exist_ok=True, parents=True)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    conv6 = list(model.named_children())[-1]
    assert conv6[0] == "conv6", "unexpected model"
    feat_dim = conv6[-1].out_channels
    if config.get("warp_val", False):
        raise NotImplementedError("was not expecting to warp validation ims")
    else:
        W = (imwidth - config["dataset"]["args"]["crop"] * 2) // 2
        H = W

    # store features as numpy arrays (will use lazy allocation)
    kpts = {"train": [], "val": []}
    num_train, num_val = train_dataset.data.shape[0], val_dataset.data.shape[0]
    loaders = {"train": train_loader, "val": val_loader}
    feats = {
        "train": np.zeros((num_train, feat_dim, H, W), dtype=np.float32),
        "val": np.zeros((num_val, feat_dim, H, W), dtype=np.float32)
    }

    with torch.no_grad():
        for key, loader in loaders.items():
            count = 0
            for ii, batch in enumerate(tqdm(loader)):
                data, target = batch["data"], batch["meta"]
                if config["limit"] and ii > config["limit"]:
                    break
                data = data.to(device)
                batch_size = data.shape[0]
                output = model(data)
                assert len(output) == 1, "output format has changed"
                if config["action"] == "cache_feats":
                    feats[key][ii:ii + batch_size] = output[0].cpu().numpy()
                    kpts[key].append(target)
                elif config["action"] == "warp_ims":
                    feats = output[0]
                    if False:
                        im = make_grid(data.cpu(), nrow=2, normalize=True)
                        plt.imshow(im.numpy().transpose((1, 2, 0)))
                        plt.title("source images")
                        zs_dispFig()
                    import ipdb; ipdb.set_trace()

                count += batch_size
            if not config["limit"]:
                msg = "Expected {} features for {}, found {}"
                msg = msg.format(feats[key].shape[0], key, count)
                assert count == feats[key].shape[0], msg

    # store features separately, to allow easier subcaching
    if config["action"] == "cache_feats":
        dump_cache(feats, kpts, dest_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str)
    parser.add_argument('-d', '--device', default="0", type=str)
    parser.add_argument('--action', type=str, choices=["cache_feats", "warp_ims"])
    parser.add_argument('--reduce_prec', action="store_true")
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    if args.config:
        config = json.load(open(args.config))
    else:
        raise AssertionError("config file needs to be specified. Add '-c config.json'")

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    config["reduce_prec"] = args.reduce_prec
    config["limit"] = args.limit
    config["action"] = args.action

    main(config)
