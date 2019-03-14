import os
import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from train import get_instance
from utils import tps
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from utils.visualization import norm_range
import torch.nn.functional as F
from utils.tps import *


def coll(batch):
    b = torch.utils.data.dataloader.default_collate(batch)
    # Flatten to be 4D
    return [bi.reshape((-1,) + bi.shape[-3:]) if isinstance(bi, torch.Tensor) else bi for bi in b]


def find_descriptor(x, y, source_descs, target_descs, stride):
    C, H, W = source_descs.shape

    x = int(np.round(x / stride))
    y = int(np.round(y / stride))

    x = min(W-1,max(x,0))
    y = min(H-1,max(y,0))

    query_desc = source_descs[:, y, x]

    corr = torch.matmul(query_desc.reshape(-1, C), target_descs.reshape(C, H * W))
    maxidx = corr.argmax()
    grid = spatial_grid_unnormalized(H, W).reshape(-1, 2) * stride
    x, y = grid[maxidx]

    return x.item(), y.item()


def main(config, resume):
    device = 'cuda'

    # setup data_loader instances
    imwidth = config['dataset']['args']['imwidth']
    crop = config['dataset']['args'].get('crop', config['warper']['args'].get('crop',None))

    # Want explicit pair warper
    warper = tps.Warper(imwidth, imwidth, warpsd_all=0.001, warpsd_subset=0.01, transsd=0.1,
                        scalesd=0.1, rotsd=5, im1_multiplier=0.5)

    warper1 = tps.WarperSingle(imwidth, imwidth, warpsd_all=0.0, warpsd_subset=0.0, transsd=0.05,
                               scalesd=0.01, rotsd=2)

    train_dataset = module_data.MAFLAligned(root='data/celeba', imwidth=imwidth, crop=crop, train=True,
                                            pair_warper=warper1, do_augmentations=False)
    val_dataset = module_data.MAFLAligned(root='data/celeba', imwidth=imwidth, crop=crop, train=False,
                                          pair_warper=warper)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    data_loader = DataLoader(val_dataset, batch_size=2, collate_fn=coll, shuffle=False)

    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    model.summary()

    # load state dict
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.train()

    # Stabilise batch norm
    reinit_bn = False
    if reinit_bn:
        torch.manual_seed(0)
        bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        for m in bns:
            m.reset_running_stats()

        train_loader_it = iter(train_loader)
        with torch.no_grad():
            for i in range(100):
                torch.manual_seed(0)
                np.random.seed(0)

                data, meta = next(train_loader_it)
                data = data.to(device)
                print(i, 'data checksum', float(data.sum()))
                output = model(data)
                print(i, 'bn checksum', float(bns[0].running_mean.sum()))

    # prepare model for testing
    model.eval()

    same_errs = []
    diff_errs = []

    torch.manual_seed(0)
    with torch.no_grad():
        for i, (data, meta) in enumerate(tqdm(data_loader)):
            if i == 0:
                # Checksum to make sure warps are deterministic
                if data.shape[2] == 64:
                    assert float(data.sum()) == -553.9221801757812
                elif data.shape[2] == 128:
                    assert float(data.sum()) == 2724.149658203125

            data = data.to(device)


            output = model(data)

            descs = output[0]
            descs1 = descs[0::2]  # 1st in pair (more warped)
            descs2 = descs[1::2]  # 2nd in pair
            ims1 = data[0::2].cpu()
            ims2 = data[1::2].cpu()
            kp1 = meta['kp1']
            kp2 = meta['kp2']

            im_source = ims1[0]
            im_same = ims2[0]
            im_diff = ims2[1]

            C, imH, imW = im_source.shape
            B, C, H, W = descs1.shape
            stride = imW / W

            desc_source = descs1[0]
            desc_same = descs2[0]
            desc_diff = descs2[1]

            kp_source = kp1[0]
            kp_same = kp2[0]
            kp_diff = kp2[1]

            fig = plt.figure()  # a new figure window
            ax1 = fig.add_subplot(1, 3, 1)
            ax2 = fig.add_subplot(1, 3, 2)
            ax3 = fig.add_subplot(1, 3, 3)

            ax1.imshow(norm_range(im_source).permute(1, 2, 0))
            ax1.scatter(kp_source[:, 0], kp_source[:, 1], c='g')

            ax2.imshow(norm_range(im_same).permute(1, 2, 0))
            ax2.scatter(kp_same[:, 0], kp_same[:, 1], c='g')

            ax3.imshow(norm_range(im_diff).permute(1, 2, 0))
            ax3.scatter(kp_diff[:, 0], kp_diff[:, 1], c='g')

            fsrc = F.normalize(desc_source, p=2, dim=0)
            fsame = F.normalize(desc_same, p=2, dim=0)
            fdiff = F.normalize(desc_diff, p=2, dim=0)

            for ki, kp in enumerate(kp_source):
                x, y = np.array(kp)
                gt_samex, gt_samey = np.array(kp_same[ki])
                gt_diffx, gt_diffy = np.array(kp_diff[ki])

                samex, samey = find_descriptor(x, y, fsrc, fsame, stride)
                ax2.scatter(samex, samey, c='b')

                same_errs.append(np.sqrt((gt_samex - samex) ** 2 + (gt_samey - samey) ** 2))

                diffx, diffy = find_descriptor(x, y, fsrc, fdiff, stride)
                ax3.scatter(diffx, diffy, c='b')

                diff_errs.append(np.sqrt((gt_diffx - diffx) ** 2 + (gt_diffy - diffy) ** 2))

            fig.savefig('/tmp/matching.pdf')

    print('same', np.mean(same_errs))
    print('diff', np.mean(diff_errs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume)
