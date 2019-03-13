import torch

def inter_ocular_error(output, meta, dataset, config):
    eyeidxs = dataset.eye_kp_idxs
    pred = output[0].detach().cpu()
    gt = meta['keypts_normalized']

    iod = ((gt[:, eyeidxs[0], :] - gt[:, eyeidxs[1], :])**2.).sum(1).sqrt()[:,None]
    err = ((pred - gt)**2.).sum(2).sqrt()
    ioderr = err/iod


    return 100*ioderr.mean()