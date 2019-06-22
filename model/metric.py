import numpy as np
import torch.nn.functional as F


def inter_ocular_error(output, meta, dataset, config):
    eyeidxs = dataset.eye_kp_idxs
    pred = output[0].detach().cpu()
    gt = meta['keypts_normalized']
    iod = ((gt[:, eyeidxs[0], :] - gt[:, eyeidxs[1], :])**2.).sum(1).sqrt()[:, None]
    err = ((pred - gt)**2.).sum(2).sqrt()
    ioderr = err / iod
    return 100 * ioderr.mean()


class runningIOU:

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        x = n_class * label_true[mask].astype(int) + label_pred[mask]
        hist = np.bincount(x, minlength=n_class ** 2)
        return hist.reshape(n_class, n_class)

    def update(self, x, meta):
        target = meta["lbls"].cpu()
        n, c, h, w = x.size()
        nt, ht, wt = target.size()
        if h != ht and w != wt:  # upsample labels
            x = F.interpolate(x, size=(ht, wt), mode="bilinear", align_corners=True)
        preds = x.data.max(1)[1].cpu().numpy()
        target = target.numpy()
        for lt, lp in zip(target, preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(),
                                                     self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))
        scores = {"acc": acc, "clsacc": acc_cls, "fwacc": fwavacc, "miou": mean_iu}
        return scores, cls_iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
