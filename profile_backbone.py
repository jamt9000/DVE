"""
python profile_backbone.py \
    --config configs/profile_smallnet_mafl_64d_evc.json \
    --device 3
"""
import os
import argparse
import torch
import numpy as np
import json
import time
import torch.nn as nn
import data_loader.data_loaders as module_data
from utils import NoGradWrapper, Up, get_instance
from torch.utils.data import DataLoader
from thop import profile
import model.model as module_arch


def get_profile_name(model_type, keypoint_reg, imwidth, upsample):
    name_map = {"SmallNet": "Ours", "HourglassNet": "Hourglass"}
    for key, val in name_map.items():
        model_type = model_type.replace(key, val)
    model_name = "{}, image size {},".format(model_type, imwidth)
    if keypoint_reg:
        model_name += " backbone+regressor"
    else:
        model_name += " backbone"
    if keypoint_reg and upsample:
        model_name += " (+upsample)"
    return model_name


def main(config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for model_type in "SmallNet", "HourglassNet":
        for imwidth in 70, 96, 128:
            for keypoint_reg in False, True:
                for upsample in False, True:

                    if upsample and not keypoint_reg:
                        continue  # not needed
                    if model_type == "HourglassNet" and imwidth not in {96, 128}:
                        continue  # not needed
                    if model_type == "SmallNet" and imwidth not in {70, 128}:
                        continue  # not needed

                    profile_name = get_profile_name(
                        model_type=model_type,
                        keypoint_reg=keypoint_reg,
                        upsample=upsample,
                        imwidth=imwidth,
                    )
                    config["dataset"]["args"] = {"imwidth": imwidth}
                    val_dataset = get_instance(
                        module=module_data,
                        name='dataset',
                        config=config,
                        train=False,
                    )
                    val_loader = DataLoader(
                        val_dataset,
                        batch_size=config["batch_size"],
                    )
                    config["arch"] = {
                        "type": model_type,
                        "args": {"num_output_channels": 64},
                    }

                    model = get_instance(module_arch, 'arch', config)

                    if keypoint_reg:
                        descdim = config['arch']['args']['num_output_channels']
                        kp_regressor = get_instance(module_arch, 'keypoint_regressor',
                                                    config,
                                                    descriptor_dimension=descdim)
                        basemodel = NoGradWrapper(model)

                        if upsample:
                            model = nn.Sequential(basemodel, Up(), kp_regressor)
                        else:
                            model = nn.Sequential(basemodel, kp_regressor)
                    # model.summary()

                    # prepare model for testing
                    model = model.to(device)
                    model.eval()
                    timings = []
                    warmup = 3
                    num_batches = 10

                    with torch.no_grad():
                        # count = 0
                        tic = time.time()
                        for ii, batch in enumerate(val_loader):
                            data = batch["data"].to(device)
                            batch_size = data.shape[0]
                            _ = model(data)
                            speed = batch_size / (time.time() - tic)
                            if ii > warmup:
                                timings.append(speed)
                            if ii > warmup + num_batches:
                                break
                            # print("speed: {:.3f}Hz".format(speed))
                            tic = time.time()
                            # count += batch_size

                    flops, params = profile(
                        model,
                        input_size=(1, 3, imwidth, imwidth),
                        verbose=False,
                    )
                    # use format so that its easy to latexify
                    template = "{} & {:.1f} & {:.1f} & ${:.1f} (\\pm {:.1f})$\\\\"
                    template = template.format(
                        profile_name,
                        params / 10**6,
                        flops / 10**9,
                        np.mean(timings),
                        np.std(timings),
                    )
                    print(template)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str)
    parser.add_argument('-d', '--device', default="0", type=str)
    args = parser.parse_args()

    if args.config:
        config = json.load(open(args.config))
    else:
        raise AssertionError("config file needs to be specified. Add '-c config.json'")

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config)
