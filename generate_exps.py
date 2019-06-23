"""Module to generate experiments"""

import json
import copy
import shutil
import numpy as np
from pathlib import Path

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", default="scarce-data")
parser.add_argument("--gen_config_dir", default="data/gen_configs")
parser.add_argument("--refresh", action="store_true")
args = parser.parse_args()

evc_celeba_ckpt = "aws-saved/smallnet_celeba_64d_evc_128in_checkpoint-epoch57.pth"
non_evc_3d_celeba_ckpt = "saved/smallnet_celeba_3d/0618_085842/checkpoint-epoch161.pth"


def update_dict(orig, updater):
    for key, val in updater.items():
        if isinstance(val, dict):
            orig[key] = update_dict(orig[key], val)
        if val == "REMOVE-KEY":
            del orig[key]
        else:
            orig[key] = val


if args.exp_name == "scarce-data":
    gen_config_root = Path(args.gen_config_dir, args.exp_name)
    template = "configs/smallnet_helen_64d_evc_128in-scarce-data-template.json"
    if args.refresh and gen_config_root.exists():
        shutil.rmtree(str(gen_config_root))
    gen_config_root.mkdir(exist_ok=True, parents=True)
    with open(template, "r") as f:
        base = json.load(f)
    restrict_to_args = [1, 10, 100, 0]
    log_dir = "data/saved-gen"
    epochs = 1
    milestones = [15]
    seeds = [0, 1, 2]
    dataset = "Helen"
    Path(log_dir).mkdir(exist_ok=True, parents=True)
    models = [
        {
            "arch": {"type": "SmallNet", "args": {"num_output_channels": 64}},
            "finetune_from": evc_celeba_ckpt,
        },
        {
            "arch": {"type": "SmallNet", "args": {"num_output_channels": 64}},
            "finetune_from": non_evc_3d_celeba_ckpt,
        },
        {
            "arch": {"type": "SmallNet", "args": {"num_output_channels": 64}},
            "finetune_from": "REMOVE-KEY",
            "segmentation_head": {"freeze_base": False},
        },
    ]
    model_names = ["smallnet_evc_64d", "smallnet_non_evc_3d", "scratch"]
    named_models = list(zip(model_names, models))
    total = np.prod(list(map(len, [restrict_to_args, seeds, named_models])))
    count = 0
    for restrict_to in restrict_to_args:
        for seed in seeds:
            for model_name, model in named_models:
                exp = copy.deepcopy(base)
                name = "{:03d}-of-{:03d}-{}-{}-restrict{}-seed{}"
                exp["name"] = name.format(count, total, model_name, dataset,
                                          restrict_to, seed)
                exp["dataset"]["args"]["restrict_to"] = restrict_to
                exp["dataset"]["args"]["restrict_seed"] = seed
                exp["trainer"]["epochs"] = epochs
                exp["trainer"]["log_dir"] = log_dir
                exp["lr_scheduler"]["milestones"] = milestones
                update_dict(exp, model)
                exp_str = json.dumps(exp)
                config_path = gen_config_root / "{}.json".format(exp["name"])
                print("{} generating config -> {}".format(count, exp["name"]))
                with open(str(config_path), "w") as f:
                    f.write(exp_str)
                count += 1
