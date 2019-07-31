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

dve_celeba_ckpt = "aws-saved/smallnet_celeba_64d_dve_128in_checkpoint-epoch57.pth"
non_dve_3d_celeba_ckpt = "saved/smallnet_celeba_3d/0618_085842/checkpoint-epoch161.pth"


def update_dict(orig, updater):
    for key, val in updater.items():
        assert key in orig, "unknown key: {}".format(key)
        if isinstance(val, dict):
            orig[key] = update_dict(orig[key], val)
        elif val == "REMOVE-KEY":
            del orig[key]
        else:
            orig[key] = val
    return orig


if args.exp_name == "scarce-data":
    gen_config_root = Path(args.gen_config_dir, args.exp_name)
    template = "configs/smallnet_helen_64d_dve_128in-scarce-data-template.json"
    if args.refresh and gen_config_root.exists():
        shutil.rmtree(str(gen_config_root))
    gen_config_root.mkdir(exist_ok=True, parents=True)
    with open(template, "r") as f:
        base = json.load(f)
    restrict_to_args = [1, 10, 100, 0]
    save_period = 100
    log_dir = "data/saved-gen"
    visualizations = []
    epochs = 20
    milestones = [15]
    tensorboardX = True
    # seeds = [0, 1, 2]
    seeds = [0, 1, 2]
    dataset = "Helen"
    Path(log_dir).mkdir(exist_ok=True, parents=True)
    named_models = [
        ("scratch_64d", {
            "arch": {"type": "SmallNet", "args": {"num_output_channels": 64}},
            "finetune_from": "REMOVE-KEY",
            "segmentation_head": {"args": {"freeze_base": False}},
        }),
        ("smallnet_non_dve_3d", {
            "arch": {"type": "SmallNet", "args": {"num_output_channels": 3}},
            "finetune_from": non_dve_3d_celeba_ckpt,
        }),
        ("smallnet_dve_64d", {
            "arch": {"type": "SmallNet", "args": {"num_output_channels": 64}},
            "finetune_from": dve_celeba_ckpt,
        }),
    ]
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
                exp["trainer"]["save_period"] = save_period
                exp["trainer"]["tensorboardX"] = tensorboardX
                exp["visualizations"] = visualizations
                exp["lr_scheduler"]["args"]["milestones"] = milestones
                update_dict(exp, model)
                exp_str = json.dumps(exp)
                config_path = gen_config_root / "{}.json".format(exp["name"])
                print("{} generating config -> {}".format(count, exp["name"]))
                with open(str(config_path), "w") as f:
                    f.write(exp_str)
                count += 1
