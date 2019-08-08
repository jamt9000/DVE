import json
import copy
import argparse
import itertools
from pathlib import Path
from collections import OrderedDict


def generate_configs(base_config, dest_dir, embeddings, grid, refresh, experiments_path):
    with open(base_config, "r") as f:
        base = json.load(f)

    with open(experiments_path, "r") as f:
        exps = json.load(f)

    model_family = {
        "smallnet": {"crop": 15, "imwidth": 100},
        "hourglass": {"crop": 20, "imwidth": 136},
    }

    for model_name, epoch in embeddings.items():

        # model naming convention: <dataset>-<model_type>-<embedding-dim>
        tokens = model_name.split("-")
        model_type, embedding_dim = tokens[1], int(tokens[2][:-1])
        preproc_kwargs = model_family[model_type]
        
        hparam_vals = [x for x in grid.values()]
        grid_vals = list(itertools.product(*hparam_vals))
        hparams = list(grid.keys())

        for cfg_vals in grid_vals:
            # dest_name = Path(base_config).stem
            config = copy.deepcopy(base)
            for hparam, val in zip(hparams, cfg_vals):
                if hparam == "smax":
                    config["keypoint_regressor"]["softmaxarg_mul"] = val
                elif hparam == "lr":
                    config["optimizer"]["args"]["lr"] = val
                elif hparam == "bs":
                    val = int(val)
                    config["batch_size"] = val
                elif hparam == "upsample":
                    val = bool(val)
                    config["keypoint_regressor_upsample"] = val
                else:
                    raise ValueError(f"unknown hparam: {hparam}")
            ckpt = f"checkpoint-epoch{epoch}.pth"
            ckpt_path = Path("data/saved/models") / model_name / exps[model_name] / ckpt
            config["dataset"]["args"].update(preproc_kwargs)
            config["finetune_from"] = str(ckpt_path)
            config["arch"]["args"]["num_output_channels"] = embedding_dim
                
            dest_path = Path(dest_dir) / f"{model_name}.json"
            dest_path.parent.mkdir(exist_ok=True, parents=True)
            if not dest_path.exists() or refresh:
                with open(str(dest_path), "w") as f:
                    json.dump(config, f, indent=4, sort_keys=False)
            else:
                print(f"config file at {str(dest_path)} exists, skipping....")
        print(f"Wrote {len(grid_vals)} configs to disk")
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default="mafl-keypoints")
    parser.add_argument('--bs', default="32")
    parser.add_argument("--experiments_path", default="misc/experiments.json")
    parser.add_argument('--smax', default="100")
    parser.add_argument('--lr', default="1E-3")
    parser.add_argument('--upsample', default="0")
    parser.add_argument('--refresh', action="store_true")
    args = parser.parse_args()

    grid_args = OrderedDict()
    for key in ["bs", "smax", "lr", "upsample"]:
        grid_args[key] = [float(x) for x in getattr(args, key).split(",")]
    dest_config_dir = Path("configs") / args.target
    base_config_path = Path("configs/templates") / f"{args.target}.json"

    pretrained_embeddings = {
        "celeba-smallnet-3d": 100,
        "celeba-smallnet-16d": 100,
        "celeba-smallnet-32d": 100,
        "celeba-smallnet-64d": 100,
        "celeba-smallnet-3d-dve": 100,
        "celeba-smallnet-16d-dve": 100,
        "celeba-smallnet-32d-dve": 100,
        "celeba-smallnet-64d-dve": 100,
        "celeba-hourglass-64d-dve": 45,
    }

    generate_configs(
        base_config=base_config_path,
        embeddings=pretrained_embeddings,
        experiments_path=args.experiments_path,
        refresh=args.refresh,
        dest_dir=dest_config_dir,
        grid=grid_args,
    )
