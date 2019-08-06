import json
import copy
import argparse
import itertools
from pathlib import Path
from collections import OrderedDict


def generate_config_grid(base_config, grid_dir, grid, refresh):
    with open(base_config, "r") as f:
        base = json.load(f)

    hparam_vals = [x for x in grid.values()]
    grid_vals = list(itertools.product(*hparam_vals))
    hparams = list(grid.keys())

    for cfg_vals in grid_vals:
        dest_name = Path(base_config).stem
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
            dest_name += f"-{hparam}-{val}"
        dest_path = Path(grid_dir) / f"{dest_name}.json"
        dest_path.parent.mkdir(exist_ok=True, parents=True)
        if not dest_path.exists() or refresh:
            with open(str(dest_path), "w") as f:
                json.dump(config, f)
        else:
            print(f"grid file at {str(dest_path)} exists, skipping....")
    print(f"Wrote {len(grid_vals)} configs to disk")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help="base config file path")
    parser.add_argument('--grid_dir', default="configs/grid")
    parser.add_argument('--bs', default="32,64,128")
    parser.add_argument('--smax', default="1,100")
    parser.add_argument('--lr', default="1E-2,3E-3,1E-3")
    parser.add_argument('--upsample', default="0,1")
    parser.add_argument('--refresh', action="store_true")
    args = parser.parse_args()

    grid = OrderedDict()
    for key in ["bs", "smax", "lr", "upsample"]:
        grid[key] = [float(x) for x in getattr(args, key).split(",")]

    generate_config_grid(
        base_config=args.config,
        grid_dir=args.grid_dir,
        grid=grid,
        refresh=args.refresh,
    )
