import argparse
import subprocess
from pathlib import Path


def main(grid_dir, max_jobs_per_device, refresh, device):
    configs = sorted(list(Path(grid_dir).glob("*.json")))
    print(f"found {len(configs)} configs in grid dir")
    jobs_launched = 0
    for config in configs:
        # exp_dir = Path("data/saved/models") / f"grid-{config.stem}"
        # if exp_dir.exists() and not refresh:
        #     continue
        std_out = Path("data/grid_log") / f"{config.stem}.txt"
        if std_out.exists():
            print(f"Found existing log for {std_out}, skipping....")
            continue
        if jobs_launched >= max_jobs_per_device:
            print("launched maximum number of jobs, exiting....")
            exit()
        python_bin = str(Path.home() / "local/anaconda3/envs/pt37/bin/python")
        cmd_args = [python_bin, "train.py", "--config", str(config), "--device", device]
        print(f"launching job with args: {' '.join(cmd_args)}")
        std_out.parent.mkdir(exist_ok=True, parents=True)
        log = open(str(std_out), "a")
        proc = subprocess.Popen(cmd_args, stdout=log, stderr=log)
        jobs_launched += 1
        print(f"Job launched successfully with pid: {proc.pid}")
        # should probably close the log file
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device')
    parser.add_argument('--refresh', action="store_true")
    parser.add_argument('--max_jobs_per_device', default=3)
    parser.add_argument('--grid_dir', default="configs/grid")
    args = parser.parse_args()

    main(
        grid_dir=args.grid_dir,
        max_jobs_per_device=args.max_jobs_per_device,
        refresh=args.refresh,
        device=args.device,
    )
