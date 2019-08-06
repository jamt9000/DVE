"""A small utility for transferring features to/from the server.
"""
import os
import time
import json
import subprocess
import argparse
from pathlib import Path


def sync_between_servers(save_dir, src_server, dest_server, local_tmp_dir, refresh,
                         ckpt_list):
    with open(ckpt_list, "r") as f:
        ckpts = json.load(f)

    filetypes = {
        "log": ["info.log"],
        "models": ["checkpoint-epoch45.pth", "config.json"]
    }
    for key, rel_dir in ckpts.items():

        # copy experiment artifacts
        for filetype, fnames in filetypes.items():
            for fname in fnames:
                if rel_dir.startswith("TODO"):
                    continue
                rel_path = Path(rel_dir) / fname
                abs_path = Path(save_dir) / filetype / key / rel_path
                print(f"{key} -> {abs_path} [{src_server} -> {dest_server}]")
                subprocess.call(["ssh", dest_server, "mkdir -p", str(abs_path.parent)])
                sync_cmd = (f"scp -3 {src_server}:{abs_path} "
                            f"{dest_server}:{abs_path}")
                print(f"running command {sync_cmd}")
                os.system(sync_cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", default="sync", choices=["sync", "fetch"])
    parser.add_argument("--ckpt_list", default="misc/server_checkpoints.json")
    parser.add_argument("--src_server", default="aws-albanie")
    parser.add_argument("--dest_server", default="shallow8")
    parser.add_argument("--local_tmp_dir", default="/tmp/server-sync")
    parser.add_argument("--refresh_server", action="store_true")
    parser.add_argument("--save_dir", default="~/data/exp/objectframe-pytorch/saved")
    args = parser.parse_args()

    refresh_targets = {
        "server": args.refresh_server,
    }

    if args.action == "sync":
        sync_between_servers(
            save_dir=args.save_dir,
            refresh=refresh_targets,
            ckpt_list=args.ckpt_list,
            src_server=args.src_server,
            local_tmp_dir=args.local_tmp_dir,
            dest_server=args.dest_server,
        )
    else:
        raise ValueError(f"unknown action: {args.action}")
