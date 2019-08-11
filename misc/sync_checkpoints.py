"""A small utility for transferring features to/from the server.
"""
import os
import time
import json
import subprocess
import argparse
from pathlib import Path


def sync_between_servers(save_dir, src_server, dest_server, refresh, ckpt_list):
    with open(ckpt_list, "r") as f:
        ckpts = json.load(f)

    for key, subdict in ckpts.items():
        rel_dir = subdict["timestamp"]
        epoch = subdict["epoch"]

        filetypes = {
            "log": ["info.log"],
            "models": [f"checkpoint-epoch{epoch}.pth", "config.json"]
        }

        # copy experiment artifacts
        for filetype, fnames in filetypes.items():
            for fname in fnames:
                if rel_dir.startswith("TODO"):
                    print(f"Checkpoint (TODO) for {key}, skipping...")
                    continue
                rel_path = Path(rel_dir) / fname
                abs_path = Path(save_dir) / filetype / key / rel_path
                print(f"{key} -> {abs_path} [{src_server} -> {dest_server}]")
                # check if destination exists
                if not dest_server == "localhost":
                    cmd = f'ssh {dest_server} "test -f {str(abs_path)}"'
                    exists = not os.system(cmd)
                else:
                    exists = abs_path.exists()
                if exists and not refresh:
                    print(f"found {abs_path} on dest server, skipping")
                    continue

                if not dest_server == "localhost":
                    subprocess.call(["ssh", dest_server, "mkdir -p", str(abs_path.parent)])
                else:
                    os.system(f"mkdir -p {str(abs_path.parent)}")

                cmd = "scp"
                if not src_server == "localhost":
                    src_path = f"{src_server}:{abs_path}"
                else:
                    src_path = abs_path

                if not dest_server == "localhost":
                    dest_path = f"{dest_server}:{abs_path}"
                else:
                    dest_path = abs_path

                if not (src_server == "localhost" or dest_server == "localhost"):
                    cmd += " -3"

                sync_cmd = f"{cmd} {src_path} {dest_path}"
                print(f"running command {sync_cmd}")

                os.system(sync_cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", default="sync", choices=["sync", "fetch"])
    parser.add_argument("--ckpt_list", default="misc/server-checkpoints.json")
    parser.add_argument("--src_server", default="aws-albanie")
    parser.add_argument("--dest_server", default="gnodeb2")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--save_dir", default="~/data/exp/objectframe-pytorch/saved")
    args = parser.parse_args()

    if args.action == "sync":
        sync_between_servers(
            save_dir=args.save_dir,
            refresh=args.refresh,
            ckpt_list=args.ckpt_list,
            src_server=args.src_server,
            dest_server=args.dest_server,
        )
    else:
        raise ValueError(f"unknown action: {args.action}")
