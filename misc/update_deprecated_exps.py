"""Parse logs from experiments in deprecated format and summarise in new format.

NOTE: This module requires tensorflow.
"""
import re
import time
import json
import argparse
import logging
import subprocess
import numpy as np
import shutil
from parse_config import ConfigParser
from pathlib import Path
from test_matching import evaluation
from logger import setup_logging
from protobuf_to_dict import protobuf_to_dict
import tensorflow as tf
from itertools import zip_longest
from collections import OrderedDict


def generate_url(root_url, target, exp_name, experiments):
    path_store = {
        "log": {"parent": "log", "fname": "info.log"},
        "config": {"parent": "models", "fname": "config.json"},
        "model": {"parent": "models", "fname": "model_best.pth"}
    }
    paths = path_store[target]
    timestamp = experiments[exp_name]
    return str(Path(root_url) / paths["parent"] / exp_name / timestamp / paths["fname"])


def parse_tboard_files(rel_dir):
    tboard_files = list(Path(rel_dir).glob("events.out.tfevents.*"))
    assert len(tboard_files) == 1, "expected a single tensorboard file"
    tboard_file_path = tboard_files[0]
    gen_log = [f"This log was generated from tensorboard file {tboard_file_path.name}"]
    count = 0
    try:
        for summary in tf.train.summary_iterator(str(tboard_file_path)):
            count += 1
            summary = protobuf_to_dict(summary)
            if count > 1000:
                break
            if "step" not in summary:
                continue
            step = summary["step"]
            ts = time.strftime('%Y-%m-%d:%Hh%Mm%Ss', time.gmtime(summary["wall_time"]))
            if "summary" in summary and summary["summary"]["value"]:
                value = summary["summary"]["value"]
                if "simple_value" in value[0]:
                    vals = [f"{x['tag']}: {x['simple_value']}" for x in value]
                    if step % 2000 == 0 or value[0]["tag"] != "train/loss":
                        row = f"{ts} step: {step}, {','.join(vals)}"
                        gen_log.append(row)
                        print(row)
                elif "image" in value[0]:
                    pass
                else:
                    import ipdb; ipdb.set_trace()
    except tf.errors.DataLossError as DLE:
        print(f"{DLE} Could not parse any further information")
    print(f"parsed {count} summaries")
    return gen_log


def modernize_exp_dir(experiments, save_dir, refresh):
    filetypes = {
        "log": ["info.log"],
        "models": ["model_best.pth", "config.json"]
    }
    for key, rel_dir in experiments.items():

        tokens = key.split("-")
        dataset, model_name = tokens[0], "-".join(tokens[1:])
        timestamp = rel_dir.split("/")[-1]
        model_dir = Path(save_dir) / "models"

        src_config = Path(rel_dir) / "config.json"
        config_path = Path(save_dir) / "models" / key / timestamp / "config.json"
        config_path.parent.mkdir(exist_ok=True, parents=True)
        if config_path.exists() and not refresh:
            print(f"transferred config found at {str(config_path)}, skipping...")

        print(f"copying config: {str(src_config)} -> {str(config_path)}")
        shutil.copyfile(str(src_config), str(config_path))
        dest_log = Path(save_dir) / "log" / key / timestamp / "info.log"

        if dest_log.exists() and not refresh:
            print(f"generated log found at {str(dest_log)}, skipping...")
            continue
        generated_log = parse_tboard_files(rel_dir)

        dest_log.parent.mkdir(exist_ok=True, parents=True)
        setup_logging(save_dir=dest_log.parent)
        logger = logging.getLogger("tboard-parser")
        for row in generated_log:
            logger.info(row)

        # re-run pixel matching evaluation
        best_ckpt_path = Path(rel_dir) / "model_best.pth"
        eval_args = argparse.ArgumentParser()
        eval_args.add_argument("--config", default=str(config_path))
        eval_args.add_argument("--device", default="0")
        eval_args.add_argument("--mini_eval", default=1)
        eval_args.add_argument("--resume", default=best_ckpt_path)
        eval_config = ConfigParser(eval_args, slave_mode=True)
        evaluation(eval_config, logger=logger)

        import ipdb; ipdb.set_trace()

        # copy experiment artifacts
        for filetype, fnames in filetypes.items():
            for fname in fnames:
                if rel_dir.startswith("TODO"):
                    continue
                rel_path = Path(rel_dir) / fname
                local_path = Path(save_dir) / filetype / key / rel_path
                server_path = Path(web_dir).expanduser() / filetype / key / rel_path
                dest = f"{webserver}:{str(server_path)}"
                print(f"{key} -> {webserver} [{local_path} -> {server_path}]")
                subprocess.call(["ssh", webserver, "mkdir -p", str(server_path.parent)])
                rsync_args = ["rsync", "-hvrPt", str(local_path), dest]
                print(f"running command {' '.join(rsync_args)}")
                subprocess.call(rsync_args)

           
def parse_log(log_path):
    with open(log_path, "r") as f:
        log = f.read().splitlines()
    results = {}
    for metric in {"same-identity", "different-identity"}:
        tag = f"Mean Pixel Error ({metric})"
        results[metric] = OrderedDict()
        presence = [tag in row for row in log]
        assert sum(presence) == 1, "expected single occurence of log tag"
        # metrics = ["R1", "R5", "R10", "R50", "MedR", "MeanR"]
        pos = np.where(presence)[0].item()
        row = log[pos]
        tokens = row.split(" ")
        val = float(tokens[-1])
        results[metric] = val
    for row in log:
        if "Trainable parameters" in row:
            results["params"] = int(row.split(" ")[-1])
    return results


def parse_results(experiments, save_dir):
    log_results = {}
    for exp_name, timestamp in experiments.items():
        if timestamp.startswith("TODO"):
            log_results[exp_name] = {"timestamp": "TODO", "results": {}}
            continue
        log_path = Path(save_dir) / "log" / exp_name / timestamp / "info.log"
        assert log_path.exists(), f"missing log file for {exp_name}: {log_path}"
        results = parse_log(log_path)
        log_results[exp_name] = {"timestamp": timestamp, "results": results}
    return log_results


def generate_readme(experiments, readme_template, root_url, readme_dest, results_path,
                    save_dir):

    results = parse_results(experiments, save_dir)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4, sort_keys=False)

    with open(readme_template, "r") as f:
        readme = f.read().splitlines()

    generated = []
    for row in readme:
        edits = []
        regex = r"\{\{(.*?)\}\}"
        for match in re.finditer(regex, row):
            groups = match.groups()
            assert len(groups) == 1, "expected single group"
            exp_name, target = groups[0].split(".")
            if results[exp_name]["timestamp"] == "TODO":
                token = "TODO"
            elif target in {"config", "model", "log"}:
                token = generate_url(root_url, target, exp_name, experiments=experiments)
            elif target in {"same-identity", "different-identity"}:
                token = f"{results[exp_name]['results'][target]:.2f}"
            elif target in {"params"}:
                token = millify(results[exp_name]["results"]["params"], precision=2)
            edits.append((match.span(), token))
        if edits:
            # invert the spans
            spans = [(None, 0)] + [x[0] for x in edits] + [(len(row), None)]
            inverse_spans = [(x[1], y[0]) for x, y in zip(spans, spans[1:])]
            tokens = [row[start:stop] for start, stop in inverse_spans]
            urls = [str(x[1]) for x in edits]
            new_row = ""
            for token, url in zip_longest(tokens, urls, fillvalue=""):
                new_row += token + url
            row = new_row

        generated.append(row)

    with open(readme_dest, "w") as f:
        f.write("\n".join(generated))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="data/saved")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--experiments_path", default="misc/experiments-deprecated.json")
    args = parser.parse_args()

    with open(args.experiments_path, "r") as f:
        experiments = json.load(f)

    modernize_exp_dir(
        save_dir=args.save_dir,
        experiments=experiments,
        refresh=args.refresh,
    )
