"""A small utility for filling in the README paths to experiment artifacts.

The template contains tags of the form {{filetype.experiment_name}}, which are then
replaced with the urls for each resource.
"""
import re
import json
import argparse
import subprocess
from millify import millify
import numpy as np
from pathlib import Path
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


def sync_files(experiments, save_dir, webserver, web_dir):
    filetypes = {
        "log": ["info.log"],
        "models": ["model_best.pth", "config.json"]
    }
    for key, rel_dir in experiments.items():

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
        msg = f"expected single occurence of Mean Pixel Error tag in {log_path}"
        assert sum(presence) == 1, msg
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
    parser.add_argument("--webserver", default="login.robots.ox.ac.uk")
    parser.add_argument("--results_path", default="misc/results.json")
    parser.add_argument("--experiments_path", default="misc/experiments.json")
    parser.add_argument("--readme_template", default="misc/README-template.md")
    parser.add_argument("--readme_dest", default="README.md")
    parser.add_argument("--task", default="generate_readme",
                        choices=["sync_files", "generate_readme"])
    parser.add_argument("--web_dir", default="/projects/vgg/vgg/WWW/research/DVE/data")
    parser.add_argument("--root_url",
                        default="http://www.robots.ox.ac.uk/~vgg/research/DVE/data")
    args = parser.parse_args()

    with open(args.experiments_path, "r") as f:
        experiments = json.load(f)

    if args.task == "sync_files":
        sync_files(
            web_dir=args.web_dir,
            save_dir=args.save_dir,
            webserver=args.webserver,
            experiments=experiments,
        )
    elif args.task == "generate_readme":
        generate_readme(
            root_url=args.root_url,
            readme_template=args.readme_template,
            readme_dest=args.readme_dest,
            results_path=args.results_path,
            save_dir=args.save_dir,
            experiments=experiments,
        )
