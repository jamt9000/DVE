"""Log parser

python parse_logs.py --log_path data/grid-logs/scarce-data.txt.2019-06-23_15-12-22
"""
import argparse
import numpy as np
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument(
    "--log_path",
    default="data/grid-logs/scarce-data.txt.2019-06-23_14-50-28",
)
parser.add_argument("--key_metric", default="miou")
args = parser.parse_args()

with open(args.log_path, "r") as f:
    log = f.read().splitlines()

metric_names = ["acc", "clsacc", "fwacc", "miou"]
best_metrics = {key: defaultdict(list) for key in metric_names}
# defaultdict(list)

exp_name = None
metrics = None
for row in log:
    if row.startswith("LAUNCHING:"):
        if metrics is not None:
            best_epoch = np.argmax(metrics[args.key_metric])
            # exp name has the format <03d>-of-<03d>-stem-seed<seed_num>
            for key, val in metrics.items():
                stem = "-".join(exp_name.split("-")[3:-1])
                best_metrics[key][stem].append(val[best_epoch])
                # print("{}: {}, {}".format(key, np.mean(val), np.std(val)))

        exp_name = row.split(" ")[1].split("/")[-1].split(".")[0]
        metrics = defaultdict(list)
        # print("parsing: {}".format(exp_name))
    if exp_name:
        for metric in metric_names:
            if row.startswith(metric):
                val = float(row.split(" ")[1])
                metrics[metric].append(val)
    # print(row)
for key, val in best_metrics.items():
    print("Metric: {}".format(key))
    for subkey, subval in val.items():
        print("{:.2f}, {:.2f} {}".format(np.mean(subval), np.std(subval), subkey))
