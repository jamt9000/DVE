"""Parse logs from experiments in deprecated format and summarise in new format.

NOTE: This module requires tensorflow.
"""
import time
import json
import argparse
import logging
import shutil
from pathlib import Path
import tensorflow as tf
from protobuf_to_dict import protobuf_to_dict
from test_matching import evaluation
from logger import setup_logging
from parse_config import ConfigParser


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
    for key, rel_dir in experiments.items():

        timestamp = rel_dir.split("/")[-1]
        model_dir = Path(save_dir) / "models"

        src_config = Path(rel_dir) / "config.json"
        config_path = model_dir / key / timestamp / "config.json"
        config_path.parent.mkdir(exist_ok=True, parents=True)
        if not config_path.exists() or refresh:
            print(f"copying config: {str(src_config)} -> {str(config_path)}")
            shutil.copyfile(str(src_config), str(config_path))
        else:
            print(f"transferred config found at {str(config_path)}, skipping...")

        src_model = Path(rel_dir) / "model_best.pth"
        model_path = model_dir / key / timestamp / "model_best.pth"
        if not model_path.exists() or refresh:
            print(f"copying model: {str(src_model)} -> {str(model_path)}")
            shutil.copyfile(str(src_model), str(model_path))
        else:
            print(f"transferred model found at {str(model_path)}, skipping...")

        dest_log = model_dir / key / timestamp / "model_best.path"
        if not dest_log.exists() or refresh:
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
        else:
            print(f"generated log found at {str(dest_log)}, skipping...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="data/saved")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--experiments_path", default="misc/experiments-deprecated.json")
    args = parser.parse_args()

    with open(args.experiments_path, "r") as f:
        old_format_experiments = json.load(f)

    modernize_exp_dir(
        save_dir=args.save_dir,
        experiments=old_format_experiments,
        refresh=args.refresh,
    )
