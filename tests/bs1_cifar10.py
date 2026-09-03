import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import CIFAR10_BS1


def _save_json(path, payload):
    from helper import NumpyEncoder
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, cls=NumpyEncoder)
    os.replace(tmp, path)
    print(f"  [saved] {path}")


def _fmt(mean, std=None):
    return f"{mean:.2f}" if std is None else f"{mean:.2f} +/- {std:.2f}"


def run(args):
    import tensorflow as tf
    import helper
    import fp32

    cfg = CIFAR10_BS1
    hp = cfg["hps"]["Backprop"]
    seeds = args.seeds if args.seeds is not None else [0]
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    eval_batch = args.eval_batch or cfg["eval_batch"]

    channels, pooling, layers = helper.build_architecture(
        cfg["channel_widths"], pool_every=cfg["pool_every"])
    ds_train, ds_test, num_classes, input_shape = helper.load_dataset(
        cfg["dataset"], batch_size=cfg["batch_size"], seed=0)
    # load_dataset batches the test set at the TRAIN batch size; at BS=1 that
    # is 10,000 one-image forward passes per evaluation. Re-batch it.
    ds_test = (ds_test.unbatch().batch(eval_batch)
               .cache().prefetch(tf.data.AUTOTUNE))

    # StoSignSGD = sign_sgd + Bernoulli pulse filter. Must be set BEFORE the
    helper.set_stochastic_threshold(hp["threshold"])

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "bs1_cifar10.json")

    payload = {
        "experiment": "bs1_cifar10",
        "config": {
            "dataset": cfg["dataset"],
            "channel_widths": cfg["channel_widths"],
            "pool_every": cfg["pool_every"],
            "batch_size": cfg["batch_size"],
            "eval_batch": eval_batch,
            "norm": cfg["norm"],
            "optimizer": cfg["optimizer"],
            "update_rule": "StoSignSGD (sign_sgd + stochastic Bernoulli "
                           "pulse filter)",
            "epochs_end_to_end": epochs,
            "seeds": seeds,
            "hyperparameters": hp,
        },
        "results": {"Backprop": {"hyperparameters": hp, "per_seed": {}}},
    }
    entry = payload["results"]["Backprop"]

    for seed in seeds:
        print(f"\n[bs1_cifar10] Backprop  seed={seed}  "
              f"lr={hp['lr_conv']:.4e}  T={hp['temperature']}  "
              f"threshold={hp['threshold']}")
        hist = fp32.train_backprop_curve(
            hp["lr_conv"], hp["temperature"], num_classes,
            ds_train, ds_test, input_shape, cfg["norm"], epochs,
            channels, pooling, cfg["optimizer"], run_seed=seed)
        entry["per_seed"][str(seed)] = hist
        finals = [h[-1]["test_acc"] for h in entry["per_seed"].values()]
        entry["final_test_acc_mean"] = float(np.mean(finals))
        entry["final_test_acc_std"] = float(np.std(finals))
        _save_json(out_path, payload)

    ref = cfg["reference"]["Backprop"]
    mine = _fmt(entry["final_test_acc_mean"],
                entry["final_test_acc_std"] if len(seeds) > 1 else None)
    print("\n" + "=" * 78)
    print(f"  CIFAR-10 CNN-7W, batch size 1, StoSignSGD (FP32) — "
          f"{len(seeds)} seed(s)")
    print("=" * 78)
    print(f"  {'algorithm':<10} {'this run':>18}   {'paper (1 run)':>14}")
    print(f"  {'Backprop':<10} {mine:>18}   {ref:>14.2f}")
    print("=" * 78)


def build_parser():
    p = argparse.ArgumentParser(
        description="Reproduce CIFAR-10 Backprop at batch size 1 under "
                    "StoSignSGD in FP32 (CNN-7W, full 50k training set). "
                    "Trains from scratch with the paper's winning "
                    "hyperparameters, writes JSON to results/ and prints a "
                    "comparison table.")
    p.add_argument("--seeds", type=int, nargs="+", default=None, metavar="N",
                   help="seeds to run (default: seed 0 only)")
    p.add_argument("--epochs", type=int, default=None,
                   help=f"override the end-to-end epoch count "
                        f"(default: {CIFAR10_BS1['epochs']})")
    p.add_argument("--eval-batch", type=int, default=None,
                   help=f"test-set batch size "
                        f"(default: {CIFAR10_BS1['eval_batch']}); the train "
                        f"batch stays 1")
    p.add_argument("--out", default="results",
                   help="output directory (default: results/)")
    return p


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
