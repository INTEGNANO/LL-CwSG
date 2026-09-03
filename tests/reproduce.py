"""Reproduction driver: one subcommand per experiment (table2_cifar10,
noise_sweep, svhn_devices, adc_quant, bs1_fp32, pcm_mnist).

Defaults run 1 seed; --full runs the 5-seed/5-run paper protocol. Results
are written incrementally to results/ and each subcommand ends with a
console table comparing against the published reference values.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import (CIFAR10_TABLE2, NOISE_SWEEP, SVHN_DEVICES, ADC_QUANT,
                    BS1_FP32, PCM_MNIST)


def _save_json(path, payload):
    from helper import NumpyEncoder
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, cls=NumpyEncoder)
    os.replace(tmp, path)
    print(f"  [saved] {path}")


def _load_data(cfg, dataset=None, batch_size=None):
    import tensorflow as tf
    import helper

    channels, pooling, layers = helper.build_architecture(
        cfg["channel_widths"], pool_every=cfg["pool_every"])
    ds_train, ds_test, num_classes, input_shape = helper.load_dataset(
        dataset or cfg["dataset"],
        batch_size=batch_size or cfg["batch_size"], seed=0)
    ds_test = ds_test.cache().prefetch(tf.data.AUTOTUNE)
    return channels, pooling, layers, ds_train, ds_test, num_classes, input_shape


def _mean_std(values):
    return float(np.mean(values)), float(np.std(values))


def _fmt(mean, std=None):
    if std is None:
        return f"{mean:.2f}"
    return f"{mean:.2f} +/- {std:.2f}"


TABLE2_ALGOS = ["Backprop", "DCL", "CwC", "CwSG"]


def _greedy_fns(algo):
    """Pick the train-step and eval builders for a greedy layer-wise rule."""
    import fp32
    if algo == "CwSG":
        return fp32.make_cwsg_train_step, fp32.make_cwsg_evaluate_fn
    if algo == "CwC":
        return fp32.make_cwc_train_step, fp32.make_cwc_evaluate_fn
    if algo == "DCL":
        return fp32.make_dcl_train_step, fp32.make_dcl_evaluate_fn
    raise ValueError(f"Unknown greedy algorithm: {algo}")


def cmd_table2_cifar10(args):
    import helper
    import fp32

    cfg = CIFAR10_TABLE2
    seeds = (args.seeds if args.seeds is not None
             else (list(range(5)) if args.full else [0]))
    epochs = cfg["epochs_per_layer"]
    algos = args.algos

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "table2_cifar10.json")

    (channels, pooling, layers, ds_train, ds_test,
     num_classes, input_shape) = _load_data(cfg)

    payload = {
        "experiment": "table2_cifar10",
        "config": {
            "dataset": cfg["dataset"],
            "channel_widths": cfg["channel_widths"],
            "pool_every": cfg["pool_every"],
            "batch_size": cfg["batch_size"],
            "norm": cfg["norm"],
            "epochs_per_layer": epochs,
            "seeds": seeds,
            "hyperparameters": cfg["hps"],
        },
        "results": {},
    }

    for opt_name in ("sgd", "sign_sgd"):
        payload["results"][opt_name] = {}
        for algo in algos:
            hp = cfg["hps"][opt_name][algo]
            helper.set_stochastic_threshold(None)
            entry = {"hyperparameters": hp, "per_seed": {}}
            payload["results"][opt_name][algo] = entry
            for seed in seeds:
                print(f"\n[table2_cifar10] optimizer={opt_name}  algo={algo}  seed={seed}")
                if algo == "Backprop":
                    hist = fp32.train_backprop_curve(
                        hp["lr"], hp["temperature"], num_classes,
                        ds_train, ds_test, input_shape, cfg["norm"], epochs,
                        channels, pooling, opt_name, run_seed=seed)
                else:
                    make_step, make_eval = _greedy_fns(algo)
                    hist = fp32.train_greedy_curve(
                        make_step, make_eval, hp["lr"], hp["temperature"],
                        num_classes, ds_train, ds_test, input_shape,
                        cfg["norm"], epochs, channels, pooling, layers,
                        opt_name, run_seed=seed, algo_label=algo)
                entry["per_seed"][str(seed)] = hist
                finals = [h[-1]["test_acc"] for h in entry["per_seed"].values()]
                entry["final_test_acc_mean"], entry["final_test_acc_std"] = _mean_std(finals)
                _save_json(out_path, payload)

    print("\n" + "=" * 78)
    print(f"  Table 2, CIFAR-10 CNN-7 columns: final test accuracy (%), "
          f"{len(seeds)} seed(s)")
    print("=" * 78)
    print(f"  {'optimizer':<10} {'algorithm':<10} {'this run':>18}   {'paper (5 seeds)':>18}")
    for opt_name in ("sgd", "sign_sgd"):
        for algo in algos:
            e = payload["results"][opt_name][algo]
            mine = _fmt(e["final_test_acc_mean"],
                        e["final_test_acc_std"] if len(seeds) > 1 else None)
            ref = _fmt(*cfg["reference"][opt_name][algo])
            print(f"  {opt_name:<10} {algo:<10} {mine:>18}   {ref:>18}")
    print("=" * 78)


def cmd_noise_sweep(args):
    import helper
    import fp32

    cfg = NOISE_SWEEP
    num_runs = 5 if args.full else 1
    epochs = cfg["epochs_per_layer"]
    read_levels = (args.read_levels if args.read_levels
                   else cfg["read_levels"])
    write_levels = cfg["write_levels"]
    noise_pairs = [(r, w) for r in read_levels for w in write_levels]
    algos = args.algos
    eval_every = 10 ** 9

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "noise_sweep.json")

    (channels, pooling, layers, ds_train, ds_test,
     num_classes, input_shape) = _load_data(cfg)

    payload = {
        "experiment": "noise_sweep",
        "config": {
            "dataset": cfg["dataset"],
            "channel_widths": cfg["channel_widths"],
            "pool_every": cfg["pool_every"],
            "batch_size": cfg["batch_size"],
            "norm": cfg["norm"],
            "epochs_per_layer": epochs,
            "num_runs": num_runs,
            "eval_every": eval_every,
            "optimizer": cfg["optimizer"],
            "update_rule": "StoSignSGD (sign_sgd + stochastic Bernoulli pulse filter)",
            "weight_clamp": cfg["weight_clamp"],
            "write_noise": "multiplicative, LR-normalized, Gaussian",
            "read_levels": read_levels,
            "write_levels": write_levels,
            "hyperparameters": cfg["hps"],
        },
        "results": {},
    }

    for algo in algos:
        hp = cfg["hps"][algo]
        helper.set_weight_clamp(cfg["weight_clamp"])
        helper.set_stochastic_threshold(hp["threshold"])
        helper.set_write_noise_lr_normalize(hp["lr_conv"])
        print(f"\n[noise_sweep] {algo}: threshold={hp['threshold']} "
              f"lr_conv={hp['lr_conv']:.4e} T={hp['temperature']} "
              f"weight_clamp={cfg['weight_clamp']}")
        payload["results"].setdefault(algo, {})
        for pair in noise_pairs:
            if algo == "Backprop":
                res = fp32.sweep_backprop(
                    [pair], num_runs, epochs, ds_train, ds_test,
                    num_classes=num_classes, input_shape=input_shape,
                    norm=cfg["norm"], eval_every=eval_every,
                    channels=channels, pooling=pooling,
                    lr=hp["lr_conv"], temperature=hp["temperature"],
                    optimizer_name=cfg["optimizer"])
            else:
                res = fp32.sweep_cwsg(
                    [pair], num_runs, epochs, ds_train, ds_test,
                    num_classes=num_classes, input_shape=input_shape,
                    norm=cfg["norm"], eval_every=eval_every,
                    channels=channels, pooling=pooling, layers=layers,
                    lr_conv=hp["lr_conv"], temperature=hp["temperature"],
                    optimizer_name=cfg["optimizer"])
            payload["results"][algo].update(res)
            _save_json(out_path, payload)

    print("\n" + "=" * 78)
    print(f"  StoSignSGD noise sweep, final test accuracy (%), "
          f"{num_runs} run(s)/cell")
    print("=" * 78)
    print(f"  {'algorithm':<10} {'read':>6} {'write':>7} {'this run':>18}   {'paper (5 runs)':>16}")
    for algo in algos:
        for read_ns, write_ns in noise_pairs:
            key = f"r{read_ns:.4f}_w{write_ns:.4f}"
            final = payload["results"][algo][key][-1]
            mine = _fmt(final["avg_test_acc"],
                        final["std_test_acc"] if num_runs > 1 else None)
            ref = cfg["reference"].get(algo, {}).get(key)
            ref_str = _fmt(*ref) if ref else "-"
            print(f"  {algo:<10} {read_ns:>6.2f} {write_ns:>7.3f} "
                  f"{mine:>18}   {ref_str:>16}")
    print("=" * 78)


SVHN_DEVICES_ALGOS = ["Backprop", "CwSG"]


def cmd_svhn_devices(args):
    import helper
    import memristor

    cfg = SVHN_DEVICES
    seeds = (args.seeds if args.seeds is not None
             else (list(range(5)) if args.full else [0]))
    epochs = cfg["epochs_per_layer"]
    devices = args.devices

    present = [d for d in devices if memristor.device_files_present(d)]
    missing = [d for d in devices if d not in present]
    if missing:
        print("\n" + "!" * 78)
        for d in missing:
            print(f"!!  {d} device data not found (devices/{d}.hdf5) - skipped.")
        print("!!  The measurement files are available from the authors on "
              "request - see README.")
        print("!" * 78)
    if not present:
        sys.exit(0)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "svhn_devices.json")

    (channels, pooling, layers, ds_train, ds_test,
     num_classes, input_shape) = _load_data(cfg)

    payload = {
        "experiment": "svhn_devices",
        "config": {
            "dataset": cfg["dataset"],
            "channel_widths": cfg["channel_widths"],
            "pool_every": cfg["pool_every"],
            "batch_size": cfg["batch_size"],
            "norm": cfg["norm"],
            "epochs_per_layer": epochs,
            "update_mechanism": cfg["update_mechanism"],
            "weight_formula": cfg["weight_formula"],
            "eval_every": cfg["eval_every"],
            "seeds": seeds,
            "hyperparameters": cfg["device_hps"],
        },
        "results": {},
    }

    for device in present:
        memristor.set_device_type(device)
        mem_path = memristor.get_memristor_path(device)
        max_pulses = cfg["max_pulses_per_device"][device]
        payload["results"][device] = {}
        for algo in SVHN_DEVICES_ALGOS:
            hp = cfg["device_hps"][device][algo]
            per_run = []
            entry = {"hyperparameters": hp, "per_run": per_run}
            payload["results"][device][algo] = entry
            for seed in seeds:
                print(f"\n[svhn_devices] device={device}  algo={algo}  seed={seed}  "
                      f"(scale={hp['scale']} threshold={hp['threshold']} "
                      f"T={hp['temperature']})")
                common = dict(
                    scale=hp["scale"], threshold=hp["threshold"],
                    temperature=hp["temperature"], num_classes=num_classes,
                    ds_train=ds_train, ds_test=ds_test, input_shape=input_shape,
                    norm_name=cfg["norm"], memristor_path=mem_path,
                    channels=channels, pooling=pooling,
                    max_pulses_per_device=max_pulses,
                    run_seed=seed, eval_every=cfg["eval_every"])
                if algo == "Backprop":
                    result = memristor.run_mem_bp(epochs=epochs, **common)
                else:
                    result = memristor.run_mem_cwsg(
                        epochs_per_layer=epochs, layers=layers, **common)
                result["run_seed"] = seed
                per_run.append(result)
                entry["aggregate_per_epoch"] = memristor.aggregate_history(per_run)
                entry["aggregate_pulse_stats"] = memristor.aggregate_pulse_stats(per_run)
                finals = [r["final_test_acc"] for r in per_run]
                entry["final_test_acc_mean"], entry["final_test_acc_std"] = _mean_std(finals)
                _save_json(out_path, payload)

    print("\n" + "=" * 78)
    print(f"  SVHN CNN-7W in-situ device training (Figure 5 c and f), "
          f"{len(seeds)} seed(s)")
    print("=" * 78)
    print(f"  {'device':<7} {'algorithm':<10} {'this run acc':>18}   {'paper acc':>16} "
          f"{'pulses/dev':>11} {'paper':>7}")
    for device in present:
        for algo in SVHN_DEVICES_ALGOS:
            e = payload["results"][device][algo]
            g = e["aggregate_pulse_stats"]["global"]
            pulses = (g["pos_mean_weighted_mean"] + g["neg_mean_weighted_mean"]) / 2.0
            mine = _fmt(e["final_test_acc_mean"],
                        e["final_test_acc_std"] if len(seeds) > 1 else None)
            ref_m, ref_s, ref_p = cfg["reference"][device][algo]
            print(f"  {device:<7} {algo:<10} {mine:>18}   {_fmt(ref_m, ref_s):>16} "
                  f"{pulses:>11.1f} {'~' + format(ref_p, '.1f'):>7}")
    print("=" * 78)


ADC_QUANT_ARMS = ["Backprop", "BPq", "CwSG"]


def cmd_adc_quant(args):
    import memristor

    cfg = ADC_QUANT
    seeds = (args.seeds if args.seeds is not None
             else (list(range(5)) if args.full else [0]))
    epochs = cfg["epochs_per_layer"]
    devices = args.devices

    present = [d for d in devices if memristor.device_files_present(d)]
    missing = [d for d in devices if d not in present]
    if missing:
        print("\n" + "!" * 78)
        for d in missing:
            print(f"!!  {d} device data not found (devices/{d}.hdf5) - skipped.")
        print("!!  The measurement files are available from the authors on "
              "request - see README.")
        print("!" * 78)
    if not present:
        sys.exit(0)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "adc_quant.json")

    (channels, pooling, layers, ds_train, ds_test,
     num_classes, input_shape) = _load_data(cfg)

    payload = {
        "experiment": "adc_quant",
        "config": {
            "dataset": cfg["dataset"],
            "channel_widths": cfg["channel_widths"],
            "pool_every": cfg["pool_every"],
            "batch_size": cfg["batch_size"],
            "norm": cfg["norm"],
            "epochs_per_layer": epochs,
            "update_mechanism": cfg["update_mechanism"],
            "weight_formula": cfg["weight_formula"],
            "eval_every": cfg["eval_every"],
            "adc": cfg["adc"],
            "arms": cfg["arms"],
            "seeds": seeds,
            "hyperparameters": cfg["device_hps"],
        },
        "results": {},
    }

    for device in present:
        memristor.set_device_type(device)
        mem_path = memristor.get_memristor_path(device)
        max_pulses = cfg["max_pulses_per_device"][device]
        payload["results"][device] = {}
        for arm in args.algos:
            arm_cfg = cfg["arms"][arm]
            hp = cfg["device_hps"][device][arm]
            adc = {"bits": cfg["adc"]["bits"],
                   "dac_bits": cfg["adc"]["dac_bits"],
                   "quant_bwd": arm_cfg["quant_bwd"],
                   "bwd_drive": arm_cfg["bwd_drive"]}
            per_run = []
            entry = {"hyperparameters": hp, "adc": adc, "per_run": per_run}
            payload["results"][device][arm] = entry
            for seed in seeds:
                print(f"\n[adc_quant] device={device}  arm={arm}  seed={seed}  "
                      f"(b={adc['bits']} scale={hp['scale']} "
                      f"threshold={hp['threshold']} T={hp['temperature']})")
                common = dict(
                    scale=hp["scale"], threshold=hp["threshold"],
                    temperature=hp["temperature"], num_classes=num_classes,
                    ds_train=ds_train, ds_test=ds_test, input_shape=input_shape,
                    norm_name=cfg["norm"], memristor_path=mem_path,
                    channels=channels, pooling=pooling,
                    max_pulses_per_device=max_pulses,
                    run_seed=seed, eval_every=cfg["eval_every"], adc=adc)
                if arm_cfg["algo"] == "Backprop":
                    result = memristor.run_mem_bp(epochs=epochs, **common)
                else:
                    result = memristor.run_mem_cwsg(
                        epochs_per_layer=epochs, layers=layers, **common)
                result["run_seed"] = seed
                per_run.append(result)
                entry["aggregate_per_epoch"] = memristor.aggregate_history(per_run)
                entry["aggregate_pulse_stats"] = memristor.aggregate_pulse_stats(per_run)
                finals = [r["final_test_acc"] for r in per_run]
                entry["final_test_acc_mean"], entry["final_test_acc_std"] = _mean_std(finals)
                _save_json(out_path, payload)

    print("\n" + "=" * 78)
    print(f"  SVHN CNN-7W in-situ device training, 8-bit ADC = DAC datapath, "
          f"{len(seeds)} seed(s)")
    print("=" * 78)
    print(f"  {'device':<7} {'arm':<10} {'this run acc':>18}   {'paper acc':>16} "
          f"{'pulses/dev':>11} {'paper':>7}")
    for device in present:
        for arm in args.algos:
            e = payload["results"][device][arm]
            g = e["aggregate_pulse_stats"]["global"]
            pulses = (g["pos_mean_weighted_mean"] + g["neg_mean_weighted_mean"]) / 2.0
            mine = _fmt(e["final_test_acc_mean"],
                        e["final_test_acc_std"] if len(seeds) > 1 else None)
            ref_m, ref_s, ref_p = cfg["reference"][device][arm]
            print(f"  {device:<7} {arm:<10} {mine:>18}   {_fmt(ref_m, ref_s):>16} "
                  f"{pulses:>11.1f} {'~' + format(ref_p, '.1f'):>7}")
    print("=" * 78)


BS1_FP32_ALGOS = ["Backprop", "CwC", "CwSG"]


def cmd_bs1_fp32(args):
    import tensorflow as tf
    import helper
    import fp32

    cfg = BS1_FP32
    seeds = (args.seeds if args.seeds is not None
             else (list(range(5)) if args.full else [0]))
    datasets = args.datasets
    epochs = cfg["epochs"]

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "bs1_fp32.json")

    payload = {
        "experiment": "bs1_fp32",
        "config": {
            "channel_widths": cfg["channel_widths"],
            "pool_every": cfg["pool_every"],
            "batch_size": cfg["batch_size"],
            "eval_batch": cfg["eval_batch"],
            "norm": cfg["norm"],
            "optimizer": cfg["optimizer"],
            "update_rule": "StoSignSGD (sign_sgd + stochastic Bernoulli pulse filter)",
            "epochs_per_layer": epochs,
            "seeds": seeds,
            "hyperparameters": cfg["hps"],
        },
        "results": {},
    }

    for dataset in datasets:
        (channels, pooling, layers, ds_train, ds_test,
         num_classes, input_shape) = _load_data(cfg, dataset=dataset)
        # re-batch the test set: at BS=1 it would be one image per eval batch
        ds_test = (ds_test.unbatch().batch(cfg["eval_batch"])
                   .cache().prefetch(tf.data.AUTOTUNE))
        payload["results"][dataset] = {}
        for algo in BS1_FP32_ALGOS:
            hp = cfg["hps"][dataset][algo]
            helper.set_stochastic_threshold(hp["threshold"])
            entry = {"hyperparameters": hp, "per_seed": {}}
            payload["results"][dataset][algo] = entry
            for seed in seeds:
                print(f"\n[bs1_fp32] dataset={dataset}  algo={algo}  seed={seed}  "
                      f"lr={hp['lr_conv']:.4e}  T={hp['temperature']}  "
                      f"threshold={hp['threshold']}")
                if algo == "Backprop":
                    hist = fp32.train_backprop_curve(
                        hp["lr_conv"], hp["temperature"], num_classes,
                        ds_train, ds_test, input_shape, cfg["norm"], epochs,
                        channels, pooling, cfg["optimizer"], run_seed=seed)
                else:
                    make_step, make_eval = _greedy_fns(algo)
                    hist = fp32.train_greedy_curve(
                        make_step, make_eval, hp["lr_conv"], hp["temperature"],
                        num_classes, ds_train, ds_test, input_shape,
                        cfg["norm"], epochs, channels, pooling, layers,
                        cfg["optimizer"], run_seed=seed, algo_label=algo)
                entry["per_seed"][str(seed)] = hist
                finals = [h[-1]["test_acc"] for h in entry["per_seed"].values()]
                entry["final_test_acc_mean"], entry["final_test_acc_std"] = _mean_std(finals)
                _save_json(out_path, payload)

    print("\n" + "=" * 78)
    print(f"  Batch size 1, StoSignSGD, FP32 CNN-7W (Table 3 FP32 column), "
          f"{len(seeds)} seed(s)")
    print("=" * 78)
    print(f"  {'dataset':<8} {'algorithm':<10} {'this run':>18}   {'paper (5 seeds)':>18}")
    for dataset in datasets:
        for algo in BS1_FP32_ALGOS:
            e = payload["results"][dataset][algo]
            mine = _fmt(e["final_test_acc_mean"],
                        e["final_test_acc_std"] if len(seeds) > 1 else None)
            ref = _fmt(*cfg["reference"][dataset][algo])
            print(f"  {dataset:<8} {algo:<10} {mine:>18}   {ref:>18}")
    print("=" * 78)


PCM_MNIST_ALGOS = ["Backprop", "CwSG"]


def cmd_pcm_mnist(args):
    import helper
    import fp32
    import memristor

    cfg = PCM_MNIST
    seeds = list(range(5)) if args.full else [0]
    epochs = cfg["epochs_per_layer"]

    os.makedirs(args.out, exist_ok=True)
    fp32_path = os.path.join(args.out, "pcm_mnist_fp32.json")
    device_path = os.path.join(args.out, "pcm_mnist_device.json")

    (channels, pooling, layers, ds_train, ds_test,
     num_classes, input_shape) = _load_data(cfg)

    base_config = {
        "dataset": cfg["dataset"],
        "channel_widths": cfg["channel_widths"],
        "pool_every": cfg["pool_every"],
        "batch_size": cfg["batch_size"],
        "norm": cfg["norm"],
        "epochs_per_layer": epochs,
        "seeds": seeds,
    }

    helper.set_stochastic_threshold(None)
    fp32_payload = {
        "experiment": "pcm_mnist_fp32",
        "config": {**base_config, "optimizer": "sign_sgd",
                   "hyperparameters": cfg["fp32_hps"]},
        "results": {},
    }
    for algo in PCM_MNIST_ALGOS:
        hp = cfg["fp32_hps"][algo]
        entry = {"hyperparameters": hp, "per_seed": {}}
        fp32_payload["results"][algo] = entry
        for seed in seeds:
            print(f"\n[pcm_mnist/fp32] algo={algo}  seed={seed}")
            if algo == "Backprop":
                hist = fp32.train_backprop_curve(
                    hp["lr"], hp["temperature"], num_classes,
                    ds_train, ds_test, input_shape, cfg["norm"], epochs,
                    channels, pooling, "sign_sgd", run_seed=seed)
            else:
                hist = fp32.train_greedy_curve(
                    fp32.make_cwsg_train_step, fp32.make_cwsg_evaluate_fn,
                    hp["lr"], hp["temperature"], num_classes,
                    ds_train, ds_test, input_shape, cfg["norm"], epochs,
                    channels, pooling, layers, "sign_sgd",
                    run_seed=seed, algo_label="CwSG")
            entry["per_seed"][str(seed)] = hist
            finals = [h[-1]["test_acc"] for h in entry["per_seed"].values()]
            entry["final_test_acc_mean"], entry["final_test_acc_std"] = _mean_std(finals)
            _save_json(fp32_path, fp32_payload)

    def _print_fp32_table():
        print("\n" + "=" * 78)
        print(f"  FP32 SignSGD baselines (MNIST CNN-7W), "
              f"final test accuracy (%), {len(seeds)} seed(s)")
        print("=" * 78)
        print(f"  {'algorithm':<10} {'this run':>18}   {'paper':>8}")
        for algo in PCM_MNIST_ALGOS:
            e = fp32_payload["results"][algo]
            mine = _fmt(e["final_test_acc_mean"],
                        e["final_test_acc_std"] if len(seeds) > 1 else None)
            print(f"  {algo:<10} {mine:>18}   {cfg['reference']['fp32'][algo]:>8.2f}")
        print("=" * 78)

    if args.fp32_only or not memristor.device_files_present(cfg["device"]):
        if not memristor.device_files_present(cfg["device"]):
            print("\n" + "!" * 78)
            print("!!  WARNING: PCM device data not bundled with this repository;")
            print("!!  available from the authors on request - see README.")
            print("!!  Running FP32 baselines only.")
            print("!" * 78)
        else:
            print("\n--fp32-only requested: skipping the device runs.")
        _print_fp32_table()
        sys.exit(0)

    memristor.set_device_type(cfg["device"])
    mem_path = memristor.get_memristor_path(cfg["device"])

    device_payload = {
        "experiment": "pcm_mnist_device",
        "config": {**base_config,
                   "device": cfg["device"],
                   "memristor_path": mem_path,
                   "update_mechanism": cfg["update_mechanism"],
                   "weight_formula": cfg["weight_formula"],
                   "max_pulses_per_device": cfg["max_pulses_per_device"],
                   "eval_every": cfg["eval_every"],
                   "hyperparameters": cfg["device_hps"]},
        "results": {},
    }
    for algo in PCM_MNIST_ALGOS:
        hp = cfg["device_hps"][algo]
        per_run = []
        entry = {"hyperparameters": hp, "per_run": per_run}
        device_payload["results"][algo] = entry
        for seed in seeds:
            print(f"\n[pcm_mnist/device] algo={algo}  seed={seed}  "
                  f"(scale={hp['scale']} threshold={hp['threshold']} "
                  f"T={hp['temperature']})")
            common = dict(
                scale=hp["scale"], threshold=hp["threshold"],
                temperature=hp["temperature"], num_classes=num_classes,
                ds_train=ds_train, ds_test=ds_test, input_shape=input_shape,
                norm_name=cfg["norm"], memristor_path=mem_path,
                channels=channels, pooling=pooling,
                max_pulses_per_device=cfg["max_pulses_per_device"],
                run_seed=seed, eval_every=cfg["eval_every"])
            if algo == "Backprop":
                result = memristor.run_mem_bp(epochs=epochs, **common)
            else:
                result = memristor.run_mem_cwsg(
                    epochs_per_layer=epochs, layers=layers, **common)
            result["run_seed"] = seed
            per_run.append(result)
            entry["aggregate_per_epoch"] = memristor.aggregate_history(per_run)
            entry["aggregate_pulse_stats"] = memristor.aggregate_pulse_stats(per_run)
            finals = [r["final_test_acc"] for r in per_run]
            entry["final_test_acc_mean"], entry["final_test_acc_std"] = _mean_std(finals)
            _save_json(device_path, device_payload)

    _print_fp32_table()
    print("\n" + "=" * 78)
    print(f"  PCM device, StoSignSGD in-situ (MNIST CNN-7W), "
          f"{len(seeds)} seed(s)")
    print("=" * 78)
    print(f"  {'algorithm':<10} {'this run acc':>18}   {'paper acc':>16} "
          f"{'pulses/dev':>11} {'paper':>6}")
    for algo in PCM_MNIST_ALGOS:
        e = device_payload["results"][algo]
        g = e["aggregate_pulse_stats"]["global"]
        pulses_per_dev = (g["pos_mean_weighted_mean"] + g["neg_mean_weighted_mean"]) / 2.0
        mine = _fmt(e["final_test_acc_mean"],
                    e["final_test_acc_std"] if len(seeds) > 1 else None)
        ref_m, ref_s, ref_p = cfg["reference"]["device"][algo]
        print(f"  {algo:<10} {mine:>18}   {_fmt(ref_m, ref_s):>16} "
              f"{pulses_per_dev:>11.1f} {'~' + format(ref_p, '.1f'):>6}")
    print("=" * 78)


def build_parser():
    p = argparse.ArgumentParser(
        prog="reproduce.py",
        description="Reproduce the paper's experiments. Each subcommand "
                    "trains from scratch with the paper's winning "
                    "hyperparameters and writes JSON results (saved "
                    "incrementally) plus a console table comparing against "
                    "the paper's numbers.",
        epilog="Default: 1 seed (seed 0). --full runs the 5-seed/5-run paper "
               "protocol.")
    sub = p.add_subparsers(dest="experiment", required=True)

    p1 = sub.add_parser(
        "table2_cifar10",
        help="Table 2, CIFAR-10 CNN-7 columns: Backprop / DCL / CwC / CwSG, "
             "each under SGD and SignSGD")
    p1.add_argument("--full", action="store_true",
                    help="paper protocol: seeds 0-4 instead of seed 0")
    p1.add_argument("--seeds", type=int, nargs="+", default=None, metavar="N",
                    help="explicit list of seeds (overrides --full)")
    p1.add_argument("--algos", nargs="+", choices=TABLE2_ALGOS,
                    default=TABLE2_ALGOS,
                    help="subset of algorithms (to split across GPUs)")
    p1.add_argument("--out", default="results",
                    help="output directory (default: results/)")

    p3 = sub.add_parser(
        "noise_sweep",
        help="Figure 3b CNN-7 panel: StoSignSGD write-noise robustness sweep "
             "on CIFAR-10 (read noise 0), Backprop vs CwSG")
    p3.add_argument("--full", action="store_true",
                    help="paper protocol: 5 runs per grid cell instead of 1")
    p3.add_argument("--algos", nargs="+", choices=["Backprop", "CwSG"],
                    default=["Backprop", "CwSG"],
                    help="subset of algorithms (to split the sweep across GPUs)")
    p3.add_argument("--read-levels", type=float, nargs="+", default=None,
                    metavar="R",
                    help="override the read-noise levels (default: 0 only)")
    p3.add_argument("--out", default="results",
                    help="output directory (default: results/)")

    p4 = sub.add_parser(
        "svhn_devices",
        help="Figure 5 c and f: SVHN CNN-7W trained in-situ on calibrated "
             "PCM and FM (RRAM) devices, StoSignSGD pulses. Devices whose "
             "measurement file is missing are skipped.")
    p4.add_argument("--full", action="store_true",
                    help="paper protocol: seeds 0-4 instead of seed 0")
    p4.add_argument("--seeds", type=int, nargs="+", default=None, metavar="N",
                    help="explicit list of seeds (overrides --full)")
    p4.add_argument("--devices", nargs="+", choices=["pcm", "rram"],
                    default=["pcm", "rram"],
                    help="subset of devices (to split across GPUs)")
    p4.add_argument("--out", default="results",
                    help="output directory (default: results/)")

    p7 = sub.add_parser(
        "adc_quant",
        help="ADC quantization at the Figure 5 operating point: SVHN CNN-7W "
             "on calibrated PCM and FM (RRAM) devices with every periphery "
             "converter at 8 bits (ADC = DAC = 8). Arms: BP (forward "
             "converters only), BPq (backward drive and readout quantized "
             "too), CwSG. Devices whose measurement file is missing are "
             "skipped.")
    p7.add_argument("--full", action="store_true",
                    help="paper protocol: seeds 0-4 instead of seed 0")
    p7.add_argument("--seeds", type=int, nargs="+", default=None, metavar="N",
                    help="explicit list of seeds (overrides --full)")
    p7.add_argument("--devices", nargs="+", choices=["pcm", "rram"],
                    default=["pcm", "rram"],
                    help="subset of devices (to split across GPUs)")
    p7.add_argument("--algos", nargs="+", choices=ADC_QUANT_ARMS,
                    default=ADC_QUANT_ARMS,
                    help="subset of arms (to split across GPUs)")
    p7.add_argument("--out", default="results",
                    help="output directory (default: results/)")

    p6 = sub.add_parser(
        "bs1_fp32",
        help="Table 3 FP32 column: MNIST and SVHN CNN-7W at batch size 1 "
             "under StoSignSGD, Backprop / CwC / CwSG (plain FP32 training "
             "loops, one epoch of updates per parameter)")
    p6.add_argument("--full", action="store_true",
                    help="paper protocol: seeds 0-4 instead of seed 0")
    p6.add_argument("--seeds", type=int, nargs="+", default=None, metavar="N",
                    help="explicit list of seeds (overrides --full)")
    p6.add_argument("--datasets", nargs="+", choices=["mnist", "svhn"],
                    default=["mnist", "svhn"],
                    help="subset of datasets (to split across GPUs)")
    p6.add_argument("--out", default="results",
                    help="output directory (default: results/)")

    p5 = sub.add_parser(
        "pcm_mnist",
        help="MNIST CNN-7W trained in-situ on calibrated PCM devices "
             "(StoSignSGD pulses) + FP32 SignSGD baselines")
    p5.add_argument("--full", action="store_true",
                    help="paper protocol: seeds 0-4 instead of seed 0")
    p5.add_argument("--fp32-only", action="store_true",
                    help="run only the FP32 baselines (device runs are also "
                         "skipped automatically when the PCM data file is "
                         "not installed)")
    p5.add_argument("--out", default="results",
                    help="output directory (default: results/)")

    return p


def main():
    args = build_parser().parse_args()
    if args.experiment == "table2_cifar10":
        cmd_table2_cifar10(args)
    elif args.experiment == "noise_sweep":
        cmd_noise_sweep(args)
    elif args.experiment == "svhn_devices":
        cmd_svhn_devices(args)
    elif args.experiment == "adc_quant":
        cmd_adc_quant(args)
    elif args.experiment == "bs1_fp32":
        cmd_bs1_fp32(args)
    elif args.experiment == "pcm_mnist":
        cmd_pcm_mnist(args)


if __name__ == "__main__":
    main()
