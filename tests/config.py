"""Experiment configurations, winning hyperparameters, and published reference values."""

# Table 1 & 2, CIFAR-10 / CNN-7 columns (SGD and SignSGD).
CIFAR10_TABLE2 = {
    "dataset": "cifar10",
    "channel_widths": [64, 128, 128, 256, 256, 256],
    "pool_every": 2,
    "batch_size": 64,
    "norm": "rmsnorm",
    "epochs_per_layer": 50,
    "hps": {
        "sgd": {
            "Backprop": {"lr": 1.3e-1,   "temperature": 1.0},
            "DCL":      {"lr": 1.4e-2, "temperature": 1.0},
            "CwC":      {"lr": 4.5e-2,  "temperature": 0.1},
            "CwSG":     {"lr": 9.2e-1,   "temperature": 0.1},
        },
        "sign_sgd": {
            "Backprop": {"lr": 1.1e-5, "temperature": 2.0},
            "DCL":      {"lr": 1.5e-3,   "temperature": 2.0},
            "CwC":      {"lr": 7.2e-5,  "temperature": 0.1},
            "CwSG":     {"lr": 7.5e-4,  "temperature": 0.1},
        },
    },
    # 5-seed mean +/- std from the paper
    "reference": {
        "sgd":      {"Backprop": (86.28, 1.60), "DCL": (77.28, 1.27),
                     "CwC": (77.77, 0.67), "CwSG": (83.66, 1.24)},
        "sign_sgd": {"Backprop": (60.65, 1.43), "DCL": (66.56, 0.41),
                     "CwC": (74.41, 0.72), "CwSG": (83.26, 0.32)},
    },
}

# Figure 3, CNN-7 panel: StoSignSGD write-noise sweep, read noise 0.
NOISE_SWEEP = {
    "dataset": "cifar10",
    "channel_widths": [64, 128, 128, 256, 256, 256],
    "pool_every": 2,
    "batch_size": 64,
    "norm": "rmsnorm",
    "epochs_per_layer": 50,
    "optimizer": "sign_sgd",
    "weight_clamp": 1.0,
    "write_levels": [0.0, 0.001, 0.003, 0.01, 0.03, 0.1],
    "read_levels": [0.0],
    "hps": {
        "Backprop": {"lr_conv": 2.6e-3, "temperature": 1.0,
                     "threshold": 100.0},
        "CwSG":     {"lr_conv": 1.7e-3, "temperature": 0.1,
                     "threshold": 10.0},
    },
    # 5 runs/level from the paper
    "reference": {
        "Backprop": {"r0.0000_w0.0000": (84.3, 2.4),
                     "r0.0000_w0.0010": (83.7, 1.2),
                     "r0.0000_w0.0030": (83.0, 0.8),
                     "r0.0000_w0.0100": (65.9, 1.1),
                     "r0.0000_w0.0300": (32.3, 1.6),
                     "r0.0000_w0.1000": (12.4, 1.2)},
        "CwSG":     {"r0.0000_w0.0000": (81.2, 0.3),
                     "r0.0000_w0.0010": (80.2, 0.4),
                     "r0.0000_w0.0030": (76.0, 0.2),
                     "r0.0000_w0.0100": (63.8, 0.6),
                     "r0.0000_w0.0300": (45.2, 0.7),
                     "r0.0000_w0.1000": (22.7, 1.2)},
    },
}

# Figure 5: SVHN / CNN-7W
# batch size 256, StoSignSGD
SVHN_DEVICES = {
    "dataset": "svhn",
    "channel_widths": [128, 256, 256, 512, 512, 512],
    "pool_every": 2,
    "batch_size": 256,
    "norm": "rmsnorm",
    "epochs_per_layer": 50,
    "update_mechanism": "stochastic",
    "weight_formula": "conductance_diff",
    "eval_every": 1,
    # PCM traces hold 10000 pulses per device, RRAM (FM) traces 5000
    "max_pulses_per_device": {"pcm": 10000, "rram": 5000},
    "device_hps": {
        "pcm": {
            "Backprop": {"scale": 10000.0, "threshold": 1000.0, "temperature": 10.0},
            "CwSG":     {"scale": 10000.0, "threshold": 10.0,   "temperature": 0.1},
        },
        "rram": {
            "Backprop": {"scale": 2500.0, "threshold": 3000.0, "temperature": 2.0},
            "CwSG":     {"scale": 1000.0, "threshold": 100.0,  "temperature": 0.1},
        },
    },
    # (acc mean, acc std, pulses/device), 5 seeds
    "reference": {
        "pcm":  {"Backprop": (82.21, 0.70, 378.0), "CwSG": (82.52, 0.54, 6.4)},
        "rram": {"Backprop": (31.49, 2.23, 3280.0), "CwSG": (84.88, 0.58, 1688.0)},
    },
}

# Table 3, FP32 column: single-sample (batch size 1), StoSignSGD,
# greedy scheduler, one epoch per layer (BP: one end-to-end epoch).
BS1_FP32 = {
    "channel_widths": [128, 256, 256, 512, 512, 512],
    "pool_every": 2,
    "batch_size": 1,
    "norm": "rmsnorm",
    "epochs": 1,
    "optimizer": "sign_sgd",
    "eval_batch": 250,
    "hps": {
        "mnist": {
            "Backprop": {"lr_conv": 3.8e-3, "temperature": 100.0,
                         "threshold": 100.0},
            "CwC":      {"lr_conv": 1.9e-3, "temperature": 0.01,
                         "threshold": 0.3},
            "CwSG":     {"lr_conv": 1.2e-2,  "temperature": 0.1,
                         "threshold": 0.3},
        },
        "svhn": {
            "Backprop": {"lr_conv": 5.9e-5, "temperature": 1.0,
                         "threshold": 1.0},
            "CwC":      {"lr_conv": 8.4e-4,   "temperature": 0.1,
                         "threshold": 3.0},
            "CwSG":     {"lr_conv": 1.5e-3,  "temperature": 0.1,
                         "threshold": 1.0},
        },
    },
    # 5-seed mean +/- std from the paper (SVHN Backprop sits at chance:
    # a single pass of one-sample sign updates is not enough for BP there)
    "reference": {
        "mnist": {"Backprop": (98.96, 0.46), "CwC": (98.59, 0.29),
                  "CwSG": (98.61, 0.13)},
        "svhn":  {"Backprop": (21.81, 2.83), "CwC": (77.85, 4.47),
                  "CwSG": (87.05, 0.32)},
    },
}

CIFAR10_BS1 = {
    # Single-sample streaming (batch size 1) under StoSignSGD, FP32, on the
    # full 50,000-image training set. This is the regime where plain SignSGD
    # collapses: with one sample per step the gradient sign carries no batch
    # averaging, and only the stochastic Bernoulli pulse filter keeps
    # Backprop trainable.
    "dataset": "cifar10",
    "channel_widths": [128, 256, 256, 512, 512, 512],   # CNN-7W (2x)
    "pool_every": 2,
    "batch_size": 1,
    "norm": "rmsnorm",
    "epochs": 5,                      # end-to-end epochs (Backprop)
    "optimizer": "sign_sgd",          # + stochastic threshold => StoSignSGD
    # ds_test inherits batch_size from load_dataset; at BS=1 that would be
    # 10,000 single-image eval batches, so the test set is re-batched.
    "eval_batch": 250,
    "hps": {
        "Backprop": {"lr_conv": 1.0e-3,
                     "temperature": 10.0, "threshold": 10.0},
    },
    # single-run HPO best (n_runs_per_trial = 1), not a seed mean
    "reference": {"Backprop": 79.01},
}

# ADC/DAC quantization at the Figure 5 operating point: SVHN / CNN-7W trained
# in-situ on the measured devices with EVERY periphery converter at 8 bits
# (joint ADC = DAC = b datapath: each array's readout, each array's drive, and
# the network input). Readouts are signed over a per-layer full scale
# B_l = C * w_rail * sqrt(9*C_in) (dense head: sqrt(C_last)); drives are
# unipolar (hidden range 4.0, input range 1.0, head range 2.0).
# Hyperparameters are the SVHN_DEVICES anchors, held fixed with no per-bit
# retuning; BPq trains with the Backprop hyperparameters.
ADC_QUANT = {
    "dataset": "svhn",
    "channel_widths": [128, 256, 256, 512, 512, 512],
    "pool_every": 2,
    "batch_size": 256,
    "norm": "rmsnorm",
    "epochs_per_layer": 50,
    "update_mechanism": "stochastic",
    "weight_formula": "conductance_diff",
    "eval_every": 1,
    "max_pulses_per_device": {"pcm": 10000, "rram": 5000},
    "adc": {"bits": 8, "dac_bits": 8, "c_design": 0.5,
            "in_range": 4.0, "input_range": 1.0, "head_range": 2.0},
    # BP = forward converters only, backward reads ideal (converter-matched
    # with CwSG, so their difference is the rule and not the periphery).
    # BPq = the physically complete backprop: the error driving the transposed
    # array and its readout are quantized too. CwSG has no cross-layer
    # backward pass, so the setting does not exist for it.
    "arms": {
        "Backprop": {"algo": "Backprop", "quant_bwd": False, "bwd_drive": False},
        "BPq":      {"algo": "Backprop", "quant_bwd": True,  "bwd_drive": True},
        "CwSG":     {"algo": "CwSG",     "quant_bwd": False, "bwd_drive": False},
    },
    "device_hps": {
        "pcm": {
            "Backprop": {"scale": 10000.0, "threshold": 1000.0, "temperature": 10.0},
            "BPq":      {"scale": 10000.0, "threshold": 1000.0, "temperature": 10.0},
            "CwSG":     {"scale": 10000.0, "threshold": 10.0,   "temperature": 0.1},
        },
        "rram": {
            "Backprop": {"scale": 2500.0, "threshold": 3000.0, "temperature": 2.0},
            "BPq":      {"scale": 2500.0, "threshold": 3000.0, "temperature": 2.0},
            "CwSG":     {"scale": 1000.0, "threshold": 100.0,  "temperature": 0.1},
        },
    },
    # (acc mean, acc std, pulses/device), 5 seeds. rram BPq is bimodal at
    # 8 bits: 3/5 runs converge (converged runs 30.18 +/- 2.86), the mean
    # below covers all five.
    "reference": {
        "pcm":  {"Backprop": (82.09, 0.74, 362.1),
                 "BPq":      (74.41, 1.42, 398.6),
                 "CwSG":     (81.60, 0.84, 6.7)},
        "rram": {"Backprop": (31.27, 2.11, 3339.1),
                 "BPq":      (27.03, 4.46, 1502.3),
                 "CwSG":     (84.54, 0.11, 1711.5)},
    },
}

PCM_MNIST = {
    "dataset": "mnist",
    "channel_widths": [128, 256, 256, 512, 512, 512],
    "pool_every": 2,
    "batch_size": 256,
    "norm": "rmsnorm",
    "epochs_per_layer": 30,
    "device": "pcm",
    "update_mechanism": "stochastic",
    "weight_formula": "conductance_diff",
    "max_pulses_per_device": 10000,
    "eval_every": 1,
    "device_hps": {
        "Backprop": {"scale": 5000.0,  "threshold": 100.0, "temperature": 10.0},
        "CwSG":     {"scale": 10000.0, "threshold": 100.0, "temperature": 0.1},
    },
    "fp32_hps": {
        "Backprop": {"lr": 3.7e-4, "temperature": 10.0},
        "CwSG":     {"lr": 2.3e-4, "temperature": 0.01},
    },
    "reference": {
        "device": {"Backprop": (98.56, 0.03, 8.0), "CwSG": (98.53, 0.06, 16.2)},
        "fp32":   {"Backprop": 99.44, "CwSG": 99.35},
    },
}
