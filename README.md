# LL-CwSG

Reference implementation and reproduction code for "TITLE (todo)": Backprop, DCL, CwC and CwSG in float32 and on
measured memristor devices (StoSignSGD pulse updates).

## Layout

| File | Contents |
|---|---|
| `helper.py` | Models, data, goodness functions, optimizers, noise injection, periphery ADC/DAC converters |
| `fp32.py` | FP32 train steps and loops (greedy and interleaved), noise-robustness sweeps |
| `memristor.py` | Device-calibrated training on measured PCM/RRAM traces |
| `tests/config.py` | Experiment configurations, winning hyperparameters, reference values |
| `tests/reproduce.py` | One subcommand per reproduced experiment |
| `tests/bs1_cifar10.py` | Standalone: CIFAR-10 Backprop at batch size 1 under StoSignSGD |

## Install

```bash
pip install -r requirements.txt
# GPU (recommended):
pip install "jax[cuda12]==0.4.30" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

MNIST/SVHN/CIFAR-10 download automatically via tensorflow-datasets. The
device measurements are available from the authors on request; place them as
`devices/pcm.hdf5` (PCM) and `devices/rram.hdf5` (filamentary memristor).
Device subcommands skip any device whose file is missing.

## Reproduce

```bash
python tests/reproduce.py table2_cifar10  # CIFAR-10 CNN-7: BP/DCL/CwC/CwSG, SGD + SignSGD
python tests/reproduce.py noise_sweep     # CNN-7, StoSignSGD write-noise sweep, BP vs CwSG
python tests/reproduce.py svhn_devices    # SVHN on calibrated PCM and FM devices
python tests/reproduce.py adc_quant       # 8-bit ADC = DAC datapath on the devices: BP, BPq, CwSG
python tests/reproduce.py bs1_fp32        # FP32 MNIST + SVHN at batch size 1
python tests/reproduce.py pcm_mnist       # MNIST on PCM devices (+ FP32 baselines)
python tests/bs1_cifar10.py               # CIFAR-10 Backprop, batch size 1, StoSignSGD (FP32)
```

Default is 1 seed; `--full` runs the 5-seed/5-run paper protocol. Most
subcommands take `--algos` / `--datasets` / `--devices` to split the work
across GPUs. Results are written incrementally to `results/*.json`;
