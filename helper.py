"""Shared building blocks: datasets, CNN models with optional read noise,
CwSG/CwC goodness functions, optimizers, write noise / clamping, evaluation.
Module globals are baked in at JIT-trace time: call the set_* functions
BEFORE building train steps and never from-import their values.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
try:
    import nvidia.cudnn as _cudnn_pkg
    if getattr(_cudnn_pkg, '__file__', None):
        _cudnn = os.path.join(os.path.dirname(_cudnn_pkg.__file__), 'lib')
        if os.path.isdir(_cudnn):
            os.environ["LD_LIBRARY_PATH"] = _cudnn + ":" + os.environ.get("LD_LIBRARY_PATH", "")
except ImportError:
    pass

import json
import functools
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from flax import linen as nn
from flax.core import FrozenDict

# Pin float32 matmuls/convs; TF32 on Ampere+ changes training trajectories.
jax.config.update("jax_default_matmul_precision", "float32")

DATASET_CHOICES = ["mnist", "svhn", "cifar10"]
TRAIN_SIZES = {"mnist": 60000, "svhn": 73257, "cifar10": 50000}

CIFAR10_MEAN = tf.constant([0.4914, 0.4822, 0.4465], dtype=tf.float32)
CIFAR10_STD  = tf.constant([0.2023, 0.1994, 0.2010], dtype=tf.float32)


def prefetch_to_device(dataset, size=128):
    """Prefetch batches to GPU via a background producer thread."""
    import threading, queue as queue_mod
    q = queue_mod.Queue(maxsize=size)

    def producer():
        for batch in tfds.as_numpy(dataset):
            q.put(jax.device_put(batch))
        q.put(None)

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is None:
            break
        yield item


def load_dataset(name, batch_size=64, seed=0, max_train_samples=None,
                 use_augmentation=True):
    """Load a dataset as batched tf.data pipelines.
    Returns (ds_train, ds_test, num_classes, input_shape)."""
    ds_name = {"mnist": "mnist", "svhn": "svhn_cropped", "cifar10": "cifar10"}
    num_classes = {"mnist": 10, "svhn": 10, "cifar10": 10}
    input_shapes = {"mnist": (28, 28, 1), "svhn": (32, 32, 3),
                    "cifar10": (32, 32, 3)}

    if name not in num_classes:
        raise ValueError(f"Unknown dataset: {name}. Choose from {DATASET_CHOICES}")

    if name == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
        img_size = 32
        pad = 4

        def preprocess_train(example):
            image = tf.cast(example["image"], tf.float32) / 255.0
            if use_augmentation:
                image = tf.pad(image, paddings=[[pad, pad], [pad, pad], [0, 0]])
                image = tf.image.random_crop(image, size=[img_size, img_size, 3])
                image = tf.image.random_flip_left_right(image)
            image = (image - mean) / std
            return image, tf.cast(example["label"], tf.int32)

        def preprocess_test(example):
            image = tf.cast(example["image"], tf.float32) / 255.0
            image = (image - mean) / std
            return image, tf.cast(example["label"], tf.int32)
    else:
        def preprocess_train(example):
            image = tf.cast(example["image"], tf.float32) / 255.0
            return image, tf.cast(example["label"], tf.int32)

        preprocess_test = preprocess_train

    # .cache() must precede the augmentation .map so augmentation is re-sampled every epoch.
    train_split = "train"
    if max_train_samples is not None:
        train_split = f"train[:{max_train_samples}]"
    ds_train = (
        tfds.load(ds_name[name], split=train_split, shuffle_files=True)
        .cache()
        .map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(max_train_samples or TRAIN_SIZES[name], seed=seed)
        .batch(batch_size).prefetch(tf.data.AUTOTUNE)
    )
    ds_test = (
        tfds.load(ds_name[name], split="test")
        .cache()
        .map(preprocess_test, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size).prefetch(tf.data.AUTOTUNE)
    )
    return ds_train, ds_test, num_classes[name], input_shapes[name]


def build_architecture(channel_widths, pool_every=2):
    """Build (channels, pooling, layers) from channel widths, pooling every
    pool_every-th layer; the dicts are FrozenDicts (static/hashable for JIT)."""
    channels = {}
    pooling = {}
    for i, ch in enumerate(channel_widths):
        name = f"conv{i+1}"
        channels[name] = ch
        pooling[name] = 1 if (i + 1) % pool_every == 0 else 0
    layers = list(channels.keys())
    return FrozenDict(channels), FrozenDict(pooling), layers


def _noisy_conv_impl(x, w, noise_std, key_fwd, stride):
    """Forward pass of noisy conv: y = conv(x, W * (1 + eps_fwd))."""
    B, H, W_sp, C_in = x.shape
    kh, kw, _, C_out = w.shape
    H_out = (H + stride[0] - 1) // stride[0]
    W_out = (W_sp + stride[1] - 1) // stride[1]
    eps = jax.random.normal(key_fwd, (B, kh, kw, C_in, C_out))
    w_noisy = w[None, ...] * (1.0 + noise_std * eps)
    x_g = x.transpose((1, 2, 0, 3)).reshape((1, H, W_sp, B * C_in))
    w_g = w_noisy.transpose((1, 2, 3, 0, 4)).reshape((kh, kw, C_in, B * C_out))
    out_g = jax.lax.conv_general_dilated(
        x_g, w_g, stride, 'SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=B
    )
    return out_g[0].reshape((H_out, W_out, B, C_out)).transpose((2, 0, 1, 3))


def noisy_conv(x, w, noise_std, key, stride=(1, 1)):
    """Conv with per-sample multiplicative read noise; forward and backward
    use independent noise realizations. w: (kh, kw, C_in, C_out)."""
    if noise_std > 0:
        key_fwd, key_bwd = jax.random.split(key)
        return _noisy_conv_custom(x, w, noise_std, key_fwd, key_bwd, stride)
    return jax.lax.conv_general_dilated(x, w, stride, 'SAME',
                                         dimension_numbers=('NHWC', 'HWIO', 'NHWC'))


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 5))
def _noisy_conv_custom(x, w, noise_std, key_fwd, key_bwd, stride):
    """Custom VJP conv: forward uses key_fwd noise, backward uses key_bwd noise."""
    return _noisy_conv_impl(x, w, noise_std, key_fwd, stride)

def _noisy_conv_fwd(x, w, noise_std, key_fwd, key_bwd, stride):
    """VJP forward rule: compute y and save residuals (both keys)."""
    y = _noisy_conv_impl(x, w, noise_std, key_fwd, stride)
    return y, (x, w, key_fwd, key_bwd)

def _noisy_conv_bwd(noise_std, stride, res, g):
    """Backward rule: dx reads W with fresh key_bwd noise; dw reuses the
    forward noise (no extra W read on hardware)."""
    x, w, key_fwd, key_bwd = res

    def fwd_for_dx(x_):
        return _noisy_conv_impl(x_, w, noise_std, key_bwd, stride)
    _, vjp_dx = jax.vjp(fwd_for_dx, x)
    (dx,) = vjp_dx(g)

    def fwd_for_dw(w_):
        return _noisy_conv_impl(x, w_, noise_std, key_fwd, stride)
    _, vjp_dw = jax.vjp(fwd_for_dw, w)
    (dw,) = vjp_dw(g)

    return dx, dw, None, None

_noisy_conv_custom.defvjp(_noisy_conv_fwd, _noisy_conv_bwd)


def noisy_matmul(x, w, noise_std, key):
    """Dense matmul with per-sample multiplicative read noise; forward and
    backward use independent noise realizations."""
    if noise_std > 0:
        key_fwd, key_bwd = jax.random.split(key)
        return _noisy_matmul_custom(x, w, noise_std, key_fwd, key_bwd)
    return jnp.dot(x, w)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _noisy_matmul_custom(x, w, noise_std, key_fwd, key_bwd):
    """Custom-VJP matmul: forward uses key_fwd noise, backward key_bwd."""
    B = x.shape[0]
    eps = jax.random.normal(key_fwd, (B,) + w.shape)
    w_noisy = w[None, ...] * (1.0 + noise_std * eps)
    return jnp.einsum('bi,bij->bj', x, w_noisy)

def _noisy_matmul_fwd(x, w, noise_std, key_fwd, key_bwd):
    """VJP forward rule: compute y and save residuals (both keys)."""
    B = x.shape[0]
    eps_fwd = jax.random.normal(key_fwd, (B,) + w.shape)
    w_noisy_fwd = w[None, ...] * (1.0 + noise_std * eps_fwd)
    y = jnp.einsum('bi,bij->bj', x, w_noisy_fwd)
    return y, (x, w, key_fwd, key_bwd)

def _noisy_matmul_bwd(noise_std, res, g):
    """dx: independent noise (fresh W read). dw: forward noise (no extra W read)."""
    x, w, key_fwd, key_bwd = res
    B = x.shape[0]

    eps_bwd = jax.random.normal(key_bwd, (B,) + w.shape)
    w_noisy_bwd = w[None, ...] * (1.0 + noise_std * eps_bwd)
    dx = jnp.einsum('bj,bij->bi', g, w_noisy_bwd)

    eps_fwd = jax.random.normal(key_fwd, (B,) + w.shape)
    dw = jnp.einsum('bi,bj->bij', x, g) * (1.0 + noise_std * eps_fwd)
    dw = dw.mean(axis=0)

    return dx, dw, None, None

_noisy_matmul_custom.defvjp(_noisy_matmul_fwd, _noisy_matmul_bwd)


def rms_norm(x, axis=-1, eps=1e-8):
    """RMS normalization along `axis`."""
    return x / (jnp.sqrt(jnp.mean(x ** 2, axis=axis, keepdims=True) + eps))

def get_norm_fn(name):
    """Return a stateless norm function. None means no normalization."""
    if name is None or name == "none": return lambda x: x
    if name == "rmsnorm": return rms_norm
    raise ValueError(f"Unknown norm: {name}")



_ADC_BITS = None          # readout resolution in bits, None = float
_ADC_DAC_BITS = None      # drive resolution in bits, None = ideal drive
_ADC_OUT_BOUND = {}       # {layer: full scale of that layer's readout}
_ADC_IN_RANGE = 4.0       # full scale of the hidden-array drives
_ADC_INPUT_RANGE = 1.0    # full scale of the network-input drive ([0,1] images)
_ADC_HEAD_RANGE = 2.0     # full scale of the dense head's drive (pooled ReLU mean)
_ADC_BWD = False          # digitize the transposed-array readouts (Backprop)
_ADC_BWD_DRIVE = False    # digitize the error DRIVING the transposed array
_ADC_BWD_BOUND_SCALE = 1.0  # backward readout bound, relative to the forward one


def set_adc_config(bits=None, dac_bits=None, out_bound=None,
                   quant_bwd=False, quant_bwd_drive=False,
                   in_range=4.0, input_range=1.0, bwd_bound_scale=1.0):
    """Configure the periphery converters (no args = float bypass). Call
    before building a train step, since JIT bakes the choice into the graph."""
    global _ADC_BITS, _ADC_DAC_BITS, _ADC_OUT_BOUND, _ADC_IN_RANGE
    global _ADC_INPUT_RANGE, _ADC_BWD, _ADC_BWD_DRIVE, _ADC_BWD_BOUND_SCALE
    _ADC_BITS = None if bits is None else int(bits)
    _ADC_DAC_BITS = None if dac_bits is None else int(dac_bits)
    _ADC_OUT_BOUND = dict(out_bound or {})
    _ADC_IN_RANGE = float(in_range)
    _ADC_INPUT_RANGE = float(input_range)
    _ADC_BWD = bool(quant_bwd)
    _ADC_BWD_DRIVE = bool(quant_bwd_drive)
    _ADC_BWD_BOUND_SCALE = float(bwd_bound_scale)
    if _ADC_BITS is not None or _ADC_DAC_BITS is not None:
        print(f"[adc] readout={_ADC_BITS} bits, drive={_ADC_DAC_BITS} bits "
              f"(range {_ADC_IN_RANGE}), backward={_ADC_BWD}"
              f"{', backward drive quantized' if _ADC_BWD_DRIVE else ''}",
              flush=True)


def get_adc_config():
    return {"bits": _ADC_BITS, "dac_bits": _ADC_DAC_BITS,
            "out_bound": dict(_ADC_OUT_BOUND), "in_range": _ADC_IN_RANGE,
            "input_range": _ADC_INPUT_RANGE, "head_range": _ADC_HEAD_RANGE,
            "quant_bwd": bool(_ADC_BWD),
            "quant_bwd_drive": bool(_ADC_BWD_DRIVE),
            "bwd_bound_scale": float(_ADC_BWD_BOUND_SCALE)}


def adc_levels(bits):
    """Code levels of a b-bit signed converter (one code spent on the bound)."""
    return None if bits is None else 2 ** int(bits) - 1


def adc_step(bound, bits):
    return 2.0 * float(bound) / (2 ** int(bits) - 2)


def quant_bound(z, bound, bits):
    """Uniform quantization to 2^bits - 1 levels over [-bound, +bound].
    Straight-through estimator, left unclipped so a saturated conversion
    still transmits gradient."""
    step = adc_step(bound, bits)
    q = jnp.clip(jnp.round(z / step) * step, -float(bound), float(bound))
    return z + jax.lax.stop_gradient(q - z)


def quant_unipolar(x, bound, bits):
    """Uniform quantization to 2^bits levels over [0, bound], zero code at
    exactly zero, for the rectified inputs of every array after the first."""
    step = float(bound) / (2 ** int(bits) - 1)
    q = jnp.clip(jnp.round(x / step) * step, 0.0, float(bound))
    return x + jax.lax.stop_gradient(q - x)


def adc_readout(z, layer_name):
    """The converter on the array output. Inert unless armed."""
    if _ADC_BITS is None:
        return z
    if layer_name not in _ADC_OUT_BOUND:
        raise KeyError(
            f"no readout bound for '{layer_name}'. Every array that is read "
            f"needs one; silently skipping it would leave part of the "
            f"datapath in full precision.")
    return quant_bound(z, float(_ADC_OUT_BOUND[layer_name]), _ADC_BITS)


def adc_dac_readout(x, is_first):
    """The converter driving a conv array's input. Both drives are unipolar:
    hidden arrays receive rectified activations and the first array a [0, 1]
    image, so a signed converter would waste half its codes."""
    if _ADC_DAC_BITS is None:
        return x
    if is_first:
        return quant_unipolar(x, _ADC_INPUT_RANGE, _ADC_DAC_BITS)
    return quant_unipolar(x, _ADC_IN_RANGE, _ADC_DAC_BITS)


def adc_head_drive(x):
    """The converter driving the dense head's array. Unipolar, the pooled
    feature being a mean of rectified activations and never negative."""
    if _ADC_DAC_BITS is None:
        return x
    return quant_unipolar(x, _ADC_HEAD_RANGE, _ADC_DAC_BITS)


def analog_mv(x, w, layer_name, mv_fn):
    """One analog matrix-vector product, with the converters a transposed
    (backward) read of the same array needs. Backprop only: the error is
    divided by its absolute maximum, optionally quantized before driving the
    transposed array (quant_bwd_drive), the readout is digitized against the
    layer's forward bound, and the result is rescaled. The weight gradient is
    an outer product formed in the update circuitry rather than a transposed
    read, so it is not digitized - but it is built from the quantized error,
    the hardware having nothing better. A no-op for the layer-local rules:
    their stop_gradient means no cotangent ever crosses a layer boundary."""
    if _ADC_BITS is None or not _ADC_BWD:
        return mv_fn(x, w)

    bits = int(_ADC_BITS)
    drive_on = bool(_ADC_BWD_DRIVE)
    bound = float(_ADC_OUT_BOUND[layer_name])

    @jax.custom_vjp
    def _f(x, w):
        return mv_fn(x, w)

    def _f_fwd(x, w):
        return mv_fn(x, w), (x, w)

    def _f_bwd(res, g):
        x, w = res
        alpha = jnp.maximum(jnp.max(jnp.abs(g)), 1e-30)
        gn = g / alpha
        if drive_on:
            gn = quant_bound(gn, 1.0, bits)
        _, vjp = jax.vjp(mv_fn, x, w)
        gx, gw = vjp(gn)
        gx = quant_bound(gx, bound * _ADC_BWD_BOUND_SCALE, bits)
        return (gx * alpha, gw * alpha)

    _f.defvjp(_f_fwd, _f_bwd)
    return _f(x, w)


class LocalCNN(nn.Module):
    """CNN feature extractor for local per-layer training; `target_layer` stops the
    gradient at that layer's input and returns its activation. Params named _conv1... (memristor injection)."""
    channels: Mapping[str, int]
    pooling_after_n_layers: Mapping[str, int]
    noise_std: float = 0.0
    norm: str = "rmsnorm"

    @nn.compact
    def __call__(self, x, train=True, target_layer=None, return_input=False):
        norm_fn = get_norm_fn(self.norm)
        cur = x
        for i, (layer_name, ch) in enumerate(self.channels.items()):
            if target_layer == layer_name:
                layer_input = cur
                cur = jax.lax.stop_gradient(cur)
            in_ch = cur.shape[-1]
            kernel = self.param(f'_{layer_name}', nn.initializers.he_normal(),
                                (3, 3, in_ch, ch))
            cur = adc_dac_readout(cur, i == 0)
            if self.noise_std > 0.0:
                _k = self.make_rng('noise')
                _mv = lambda a, b: noisy_conv(a, b, self.noise_std, _k)
            else:
                _mv = lambda a, b: jax.lax.conv_general_dilated(
                    a, b, (1, 1), 'SAME',
                    dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
            cur = analog_mv(cur, kernel, layer_name, _mv)
            cur = adc_readout(cur, layer_name)
            cur = norm_fn(cur)
            cur = nn.relu(cur)
            if target_layer == layer_name:
                if return_input:
                    return (jax.lax.stop_gradient(layer_input), cur)
                return cur
            if self.pooling_after_n_layers[layer_name]:
                cur = nn.max_pool(cur, (2, 2), (2, 2))
        return cur


class CNNClassifier(nn.Module):
    """Plain CNN with GAP + dense head for Backprop training."""
    channels: Mapping[str, int]
    pooling_after_n_layers: Mapping[str, int]
    num_classes: int = 10
    noise_std: float = 0.0
    norm: str = "rmsnorm"

    @nn.compact
    def __call__(self, x, train=True):
        norm_fn = get_norm_fn(self.norm)
        cur = x
        for i, (layer_name, ch) in enumerate(self.channels.items()):
            in_ch = cur.shape[-1]
            kernel = self.param(f'_{layer_name}', nn.initializers.he_normal(),
                                (3, 3, in_ch, ch))
            cur = adc_dac_readout(cur, i == 0)
            if self.noise_std > 0.0:
                _k = self.make_rng('noise')
                _mv = lambda a, b: noisy_conv(a, b, self.noise_std, _k)
            else:
                _mv = lambda a, b: jax.lax.conv_general_dilated(
                    a, b, (1, 1), 'SAME',
                    dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
            cur = analog_mv(cur, kernel, layer_name, _mv)
            cur = adc_readout(cur, layer_name)
            cur = norm_fn(cur)
            cur = nn.relu(cur)
            if self.pooling_after_n_layers[layer_name]:
                cur = nn.max_pool(cur, (2, 2), (2, 2))
        cur = jnp.mean(cur, axis=(1, 2))
        # The dense head is a memristive array like any other: its drive and
        # forward readout carry the same pair of converters.
        cur = adc_head_drive(cur)
        dense_features = cur.shape[-1]
        dense_w = self.param('classifier_kernel', nn.initializers.he_normal(),
                             (dense_features, self.num_classes))
        if self.noise_std > 0.0:
            _k = self.make_rng('noise')
            _mv = lambda a, b: noisy_matmul(a, b, self.noise_std, _k)
        else:
            _mv = lambda a, b: jnp.dot(a, b)
        logits = analog_mv(cur, dense_w, 'classifier', _mv)
        logits = adc_readout(logits, 'classifier')
        return logits


MODEL_CHOICES = ["cnn"]


def create_local_model(model_name, **kwargs):
    """Create a feature-extractor model (target_layer support) for local
    per-layer training (CwSG / CwC)."""
    if model_name != "cnn":
        raise ValueError(f"Unknown model: {model_name}. Choose from {MODEL_CHOICES}")
    return LocalCNN(**kwargs)


def create_classifier_model(model_name, **kwargs):
    """Create a classifier model (GAP + dense head) for Backprop."""
    if model_name != "cnn":
        raise ValueError(f"Unknown model: {model_name}. Choose from {MODEL_CHOICES}")
    return CNNClassifier(**kwargs)


# Write-noise LR divisor: None to noise std sigma*|update|; lr_ref to sigma*|update|/lr_ref. Baked at JIT-trace time.
WRITE_NOISE_LR_DIVISOR = None


def add_write_noise(updates, write_noise_std, key):
    """Multiplicative write noise on updates: std sigma*|update|, or the
    LR-normalized sigma*|update|/lr_ref when WRITE_NOISE_LR_DIVISOR is set."""
    if write_noise_std <= 0:
        return updates
    lr_div = WRITE_NOISE_LR_DIVISOR
    normalize = (lr_div is not None) and (lr_div > 0.0)
    leaves, treedef = jax.tree_util.tree_flatten(updates)
    keys = jax.random.split(key, len(leaves))
    noisy_leaves = []
    for leaf, k in zip(leaves, keys):
        eps = jax.random.normal(k, leaf.shape)
        if normalize:
            noisy_leaves.append(leaf + write_noise_std * (leaf / lr_div) * eps)
        else:
            noisy_leaves.append(leaf * (1.0 + write_noise_std * eps))
    return treedef.unflatten(noisy_leaves)


def set_write_noise_lr_normalize(lr_ref):
    """Enable LR-normalized write noise (sigma divided by lr_ref; None disables).
    Call once at training start."""
    global WRITE_NOISE_LR_DIVISOR
    WRITE_NOISE_LR_DIVISOR = lr_ref


WEIGHT_CLAMP = None


def set_weight_clamp(value):
    """Set the global weight clamp range (None disables). Call BEFORE
    building train steps - the value is baked in at JIT trace time."""
    global WEIGHT_CLAMP
    WEIGHT_CLAMP = value


def clamp_weights(params):
    """Clamp all weight values to [-c, +c] if WEIGHT_CLAMP is set."""
    if WEIGHT_CLAMP is None:
        return params
    c = WEIGHT_CLAMP
    return jax.tree.map(lambda w: jnp.clip(w, -c, c), params)


_STOCH_THRESHOLD = None


def set_stochastic_threshold(value):
    """Set the StoSignSGD Bernoulli pulse-probability gain (None restores
    deterministic sign_sgd). Baked in at train-step build time."""
    global _STOCH_THRESHOLD
    _STOCH_THRESHOLD = float(value) if value is not None else None


def get_stochastic_threshold():
    """Return the current StoSignSGD pulse-probability gain (None = off)."""
    return _STOCH_THRESHOLD


_SEED_OFFSET = 0


def set_seed_offset(offset):
    """Set the offset added to run_idx when sweep functions derive per-run seeds."""
    global _SEED_OFFSET
    _SEED_OFFSET = int(offset)


def get_seed_offset():
    """Return the current per-run seed offset (read at sweep-call time)."""
    return _SEED_OFFSET


def make_optimizer(name, lr, total_steps=0):
    """Build an optax optimizer: "sgd" or "sign_sgd" (StoSignSGD when a
    stochastic threshold is set)."""
    if name == "sign_sgd":
        if _STOCH_THRESHOLD is not None:
            thr = float(_STOCH_THRESHOLD)

            def _init(params):
                return jax.random.PRNGKey(0)

            def _update(updates, state, params=None):
                rng = state
                rng, sub = jax.random.split(rng)
                leaves, treedef = jax.tree_util.tree_flatten(updates)
                keys = jax.random.split(sub, len(leaves))
                new_leaves = [
                    -lr * jnp.sign(g) * (jax.random.uniform(k, g.shape) <
                                          jnp.clip(jnp.abs(g) * thr, 0.0, 1.0))
                    for g, k in zip(leaves, keys)
                ]
                return jax.tree_util.tree_unflatten(treedef, new_leaves), rng

            return optax.GradientTransformation(_init, _update)
        return optax.sign_sgd(lr)
    elif name == "sgd":
        return optax.sgd(lr)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def mask_local_grads(grads, target_layer):
    """Zero grads for all params except the target layer; the prefix match
    keeps '_conv3'/'_conv3_dw' but not '_conv30' (matters for >9 layers)."""
    masked = {}
    prefix = f"_{target_layer.lower()}"
    for name, g in grads.items():
        n = name.lower()
        keep = (n == prefix) or n.startswith(prefix + "_")
        masked[name] = g if keep else jax.tree_util.tree_map(jnp.zeros_like, g)
    return masked


def _model_has_noise(model):
    """True if the model has read noise (noise_std > 0). Passing a 'noise'
    rng when noise_std=0 still perturbs the graph, so skip it then."""
    m = model.module if hasattr(model, 'module') else model
    return getattr(m, 'noise_std', 0.0) > 0


def _noise_rngs(has_noise, key):
    """Return rngs dict for model.apply - empty if no noise."""
    return {'noise': key} if has_noise else {}


@functools.partial(jax.jit, static_argnums=(1,))
def channel_group_goodness(act, num_classes):
    """Mean squared activation per channel group - the CwC class scores.
    Accepts 4D (B, H, W, C) or 2D (B, C); returns (B, num_classes)."""
    if act.ndim == 2:
        B, C = act.shape
        cpc = C // num_classes
        used_C = num_classes * cpc
        act = act[:, :used_C]
        return jnp.mean((act ** 2).reshape(B, num_classes, cpc), axis=-1)
    B, H, W, C = act.shape
    cpc = C // num_classes
    used_C = num_classes * cpc
    act = act[..., :used_C]
    return jnp.mean((act ** 2).reshape(B, H, W, num_classes, cpc), axis=(1, 2, 4))


@functools.partial(jax.jit, static_argnums=(1,))
def spatial_gradient_energy(act, num_classes):
    """Mean squared activation differences per channel group - the CwSG class
    scores. 4D: central differences over (H, W); 2D: cyclic neighbor difference."""
    if act.ndim == 2:
        B, C = act.shape
        cpc = C // num_classes
        used_C = num_classes * cpc
        act_grouped = act[:, :used_C].reshape(B, num_classes, cpc)
        neighbor = jnp.roll(act_grouped, shift=-1, axis=-1)
        return jnp.mean((act_grouped - neighbor) ** 2, axis=-1)

    B, H, W, C = act.shape
    cpc = C // num_classes
    used_C = num_classes * cpc
    act_used = act[..., :used_C]

    grad_h = jnp.zeros_like(act_used)
    grad_h = grad_h.at[:, 1:-1, :, :].set(
        act_used[:, 2:, :, :] - act_used[:, :-2, :, :])

    grad_w = jnp.zeros_like(act_used)
    grad_w = grad_w.at[:, :, 1:-1, :].set(
        act_used[:, :, 2:, :] - act_used[:, :, :-2, :])

    grad_energy = grad_h ** 2 + grad_w ** 2

    return jnp.mean(grad_energy.reshape(B, H, W, num_classes, cpc), axis=(1, 2, 4))


@functools.partial(jax.jit, static_argnums=(0, 5))
def bp_predict(model, params, batch_stats, imgs, rng_key, has_noise=False):
    """Predict class labels with the Backprop classifier (argmax logits)."""
    logits = model.apply({"params": params, "batch_stats": batch_stats},
                         imgs, train=False, rngs=_noise_rngs(has_noise, rng_key))
    return jnp.argmax(logits, axis=-1)


def compute_bp_accuracy(model, params, batch_stats, ds, rng_key):
    """Test-set accuracy (%) of the Backprop classifier over dataset `ds`."""
    _has_noise = _model_has_noise(model)
    correct, total = jnp.int32(0), 0
    for imgs, labels in prefetch_to_device(ds):
        rng_key, eval_key = jax.random.split(rng_key)
        preds = bp_predict(model, params, batch_stats, imgs, eval_key, _has_noise)
        correct += jnp.sum(preds == labels)
        total += len(labels)
    return float(correct / total * 100), rng_key


def compute_prototype_accuracy(evaluate_fn, params, batch_stats, ds, rng_key):
    """Accuracy (%) for the goodness-based evaluate functions (CwSG / CwC)."""
    correct, total = jnp.int32(0), 0
    for imgs, labels in prefetch_to_device(ds):
        rng_key, eval_key = jax.random.split(rng_key)
        preds = evaluate_fn(params, batch_stats, imgs, eval_key)
        correct += jnp.sum(preds == labels)
        total += len(labels)
    return float(correct / total * 100), rng_key


def average_metrics(all_run_metrics):
    """Average per-epoch metric dicts across runs (mean/std of loss and
    train/test accuracy, plus the per-run values)."""
    num_epochs = len(all_run_metrics[0])
    averaged = []
    for i in range(num_epochs):
        losses = [run[i]["loss"] for run in all_run_metrics]
        train_accs = [run[i]["train_acc"] for run in all_run_metrics]
        test_accs = [run[i]["test_acc"] for run in all_run_metrics]
        averaged.append({
            "global_epoch": all_run_metrics[0][i]["global_epoch"],
            "layer": all_run_metrics[0][i].get("layer"),
            "local_epoch": all_run_metrics[0][i]["local_epoch"],
            "avg_loss": float(np.mean(losses)),
            "std_loss": float(np.std(losses)),
            "avg_train_acc": float(np.mean(train_accs)),
            "std_train_acc": float(np.std(train_accs)),
            "avg_test_acc": float(np.mean(test_accs)),
            "std_test_acc": float(np.std(test_accs)),
            "per_run_losses": [float(v) for v in losses],
            "per_run_train_accs": [float(v) for v in train_accs],
            "per_run_test_accs": [float(v) for v in test_accs],
        })
    return averaged


class NumpyEncoder(json.JSONEncoder):
    """json.JSONEncoder that also handles numpy/jax scalars and arrays."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating, jnp.floating)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)
