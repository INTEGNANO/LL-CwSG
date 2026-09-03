"""Device-calibrated memristor training on measured PCM/RRAM HDF5 traces:
DevicePool / MemristorWeight device model, StoSignSGD in-situ pulse updates,
memristor Backprop / CwSG train stacks and training loops. Call
set_device_type BEFORE building train steps - polarity is baked at JIT-trace time.
"""

import os
import math
from pathlib import Path
from typing import NamedTuple

# helper must be imported before jax: it sets the TF/XLA env vars.
from helper import (
    create_local_model, create_classifier_model,
    spatial_gradient_energy, mask_local_grads,
    prefetch_to_device,
    set_adc_config, get_adc_config, adc_levels, adc_step,
)

import h5py
import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
import optax
from tqdm import tqdm


DEVICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "devices")

DEVICE_CONFIGS = {
    "rram": {
        "path": os.path.join(DEVICES_DIR, "rram.hdf5"),
        "description": "RRAM weak-reset (high to low conductance)",
    },
    "pcm": {
        "path": os.path.join(DEVICES_DIR, "pcm.hdf5"),
        "description": "PCM weak-SET (low to high conductance)",
    },
}
DEVICE_CHOICES = list(DEVICE_CONFIGS.keys())


def device_files_present(device_type):
    """True if the calibrated HDF5 file for `device_type` is in devices/."""
    return os.path.isfile(DEVICE_CONFIGS[device_type]["path"])


# PCM and RRAM have inverted pulse polarity; baked in at JIT-trace time.
_DEVICE_TYPE = "rram"


def set_device_type(device_type):
    """Set the global device polarity. MUST be called before building any
    memristor train step - the value is baked in at JIT trace time."""
    global _DEVICE_TYPE
    assert device_type in DEVICE_CHOICES, f"Unknown device: {device_type}. Choose from {DEVICE_CHOICES}"
    _DEVICE_TYPE = device_type


def get_memristor_path(device_type=None):
    """Return the HDF5 path for the given (or current) device type."""
    dt = device_type or _DEVICE_TYPE
    return DEVICE_CONFIGS[dt]["path"]


class DevicePool:
    """Pool of measured device resistance traces from an HDF5 file; hands out
    differential (pos, neg) device pairs, reshuffling/recycling when exhausted."""

    def __init__(self, filename, key):
        self._key = key
        self._filename = Path(filename)

        with h5py.File(self._filename, "r") as f:
            h5_devices = list(f.values())

            self.devices = jnp.asarray(h5_devices)[:, :]

            self._devices_metadata = np.array(
                [dict(d.attrs.items()) for d in h5_devices]
            )

        if self.devices.ndim != 2:
            raise ValueError(
                f"Invalid devices shape, expected 2 dimensions, got {self.devices.ndim}"
            )

        self.num_devices, self.num_measurement = self.devices.shape

        self.request_metadatas = dict()
        self._request_anonym_index = 0

        self._devices_order = np.arange(self.num_devices, dtype=np.uint32)
        self._shuffle_devices_order()

        self.pool_index = 0

    def _shuffle_devices_order(self):
        """Reshuffle the device hand-out order (consumes one PRNG split)."""
        self._key, permutation_key = jrand.split(self._key)
        permut = jax.random.permutation(
            permutation_key, self._devices_order, independent=True
        )
        self._devices_order = self._devices_order[permut]

        return self._devices_order

    def request_couple(self, shape=(1,), uid=None):
        """Allocate a (positive, negative) device pair for every element of
        `shape`; returns device indices of shape (2, *shape)."""
        if not isinstance(shape, (tuple, list)):
            shape = (shape,)

        total_req_size = 2 * math.prod(shape)
        devices_left_in_pool = len(self._devices_order) - self.pool_index

        if total_req_size > devices_left_in_pool:
            print(
                "[WARN] Out-of-device: reshuffling the pool for request of shape", shape
            )

            sel_dev_chunks = [self._devices_order[self.pool_index :]]
            total_req_size -= devices_left_in_pool
            self._shuffle_devices_order()

            while total_req_size > len(self._devices_order):
                sel_dev_chunks.append(self._devices_order[:])
                total_req_size -= len(self._devices_order)

                self._shuffle_devices_order()

            self.pool_index = total_req_size
            sel_dev_chunks.append(self._devices_order[: self.pool_index])

            selected_devices = np.concatenate(sel_dev_chunks)
        else:
            selected_devices = self._devices_order[
                self.pool_index : self.pool_index + total_req_size
            ]
            self.pool_index += total_req_size

        if uid is None:
            id = self._request_anonym_index
            self._request_anonym_index += 1
        else:
            id = uid

        self.request_metadatas[id] = (
            selected_devices,
            self._devices_metadata[selected_devices[:len(self._devices_metadata)]],
        )
        return jax.device_put(selected_devices.reshape((2, *shape)))


class _MemristorWeightStatics(NamedTuple):
    """Static (non-trained) part of a MemristorWeight: device ids, pool,
    weight formula (always "conductance_diff") and scale factor."""
    pos_dev_id: jnp.ndarray
    neg_dev_id: jnp.ndarray

    name: str
    pool: 'DevicePool'

    weight_formula: str
    scale_factor: float


class MemristorWeight(NamedTuple):
    """Differential memristor weight: two pulse-counter arrays plus statics.
    Logical weight W = scale * (1/R_neg - 1/R_pos) at the current pulse indices."""
    pos_pulse_id: jnp.ndarray
    neg_pulse_id: jnp.ndarray
    statics: _MemristorWeightStatics

    @staticmethod
    def init(
        device_pool,
        name,
        shape,
        scale_factor,
    ):
        """Allocate a fresh device pair per weight element (pulse counters at 0)."""
        pos, neg = device_pool.request_couple(shape, uid=name)
        return MemristorWeight.from_devices(
            (pos, neg), device_pool, name, scale_factor
        )

    @staticmethod
    def from_devices(
        devices,
        device_pool,
        name,
        scale_factor,
    ):
        """Build a MemristorWeight from pre-allocated (pos, neg) device ids."""
        pos, neg = devices
        shape = pos.shape
        return MemristorWeight(
            jnp.zeros(shape, dtype=np.uint16),
            jnp.zeros(shape, dtype=np.uint16),
            _MemristorWeightStatics(
                pos, neg, name, device_pool, "conductance_diff", scale_factor,
            ),
        )

    @property
    def weight(self):
        """Materialise W = scale * (1/R_neg - 1/R_pos) via lax.scan over the
        output-channel axis to keep memory flat (prevents OOM)."""
        p_ids_T = jnp.moveaxis(self.pos_pulse_id, -1, 0)
        n_ids_T = jnp.moveaxis(self.neg_pulse_id, -1, 0)
        p_dev_T = jnp.moveaxis(self.statics.pos_dev_id, -1, 0)
        n_dev_T = jnp.moveaxis(self.statics.neg_dev_id, -1, 0)

        def get_slice(carry, inputs):
            (ids_p, ids_n, dev_p, dev_n) = inputs

            wp = self.statics.pool.devices[dev_p, ids_p]
            wn = self.statics.pool.devices[dev_n, ids_n]

            w_slice = 1.0 / wn - 1.0 / wp

            return carry, w_slice

        _, w_stacked = jax.lax.scan(
            get_slice,
            None,
            (p_ids_T, n_ids_T, p_dev_T, n_dev_T)
        )

        w_final = jnp.moveaxis(w_stacked, 0, -1)

        return self.statics.scale_factor * w_final

    def incr_all(self, indices):
        """Add one positive-device pulse wherever `indices` is 1/True."""
        indices = jnp.asarray(indices)
        return MemristorWeight(
            self.pos_pulse_id + indices,
            self.neg_pulse_id,
            self.statics,
        )

    def decr_all(self, indices):
        """Add one negative-device pulse wherever `indices` is 1/True."""
        indices = jnp.asarray(indices)
        return MemristorWeight(
            self.pos_pulse_id,
            self.neg_pulse_id + indices,
            self.statics,
        )


def _device_pool_flatten(pool):
    """Flatten DevicePool: the devices tensor is the only dynamic leaf."""
    children = (pool.devices,)
    aux_data = (
        str(pool._filename),
        pool.num_devices,
        pool.num_measurement
    )
    return children, aux_data

def _device_pool_unflatten(aux_data, children):
    """Rebuild a DevicePool shell from children + aux; allocation state is reset."""
    pool = DevicePool.__new__(DevicePool)
    pool.devices = children[0]
    (
        pool._filename,
        pool.num_devices,
        pool.num_measurement
    ) = aux_data

    pool._key = None
    pool._devices_metadata = None
    pool.request_metadatas = {}
    pool._request_anonym_index = 0
    pool._devices_order = None
    pool.pool_index = 0

    return pool


def _mw_statics_flatten(statics):
    """Flatten _MemristorWeightStatics: pool + device ids as children,
    (name, formula, scale) as aux_data."""
    children = (statics.pool, statics.pos_dev_id, statics.neg_dev_id)
    aux_data = (
        statics.name,
        statics.weight_formula,
        statics.scale_factor
    )
    return children, aux_data

def _mw_statics_unflatten(aux_data, children):
    """Rebuild _MemristorWeightStatics from children + aux."""
    pool, pos_dev_id, neg_dev_id = children
    name, weight_formula, scale_factor = aux_data

    return _MemristorWeightStatics(
        pos_dev_id,
        neg_dev_id,
        name,
        pool,
        weight_formula,
        scale_factor
    )


def _mw_flatten(mw):
    """Flatten MemristorWeight: pulse counters + statics are all children."""
    children = (mw.pos_pulse_id, mw.neg_pulse_id, mw.statics)
    return children, None

def _mw_unflatten(aux, children):
    """Rebuild a MemristorWeight from (pos_pulse_id, neg_pulse_id, statics)."""
    return MemristorWeight(children[0], children[1], children[2])


jax.tree_util.register_pytree_node(DevicePool, _device_pool_flatten, _device_pool_unflatten)

jax.tree_util.register_pytree_node(_MemristorWeightStatics, _mw_statics_flatten, _mw_statics_unflatten)

jax.tree_util.register_pytree_node(MemristorWeight, _mw_flatten, _mw_unflatten)


def init_conv_memristors(model_params, memristor_path, scale, rng_key):
    """Create a MemristorWeight for each '_conv*' kernel (LocalCNN naming);
    returns ({param_name: MemristorWeight}, rng_key)."""
    mem_devices = {}
    param_names = sorted([k for k in model_params.keys() if k.startswith('_conv')])

    for name in param_names:
        param = model_params[name]
        rng_key, pool_key = jax.random.split(rng_key)
        pool = DevicePool(memristor_path, pool_key)
        mem_devices[name] = MemristorWeight.init(
            pool, name, param.shape,
            scale_factor=scale,
        )

    return mem_devices, rng_key


def init_dense_memristors(model_params, memristor_path, scale, rng_key):
    """Create a MemristorWeight for the 'classifier_kernel' dense layer;
    returns ({param_name: MemristorWeight}, rng_key)."""
    mem_devices = {}

    for name in ['classifier_kernel']:
        if name in model_params:
            param = model_params[name]
            rng_key, pool_key = jax.random.split(rng_key)
            pool = DevicePool(memristor_path, pool_key)
            mem_devices[name] = MemristorWeight.init(
                pool, name, param.shape,
                scale_factor=scale,
            )

    return mem_devices, rng_key


def eval_memristor_dict(mem_devices):
    """Evaluate all MemristorWeights in a dict to {name: jax.Array}."""
    return {name: dev.weight for name, dev in mem_devices.items()}


def inject_memristor_weights(model_params, mem_values):
    """Return a new (unfrozen) params dict with memristor-evaluated weight
    arrays injected for matching names."""
    new_params = dict(model_params)
    for name, val in mem_values.items():
        if name in new_params:
            new_params[name] = val
    return new_params


def _stochastic_update_memristors(mem_devices, grads, lr_scale, rng_key):
    """Stochastic pulse coincidence: pulse each weight with probability
    p = clip(|grad|*lr_scale, 0, 1) toward -sign(grad); memoryless.
    Returns (new_devices, pulse_ratio, expected per-layer pulse counts)."""
    names = sorted(mem_devices.keys())
    keys = jax.random.split(rng_key, len(names))
    key_by_name = {n: k for n, k in zip(names, keys)}

    total_params = sum(x.size for x in jax.tree_util.tree_leaves(grads))
    per_layer_pulses = {}
    for n in names:
        g = grads[n]
        p = jnp.clip(jnp.abs(g) * lr_scale, 0.0, 1.0)
        per_layer_pulses[n] = jnp.sum(p)
    pulse_ratio = sum(per_layer_pulses.values()) / max(total_params, 1)

    new_devices = dict(mem_devices)
    for n in names:
        weight = mem_devices[n]
        grad = grads[n]
        key = key_by_name[n]
        p = jnp.clip(jnp.abs(grad) * lr_scale, 0.0, 1.0)
        r = jax.random.uniform(key, shape=grad.shape)
        pulse = r < p
        if _DEVICE_TYPE == "pcm":
            pulse_incr = pulse & (grad > 0)
            pulse_decr = pulse & (grad < 0)
        else:
            pulse_incr = pulse & (grad < 0)
            pulse_decr = pulse & (grad > 0)
        new_devices[n] = weight.incr_all(pulse_incr).decr_all(pulse_decr)

    return new_devices, pulse_ratio, per_layer_pulses


def update_memristors(mem_devices, grads, threshold, rng_key=None,
                       return_per_layer=False):
    """Apply stochastic write pulses (p = clip(|grad|*threshold, 0, 1)).
    Returns (new_devices, pulse_ratio[, per_layer_pulses])."""
    assert rng_key is not None, (
        "Stochastic update mechanism requires rng_key. "
        "Pass a PRNGKey to update_memristors."
    )
    new_dev, pr, per_layer = _stochastic_update_memristors(
        mem_devices, grads, threshold, rng_key)
    if return_per_layer:
        return new_dev, pr, per_layer
    return new_dev, pr


def init_pulse_counts(mem_devices):
    """Allocate per-layer cumulative-pulse scalar trackers (jnp.float32)."""
    return {name: jnp.float32(0.0) for name in mem_devices}


def compute_pulse_stats(mem_devices):
    """Per-layer min/max/mean/std of the pos/neg pulse counters, keyed by
    layer name (JSON-friendly ints/floats)."""
    stats = {}
    for name, dev in mem_devices.items():
        pos = np.asarray(dev.pos_pulse_id)
        neg = np.asarray(dev.neg_pulse_id)
        stats[name] = {
            "pos_min":  int(pos.min()),
            "pos_max":  int(pos.max()),
            "pos_mean": float(pos.mean()),
            "pos_std":  float(pos.std()),
            "neg_min":  int(neg.min()),
            "neg_max":  int(neg.max()),
            "neg_mean": float(neg.mean()),
            "neg_std":  float(neg.std()),
            "n_cells":  int(pos.size),
        }
    return stats


def compute_weight_stats(mem_devices):
    """Per-layer stats of the materialised weights: range/spread, effective
    bit-width, and pulse-counter saturation fractions."""
    stats = {}
    for name, dev in mem_devices.items():
        w = np.asarray(dev.weight)
        pos = np.asarray(dev.pos_pulse_id)
        neg = np.asarray(dev.neg_pulse_id)

        rounded = np.round(w, decimals=4)
        n_unique = int(np.unique(rounded).size)

        try:
            max_pulse_idx = int(dev.statics.pool.devices.shape[-1] - 1)
        except Exception:
            max_pulse_idx = None

        s = {
            "w_min":      float(w.min()),
            "w_max":      float(w.max()),
            "w_mean":     float(w.mean()),
            "w_std":      float(w.std()),
            "w_n_unique": n_unique,
            "w_eff_bits": float(np.log2(n_unique)) if n_unique > 0 else 0.0,
            "n_cells":    int(w.size),
        }
        if max_pulse_idx is not None:
            s["sat_pos_frac"] = float((pos >= max_pulse_idx).mean())
            s["sat_neg_frac"] = float((neg >= max_pulse_idx).mean())
            s["max_pulse_idx"] = max_pulse_idx
        stats[name] = s
    return stats


def layer_cell_counts(mem_devices):
    """Number of memristor cells per layer (Python ints, fixed at init)."""
    mem_values = eval_memristor_dict(mem_devices)
    return {name: int(np.prod(v.shape)) for name, v in mem_values.items()}


# Headroom constant of the ADC readout full scale, B_l = C * w_rail * sqrt(F_l).
# C is headroom, and headroom costs resolution: halving it is one bit of
# effective resolution, paid for in saturation.
ADC_C_DESIGN = 0.5


def device_weight_rail(mem_devices):
    """scale * (Gmax - Gmin), the representable range of a differential pair.
    A property of the conductance window, not of any programmed weight."""
    m = next(iter(mem_devices.values()))
    R = m.statics.pool.devices
    return float(m.statics.scale_factor * (1.0 / jnp.min(R) - 1.0 / jnp.max(R)))


def adc_readout_bounds(mem_devices, channels, layers):
    """Per-layer ADC readout full scale B_l = C * w_rail * sqrt(F_l). A column
    current accumulates F = 9*C_in products, so its scale grows as sqrt(F) and
    the range of each layer follows that and nothing else. The dense head
    accumulates one product per input channel (no kernel window), so its
    fan-in is the last convolution's channel count."""
    rail = device_weight_rail(mem_devices)
    chan, bounds, c_in = list(channels.values()), {}, 3
    for i, name in enumerate(layers):
        bounds[name] = ADC_C_DESIGN * rail * ((9 * c_in) ** 0.5)
        c_in = chan[i]
    bounds["classifier"] = ADC_C_DESIGN * rail * (float(chan[-1]) ** 0.5)
    return bounds, rail


def arm_adc(adc, mem_devices, channels, layers):
    """Arm the periphery converters from an `adc` config dict (keys: bits,
    dac_bits, quant_bwd, bwd_drive) and the device model's conductance window;
    returns a JSON-friendly info dict. adc=None resets to the float bypass."""
    if adc is None:
        set_adc_config()
        return None
    bounds, rail = adc_readout_bounds(mem_devices, channels, layers)
    quant_bwd = bool(adc.get("quant_bwd", False))
    set_adc_config(bits=adc["bits"], dac_bits=adc.get("dac_bits"),
                   out_bound=bounds,
                   quant_bwd=quant_bwd,
                   quant_bwd_drive=quant_bwd and bool(adc.get("bwd_drive", False)))
    info = dict(get_adc_config())
    info["levels"] = adc_levels(adc["bits"])
    info["weight_rail"] = rail
    info["c_design"] = ADC_C_DESIGN
    info["step"] = {k: adc_step(v, adc["bits"]) for k, v in bounds.items()}
    return info


def check_pulse_endurance(pulse_counts, n_cells_per_layer,
                           max_pulses_per_device, already_warned, run_label=""):
    """Pigeonhole endurance check: warn once when a layer's average
    pulses/cell reaches the limit; returns the updated warned flag."""
    if max_pulses_per_device is None or already_warned:
        return already_warned
    worst_layer = None
    worst_avg = -1.0
    for name, count in pulse_counts.items():
        avg = float(count) / max(n_cells_per_layer[name], 1)
        if avg > worst_avg:
            worst_avg = avg
            worst_layer = name
    if worst_avg >= max_pulses_per_device:
        print(f"    [{run_label}] memristor endurance warning: "
              f"layer '{worst_layer}' averaging {worst_avg:.0f} pulses/cell "
              f"(threshold {max_pulses_per_device}) - at least one cell has "
              f"accumulated >= {max_pulses_per_device} pulses.")
        return True
    return already_warned


def make_mem_bp_train_step(model, threshold, temperature=1.0):
    """JIT Backprop train step with all weights (conv + dense) on memristors;
    updates are purely pulse-based (no FP32 optimizer)."""
    @jax.jit
    def step(mem_devices, batch_stats, pulse_counts, imgs, labels, rng_key):
        noise_key, pulse_key = jax.random.split(rng_key)
        mem_values = eval_memristor_dict(mem_devices)

        def loss_fn(model_params):
            out, new_state = model.apply(
                {"params": model_params, "batch_stats": batch_stats},
                imgs, train=True, mutable=["batch_stats"],
                rngs={'noise': noise_key}
            )
            loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
                out / temperature, labels))
            return loss, new_state

        (loss, new_state), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(mem_values)

        new_mem_devices, pulse_ratio, per_layer_pulses = update_memristors(
            mem_devices, grads, threshold, pulse_key, return_per_layer=True)
        new_pulse_counts = {k: pulse_counts[k] + per_layer_pulses[k]
                            for k in pulse_counts}

        return (new_mem_devices, new_state.get("batch_stats", {}),
                new_pulse_counts, loss, pulse_ratio)

    return step


def make_mem_bp_evaluate_fn(model):
    """JIT Backprop eval function with memristor weights."""
    @jax.jit
    def evaluate_batch(mem_values, batch_stats, imgs, rng_key):
        logits = model.apply(
            {"params": mem_values, "batch_stats": batch_stats},
            imgs, train=False, rngs={'noise': rng_key}
        )
        return jnp.argmax(logits, axis=-1)

    return evaluate_batch


def compute_mem_bp_accuracy(evaluate_fn, mem_values, batch_stats, ds, rng_key):
    """Accuracy (%) for Backprop with memristor weights."""
    correct, total = jnp.int32(0), 0
    for imgs, labels in prefetch_to_device(ds):
        rng_key, eval_key = jax.random.split(rng_key)
        preds = evaluate_fn(mem_values, batch_stats, imgs, eval_key)
        correct += jnp.sum(preds == labels)
        total += len(labels)
    return float(correct / total * 100), rng_key


def make_mem_cwsg_train_step(model, layer_name, threshold,
                             temperature, num_classes):
    """JIT CwSG train step: CE on spatial_gradient_energy, pulsing only the
    current layer's conv memristors."""
    @jax.jit
    def step(mem_devices, params, batch_stats, pulse_counts,
             imgs, labels, rng_key):
        noise_key, pulse_key = jax.random.split(rng_key)
        mem_values = eval_memristor_dict(mem_devices)
        model_params_with_mem = inject_memristor_weights(params, mem_values)

        def loss_fn(model_p):
            act, new_state = model.apply(
                {"params": model_p, "batch_stats": batch_stats},
                imgs, train=True, target_layer=layer_name,
                mutable=["batch_stats"], rngs={'noise': noise_key}
            )
            goodness = spatial_gradient_energy(act, num_classes)
            logits = goodness / temperature
            loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
                logits, labels))
            return loss, new_state

        (loss, new_state), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(model_params_with_mem)

        grads = mask_local_grads(grads, layer_name)

        conv_grads_for_pulsing = {}
        for name in mem_devices:
            if name in grads:
                conv_grads_for_pulsing[name] = grads[name]

        new_mem_devices, pulse_ratio, per_layer_pulses = update_memristors(
            mem_devices, conv_grads_for_pulsing, threshold, pulse_key,
            return_per_layer=True)
        new_pulse_counts = {k: pulse_counts[k] + per_layer_pulses[k]
                            for k in pulse_counts}

        return (new_mem_devices, params, new_state["batch_stats"],
                new_pulse_counts, loss, pulse_ratio)

    return step


def make_mem_cwsg_interleaved_step(model, layers, threshold,
                                   temperature, num_classes):
    """JIT interleaved CwSG train step: sum of per-layer CE losses, one
    value_and_grad, one update_memristors call on the full conv-grad dict.
    All layers pulse in parallel from the same pre-update devices; the first
    rng half drives every layer's forward, the second the pulse draws.
    stop_gradient at each target layer's input makes grad masking redundant."""
    layers = list(layers)

    @jax.jit
    def step(mem_devices, params, batch_stats, pulse_counts,
             imgs, labels, rng_key):
        noise_key, pulse_key = jax.random.split(rng_key)
        mem_values = eval_memristor_dict(mem_devices)
        model_params_with_mem = inject_memristor_weights(params, mem_values)

        def loss_fn(model_p):
            total = jnp.float32(0.0)
            new_state = {"batch_stats": batch_stats}
            for layer_name in layers:
                act, ns = model.apply(
                    {"params": model_p, "batch_stats": batch_stats},
                    imgs, train=True, target_layer=layer_name,
                    mutable=["batch_stats"], rngs={'noise': noise_key}
                )
                goodness = spatial_gradient_energy(act, num_classes)
                logits = goodness / temperature
                total = total + jnp.mean(
                    optax.softmax_cross_entropy_with_integer_labels(
                        logits, labels))
                new_state = ns
            return total, new_state

        (loss, new_state), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(model_params_with_mem)

        conv_grads_for_pulsing = {}
        for name in mem_devices:
            if name in grads:
                conv_grads_for_pulsing[name] = grads[name]

        new_mem_devices, pulse_ratio, per_layer_pulses = update_memristors(
            mem_devices, conv_grads_for_pulsing, threshold, pulse_key,
            return_per_layer=True)
        new_pulse_counts = {k: pulse_counts[k] + per_layer_pulses[k]
                            for k in pulse_counts}

        return (new_mem_devices, params, new_state["batch_stats"],
                new_pulse_counts, loss, pulse_ratio)

    return step


def make_mem_cwsg_evaluate_fn(model, eval_layer, num_classes):
    """JIT CwSG eval function with memristor weights."""
    @jax.jit
    def evaluate_batch(mem_values, params, batch_stats, imgs, rng_key):
        model_params_with_mem = inject_memristor_weights(params, mem_values)
        act = model.apply(
            {"params": model_params_with_mem, "batch_stats": batch_stats},
            imgs, train=False, target_layer=eval_layer,
            rngs={'noise': rng_key}
        )
        goodness = spatial_gradient_energy(act, num_classes)
        return jnp.argmax(goodness, axis=-1)

    return evaluate_batch


def compute_mem_cwsg_accuracy(evaluate_fn, mem_values, params,
                              batch_stats, ds, rng_key):
    """Accuracy (%) for CwSG with memristor weights."""
    correct, total = jnp.int32(0), 0
    for imgs, labels in prefetch_to_device(ds):
        rng_key, eval_key = jax.random.split(rng_key)
        preds = evaluate_fn(mem_values, params, batch_stats, imgs, eval_key)
        correct += jnp.sum(preds == labels)
        total += len(labels)
    return float(correct / total * 100), rng_key


def _per_epoch_record(epoch, global_epoch, layer_idx, layer_name,
                      epoch_in_layer, train_loss, test_acc):
    """One history entry of a replay run (JSON-friendly scalars)."""
    return {
        "epoch": int(epoch),
        "global_epoch": int(global_epoch),
        "layer_idx": layer_idx,
        "layer_name": layer_name,
        "epoch_in_layer": epoch_in_layer,
        "train_loss": float(train_loss),
        "test_acc": (float(test_acc) if test_acc is not None else None),
    }


def run_mem_bp(*, scale, threshold, temperature, num_classes,
               ds_train, ds_test, input_shape, norm_name,
               epochs, memristor_path, channels, pooling,
               max_pulses_per_device, run_seed,
               eval_every, adc=None):
    """Single Backprop-on-device run (all weights on devices, pulse updates);
    returns {history, final_test_acc, pulse_stats, weight_stats}. `adc` arms
    the periphery converters (see arm_adc); None = float periphery."""
    set_adc_config()
    model = create_classifier_model(
        "cnn", channels=channels, pooling_after_n_layers=pooling,
        num_classes=num_classes, noise_std=0.0, norm=norm_name,
    )
    init_variables = model.init(
        {"params": jax.random.PRNGKey(run_seed),
         "noise": jax.random.PRNGKey(1)},
        jnp.ones((1,) + input_shape),
    )
    batch_stats = init_variables.get("batch_stats", {})

    rng_key = jax.random.PRNGKey(run_seed + 1000)
    mem_devices_conv, rng_key = init_conv_memristors(
        init_variables["params"], memristor_path, scale, rng_key)
    mem_devices_dense, rng_key = init_dense_memristors(
        init_variables["params"], memristor_path, scale, rng_key)
    mem_devices = {**mem_devices_conv, **mem_devices_dense}

    pulse_counts = init_pulse_counts(mem_devices)
    n_cells_layer = layer_cell_counts(mem_devices)
    pulse_warned = False

    adc_info = arm_adc(adc, mem_devices, channels, list(channels.keys()))
    train_step = make_mem_bp_train_step(model, threshold, temperature)
    evaluate_fn = make_mem_bp_evaluate_fn(model)

    rng_key = jax.random.PRNGKey(run_seed)
    history = []
    pbar = tqdm(range(epochs), desc=f"BP seed{run_seed}", ncols=100)
    test_acc = None
    for epoch in pbar:
        epoch_loss = 0.0
        nb = 0
        for imgs, labels in prefetch_to_device(ds_train):
            rng_key, step_key = jax.random.split(rng_key)
            (mem_devices, batch_stats, pulse_counts,
             loss, _) = train_step(
                mem_devices, batch_stats, pulse_counts,
                imgs, labels, step_key)
            epoch_loss += float(loss); nb += 1
        avg_loss = epoch_loss / max(nb, 1)

        if (eval_every and (epoch + 1) % eval_every == 0) or epoch == epochs - 1:
            mem_values = eval_memristor_dict(mem_devices)
            test_acc, rng_key = compute_mem_bp_accuracy(
                evaluate_fn, mem_values, batch_stats, ds_test, rng_key)
            pbar.set_postfix(loss=f"{avg_loss:.3f}",
                             acc=f"{test_acc:.2f}%")
        else:
            pbar.set_postfix(loss=f"{avg_loss:.3f}")

        history.append(_per_epoch_record(
            epoch=epoch, global_epoch=epoch,
            layer_idx=None, layer_name=None, epoch_in_layer=None,
            train_loss=avg_loss, test_acc=test_acc))

        pulse_warned = check_pulse_endurance(
            pulse_counts, n_cells_layer, max_pulses_per_device,
            pulse_warned, run_label=f"BP seed{run_seed}")
    pbar.close()

    return {
        "history": history,
        "final_test_acc": float(test_acc) if test_acc is not None else None,
        "pulse_stats": compute_pulse_stats(mem_devices),
        "weight_stats": compute_weight_stats(mem_devices),
        "adc": adc_info,
    }


def run_mem_cwsg(*, scale, threshold, temperature, num_classes,
                 ds_train, ds_test, input_shape, norm_name,
                 epochs_per_layer, memristor_path,
                 channels, pooling, layers,
                 max_pulses_per_device, run_seed,
                 eval_every, adc=None):
    """Single greedy CwSG-on-device run (conv weights on devices, pulse
    updates); returns {history, final_test_acc, pulse_stats, weight_stats}.
    `adc` arms the periphery converters (see arm_adc); None = float."""
    set_adc_config()
    model = create_local_model("cnn", channels=channels,
                             pooling_after_n_layers=pooling,
                             noise_std=0.0, norm=norm_name)
    init_variables = model.init(
        {"params": jax.random.PRNGKey(run_seed),
         "noise": jax.random.PRNGKey(1)},
        jnp.ones((1,) + input_shape))
    params = init_variables["params"]
    batch_stats = init_variables.get("batch_stats", {})

    rng_key = jax.random.PRNGKey(run_seed + 1000)
    mem_devices, rng_key = init_conv_memristors(
        params, memristor_path, scale, rng_key)

    pulse_counts = init_pulse_counts(mem_devices)
    n_cells_layer = layer_cell_counts(mem_devices)
    pulse_warned = False

    adc_info = arm_adc(adc, mem_devices, channels, layers)
    rng_key = jax.random.PRNGKey(run_seed)
    history = []
    test_acc = None
    global_epoch = 0
    for layer_idx, layer_name in enumerate(layers):
        train_step = make_mem_cwsg_train_step(
            model, layer_name, threshold, temperature, num_classes)
        evaluate_fn = make_mem_cwsg_evaluate_fn(
            model, layer_name, num_classes)

        pbar = tqdm(range(epochs_per_layer),
                    desc=f"CwSG {layer_name} seed{run_seed}", ncols=110)
        for epoch in pbar:
            epoch_loss = jnp.float32(0.0)
            nb = 0
            for imgs, labels in prefetch_to_device(ds_train):
                rng_key, step_key = jax.random.split(rng_key)
                (mem_devices, params, batch_stats, pulse_counts,
                 loss, _) = train_step(
                    mem_devices, params, batch_stats, pulse_counts,
                    imgs, labels, step_key)
                epoch_loss += loss; nb += 1
            avg_loss = float(epoch_loss) / max(nb, 1)

            if (eval_every and (epoch + 1) % eval_every == 0) or epoch == epochs_per_layer - 1:
                mem_values = eval_memristor_dict(mem_devices)
                test_acc, rng_key = compute_mem_cwsg_accuracy(
                    evaluate_fn, mem_values, params, batch_stats,
                    ds_test, rng_key)
                pbar.set_postfix(loss=f"{avg_loss:.3f}",
                                 acc=f"{test_acc:.2f}%")
            else:
                pbar.set_postfix(loss=f"{avg_loss:.3f}")

            history.append(_per_epoch_record(
                epoch=epoch, global_epoch=global_epoch,
                layer_idx=layer_idx, layer_name=layer_name,
                epoch_in_layer=epoch,
                train_loss=avg_loss, test_acc=test_acc))
            global_epoch += 1

            pulse_warned = check_pulse_endurance(
                pulse_counts, n_cells_layer, max_pulses_per_device,
                pulse_warned, run_label=f"CwSG {layer_name} seed{run_seed}")
        pbar.close()

    return {
        "history": history,
        "final_test_acc": float(test_acc) if test_acc is not None else None,
        "pulse_stats": compute_pulse_stats(mem_devices),
        "weight_stats": compute_weight_stats(mem_devices),
        "adc": adc_info,
    }


def run_mem_cwsg_interleaved(*, scale, threshold, temperature, num_classes,
                             ds_train, ds_test, input_shape, norm_name,
                             epochs, memristor_path, channels, pooling, layers,
                             max_pulses_per_device, run_seed,
                             eval_every):
    """Single interleaved CwSG-on-device run: each batch pulses ALL conv
    layers at once (summed per-layer loss from the pre-update devices);
    `epochs` = total passes. Eval reads out the last layer with fresh
    per-pass keys, so eval cadence never touches the training rng.
    Returns {history, final_test_acc, pulse_stats, weight_stats}."""
    set_adc_config()
    model = create_local_model("cnn", channels=channels,
                               pooling_after_n_layers=pooling,
                               noise_std=0.0, norm=norm_name)
    init_variables = model.init(
        {"params": jax.random.PRNGKey(run_seed),
         "noise": jax.random.PRNGKey(1)},
        jnp.ones((1,) + input_shape))
    params = init_variables["params"]
    batch_stats = init_variables.get("batch_stats", {})

    rng_key = jax.random.PRNGKey(run_seed + 1000)
    mem_devices, rng_key = init_conv_memristors(
        params, memristor_path, scale, rng_key)

    pulse_counts = init_pulse_counts(mem_devices)
    n_cells_layer = layer_cell_counts(mem_devices)
    pulse_warned = False

    train_step = make_mem_cwsg_interleaved_step(
        model, layers, threshold, temperature, num_classes)
    evaluate_fn = make_mem_cwsg_evaluate_fn(
        model, layers[-1], num_classes)

    rng_key = jax.random.PRNGKey(run_seed)
    history = []
    test_acc = None
    pbar = tqdm(range(epochs), desc=f"CwSG interleaved seed{run_seed}", ncols=110)
    for epoch in pbar:
        epoch_loss = jnp.float32(0.0)
        nb = 0
        for imgs, labels in prefetch_to_device(ds_train):
            rng_key, step_key = jax.random.split(rng_key)
            (mem_devices, params, batch_stats, pulse_counts,
             loss, _) = train_step(
                mem_devices, params, batch_stats, pulse_counts,
                imgs, labels, step_key)
            epoch_loss += loss; nb += 1
        avg_loss = float(epoch_loss) / max(nb, 1)

        if (eval_every and (epoch + 1) % eval_every == 0) or epoch == epochs - 1:
            mem_values = eval_memristor_dict(mem_devices)
            test_acc, _ = compute_mem_cwsg_accuracy(
                evaluate_fn, mem_values, params, batch_stats,
                ds_test, jax.random.PRNGKey(7 + epoch))
            pbar.set_postfix(loss=f"{avg_loss:.3f}",
                             acc=f"{test_acc:.2f}%")
        else:
            pbar.set_postfix(loss=f"{avg_loss:.3f}")

        history.append(_per_epoch_record(
            epoch=epoch, global_epoch=epoch,
            layer_idx=None, layer_name="interleaved",
            epoch_in_layer=epoch,
            train_loss=avg_loss, test_acc=test_acc))

        pulse_warned = check_pulse_endurance(
            pulse_counts, n_cells_layer, max_pulses_per_device,
            pulse_warned, run_label=f"CwSG interleaved seed{run_seed}")
    pbar.close()

    return {
        "history": history,
        "final_test_acc": float(test_acc) if test_acc is not None else None,
        "pulse_stats": compute_pulse_stats(mem_devices),
        "weight_stats": compute_weight_stats(mem_devices),
    }


def aggregate_history(per_run):
    """Stack per-run histories into per-epoch mean/std (runs share the same
    epoch schedule, aligned by index)."""
    n_runs = len(per_run)
    if n_runs == 0:
        return []
    n_epochs = len(per_run[0]["history"])
    out = []
    for i in range(n_epochs):
        ref = per_run[0]["history"][i]
        accs = []
        losses = []
        for r in per_run:
            rec = r["history"][i]
            if rec.get("test_acc") is not None:
                accs.append(rec["test_acc"])
            losses.append(rec["train_loss"])
        agg = {
            "epoch": ref["epoch"],
            "global_epoch": ref["global_epoch"],
            "layer_idx": ref["layer_idx"],
            "layer_name": ref["layer_name"],
            "epoch_in_layer": ref["epoch_in_layer"],
            "n_runs_with_acc": len(accs),
            "train_loss_mean": float(np.mean(losses)),
            "train_loss_std":  float(np.std(losses)),
            "test_acc_mean":  float(np.mean(accs)) if accs else None,
            "test_acc_std":   float(np.std(accs)) if accs else None,
            "per_run_test_acc": [
                r["history"][i].get("test_acc") for r in per_run],
            "per_run_train_loss": [
                r["history"][i]["train_loss"] for r in per_run],
        }
        out.append(agg)
    return out


def aggregate_pulse_stats(per_run):
    """Aggregate per-layer pulse stats across runs (mean/std) and
    cell-weighted global averages."""
    runs_pulse = [r["pulse_stats"] for r in per_run]
    if not runs_pulse:
        return {"per_layer": {}, "global": {}}
    layer_names = list(runs_pulse[0].keys())

    per_layer_agg = {}
    keys = ("pos_min", "pos_max", "pos_mean", "pos_std",
            "neg_min", "neg_max", "neg_mean", "neg_std")
    for ln in layer_names:
        agg = {"n_cells": int(runs_pulse[0][ln]["n_cells"])}
        for k in keys:
            vals = [float(rp[ln][k]) for rp in runs_pulse]
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"]  = float(np.std(vals))
            agg[f"{k}_per_run"] = vals
        per_layer_agg[ln] = agg

    pos_globals, neg_globals = [], []
    pos_hot, neg_hot = [], []
    total_pulses_per_run = []
    for rp in runs_pulse:
        n_cells_total = sum(s["n_cells"] for s in rp.values())
        pos_mean_w = (sum(s["pos_mean"] * s["n_cells"] for s in rp.values())
                      / n_cells_total)
        neg_mean_w = (sum(s["neg_mean"] * s["n_cells"] for s in rp.values())
                      / n_cells_total)
        pos_globals.append(float(pos_mean_w))
        neg_globals.append(float(neg_mean_w))
        pos_hot.append(int(max(s["pos_max"] for s in rp.values())))
        neg_hot.append(int(max(s["neg_max"] for s in rp.values())))
        total = sum((s["pos_mean"] + s["neg_mean"]) * s["n_cells"]
                    for s in rp.values())
        total_pulses_per_run.append(float(total))

    global_agg = {
        "pos_mean_weighted_per_run":  pos_globals,
        "neg_mean_weighted_per_run":  neg_globals,
        "pos_mean_weighted_mean":     float(np.mean(pos_globals)),
        "pos_mean_weighted_std":      float(np.std(pos_globals)),
        "neg_mean_weighted_mean":     float(np.mean(neg_globals)),
        "neg_mean_weighted_std":      float(np.std(neg_globals)),
        "hottest_pos_pulse_per_run":  pos_hot,
        "hottest_neg_pulse_per_run":  neg_hot,
        "total_pulses_per_run":       total_pulses_per_run,
        "total_pulses_mean":          float(np.mean(total_pulses_per_run)),
        "total_pulses_std":           float(np.std(total_pulses_per_run)),
        "n_cells_total":              int(sum(
            per_layer_agg[ln]["n_cells"] for ln in layer_names)),
    }
    return {"per_layer": per_layer_agg, "global": global_agg}
