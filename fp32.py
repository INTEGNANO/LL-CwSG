"""FP32 (no device model) training: JIT train-step/eval builders for
Backprop, CwC and CwSG, training-curve loops, and read/write-noise sweeps.
helper's module globals (clamp, stochastic threshold, write-noise LR divisor)
are baked in at JIT-trace/build time - call helper.set_* BEFORE anything here.
"""

import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

import helper
from helper import (
    prefetch_to_device,
    create_local_model,
    create_classifier_model,
    make_optimizer,
    add_write_noise,
    clamp_weights,
    mask_local_grads,
    _model_has_noise,
    _noise_rngs,
    channel_group_goodness,
    spatial_gradient_energy,
    compute_bp_accuracy,
    compute_prototype_accuracy,
    average_metrics,
)


def make_bp_train_step(model, optimizer, write_noise_std, temperature=1.0):
    """Build the JIT-compiled Backprop train step: end-to-end CE on the
    classifier logits, with optional write noise and weight clamping."""
    _has_noise = _model_has_noise(model)
    @jax.jit
    def step(params, batch_stats, opt_state, imgs, labels, rng_key):
        rng_key, write_key = jax.random.split(rng_key)
        def loss_fn(p):
            apply_kwargs = {"params": p, "batch_stats": batch_stats}
            out, new_state = model.apply(
                apply_kwargs, imgs, train=True, mutable=["batch_stats"],
                rngs=_noise_rngs(_has_noise, rng_key))
            loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
                out / temperature, labels))
            return loss, new_state
        (loss, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        updates = add_write_noise(updates, write_noise_std, write_key)
        new_params = clamp_weights(optax.apply_updates(params, updates))
        return new_params, new_state.get("batch_stats", {}), new_opt_state, loss
    return step


def make_cwc_train_step(model, opt_conv, layer_name,
                        read_noise_std, write_noise_std,
                        temperature, num_classes):
    """Train step for CwC: CE on raw channel-group goodness.
    Gradients are masked so only `layer_name`'s weights update."""
    _has_noise = _model_has_noise(model)

    @jax.jit
    def step(params, batch_stats, opt_state_conv, imgs, labels, rng_key):
        rng_key, write_key = jax.random.split(rng_key)

        def loss_fn(p):
            conv_key = rng_key
            act, new_state = model.apply(
                {"params": p, "batch_stats": batch_stats},
                imgs, train=True, target_layer=layer_name,
                mutable=["batch_stats"], rngs=_noise_rngs(_has_noise, conv_key)
            )
            goodness = channel_group_goodness(act, num_classes)
            logits = goodness / temperature
            loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
            return loss, new_state

        (loss, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grads = mask_local_grads(grads, layer_name)
        updates, new_opt = opt_conv.update(grads, opt_state_conv, params)
        updates = add_write_noise(updates, write_noise_std, write_key)
        new_params = clamp_weights(optax.apply_updates(params, updates))
        return new_params, new_state["batch_stats"], new_opt, loss

    return step


def make_cwc_interleaved_step(model, opt_conv, layers,
                              read_noise_std, write_noise_std,
                              temperature, num_classes):
    """Interleaved train step for CwC: sum per-layer CE losses over all layers.
    All layers update in parallel from the same pre-update params."""
    _has_noise = _model_has_noise(model)

    @jax.jit
    def step(params, batch_stats, opt_state_conv, imgs, labels, rng_key):
        rng_key, write_key = jax.random.split(rng_key)

        def loss_fn(p):
            total = jnp.float32(0.0)
            new_state = {"batch_stats": batch_stats}
            for layer_name in layers:
                conv_key = rng_key
                act, ns = model.apply(
                    {"params": p, "batch_stats": batch_stats},
                    imgs, train=True, target_layer=layer_name,
                    mutable=["batch_stats"], rngs=_noise_rngs(_has_noise, conv_key)
                )
                goodness = channel_group_goodness(act, num_classes)
                logits = goodness / temperature
                total = total + jnp.mean(
                    optax.softmax_cross_entropy_with_integer_labels(logits, labels))
                new_state = ns
            return total, new_state

        (loss, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt = opt_conv.update(grads, opt_state_conv, params)
        updates = add_write_noise(updates, write_noise_std, write_key)
        new_params = clamp_weights(optax.apply_updates(params, updates))
        return new_params, new_state["batch_stats"], new_opt, loss

    return step


def make_cwc_evaluate_fn(model, eval_layer, num_classes, temperature=1.0):
    """Eval function for CwC: argmax of channel-group goodness at `eval_layer`."""
    _has_noise = _model_has_noise(model)

    @jax.jit
    def evaluate_batch(params, batch_stats, imgs, rng_key):
        act = model.apply(
            {"params": params, "batch_stats": batch_stats},
            imgs, train=False, target_layer=eval_layer, rngs=_noise_rngs(_has_noise, rng_key)
        )
        goodness = channel_group_goodness(act, num_classes)
        return jnp.argmax(goodness, axis=-1)
    return evaluate_batch


def _dcl_dictionary(model, num_classes):
    """Fixed random class dictionary for DCL: one Gaussian vector per class,
    dimension = widest layer, drawn once with seed 42 (same for every run).
    Never trained."""
    embed_dim = max(model.channels.values())
    return jax.random.normal(jax.random.PRNGKey(42), (num_classes, embed_dim))


def _pool_dictionary(dictionary, target_dim):
    """Shrink the dictionary to a narrower layer by averaging groups of
    columns. Widths must divide the dictionary dimension (64/128/256 do)."""
    K, C_D = dictionary.shape
    if C_D == target_dim:
        return dictionary
    assert C_D % target_dim == 0
    return dictionary.reshape(K, target_dim, C_D // target_dim).mean(axis=-1)


def _dcl_logits(act, dictionary, temperature):
    """DCL class scores: spatially averaged activation dotted with the
    (pooled) class dictionary."""
    h = jnp.mean(act, axis=(1, 2)) if act.ndim == 4 else act
    pooled = _pool_dictionary(dictionary, h.shape[-1])
    return (h @ pooled.T) / temperature


def make_dcl_train_step(model, opt_conv, layer_name,
                        read_noise_std, write_noise_std,
                        temperature, num_classes):
    """Train step for DCL: CE on the dictionary logits of the layer's
    spatially averaged activation. Only `layer_name`'s weights update."""
    _has_noise = _model_has_noise(model)
    dictionary = _dcl_dictionary(model, num_classes)

    @jax.jit
    def step(params, batch_stats, opt_state_conv, imgs, labels, rng_key):
        rng_key, write_key = jax.random.split(rng_key)

        def loss_fn(p):
            act, new_state = model.apply(
                {"params": p, "batch_stats": batch_stats},
                imgs, train=True, target_layer=layer_name,
                mutable=["batch_stats"], rngs=_noise_rngs(_has_noise, rng_key)
            )
            logits = _dcl_logits(act, dictionary, temperature)
            loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
            return loss, new_state

        (loss, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grads = mask_local_grads(grads, layer_name)
        updates, new_opt = opt_conv.update(grads, opt_state_conv, params)
        updates = add_write_noise(updates, write_noise_std, write_key)
        new_params = clamp_weights(optax.apply_updates(params, updates))
        return new_params, new_state["batch_stats"], new_opt, loss

    return step


def make_dcl_evaluate_fn(model, eval_layer, num_classes, temperature=1.0):
    """Eval function for DCL: argmax of the dictionary logits at `eval_layer`."""
    _has_noise = _model_has_noise(model)
    dictionary = _dcl_dictionary(model, num_classes)

    @jax.jit
    def evaluate_batch(params, batch_stats, imgs, rng_key):
        act = model.apply(
            {"params": params, "batch_stats": batch_stats},
            imgs, train=False, target_layer=eval_layer, rngs=_noise_rngs(_has_noise, rng_key)
        )
        logits = _dcl_logits(act, dictionary, 1.0)
        return jnp.argmax(logits, axis=-1)
    return evaluate_batch


def make_cwsg_train_step(model, opt_conv, layer_name,
                         read_noise_std, write_noise_std,
                         temperature, num_classes):
    """Train step for CwSG: CE on raw spatial gradient energy goodness.
    Gradients are masked so only `layer_name`'s weights update."""
    _has_noise = _model_has_noise(model)

    @jax.jit
    def step(params, batch_stats, opt_state_conv, imgs, labels, rng_key):
        rng_key, write_key = jax.random.split(rng_key)

        def loss_fn(p):
            conv_key = rng_key
            act, new_state = model.apply(
                {"params": p, "batch_stats": batch_stats},
                imgs, train=True, target_layer=layer_name,
                mutable=["batch_stats"], rngs=_noise_rngs(_has_noise, conv_key)
            )
            goodness = spatial_gradient_energy(act, num_classes)
            logits = goodness / temperature
            loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
            return loss, new_state

        (loss, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grads = mask_local_grads(grads, layer_name)
        updates, new_opt = opt_conv.update(grads, opt_state_conv, params)
        updates = add_write_noise(updates, write_noise_std, write_key)
        new_params = clamp_weights(optax.apply_updates(params, updates))
        return new_params, new_state["batch_stats"], new_opt, loss

    return step


def make_cwsg_interleaved_step(model, opt_conv, layers,
                               read_noise_std, write_noise_std,
                               temperature, num_classes):
    """Interleaved train step for CwSG: sum per-layer CE losses over all layers.
    All layers update in parallel from the same pre-update params."""
    _has_noise = _model_has_noise(model)

    @jax.jit
    def step(params, batch_stats, opt_state_conv, imgs, labels, rng_key):
        rng_key, write_key = jax.random.split(rng_key)

        def loss_fn(p):
            total = jnp.float32(0.0)
            new_state = {"batch_stats": batch_stats}
            for layer_name in layers:
                conv_key = rng_key
                act, ns = model.apply(
                    {"params": p, "batch_stats": batch_stats},
                    imgs, train=True, target_layer=layer_name,
                    mutable=["batch_stats"], rngs=_noise_rngs(_has_noise, conv_key)
                )
                goodness = spatial_gradient_energy(act, num_classes)
                logits = goodness / temperature
                total = total + jnp.mean(
                    optax.softmax_cross_entropy_with_integer_labels(logits, labels))
                new_state = ns
            return total, new_state

        (loss, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt = opt_conv.update(grads, opt_state_conv, params)
        updates = add_write_noise(updates, write_noise_std, write_key)
        new_params = clamp_weights(optax.apply_updates(params, updates))
        return new_params, new_state["batch_stats"], new_opt, loss

    return step


def make_cwsg_evaluate_fn(model, eval_layer, num_classes, temperature=1.0):
    """Eval function for CwSG: argmax of spatial gradient energy goodness at `eval_layer`."""
    _has_noise = _model_has_noise(model)

    @jax.jit
    def evaluate_batch(params, batch_stats, imgs, rng_key):
        act = model.apply(
            {"params": params, "batch_stats": batch_stats},
            imgs, train=False, target_layer=eval_layer, rngs=_noise_rngs(_has_noise, rng_key)
        )
        goodness = spatial_gradient_energy(act, num_classes)
        return jnp.argmax(goodness, axis=-1)
    return evaluate_batch


def train_backprop_curve(lr, temperature, num_classes, ds_train, ds_test,
                         input_shape, norm_name, epochs, channels, pooling,
                         optimizer_name, run_seed=0):
    """One end-to-end Backprop training pass; returns the per-epoch history
    of train loss and test accuracy."""
    model = create_classifier_model(
        "cnn", channels=channels, pooling_after_n_layers=pooling,
        num_classes=num_classes, noise_std=0.0, norm=norm_name)
    init_variables = model.init(
        {"params": jax.random.PRNGKey(run_seed), "noise": jax.random.PRNGKey(1)},
        jnp.ones((1,) + input_shape))
    optimizer = make_optimizer(optimizer_name, lr, 0)
    train_step = make_bp_train_step(model, optimizer, 0.0, temperature)

    params = init_variables["params"]
    batch_stats = init_variables.get("batch_stats", {})
    opt_state = optimizer.init(params)
    rng_key = jax.random.PRNGKey(run_seed)

    history = []
    pbar = tqdm(range(epochs), desc="BP", ncols=100)
    for epoch in pbar:
        epoch_loss = 0.0
        num_batches = 0
        for imgs, labels in prefetch_to_device(ds_train):
            rng_key, step_key = jax.random.split(rng_key)
            params, batch_stats, opt_state, loss = train_step(
                params, batch_stats, opt_state, imgs, labels, step_key)
            epoch_loss += float(loss)
            num_batches += 1
        avg_loss = epoch_loss / max(num_batches, 1)
        test_acc, rng_key = compute_bp_accuracy(model, params, batch_stats, ds_test, rng_key)
        history.append({
            "epoch": epoch,
            "global_epoch": epoch,
            "layer_idx": None,
            "layer_name": None,
            "epoch_in_layer": None,
            "train_loss": avg_loss,
            "test_acc": float(test_acc),
        })
        pbar.set_postfix(loss=f"{avg_loss:.3f}", acc=f"{test_acc:.2f}%")
    pbar.close()
    return history


def train_greedy_curve(make_step_fn, make_eval_fn, lr_conv, temperature,
                       num_classes, ds_train, ds_test, input_shape, norm_name,
                       epochs_per_layer, channels, pooling, layers,
                       optimizer_name, run_seed=0,
                       algo_label="CwSG"):
    """One greedy layer-wise pass; make_step_fn/make_eval_fn select CwC or
    CwSG. Returns per-epoch history; test read-out tracks the trained layer."""
    model = create_local_model("cnn", channels=channels,
                             pooling_after_n_layers=pooling,
                             noise_std=0.0, norm=norm_name)
    init_variables = model.init(
        {"params": jax.random.PRNGKey(run_seed), "noise": jax.random.PRNGKey(1)},
        jnp.ones((1,) + input_shape))
    params = init_variables["params"]
    batch_stats = init_variables.get("batch_stats", {})
    rng_key = jax.random.PRNGKey(run_seed)

    history = []
    global_epoch = 0
    for layer_idx, layer_name in enumerate(layers):
        C_L = channels[layer_name]
        if C_L < num_classes:
            print(f"  [{layer_name}] C={C_L} < K={num_classes}, FROZEN")
            continue
        opt_conv = make_optimizer(optimizer_name, lr_conv, 0)
        train_step = make_step_fn(
            model, opt_conv, layer_name,
            0.0, 0.0, temperature, num_classes)
        evaluate_fn = make_eval_fn(
            model, layer_name, num_classes, temperature)
        opt_state = opt_conv.init(params)

        pbar = tqdm(range(epochs_per_layer),
                    desc=f"{algo_label} {layer_name}", ncols=110)
        for epoch in pbar:
            epoch_loss = 0.0
            num_batches = 0
            for imgs, labels in prefetch_to_device(ds_train):
                rng_key, step_key = jax.random.split(rng_key)
                params, batch_stats, opt_state, loss = train_step(
                    params, batch_stats, opt_state, imgs, labels, step_key)
                epoch_loss += float(loss)
                num_batches += 1
            avg_loss = epoch_loss / max(num_batches, 1)
            rng_key, eval_rng = jax.random.split(rng_key)
            test_acc, _ = compute_prototype_accuracy(
                evaluate_fn, params, batch_stats, ds_test, eval_rng)
            history.append({
                "epoch": epoch,
                "global_epoch": global_epoch,
                "layer_idx": layer_idx,
                "layer_name": layer_name,
                "epoch_in_layer": epoch,
                "train_loss": avg_loss,
                "test_acc": float(test_acc),
            })
            global_epoch += 1
            pbar.set_postfix(loss=f"{avg_loss:.3f}", acc=f"{test_acc:.2f}%")
        pbar.close()
    return history


def train_interleaved_curve(make_interleaved_step_fn, make_eval_fn, lr_conv,
                            temperature, num_classes, ds_train, ds_test,
                            input_shape, norm_name,
                            epochs, channels, pooling, layers,
                            optimizer_name, run_seed=0,
                            algo_label="CwSG"):
    """One interleaved pass: each batch updates ALL layers at once (summed
    per-layer loss, parallel updates from the pre-update params, one shared
    optimizer); `epochs` = total passes over the data. make_interleaved_step_fn/
    make_eval_fn select CwC or CwSG (make_*_interleaved_step / make_*_evaluate_fn).
    Eval reads out the last layer with fresh per-pass keys, so eval cadence
    never touches the training rng. Returns per-pass history."""
    model = create_local_model("cnn", channels=channels,
                               pooling_after_n_layers=pooling,
                               noise_std=0.0, norm=norm_name)
    init_variables = model.init(
        {"params": jax.random.PRNGKey(run_seed), "noise": jax.random.PRNGKey(1)},
        jnp.ones((1,) + input_shape))
    params = init_variables["params"]
    batch_stats = init_variables.get("batch_stats", {})
    rng_key = jax.random.PRNGKey(run_seed)

    opt_conv = make_optimizer(optimizer_name, lr_conv, 0)
    train_step = make_interleaved_step_fn(
        model, opt_conv, list(layers), 0.0, 0.0, temperature, num_classes)
    evaluate_fn = make_eval_fn(model, layers[-1], num_classes, temperature)
    opt_state = opt_conv.init(params)

    history = []
    pbar = tqdm(range(epochs), desc=f"{algo_label} interleaved", ncols=110)
    for epoch in pbar:
        epoch_loss = 0.0
        num_batches = 0
        for imgs, labels in prefetch_to_device(ds_train):
            rng_key, step_key = jax.random.split(rng_key)
            params, batch_stats, opt_state, loss = train_step(
                params, batch_stats, opt_state, imgs, labels, step_key)
            epoch_loss += float(loss)
            num_batches += 1
        avg_loss = epoch_loss / max(num_batches, 1)
        test_acc, _ = compute_prototype_accuracy(
            evaluate_fn, params, batch_stats, ds_test,
            jax.random.PRNGKey(7 + epoch))
        history.append({
            "epoch": epoch,
            "global_epoch": epoch,
            "layer_idx": None,
            "layer_name": "interleaved",
            "epoch_in_layer": epoch,
            "train_loss": avg_loss,
            "test_acc": float(test_acc),
        })
        pbar.set_postfix(loss=f"{avg_loss:.3f}", acc=f"{test_acc:.2f}%")
    pbar.close()
    return history


def sweep_backprop(noise_pairs, num_runs, epochs,
                   ds_train, ds_test, num_classes=10,
                   input_shape=(32, 32, 3), norm="rmsnorm", eval_every=1,
                   channels=None, pooling=None, lr=1e-3,
                   temperature=1.0, optimizer_name="sign_sgd", total_steps=0):
    """Backprop noise sweep: per (read, write) pair, train num_runs classifiers
    and return run-averaged per-epoch metrics keyed by noise pair."""
    if channels is None or pooling is None:
        raise ValueError("channels and pooling are required - pass the "
                         "values returned by helper.build_architecture")
    results = {}
    LR = lr

    for read_ns, write_ns in noise_pairs:
        noise_key = f"r{read_ns:.4f}_w{write_ns:.4f}"
        print(f"\n  [Backprop] read_noise={read_ns:.4f}, write_noise={write_ns:.4f}")
        model = create_classifier_model(
            "cnn", channels=channels, pooling_after_n_layers=pooling,
            num_classes=num_classes, noise_std=read_ns, norm=norm
        )
        optimizer = make_optimizer(optimizer_name, LR, total_steps)
        train_step = make_bp_train_step(model, optimizer, write_ns, temperature)

        all_run_metrics = []
        for run_idx in range(num_runs):
            init_variables = model.init(
                {'params': jax.random.PRNGKey(run_idx + helper.get_seed_offset()), 'noise': jax.random.PRNGKey(1)},
                jnp.ones((1,) + input_shape)
            )
            params = init_variables["params"]
            batch_stats = init_variables.get("batch_stats", {})
            opt_state = optimizer.init(params)
            rng_key = jax.random.PRNGKey(run_idx + helper.get_seed_offset())

            run_metrics = []
            pbar = tqdm(range(epochs),
                        desc=f"    Run {run_idx+1}/{num_runs} Backprop",
                        leave=False)
            for epoch in pbar:
                epoch_loss = jnp.float32(0.0)
                num_batches = 0
                for imgs, labels in prefetch_to_device(ds_train):
                    rng_key, step_key = jax.random.split(rng_key)
                    params, batch_stats, opt_state, loss = train_step(
                        params, batch_stats, opt_state, imgs, labels, step_key
                    )
                    epoch_loss += loss
                    num_batches += 1
                avg_loss = float(epoch_loss) / num_batches
                pbar.set_postfix({"Loss": f"{avg_loss:.3f}"})

                if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
                    rng_key, eval_rng = jax.random.split(rng_key)
                    train_acc, eval_rng = compute_bp_accuracy(model, params, batch_stats, ds_train, eval_rng)
                    test_acc, _ = compute_bp_accuracy(model, params, batch_stats, ds_test, eval_rng)
                else:
                    train_acc, test_acc = float('nan'), float('nan')

                run_metrics.append({
                    "global_epoch": epoch,
                    "layer": None,
                    "local_epoch": epoch,
                    "loss": avg_loss,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                })

            all_run_metrics.append(run_metrics)
            tqdm.write(f"    Run {run_idx+1} -> Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")

        averaged = average_metrics(all_run_metrics)
        results[noise_key] = averaged

        final = averaged[-1]
        print(f"  [Backprop] read={read_ns:.4f} write={write_ns:.4f} | "
              f"Train: {final['avg_train_acc']:.2f}% +/- {final['std_train_acc']:.2f}% | "
              f"Test: {final['avg_test_acc']:.2f}% +/- {final['std_test_acc']:.2f}%")

    return results


def sweep_cwsg(noise_pairs, num_runs, epochs_per_layer,
               ds_train, ds_test, num_classes=10,
               input_shape=(32, 32, 3), norm="rmsnorm", eval_every=1,
               channels=None, pooling=None, layers=None,
               lr_conv=1e-3, temperature=1.0,
               optimizer_name="sign_sgd", total_steps=0):
    """CwSG noise sweep: per (read, write) pair, run num_runs greedy layer-wise
    trainings and return run-averaged per-epoch metrics keyed by noise pair."""
    if channels is None or pooling is None or layers is None:
        raise ValueError("channels, pooling and layers are required - pass "
                         "the values returned by helper.build_architecture")
    results = {}

    for read_ns, write_ns in noise_pairs:
        noise_key = f"r{read_ns:.4f}_w{write_ns:.4f}"
        print(f"\n  [CwSG] read_noise={read_ns:.4f}, write_noise={write_ns:.4f}")
        model = create_local_model("cnn", channels=channels,
                                 pooling_after_n_layers=pooling,
                                 noise_std=read_ns, norm=norm)

        layer_fns = {}
        layer_eval_fns = {}
        for layer_name in layers:
            opt_conv = make_optimizer(optimizer_name, lr_conv, total_steps)
            layer_fns[layer_name] = (
                make_cwsg_train_step(model, opt_conv, layer_name,
                                     read_ns, write_ns, temperature, num_classes),
                opt_conv)
            layer_eval_fns[layer_name] = make_cwsg_evaluate_fn(
                model, layer_name, num_classes, temperature)

        all_run_metrics = []
        for run_idx in range(num_runs):
            init_variables = model.init(
                {'params': jax.random.PRNGKey(run_idx + helper.get_seed_offset()), 'noise': jax.random.PRNGKey(1)},
                jnp.ones((1,) + input_shape))
            params = init_variables["params"]
            batch_stats = init_variables.get("batch_stats", {})
            rng_key = jax.random.PRNGKey(run_idx + helper.get_seed_offset())

            run_metrics = []
            global_epoch = 0

            for layer_name in layers:
                evaluate_fn = layer_eval_fns[layer_name]
                train_step, opt_conv = layer_fns[layer_name]
                opt_s_conv = opt_conv.init(params)

                pbar = tqdm(range(epochs_per_layer),
                            desc=f"    Run {run_idx+1}/{num_runs} {layer_name}",
                            leave=False)
                for epoch in pbar:
                    epoch_loss = jnp.float32(0.0)
                    num_batches = 0
                    for imgs, labels in prefetch_to_device(ds_train):
                        rng_key, step_key = jax.random.split(rng_key)
                        params, batch_stats, opt_s_conv, loss = train_step(
                            params, batch_stats, opt_s_conv, imgs, labels, step_key)
                        epoch_loss += loss
                        num_batches += 1
                    avg_loss = float(epoch_loss) / num_batches
                    pbar.set_postfix({"Loss": f"{avg_loss:.3f}"})

                    if (epoch + 1) % eval_every == 0 or epoch == epochs_per_layer - 1:
                        rng_key, eval_rng = jax.random.split(rng_key)
                        train_acc, eval_rng = compute_prototype_accuracy(
                            evaluate_fn, params, batch_stats, ds_train, eval_rng)
                        test_acc, _ = compute_prototype_accuracy(
                            evaluate_fn, params, batch_stats, ds_test, eval_rng)
                    else:
                        train_acc, test_acc = float('nan'), float('nan')

                    run_metrics.append({
                        "global_epoch": global_epoch,
                        "layer": layer_name,
                        "local_epoch": epoch,
                        "loss": avg_loss,
                        "train_acc": train_acc,
                        "test_acc": test_acc,
                    })
                    global_epoch += 1

                tqdm.write(f"    Run {run_idx+1} -> {layer_name} "
                           f"Test Acc: {run_metrics[-1]['test_acc']:.2f}%")

            all_run_metrics.append(run_metrics)

        averaged = average_metrics(all_run_metrics)
        results[noise_key] = averaged

        final = averaged[-1]
        print(f"  [CwSG] read={read_ns:.4f} write={write_ns:.4f} | "
              f"Train: {final['avg_train_acc']:.2f}% +/- {final['std_train_acc']:.2f}% | "
              f"Test: {final['avg_test_acc']:.2f}% +/- {final['std_test_acc']:.2f}%")

    return results
