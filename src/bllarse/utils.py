import equinox as eqx
import jax.tree as jtu
from jax import nn, lax, numpy as jnp, vmap, random as jr
from tensorflow_probability.substrates.jax.stats import expected_calibration_error as compute_ece
from functools import partial
import optax
try:
    import wandb
    no_wandb = False
except:
    no_wandb = True
import numpy as onp

from typing import Mapping, Optional

from blrax.utils import noisy_value_and_grad
from bllarse.losses import Classical

import jax.scipy.linalg as linalg
from jax.lax.linalg import triangular_solve
import augmax
from math import prod

# these values were copied from scaling mlps repo https://github.com/gregorbachmann/scaling_mlps
MEAN_DICT = {
    "imagenet21": jnp.asarray([0.485, 0.456, 0.406]),
    "imagenet": jnp.asarray([0.485, 0.456, 0.406]),
    "imagenet_real": jnp.asarray([0.485, 0.456, 0.406]),
    "tinyimagenet": jnp.asarray([0.485, 0.456, 0.406]),
    "cifar10": jnp.asarray([0.49139968, 0.48215827, 0.44653124]),
    "cifar100": jnp.asarray([0.49139968, 0.48215827, 0.44653124]),
    "stl10": jnp.asarray([0.4914, 0.4822, 0.4465]),
}


STD_DICT = {
    "imagenet21": jnp.asarray([0.229, 0.224, 0.225]),
    "imagenet": jnp.asarray([0.229, 0.224, 0.225]),
    "imagenet_real": jnp.asarray([0.229, 0.224, 0.225]),
    "tinyimagenet": jnp.asarray([0.229, 0.224, 0.225]),
    "cifar10": jnp.asarray([0.24703233, 0.24348505, 0.26158768]),
    "cifar100": jnp.asarray([0.24703233, 0.24348505, 0.26158768]),
    "stl10": jnp.asarray([0.2471, 0.2435, 0.2616]),
}

def stable_inverse(arr_to_invert):
    """ Assumes `arr_to_invert` is a batch of matrices of shape (..., dim, dim)"""
    dim = arr_to_invert.shape[-1]
    L_chol = linalg.cho_factor(arr_to_invert, lower=True)
    return linalg.cho_solve(L_chol, jnp.broadcast_to(jnp.eye(dim), arr_to_invert.shape, dtype=arr_to_invert.dtype))

def solve_precision(L, b):
    """Solve (LLᵀ)x = b for x given lower-triangular L (Cholesky of Λ)."""
    y = triangular_solve(L, b, left_side=True, lower=True)
    x = triangular_solve(L, y, left_side=True, lower=True, transpose_a=True)
    return x

def get_number_of_parameters(model):
    params, _ = eqx.partition(model, eqx.is_array)
    leaves = jtu.leaves(params)

    return sum([prod(l.shape) for l in leaves])

def resize_images(img, img_size):
    func = augmax.Chain(
        augmax.ByteToFloat(),
        augmax.Resize(img_size, img_size)
    )

    return vmap(func, in_axes=(None, 0))(jr.PRNGKey(0), img)

# define data augmentation
def augmentdata(img, key=None, **kwargs):
    img_size = img.shape[1:-1]
    norm = augmax.Normalize(**kwargs)
    if key is None:
        func = augmax.Chain(
            norm,
        )
        return vmap(func, in_axes=(None, 0))(jr.PRNGKey(0), img)
    else:
        keys = jr.split(key, img.shape[0])
        func = augmax.Chain(
                    norm,
                    augmax.RandomSizedCrop(*img_size),
                    augmax.HorizontalFlip()
                )
        return vmap(func)(keys, img)

def evaluate_model(data_augmentation, loss_fn, nnet, images, labels, loss_type=3):
    aug_images = data_augmentation(images, key=None)
    features = extract_features(nnet, aug_images)
    loss, logits = loss_fn(features, labels, with_logits=True, loss_type=loss_type)
    predictions = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(predictions == labels)
    ece = compute_ece(20, logits=logits, labels_true=labels, labels_predicted=predictions)
    return acc, loss.mean(), ece

def extract_features(nnet, x):
    # vmap to keep things simple (batch, …) → (batch, embed_dim)
    return vmap(nnet)(x)

def run_training(
    key,
    last_layer,
    pretrained_nnet,
    loss_fn,
    optim,
    data_augmentation,
    train_ds,
    test_ds,
    opt_state=None,
    mc_samples=1,
    num_epochs=1,
    batch_size=32,
    log_to_wandb=False,
    ):
    """
    Train a neural network using Equinox and Optax.
    
    Args:
        key: JAX PRNG key
        last_layer: Equinox neural network
        pretrained_nnet: Equinox neural network
        loss_fn: Equinox loss function
        optim: Optax (or blrax) optimizer
        data_augmentation: Jax compatible data augmentation function
        train_ds: Training dataset dictionary with 'image' and 'label' keys
        test_ds: Test dataset dictionary with 'image' and 'label' keys
        opt_state: Initial optimizer state, if None it is initiated localy
        num_epochs: Number of epochs to train
        batch_size: Batch size for training
        log_to_wandb: Whether to log metrics to Weights & Biases
    """
    
    params, static = eqx.partition(last_layer, eqx.is_array)
    opt_state = optim.init(params) if opt_state is None else opt_state  # initialize optimizer state

    def local_loss(params, x, y, *args, **kwargs):
        ll = eqx.combine(params, static)
        return loss_fn(partial(ll, pretrained_nnet), x, y)

    # Training step function
    @eqx.filter_jit
    def train_step(params, opt_state, x, y, key):
        loss_value, grads, opt_state = noisy_value_and_grad(
            local_loss,
            opt_state,
            params, 
            key, 
            x, 
            y, 
            mc_samples=mc_samples
        )
        updates, opt_state = optim.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value
    
    # Evaluation function
    @eqx.filter_jit
    def evaluate(model, images, labels):
        return evaluate_model(data_augmentation, loss_fn, model, images, labels)
    
    # Inner training loop (one epoch)
    def train_epoch(carry, xs):
        params, opt_state = carry
        key, epoch = xs
        
        # Shuffle training data
        key, _key = jr.split(key)
        img_shape = train_ds['image'].shape[1:]
        num_datapoints = len(train_ds['label'])
        steps_per_epoch = num_datapoints // batch_size
        perm = jr.permutation(_key, num_datapoints)
        train_images = train_ds['image'][perm]
        train_labels = train_ds['label'][perm]
        
        def train_step_scan(carry, xs):
            params, opt_state, key = carry
            aug_key, grad_eval_key = jr.split(key)
            
            batch_images, batch_labels = xs
            aug_batch_images = data_augmentation(batch_images, key=aug_key)
            new_params, new_opt_state, loss_value = train_step(
                params, opt_state, aug_batch_images, batch_labels, grad_eval_key
            )

            return (new_params, new_opt_state, key), loss_value
        
        # Run training steps for one epoch
        data = (
            train_images[:steps_per_epoch * batch_size].reshape(steps_per_epoch, batch_size, *img_shape),
            train_labels[:steps_per_epoch * batch_size].reshape(steps_per_epoch, batch_size)
        )
        init_carry = (params, opt_state, key)
        (params, opt_state, key), losses = lax.scan(
            train_step_scan,
            init_carry,
            data
        )
        
        # Calculate metrics
        key, _key = jr.split(key)
        nnet = partial(eqx.combine(params, static), pretrained_nnet)
        acc, nll, ece = evaluate(
            nnet,
            test_ds['image'],
            test_ds['label'],
        )
        
        metrics = {
            'loss': losses.mean(),
            'acc': acc,
            'ece': ece,
            'nll': nll,
        }
        
        return (params, opt_state), metrics
    
    # Run training for multiple epochs
    keys = jr.split(key, num_epochs)
    init_carry = (params, opt_state)
    (params, final_opt_state), metrics_seq = lax.scan(
        train_epoch,
        init_carry,
        (keys, jnp.arange(num_epochs))
    )
    if log_to_wandb and not no_wandb:
        # turn DeviceArrays → python scalars so wandb is happy
        metrics_np = jtu.map(lambda x: onp.asarray(x), metrics_seq)
        for ep in range(num_epochs):
            wandb.log(
                {
                    "epoch": ep + 1,
                    "loss": float(metrics_np["loss"][ep]),
                    "nll":  float(metrics_np["nll"][ep]),
                    "acc":  float(metrics_np["acc"][ep]),
                    "ece":  float(metrics_np["ece"][ep]),
                }
            )

    trained_last_layer = eqx.combine(params, static)
    return trained_last_layer, final_opt_state, metrics_seq

def run_bayesian_training(
    key,
    pretrained_nnet,
    loss_model,
    data_augmentation,
    train_ds,
    test_ds,
    *,  
    optimizer = None,
    opt_state = None,
    tune_last_layer_only: bool = False,
    loss_type: int = 3,
    num_epochs: int = 1,
    batch_size: int = 32,
    num_update_iters: int = 32,
    mc_samples: int = 1,
    log_to_wandb=False,
):
    """
    Unified training function that handles both Bayesian and classical losses,
    with support for last-layer-only or full-network fine-tuning.

    Parameters
    ----------
    key : JAX PRNG key
    pretrained_nnet : Pretrained network (with fc layer)
    loss_model : Loss function (Bayesian: IBProbit, etc. or classical: MSE, CrossEntropy)
    data_augmentation : Data augmentation function
    train_ds : Training dataset dict with 'image' and 'label' keys
    test_ds : Test dataset dict with 'image' and 'label' keys
    optimizer : Optimizer (None for CAVI-only with Bayesian + tune_last_layer_only)
    opt_state : Optional initial optimizer state  
    last_layer : LastLayer wrapper (required for classical + tune_last_layer_only)
    tune_last_layer_only : If True, freeze backbone and only optimize last layer
    loss_type : Loss type parameter for Bayesian losses (ignored for classical)
    num_epochs : Number of training epochs
    batch_size : Batch size
    num_update_iters : Number of CAVI/PG iterations per batch (Bayesian only)
    mc_samples : MC samples for stochastic gradient estimation
    log_to_wandb : Whether to log to Weights & Biases

    Returns
    -------
    trained_loss_model : Updated loss model
    trained_nnet : Updated network (or unchanged if tune_last_layer_only)
    opt_state : Final optimizer state
    metrics : Training metrics per epoch
    
    Cases handled:
    1. Bayesian + tune_last_layer_only + optimizer=None: CAVI only, freeze network
    2. Bayesian + tune_last_layer_only + optimizer: CAVI + optimize network
    3. Bayesian + full network: CAVI + optimize network
    4. Classical + tune_last_layer_only: Optimize last layer params
    5. Classical + full network: Optimize full network params
    """
    
    # Detect loss type
    is_bayesian_loss = not isinstance(loss_model, Classical)

    last_layer = pretrained_nnet.fc
    
    # Setup: extract backbone (without fc) and determine what to optimize
    headless_nnet = eqx.nn.inference_mode(
        eqx.tree_at(lambda m: m.fc, pretrained_nnet, eqx.nn.Identity()),
        True,
    )
    nnet_params, nnet_static = eqx.partition(headless_nnet, eqx.is_array)
    
    # Setup parameters to optimize
    if tune_last_layer_only:
        if is_bayesian_loss:
            # Bayesian + last layer: don't optimize network params
            params_to_optimize = None
            params_static = None
        else:
            params_to_optimize, params_static = eqx.partition(last_layer, eqx.is_array)
    else:
        # Full network: optimize network params
        if is_bayesian_loss:
            params_to_optimize = nnet_params
            params_static = nnet_static
        else:
            params_to_optimize, params_static = eqx.partition(pretrained_nnet, eqx.is_array)

    def get_nnet(params):
        if tune_last_layer_only:
            if is_bayesian_loss:
                return headless_nnet
            else:
                ll = eqx.combine(params, params_static)
                return eqx.tree_at(lambda m: m.fc, headless_nnet, ll)
        else:
            return eqx.combine(params, params_static)
    
    #Initialize optimizer
    if params_to_optimize is not None:
        opt_state = optimizer.init(params_to_optimize) if opt_state is None else opt_state
    else:
        opt_state = None
        optimizer = None
    
    loss_params, loss_static = eqx.partition(loss_model, eqx.is_array)
    
    # Batch update function
    def batch_update(current_loss_params, current_params, opt_state, batch_imgs, batch_labels, key):
        aug_key, grad_eval_key = jr.split(key)
        aug_imgs = data_augmentation(batch_imgs, key=aug_key)
        
        # Reconstruct models
        current_loss = eqx.combine(current_loss_params, loss_static)
        if is_bayesian_loss:
            nnet = get_nnet(current_params)
            features = extract_features(nnet, aug_imgs)
            updated_loss = current_loss.update(features, batch_labels, num_iters=num_update_iters)
            updated_loss_params = eqx.filter(updated_loss, eqx.is_array)
        else:
            updated_loss = current_loss
            updated_loss_params = current_loss_params
            
        if optimizer is not None:
            # Compute logits and loss
            def loss_fn(params, images, labels, *args, **kwargs):
                nnet = get_nnet(params)
                logits = extract_features(nnet, images)
                return updated_loss(logits, labels, loss_type=loss_type).mean()
                
            loss_value, grads, opt_state = noisy_value_and_grad(
                    loss_fn, opt_state, current_params, grad_eval_key,
                    aug_imgs, batch_labels, mc_samples=mc_samples
                )
            
            updates, opt_state = optimizer.update(grads, opt_state, current_params)
            updated_params = optax.apply_updates(current_params, updates)
        else:
            updated_params = current_params
            loss_value = updated_loss(features, batch_labels, loss_type=loss_type).mean()
                
        return updated_loss_params, updated_params, opt_state, loss_value

    # Evaluation
    def evaluate(loss_params, net_params, images, labels):
        loss_fn = eqx.combine(loss_params,loss_static)
        nnet = get_nnet(net_params)
        return evaluate_model(data_augmentation, loss_fn, nnet, images, labels, loss_type=loss_type)
    
    # Training epoch
    def epoch_body(carry, epoch_key):
        curr_loss_params, curr_params, opt_state = carry
        ds_size = train_ds["label"].shape[0]
        img_shape = train_ds["image"].shape[1:]
        n_batches = ds_size // batch_size
        
        epoch_key, perm_key, batch_key = jr.split(epoch_key, 3)
        perm = jr.permutation(perm_key, ds_size)
        shuf_imgs = train_ds["image"][perm]
        shuf_labels = train_ds["label"][perm]
        
        img_batches = shuf_imgs[: n_batches * batch_size].reshape(n_batches, batch_size, *img_shape)
        label_batches = shuf_labels[: n_batches * batch_size].reshape(n_batches, batch_size)
        batch_keys = jr.split(batch_key, n_batches)
        
        def batch_body(carry, scans):
            loss_params, params, opt_state = carry
            imgs, labels, key = scans
            updated_loss_params, updated_params, opt_state, loss = batch_update(
                loss_params, params, opt_state, imgs, labels, key
            )
            return (updated_loss_params, updated_params, opt_state), loss
        
        init = (curr_loss_params, curr_params, opt_state)
        (updated_loss_params, updated_params, opt_state), batch_losses = lax.scan(
            batch_body, init, (img_batches, label_batches, batch_keys)
        )
        
        acc, nll, ece = evaluate(updated_loss_params, updated_params, test_ds['image'], test_ds['label'])
        
        metrics = {
            'loss': batch_losses.mean(),
            'acc': acc,
            'ece': ece,
            'nll': nll,
        }
        
        return (updated_loss_params, updated_params, opt_state), metrics
    
    # Main training loop
    epoch_keys = jr.split(key, num_epochs)
    init = (loss_params, params_to_optimize, opt_state)
    (final_loss_params, final_params, final_opt_state), metrics_seq = lax.scan(
        epoch_body, init, epoch_keys
    )
    
    if log_to_wandb and not no_wandb:
        metrics_np = jtu.map(lambda x: onp.asarray(x), metrics_seq)
        for ep in range(num_epochs):
            wandb.log({
                "epoch": ep + 1,
                "loss": float(metrics_np["loss"][ep]),
                "nll": float(metrics_np["nll"][ep]),
                "acc": float(metrics_np["acc"][ep]),
                "ece": float(metrics_np["ece"][ep]),
            })
    
    trained_loss_model = eqx.combine(final_loss_params, loss_static)
    trained_nnet = get_nnet(final_params)
    
    return trained_loss_model, trained_nnet, final_opt_state, metrics_seq

def save_checkpoint_bundle(path, *, models: Mapping[str, eqx.Module], opt_state=None):
    """Serialise one or more Equinox modules plus an optional optimizer state.

    Args:
        path: Destination filepath.
        models: Mapping from user-defined names to Equinox modules.
        opt_state: PyTree optimizer state aligned with one of the models (or None).
    """
    if not models:
        raise ValueError("Expected at least one model to save.")
    filtered_models = {name: eqx.filter(model, eqx.is_array) for name, model in models.items()}
    payload = {"models": filtered_models, "opt_state": opt_state}
    eqx.tree_serialise_leaves(path, payload)


def load_checkpoint_bundle(
    path,
    *,
    model_likes: Mapping[str, eqx.Module],
    optim=None,
    opt_target: Optional[str] = None,
):
    """Deserialise checkpoints written via `save_checkpoint_bundle`.

    Args:
        path: Path to the checkpoint file.
        model_likes: Mapping from names used at save-time to like-structured modules.
        optim: Optional optimizer instance used to rebuild the opt_state tree.
        opt_target: Name of the model whose parameters were optimised. Required when
            `optim` is provided and multiple models were saved.
    Returns:
        restored_models: dict mapping names → reconstructed Equinox modules.
        opt_state: Restored optimizer state (or None).
    """
    if not model_likes:
        raise ValueError("Expected at least one model to load.")

    filtered_models = {name: eqx.filter(model, eqx.is_array) for name, model in model_likes.items()}

    opt_state_like = None
    if optim is not None:
        target = opt_target
        if target is None:
            if len(model_likes) != 1:
                raise ValueError(
                    "opt_target must be specified when loading multiple models with an optimizer."
                )
            target = next(iter(model_likes))
        if target not in filtered_models:
            raise KeyError(f"opt_target '{target}' not present in provided model_likes.")
        opt_state_like = optim.init(filtered_models[target])

    like = {"models": filtered_models, "opt_state": opt_state_like}
    ckpt = eqx.tree_deserialise_leaves(path, like)

    restored_models = {}
    for name, model_like in model_likes.items():
        params = ckpt["models"][name]
        _, static = eqx.partition(model_like, eqx.is_array)
        restored_models[name] = eqx.combine(params, static)

    return restored_models, ckpt["opt_state"]


def save_ivon_checkpoint(path, model, opt_state):
    """Backward-compatible helper for single-model IVON checkpoints."""
    save_checkpoint_bundle(path, models={"model": model}, opt_state=opt_state)


def load_ivon_checkpoint(path, model_like, optim):
    """Backward-compatible helper for single-model IVON checkpoints."""
    restored, opt_state = load_checkpoint_bundle(
        path, model_likes={"model": model_like}, optim=optim, opt_target="model"
    )
    return restored["model"], opt_state
