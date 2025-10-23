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

from blrax.utils import noisy_value_and_grad

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

def evaluate_model(data_augmentation, loss_fn, nnet, images, labels):
    aug_images = data_augmentation(images, key=None)
    loss, logits = loss_fn(nnet, aug_images, labels, with_logits=True)
    predictions = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(predictions == labels)
    ece = compute_ece(20, logits=logits, labels_true=labels, labels_predicted=predictions)
    return acc, loss, ece

def evaluate_bayesian_model(data_augmentation, loss_fn, pretrained_nnet, images, labels):
    """
    Bayesian version of `evaluate_model`. 
    #TODO: Re-write `evaluate_model` and `evaluate_bayesian_model()` into a single function
    with branching on `if any(hasattr(loss_fn, attr) for attr in ["eta", "mu"])`:
    """
    aug_images = data_augmentation(images, key=None)
    feats    = extract_features(pretrained_nnet, aug_images)
    loss, logits = loss_fn(feats, labels, with_logits=True)
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
    bayesian_loss_model,                 # e.g. IBProbit / IBPolyaGamma / MultinomialPolyaGamma / BayesianLinearRegression
    data_augmentation,
    train_ds,
    test_ds,
    *,  
    optimizer = None,
    opt_state = None,
    loss_type: int = 3,
    num_epochs: int = 1,
    batch_size: int = 32,
    num_update_iters: int = 32,  # CAVI/PG iterations per mini-batch
    mc_samples: int = 1,
    log_to_wandb=False,
):
    """
    Variational Bayes fine-tuning of the last layer (`loss_fn`) while the
    pretrained feature extractor stays frozen.

    Returns
    -------
    trained_loss_fn : updated Bayesian layer (same type as `loss_fn`)
    metrics         : dict with per-epoch jnp arrays (shape (num_epochs,))
    """

    # ----------------------------------------------------------------
    # 1)  Freeze the feature extractor (remove its final `fc`)
    # ----------------------------------------------------------------
    headless_nnet = eqx.nn.inference_mode(
        eqx.tree_at(lambda m: m.fc, pretrained_nnet, eqx.nn.Identity()),
        True,            # turn off dropout / BN statistics
    )

    nnet_params, nnet_static = eqx.partition(headless_nnet, eqx.is_array)
    if opt_state is None:
        opt_state = None if optimizer is None else optimizer.init(nnet_params) 

    # ----------------------------------------------------------------
    # 2)  Helper: evaluate on (full) test set
    # ----------------------------------------------------------------
    # Evaluation function
    def evaluate(bll_model, nnet_params, images, labels):
        nnet = eqx.combine(nnet_params, nnet_static)
        return evaluate_bayesian_model(
            data_augmentation, partial(bll_model, loss_type=loss_type), nnet, images, labels
        )
    
    # ----------------------------------------------------------------
    # 3)  One mini-batch update (CAVI / PG)
    # ----------------------------------------------------------------
    def batch_update(current_loss_model, nnet_params, opt_state, batch_imgs, batch_labels, key):

        # split batch-specific key into a key for data augmentation and a key for gradient evaluation
        aug_key, grad_eval_key = jr.split(key)
        aug_imgs   = data_augmentation(batch_imgs, key=aug_key)

        nnet = eqx.combine(nnet_params, nnet_static)
        feats = extract_features(nnet, aug_imgs)
        updated_loss_model = current_loss_model.update(
            feats,
            batch_labels,
            num_iters=num_update_iters
        )

        # training NLL (after the BLL update)
        if optimizer is None:
            loss = updated_loss_model(feats, batch_labels, loss_type=loss_type).mean()
        else:
            def loss_fn(params, images, labels, *args):
                nnet = eqx.combine(params, nnet_static)
                feats = extract_features(nnet, images)
                return updated_loss_model(feats, labels, loss_type=loss_type).mean()
            
            loss, grads, opt_state = \
                noisy_value_and_grad(
                    loss_fn,
                    opt_state,
                    nnet_params,
                    grad_eval_key,
                    aug_imgs,
                    batch_labels,
                    mc_samples=mc_samples,
                )

            updates, opt_state = optimizer.update(grads, opt_state, nnet_params)
            nnet_params = optax.apply_updates(nnet_params, updates)

        return updated_loss_model, nnet_params, opt_state, jnp.mean(loss)

    # ----------------------------------------------------------------
    # 4)  One training epoch (scan over mini-batches)
    # ----------------------------------------------------------------

    loss_params, loss_static = eqx.partition(bayesian_loss_model, eqx.is_array)
    def epoch_body(carry, epoch_key):
        current_loss_params, current_nnet_params, opt_state = carry
        ds_size         = train_ds["label"].shape[0]
        img_shape       = train_ds["image"].shape[1:]
        n_batches       = ds_size // batch_size

        # shuffle dataset
        epoch_key, perm_key, batch_key = jr.split(epoch_key, 3)
        perm        = jr.permutation(perm_key, ds_size)
        shuf_imgs   = train_ds["image"][perm]
        shuf_labels = train_ds["label"][perm]

        # reshape into (n_batches, batch, …)
        img_batches   = shuf_imgs[: n_batches * batch_size].reshape(
            n_batches, batch_size, *img_shape
        )
        label_batches = shuf_labels[: n_batches * batch_size].reshape(
            n_batches, batch_size
        )

        # mini-batch keys for stochastic augmentation and gradient evaluation
        batch_keys = jr.split(batch_key, n_batches)

        # scan over mini-batches
        def batch_body(carry, scans):
            loss_params, nnet_params, opt_state = carry
            loss_model = eqx.combine(loss_params, loss_static)
            imgs, labels, key = scans
            updated_loss_model, updated_nnet_params, opt_state, loss = batch_update(
                loss_model, nnet_params, opt_state, imgs, labels, key
            )
            updated_loss_params = eqx.filter(updated_loss_model, eqx.is_array)
            return (updated_loss_params, updated_nnet_params, opt_state), loss
        
        init = (current_loss_params, current_nnet_params, opt_state)
        (updated_loss_params, updated_nnet_params, opt_state), batch_losses = lax.scan(
            batch_body, 
            init, 
            (img_batches, label_batches, batch_keys)
        )

        updated_model = eqx.combine(updated_loss_params, loss_static)

        # validation metrics (after epoch)
        acc, nll, ece = evaluate(
            updated_model,
            updated_nnet_params,
            test_ds['image'],
            test_ds['label'],
        )

        metrics = {
            'loss': batch_losses.mean(),
            'acc': acc,
            'ece': ece,
            'nll': nll,
        }

        return (updated_loss_params, updated_nnet_params, opt_state), metrics

    # ----------------------------------------------------------------
    # 5)  Main training loop (scan over epochs)
    # ----------------------------------------------------------------
    epoch_keys = jr.split(key, num_epochs)
    init = (loss_params, nnet_params, opt_state)
    (updated_loss_params, updated_nnet_params, opt_state), metrics_seq = lax.scan(epoch_body, init, epoch_keys)

    trained_loss_model = eqx.combine(updated_loss_params, loss_static)
    trained_nnet = eqx.combine(nnet_params, nnet_static)

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

    return trained_loss_model, trained_nnet, opt_state, metrics_seq

def save_ivon_checkpoint(path, last_layer, opt_state):
    """Save only array leaves: params from last_layer + opt_state (PyTrees)."""
    params = eqx.filter(last_layer, eqx.is_array)  # array-only view of the module
    ckpt = {"params": params, "opt_state": opt_state}
    eqx.tree_serialise_leaves(path, ckpt)

def load_ivon_checkpoint(path, last_layer_like, optim):
    """Load back into a like-structured PyTree and reassemble the module.

    last_layer_like: same structure/hparams as your LastLayer (untrained or current).
    optim: the Optax/IVON optimizer to reconstruct opt_state shape if needed.
    """
    # Prepare skeletons ( the ‘like’ PyTrees) for deserialisation:
    params_like = eqx.filter(last_layer_like, eqx.is_array)
    opt_state_like = None if optim is None else optim.init(params_like)
    like = {"params": params_like, "opt_state": opt_state_like}

    ckpt = eqx.tree_deserialise_leaves(path, like) 

    # Recombine arrays + statics to get a full LastLayer module back.
    # (params, static) = partition; then combine(restored_params, static)
    _, static = eqx.partition(last_layer_like, eqx.is_array)
    restored_last_layer = eqx.combine(ckpt["params"], static)
    restored_opt_state = ckpt["opt_state"]
    return restored_last_layer, restored_opt_state
