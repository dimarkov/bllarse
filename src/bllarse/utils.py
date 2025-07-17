import equinox as eqx
import jax.tree_util as jtu
from jax import nn, lax, numpy as jnp, vmap, random as jr
from tensorflow_probability.substrates.jax.stats import expected_calibration_error as compute_ece
from functools import partial
import optax

from blrax.states import ScaleByIvonState
from blrax.utils import noisy_value_and_grad, get_scale, sample_posterior

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
    leafs = jtu.tree_flatten(params)[0]

    return sum([prod(l.shape) for l in leafs])

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
    pretrained_nnet,
    loss_fn,
    optim,
    data_augmentation,
    train_ds,
    test_ds,
    opt_state=None,
    mc_samples=(),
    num_epochs=1,
    batch_size=32,
    ):
    """
    Train a neural network using Equinox and Optax.
    
    Args:
        key: JAX PRNG key
        last_layer: Equinox neural network
        pretrauned_nnet: Equinox neural network
        loss_fn: Eqionox loss function
        optim: Optax (or blrax) optimizer
        data_augmentation: Jax compatible data augmentation function
        train_ds: Training dataset dictionary with 'image' and 'label' keys
        test_ds: Test dataset dictionary with 'image' and 'label' keys
        opt_state: Initial optimizer state, if None it is initiated localy
        num_epochs: Number of epochs to train
        batch_size: Batch size for training
    """
    
    
    if hasattr(loss_fn, 'mu'): # Bayesian case
        params, static = eqx.partition(loss_fn, eqx.is_array)
    else: # Classical case
        params, static = eqx.partition(loss_fn, eqx.is_array)

    opt_state = optim.init(params) if opt_state is None else opt_state  # initialize optimizer state

    def local_loss(params, x, y, *args, **kwargs):
        loss_module = eqx.combine(params, static)
        model = pretrained_nnet
        if hasattr(loss_module, 'mu'): # Bayesian case
            loss, _, new_loss_module = loss_module(model, x, y, *args, **kwargs)
            return loss, new_loss_module
        else: # Classical case
            return loss_module(model, x, y, *args, **kwargs)

    
    # Training step function
    @eqx.filter_jit
    def train_step(loss_fn, params, opt_state, x, y, key):
        keys = jr.split(key, mc_samples)
        (loss_value, new_loss_module), grads = noisy_value_and_grad(loss_fn, opt_state[0], params, x, y, key=keys)
        updates, opt_state = optim.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        if hasattr(new_loss_module, 'mu'):
            params, _ = eqx.partition(new_loss_module, eqx.is_array)
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
            
            batch_images, batch_labels = xs
            key, _key = jr.split(key)
            aug_batch_images = data_augmentation(batch_images, key=_key)
            key, _key = jr.split(key)
            params, opt_state, loss_value = train_step(
                local_loss, params, opt_state, aug_batch_images, batch_labels, _key
            )
            return (params, opt_state, key), loss_value
        
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
        loss_module = eqx.combine(params, static)
        acc, nll, ece = evaluate(
            pretrained_nnet,
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
    (params, final_opt_state), metrics = lax.scan(
        train_epoch,
        init_carry,
        (keys, jnp.arange(num_epochs))
    )
    trained_loss_module = eqx.combine(params, static)
    return trained_loss_module, final_opt_state, metrics

def run_bayesian_training(
    key,
    pretrained_nnet,
    bayesian_model,                 # e.g. IBProbit / IBPolyaGamma / MultinomialPolyaGamma
    data_augmentation,
    train_ds,
    test_ds,
    *,
    num_epochs: int = 1,
    batch_size: int = 32,
    num_update_iters: int = 32,  # CAVI/PG iterations per mini-batch
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

    # ----------------------------------------------------------------
    # 2)  Helper: evaluate on (full) test set
    # ----------------------------------------------------------------
    # Evaluation function
    @eqx.filter_jit
    def evaluate(model, images, labels):
        return evaluate_bayesian_model(
            data_augmentation, model, headless_nnet, images, labels
        )
    # ----------------------------------------------------------------
    # 3)  One mini-batch update (CAVI / PG)
    # ----------------------------------------------------------------
    @eqx.filter_jit
    def batch_update(current_model, batch_imgs, batch_labels, k):
        aug_imgs   = data_augmentation(batch_imgs, key=k)
        feats      = extract_features(headless_nnet, aug_imgs)
        # closed-form VI update; returns *new* model where prior parameters = posterior params at the end of updating
        updated_model = current_model.update(feats, batch_labels,
                                            num_iters=num_update_iters)
        # training NLL (after the update)
        loss = updated_model(feats, batch_labels).mean()
        return updated_model, loss

    # ----------------------------------------------------------------
    # 4)  One training epoch (scan over mini-batches)
    # ----------------------------------------------------------------
    def epoch_body(carry, epoch_key):
        current_loss_fn = carry
        ds_size         = train_ds["label"].shape[0]
        img_shape       = train_ds["image"].shape[1:]
        n_batches       = ds_size // batch_size

        # shuffle dataset
        epoch_key, perm_key, aug_key = jr.split(epoch_key, 3)
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

        # mini-batch keys for stochastic augmentation
        batch_keys = jr.split(aug_key, n_batches)

        # scan over mini-batches
        def batch_body(loss_fn_b, scans):
            imgs, labs, k = scans
            return batch_update(loss_fn_b, imgs, labs, k)

        current_loss_fn, batch_losses = lax.scan(
            batch_body, current_loss_fn, (img_batches, label_batches, batch_keys)
        )

        # validation metrics (after epoch)
        nll, acc, ece, _ = evaluate(current_loss_fn)

        metrics = dict(
            loss=batch_losses.mean(),
            nll=nll,
            acc=acc,
            ece=ece,
        )
        return current_loss_fn, metrics

    # ----------------------------------------------------------------
    # 5)  Main training loop (scan over epochs)
    # ----------------------------------------------------------------
    epoch_keys = jr.split(key, num_epochs)
    trained_loss_fn, metrics_seq = lax.scan(epoch_body, bayesian_model, epoch_keys)

    # `lax.scan` stacks dict‐values; convert to {k:jnp.ndarray}
    stacked_metrics = {
        k: jnp.stack([m[k] for m in metrics_seq]) for k in metrics_seq[0]
    }
    return trained_loss_fn, stacked_metrics
