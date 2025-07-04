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
    nll, logits = loss_fn(nnet, aug_images, labels, with_logits=True)
    
    predictions = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(predictions == labels)
    ece = compute_ece(20, logits=logits, labels_true=labels, labels_predicted=predictions)
    return acc, nll, ece

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
    
    
    params, static = eqx.partition(last_layer, eqx.is_array)  # split model into params and static fields
    opt_state = optim.init(params) if opt_state is None else opt_state  # initialize optimizer state

    def local_loss(params, x, y, *args, **kwargs):
        ll = eqx.combine(params, static)
        return loss_fn(partial(ll, pretrained_nnet), x, y)

    
    # Training step function
    @eqx.filter_jit
    def train_step(loss_fn, params, opt_state, x, y, key):
        keys = jr.split(key, mc_samples)
        loss_value, grads = noisy_value_and_grad(loss_fn, opt_state[0], params, x, y, key=keys)
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
    (params, final_opt_state), metrics = lax.scan(
        train_epoch,
        init_carry,
        (keys, jnp.arange(num_epochs))
    )
    trained_model = eqx.combine(params, static)
    return trained_model, final_opt_state, metrics