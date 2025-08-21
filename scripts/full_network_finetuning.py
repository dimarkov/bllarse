import os
import argparse

# do not prealocate memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax.numpy as jnp
import equinox as eqx
import optax
import augmax
import jax.tree_util as jtu

from functools import partial
from math import prod
from datasets import load_dataset

from jax import random as jr, nn, vmap, config

from blrax.optim import ivon
from mlpox.load_models import load_model

from bllarse.losses import MSE, CrossEntropy, IBProbit
from bllarse.layers import LastLayer
from bllarse.utils import run_training, run_bayesian_training, resize_images, augmentdata, get_number_of_parameters, evaluate_model, evaluate_bayesian_model, MEAN_DICT, STD_DICT

config.update("jax_default_matmul_precision", "highest")

def main(args, m_config, o_config):
    dataset = args.dataset
    seed = args.seed
    num_epochs = args.epochs
    num_warmup_epochs = args.warmup
    batch_size = args.batch_size
    save_every = args.save_every
    platform = args.device

    key = jr.PRNGKey(seed)

    # load data
    ds = load_dataset(dataset).with_format("jax")
    label_key = 'fine_label' if dataset == 'cifar100' else 'label'
    train_ds = {'image': ds['train']['img'][:].astype(jnp.float32), 'label': ds['train'][label_key][:] }
    test_ds = {'image': ds['test']['img'][:].astype(jnp.float32), 'label': ds['test'][label_key][:] }

    datasize = ds['train'].num_rows
    num_iters = num_epochs * datasize // batch_size
    warmup_steps = num_warmup_epochs * datasize // batch_size

    # define data augmentation
    train_ds['image'] = resize_images(train_ds['image'], m_config['img_size'])
    test_ds['image'] = resize_images(test_ds['image'], m_config['img_size'])

    mean = MEAN_DICT[args.dataset]
    std = STD_DICT[args.dataset]
    augdata = partial(augmentdata, mean=mean, std=std)
        
    # load model
    name = f"B_{m_config['num_blocks']}-Wi_{m_config['embed_dim']}_res_64_in21k"
    if args.pretrained == 'in21k':
        pretrained_nnet = eqx.nn.inference_mode(load_model(name), True)
        key, _key = jr.split(key)
        last_layer = LastLayer(
            eqx.nn.Linear(
                m_config['embed_dim'],
                m_config['num_classes'],
                key=_key
            )
        )
    
    elif args.pretrained == 'in21k_cifar':
        name = name + '_' + args.dataset
        pretrained_nnet = eqx.nn.inference_mode(load_model(name), True)
        if args.reinitialize:
            key, _key = jr.split(key)
            last_layer = LastLayer(
                eqx.nn.Linear(
                    m_config['embed_dim'],
                    m_config['num_classes'],
                    key=_key
                )
            )
        else:
            last_layer = LastLayer(pretrained_nnet.fc)

    # specify loss function
    if args.loss_function == 'MSE':
        loss_fn = MSE(m_config['num_classes'])
    if args.loss_function == 'CrossEntropy':
        loss_fn = CrossEntropy(args.label_smooth, m_config['num_classes'])
    if args.loss_function == 'IBProbit':
        key, _key = jr.split(key)
        loss_fn = IBProbit(m_config['embed_dim'], m_config['num_classes'], key=_key)

    if args.nodataaug:
        _augdata = lambda img, key=None, **kwargs: augdata(img, key=None, **kwargs)
    else:
        _augdata = augdata
    
    # evaluate original model
    nnet = partial(last_layer, pretrained_nnet)
    _loss_fn = CrossEntropy(0.0, m_config['num_classes'])
    acc, nll, ece = evaluate_model(_augdata, _loss_fn, nnet, test_ds['image'], test_ds['label'])
    print(f'pre-trained test acc={acc:.3f}, ece={ece:.3f}, nll={nll:.3f}')

    # set optimizer
    if 'lion' in o_config:
        optim = optax.lion(**o_config['lion'])
        mc_samples = ()
    elif 'ivon' in o_config:
        lr_conf = o_config['lr']
        lr_conf['decay_steps'] = num_iters
        lr_conf['warmup_steps'] = num_iters // 10
        lr_schd = optax.schedules.warmup_cosine_decay_schedule(
            **lr_conf
        )
        key, _key = jr.split(key)
        conf = o_config['ivon']
        conf['num_data'] = num_epochs * datasize
        optim = ivon(_key, lr_schd, **conf)
        mc_samples = o_config['ivon']['mc_samples']

    num_update_iters = args.num_update_iters  
    num_params = get_number_of_parameters(pretrained_nnet)
    print(f"Number of parameters of {name} is {num_params}.")

    # run training
    opt_state = None
    trained_loss_fn = loss_fn
    trained_nnet = pretrained_nnet
    for i in range(num_epochs // save_every):
        key, _key = jr.split(key)
        trained_loss_fn, trained_nnet, opt_state, metrics = run_bayesian_training(
            _key,
            trained_nnet,
            trained_loss_fn,
            _augdata,
            train_ds,
            test_ds,
            optimizer=optim,
            opt_state=opt_state,
            loss_type=3,
            num_epochs=save_every,
            batch_size=batch_size,
            num_update_iters=num_update_iters,
        )

        #TODO: save model checkpoint, opt_state, and test metrics
        to_save = {"loss_fn": trained_loss_fn, "opt_state": opt_state, "metrics": metrics}
        vals = jtu.tree_map(lambda x: x[-1], metrics)
        print(i, [(name, f'{vals[name].item():.3f}') for name in vals])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="deep MLP training")
    parser.add_argument("-o", "--optimizer", choices=['ivon', 'lion'], default='lion', type=str)
    parser.add_argument("--loss-function", choices=['MSE', 'CrossEntropy', 'IBProbit'], default='IBProbit', type=str)
    parser.add_argument('--num-blocks', choices=[6, 12], default=6, type=int, help='Allowed number of blocks/layers')
    parser.add_argument('--embed-dim', choices=[512, 1024], default=512, type=int, help='Allowed embedding dimensions')
    parser.add_argument("--device", nargs='?', default='gpu', type=str)
    parser.add_argument("--seed", nargs='?', default=137, type=int)
    parser.add_argument("-ds", "--dataset", nargs='?', default='cifar10', type=str)
    parser.add_argument("--save-every", nargs='?', default=10, type=int)
    parser.add_argument("-e", "--epochs", nargs='?', default=100, type=int)
    parser.add_argument("-w", "--warmup", nargs='?', default=10, type=int)
    parser.add_argument("-bs", "--batch-size", nargs='?', default=64, type=int)
    parser.add_argument("-ls", "--label-smooth", nargs='?', default=0.0, type=float)
    parser.add_argument("-mc", "--mc-samples", nargs='?', default=1, type=int)
    parser.add_argument("--num-update-iters", nargs='?', default=32, type=int, help='Number of CAVI iterations per mini-batch for Bayesian last layer')
    parser.add_argument("--pretrained", nargs='?', choices=['in21k', 'in21k_cifar'], default='in21k', type=str)
    parser.add_argument("--reinitialize", action="store_true")
    parser.add_argument("--nodataaug", action="store_true")

    args = parser.parse_args()
    config.update("jax_platform_name", args.device)

    num_classes = 1
    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    
    # specify configurations for different neural networks
    # for now this is shared across both types, but it should 
    # become obsolete if we simply load pretrained models
    model_config = {
        'img_size': 64,
        'in_chans': 3,
        'embed_dim': args.embed_dim,
        'num_blocks': args.num_blocks,
        'num_classes': num_classes
        }

    if args.optimizer == 'lion':
        opt_config = {'lion': {'learning_rate': 5e-5, 'weight_decay': 1e-2}}
    if args.optimizer == 'ivon':
        opt_config = {
            'ivon': {'s0': 1., 'h0': 1., 'mc_samples': args.mc_samples, 'clip_radius': 1e3},
            'lr': {
                'init_value': 5e-4,
                'peak_value': 1e-2,
                'end_value': 5e-5
            }
        }
        
    main(args, model_config, opt_config)
