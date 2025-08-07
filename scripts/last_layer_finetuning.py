import os
import argparse

# do not preallocate memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax.numpy as jnp
import equinox as eqx
import optax
import wandb
import jax.tree_util as jtu

from functools import partial
from math import prod
from datasets import load_dataset

from jax import random as jr, nn, vmap, config

from blrax.optim import ivon
from mlpox.load_models import load_model

from bllarse.losses import MSE, CrossEntropy, IBProbit
from bllarse.layers import LastLayer
from bllarse.utils import run_training, run_bayesian_training, resize_images, augmentdata, get_number_of_parameters, evaluate_model, MEAN_DICT, STD_DICT

config.update("jax_default_matmul_precision", "highest")

def main(args, m_config, o_config):
    dataset = args.dataset
    seed = args.seed
    num_epochs = args.epochs
    num_warmup_epochs = args.warmup
    batch_size = args.batch_size
    save_every = args.save_every
    platform = args.device

    if args.enable_wandb:
        wandb.init(
            project="bllarse_experiments",
            id=args.uid if args.uid else wandb.util.generate_id(),
            group=args.group_id,      # <-- lets you filter by group_id in the UI
            config=dict(              # everything you later want to slice on
                dataset=args.dataset,
                seed=args.seed,
                batch_size=args.batch_size,
                num_vb_iters=args.num_update_iters,
                optimizer=args.optimizer,
                loss_fn=args.loss_function,
                embed_dim=args.embed_dim,
                num_blocks=args.num_blocks,
                pretrained=args.pretrained,
                label_smooth=args.label_smooth,
            ),
            # Entity / mode / tags can be added here if you use them
            reinit=True,
      )
        wandb.define_metric("epoch")                 # x-axis
        wandb.define_metric("loss", step_metric="epoch", summary="min")
        wandb.define_metric("nll",  step_metric="epoch", summary="min")
        wandb.define_metric("acc",  step_metric="epoch", summary="max")
        wandb.define_metric("ece",  step_metric="epoch", summary="min")

    key = jr.PRNGKey(seed)

    # load data
    ds = load_dataset(dataset).with_format("jax")
    train_ds = {'image': ds['train']['img'][:].astype(jnp.float32), 'label': ds['train']['label'][:] }
    test_ds = {'image': ds['test']['img'][:].astype(jnp.float32), 'label': ds['test']['label'][:] }

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
        assert "cavi" in o_config, "Bayesian last layer requires CAVI optimizer"
        key, _key = jr.split(key)
        loss_fn = IBProbit(m_config['embed_dim'], m_config['num_classes'], key=_key)

    # get pretrained network test stats
    if hasattr(loss_fn, "update"):  
        assert "cavi" in o_config, "Bayesian last layer requires CAVI optimizer"
    else:
        nnet = partial(last_layer, pretrained_nnet)
    
    # evaluate original model
    nnet = partial(last_layer, pretrained_nnet)
    _loss_fn = CrossEntropy(0.0, m_config['num_classes'])
    acc, nll, ece = evaluate_model(augdata, _loss_fn, nnet, test_ds['image'], test_ds['label'])
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
    elif 'cavi' in o_config:
        optim = None
        num_update_iters = o_config['cavi']['num_update_iters']
        assert hasattr(loss_fn, "update"), "Using CAVI optimizer requires Bayesian loss function for last layer"
    
    num_params = get_number_of_parameters(pretrained_nnet)
    print(f"Number of parameters of {name} is {num_params}.")

    # run training
    opt_state = None
    trained_loss_fn = loss_fn
    for i in range(num_epochs // save_every):
        key, _key = jr.split(key)
        if hasattr(trained_loss_fn, "update"):     # ==> Bayesian last layer
            trained_loss_fn, metrics = run_bayesian_training(
                _key,
                pretrained_nnet,
                trained_loss_fn,
                augdata,
                train_ds,
                test_ds,
                num_epochs=save_every,
                batch_size=batch_size,
                num_update_iters=num_update_iters,
                log_to_wandb=args.enable_wandb,
            )
            opt_state = None                       # keep interface untouched
        else:                                      # ==> classical optimiser-based
            trained_loss_fn, opt_state, metrics = run_training(
                _key,
                pretrained_nnet,
                trained_loss_fn,
                optim,
                augdata,
                train_ds,
                test_ds,
                opt_state=opt_state,
                mc_samples=mc_samples,
                num_epochs=save_every,
                batch_size=batch_size,
            )


        #TODO: save model checkpoint, opt_state, and test metrics
        to_save = {"loss_fn": trained_loss_fn, "opt_state": opt_state, "metrics": metrics}
        vals = jtu.tree_map(lambda x: x[-1], metrics)

        if args.enable_wandb:
            wandb.log({"epoch": (i + 1) * save_every, **vals})
        else: 
            # only print to console if not logging to wandb
            print(i, [(name, f'{vals[name].item():.3f}') for name in vals])


def build_argparser():
    parser = argparse.ArgumentParser(description="last layer finetuning")
    parser.add_argument("-o", "--optimizer", choices=['ivon', 'lion', "cavi"], default='cavi', type=str)
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
    parser.add_argument("--pretrained", nargs='?', choices=['in21k', 'in21k_cifar'], default='in21k_cifar', type=str)
    parser.add_argument("--reinitialize", action="store_true")
    parser.add_argument("--enable_wandb", "--enable-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--group-id", type=str, default="vb_sweep", help="Put all runs of this sweep in the same W&B group")
    parser.add_argument("--uid", type=str, default=None, help="Unique identifier for the W&B run. If not provided, a random one will be generated.")
    return parser

def build_configs(args):
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
    if args.optimizer == 'cavi':
        opt_config = {'cavi': {'num_update_iters': args.num_update_iters}}
    
    return model_config, opt_config

if __name__ == '__main__':
    parser = build_argparser()
    args = parser.parse_args()
    config.update("jax_platform_name", args.device)

    model_conf, opt_conf = build_configs(args)

    main(args, model_conf, opt_conf)
