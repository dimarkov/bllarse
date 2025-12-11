import os
import argparse
import warnings

# do not preallocate memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax.numpy as jnp
import equinox as eqx
import optax
try:
    import wandb
    no_wandb = False
except:
    print('wandb not installed')
    no_wandb = True
import jax.tree_util as jtu

from functools import partial
from datasets import load_dataset

from jax import random as jr, config

from blrax.optim import ivon
from mlpox.load_models import load_model

from bllarse.losses import MSE, CrossEntropy, IBProbit
from bllarse.utils import (
    run_training,
    resize_images,
    augmentdata,
    get_number_of_parameters,
    evaluate_model,
    MEAN_DICT,
    STD_DICT,
    save_ivon_checkpoint,
    save_checkpoint_bundle,
)

config.update("jax_default_matmul_precision", "highest")


def _build_wandb_config(args, o_config):
    """Return shared run metadata plus optimizer-specific hyperparameters."""
    config_dict = dict(
        dataset=args.dataset,
        seed=args.seed,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        loss_fn=args.loss_fn,
        embed_dim=args.embed_dim,
        num_blocks=args.num_blocks,
        pretrained=args.pretrained,
        label_smooth=args.label_smooth,
        epochs=args.epochs,
        nodataaug=args.nodataaug,
        tune_mode=args.tune_mode,
        sequential_update=args.sequential_update,
        reset_loss_per_epoch=args.reset_loss_per_epoch,
    )

    if args.loss_fn == 'IBProbit':
        config_dict.update(dict(num_vb_iters=args.num_update_iters))

    if 'ivon' in o_config:
        config_dict.update(
            dict(
                ivon_weight_decay=args.ivon_weight_decay,
                ivon_hess_init=args.ivon_hess_init,
                mc_samples=args.mc_samples,
            )
        )
    else:
        # Lion and AdamW share the same hyperparameters.
        config_dict.update(
            dict(
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
            )
        )

    return config_dict


def main(args, m_config, o_config):
    dataset = args.dataset
    seed = args.seed
    num_epochs = args.epochs
    batch_size = args.batch_size
    save_every = args.save_every
    platform = args.device
    tune_last_layer_only = (args.tune_mode == 'last_layer')
    use_ivon = 'ivon' in o_config
    log_ivon_checkpoints = args.log_checkpoints and args.enable_wandb and not no_wandb and use_ivon
    
    # Validate arguments
    if args.sequential_update:
        if args.loss_fn != 'IBProbit':
            raise ValueError("--sequential-update requires --loss-fn IBProbit")
        if args.tune_mode != 'full_network':
            raise ValueError("--sequential-update requires --tune-mode full_network")
    
    if args.reset_loss_per_epoch and args.loss_fn != 'IBProbit':
        raise ValueError("--reset-loss-per-epoch requires --loss-fn IBProbit")
    
    if args.loss_fn == 'IBProbit' and tune_last_layer_only:
        warnings.warn(
            "When using IBProbit with --tune-mode last_layer, the --optimizer argument and relevant hyperparameters will be ignored."
        )

    if args.enable_wandb and not no_wandb:
        wandb.init(
            project="bllarse_experiments",
            id=args.uid if args.uid else wandb.util.generate_id(),
            group=args.group_id,
            config=_build_wandb_config(args, o_config),
            reinit=True,
        )
        wandb.define_metric("epoch")
        wandb.define_metric("loss", step_metric="epoch", summary="min")
        wandb.define_metric("nll", step_metric="epoch", summary="min")
        wandb.define_metric("acc", step_metric="epoch", summary="max")
        wandb.define_metric("ece", step_metric="epoch", summary="min")

    key = jr.PRNGKey(seed)

    # load data
    ds = load_dataset(dataset).with_format("jax")
    label_key = 'fine_label' if dataset == 'cifar100' else 'label'
    train_ds = {'image': ds['train']['img'][:].astype(jnp.float32), 'label': ds['train'][label_key][:]}
    test_ds = {'image': ds['test']['img'][:].astype(jnp.float32), 'label': ds['test'][label_key][:]}

    datasize = ds['train'].num_rows
    num_iters = num_epochs * datasize // batch_size

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
        last_layer = eqx.nn.Linear(
                m_config['embed_dim'],
                m_config['num_classes'],
                key=_key
            )

    elif args.pretrained == 'in21k_cifar':
        name = name + '_' + args.dataset
        pretrained_nnet = eqx.nn.inference_mode(load_model(name), True)
        if args.reinitialize:
            key, _key = jr.split(key)
            last_layer = eqx.nn.Linear(
                    m_config['embed_dim'],
                    m_config['num_classes'],
                    key=_key
                )
        else:
            last_layer = pretrained_nnet.fc

    # specify loss function
    if args.loss_fn == 'MSE':
        loss_fn = MSE(m_config['num_classes'])
    elif args.loss_fn == 'CrossEntropy':
        loss_fn = CrossEntropy(args.label_smooth, m_config['num_classes'])
    elif args.loss_fn == 'IBProbit':
        key, _key = jr.split(key)
        loss_fn = IBProbit(m_config['embed_dim'], m_config['num_classes'], key=_key)

    if args.nodataaug:
        _augdata = lambda img, key=None, **kwargs: augdata(img, key=None, **kwargs)
    else:
        _augdata = augdata

    # evaluate original model
    nnet = eqx.tree_at(lambda m: m.fc, pretrained_nnet, last_layer)
    _loss_fn = CrossEntropy(0.0, m_config['num_classes'])
    acc, nll, ece = evaluate_model(_augdata, _loss_fn, nnet, test_ds['image'], test_ds['label'])
    print(f'pre-trained test acc={acc:.3f}, ece={ece:.3f}, nll={nll:.3f}')

    # set optimizer
    if 'lion' in o_config:
        optim = optax.lion(**o_config['lion'])
        mc_samples = 1
    elif 'adamw' in o_config:
        optim = optax.adamw(**o_config['adamw'])
        mc_samples = 1
    elif 'ivon' in o_config:
        lr_conf = o_config['lr']
        lr_conf['decay_steps'] = num_iters
        lr_conf['warmup_steps'] = num_iters // 10
        lr_schd = optax.schedules.warmup_cosine_decay_schedule(**lr_conf)
        conf = o_config['ivon']
        mc_samples = conf.pop('mc_samples')
        conf['ess'] = datasize
        optim = ivon(lr_schd, **conf)

    num_update_iters = args.num_update_iters
    num_params = get_number_of_parameters(pretrained_nnet)
    print('Loss Function:', args.loss_fn, 'Optimizer:', args.optimizer, 'Finetune Mode:', args.tune_mode)
    print(f"Number of parameters of {name} is {num_params}.")

    # run training
    opt_state = None
    trained_loss_fn = loss_fn
    trained_model = nnet
    
    for i in range(num_epochs // save_every):
        key, _key = jr.split(key)
        trained_loss_fn, trained_model, opt_state, metrics = run_training(
            _key,
            trained_model,
            trained_loss_fn,
            _augdata,
            train_ds,
            test_ds,
            optimizer=optim,
            opt_state=opt_state,
            tune_last_layer_only=tune_last_layer_only,
            loss_type=3,
            num_epochs=save_every,
            batch_size=batch_size,
            num_update_iters=num_update_iters,
            mc_samples=mc_samples,
            log_to_wandb=args.enable_wandb,
            sequential_update=args.sequential_update,
            reset_loss_per_epoch=args.reset_loss_per_epoch,
        )

        # Save checkpoint
        is_bayesian = isinstance(trained_loss_fn, IBProbit)
        if is_bayesian:
            to_save = {"loss_fn": trained_loss_fn, "nnet": trained_model, "opt_state": opt_state, "metrics": metrics}
        else:
            if tune_last_layer_only:
                to_save = {"last_layer": trained_model, "opt_state": opt_state, "metrics": metrics}
            else:
                to_save = {"nnet": trained_model, "opt_state": opt_state, "metrics": metrics}

        vals = jtu.tree_map(lambda x: x[-1], metrics)

        if args.enable_wandb:
            wandb.log({"epoch": (i + 1) * save_every, **vals})
        else:
            # only print to console if not logging to wandb
            print(i, [(name, f'{vals[name].item():.3f}') for name in vals])
        
        if log_ivon_checkpoints:
            epoch = (i + 1) * save_every
            ckpt_name = f"ivon_epoch_{epoch}.eqx"
            ckpt_path = os.path.join(wandb.run.dir, ckpt_name)

            if is_bayesian and not tune_last_layer_only:
                # Full network + Bayesian head
                save_checkpoint_bundle(
                    ckpt_path,
                    models={"backbone": trained_model, "bayes_head": trained_loss_fn},
                    opt_state=opt_state,
                )
            elif tune_last_layer_only and not is_bayesian:
                # Last layer only (classical)
                save_ivon_checkpoint(ckpt_path, trained_model, opt_state)
            elif not tune_last_layer_only and not is_bayesian:
                # Full network (classical)
                save_ivon_checkpoint(ckpt_path, trained_model, opt_state)
            else:
                # Bayesian + last layer only
                save_checkpoint_bundle(
                    ckpt_path,
                    models={"bayes_head": trained_loss_fn},
                    opt_state=opt_state,
                )

            # Track the single .eqx file as a W&B artifact
            artifact = wandb.Artifact(f"{wandb.run.id}-ivon-{epoch}", type="model")
            artifact.add_file(ckpt_path, name=ckpt_name)
            wandb.run.log_artifact(artifact)


def build_argparser():
    parser = argparse.ArgumentParser(description="Finetuning script")
    parser.add_argument("-o", "--optimizer", choices=['ivon', 'lion', "adamw"], default='adamw', type=str)
    parser.add_argument("--loss-fn", choices=['MSE', 'CrossEntropy', 'IBProbit'], default='CrossEntropy', type=str)
    parser.add_argument("--tune-mode", choices=['last_layer', 'full_network'], default='last_layer', type=str,
                       help='Whether to tune only the last layer or the full network')
    parser.add_argument('--num-blocks', choices=[6, 12], default=6, type=int, help='Allowed number of blocks/layers')
    parser.add_argument('--embed-dim', choices=[512, 1024], default=512, type=int, help='Allowed embedding dimensions')
    parser.add_argument("--device", nargs='?', default='gpu', type=str)
    parser.add_argument("--seed", nargs='?', default=137, type=int)
    parser.add_argument("-ds", "--dataset", nargs='?', default='cifar10', type=str)
    parser.add_argument("--save-every", nargs='?', default=10, type=int)
    parser.add_argument("-e", "--epochs", nargs='?', default=100, type=int)
    parser.add_argument("-bs", "--batch-size", nargs='?', default=64, type=int)
    parser.add_argument("-ls", "--label-smooth", nargs='?', default=0.0, type=float)
    parser.add_argument("-lr", "--learning-rate", nargs='?', default=1e-3, type=float, 
                       help='Learning rate for AdamW or Lion optimizers')
    parser.add_argument("-wd", "--weight-decay", nargs='?', default=1e-2, type=float, 
                       help='Weight decay for AdamW or Lion optimizers')
    parser.add_argument("-mc", "--mc-samples", nargs='?', default=1, type=int)
    parser.add_argument("--ivon-peak-lr", nargs='?', default=1e-2, type=float)
    parser.add_argument("--ivon-weight-decay", nargs='?', default=1e-6, type=float, 
                       help='Weight decay for IVON optimizer')
    parser.add_argument("--ivon-hess-init", nargs='?', default=1.0, type=float, 
                       help='Hessian initialisation scale for IVON optimizer')
    parser.add_argument("--ivon-b2", nargs='?', default=0.999, type=float, 
                       help='Beta2 parameter for IVON optimizer')
    parser.add_argument("--num-update-iters", nargs='?', default=16, type=int, 
                       help='Number of CAVI iterations per mini-batch for Bayesian last layer')
    parser.add_argument("--pretrained", nargs='?', choices=['in21k', 'in21k_cifar'], default='in21k_cifar', type=str)
    parser.add_argument("--reinitialize", action="store_true")
    parser.add_argument("--nodataaug", action="store_true")
    parser.add_argument("--sequential-update", action="store_true",
                       help='Enable two-pass training per epoch: first update loss model, then update network (requires IBProbit + full_network)')
    parser.add_argument("--reset-loss-per-epoch", action="store_true",
                       help='Reset loss model parameters at the start of each epoch (requires IBProbit)')
    parser.add_argument("--enable_wandb", "--enable-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--log-checkpoints", "--log_checkpoints", action="store_true", help="Log IVON checkpoints to W&B/locally")
    parser.add_argument("--group-id", type=str, default="finetuning_test", help="Put all runs of this sweep in the same W&B group")
    parser.add_argument("--uid", type=str, default=None, help="Unique identifier for the W&B run. If not provided, a random one will be generated.")
    return parser


def build_configs(args):
    num_classes = 1
    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100

    model_config = {
        'img_size': 64,
        'in_chans': 3,
        'embed_dim': args.embed_dim,
        'num_blocks': args.num_blocks,
        'num_classes': num_classes
    }

    if args.optimizer == 'lion':
        opt_config = {'lion': {'learning_rate': args.learning_rate, 'weight_decay': args.weight_decay}}
    elif args.optimizer == 'adamw':
        opt_config = {'adamw': {'learning_rate': args.learning_rate, 'weight_decay': args.weight_decay}}
    elif args.optimizer == 'ivon':
        opt_config = {
            'ivon': {
                'weight_decay': args.ivon_weight_decay,
                'hess_init': args.ivon_hess_init,
                'mc_samples': args.mc_samples,
                'b2': args.ivon_b2,
                'clip_radius': 1e3
            },
            'lr': {
                'init_value': args.ivon_peak_lr / 20,
                'peak_value': args.ivon_peak_lr,
                'end_value': args.ivon_peak_lr / 10
            }
        }

    return model_config, opt_config


if __name__ == '__main__':
    parser = build_argparser()
    args = parser.parse_args()
    config.update("jax_platform_name", args.device)

    model_conf, opt_conf = build_configs(args)

    main(args, model_conf, opt_conf)
