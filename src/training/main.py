import glob
import logging
import os
import random
import re
import subprocess
import sys
from datetime import datetime

import numpy as np
import torch
from loguru import logger
from torch import optim
from torch.cuda.amp import GradScaler

from training.data import get_data
from training.distributed import broadcast_object, init_distributed_device, is_master
from training.logger import setup_logging
from training.losses import create_loss
from training.models import create_model_and_transforms
from training.params import parse_args
from training.scheduler import const_lr, const_lr_cooldown, cosine_lr
from training.train import evaluate, train_one_epoch

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote: bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(
            ["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [
            os.path.join(path,
                         x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]
        ]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def main(args):
    args = parse_args(args)
    if args.pretrained and args.pretrained == "openai":  # or siglip?
        args.model_cache_dir = os.path.expanduser("~/.cache/torch/hub/OpenCLIP/openai")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = '-'.join(
            [
                date_str,
                f"model_{model_name_safe}",
                f"lr_{args.lr}",
                f"b_{args.batch_size}",
                f"j_{args.workers}",
                f"p_{args.precision}",
            ]
        )

    resume_latest = args.resume == 'latest'
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not resume_latest:
            print("Error. Experiment already exists. Use --name {} to specify a new experiment.")
            return -1

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        args.tensorboard_path = os.path.join(
            log_base_path, "tensorboard"
        ) if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    random_seed(args.seed, 0)

    model, preprocess_train, preprocess_val, tokenizer = create_model_and_transforms(args, device)

    dist_model = None
    args.distill = dist_model is not None
    random_seed(args.seed, args.rank)

    if is_master(args):
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                # logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")
        logger.info(f"All params are saved to: {params_file}")

    # create optimizer and scaler
    optimizer = None
    scaler = None

    if args.train_data or args.dataset_type == "synthetic":
        assert not args.trace, 'Cannot train with traced model'
        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n  # noqa
        include = lambda n, p: not exclude(n, p)  # noqa

        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
        params_list = [
            {
                "params": gain_or_bias_params,
                "weight_decay": 0.
            },
            {
                "params": rest_params,
                "weight_decay": args.wd
            },
        ]

        optimizer = optim.AdamW(
            params_list,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        scaler = GradScaler() if args.precision == "amp" else None

    start_epoch = 0

    # initialize datasets
    data = get_data(
        args,
        (preprocess_train, preprocess_val),
        epoch=start_epoch,
        tokenizer=tokenizer,
    )
    assert len(data) or args.test_full_comp, 'At least one train or eval dataset must be specified.'

    # create scheduler if train
    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
        if args.lr_scheduler == "cosine":
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const":
            scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const-cooldown":
            assert args.epochs_cooldown is not None,\
                "Please specify the number of cooldown epochs for this lr schedule."
            cooldown_steps = (
                data["train"].dataloader.num_batches // args.accum_freq
            ) * args.epochs_cooldown
            scheduler = const_lr_cooldown(
                optimizer, args.lr, args.warmup, total_steps, cooldown_steps,
                args.lr_cooldown_power, args.lr_cooldown_end
            )
        else:
            logging.error(
                f"Unknown scheduler, {args.lr_scheduler}. Available options are: "
                "cosine, const, const-cooldown."
            )
            exit(1)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples

        # you will have to configure this for your project!
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume='auto' if args.resume == "latest" else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    # For compatibility, we save state_dict() of the original model, which shares the
    # weights without the prefix.
    original_model = model
    if args.torchcompile:
        logging.info('Compiling model...')
        model = torch.compile(original_model)

    if 'train' not in data:
        # If using int8, convert to inference mode.
        if args.use_bnb_linear is not None:
            from open_clip.utils import convert_int8_model_to_inference_mode
            convert_int8_model_to_inference_mode(model)
        # Evaluate; consider as if the last epoch
        logger.info("No training data is found. Only evaluation at thest phase is performed...")
        evaluate(model, data, args.epochs, args, tb_writer=writer, tokenizer=tokenizer)
        return

    loss = create_loss(args)
    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch} | name: {args.name}')
        train_one_epoch(
            model,
            data,
            loss,
            epoch,
            optimizer,
            scaler,
            scheduler,
            dist_model,
            args,
            tb_writer=writer
        )

        completed_epoch = epoch + 1
        if (
            any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2'))
            or args.comp_validation_datasets or args.test_full_comp
        ):
            evaluate(model, data, completed_epoch, args, tb_writer=writer, tokenizer=tokenizer)

        # Saving checkpoints.
        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": original_model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                if not args.skip_saving_checkpoint:
                    torch.save(
                        checkpoint_dict,
                        os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                    )
            if args.delete_previous_checkpoint:
                previous_checkpoint = os.path.join(
                    args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt"
                )
                if os.path.exists(previous_checkpoint):
                    os.remove(previous_checkpoint)

            if args.save_most_recent:
                # try not to corrupt the latest checkpoint if save fails
                tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                torch.save(checkpoint_dict, tmp_save_path)
                os.replace(tmp_save_path, latest_save_path)

    if args.wandb and is_master(args):
        wandb.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
