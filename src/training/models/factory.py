import os

import numpy as np
import torch
from loguru import logger
from open_clip import create_model, create_transforms, get_tokenizer, merge_preprocess_kwargs

from vl_compo.models.build import get_model

from .lora_utils import mark_only_lora_as_trainable, patch_vit_clip_lora
from .model_utils import adjust_context_length, count_parameters, patch_wise_ft, unwrap_model


def created_finetuned_model_and_transforms(args, device=None):
    base_cache_dir = os.path.expanduser("~/.cache")
    assert not args.train_data, "Test only"

    if "CLoVe" in args.model:
        assert args.force_context_length == 64, (
            "set context length properly to prevent further errors."
        )

    model_wrapper, val_preprocess = get_model(
        args.model,
        args.pretrained,
        args.comp_model_family,
        base_cache_dir,
        amp=False,
    )

    model = model_wrapper.model.eval()
    if device is not None:
        model = model.to(device)

    tokenizer = model_wrapper.tokenizer
    if tokenizer is None:
        logger.warning("Undefined tokenizer! Try using the default ViT-B-32 tokenizer.")
        tokenizer = get_tokenizer("ViT-B-32", context_length=args.force_context_length)
    return model, val_preprocess, tokenizer


def _create_model(args, device):
    if args.precision == 'fp16':
        logger.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.'
        )

    if args.distributed:
        logger.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.'
        )
    else:
        logger.info(f'Running with a single process. Device {args.device}.')

    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]

    model_kwargs = {"return_dense_tokens": args.return_dense_tokens}
    if args.siglip:
        model_kwargs['init_logit_scale'] = np.log(10)  # different from CLIP
        model_kwargs['init_logit_bias'] = -10

    force_preprocess_cfg = merge_preprocess_kwargs(
        {},
        mean=args.image_mean,
        std=args.image_std,
        interpolation=args.image_interpolation,
        resize_mode=args.image_resize_mode  # only effective for inference
    )

    pretrained = args.pretrained
    if args.lora_rank > 0 and pretrained != "openai" and os.path.exists(pretrained):
        # loading lora checkpoints: first initialize a new CLIP model and load the state dict
        pretrained = "openai"  # FIX! assuming initial pretrained model with quick_gelu=True
        # may not work properly when loading model without pre-trained/fine-tuned with this...
        model_state_dict = torch.load(args.pretrained, map_location="cpu")["state_dict"]

    model = create_model(
        args.model,
        pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        force_preprocess_cfg=force_preprocess_cfg,
        pretrained_image=args.pretrained_image,
        pretrained_hf=True,
        cache_dir=args.model_cache_dir,
        output_dict=True,
        **model_kwargs,
    )

    # post-hoc patching lora layers. It maintains the parameters in the existing layer
    if args.lora_rank > 0:
        model = patch_vit_clip_lora(model, lora=args.lora_rank, device=device)
        mark_only_lora_as_trainable(model)

        # loading lora fine-tuned checkpoint, if specified.
        if args.pretrained != "openai" and os.path.exists(args.pretrained):
            msg = model.load_state_dict(model_state_dict, strict=False)
            logger.info(
                f"Loading lora checkpoints. unexpected keys: {msg.unexpected_keys}, "
                f"missing_keys: {msg.missing_keys}"
            )

    if args.force_context_length is not None:
        adjust_context_length(model, context_length=args.force_context_length)
        logger.info(
            f"custom context length to text encoder and tokenizer: {args.force_context_length}"
        )

    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats
        )
        logger.info("Lock visual encoder")

    if args.lock_text:
        model.lock_text_tower(
            unlocked_layers=args.lock_text_unlocked_layers,
            freeze_layer_norm=args.lock_text_freeze_layer_norm
        )
        logger.info("lock text encoder")

    if args.grad_checkpointing:
        model.set_grad_checkpointing()
        logger.info("use grad checkpointing")

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)
        logger.info(f"enabled ddp; world_size: {args.world_size}, rank: {args.rank}.")
    logger.info(f"Trainable parameters: {count_parameters(model) / 1024 / 1024:.3f}M")

    return model


def create_model_and_transforms(args, device):

    if args.comp_model_family is not None:
        # load model from vl_compo module
        model, preprocess_val, tokenizer = created_finetuned_model_and_transforms(
            args, device=device
        )
        preprocess_train = None  # currently, eval only
    else:
        # load from open_clip
        model = _create_model(args, device)
        preprocess_cfg = unwrap_model(model).visual.preprocess_cfg
        preprocess_train, preprocess_val = create_transforms(preprocess_cfg, aug_cfg=args.aug_cfg)
        tokenizer = get_tokenizer(args.model, context_length=args.force_context_length)

    # apply weight-space interpolation between pretrained and finetuned if specified.
    if args.wise_ft_alpha is not None:
        assert args.wise_ft_alpha > 0 and args.wise_ft_alpha < 1.0
        logger.info("applying weight-space interpolation with alpha={}".format(args.wise_ft_alpha))
        model = patch_wise_ft(model, args, device)

    return model, preprocess_train, preprocess_val, tokenizer
