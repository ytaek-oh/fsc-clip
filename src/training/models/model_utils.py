import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from open_clip import create_model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _resize_text_pos_embed(
    model, new_context_length: int, interpolation: str = 'linear', antialias: bool = False
):
    # FIXME add support for text cls_token
    old_pos_embed = getattr(model, 'positional_embedding', None)
    assert old_pos_embed is not None, "Currently, only support with `CLIP` class, not Custom Text."

    old_num_pos, old_width = old_pos_embed.size()
    if old_num_pos == new_context_length:
        return

    logger.info(
        'Resizing text position embedding from {} to {}'.format(old_num_pos, new_context_length)
    )
    new_pos_embed = old_pos_embed.clone()
    if new_context_length > old_num_pos:
        new_pos_embed = new_pos_embed.reshape(1, old_num_pos, old_width).permute(0, 2, 1)
        new_pos_embed = F.interpolate(
            new_pos_embed,
            size=new_context_length,
            mode=interpolation,
            antialias=antialias,
            align_corners=False,
        )
        new_pos_embed = new_pos_embed.permute(0, 2, 1)[0]
    else:
        new_pos_embed = new_pos_embed[:new_context_length, :]

    setattr(model, "positional_embedding", nn.Parameter(new_pos_embed))
    model.context_length = new_context_length
    return


def _build_causal_mask(context_length: int):
    # lazily create causal attention mask, with full attention between the tokens
    # pytorch uses additive attention mask; fill with -inf
    mask = torch.empty(context_length, context_length)
    mask.fill_(float("-inf"))
    mask.triu_(1)  # zero out the lower diagonal
    return mask


def adjust_context_length(model, context_length: int):
    # modify text positional encoding
    _resize_text_pos_embed(model, context_length)

    # modify attn_mask if exists
    if not hasattr(model, "attn_mask"):
        return

    attn_mask = getattr(model, "attn_mask", None)
    if attn_mask is not None:
        new_attn_mask = _build_causal_mask(context_length)
        setattr(model, "attn_mask", new_attn_mask)
    return model


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def merge(alpha, theta_0, theta_1):  # (pretrained, finetuned)
    # in CLoVe model, fine-tuned context length is 64, while openai model has 77
    # -> only take the first 64 dim for the pretrained checkpoint
    if "positional_embedding" in theta_1:
        pos_emb_0, pos_emb_1 = theta_0["positional_embedding"], theta_1["positional_embedding"]
        if pos_emb_1.shape[0] < pos_emb_0.shape[0]:
            # prune positional embedding of pretrained checkpoint
            theta_0["positional_embedding"] = theta_0["positional_embedding"][:pos_emb_1.shape[0]]

    # interpolate between all weights in the checkpoints
    return {key: (1 - alpha) * theta_0[key] + alpha * theta_1[key] for key in theta_0.keys()}


def patch_wise_ft(model, args, device):
    with_lora = ("DAC" in args.model or "TSVLC" in args.model) or args.lora_rank > 0
    finetuned_ckpt = model.state_dict()
    if with_lora > 0:
        # LoRA weights for pretrained model as zero, and interpolate lora weights only, keeping
        #  original pre-trained weight the same.
        finetuned_ckpt = {k: v.clone() for k, v in finetuned_ckpt.items() if "lora" in k}
        pretrained_ckpt = {k: torch.zeros_like(v) for k, v in finetuned_ckpt.items()}
    else:
        # load pretrained model from open_clip
        pretrained, arch = args.zeroshot_model.split(":")
        pretrained_ckpt = create_model(
            arch,
            pretrained,
            precision=args.precision,
            device=device,
            cache_dir=args.model_cache_dir
        ).state_dict()

    assert set(finetuned_ckpt.keys()) == set(pretrained_ckpt.keys())
    merged = merge(args.wise_ft_alpha, pretrained_ckpt, finetuned_ckpt)
    model.load_state_dict(merged, strict=(not with_lora))
    return model
