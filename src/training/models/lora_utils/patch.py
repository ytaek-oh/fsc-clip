import torch.nn as nn

from .lora_layers import Conv2d, Linear, MultiheadAttention


def _get_original_params(old: nn.Module, new: nn.Module):
    old_params = {n: p for n, p in old.named_parameters()}
    for name, target_param in new.named_parameters():
        if name in old_params:
            target_param.data.copy_(old_params[name].data)


def _patch_linear_lora(model: nn.Module, lora: int, skip_attn=True):
    target_modules = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    if skip_attn:
        target_modules = [n for n in target_modules if "attn" not in n]

    for name in target_modules:
        *parent, child = name.split(".")
        assert len(parent)
        parent = model.get_submodule(".".join(parent))
        old = getattr(parent, child)
        new = Linear(old.in_features, old.out_features, r=lora)
        _get_original_params(old, new)
        setattr(parent, child, new)


def _patch_conv2d_lora(model: nn.Module, lora: int):
    target_modules = [n for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    for name in target_modules:
        *parent, child = name.split(".")
        assert len(parent)
        parent = model.get_submodule(".".join(parent))
        old = getattr(parent, child)
        new = Conv2d(
            old.in_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            r=lora,
            stride=old.stride,
            bias=False,
        )
        _get_original_params(old, new)
        setattr(parent, child, new)


def _patch_transformer_attn_lora(transformer, lora: int):
    assert getattr(transformer, "resblocks")
    for module in transformer.resblocks:
        new_module = MultiheadAttention(module.attn.embed_dim, module.attn.num_heads, r=lora)
        _get_original_params(module.attn, new_module)
        module.attn = new_module
    if not hasattr(transformer, "cross_attn"):
        return

    # cross attention layers
    for module in transformer.cross_attn:
        new_module = MultiheadAttention(module.attn.embed_dim, module.attn.num_heads, r=lora)
        _get_original_params(module.attn, new_module)
        module.attn = new_module


def patch_vit_clip_lora(model: nn.Module, lora: int, device=None):
    """ patch function for ViT CLIP models. It does *not* work for ResNet CLIP models. """

    # inject lora weights for linear and conv layers except for inside attention layers
    _patch_linear_lora(model, lora=lora, skip_attn=True)
    _patch_conv2d_lora(model, lora=lora)

    # update atten layers separately
    _patch_transformer_attn_lora(model.visual.transformer, lora=lora)
    if hasattr(model, "transformer"):
        _patch_transformer_attn_lora(model.transformer, lora=lora)
    if hasattr(model, "text"):
        _patch_transformer_attn_lora(model.text.transformer, lora=lora)
    if hasattr(model, "text_decoder"):
        _patch_transformer_attn_lora(model.text_decoder, lora=lora)
    if device is not None:
        model = model.to(device)
    return model
