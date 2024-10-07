from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def _cosine_similarity(x, y, logit_scale=None):
    return x @ y.T if logit_scale is None else logit_scale * x @ y.T


def _cross_entropy_loss(logits, labels, reduction="mean", **kwargs):
    assert reduction in ["mean", "sum", "none"]
    if labels.ndim == 1:
        return F.cross_entropy(logits, labels, reduction=reduction)
    elif labels.ndim == 2:
        # log softmax
        assert reduction == "mean"
        return -1 * torch.sum(labels * F.log_softmax(logits, dim=-1), dim=-1).mean()
    else:
        raise ValueError


def _focal_loss(logits, labels, gamma: float = 1.0, reduction="mean", use_sigmoid=False, **kwargs):
    assert reduction in ["none", "mean", "sum"]
    if use_sigmoid:
        raise NotImplementedError

    # focal loss with softmax-cross entropy
    assert labels.ndim == 1
    ce_loss = _cross_entropy_loss(logits, labels, reduction="none")  # -log(p)
    ce_loss *= torch.pow(1.0 - torch.exp(-ce_loss), gamma)
    if reduction == "none":
        return ce_loss
    divider = len(ce_loss) if reduction == "mean" else 1.0
    return ce_loss.sum() / divider


def _cross_entropy_loss_with_nsr(
    logits, labels, reduction="mean", label_smoothing=0.1, valid_neg_mask=None, **kwargs
):
    # used for neg_logits, always 0 as labels
    assert labels.ndim == 1 and torch.all(labels == 0)
    num_cls_per_row = valid_neg_mask.sum(dim=-1) + 1

    num_batch, num_cls = logits.size(0), logits.size(1)
    labels_2d = torch.zeros((num_batch, num_cls), device=logits.device).float()

    mixture_dist = 1. / num_cls_per_row.unsqueeze(-1)  # uniform distribution
    labels_2d[:, :1] = (1.0 - label_smoothing) + label_smoothing * mixture_dist
    labels_2d[:, 1:] = label_smoothing * mixture_dist
    labels_2d[:, 1:] *= valid_neg_mask

    loss = -1 * torch.sum(labels_2d * F.log_softmax(logits, dim=-1), dim=-1)
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise NotImplementedError


def _focal_loss_with_nsr(
    logits,
    labels,
    gamma=2.0,
    label_smoothing=0.1,
    valid_neg_mask=None,
    use_sigmoid=False,
    reduction="mean"
):
    assert use_sigmoid is False
    assert labels.ndim == 1 and torch.all(labels == 0)
    num_cls_per_row = valid_neg_mask.sum(dim=-1) + 1
    num_batch, num_cls = logits.size(0), logits.size(1)
    labels_2d = torch.zeros((num_batch, num_cls), device=logits.device).float()
    labels_2d[:, 0] = (1.0 - label_smoothing) + label_smoothing / num_cls_per_row
    labels_2d[:, 1:] = label_smoothing / num_cls_per_row.unsqueeze(-1)
    labels_2d[:, 1:] *= valid_neg_mask

    celoss_2d = -1.0 * F.log_softmax(logits, dim=-1)  # -log(p)
    celoss_2d *= torch.pow(1.0 - torch.exp(-celoss_2d), gamma)  # -(1-p)^r log(p)

    loss = torch.sum(labels_2d * celoss_2d, dim=-1)
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise NotImplementedError


def build_neg_loss(loss_name: str, neg_label_smoothing=None, **kwargs):
    assert loss_name in ["cross_entropy", "focal_loss"]
    if neg_label_smoothing is not None:
        if loss_name == "cross_entropy":
            return partial(
                _cross_entropy_loss_with_nsr,
                label_smoothing=neg_label_smoothing,
            )
        elif loss_name == "focal_loss":
            gamma = kwargs.pop("focal_gamma", 2.0)
            return partial(
                _focal_loss_with_nsr,
                label_smoothing=neg_label_smoothing,
                gamma=gamma,
            )

        else:
            raise NotImplementedError

    if loss_name == "cross_entropy":
        return _cross_entropy_loss
    elif loss_name == "focal_loss":
        gamma = kwargs.pop("focal_gamma", 2.0)
        return partial(_focal_loss, gamma=gamma, use_sigmoid=False)


def _clip_loss(logits_per_image, logits_per_text, labels):
    loss_arg_pairs = [(logits_per_image, labels), (logits_per_text, labels)]
    return sum([_cross_entropy_loss(*arg) for arg in loss_arg_pairs]) / len(loss_arg_pairs)


def gather_features(feature_list, **kwargs):
    gathered_features = [_gather_features(f, **kwargs) for f in feature_list]
    return tuple(gathered_features)


def _gather_features(
    features, local_loss=False, gather_with_grad=False, rank=0, world_size=1, use_horovod=False
):
    if world_size == 1:
        return features

    assert has_distributed, (
        'torch.distributed did not import correctly, please use a PyTorch version with support.'
    )
    gathered_features = [torch.zeros_like(features) for _ in range(world_size)]
    dist.all_gather(gathered_features, features)
    if not local_loss:
        # ensure grads for local rank when all_* features don't have a gradient
        gathered_features[rank] = features
    all_features = torch.cat(gathered_features, dim=0)

    return all_features


class LossFunction(nn.Module):
    """ basic utils for distributed training, implementing all_gather ops """

    def __init__(
        self,
        local_batch_size=0,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False,
        **kwargs
    ):
        super().__init__()
        assert local_batch_size > 0
        self.local_batch_size = local_batch_size
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

    def gather(self, features):
        if not isinstance(features, list):
            features = [features]
        all_features = gather_features(
            features,
            local_loss=self.local_loss,
            gather_with_grad=self.gather_with_grad,
            rank=self.rank,
            world_size=self.world_size,
            use_horovod=self.use_horovod
        )
        if len(all_features) == 1:
            return all_features[0]
        return all_features


class ClipLoss(LossFunction):
    """ default clip loss; image-text contrastive """

    def __init__(self, cache_labels=False, **kwargs):
        super().__init__(**kwargs)
        # cache state
        self.cache_labels = cache_labels

        self.prev_num_logits = 0
        self.labels = {}

    def _fill_empty_negatives(self, neg_text_features, neg_mask):
        # neg_mask: (N * B, ) where N: num. types of neg augmentations, with value 1 on valid negs.
        new_neg_shape = (neg_mask.size(0), )
        if neg_text_features.ndim > 1:
            new_neg_shape += neg_text_features.shape[1:]

        new_neg_text_features = torch.zeros(
            new_neg_shape, device=neg_text_features.device, dtype=neg_text_features.dtype
        )
        new_neg_text_features[neg_mask > 0] = neg_text_features
        return new_neg_text_features

    def _rearrange_text_features(self, text_features, global_batch_size):
        if self.world_size == 1:
            return text_features

        # consider, no additional positive texts
        batch_per_gpu = global_batch_size // self.world_size
        num_neg_types = (text_features.size(0) - global_batch_size) // global_batch_size

        text_types = torch.Tensor(
            ([1] * batch_per_gpu + [2] * num_neg_types * batch_per_gpu) * self.world_size
        ).long().to(text_features.device)  # 1: original text type, 2: negative text type
        assert text_types.size(0) == text_features.size(0)
        return torch.cat((text_features[text_types == 1], text_features[text_types == 2]), dim=0)

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        logits_per_image = _cosine_similarity(image_features, text_features, logit_scale)

        if text_features.size(0) > image_features.size(0):
            # exclude similarity logits for the negative texts
            logits_per_text = _cosine_similarity(
                text_features[:image_features.size(0)], image_features, logit_scale
            )
        else:
            logits_per_text = _cosine_similarity(text_features, image_features, logit_scale)
        return logits_per_image, logits_per_text

    def forward(
        self,
        image_features,
        text_features,
        logit_scale,
        neg_text_masks=None,
        output_dict=False,
        return_logits=False,
        **kwargs,
    ):
        assert return_logits is False
        device = image_features.device
        lbs = self.local_batch_size

        # additional hard negative texts
        if text_features.size(0) > lbs:
            assert neg_text_masks is not None
            text_features, neg_text_features = text_features[:lbs], text_features[lbs:]
            neg_text_features = self._fill_empty_negatives(neg_text_features, neg_text_masks)
            text_features = torch.cat((text_features, neg_text_features), dim=0)

        all_img_feats, all_text_feats = self.gather([image_features, text_features])
        gbs = all_img_feats.size(0)

        # additional negative text features -> rearrange for loss calculation in ddp
        if all_text_feats.size(0) > gbs:
            all_text_feats = self._rearrange_text_features(all_text_feats, gbs)

        logits_per_img, logits_per_txt = self.get_logits(all_img_feats, all_text_feats, logit_scale)
        labels = self.get_ground_truth(device, logits_per_img.shape[0])
        total_loss = _clip_loss(logits_per_img, logits_per_txt, labels)

        return {"clip_loss": total_loss} if output_dict else total_loss


class Global_HNLoss(ClipLoss):
    """ CLIP loss with (optional) separate global HN loss """

    def __init__(
        self,
        apply_global_neg_loss=False,
        neg_loss_weight=1.0,
        neg_loss_name="cross_entropy",
        focal_gamma=1.0,
        neg_label_smoothing=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.apply_global_neg_loss = apply_global_neg_loss
        self.neg_loss_weight = neg_loss_weight

        # build custom negative loss
        self.neg_loss_fn = build_neg_loss(
            loss_name=neg_loss_name,
            focal_gamma=focal_gamma,
            neg_label_smoothing=neg_label_smoothing
        )

    def forward(
        self,
        image_features=None,
        text_features=None,
        neg_text_masks=None,
        logit_scale=None,
        output_dict=False,
        return_logits=False,
        **kwargs
    ):
        assert output_dict
        device = image_features.device
        lbs = self.local_batch_size

        if text_features.size(0) > lbs:
            assert neg_text_masks is not None
            text_features, neg_text_features = text_features[:lbs], text_features[lbs:]
            neg_text_features = self._fill_empty_negatives(neg_text_features, neg_text_masks)

        total_loss = {}

        # contrastive loss after all-gather ops
        clip_loss = super().forward(image_features, text_features, logit_scale, output_dict=True)
        if return_logits:
            clip_loss, clip_logits = clip_loss
        total_loss.update(clip_loss)

        if self.apply_global_neg_loss:
            # transpose neg text features (N*B, D) -> (B, N, D)
            neg_text_features = torch.stack(neg_text_features.split(lbs, dim=0), dim=1)

            # image-text logits: (B, 1, D) * (B, 1+N, D) = (B, 1, 1+N) -> (B, 1+N)
            neg_logits_per_img = logit_scale * (
                torch.einsum(
                    "bkd,bld->bkl",
                    image_features.unsqueeze(1),
                    torch.cat((text_features.unsqueeze(1), neg_text_features), dim=1),
                ).squeeze(1)
            )
            # dimension 0 -> always the positive text
            neg_labels = torch.zeros(lbs, device=device, dtype=torch.long)

            # consider loss for the rows that have at least one non-zero text hard negative,
            # for the softmax-ce-loss works properly.
            valid_neg_mask = torch.stack(neg_text_masks.split(lbs, dim=0), dim=1)
            valid_neg_weight = torch.any(valid_neg_mask > 0, dim=1)

            # (n+1)-class classification, distinguishing positive assignment over negative ones
            clip_neg_loss = self.neg_loss_fn(
                neg_logits_per_img[valid_neg_weight],
                neg_labels[valid_neg_weight],
                valid_neg_mask=valid_neg_mask[valid_neg_weight]
            )
            total_loss["clip_neg_loss"] = self.neg_loss_weight * clip_neg_loss

        if return_logits:
            return total_loss, clip_logits
        return total_loss


class Local_HNLoss(Global_HNLoss):

    def __init__(
        self,
        local_neg_weight=1.0,
        sim_normalizer="minmax",
        sim_sparsify=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.local_neg_weight = local_neg_weight
        self.sim_normalizer = sim_normalizer
        self.sim_sparsify = sim_sparsify

    def get_per_token_visual_patches(self, image_tokens, text_tokens):
        # language-grouped visual patches
        # B: batch_size, P: num. patches of visual feature, T: num. tokens of text feature

        # (B, T, P)
        sim_map = torch.einsum("btd,bpd->btp", text_tokens, image_tokens).detach()

        # max-min normalize across patches dim
        if self.sim_normalizer == "minmax":
            sim_max = sim_map.max(dim=-1, keepdim=True)[0]
            sim_min = sim_map.min(dim=-1, keepdim=True)[0]
            sim_map = (sim_map - sim_min) / (sim_max - sim_min)
            if self.sim_sparsify:
                num_patches = image_tokens.size(1)
                sim_map[sim_map < 1. / num_patches] = 0.
            sim_map = sim_map / sim_map.sum(dim=-1, keepdim=True)
        else:
            assert self.sim_normalizer == "softmax"
            sim_map = sim_map.softmax(dim=-1)

        # weighted average of visual patches per token.
        image_tokens = torch.einsum("btp,bpd->btd", sim_map, image_tokens)

        return F.normalize(image_tokens, dim=-1)

    def get_aggregated_local_score(self, image_tokens, text_tokens, attn_mask, logit_scale):
        # image_tokens and text_tokens: (B, T, D), attn_mask: (B, T)
        text_tokens = F.normalize(text_tokens, dim=-1)  # which was un-normalized,

        batch_size, context_len = image_tokens.shape[:2]
        local_sim = logit_scale * torch.einsum(
            "bd,bd->b",
            image_tokens.view(batch_size * context_len, -1),  # (BT, D)
            text_tokens.view(batch_size * context_len, -1),  # (BT, D)
        ).view(batch_size, context_len)  # (B, T)

        # mask out <pad> tokens. Note: Any row of `local_sim` does not have all-zero scores
        #  which causes inf after `torch.logsumexp`, especifally for handling neg. caption tokens.
        additive_attn_mask = torch.where(attn_mask > 0, 0., float("-inf"))
        return torch.logsumexp(local_sim + additive_attn_mask, dim=1)

    def forward(
        self,
        image_features=None,
        image_tokens=None,
        text_features=None,
        text_tokens=None,
        attn_mask=None,
        neg_text_masks=None,
        logit_scale=None,
        output_dict=False,
        return_logits=False,
        **kwargs
    ):
        assert output_dict
        assert image_tokens is not None
        assert text_tokens is not None
        assert attn_mask is not None
        device = image_features.device
        lbs = self.local_batch_size

        # compute global losses, including positive and optional neg loss.
        total_loss = {}
        global_loss = super().forward(
            image_features=image_features,
            text_features=text_features,
            neg_text_masks=neg_text_masks,
            logit_scale=logit_scale,
            output_dict=True
        )

        if return_logits:
            global_loss, clip_logits = global_loss
        total_loss.update(global_loss)

        # compute local counterpart
        if text_tokens.size(0) > lbs:
            assert neg_text_masks is not None
            text_tokens, neg_text_tokens = text_tokens[:lbs], text_tokens[lbs:]
            attn_mask, neg_attn_mask = attn_mask[:lbs], attn_mask[lbs:]

        # process local features - token-patches for positive pair
        new_image_tokens = self.get_per_token_visual_patches(image_tokens, text_tokens)

        # image-text aggregated local alignment score
        pos_local_scores = self.get_aggregated_local_score(
            new_image_tokens, text_tokens, attn_mask, logit_scale
        )

        # do the same thing with negatives; More efficient way of such calculation without for loop?
        neg_chunk_sizes = [neg_mask.sum().item() for neg_mask in neg_text_masks.split(lbs, dim=0)]

        neg_local_score_list = []
        for _neg_text_masks, _neg_text_tokens, _neg_attn_mask in zip(
            neg_text_masks.split(lbs, dim=0),
            neg_text_tokens.split(neg_chunk_sizes, dim=0),
            neg_attn_mask.split(neg_chunk_sizes, dim=0),
        ):
            # select images corresponding to neg text captions
            _image_tokens = image_tokens[_neg_text_masks > 0]
            assert _image_tokens.size(0) == _neg_text_tokens.size(0) == _neg_attn_mask.size(0)
            neg_img_tokens = self.get_per_token_visual_patches(_image_tokens, _neg_text_tokens)
            neg_local_scores = self.get_aggregated_local_score(
                neg_img_tokens, _neg_text_tokens, _neg_attn_mask, logit_scale
            )  # calculate the aggregated token-wise alignment score
            neg_local_score_list.append(neg_local_scores)
        neg_local_score_list = torch.cat(neg_local_score_list, dim=0)
        assert neg_local_score_list.size(0) == neg_text_tokens.size(0)

        # fill zero values on the positions of empty negative texts on a batch
        neg_local_score_list = self._fill_empty_negatives(neg_local_score_list, neg_text_masks)

        # apply neg loss in the same way as global contrastive; transpose.
        neg_local_score_list = torch.stack(neg_local_score_list.split(lbs, dim=0), dim=1)
        neg_logits = torch.cat((pos_local_scores.unsqueeze(1), neg_local_score_list), dim=1)
        neg_labels = torch.zeros(lbs, device=device, dtype=torch.long)

        # consider loss for the rows that have at least one non-zero text hard negative,
        # for the softmax-ce-loss works properly
        valid_neg_mask = torch.stack(neg_text_masks.split(lbs, dim=0), dim=1)
        valid_neg_weight = torch.any(valid_neg_mask > 0, dim=1)

        # (n+1)-class classification, distinguishing positive assignment over negative ones
        local_neg_loss = self.neg_loss_fn(
            neg_logits[valid_neg_weight],
            neg_labels[valid_neg_weight],
            valid_neg_mask=valid_neg_mask[valid_neg_weight]
        )
        total_loss["local_neg_loss"] = self.local_neg_weight * local_neg_loss

        if return_logits:
            return total_loss, clip_logits
        return total_loss
