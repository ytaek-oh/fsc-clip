from .loss import ClipLoss, Global_HNLoss, Local_HNLoss


def create_loss(args):
    if args.loss_name == "clip":
        # single contrasive loss, optionally containing hard negative texts
        return ClipLoss(
            local_batch_size=args.batch_size,
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
        )

    # clip loss with only image - positive text pairs plus, separate hard negative loss
    assert args.loss_name == "fsc-clip"

    if args.apply_local_neg_loss:
        assert args.return_dense_tokens  # enforce models to output token representations
        return Local_HNLoss(
            # clip loss params
            local_batch_size=args.batch_size,
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
            # separate neg loss params
            apply_global_neg_loss=args.apply_global_neg_loss,
            neg_loss_weight=args.neg_loss_weight,  # global neg loss weight
            neg_loss_name=args.neg_loss_name,
            focal_gamma=args.focal_gamma,
            neg_label_smoothing=args.neg_label_smoothing,
            # local neg loss params
            local_neg_weight=args.local_neg_weight,
            sim_normalizer=args.sim_normalizer,
            sim_sparsify=args.sim_sparsify
        )

    assert args.apply_global_neg_loss
    return Global_HNLoss(
        # clip loss params
        local_batch_size=args.batch_size,
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod,
        # separate neg loss params
        apply_global_neg_loss=args.apply_global_neg_loss,
        neg_loss_weight=args.neg_loss_weight,  # global neg loss weight
        neg_loss_name=args.neg_loss_name,
        focal_gamma=args.focal_gamma,
        neg_label_smoothing=args.neg_label_smoothing,
    )
