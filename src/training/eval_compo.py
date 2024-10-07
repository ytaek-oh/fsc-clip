import json
import os

import numpy as np
import torch
from loguru import logger
from open_clip import create_transforms, get_tokenizer

from training.models.model_utils import unwrap_model
from vl_compo.datasets.catalog import build_dataset, get_task_key
from vl_compo.evaluation import extract_meta_avg, get_evaluator
from vl_compo.model_wrappers import CLIPWrapper


def compo_eval(model, epoch, args, val_preprocess=None, tokenizer=None, phase="val"):
    if phase == "val" and not args.comp_validation_datasets:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}

    model_without_ddp = unwrap_model(model)

    if val_preprocess is None:
        _, val_preprocess = create_transforms(model_without_ddp.visual.preprocess_cfg)

    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    # wrap model
    device = (torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu"))
    model = CLIPWrapper(model_without_ddp, device=device, tokenizer=tokenizer, amp=False)

    # build compositionality datasets for evaluation
    eval_datasets = [
        "aro", "crepe", "eqben", "imagecode", "sugarcrepe", "svo_probes", "valse", "vl_checklist",
        "whatsup", "winoground", "spec", "coco_retrieval_karpathy", "flickr30k_retrieval",
        "coco_counterfactual_retrieval", "elevater"
    ]
    if phase == "val":
        eval_datasets = args.comp_validation_datasets
    logger.info("Benchmarks to evaluate: {}".format(",".join(eval_datasets)))

    comp_metrics = {}
    results_all = {}
    for data_name in eval_datasets:
        logger.info(f"Start Evaluating {data_name}...")

        # build dataset
        dataset, summarizer = build_dataset(data_name, args.comp_data_root, val_preprocess)

        # build evaluator
        evaluator = get_evaluator(
            get_task_key(data_name), batch_size=args.comp_batch_size, num_workers=args.workers
        )

        # perform evaluation
        results_dict, _ = evaluator(dataset, model, summarizer=summarizer, save_scores=False)
        results_all[data_name] = results_dict

        # update metrics for logging
        summary = results_dict["summary"]
        logger.info(f"summary of the evaluation results on {data_name}:")
        print(json.dumps(summary))

        meta_avg = extract_meta_avg(summary, data_name)
        comp_metrics.update(meta_avg)

    if phase == "val":
        return comp_metrics

    # aggregation for test phase
    comp_datasets = [data for data in eval_datasets if get_task_key(data) == "compositional"]
    comp_avg = np.mean([comp_metrics[d] for d in comp_datasets])
    comp_metrics.update({"comp_avg": comp_avg})

    # average retrieval tasks
    i2t_keys = [k for k in comp_metrics.keys() if "i2t/R@1" in k]
    comp_metrics.update({"i2t_avg": np.mean([comp_metrics[k] for k in i2t_keys])})
    t2i_keys = [k for k in comp_metrics.keys() if "t2i/R@1" in k]
    comp_metrics.update({"t2i_avg": np.mean([comp_metrics[k] for k in t2i_keys])})

    # save full results
    results_all["experiment_name"] = args.name
    out_file = os.path.join(args.checkpoint_path, "test_compo_detailed_eval_results.txt")
    with open(out_file, "w") as f:
        f.write(json.dumps(results_all, indent=2))
    return comp_metrics
