import ast
import glob
import json
import logging
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from functools import partial
from multiprocessing import Value

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
from huggingface_hub import snapshot_download
from loguru import logger
from PIL import Image
from torch.utils.data import (
    DataLoader, Dataset, IterableDataset, SubsetRandomSampler, get_worker_info,
)
from torch.utils.data.distributed import DistributedSampler

import braceexpand
import webdataset as wds
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, tar_file_expander, url_opener, valid_sample

from .text_negatives import add_random_text_hard_negatives

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


class CsvDataset(Dataset):

    def __init__(
        self,
        input_filename,
        transforms,
        img_key,
        caption_key,
        sep="\t",
        tokenizer=None,
        image_root=""
    ):
        df = pd.read_csv(input_filename, sep=sep)
        logger.info(f'Csv data from {input_filename}, size: {len(df)}.')

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer
        self.image_root = image_root

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image_file = os.path.join(self.image_root, str(self.images[idx]))
        images = self.transforms(Image.open(image_file))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts


class CocoDataset(CsvDataset):

    def __init__(
        self,
        input_filename,
        transforms,
        img_key,
        caption_key,
        sep="\t",
        tokenizer=None,
        image_root="",
        text_hard_negatives=None,
        is_train=False
    ):
        df = pd.read_csv(input_filename, sep=sep, converters={"neg_caption": ast.literal_eval})
        logger.info(f'Csv data from {input_filename}, size: {len(df)}.')

        # coco-specific operation
        df[img_key] = df[img_key].apply(lambda x: x.replace("coco/", ""))

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.csv_neg_captions = df["neg_caption"].tolist()
        self.transforms = transforms

        self.tokenize = tokenizer
        self.image_root = image_root

        # additional ops
        self.text_hard_negatives = text_hard_negatives
        self.is_train = is_train

    def __getitem__(self, idx):
        assert self.is_train
        image_file = os.path.join(self.image_root, str(self.images[idx]))
        images = self.transforms(Image.open(image_file))
        texts = str(self.captions[idx])

        if self.text_hard_negatives is None:
            return images, texts

        if self.text_hard_negatives == "csv_negclip":
            neg_caption = random.choice(self.csv_neg_captions[idx])
            return images, texts, neg_caption

        return images, texts

    @classmethod
    def collate_fn(cls, text_hard_negatives=None, tokenizer=None):

        def func(batch):
            neg_texts, neg_masks = [], None
            batches = zip(*batch)
            if text_hard_negatives == "csv_negclip":
                images, texts, neg_texts = batches
                neg_texts = list(neg_texts)
                neg_masks = [1 for _ in range(len(texts))]
            else:
                images, texts = batches

            texts = list(texts)

            if text_hard_negatives is not None and text_hard_negatives != "csv_negclip":
                neg_texts, neg_masks = add_random_text_hard_negatives(
                    texts, style=text_hard_negatives, text_batch_size=len(texts)
                )

            outputs = {
                "images": torch.stack(images, dim=0),
                "texts": tokenizer(texts + neg_texts),
                "neg_masks": torch.Tensor(neg_masks).long() if neg_masks is not None else None
            }
            return outputs

        return func


class SharedEpoch:

    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist), (
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) "
            "to match."
        )
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = 0
        for shard in shards_list:
            # in case of shards downloaded via img2dataset
            shard_stat_file = shard.replace(".tar", "_stats.json")
            if not os.path.exists(shard_stat_file):
                total_size = None  # num samples undefined
                break
            total_size += int(json.load(open(shard_stat_file))["successes"])
        logger.info(f"Total training samples: {total_size}")

    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):

    def __init__(
        self,
        bufsize=1000,
        initial=100,
        seed=0,
        epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or
            # train) situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all
            # nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights),\
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess
            # (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by
                # arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


def _replace_caption_col(sample, caption_col=None, data_root=None):
    assert caption_col is not None
    item = sample["json"]

    # first check if 'caption_col' is present in the loaded item
    if caption_col in item:
        sample["txt"] = sample["json"][caption_col]
        return sample

    # try to load from local directory
    assert data_root is not None
    assert os.path.exists(os.path.join(data_root, caption_col))
    caption_file = os.path.join(data_root, caption_col, f"{item['key']}.txt")
    with open(caption_file, "r") as f:  # is it bottleneck? then need to pre-load captions
        sample["txt"] = f.readline()
    return sample


def _process_texts(sample, neg_append_fn=None, tokenizer=None):
    # input: ("image", "text"),  output: ("image", "text", "neg_text_mask")
    out_tuple = (sample[0], )

    text = sample[1]
    neg_text_mask = None

    assert len(sample) == 2
    if neg_append_fn is not None:
        neg_text, neg_text_mask = neg_append_fn(text)
        neg_text_mask = torch.Tensor(neg_text_mask).to(torch.long)
        text += neg_text
    text = tokenizer(text)
    out_tuple += (text, )
    if neg_text_mask is not None:
        out_tuple += (neg_text_mask, )
    return out_tuple


def get_wds_dataset(
    args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None, input_shards=None
):
    if input_shards is None:
        input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training '
                    'dataset. Please specify it via `--train-num-samples` if no dataset length '
                    'info is present.'
                )
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0
        if num_samples == 0:
            num_samples, _ = get_dataset_size(input_shards)

    # create a shared epoch store to sync epoch to dataloader worker proc
    shared_epoch = SharedEpoch(epoch=epoch)

    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled, (
            "--train_data_upsampling_factors is only supported when sampling with replacement "
            "(with --dataset-resampled)."
        )

    if resampled:
        pipeline = [
            ResampledShards2(
                input_shards,
                weights=args.train_data_upsampling_factors,
                deterministic=True,
                epoch=shared_epoch,
            )
        ]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend(
                [
                    detshuffle2(
                        bufsize=_SHARD_SHUFFLE_SIZE,
                        initial=_SHARD_SHUFFLE_INITIAL,
                        seed=args.seed,
                        epoch=shared_epoch,
                    ),
                    wds.split_by_node,
                    wds.split_by_worker,
                ]
            )
        # at this point, we have an iterator over the shards assigned to each worker at each node
        pipeline.extend(
            [
                tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
                wds.shuffle(bufsize=_SAMPLE_SHUFFLE_SIZE, initial=_SAMPLE_SHUFFLE_INITIAL),
            ]
        )
    else:
        # at this point, we have an iterator over the shards assigned to each worker
        pipeline.extend([wds.split_by_worker, wds.tarfile_to_samples(handler=log_and_continue)])

    pipeline.extend(
        [
            wds.select(filter_no_caption_or_no_image),
            wds.decode("pilrgb", handler=log_and_continue),
        ]
    )

    # set caption_key and add pos_caption
    if is_train:
        if args.caption_key != "caption":
            tar_path = os.path.dirname(expand_urls(input_shards)[0][0])

            # replace caption column with new caption_key
            pipeline.append(
                wds.map(
                    partial(_replace_caption_col, caption_col=args.caption_key, data_root=tar_path)
                )
            )

    pipeline.extend(
        [
            wds.rename(image="jpg;png;jpeg;webp", text="txt"),
            wds.map_dict(image=preprocess_img),
            wds.to_tuple("image", "text"),
            wds.batched(args.batch_size, partial=not is_train),
        ]
    )

    # add negative captions to the original caption
    neg_append_fn = None
    if is_train and args.add_random_text_hard_negatives:
        neg_append_fn = partial(
            add_random_text_hard_negatives,
            style=args.add_random_text_hard_negatives,
            text_batch_size=args.batch_size
        )

    # concat all texts and tokenize
    pipeline.append(
        wds.map(partial(_process_texts, neg_append_fn=neg_append_fn, tokenizer=tokenizer))
    )

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args.workers * args.world_size, (
                'number of shards must be >= total workers'
            )

        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(
    args,
    preprocess_fn,
    is_train,
    epoch=0,
    tokenizer=None,
    constructor=CsvDataset,
    collate_fn=None,
):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename

    # ["CsvDataset", "CocoDataset"]
    dataset = constructor(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer,
        image_root=args.image_root
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    if collate_fn is not None:
        dataloader = DataLoader(
            dataset,
            collate_fn=collate_fn(tokenizer=tokenizer),
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
            pin_memory=True,
            sampler=sampler,
            drop_last=is_train,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
            pin_memory=True,
            sampler=sampler,
            drop_last=is_train,
        )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def init_wds_train_dataset(data_name, data_path):
    os.makedirs(data_path, exist_ok=True)
    hf_repo_id = f"ytaek-oh/{data_name}-subset-100k"
    snapshot_download(
        repo_id=hf_repo_id,
        repo_type="dataset",
        local_dir=data_path,
        local_dir_use_symlinks=False,
    )

    # extract coca captions
    if data_name == "cc3m":
        subprocess.call(
            ["tar", "-zxf", os.path.join(data_path, "coca_captions.tar.gz")], cwd=data_path
        )


def _get_laioncoco_dataset_fn(train_data, data_root, is_train=False, args=None):
    data_root = os.path.join(data_root, "laioncoco")
    if not is_train:
        val_data = os.path.join(data_root, "eval", "{00000..00001}.tar")
        return partial(get_wds_dataset, input_shards=val_data)

    subset_name = train_data.split("_")[-1].upper().replace(".", "_")
    assert subset_name in ["100K"], f"invalid subset_name {subset_name}"
    data_path = os.path.join(data_root, f"train_subset_{subset_name}")
    tar_list = sorted(glob.glob(os.path.join(data_path, "*.tar")))

    # initialize dataset if not exists
    if not os.path.exists(data_path) or len(tar_list) < 1:
        init_wds_train_dataset("laioncoco", data_path)

    def _extract_number(tar_path):
        return os.path.basename(tar_path).split(".")[0]

    def _wrap_together(tar_paths):
        left = _extract_number(tar_paths[0])
        right = _extract_number(tar_paths[-1])
        return "{" + left + ".." + right + "}.tar"

    train_data = os.path.join(data_path, _wrap_together(tar_list))
    return partial(get_wds_dataset, input_shards=train_data)


def _get_cc_dataset_fn(train_data, data_root, is_train=False, args=None, data=None):
    assert data in ["cc3m"]
    data_root = os.path.join(data_root, data)
    subset_name = train_data.split("_")[-1].upper().replace(".", "_")
    assert subset_name in ["100K"], f"invalid subset_name {subset_name}"
    data_path = os.path.join(data_root, f"train_subset_{subset_name}")
    tar_list = sorted(glob.glob(os.path.join(data_path, "*.tar")))

    # initialize dataset if not exists
    if not os.path.exists(data_path) or len(tar_list) < 1:
        init_wds_train_dataset(data, data_path)

    def _extract_number(tar_path):
        return os.path.basename(tar_path).split(".")[0]

    def _wrap_together(tar_paths):
        left = _extract_number(tar_paths[0])
        right = _extract_number(tar_paths[-1])
        return "{" + left + ".." + right + "}.tar"

    train_data = os.path.join(data_path, _wrap_together(tar_list))
    return partial(get_wds_dataset, input_shards=train_data)


def _get_negclip_coco_dataset_fn(train_data, data_root, is_train=False, args=None):
    assert args is not None
    assert train_data in ["train_neg_clip"]
    data_root = os.path.join(data_root, "coco")
    os.makedirs(data_root, exist_ok=True)

    if (
        (not os.path.exists(os.path.join(data_root, "images/train2014")))
        or (not os.path.exists(os.path.join(data_root, "images/val2014")))
    ):
        from vl_compo.datasets.dataset_downloader import download_coco_datasets
        download_coco_datasets(data_root, version=2014, splits=["train", "val"])

    ann_file = os.path.join(data_root, f"{train_data}.tsv")
    if not os.path.exists(ann_file):
        from vl_compo.utils import download_url
        ann_url = "https://raw.githubusercontent.com/mertyg/vision-language-models-are-bows/refs/heads/main/temp_data/train_neg_clip.tsv"  # noqa
        ann_file = download_url(ann_url, root_dir=ann_file)

    # modify args proper to negclip_coco annotation
    args.caption_key = "title"
    args.csv_img_key = "filepath"
    args.csv_separator = "\t"
    args.image_root = os.path.join(data_root, "images")
    assert is_train
    args.train_data = ann_file

    # options
    constructor = partial(
        CocoDataset, text_hard_negatives=args.add_random_text_hard_negatives, is_train=is_train
    )
    collate_fn = partial(
        CocoDataset.collate_fn, text_hard_negatives=args.add_random_text_hard_negatives
    )
    return partial(get_csv_dataset, constructor=constructor, collate_fn=collate_fn)


def get_dataset_fn(args, is_train=False):
    data_path = args.train_data if is_train else args.val_data
    data_root = args.train_data_root
    dataset_type = args.dataset_type

    # custom webdataset format downloaded by img2dataset
    if "laioncoco" in data_path:
        assert dataset_type == "webdataset"  # currently, only support webdataset format
        return _get_laioncoco_dataset_fn(data_path, data_root, is_train=is_train, args=args)
    elif "cc3m" in data_path:
        assert dataset_type == "webdataset"
        return _get_cc_dataset_fn(data_path, data_root, is_train=is_train, args=args, data="cc3m")

    # custom csv dataset: negclip_coco and pug_ar4t, DAC
    elif "neg_clip" in data_path:
        return _get_negclip_coco_dataset_fn(data_path, data_root, is_train=is_train, args=args)

    elif dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "auto":
        raise NotImplementedError
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.dataset_type == "synthetic":
        train_data_fn = get_dataset_fn(args, is_train=True)
        data["train"] = train_data_fn(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer
        )

    if args.val_data:
        val_data_fn = get_dataset_fn(args, is_train=False)
        data["val"] = val_data_fn(args, preprocess_val, is_train=False, tokenizer=tokenizer)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data
