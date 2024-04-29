"""
prepares common voice dataloaders

@Author: Dominik Lau
@Links: 
    https://github.com/lifeiteng/vall-e/blob/6dc715a6213864409a94c7ed4a21134eed476b9f/valle/data/datamodule.py
"""
from lhotse.dataset import (
    CutConcatenate,
    DynamicBucketingSampler
)
from lhotse.dataset.input_strategies import OnTheFlyFeatures
from torch.utils.data import DataLoader

from tts.dataset.common_voice import train_cuts, test_cuts, val_cuts
from tts.dataset.dataset import TTSDataset


def loaders():
    return {
        "train": __train_dataloader(),
        "test": __test_dataloader(),
        "val": __val_dataloader()
    }


def __train_dataloader(path="data/prepared"):
    return __make_loader(train_cuts(path), True)
    

def __test_dataloader(path="data/prepared"):
    return __make_loader(test_cuts(path), False)


def __val_dataloader(path="data/prepared"):
    return __make_loader(val_cuts(path), False)


def __make_loader(cuts, shuffle):
    sampler = DynamicBucketingSampler(
        cuts,
        shuffle=shuffle,
        max_duration=100.0,
        num_buckets=10,
    )

    ds = TTSDataset()

    dl = DataLoader(
        ds,
        sampler=sampler,
        batch_size=None
    )

    return dl
