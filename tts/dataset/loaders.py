"""
prepares common voice dataloaders

@Author: Dominik Lau
@Links: 
    https://github.com/lifeiteng/vall-e/blob/6dc715a6213864409a94c7ed4a21134eed476b9f/valle/data/datamodule.py
"""

from tts.dataset.common_voice import train_cuts, test_cuts
from lhotse.dataset import (
    CutConcatenate,
    DynamicBucketingSampler,
    SpeechSynthesisDataset
)
from lhotse.dataset.input_strategies import OnTheFlyFeatures
from torch.utils.data import DataLoader
from lhotse import Fbank


transforms = [
    CutConcatenate(duration_factor=1.0, gap=0.1)
]


def train_dataloader(path):
    cuts = train_cuts(path)
    sampler = DynamicBucketingSampler(
        cuts,
        shuffle=True,
        max_duration=100.0,
        num_buckets=10,
    )

    ds = SpeechSynthesisDataset(
        cut_transforms=transforms,
        feature_input_strategy=OnTheFlyFeatures(Fbank())
    )

    dl = DataLoader(
        ds,
        sampler=sampler,
        batch_size=None
    )

    return dl
    

def test_dataloader(path):
    sampler = DynamicBucketingSampler(
        test_cuts(path),
        shuffle=True,
        max_duration=100.0,
        num_buckets=10,
    )

    ds = SpeechSynthesisDataset(
        cut_transforms=transforms,
        feature_input_strategy=OnTheFlyFeatures(Fbank())
    )

    dl = DataLoader(
        ds,
        sampler=sampler,
        batch_size=None
    )

    return dl


if __name__ == "__main__":
    dl_train, dl_test = train_dataloader("data/prepared"), test_dataloader("data/prepared")
    # TODO: tokenizer & token collater
    print(next(iter(dl_train)))
