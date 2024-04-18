from lhotse.dataset import (
    CutConcatenate,
    DynamicBucketingSampler,
    SpeechSynthesisDataset
)
from lhotse.dataset.input_strategies import OnTheFlyFeatures
from tts.dataset.collation import get_text_token_collater


class TTSDataset(SpeechSynthesisDataset):
    def __init__(token_path):
        super().__init__(
            cut_transforms=[
                CutConcatenate(duration_factor=1.0, gap=0.1)
            ],
            feature_input_strategy=OnTheFlyFeatures(Fbank())
        )
        self.collater = get_text_token_collater(token_path)

