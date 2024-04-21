import json

from lhotse import CutSet, Fbank
from lhotse.dataset import (
    CutConcatenate,
    DynamicBucketingSampler,
    SpeechSynthesisDataset
)
from lhotse.dataset.collation import collate_audio

from lhotse.dataset.input_strategies import OnTheFlyFeatures
from tts.dataset.collation import TextTokenCollater


TOKEN_PATH='data/unique_tokens'

class TTSDataset(SpeechSynthesisDataset):
    '''
    TODO: sampling rate is different for some audio, must be resampled to even (otherwise everything fails)
    '''
    def __init__(self, token_path=TOKEN_PATH):
        super().__init__(
            cut_transforms=[
                CutConcatenate(duration_factor=1.0, gap=0.1)
            ],
            feature_input_strategy=OnTheFlyFeatures(Fbank()),
            return_text = False,
            return_tokens=False,
            return_spk_ids = False
        )

        self.collater = TextTokenCollater(
            self.__get_unique_tokens(token_path), 
            add_bos=True, 
            add_eos=True
        )

    def __getitem__(self, cuts):
        batch = super().__getitem__(cuts)
        text = [cut.supervisions[0].text for cut in cuts]
        phonemes = [cut.supervisions[0].custom["phonemes"]["text"] for cut in cuts]
        tokens = [self.collater(p) for p in phonemes]
        return {
            "audio": batch["audio"],
            "audio_lens": batch["audio_lens"],
            "text": text,
            "tokens": tokens,
            "phonemes": phonemes
        }

    def __get_unique_tokens(self, path):
        with open(path, 'r') as f:
            tokens = f.read()
        return json.loads(tokens)