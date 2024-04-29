"""
prepares a common voice dataset
also QoL functions for commonvoice cuts

@Author: Dominik Lau
@Links: 
    https://lhotse-speech.github.io/2020/11/03/introduction.html
    https://github.com/lhotse-speech/lhotse/blob/master/examples/00-basic-workflow.ipynb
"""
import os
import json

from lhotse.recipes.commonvoice import prepare_commonvoice
from lhotse.dataset.speech_synthesis import SpeechSynthesisDataset
from lhotse import CutSet, SupervisionSet, RecordingSet

from tts.dataset.phonemes import TextPhonemizer
from tts.dataset.dataset import TOKEN_PATH


DATA_DIR = "data/cv-corpus-17.0-2024-03-15-pl/cv-corpus-17.0-2024-03-15"
OUTPUT_DIR = "data/prepared"
SPLITS = ("test", "dev", "train")
LANGS = "pl"

SAMPLE_RATE=24000 # 24kHz

# DATA_DIR = # path to downloaded common voice
# OUTPUT_DIR = # output path for manifests
# SPLITS = ("test", "dev", "train")
# LANGS = # languages of dataset, example: "pl", ["pl", "en"]


def main():
    prepare_commonvoice(
        corpus_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        languages=LANGS,
        splits=SPLITS
    )
    phonemes_train = __prepare(OUTPUT_DIR, "train")
    phonemes_test = __prepare(OUTPUT_DIR, "test")
    phonemes_val = __prepare(OUTPUT_DIR, "dev")

    with open(TOKEN_PATH, 'w') as f:
        f.write(json.dumps(list(phonemes_train | phonemes_test | phonemes_val)))


def train_cuts(manifest_dir, lang="pl"):
    return __get_dataset(manifest_dir, "train", lang)


def test_cuts(manifest_dir, lang="pl"):
    return __get_dataset(manifest_dir, "test", lang)


def val_cuts(manifest_dir, lang="pl"):
    return __get_dataset(manifest_dir, "dev", lang)


def __prepare(manifest_dir, t, lang="pl"):
    '''
    performs text phonemization and adds it to the common voice dataset
    '''
    supervisions = SupervisionSet.from_jsonl(f"{manifest_dir}/cv-{lang}_supervisions_{t}.jsonl.gz")
    pho = TextPhonemizer()
    tokens = set()
    for supervision in supervisions:
        phons = pho(supervision.text)
        supervision.custom["phonemes"] = {"text": phons}
        tokens |= {char for char in phons}
    supervisions.to_file(f"{manifest_dir}/cv-{lang}_supervisions_{t}.jsonl.gz")
    return tokens


def __get_dataset(manifest_dir, t, lang):
    manifest_path = os.path.join(manifest_dir, f"cv-{lang}_")
    supervisions = SupervisionSet.from_jsonl(manifest_path+f"supervisions_{t}.jsonl.gz")
    recordings = RecordingSet.from_jsonl(manifest_path+f"recordings_{t}.jsonl.gz")
    cuts = CutSet.from_manifests(recordings, supervisions)
    return cuts.resample(SAMPLE_RATE)


if __name__ == "__main__":
    main()
