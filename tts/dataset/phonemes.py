"""
phonemizes dataset
defines phonemizer (text to phonemes)
@Author: Dominik Lau
@Links: 
    adapted from https://github.com/lifeiteng/vall-e/blob/9c69096d603ce13174fb5cb025f185e2e9b36ac7/valle/data/tokenizer.py
    reader: http://ipa-reader.xyz/
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Pattern, Union

import numpy as np
from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.language_switch import LanguageSwitch
from phonemizer.backend.espeak.words_mismatch import WordMismatch
from phonemizer.punctuation import Punctuation
from common_voice import train_cuts, test_cuts
from lhotse import SupervisionSet


PHONEMIZATION_DATASET_OUTPUT_DIR="data/phonemized"


class TextPhonemizer(EspeakBackend):
    """
    text to phonemes in IPA notation
    example: Eśąćż, źdźbło jabłko -> ɛɕɔɲtɕʃ, ʑdʑbwɔ japkɔ
    """
    def __init__(
        self,
        language="pl"
    ):
        super().__init__(language,
            punctuation_marks=Punctuation.default_marks(),
            preserve_punctuation=True,
            with_stress=False,
            tie=False,
            language_switch="keep-flags",
            words_mismatch="ignore"
        )
   
    def __call__(self, text, strip=True):
        return self.phonemize(
            [text.strip()], strip=True
        )[0]


def main():
    __prepare("data/prepared", "train")
    __prepare("data/prepared", "test")


def __prepare(manifest_dir, t):
    supervisions = SupervisionSet.from_jsonl(f"{manifest_dir}/cv-pl_supervisions_{t}.jsonl.gz")

    pho = TextPhonemizer()
    for supervision in supervisions:
        supervision.custom["phonemes"] = {"text": pho(supervision.text)}
    supervisions.to_file(f"{PHONEMIZATION_DATASET_OUTPUT_DIR}/cv-pl_supervisions_{t}.jsonl_gz")


if __name__ == "__main__":
    main()