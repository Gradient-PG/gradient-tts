# Gradient TTS
An implementation of *Wang, C., Chen, S., Wu, Y., (2023), Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers*

## Dataset
[polish dataset from common voice](https://commonvoice.mozilla.org/pl/datasets)
## Weights
Link to [download](TODO) the weights
## Dependencies
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
also [espeak](https://aur.archlinux.org/packages/espeak) and [mpdecimal](https://archlinux.org/packages/core/x86_64/mpdecimal)
## Prepare dataset
extract common voice to any directory, then fix it [lhotse bug](https://github.com/lhotse-speech/lhotse/pull/1328)
```
./dataset/fix_common_voice.sh
```
set args in [common_voice.py](tts/dataset/common_voice.py), then
```
python -m tts.dataset.common_voice
```

## Result
TODO
