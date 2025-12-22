""" from https://github.com/keithito/tacotron

Phonemizers convert input text to phonemes at both training and eval time.

Phonemizers can be selected by passing a comma-delimited list of phonemizer names as the "phonemizers"
hyperparameter in your corpus descriptor:
phonemizers: [multilingual_phonemizer] 
"""

import logging
import re

import phonemizer

logging.basicConfig()
logger = logging.getLogger("phonemizer")
logger.setLevel(logging.ERROR)

# Initializing the phonemizer globally significantly improves the speed.
phonemizers = {}
phonemizers["en-us"] = phonemizer.backend.EspeakBackend(
    language="en-us",
    preserve_punctuation=True,
    with_stress=True,
    language_switch="remove-flags",
    logger=logger,
)

# eSpeak supports only Hiragana or Katakana for Japanese, but does not support Kanji.
# List of supported languages: 
# English, Spanish, Portuguese, French, German, Italian, Romanian, Japanese, Hebrew
def multilingual_phonemizer(text, language="en-us"):
    phonemizer = phonemizers[language]
    if not phonemizer:
        raise Exception(f"Unsupported {language=}")
    phonemes = phonemizer.phonemize([text], strip=True, njobs=1)[0]
    return phonemes