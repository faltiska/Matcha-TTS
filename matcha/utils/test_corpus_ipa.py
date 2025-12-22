"""
Test corpus IPA symbols against symbols.py.

Usage:
  python -m matcha.utils.test_corpus_ipa -i corpus.csv
  python -m matcha.utils.test_corpus_ipa -i corpus.csv --language en-us
"""

import argparse
from pathlib import Path

from matcha.text.phonemizers import multilingual_phonemizer
from matcha.text.symbols import symbols


def parse_filelist(filelist_path: Path, split_char: str = "|"):
    with open(filelist_path, encoding="utf-8") as f:
        return [line.strip().split(split_char) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Test corpus IPA symbols against symbols.py.")
    parser.add_argument("-i", "--input", required=True, help="Path to corpus CSV file")
    parser.add_argument("--language", default="en-us", help="Language code (default: en-us)")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    entries = parse_filelist(input_path)
    total = len(entries)
    symbol_set = set(symbols)
    unknown_symbols = set()

    print(f"[test_corpus_ipa] Input: {input_path}")
    print(f"[test_corpus_ipa] Language: {args.language}")
    print(f"[test_corpus_ipa] Processing {total} entries...")

    for i, parts in enumerate(entries, start=1):
        if len(parts) < 3:
            print(f"[test_corpus_ipa] WARNING: Skipping malformed line {i}: {parts}")
            continue

        text = parts[2]
        ipa = multilingual_phonemizer(text, language=args.language)

        for char in ipa:
            if char not in symbol_set:
                unknown_symbols.add(char)

        if i % 100 == 0:
            print(f"\r[test_corpus_ipa] {i}/{total} done.", end="", flush=True)

    print(f"\n[test_corpus_ipa] Finished.")

    if unknown_symbols:
        print(f"[test_corpus_ipa] WARNING: Found {len(unknown_symbols)} unknown symbols not in symbols.py:")
        print(f"[test_corpus_ipa] {sorted(unknown_symbols)}")
    else:
        print(f"[test_corpus_ipa] All symbols are valid.")


if __name__ == "__main__":
    main()
