import sys
import os
import wave
import argparse

def get_wav_duration(file_path):
    """Returns the duration of a wav file in seconds."""
    try:
        with wave.open(file_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            return frames / float(rate)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def main():
    """
    python -m matcha.utils.filter_by_wav_duration ./data/corpus/train.csv 10
    """
    parser = argparse.ArgumentParser(description="Filter metadata based on wav duration.")
    parser.add_argument("input_file", help="Path to the input text file")
    parser.add_argument("max_duration", type=float, help="Max duration threshold in seconds")

    args = parser.parse_args()

    # Determine output path and the base directory of the input file
    output_file = args.input_file + ".out"
    input_dir = os.path.dirname(os.path.abspath(args.input_file))

    lines_processed = 0
    lines_kept = 0

    with open(args.input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            parts = line.split('|')
            full_path = os.path.join(input_dir, "wav", parts[0]) + ".wav"

            duration = get_wav_duration(full_path)

            if duration is not None and duration < args.max_duration:
                f_out.write(line + '\n')
                lines_kept += 1

            lines_processed += 1

    print(f"Process complete.")
    print(f"Saved to: {output_file}")
    print(f"Kept {lines_kept} of {lines_processed} lines.")

if __name__ == "__main__":
    main()