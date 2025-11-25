r"""
The file creates a pickle file where the values needed for loading of dataset is stored and the model can load it
when needed.

Parameters from hparam.py will be used
"""
import argparse
import json
import os
import sys
from pathlib import Path

import rootutils
import torch
from hydra import compose, initialize
from omegaconf import open_dict
from tqdm.auto import tqdm

from matcha.data.text_mel_datamodule import TextMelDataModule
from matcha.utils.logging_utils import pylogger

log = pylogger.get_pylogger(__name__)


def compute_data_statistics(data_loader: torch.utils.data.DataLoader, out_channels: int):
    """Generate data mean and standard deviation helpful in data normalisation

    Args:
        data_loader (torch.utils.data.Dataloader): _description_
        out_channels (int): mel spectrogram channels
    """
    total_mel_sum = 0
    total_mel_sq_sum = 0
    total_mel_len = 0

    for batch in tqdm(data_loader, leave=False):
        mels = batch["y"]
        mel_lengths = batch["y_lengths"]

        total_mel_len += torch.sum(mel_lengths)
        total_mel_sum += torch.sum(mels)
        total_mel_sq_sum += torch.sum(torch.pow(mels, 2))

    data_mean = total_mel_sum / (total_mel_len * out_channels)
    data_std = torch.sqrt((total_mel_sq_sum / (total_mel_len * out_channels)) - torch.pow(data_mean, 2))

    return {"mel_mean": data_mean.item(), "mel_std": data_std.item()}


def compute_f0_statistics(wav_paths, sample_rate, hop_length, f0_fmin, f0_fmax):
    """
    Compute F0 mean and std across all voiced frames in the dataset.
    Uses torchaudio.functional.detect_pitch_frequency as in precompute_corpus.py.
    """
    import torchaudio as ta
    import torch

    all_voiced_f0s = []

    for wav_path in tqdm(wav_paths, desc="F0 stats", leave=False):
        audio, sr = ta.load(str(wav_path))
        if sr != sample_rate:
            raise RuntimeError(f"Sample rate mismatch {sr} != {sample_rate} in {wav_path}")

        if audio.dim() == 2 and audio.size(0) > 1:
            audio = audio[:1, :]  # use only first channel 

        frame_time = hop_length / float(sample_rate)
        win_len = 5  # as in precompute_corpus, median smoothing

        f0 = ta.functional.detect_pitch_frequency(
            audio,
            sample_rate,
            frame_time=frame_time,
            win_length=win_len,
            freq_low=f0_fmin,
            freq_high=f0_fmax,
        )[0]
        # Only use voiced frames
        voiced = f0[f0 > 0]
        if len(voiced) > 0:
            all_voiced_f0s.append(voiced.cpu())

    if not all_voiced_f0s:
        raise RuntimeError("No voiced F0 frames found in the dataset.")

    all_voiced_f0s = torch.cat(all_voiced_f0s)
    f0_mean = all_voiced_f0s.mean().item()
    f0_std  = all_voiced_f0s.std(unbiased=False).item()  # match np.std, population std

    return {"f0_mean": f0_mean, "f0_std": f0_std}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-config",
        type=str,
        default="vctk.yaml",
        help="The name of the yaml config file under configs/data",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default="256",
        help="Can have increased batch size for faster computation",
    )

    args = parser.parse_args()
    output_file = Path(args.input_config).with_suffix(".json")

    with initialize(version_base="1.3", config_path="../../configs/data"):
        cfg = compose(config_name=args.input_config, return_hydra_config=True, overrides=[])

    root_path = rootutils.find_root(search_from=__file__, indicator=".project-root")

    with open_dict(cfg):
        del cfg["hydra"]
        del cfg["_target_"]
        cfg["data_statistics"] = None
        cfg["seed"] = 1234
        cfg["batch_size"] = args.batch_size
        cfg["train_filelist_path"] = str(os.path.join(root_path, cfg["train_filelist_path"]))
        cfg["valid_filelist_path"] = str(os.path.join(root_path, cfg["valid_filelist_path"]))
        cfg["load_durations"] = False

    mel_dir = cfg.get("mel_dir")
    f0_dir  = cfg.get("f0_dir")
    for check_dir in [mel_dir, f0_dir]:
        if check_dir and os.path.exists(check_dir):
            # not only stats are not needed; when the mel files exist, the method that returns mels will use them  
            # and the stats will be wrong; we must compute the stats from the wav files, not from mels. 
            print(f"ERROR: Directory '{check_dir}' already exists, will not compute statistics.")
            sys.exit(1)

    # Compute mel statistics
    text_mel_datamodule = TextMelDataModule(**cfg)
    text_mel_datamodule.setup()
    data_loader = text_mel_datamodule.train_dataloader()
    log.info("Dataloader loaded! Now computing mel stats...")
    params = compute_data_statistics(data_loader, cfg["n_feats"])
    print(params)

    # Compute F0 statistics
    def parse_filelist(filelist_path):
        filelist = []
        with open(filelist_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                filelist.append(line.strip().split("|")[0])  # Assume wav path is the first column
        return filelist

    train_wavs = parse_filelist(cfg["train_filelist_path"])
    valid_wavs = parse_filelist(cfg["valid_filelist_path"])
    wav_paths = list(sorted(set(train_wavs + valid_wavs)))
    sample_rate = cfg["sample_rate"]
    hop_length = cfg["hop_length"]
    # Use F0 extraction range from config or sensible defaults
    f0_fmin = float(cfg.get("f0_fmin", 50.0))
    f0_fmax = float(cfg.get("f0_fmax", 1100.0))

    f0_stats = compute_f0_statistics(wav_paths, sample_rate, hop_length, f0_fmin, f0_fmax)
    print(f0_stats)
    # Optionally, params.update(f0_stats) to have all together

    with open(output_file, "w", encoding="utf-8") as dumpfile:
        json.dump({**params, **f0_stats}, dumpfile)


if __name__ == "__main__":
    main()
