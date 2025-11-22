import torch
import torch.nn as nn
import json
from huggingface_hub import hf_hub_download
from pathlib import Path


class iSTFTNetWrapper(nn.Module):
    """
    Wrapper for iSTFTNet vocoder that takes mel-spectrograms and outputs audio.
    """
    def __init__(self, generator, stft, config):
        super().__init__()
        self.generator = generator
        self.stft = stft
        self.config = config

    def forward(self, mel):
        """
        Args:
            mel: Mel-spectrogram tensor of shape (batch, n_mels, time)
        
        Returns:
            audio: Waveform tensor of shape (batch, samples)
        """
        # Generator outputs magnitude spectrogram and phase
        spec, phase = self.generator(mel)

        # Apply inverse STFT to get waveform
        # Manually move window to correct device to fix device mismatch bug
        if hasattr(self.stft, 'window') and self.stft.window is not None:
            if self.stft.window.device != spec.device:
                self.stft.window = self.stft.window.to(spec.device)

        audio = self.stft.inverse(spec, phase)

        # Remove channel dimension if present
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        return audio

    @torch.no_grad()
    def inference(self, mel):
        """Convenience method for inference mode"""
        return self.forward(mel)


def get_default_config():
    """
    Returns the default iSTFTNet configuration matching the Uberduck models.
    These are trained on 22kHz audio with 2 upsampling layers.
    """
    return {
        "resblock": "1",
        "num_gpus": 1,
        "batch_size": 16,
        "learning_rate": 0.0002,
        "adam_b1": 0.8,
        "adam_b2": 0.99,
        "lr_decay": 0.999,
        "seed": 1234,

        # Uberduck models use 2 upsampling layers (not 4)
        # Total upsampling factor: 8 * 8 = 64
        "upsample_rates": [8, 8],
        "upsample_kernel_sizes": [16, 16],
        "upsample_initial_channel": 512,
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],

        "gen_istft_n_fft": 16,
        "gen_istft_hop_size": 4,

        "segment_size": 8192,
        "num_mels": 80,
        "n_fft": 1024,
        "hop_size": 256,
        "win_size": 1024,

        "sampling_rate": 22050,

        "fmin": 0,
        "fmax": 8000,
        "fmax_for_loss": None,

        "num_workers": 4,

        "dist_config": {
            "dist_backend": "nccl",
            "dist_url": "tcp://localhost:54321",
            "world_size": 1
        }
    }


class AttrDict(dict):
    """Dictionary that allows attribute-style access"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def load_vocoder(
        model_id="Uberduck/iSTFTNet",
        checkpoint_name="g_00030000",
        config_dict=None,
        device="cuda"
):
    """
    Load iSTFTNet vocoder from Hugging Face Hub.
    
    Args:
        model_id: HuggingFace model repository ID
        checkpoint_name: Name of checkpoint file. Available options from Uberduck/iSTFTNet:
            - "g_00010000" or "g_00030000" (general speech)
            - "music_g_00031000", "music_g_00033000", "music_g_00034000", "music_g_00037000" (music)
        config_dict: Optional config dictionary. If None, uses default config.
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        iSTFTNetWrapper: Wrapped vocoder model ready for inference
    """
    # Import required modules from rishikksh20's implementation
    # Note: You'll need to have the models.py and stft.py from 
    # https://github.com/rishikksh20/iSTFTNet-pytorch in your path
    try:
        from models import Generator
        from stft import TorchSTFT
    except ImportError:
        raise ImportError(
            "Please download models.py and stft.py from "
            "https://github.com/rishikksh20/iSTFTNet-pytorch "
            "and add them to your Python path"
        )

    # Use provided config or default
    if config_dict is None:
        config_dict = get_default_config()

    config = AttrDict(config_dict)

    # Download checkpoint from HuggingFace
    print(f"Downloading checkpoint {checkpoint_name} from {model_id}...")
    checkpoint_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)

    # Initialize generator
    print("Initializing generator...")
    generator = Generator(config).to(device)

    # Initialize STFT module
    stft = TorchSTFT(
        filter_length=config.gen_istft_n_fft,
        hop_length=config.gen_istft_hop_size,
        win_length=config.gen_istft_n_fft
    ).to(device)

    # Ensure all STFT buffers are on the correct device
    for name, buffer in stft.named_buffers():
        buffer.data = buffer.data.to(device)

    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint_dict = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint_dict['generator'])

    # Set to eval mode and remove weight norm for inference
    generator.eval()
    generator.remove_weight_norm()

    # Create wrapper
    wrapper = iSTFTNetWrapper(generator, stft, config)
    wrapper.eval()

    print("Vocoder loaded successfully!")
    return wrapper


def load_vocoder_with_local_checkpoint(checkpoint_path, config_dict=None, device="cuda"):
    """
    Load iSTFTNet vocoder from a local checkpoint file.
    
    Args:
        checkpoint_path: Path to local checkpoint file
        config_dict: Optional config dictionary. If None, uses default config.
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        iSTFTNetWrapper: Wrapped vocoder model ready for inference
    """
    try:
        from models import Generator
        from stft import TorchSTFT
    except ImportError:
        raise ImportError(
            "Please download models.py and stft.py from "
            "https://github.com/rishikksh20/iSTFTNet-pytorch "
            "and add them to your Python path"
        )

    # Use provided config or default
    if config_dict is None:
        config_dict = get_default_config()

    config = AttrDict(config_dict)

    # Initialize generator
    print("Initializing generator...")
    generator = Generator(config).to(device)

    # Initialize STFT module
    stft = TorchSTFT(
        filter_length=config.gen_istft_n_fft,
        hop_length=config.gen_istft_hop_size,
        win_length=config.gen_istft_n_fft
    ).to(device)

    # Ensure all STFT buffers are on the correct device
    for name, buffer in stft.named_buffers():
        buffer.data = buffer.data.to(device)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint_dict = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint_dict['generator'])

    # Set to eval mode and remove weight norm for inference
    generator.eval()
    generator.remove_weight_norm()

    # Create wrapper
    wrapper = iSTFTNetWrapper(generator, stft, config)
    wrapper.eval()

    print("Vocoder loaded successfully!")
    return wrapper


# Example usage:
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Option 1: Load from HuggingFace Hub (general speech model)
    vocoder = load_vocoder(
        model_id="Uberduck/iSTFTNet",
        checkpoint_name="g_00030000",  # or "g_00010000" for earlier checkpoint
        device=device
    )

    # Option 1b: Load music-trained model
    # vocoder = load_vocoder(
    #     model_id="Uberduck/iSTFTNet",
    #     checkpoint_name="music_g_00037000",  # or music_g_00031000, music_g_00033000, music_g_00034000
    #     device=device
    # )

    # Option 2: Load from local checkpoint
    # vocoder = load_vocoder_with_local_checkpoint(
    #     checkpoint_path="/path/to/checkpoint",
    #     device=device
    # )

    # Load mel spectrogram from file
    mel_path = "data/Corpus-small/mels/-944126494-0002-00001.npy"  # Change to your actual path

    # Determine file type and load accordingly
    if mel_path.endswith('.npy'):
        # If saved as numpy array (.npy)
        import numpy as np
        mel = torch.from_numpy(np.load(mel_path)).float().to(device)
    elif mel_path.endswith(('.pt', '.pth')):
        # If saved as PyTorch tensor (.pt, .pth)
        mel = torch.load(mel_path, map_location=device, weights_only=False)
    else:
        raise ValueError(f"Unsupported file format: {mel_path}")

    # Ensure correct shape: (batch, n_mels, time)
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)  # Add batch dimension

    print(f"Loaded mel spectrogram shape: {mel.shape}")
    print(f"Mel spectrogram - min: {mel.min():.4f}, max: {mel.max():.4f}, mean: {mel.mean():.4f}")

    # Check if mel needs to be transposed
    if mel.shape[1] != 80:
        print(f"Warning: Expected 80 mel bins, but got {mel.shape[1]}. Trying to transpose...")
        if mel.shape[2] == 80:
            mel = mel.transpose(1, 2)
            print(f"Transposed mel shape: {mel.shape}")

    # Undo dataset normalization if present (precompute_corpus.py normalizes with dataset mean/std)
    meta_path = Path(mel_path).parent / "metadata.json"
    mel_denorm_applied = False
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            mel_mean = float(meta.get("mel_mean", 0.0))
            mel_std = float(meta.get("mel_std", 1.0))
            # mel saved = (log_mel - mean)/std, so denormalize back to log_mel domain
            mel = mel * mel_std + mel_mean
            mel_denorm_applied = True
            print(f"Denormalized mel using mean={mel_mean:.6f}, std={mel_std:.6f}")
        except Exception as e:
            print(f"Warning: failed to read mel normalization metadata at {meta_path}: {e}")
    else:
        print(f"Warning: metadata.json not found next to mel file at {meta_path}. Assuming mel is not z-normalized.")

    # iSTFTNet expects log-mel amplitude (natural log) features without per-dataset z-norm.
    # Do NOT exponentiate to linear or convert from dB unless your mel was actually saved in those domains.

    # Generate audio
    with torch.no_grad():
        audio = vocoder(mel)
        print(f"Generated audio shape: {audio.shape}")

        # Save the audio
        import torchaudio
        torchaudio.save("output.wav", audio.cpu(), 22050)
