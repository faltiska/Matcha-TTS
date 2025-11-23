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
        # Prefer absolute import when package is imported (e.g., via matcha.cli)
        from matcha.istftnet.models import Generator  # type: ignore
        from matcha.istftnet.stft import TorchSTFT  # type: ignore
    except Exception:
        try:
            # When imported as a package, relative imports also work
            from .models import Generator  # type: ignore
            from .stft import TorchSTFT  # type: ignore
        except Exception:
            # When executed as a script (python matcha/istftnet/wrapper.py), fall back to local imports
            try:
                from models import Generator  # type: ignore
                from stft import TorchSTFT  # type: ignore
            except Exception as e:
                raise ImportError(
                    "Could not import iSTFTNet modules. Ensure matcha/istftnet is a package and "
                    "import via 'from matcha.istftnet import wrapper' or run as a module.\n"
                    "Expected files: matcha/istftnet/models.py and matcha/istftnet/stft.py"
                ) from e

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
        # Prefer absolute import when package is imported (e.g., via matcha.cli)
        from matcha.istftnet.models import Generator  # type: ignore
        from matcha.istftnet.stft import TorchSTFT  # type: ignore
    except Exception:
        try:
            # When imported as a package, relative imports also work
            from .models import Generator  # type: ignore
            from .stft import TorchSTFT  # type: ignore
        except Exception:
            # When executed as a script (python matcha/istftnet/wrapper.py), fall back to local imports
            try:
                from models import Generator  # type: ignore
                from stft import TorchSTFT  # type: ignore
            except Exception as e:
                raise ImportError(
                    "Could not import iSTFTNet modules. Ensure matcha/istftnet is a package and "
                    "import via 'from matcha.istftnet import wrapper' or run as a module.\n"
                    "Expected files: matcha/istftnet/models.py and matcha/istftnet/stft.py"
                ) from e

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
