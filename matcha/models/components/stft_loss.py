import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss for catching spectral artifacts.
    
    Computes STFT at multiple window sizes and measures magnitude differences.
    This catches harmonic distortions and metallic resonance that MSE loss misses.
    """
    
    def __init__(self, fft_sizes=[512, 1024, 2048], hop_sizes=None, win_lengths=None):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes or [fs // 4 for fs in fft_sizes]
        self.win_lengths = win_lengths or fft_sizes
        
        self.hann_windows = {}
    
    def _get_hann_window(self, fft_size, device):
        """Get or create Hann window for given FFT size."""
        key = (fft_size, device.type, device.index if device.index is not None else 0)
        if key not in self.hann_windows:
            self.hann_windows[key] = torch.hann_window(fft_size).to(device)
        
        return self.hann_windows[key]
    
    def forward(self, pred, target, mask=None):
        """Compute multi-resolution STFT loss.
        
        Args:
            pred: Predicted mel-spectrogram (batch, n_feats, time)
            target: Target mel-spectrogram (batch, n_feats, time)
            mask: Optional mask (batch, 1, time)
        
        Returns:
            Multi-resolution STFT loss
        """
        # Apply mask to inputs if provided
        if mask is not None:
            pred = pred * mask
            target = target * mask
        
        loss = 0.0
        
        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            # Compute STFT magnitude
            pred_stft = self._stft(pred, fft_size, hop_size, win_length)
            target_stft = self._stft(target, fft_size, hop_size, win_length)
            
            # L1 loss on magnitude
            loss += F.l1_loss(pred_stft, target_stft, reduction="mean")
        
        return loss / len(self.fft_sizes)
    
    def _stft(self, x, fft_size, hop_size, win_length):
        """Compute STFT magnitude."""
        device = x.device
        dtype = x.dtype
        window = self._get_hann_window(fft_size, device)
        
        # Flatten batch and feature dimensions for STFT
        b, c, t = x.shape
        x_flat = x.reshape(b * c, t)
        
        # STFT doesn't support bf16, convert to float32
        if dtype == torch.bfloat16:
            x_flat = x_flat.float()
        
        # Compute STFT
        spec = torch.stft(
            x_flat,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        
        # Return magnitude in original dtype
        mag = torch.abs(spec)
        return mag.to(dtype) if dtype == torch.bfloat16 else mag
