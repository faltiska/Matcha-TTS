# Multi-Resolution STFT Loss

## Overview

Multi-resolution STFT loss addresses spectral artifacts like metallic resonance by comparing spectrograms at multiple window sizes. 
STFT loss directly penalizes frequency-domain mismatches.
I am using 3 different windows sizes, to capture fine (512), medium (1024) and coarse (2048) frequency details.

## How It Works

1. Computes STFT at multiple window sizes: [512, 1024, 2048]
2. Compares magnitude spectrograms between predicted and target
3. Combines L1 losses across all resolutions
4. Added to CFM loss: `total_loss = cfm_loss + lambda_stft * stft_loss`

## Configuration

Add to your model config (e.g., `configs/train.yaml`):

```yaml
model:
  use_stft_loss: true
  lambda_stft: 0.1  # Start with 0.1, adjust based on results
```

Or via command line:

```bash
python -m matcha.train model.use_stft_loss=true model.lambda_stft=0.1
```

## Parameters

- `use_stft_loss`: Enable/disable multi-resolution STFT loss
- `lambda_stft`: Weight coefficient for STFT loss in total loss

## Tips

- Start with `lambda_stft=0.1` and adjust based on validation loss
- Higher values (0.2-0.5) give more weight to spectral quality
- STFT loss adds minimal computational overhead
- Works best combined with existing losses (prior, diff, pitch)
- Monitor `sub_loss/train_diff_loss` to see CFM loss component

## Implementation

- Loss module: `matcha/models/components/stft_loss.py`
- Integration: `matcha/models/components/flow_matching.py` (compute_loss method)
- Model config: `matcha/models/matcha_tts.py`
