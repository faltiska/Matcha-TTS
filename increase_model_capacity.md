# Increasing MatchaTTS Model Capacity

---

## 1. Will increasing model capacity improve quality?
Yes—up to the point where there is sufficient training data, and GPU memory / training time remain practical.  
Bigger models mainly improve:  
  * **spectral detail** (timbre, noise floor), and  
  * **prosody modelling** (intonation, rhythm)  
  if extra parameters are placed in the right sub-modules.  
* Diminishing returns: doubling parameters rarely doubles MOS. Beyond ~50 M parameters for single-speaker systems, data cleanliness and training stability matter more than sheer size.

## 2. Main components in MatchaTTS
| Stage | File | Purpose |
|-------|------|---------|
| Text side | `matcha/models/components/text_encoder.py` | TextEncoder (Conv/MHA stack) + DurationPredictor |
| Latent/Flow | `matcha/models/components/flow_matching.py` | CFM / BASECFM transforms mel ↔ latent **z** |
| Acoustic decoder | `matcha/models/components/decoder.py` | ResNet / Conformer U-Net conditioned on diffusion timestep; outputs mel |
| Vocoder | `matcha/hifigan/*` | HiFi-GAN converts mel → waveform |
| Speaker embedding (optional) | within model | Embedding table if `n_spks > 1` |

_Default parameter split (~28 M parameters): TextEncoder ≈5 M, CFM ≈5 M, Decoder ≈15 M, others ≈3 M._

## 3. Where extra capacity pays off
### A. Natural timbre / lower noise floor
* **Decoder** is the main bottleneck – enlarge its base channel `dim`, number of ResNet/Conformer blocks, and attention-head dimension.  
* **CFM** latent size (`in_channels`, `out_channel`) should scale to avoid information loss.  
* If high-frequency artefacts persist, increase HiFi-GAN generator channels (`resblock_channels`, etc.).

### B. Richer prosody (intonation, rhythm)
* **TextEncoder** depth + attention heads matter most (`n_layers`, `hidden_channels`, `n_heads`).  
* Add/expand variance predictors (duration, pitch, energy) if ground truth is available.  
* Decoder Conformer attention blocks also help long-range prosody; raise `attention_head_dim` or stack more blocks.

## 4. Practical upgrade presets
| Module | Parameter | Default → Recommended | Δ Params | Δ VRAM |
|--------|-----------|-----------------|----------|--------|
| **Decoder** | `dim` | 256 → 384 | +2 M | +10 % |
|  | `num_res_blocks_per_level` | 2 → 3 | +2 M | +10 % |
|  | `attention_head_dim` | 64 → 80 | +\<1 M | negligible |
| **TextEncoder** | `n_layers` | 6 → 10 | +1.5 M | modest |
|  | `hidden_channels` | 256 → 384 | +1 M | modest |
|  | `n_heads` | 4 → 8 | +0.5 M | negligible |
| **CFM** | `in_channels / out_channel` | 128 → 192 | +1 M | small |

These settings fit on a 24 GB GPU for LJ-Speech-sized corpora.

## 5. When to stop scaling
* MOS/SSIM plateaus or even degrades beyond ~60 M parameters on corpora \<40 h.  
* Training instability (loss spikes) usually indicates data or LR schedule issues, not size.  
* Inference latency grows with Decoder depth; test for realtime use-cases.

## 6. Summary
* Increasing capacity **does** improve naturalness.  
* Prioritise **Decoder** (timbre) then **TextEncoder** (prosody).  
* Scale **CFM** and **HiFi-GAN** proportionally so they do not bottleneck.  
* Monitor data adequacy and compute budget; stop when returns taper.

---
