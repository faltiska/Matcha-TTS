# Vocoder

Matcha-TTS is an acoustic model that generates mel-spectrograms.
The final audio fidelity (timbre) is determined by the vocoder (e.g., HiFi-GAN, UnivNet).
Ensure you are using a state-of-the-art, well-trained vocoder that is robust to the spectrograms produced by your current Matcha-TTS checkpoint.
A mismatch or a poor vocoder is the primary source of metallic or artifact-laden voices.

## Vocos
Vocos is a strong alternative to HiFi-GAN that can work with Matcha-TTS. 
Vocos is designed as a fast neural vocoder that generates spectral coefficients instead of modeling audio samples in the time domain, 
facilitating rapid audio reconstruction through inverse Fourier transform, and aims to provide an alternative to HiFi-GAN that is faster and 
compatible with several TTS models Medium.
Key advantages of Vocos:

Speed: Vocos is faster than typical GAN-based vocoders like HiFi-GAN Medium
Compatibility: The Vocos mel-22khz version uses 80-bin mel spectrograms as acoustic features which are widespread in the TTS domain since 
the introduction of HiFi-GAN Hugging Face
Real-world usage: Teams have successfully used Vocos with Matcha-TTS and achieved significant performance improvements Medium

Important training consideration:
If you plan to train Vocos for use with Matcha-TTS, be aware that there can be mismatches between Matcha-TTS-generated mel spectrograms and 
Vocos's expected mel spectrogram parameters GitHub. The GitHub issue at gemelo-ai/vocos includes custom feature extractor code to properly 
align the mel spectrogram parameters between Matcha-TTS and Vocos.

BSC-LT provides a pre-trained Vocos model specifically for 80-bin mel spectrograms at 22kHz that works with Matcha-TTS
https://huggingface.co/BSC-LT/vocos-mel-22khz
Several production systems are already using this combination successfully

## HiFiGAN  

There are other pretrained models here:
https://github.com/jik876/hifi-gan?tab=readme-ov-file

## Max Frequency

Right now, the model uses an f_max of 8000, but I can go as high as 11025, but I have to find a Vocoder trained with 11025.
I should look into n_feats: 80, it is possible that increasing it may help. 

# Trainer FP precision

I changed from fp16 to bf16 in configs/trainer/default.yaml

# Increase model capacity
Increase Decoder Parameters (Model Capacity): The Matcha-TTS paper notes often mention that "light" models can be underparameterized. 
For high-quality, high-fidelity results, consider increasing the model capacity, particularly the decoder. 
One suggestion found in community notes is to increase the decoder's parameter count (e.g., from 10M to around 40M). 
This allows the model to better map the conditional flow matching to the highly complex mel-spectrogram target distribution, improving timbre fidelity.

# Prosody (intonation, rhythm, pauses)
This is often the hardest aspect to make sound truly human. 
Matcha-TTS uses Monotonic Alignment Search (MAS), which can be sensitive to small, high-quality, or emotionally diverse datasets.
Curate for Prosodic Diversity: Your "small high quality dataset" might lack the necessary prosodic variation. 
To improve prosody, the training data should contain a mix of:

Various Sentence Lengths: To teach natural phrasing and breathing.
Different Punctuation: To model terminal (period, question mark) and internal (comma) phrase boundaries.
Expressive Reading Styles: If possible, include some emotional or conversational speech.
Focus Fine-tuning on Prosody: One study specifically used Matcha-TTS as an example to show that fine-tuning with a customized training dataset tailored to 
emphasize prosodic cues (like pause duration) can significantly enhance the model's ability to produce consistent human prosodic patterns.
Consider an External Aligner: If you're having noticeable issues with choppy speech or strange pacing, the internal MAS alignment might be 
failing. In modern systems, using an ASR-based forced aligner (like MFA) to generate fixed, high-quality phoneme durations and alignments 
before training can often drastically stabilize and improve prosody. 
This would be a significant modification to the pipeline but could break the loss plateau.

# Validation frequency and early stopping

I have `check_val_every_n_epoch = 5`, in trainer/default

Validation in PyTorch-Lightning is only an evaluation pass; training does not halt automatically. Add an `EarlyStopping` callback to stop 
when the validation metric degrades. Setting `check_val_every_n_epoch = 1` runs validation every epoch, giving faster feedback and allowing 
EarlyStopping or ModelCheckpoint to act immediately; higher values trade evaluation overhead for slower feedback.