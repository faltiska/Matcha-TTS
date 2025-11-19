# Vocoder Max Frequency

Right now, the model uses an f_max of 8000, but I can go as high as 11025, but I have to find a Vocoder trained with 11025.
I should look into n_feats: 80, it is possible that increasing it may help. 

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