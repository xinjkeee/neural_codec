# Neural Audio Codec

Neural audio codec built from scratch in PyTorch. Compresses raw 16kHz speech into discrete tokens and reconstructs it back.

## Architecture

- **Encoder** — 4× downsampling conv blocks with residual connections (total stride 160×)
- **RVQ** — Residual Vector Quantization, 4 quantizers, codebook size 512, EMA updates, dead code resetting
- **Decoder** — symmetric upsampling with transposed convolutions
- **Discriminators** — Multi-Scale (MSD) + Multi-Period (MPD) + MRSTFT

## Training

Warmup (5 epochs, reconstruction only) → GAN ramp-up (10 epochs) → full GAN (50 epochs total).

Losses: Multi-Scale STFT (5 scales), L1, feature matching, adversarial.

```bash
pip install -r requirements.txt
python main.py  # trains on ./dev-clean
```

## Dataset

LibriSpeech `dev-clean`, 16kHz, 2-second clips.
