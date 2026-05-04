import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import gc

from dataset import AudioDataset
from audio import NeuralAudioCodec
from loss import MultiScaleSTFTLoss
from discriminator import CompleteDiscriminator

import os

torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

TOTAL_STRIDE = 160
BATCH_SIZE = 16
EPOCHS = 50
WARMUP_EPOCHS = 5
LR_G = 2e-4
LR_D = 1e-4         
BETAS = (0.5, 0.9)

HIDDEN_DIM = 512
EMBED_DIM = 128
CODEBOOK_SIZE = 512
NUM_QUANTIZERS = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

dataset = AudioDataset("./dev-clean", sample_rate=16000, duration=2, total_stride=TOTAL_STRIDE)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, num_workers=2,
    pin_memory=True, shuffle=True
)

model = NeuralAudioCodec(
    num_quantizers=NUM_QUANTIZERS,
    codebook_size=CODEBOOK_SIZE,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM
).to(device)
discriminator = CompleteDiscriminator(use_mrstft=False).to(device)
stft_criterion = MultiScaleSTFTLoss().to(device)
l1_criterion = nn.L1Loss().to(device)

opt_g = torch.optim.AdamW(model.parameters(), lr=LR_G, betas=BETAS, weight_decay=1e-4)
opt_d = torch.optim.AdamW(discriminator.parameters(), lr=LR_D, betas=BETAS, weight_decay=1e-4)

scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=EPOCHS, eta_min=1e-5)
scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=EPOCHS, eta_min=1e-5)

os.makedirs("checkpoints", exist_ok=True)

W_STFT = 3.0
W_L1 = 1.0
W_RVQ = 10.0

MAX_W_ADV = 0.5
MAX_W_FM = 2.0
GAN_RAMP_EPOCHS = 10  

print(f"Параметры модели: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

for epoch in range(EPOCHS):
    model.train()
    discriminator.train()
    total_g_loss = 0.0
    total_d_loss = 0.0
    total_perplexity = [0.0] * NUM_QUANTIZERS

    for batch_idx, (audio, _) in enumerate(dataloader):
        audio = audio.unsqueeze(1).to(device)

        if epoch < WARMUP_EPOCHS:
            opt_g.zero_grad()
            with torch.amp.autocast('cuda', enabled=True):
                x_hat, rvq_loss, indices = model(audio)
                l1_loss = l1_criterion(x_hat, audio)
                stft_loss = stft_criterion(x_hat, audio)
                loss_g = W_STFT * stft_loss + W_L1 * l1_loss + W_RVQ * rvq_loss

            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt_g.step()
            total_g_loss += loss_g.item()

            if batch_idx % 200 == 0:
                model.rvq_reset_dead_codes(audio)
                torch.cuda.empty_cache()

        else:
            if epoch < WARMUP_EPOCHS + GAN_RAMP_EPOCHS:
                alpha = (epoch - WARMUP_EPOCHS) / GAN_RAMP_EPOCHS
                w_adv = MAX_W_ADV * alpha
                w_fm = MAX_W_FM * alpha
            else:
                w_adv = MAX_W_ADV
                w_fm = MAX_W_FM

            with torch.amp.autocast('cuda', enabled=True):
                x_hat, rvq_loss, indices = model(audio)

            opt_d.zero_grad()
            with torch.amp.autocast('cuda', enabled=True):
                real_scores, _ = discriminator(audio)
                fake_scores_d, _ = discriminator(x_hat.detach())
                loss_d = sum(F.mse_loss(rs, torch.ones_like(rs)) +
                             F.mse_loss(fs, torch.zeros_like(fs))
                             for rs, fs in zip(real_scores, fake_scores_d))

            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 10.0)
            opt_d.step()
            total_d_loss += loss_d.item()

            opt_g.zero_grad()
            with torch.amp.autocast('cuda', enabled=True):
                fake_scores, fake_fmaps = discriminator(x_hat)
                with torch.no_grad():
                    _, real_fmaps = discriminator(audio)

                stft_loss = stft_criterion(x_hat, audio)
                l1_loss = l1_criterion(x_hat, audio)

                fm_loss = sum(F.l1_loss(f, r) for f, r in zip(fake_fmaps, real_fmaps))
                fm_loss = fm_loss / max(len(fake_fmaps), 1)

                adv_loss = sum(F.mse_loss(fs, torch.ones_like(fs)) for fs in fake_scores)

                loss_g = (W_STFT * stft_loss + W_L1 * l1_loss +
                          W_RVQ * rvq_loss + w_fm * fm_loss + w_adv * adv_loss)

            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt_g.step()
            total_g_loss += loss_g.item()

            if batch_idx % 200 == 0:
                model.rvq_reset_dead_codes(audio)
                torch.cuda.empty_cache()

        with torch.no_grad():
            for i, idx in enumerate(indices):
                one_hot = F.one_hot(idx, CODEBOOK_SIZE).float()
                probs = one_hot.mean(0)
                perp = torch.exp(-(probs * torch.log(probs + 1e-10)).sum())
                total_perplexity[i] += perp.item()

        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    scheduler_g.step()
    scheduler_d.step()

    avg_g = total_g_loss / len(dataloader)
    avg_d = total_d_loss / len(dataloader) if epoch >= WARMUP_EPOCHS else 0.0
    avg_perp = [p / len(dataloader) for p in total_perplexity]

    w_adv_str = f" | w_adv={MAX_W_ADV:.1f}" if epoch >= WARMUP_EPOCHS + GAN_RAMP_EPOCHS else ""
    print(f"Epoch {epoch+1}/{EPOCHS} | Gen: {avg_g:.4f} | Disc: {avg_d:.4f}"
          f" | Perp: {[f'{p:.0f}' for p in avg_perp]}{w_adv_str}")

    if (epoch+1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'discriminator': discriminator.state_dict(),
            'optimizer_g': opt_g.state_dict(),
            'optimizer_d': opt_d.state_dict(),
        }, f"checkpoints/codec_gan_epoch_{epoch+1}.pt")

torch.save(model.state_dict(), "neural_codec_final.pt")
print("Обучение завершено!")

def save_reconstruction(model, dataset, index=0, device="cuda"):
    model.eval()
    with torch.no_grad():
        audio, _ = dataset[index]
        audio = audio.unsqueeze(0).unsqueeze(0).to(device)
        x_hat, _, _ = model(audio)

        x_hat = torchaudio.functional.deemphasis(x_hat.squeeze(1), coeff=0.85).unsqueeze(1)
        origin = torchaudio.functional.deemphasis(audio.squeeze(1), coeff=0.85).unsqueeze(1)

        x_hat = x_hat.squeeze(0).cpu()
        origin = origin.squeeze(0).cpu()
        torchaudio.save("original.wav", origin, 16000)
        torchaudio.save("reconstructed.wav", x_hat, 16000)

save_reconstruction(model, dataset, index=0, device=device)