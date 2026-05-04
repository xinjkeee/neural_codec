import torch
import torchaudio
import torchaudio.functional as F
import torch.nn.functional as tF
import os
import glob
import random

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, folder, sample_rate=16000, duration=2, total_stride=160):
        self.files = glob.glob(os.path.join(folder, "**/*.flac"), recursive=True)
        self.sample_rate = sample_rate
        self.num_samples = (sample_rate * duration // total_stride) * total_stride

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        audio, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            audio = F.resample(audio, sr, self.sample_rate)

        audio = audio.mean(dim=0)

        peak = torch.max(torch.abs(audio)) + 1e-8
        audio = audio / peak * 0.95

        audio = F.preemphasis(audio, coeff=0.85)

        total_samples = audio.shape[-1]
        if total_samples > self.num_samples:
            max_start = total_samples - self.num_samples
            start_idx = random.randint(0, max_start)
            audio = audio[start_idx : start_idx + self.num_samples]
        else:
            pad = self.num_samples - total_samples
            audio = tF.pad(audio, (0, pad))

        return audio, 0