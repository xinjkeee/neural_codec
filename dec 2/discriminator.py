import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

class DiscriminatorBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(1, 16, 15, 1, padding=7)),
            weight_norm(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            weight_norm(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            weight_norm(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            weight_norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1))
        ])
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        fmaps = []
        for conv in self.convs[:-1]:
            x = conv(x)
            x = self.lrelu(x)
            fmaps.append(x)
        out = self.convs[-1](x)
        return out, fmaps

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorBlock() for _ in range(3)])
        self.poolings = nn.ModuleList([
            nn.Identity(),
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])
    def forward(self, x):
        scores, fmaps = [], []
        for pool, disc in zip(self.poolings, self.discriminators):
            x_pooled = pool(x)
            score, fmap = disc(x_pooled)
            scores.append(score)
            fmaps.extend(fmap)
        return scores, fmaps

class PeriodDiscriminator(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmaps = []
        b, c, t = x.shape
        if t % self.period != 0:
            pad = self.period - (t % self.period)
            x = F.pad(x, (0, pad), "reflect")
            t = t + pad
        x = x.view(b, c, t // self.period, self.period)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.2)
            fmaps.append(x)
        x = self.conv_post(x)
        fmaps.append(x)
        return x, fmaps

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(2),
            PeriodDiscriminator(3),
            PeriodDiscriminator(5),
            PeriodDiscriminator(7),
            PeriodDiscriminator(11),
        ])
    def forward(self, x):
        scores, fmaps = [], []
        for d in self.discriminators:
            score, fmap = d(x)
            scores.append(score)
            fmaps.extend(fmap)
        return scores, fmaps

class MRSTFTDiscriminator(nn.Module):
    def __init__(self, fft_sizes=[512, 256], hop_sizes=[128, 64],
                 win_lengths=[512, 256]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        
        self.convs = nn.ModuleList()
        for _ in range(len(fft_sizes)):
            convs = nn.ModuleList([
                weight_norm(nn.Conv2d(2, 8, (3, 5), padding=(1, 2))),
                nn.LeakyReLU(0.2),
                weight_norm(nn.Conv2d(8, 16, (3, 5), stride=(1,2), padding=(1, 2))),
                nn.LeakyReLU(0.2),
                weight_norm(nn.Conv2d(16, 32, (3, 5), stride=(1,2), padding=(1, 2))),
                nn.LeakyReLU(0.2),
                weight_norm(nn.Conv2d(32, 32, (3, 3), padding=(1,1))),
                nn.LeakyReLU(0.2),
                weight_norm(nn.Conv2d(32, 1, (3, 3), padding=(1,1))),
            ])
            self.convs.append(convs)

    def forward(self, x):
        scores, fmaps = [], []
        
        for fft_size, hop_size, win_length, conv_layers in zip(
                self.fft_sizes, self.hop_sizes, self.win_lengths, self.convs):
            
            window = torch.hann_window(win_length, device=x.device)
            
            with torch.no_grad():
                stft = torch.stft(
                    x.squeeze(1), 
                    fft_size, 
                    hop_size, 
                    win_length, 
                    window, 
                    return_complex=True
                )
            
            real = stft.real.unsqueeze(1).detach()
            imag = stft.imag.unsqueeze(1).detach()
            mag = torch.cat([real, imag], dim=1) 
            
            for i, layer in enumerate(conv_layers):
                mag = layer(mag)
                if isinstance(layer, nn.LeakyReLU):
                    fmaps.append(mag)
            
            scores.append(mag)
        
        return scores, fmaps

class CompleteDiscriminator(nn.Module):
    def __init__(self, use_mrstft=True):
        super().__init__()
        self.msd = MultiScaleDiscriminator()
        self.mpd = MultiPeriodDiscriminator()
        self.use_mrstft = use_mrstft
        if use_mrstft:
            self.mrstft = MRSTFTDiscriminator()

    def forward(self, x):
        msd_scores, msd_fmaps = self.msd(x)
        mpd_scores, mpd_fmaps = self.mpd(x)
        
        all_scores = msd_scores + mpd_scores
        all_fmaps = msd_fmaps + mpd_fmaps
        
        if self.use_mrstft:
            mrstft_scores, mrstft_fmaps = self.mrstft(x)
            all_scores = all_scores + mrstft_scores
            all_fmaps = all_fmaps + mrstft_fmaps
        
        return all_scores, all_fmaps