import torch
import torch.nn as nn
import torch.nn.functional as F

class STFTLoss(nn.Module):
    def __init__(self, fft_size, shift_size, win_length):
        super().__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, x, y):
        x_stft = torch.stft(x.squeeze(1), self.fft_size, self.shift_size, self.win_length,
                            self.window, return_complex=True)
        y_stft = torch.stft(y.squeeze(1), self.fft_size, self.shift_size, self.win_length,
                            self.window, return_complex=True)

        x_mag = torch.sqrt(x_stft.abs()**2 + 1e-7)
        y_mag = torch.sqrt(y_stft.abs()**2 + 1e-7)

        sc_loss = torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")
        log_loss = F.l1_loss(torch.log(y_mag), torch.log(x_mag))
        return sc_loss + log_loss

class MultiScaleSTFTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.stft_losses = nn.ModuleList([
            STFTLoss(2048, 512, 2048),
            STFTLoss(1024, 256, 1024),
            STFTLoss(512, 128, 512),
            STFTLoss(256, 64, 256),
            STFTLoss(128, 32, 128),
        ])

    def forward(self, x_hat, x):
        loss = 0.0
        for f in self.stft_losses:
            loss += f(x_hat, x)
        return loss / len(self.stft_losses)