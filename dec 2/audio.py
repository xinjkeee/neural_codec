import torch
import torch.nn as nn
from enc_dec import Encoder, Decoder
from rvq import RVQ

class NeuralAudioCodec(nn.Module):
    def __init__(self, num_quantizers=4, codebook_size=1024, embed_dim=128,
                 hidden_dim=512):
        super().__init__()
        self.encoder = Encoder(in_channels=1, hidden_dim=hidden_dim, embed_dim=embed_dim)
        self.rvq = RVQ(num_quantizers, codebook_size, embed_dim)
        self.decoder = Decoder(embed_dim=embed_dim, hidden_dim=hidden_dim)

    def forward(self, x):
        z = self.encoder(x)
        z_q, rvq_loss, indices = self.rvq(z)
        x_hat = self.decoder(z_q)

        if x_hat.shape[-1] != x.shape[-1]:
            x_hat = x_hat[..., :x.shape[-1]]
        x_hat = torch.tanh(x_hat)
        return x_hat, rvq_loss, indices

    def rvq_reset_dead_codes(self, x):
        """Сброс мёртвых кодов с использованием энкодера."""
        with torch.no_grad():
            z = self.encoder(x)
        training = self.rvq.training
        self.rvq.eval()
        with torch.no_grad():
            _, _, _ = self.rvq(z, reset_dead=True)
        if training:
            self.rvq.train()