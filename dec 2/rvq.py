import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizerEMA(nn.Module):
    def __init__(self, codebook_size, embed_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, embed_dim)
        nn.init.uniform_(self.codebook.weight, -0.1, 0.1)

        self.register_buffer('cluster_size', torch.zeros(codebook_size))
        self.register_buffer('embed_sum', self.codebook.weight.clone())
        self.decay = decay
        self.eps = eps
        self.codebook_size = codebook_size

    def forward(self, z):
        # z: [B, C, T]
        B, C, T = z.shape
        z_flat = z.transpose(1, 2).contiguous().view(-1, C)  # [B*T, C]

        # Расстояния
        d = (z_flat**2).sum(dim=1, keepdim=True) + \
            (self.codebook.weight**2).sum(dim=1) - \
            2 * torch.matmul(z_flat, self.codebook.weight.t())
        idx = d.argmin(dim=1)  # [B*T]

        z_q = self.codebook(idx).view(B, T, C).transpose(1, 2)  # [B, C, T]

        if self.training:
            one_hot = F.one_hot(idx, self.codebook_size).float()
            cluster_size = one_hot.sum(dim=0)
            embed_sum = torch.matmul(one_hot.t(), z_flat)

            self.cluster_size.data.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
            self.embed_sum.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n
            embed_normalized = self.embed_sum / cluster_size.unsqueeze(1)
            self.codebook.weight.data.copy_(embed_normalized)

        commitment_loss = F.mse_loss(z_q.detach(), z)
        return z_q, commitment_loss, idx

    def reset_dead_codes(self, z_flat_current, threshold=2):

        with torch.no_grad():
            usage = self.cluster_size
            dead = usage < threshold
            n_dead = dead.sum().item()
            
            if n_dead > 0:
                n_available = z_flat_current.size(0)
                
                if n_available >= n_dead:
                    perm = torch.randperm(n_available, device=z_flat_current.device)[:n_dead]
                    new_codes = z_flat_current[perm].clone()
                else:
                    repeats = (n_dead // n_available) + 1
                    new_codes = z_flat_current.repeat(repeats, 1)[:n_dead].clone()
                
                self.codebook.weight[dead] = new_codes
                self.cluster_size[dead] = 1.0
                self.embed_sum[dead] = self.codebook.weight[dead].clone()
                
                return n_dead
            return 0


class RVQ(nn.Module):
    def __init__(self, num_quantizers=4, codebook_size=1024, embed_dim=128):
        super().__init__()
        self.quantizers = nn.ModuleList([
            VectorQuantizerEMA(codebook_size, embed_dim) for _ in range(num_quantizers)
        ])

    def forward(self, z, reset_dead=False):
        quantized_out = 0
        residual = z
        total_commit_loss = 0
        all_indices = []

        for i, quantizer in enumerate(self.quantizers):
            z_q, commit_loss, idx = quantizer(residual)
            quantized_out = quantized_out + z_q
            residual = residual - z_q.detach()
            total_commit_loss = total_commit_loss + commit_loss

            if reset_dead:
                B, C, T = residual.shape
                residual_flat = residual.transpose(1, 2).contiguous().view(-1, C)
                n_reset = quantizer.reset_dead_codes(residual_flat)
                if n_reset > 0:
                    pass 
            
            all_indices.append(idx)

        total_commit_loss = total_commit_loss / len(self.quantizers)
        quantized_out = z + (quantized_out - z).detach()
        return quantized_out, total_commit_loss, all_indices