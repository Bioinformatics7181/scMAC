import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------
# Activation Functions
# ------------------------

class MeanAct(nn.Module):
    """Activation for decoder mean output: exp with clamping."""
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(nn.Module):
    """Activation for decoder dispersion output: softplus with clamping."""
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)

# ------------------------
# Attention & Embedding Blocks
# ------------------------

class ScaleAttention(nn.Module):
    """
    Attention mechanism to fuse multi-branch convolution features.
    Input shape: (N, B, F), where B is number of convolution branches.
    """
    def __init__(self, feature_dim):
        super(ScaleAttention, self).__init__()
        self.fc = nn.Linear(feature_dim, 1)

    def forward(self, features):
        scores = self.fc(features)  # (N, B, 1)
        scores = scores - torch.max(scores, dim=1, keepdim=True).values
        scores = torch.clamp(scores, min=-50, max=50)
        weights = torch.softmax(scores, dim=1)
        weights = weights + 1e-8
        weights = weights / weights.sum(dim=1, keepdim=True)
        fused = (features * weights).sum(dim=1)  # (N, F)
        return fused, weights


class GeneEncoder(nn.Module):
    """Gene encoder that projects high-dimensional embeddings to a latent space."""
    def __init__(self, in_dim=3072, hid=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, out_dim)
        )

    def forward(self, E):
        return self.net(E)


class CrossAttn(nn.Module):
    """Cross-attention from cell features to gene embeddings."""
    def __init__(self, q_dim, k_dim, attn_dim):
        super().__init__()
        self.Wq = nn.Linear(q_dim, attn_dim, bias=False)
        self.Wk = nn.Linear(k_dim, attn_dim, bias=False)
        self.Wv = nn.Linear(k_dim, attn_dim, bias=False)
        self.scale = attn_dim ** 0.5

    def forward(self, Q, K, V):
        q = self.Wq(Q).unsqueeze(1)            # (N, 1, h)
        k = self.Wk(K).unsqueeze(0)            # (1, k, h)
        v = self.Wv(V).unsqueeze(0)            # (1, k, h)
        attn = torch.softmax((q @ k.transpose(-1, -2)) / self.scale, dim=-1)  # (N, 1, k)
        ctx = (attn @ v).squeeze(1)            # (N, h)
        return ctx, attn.squeeze(1)            # (N, h), (N, k)

# ------------------------
# Main Model
# ------------------------

class scMACLLMNet(nn.Module):
    """
    scMAC-LLM: A Multi-scale Attention-guided Convolutional autoencoder for single-cell RNA-seq data analysis enhanced by LLM.
    Input: 
        x_cnt (N, 1, k) - raw count matrix
        E_raw (k, embed_dim) - pretrained gene embeddings
    Output: 
        mean, dispersion, latent, E_low, x_recon, e_recon, attn_dict
    """
    def __init__(self, k=5000,
                 big_kernel_sizes=[500, 1000, 1500, 2000, 2500, 3000],
                 small_kernel_sizes=[2, 4],
                 num_filters_coarse=16, num_filters_fine=32,
                 embed_dim=3072, attn_hidden_dim=128, attn_dim=64,
                 latent_dim=64, dec_hidden_dim=128):
        super().__init__()
        self.k = k
        self.embed_dim = embed_dim
        self.attn_dim = attn_dim
        self.num_filters_coarse = num_filters_coarse
        self.num_filters_fine = num_filters_fine
        self.F = num_filters_fine * len(small_kernel_sizes)

        self.in_channels = 1
        self.pool = nn.AdaptiveMaxPool1d(1)

        assert embed_dim > 0, "Embed_dim must be greater than 0 even when not using embeddings."

        # Build convolutional branches
        self.coarse_convs = self._build_coarse_convs(big_kernel_sizes)
        self.fine_convs = self._build_fine_convs(big_kernel_sizes, small_kernel_sizes)

        self.attn_scale = ScaleAttention(self.F)
        self.gene_enc = GeneEncoder(embed_dim, attn_hidden_dim, attn_dim)
        self.cross_attn = CrossAttn(self.F, attn_dim, attn_dim)

        # Encoder and decoder
        self.fc_enc_with_ctx = nn.Linear(self.F + attn_dim, latent_dim)
        self.fc_enc_no_ctx = nn.Linear(self.F, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, dec_hidden_dim)
        self.dec_mean = nn.Linear(dec_hidden_dim, k)
        self.dec_disp = nn.Linear(dec_hidden_dim, k)
        self.dec_embed = nn.Linear(dec_hidden_dim, k * attn_dim)
        self.bias_dec1 = nn.Parameter(torch.zeros(attn_hidden_dim))
        self.bias_dec2 = nn.Parameter(torch.zeros(embed_dim))


    def _build_coarse_convs(self, kernel_sizes):
        return nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.in_channels, self.num_filters_coarse, K, stride=500),
                nn.BatchNorm1d(self.num_filters_coarse),
                nn.LeakyReLU(0.01)
            ) for K in kernel_sizes
        ])

    def _build_fine_convs(self, branches, fine_ks):
        return nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(self.num_filters_coarse, self.num_filters_fine, ks),
                    nn.BatchNorm1d(self.num_filters_fine),
                    nn.LeakyReLU(0.01)
                ) for ks in fine_ks
            ]) for _ in branches
        ])

    def forward(self, x_cnt, E_raw=None):
        """
        Forward pass.

        Args:
            x_cnt: Tensor of shape (N, 1, k), cell input counts
            E_raw: (optional) Tensor of shape (k, embed_dim), gene embeddings

        Returns:
            Depending on E_raw:
                With embedding (scMAC-LLM): mean, disp, latent, E_low, x_recon, e_recon, attn_dict
                Without embedding (scMAC): mean, disp, latent, None, x_recon, None, attn_dict
        """
        N = x_cnt.size(0)

        # 1. Multi-scale convolution and fusion
        branch = []
        for c_conv, f_list in zip(self.coarse_convs, self.fine_convs):
            c_feat = c_conv(x_cnt)                                                   # (N, Cc, Lc)
            f_cat = torch.cat([self.pool(f(c_feat)).squeeze(2) for f in f_list], 1)  # (N, F)
            branch.append(f_cat)
        feats = torch.stack(branch, 1)                                               # (N, B, F)
        fused, w_scale = self.attn_scale(feats)                                      # (N, F)

        attn_dict = {"scale_attn": w_scale}
        E_low, e_recon = None, None

        if E_raw is not None:
            # With embedding:
            # 2. Gene encoding
            E_low = self.gene_enc(E_raw)                                             # (k, attn_dim)

            # 3. Cross-attention from cell features to gene embeddings.
            ctx, gene_attn = self.cross_attn(fused, E_low, E_low)                    # (N, attn_dim)
            fused_all = torch.cat([fused, ctx], 1)                                   # (N, F + attn_dim)
            latent = F.relu(self.fc_enc_with_ctx(fused_all))                         # (N, latent_dim)
            attn_dict["gene_attn"] = gene_attn
        else:
            # Without embedding:
            # 4. Latent encoding
            latent = F.relu(self.fc_enc_no_ctx(fused))                               # (N, latent_dim)

        # 5. Count decoder
        h = F.relu(self.fc_dec(latent))                                              # (N, dec_hidden_dim)
        mean = MeanAct()(self.dec_mean(h))                                           # (N, k)
        disp = DispAct()(self.dec_disp(h))                                           # (N, k)
        x_recon = mean.view(N, 1, self.k)                                            # (N, 1, k)

        # 6. (optional) Embedding decoder (reconstruct gene embeddings)
        if E_raw is not None:
            e_pred = self.dec_embed(h).view(N, self.k, self.attn_dim)                # (N, k, attn_dim)
            e_low_rec = e_pred.mean(0)                                               # (k, attn_dim)

            # Use encoder weights to decode back to raw gene space
            weight_dec1 = self.gene_enc.net[2].weight                                # (attn_dim, hid)
            e_mid = F.relu(e_low_rec @ weight_dec1 + self.bias_dec1)                 # (k, hid)
            
            weight_dec2 = self.gene_enc.net[0].weight                                # (hid, embed_dim)
            e_recon = e_mid @ weight_dec2 + self.bias_dec2                           # (k, embed_dim)

        return (mean, disp, latent, E_low, x_recon, e_recon, attn_dict)
