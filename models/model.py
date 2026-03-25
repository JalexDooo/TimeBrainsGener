import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for continuous time delta."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)


# ============================================================
# Building Blocks
# ============================================================

class DoubleConv3D(nn.Module):
    """Standard unconditioned double convolution for encoder."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, padding_mode='replicate'),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, padding_mode='replicate'),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.LeakyReLU(0.2, inplace=False),
        )
    def forward(self, x):
        return self.net(x)


class FiLMConv3D(nn.Module):
    """Double Conv3D with FiLM conditioning at each GroupNorm."""
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, padding_mode='replicate')
        self.gn1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act1 = nn.LeakyReLU(0.2, inplace=False)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, padding_mode='replicate')
        self.gn2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act2 = nn.LeakyReLU(0.2, inplace=False)
        self.film1 = nn.Linear(cond_dim, out_ch * 2)
        self.film2 = nn.Linear(cond_dim, out_ch * 2)

    def forward(self, x, cond):
        h = self.gn1(self.conv1(x))
        g1, b1 = self.film1(cond).chunk(2, dim=1)
        h = g1[..., None, None, None] * h + b1[..., None, None, None]
        h = self.act1(h)
        h = self.gn2(self.conv2(h))
        g2, b2 = self.film2(cond).chunk(2, dim=1)
        h = g2[..., None, None, None] * h + b2[..., None, None, None]
        return self.act2(h)


class ConditionGatedSkip(nn.Module):
    """
    Condition-dependent gate for skip connections.
    Instead of directly concatenating skip features (which enables identity mapping),
    the condition vector controls how much information from the encoder flows through.
    gate = sigmoid(W @ cond + b)  ->  gated_skip = gate * skip_features
    """
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(cond_dim, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
            nn.Sigmoid()
        )

    def forward(self, skip, cond):
        g = self.gate(cond)  # (B, C)
        return g[..., None, None, None] * skip


class AdaLNTransformerBlock(nn.Module):
    """
    Transformer block with Adaptive Layer Normalization (AdaLN), as in DiT.
    The condition modulates LayerNorm parameters (scale/shift) and output gates.
    This gives the condition direct, fine-grained control over feature processing.
    """
    def __init__(self, dim, num_heads, cond_dim, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )
        # AdaLN: condition -> 6 modulation vectors (scale1, shift1, gate1, scale2, shift2, gate2)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim * 6)
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, cond):
        params = self.adaLN(cond).unsqueeze(1)  # (B, 1, dim*6)
        s1, b1, g1, s2, b2, g2 = params.chunk(6, dim=-1)

        # Self-attention with modulated norm
        h = self.norm1(x) * (1 + s1) + b1
        h = self.attn(h, h, h, need_weights=False)[0]
        x = x + g1 * h

        # FFN with modulated norm
        h = self.norm2(x) * (1 + s2) + b2
        h = self.ffn(h)
        x = x + g2 * h
        return x


# ============================================================
# Main Model
# ============================================================

class TimeBrainsModel(nn.Module):
    """
    Hybrid CNN-Transformer architecture for longitudinal MRI generation.

    Key design differences from the previous UNet:
    1. Encoder is deeper (4 levels) and unconditioned
    2. Bottleneck uses Transformer self-attention with AdaLN conditioning
       (condition directly modulates every attention and FFN operation)
    3. Skip connections are GATED by the condition vector
       (prevents identity mapping; condition controls information flow)
    4. Decoder uses FiLM conditioning at every layer

    Data flow:
      Input(t0) → CNN Encoder [e1,e2,e3,e4] → Flatten → Transformer(cond) →
      Reshape → CNN Decoder(gated_skips, cond) → [delta_img, seg_logits]
    """
    def __init__(self, in_channels=7, out_channels=3, num_classes=4,
                 base_feat=48, max_treatments=20,
                 num_transformer_blocks=4, num_heads=8):
        super().__init__()
        bf = base_feat
        cond_dim = bf * 4  # 192
        trans_dim = bf * 8  # 384

        # --- Conditioning ---
        self.dt_embed = SinusoidalTimeEmbedding(cond_dim)
        self.trt_embed = nn.Embedding(max_treatments, cond_dim)
        self.cond_fuse = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim)
        )

        # --- Encoder (4 levels, unconditioned) ---
        self.enc1 = DoubleConv3D(in_channels, bf)        # 48  @ full
        self.down1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv3D(bf, bf * 2)             # 96  @ /2
        self.down2 = nn.MaxPool3d(2)
        self.enc3 = DoubleConv3D(bf * 2, bf * 4)         # 192 @ /4
        self.down3 = nn.MaxPool3d(2)
        self.enc4 = DoubleConv3D(bf * 4, trans_dim)      # 384 @ /8
        self.down4 = nn.MaxPool3d(2)                     # → /16 ≈ 11×13×11

        # --- Transformer Bottleneck ---
        max_tokens = 2048  # 11*13*11 = 1573 in our case
        self.pos_embed = nn.Parameter(torch.randn(1, max_tokens, trans_dim) * 0.02)
        self.transformer_blocks = nn.ModuleList([
            AdaLNTransformerBlock(trans_dim, num_heads, cond_dim)
            for _ in range(num_transformer_blocks)
        ])
        self.trans_norm = nn.LayerNorm(trans_dim)

        # --- Condition-Gated Skip Connections ---
        self.gate4 = ConditionGatedSkip(trans_dim, cond_dim)
        self.gate3 = ConditionGatedSkip(bf * 4, cond_dim)
        self.gate2 = ConditionGatedSkip(bf * 2, cond_dim)
        self.gate1 = ConditionGatedSkip(bf, cond_dim)

        # --- Decoder (4 levels, FiLM conditioned) ---
        self.up4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.dec4 = FiLMConv3D(trans_dim * 2, bf * 4, cond_dim)

        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.dec3 = FiLMConv3D(bf * 4 * 2, bf * 2, cond_dim)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.dec2 = FiLMConv3D(bf * 2 * 2, bf, cond_dim)

        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.dec1 = FiLMConv3D(bf * 2, bf, cond_dim)

        # --- Output Heads ---
        self.final_img = nn.Conv3d(bf, out_channels, kernel_size=1)
        self.final_seg = nn.Conv3d(bf, num_classes, kernel_size=1)

    def forward(self, x, dt, treatment):
        # 1. Fuse conditioning
        cond = self.cond_fuse(torch.cat([self.dt_embed(dt), self.trt_embed(treatment)], dim=1))

        # 2. Encode (no conditioning)
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        z = self.down4(e4)  # (B, 384, D', H', W')

        # 3. Transformer bottleneck
        B, C, D, H, W = z.shape
        tokens = z.flatten(2).transpose(1, 2)  # (B, N, C)
        N = tokens.shape[1]
        tokens = tokens + self.pos_embed[:, :N, :]

        for block in self.transformer_blocks:
            tokens = block(tokens, cond)
        tokens = self.trans_norm(tokens)

        z_out = tokens.transpose(1, 2).view(B, C, D, H, W)

        # 4. Decode with gated skip connections + FiLM
        d4 = self._up_and_cat(self.up4(z_out), self.gate4(e4, cond))
        d4 = self.dec4(d4, cond)

        d3 = self._up_and_cat(self.up3(d4), self.gate3(e3, cond))
        d3 = self.dec3(d3, cond)

        d2 = self._up_and_cat(self.up2(d3), self.gate2(e2, cond))
        d2 = self.dec2(d2, cond)

        d1 = self._up_and_cat(self.up1(d2), self.gate1(e1, cond))
        d1 = self.dec1(d1, cond)

        return self.final_img(d1), self.final_seg(d1), e4, z_out

    @staticmethod
    def _up_and_cat(up, skip):
        if up.shape[-3:] != skip.shape[-3:]:
            up = F.interpolate(up, size=skip.shape[-3:], mode='trilinear', align_corners=False)
        return torch.cat([up, skip], dim=1)


# ============================================================
# Discriminator (unchanged)
# ============================================================

class Discriminator(nn.Module):
    """5-layer PatchGAN with Spectral Normalization."""
    def __init__(self, in_channels=7):
        super().__init__()

        def spectral_conv(in_c, out_c, k=4, s=2, p=1):
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv3d(in_c, out_c, k, s, p)),
                nn.LeakyReLU(0.2, inplace=False)
            )

        self.net = nn.Sequential(
            spectral_conv(in_channels, 32),
            spectral_conv(32, 64),
            spectral_conv(64, 128),
            spectral_conv(128, 256),
            nn.utils.spectral_norm(nn.Conv3d(256, 1, 4, 1, 1))
        )

    def forward(self, x):
        return self.net(x)
