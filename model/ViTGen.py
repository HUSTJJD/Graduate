import numpy as np
from einops import rearrange, repeat

import torch
import torch.nn as nn


class SLN(nn.Module):
    """
    Self-modulated LayerNorm
    """
    def __init__(self, num_features):
        super(SLN, self).__init__()
        self.ln = nn.LayerNorm(num_features)
        # self.gamma = nn.Parameter(torch.FloatTensor(1, 1, 1))
        # self.beta = nn.Parameter(torch.FloatTensor(1, 1, 1))
        self.gamma = nn.Parameter(torch.randn(1, 1, 1))  #.to("cuda")
        self.beta = nn.Parameter(torch.randn(1, 1, 1))  #.to("cuda")

    def forward(self, hl, w):
        return self.gamma * w * self.ln(hl) + self.beta * w


class MLP(nn.Module):
    def __init__(self, in_feat, hid_feat=None, out_feat=None, dropout=0.):
        super().__init__()
        if not hid_feat:
            hid_feat = in_feat
        if not out_feat:
            out_feat = in_feat
        self.linear1 = nn.Linear(in_feat, hid_feat)
        # self.activation = nn.GELU()
        self.activation = nn.LeakyReLU()
        self.linear2 = nn.Linear(hid_feat, out_feat)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout(x)


class Attention(nn.Module):
    """
    Implement multi head self attention layer using the "Einstein summation convention".

    Parameters
    ----------
    dim:
        Token's dimension, EX: word embedding vector size
    num_heads:
        The number of distinct representations to learn
    dim_head:
        The dimension of the each head
    discriminator:
        Used in discriminator or not.
    """
    def __init__(self, dim, num_heads=4, dim_head=None, discriminator=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.dim_head = int(dim / num_heads) if dim_head is None else dim_head
        self.weight_dim = self.num_heads * self.dim_head
        self.to_qkv = nn.Linear(dim, self.weight_dim * 3, bias=False)
        self.scale_factor = dim**-0.5
        self.discriminator = discriminator
        self.w_out = nn.Linear(self.weight_dim, dim, bias=True)

        if discriminator:
            u, s, v = torch.svd(self.to_qkv.weight)
            self.init_spect_norm = torch.max(s)

    def forward(self, x):
        assert x.dim() == 3

        if self.discriminator:
            u, s, v = torch.svd(self.to_qkv.weight)
            self.to_qkv.weight = torch.nn.Parameter(
                self.to_qkv.weight * self.init_spect_norm / torch.max(s))

        # Generate the q, k, v vectors
        qkv = self.to_qkv(x)
        q, k, v = tuple(
            rearrange(qkv, 'b t (d k h) -> k b h t d', k=3, h=self.num_heads))

        # Enforcing Lipschitzness of Transformer Discriminator
        # Due to Lipschitz constant of standard dot product self-attention
        # layer can be unbounded, so adopt the l2 attention replace the dot product.
        if self.discriminator:
            attn = torch.cdist(q, k, p=2)
        else:
            attn = torch.einsum("... i d, ... j d -> ... i j", q, k)
        scale_attn = attn * self.scale_factor
        scale_attn_score = torch.softmax(scale_attn, dim=-1)
        result = torch.einsum("... i j, ... j d -> ... i d", scale_attn_score,
                              v)

        # re-compose
        result = rearrange(result, "b h t d -> b t (h d)")
        return nn.LeakyReLU()(self.w_out(result))


class GEncoderBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=4,
                 dim_head=None,
                 dropout=0.,
                 mlp_ratio=4):
        super(GEncoderBlock, self).__init__()
        self.attn = Attention(dim, num_heads, dim_head)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = SLN(dim)
        self.norm2 = SLN(dim)

        self.mlp = MLP(dim, dim * mlp_ratio, dropout=dropout)

    def forward(self, hl, x):
        hl_temp = self.dropout(self.attn(self.norm1(hl, x))) + hl
        hl_final = self.mlp(self.norm2(hl_temp, x)) + hl_temp
        return x, hl_final


class GTransformerEncoder(nn.Module):
    def __init__(self, dim, blocks=6, num_heads=8, dim_head=None, dropout=0):
        super(GTransformerEncoder, self).__init__()
        self.blocks = self._make_layers(dim, blocks, num_heads, dim_head,
                                        dropout)

    def _make_layers(self,
                     dim,
                     blocks=6,
                     num_heads=8,
                     dim_head=None,
                     dropout=0):
        layers = []
        for _ in range(blocks):
            layers.append(GEncoderBlock(dim, num_heads, dim_head, dropout))
        return nn.Sequential(*layers)

    def forward(self, hl, x):
        for block in self.blocks:
            x, hl = block(hl, x)
        return x, hl


class SineLayer(nn.Module):
    """
    Paper: Implicit Neural Representation with Periodic Activ ation Function (SIREN)
    """
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 is_first=False,
                 omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class Decoder(nn.Module):
    def __init__(self,
                 initialize_size=16,
                 dim=384,
                 blocks=12,
                 num_heads=6,
                 dim_head=None,
                 dropout=0,
                 out_channels=3):
        super(Decoder, self).__init__()
        self.initialize_size = initialize_size
        self.dim = dim
        self.blocks = blocks
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.out_channels = out_channels

        self.pos_emb1D = nn.Parameter(
            torch.randn(self.initialize_size * 8, dim))

        self.mlp = nn.Linear(14, (self.initialize_size * 8) * self.dim)
        self.Transformer_Encoder = GTransformerEncoder(dim, blocks, num_heads,
                                                       dim_head, dropout)

        # Implicit Neural Representation
        self.w_out = nn.Sequential(
            SineLayer(dim, dim * 2, is_first=True, omega_0=30.),
            SineLayer(dim * 2,
                      self.initialize_size * 8 * self.out_channels,
                      is_first=False,
                      omega_0=30))
        self.sln_norm = SLN(self.dim)


    def forward(self, noise):
        x = self.mlp(noise).view(-1, self.initialize_size * 8, self.dim)
        x, hl = self.Transformer_Encoder(self.pos_emb1D, x)
        x = self.sln_norm(hl, x)
        x = self.w_out(x)  # Replace to siren
        result = x.view(x.shape[0], 3, self.initialize_size * 8,
                        self.initialize_size * 8)
        # result = self.deconv(result)
        return result


# def test_both():
#     B, dim = 1, 14
#     G = Decoder(initialize_size=16, dropout=0.1)
#     noise = torch.FloatTensor(np.random.normal(0, 1, (B, dim)))
#     fake_img = G(noise)

# D = Discriminator(patch_size=8, dropout=0.1)
# D_logits = D(fake_img)
# print(D_logits)
# print(f"Max: {torch.max(D_logits)}, Min: {torch.min(D_logits)}")
