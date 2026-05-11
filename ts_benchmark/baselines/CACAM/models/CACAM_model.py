import torch
import torch.nn as nn


class CACAMEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.attn_norm(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.ffn_norm(x + self.dropout(ffn_out))


class CACAMModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        if configs.d_model % configs.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.seq_len = configs.seq_len
        self.input_proj = nn.Linear(configs.c_in, configs.d_model)
        self.position = nn.Parameter(torch.zeros(1, configs.seq_len, configs.d_model))
        self.layers = nn.ModuleList(
            [
                CACAMEncoderLayer(
                    d_model=configs.d_model,
                    n_heads=configs.n_heads,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                )
                for _ in range(configs.e_layers)
            ]
        )
        self.norm = nn.LayerNorm(configs.d_model)
        self.output_proj = nn.Linear(configs.d_model, configs.c_in)

    def forward(self, x):
        # x: [batch, seq_len, channels], output keeps the same reconstruction shape.
        h = self.input_proj(x)
        h = h + self.position[:, : h.size(1), :]
        for layer in self.layers:
            h = layer(h)
        return self.output_proj(self.norm(h))
