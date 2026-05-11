import torch
import torch.nn as nn


class FeatureTemporalEncoderLayer(nn.Module):
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


class FeatureTemporalTransformerModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        if configs.d_model % configs.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.seq_len = configs.seq_len
        self.c_in = configs.c_in
        self.feature_layers = max(0, int(getattr(configs, "feature_layers", 1)))
        self.temporal_layer_count = max(1, int(getattr(configs, "temporal_layers", 2)))

        self.feature_mixers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.c_in),
                    nn.Linear(self.c_in, configs.d_model),
                    nn.GELU(),
                    nn.Dropout(configs.dropout),
                    nn.Linear(configs.d_model, self.c_in),
                )
                for _ in range(self.feature_layers)
            ]
        )

        self.temporal_input_proj = nn.Linear(self.c_in, configs.d_model)
        self.temporal_position = nn.Parameter(torch.zeros(1, self.seq_len, configs.d_model))
        self.temporal_layers = nn.ModuleList(
            [
                FeatureTemporalEncoderLayer(
                    d_model=configs.d_model,
                    n_heads=configs.n_heads,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                )
                for _ in range(self.temporal_layer_count)
            ]
        )
        self.temporal_norm = nn.LayerNorm(configs.d_model)
        self.output_proj = nn.Linear(configs.d_model, self.c_in)

    def forward(self, x):
        # x: [batch, seq_len, channels], output keeps the same reconstruction shape.
        batch_size, seq_len, channels = x.shape
        if channels != self.c_in:
            raise ValueError(f"Expected {self.c_in} channels, got {channels}")
        if seq_len > self.seq_len:
            raise ValueError(f"Expected seq_len <= {self.seq_len}, got {seq_len}")

        feature_values = x
        for mixer in self.feature_mixers:
            feature_values = feature_values + mixer(feature_values)

        temporal_tokens = self.temporal_input_proj(feature_values)
        temporal_tokens = temporal_tokens + self.temporal_position[:, :seq_len, :]
        for layer in self.temporal_layers:
            temporal_tokens = layer(temporal_tokens)
        return self.output_proj(self.temporal_norm(temporal_tokens))
