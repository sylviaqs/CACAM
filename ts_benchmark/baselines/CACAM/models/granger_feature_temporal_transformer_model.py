from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_granger_attention_bias(G, mask_type="soft", bias_value=2.0):
    """
    G: [C, C], where G[src, tgt] = 1 means src Granger-causes tgt.
    Return attn_mask: [C, C], where attn_mask[query=tgt, key=src].
    """
    if G.dim() != 2 or G.size(0) != G.size(1):
        raise ValueError(f"Expected square Granger matrix [C, C], got {tuple(G.shape)}")

    allowed = G.T.bool().clone()
    allowed.fill_diagonal_(True)

    mask = torch.zeros(G.size(0), G.size(1), device=G.device, dtype=torch.float32)
    if mask_type == "hard":
        return mask.masked_fill(~allowed, float("-inf"))
    if mask_type == "soft":
        return mask.masked_fill(~allowed, -float(bias_value))
    raise ValueError(f"Unknown granger_mask_type '{mask_type}', expected 'soft' or 'hard'")


def load_granger_graph(path):
    graph_path = Path(path)
    if not graph_path.exists():
        raise FileNotFoundError(f"Granger graph file not found: {graph_path}")

    suffix = graph_path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        graph = torch.load(graph_path, map_location="cpu")
        if isinstance(graph, dict):
            graph = graph.get("granger_G", graph.get("G"))
            if graph is None:
                raise ValueError("Granger graph dict must contain key 'granger_G' or 'G'")
        return torch.as_tensor(graph, dtype=torch.float32)
    if suffix == ".npy":
        return torch.from_numpy(np.load(graph_path)).float()
    return torch.from_numpy(np.loadtxt(graph_path, delimiter=",")).float()


class TemporalEncoderLayer(nn.Module):
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


class GrangerFeatureEncoderLayer(nn.Module):
    def __init__(self, seq_len, c_in, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.c_in = c_in
        self.history_proj = nn.Linear(seq_len, d_model)
        self.feature_embedding = nn.Parameter(torch.zeros(1, c_in, d_model))
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
        self.history_out = nn.Linear(d_model, seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, granger_mask=None):
        # x: [B, L, C]. Feature tokens are channels; each token owns a length-L history.
        batch_size, seq_len, channels = x.shape
        if channels != self.c_in:
            raise ValueError(f"Expected {self.c_in} channels, got {channels}")
        if seq_len > self.seq_len:
            raise ValueError(f"Expected seq_len <= {self.seq_len}, got {seq_len}")

        if seq_len < self.seq_len:
            x = F.pad(x, (0, 0, 0, self.seq_len - seq_len))

        feature_tokens = self.history_proj(x.transpose(1, 2))
        feature_tokens = feature_tokens + self.feature_embedding

        attn_mask = None
        if granger_mask is not None:
            attn_mask = granger_mask.to(device=x.device, dtype=feature_tokens.dtype)
        attn_out, _ = self.attn(
            feature_tokens,
            feature_tokens,
            feature_tokens,
            attn_mask=attn_mask,
            need_weights=False,
        )
        feature_tokens = self.attn_norm(feature_tokens + self.dropout(attn_out))
        ffn_out = self.ffn(feature_tokens)
        feature_tokens = self.ffn_norm(feature_tokens + self.dropout(ffn_out))

        out = self.history_out(feature_tokens).transpose(1, 2)
        return out[:, :seq_len, :]


class GrangerFeatureTemporalTransformerModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        if configs.d_model % configs.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.seq_len = configs.seq_len
        self.c_in = configs.c_in
        self.granger_mask_type = getattr(configs, "granger_mask_type", "soft")
        self.granger_bias = float(getattr(configs, "granger_bias", 2.0))
        self.feature_layer_count = max(0, int(getattr(configs, "feature_layers", 1)))
        self.temporal_layer_count = max(1, int(getattr(configs, "temporal_layers", 2)))
        granger_graph_path = getattr(configs, "granger_graph_path", None)
        default_granger_G = (
            load_granger_graph(granger_graph_path)
            if granger_graph_path not in (None, "")
            else None
        )
        if default_granger_G is not None and tuple(default_granger_G.shape) != (self.c_in, self.c_in):
            raise ValueError(
                f"Expected Granger graph shape {(self.c_in, self.c_in)}, "
                f"got {tuple(default_granger_G.shape)}"
            )
        self.register_buffer("default_granger_G", default_granger_G, persistent=False)

        self.feature_layers = nn.ModuleList(
            [
                GrangerFeatureEncoderLayer(
                    seq_len=self.seq_len,
                    c_in=self.c_in,
                    d_model=configs.d_model,
                    n_heads=configs.n_heads,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                )
                for _ in range(self.feature_layer_count)
            ]
        )

        self.temporal_input_proj = nn.Linear(self.c_in, configs.d_model)
        self.temporal_layers = nn.ModuleList(
            [
                TemporalEncoderLayer(
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

    def forward(self, x, granger_G=None):
        # x: [batch, seq_len, channels], output keeps the same reconstruction shape.
        batch_size, seq_len, channels = x.shape
        if channels != self.c_in:
            raise ValueError(f"Expected {self.c_in} channels, got {channels}")
        if seq_len > self.seq_len:
            raise ValueError(f"Expected seq_len <= {self.seq_len}, got {seq_len}")

        granger_mask = None
        if granger_G is None:
            granger_G = self.default_granger_G
        if granger_G is not None:
            if tuple(granger_G.shape) != (channels, channels):
                raise ValueError(
                    f"Expected Granger graph shape {(channels, channels)}, got {tuple(granger_G.shape)}"
                )
            granger_mask = build_granger_attention_bias(
                granger_G.to(device=x.device),
                mask_type=self.granger_mask_type,
                bias_value=self.granger_bias,
            )

        z = x
        for layer in self.feature_layers:
            z = layer(z, granger_mask)

        temporal_tokens = self.temporal_input_proj(z)
        for layer in self.temporal_layers:
            temporal_tokens = layer(temporal_tokens)

        return self.output_proj(self.temporal_norm(temporal_tokens))
