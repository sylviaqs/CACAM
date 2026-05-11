from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import f as f_distribution


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def build_granger_attention_bias(G, mask_type="soft", bias_value=2.0):
    """
    G: [C, C] or [B, C, C], where G[src, tgt] = 1 means src Granger-causes tgt.
    Return attn_mask with query=tgt and key=src.
    """
    if G.dim() not in {2, 3} or G.size(-1) != G.size(-2):
        raise ValueError(f"Expected square Granger matrix [C, C] or [B, C, C], got {tuple(G.shape)}")

    channels = G.size(-1)
    allowed = G.transpose(-1, -2).bool().clone()
    eye = torch.eye(channels, device=G.device, dtype=torch.bool)
    if G.dim() == 3:
        eye = eye.unsqueeze(0)
    allowed = allowed | eye

    mask = torch.zeros_like(G, dtype=torch.float32)
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


def _least_squares_rss(design, target):
    target = target.unsqueeze(-1)
    try:
        coef = torch.linalg.lstsq(design, target).solution
    except RuntimeError:
        coef = torch.matmul(torch.linalg.pinv(design), target)
    residual = target - torch.matmul(design, coef)
    return residual.squeeze(-1).pow(2).sum(dim=1)


def estimate_window_granger_graph(x, max_lag=3, alpha=0.05, eps=1e-8, standardize=True):
    """
    Estimate a pairwise Granger graph for each current input window.

    x: [B, L, C]
    Return G: [B, C, C], where G[b, src, tgt] = 1 means src Granger-causes tgt
    within window b under a pairwise F-test.
    """
    if x.dim() != 3:
        raise ValueError(f"Expected input shape [B, L, C], got {tuple(x.shape)}")
    max_lag = int(max_lag)
    if max_lag < 1:
        raise ValueError(f"granger_lag must be >= 1, got {max_lag}")
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"granger_alpha must be in (0, 1), got {alpha}")

    batch_size, seq_len, channels = x.shape
    graph = torch.zeros(batch_size, channels, channels, device=x.device, dtype=torch.float32)
    n_obs = seq_len - max_lag
    df_den = n_obs - (2 * max_lag + 1)
    if channels < 2 or n_obs <= 0 or df_den <= 0:
        return graph

    with torch.no_grad():
        series = x.detach().float()
        if standardize:
            series = series - series.mean(dim=1, keepdim=True)
            scale = series.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
            series = series / scale
            series = torch.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)

        y = series[:, max_lag:, :]
        lagged = torch.stack(
            [series[:, max_lag - lag : seq_len - lag, :] for lag in range(1, max_lag + 1)],
            dim=-1,
        )
        ones = torch.ones(batch_size, n_obs, 1, device=x.device, dtype=series.dtype)
        f_critical = float(f_distribution.ppf(1.0 - alpha, max_lag, df_den))

        for tgt in range(channels):
            target = y[:, :, tgt]
            target_lags = lagged[:, :, tgt, :]
            restricted_design = torch.cat([ones, target_lags], dim=-1)
            rss_restricted = _least_squares_rss(restricted_design, target)

            for src in range(channels):
                if src == tgt:
                    continue
                src_lags = lagged[:, :, src, :]
                unrestricted_design = torch.cat([ones, target_lags, src_lags], dim=-1)
                rss_unrestricted = _least_squares_rss(unrestricted_design, target)
                numerator = (rss_restricted - rss_unrestricted).clamp_min(0.0) / max_lag
                denominator = rss_unrestricted.clamp_min(eps) / df_den
                f_stat = numerator / denominator
                graph[:, src, tgt] = (f_stat > f_critical).float()

    return graph


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
        self.n_heads = n_heads
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

    def forward(self, x, granger_mask=None, debug_print=False):
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
            raw_attn_mask_shape = tuple(attn_mask.shape)
            if attn_mask.dim() == 3:
                if attn_mask.size(0) != batch_size or tuple(attn_mask.shape[-2:]) != (channels, channels):
                    raise ValueError(
                        f"Expected batch Granger mask shape {(batch_size, channels, channels)}, "
                        f"got {tuple(attn_mask.shape)}"
                    )
                attn_mask = attn_mask.repeat_interleave(self.n_heads, dim=0)
            if debug_print:
                print(
                    "[GrangerDebug] Feature layer attention: "
                    f"feature_tokens={tuple(feature_tokens.shape)} "
                    f"raw_attn_mask={raw_attn_mask_shape} "
                    f"mha_attn_mask={tuple(attn_mask.shape)} "
                    "passed_to=nn.MultiheadAttention(attn_mask=...)",
                    flush=True,
                )
        elif debug_print:
            print(
                "[GrangerDebug] Feature layer attention: "
                f"feature_tokens={tuple(feature_tokens.shape)} attn_mask=None",
                flush=True,
            )
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
        self.dynamic_granger = _to_bool(getattr(configs, "dynamic_granger", True))
        self.granger_lag = int(getattr(configs, "granger_lag", 3))
        self.granger_alpha = float(getattr(configs, "granger_alpha", 0.05))
        self.granger_standardize = _to_bool(getattr(configs, "granger_standardize", True))
        self.granger_debug_print = _to_bool(getattr(configs, "granger_debug_print", False))
        self._granger_debug_printed = False
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
        if granger_G is None and self.dynamic_granger and self.feature_layer_count > 0:
            granger_G = estimate_window_granger_graph(
                x,
                max_lag=self.granger_lag,
                alpha=self.granger_alpha,
                standardize=self.granger_standardize,
            )
        if granger_G is not None:
            expected_shapes = {(channels, channels), (batch_size, channels, channels)}
            if tuple(granger_G.shape) not in expected_shapes:
                raise ValueError(
                    f"Expected Granger graph shape {(channels, channels)} or "
                    f"{(batch_size, channels, channels)}, got {tuple(granger_G.shape)}"
                )
            granger_mask = build_granger_attention_bias(
                granger_G.to(device=x.device),
                mask_type=self.granger_mask_type,
                bias_value=self.granger_bias,
            )
        debug_print = self.granger_debug_print and not self._granger_debug_printed
        if debug_print:
            print(
                "[GrangerDebug] Forward: "
                f"x={tuple(x.shape)} "
                f"granger_G={None if granger_G is None else tuple(granger_G.shape)} "
                f"granger_mask={None if granger_mask is None else tuple(granger_mask.shape)} "
                f"mask_type={self.granger_mask_type} "
                f"dynamic_granger={self.dynamic_granger} "
                f"default_granger_G={None if self.default_granger_G is None else tuple(self.default_granger_G.shape)}",
                flush=True,
            )

        z = x
        for layer_idx, layer in enumerate(self.feature_layers):
            z = layer(z, granger_mask, debug_print=debug_print and layer_idx == 0)
        if debug_print:
            self._granger_debug_printed = True

        temporal_tokens = self.temporal_input_proj(z)
        for layer in self.temporal_layers:
            temporal_tokens = layer(temporal_tokens)

        return self.output_proj(self.temporal_norm(temporal_tokens))
