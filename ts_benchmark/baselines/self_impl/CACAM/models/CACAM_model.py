import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fft import fft, ifft


class CausalConstraintAttention(nn.Module):
    def __init__(self, d_model, n_heads=1, dropout=0.1):
        super(CausalConstraintAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, causal_weight):
        # x: [B, C, d_model]
        # causal_weight: [B, C, C]
        B, C, _ = x.shape
        
        Q = self.q_proj(x).view(B, C, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        K = self.k_proj(x).view(B, C, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        V = self.v_proj(x).view(B, C, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_model ** 0.5)
        
        cw_log = torch.log(causal_weight.unsqueeze(1) + 1e-8)
        scores = scores + cw_log
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V).transpose(1, 2).reshape(B, C, self.d_model)
        return self.out_proj(out), attn


class FrequencyCausalAttention(nn.Module):
    """
    频域因果注意力模块 - 在通道维度上进行注意力计算
    """
    def __init__(self, d_model, n_heads=1, dropout=0.1):
        super(FrequencyCausalAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, causal_weight):
        # x: [B, C, d_model] 通道维度作为"序列"
        # causal_weight: [B, C, C] 通道间的因果掩码
        B, C, _ = x.shape
        
        Q = self.q_proj(x).view(B, C, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        K = self.k_proj(x).view(B, C, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        V = self.v_proj(x).view(B, C, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_model ** 0.5)
        
        if causal_weight is not None:
            cw_log = torch.log(causal_weight.unsqueeze(1) + 1e-8)
            scores = scores + cw_log
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V).transpose(1, 2).reshape(B, C, self.d_model)
        return self.out_proj(out), attn


class Basic_CACAM(nn.Module):
    def __init__(self, configs):
        super(Basic_CACAM, self).__init__()
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.dropout = configs.dropout
        self.causal_method = getattr(configs, "causal_method", "pcmci")
        self.causal_max_lag = getattr(configs, "causal_max_lag", 3)
        self.causal_pc_alpha = getattr(configs, "causal_pc_alpha", 0.05)
        self.freq_weight = getattr(configs, "freq_weight", 0.5)
        
        # 时域特征投影
        self.feature_proj = nn.Sequential(
            nn.Linear(self.seq_len, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        # 频域特征投影 - 对每个通道的频域特征做映射: [B*C, T*2] -> [B*C, d_model]
        self.freq_proj = nn.Sequential(
            nn.Linear(self.seq_len * 2, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        # 频域重建层: [B*C, d_model] -> [B*C, T*2]
        self.freq_reconstruct_proj = nn.Sequential(
            nn.Linear(self.d_model, self.seq_len * 2)
        )
        
        # 时域因果注意力
        self.causal_attn = CausalConstraintAttention(
            self.d_model, 
            n_heads=self.n_heads, 
            dropout=self.dropout
        )
        
        # 频域因果注意力
        self.freq_causal_attn = FrequencyCausalAttention(
            self.d_model,
            n_heads=self.n_heads,
            dropout=self.dropout
        )
        
        # 时域重建层
        self.reconstruct_proj = nn.Sequential(
            nn.Linear(self.d_model, self.seq_len)
        )
        
        # 频域重建层
        self.freq_reconstruct_proj = nn.Sequential(
            nn.Linear(self.d_model, self.seq_len * 2)
        )
    
    def _normalize_causal_weight(self, weight):
        weight = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)
        weight = torch.clamp(weight, min=0.0)
        row_sum = weight.sum(dim=-1, keepdim=True)
        uniform = torch.full_like(weight, 1.0 / weight.shape[-1])
        return torch.where(row_sum > 1e-8, weight / (row_sum + 1e-8), uniform)

    def _to_frequency(self, x):
        """将时域信号转换到频域"""
        # x: [B, T, C]
        B, T, C = x.shape
        
        x_t = x.transpose(1, 2)  # [B, C, T]
        
        freq_real_list = []
        freq_imag_list = []
        for b in range(B):
            x_t_b_detached = x_t[b].detach().cpu().numpy()
            fft_result = fft(x_t_b_detached, axis=-1)
            freq_real_list.append(torch.from_numpy(np.real(fft_result)).float())
            freq_imag_list.append(torch.from_numpy(np.imag(fft_result)).float())
        
        freq_real = torch.stack(freq_real_list, dim=0).to(x.device)
        freq_imag = torch.stack(freq_imag_list, dim=0).to(x.device)
        
        freq_complex = torch.cat([freq_real, freq_imag], dim=-1)  # [B, C, T*2]
        
        return freq_complex

    def _from_frequency(self, freq_feat):
        """将频域特征转换回时域"""
        # freq_feat: [B, C, T*2]
        B, C, _ = freq_feat.shape
        T = self.seq_len
        
        freq_real = freq_feat[:, :, :T]
        freq_imag = freq_feat[:, :, T:]
        freq_complex = freq_real + 1j * freq_imag
        
        time_signal_list = []
        for b in range(B):
            freq_b_detached = freq_complex[b].detach().cpu().numpy()
            ifft_result = ifft(freq_b_detached, axis=-1)
            time_signal_list.append(torch.from_numpy(np.real(ifft_result)).float())
        
        time_signal = torch.stack(time_signal_list, dim=0).to(freq_feat.device)
        time_signal = time_signal.transpose(1, 2)  # [B, T, C]
        
        return time_signal

    def compute_causal_weight(self, x):
        method = str(self.causal_method).lower()
        if method in ("corr", "correlation", "pearson"):
            return self.compute_causal_weight_correlation(x)
        if method in ("partial_corr", "partial_correlation", "precision"):
            return self.compute_causal_weight_partial_correlation(x)
        if method in ("granger", "granger_causality"):
            return self.compute_causal_weight_granger(x)
        if method == "pcmci":
            return self.compute_causal_weight_pcmci(x)
        if method == "identity":
            return self.compute_causal_weight_identity(x)
        if method == "uniform":
            return self.compute_causal_weight_uniform(x)
        raise ValueError(
            f"Unsupported causal_method={self.causal_method!r}. "
            "Use one of: correlation, partial_correlation, granger, pcmci, identity, uniform."
        )

    def compute_causal_weight_identity(self, x):
        B, _, C = x.shape
        return torch.eye(C, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, -1, -1)

    def compute_causal_weight_uniform(self, x):
        B, _, C = x.shape
        return torch.full((B, C, C), 1.0 / C, device=x.device, dtype=x.dtype)

    def compute_causal_weight_pcmci(self, x):
        """
        使用 PCMCI 算法估计时域因果矩阵
        """
        B, T, C = x.shape
        
        try:
            import tigramite.data_processing as dp
            from tigramite.pcmci import PCMCI
            from tigramite.independence_tests.parcorr import ParCorr
        except ImportError as e:
            print(f"Warning: Failed to import tigramite. Error: {e}")
            print("Please ensure tigramite is installed: pip install tigramite")
            return self.compute_causal_weight_correlation(x)
        
        data = x.detach().cpu().numpy().mean(axis=0)  # [T, C]
        
        dataframe = dp.DataFrame(data)
        cond_ind_test = ParCorr(significance='analytic')
        
        pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=cond_ind_test
        )
        
        results = pcmci.run_pcmci(tau_max=self.causal_max_lag, pc_alpha=self.causal_pc_alpha)
        
        causal_matrix = torch.zeros(C, C, device=x.device)

        val_matrix = results.get("val_matrix")
        p_matrix = results.get("p_matrix")
        if val_matrix is None:
            return self.compute_causal_weight_correlation(x)

        for source in range(C):
            for target in range(C):
                if source == target:
                    continue
                strengths = []
                for tau in range(1, min(self.causal_max_lag, val_matrix.shape[2] - 1) + 1):
                    p_value = 0.0 if p_matrix is None else p_matrix[source, target, tau]
                    if p_matrix is None or p_value <= self.causal_pc_alpha:
                        strengths.append(abs(val_matrix[source, target, tau]))
                if strengths:
                    causal_matrix[target, source] = float(max(strengths))

        causal_matrix.fill_diagonal_(1.0)
        causal_weight = self._normalize_causal_weight(causal_matrix).unsqueeze(0).expand(B, -1, -1)
        return causal_weight

    def compute_causal_weight_correlation(self, x):
        B, T, C = x.shape
        
        x_mean = x.mean(dim=1, keepdim=True)
        x_centered = x - x_mean
        
        x_var = (x_centered ** 2).sum(dim=1, keepdim=True)
        x_std = torch.sqrt(x_var + 1e-8)
        
        x_normalized = x_centered / x_std
        
        corr = torch.bmm(x_normalized.transpose(1, 2), x_normalized)
        
        causal_weight = torch.abs(corr)

        return self._normalize_causal_weight(causal_weight)

    def compute_causal_weight_partial_correlation(self, x):
        B, T, C = x.shape
        x_centered = x - x.mean(dim=1, keepdim=True)
        cov = torch.bmm(x_centered.transpose(1, 2), x_centered) / max(T - 1, 1)
        eye = torch.eye(C, device=x.device, dtype=x.dtype).unsqueeze(0)
        cov = cov + 1e-3 * eye
        precision = torch.linalg.pinv(cov)
        diag = torch.diagonal(precision, dim1=-2, dim2=-1).clamp_min(1e-8)
        denom = torch.sqrt(diag.unsqueeze(-1) * diag.unsqueeze(-2))
        partial_corr = -precision / denom
        partial_corr = torch.abs(partial_corr)
        partial_corr.diagonal(dim1=-2, dim2=-1).fill_(1.0)
        return self._normalize_causal_weight(partial_corr)

    def compute_causal_weight_granger(self, x):
        B, _, C = x.shape
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
        except ImportError as e:
            print(f"Warning: Failed to import statsmodels. Error: {e}")
            return self.compute_causal_weight_correlation(x)

        data = x.detach().cpu().numpy().mean(axis=0)
        causal_matrix = torch.eye(C, device=x.device, dtype=x.dtype)
        max_lag = max(1, int(self.causal_max_lag))

        for target in range(C):
            for source in range(C):
                if source == target:
                    continue
                pair = data[:, [target, source]]
                try:
                    result = grangercausalitytests(pair, maxlag=max_lag, verbose=False)
                    p_values = [
                        result[lag][0]["ssr_ftest"][1]
                        for lag in range(1, max_lag + 1)
                        if lag in result
                    ]
                    if p_values:
                        p_value = max(min(p_values), 1e-12)
                        causal_matrix[target, source] = float(-torch.log(torch.tensor(p_value)))
                except Exception:
                    continue

        return self._normalize_causal_weight(causal_matrix).unsqueeze(0).expand(B, -1, -1)
        
    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.shape
        
        # 1. 时域因果权重 (PCMCI)
        causal_weight = self.compute_causal_weight(x)
        
        # 2. 时域分支
        x_t = x.transpose(1, 2)  # [B, C, T]
        h_time = self.feature_proj(x_t)  # [B, C, d_model]
        h_time_attn, _ = self.causal_attn(h_time, causal_weight)
        h_time_out = F.gelu(h_time + h_time_attn)
        x_time_recon = self.reconstruct_proj(h_time_out)  # [B, C, T]
        x_time_recon = x_time_recon.transpose(1, 2)  # [B, T, C]
        
        # 3. 频域分支
        freq_feat = self._to_frequency(x)  # [B, C, T*2]
        
        # 计算频域因果掩码 (皮尔逊相关性) - 基于频域特征的通道相关性
        freq_pearson_mask = self._compute_pearson_mask(freq_feat)  # [B, C, C]
        
        # 频域特征投影: [B, C, T*2] -> [B, C, d_model]
        # 对每个频率点(维度T*2)做线性投影，得到每个通道的d_model表示
        # freq_feat: [B, C, T*2] -> 转换到 [B, C, T*2] -> 通过线性层 [B, C, T*2] -> [B, C, d_model]
        # 方法：reshape成 [B*C, T*2] 再做线性变换，再reshape回来
        B, C, FT = freq_feat.shape
        h_freq_flat = freq_feat.reshape(B * C, FT)
        h_freq = self.freq_proj(h_freq_flat).reshape(B, C, self.d_model)  # [B, C, d_model]
        
        # 频域因果注意力 (在C维度上，用频域通道掩码)
        h_freq_attn, _ = self.freq_causal_attn(h_freq, freq_pearson_mask)  # [B, C, d_model]
        h_freq_out = F.gelu(h_freq + h_freq_attn)  # [B, C, d_model]
        
        # 重建频域特征: [B, C, d_model] -> [B, C, T*2]
        freq_recon_flat = self.freq_reconstruct_proj(h_freq_out.reshape(B * C, self.d_model))
        freq_recon_feat = freq_recon_flat.reshape(B, C, FT)  # [B, C, T*2]
        
        # 转换回时域
        x_freq_recon = self._from_frequency(freq_recon_feat)  # [B, T, C]
        
        # 4. 时域和频域加权融合
        x_recon = self.freq_weight * x_freq_recon + (1 - self.freq_weight) * x_time_recon
        
        return x_recon, causal_weight, x_time_recon, x_freq_recon
    
    def _compute_pearson_mask(self, x):
        """计算皮尔逊相关性掩码用于频域注意力"""
        # x: [B, C, FT] 频域特征
        B, C, FT = x.shape

        # 在频域维度上聚合，计算通道间的相关性
        # x: [B, C, FT] -> 计算 C x C 相关矩阵
        x_mean = x.mean(dim=-1, keepdim=True)  # [B, C, 1]
        x_centered = x - x_mean  # [B, C, FT]
        x_std = torch.sqrt((x_centered ** 2).sum(dim=-1, keepdim=True) + 1e-8)  # [B, C, 1]
        x_normalized = x_centered / x_std  # [B, C, FT]

        # 批量矩阵乘法: [B, C, FT] x [B, FT, C] -> [B, C, C]
        corr = torch.bmm(x_normalized, x_normalized.transpose(1, 2))
        corr_abs = torch.abs(corr)

        corr_abs = torch.nan_to_num(corr_abs, nan=0.0, posinf=0.0, neginf=0.0)
        corr_abs = torch.clamp(corr_abs, min=0.0)
        row_sum = corr_abs.sum(dim=-1, keepdim=True)
        uniform = torch.full_like(corr_abs, 1.0 / C)
        corr_abs = torch.where(row_sum > 1e-8, corr_abs / (row_sum + 1e-8), uniform)

        return corr_abs
