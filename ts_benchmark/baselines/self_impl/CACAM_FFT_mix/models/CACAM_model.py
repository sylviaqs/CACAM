import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        
        # Reshape for multi-head attention
        Q = self.q_proj(x).view(B, C, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        K = self.k_proj(x).view(B, C, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        V = self.v_proj(x).view(B, C, self.n_heads, self.d_model // self.n_heads).transpose(1, 2)
        
        # Calculate attention scores: [B, n_heads, C, C]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_model ** 0.5)
        
        # Apply causal constraint
        # causal_weight represents the statistical dependency between channels.
        # We transform it to log space to act as an additive mask, penalizing weak dependencies.
        # A small epsilon prevents log(0).
        cw_log = torch.log(causal_weight.unsqueeze(1) + 1e-8)
        
        # Constraint superimposed on original attention scores
        scores = scores + cw_log
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Context vectors
        out = torch.matmul(attn, V).transpose(1, 2).reshape(B, C, self.d_model)
        return self.out_proj(out), attn


class Basic_CACAM(nn.Module):
    def __init__(self, configs):
        super(Basic_CACAM, self).__init__()
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.dropout = configs.dropout
        self.causal_method = getattr(configs, "causal_method", "correlation")
        self.causal_max_lag = getattr(configs, "causal_max_lag", 3)
        self.causal_pc_alpha = getattr(configs, "causal_pc_alpha", 0.05)
        
        # 时域特征投影
        self.feature_proj = nn.Sequential(
            nn.Linear(self.seq_len, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        # 频域特征投影 - 使用卷积提取频域模式
        # 输入: [B, C, freq_len] 其中 freq_len = seq_len//2 + 1 (rfft结果)
        freq_len = self.seq_len // 2 + 1
        self.freq_proj = nn.Sequential(
            nn.Conv1d(
                in_channels=1,  # 每个通道单独处理
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),  # 固定输出长度为1，避免MPS不可整除问题
            nn.Flatten(start_dim=1),
            nn.Linear(16, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        # 融合时频特征的投影层
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        # Causal Attention layer
        self.causal_attn = CausalConstraintAttention(
            self.d_model, 
            n_heads=self.n_heads, 
            dropout=self.dropout
        )
        
        # Reconstruction layer
        self.reconstruct_proj = nn.Sequential(
            nn.Linear(self.d_model, self.seq_len)
        )
        
    def _normalize_causal_weight(self, weight):
        weight = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)
        weight = torch.clamp(weight, min=0.0)
        row_sum = weight.sum(dim=-1, keepdim=True)
        uniform = torch.full_like(weight, 1.0 / weight.shape[-1])
        return torch.where(row_sum > 1e-8, weight / (row_sum + 1e-8), uniform)

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
        使用 PCMCI 算法估计因果矩阵。
        PCMCI (Peter-Clark Momentary Conditional Independence) 是一种因果发现算法，
        专门用于从时间序列数据中推断变量间的因果关系。
        
        Returns:
            causal_weight: [B, C, C] 因果强度矩阵，其中元素 (i,j) 表示从变量 j 到变量 i 的因果强度
        """
        # x: [B, T, C]
        B, T, C = x.shape
        
        try:
            import tigramite.data_processing as dp
            from tigramite.pcmci import PCMCI
            from tigramite.independence_tests.parcorr import ParCorr
        except ImportError as e:
            print(f"Warning: Failed to import tigramite. Error: {e}")
            print("Please ensure tigramite is installed: pip install tigramite")
            return self.compute_causal_weight_correlation(x)
        
        # 数据格式转换: [B, T, C] -> [T, C] (PCMCI 期望 [time x variable])
        # 使用 batch 中所有样本的平均作为因果结构（更稳定）
        data = x.detach().cpu().numpy().mean(axis=0)  # [T, C]
        
        # 将多滞后阶数的因果信息聚合到 0 滞后

        # 构建 tigramite 数据框架
        dataframe = dp.DataFrame(data)
        # ParCorr (Partial Correlation) 适用于连续型时间序列数据
        cond_ind_test = ParCorr(significance='analytic')
        
        # 初始化 PCMCI 算法
        pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=cond_ind_test
        )
        
        # 运行 PCMCI 算法
        # - pc_alpha: PC 算法条件独立性检验的显著性水平
        # - max_cond_px: MCI 条件集的最大大小
        # - max_lags: 最大滞后阶数（考虑过去多久的因果影响）
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
        """
        使用皮尔逊相关系数估计因果矩阵（原始方法，保留作为回退方案）。
        
        注意：相关性不等于因果性，此方法仅作为 tigramite 不可用时的备选。
        相关系数仅能捕捉变量间的统计关联，无法区分因果方向。
        """
        # x: [B, T, C]
        B, T, C = x.shape
        
        # 按时间维度中心化
        x_mean = x.mean(dim=1, keepdim=True)
        x_centered = x - x_mean
        
        # 计算方差和标准差
        x_var = (x_centered ** 2).sum(dim=1, keepdim=True)
        x_std = torch.sqrt(x_var + 1e-8)
        
        # 标准化
        x_normalized = x_centered / x_std
        
        # 计算相关系数矩阵: [B, C, C]
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
        
        # 1. Compute causal or dependency weight matrix.
        causal_weight = self.compute_causal_weight(x)
        
        # 2. 时域特征提取
        x_t = x.transpose(1, 2)  # [B, C, T]
        h_time = self.feature_proj(x_t)  # [B, C, d_model]
        
        # 3. 频域特征提取 - 对每个通道单独做rfft，取幅值
        # rfft on time dimension: [B, C, T] -> [B, C, freq_len]
        freq_len = self.seq_len // 2 + 1
        fft_result = torch.fft.rfft(x_t, dim=-1)
        freq_magnitude = torch.abs(fft_result)  # [B, C, freq_len]
        
        # 频域特征通过卷积提取模式
        # Reshape to [B*C, 1, freq_len]
        freq_magnitude_flat = freq_magnitude.reshape(B * C, 1, freq_len)
        h_freq_conv = self.freq_proj(freq_magnitude_flat)  # [B*C, d_model]
        h_freq = h_freq_conv.reshape(B, C, self.d_model)  # [B, C, d_model]
        
        # 4. 时频特征融合
        h_fused = torch.cat([h_time, h_freq], dim=-1)  # [B, C, d_model*2]
        h_fused = self.fusion_proj(h_fused)  # [B, C, d_model]
        
        # 5. Apply Causal Constraint Attention on fused features
        h_attn, attn_weights = self.causal_attn(h_fused, causal_weight)
        
        # 6. Residual connection & Reconstruct
        h_out = F.gelu(h_fused + h_attn)
        x_recon = self.reconstruct_proj(h_out)  # [B, C, T]
        
        # Transpose back to [B, T, C]
        x_recon = x_recon.transpose(1, 2)
        
        return x_recon, causal_weight
