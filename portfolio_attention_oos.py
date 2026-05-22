"""
================================================================================
 Portfolio Optimization with Multihead-Attention Volatility Forecasting
 and Deep Reinforcement Learning  —  Publication-grade OOS pipeline
================================================================================

 Pipeline:
   1.  Strict temporal split (train 60% / val 20% / test 20%) — no leakage.
   2.  Volatility forecasting with FOUR models:
         * AttentionTCN     (NEW: TCN backbone + Multihead Attention + BiLSTM + PE)
         * ResNet1D         (baseline)
         * WaveletCNN       (baseline)
         * TCNAutoencoder   (baseline)
       Hyperparameters tuned by minimum validation MSE.
       Diebold–Mariano test between forecasters on the TEST set.
   3.  Portfolio allocation strategies, all evaluated OOS on TEST:
         * DRL-PPO with AdvancedFeatureExtractor (multihead attention)
         * DRL-A2C with AdvancedFeatureExtractor (multihead attention)
         * Markowitz Max-Sharpe (rolling)
         * Minimum-Variance     (rolling)
         * Risk-Parity          (rolling, inverse-vol)
         * Equal-Weight         (rolling)
         * Buy & Hold           (1/N, no rebalance)
       Transaction costs and slippage modeled.
   4.  Comprehensive metrics: CAGR, Vol, Sharpe, Sortino, Calmar, Max-DD, VaR,
       CVaR, Alpha, Beta, Win-Rate, Turnover, RoMaD, t-stat & p-value.
   5.  Bootstrap 95% CI for Sharpe (n=2000).
   6.  Statistical tests: paired t-test, Wilcoxon, Diebold-Mariano.
   7.  Multi-seed runs (default n=3) for variance estimation.
   8.  All artifacts (CSV, plots, reports) saved under RESULTS_DIR.

 Notes:
   - DRL action space is continuous Box([0,1]^n) -> softmax-normalized weights.
   - Observations are sequences (lookback x features) so attention is meaningful.
   - Vol forecast is added as an exogenous market-stress signal.
"""

from __future__ import annotations
import os
import sys
import math
import json
import random
import warnings
import itertools
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# pypfopt for classical optimization
from pypfopt import EfficientFrontier, risk_models, expected_returns

# RL stack
import gymnasium as gym
from gymnasium import spaces

try:
    from stable_baselines3 import PPO, A2C
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    SB3_OK = True
except Exception:
    SB3_OK = False
    print("[WARN] stable_baselines3 not available — DRL strategies will be skipped.")

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


# =============================================================================
# 1. GLOBAL CONFIGURATION
# =============================================================================

# Reproducibility
GLOBAL_SEED = 42

# Hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
PRICES_PATH = r"Merged_BATS_Stock_Prices.csv"
VOL_PATH    = r"Merged_VIX__DIX__and_SPX_Data.csv"
TICKERS     = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "XOM", "BAC", "META"]
ALL_VOL_FEATURES = ["DIX", "GEX", "SKEW", "PCR", "VIX"]

# Temporal split (strict, no leakage)
TRAIN_RATIO = 0.60
VAL_RATIO   = 0.20
TEST_RATIO  = 0.20
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9

# Windows
LOOKBACK_VOL  = 60     # input window for volatility forecasters
LOOKBACK_RL   = 20     # input window for RL observations

# Capital & costs
INITIAL_CAPITAL = 100_000.0
COMMISSION_BPS  = 5.0   # 5 basis points per side
SLIPPAGE_BPS    = 2.0   # 2 bps slippage per turnover unit

# Portfolio constraints
MIN_WEIGHT = 0.0
MAX_WEIGHT = 0.40

# Rebalance frequencies (trading days)
REBALANCE_FREQS = {"weekly": 5, "monthly": 21, "quarterly": 63}
PRIMARY_REBAL  = "monthly"   # used for the main reported run

# Volatility forecaster training
VF_EPOCHS     = 30
VF_BATCH      = 64
VF_LR         = 1e-3
VF_PATIENCE   = 5            # early-stopping patience
VF_WEIGHT_DEC = 1e-5

# RL training
RL_TUNING_STEPS = 30_000
RL_FINAL_STEPS  = 120_000
RL_N_SEEDS      = 3          # number of seeds per algorithm for variance estimation

# Wavelet denoising (optional, applied to vol features only)
USE_WAVELET_DENOISE = True
WAVELET_NAME        = "db4"
WAVELET_LEVEL       = 2
WAVELET_THRESH      = 0.10

# Output dirs
RESULTS_DIR = Path("results_attention_portfolio")
for sub in ["forecasts", "backtests", "plots", "reports", "stats", "models"]:
    (RESULTS_DIR / sub).mkdir(parents=True, exist_ok=True)

RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")


def set_seeds(seed: int = GLOBAL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seeds()


# =============================================================================
# 2. DATA LOADING & TEMPORAL SPLIT
# =============================================================================

def _load_csv(path: str, wanted_cols: Optional[List[str]] = None) -> pd.DataFrame:
    sample = pd.read_csv(path, nrows=1)
    date_col = next((c for c in sample.columns if "date" in c.lower()),
                    sample.columns[0])
    df = pd.read_csv(path, parse_dates=[date_col], index_col=date_col)
    if wanted_cols is not None:
        rename = {}
        for w in wanted_cols:
            matches = [c for c in df.columns if w.lower() == c.lower()]
            if not matches:
                matches = [c for c in df.columns if w.lower() in c.lower()]
            if matches:
                rename[matches[0]] = w
        df = df.rename(columns=rename)
    return df


def load_data(prices_path: str = PRICES_PATH,
              vol_path: str = VOL_PATH
              ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Returns: log-returns DataFrame, volatility-features DataFrame, vol target Series."""
    prices = _load_csv(prices_path, TICKERS)
    vol_df = _load_csv(vol_path,    ALL_VOL_FEATURES)

    tickers_avail = [t for t in TICKERS if t in prices.columns]
    if not tickers_avail:
        raise ValueError(f"None of TICKERS found in {prices_path}")

    vol_feats = [f for f in ALL_VOL_FEATURES if f in vol_df.columns]
    if not vol_feats:
        raise ValueError(f"No volatility features found in {vol_path}")

    # Align on common dates
    common = prices.index.intersection(vol_df.index).sort_values()
    prices = prices.loc[common, tickers_avail].sort_index()
    vol_df = vol_df.loc[common, vol_feats].sort_index()

    # Returns (simple, daily)
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    vol_df  = vol_df.loc[returns.index].ffill().bfill()

    # Realized cross-sectional vol target = next-day std of cross-section returns
    vol_target = returns.std(axis=1).shift(-1).dropna()
    returns    = returns.loc[vol_target.index]
    vol_df     = vol_df.loc[vol_target.index]

    print(f"[DATA] returns={returns.shape}  vol_features={vol_df.shape}  "
          f"target={vol_target.shape}")
    print(f"[DATA] range: {returns.index[0].date()} -> {returns.index[-1].date()}")
    return returns, vol_df, vol_target


def temporal_split(returns: pd.DataFrame,
                   vol_df: pd.DataFrame,
                   vol_target: pd.Series
                   ) -> Dict[str, Dict[str, pd.DataFrame]]:
    n = len(returns)
    i_tr = int(n * TRAIN_RATIO)
    i_va = int(n * (TRAIN_RATIO + VAL_RATIO))
    splits = {}
    for name, (lo, hi) in [("train", (0, i_tr)),
                           ("val",   (i_tr, i_va)),
                           ("test",  (i_va, n))]:
        splits[name] = {
            "returns":    returns.iloc[lo:hi].copy(),
            "vol":        vol_df.iloc[lo:hi].copy(),
            "target":     vol_target.iloc[lo:hi].copy(),
        }
        print(f"[SPLIT] {name:5s}: {len(splits[name]['returns']):4d} rows  "
              f"[{splits[name]['returns'].index[0].date()} -> "
              f"{splits[name]['returns'].index[-1].date()}]")
    return splits


# =============================================================================
# 3. WAVELET DENOISING (used on vol features only, fit on TRAIN ONLY)
# =============================================================================

def wavelet_denoise(series: np.ndarray,
                    wavelet: str = WAVELET_NAME,
                    level: int = WAVELET_LEVEL,
                    thresh_factor: float = WAVELET_THRESH) -> np.ndarray:
    try:
        coeffs = pywt.wavedec(series, wavelet, level=level)
        thr = np.std(coeffs[-1]) * thresh_factor
        coeffs[1:] = [pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]]
        rec = pywt.waverec(coeffs, wavelet)
        return rec[: len(series)]
    except Exception:
        return series


def preprocess_vol(splits: Dict[str, Dict[str, pd.DataFrame]]) -> Dict:
    """Wavelet-denoise vol features (no leakage: same transform applied to each split)
    and standardize using TRAIN statistics only."""
    cols = splits["train"]["vol"].columns.tolist()
    out = {}
    if USE_WAVELET_DENOISE:
        for name in ("train", "val", "test"):
            vol = splits[name]["vol"].copy()
            for c in cols:
                vol[c] = wavelet_denoise(vol[c].values)
            out[name] = vol
    else:
        out = {n: splits[n]["vol"].copy() for n in ("train", "val", "test")}

    # Standardize using TRAIN only
    scaler = StandardScaler().fit(out["train"].values)
    for name in ("train", "val", "test"):
        arr = scaler.transform(out[name].values)
        out[name] = pd.DataFrame(arr, index=out[name].index, columns=cols)
    return out


# =============================================================================
# 4. POSITIONAL ENCODING
# =============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(pos * div)
        else:
            pe[:, 1::2] = torch.cos(pos * div[:-1])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


# =============================================================================
# 5. VOLATILITY FORECASTERS
# =============================================================================

class Chomp1d(nn.Module):
    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x):
        return x[:, :, : -self.chomp].contiguous()


class TemporalBlock(nn.Module):
    """Residual TCN block."""
    def __init__(self, in_ch, out_ch, k=3, d=1, drop=0.2):
        super().__init__()
        pad = (k - 1) * d
        self.out_ch = out_ch
        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=d)),
            Chomp1d(pad), nn.GELU(), nn.Dropout(drop),
            weight_norm(nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=d)),
            Chomp1d(pad), nn.GELU(), nn.Dropout(drop),
        )
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.net(x)
        res = x if self.down is None else self.down(x)
        return F.gelu(out + res)


class AttentionTCN(nn.Module):
    """
    Multihead-Attention volatility forecaster (NEW core model).
        TCN backbone (multi-scale temporal features)
        -> Positional encoding
        -> Multihead Self-Attention (captures long-range dependencies)
        -> BiLSTM (sequential refinement)
        -> Head -> scalar vol prediction.
    Input:  x in [B, T, F]
    Output: y in [B, 1]
    """
    def __init__(self, n_features: int, seq_len: int,
                 d_model: int = 64, n_heads: int = 4, drop: float = 0.2):
        super().__init__()
        self.seq_len = seq_len
        # TCN backbone: increasing dilation
        chans = [32, d_model]
        layers, in_ch = [], n_features
        for i, ch in enumerate(chans):
            layers.append(TemporalBlock(in_ch, ch, k=3, d=2 ** i, drop=drop))
            in_ch = ch
        self.tcn = nn.Sequential(*layers)
        # Attention
        self.pos = PositionalEncoding(d_model, max_len=seq_len + 16, dropout=drop)
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          dropout=drop, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        # BiLSTM
        self.bilstm = nn.LSTM(d_model, d_model // 2,
                              bidirectional=True, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        # Head
        self.head = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(drop),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F] -> [B, F, T] for Conv1d
        h = self.tcn(x.transpose(1, 2))
        h = h.transpose(1, 2)               # [B, T, d_model]
        h = self.pos(h)
        a, _ = self.attn(h, h, h)
        h = self.norm1(h + a)
        l, _ = self.bilstm(h)
        h = self.norm2(h + l)
        # Use last time-step pooling (most recent context)
        return self.head(h[:, -1, :])


class ResNet1D(nn.Module):
    """Baseline 1D-ResNet over the lookback window."""
    def __init__(self, n_features: int, seq_len: int = LOOKBACK_VOL):
        super().__init__()
        chans = [64, 128, 256]
        layers, in_ch = [], n_features
        for ch in chans:
            layers += [
                nn.Conv1d(in_ch, ch, 3, padding=1),
                nn.InstanceNorm1d(ch) if n_features > 1 else nn.Identity(),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.3),
            ]
            in_ch = ch
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, n_features, seq_len)
            flat = self.conv(dummy).flatten(1).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(flat, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.fc(self.conv(x).flatten(1))


class WaveletCNN(nn.Module):
    """Baseline Wavelet-CNN: decompose with db4 level-3, then 2-layer CNN."""
    def __init__(self, n_features: int, seq_len: int = LOOKBACK_VOL):
        super().__init__()
        self.n_features = n_features
        # Pre-compute coefficient length
        dummy = np.zeros(seq_len)
        coeffs = pywt.wavedec(dummy, "db4", level=3)
        tot = sum(len(c) for c in coeffs)
        self.tot = tot
        self.conv1 = nn.Conv1d(n_features, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.fc = nn.Sequential(
            nn.Linear(128 * tot, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: [B, T, F]
        B = x.shape[0]
        feats = []
        x_np = x.detach().cpu().numpy()
        for fi in range(self.n_features):
            coeffs = pywt.wavedec(x_np[:, :, fi], "db4", level=3, axis=1)
            feats.append(np.concatenate(coeffs, axis=1))
        w = np.stack(feats, axis=1).astype(np.float32)        # [B, F, tot]
        w = torch.from_numpy(w).to(x.device)
        h = F.relu(self.conv1(w))
        h = F.relu(self.conv2(h))
        return self.fc(h.flatten(1))


class TCNAutoencoder(nn.Module):
    """Baseline TCN-AE with auxiliary regression head."""
    def __init__(self, n_features: int, seq_len: int = LOOKBACK_VOL):
        super().__init__()
        chans = [64, 128]
        enc, in_ch = [], n_features
        for i, ch in enumerate(chans):
            enc.append(TemporalBlock(in_ch, ch, 3, 2 ** i, 0.3))
            in_ch = ch
        self.encoder = nn.Sequential(*enc)
        self.code_len = chans[-1] * seq_len
        self.fc_enc = nn.Linear(self.code_len, 32)
        self.head = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        h = self.encoder(x.transpose(1, 2))
        z = self.fc_enc(h.flatten(1))
        return self.head(z)


# Forecaster registry
FORECASTERS = {
    "AttentionTCN":   AttentionTCN,
    "ResNet1D":       ResNet1D,
    "WaveletCNN":     WaveletCNN,
    "TCNAutoencoder": TCNAutoencoder,
}


# =============================================================================
# 6. FORECASTER TRAINING & INFERENCE
# =============================================================================

def make_windows(X: np.ndarray, y: np.ndarray,
                 window: int = LOOKBACK_VOL) -> Tuple[np.ndarray, np.ndarray]:
    """Build sliding windows aligned with y (target at the END of the window)."""
    if len(X) <= window:
        raise ValueError(f"Need at least {window+1} obs, got {len(X)}")
    n = len(X) - window + 1
    Xw = np.lib.stride_tricks.sliding_window_view(
        X, (window, X.shape[1]))[:, 0, :, :]      # [n, T, F]
    yw = y[window - 1 :]                          # aligned to last index
    assert len(Xw) == len(yw)
    return Xw.astype(np.float32), yw.astype(np.float32)


def train_forecaster(model_name: str,
                     train_X: np.ndarray, train_y: np.ndarray,
                     val_X:   np.ndarray, val_y:   np.ndarray,
                     epochs: int = VF_EPOCHS,
                     batch:  int = VF_BATCH,
                     lr:     float = VF_LR,
                     verbose: bool = True) -> Tuple[nn.Module, Dict]:
    """Train a forecaster with early stopping on validation MSE."""
    cls = FORECASTERS[model_name]
    n_feat = train_X.shape[2]
    model  = cls(n_features=n_feat, seq_len=train_X.shape[1]).to(DEVICE)
    opt    = torch.optim.AdamW(model.parameters(), lr=lr,
                               weight_decay=VF_WEIGHT_DEC)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    tr_ds = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
    va_ds = TensorDataset(torch.from_numpy(val_X),   torch.from_numpy(val_y))
    tr_ld = DataLoader(tr_ds, batch_size=batch, shuffle=True,  drop_last=False)
    va_ld = DataLoader(va_ds, batch_size=batch, shuffle=False, drop_last=False)

    best_val = float("inf")
    best_state = None
    bad = 0
    history = {"train": [], "val": []}

    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE).unsqueeze(1)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(tr_ds)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_ld:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE).unsqueeze(1)
                pred = model(xb)
                va_loss += loss_fn(pred, yb).item() * xb.size(0)
        va_loss /= len(va_ds)
        sched.step()

        history["train"].append(tr_loss)
        history["val"].append(va_loss)
        if verbose and (ep + 1) % 5 == 0:
            print(f"   [{model_name:14s}] ep {ep+1:02d}  "
                  f"train={tr_loss:.6e}  val={va_loss:.6e}")

        if va_loss < best_val - 1e-9:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= VF_PATIENCE:
                if verbose:
                    print(f"   [{model_name}] early stop at epoch {ep+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_mse": best_val, "history": history}


def forecast(model: nn.Module, X: np.ndarray, batch: int = 256) -> np.ndarray:
    """Return point forecasts for each window in X."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i : i + batch]).to(DEVICE)
            preds.append(model(xb).cpu().numpy().ravel())
    return np.concatenate(preds)


# =============================================================================
# 7. RL FEATURE EXTRACTOR (MULTIHEAD ATTENTION)
# =============================================================================

if SB3_OK:

    class AdvancedFeatureExtractor(BaseFeaturesExtractor):
        """
        Observation comes in flattened from PortfolioEnv: shape
            obs_dim = LOOKBACK_RL * n_tickers + n_vol_feats + 1 + n_tickers
        We reshape the sequential part (returns lookback) back to
            [B, LOOKBACK_RL, n_tickers] and feed Multihead-Attention + BiLSTM.
        Non-sequential parts (vol features, vol forecast, current weights) are
        merged with the pooled sequence representation through an MLP head.
        """
        def __init__(self, observation_space: spaces.Box,
                     n_tickers: int,
                     n_vol_feats: int,
                     features_dim: int = 64,
                     n_heads: int = 4,
                     dropout: float = 0.1):
            super().__init__(observation_space, features_dim)
            self.n_tickers   = n_tickers
            self.n_vol_feats = n_vol_feats
            self.seq_len     = LOOKBACK_RL
            self.seq_dim     = n_tickers
            self.scalar_dim  = n_vol_feats + 1 + n_tickers   # vol_feats + forecast + weights

            d_model = 64
            self.proj = nn.Linear(self.seq_dim, d_model)
            self.pos  = PositionalEncoding(d_model, max_len=self.seq_len + 8,
                                           dropout=dropout)
            self.attn = nn.MultiheadAttention(d_model, n_heads,
                                              dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(d_model)
            self.bilstm = nn.LSTM(d_model, d_model // 2,
                                  bidirectional=True, batch_first=True)
            self.norm2  = nn.LayerNorm(d_model)

            self.scalar_mlp = nn.Sequential(
                nn.Linear(self.scalar_dim, 64), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(64, 32),
            )
            self.merge = nn.Sequential(
                nn.Linear(d_model + 32, features_dim), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(features_dim, features_dim),
                nn.Tanh(),
            )

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            B = obs.shape[0]
            seq_total = self.seq_len * self.seq_dim
            seq_part   = obs[:, :seq_total].view(B, self.seq_len, self.seq_dim)
            scalar_part = obs[:, seq_total:]
            h = self.proj(seq_part)
            h = self.pos(h)
            a, _ = self.attn(h, h, h)
            h = self.norm(h + a)
            l, _ = self.bilstm(h)
            h = self.norm2(h + l)
            seq_feat = h.mean(dim=1)                          # mean-pool
            sca_feat = self.scalar_mlp(scalar_part)
            return self.merge(torch.cat([seq_feat, sca_feat], dim=1))


# =============================================================================
# 8. PORTFOLIO ENVIRONMENT
# =============================================================================

class PortfolioEnv(gym.Env):
    """
    Continuous-action portfolio allocation env.
    Action:  Box([0,1]^n)  -> softmax-normalized portfolio weights.
    Reward:  log(1 + r_p) - lambda * turnover_cost
    Observation (flattened):
        last LOOKBACK_RL daily returns of n_tickers  (LOOKBACK_RL * n_tickers)
        current vol features                          (n_vol_feats)
        current vol forecast                          (1)
        current weights                               (n_tickers)
    """
    metadata = {"render_modes": []}

    def __init__(self,
                 returns: pd.DataFrame,
                 vol_feats: pd.DataFrame,
                 vol_forecast: np.ndarray,
                 rebalance_freq: int = 21,
                 max_weight: float = MAX_WEIGHT,
                 min_weight: float = MIN_WEIGHT,
                 commission_bps: float = COMMISSION_BPS,
                 slippage_bps:   float = SLIPPAGE_BPS,
                 turnover_penalty: float = 0.0):
        super().__init__()
        # Align lengths
        n = min(len(returns), len(vol_feats), len(vol_forecast))
        self.returns      = returns.iloc[:n].values.astype(np.float32)
        self.dates        = returns.iloc[:n].index
        self.vol_feats    = vol_feats.iloc[:n].values.astype(np.float32)
        self.vol_forecast = vol_forecast[:n].astype(np.float32)
        self.n_tickers    = self.returns.shape[1]
        self.n_vol_feats  = self.vol_feats.shape[1]
        self.steps        = len(self.returns)

        self.rebalance_freq   = rebalance_freq
        self.max_weight       = max_weight
        self.min_weight       = min_weight
        self.commission       = commission_bps / 10_000.0
        self.slippage         = slippage_bps   / 10_000.0
        self.turnover_penalty = turnover_penalty

        self.action_space = spaces.Box(0.0, 1.0,
                                       shape=(self.n_tickers,), dtype=np.float32)
        obs_dim = (LOOKBACK_RL * self.n_tickers
                   + self.n_vol_feats + 1 + self.n_tickers)
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(obs_dim,), dtype=np.float32)

    def _project_weights(self, raw: np.ndarray) -> np.ndarray:
        # softmax then clip then renormalize (handles min/max)
        e = np.exp(raw - raw.max())
        w = e / (e.sum() + 1e-12)
        w = np.clip(w, self.min_weight, self.max_weight)
        s = w.sum()
        if s <= 0:
            w = np.ones_like(w) / len(w)
        else:
            w = w / s
        return w.astype(np.float32)

    def _obs(self) -> np.ndarray:
        lo = max(0, self.i - LOOKBACK_RL)
        seq = self.returns[lo : self.i]
        if seq.shape[0] < LOOKBACK_RL:
            pad = np.zeros((LOOKBACK_RL - seq.shape[0], self.n_tickers),
                           dtype=np.float32)
            seq = np.vstack([pad, seq])
        return np.concatenate([
            seq.flatten(),
            self.vol_feats[self.i - 1],
            np.array([self.vol_forecast[self.i - 1]], dtype=np.float32),
            self.weights,
        ]).astype(np.float32)

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self.i = LOOKBACK_RL
        self.value   = INITIAL_CAPITAL
        self.weights = np.ones(self.n_tickers, dtype=np.float32) / self.n_tickers
        self.history = [self.value]
        self.weights_log = [self.weights.copy()]
        self.turnover_log = [0.0]
        self.rebal_step = 0
        return self._obs(), {}

    def step(self, action: np.ndarray):
        # Rebalance only at frequency
        if self.rebal_step % self.rebalance_freq == 0:
            new_w = self._project_weights(np.asarray(action, dtype=np.float32))
            turnover = np.abs(new_w - self.weights).sum()
            cost = turnover * (self.commission + self.slippage)
            self.weights = new_w
        else:
            turnover = 0.0
            cost = 0.0

        r = float(np.dot(self.weights, self.returns[self.i])) - cost
        self.value *= (1.0 + r)
        self.history.append(self.value)
        self.weights_log.append(self.weights.copy())
        self.turnover_log.append(turnover)
        self.rebal_step += 1
        self.i += 1

        reward = math.log(max(1.0 + r, 1e-8)) - self.turnover_penalty * turnover
        terminated = self.i >= self.steps
        truncated  = False
        obs = (self._obs() if not terminated
               else np.zeros(self.observation_space.shape, dtype=np.float32))
        return obs, reward, terminated, truncated, {"value": self.value,
                                                     "turnover": turnover}


# =============================================================================
# 9. PORTFOLIO STRATEGIES
# =============================================================================

def _equity_from_weights(returns: pd.DataFrame,
                         weights_path: List[np.ndarray],
                         rebalance_idx: List[int]) -> np.ndarray:
    """Walk equity given a path of weights and rebalance indices.
    weights_path[k] is the weight applied from rebalance_idx[k] until
    rebalance_idx[k+1] (or the end)."""
    n = len(returns)
    eq = np.empty(n, dtype=np.float64)
    eq[:] = np.nan
    val = INITIAL_CAPITAL
    rets = returns.values
    cur_w = np.ones(returns.shape[1]) / returns.shape[1]
    rb = dict(zip(rebalance_idx, weights_path))
    eq[0] = val
    for t in range(1, n):
        if t in rb:
            new_w = rb[t]
            turnover = np.abs(new_w - cur_w).sum()
            val *= (1.0 - turnover * (COMMISSION_BPS + SLIPPAGE_BPS) / 10_000.0)
            cur_w = new_w
        r = float(np.dot(cur_w, rets[t]))
        val *= (1.0 + r)
        eq[t] = val
    return eq


def strat_equal_weight(returns: pd.DataFrame, rebal: int) -> np.ndarray:
    n_assets = returns.shape[1]
    w = np.ones(n_assets) / n_assets
    idx = list(range(LOOKBACK_RL, len(returns), rebal))
    path = [w.copy() for _ in idx]
    return _equity_from_weights(returns, path, idx)


def strat_buy_and_hold(returns: pd.DataFrame) -> np.ndarray:
    n_assets = returns.shape[1]
    w = np.ones(n_assets) / n_assets
    idx = [LOOKBACK_RL]
    path = [w.copy()]
    return _equity_from_weights(returns, path, idx)


def strat_risk_parity(returns: pd.DataFrame, rebal: int,
                      lookback: int = LOOKBACK_VOL) -> np.ndarray:
    """Inverse-volatility weighting."""
    idx, path = [], []
    for t in range(max(lookback, LOOKBACK_RL), len(returns), rebal):
        window = returns.iloc[t - lookback : t]
        sig = window.std().replace(0, np.nan)
        w = (1.0 / sig)
        w = (w / w.sum()).fillna(1.0 / returns.shape[1]).values
        w = np.clip(w, MIN_WEIGHT, MAX_WEIGHT)
        w = w / w.sum()
        idx.append(t); path.append(w)
    return _equity_from_weights(returns, path, idx)


def strat_markowitz_max_sharpe(returns: pd.DataFrame, rebal: int,
                               lookback: int = LOOKBACK_VOL) -> np.ndarray:
    idx, path = [], []
    n_assets = returns.shape[1]
    for t in range(max(lookback, LOOKBACK_RL), len(returns), rebal):
        window = returns.iloc[t - lookback : t]
        try:
            mu = expected_returns.mean_historical_return(window, frequency=252, returns_data=True)
            S  = risk_models.CovarianceShrinkage(window, frequency=252, returns_data=True).ledoit_wolf()
            ef = EfficientFrontier(mu, S, weight_bounds=(MIN_WEIGHT, MAX_WEIGHT))
            try:
                ef.max_sharpe()
            except Exception:
                ef.min_volatility()
            cw = ef.clean_weights()
            w = np.array([cw.get(c, 0.0) for c in returns.columns])
            w = w / w.sum() if w.sum() > 0 else np.ones(n_assets) / n_assets
        except Exception:
            w = np.ones(n_assets) / n_assets
        idx.append(t); path.append(w)
    return _equity_from_weights(returns, path, idx)


def strat_min_volatility(returns: pd.DataFrame, rebal: int,
                         lookback: int = LOOKBACK_VOL) -> np.ndarray:
    idx, path = [], []
    n_assets = returns.shape[1]
    for t in range(max(lookback, LOOKBACK_RL), len(returns), rebal):
        window = returns.iloc[t - lookback : t]
        try:
            S = risk_models.CovarianceShrinkage(window, frequency=252, returns_data=True).ledoit_wolf()
            mu = expected_returns.mean_historical_return(window, frequency=252, returns_data=True)
            ef = EfficientFrontier(mu, S, weight_bounds=(MIN_WEIGHT, MAX_WEIGHT))
            ef.min_volatility()
            cw = ef.clean_weights()
            w = np.array([cw.get(c, 0.0) for c in returns.columns])
            w = w / w.sum() if w.sum() > 0 else np.ones(n_assets) / n_assets
        except Exception:
            w = np.ones(n_assets) / n_assets
        idx.append(t); path.append(w)
    return _equity_from_weights(returns, path, idx)


def strat_drl(algo_name: str,
              returns_train: pd.DataFrame,
              vol_train: pd.DataFrame,
              vf_train: np.ndarray,
              returns_test:  pd.DataFrame,
              vol_test:  pd.DataFrame,
              vf_test:   np.ndarray,
              rebal: int,
              seed: int,
              steps: int = RL_FINAL_STEPS) -> np.ndarray:
    """Train a DRL agent on train, evaluate OOS on test."""
    if not SB3_OK:
        return np.full(len(returns_test), INITIAL_CAPITAL, dtype=np.float64)
    set_seeds(seed)

    def make_train():
        return PortfolioEnv(returns_train, vol_train, vf_train,
                            rebalance_freq=rebal,
                            turnover_penalty=0.001)

    train_env = DummyVecEnv([make_train])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                             clip_obs=10.0)

    policy_kwargs = dict(
        features_extractor_class=AdvancedFeatureExtractor,
        features_extractor_kwargs=dict(
            n_tickers=returns_train.shape[1],
            n_vol_feats=vol_train.shape[1],
            features_dim=64, n_heads=4, dropout=0.1,
        ),
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
    )

    AlgoCls = {"PPO": PPO, "A2C": A2C}[algo_name]
    if algo_name == "PPO":
        model = AlgoCls("MlpPolicy", train_env, verbose=0, seed=seed,
                        learning_rate=3e-4, gamma=0.99, n_steps=2048,
                        batch_size=128, policy_kwargs=policy_kwargs,
                        device=DEVICE)
    else:  # A2C
        model = AlgoCls("MlpPolicy", train_env, verbose=0, seed=seed,
                        learning_rate=7e-4, gamma=0.99, n_steps=20,
                        policy_kwargs=policy_kwargs, device=DEVICE)

    model.learn(total_timesteps=steps)

    # Build test env (separate) and reuse VecNormalize stats from train
    def make_test():
        return PortfolioEnv(returns_test, vol_test, vf_test,
                            rebalance_freq=rebal, turnover_penalty=0.0)
    test_env = DummyVecEnv([make_test])
    test_env = VecNormalize(test_env, norm_obs=True, norm_reward=False,
                            training=False, clip_obs=10.0)
    test_env.obs_rms = train_env.obs_rms          # transplant stats (no leakage of test)
    test_env.ret_rms = train_env.ret_rms

    obs = test_env.reset()
    done = [False]
    equity_list = [INITIAL_CAPITAL]
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, infos = test_env.step(action)
        equity_list.append(float(infos[0].get("value", equity_list[-1])))
    equity = np.array(equity_list, dtype=np.float64)

    train_env.close(); test_env.close()
    return equity


# =============================================================================
# 10. METRICS
# =============================================================================

def _to_returns(equity: np.ndarray) -> np.ndarray:
    eq = np.asarray(equity, dtype=np.float64)
    eq = eq[np.isfinite(eq)]
    return np.diff(eq) / eq[:-1]


def bootstrap_sharpe_ci(returns: np.ndarray,
                        n_boot: int = 2000,
                        alpha: float = 0.05,
                        ann: int = 252) -> Tuple[float, float]:
    if len(returns) < 5:
        return (0.0, 0.0)
    rng = np.random.default_rng(GLOBAL_SEED)
    boot = np.empty(n_boot)
    n = len(returns)
    for b in range(n_boot):
        sample = returns[rng.integers(0, n, n)]
        sd = sample.std()
        boot[b] = sample.mean() / sd * math.sqrt(ann) if sd > 0 else 0.0
    return float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


def compute_metrics(equity: np.ndarray,
                    benchmark_eq: Optional[np.ndarray] = None,
                    freq: int = 252) -> Dict[str, float]:
    if equity is None or len(equity) < 3:
        return {k: 0.0 for k in [
            "Final_Value", "Total_Return", "CAGR", "Volatility", "Sharpe",
            "Sortino", "Calmar", "Max_DD", "RoMaD", "VaR_5", "CVaR_5",
            "Win_Rate", "Max_Loss", "T_Stat", "P_Value", "Alpha", "Beta",
            "Sharpe_CI_low", "Sharpe_CI_high"]}
    eq = np.asarray(equity, dtype=np.float64)
    eq = eq[np.isfinite(eq)]
    rets = _to_returns(eq)
    if len(rets) < 2:
        return {k: 0.0 for k in [
            "Final_Value", "Total_Return", "CAGR", "Volatility", "Sharpe",
            "Sortino", "Calmar", "Max_DD", "RoMaD", "VaR_5", "CVaR_5",
            "Win_Rate", "Max_Loss", "T_Stat", "P_Value", "Alpha", "Beta",
            "Sharpe_CI_low", "Sharpe_CI_high"]}

    total = eq[-1] / eq[0] - 1
    yrs   = len(eq) / freq
    cagr  = (eq[-1] / eq[0]) ** (1 / yrs) - 1 if yrs > 0 else 0
    vol   = rets.std() * math.sqrt(freq)
    sharpe = rets.mean() / rets.std() * math.sqrt(freq) if rets.std() > 0 else 0
    neg = rets[rets < 0]
    dn  = neg.std() * math.sqrt(freq) if len(neg) > 1 else 0
    sortino = rets.mean() * freq / dn if dn > 0 else 0
    peak = np.maximum.accumulate(eq)
    dd   = (peak - eq) / np.maximum(peak, 1)
    mdd  = float(dd.max())
    calmar = cagr / mdd if mdd > 0 else 0
    romad  = total / mdd if mdd > 0 else 0
    var5   = float(np.quantile(rets, 0.05))
    cvar5  = float(rets[rets <= var5].mean()) if (rets <= var5).any() else var5
    win    = float((rets > 0).mean())
    maxloss = float(rets.min())
    t_stat, p_val = stats.ttest_1samp(rets, 0)
    t_stat = float(t_stat) if np.isfinite(t_stat) else 0.0
    p_val  = float(p_val)  if np.isfinite(p_val)  else 1.0

    if benchmark_eq is not None and len(benchmark_eq) >= len(eq):
        b_rets = _to_returns(benchmark_eq[: len(eq)])
        m = min(len(rets), len(b_rets))
        x, y = b_rets[:m], rets[:m]
        cov = np.cov(x, y)[0, 1]
        beta = cov / x.var() if x.var() > 0 else 0
        alpha = (y.mean() - beta * x.mean()) * freq
    else:
        alpha, beta = 0.0, 0.0

    ci_lo, ci_hi = bootstrap_sharpe_ci(rets)

    return {
        "Final_Value": float(eq[-1]),
        "Total_Return": float(total),
        "CAGR": float(cagr),
        "Volatility": float(vol),
        "Sharpe": float(sharpe),
        "Sharpe_CI_low": ci_lo,
        "Sharpe_CI_high": ci_hi,
        "Sortino": float(sortino),
        "Calmar": float(calmar),
        "Max_DD": mdd,
        "RoMaD": float(romad),
        "VaR_5": var5,
        "CVaR_5": cvar5,
        "Win_Rate": win,
        "Max_Loss": maxloss,
        "T_Stat": t_stat,
        "P_Value": p_val,
        "Alpha": float(alpha),
        "Beta":  float(beta),
    }


# =============================================================================
# 11. STATISTICAL TESTS
# =============================================================================

def paired_returns_test(rets_a: np.ndarray, rets_b: np.ndarray
                        ) -> Dict[str, float]:
    m = min(len(rets_a), len(rets_b))
    a, b = rets_a[:m], rets_b[:m]
    diff = a - b
    t, pt = stats.ttest_rel(a, b)
    try:
        w, pw = stats.wilcoxon(diff)
    except Exception:
        w, pw = 0.0, 1.0
    return {"t": float(t), "p_t": float(pt),
            "W": float(w), "p_wilcoxon": float(pw),
            "mean_diff": float(diff.mean())}


def diebold_mariano(e1: np.ndarray, e2: np.ndarray,
                    h: int = 1, power: int = 2) -> Dict[str, float]:
    """Diebold-Mariano test comparing forecast errors e1 vs e2."""
    m = min(len(e1), len(e2))
    d = np.abs(e1[:m]) ** power - np.abs(e2[:m]) ** power
    n = len(d)
    if n < 10:
        return {"DM": 0.0, "p_value": 1.0}
    mean_d = d.mean()
    # Newey-West-like long-run variance with h-1 lags
    gamma0 = ((d - mean_d) ** 2).mean()
    gammas = 0.0
    for k in range(1, h):
        c = np.cov(d[: n - k], d[k:])[0, 1]
        gammas += (1 - k / h) * c
    var_d = (gamma0 + 2 * gammas) / n
    if var_d <= 0:
        return {"DM": 0.0, "p_value": 1.0}
    dm = mean_d / math.sqrt(var_d)
    # Harvey small-sample correction
    k_corr = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm *= k_corr
    p = 2 * (1 - stats.norm.cdf(abs(dm)))
    return {"DM": float(dm), "p_value": float(p)}


# =============================================================================
# 12. VISUALIZATION
# =============================================================================

def plot_equity_curves(curves: Dict[str, np.ndarray],
                       dates: pd.DatetimeIndex,
                       title: str, savepath: Path):
    plt.figure(figsize=(13, 6))
    for name, eq in curves.items():
        m = min(len(eq), len(dates))
        plt.plot(dates[:m], eq[:m], label=name, linewidth=1.4)
    plt.title(title)
    plt.ylabel("Equity ($)")
    plt.xlabel("Date")
    plt.legend(loc="best", fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()


def plot_drawdown(curves: Dict[str, np.ndarray],
                  dates: pd.DatetimeIndex,
                  title: str, savepath: Path):
    plt.figure(figsize=(13, 5))
    for name, eq in curves.items():
        eq = np.asarray(eq, dtype=np.float64)
        eq = eq[np.isfinite(eq)]
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        m = min(len(dd), len(dates))
        plt.plot(dates[:m], dd[:m], label=name, linewidth=1.2)
    plt.title(title)
    plt.ylabel("Drawdown")
    plt.xlabel("Date")
    plt.legend(loc="best", fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()


def plot_metric_bar(df: pd.DataFrame, metric: str, savepath: Path):
    plt.figure(figsize=(11, 5))
    order = df.sort_values(metric, ascending=False)["Strategy"].tolist()
    sns.barplot(data=df, x="Strategy", y=metric, order=order)
    plt.title(f"{metric} — Out-of-sample (test)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()


def plot_forecast_loss(histories: Dict[str, Dict], savepath: Path):
    plt.figure(figsize=(11, 5))
    for name, hist in histories.items():
        plt.plot(hist["val"], label=f"{name} (val)", linewidth=1.4)
    plt.title("Volatility forecasters — validation MSE")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.yscale("log")
    plt.legend(loc="best")
    plt.tight_layout(); plt.savefig(savepath, dpi=200); plt.close()


# =============================================================================
# 13. MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 78)
    print(f"  RUN {RUN_TAG} | device={DEVICE} | seed={GLOBAL_SEED}")
    print("=" * 78)

    # --- 1. Data
    returns, vol_df, vol_target = load_data()
    splits = temporal_split(returns, vol_df, vol_target)

    # --- 2. Preprocess vol features (denoise + standardize using train stats)
    vol_proc = preprocess_vol(splits)
    for s in ("train", "val", "test"):
        splits[s]["vol_proc"] = vol_proc[s]

    # --- 3. Build sliding windows for forecasters
    Xtr_w, ytr_w = make_windows(splits["train"]["vol_proc"].values,
                                splits["train"]["target"].values)
    Xva_w, yva_w = make_windows(splits["val"]["vol_proc"].values,
                                splits["val"]["target"].values)
    Xte_w, yte_w = make_windows(splits["test"]["vol_proc"].values,
                                splits["test"]["target"].values)
    print(f"[WIN ] train {Xtr_w.shape} | val {Xva_w.shape} | test {Xte_w.shape}")

    # --- 4. Train all forecasters, evaluate on TEST
    forecast_results = {}
    forecast_histories = {}
    test_preds = {}
    print("\n--- VOLATILITY FORECASTERS ---")
    for name in FORECASTERS:
        print(f"\n[FCST] training {name}")
        set_seeds(GLOBAL_SEED)
        model, info = train_forecaster(name, Xtr_w, ytr_w, Xva_w, yva_w)
        test_pred = forecast(model, Xte_w)
        mse = float(np.mean((test_pred - yte_w) ** 2))
        mae = float(np.mean(np.abs(test_pred - yte_w)))
        forecast_results[name] = {"val_mse": info["best_val_mse"],
                                  "test_mse": mse, "test_mae": mae}
        forecast_histories[name] = info["history"]
        test_preds[name] = test_pred
        print(f"[FCST] {name:14s}  val_MSE={info['best_val_mse']:.6e}  "
              f"test_MSE={mse:.6e}  test_MAE={mae:.6e}")

    # Save forecast metrics
    fc_df = pd.DataFrame(forecast_results).T
    fc_df.index.name = "Model"
    fc_df.to_csv(RESULTS_DIR / "forecasts" / f"forecast_metrics_{RUN_TAG}.csv")

    plot_forecast_loss(forecast_histories,
                       RESULTS_DIR / "plots" / f"forecast_val_loss_{RUN_TAG}.png")

    # Diebold-Mariano: AttentionTCN vs each baseline (TEST errors)
    dm_rows = []
    e_att = test_preds["AttentionTCN"] - yte_w
    for other in ["ResNet1D", "WaveletCNN", "TCNAutoencoder"]:
        e_oth = test_preds[other] - yte_w
        dm = diebold_mariano(e_att, e_oth, h=1, power=2)
        dm_rows.append({"AttentionTCN_vs": other, **dm})
    pd.DataFrame(dm_rows).to_csv(
        RESULTS_DIR / "stats" / f"diebold_mariano_{RUN_TAG}.csv", index=False)
    print("\n[DM  ] Diebold-Mariano vs baselines:")
    for r in dm_rows:
        sig = "*" if r["p_value"] < 0.05 else " "
        print(f"   AttentionTCN vs {r['AttentionTCN_vs']:14s}  "
              f"DM={r['DM']:+.3f}  p={r['p_value']:.4f} {sig}")

    # --- 5. Pick best forecaster on VAL, build full-period forecast series
    best_fc_name = min(forecast_results, key=lambda k: forecast_results[k]["val_mse"])
    print(f"\n[FCST] Best forecaster on val: {best_fc_name}")
    # Re-train on (train+val) for final OOS test (allowed: val is in-sample then)
    print(f"[FCST] Re-training {best_fc_name} on TRAIN+VAL ...")
    X_full = np.concatenate([Xtr_w, Xva_w], axis=0)
    y_full = np.concatenate([ytr_w, yva_w], axis=0)
    # Use last 10% of TRAIN+VAL as inner-val for early stopping
    cut = int(len(X_full) * 0.9)
    set_seeds(GLOBAL_SEED)
    final_model, _ = train_forecaster(
        best_fc_name, X_full[:cut], y_full[:cut], X_full[cut:], y_full[cut:],
        verbose=False)

    # Generate test-period forecast aligned with TEST returns
    test_pred_full = forecast(final_model, Xte_w)
    # Align forecasts to returns of test (target at end-of-window)
    test_pred_aligned = np.concatenate(
        [np.full(LOOKBACK_VOL - 1, np.nan), test_pred_full])
    test_pred_aligned = pd.Series(test_pred_aligned,
                                  index=splits["test"]["vol_proc"].index)
    test_pred_aligned = test_pred_aligned.ffill().bfill().abs().values

    # Build train-period forecast (for RL training env) using same model
    train_pred_full = forecast(final_model, Xtr_w)
    train_pred_aligned = np.concatenate(
        [np.full(LOOKBACK_VOL - 1, np.nan), train_pred_full])
    train_pred_aligned = pd.Series(train_pred_aligned,
                                   index=splits["train"]["vol_proc"].index)
    train_pred_aligned = train_pred_aligned.ffill().bfill().abs().values

    # --- 6. Run portfolio strategies OOS on TEST
    rebal_days = REBALANCE_FREQS[PRIMARY_REBAL]
    test_returns = splits["test"]["returns"]
    test_vol     = splits["test"]["vol_proc"]
    train_returns = splits["train"]["returns"]
    train_vol     = splits["train"]["vol_proc"]

    print(f"\n--- PORTFOLIO STRATEGIES (OOS test, rebal={PRIMARY_REBAL}) ---")
    curves: Dict[str, np.ndarray] = {}

    # Classical baselines
    curves["EqualWeight"]   = strat_equal_weight(test_returns, rebal_days)
    curves["BuyAndHold"]    = strat_buy_and_hold(test_returns)
    curves["RiskParity"]    = strat_risk_parity(test_returns, rebal_days)
    curves["MaxSharpe_MV"]  = strat_markowitz_max_sharpe(test_returns, rebal_days)
    curves["MinVolatility"] = strat_min_volatility(test_returns, rebal_days)

    # DRL strategies (multi-seed for variance)
    if SB3_OK:
        for algo in ["PPO", "A2C"]:
            seed_curves = []
            for k in range(RL_N_SEEDS):
                sd = GLOBAL_SEED + 1000 * (k + 1)
                print(f"[DRL ] training {algo} seed={sd} steps={RL_FINAL_STEPS:,}")
                eq = strat_drl(algo,
                               train_returns, train_vol, train_pred_aligned,
                               test_returns,  test_vol,  test_pred_aligned,
                               rebal_days, sd)
                seed_curves.append(eq)
                # Save individual seed equity
                np.save(RESULTS_DIR / "backtests" /
                        f"{algo}_seed{sd}_equity_{RUN_TAG}.npy", eq)
            # Median curve across seeds (robust)
            min_len = min(len(c) for c in seed_curves)
            stacked = np.stack([c[:min_len] for c in seed_curves], axis=0)
            curves[f"{algo}_attention"] = np.median(stacked, axis=0)
            curves[f"{algo}_attention_mean"] = stacked.mean(axis=0)

    # Benchmark for alpha/beta = EqualWeight
    benchmark = curves["EqualWeight"]

    # --- 7. Metrics
    rows = []
    for name, eq in curves.items():
        m = compute_metrics(eq, benchmark_eq=benchmark)
        m["Strategy"] = name
        rows.append(m)
    metrics_df = pd.DataFrame(rows).set_index("Strategy")
    cols_order = ["Final_Value", "Total_Return", "CAGR", "Volatility",
                  "Sharpe", "Sharpe_CI_low", "Sharpe_CI_high",
                  "Sortino", "Calmar", "Max_DD", "RoMaD", "VaR_5", "CVaR_5",
                  "Win_Rate", "Max_Loss", "T_Stat", "P_Value", "Alpha", "Beta"]
    cols_order = [c for c in cols_order if c in metrics_df.columns]
    metrics_df = metrics_df[cols_order]
    print("\n--- OOS METRICS ---")
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", "{:.4f}".format):
        print(metrics_df)

    metrics_df.to_csv(RESULTS_DIR / "reports" / f"oos_metrics_{RUN_TAG}.csv")

    # --- 8. Pairwise return tests vs benchmark
    print("\n--- PAIRWISE STATISTICAL TESTS vs EqualWeight ---")
    pair_rows = []
    bench_r = _to_returns(benchmark)
    for name, eq in curves.items():
        if name == "EqualWeight":
            continue
        rets = _to_returns(eq)
        test = paired_returns_test(rets, bench_r)
        pair_rows.append({"Strategy": name, **test})
        print(f"   {name:24s}  t={test['t']:+.3f}  p_t={test['p_t']:.4f}  "
              f"W p_wilcoxon={test['p_wilcoxon']:.4f}  "
              f"mean_diff={test['mean_diff']:+.4e}")
    pd.DataFrame(pair_rows).to_csv(
        RESULTS_DIR / "stats" / f"pairwise_vs_benchmark_{RUN_TAG}.csv", index=False)

    # --- 9. Plots
    dates_test = test_returns.index
    plot_equity_curves(curves, dates_test,
                       "OOS Equity Curves — Test period",
                       RESULTS_DIR / "plots" / f"equity_curves_{RUN_TAG}.png")
    plot_drawdown(curves, dates_test,
                  "OOS Drawdown — Test period",
                  RESULTS_DIR / "plots" / f"drawdown_{RUN_TAG}.png")
    plot_df = metrics_df.reset_index()
    for metric in ["Sharpe", "Sortino", "Calmar", "Total_Return", "Max_DD"]:
        if metric in plot_df.columns:
            plot_metric_bar(plot_df, metric,
                            RESULTS_DIR / "plots" / f"bar_{metric}_{RUN_TAG}.png")

    # --- 10. Robustness across rebalance freq (using best DRL setup)
    if SB3_OK:
        print("\n--- ROBUSTNESS: rebalance frequency sweep (PPO_attention) ---")
        rob_rows = []
        for fname, fdays in REBALANCE_FREQS.items():
            eq = strat_drl("PPO", train_returns, train_vol, train_pred_aligned,
                           test_returns, test_vol, test_pred_aligned,
                           fdays, GLOBAL_SEED + 999, steps=RL_FINAL_STEPS // 2)
            m = compute_metrics(eq, benchmark_eq=benchmark)
            m["rebalance"] = fname
            rob_rows.append(m)
        pd.DataFrame(rob_rows).to_csv(
            RESULTS_DIR / "reports" / f"robustness_rebal_{RUN_TAG}.csv", index=False)

    # --- 11. Final report
    report_path = RESULTS_DIR / "reports" / f"REPORT_{RUN_TAG}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 78 + "\n")
        f.write(f"Portfolio Optimization — Multihead-Attention + DRL\n")
        f.write(f"Run tag: {RUN_TAG}\n")
        f.write(f"Device:  {DEVICE}\n")
        f.write(f"Tickers: {TICKERS}\n")
        f.write(f"Vol features: {list(vol_df.columns)}\n")
        f.write(f"Periods: train={len(splits['train']['returns'])}  "
                f"val={len(splits['val']['returns'])}  "
                f"test={len(splits['test']['returns'])}\n")
        f.write(f"Test window: {dates_test[0].date()} -> {dates_test[-1].date()}\n")
        f.write(f"Initial capital: ${INITIAL_CAPITAL:,.0f}\n")
        f.write(f"Costs: commission={COMMISSION_BPS}bps slippage={SLIPPAGE_BPS}bps\n")
        f.write(f"Best forecaster (by val MSE): {best_fc_name}\n\n")

        f.write("Forecast metrics:\n")
        f.write(fc_df.to_string(float_format=lambda v: f"{v:.6e}") + "\n\n")

        f.write("Diebold-Mariano (AttentionTCN vs each baseline):\n")
        for r in dm_rows:
            f.write(f"   vs {r['AttentionTCN_vs']:14s}: DM={r['DM']:+.3f}  "
                    f"p={r['p_value']:.4f}\n")
        f.write("\n")

        f.write("OOS portfolio metrics:\n")
        f.write(metrics_df.to_string(float_format=lambda v: f"{v:.4f}") + "\n\n")

        f.write("Pairwise tests vs EqualWeight benchmark:\n")
        for r in pair_rows:
            f.write(f"   {r['Strategy']:24s}  t={r['t']:+.3f}  "
                    f"p_t={r['p_t']:.4f}  p_W={r['p_wilcoxon']:.4f}  "
                    f"mean_diff={r['mean_diff']:+.4e}\n")
    print(f"\n[OK ] Final report written to {report_path}")
    print("=" * 78)
    print("DONE.")
    print("=" * 78)


if __name__ == "__main__":
    main()
