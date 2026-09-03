# ==============================================================================
# PORTFOLIO VIX ENGINE
# Indice de volatilidad implicita "model-free" (metodologia CBOE extendida a
# portafolios de N acciones / ETFs) sobre cadenas de opciones de Polygon.io.
# ==============================================================================

from __future__ import annotations

import argparse
import math
import os
import time
import warnings
import webbrowser
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _HAS_PLOTLY = True
except Exception:  # pragma: no cover
    _HAS_PLOTLY = False

try:
    import yfinance as yf

    _HAS_YF = True
except Exception:
    _HAS_YF = False

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


# ==============================================================================
# BLOQUE 1: CONFIGURACION  <<<  EDITA AQUI  >>>
# Composicion del portafolio, tasa libre de riesgo, rutas de salida y
# parametros de descarga de Polygon. Es el unico bloque pensado para tocar.
# ==============================================================================
PORTFOLIO_HOLDINGS: Dict[str, Optional[float]] = {
    "XLU": 0.1200,
    "GLD": 0.1200,
    "T": 0.1031,
    "GILD": 0.0925,
    "FXI": 0.0779,
    "MRK": 0.0734,
    "ADP": 0.0595,
    "KO": 0.0578,
    "ABT": 0.0562,
    "VRTX": 0.0541,
    "AMGN": 0.0509,
    "UNP": 0.0485,
    "NEE": 0.0484,
    "ABBV": 0.0235,
    "TMO": 0.0142,
}

EQUAL_WEIGHTS: bool = False              # True -> ignora los pesos de arriba
RISK_FREE_RATE: float = 0.045            # tasa libre de riesgo continua
USE_ESCROWED_DIVIDENDS: bool = True      # dividendos discretos vs. yield continuo
HISTORY_LOOKBACK: str = "2y"             # ventana histórica para la correlación

# --- salidas ------------------------------------------------------------------
OUTPUT_HTML: str = "portfolio_vix_report.html"   # informe interactivo (Plotly)
SHOW_PLOT: bool = True                   # abre el informe interactivo al terminar

# --- Polygon.io ---------------------------------------------------------------
POLYGON_API_KEY: Optional[str] = os.environ.get("POLYGON_API_KEY")
POLYGON_BASE_URL: str = "https://api.polygon.io"

if not POLYGON_API_KEY:
    print("ADVERTENCIA: No hay POLYGON_API_KEY configurada. Las cadenas de opciones no")
    print("             se podran descargar y el motor caera a yfinance / datos sinteticos.")

OPT_MIN_DTE: float = 7.0                 # días mínimos a vencimiento admitidos
OPT_MAX_DTE: float = 90.0                # horizonte máximo al buscar vencimientos
OPT_STRIKE_RANGE_PCT: float = 0.40       # ventana de strikes pedida a la API
OPT_MAX_MONEYNESS_SD: float = 4.0        # recorte final del strip, en sigma_atm*sqrt(T)
IV_SANITY_MIN: float = 0.02              # descarta IVs publicadas fuera de rango
IV_SANITY_MAX: float = 3.00
QUOTE_SPREAD_LIQUID: float = 0.02        # semi-spread sintético (OI o volumen > 0)
QUOTE_SPREAD_ILLIQUID: float = 0.12      # semi-spread sintético (contrato sin actividad)
QUOTE_MIN_HALF_SPREAD: float = 0.01      # tick mínimo
POLYGON_MAX_PAGES: int = 25              # tope de paginación por petición
STOCKS_THROTTLE_SECONDS: float = 13.0    # respeta el límite de 5/min del tier Stocks


# ==============================================================================
# BLOQUE 2: CONSTANTES DE CALENDARIO Y TIPOS
# Convencion CBOE: el tiempo se mide en minutos sobre base 365. Aqui viven
# tambien los alias de tipos y los helpers que leen PORTFOLIO_HOLDINGS.
# ==============================================================================
MINUTES_PER_YEAR: float = 525_600.0      # N_365
MINUTES_PER_30D: float = 43_200.0        # N_30
SECONDS_PER_YEAR: float = 365.0 * 24 * 3600

VolMethod = Literal["spline", "svi"]
CorrMethod = Literal["ewma", "rmt", "sample"]
AmericanEngine = Literal["bs1993", "crr"]
DataSource = Literal["polygon", "yahoo", "synthetic"]


def portfolio_tickers() -> List[str]:
    return list(PORTFOLIO_HOLDINGS.keys())


def portfolio_weights() -> Optional[List[float]]:
    if EQUAL_WEIGHTS:
        return None
    raw = list(PORTFOLIO_HOLDINGS.values())
    if any(w is None for w in raw):
        return None
    vals = np.asarray([float(w) for w in raw], dtype=float)
    total = float(vals.sum())
    if total <= 0:
        return None
    return list(vals / total)


# ==============================================================================
# BLOQUE 3: NUCLEO DE VALORACION
# Black-Scholes-Merton y Black-76 para europeas; Bjerksund-Stensland 1993 y
# arbol CRR para americanas; inversion a volatilidad implicita por brentq.
# ==============================================================================
def bs_price(
    S: float, K: float, T: float, r: float, q: float, sigma: float, kind: str
) -> float:
    if T <= 0 or sigma <= 0:
        intrinsic = (S - K) if kind == "call" else (K - S)
        return max(intrinsic, 0.0)
    sT = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sT
    d2 = d1 - sT
    if kind == "call":
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)


def black76_price(F: float, K: float, T: float, r: float, sigma: float, kind: str) -> float:
    if T <= 0 or sigma <= 0:
        intrinsic = (F - K) if kind == "call" else (K - F)
        return math.exp(-r * T) * max(intrinsic, 0.0)
    sT = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma**2 * T) / sT
    d2 = d1 - sT
    df = math.exp(-r * T)
    if kind == "call":
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def _euro_call_carry(S: float, K: float, T: float, r: float, b: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    sT = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (b + 0.5 * sigma**2) * T) / sT
    d2 = d1 - sT
    return S * math.exp((b - r) * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def _phi(
    S: float, T: float, gamma: float, H: float, I: float, r: float, b: float, sigma: float
) -> float:
    v2 = sigma**2
    sT = sigma * math.sqrt(T)
    lam = (-r + gamma * b + 0.5 * gamma * (gamma - 1.0) * v2) * T
    kappa = 2.0 * b / v2 + (2.0 * gamma - 1.0)
    d1 = -(math.log(S / H) + (b + (gamma - 0.5) * v2) * T) / sT
    d2 = d1 - 2.0 * math.log(I / S) / sT
    return math.exp(lam) * (S**gamma) * (norm.cdf(d1) - ((I / S) ** kappa) * norm.cdf(d2))


def bjerksund_stensland_call(
    S: float, K: float, T: float, r: float, b: float, sigma: float
) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    if b >= r:  # sin dividendos el ejercicio anticipado de la call nunca es óptimo
        return _euro_call_carry(S, K, T, r, b, sigma)

    v2 = sigma**2
    beta = (0.5 - b / v2) + math.sqrt((b / v2 - 0.5) ** 2 + 2.0 * r / v2)
    if abs(beta - 1.0) < 1e-10:
        return _euro_call_carry(S, K, T, r, b, sigma)

    b_inf = beta / (beta - 1.0) * K
    b_zero = max(K, r / (r - b) * K)
    if b_inf <= b_zero:
        return _euro_call_carry(S, K, T, r, b, sigma)
    h = -(b * T + 2.0 * sigma * math.sqrt(T)) * (b_zero / (b_inf - b_zero))
    trigger = b_zero + (b_inf - b_zero) * (1.0 - math.exp(h))

    if S >= trigger:
        return S - K

    alpha = (trigger - K) * trigger ** (-beta)
    val = (
        alpha * S**beta
        - alpha * _phi(S, T, beta, trigger, trigger, r, b, sigma)
        + _phi(S, T, 1.0, trigger, trigger, r, b, sigma)
        - _phi(S, T, 1.0, K, trigger, r, b, sigma)
        - K * _phi(S, T, 0.0, trigger, trigger, r, b, sigma)
        + K * _phi(S, T, 0.0, K, trigger, r, b, sigma)
    )
    return max(val, _euro_call_carry(S, K, T, r, b, sigma))


def bjerksund_stensland_price(
    S: float, K: float, T: float, r: float, q: float, sigma: float, kind: str
) -> float:
    b = r - q
    if kind == "call":
        return bjerksund_stensland_call(S, K, T, r, b, sigma)
    return bjerksund_stensland_call(K, S, T, r - b, -b, sigma)


def crr_american(
    S: float, K: float, T: float, r: float, q: float, sigma: float, kind: str, steps: int = 201
) -> float:
    if T <= 0 or sigma <= 0:
        return max((S - K) if kind == "call" else (K - S), 0.0)
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    p = min(max(p, 0.0), 1.0)

    j = np.arange(steps + 1)
    prices = S * (u ** (steps - j)) * (d**j)
    values = np.maximum(prices - K, 0.0) if kind == "call" else np.maximum(K - prices, 0.0)
    for i in range(steps - 1, -1, -1):
        jj = np.arange(i + 1)
        prices = S * (u ** (i - jj)) * (d**jj)
        values = disc * (p * values[: i + 1] + (1.0 - p) * values[1 : i + 2])
        exercise = np.maximum(prices - K, 0.0) if kind == "call" else np.maximum(K - prices, 0.0)
        values = np.maximum(values, exercise)
    return float(values[0])


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    kind: str,
    style: Literal["american", "european"] = "american",
    engine: AmericanEngine = "bs1993",
    lo: float = 1e-3,
    hi: float = 5.0,
) -> float:
    if not np.isfinite(price) or price <= 0.0 or T <= 0.0:
        return float("nan")

    if style == "european":
        def f(s: float) -> float:
            return bs_price(S, K, T, r, q, s, kind) - price
    elif engine == "crr":
        def f(s: float) -> float:
            return crr_american(S, K, T, r, q, s, kind) - price
    else:
        def f(s: float) -> float:
            return bjerksund_stensland_price(S, K, T, r, q, s, kind) - price

    try:
        f_lo, f_hi = f(lo), f(hi)
    except (ValueError, OverflowError, ZeroDivisionError):
        return float("nan")
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0:
        return float("nan")
    try:
        return float(brentq(f, lo, hi, xtol=1e-8, rtol=1e-10, maxiter=200))
    except (ValueError, RuntimeError):
        return float("nan")


# ==============================================================================
# BLOQUE 4: ESTRUCTURAS DE DATOS
# Contenedores que viajan entre modulos: corte por vencimiento, datos de
# mercado de un activo (con su calendario de dividendos) y ajuste de sonrisa.
# ==============================================================================
@dataclass
class ExpirySlice:
    T: float
    chain: pd.DataFrame
    label: str = ""


@dataclass
class AssetMarketData:
    ticker: str
    spot: float
    q: float
    slices: List[ExpirySlice]
    history: Optional[pd.Series] = None
    dividends: List[Tuple[pd.Timestamp, float]] = field(default_factory=list)


@dataclass
class SmileFit:
    T: float
    forward: float
    K0: float
    strikes_dense: np.ndarray
    iv_dense: np.ndarray
    strikes_obs: np.ndarray
    iv_obs: np.ndarray
    otm_prices: np.ndarray
    dk: np.ndarray
    variance: float
    method: str
    label: str = ""


# ==============================================================================
# BLOQUE 5: LIMPIEZA DE CADENAS Y DE-AMERICANIZACION
# Filtros de no-arbitraje estatico (cotizaciones, spread, cotas, monotonia y
# convexidad) y extraccion de la volatilidad europea equivalente: se invierte
# el precio con modelo americano y se repricia como europea, lo que elimina
# la prima de ejercicio anticipado. Resuelve el sesgo de estilo americano.
# ==============================================================================
class OptionChainCleaner:
    def __init__(
        self, max_rel_spread: float = 0.85, min_points_per_side: int = 4, verbose: bool = True
    ) -> None:
        self.max_rel_spread = max_rel_spread
        self.min_points_per_side = min_points_per_side
        self.verbose = verbose

    # -- utilidades ------------------------------------------------------------
    @staticmethod
    def _monotonic_mask(strikes: np.ndarray, prices: np.ndarray, kind: str) -> np.ndarray:
        keep = np.ones(len(strikes), dtype=bool)
        if kind == "call":  # C(K) decreciente en K
            best = np.inf
            for i in range(len(strikes)):
                if prices[i] <= best + 1e-12:
                    best = prices[i]
                else:
                    keep[i] = False
        else:  # P(K) creciente en K
            best = -np.inf
            for i in range(len(strikes)):
                if prices[i] >= best - 1e-12:
                    best = prices[i]
                else:
                    keep[i] = False
        return keep

    @staticmethod
    def _convexity_mask(strikes: np.ndarray, prices: np.ndarray) -> np.ndarray:
        keep = np.ones(len(strikes), dtype=bool)
        changed = True
        while changed and keep.sum() >= 3:
            changed = False
            idx = np.where(keep)[0]
            k, p = strikes[idx], prices[idx]
            for j in range(1, len(idx) - 1):
                k1, k2, k3 = k[j - 1], k[j], k[j + 1]
                bound = ((k3 - k2) * p[j - 1] + (k2 - k1) * p[j + 1]) / (k3 - k1)
                if p[j] > bound + 1e-8:
                    keep[idx[j]] = False
                    changed = True
                    break
        return keep

    # -- API pública -----------------------------------------------------------
    def clean(
        self, chain: pd.DataFrame, S: float, T: float, r: float, q: float, tag: str = ""
    ) -> pd.DataFrame:
        df = chain.copy()
        n0 = len(df)
        for col in ("strike", "bid", "ask"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["strike", "bid", "ask", "type"])

        # (1) cotizaciones inválidas
        df = df[(df["bid"] > 0.0) & (df["ask"] > df["bid"]) & (df["strike"] > 0.0)].copy()
        df["mid"] = 0.5 * (df["bid"] + df["ask"])
        df = df[df["mid"] > 0.0]
        if df.empty:
            return df

        # (2) spread relativo
        rel = (df["ask"] - df["bid"]) / df["mid"]
        df = df[rel <= self.max_rel_spread]
        if df.empty:
            return df

        # (3) límites de no-arbitraje estático
        disc_r, disc_q = math.exp(-r * T), math.exp(-q * T)
        is_call = df["type"].eq("call").values
        lower = np.where(
            is_call,
            np.maximum(S * disc_q - df["strike"].values * disc_r, 0.0),
            np.maximum(df["strike"].values * disc_r - S * disc_q, 0.0),
        )
        upper = np.where(is_call, S * disc_q, df["strike"].values * disc_r)
        df = df[(df["mid"].values >= lower - 1e-6) & (df["mid"].values <= upper + 1e-6)]

        # (4) y (5) por lado
        out = []
        for kind, g in df.groupby("type"):
            g = g.sort_values("strike")
            if len(g) >= 2:
                g = g[self._monotonic_mask(g["strike"].values, g["mid"].values, str(kind))]
            if len(g) >= 3:
                g = g[self._convexity_mask(g["strike"].values, g["mid"].values)]
            if len(g) < self.min_points_per_side:
                warnings.warn(
                    f"[{tag}] Sólo {len(g)} {kind}s sobreviven a los filtros de no-arbitraje; "
                    "la sonrisa puede ser poco fiable.",
                    RuntimeWarning,
                )
            out.append(g)

        res = pd.concat(out, ignore_index=True) if out else df.iloc[0:0]
        if self.verbose:
            print(f"    [clean]   {tag}: {n0} -> {len(res)} cotizaciones válidas")
        return res.sort_values(["type", "strike"]).reset_index(drop=True)


class DeAmericanizer:
    def __init__(self, engine: AmericanEngine = "bs1993", verbose: bool = True) -> None:
        self.engine = engine
        self.verbose = verbose

    def transform(
        self, chain: pd.DataFrame, S: float, T: float, r: float, q: float, tag: str = ""
    ) -> pd.DataFrame:
        ivs, prices_eu, premia = [], [], []
        for _, row in chain.iterrows():
            k, kind, mid = float(row["strike"]), str(row["type"]), float(row["mid"])
            iv = implied_vol(mid, S, k, T, r, q, kind, "american", self.engine)
            if not np.isfinite(iv):
                iv = implied_vol(mid, S, k, T, r, q, kind, "european")
            p_eu = bs_price(S, k, T, r, q, iv, kind) if np.isfinite(iv) else float("nan")
            ivs.append(iv)
            prices_eu.append(p_eu)
            premia.append(mid - p_eu)

        out = chain.copy()
        out["iv"] = ivs
        out["price_eu"] = prices_eu
        out["early_ex_premium"] = premia
        n_bad = int(out["iv"].isna().sum())
        out = out.dropna(subset=["iv", "price_eu"])
        if self.verbose:
            avg = float(np.nanmean(out["early_ex_premium"])) if len(out) else 0.0
            print(
                f"    [de-amer] {tag}: {n_bad} sin IV | prima media de "
                f"ejercicio anticipado eliminada = {avg:.4f}"
            )
        return out.reset_index(drop=True)


# ==============================================================================
# BLOQUE 6: CONSTRUCCION DE LA SONRISA
# Ajuste SVI raw (con test de mariposa de Gatheral y guardia de R2 contra
# ajustes planos) o spline cubico sobre varianza total, y generacion de una
# grilla densa de strikes. Resuelve los agujeros de las cadenas iliquidas:
# la integral corre sobre la grilla densa, no sobre los strikes observados.
# ==============================================================================
class SmileBuilder:
    def __init__(
        self,
        method: VolMethod = "svi",
        n_grid: int = 401,
        wing_ext_sd: float = 1.0,
        max_sd: float = 5.0,
        min_svi_r2: float = 0.60,
        verbose: bool = True,
    ) -> None:
        self.method = method
        self.n_grid = n_grid
        self.wing_ext_sd = wing_ext_sd
        self.max_sd = max_sd
        self.min_svi_r2 = min_svi_r2
        self.verbose = verbose

    # -- SVI -------------------------------------------------------------------
    @staticmethod
    def _svi_w(params: np.ndarray, k: np.ndarray) -> np.ndarray:
        a, b, rho, m, s = params
        return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s**2))

    def _fit_svi(self, k: np.ndarray, w: np.ndarray) -> Optional[np.ndarray]:
        if len(k) < 5:
            return None
        w_atm = float(np.interp(0.0, k, w))
        lb = np.array([1e-8, 1e-6, -0.999, -1.5, 1e-3])
        ub = np.array([max(float(w.max()) * 2.0, 1e-4), 5.0, 0.999, 1.5, 2.0])
        x0 = np.clip(np.array([max(w_atm * 0.5, 1e-6), 0.1, -0.5, 0.0, 0.1]), lb, ub)
        try:
            sol = least_squares(
                lambda p: self._svi_w(p, k) - w,
                x0=x0,
                bounds=(lb, ub),
                loss="soft_l1",
                f_scale=max(1e-5, 0.05 * float(np.mean(w))),
                max_nfev=5000,
            )
        except Exception:  # noqa: BLE001
            return None
        return np.asarray(sol.x, dtype=float)

    @staticmethod
    def _butterfly_g(params: np.ndarray, k: np.ndarray) -> np.ndarray:
        a, b, rho, m, s = params
        x = k - m
        root = np.sqrt(x**2 + s**2)
        w = np.maximum(a + b * (rho * x + root), 1e-12)
        wp = b * (rho + x / root)
        wpp = b * s**2 / root**3
        return (1.0 - k * wp / (2.0 * w)) ** 2 - (wp**2 / 4.0) * (1.0 / w + 0.25) + wpp / 2.0

    # -- Spline ----------------------------------------------------------------
    @staticmethod
    def _spline_fn(k: np.ndarray, w: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        if len(k) >= 4:
            cs = CubicSpline(k, w, bc_type="natural", extrapolate=False)
            d_lo, d_hi = float(cs(k[0], 1)), float(cs(k[-1], 1))
        else:
            cs, d_lo, d_hi = None, 0.0, 0.0

        def fn(kk: np.ndarray) -> np.ndarray:
            kk = np.atleast_1d(np.asarray(kk, dtype=float))
            if cs is None:
                out = np.interp(kk, k, w)
            else:
                out = np.asarray(cs(kk), dtype=float)
                left, right = kk < k[0], kk > k[-1]
                out[left] = w[0] + d_lo * (kk[left] - k[0])
                out[right] = w[-1] + d_hi * (kk[right] - k[-1])
            return np.maximum(out, 1e-8)

        return fn

    # -- API -------------------------------------------------------------------
    def build(
        self, strikes: np.ndarray, ivs: np.ndarray, F: float, T: float, tag: str = ""
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        order = np.argsort(strikes)
        k_raw = np.log(np.asarray(strikes, dtype=float)[order] / F)
        iv_raw = np.asarray(ivs, dtype=float)[order]
        ok = np.isfinite(k_raw) & np.isfinite(iv_raw) & (iv_raw > 0)
        k_raw, iv_raw = k_raw[ok], iv_raw[ok]
        k, inv = np.unique(k_raw, return_inverse=True)          # colapsa duplicados
        iv = np.bincount(inv, weights=iv_raw) / np.bincount(inv)
        if len(k) < 3:
            raise ValueError("menos de 3 puntos válidos para construir la sonrisa")

        w = iv**2 * T
        sd = float(np.interp(0.0, k, iv)) * math.sqrt(T)  # sigma_atm * sqrt(T)
        k_lo = max(k.min() - self.wing_ext_sd * sd, -self.max_sd * sd)
        k_hi = min(k.max() + self.wing_ext_sd * sd, self.max_sd * sd)
        k_lo, k_hi = min(k_lo, k.min()), max(k_hi, k.max())
        k_dense = np.linspace(k_lo, k_hi, self.n_grid)

        used = self.method
        w_dense: Optional[np.ndarray] = None
        if self.method == "svi":
            params = self._fit_svi(k, w)
            if params is None:
                warnings.warn(f"[{tag}] Ajuste SVI fallido; se usa cubic spline.", RuntimeWarning)
                used = "spline (fallback de SVI)"
            else:
                g_min = float(np.nanmin(self._butterfly_g(params, k_dense)))
                if g_min < -1e-4:
                    warnings.warn(
                        f"[{tag}] SVI con arbitraje mariposa (min g = {g_min:.4f}); "
                        "se usa cubic spline.",
                        RuntimeWarning,
                    )
                    used = "spline (fallback de SVI)"
                else:
                    resid = self._svi_w(params, k) - w
                    ss_res = float(resid @ resid)
                    ss_tot = float(np.sum((w - w.mean()) ** 2))
                    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else 1.0
                    if r2 < self.min_svi_r2:
                        # la pérdida robusta ha tratado toda la sonrisa como
                        # atípica y ha devuelto una recta: se descarta el ajuste
                        warnings.warn(
                            f"[{tag}] Ajuste SVI degenerado (R² = {r2:.2f}); "
                            "se usa cubic spline.",
                            RuntimeWarning,
                        )
                        used = "spline (fallback de SVI)"
                    else:
                        w_dense = self._svi_w(params, k_dense)

        if w_dense is None:
            w_dense = self._spline_fn(k, w)(k_dense)

        iv_dense = np.sqrt(np.maximum(w_dense, 1e-10) / T)
        strikes_dense = F * np.exp(k_dense)
        if self.verbose:
            print(
                f"    [smile]   {tag}: {len(k)} puntos -> grilla de {len(k_dense)} strikes "
                f"[{strikes_dense[0]:.2f}, {strikes_dense[-1]:.2f}] ({used})"
            )
        return strikes_dense, iv_dense, used


# ==============================================================================
# BLOQUE 7: MOTOR CBOE MODEL-FREE VARIANCE
# Forward implicito por paridad put-call (no depende del dividend yield),
# seleccion OTM alrededor de K0, integracion discreta con pesos dK/K^2 e
# interpolacion temporal en minutos hacia el horizonte de 30 dias.
# ==============================================================================
class CBOEVarianceEngine:
    def __init__(self, r: float, verbose: bool = True) -> None:
        self.r = r
        self.verbose = verbose

    # -- forward ---------------------------------------------------------------
    def implied_forward(self, chain: pd.DataFrame, T: float) -> Tuple[float, float]:
        piv = chain.pivot_table(
            index="strike", columns="type", values="price_eu", aggfunc="mean"
        ).dropna()
        if piv.empty or not {"call", "put"}.issubset(set(piv.columns)):
            raise ValueError("no hay strikes con call y put simultáneas para extraer el forward")
        k_star = float((piv["call"] - piv["put"]).abs().idxmin())
        f = k_star + math.exp(self.r * T) * float(
            piv.loc[k_star, "call"] - piv.loc[k_star, "put"]
        )
        if not np.isfinite(f) or f <= 0:
            raise ValueError(f"forward implícito no válido ({f})")
        return f, k_star

    # -- integración -----------------------------------------------------------
    @staticmethod
    def _delta_k(strikes: np.ndarray) -> np.ndarray:
        dk = np.empty_like(strikes, dtype=float)
        dk[1:-1] = (strikes[2:] - strikes[:-2]) / 2.0
        dk[0] = strikes[1] - strikes[0]
        dk[-1] = strikes[-1] - strikes[-2]
        return dk

    def variance(
        self, strikes: np.ndarray, iv: np.ndarray, F: float, T: float
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        below = strikes[strikes <= F]
        if below.size == 0:
            raise ValueError("ningún strike por debajo del forward: grilla insuficiente")
        k0 = float(below.max())

        prices = np.empty_like(strikes, dtype=float)
        for i, (k, v) in enumerate(zip(strikes, iv)):
            if k < k0:
                prices[i] = black76_price(F, float(k), T, self.r, float(v), "put")
            elif k > k0:
                prices[i] = black76_price(F, float(k), T, self.r, float(v), "call")
            else:
                prices[i] = 0.5 * (
                    black76_price(F, float(k), T, self.r, float(v), "call")
                    + black76_price(F, float(k), T, self.r, float(v), "put")
                )

        dk = self._delta_k(strikes)
        contrib = (dk / strikes**2) * math.exp(self.r * T) * prices
        sigma2 = (2.0 / T) * float(contrib.sum()) - (1.0 / T) * (F / k0 - 1.0) ** 2
        return float(sigma2), k0, prices, dk

    # -- interpolación temporal ------------------------------------------------
    @staticmethod
    def interpolate_30d(
        T1: float, var1: float, T2: Optional[float], var2: Optional[float]
    ) -> float:
        if T2 is None or var2 is None:
            warnings.warn(
                "Sólo hay un vencimiento disponible: se usa su varianza sin interpolar "
                "al horizonte de 30 días (cobertura mínima no satisfecha).",
                RuntimeWarning,
            )
            return float(var1)

        n1, n2 = T1 * MINUTES_PER_YEAR, T2 * MINUTES_PER_YEAR
        if abs(n2 - n1) < 1e-9:
            return 0.5 * (var1 + var2)
        if not (n1 <= MINUTES_PER_30D <= n2):
            warnings.warn(
                f"Los vencimientos ({T1*365:.1f}d, {T2*365:.1f}d) no encierran los 30 días: "
                "se extrapola linealmente en varianza total (menor fiabilidad).",
                RuntimeWarning,
            )
        term = (
            T1 * var1 * (n2 - MINUTES_PER_30D) / (n2 - n1)
            + T2 * var2 * (MINUTES_PER_30D - n1) / (n2 - n1)
        )
        return float(term * MINUTES_PER_YEAR / MINUTES_PER_30D)


class SingleAssetVIX:
    def __init__(
        self,
        r: float,
        cleaner: OptionChainCleaner,
        deamer: DeAmericanizer,
        smiler: SmileBuilder,
        verbose: bool = True,
    ) -> None:
        self.r = r
        self.cleaner = cleaner
        self.deamer = deamer
        self.smiler = smiler
        self.engine = CBOEVarianceEngine(r, verbose=verbose)
        self.verbose = verbose

    def compute(self, data: AssetMarketData) -> Tuple[float, List[SmileFit]]:
        if self.verbose:
            div_txt = (
                f"{len(data.dividends)} dividendo(s) discreto(s)"
                if data.dividends
                else f"q = {data.q:.4f}"
            )
            print(f"\n  >> {data.ticker}  (S = {data.spot:.2f}, {div_txt})")
        if len(data.slices) < 2:
            warnings.warn(
                f"[{data.ticker}] Cobertura mínima de vencimientos no satisfecha "
                f"({len(data.slices)} disponible/s; se requieren 2 para interpolar a 30d).",
                RuntimeWarning,
            )

        fits: List[SmileFit] = []
        for sl in data.slices:
            tag = f"{data.ticker} {sl.label or f'T={sl.T*365:.0f}d'}"
            try:
                # Dividendos discretos: se descuenta del spot el valor presente de
                # los pagos anteriores al vencimiento (escrowed) en vez de usar un
                # yield continuo. El forward sigue saliendo de la paridad put-call.
                s_eff, q_eff = escrowed_inputs(
                    data.spot, data.q, data.dividends, sl.T, self.r
                )
                if self.verbose and s_eff != data.spot:
                    print(
                        f"    [div]     {tag}: S escrowed = {s_eff:.2f} "
                        f"(PV dividendos = {data.spot - s_eff:.4f})"
                    )
                clean = self.cleaner.clean(sl.chain, s_eff, sl.T, self.r, q_eff, tag)
                if clean.empty:
                    raise ValueError("cadena vacía tras la limpieza")
                euro = self.deamer.transform(clean, s_eff, sl.T, self.r, q_eff, tag)
                fwd, _ = self.engine.implied_forward(euro, sl.T)

                # sólo la rama OTM observada calibra la sonrisa (evita mezclar ITM ruidosas)
                otm = euro[
                    ((euro["type"] == "put") & (euro["strike"] <= fwd))
                    | ((euro["type"] == "call") & (euro["strike"] >= fwd))
                ]
                if len(otm) < 4:
                    otm = euro
                k_d, iv_d, method = self.smiler.build(
                    otm["strike"].values, otm["iv"].values, fwd, sl.T, tag
                )
                var, k0, prices, dk = self.engine.variance(k_d, iv_d, fwd, sl.T)
                if not np.isfinite(var) or var <= 0:
                    raise ValueError(f"varianza no válida ({var})")

                fits.append(
                    SmileFit(
                        T=sl.T,
                        forward=fwd,
                        K0=k0,
                        strikes_dense=k_d,
                        iv_dense=iv_d,
                        strikes_obs=otm["strike"].values,
                        iv_obs=otm["iv"].values,
                        otm_prices=prices,
                        dk=dk,
                        variance=var,
                        method=method,
                        label=sl.label,
                    )
                )
                if self.verbose:
                    print(
                        f"    [cboe]    {tag}: F = {fwd:.2f} | K0 = {k0:.2f} | "
                        f"sigma_T = {100*math.sqrt(var):.2f}"
                    )
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"[{tag}] Vencimiento descartado: {exc}", RuntimeWarning)

        if not fits:
            raise ValueError("ningún vencimiento utilizable")

        fits.sort(key=lambda f: f.T)
        near, nxt = fits[0], (fits[1] if len(fits) > 1 else None)
        var30 = self.engine.interpolate_30d(
            near.T, near.variance, nxt.T if nxt else None, nxt.variance if nxt else None
        )
        if self.verbose:
            print(f"    [30d]     VIX_{data.ticker} = {100*math.sqrt(max(var30, 0)):.2f}")
        return max(var30, 1e-12), fits


# ==============================================================================
# BLOQUE 8: AGREGACION DEL PORTAFOLIO
# Matriz de correlacion (EWMA, muestral o filtrada por Random Matrix Theory),
# covarianza implicita y atribucion de riesgo por descomposicion de Euler.
# ==============================================================================
class ImpliedCorrelationEstimator:
    def __init__(self, lam: float = 0.94, eig_floor: float = 1e-8, verbose: bool = True) -> None:
        self.lam = lam
        self.eig_floor = eig_floor
        self.verbose = verbose

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
        d = np.sqrt(np.clip(np.diag(cov), 1e-16, None))
        corr = cov / np.outer(d, d)
        np.fill_diagonal(corr, 1.0)
        return np.clip(corr, -1.0, 1.0)

    def _nearest_pd_corr(self, corr: np.ndarray) -> np.ndarray:
        corr = 0.5 * (corr + corr.T)
        vals, vecs = np.linalg.eigh(corr)
        vals = np.clip(vals, self.eig_floor, None)
        return self._cov_to_corr(vecs @ np.diag(vals) @ vecs.T)

    def _ewma_cov(self, rets: np.ndarray) -> np.ndarray:
        t = rets.shape[0]
        w = (1.0 - self.lam) * self.lam ** np.arange(t - 1, -1, -1)
        w /= w.sum()
        x = rets - np.average(rets, axis=0, weights=w)
        return (x * w[:, None]).T @ x

    def _rmt_filter(self, corr: np.ndarray, t: int) -> np.ndarray:
        n = corr.shape[0]
        if t <= n + 1:
            warnings.warn(
                f"T = {t} <= N + 1 = {n+1}: la matriz muestral es singular y el filtrado "
                "RMT es poco informativo. Se aplica sólo proyección definida positiva.",
                RuntimeWarning,
            )
            return self._nearest_pd_corr(corr)
        vals, vecs = np.linalg.eigh(corr)
        qq = n / t
        lam_plus = (1.0 + math.sqrt(qq)) ** 2
        noise = vals < lam_plus
        noise[-1] = False  # el modo de mercado siempre se conserva
        if noise.sum() > 0:
            trace_noise = vals[noise].sum()
            vals = vals.copy()
            vals[noise] = trace_noise / noise.sum()
        if self.verbose:
            print(
                f"    [rmt] lambda+ = {lam_plus:.3f} | autovalores filtrados: "
                f"{int(noise.sum())}/{n}"
            )
        return self._nearest_pd_corr(vecs @ np.diag(vals) @ vecs.T)

    # -- API -------------------------------------------------------------------
    def estimate(self, prices: pd.DataFrame, method: CorrMethod = "ewma") -> np.ndarray:
        n = prices.shape[1]
        if n == 1:
            return np.ones((1, 1))
        rets = np.log(prices.astype(float)).diff().dropna().values
        if rets.shape[0] < 20:
            warnings.warn(
                "Menos de 20 retornos disponibles: se asume matriz identidad "
                "(correlación nula). Revise el histórico.",
                RuntimeWarning,
            )
            return np.eye(n)

        if method == "sample":
            corr = self._cov_to_corr(np.cov(rets, rowvar=False))
        elif method == "ewma":
            corr = self._cov_to_corr(self._ewma_cov(rets))
        elif method == "rmt":
            corr = self._rmt_filter(self._cov_to_corr(self._ewma_cov(rets)), rets.shape[0])
        else:
            raise ValueError(f"método de correlación desconocido: {method}")
        return self._nearest_pd_corr(corr)


class PortfolioVIX:
    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

    def aggregate(
        self, tickers: Sequence[str], w: np.ndarray, sig30: np.ndarray, corr: np.ndarray
    ) -> Tuple[float, np.ndarray, pd.DataFrame, Dict[str, float]]:
        w = np.asarray(w, dtype=float)
        if not np.isclose(w.sum(), 1.0):
            warnings.warn(f"Los pesos suman {w.sum():.4f}; se normalizan a 1.", RuntimeWarning)
            w = w / w.sum()

        cov = np.outer(sig30, sig30) * corr
        var_p = float(w @ cov @ w)
        sig_p = math.sqrt(max(var_p, 1e-16))
        vix_p = 100.0 * sig_p

        # Atribución de riesgo por descomposición de Euler
        mcr = (cov @ w) / sig_p          # d sigma_p / d w_i
        ctr = w * mcr                    # suma exactamente sigma_p
        ctr_pct = ctr / sig_p

        weighted_avg_vol = float(np.abs(w) @ sig30)
        div_ratio = weighted_avg_vol / sig_p
        num = var_p - float(np.sum(w**2 * sig30**2))
        den = 2.0 * float(np.sum(np.triu(np.outer(w, w) * np.outer(sig30, sig30), k=1)))
        implied_corr = num / den if abs(den) > 1e-14 else float("nan")

        breakdown = pd.DataFrame(
            {
                "ticker": list(tickers),
                "peso": w,
                "VIX_individual": 100.0 * sig30,
                "sigma_30d": sig30,
                "MCR": mcr,
                "CTR": ctr,
                "CTR_VIX_pts": 100.0 * ctr,
                "CTR_%": 100.0 * ctr_pct,
            }
        ).set_index("ticker")

        metrics = {
            "VIX_portfolio": vix_p,
            "sigma_portfolio_30d": sig_p,
            "VIX_medio_ponderado": 100.0 * weighted_avg_vol,
            "ratio_diversificacion": div_ratio,
            "beneficio_diversificacion_pts": 100.0 * (weighted_avg_vol - sig_p),
            "correlacion_implicita_media": implied_corr,
        }
        return vix_p, cov, breakdown, metrics


# ==============================================================================
# BLOQUE 9: CARGA DE DATOS
# Fuente principal Polygon.io: vencimientos con criterio CBOE, cadenas por
# snapshot y dividendos discretos con modelo escrowed (se descuenta del spot
# el valor presente de los pagos previos al vencimiento en vez de usar un
# yield continuo). Respaldos en cascada: polygon -> yahoo -> sintetico.
# ==============================================================================
def year_fraction(expiry: pd.Timestamp, now: Optional[pd.Timestamp] = None) -> float:
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    exp = pd.Timestamp(expiry)
    exp = exp.tz_localize("UTC") if exp.tzinfo is None else exp.tz_convert("UTC")
    exp = exp.normalize() + pd.Timedelta(hours=20)  # 16:00 ET ~ 20:00 UTC
    return max((exp - now).total_seconds(), 60.0) / SECONDS_PER_YEAR


def select_cboe_expiries(expiries: Sequence[str], min_days: float = OPT_MIN_DTE) -> List[str]:
    now = pd.Timestamp.now(tz="UTC")
    cand = [(e, year_fraction(pd.Timestamp(e), now) * 365.0) for e in sorted(set(expiries))]
    cand = [(e, d) for e, d in cand if d >= min_days]
    if not cand:
        return []
    near = [e for e, d in cand if d <= 30.0]
    nxt = [e for e, d in cand if d > 30.0]
    if near and nxt:
        return [near[-1], nxt[0]]
    return [e for e, _ in cand[:2]]


def clamp_iv(iv: Optional[float]) -> float:
    if iv is None:
        return float("nan")
    try:
        v = float(iv)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(v) or not (IV_SANITY_MIN <= v <= IV_SANITY_MAX):
        return float("nan")
    return v


def escrowed_inputs(
    spot: float,
    q: float,
    dividends: Sequence[Tuple[pd.Timestamp, float]],
    T: float,
    r: float,
    now: Optional[pd.Timestamp] = None,
) -> Tuple[float, float]:
    if not USE_ESCROWED_DIVIDENDS or not dividends:
        return spot, q

    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    pv = 0.0
    for ex, amount in dividends:
        ex_ts = pd.Timestamp(ex)
        ex_ts = ex_ts.tz_localize("UTC") if ex_ts.tzinfo is None else ex_ts.tz_convert("UTC")
        t = (ex_ts.normalize() - now).total_seconds() / SECONDS_PER_YEAR
        if 0.0 < t <= T:
            pv += float(amount) * math.exp(-r * t)

    # ningún pago dentro del vencimiento, o calendario inverosímil: se deja q
    if pv <= 0.0 or pv > 0.25 * spot:
        return spot, q
    return spot - pv, 0.0


class PolygonMarketLoader:
    def __init__(
        self,
        api_key: Optional[str] = None,
        r: float = RISK_FREE_RATE,
        engine: AmericanEngine = "bs1993",
        min_days: float = OPT_MIN_DTE,
        max_days: float = OPT_MAX_DTE,
        strike_range_pct: float = OPT_STRIKE_RANGE_PCT,
        verbose: bool = True,
    ) -> None:
        self.api_key = api_key or POLYGON_API_KEY
        if not self.api_key:
            raise RuntimeError(
                "No se encontró POLYGON_API_KEY (defínela en el archivo .env o en el entorno)"
            )
        self.r = r
        self.engine = engine
        self.min_days = min_days
        self.max_days = max_days
        self.strike_range_pct = strike_range_pct
        self.verbose = verbose
        self.session = requests.Session()
        self._last_stock_call = 0.0

    # -- transporte ------------------------------------------------------------
    def _get(self, url: str, params: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        for attempt in range(4):
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = STOCKS_THROTTLE_SECONDS * (attempt + 1)
                if self.verbose:
                    print(f"    [polygon] límite de peticiones alcanzado; espera {wait:.0f}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"límite de peticiones de Polygon persistente en {url}")

    def _paginate(
        self, url: str, params: Dict[str, object], max_pages: int = POLYGON_MAX_PAGES
    ) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        next_url: Optional[str] = url
        page = 0
        while next_url and page < max_pages:
            payload = self._get(next_url, params if page == 0 else {"apiKey": self.api_key})
            results = payload.get("results") or []
            out.extend(results)
            next_url = payload.get("next_url")  # type: ignore[assignment]
            page += 1
        return out

    def _throttle_stocks(self) -> None:
        elapsed = time.time() - self._last_stock_call
        if self._last_stock_call and elapsed < STOCKS_THROTTLE_SECONDS:
            time.sleep(STOCKS_THROTTLE_SECONDS - elapsed)
        self._last_stock_call = time.time()

    # -- datos del subyacente --------------------------------------------------
    @staticmethod
    def _lookback_days(lookback: str) -> int:
        s = str(lookback).strip().lower()
        try:
            if s.endswith("mo"):
                return int(round(float(s[:-2]) * 30.44))
            if s.endswith("y"):
                return int(round(float(s[:-1]) * 365))
            if s.endswith("d"):
                return int(float(s[:-1]))
            return int(float(s))
        except ValueError:
            return 730

    def _history_polygon(self, ticker: str, lookback: str) -> pd.Series:
        end = date.today()
        start = end - timedelta(days=self._lookback_days(lookback) + 10)
        self._throttle_stocks()
        payload = self._get(
            f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/"
            f"{start.isoformat()}/{end.isoformat()}",
            {"apiKey": self.api_key, "adjusted": "true", "sort": "asc", "limit": 50000},
        )
        rows = payload.get("results") or []
        if not rows:
            raise RuntimeError(f"Polygon no devolvió histórico para {ticker}")
        idx = pd.to_datetime([r["t"] for r in rows], unit="ms").normalize()
        return pd.Series([float(r["c"]) for r in rows], index=idx, name=ticker)

    def _history(self, ticker: str, lookback: str) -> pd.Series:
        if _HAS_YF:
            try:
                hist = yf.Ticker(ticker).history(period=lookback, auto_adjust=True)["Close"]
                hist = hist.dropna()
                if not hist.empty:
                    # fechas normalizadas para que casen con el respaldo de Polygon
                    hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
                    return hist
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"[{ticker}] yfinance falló ({exc}); se usa el histórico de Polygon.",
                    RuntimeWarning,
                )
        return self._history_polygon(ticker, lookback)

    def _dividend_schedule(
        self, ticker: str, spot: float
    ) -> Tuple[float, List[Tuple[pd.Timestamp, float]]]:
        try:
            payload = self._get(
                f"{POLYGON_BASE_URL}/v3/reference/dividends",
                {
                    "apiKey": self.api_key,
                    "ticker": ticker,
                    "limit": 12,
                    "order": "desc",
                    "sort": "ex_dividend_date",
                },
            )
        except Exception:  # noqa: BLE001
            return 0.0, []
        rows = payload.get("results") or []
        if not rows or spot <= 0:
            return 0.0, []

        today = date.today()
        horizon = today + timedelta(days=int(self.max_days))
        cutoff = today - timedelta(days=365)

        paid, past, future = 0.0, [], []
        for row in rows:
            try:
                ex = date.fromisoformat(str(row.get("ex_dividend_date")))
                amount = float(row.get("cash_amount") or 0.0)
            except (TypeError, ValueError):
                continue
            if amount <= 0:
                continue
            if ex > today:
                future.append((ex, amount, float(row.get("frequency") or 0.0)))
            else:
                past.append((ex, amount, float(row.get("frequency") or 0.0)))
                if ex >= cutoff:
                    paid += amount

        if paid <= 0 and past:  # sin pagos en 12 meses: se anualiza el último
            paid = past[0][1] * past[0][2]
        q = max(paid / spot, 0.0)

        schedule = [
            (pd.Timestamp(ex), amount) for ex, amount, _ in future if ex <= horizon
        ]

        # Ningún dividendo declarado dentro del horizonte: se proyecta el
        # siguiente a partir de la cadencia histórica (frequency = pagos/año).
        if not schedule and past:
            last_ex, last_amount, freq = past[0]
            if freq > 0:
                step = timedelta(days=365.0 / freq)
                nxt = last_ex + step
                while nxt <= horizon:
                    if nxt > today:
                        schedule.append((pd.Timestamp(nxt), last_amount))
                    nxt += step

        schedule.sort(key=lambda x: x[0])
        return q, schedule

    # -- cadena de opciones ----------------------------------------------------
    def _expirations(self, ticker: str) -> List[str]:
        today = date.today()
        params = {
            "apiKey": self.api_key,
            "underlying_ticker": ticker,
            "contract_type": "call",
            "expired": "false",
            "expiration_date.gte": (today + timedelta(days=int(self.min_days))).isoformat(),
            "expiration_date.lte": (today + timedelta(days=int(self.max_days))).isoformat(),
            "limit": 1000,
            "sort": "expiration_date",
            "order": "asc",
        }
        contracts = self._paginate(
            f"{POLYGON_BASE_URL}/v3/reference/options/contracts", params, max_pages=3
        )
        return sorted({str(c.get("expiration_date")) for c in contracts if c.get("expiration_date")})

    def _chain(self, ticker: str, expiration: str, spot: float, T: float, q: float) -> pd.DataFrame:
        params = {
            "apiKey": self.api_key,
            "limit": 250,
            "expiration_date": expiration,
            "strike_price.gte": round(spot * (1.0 - self.strike_range_pct), 2),
            "strike_price.lte": round(spot * (1.0 + self.strike_range_pct), 2),
        }
        contracts = self._paginate(
            f"{POLYGON_BASE_URL}/v3/snapshot/options/{ticker}", params
        )

        rows: List[Dict[str, object]] = []
        n_from_close = 0
        for c in contracts:
            details = c.get("details") or {}
            strike, kind = details.get("strike_price"), details.get("contract_type")
            if strike is None or kind not in ("call", "put"):
                continue
            style = "american" if details.get("exercise_style") != "european" else "european"

            day = c.get("day") or {}
            oi = float(c.get("open_interest") or 0.0)
            vol = float(day.get("volume") or 0.0)
            close = day.get("close")

            iv = clamp_iv(c.get("implied_volatility"))
            if np.isfinite(iv):
                if style == "american":
                    price = bjerksund_stensland_price(
                        spot, float(strike), T, self.r, q, iv, str(kind)
                    )
                else:
                    price = bs_price(spot, float(strike), T, self.r, q, iv, str(kind))
            elif close and float(close) > 0.0:
                # sin IV publicada: se usa el último cierre negociado como precio observado
                price = float(close)
                iv = clamp_iv(
                    implied_vol(
                        price, spot, float(strike), T, self.r, q, str(kind), style, self.engine
                    )
                )
                if not np.isfinite(iv):
                    continue
                n_from_close += 1
            else:
                continue

            if not np.isfinite(price) or price <= 0.0:
                continue

            rel = QUOTE_SPREAD_LIQUID if (oi > 0.0 or vol > 0.0) else QUOTE_SPREAD_ILLIQUID
            half = max(QUOTE_MIN_HALF_SPREAD, rel * price)
            rows.append(
                {
                    "strike": float(strike),
                    "type": str(kind),
                    "bid": price - half,
                    "ask": price + half,
                    "iv_polygon": iv,
                    "open_interest": oi,
                    "volume": vol,
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # Recorte de alas: fuera de +-max_sd * sigma_atm * sqrt(T) las IVs publicadas
        # proceden de contratos sin mercado y sólo aportan ruido al strip model-free
        # (es el análogo al truncamiento por bids en cero del White Paper de la CBOE).
        k = np.log(df["strike"].values / spot)
        atm = df.loc[np.abs(k) <= 0.03, "iv_polygon"]
        sigma_atm = float(atm.median()) if len(atm) else float(df["iv_polygon"].median())
        n_full = len(df)
        if np.isfinite(sigma_atm) and sigma_atm > 0:
            band = OPT_MAX_MONEYNESS_SD * sigma_atm * math.sqrt(max(T, 1e-8))
            df = df[np.abs(k) <= band].reset_index(drop=True)

        if self.verbose and len(df):
            print(
                f"    [polygon] {ticker} {expiration}: {len(df)} contratos "
                f"(de {n_full}; {n_from_close} valorados con el cierre del día; "
                f"sigma_atm = {sigma_atm:.2%})"
            )
        return df

    # -- API pública -----------------------------------------------------------
    def load(
        self, tickers: Sequence[str], lookback: str = HISTORY_LOOKBACK
    ) -> Tuple[List[AssetMarketData], pd.DataFrame]:
        assets: List[AssetMarketData] = []
        closes: Dict[str, pd.Series] = {}
        now = pd.Timestamp.now(tz="UTC")

        for tk in tickers:
            try:
                hist = self._history(tk, lookback)
                spot = float(hist.iloc[-1])
                q, dividends = self._dividend_schedule(tk, spot)

                exps = select_cboe_expiries(self._expirations(tk), self.min_days)
                if not exps:
                    raise RuntimeError("sin vencimientos dentro de la ventana configurada")
                if len(exps) < 2:
                    warnings.warn(
                        f"[{tk}] Sólo {len(exps)} vencimiento válido; la interpolación a 30 "
                        "días será una extrapolación.",
                        RuntimeWarning,
                    )

                slices: List[ExpirySlice] = []
                for e in exps:
                    T = year_fraction(pd.Timestamp(e), now)
                    s_eff, q_eff = escrowed_inputs(spot, q, dividends, T, self.r, now)
                    chain = self._chain(tk, e, s_eff, T, q_eff)
                    if chain.empty:
                        warnings.warn(
                            f"[{tk} {e}] Polygon no devolvió contratos utilizables.",
                            RuntimeWarning,
                        )
                        continue
                    slices.append(ExpirySlice(T=T, chain=chain, label=e))

                if not slices:
                    raise RuntimeError("ninguna cadena utilizable")

                closes[tk] = hist
                assets.append(
                    AssetMarketData(
                        ticker=tk, spot=spot, q=q, slices=slices, dividends=dividends
                    )
                )
                if self.verbose:
                    div_txt = (
                        ", ".join(f"{d.date()}: {a:.2f}" for d, a in dividends)
                        if dividends
                        else f"sin dividendos en el horizonte (q = {q:.4f})"
                    )
                    print(
                        f"  [polygon] {tk}: spot = {spot:.2f} | {div_txt} | "
                        f"vencimientos = {[s.label for s in slices]}"
                    )
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"[{tk}] Excluido de la carga de datos: {exc}", RuntimeWarning)

        if not assets:
            raise RuntimeError("Polygon no devolvió datos utilizables para ningún activo")

        prices = pd.DataFrame(closes).dropna()
        return assets, prices


class YahooMarketLoader:
    def __init__(self, min_days: float = 7.0, verbose: bool = True) -> None:
        self.min_days = min_days
        self.verbose = verbose

    def _pick_expiries(self, expiries: Sequence[str]) -> List[str]:
        return select_cboe_expiries(expiries, self.min_days)

    def load(
        self, tickers: Sequence[str], lookback: str = "2y"
    ) -> Tuple[List[AssetMarketData], pd.DataFrame]:
        if not _HAS_YF:
            raise RuntimeError("yfinance no está instalado")
        assets: List[AssetMarketData] = []
        closes: Dict[str, pd.Series] = {}
        now = pd.Timestamp.now(tz="UTC")

        for tk in tickers:
            t = yf.Ticker(tk)
            hist = t.history(period=lookback, auto_adjust=True)["Close"].dropna()
            if hist.empty:
                raise RuntimeError(f"sin histórico para {tk}")
            hist.index = pd.to_datetime(hist.index).tz_localize(None)
            closes[tk] = hist
            spot = float(hist.iloc[-1])

            try:
                dy = t.info.get("dividendYield", 0.0) or 0.0
                q = float(dy) if float(dy) < 1.0 else float(dy) / 100.0
            except Exception:  # noqa: BLE001
                q = 0.0

            exps = self._pick_expiries(list(t.options))
            if len(exps) < 2:
                warnings.warn(
                    f"[{tk}] Menos de 2 vencimientos válidos disponibles en la fuente.",
                    RuntimeWarning,
                )
            slices: List[ExpirySlice] = []
            for e in exps:
                oc = t.option_chain(e)
                calls = oc.calls[["strike", "bid", "ask"]].assign(type="call")
                puts = oc.puts[["strike", "bid", "ask"]].assign(type="put")
                slices.append(
                    ExpirySlice(
                        T=year_fraction(pd.Timestamp(e), now),
                        chain=pd.concat([calls, puts], ignore_index=True),
                        label=e,
                    )
                )
            assets.append(AssetMarketData(ticker=tk, spot=spot, q=q, slices=slices))
            if self.verbose:
                print(f"  [yfinance] {tk}: spot = {spot:.2f}, q = {q:.4f}, vencimientos = {exps}")

        return assets, pd.DataFrame(closes).dropna()


class SyntheticMarketGenerator:
    def __init__(self, r: float = 0.045, seed: int = 7, verbose: bool = True) -> None:
        self.r = r
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose

    def _svi_iv(
        self, k: np.ndarray, T: float, base: float, skew: float, curv: float
    ) -> np.ndarray:
        a = base**2 * T * 0.6
        b = curv * T
        rho, m, s = skew, 0.02, 0.14
        w = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s**2))
        return np.sqrt(np.maximum(w, 1e-8) / T)

    def generate(
        self,
        tickers: Sequence[str],
        spots: Optional[Sequence[float]] = None,
        base_vols: Optional[Sequence[float]] = None,
        divs: Optional[Sequence[float]] = None,
        days: Tuple[float, float] = (23.0, 37.0),
        n_hist: int = 750,
        drop_frac: float = 0.28,
    ) -> Tuple[List[AssetMarketData], pd.DataFrame]:
        n = len(tickers)
        spots = list(spots) if spots is not None else list(self.rng.uniform(80, 400, n))
        base_vols = (
            list(base_vols) if base_vols is not None else list(self.rng.uniform(0.22, 0.45, n))
        )
        divs = list(divs) if divs is not None else list(self.rng.uniform(0.0, 0.025, n))

        # --- histórico con estructura factorial (mercado + idiosincrásico)
        beta = self.rng.uniform(0.7, 1.3, n)
        mkt = self.rng.normal(0.0, 0.011, n_hist)
        idio = self.rng.normal(0, 1, (n_hist, n)) * (np.array(base_vols) / math.sqrt(252)) * 0.7
        rets = mkt[:, None] * beta[None, :] + idio
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_hist)
        cum = np.cumsum(rets, axis=0)
        prices = pd.DataFrame(
            np.array(spots)[None, :] * np.exp(cum - cum[-1][None, :]),
            index=idx,
            columns=list(tickers),
        )

        assets: List[AssetMarketData] = []
        for i, tk in enumerate(tickers):
            s0, vol, q = float(spots[i]), float(base_vols[i]), float(divs[i])
            skew = float(self.rng.uniform(-0.75, -0.35))
            curv = float(self.rng.uniform(0.9, 1.8))
            slices: List[ExpirySlice] = []
            for dte in days:
                T = dte / 365.0
                step = max(round(s0 * 0.025 / 2.5) * 2.5, 1.0)
                lo = s0 * (1.0 - 5.0 * vol * math.sqrt(T))
                hi = s0 * (1.0 + 5.0 * vol * math.sqrt(T))
                strikes = np.arange(math.floor(lo / step) * step, hi + step, step)
                strikes = strikes[strikes > 0]
                fwd = s0 * math.exp((self.r - q) * T)
                iv_true = self._svi_iv(np.log(strikes / fwd), T, vol, skew, curv)

                rows = []
                for k, v in zip(strikes, iv_true):
                    for kind in ("call", "put"):
                        px = bjerksund_stensland_price(s0, float(k), T, self.r, q, float(v), kind)
                        if px < 0.02:  # sin mercado en las alas profundas
                            bid, ask = 0.0, 0.05
                        else:
                            half = max(0.01, 0.012 * px + 0.01)
                            noise = float(self.rng.normal(0.0, 0.15 * half))
                            bid, ask = max(px - half + noise, 0.0), px + half + noise
                        rows.append({"strike": float(k), "type": kind, "bid": bid, "ask": ask})

                chain = pd.DataFrame(rows)
                chain = chain[self.rng.random(len(chain)) > drop_frac].reset_index(drop=True)
                slices.append(ExpirySlice(T=T, chain=chain, label=f"synt+{int(dte)}d"))

            assets.append(
                AssetMarketData(ticker=tk, spot=s0, q=q, slices=slices, history=prices[tk])
            )
            if self.verbose:
                print(
                    f"  [synth] {tk}: S = {s0:.2f} | vol_atm = {vol:.2%} | "
                    f"q = {q:.2%} | skew = {skew:.2f}"
                )
        return assets, prices


# ==============================================================================
# BLOQUE 10: INFORME INTERACTIVO
# Panel Plotly con las sonrisas por activo, la ponderacion dK*Q(K)/K^2, la
# matriz de correlacion y la atribucion de riesgo. Se guarda como HTML.
# ==============================================================================
class ReportPlotter:
    NEAR_COLOR = "#3987e5"
    NEXT_COLOR = "#e6893c"

    @staticmethod
    def build_figure(
        fits_by_asset: Dict[str, List[SmileFit]],
        breakdown: pd.DataFrame,
        metrics: Dict[str, float],
        corr: np.ndarray,
    ) -> "go.Figure":
        tickers = list(fits_by_asset.keys())
        n = len(tickers)
        rows_smiles = max((n + 2) // 3, 1)
        total_rows = rows_smiles + 2

        titles: List[str] = []
        for i in range(rows_smiles * 3):
            titles.append(
                f"{tickers[i]} — sonrisa OTM de-americanizada" if i < n else ""
            )
        # las dos últimas filas usan colspan: sólo tienen dos subplots reales
        titles += ["Ponderación ΔK·Q(K)/K²  (near-term, normalizada)", "Correlación implícita"]
        titles += ["Atribución de riesgo (Volatility Budgeting)", "Reparto del riesgo total"]

        specs: List[List[Optional[Dict[str, object]]]] = [
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}] for _ in range(rows_smiles)
        ]
        specs.append([{"type": "xy", "colspan": 2}, None, {"type": "xy"}])
        specs.append([{"type": "xy", "colspan": 2}, None, {"type": "domain"}])

        fig = make_subplots(
            rows=total_rows,
            cols=3,
            specs=specs,
            subplot_titles=titles,
            vertical_spacing=min(0.09, 0.5 / max(total_rows - 1, 1)),
            horizontal_spacing=0.08,
        )

        # --- sonrisas ---------------------------------------------------------
        for i, tk in enumerate(tickers):
            row, col = i // 3 + 1, i % 3 + 1
            for j, f in enumerate(sorted(fits_by_asset[tk], key=lambda x: x.T)):
                color = ReportPlotter.NEAR_COLOR if j == 0 else ReportPlotter.NEXT_COLOR
                name = "Near-term" if j == 0 else "Next-term"
                label = f.label or f"T={f.T*365:.0f}d"
                fig.add_trace(
                    go.Scatter(
                        x=f.strikes_dense,
                        y=100.0 * f.iv_dense,
                        mode="lines",
                        line=dict(color=color, width=2),
                        name=f"{name} (ajuste)",
                        legendgroup=name,
                        showlegend=(i == 0),
                        hovertemplate=f"<b>{tk} · {label}</b><br>K = %{{x:.2f}}"
                        "<br>IV = %{y:.2f}%<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )
                fig.add_trace(
                    go.Scatter(
                        x=f.strikes_obs,
                        y=100.0 * f.iv_obs,
                        mode="markers",
                        marker=dict(color=color, size=5, line=dict(color="#333", width=0.4)),
                        name=f"{name} (mercado)",
                        legendgroup=name,
                        showlegend=(i == 0),
                        hovertemplate=f"<b>{tk} · {label}</b><br>K = %{{x:.2f}}"
                        "<br>IV mercado = %{y:.2f}%<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )
                fig.add_vline(
                    x=f.forward,
                    line=dict(color=color, width=1, dash="dot"),
                    opacity=0.6,
                    row=row,
                    col=col,
                )
            fig.update_xaxes(title_text="Strike", row=row, col=col, title_font_size=10)
            fig.update_yaxes(title_text="IV (%)", row=row, col=col, title_font_size=10)

        # --- ponderación 1/K^2 ------------------------------------------------
        wrow = rows_smiles + 1
        for tk in tickers:
            f = sorted(fits_by_asset[tk], key=lambda x: x.T)[0]
            contrib = (f.dk / f.strikes_dense**2) * f.otm_prices
            denom = float(np.max(contrib)) if np.max(contrib) > 0 else 1e-12
            fig.add_trace(
                go.Scatter(
                    x=f.strikes_dense / f.forward,
                    y=contrib / denom,
                    mode="lines",
                    name=tk,
                    legendgroup=tk,
                    hovertemplate=f"<b>{tk}</b><br>K/F = %{{x:.3f}}"
                    "<br>peso relativo = %{y:.3f}<extra></extra>",
                ),
                row=wrow,
                col=1,
            )
        fig.update_xaxes(title_text="Moneyness  K / F", row=wrow, col=1, title_font_size=10)
        fig.update_yaxes(title_text="Contribución relativa", row=wrow, col=1, title_font_size=10)

        # --- matriz de correlación -------------------------------------------
        labels = list(breakdown.index)
        fig.add_trace(
            go.Heatmap(
                z=corr,
                x=labels,
                y=labels,
                zmin=-1.0,
                zmax=1.0,
                colorscale="RdBu",
                reversescale=True,
                showscale=True,
                colorbar=dict(
                    len=0.8 / total_rows,
                    y=1.0 - (wrow - 0.5) / total_rows,
                    yanchor="middle",
                    thickness=12,
                ),
                hovertemplate="ρ(%{y}, %{x}) = %{z:.3f}<extra></extra>",
            ),
            row=wrow,
            col=3,
        )
        fig.update_xaxes(
            tickmode="array", tickvals=labels, tickfont_size=9, row=wrow, col=3
        )
        fig.update_yaxes(
            tickmode="array",
            tickvals=labels,
            tickfont_size=9,
            autorange="reversed",
            row=wrow,
            col=3,
        )

        # --- atribución de riesgo --------------------------------------------
        arow = rows_smiles + 2
        fig.add_trace(
            go.Bar(
                x=labels,
                y=breakdown["VIX_individual"].values,
                name="VIX individual",
                marker_color="#3987e5",
                hovertemplate="<b>%{x}</b><br>VIX individual = %{y:.2f}<extra></extra>",
            ),
            row=arow,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=labels,
                y=breakdown["CTR_VIX_pts"].values,
                name="Contribución al VIX (pts)",
                marker_color="#e66767",
                customdata=np.stack(
                    [breakdown["peso"].values * 100.0, breakdown["CTR_%"].values], axis=-1
                ),
                hovertemplate="<b>%{x}</b><br>contribución = %{y:.2f} pts"
                "<br>peso = %{customdata[0]:.2f}%<br>CTR = %{customdata[1]:.1f}%<extra></extra>",
            ),
            row=arow,
            col=1,
        )
        fig.add_hline(
            y=metrics["VIX_portfolio"],
            line=dict(color="#383835", width=1.4, dash="dash"),
            annotation_text=f"VIX portafolio = {metrics['VIX_portfolio']:.2f}",
            annotation_position="top left",
            annotation_font_size=10,
            row=arow,
            col=1,
        )
        fig.update_yaxes(title_text="Puntos de VIX", row=arow, col=1, title_font_size=10)

        fig.add_trace(
            go.Pie(
                labels=labels,
                values=np.abs(breakdown["CTR_%"].values),
                textinfo="label+percent",
                textfont_size=9,
                hovertemplate="<b>%{label}</b><br>%{percent} del riesgo total<extra></extra>",
                showlegend=False,
            ),
            row=arow,
            col=3,
        )

        fig.update_layout(
            title=dict(
                text=(
                    f"<b>VIX de portafolio = {metrics['VIX_portfolio']:.2f}</b>"
                    f"   |   VIX medio ponderado = {metrics['VIX_medio_ponderado']:.2f}"
                    f"   |   ratio de diversificación = {metrics['ratio_diversificacion']:.3f}"
                    f"   |   correlación implícita media = "
                    f"{metrics['correlacion_implicita_media']:.3f}"
                ),
                x=0.5,
                xanchor="center",
                font=dict(size=17),
            ),
            template="plotly_white",
            barmode="group",
            height=max(760, 330 * total_rows),
            width=1500,
            margin=dict(l=70, r=60, t=110, b=60),
            legend=dict(orientation="v", yanchor="top", y=1.0, x=1.005, font=dict(size=10)),
            hovermode="closest",
        )
        fig.update_annotations(font_size=11)
        return fig

    @staticmethod
    def plot(
        fits_by_asset: Dict[str, List[SmileFit]],
        breakdown: pd.DataFrame,
        metrics: Dict[str, float],
        corr: np.ndarray,
        html_file: str = OUTPUT_HTML,
        show: bool = False,
    ) -> Optional[str]:
        if not _HAS_PLOTLY:
            warnings.warn(
                "plotly no está instalado: se omite el informe gráfico "
                "(`pip install plotly`).",
                RuntimeWarning,
            )
            return None

        fig = ReportPlotter.build_figure(fits_by_asset, breakdown, metrics, corr)

        out_html = os.path.abspath(html_file)
        fig.write_html(out_html, include_plotlyjs="cdn", full_html=True)

        if show:
            try:
                fig.show()
            except Exception:  # noqa: BLE001
                webbrowser.open(f"file://{out_html}")
        return out_html


# ==============================================================================
# BLOQUE 11: ORQUESTADOR
# Configuracion efectiva del calculo y encadenado de los bloques 5 a 10,
# excluyendo los activos que no se puedan valorar y renormalizando pesos.
# ==============================================================================
@dataclass
class VIXConfig:
    tickers: List[str] = field(default_factory=portfolio_tickers)
    weights: Optional[List[float]] = field(default_factory=portfolio_weights)
    r: float = RISK_FREE_RATE
    source: DataSource = "polygon"
    vol_method: VolMethod = "svi"
    corr_method: CorrMethod = "ewma"
    american_engine: AmericanEngine = "bs1993"
    n_grid: int = 401
    ewma_lambda: float = 0.94
    lookback: str = HISTORY_LOOKBACK
    verbose: bool = True
    html_file: str = OUTPUT_HTML
    show_plot: bool = SHOW_PLOT


class PortfolioVIXCalculator:
    def __init__(self, config: VIXConfig) -> None:
        self.cfg = config
        self.cleaner = OptionChainCleaner(verbose=config.verbose)
        self.deamer = DeAmericanizer(engine=config.american_engine, verbose=config.verbose)
        self.smiler = SmileBuilder(
            method=config.vol_method, n_grid=config.n_grid, verbose=config.verbose
        )
        self.single = SingleAssetVIX(
            config.r, self.cleaner, self.deamer, self.smiler, verbose=config.verbose
        )
        self.corr_est = ImpliedCorrelationEstimator(lam=config.ewma_lambda, verbose=config.verbose)
        self.aggregator = PortfolioVIX(verbose=config.verbose)
        self.source_used: str = config.source

    # -- datos -----------------------------------------------------------------
    def _load(self) -> Tuple[List[AssetMarketData], pd.DataFrame]:
        src = self.cfg.source

        if src == "polygon":
            try:
                loader = PolygonMarketLoader(
                    r=self.cfg.r,
                    engine=self.cfg.american_engine,
                    verbose=self.cfg.verbose,
                )
                data = loader.load(self.cfg.tickers, self.cfg.lookback)
                self.source_used = "polygon"
                return data
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"Fallo al descargar opciones de Polygon ({exc}); se prueba con yfinance.",
                    RuntimeWarning,
                )
                src = "yahoo"

        if src == "yahoo":
            if _HAS_YF:
                try:
                    data = YahooMarketLoader(verbose=self.cfg.verbose).load(
                        self.cfg.tickers, self.cfg.lookback
                    )
                    self.source_used = "yahoo"
                    return data
                except Exception as exc:  # noqa: BLE001
                    warnings.warn(
                        f"Fallo al descargar datos de mercado ({exc}); se conmuta a datos "
                        "sintéticos autogenerados.",
                        RuntimeWarning,
                    )
            else:
                warnings.warn(
                    "yfinance no está instalado; se usan datos sintéticos autogenerados "
                    "(`pip install yfinance` para datos reales).",
                    RuntimeWarning,
                )

        self.source_used = "synthetic"
        return SyntheticMarketGenerator(r=self.cfg.r, verbose=self.cfg.verbose).generate(
            self.cfg.tickers
        )

    # -- ejecución -------------------------------------------------------------
    def run(self) -> Dict[str, object]:
        assets, prices = self._load()

        sig30: List[float] = []
        used: List[str] = []
        fits_by_asset: Dict[str, List[SmileFit]] = {}
        for a in assets:
            try:
                var30, fits = self.single.compute(a)
                sig30.append(math.sqrt(var30))
                used.append(a.ticker)
                fits_by_asset[a.ticker] = fits
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"[{a.ticker}] Excluido del portafolio: {exc}", RuntimeWarning)

        if not used:
            raise RuntimeError("ningún activo pudo ser valorado; revise los datos de entrada")

        # pesos alineados con los activos efectivamente valorados
        if self.cfg.weights is None:
            w = np.full(len(used), 1.0 / len(used))
        else:
            wmap = dict(zip(self.cfg.tickers, self.cfg.weights))
            w = np.array([wmap.get(t, 0.0) for t in used], dtype=float)
            if w.sum() <= 0:
                w = np.full(len(used), 1.0 / len(used))

        prices = prices[used]
        corr = self.corr_est.estimate(prices, self.cfg.corr_method)
        vix, cov, breakdown, metrics = self.aggregator.aggregate(
            used, w, np.array(sig30), corr
        )
        figure = (
            ReportPlotter.build_figure(fits_by_asset, breakdown, metrics, corr)
            if _HAS_PLOTLY
            else None
        )
        plot = ReportPlotter.plot(
            fits_by_asset,
            breakdown,
            metrics,
            corr,
            self.cfg.html_file,
            self.cfg.show_plot,
        )

        if self.cfg.verbose:
            self._print_report(breakdown, metrics, corr, used, plot)

        return {
            "vix": vix,
            "breakdown": breakdown,
            "metrics": metrics,
            "covariance": pd.DataFrame(cov, index=used, columns=used),
            "correlation": pd.DataFrame(corr, index=used, columns=used),
            "fits": fits_by_asset,
            "figure": figure,
            "plot": plot,
            "source": self.source_used,
        }

    @staticmethod
    def _print_report(
        breakdown: pd.DataFrame,
        metrics: Dict[str, float],
        corr: np.ndarray,
        tickers: Sequence[str],
        plot: Optional[str],
    ) -> None:
        with pd.option_context("display.float_format", lambda v: f"{v:,.4f}"):
            print("\n" + "=" * 80)
            print(" RESULTADO — VIX DE PORTAFOLIO (30 días, model-free)")
            print("=" * 80)
            print(f"\n  VIX_port = {metrics['VIX_portfolio']:.2f}\n")
            print("--- Desglose y atribución de riesgo ---")
            print(breakdown.to_string())
            print("\n--- Matriz de correlación ---")
            print(pd.DataFrame(corr, index=list(tickers), columns=list(tickers)).to_string())
            print("\n--- Métricas de diversificación ---")
            for k, v in metrics.items():
                print(f"  {k:<32s} : {v:,.4f}")
            if plot:
                print(f"\n  Informe interactivo: {plot}")
            print("=" * 80)


# ==============================================================================
# BLOQUE 12: CLI
# Argumentos de linea de comandos. Sin argumentos se usa el portafolio y las
# rutas declaradas en el BLOQUE 1.
# ==============================================================================
def _parse_args() -> VIXConfig:
    p = argparse.ArgumentParser(description="VIX de portafolio model-free (CBOE extendido)")
    p.add_argument("--tickers", nargs="+", default=None, help="sobrescribe PORTFOLIO_HOLDINGS")
    p.add_argument("--weights", nargs="+", type=float, default=None)
    p.add_argument("--r", type=float, default=RISK_FREE_RATE, help="tasa libre de riesgo continua")
    p.add_argument("--source", choices=["polygon", "yahoo", "synthetic"], default="polygon")
    p.add_argument("--vol-method", choices=["svi", "spline"], default="svi")
    p.add_argument("--corr-method", choices=["ewma", "rmt", "sample"], default="ewma")
    p.add_argument("--american-engine", choices=["bs1993", "crr"], default="bs1993")
    p.add_argument("--lookback", default=HISTORY_LOOKBACK)
    p.add_argument(
        "--synthetic", action="store_true", help="atajo para --source synthetic (sin red)"
    )
    p.add_argument("--show", dest="show", action="store_true", default=None,
                   help="abrir el informe interactivo")
    p.add_argument("--no-show", dest="show", action="store_false",
                   help="no abrir el informe interactivo")
    p.add_argument("--html", default=OUTPUT_HTML, help="ruta del informe interactivo")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()

    tickers = a.tickers if a.tickers else portfolio_tickers()
    if a.weights is not None:
        weights: Optional[List[float]] = list(a.weights)
    elif a.tickers:
        weights = None  # portafolio ad hoc por CLI: equiponderado salvo --weights
    else:
        weights = portfolio_weights()

    return VIXConfig(
        tickers=tickers,
        weights=weights,
        r=a.r,
        source="synthetic" if a.synthetic else a.source,
        vol_method=a.vol_method,
        corr_method=a.corr_method,
        american_engine=a.american_engine,
        lookback=a.lookback,
        verbose=not a.quiet,
        html_file=a.html,
        show_plot=SHOW_PLOT if a.show is None else a.show,
    )


if __name__ == "__main__":
    warnings.simplefilter("always", RuntimeWarning)
    cfg = _parse_args()
    pesos_txt = (
        "equiponderado"
        if cfg.weights is None
        else ", ".join(f"{t} {100*w:.2f}%" for t, w in zip(cfg.tickers, cfg.weights))
    )
    print("=" * 80)
    print(" PORTFOLIO VIX ENGINE — CBOE model-free variance con ajuste de estilo americano")
    print("=" * 80)
    print(
        f"  Activos         : {len(cfg.tickers)} -> {cfg.tickers}\n"
        f"  Pesos           : {pesos_txt}\n"
        f"  r               : {cfg.r:.4f}\n"
        f"  Sonrisa         : {cfg.vol_method}\n"
        f"  Correlación     : {cfg.corr_method}\n"
        f"  Motor americano : {cfg.american_engine}\n"
        f"  Fuente opciones : {cfg.source}"
    )
    results = PortfolioVIXCalculator(cfg).run()
