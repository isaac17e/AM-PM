# ===============================================
# Script: Optimizacion de portafolio - Utilidad Cuadratica Seasonal Version
# ===============================================

import warnings
warnings.filterwarnings("ignore")

import time
import math
import itertools
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

import yfinance as yf
import pandas_datareader.data as pdr
import statsmodels.api as sm
from scipy.stats import norm
import quadprog

import matplotlib.pyplot as plt

# ------------------------------------------------
# API KEY - Polygon.io
# ------------------------------------------------
import os
from dotenv import load_dotenv
load_dotenv()
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY")

if not POLYGON_API_KEY:
    print("ADVERTENCIA: No hay POLYGON_API_KEY configurada. Todas las consultas de opciones")
    print("             fallaran y cada activo caera a fallback historico (sin BKM real).")

polygon_dte_tol = 10  # +/- dias alrededor del tenor objetivo

# ================================================
# PARAMETROS CONFIGURABLES
# ================================================

n_top_nasdaq = 90
n_top_sp500 = 90
n_top_int = 50

rebalance_months = [9]

target_years = list(range(2019, 2026))
mdd_start_year = 2019
rf_rate = 0.046
seed = 123
max_weight = 0.30
n_sim = 5000
target_total_tickers = 500

lookback_months = None

lambda_ = 0.8
weight_sharpe = 0.55
weight_low_vol = 0.15
weight_decorr = 0.30

n_pre_seasonal = 45
n_divers_candidates = 30

# Seleccion conjunta de candidatos (h_i / J_ij) via QUBO/Ising.
# Por debajo del umbral: fuerza bruta (optimo exacto). Por encima: Simulated Annealing.
qubo_exact_threshold = 2e6
qubo_sa_iterations = 20000

volatility_percentile = 0.85
correlation_percentile = 0.85

seasonal_min_weeks = 40

# Umbral de exclusion estacional: si vol_estacional_anualizada / vol_general_anualizada
# de un activo supera este multiplicador, se descarta (no se penaliza, se excluye)
seasonal_vol_ratio_max = 1.80

# Piso minimo de activos que deben sobrevivir al filtro estacional; si el umbral
# deja menos que esto, se relaja tomando los siguientes mejores por ratio hasta alcanzarlo
seasonal_min_survivors = 18

# Filtro MFIS (Bakshi-Kapadia-Madan) sobre el grupo final, antes de la matriz de covarianza
bkm_moneyness_lo = 0.70          # limite inferior de moneyness K/S para strikes OTM (integracion BKM)
bkm_moneyness_hi = 1.40          # limite superior de moneyness K/S para strikes OTM (integracion BKM)
bkm_hist_moneyness_grid = np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15])  # grid reducido, reconstruccion historica
bkm_min_options_per_side = 3     # minimo de strikes OTM por lado (calls/puts) para integracion valida
bkm_mfis_clip = 10.0              # cota de winsorizacion para MFIS (estabilidad numerica en activos de baja MFIV)
bkm_mfik_clip = 30.0              # cota de winsorizacion para MFIK (idem)
bkm_lookback_months = 12         # ventana para reconstruir el MFIS historico "normal" del activo
bkm_hist_sample_freq = "2W"      # frecuencia de muestreo historico BKM (2W=quincenal; toleramos huecos, min. 8 puntos validos)
bkm_max_workers = 6              # tickers evaluados en paralelo durante el filtro BKM (ajustar segun rate limit del plan Polygon)
bkm_z_threshold = 1.75           # se descarta si (MFIS_hoy - media_hist) / sd_hist supera esto
bkm_min_survivors = 14           # piso minimo tras el filtro BKM (se repone desde el pool estacional)
tail_penalty_theta1 = 0.005       # theta1: escala de penalizacion por asimetria implicita (MFIS) en mu
tail_penalty_theta2 = 0.001       # theta2: escala de penalizacion por curtosis implicita (MFIK) en mu
cornish_fisher_confidence = 0.95  # nivel de confianza para VaR/CVaR ajustados por Cornish-Fisher

correlation_order = 0

min_observations = 24
ideal_observations = 60

use_delta_filter = True
delta_min = 0.30
delta_strike_mode = "rf"
iv_outlier_multiplier = 6.0

max_region_weight = 0.80

# Porcentaje deseado de ETFs en el portafolio FINAL (resto queda en acciones individuales)
pct_etf_deseado = 0.10     # ej. 15% ETFs / 85% acciones individuales
pct_etf_tolerancia = 0.10  # banda +/- alrededor del porcentaje deseado

# === VALIDACION DE PARAMETROS ===
if not (1 <= len(rebalance_months) <= 3):
    raise ValueError("Error: rebalance_months debe contener 1, 2 o 3 meses")
if any(m < 1 or m > 12 for m in rebalance_months):
    raise ValueError("Error: Los meses deben estar entre 1 y 12")
if not (0 <= pct_etf_deseado <= 1):
    raise ValueError("Error: pct_etf_deseado debe estar entre 0 y 1 (ej. 0.30 = 30%)")
if not (0 <= pct_etf_tolerancia <= 0.5):
    raise ValueError("Error: pct_etf_tolerancia debe estar entre 0 y 0.5")

horizon_months = len(rebalance_months)
MONTH_ABB = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_NAME = ["", "January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"]
horizon_label = "-".join(MONTH_ABB[m] for m in rebalance_months)
target_months = rebalance_months

print("\n" + "=" * 60)
print(f"MES(ES) DE REBALANCEO: {horizon_label}")
lb_txt = f" (ultimos {lookback_months} meses)" if lookback_months is not None else " (sin restriccion de ventana)"
print("ENTRENAMIENTO: todo el historico mensual disponible" + lb_txt)
print("=" * 60 + "\n")

rf_rate_monthly = rf_rate / 12


# ================================================
# FUNCION AUXILIAR: Web scraping seguro
# ================================================
def safe_scrape_table(url, fallback=None, headers=None):
    try:
        headers = headers or {"User-Agent": "Mozilla/5.0 (compatible; PortfolioBot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        if not tables:
            return fallback
        return tables[0]
    except Exception as e:
        print(f"Error al obtener datos de {url}: {e}")
        return fallback


# ================================================
# FUNCIONES AUXILIARES: Polygon.io
# ================================================
import re


def is_us_ticker(ticker):
    return not re.search(r"\.(TO|DE|L|PA|MC|T)$", ticker)


def polygon_format_ticker(ticker):
    return ticker.replace("-", ".")


polygon_diag = {
    "status_counts": {},
    "sample_errors": [],
    "total_attempts": 0,
    "empty_match_count": 0,
}
polygon_diag_lock = threading.Lock()  # protege polygon_diag: el filtro BKM ahora evalua tickers en paralelo


def polygon_log_status(status, body=None):
    key = str(status)
    with polygon_diag_lock:
        polygon_diag["status_counts"][key] = polygon_diag["status_counts"].get(key, 0) + 1
        if status != 200 and len(polygon_diag["sample_errors"]) < 5:
            polygon_diag["sample_errors"].append({"status": status, "body": body})


def polygon_api_request(url, max_retries=5, sleep_between=0.3):
    with polygon_diag_lock:
        polygon_diag["total_attempts"] += 1
    for intento in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=20)
        except Exception:
            time.sleep(sleep_between * intento)
            continue
        status = resp.status_code
        if status == 200:
            polygon_log_status(status)
            time.sleep(sleep_between)
            return resp
        if status == 429:
            retry_after = resp.headers.get("retry-after")
            try:
                retry_after = float(retry_after)
            except (TypeError, ValueError):
                retry_after = None
            espera = retry_after if (retry_after and retry_after > 0) else sleep_between * (2 ** intento)
            time.sleep(espera)
            continue
        try:
            body_txt = resp.text
        except Exception:
            body_txt = None
        polygon_log_status(status, body_txt)
        time.sleep(sleep_between)
        return resp
    polygon_log_status(429, "limite de reintentos agotado tras 429 repetido")
    time.sleep(sleep_between)
    return None


def polygon_get_atm_option(ticker, target_dte, dte_tol=10, api_key=None, contract_type="call"):
    api_key = api_key or POLYGON_API_KEY
    vacio = dict(iv=np.nan, delta=np.nan, gamma=np.nan, vega=np.nan, theta=np.nan,
                 dte=np.nan, strike=np.nan, ok=False)

    hoy = date.today()
    fecha_min = (hoy + timedelta(days=max(target_dte - dte_tol, 1))).strftime("%Y-%m-%d")
    fecha_max = (hoy + timedelta(days=target_dte + dte_tol)).strftime("%Y-%m-%d")

    url_chain = (
        f"https://api.polygon.io/v3/snapshot/options/{polygon_format_ticker(ticker)}?"
        f"contract_type={contract_type}&"
        f"expiration_date.gte={fecha_min}&expiration_date.lte={fecha_max}&"
        f"limit=250&apiKey={api_key}"
    )

    try:
        resp = polygon_api_request(url_chain)
        if resp is None or resp.status_code != 200:
            return vacio
        data = resp.json()
        results = data.get("results")
        if not results:
            polygon_diag["empty_match_count"] += 1
            return vacio
        df = pd.json_normalize(results)
        if df.empty:
            polygon_diag["empty_match_count"] += 1
            return vacio

        df["dte"] = (pd.to_datetime(df["details.expiration_date"]) - pd.Timestamp(hoy)).dt.days
        df = df[df.get("implied_volatility").notna() & (df.get("implied_volatility") > 0) &
                 df.get("greeks.delta").notna()].copy()
        if df.empty:
            polygon_diag["empty_match_count"] += 1
            return vacio

        df["_score1"] = (df["greeks.delta"] - 0.50).abs()
        df["_score2"] = (df["dte"] - target_dte).abs()
        df = df.sort_values(["_score1", "_score2"])
        c1 = df.iloc[0]

        return dict(
            iv=c1.get("implied_volatility", np.nan),
            delta=c1.get("greeks.delta", np.nan),
            gamma=c1.get("greeks.gamma", np.nan),
            vega=c1.get("greeks.vega", np.nan),
            theta=c1.get("greeks.theta", np.nan),
            dte=c1.get("dte", np.nan),
            strike=c1.get("details.strike_price", np.nan),
            ok=True,
        )
    except Exception:
        return vacio


# ================================================
# FUNCIONES AUXILIARES: BKM (Black-Scholes, contratos historicos)
# ================================================
def bs_price(S, K, T_yrs, r, sigma, tipo="call"):
    if T_yrs <= 0 or sigma <= 0:
        return np.nan
    d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T_yrs) / (sigma * math.sqrt(T_yrs))
    d2 = d1 - sigma * math.sqrt(T_yrs)
    if tipo == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T_yrs) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T_yrs) * norm.cdf(-d2) - S * norm.cdf(-d1)


def polygon_find_contract_asof(ticker, contract_type, strike_target, expiration_target,
                                as_of_date, strike_band=0.06, expiration_band_days=12,
                                api_key=None):
    api_key = api_key or POLYGON_API_KEY
    strike_min = strike_target * (1 - strike_band)
    strike_max = strike_target * (1 + strike_band)
    exp_target_d = pd.Timestamp(expiration_target)
    exp_min = (exp_target_d - timedelta(days=expiration_band_days)).strftime("%Y-%m-%d")
    exp_max = (exp_target_d + timedelta(days=expiration_band_days)).strftime("%Y-%m-%d")
    url_ref = (
        f"https://api.polygon.io/v3/reference/options/contracts?"
        f"underlying_ticker={polygon_format_ticker(ticker)}&contract_type={contract_type}&"
        f"strike_price.gte={strike_min:.4f}&strike_price.lte={strike_max:.4f}&"
        f"expiration_date.gte={exp_min}&expiration_date.lte={exp_max}&"
        f"as_of={pd.Timestamp(as_of_date).strftime('%Y-%m-%d')}&limit=100&apiKey={api_key}"
    )
    try:
        resp = polygon_api_request(url_ref)
        if resp is None or resp.status_code != 200:
            return None
        data = resp.json()
        results = data.get("results")
        if not results:
            return None
        df = pd.json_normalize(results)
        if df.empty:
            return None
        df["d_strike"] = (df["strike_price"] - strike_target).abs()
        df["d_exp"] = (pd.to_datetime(df["expiration_date"]) - exp_target_d).abs().dt.days
        df = df.sort_values(["d_strike", "d_exp"])
        return df.iloc[0]["ticker"]
    except Exception:
        return None


def polygon_get_contract_close_near(contract_ticker, target_date, window_days=5, api_key=None):
    api_key = api_key or POLYGON_API_KEY
    target_d = pd.Timestamp(target_date)
    from_d = (target_d - timedelta(days=window_days)).strftime("%Y-%m-%d")
    to_d = (target_d + timedelta(days=window_days)).strftime("%Y-%m-%d")
    url_aggs = (
        f"https://api.polygon.io/v2/aggs/ticker/{contract_ticker}/range/1/day/"
        f"{from_d}/{to_d}?adjusted=true&sort=asc&limit=50&apiKey={api_key}"
    )
    try:
        resp = polygon_api_request(url_aggs)
        if resp is None or resp.status_code != 200:
            return np.nan
        data = resp.json()
        results = data.get("results")
        if not results:
            return np.nan
        df = pd.json_normalize(results)
        if df.empty:
            return np.nan
        df["fecha"] = pd.to_datetime(df["t"], unit="ms").dt.date
        df["d_dias"] = (pd.to_datetime(df["fecha"]) - target_d).abs().dt.days
        df = df.sort_values("d_dias")
        return df.iloc[0]["c"]
    except Exception:
        return np.nan


def get_spot_safe_bkm(ticker):
    try:
        fi = yf.Ticker(ticker).fast_info
        px = fi.get("lastPrice") if hasattr(fi, "get") else None
        if px is None:
            px = getattr(fi, "last_price", None)
        if px is None:
            info = yf.Ticker(ticker).info
            px = info.get("regularMarketPrice") or info.get("currentPrice")
        return float(px) if px is not None else np.nan
    except Exception:
        return np.nan


# ================================================
# FUNCIONES BKM (Bakshi, Kapadia y Madan, 2003)
#
# Momentos model-free a partir de la curva continua de precios OTM (calls K>=S, puts K<S):
#
#   V(T) = int_S^inf [2*(1-ln(K/S))/K^2] C(K) dK + int_0^S [2*(1+ln(S/K))/K^2] P(K) dK
#   W(T) = int_S^inf [6*ln(K/S) - 3*ln(K/S)^2]/K^2 C(K) dK - int_0^S [6*ln(S/K) + 3*ln(S/K)^2]/K^2 P(K) dK
#   X(T) = int_S^inf [12*ln(K/S)^2 - 4*ln(K/S)^3]/K^2 C(K) dK + int_0^S [12*ln(S/K)^2 + 4*ln(S/K)^3]/K^2 P(K) dK
#
#   mu(T)   = e^{rT} - 1 - (e^{rT}/2)*V - (e^{rT}/6)*W - (e^{rT}/24)*X
#   MFIV(T) = e^{rT}*V - mu(T)^2
#   MFIS(T) = [e^{rT}*W - 3*mu(T)*e^{rT}*V + 2*mu(T)^3] / MFIV(T)^{3/2}
#   MFIK(T) = [e^{rT}*X - 4*mu(T)*e^{rT}*W + 6*e^{rT}*mu(T)^2*V - 3*mu(T)^4] / MFIV(T)^2
#
# La integracion se resuelve numericamente (regla del Trapecio, np.trapezoid) sobre los
# strikes OTM disponibles en cada lado (calls/puts).
# ================================================
def bkm_iv_chain_to_prices(S, r, T, chain_df):
    if chain_df is None or chain_df.empty:
        return chain_df
    chain_df = chain_df.copy()
    chain_df["price"] = [
        bs_price(S, k, T, r, iv, tipo)
        for k, iv, tipo in zip(chain_df["strike"], chain_df["iv"], chain_df["type"])
    ]
    chain_df = chain_df[chain_df["price"].notna() & (chain_df["price"] > 0)]
    return chain_df


def bkm_compute_moments(S, r, T, calls_df, puts_df):
    if calls_df is None or puts_df is None or len(calls_df) < bkm_min_options_per_side \
            or len(puts_df) < bkm_min_options_per_side or T <= 0 or S <= 0:
        return dict(mfiv=np.nan, mfis=np.nan, mfik=np.nan, mu=np.nan, ok=False)

    calls_df = calls_df.sort_values("strike")
    puts_df = puts_df.sort_values("strike")
    Kc, Cc = calls_df["strike"].values.astype(float), calls_df["price"].values.astype(float)
    Kp, Pp = puts_df["strike"].values.astype(float), puts_df["price"].values.astype(float)

    fC_V = (2 * (1 - np.log(Kc / S))) / Kc ** 2 * Cc
    fP_V = (2 * (1 + np.log(S / Kp))) / Kp ** 2 * Pp
    fC_W = (6 * np.log(Kc / S) - 3 * np.log(Kc / S) ** 2) / Kc ** 2 * Cc
    fP_W = (6 * np.log(S / Kp) + 3 * np.log(S / Kp) ** 2) / Kp ** 2 * Pp
    fC_X = (12 * np.log(Kc / S) ** 2 - 4 * np.log(Kc / S) ** 3) / Kc ** 2 * Cc
    fP_X = (12 * np.log(S / Kp) ** 2 + 4 * np.log(S / Kp) ** 3) / Kp ** 2 * Pp

    try:
        V = np.trapezoid(fC_V, Kc) + np.trapezoid(fP_V, Kp)
        W = np.trapezoid(fC_W, Kc) - np.trapezoid(fP_W, Kp)
        X = np.trapezoid(fC_X, Kc) + np.trapezoid(fP_X, Kp)
    except Exception:
        return dict(mfiv=np.nan, mfis=np.nan, mfik=np.nan, mu=np.nan, ok=False)

    erT = math.exp(r * T)
    mu = erT - 1 - erT / 2 * V - erT / 6 * W - erT / 24 * X
    mfiv = erT * V - mu ** 2
    if not np.isfinite(mfiv) or mfiv <= 0:
        return dict(mfiv=np.nan, mfis=np.nan, mfik=np.nan, mu=np.nan, ok=False)

    mfis = (erT * W - 3 * mu * erT * V + 2 * mu ** 3) / mfiv ** 1.5
    mfik = (erT * X - 4 * mu * erT * W + 6 * erT * mu ** 2 * V - 3 * mu ** 4) / mfiv ** 2

    if not (np.isfinite(mfis) and np.isfinite(mfik)):
        return dict(mfiv=np.nan, mfis=np.nan, mfik=np.nan, mu=np.nan, ok=False)

    mfis = float(np.clip(mfis, -bkm_mfis_clip, bkm_mfis_clip))
    mfik = float(np.clip(mfik, 0.0, bkm_mfik_clip))

    return dict(mfiv=mfiv, mfis=mfis, mfik=mfik, mu=mu, ok=True)


def bkm_fetch_otm_chain(ticker, target_dte, dte_tol, S, moneyness_lo, moneyness_hi, api_key=None):
    api_key = api_key or POLYGON_API_KEY
    hoy = date.today()
    fecha_min = (hoy + timedelta(days=max(target_dte - dte_tol, 1))).strftime("%Y-%m-%d")
    fecha_max = (hoy + timedelta(days=target_dte + dte_tol)).strftime("%Y-%m-%d")
    strike_min = S * moneyness_lo
    strike_max = S * moneyness_hi

    resultado = {"call": pd.DataFrame(columns=["strike", "iv", "type"]),
                 "put": pd.DataFrame(columns=["strike", "iv", "type"])}
    for tipo in ("call", "put"):
        url = (
            f"https://api.polygon.io/v3/snapshot/options/{polygon_format_ticker(ticker)}?"
            f"contract_type={tipo}&"
            f"strike_price.gte={strike_min:.4f}&strike_price.lte={strike_max:.4f}&"
            f"expiration_date.gte={fecha_min}&expiration_date.lte={fecha_max}&"
            f"limit=250&apiKey={api_key}"
        )
        resp = polygon_api_request(url)
        if resp is None or resp.status_code != 200:
            continue
        try:
            results = resp.json().get("results")
            if not results:
                continue
            df = pd.json_normalize(results)
            df = df[df.get("implied_volatility").notna() & (df.get("implied_volatility") > 0)].copy()
            if df.empty:
                continue
            df["strike"] = df["details.strike_price"]
            df["iv"] = df["implied_volatility"]
            df["type"] = tipo
            df = df[df["strike"] >= S] if tipo == "call" else df[df["strike"] < S]
            resultado[tipo] = df[["strike", "iv", "type"]].drop_duplicates("strike")
        except Exception:
            continue
    return resultado["call"], resultado["put"]


def bkm_get_current_moments(ticker, target_dte, dte_tol, rf):
    S = get_spot_safe_bkm(ticker)
    if pd.isna(S) or S <= 0:
        return dict(mfiv=np.nan, mfis=np.nan, mfik=np.nan, mu=np.nan, ok=False, spot=np.nan)
    calls_df, puts_df = bkm_fetch_otm_chain(ticker, target_dte, dte_tol, S, bkm_moneyness_lo, bkm_moneyness_hi)
    T = target_dte / 365
    calls_df = bkm_iv_chain_to_prices(S, rf, T, calls_df)
    puts_df = bkm_iv_chain_to_prices(S, rf, T, puts_df)
    mom = bkm_compute_moments(S, rf, T, calls_df, puts_df)
    mom["spot"] = S
    return mom


def bkm_reconstruct_mfis_history(ticker, spot_series, sample_dates, target_dte, moneyness_grid, rf):
    mfis_hist = [np.nan] * len(sample_dates)
    for i, fecha_i in enumerate(sample_dates):
        fecha_i = pd.Timestamp(fecha_i)
        spot_row = spot_series[spot_series["date"] <= fecha_i].sort_values("date", ascending=False)
        if spot_row.empty:
            continue
        S_i = spot_row.iloc[0]["adjusted"]
        exp_target = fecha_i + timedelta(days=target_dte)
        T_i = target_dte / 365

        call_rows, put_rows = [], []
        for m in moneyness_grid:
            strike_target = S_i * m
            tipo = "call" if m >= 1.0 else "put"
            ct = polygon_find_contract_asof(ticker, tipo, strike_target, exp_target, fecha_i)
            if ct is None:
                continue
            px = polygon_get_contract_close_near(ct, fecha_i)
            if pd.isna(px) or px <= 0:
                continue
            (call_rows if tipo == "call" else put_rows).append(dict(strike=strike_target, price=px))

        calls_df = pd.DataFrame(call_rows)
        puts_df = pd.DataFrame(put_rows)
        mom = bkm_compute_moments(S_i, rf, T_i, calls_df, puts_df)
        if mom["ok"]:
            mfis_hist[i] = mom["mfis"]
    return np.array(mfis_hist)


# Mapeo heuristico de SIC description (Polygon reference endpoint) a ETF sectorial SPDR.
sector_keywords = {
    "XLK": ["SEMICONDUCTOR", "COMPUTER", "SOFTWARE", "ELECTRONIC COMPONENTS", "COMPUTER PROGRAMMING", "COMPUTER PERIPHERAL"],
    "XLV": ["PHARMACEUTICAL", "BIOLOGICAL", "MEDICAL", "HOSPITAL", "HEALTH", "SURGICAL", "DRUG"],
    "XLF": ["BANK", "INSURANCE", "INVESTMENT OFFICE", "FINANCE", "SECURITY BROKER", "SAVINGS INSTITUTION"],
    "XLE": ["PETROLEUM", "CRUDE", "NATURAL GAS", "DRILLING", "OIL"],
    "XLY": ["RETAIL", "APPAREL", "HOTEL", "RESTAURANT", "AUTOMOTIVE", "MOTOR VEHICLE", "DEPARTMENT STORE"],
    "XLP": ["FOOD", "BEVERAGE", "TOBACCO", "HOUSEHOLD", "GROCERY", "SOAP"],
    "XLI": ["MACHINERY", "AEROSPACE", "INDUSTRIAL", "TRANSPORTATION", "CONSTRUCTION", "DEFENSE", "RAILROAD", "AIR TRANSPORT"],
    "XLB": ["CHEMICAL", "MINING", "METAL", "PAPER", "STEEL"],
    "XLU": ["ELECTRIC", "UTILITY", "GAS AND ELECTRIC", "WATER SUPPLY"],
    "XLRE": ["REAL ESTATE", "REIT", "LESSORS OF REAL PROPERTY"],
    "XLC": ["TELEPHONE", "TELECOMMUNICATIONS", "BROADCASTING", "CABLE", "MOTION PICTURE", "PUBLISHING", "ADVERTISING", "INTERNET", "RADIOTELEPHONE"],
}


def polygon_get_sic_description(ticker, api_key=None):
    api_key = api_key or POLYGON_API_KEY
    url_ref = f"https://api.polygon.io/v3/reference/tickers/{polygon_format_ticker(ticker)}?apiKey={api_key}"
    try:
        resp = polygon_api_request(url_ref)
        if resp is None or resp.status_code != 200:
            return None
        data = resp.json()
        results = data.get("results")
        if not results or not results.get("sic_description"):
            return None
        return results["sic_description"].upper()
    except Exception:
        return None


def polygon_get_sector_etf(ticker):
    sic = polygon_get_sic_description(ticker)
    if sic is None:
        return None
    for etf, keywords in sector_keywords.items():
        if any(kw in sic for kw in keywords):
            return etf
    return None


# ================================================
# OBTENER TICKERS: S&P 500
# ================================================
print("Obteniendo tickers del S&P 500...")
sp500_tbl = safe_scrape_table("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")

if sp500_tbl is None or "Symbol" not in sp500_tbl.columns:
    raise RuntimeError("No se pudo obtener la lista de miembros del S&P 500 desde Wikipedia.")

sp500_members = (
    sp500_tbl["Symbol"].astype(str).str.replace(".", "-", regex=False).str.upper().unique().tolist()
)
print(f"Miembros del S&P 500 obtenidos: {len(sp500_members)}")
print("Consultando market cap (Yahoo Finance) para ordenar por tamaño...")


def get_market_cap(ticker):
    try:
        fi = yf.Ticker(ticker).fast_info
        mc = fi.get("marketCap") if hasattr(fi, "get") else None
        if mc is None:
            mc = getattr(fi, "market_cap", None)
        if mc is None or (isinstance(mc, float) and np.isnan(mc)):
            raise ValueError
        return float(mc)
    except Exception:
        try:
            info = yf.Ticker(ticker).info
            mc = info.get("marketCap")
            return float(mc) if mc is not None else np.nan
        except Exception:
            return np.nan


market_cap_rows = []
for i, tk in enumerate(sp500_members, start=1):
    market_cap_rows.append({"symbol": tk, "market_cap": get_market_cap(tk)})
    if i % 50 == 0 or i == len(sp500_members):
        print(f"   Market cap: {i}/{len(sp500_members)}")

sp500_marketcap = pd.DataFrame(market_cap_rows).dropna(subset=["market_cap"])

if len(sp500_marketcap) > 0:
    sp500_tickers = (
        sp500_marketcap.sort_values("market_cap", ascending=False)
        .head(n_top_sp500)["symbol"].tolist()
    )
    print(f"S&P 500: {n_top_sp500} tickers objetivo, {len(sp500_tickers)} unicos obtenidos "
          f"(ordenados por market cap real)")
else:
    print("Error consultando market cap. Se usara el orden de la tabla original.")
    sp500_tickers = sp500_members[:n_top_sp500]
    print(f"S&P 500: {n_top_sp500} tickers objetivo, {len(sp500_tickers)} unicos obtenidos "
          f"(orden de tabla, sin market cap)")

# ================================================
# OBTENER TICKERS: NASDAQ
# ================================================
print("\nObteniendo tickers del NASDAQ...")
nasdaq_tbl = safe_scrape_table("https://stockanalysis.com/list/nasdaq-stocks/")

if nasdaq_tbl is not None:
    nasdaq_tbl_clean = nasdaq_tbl.copy()
    nasdaq_tbl_clean.columns = [str(c).strip().lower().replace(" ", "_") for c in nasdaq_tbl_clean.columns]
    symbol_col = "symbol" if "symbol" in nasdaq_tbl_clean.columns else nasdaq_tbl_clean.columns[1]
    nasdaq_tbl_clean = nasdaq_tbl_clean.rename(columns={symbol_col: "symbol"})
    nasdaq_tbl_clean = nasdaq_tbl_clean[
        nasdaq_tbl_clean["symbol"].notna()
        & ~nasdaq_tbl_clean["symbol"].astype(str).str.contains(r"\^|\$", regex=True)
        & (nasdaq_tbl_clean["symbol"].astype(str).str.len() >= 1)
        & (nasdaq_tbl_clean["symbol"].astype(str).str.len() <= 5)
        & ~nasdaq_tbl_clean["symbol"].astype(str).str.match(r"^[0-9]")
    ].copy()
    nasdaq_tbl_clean["symbol"] = nasdaq_tbl_clean["symbol"].astype(str).str.upper().str.replace(".", "-", regex=False)

    nasdaq_tickers = nasdaq_tbl_clean["symbol"].iloc[: min(n_top_nasdaq, len(nasdaq_tbl_clean))].unique().tolist()
    print(f"NASDAQ: {n_top_nasdaq} tickers objetivo, {len(nasdaq_tickers)} unicos obtenidos")

    shortage_nasdaq = n_top_nasdaq - len(nasdaq_tickers)
    if shortage_nasdaq > 0 and len(nasdaq_tbl_clean) > n_top_nasdaq:
        additional_needed = min(shortage_nasdaq * 2, len(nasdaq_tbl_clean) - n_top_nasdaq)
        additional_nasdaq = (
            nasdaq_tbl_clean["symbol"]
            .iloc[n_top_nasdaq: n_top_nasdaq + additional_needed]
            .unique().tolist()
        )
        additional_nasdaq_unique = [t for t in additional_nasdaq if t not in nasdaq_tickers]
        if additional_nasdaq_unique:
            to_add = additional_nasdaq_unique[: shortage_nasdaq]
            nasdaq_tickers = nasdaq_tickers + to_add
else:
    nasdaq_tbl_clean = None
    nasdaq_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
                       "AVGO", "ASML", "COST", "NFLX", "AMD", "PEP", "ADBE", "CSCO"]
    print("Usando NASDAQ fallback")

# ================================================
# ETFs
# ================================================
etf_core = ["SPY", "QQQ", "VOO", "VTI", "VYM", "IWM", "GLD", "SLV", "USO", "PDBC", "HYG", "VNQ"]

# VOX (Vanguard Communication Services ETF, inception 2004) se agrega junto a XLC:
# XLC sigue siendo el ETF de referencia para el mapeo SIC->sector de acciones US
# (sector_keywords), pero el mapeo manual de internacionales usa VOX para el
# sector de Comunicaciones (ver international_sector_etf_map mas abajo), asi
# que VOX debe estar en el universo para poder descargarse y cotizar su IV.
etf_sectoriales = ["XLK", "XLV", "XLF", "XLE", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC", "VOX"]

etf_subsectoriales = [
    "SMH", "SOXX", "IGV", "CIBR", "HACK", "SKYY", "BOTZ", "ROBO", "BUG",
    "XBI", "IBB", "PPH", "IHI", "IHF", "ARKG",
    "KBE", "KRE", "KIE", "IAI", "FINX",
    "XOP", "AMLP", "MLPX", "OIH", "ICLN", "TAN", "FAN",
    "XRT", "XHB", "ITB", "PEJ", "ONLN", "CARZ",
    "MOO",
    "ITA", "PPA", "IYT", "PAVE",
    "GDX", "GDXJ", "LIT", "SLX", "COPX",
    "REM",
    "ARKW", "ARKF",
]

etf_geograficos = [
    "VWO", "EEM",
    "EFA", "VGK", "EZU",
    "AAXJ", "EWJ",
    "MCHI", "FXI", "INDA",
    "ILF", "EWZ",
    "VXUS", "ACWX", "VT", "FM",
]

etf_tickers = list(dict.fromkeys(etf_core + etf_sectoriales + etf_subsectoriales + etf_geograficos))

# ================================================
# COMMODITIES
# ================================================
commodity_tickers = ["SLV", "UNG"]

# ================================================
# TICKERS INTERNACIONALES
# ================================================
international_tickers_full = [
    "RY.TO", "SHOP.TO", "TD.TO", "BN.TO", "ENB.TO", "TRI.TO", "BNS.TO",
    "CP.TO", "CNQ.TO", "AEM.TO", "SU.TO", "TRP.TO", "WCN.TO", "FNV.TO",
    "SAP.TO", "SIE.DE", "DTE.DE", "ALV.DE", "MBG.DE", "IFX.DE", "BMW.DE",
    "DB1.DE", "DHL.DE", "DBK.DE", "MUV2.DE", "AZN.L", "HSBC", "ULVR.L",
    "BP", "GSK.L", "RIO.L", "BATS.L", "GLEN.L", "DGE.L", "NG.L", "MC.PA",
    "TTE.PA", "SAN.PA", "OR.PA", "SU.PA", "AI.PA", "BNP.PA", "RMS.PA",
    "CS.PA", "SAF.PA", "CAP.PA", "ITX.MC", "IBE.MC", "BBVA.MC", "SAN.MC",
    "7203.T", "6758.T", "6861.T", "8306.T", "9984.T", "6367.T", "6098.T",
    "4063.T", "7974.T", "9432.T", "6501.T", "7267.T", "8316.T", "4568.T",
    "6902.T", "4502.T", "8031.T",
]

international_tickers = international_tickers_full[:n_top_int]

# ================================================
# MAPEO DE MONEDA POR SUFIJO DE TICKER + PARES FX (para conversion a USD)
# ================================================

fx_pairs = {
    "CAD": {"ticker": "CAD=X", "invert": True},
    "EUR": {"ticker": "EURUSD=X", "invert": False},
    "GBP": {"ticker": "GBPUSD=X", "invert": False},
    "JPY": {"ticker": "JPY=X", "invert": True},
}
ticker_currency_by_suffix = {
    ".TO": "CAD",
    ".DE": "EUR",
    ".PA": "EUR",
    ".MC": "EUR",
    ".L": "GBP",
    ".T": "JPY",
}
ticker_currency_override = {
    "HSBC": "GBP",
    "BP": "GBP",
}


def get_currency_for_ticker(ticker):
    if ticker in ticker_currency_override:
        return ticker_currency_override[ticker]
    for suf, cur in ticker_currency_by_suffix.items():
        if ticker.endswith(suf):
            return cur
    return "USD"

# ================================================
# COMBINAR Y LIMPIAR
# ================================================
print("\nCombinando y limpiando tickers...")

sp500_tickers_clean = list(dict.fromkeys(t.upper() for t in sp500_tickers))
nasdaq_tickers_clean = list(dict.fromkeys(t.upper() for t in nasdaq_tickers))
etf_tickers_clean = list(dict.fromkeys(t.upper() for t in etf_tickers))
commodity_tickers_clean = list(dict.fromkeys(t.upper() for t in commodity_tickers))
international_tickers_clean = list(dict.fromkeys(t.upper() for t in international_tickers))

all_tickers_raw = (sp500_tickers_clean + nasdaq_tickers_clean + etf_tickers_clean
                    + commodity_tickers_clean + international_tickers_clean)

all_tickers = list(dict.fromkeys(all_tickers_raw))

all_tickers = [
    t for t in all_tickers
    if not re.search(r"\^|\$", t) and 1 <= len(t) <= 5 and not re.match(r"^[0-9]", t) and t != ""
]

if len(all_tickers) < target_total_tickers:
    shortage = target_total_tickers - len(all_tickers)
    if len(international_tickers_full) > n_top_int:
        additional_int_available = international_tickers_full[n_top_int:]
        additional_int_unique = [t for t in dict.fromkeys(x.upper() for x in additional_int_available)
                                  if t not in all_tickers]
        if additional_int_unique:
            to_add_int = additional_int_unique[: shortage]
            all_tickers = all_tickers + to_add_int
            shortage -= len(to_add_int)
    if shortage > 0 and nasdaq_tbl_clean is not None and len(nasdaq_tbl_clean) > len(nasdaq_tickers):
        additional_nasdaq_available = nasdaq_tbl_clean["symbol"].iloc[len(nasdaq_tickers):].unique().tolist()
        additional_nasdaq_unique = [t for t in additional_nasdaq_available if t not in all_tickers]
        if additional_nasdaq_unique:
            to_add_nasdaq = additional_nasdaq_unique[: shortage]
            all_tickers = all_tickers + to_add_nasdaq

all_tickers = list(dict.fromkeys(all_tickers))
print(f"Total tickers FINAL (unicos): {len(all_tickers)}")


# ================================================
# DESCARGA DE PRECIOS (helper generico yfinance)
# ================================================
def download_fx_prices(start, end, period="1mo"):
    """
    Descarga los pares FX necesarios a la misma granularidad que los precios
    que se van a convertir, y devuelve un dict {moneda: Series} ya en
    USD-por-unidad-de-moneda-local (invertido cuando corresponde).
    """
    fx_prices = {}
    for cur, info in fx_pairs.items():
        try:
            hist = yf.Ticker(info["ticker"]).history(start=start, end=end, interval=period, auto_adjust=True)
            if hist is None or hist.empty:
                continue
            s = hist["Close"].copy()
            s.index = pd.to_datetime(s.index).tz_localize(None)
            if info["invert"]:
                s = 1 / s
            fx_prices[cur] = s.sort_index()
        except Exception:
            pass
    return fx_prices


def download_period_returns(tickers, start, end, period="1mo", fx_prices=None):
    """
    Descarga precios ajustados y calcula retornos por periodo (mensual o semanal).
    Si fx_prices trae la serie de la moneda de un ticker no-USD, el precio se
    convierte a USD (ultimo valor FX conocido, ffill) antes de calcular el
    retorno - asi el retorno queda en base USD real, no en moneda local.
    Devuelve DataFrame largo: columns = ['symbol', 'date', 'return', 'adjusted']
    """
    all_rows = []
    n = len(tickers)
    n_convertidos = 0
    for i, tk in enumerate(tickers, start=1):
        try:
            hist = yf.Ticker(tk).history(start=start, end=end, interval=period, auto_adjust=True)
            if hist is None or hist.empty:
                continue
            hist = hist[["Close"]].rename(columns={"Close": "adjusted"})
            hist.index = pd.to_datetime(hist.index).tz_localize(None)

            cur = get_currency_for_ticker(tk)
            if cur != "USD" and fx_prices is not None and cur in fx_prices:
                fx_aligned = fx_prices[cur].reindex(
                    fx_prices[cur].index.union(hist.index)
                ).sort_index().ffill().reindex(hist.index)
                hist["adjusted"] = hist["adjusted"] * fx_aligned
                n_convertidos += 1

            hist["return"] = hist["adjusted"].pct_change()
            hist = hist.dropna(subset=["return"])
            hist["symbol"] = tk
            hist = hist.reset_index().rename(columns={"Date": "date", "index": "date"})
            hist["date"] = pd.to_datetime(hist["date"]).dt.tz_localize(None)
            all_rows.append(hist[["symbol", "date", "return", "adjusted"]])
        except Exception:
            pass
        if i % 25 == 0 or i == n:
            print(f"   Descargados {i}/{n} tickers...")
    if n_convertidos > 0:
        print(f"   OK Tickers convertidos a USD en esta descarga: {n_convertidos}")
    if not all_rows:
        return pd.DataFrame(columns=["symbol", "date", "return", "adjusted"])
    return pd.concat(all_rows, ignore_index=True)


start_date = f"{min(target_years)}-01-01"
end_date = f"{max(target_years) + 1}-12-31"

print("Descargando pares FX (mensual) para conversion a USD...")
fx_prices_monthly = download_fx_prices(start_date, end_date, period="1mo")
print(f"   OK Pares FX mensuales disponibles: {', '.join(fx_prices_monthly.keys()) if fx_prices_monthly else '(ninguno)'}")

print("Descargando datos de precios MENSUALES...")
df_prices_monthly = download_period_returns(all_tickers, start_date, end_date, period="1mo",
                                             fx_prices=fx_prices_monthly)
df_prices_monthly = df_prices_monthly.rename(columns={"return": "monthly_return"})

df_prices = df_prices_monthly.copy()
df_prices["year"] = df_prices["date"].dt.year
df_prices = df_prices[df_prices["year"].isin(target_years)]
if lookback_months is not None:
    cutoff_date = df_prices["date"].max() - relativedelta(months=lookback_months - 1)
    df_prices = df_prices[df_prices["date"] >= cutoff_date]
df_prices = df_prices[["symbol", "date", "monthly_return"]]

print(f"Meses disponibles en entrenamiento: {df_prices['date'].nunique()}")
print(f"Tickers con al menos 1 observacion: {df_prices['symbol'].nunique()}")

# === BENCHMARK (SPY) ===
print("Procesando benchmark (SPY)...")
benchmark_prices = download_period_returns(["SPY"], start_date, end_date, period="1mo",
                                            fx_prices=fx_prices_monthly)
benchmark_prices = benchmark_prices.rename(columns={"return": "benchmark_return"})[["date", "benchmark_return"]]
if lookback_months is not None:
    cutoff_date = benchmark_prices["date"].max() - relativedelta(months=lookback_months - 1)
    benchmark_prices = benchmark_prices[benchmark_prices["date"] >= cutoff_date]

# === RETORNOS SEMANALES PARA FILTRO ESTACIONAL ===
print(f"Descargando pares FX (semanal) para conversion a USD...")
fx_prices_weekly = download_fx_prices(start_date, end_date, period="1wk")

print(f"Descargando retornos SEMANALES para filtro estacional ({horizon_label})...")
df_weekly_seasonal = download_period_returns(all_tickers, start_date, end_date, period="1wk",
                                              fx_prices=fx_prices_weekly)
df_weekly_seasonal = df_weekly_seasonal.rename(columns={"return": "weekly_return"})
df_weekly_seasonal = df_weekly_seasonal[df_weekly_seasonal["date"].dt.month.isin(rebalance_months)]

seasonal_sd_stats = (
    df_weekly_seasonal.groupby("symbol")["weekly_return"]
    .agg(seasonal_sd="std", seasonal_n_obs="count")
    .reset_index()
)
seasonal_sd_stats = seasonal_sd_stats[
    (seasonal_sd_stats["seasonal_n_obs"] >= seasonal_min_weeks)
    & (seasonal_sd_stats["seasonal_sd"] > 0)
    & seasonal_sd_stats["seasonal_sd"].notna()
].sort_values("seasonal_sd").reset_index(drop=True)

print(f"Tickers con SD estacional calculable: {len(seasonal_sd_stats)}")

# ================================================
# ESTADISTICAS DESCRIPTIVAS
# ================================================
rf_rate_period = rf_rate_monthly

summary_stats = (
    df_prices.groupby("symbol")["monthly_return"]
    .agg(mean_return="mean", sd_return="std", n_obs="count")
    .reset_index()
)
summary_stats = summary_stats[(summary_stats["n_obs"] >= min_observations) & (summary_stats["sd_return"] > 0)].copy()
summary_stats["sharpe_ratio"] = (summary_stats["mean_return"] - rf_rate_monthly) / summary_stats["sd_return"]
summary_stats["data_quality_penalty"] = (summary_stats["n_obs"] / ideal_observations).clip(upper=1.0)

if len(summary_stats) > 0:
    tickers_total = len(summary_stats)
    obs_median = summary_stats["n_obs"].median()
    pct_ideal = (summary_stats["n_obs"] >= ideal_observations).mean() * 100
    pct_acceptable = (summary_stats["n_obs"] >= min_observations).mean() * 100
    print(f"Tickers totales: {tickers_total} | Obs mediana: {obs_median:.0f} | "
          f"Ideal: {pct_ideal:.1f}% | Minimo: {pct_acceptable:.1f}%")

    plt.figure(figsize=(8, 5))
    plt.hist(summary_stats["n_obs"], bins=20, color="steelblue", alpha=0.7, edgecolor="white")
    plt.axvline(ideal_observations, color="darkgreen", linestyle="--", linewidth=1.5)
    plt.axvline(min_observations, color="orange", linestyle="--", linewidth=1.5)
    plt.title("Distribucion de Observaciones por Ticker")
    plt.xlabel("Numero de Observaciones")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.show()
else:
    raise RuntimeError("No hay tickers con datos suficientes. Ajusta los parametros.")

# ================================================
# CORRELACIONES
# ================================================
returns_wide = df_prices.pivot_table(index="date", columns="symbol", values="monthly_return")
cor_matrix_full = returns_wide.corr(min_periods=1)

# El indice y las columnas de cor_matrix_full ambos heredan el nombre "symbol"
# del pivot_table; hay que renombrarlos a algo distinto antes de apilar,
# o reset_index() intenta crear dos columnas llamadas "symbol" y falla.
cor_matrix_stack = cor_matrix_full.rename_axis(index="Var1", columns="Var2")
corr_pairs = cor_matrix_stack.stack().reset_index()
corr_pairs.columns = ["Var1", "Var2", "Freq"]
corr_pairs = corr_pairs[corr_pairs["Var1"].astype(str) < corr_pairs["Var2"].astype(str)]

avg_cor_by_ticker = (
    corr_pairs.assign(abs_freq=corr_pairs["Freq"].abs())
    .groupby("Var1")["abs_freq"].mean().reset_index()
    .rename(columns={"Var1": "symbol", "abs_freq": "avg_cor"})
)

# ================================================
# FAMA-FRENCH
# ================================================
print("Descargando datos Fama-French...")
try:
    ff_raw = pdr.DataReader("F-F_Research_Data_Factors", "famafrench",
                             start=f"{min(target_years)}-01-01",
                             end=f"{max(target_years)}-12-31")[0]
    ff_raw = ff_raw.copy()
    ff_raw.index = ff_raw.index.to_timestamp()
    ff_raw = ff_raw.rename(columns={"Mkt-RF": "Mkt-RF", "SMB": "SMB", "HML": "HML"})
    ff_raw = ff_raw[ff_raw.index.year.isin(target_years)]
    if lookback_months is not None:
        cutoff_date = ff_raw.index.max() - relativedelta(months=lookback_months - 1)
        ff_raw = ff_raw[ff_raw.index >= cutoff_date]
    ff_raw = ff_raw / 100.0
    ff_data = ff_raw[["Mkt-RF", "SMB", "HML"]]
except Exception as e:
    print(f"Error con Fama-French ({e}). Continuando sin ellos.")
    ff_data = None

# ================================================
# BETAS FF3
# ================================================
if ff_data is not None:
    # Alinear por mes (fin de mes) para el merge
    ff_data_m = ff_data.copy()
    ff_data_m.index = ff_data_m.index.to_period("M")
    ff_data_m.index.name = "period"
    ff_data_m_reset = ff_data_m.reset_index()

    df_prices_m = df_prices.copy()
    df_prices_m["period"] = df_prices_m["date"].dt.to_period("M")
    df_prices_m = df_prices_m.merge(benchmark_prices.assign(period=benchmark_prices["date"].dt.to_period("M")),
                                     on="period", how="left", suffixes=("", "_bench"))
    df_prices_m["excess_return"] = df_prices_m["monthly_return"] - rf_rate_period
    df_prices_m = df_prices_m[df_prices_m["period"].isin(ff_data_m_reset["period"])]

    rows = []
    if len(df_prices_m) > 0:
        for sym, g in df_prices_m.groupby("symbol"):
            if len(g) < 3:
                continue
            merged = g.merge(ff_data_m_reset, on="period", how="inner")
            if len(merged) < 3:
                continue
            try:
                X = sm.add_constant(merged[["Mkt-RF", "SMB", "HML"]])
                y = merged["excess_return"]
                model = sm.OLS(y, X, missing="drop").fit()
                rows.append(dict(symbol=sym, beta_mkt=model.params.get("Mkt-RF", np.nan),
                                  beta_smb=model.params.get("SMB", np.nan),
                                  beta_hml=model.params.get("HML", np.nan)))
            except Exception:
                rows.append(dict(symbol=sym, beta_mkt=np.nan, beta_smb=np.nan, beta_hml=np.nan))

    ff_stats = pd.DataFrame(rows, columns=["symbol", "beta_mkt", "beta_smb", "beta_hml"])
    if len(ff_stats) > 0:
        market_premium = ff_data["Mkt-RF"].mean()
        smb_premium = ff_data["SMB"].mean()
        hml_premium = ff_data["HML"].mean()
        ff_stats["ff_expected_return"] = (rf_rate_period
                                           + ff_stats["beta_mkt"] * market_premium
                                           + ff_stats["beta_smb"] * smb_premium
                                           + ff_stats["beta_hml"] * hml_premium)
        print(f"Betas calculados para {len(ff_stats)} activos")
    else:
        ff_stats = pd.DataFrame(columns=["symbol", "beta_mkt", "beta_smb", "beta_hml", "ff_expected_return"])
else:
    ff_stats = pd.DataFrame(columns=["symbol", "beta_mkt", "beta_smb", "beta_hml", "ff_expected_return"])

# ================================================
# COMBINAR METRICAS
# ================================================
combined_stats = summary_stats.merge(avg_cor_by_ticker, on="symbol", how="left")
combined_stats = combined_stats.merge(
    ff_stats[["symbol", "beta_mkt", "beta_smb", "beta_hml", "ff_expected_return"]], on="symbol", how="left"
)
combined_stats["avg_cor"] = combined_stats["avg_cor"].fillna(combined_stats["avg_cor"].median())
combined_stats["ff_expected_return"] = combined_stats["ff_expected_return"].fillna(combined_stats["mean_return"])
combined_stats["adjusted_return"] = np.where(
    combined_stats["ff_expected_return"].notna(),
    (combined_stats["mean_return"] + combined_stats["ff_expected_return"]) / 2,
    combined_stats["mean_return"],
)
combined_stats["sharpe_ratio_adjusted"] = (
    (combined_stats["adjusted_return"] - rf_rate_period) / combined_stats["sd_return"]
)
combined_stats["is_etf_commodity"] = combined_stats["symbol"].isin(etf_tickers + commodity_tickers)

if combined_stats is None or len(combined_stats) == 0:
    raise RuntimeError("combined_stats no se creo correctamente")
print(f"combined_stats creado: {len(combined_stats)} activos")


# ================================================
# SELECCION CONJUNTA VIA QUBO/ISING
# ================================================
def qubo_energy(idx_sel, h, J):
    idx_sel = list(idx_sel)
    if len(idx_sel) < 2:
        return -np.sum(h[idx_sel])
    sub_J = J[np.ix_(idx_sel, idx_sel)]
    return -np.sum(h[idx_sel]) + np.sum(np.triu(sub_J, k=1))


def select_candidates_qubo(symbols, h_dict, cor_matrix, n_select, weight_decorr_):
    n_pool = len(symbols)
    if n_select >= n_pool:
        return symbols

    J = weight_decorr_ * cor_matrix.loc[symbols, symbols].abs().values
    np.fill_diagonal(J, 0)
    h = np.array([h_dict[s] for s in symbols])

    n_combos = math.comb(n_pool, n_select)
    print(f"   QUBO: pool={n_pool}, seleccionar={n_select}, combinaciones={n_combos:.3e}")

    if n_combos <= qubo_exact_threshold:
        print("   Combinaciones dentro del umbral - resolviendo por fuerza bruta (optimo exacto)")
        best_energy = np.inf
        best_combo = None
        combos = itertools.combinations(range(n_pool), n_select)
        count = 0
        for combo in combos:
            e = qubo_energy(combo, h, J)
            if e < best_energy:
                best_energy = e
                best_combo = combo
            count += 1
            if count % 5000 == 0:
                print(f"      ... {count}/{n_combos:.0f} combinaciones evaluadas")
        print(f"   Optimo exacto - energia={best_energy:.4f}")
        return [symbols[i] for i in best_combo]

    print(f"   Combinaciones exceden el umbral - resolviendo via Simulated Annealing ({qubo_sa_iterations} iteraciones)")
    rng = np.random.default_rng(seed)
    idx_current = list(rng.choice(n_pool, n_select, replace=False))
    e_current = qubo_energy(idx_current, h, J)
    best_idx = idx_current.copy()
    best_energy = e_current
    e_inicial = e_current

    temp_init = max(abs(e_current), 1e-6)
    temp_final = temp_init * 1e-4

    idx_set = set(idx_current)
    for it in range(1, qubo_sa_iterations + 1):
        temp = temp_init * (temp_final / temp_init) ** (it / qubo_sa_iterations)
        out_pos = idx_current[rng.integers(0, len(idx_current))]
        remaining = [x for x in range(n_pool) if x not in idx_set]
        in_pos = remaining[rng.integers(0, len(remaining))]
        idx_prop = [x for x in idx_current if x != out_pos] + [in_pos]
        e_prop = qubo_energy(idx_prop, h, J)
        delta_e = e_prop - e_current

        if delta_e < 0 or rng.random() < np.exp(-delta_e / temp):
            idx_current = idx_prop
            idx_set = set(idx_current)
            e_current = e_prop
            if e_current < best_energy:
                best_energy = e_current
                best_idx = idx_current.copy()

    print(f"   SA completado - energia inicial={e_inicial:.4f} | mejor energia={best_energy:.4f}")
    return [symbols[i] for i in best_idx]


def select_optimal_candidates(df, n_candidates):
    df_filtered = df[(df["n_obs"] >= min_observations) & (df["sd_return"] > 0)
                      & df["sharpe_ratio_adjusted"].notna() & np.isfinite(df["sharpe_ratio_adjusted"])].copy()

    sd_threshold = df_filtered["sd_return"].quantile(volatility_percentile)
    cor_threshold = df_filtered["avg_cor"].quantile(correlation_percentile)

    if correlation_order == 1:
        df_candidates = df_filtered[(df_filtered["sd_return"] <= sd_threshold) & (df_filtered["avg_cor"] >= cor_threshold)]
    else:
        df_candidates = df_filtered[(df_filtered["sd_return"] <= sd_threshold) & (df_filtered["avg_cor"] <= cor_threshold)]

    if len(df_candidates) < n_candidates * 0.5:
        sd_threshold = df_filtered["sd_return"].quantile(0.75)
        cor_threshold = df_filtered["avg_cor"].quantile(0.80)
        if correlation_order == 1:
            df_candidates = df_filtered[(df_filtered["sd_return"] <= sd_threshold) & (df_filtered["avg_cor"] >= cor_threshold)]
        else:
            df_candidates = df_filtered[(df_filtered["sd_return"] <= sd_threshold) & (df_filtered["avg_cor"] <= cor_threshold)]

    df_candidates = df_candidates.copy()
    sr = df_candidates["sharpe_ratio_adjusted"]
    sd = df_candidates["sd_return"]
    sharpe_norm = (sr - sr.min()) / (sr.max() - sr.min())
    vol_norm = 1 - (sd - sd.min()) / (sd.max() - sd.min())
    df_candidates["sharpe_norm"] = sharpe_norm.fillna(0.5)
    df_candidates["vol_norm"] = vol_norm.fillna(0.5)
    df_candidates["h_score"] = (
        (weight_sharpe * df_candidates["sharpe_norm"] + weight_low_vol * df_candidates["vol_norm"])
        * df_candidates["data_quality_penalty"]
    )
    df_candidates = df_candidates.sort_values("h_score", ascending=False)

    n_to_select = min(n_candidates, len(df_candidates))

    h_vec = dict(zip(df_candidates["symbol"], df_candidates["h_score"]))
    selected = select_candidates_qubo(
        symbols=list(df_candidates["symbol"]),
        h_dict=h_vec,
        cor_matrix=cor_matrix_full,
        n_select=n_to_select,
        weight_decorr_=weight_decorr,
    )

    print("\nTop 10 activos seleccionados (via QUBO):")
    top10 = (df_candidates[df_candidates["symbol"].isin(selected)]
             .sort_values("h_score", ascending=False)
             .head(10)[["symbol", "sharpe_ratio_adjusted", "sd_return", "avg_cor", "n_obs", "h_score"]])
    print(top10.to_string(index=False))
    return selected


ticker_candidates = select_optimal_candidates(combined_stats, n_pre_seasonal)
print(f"\nPool pre-filtro estacional: {len(ticker_candidates)} candidatos\n")

# ================================================
# FILTRO ESTACIONAL (primera pasada, informativa)
# ================================================
print(f"Aplicando filtro estacional ({horizon_label})...")

seasonal_candidates = (
    seasonal_sd_stats[seasonal_sd_stats["symbol"].isin(ticker_candidates)]
    .sort_values("seasonal_sd").head(n_divers_candidates)
)

n_con_seasonal = len(seasonal_candidates)
print(f"Candidatos previos: {len(ticker_candidates)} | Con SD estacional: {n_con_seasonal} | "
      f"Seleccionados: {n_con_seasonal}")

if n_con_seasonal < n_divers_candidates * 0.5:
    print(f"Advertencia: solo {n_con_seasonal} tickers sobrevivieron el filtro estacional.")

seasonal_display = seasonal_candidates.merge(
    combined_stats[["symbol", "sharpe_ratio_adjusted", "sd_return", "n_obs"]], on="symbol", how="left"
)
print(seasonal_display.to_string(index=False))

ticker_candidates = seasonal_candidates["symbol"].tolist()
print(f"\nConjunto FINAL para optimizacion: {len(ticker_candidates)} tickers\n")

# Recalcular pool completo pre-estacional (se vuelve a usar el pool completo mas abajo,
# igual que en el script R original)
ticker_candidates = select_optimal_candidates(combined_stats, n_pre_seasonal)
print(f"\nPool pre-filtro estacional: {len(ticker_candidates)} candidatos\n")

# ================================================
# DATOS DE MERCADO VIA POLYGON (IV, GRIEGAS, SECTOR)
# ================================================
print("\nConsultando datos de mercado (Polygon) para tickers estadounidenses...")

target_dte_polygon = max(round(horizon_months * 30), 15)

us_candidates = [t for t in ticker_candidates if is_us_ticker(t)]
intl_candidates = [t for t in ticker_candidates if not is_us_ticker(t)]

print(f"  Tickers US: {len(us_candidates)} | Tickers internacionales (sin cobertura Polygon): {len(intl_candidates)}")

polygon_market_list = {}
for i, tk in enumerate(us_candidates, start=1):
    opt_tk = polygon_get_atm_option(tk, target_dte_polygon, polygon_dte_tol)
    polygon_market_list[tk] = {"symbol": tk, **opt_tk}
    if i % 10 == 0 or i == len(us_candidates):
        print(f"   Polygon IV: {i}/{len(us_candidates)}")

polygon_market_df = pd.DataFrame(list(polygon_market_list.values()))
if "ok" not in polygon_market_df.columns:
    polygon_market_df["ok"] = False

n_polygon_ok = polygon_market_df["ok"].fillna(False).sum()
print(f"  Datos de opciones obtenidos: {n_polygon_ok} de {len(us_candidates)} tickers US")

stock_us_candidates = [t for t in us_candidates if t not in (etf_tickers + commodity_tickers)]
print(f"  Clasificando sector de {len(stock_us_candidates)} acciones US (Polygon SIC)...")

sector_map = {}
for i, tk in enumerate(stock_us_candidates, start=1):
    sector_map[tk] = polygon_get_sector_etf(tk)
    if i % 10 == 0 or i == len(stock_us_candidates):
        print(f"   Sector: {i}/{len(stock_us_candidates)}")

n_sector_ok = sum(1 for v in sector_map.values() if v is not None)
print(f"  Sector identificado (Polygon SIC, US): {n_sector_ok} de {len(stock_us_candidates)} acciones")

# === MAPEO DE SECTOR PARA CANDIDATOS INTERNACIONALES (mapeo manual, mismos
# ETFs sectoriales ya presentes en etf_sectoriales - Polygon no tiene SIC
# para emisores no listados en EE.UU., asi que estos siempre quedaban sin
# sector y caian al promedio implicito global) ===
international_sector_etf_map = {
    # Canada (.TO)
    "RY.TO": "XLF", "SHOP.TO": "XLK", "TD.TO": "XLF", "BN.TO": "XLF",
    "ENB.TO": "XLE", "TRI.TO": "XLI", "BNS.TO": "XLF", "CP.TO": "XLI",
    "CNQ.TO": "XLE", "AEM.TO": "XLB", "SU.TO": "XLE", "TRP.TO": "XLE",
    "WCN.TO": "XLI", "FNV.TO": "XLB", "SAP.TO": "XLP",
    # Alemania (.DE)
    "SIE.DE": "XLI", "DTE.DE": "VOX", "ALV.DE": "XLF", "MBG.DE": "XLY",
    "IFX.DE": "XLK", "BMW.DE": "XLY", "DB1.DE": "XLF", "DHL.DE": "XLI",
    "DBK.DE": "XLF", "MUV2.DE": "XLF",
    # Reino Unido (.L, y HSBC/BP sin sufijo)
    "AZN.L": "XLV", "HSBC": "XLF", "ULVR.L": "XLP", "BP": "XLE",
    "GSK.L": "XLV", "RIO.L": "XLB", "BATS.L": "XLP", "GLEN.L": "XLB",
    "DGE.L": "XLP", "NG.L": "XLU",
    # Francia (.PA)
    "MC.PA": "XLY", "TTE.PA": "XLE", "SAN.PA": "XLV", "OR.PA": "XLP",
    "SU.PA": "XLI", "AI.PA": "XLB", "BNP.PA": "XLF", "RMS.PA": "XLY",
    "CS.PA": "XLF", "SAF.PA": "XLI", "CAP.PA": "XLK",
    # Espana (.MC)
    "ITX.MC": "XLY", "IBE.MC": "XLU", "BBVA.MC": "XLF", "SAN.MC": "XLF",
    # Japon (.T)
    "7203.T": "XLY", "6758.T": "XLY", "6861.T": "XLK", "8306.T": "XLF",
    "9984.T": "VOX", "6367.T": "XLI", "6098.T": "XLI", "4063.T": "XLB",
    "7974.T": "VOX", "9432.T": "VOX", "6501.T": "XLI", "7267.T": "XLY",
    "8316.T": "XLF", "4568.T": "XLV", "6902.T": "XLY", "4502.T": "XLV",
    "8031.T": "XLI",
}

stock_intl_candidates = [t for t in intl_candidates if t not in (etf_tickers + commodity_tickers)]
for tk in stock_intl_candidates:
    sector_map[tk] = international_sector_etf_map.get(tk)

n_sector_intl_ok = sum(1 for t in stock_intl_candidates if sector_map.get(t) is not None)
print(f"  Sector identificado (mapeo manual, internacionales): {n_sector_intl_ok} de {len(stock_intl_candidates)} acciones")


print(f"  Llamadas HTTP intentadas en total: {polygon_diag['total_attempts']}")
print("  Diagnostico de llamadas a Polygon (status HTTP acumulado):")
for k, v in polygon_diag["status_counts"].items():
    print(f"     {k}: {v}")
print(f"  Respuestas 200 sin contrato dentro de la ventana strike/DTE: {polygon_diag['empty_match_count']}")
if polygon_diag["sample_errors"]:
    print("  Ejemplos de error (hasta 5, para identificar la causa real):")
    for e in polygon_diag["sample_errors"]:
        cuerpo = (e["body"] or "(sin cuerpo)")[:200] if e["body"] else "(sin cuerpo)"
        print(f"     status={e['status']} | {cuerpo}")

# ================================================
# FILTRO DELTA
# ================================================
if use_delta_filter:
    print(f"\nAplicando filtro Delta (delta_min={delta_min:.2f}, T={horizon_months / 12:.3f} anios)...")
    print("   Fuente: delta real de mercado (Polygon) para US, formula BS con vol historica para el resto")

    T_horizon = horizon_months / 12

    mu_for_delta = (
        df_prices[df_prices["symbol"].isin(ticker_candidates)]
        .groupby("symbol")["monthly_return"].mean().rename("mu_monthly").reset_index()
    )
    hist_vol_for_delta = (
        df_prices[df_prices["symbol"].isin(ticker_candidates)]
        .groupby("symbol")["monthly_return"].std().mul(math.sqrt(12)).rename("vol_hist").reset_index()
    )

    delta_rows = []
    poly_ok_set = set(polygon_market_df.loc[polygon_market_df["ok"] == True, "symbol"]) if "ok" in polygon_market_df.columns else set()

    for ticker in ticker_candidates:
        usa_polygon = ticker in poly_ok_set
        if usa_polygon:
            fila = polygon_market_df[polygon_market_df["symbol"] == ticker].iloc[0]
            delta_rows.append(dict(symbol=ticker, delta=fila["delta"], strike_mode="polygon_real", iv_used=fila["iv"]))
            continue

        mu_i_series = mu_for_delta.loc[mu_for_delta["symbol"] == ticker, "mu_monthly"]
        mu_i = mu_i_series.iloc[0] if len(mu_i_series) and not pd.isna(mu_i_series.iloc[0]) else 0.0

        iv_series = hist_vol_for_delta.loc[hist_vol_for_delta["symbol"] == ticker, "vol_hist"]
        iv = iv_series.iloc[0] if len(iv_series) and not pd.isna(iv_series.iloc[0]) and iv_series.iloc[0] > 0 else np.nan

        if pd.isna(iv):
            delta_rows.append(dict(symbol=ticker, delta=np.nan, strike_mode="sin_datos", iv_used=np.nan))
            continue

        if delta_strike_mode == "mu":
            drift_term = mu_i * horizon_months
        elif delta_strike_mode == "rf":
            drift_term = rf_rate * (horizon_months / 12)
        else:
            drift_term = 0

        d1 = (T_horizon * (rf_rate + iv ** 2 / 2) - drift_term) / (iv * math.sqrt(T_horizon))
        delta_i = norm.cdf(d1)
        delta_rows.append(dict(symbol=ticker, delta=delta_i, strike_mode=f"bs_{delta_strike_mode}", iv_used=iv))

    delta_df = pd.DataFrame(delta_rows)
    delta_named = dict(zip(delta_df["symbol"], delta_df["delta"]))

    n_delta_ok = delta_df["delta"].notna().sum()
    n_delta_na = delta_df["delta"].isna().sum()
    n_below = ((delta_df["delta"].notna()) & (delta_df["delta"] < delta_min)).sum()
    n_pass = ((delta_df["delta"].notna()) & (delta_df["delta"] >= delta_min)).sum()
    n_real = (delta_df["strike_mode"] == "polygon_real").sum()

    print(f"  Deltas de mercado real (Polygon): {n_real} | Deltas via formula BS "
          f"(modo '{delta_strike_mode}'): {(delta_df['strike_mode'] == f'bs_{delta_strike_mode}').sum()}")
    print(f"  Deltas calculados: {n_delta_ok} | Sin datos (se conservan): {n_delta_na}")
    print(f"  Descartados (delta < {delta_min:.2f}): {n_below}")
    print(f"  Superan el filtro (delta >= {delta_min:.2f}): {n_pass}")

    discarded_delta = delta_df[(delta_df["delta"].notna()) & (delta_df["delta"] < delta_min)].sort_values("delta")
    if len(discarded_delta) > 0:
        disp = discarded_delta.copy()
        disp["Delta"] = disp["delta"].map(lambda x: f"{x:.3f}")
        disp["IV_anual"] = disp["iv_used"].map(lambda x: f"{x * 100:.1f}%")
        print("\nActivos descartados por Delta insuficiente:")
        print(disp[["symbol", "Delta", "IV_anual", "strike_mode"]]
              .rename(columns={"symbol": "Symbol", "strike_mode": "Fuente"}).to_string(index=False))

    tickers_pass_delta = delta_df[(delta_df["delta"].isna()) | (delta_df["delta"] >= delta_min)]["symbol"].tolist()
    ticker_candidates = tickers_pass_delta
    print(f"\nTras filtro Delta (sobre pool pre-estacional): {len(ticker_candidates)} tickers disponibles "
          f"para el filtro estacional\n")
else:
    print("Filtro Delta desactivado\n")
    delta_named = {t: np.nan for t in ticker_candidates}

# ================================================
# FILTRO ESTACIONAL (segunda pasada, por umbral de ratio)
# ================================================
print(f"Aplicando filtro estacional ({horizon_label}) - exclusion por umbral (ratio max={seasonal_vol_ratio_max:.2f}x)...")

seasonal_ratio_stats = (
    seasonal_sd_stats[seasonal_sd_stats["symbol"].isin(ticker_candidates)]
    .merge(combined_stats[["symbol", "sharpe_ratio_adjusted", "sd_return", "n_obs"]], on="symbol", how="left")
)
seasonal_ratio_stats = seasonal_ratio_stats[seasonal_ratio_stats["sd_return"].notna() & (seasonal_ratio_stats["sd_return"] > 0)].copy()
seasonal_ratio_stats["seasonal_vol_anual"] = seasonal_ratio_stats["seasonal_sd"] * math.sqrt(52)
seasonal_ratio_stats["general_vol_anual"] = seasonal_ratio_stats["sd_return"] * math.sqrt(12)
seasonal_ratio_stats["seasonal_vol_ratio"] = (
    seasonal_ratio_stats["seasonal_vol_anual"] / seasonal_ratio_stats["general_vol_anual"]
)
seasonal_ratio_stats = seasonal_ratio_stats.sort_values("seasonal_vol_ratio")

n_pool_post_delta = len(ticker_candidates)
n_con_seasonal = len(seasonal_ratio_stats)
print(f"Candidatos previos (post-Delta): {n_pool_post_delta} | Con ratio estacional calculable: {n_con_seasonal}")

seasonal_pass = seasonal_ratio_stats[seasonal_ratio_stats["seasonal_vol_ratio"] <= seasonal_vol_ratio_max]

n_descartados = n_con_seasonal - len(seasonal_pass)
print(f"Descartados por umbral (ratio > {seasonal_vol_ratio_max:.2f}x): {n_descartados} | "
      f"Superan el filtro: {len(seasonal_pass)}")

if len(seasonal_pass) < seasonal_min_survivors:
    print(f"Advertencia: solo {len(seasonal_pass)} tickers superan el umbral - relajando hasta "
          f"el piso minimo ({seasonal_min_survivors})")
    seasonal_pass = seasonal_ratio_stats.head(min(seasonal_min_survivors, n_con_seasonal))

if len(seasonal_pass) > n_divers_candidates:
    print(f"Pool post-umbral ({len(seasonal_pass)}) supera el techo n_divers_candidates "
          f"({n_divers_candidates}) - recortando por ratio ascendente")
    seasonal_pass = seasonal_pass.sort_values("seasonal_vol_ratio").head(n_divers_candidates)

seasonal_candidates = seasonal_pass.copy()

disp = seasonal_candidates.copy()
disp["SD_Estacional"] = disp["seasonal_sd"].map(lambda x: f"{x:.4f}")
disp["SD_General"] = disp["sd_return"].map(lambda x: f"{x:.4f}")
disp["Ratio_Anualizado"] = disp["seasonal_vol_ratio"].map(lambda x: f"{x:.2f}x")
disp["Sharpe"] = disp["sharpe_ratio_adjusted"].map(lambda x: f"{x:.3f}")
print(disp.rename(columns={"symbol": "Symbol", "seasonal_n_obs": "Semanas", "n_obs": "Obs_Mensuales"})
      [["Symbol", "SD_Estacional", "Semanas", "SD_General", "Ratio_Anualizado", "Sharpe", "Obs_Mensuales"]]
      .to_string(index=False))

ticker_candidates = seasonal_candidates["symbol"].tolist()
print(f"\nConjunto FINAL para optimizacion: {len(ticker_candidates)} tickers\n")

# ================================================
# FILTRO MFIS (BKM) - cobertura anomala vs especulacion
# ================================================
print(f"\nAplicando filtro MFIS (Bakshi-Kapadia-Madan) (z_threshold={bkm_z_threshold:.2f}, "
      f"lookback={bkm_lookback_months} meses)...")

hoy_ts = pd.Timestamp(date.today())
sample_dates_bkm = pd.date_range(
    hoy_ts - relativedelta(months=bkm_lookback_months), hoy_ts - timedelta(days=7), freq=bkm_hist_sample_freq
)
print(f"   Fechas de muestreo historico: {len(sample_dates_bkm)} (freq={bkm_hist_sample_freq})")

reponer_pool_bkm = (
    seasonal_ratio_stats[~seasonal_ratio_stats["symbol"].isin(ticker_candidates)]
    .sort_values("seasonal_vol_ratio")["symbol"].tolist()
)

bkm_current_moments = {}


def evaluar_bkm_activo(ticker):
    mom_actual = bkm_get_current_moments(ticker, target_dte_polygon, polygon_dte_tol, rf_rate)
    bkm_current_moments[ticker] = mom_actual
    if not mom_actual["ok"]:
        return dict(symbol=ticker, decision="mantener", motivo="sin_mfis_actual", z=np.nan)

    try:
        start_sk = (hoy_ts - relativedelta(months=bkm_lookback_months + 1)).strftime("%Y-%m-%d")
        end_sk = hoy_ts.strftime("%Y-%m-%d")
        hist = yf.Ticker(ticker).history(start=start_sk, end=end_sk, auto_adjust=True)
        if hist is None or hist.empty or len(hist) < 20:
            return dict(symbol=ticker, decision="mantener", motivo="sin_precio_historico", z=np.nan)
        spot_series_tk = hist[["Close"]].rename(columns={"Close": "adjusted"}).reset_index()
        spot_series_tk = spot_series_tk.rename(columns={"Date": "date"})
        spot_series_tk["date"] = pd.to_datetime(spot_series_tk["date"]).dt.tz_localize(None)
    except Exception:
        return dict(symbol=ticker, decision="mantener", motivo="sin_precio_historico", z=np.nan)

    hist_mfis = bkm_reconstruct_mfis_history(
        ticker, spot_series_tk, sample_dates_bkm, target_dte_polygon, bkm_hist_moneyness_grid, rf_rate
    )
    hist_mfis_validos = hist_mfis[~np.isnan(hist_mfis)]

    if len(hist_mfis_validos) < 8:
        return dict(symbol=ticker, decision="mantener", motivo="muestra_insuficiente", z=np.nan)

    media_hist = np.mean(hist_mfis_validos)
    sd_hist = np.std(hist_mfis_validos, ddof=1)
    if sd_hist <= 0 or np.isnan(sd_hist):
        return dict(symbol=ticker, decision="mantener", motivo="sd_hist_invalida", z=np.nan)

    z = (mom_actual["mfis"] - media_hist) / sd_hist

    if np.isfinite(z) and z > bkm_z_threshold:
        return dict(symbol=ticker, decision="descartar", motivo="cobertura_anomala", z=z)
    else:
        return dict(symbol=ticker, decision="mantener", motivo="normal_o_especulativo", z=z)


full_set = list(ticker_candidates)
resultados_log = []
i_reponer = 0
por_evaluar = list(ticker_candidates)
ronda_bkm = 1

while True:
    print(f"   [Ronda {ronda_bkm}] Evaluando MFIS de {len(por_evaluar)} activo(s) "
          f"({bkm_max_workers} en paralelo)...")

    resultados = []
    with ThreadPoolExecutor(max_workers=bkm_max_workers) as executor:
        for k, r in enumerate(executor.map(evaluar_bkm_activo, por_evaluar), start=1):
            resultados.append(r)
            if k % 10 == 0 or k == len(por_evaluar):
                print(f"      ... {k}/{len(por_evaluar)}")

    resultados_log.extend(resultados)
    descartados = [r for r in resultados if r["decision"] == "descartar"]

    if not descartados:
        break

    reemplazos_nuevos = []
    for d in descartados:
        print(f"   Descartado por MFIS anomalo: {d['symbol']} (z={d['z']:.2f})")
        full_set = [s for s in full_set if s != d["symbol"]]

        if len(full_set) + len(reemplazos_nuevos) < bkm_min_survivors and i_reponer < len(reponer_pool_bkm):
            reemplazo = reponer_pool_bkm[i_reponer]
            i_reponer += 1
            print(f"   Reponiendo con: {reemplazo} (pool estacional validado)")
            reemplazos_nuevos.append(reemplazo)

    if not reemplazos_nuevos:
        break
    full_set = list(dict.fromkeys(full_set + reemplazos_nuevos))
    por_evaluar = list(dict.fromkeys(reemplazos_nuevos))
    ronda_bkm += 1

bkm_log_df = pd.DataFrame(resultados_log)
n_descartados_bkm = (bkm_log_df["decision"] == "descartar").sum() if len(bkm_log_df) else 0
print(f"\n   Evaluados: {len(bkm_log_df)} | Descartados por MFIS: {n_descartados_bkm} | Repuestos: {i_reponer}")

if len(full_set) < bkm_min_survivors:
    print(f"   Advertencia: solo {len(full_set)} tickers tras filtro BKM "
          f"(piso deseado: {bkm_min_survivors}) - pool de reposicion agotado")

ticker_candidates = full_set
print(f"\nConjunto FINAL tras filtro BKM (MFIS): {len(ticker_candidates)} tickers\n")

for tk in ticker_candidates:
    if tk not in bkm_current_moments or not bkm_current_moments[tk]["ok"]:
        bkm_current_moments[tk] = bkm_get_current_moments(tk, target_dte_polygon, polygon_dte_tol, rf_rate)

# ================================================
# CONSTRUCCION DE MATRIZ DE RETORNOS
# ================================================
df_wide = (
    df_prices[df_prices["symbol"].isin(ticker_candidates)]
    .pivot_table(index="date", columns="symbol", values="monthly_return")
    .sort_index()
    .dropna()
)

dates = df_wide.index
ticker_candidates = list(df_wide.columns)
df_xts = df_wide.copy()

print(f"Matriz de retornos: {df_xts.shape[0]} observaciones x {df_xts.shape[1]} activos")

# === MEDIA ===
assets = ticker_candidates
mu = df_xts[assets].mean().values
mu = np.where(np.isfinite(mu), mu, 0.0)
n_assets = len(assets)

# ================================================
# MATRIZ DE COVARIANZA: SHRINKAGE MFIV (BKM) + HISTORICA
# ================================================
iv_rank_low = 0.20
iv_rank_high = 0.80
alpha_min = 0.20
alpha_max = 0.80
iv_cap_multiplier = iv_outlier_multiplier

print("\nEstimando covarianza robusta: Shrinkage MFIV (BKM) + Historica...")
print(f"   Tenor objetivo: {target_dte_polygon} dias (consistente con horizonte de rebalanceo) | "
      f"Cap IV: {iv_cap_multiplier:.1f}x vol historica")

T_bkm = target_dte_polygon / 365

iv_assets_implied = {a: np.nan for a in assets}
for a in assets:
    mom_a = bkm_current_moments.get(a)
    if mom_a is not None and mom_a.get("ok") and mom_a.get("mfiv", np.nan) > 0:
        vol_hist_a = df_prices.loc[df_prices["symbol"] == a, "monthly_return"].std() * math.sqrt(12)
        vol_mfiv_annual_a = math.sqrt(mom_a["mfiv"] / T_bkm)
        iv_cap_a = vol_hist_a * iv_cap_multiplier
        if np.isfinite(iv_cap_a) and vol_mfiv_annual_a <= iv_cap_a:
            iv_assets_implied[a] = vol_mfiv_annual_a

n_iv_ok = sum(1 for v in iv_assets_implied.values() if not pd.isna(v))
n_iv_na = len(assets) - n_iv_ok
print(f"   MFIV (BKM) validas: {n_iv_ok} | Sin datos (fallback historico): {n_iv_na}")

sd_hist_annual = df_xts[assets].std().values * math.sqrt(12)
iv_final = np.array([iv_assets_implied[a] if not pd.isna(iv_assets_implied[a]) else sd_hist_annual[i]
                      for i, a in enumerate(assets)])
iv_final_dict = dict(zip(assets, iv_final))

print(f"   IV final - Min: {iv_final.min() * 100:.1f}% | Mediana: {np.median(iv_final) * 100:.1f}% | "
      f"Max: {iv_final.max() * 100:.1f}%")

mom_spy = bkm_get_current_moments("SPY", target_dte_polygon, polygon_dte_tol, rf_rate)
if mom_spy["ok"] and mom_spy["mfiv"] > 0:
    iv_spy_implied = math.sqrt(mom_spy["mfiv"] / T_bkm)
else:
    print("   Sin MFIV de mercado para SPY - usando vol historica mensual")
    iv_spy_implied = benchmark_prices["benchmark_return"].std() * math.sqrt(12)


def get_iv_rank(ticker, iv_current):
    try:
        hist_ret = df_prices.loc[df_prices["symbol"] == ticker, "monthly_return"].values
        if len(hist_ret) < 12:
            return 0.5
        n_roll = 3
        vol_history = []
        for i in range(n_roll - 1, len(hist_ret)):
            window = hist_ret[i - n_roll + 1: i + 1]
            vol_history.append(np.std(window, ddof=1) * math.sqrt(12))
        vol_history = np.array(vol_history)
        vol_history = vol_history[np.isfinite(vol_history) & (vol_history > 0)]
        if len(vol_history) < 6:
            return 0.5
        return float(np.mean(vol_history <= iv_current))
    except Exception:
        return 0.5


print("   Calculando IV Percentile Rank...")
iv_ranks = {t: get_iv_rank(t, iv_final_dict[t]) for t in assets}

has_iv_real = np.array([not pd.isna(iv_assets_implied[a]) for a in assets])

alpha_per_asset = np.zeros(n_assets)
for i, t in enumerate(assets):
    rank = iv_ranks[t]
    if not has_iv_real[i]:
        alpha_per_asset[i] = alpha_min
        continue
    if iv_rank_low <= rank <= iv_rank_high:
        center_dist = 1 - 2 * abs(rank - 0.5)
        alpha_per_asset[i] = alpha_min + (alpha_max - alpha_min) * center_dist
    else:
        alpha_per_asset[i] = alpha_min

alpha_global = np.nanmean(alpha_per_asset)
alpha_global = max(alpha_min, min(alpha_max, alpha_global))

n_alpha_full = int((alpha_per_asset > alpha_min).sum())
n_alpha_min = int((alpha_per_asset <= alpha_min).sum())
print(f"   IV Rank - Min: {min(iv_ranks.values()):.2f} | Mediana: {np.median(list(iv_ranks.values())):.2f} | "
      f"Max: {max(iv_ranks.values()):.2f}")
print(f"   Alpha por activo - Pleno (>{alpha_min:.2f}): {n_alpha_full} | Minimo ({alpha_min:.2f}): "
      f"{n_alpha_min} (sin IV real o IV extrema)")
print(f"   Alpha shrinkage global (promedio): {alpha_global:.3f} ({alpha_global * 100:.0f}% implied / "
      f"{(1 - alpha_global) * 100:.0f}% historica)")

Sigma_hist = df_xts[assets].cov().values * 12

# --- Correlacion implicita global via dispersion de SPY ---
w_disp = np.repeat(1 / n_assets, n_assets)
sigma_i = iv_final

var_spy_impl = iv_spy_implied ** 2
weighted_var_i = np.sum(w_disp ** 2 * sigma_i ** 2)
sigma_total_sq = (np.sum(w_disp * sigma_i)) ** 2

R_hist_full = df_xts[assets].corr().values

if sigma_total_sq > weighted_var_i and sigma_total_sq > 0:
    rho_implied_avg = (var_spy_impl - weighted_var_i) / (sigma_total_sq - weighted_var_i)
    rho_implied_avg = max(-0.999, min(0.999, rho_implied_avg))
    print(f"   Correlacion promedio implicita (dispersion SPY): {rho_implied_avg:.4f}")
else:
    tril_idx = np.tril_indices(n_assets, k=-1)
    rho_implied_avg = np.nanmean(R_hist_full[tril_idx])
    rho_implied_avg = max(-0.999, min(0.999, rho_implied_avg))
    print(f"   Correlacion promedio implicita (fallback historico): {rho_implied_avg:.4f}")

tril_idx = np.tril_indices(n_assets, k=-1)
rho_hist_avg_ref = np.nanmean(np.abs(R_hist_full[tril_idx]))

# --- Correlacion implicita SECTORIAL (ETFs sectoriales SPDR + VOX), via MFIV (BKM) ---
print("   Calculando correlacion implicita por sector (MFIV de ETFs sectoriales)...")

rho_sector_lookup = {etf: np.nan for etf in etf_sectoriales}
sector_etf_iv = {etf: np.nan for etf in etf_sectoriales}
for etf in etf_sectoriales:
    mom_etf = bkm_get_current_moments(etf, target_dte_polygon, polygon_dte_tol, rf_rate)
    if mom_etf["ok"] and mom_etf["mfiv"] > 0:
        sector_etf_iv[etf] = math.sqrt(mom_etf["mfiv"] / T_bkm)

for etf in etf_sectoriales:
    stocks_sector = [t for t in assets if sector_map.get(t) == etf]
    if len(stocks_sector) < 2 or pd.isna(sector_etf_iv.get(etf)):
        continue

    idxs = [assets.index(s) for s in stocks_sector]
    sigma_sector = iv_final[idxs]
    var_etf_impl = sector_etf_iv[etf] ** 2
    w_disp_sector = np.repeat(1 / len(stocks_sector), len(stocks_sector))
    weighted_var_s = np.sum(w_disp_sector ** 2 * sigma_sector ** 2)
    sigma_total_s = (np.sum(w_disp_sector * sigma_sector)) ** 2

    if sigma_total_s > weighted_var_s and sigma_total_s > 0:
        rho_sec = (var_etf_impl - weighted_var_s) / (sigma_total_s - weighted_var_s)
        rho_sector_lookup[etf] = max(-0.999, min(0.999, rho_sec))
        print(f"      {etf:<5}: rho implicita = {rho_sector_lookup[etf]:.4f} ({len(stocks_sector)} acciones)")

n_sectores_ok = sum(1 for v in rho_sector_lookup.values() if not pd.isna(v))
print(f"   Correlacion sectorial calculada para {n_sectores_ok} de {len(etf_sectoriales)} sectores")

# Construir R_implied
R_implied = np.eye(n_assets)
for i in range(n_assets):
    for j in range(n_assets):
        if i == j:
            continue
        rho_hist_ij = R_hist_full[i, j]
        if pd.isna(rho_hist_ij) or not np.isfinite(rho_hist_ij):
            rho_hist_ij = rho_implied_avg

        sector_i = sector_map.get(assets[i])
        sector_j = sector_map.get(assets[j])

        if sector_i is not None and sector_j is not None and sector_i == sector_j and not pd.isna(rho_sector_lookup.get(sector_i)):
            rho_ancla = rho_sector_lookup[sector_i]
        else:
            rho_ancla = rho_implied_avg

        scale_factor = (rho_hist_ij / rho_hist_avg_ref) if rho_hist_avg_ref > 0 else 1.0
        rho_ij_impl = max(-0.999, min(0.999, rho_ancla * scale_factor))
        R_implied[i, j] = rho_ij_impl

R_implied = (R_implied + R_implied.T) / 2
np.fill_diagonal(R_implied, 1.0)
eig_vals_R, eig_vecs_R = np.linalg.eigh(R_implied)
if np.any(eig_vals_R < 0):
    eig_vals_R = np.maximum(eig_vals_R, 1e-8)
    R_implied = eig_vecs_R @ np.diag(eig_vals_R) @ eig_vecs_R.T
    D_norm = np.diag(1 / np.sqrt(np.diag(R_implied)))
    R_implied = D_norm @ R_implied @ D_norm
    np.fill_diagonal(R_implied, 1.0)

D_implied = np.diag(sigma_i)
Sigma_impl = D_implied @ R_implied @ D_implied

Sigma_blend = np.zeros((n_assets, n_assets))
for i in range(n_assets):
    for j in range(n_assets):
        a_ij = (alpha_per_asset[i] + alpha_per_asset[j]) / 2
        Sigma_blend[i, j] = a_ij * Sigma_impl[i, j] + (1 - a_ij) * Sigma_hist[i, j]

cov_mat = Sigma_blend / 12

vol_impl_pct = np.sqrt(np.diag(Sigma_impl)) * 100
vol_hist_pct = np.sqrt(np.diag(Sigma_hist)) * 100
vol_blend_pct = np.sqrt(np.diag(Sigma_blend)) * 100

print("\n   Volatilidades anualizadas (promedio):")
print(f"      Implied (MFIV): {vol_impl_pct.mean():.1f}%")
print(f"      Historica: {vol_hist_pct.mean():.1f}%")
print(f"      Blend:     {vol_blend_pct.mean():.1f}% (a_global={alpha_global:.2f})")

eig_vals = np.linalg.eigvalsh(cov_mat)
print(f"   Eigenvalue minimo (Sigma blend): {eig_vals.min():.6f}")
if not np.all(eig_vals >= -1e-8):
    print("   Sigma blend no es PSD - aplicando correccion Higham...")
    eig_vals_full, eig_vecs_full = np.linalg.eigh(cov_mat)
    eig_vals_full = np.maximum(eig_vals_full, 1e-8)
    cov_mat = eig_vecs_full @ np.diag(eig_vals_full) @ eig_vecs_full.T

if np.any(~np.isfinite(cov_mat)):
    print("   Limpiando valores no finitos en cov_mat...")
    cov_mat = np.nan_to_num(cov_mat, nan=0.0, posinf=0.0, neginf=0.0)

print(f"   cov_mat blend construida: {cov_mat.shape[0]} x {cov_mat.shape[1]} activos")
print(f"   Rango MFIV anualizada (post-cap): {vol_impl_pct.min():.1f}% - {vol_impl_pct.max():.1f}%")
tril_R = R_implied[np.tril_indices(n_assets, k=-1)]
print(f"   Rango correlaciones implicitas (off-diagonal R_implied): {tril_R.min():.3f} - {tril_R.max():.3f}")

# ================================================
# OPTIMIZACION CUADRATICA (quadprog)
# ================================================
print(f"\nEjecutando optimizacion cuadratica (quadprog) con lambda={lambda_:.2f}...")

etf_commodity_assets = [i for i, a in enumerate(assets) if a in (etf_tickers + commodity_tickers)]
stock_assets = [i for i, a in enumerate(assets) if a not in (etf_tickers + commodity_tickers)]

canada_assets = [i for i, a in enumerate(assets) if a.endswith(".TO")]
europe_assets = [i for i, a in enumerate(assets) if re.search(r"\.(DE|L|PA|MC)$", a)]
japan_assets = [i for i, a in enumerate(assets) if a.endswith(".T")]

geo_union = set(canada_assets) | set(europe_assets) | set(japan_assets)
print(f"   Grupos geograficos - CA: {len(canada_assets)} | EU: {len(europe_assets)} | "
      f"JP: {len(japan_assets)} | US/ETF: {n_assets - len(geo_union)}")

n = n_assets
Dmat = cov_mat + np.eye(n) * 1e-8

# --- Delta como ponderador de retorno esperado en el vector dvec ---
delta_aligned = np.array([delta_named.get(a, np.nan) for a in assets])

delta_valid = delta_aligned[~pd.isna(delta_aligned)]
if len(delta_valid) >= 2:
    d_min_obs = delta_valid.min()
    d_max_obs = delta_valid.max()
    if d_max_obs > d_min_obs:
        delta_scaled = (delta_aligned - d_min_obs) / (d_max_obs - d_min_obs) * (1 - delta_min) + delta_min
    else:
        delta_scaled = np.repeat(1.0, len(delta_aligned))
else:
    delta_scaled = np.repeat(1.0, len(delta_aligned))
delta_scaled = np.where(np.isnan(delta_scaled), 1.0, delta_scaled)

mu_delta_adjusted = mu * delta_scaled

# Penalizacion por riesgo de cola (MFIS, MFIK de BKM):
# mu_final = mu_delta_adjusted - theta1*MFIS - theta2*MFIK
mfis_aligned = np.array([bkm_current_moments.get(a, {}).get("mfis", np.nan) for a in assets])
mfik_aligned = np.array([bkm_current_moments.get(a, {}).get("mfik", np.nan) for a in assets])
mfis_aligned = np.where(np.isnan(mfis_aligned), 0.0, mfis_aligned)
mfik_aligned = np.where(np.isnan(mfik_aligned), 3.0, mfik_aligned)
mu_final = mu_delta_adjusted - tail_penalty_theta1 * mfis_aligned - tail_penalty_theta2 * mfik_aligned

dvec = mu_final / lambda_

print(f"   Delta scaling aplicado - activos con delta real: {int((~pd.isna(delta_aligned)).sum())} | "
      f"fallback (sin penalizacion): {int(pd.isna(delta_aligned).sum())}")
print(f"   Penalizacion de cola (BKM) aplicada - theta1={tail_penalty_theta1:.3f}, theta2={tail_penalty_theta2:.3f}")

A_cols = []
b_vals = []

A_cols.append(np.ones(n))
b_vals.append(1.0)

for i in range(n):
    v = np.zeros(n)
    v[i] = 1
    A_cols.append(v)
    b_vals.append(0.0)

for i in range(n):
    v = np.zeros(n)
    v[i] = -1
    A_cols.append(v)
    b_vals.append(-max_weight)

if len(etf_commodity_assets) > 0 and len(stock_assets) > 0:
    etf_lo = max(0, pct_etf_deseado - pct_etf_tolerancia)
    etf_hi = min(1, pct_etf_deseado + pct_etf_tolerancia)
    stk_lo = 1 - etf_hi
    stk_hi = 1 - etf_lo

    v_etf_lo = np.zeros(n); v_etf_lo[etf_commodity_assets] = 1
    v_etf_hi = np.zeros(n); v_etf_hi[etf_commodity_assets] = -1
    v_stk_lo = np.zeros(n); v_stk_lo[stock_assets] = 1
    v_stk_hi = np.zeros(n); v_stk_hi[stock_assets] = -1
    A_cols += [v_etf_lo, v_etf_hi, v_stk_lo, v_stk_hi]
    b_vals += [etf_lo, -etf_hi, stk_lo, -stk_hi]

    print(f"   Restriccion ETFs: {etf_lo * 100:.0f}% - {etf_hi * 100:.0f}% del portafolio "
          f"(objetivo {pct_etf_deseado * 100:.0f}% +/- {pct_etf_tolerancia * 100:.0f}%)")

geo_groups = {"Canada": canada_assets, "Europa": europe_assets, "Japon": japan_assets}

for region_name, idx_region in geo_groups.items():
    if len(idx_region) > 0:
        v_geo = np.zeros(n)
        v_geo[idx_region] = -1
        A_cols.append(v_geo)
        b_vals.append(-max_region_weight)
        print(f"   Restriccion {region_name}: max {max_region_weight * 100:.0f}% del portafolio")

Amat = np.column_stack(A_cols)
bvec = np.array(b_vals)
meq = 1

try:
    sol = quadprog.solve_qp(Dmat, dvec, Amat, bvec, meq)
except Exception as e:
    print(f"  quadprog fallo ({e}) - reintentando con nugget mayor...")
    sol = quadprog.solve_qp(cov_mat + np.eye(n) * 1e-6, dvec, Amat, bvec, meq)

weights_opt = sol[0]
weights_opt = np.maximum(weights_opt, 0)
weights_opt = weights_opt / weights_opt.sum()
weights_opt = pd.Series(weights_opt, index=assets)

print("  Optimizacion completada")
print(f"  Activos con peso > 1%: {(weights_opt > 0.01).sum()}")
print(f"  Peso maximo: {weights_opt.max() * 100:.2f}% ({weights_opt.idxmax()})")

top_assets_idx = weights_opt.sort_values(ascending=False).head(min(10, n)).index.tolist()
print("\n  Top activos - efecto delta scaling y penalizacion de cola en retorno esperado:")
print(f"  {'Ticker':<8}  {'Peso%':>8}  {'mu%':>8}  {'delta':>8}  {'MFIS':>8}  {'MFIK':>8}  {'mu_final%':>10}")
print("  " + "-" * 68)
for t_i in top_assets_idx:
    i = assets.index(t_i)
    d_i = f"{delta_aligned[i]:.3f}" if not np.isnan(delta_aligned[i]) else "fallbk"
    print(f"  {t_i:<8}  {weights_opt[t_i] * 100:7.2f}%  {mu[i] * 100:7.3f}%  {d_i:>7}  "
          f"{mfis_aligned[i]:8.3f}  {mfik_aligned[i]:8.3f}  {mu_final[i] * 100:9.3f}%")

for region_name, idx_region in geo_groups.items():
    if len(idx_region) > 0:
        print(f"  Peso {region_name}: {weights_opt.iloc[idx_region].sum() * 100:.1f}%")

# ================================================
# RESULTADOS
# ================================================
w_vec = weights_opt.values
ret_opt = float(np.sum(w_vec * mu))
sd_opt = float(np.sqrt(w_vec @ cov_mat @ w_vec))
sharpe_opt = (ret_opt - rf_rate_period) / sd_opt
utility_opt = ret_opt - (lambda_ / 2) * (sd_opt ** 2)


def portfolio_returns_series(returns_df, weights_series):
    aligned = returns_df[weights_series.index].dropna()
    return aligned.dot(weights_series)


portfolio_returns_full = portfolio_returns_series(df_xts[ticker_candidates], weights_opt)
var_parametric = ret_opt - norm.ppf(0.95) * sd_opt
q05 = portfolio_returns_full.quantile(0.05)
cvar_95 = portfolio_returns_full[portfolio_returns_full <= q05].mean()

# ================================================
# VaR/CVaR prospectivos ajustados por Cornish-Fisher (BKM)
# Cornish-Fisher (1937): z_cf = z_a + (z_a^2-1)/6*S + (z_a^3-3z_a)/24*K_exc - (2z_a^3-5z_a)/36*S^2
# ES modificado (Boudt, Peterson & Croux, 2008):
#   MES_a = -phi(z_a)/a * (1 + S/6*z_a^2 + K_exc/24*(z_a^3-3z_a) - S^2/36*(2z_a^3-5z_a))
# ================================================
alpha_cf = 1 - cornish_fisher_confidence
z_a = norm.ppf(alpha_cf)

w_full = weights_opt.reindex(assets).fillna(0.0).values
mfis_w = np.array([bkm_current_moments.get(a, {}).get("mfis", np.nan) for a in assets])
mfik_w = np.array([bkm_current_moments.get(a, {}).get("mfik", np.nan) for a in assets])

mask_bkm_ok = np.isfinite(mfis_w) & np.isfinite(mfik_w)
w_bkm_norm = w_full[mask_bkm_ok]
if w_bkm_norm.sum() > 0:
    w_bkm_norm = w_bkm_norm / w_bkm_norm.sum()
    port_skew = float(np.sum(w_bkm_norm * mfis_w[mask_bkm_ok]))
    port_kurt_exc = float(np.sum(w_bkm_norm * mfik_w[mask_bkm_ok])) - 3.0
    port_sd_cf = sd_opt  # sigma diversificada (cov_mat), no el promedio ponderado de MFIV individuales

    z_cf = (z_a + (z_a ** 2 - 1) / 6 * port_skew
            + (z_a ** 3 - 3 * z_a) / 24 * port_kurt_exc
            - (2 * z_a ** 3 - 5 * z_a) / 36 * port_skew ** 2)
    var_cf = ret_opt + z_cf * port_sd_cf

    mes_a = (norm.pdf(z_a) / alpha_cf) * (
        1 + port_skew / 6 * z_a ** 2
        + port_kurt_exc / 24 * (z_a ** 3 - 3 * z_a)
        - port_skew ** 2 / 36 * (2 * z_a ** 3 - 5 * z_a)
    )
    cvar_cf = ret_opt - mes_a * port_sd_cf
else:
    var_cf, cvar_cf = np.nan, np.nan
    print("   Advertencia: sin cobertura BKM suficiente en el portafolio final - VaR/CVaR Cornish-Fisher = NaN")

benchmark_series = benchmark_prices.set_index("date")["benchmark_return"]
df_with_bench = df_xts[ticker_candidates].join(benchmark_series.rename("benchmark"), how="inner").dropna()
ticker_cols_in_bench = [c for c in df_with_bench.columns if c != "benchmark"]
benchmark_aligned = df_with_bench["benchmark"]
portfolio_returns_aligned = df_with_bench[ticker_cols_in_bench].dot(weights_opt[ticker_cols_in_bench])
tracking_error = (portfolio_returns_aligned - benchmark_aligned).std()
relative_returns = portfolio_returns_aligned - benchmark_aligned
relative_var = relative_returns.mean() - norm.ppf(0.95) * relative_returns.std()

pesos = (
    pd.DataFrame({"symbol": weights_opt.index, "weight": weights_opt.values})
    .query("weight > 0.01")
    .sort_values("weight", ascending=False)
    .reset_index(drop=True)
)

annualization_factor = 12

downside_returns = portfolio_returns_full.values - rf_rate_monthly
downside_neg = downside_returns[downside_returns < 0]
downside_dev = np.sqrt(np.mean(downside_neg ** 2)) if len(downside_neg) else np.nan
sortino_opt = (ret_opt - rf_rate_monthly) / downside_dev

# ================================================
# PORTAFOLIO FINAL
# ================================================
print("\n" + "=" * 60)
print("PORTAFOLIO OPTIMO - PONDERACIONES FINALES")
print("=" * 60)
print(f"    Entrenado con: {min(target_years)}-{max(target_years)} (retornos mensuales)")
print(f"    Mes de ejecucion: {horizon_label} | lambda = {lambda_:.2f}\n")

print(f"  {'Ticker':<8}  {'Peso':>8}")
print("  " + "-" * 50)
for _, row in pesos.iterrows():
    barra = "#" * round(row["weight"] * 100 / 2)
    print(f"  {row['symbol']:<8}  {row['weight'] * 100:6.2f}%  {barra}")
print("  " + "-" * 50)
print(f"  {'TOTAL':<8}  {pesos['weight'].sum() * 100:6.2f}%\n")
print(f"  Activos en portafolio: {len(pesos)}")
print(f"  Peso maximo: {pesos['weight'].max() * 100:.2f}% ({pesos.iloc[0]['symbol']})")
print(f"  Concentracion top 5: {pesos['weight'].head(5).sum() * 100:.2f}%")
print("=" * 60 + "\n")

print(f"\n=== METRICAS DEL PORTAFOLIO ({min(target_years)}-{max(target_years)}, base mensual) ===")
print(f"  Mes de ejecucion: {horizon_label}")
print(f"  Lambda: {lambda_:.2f}")
print(f"  Retorno Esperado mensual: {ret_opt * 100:.4f}%")
print(f"  Volatilidad mensual:      {sd_opt * 100:.4f}%")
print(f"  Sharpe Ratio (mensual):   {sharpe_opt:.4f}")
print(f"  Sortino Ratio (mensual):  {sortino_opt:.4f}")
print(f"  Utilidad Cuadratica:      {utility_opt:.6f}")
print(f"  VaR (95%, Normal):        {var_parametric * 100:.4f}%")
print(f"  CVaR (95%, Historico):    {cvar_95 * 100:.4f}%")
print(f"  VaR Cornish-Fisher ({cornish_fisher_confidence * 100:.0f}%): {var_cf * 100:.4f}%")
print(f"  CVaR Cornish-Fisher ({cornish_fisher_confidence * 100:.0f}%): {cvar_cf * 100:.4f}%")
print(f"  Tracking Error:           {tracking_error * 100:.4f}%")
print(f"  Relative VaR (95%):      {relative_var * 100:.4f}%")

print("\n=== METRICAS ANUALIZADAS (x12) ===")
print(f"  Retorno anual:     {ret_opt * annualization_factor * 100:.2f}%")
print(f"  Volatilidad anual: {sd_opt * math.sqrt(annualization_factor) * 100:.2f}%")

# ================================================
# ATRIBUCION DE RIESGO POR GRIEGAS
# ================================================
print("\n=== ATRIBUCION DE RIESGO POR GRIEGAS (BLACK-SCHOLES) ===")

griegas_df = pd.DataFrame({"symbol": assets, "weight": weights_opt.reindex(assets).values})
if len(polygon_market_df):
    griegas_df = griegas_df.merge(
        polygon_market_df[["symbol", "delta", "gamma", "vega", "theta", "ok"]], on="symbol", how="left"
    )
else:
    for col in ["delta", "gamma", "vega", "theta", "ok"]:
        griegas_df[col] = np.nan

delta_portfolio = np.nansum(weights_opt.reindex(assets).values * delta_aligned)
print(f"  Delta de portafolio (ponderado, incl. fallback BS): {delta_portfolio:.4f}")

peso_con_griegas = griegas_df.loc[griegas_df["gamma"].notna(), "weight"].sum()

if peso_con_griegas > 0:
    vega_portfolio = np.nansum(griegas_df["weight"] * griegas_df["vega"])
    gamma_portfolio = np.nansum(griegas_df["weight"] * griegas_df["gamma"])
    print(f"  Vega de portafolio:  {vega_portfolio:.4f}  (sobre {peso_con_griegas * 100:.1f}% del peso con datos de opciones)")
    print(f"  Gamma de portafolio: {gamma_portfolio:.6f} (sobre {peso_con_griegas * 100:.1f}% del peso con datos de opciones)")
else:
    print("  Sin cobertura suficiente de opciones para reportar Vega/Gamma agregados")

top_griegas = griegas_df[griegas_df["weight"] > 0.01].sort_values("weight", ascending=False).copy()
top_griegas["Peso"] = top_griegas["weight"].map(lambda x: f"{x * 100:.2f}%")
top_griegas["Delta"] = top_griegas["delta"].map(lambda x: "fallback" if pd.isna(x) else f"{x:.3f}")
top_griegas["Gamma"] = top_griegas["gamma"].map(lambda x: "-" if pd.isna(x) else f"{x:.5f}")
top_griegas["Vega"] = top_griegas["vega"].map(lambda x: "-" if pd.isna(x) else f"{x:.3f}")

print("\n  Griegas por activo (peso > 1%):")
print(top_griegas.rename(columns={"symbol": "Symbol"})[["Symbol", "Peso", "Delta", "Gamma", "Vega"]].to_string(index=False))

# ================================================
# FRONTERA EFICIENTE
# ================================================
print("\nGenerando frontera eficiente...")


def solve_qp_portfolio(lambda_val):
    dv = mu_final / lambda_val
    Dm = cov_mat + np.eye(n) * 1e-8
    try:
        s = quadprog.solve_qp(Dm, dv, Amat, bvec, meq)
        w = np.maximum(s[0], 0)
        w = w / w.sum()
        return pd.Series(w, index=assets)
    except Exception:
        return None


rng = np.random.default_rng(seed)
random_weights_mat = np.zeros((n_sim, n))
valid_count = 0
attempt = 0

while valid_count < n_sim and attempt < n_sim * 10:
    attempt += 1

    if attempt % 2 == 0:
        n_active = rng.integers(4, min(10, n) + 1)
        idx = rng.choice(n, n_active, replace=False)
        w_raw = np.zeros(n)
        w_raw[idx] = rng.uniform(size=n_active)
        w_raw = w_raw / w_raw.sum()
    else:
        w_raw = rng.uniform(size=n)
        w_raw = w_raw / w_raw.sum()

    w_raw = np.minimum(w_raw, max_weight)
    if w_raw.sum() == 0:
        continue
    w_raw = w_raw / w_raw.sum()

    if np.all(w_raw >= 0) and abs(w_raw.sum() - 1) < 1e-6:
        random_weights_mat[valid_count, :] = w_raw
        valid_count += 1

random_weights_mat = random_weights_mat[:valid_count, :]

returns_vals = random_weights_mat @ mu
risk_vals = np.sqrt(np.einsum("ij,jk,ik->i", random_weights_mat, cov_mat, random_weights_mat))
utility_vals = returns_vals - (lambda_ / 2) * (risk_vals ** 2)

frontier_df = pd.DataFrame({"risk": risk_vals, "ret": returns_vals, "utility": utility_vals}).dropna()

fv = frontier_df.sort_values("risk").reset_index(drop=True)
efficient_points = fv.iloc[[0]].copy()
running_max_ret = efficient_points["ret"].max()
for i in range(1, len(fv)):
    if fv.loc[i, "ret"] > running_max_ret:
        efficient_points = pd.concat([efficient_points, fv.iloc[[i]]], ignore_index=True)
        running_max_ret = fv.loc[i, "ret"]

opt_point = pd.DataFrame({"risk": [sd_opt], "ret": [ret_opt], "utility": [utility_opt]})

x_max = max(risk_vals.max(), sd_opt) * 1.15
y_min = min(returns_vals.min(), ret_opt) * 0.95
y_max = max(returns_vals.max(), ret_opt) * 1.10

fig, ax = plt.subplots(figsize=(9, 6))
sc = ax.scatter(frontier_df["risk"], frontier_df["ret"], c=frontier_df["utility"],
                cmap="RdYlGn", alpha=0.3, s=15)
plt.colorbar(sc, label=f"Utilidad\n(lambda={lambda_:.1f})")
ax.plot(efficient_points["risk"], efficient_points["ret"], color="darkgreen", linestyle="--", linewidth=1.5)
ax.scatter([opt_point["risk"][0]], [opt_point["ret"][0]], color="red", s=120, marker="D", zorder=5)
ax.annotate(f"Optimo\nU={opt_point['utility'][0]:.4f}", (opt_point["risk"][0], opt_point["ret"][0]),
            textcoords="offset points", xytext=(8, 8), fontsize=9, fontweight="bold")
ax.set_xlim(0, x_max)
ax.set_ylim(y_min, y_max)
ax.set_title("Frontera Eficiente - Utilidad Cuadratica", fontweight="bold", fontsize=14)
ax.set_xlabel(f"Riesgo ({horizon_months} meses)")
ax.set_ylabel(f"Retorno Esperado ({horizon_months} meses)")
plt.suptitle(f"Horizonte: {horizon_months} mes(es) ({horizon_label}) | lambda={lambda_:.2f} | "
             f"{min(target_years)}-{max(target_years)}", y=0.94, fontsize=10)
plt.tight_layout()
plt.show()

# ================================================
# COMPARACION DE LAMBDAS
# ================================================
print("\n" + "=" * 70)
print("ANALISIS COMPARATIVO: Portafolios por nivel de lambda")
print("=" * 70 + "\n")

lambda_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0]
results_comparison_rows = []

for lambda_test in lambda_values:
    print(f"  Optimizando lambda={lambda_test:.1f}... ", end="")
    w = solve_qp_portfolio(lambda_test)
    if w is not None and w.sum() > 0.9:
        wv = w.values
        ret = float(np.sum(wv * mu))
        vol = float(np.sqrt(wv @ cov_mat @ wv))
        sharpe = (ret - rf_rate_period) / vol
        utility = ret - (lambda_test / 2) * (vol ** 2)
        results_comparison_rows.append(dict(
            lambda_=lambda_test, retorno=ret * 100, volatilidad=vol * 100,
            sharpe=sharpe, utilidad=utility, n_activos=int((w > 0.01).sum()), max_peso=w.max() * 100
        ))
        print(f"R={ret * 100:.2f}% | sigma={vol * 100:.2f}% | U={utility:.4f}")
    else:
        print("No se encontro solucion feasible")

results_comparison = pd.DataFrame(results_comparison_rows)

if len(results_comparison) > 0:
    print("\n" + "=" * 70)
    print("RESULTADOS COMPARATIVOS")
    print("=" * 70 + "\n")
    disp = results_comparison.copy()
    for col in ["retorno", "volatilidad", "sharpe", "utilidad", "max_peso"]:
        disp[col] = disp[col].map(lambda x: f"{x:.4f}")
    print(disp.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(results_comparison["volatilidad"], results_comparison["retorno"],
            color="steelblue", alpha=0.6, linewidth=1.5)
    sc = ax.scatter(results_comparison["volatilidad"], results_comparison["retorno"],
                     c=results_comparison["utilidad"], cmap="RdYlGn",
                     s=results_comparison["lambda_"] * 15 + 40)
    for _, row in results_comparison.iterrows():
        ax.annotate(f"{row['lambda_']:.1f}", (row["volatilidad"], row["retorno"]),
                    textcoords="offset points", xytext=(6, 0), fontsize=8)
    plt.colorbar(sc, label="Utilidad")
    ax.set_title("Frontera de Portafolios por Nivel de Aversion al Riesgo", fontweight="bold", fontsize=13)
    ax.set_xlabel("Volatilidad (%)")
    ax.set_ylabel("Retorno Esperado (%)")
    plt.figtext(0.5, -0.02, "lambda bajo = Agresivo | lambda alto = Conservador", ha="center", fontsize=9)
    plt.suptitle(f"Horizonte: {horizon_months} mes(es) ({horizon_label}) | {min(target_years)}-{max(target_years)}",
                 y=0.96, fontsize=10)
    plt.tight_layout()
    plt.show()

    best_row = results_comparison.loc[results_comparison["utilidad"].idxmax()]
    print("\n" + "=" * 70)
    print("RECOMENDACION BASADA EN UTILIDAD MAXIMA")
    print("=" * 70 + "\n")
    print(f"  lambda optimo para estos datos: {best_row['lambda_']:.1f}")
    print(f"  - Retorno esperado: {best_row['retorno']:.2f}% ({best_row['retorno'] * (12 / horizon_months):.1f}% anual)")
    print(f"  - Volatilidad: {best_row['volatilidad']:.2f}% ({best_row['volatilidad'] * math.sqrt(12 / horizon_months):.1f}% anual)")
    print(f"  - Sharpe Ratio: {best_row['sharpe']:.3f}")
    print(f"  - Utilidad: {best_row['utilidad']:.4f} (maxima)")
    print(f"  - Activos en portafolio: {int(best_row['n_activos'])}")
    if abs(best_row["lambda_"] - lambda_) > 0.5:
        print(f"\n  El lambda actual ({lambda_:.1f}) difiere del optimo ({best_row['lambda_']:.1f})")
        print(f"     Considera ajustar lambda <- {best_row['lambda_']:.1f}")
    else:
        print(f"\n  El lambda actual ({lambda_:.1f}) esta cerca del optimo")

print("\nOptimizacion completada exitosamente")

# ================================================
# ANALISIS DE MDD
# ================================================
print("\n\n=== ANALISIS DE MAXIMUM DRAWDOWN DEL PORTAFOLIO ===")

tickers_portfolio = pesos["symbol"].tolist()
weights_dict = dict(zip(pesos["symbol"], pesos["weight"]))


def calc_mdd(returns_vector):
    returns_vector = np.asarray(returns_vector, dtype=float)
    valid = returns_vector[~np.isnan(returns_vector)]
    if len(valid) < 2:
        return np.nan
    cumulative_values = np.cumprod(1 + returns_vector)
    peak = np.maximum.accumulate(cumulative_values)
    drawdown = (cumulative_values - peak) / peak
    mdd = np.nanmin(drawdown)
    if not np.isfinite(mdd):
        return np.nan
    return mdd


portfolio_hist = df_prices_monthly[
    df_prices_monthly["symbol"].isin(tickers_portfolio)
    & (df_prices_monthly["date"].dt.year >= mdd_start_year)
    & (df_prices_monthly["date"].dt.year <= max(target_years))
]

print(f"  Datos historicos: {len(portfolio_hist)} observaciones | "
      f"{portfolio_hist['date'].min():%Y-%m} a {portfolio_hist['date'].max():%Y-%m}")

portfolio_wide = portfolio_hist.pivot_table(index="date", columns="symbol", values="monthly_return").sort_index()

rows = []
for dt_, row in portfolio_wide.iterrows():
    valid_cols = [t for t in tickers_portfolio if t in portfolio_wide.columns and not pd.isna(row[t])]
    available_tickers = len(valid_cols)
    if available_tickers > 0:
        valid_weights = np.array([weights_dict[t] for t in valid_cols])
        normalized_weights = valid_weights / valid_weights.sum()
        valid_returns = np.array([row[t] for t in valid_cols])
        portfolio_return = float(np.sum(valid_returns * normalized_weights))
    else:
        portfolio_return = np.nan
    rows.append(dict(date=dt_, year=dt_.year, portfolio_return=portfolio_return, available_tickers=available_tickers))

portfolio_returns_by_year = pd.DataFrame(rows)

valid_returns_mdd = portfolio_returns_by_year[
    portfolio_returns_by_year["portfolio_return"].notna() & np.isfinite(portfolio_returns_by_year["portfolio_return"])
]

if len(valid_returns_mdd) >= 2:
    cumulative_values = np.cumprod(1 + valid_returns_mdd["portfolio_return"].values)
    peak = np.maximum.accumulate(cumulative_values)
    drawdown = (cumulative_values - peak) / peak
    global_mdd = np.nanmin(drawdown)
    print(f"\n  MDD Global ({valid_returns_mdd['year'].min()}-{valid_returns_mdd['year'].max()}): {global_mdd * 100:.2f}%")

all_years = pd.DataFrame({"year": range(mdd_start_year, max(target_years) + 1)})

yearly_mdd_calculated = (
    valid_returns_mdd.groupby("year")
    .agg(mdd=("portfolio_return", calc_mdd), n_obs=("portfolio_return", "count"),
         avg_tickers=("available_tickers", "mean"))
    .reset_index()
)

yearly_mdd = all_years.merge(yearly_mdd_calculated, on="year", how="left")
yearly_mdd["has_data"] = yearly_mdd["mdd"].notna()
yearly_mdd["n_obs"] = yearly_mdd["n_obs"].fillna(0)

yearly_mdd_valid = yearly_mdd[yearly_mdd["has_data"] & np.isfinite(yearly_mdd["mdd"])]

if len(yearly_mdd_valid) >= 1:
    print("\n=== ESTADISTICAS DE MDD ===")

    if len(yearly_mdd_valid) >= 3:
        q1 = yearly_mdd_valid["mdd"].quantile(0.25)
        q3 = yearly_mdd_valid["mdd"].quantile(0.75)
        iqr = q3 - q1

        yearly_mdd_clean = yearly_mdd_valid[
            (yearly_mdd_valid["mdd"] >= (q1 - 1.5 * iqr)) & (yearly_mdd_valid["mdd"] <= (q3 + 1.5 * iqr))
        ]
        if len(yearly_mdd_clean) < 2:
            yearly_mdd_clean = yearly_mdd_valid

        median_mdd = yearly_mdd_clean["mdd"].median()
        p90_mdd = yearly_mdd_clean["mdd"].quantile(0.90)
        mean_mdd = yearly_mdd_clean["mdd"].mean()
        min_mdd = yearly_mdd_clean["mdd"].min()
        max_mdd = yearly_mdd_clean["mdd"].max()

        print(f"  Peor escenario historico:    {min_mdd * 100:.2f}%")
        print(f"  Escenario Conservador (P90): {p90_mdd * 100:.2f}%")
        print(f"  Escenario Tipico (Mediana):  {median_mdd * 100:.2f}%")
        print(f"  Promedio:                    {mean_mdd * 100:.2f}%")
        print(f"  Mejor escenario historico:   {max_mdd * 100:.2f}%")
    else:
        mean_mdd = yearly_mdd_valid["mdd"].mean()
        min_mdd = yearly_mdd_valid["mdd"].min()
        max_mdd = yearly_mdd_valid["mdd"].max()
        median_mdd = mean_mdd
        p90_mdd = max_mdd
        print(f"  Peor escenario:   {max_mdd * 100:.2f}%")
        print(f"  Promedio:         {mean_mdd * 100:.2f}%")
        print(f"  Mejor escenario:  {min_mdd * 100:.2f}%")

    last_year_data = yearly_mdd_valid.sort_values("year", ascending=False)
    if len(last_year_data) > 0:
        print(f"  MDD mas reciente ({int(last_year_data.iloc[0]['year'])}): {last_year_data.iloc[0]['mdd'] * 100:.2f}%")

    plot_data = yearly_mdd.copy()
    plot_data["mdd_pct"] = np.where(plot_data["has_data"], plot_data["mdd"] * 100, np.nan)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(plot_data["year"], plot_data["mdd_pct"], color="darkred", linewidth=1.5)
    colors = plot_data["has_data"].map({True: "darkred", False: "gray"})
    ax.scatter(plot_data["year"], plot_data["mdd_pct"], c=colors, s=40)

    if len(yearly_mdd_valid) >= 3:
        ax.axhline(median_mdd * 100, linestyle="--", color="blue", linewidth=1.2)
        ax.axhline(p90_mdd * 100, linestyle="--", color="orange", linewidth=1.2)
        ax.text(yearly_mdd["year"].min() + 2, median_mdd * 100 - 0.3, f"Mediana: {median_mdd * 100:.2f}%",
                color="blue", fontsize=9)
        ax.text(yearly_mdd["year"].min() + 2, p90_mdd * 100 - 0.3, f"P90: {p90_mdd * 100:.2f}%",
                color="orange", fontsize=9)

    ax.set_title("Maximum Drawdown Historico del Portafolio", fontweight="bold", fontsize=14)
    ax.set_xlabel("Ano")
    ax.set_ylabel("MDD (%)")
    ax.set_xticks(range(mdd_start_year, max(target_years) + 1, 2))
    plt.xticks(rotation=45)
    plt.suptitle(f"Horizonte: {horizon_months} mes(es) ({horizon_label}) | MDD: {mdd_start_year}-{max(target_years)} | "
                 f"{int(yearly_mdd['has_data'].sum())} anos con datos", y=0.96, fontsize=10)
    plt.tight_layout()
    plt.show()

    yearly_summary = yearly_mdd.sort_values("year", ascending=False).copy()
    yearly_summary["MDD"] = np.where(yearly_summary["has_data"],
                                      yearly_summary["mdd"].map(lambda x: f"{x * 100:.2f}%"), "Sin datos")
    yearly_summary["Obs"] = np.where(yearly_summary["has_data"], yearly_summary["n_obs"].astype(int).astype(str), "-")
    yearly_summary["Tickers_Prom"] = np.where(yearly_summary["has_data"],
                                               yearly_summary["avg_tickers"].map(lambda x: f"{x:.1f}"), "-")
    yearly_summary = yearly_summary.rename(columns={"year": "Anio"})[["Anio", "MDD", "Obs", "Tickers_Prom"]]
    print(yearly_summary.to_string(index=False))

print("\nAnalisis completado exitosamente!")

# ================================================
# RESUMEN EJECUTIVO
# ================================================
print("\n" + "=" * 60)
print("RESUMEN EJECUTIVO FINAL")
print("=" * 60)
print(f"   Periodo de entrenamiento: {min(target_years)}-{max(target_years)}")
print(f"   Mes de ejecucion: {horizon_label}")
print(f"   Activos en portafolio: {len(tickers_portfolio)}")
print(f"   Retorno esperado mensual: {ret_opt * 100:.2f}%")
print(f"   Retorno anualizado:       {ret_opt * annualization_factor * 100:.2f}%")
print(f"   Volatilidad mensual:      {sd_opt * 100:.2f}%")
print(f"   Volatilidad anualizada:   {sd_opt * math.sqrt(annualization_factor) * 100:.2f}%")
print(f"   Sharpe Ratio:             {sharpe_opt:.4f}")
print(f"   Sortino Ratio:            {sortino_opt:.4f}")
print(f"   VaR Cornish-Fisher ({cornish_fisher_confidence * 100:.0f}%): {var_cf * 100:.4f}%")
print(f"   CVaR Cornish-Fisher ({cornish_fisher_confidence * 100:.0f}%): {cvar_cf * 100:.4f}%")
if "median_mdd" in dir():
    print(f"   MDD tipico historico:     {median_mdd * 100:.2f}%")

print(f"\n--- PORTAFOLIO A EJECUTAR EN {MONTH_NAME[max(rebalance_months)].upper()} ---\n")
for _, row in pesos.iterrows():
    print(f"   {row['symbol']:<8}  {row['weight'] * 100:.2f}%")
print("=" * 60)

# ================================================
# CALIDAD DE DATOS EN PORTAFOLIO FINAL
# ================================================
print("\n=== CALIDAD DE DATOS EN PORTAFOLIO FINAL ===")

portfolio_data_quality = (
    combined_stats[combined_stats["symbol"].isin(tickers_portfolio)]
    [["symbol", "n_obs", "data_quality_penalty", "sharpe_ratio_adjusted", "sd_return"]]
    .merge(pesos, on="symbol", how="left")
    .sort_values("weight", ascending=False)
)

print(f"  Activos totales: {len(portfolio_data_quality)}")
print(f"  Con datos ideales (>={ideal_observations} obs): "
      f"{(portfolio_data_quality['n_obs'] >= ideal_observations).sum()} "
      f"({(portfolio_data_quality['n_obs'] >= ideal_observations).mean() * 100:.1f}%)")
print(f"  Con penalizacion: {(portfolio_data_quality['data_quality_penalty'] < 1.0).sum()} "
      f"({(portfolio_data_quality['data_quality_penalty'] < 1.0).mean() * 100:.1f}%)")

if (portfolio_data_quality["data_quality_penalty"] < 1.0).any():
    limited_data = portfolio_data_quality[portfolio_data_quality["data_quality_penalty"] < 1.0].copy()
    limited_data["Peso"] = limited_data["weight"].map(lambda x: f"{x * 100:.2f}%")
    limited_data["Penalizacion"] = limited_data["data_quality_penalty"].map(lambda x: f"{x * 100:.1f}%")
    print(limited_data.rename(columns={"symbol": "Symbol", "n_obs": "Obs", "sharpe_ratio_adjusted": "Sharpe"})
          [["Symbol", "Peso", "Obs", "Penalizacion", "Sharpe"]].to_string(index=False))

print("\nScript completado con exito!")
