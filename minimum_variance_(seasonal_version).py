# ==============================================================================
# OPTIMIZACION DE PORTAFOLIOS - MINIMA VARIANZA (Seasonal Version)
# ==============================================================================

import warnings
warnings.filterwarnings("ignore")

import re
import time
import math
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

import yfinance as yf
import statsmodels.api as sm
from scipy.stats import norm, skew, kurtosis
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import quadprog

import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------
# API KEY - Polygon.io
# ------------------------------------------------
import os
from dotenv import load_dotenv
load_dotenv()
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY")

if not POLYGON_API_KEY:
    print("ADVERTENCIA: No hay POLYGON_API_KEY configurada. Todas las consultas de opciones")
    print("             fallaran y cada activo caera a volatilidad historica (sin IV real).")

# ==============================================================================
# SECCION 1: PARAMETROS CONFIGURABLES
# ==============================================================================

benchmark = "SPY"

n_top_sp500 = 50
n_top_nasdaq = 50
n_top_international = 30
target_total_tickers = 130

# HORIZONTE DE DATOS HISTORICOS
start_date = "2018-01-01"
end_date = date.today()

# MESES DE EJECUCION: define el horizonte de inversion.
execution_months = [9]

# PARAMETROS FINANCIEROS
risk_free_rate = 0.045
risk_free_rate_weekly = risk_free_rate / 52

n_pre_seasonal = 45
n_divers_candidates = 45

volatility_percentile = 0.97
correlation_percentile = 0.90
max_assets_in_portfolio = 6

seasonal_min_weeks = 10

# === PARAMETROS BKM Y TAIL RISK (VaR_CF Cornish-Fisher, fusionado con la ventana estacional) ===
bkm_moneyness_lo = 0.70            # limite inferior de moneyness K/S para strikes OTM
bkm_moneyness_hi = 1.30            # limite superior de moneyness K/S para strikes OTM
bkm_min_options_per_side = 3       # minimo de strikes OTM por lado (calls/puts) para integracion valida
bkm_mfis_clip = 10.0               # cota de winsorizacion para MFIS (estabilidad numerica en activos de baja MFIV)
bkm_mfik_clip = 30.0               # cota de winsorizacion para MFIK (idem)
tail_risk_filter_confidence = 0.99  # confianza del VaR_CF usado para rankear/filtrar candidatos (reemplaza SD estacional)
cornish_fisher_confidence = 0.95    # confianza del VaR/CVaR prospectivo del portafolio final
cornish_fisher_mfis_clip = 5.0      # cota de MFIS para el termino de Cornish-Fisher (la expansion pierde validez con colas extremas)
cornish_fisher_mfik_clip = 15.0     # cota de MFIK-3 (exceso) para el termino de Cornish-Fisher (idem)
tail_risk_alpha = 0.05              # alpha: aversion a curtosis implicita (MFIK) en diag(Sigma)
tail_risk_beta = 0.05               # beta: aversion a asimetria implicita negativa (MFIS) en diag(Sigma)

# RESTRICCIONES DE PONDERACION
max_weight_per_asset = 0.35
min_weight_per_asset = 0.001

# ESTRATEGIA DE PONDERACION
require_full_investment = False
min_total_weight = 1.00
max_total_weight = 1.00

# === RESTRICCION DE PARTICIPACION DE ETFs EN EL PORTAFOLIO FINAL ===
use_etf_constraint = True
etf_min_weight = 0.00
etf_max_weight = 0.05

# === RESTRICCION DE EXPOSICION CAMBIARIA (tickers no denominados en USD) ===
use_fx_factor = True
max_fx_exposure = 0.45

annualization_factor = 52

# === FILTRO DELTA (Black-Scholes) ===
use_delta_filter = True
delta_min = 0.30
delta_strike_mode = "atm"
target_dte_iv = 30
dte_tol_iv = 7
moneyness_tol_iv = 0.02

# === SHRINKAGE COVARIANZA: IMPLIED vs HISTORICA ===
use_iv_shrinkage = True
shrinkage_max = 0.40
shrinkage_min = 0.02
ratio_band = 0.20

# === CORRELACION IMPLICITA DE FACTORES (MERCADO + SECTOR + PAIS + FX) ===
use_sector_factor = True
use_country_factor = True

# ==============================================================================
# SECCION 2: VALIDACION DEL HORIZONTE
# ==============================================================================

if not (1 <= len(execution_months) <= 3):
    raise ValueError("Error: execution_months debe contener entre 1 y 3 meses.")
if any(m < 1 or m > 12 for m in execution_months):
    raise ValueError("Error: Cada mes debe ser un entero entre 1 y 12.")


def check_consecutive(months):
    if len(months) == 1:
        return True
    diffs = [(months[i + 1] - months[i]) % 12 for i in range(len(months) - 1)]
    return all(d == 1 or d == -11 % 12 for d in diffs) or all(
        (b - a) % 12 == 1 for a, b in zip(months, months[1:])
    )


if not check_consecutive(execution_months):
    raise ValueError(
        "Error: Los meses deben ser consecutivos.\n"
        f"   Recibido: {', '.join(map(str, execution_months))}\n"
        "   Ejemplos validos: [3,4,5] | [11,12,1] | [12,1,2]"
    )

horizon_months = len(execution_months)
horizon_weeks = horizon_months * (52 / 12)
horizon_factor = horizon_weeks
horizon_sqrt = math.sqrt(horizon_weeks)
rf_horizon = risk_free_rate * (horizon_months / 12)

# El horizonte de esta variante siempre es <=3 meses (validado arriba), por lo que
# siempre cae dentro de la cobertura tipica de opciones - no hace falta un cap
# analogo a options_horizon_cap_months de la version general.
use_iv_for_horizon = True
T_options = horizon_months / 12

MONTH_NAME = ["", "January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"]

execution_label = "-".join(MONTH_NAME[m] for m in execution_months)
horizon_label = f"{horizon_months} {'mes' if horizon_months == 1 else 'meses'} ({execution_label})"

print("\n" + "=" * 68)
print("     OPTIMIZACION DE PORTAFOLIO - MINIMA VARIANZA")
print("     Retornos: SEMANALES | Entrenamiento: Historico completo")
print("=" * 68 + "\n")
print(f"Horizonte de inversion : {horizon_label}")
print(f"Semanas del horizonte  : {horizon_weeks:.1f} semanas")
print(f"Entrenamiento          : {start_date} -> {end_date:%Y-%m-%d}")
print(f"Anualizacion base      : x{annualization_factor} | Horizonte: x{horizon_weeks:.1f} semanas\n")

# ==============================================================================
# SECCION 3: UNIVERSO DE INVERSION
# ==============================================================================


def safe_scrape_table(url, fallback=None):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"}
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            print(f"[ADVERTENCIA] HTTP {resp.status_code} al obtener {url}")
            return fallback
        tables = pd.read_html(resp.text)
        if not tables:
            return fallback
        return tables[0]
    except Exception as e:
        print(f"[ADVERTENCIA] Error al obtener datos de {url}: {e}")
        return fallback


def clean_symbol_table(tbl):
    tbl = tbl.copy()
    tbl.columns = [str(c).strip().lower().replace(" ", "_") for c in tbl.columns]
    symbol_col = "symbol" if "symbol" in tbl.columns else tbl.columns[1]
    tbl = tbl.rename(columns={symbol_col: "symbol"})
    tbl = tbl[
        tbl["symbol"].notna()
        & ~tbl["symbol"].astype(str).str.contains(r"\^|\$", regex=True)
        & (tbl["symbol"].astype(str).str.len() >= 1)
        & (tbl["symbol"].astype(str).str.len() <= 5)
        & ~tbl["symbol"].astype(str).str.match(r"^[0-9]")
    ].copy()
    tbl["symbol"] = tbl["symbol"].astype(str).str.upper().str.replace(".", "-", regex=False)
    return tbl


# === OBTENER TICKERS: S&P 500 (ordenado por cap. de mercado) ===
print("[INFO] Obteniendo tickers del S&P 500 (stockanalysis.com)...")
sp500_tbl = safe_scrape_table("https://stockanalysis.com/list/sp-500-stocks/")

if sp500_tbl is None:
    print("  Reintentando con slickcharts.com como fuente alterna...")
    sp500_tbl = safe_scrape_table("https://www.slickcharts.com/sp500")

if sp500_tbl is not None:
    sp500_tbl_clean = clean_symbol_table(sp500_tbl)
    sp500_tickers = sp500_tbl_clean["symbol"].iloc[: min(n_top_sp500, len(sp500_tbl_clean))].unique().tolist()
    print(f"  OK S&P 500: {n_top_sp500} objetivo, {len(sp500_tickers)} unicos obtenidos")
else:
    sp500_tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "BRK-B", "LLY", "AVGO",
                      "TSLA", "JPM", "UNH", "V", "XOM", "MA", "JNJ", "PG", "COST", "HD"]
    print("  ADVERTENCIA: Scraping fallo en ambas fuentes - usando fallback S&P 500 (20 tickers hardcodeados)")

# === OBTENER TICKERS: NASDAQ (ordenado por cap. de mercado) ===
print("\n[INFO] Obteniendo tickers del NASDAQ (stockanalysis.com)...")
nasdaq_tbl = safe_scrape_table("https://stockanalysis.com/list/nasdaq-stocks/")

if nasdaq_tbl is not None:
    nasdaq_tbl_clean = clean_symbol_table(nasdaq_tbl)
    nasdaq_tickers = nasdaq_tbl_clean["symbol"].iloc[: min(n_top_nasdaq, len(nasdaq_tbl_clean))].unique().tolist()
    print(f"  OK NASDAQ: {n_top_nasdaq} objetivo, {len(nasdaq_tickers)} unicos obtenidos")
else:
    nasdaq_tbl_clean = None
    nasdaq_tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "AVGO", "TSLA", "COST",
                       "NFLX", "AMD", "ADBE", "QCOM", "INTU", "AMAT", "TXN", "MU", "BKNG", "NOW"]
    print("  ADVERTENCIA: Scraping fallo - usando fallback NASDAQ (20 tickers hardcodeados)")

# === ETFs ===
etf_core = ["SPY", "QQQ", "VOO", "VTI", "VYM", "IWM", "GLD", "SLV", "USO", "PDBC", "HYG", "VNQ"]
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
    "EWC", "EWG", "EWU", "EWQ", "EWP",
]
etf_tickers = list(dict.fromkeys(etf_core + etf_sectoriales + etf_subsectoriales + etf_geograficos))

# === COMMODITIES ===
commodity_tickers = ["SLV", "UNG"]

# === TICKERS INTERNACIONALES ===
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

n_top_international = min(n_top_international, len(international_tickers_full))
international_tickers = international_tickers_full[:n_top_international]

etf_universe_tickers = list(dict.fromkeys(etf_tickers + commodity_tickers))

# === COMBINAR Y LIMPIAR ===
print("\n[INFO] Combinando y limpiando tickers...")

sp500_tickers_clean = list(dict.fromkeys(t.upper() for t in sp500_tickers))
nasdaq_tickers_clean = list(dict.fromkeys(t.upper() for t in nasdaq_tickers))
etf_tickers_clean = list(dict.fromkeys(t.upper() for t in etf_tickers))
commodity_tickers_clean = list(dict.fromkeys(t.upper() for t in commodity_tickers))
international_tickers_clean = list(dict.fromkeys(t.upper() for t in international_tickers))

tickers_domesticos = list(dict.fromkeys(
    sp500_tickers_clean + nasdaq_tickers_clean + etf_tickers_clean + commodity_tickers_clean
))
tickers_domesticos_ok = [
    t for t in tickers_domesticos
    if not re.search(r"\^|\$", t) and 1 <= len(t) <= 5 and not re.match(r"^[0-9]", t) and t != ""
]

all_tickers = list(dict.fromkeys(tickers_domesticos_ok + international_tickers_clean))

if target_total_tickers > 0 and len(all_tickers) < target_total_tickers:
    shortage = target_total_tickers - len(all_tickers)
    print(f"[INFO] Poblacion ({len(all_tickers)}) por debajo del objetivo ({target_total_tickers}) "
          f"- completando {shortage} tickers...")

    if len(international_tickers_full) > n_top_international:
        extra_intl = [t for t in dict.fromkeys(x.upper() for x in international_tickers_full[n_top_international:])
                      if t not in all_tickers]
        if extra_intl:
            to_add = extra_intl[:shortage]
            all_tickers = list(dict.fromkeys(all_tickers + to_add))
            shortage -= len(to_add)
            print(f"  + {len(to_add)} internacionales adicionales")

    if shortage > 0 and nasdaq_tbl_clean is not None and len(nasdaq_tbl_clean) > len(nasdaq_tickers):
        extra_nasdaq = [t for t in nasdaq_tbl_clean["symbol"].iloc[len(nasdaq_tickers):].unique().tolist()
                         if t not in all_tickers]
        if extra_nasdaq:
            to_add = extra_nasdaq[:shortage]
            all_tickers = list(dict.fromkeys(all_tickers + to_add))
            shortage -= len(to_add)
            print(f"  + {len(to_add)} NASDAQ adicionales")

    if shortage > 0:
        print(f"  ADVERTENCIA: No se pudo alcanzar target_total_tickers; faltan {shortage}")

all_tickers = list(dict.fromkeys(all_tickers))
print(f"[INFO] Total de tickers FINAL (unicos): {len(all_tickers)}\n")

# ==============================================================================
# MAPEO DE MONEDA POR SUFIJO DE TICKER + PARES FX (para conversion a USD)
# ==============================================================================

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


# ==============================================================================
# SECCION 4: REPORTE DEL UNIVERSO
# ==============================================================================

print("\n" + "=" * 67)
print("UNIVERSO DE INVERSION")
print("=" * 67)
print(f"- Tickers del S&P 500 (top {n_top_sp500} por cap.):      {len(sp500_tickers)}")
print(f"- Tickers del NASDAQ (top {n_top_nasdaq} por cap.):      {len(nasdaq_tickers)}")
print(f"- ETFs (todas las categorias):                {len(etf_tickers)}")
print(f"- Commodities:                                {len(commodity_tickers)}")
print(f"- Internacionales (top {n_top_international} de {len(international_tickers_full)}): "
      f"{len(international_tickers)}")
print(f"- Objetivo de poblacion total (target_total_tickers): {target_total_tickers}")
print(f"- Total unicos tras combinar + rellenar:      {len(all_tickers)}\n")

# ==============================================================================
# SECCION 5: DESCARGA Y PREPARACION DE DATOS
# ==============================================================================

print("=" * 67)
print("DESCARGA DE DATOS HISTORICOS")
print("=" * 67 + "\n")

print("[INFO] Descargando precios diarios desde Yahoo Finance...")
print(f"[INFO] Periodo: {start_date} -> {end_date:%Y-%m-%d}")
print("[INFO] Se convertiran a retornos SEMANALES para el entrenamiento\n")


def download_ticker_data(ticker, start, end, max_retries=3):
    for attempt in range(max_retries):
        try:
            data = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
            if data is not None and len(data) > 0:
                data = data[["Close"]].rename(columns={"Close": "adjusted"}).reset_index()
                data = data.rename(columns={"Date": "date"})
                data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None)
                data["symbol"] = ticker
                return data[["date", "symbol", "adjusted"]]
        except Exception:
            time.sleep(1)
    return None


stock_data_list = {}
successful_tickers = []
failed_tickers = []

n_total = len(all_tickers)
for i, ticker in enumerate(all_tickers, start=1):
    data = download_ticker_data(ticker, start_date, end_date)
    if data is not None and len(data) > 0:
        stock_data_list[ticker] = data
        successful_tickers.append(ticker)
    else:
        failed_tickers.append(ticker)
    if i % 25 == 0 or i == n_total:
        print(f"   Descargados {i}/{n_total}...")

print(f"\nOK Descargados exitosamente: {len(successful_tickers)} tickers")
if failed_tickers:
    print(f"X Fallaron: {len(failed_tickers)} tickers")
    print(f"   Tickers fallidos: {', '.join(failed_tickers[:10])}"
          f"{'...' if len(failed_tickers) > 10 else ''}")

stock_data = pd.concat(stock_data_list.values(), ignore_index=True) if stock_data_list else pd.DataFrame()

print("\n[INFO] Descargando benchmark (SPY)...")
benchmark_data = download_ticker_data(benchmark, start_date, end_date)

if len(stock_data) == 0 or benchmark_data is None or len(benchmark_data) == 0:
    raise RuntimeError("Error: No se descargaron datos suficientes.")

# ---------------------------------------------------------------------------
# DESCARGA DE PARES FX (para convertir precios de internacionales a USD)
# ---------------------------------------------------------------------------
print("\n[INFO] Descargando pares FX para conversion a USD...")
fx_data = {}
for cur, info in fx_pairs.items():
    d = download_ticker_data(info["ticker"], start_date, end_date)
    if d is not None and len(d) > 0:
        fx_data[cur] = d.set_index("date")["adjusted"]
        print(f"  OK {cur} ({info['ticker']})")
    else:
        print(f"  ADVERTENCIA: no se pudo descargar {info['ticker']} para {cur} "
              f"- los tickers en {cur} quedaran en moneda local")

# ---------------------------------------------------------------------------
# CONVERTIR A RETORNOS SEMANALES (precios internacionales convertidos a USD)
# ---------------------------------------------------------------------------
print("\n[INFO] Convirtiendo precios diarios -> retornos semanales...")

prices_wide = stock_data.pivot_table(index="date", columns="symbol", values="adjusted")

n_expected = len(prices_wide)
na_counts = prices_wide.isna().sum()
valid_tickers = na_counts[na_counts < 0.2 * n_expected].index.tolist()
print(f"[INFO] Tickers con datos completos (>80%): {len(valid_tickers)}")

prices_wide_clean = prices_wide[valid_tickers].dropna()

if len(prices_wide_clean) == 0:
    raise RuntimeError("Error: No hay datos despues de eliminar NAs.")

prices_weekly = prices_wide_clean.resample("W").last()

fx_weekly_price_usd = {}
for cur, s in fx_data.items():
    w = s.resample("W").last()
    fx_weekly_price_usd[cur] = (1 / w) if fx_pairs[cur]["invert"] else w
fx_weekly_price_usd = pd.DataFrame(fx_weekly_price_usd)

print("[INFO] Convirtiendo precios de tickers no-USD a USD...")
prices_weekly_usd = prices_weekly.copy()
n_convertidos = 0
for t in prices_weekly_usd.columns:
    cur = get_currency_for_ticker(t)
    if cur != "USD":
        if cur in fx_weekly_price_usd.columns:
            fx_series = fx_weekly_price_usd[cur].reindex(prices_weekly_usd.index).ffill()
            prices_weekly_usd[t] = prices_weekly_usd[t] * fx_series
            n_convertidos += 1
        else:
            print(f"  ADVERTENCIA: sin serie FX para {t} ({cur}) - queda en moneda local")
print(f"  OK Tickers convertidos a USD: {n_convertidos}")

weekly_returns = np.log(prices_weekly_usd / prices_weekly_usd.shift(1)).dropna(how="all")
weekly_returns = weekly_returns.dropna()

fx_weekly_returns = np.log(fx_weekly_price_usd / fx_weekly_price_usd.shift(1)).dropna(how="all")

benchmark_daily = benchmark_data.set_index("date")["adjusted"]
benchmark_weekly_prices = benchmark_daily.resample("W").last()
benchmark_returns_full = np.log(benchmark_weekly_prices / benchmark_weekly_prices.shift(1)).dropna()
benchmark_returns_full = benchmark_returns_full.rename("SPY").to_frame()

common_dates = weekly_returns.index.intersection(benchmark_returns_full.index)
weekly_returns = weekly_returns.loc[common_dates]
benchmark_returns_full = benchmark_returns_full.loc[common_dates]

n_weeks = len(weekly_returns)
n_years = round(n_weeks / 52, 1)
print(f"[INFO] Semanas disponibles para entrenamiento: {n_weeks} (~{n_years} anios)")
print(f"[INFO] Tickers validos: {weekly_returns.shape[1]}\n")

log_returns = weekly_returns.copy()
benchmark_returns = benchmark_returns_full.copy()

print(f"[INFO] Entrenamiento con {len(log_returns)} semanas completas "
      f"({log_returns.index.min():%Y-%m-%d} -> {log_returns.index.max():%Y-%m-%d})")
print(f"[INFO] Horizonte de inversion: {horizon_label}\n")

if log_returns.isna().any().any() or np.isinf(log_returns.values).any():
    print("[ADVERTENCIA] Limpiando NAs/Inf...")
    bad_mask = log_returns.isna().any() | np.isinf(log_returns).any()
    bad_cols = log_returns.columns[bad_mask]
    good_cols = [c for c in log_returns.columns if c not in bad_cols]
    log_returns = log_returns[good_cols]
    print(f"   Tickers restantes: {log_returns.shape[1]}")

if log_returns.shape[1] < 5:
    raise RuntimeError("Error: Quedan menos de 5 tickers validos.")

# ==============================================================================
# SECCION 6: ANALISIS DESCRIPTIVO Y SELECCION
# ==============================================================================

print("\n" + "=" * 67)
print("ANALISIS DE SELECCION DE ACTIVOS")
print(f"Entrenamiento: {n_weeks} semanas | Horizonte: {horizon_label} (~{horizon_weeks:.1f} semanas)")
print("=" * 67 + "\n")


def max_drawdown_from_returns(returns_series):
    r = pd.Series(returns_series).dropna()
    if len(r) < 2:
        return np.nan
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()


asset_stats = pd.DataFrame({
    "Symbol": log_returns.columns,
    "Mean_Return": log_returns.mean().values * annualization_factor,
    "Volatility": log_returns.std().values * math.sqrt(annualization_factor),
})
asset_stats["Sharpe"] = (asset_stats["Mean_Return"] - risk_free_rate) / asset_stats["Volatility"]

correlations = log_returns.apply(lambda col: col.corr(benchmark_returns.iloc[:, 0]))
asset_stats["Correlation_SPY"] = correlations.values

mdd_values = log_returns.apply(lambda col: max_drawdown_from_returns(col))
asset_stats["Max_Drawdown"] = mdd_values.values

asset_stats = asset_stats[
    asset_stats["Sharpe"].notna() & asset_stats["Volatility"].notna() & asset_stats["Correlation_SPY"].notna()
    & np.isfinite(asset_stats["Sharpe"]) & np.isfinite(asset_stats["Volatility"])
].sort_values("Correlation_SPY").reset_index(drop=True)

print(f"[INFO] Activos con metricas validas: {len(asset_stats)}\n")

vol_threshold = asset_stats["Volatility"].quantile(volatility_percentile)
corr_threshold = asset_stats["Correlation_SPY"].abs().quantile(correlation_percentile)

asset_stats["Passes_Vol"] = asset_stats["Volatility"] <= vol_threshold
asset_stats["Passes_Corr"] = asset_stats["Correlation_SPY"].abs() <= corr_threshold
asset_stats["Passes_Return"] = asset_stats["Mean_Return"] > 0
asset_stats["Passes_All"] = asset_stats["Passes_Vol"] & asset_stats["Passes_Corr"] & asset_stats["Passes_Return"]
asset_stats = asset_stats.sort_values("Volatility").reset_index(drop=True)

selected_pre_seasonal = (
    asset_stats[asset_stats["Passes_All"]].sort_values("Volatility").head(n_pre_seasonal)["Symbol"].tolist()
)

if len(selected_pre_seasonal) < 5:
    print("\n[ADVERTENCIA] Muy pocos activos pasan todos los filtros. Relajando filtro de correlacion...")
    selected_pre_seasonal = (
        asset_stats[asset_stats["Passes_Vol"] & asset_stats["Passes_Return"]]
        .sort_values("Volatility").head(n_pre_seasonal)["Symbol"].tolist()
    )

print(f"\n[INFO] Pool pre-filtro estacional: {len(selected_pre_seasonal)} candidatos "
      f"(max configurado: {n_pre_seasonal})")

# === FILTRO DELTA (Black-Scholes) ============================================


def get_spot_safe(ticker):
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


_polygon_cache = {}


def get_polygon_option_snapshot(ticker):
    if ticker in _polygon_cache:
        return _polygon_cache[ticker]

    resultado = None
    try:
        S = get_spot_safe(ticker)
        if pd.isna(S) or S <= 0:
            raise ValueError("sin spot Yahoo para filtrar la llamada a Polygon")

        hoy = date.today()
        fecha_min = (hoy + timedelta(days=target_dte_iv - dte_tol_iv)).strftime("%Y-%m-%d")
        fecha_max = (hoy + timedelta(days=target_dte_iv + dte_tol_iv)).strftime("%Y-%m-%d")
        strike_min = round(S * (1 - moneyness_tol_iv), 2)
        strike_max = round(S * (1 + moneyness_tol_iv), 2)

        url = (
            f"https://api.polygon.io/v3/snapshot/options/{ticker}?"
            f"contract_type=call&"
            f"strike_price.gte={strike_min:.2f}&strike_price.lte={strike_max:.2f}&"
            f"expiration_date.gte={fecha_min}&expiration_date.lte={fecha_max}&"
            f"limit=250&apiKey={POLYGON_API_KEY}"
        )

        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            raise ValueError("HTTP != 200")

        data_json = resp.json()
        results = data_json.get("results")
        if not results:
            raise ValueError("sin resultados")

        df = pd.json_normalize(results)
        hoy_ts = pd.Timestamp(hoy)
        df["strike"] = df["details.strike_price"]
        df["expiracion"] = pd.to_datetime(df["details.expiration_date"])
        df["dte"] = (df["expiracion"] - hoy_ts).dt.days

        df = df[(df["dte"] >= target_dte_iv - dte_tol_iv) & (df["dte"] <= target_dte_iv + dte_tol_iv)].copy()
        if len(df) == 0:
            raise ValueError("sin contratos en la ventana ~30 DTE")

        df["_s1"] = (df["dte"] - target_dte_iv).abs()
        df["_s2"] = (df["strike"] / S - 1).abs()
        df = df.sort_values(["_s1", "_s2"])
        elegido = df.iloc[0]

        spot_final = elegido.get("underlying_asset.price", np.nan)
        spot_final = float(spot_final) if not pd.isna(spot_final) else S

        resultado = dict(
            spot=spot_final,
            iv=float(elegido.get("implied_volatility", np.nan)) if not pd.isna(elegido.get("implied_volatility", np.nan)) else np.nan,
            delta=float(elegido.get("greeks.delta", np.nan)) if not pd.isna(elegido.get("greeks.delta", np.nan)) else np.nan,
            gamma=float(elegido.get("greeks.gamma", np.nan)) if not pd.isna(elegido.get("greeks.gamma", np.nan)) else np.nan,
            vega=float(elegido.get("greeks.vega", np.nan)) if not pd.isna(elegido.get("greeks.vega", np.nan)) else np.nan,
            theta=float(elegido.get("greeks.theta", np.nan)) if not pd.isna(elegido.get("greeks.theta", np.nan)) else np.nan,
        )
    except Exception:
        resultado = None

    _polygon_cache[ticker] = resultado
    return resultado


def get_atm_iv_safe(ticker):
    if get_currency_for_ticker(ticker) != "USD":
        # Polygon solo cubre opciones de emisores listados en EE.UU. Para
        # tickers internacionales no se consulta - el fallback historico
        # ya existente en cada punto de uso se activa automaticamente
        # al recibir NaN aqui.
        return np.nan
    poly = get_polygon_option_snapshot(ticker)
    if poly is not None and not pd.isna(poly["iv"]) and poly["iv"] > 0:
        return poly["iv"]
    return np.nan


# ==============================================================================
# FUNCIONES BKM (Bakshi, Kapadia y Madan, 2003) - extraccion de momentos
# implicitos (MFIV/MFIS/MFIK) a partir de la cadena de opciones OTM.
#
#   V(T) = int_S^inf [2*(1-ln(K/S))/K^2] C(K) dK + int_0^S [2*(1+ln(S/K))/K^2] P(K) dK
#   W(T) = int_S^inf [6*ln(K/S)-3*ln(K/S)^2]/K^2 C(K) dK - int_0^S [6*ln(S/K)+3*ln(S/K)^2]/K^2 P(K) dK
#   X(T) = int_S^inf [12*ln(K/S)^2-4*ln(K/S)^3]/K^2 C(K) dK + int_0^S [12*ln(S/K)^2+4*ln(S/K)^3]/K^2 P(K) dK
#
#   mu(T)   = e^{rT} - 1 - (e^{rT}/2)*V - (e^{rT}/6)*W - (e^{rT}/24)*X
#   MFIV(T) = e^{rT}*V - mu(T)^2
#   MFIS(T) = [e^{rT}*W - 3*mu(T)*e^{rT}*V + 2*mu(T)^3] / MFIV(T)^{3/2}
#   MFIK(T) = [e^{rT}*X - 4*mu(T)*e^{rT}*W + 6*e^{rT}*mu(T)^2*V - 3*mu(T)^4] / MFIV(T)^2
#
# Integracion numerica via regla del Trapecio (np.trapezoid) sobre strikes OTM disponibles.
# ==============================================================================
def bs_price(S, K, T_yrs, r, sigma, tipo="call"):
    if T_yrs <= 0 or sigma <= 0:
        return np.nan
    d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T_yrs) / (sigma * math.sqrt(T_yrs))
    d2 = d1 - sigma * math.sqrt(T_yrs)
    if tipo == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T_yrs) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T_yrs) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bkm_fetch_otm_chain(ticker, target_dte, dte_tol, S, moneyness_lo, moneyness_hi):
    resultado = {"call": pd.DataFrame(columns=["strike", "iv", "type"]),
                 "put": pd.DataFrame(columns=["strike", "iv", "type"])}
    if pd.isna(S) or S <= 0:
        return resultado["call"], resultado["put"]
    hoy = date.today()
    fecha_min = (hoy + timedelta(days=target_dte - dte_tol)).strftime("%Y-%m-%d")
    fecha_max = (hoy + timedelta(days=target_dte + dte_tol)).strftime("%Y-%m-%d")
    strike_min = round(S * moneyness_lo, 2)
    strike_max = round(S * moneyness_hi, 2)
    for tipo in ("call", "put"):
        try:
            url = (
                f"https://api.polygon.io/v3/snapshot/options/{ticker}?"
                f"contract_type={tipo}&"
                f"strike_price.gte={strike_min:.2f}&strike_price.lte={strike_max:.2f}&"
                f"expiration_date.gte={fecha_min}&expiration_date.lte={fecha_max}&"
                f"limit=250&apiKey={POLYGON_API_KEY}"
            )
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                continue
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


def bkm_get_current_moments(ticker, target_dte, dte_tol, rf):
    S = get_spot_safe(ticker)
    if pd.isna(S) or S <= 0:
        return dict(mfiv=np.nan, mfis=np.nan, mfik=np.nan, mu=np.nan, ok=False, spot=np.nan)
    calls_df, puts_df = bkm_fetch_otm_chain(ticker, target_dte, dte_tol, S, bkm_moneyness_lo, bkm_moneyness_hi)
    T = target_dte / 365
    calls_df = bkm_iv_chain_to_prices(S, rf, T, calls_df)
    puts_df = bkm_iv_chain_to_prices(S, rf, T, puts_df)
    mom = bkm_compute_moments(S, rf, T, calls_df, puts_df)
    mom["spot"] = S
    return mom


bkm_moments_cache = {}


def bkm_get_current_moments_cached(ticker):
    if ticker in bkm_moments_cache:
        return bkm_moments_cache[ticker]
    if not use_iv_for_horizon or get_currency_for_ticker(ticker) != "USD":
        mom = dict(mfiv=np.nan, mfis=np.nan, mfik=np.nan, mu=np.nan, ok=False, spot=np.nan)
    else:
        mom = bkm_get_current_moments(ticker, target_dte_iv, dte_tol_iv, risk_free_rate)
    bkm_moments_cache[ticker] = mom
    return mom


def bkm_annual_vol(ticker):
    mom = bkm_get_current_moments_cached(ticker)
    if mom["ok"] and mom["mfiv"] > 0:
        return math.sqrt(mom["mfiv"] / T_options)
    return np.nan


if use_delta_filter:
    print(f"\nAplicando filtro Delta BS (modo='{delta_strike_mode}', delta_min={delta_min:.2f}, "
          f"T={horizon_months / 12:.3f} anios)...")

    T_horizon = horizon_months / 12

    mu_weekly_for_delta = log_returns[selected_pre_seasonal].mean()

    print(f"  Consultando spot + IV ATM (~{target_dte_iv} DTE) para {len(selected_pre_seasonal)} activos "
          f"(pool pre-estacional)...")

    spot_cache = {t: get_spot_safe(t) for t in selected_pre_seasonal}
    iv_cache = {t: get_atm_iv_safe(t) for t in selected_pre_seasonal}

    delta_rows = []
    for ticker in selected_pre_seasonal:
        S = spot_cache[ticker]
        iv = iv_cache[ticker]

        if pd.isna(iv) or iv <= 0:
            hist_ret = log_returns[ticker].values
            iv = np.nanstd(hist_ret) * math.sqrt(annualization_factor)

        if pd.isna(S) or S <= 0 or pd.isna(iv) or iv <= 0:
            delta_rows.append(dict(symbol=ticker, delta=np.nan, strike_mode=delta_strike_mode, iv_used=iv))
            continue

        mu_i = mu_weekly_for_delta.get(ticker, 0.0)
        mu_i = 0.0 if pd.isna(mu_i) else mu_i

        if delta_strike_mode == "atm":
            K = S
        elif delta_strike_mode == "rf":
            K = S * (1 + risk_free_rate * T_horizon)
        elif delta_strike_mode == "mu":
            K = S * (1 + mu_i * horizon_weeks)
        else:
            K = S

        d1 = (math.log(S / K) + (risk_free_rate + iv ** 2 / 2) * T_horizon) / (iv * math.sqrt(T_horizon))
        delta_i = norm.cdf(d1)

        delta_rows.append(dict(symbol=ticker, delta=delta_i, strike_mode=delta_strike_mode, iv_used=iv))

    delta_df = pd.DataFrame(delta_rows)

    n_delta_ok = delta_df["delta"].notna().sum()
    n_delta_na = delta_df["delta"].isna().sum()
    n_below = ((delta_df["delta"].notna()) & (delta_df["delta"] < delta_min)).sum()
    n_pass = ((delta_df["delta"].notna()) & (delta_df["delta"] >= delta_min)).sum()

    print(f"  OK Deltas calculados: {n_delta_ok} | Sin datos (se conservan): {n_delta_na}")
    print(f"  X Descartados (delta < {delta_min:.2f}): {n_below}")
    print(f"  OK Superan el filtro (delta >= {delta_min:.2f}): {n_pass}")

    discarded_delta = delta_df[(delta_df["delta"].notna()) & (delta_df["delta"] < delta_min)].sort_values("delta")
    if len(discarded_delta) > 0:
        disp = discarded_delta.copy()
        disp["Delta"] = disp["delta"].map(lambda x: f"{x:.3f}")
        disp["IV_anual"] = disp["iv_used"].map(lambda x: f"{x * 100:.1f}%")
        print("\n  Activos descartados por Delta insuficiente:")
        print(disp.rename(columns={"symbol": "Symbol"})[["Symbol", "Delta", "IV_anual"]].to_string(index=False))

    top_delta = (
        delta_df[(delta_df["delta"].isna()) | (delta_df["delta"] >= delta_min)]
        .sort_values("delta", ascending=False)
        .head(10)
        .dropna(subset=["delta"])
        .copy()
    )
    if len(top_delta) > 0:
        top_delta["Delta"] = top_delta["delta"].map(lambda x: f"{x:.3f}")
        top_delta["IV_anual"] = top_delta["iv_used"].map(lambda x: f"{x * 100:.1f}%")
        print("\n  Top 10 activos por Delta (mayor probabilidad de superar K):")
        print(top_delta.rename(columns={"symbol": "Symbol"})[["Symbol", "Delta", "IV_anual"]].to_string(index=False))

    selected_pre_seasonal = delta_df[(delta_df["delta"].isna()) | (delta_df["delta"] >= delta_min)]["symbol"].tolist()

    spot_cache = {t: spot_cache[t] for t in selected_pre_seasonal}
    iv_cache = {t: iv_cache[t] for t in selected_pre_seasonal}

    print(f"\nTras filtro Delta (sobre pool pre-estacional): {len(selected_pre_seasonal)} tickers disponibles "
          f"para el filtro estacional\n")

else:
    print("\nFiltro Delta desactivado (use_delta_filter = False)\n")
    print(f"  Consultando spot + IV ATM (~{target_dte_iv} DTE) para {len(selected_pre_seasonal)} activos "
          f"(uso: covarianza)...")
    spot_cache = {t: get_spot_safe(t) for t in selected_pre_seasonal}
    iv_cache = {t: get_atm_iv_safe(t) for t in selected_pre_seasonal}

# ==============================================================================
# FILTRO ESTACIONAL DE TAIL RISK BKM: VaR_CF (Cornish-Fisher) CON mu/HV
# DE LA VENTANA ESTACIONAL + MFIS/MFIK FORWARD DE BKM
# (reemplaza el ranking por SD estacional simple: la mu y la HV historica que
#  alimentan el VaR_CF se calculan SOLO con las semanas dentro de execution_months,
#  preservando la senal de estacionalidad, y se le suma la vista prospectiva de
#  asimetria/curtosis implicita del mercado de opciones vigente)
# ==============================================================================
print(f"\nAplicando filtro de Tail Risk BKM estacional ({execution_label}) - "
      f"VaR_CF a {tail_risk_filter_confidence * 100:.0f}% de confianza...")

log_returns_seasonal = log_returns.loc[log_returns.index.month.isin(execution_months), selected_pre_seasonal]

n_seasonal_weeks = len(log_returns_seasonal)
print(f"  OK Semanas dentro de {execution_label} disponibles: {n_seasonal_weeks}")

z_a_filter = norm.ppf(1 - tail_risk_filter_confidence)

tail_risk_rows = []
for ticker in selected_pre_seasonal:
    r_seasonal = log_returns_seasonal[ticker].dropna()
    n_obs = len(r_seasonal)
    hv_annual = r_seasonal.std() * math.sqrt(annualization_factor) if n_obs > 5 else np.nan
    mu_weekly_seasonal = r_seasonal.mean() if n_obs > 5 else np.nan

    mom = bkm_get_current_moments_cached(ticker)

    if n_obs < seasonal_min_weeks or not mom["ok"]:
        tail_risk_rows.append(dict(Symbol=ticker, MFIV=mom.get("mfiv", np.nan), MFIS=mom.get("mfis", np.nan),
                                    MFIK=mom.get("mfik", np.nan), VaR_CF=np.nan,
                                    Seasonal_SD=r_seasonal.std() if n_obs > 1 else np.nan, N_Obs=n_obs))
        continue

    mu_T = mu_weekly_seasonal * (target_dte_iv / 7) if not pd.isna(mu_weekly_seasonal) else 0.0
    sigma_T = math.sqrt(mom["mfiv"])
    S_skew = mom["mfis"]
    K_exc = mom["mfik"] - 3.0

    z_cf = (z_a_filter + (z_a_filter ** 2 - 1) / 6 * S_skew
            + (z_a_filter ** 3 - 3 * z_a_filter) / 24 * K_exc
            - (2 * z_a_filter ** 3 - 5 * z_a_filter) / 36 * S_skew ** 2)
    var_cf_i = -(mu_T + sigma_T * z_cf)

    tail_risk_rows.append(dict(Symbol=ticker, MFIV=mom["mfiv"], MFIS=mom["mfis"], MFIK=mom["mfik"],
                                VaR_CF=var_cf_i, Seasonal_SD=r_seasonal.std(), N_Obs=n_obs))

tail_risk_stats = pd.DataFrame(tail_risk_rows)
tail_risk_stats = tail_risk_stats[
    (tail_risk_stats["N_Obs"] >= seasonal_min_weeks) & tail_risk_stats["VaR_CF"].notna()
].sort_values("VaR_CF").reset_index(drop=True)

n_con_tail = len(tail_risk_stats)
n_sin_tail = len(selected_pre_seasonal) - n_con_tail

print(f"  OK Candidatos previos al filtro (post-Delta): {len(selected_pre_seasonal)}")
print(f"  OK Con VaR_CF calculable (BKM + estacional, >={seasonal_min_weeks} sem): {n_con_tail}")
print(f"  ADVERTENCIA Descartados (sin cobertura de opciones BKM o datos estacionales insuficientes): {n_sin_tail}")

tail_risk_final = tail_risk_stats.head(n_divers_candidates)

print(f"  OK Seleccionados (<={n_divers_candidates}, menor VaR_CF estacional): {len(tail_risk_final)}")

if len(tail_risk_final) < n_divers_candidates * 0.5:
    print(f"\n  ADVERTENCIA: solo {len(tail_risk_final)} tickers sobrevivieron el filtro de Tail Risk estacional.")
    print(f"     Considera reducir seasonal_min_weeks (actual: {seasonal_min_weeks}), ampliar start_date,")
    print("     o ampliar bkm_moneyness_lo/hi.")

tail_risk_final = tail_risk_final.merge(asset_stats[["Symbol", "Volatility", "Sharpe"]], on="Symbol", how="left")

print(f"\n  Candidatos finales ordenados por VaR_CF estacional (Tail Risk Score, {execution_label}):")
seasonal_display = tail_risk_final.copy()
seasonal_display["VaR_CF"] = seasonal_display["VaR_CF"].map(lambda x: f"{x * 100:.3f}%")
seasonal_display["MFIV"] = seasonal_display["MFIV"].map(lambda x: f"{x:.5f}")
seasonal_display["MFIS"] = seasonal_display["MFIS"].map(lambda x: f"{x:.3f}")
seasonal_display["MFIK"] = seasonal_display["MFIK"].map(lambda x: f"{x:.3f}")
seasonal_display["SD_Estacional"] = seasonal_display["Seasonal_SD"].map(lambda x: f"{x:.4f}")
seasonal_display["Sharpe_fmt"] = seasonal_display["Sharpe"].map(lambda x: f"{x:.3f}")
print(seasonal_display.rename(columns={"N_Obs": "Semanas", "Sharpe_fmt": "Sharpe"})
      [["Symbol", "VaR_CF", "MFIV", "MFIS", "MFIK", "SD_Estacional", "Semanas", "Sharpe"]]
      .to_string(index=False))

selected_tickers = tail_risk_final["Symbol"].tolist()
spot_cache = {t: spot_cache[t] for t in selected_tickers}
iv_cache = {t: iv_cache[t] for t in selected_tickers}
print(f"\nOK Conjunto tras filtro de Tail Risk BKM estacional: {len(selected_tickers)} tickers\n")

log_returns_selected = log_returns[selected_tickers]

# ==============================================================================
# SECCION 7: ESTADISTICAS DESCRIPTIVAS
# ==============================================================================

print("\n" + "=" * 67)
print("ESTADISTICAS DESCRIPTIVAS DE ACTIVOS SELECCIONADOS")
print(f"Anualizacion x{annualization_factor} | Horizonte: {horizon_label} (~{horizon_weeks:.1f} sem)")
print("=" * 67 + "\n")

descriptive_stats = pd.DataFrame({
    "Symbol": log_returns_selected.columns,
    "Media_Semanal": log_returns_selected.mean().values,
    "SD_Semanal": log_returns_selected.std().values,
    "Retorno_Horizonte": log_returns_selected.mean().values * horizon_weeks,
    "Volatilidad_Horizonte": log_returns_selected.std().values * horizon_sqrt,
    "Retorno_Anual": log_returns_selected.mean().values * annualization_factor,
    "Volatilidad_Anual": log_returns_selected.std().values * math.sqrt(annualization_factor),
    "Asimetria_P": [skew(log_returns_selected[c].dropna()) for c in log_returns_selected.columns],
    "Curtosis_P": [kurtosis(log_returns_selected[c].dropna()) for c in log_returns_selected.columns],
    "Max_Drawdown": [max_drawdown_from_returns(log_returns_selected[c]) for c in log_returns_selected.columns],
})

# Metricas prospectivas BKM (medida Q) junto a las historicas (medida P) de arriba
descriptive_stats["MFIV_T"] = [bkm_moments_cache.get(c, {}).get("mfiv", np.nan) for c in log_returns_selected.columns]
descriptive_stats["MFIS_Q"] = [bkm_moments_cache.get(c, {}).get("mfis", np.nan) for c in log_returns_selected.columns]
descriptive_stats["MFIK_Q"] = [bkm_moments_cache.get(c, {}).get("mfik", np.nan) for c in log_returns_selected.columns]
descriptive_stats["MFIV_Vol_Anual_Q"] = np.sqrt(descriptive_stats["MFIV_T"] / T_options)

disp = descriptive_stats.copy()
for col in ["Retorno_Horizonte", "Volatilidad_Horizonte", "Retorno_Anual", "Volatilidad_Anual",
            "Max_Drawdown", "MFIV_Vol_Anual_Q"]:
    disp[col] = disp[col].map(lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "N/D")
disp["Asimetria_P"] = disp["Asimetria_P"].round(2)
disp["Curtosis_P"] = disp["Curtosis_P"].round(2)
disp["MFIS_Q"] = disp["MFIS_Q"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/D")
disp["MFIK_Q"] = disp["MFIK_Q"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/D")
print(disp[["Symbol", "Retorno_Horizonte", "Volatilidad_Horizonte", "Retorno_Anual", "Volatilidad_Anual",
            "Asimetria_P", "Curtosis_P", "Max_Drawdown", "MFIV_Vol_Anual_Q", "MFIS_Q", "MFIK_Q"]]
      .to_string(index=False))

# ==============================================================================
# SECCION 8: OPTIMIZACION DE PORTAFOLIOS
# ==============================================================================

print("\n" + "=" * 67)
print("OPTIMIZACION DE PORTAFOLIOS - MINIMA VARIANZA")
print(f"Datos: {n_weeks} semanas | Horizonte: {horizon_label} (~{horizon_weeks:.1f} sem) | "
      f"Activos: {len(selected_tickers)}")
print("=" * 67 + "\n")

mean_ret = log_returns_selected.mean()
sd_ret = log_returns_selected.std()

# --- MATRIZ DE COVARIANZA: MFIV (BKM) model-free implied variance -----------
benchmark_iv = "SPY"

print("\nEstimando MFIV (BKM) para covarianza forward-looking...")

mfik_arr = np.array([bkm_moments_cache.get(t, {}).get("mfik", np.nan) for t in selected_tickers])
mfis_arr = np.array([bkm_moments_cache.get(t, {}).get("mfis", np.nan) for t in selected_tickers])

iv_assets_implied = np.array([bkm_annual_vol(t) for t in selected_tickers])

n_iv_ok = np.sum(~pd.isna(iv_assets_implied))
n_iv_na = np.sum(pd.isna(iv_assets_implied))
print(f"  OK MFIV (BKM) validas: {n_iv_ok} | Sin datos (fallback historico): {n_iv_na}")

iv_spy_implied = bkm_annual_vol(benchmark_iv) if use_iv_for_horizon else np.nan

sd_hist_annual = sd_ret.values * math.sqrt(annualization_factor)
iv_final = np.where(~pd.isna(iv_assets_implied), iv_assets_implied, sd_hist_annual)
iv_final = np.where(pd.isna(iv_final), sd_hist_annual, iv_final)

print("\n  DIAGNOSTICO DE ESCALA:")
print(f"     sd_ret (semanal, primeros 3):      {sd_ret.values[0]:.6f} | {sd_ret.values[1]:.6f} | {sd_ret.values[2]:.6f}")
print(f"     sd_hist_annual (primeros 3):       {sd_hist_annual[0]:.6f} | {sd_hist_annual[1]:.6f} | {sd_hist_annual[2]:.6f}")
print(f"     iv_final (primeros 3):             {iv_final[0]:.6f} | {iv_final[1]:.6f} | {iv_final[2]:.6f}")
print(f"     iv_spy_implied:                    {iv_spy_implied if not pd.isna(iv_spy_implied) else np.nan}")

if pd.isna(iv_spy_implied):
    print("  ADVERTENCIA Sin IV para SPY - usando vol historica como fallback")
    if "SPY" in log_returns_selected.columns:
        iv_spy_implied = log_returns_selected["SPY"].std() * math.sqrt(annualization_factor)
    else:
        spy_hist = download_ticker_data("SPY", start_date, end_date)
        spy_hist = spy_hist.set_index("date")["adjusted"].resample("W").last()
        spy_ret = np.log(spy_hist / spy_hist.shift(1)).dropna()
        iv_spy_implied = spy_ret.std() * math.sqrt(annualization_factor)

# ==============================================================================
# MODELO DE CORRELACION IMPLICITA: FACTORES (MERCADO + SECTOR + PAIS + FX)
# ==============================================================================
n_sel = len(selected_tickers)

print("\nConstruyendo modelo de correlacion de factores (mercado + sector + pais + fx)...")

sector_etf_map = {
    "Information Technology": "XLK",
    "Health Care": "XLV",
    "Financials": "XLF",
    "Energy": "XLE",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    # XLC (lanzado en 2018) no cubre start_date=2014 con suficiente historia
    # y queda descartado en el filtro de completitud - se usa VOX (Vanguard
    # Communication Services ETF, inception 2004) como proxy equivalente.
    "Communication Services": "VOX",
}

ticker_sector_etf = {}
if use_sector_factor:
    try:
        sector_tbl = safe_scrape_table("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        if sector_tbl is None:
            raise ValueError("scraping de sectores fallo")
        sector_tbl = sector_tbl.copy()
        sector_tbl.columns = [str(c).strip().lower().replace(" ", "_") for c in sector_tbl.columns]
        sector_col = "gics_sector" if "gics_sector" in sector_tbl.columns else None
        if sector_col is None:
            raise ValueError("no se encontro columna GICS Sector")
        sector_tbl["symbol"] = sector_tbl["symbol"].astype(str).str.upper().str.replace(".", "-", regex=False)
        sector_tbl["sector_etf"] = sector_tbl[sector_col].map(sector_etf_map)
        sector_tbl = sector_tbl[sector_tbl["sector_etf"].notna()]
        ticker_sector_etf = dict(zip(sector_tbl["symbol"], sector_tbl["sector_etf"]))
    except Exception:
        print("  ADVERTENCIA No se pudo obtener el mapeo de sectores (Wikipedia) - cae a modelo de 1 factor (solo mercado) para domesticos")
        ticker_sector_etf = {}

    # === MAPEO DE SECTOR PARA TICKERS INTERNACIONALES (mismos ETFs sectoriales
    # ya presentes en etf_sectoriales, no se abren nuevos conjuntos) ===
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
    ticker_sector_etf.update(international_sector_etf_map)

asset_sector_etf = {t: ticker_sector_etf.get(t) for t in selected_tickers}

sectores_presentes = sorted(set(v for v in asset_sector_etf.values() if v is not None))
sectores_sin_precio = [s for s in sectores_presentes if s not in log_returns.columns]
if sectores_sin_precio:
    print(f"  ADVERTENCIA ETFs sectoriales sin precios descargados (se ignoran): {', '.join(sectores_sin_precio)}")
    asset_sector_etf = {t: (None if v in sectores_sin_precio else v) for t, v in asset_sector_etf.items()}

sectores_unicos = sorted(set(v for v in asset_sector_etf.values() if v is not None))
k_sectores = len(sectores_unicos)

n_con_sector = sum(1 for v in asset_sector_etf.values() if v is not None)
print(f"  OK Sector asignado: {n_con_sector} de {n_sel} activos | {k_sectores} sectores distintos en el pool"
      + (f" ({', '.join(sectores_unicos)})" if k_sectores > 0 else ""))

# ==============================================================================
# MAPEO DE PAIS/REGION PARA TICKERS INTERNACIONALES
# ==============================================================================

country_etf_by_suffix = {
    ".TO": "EWC",
    ".DE": "EWG",
    ".L": "EWU",
    ".PA": "EWQ",
    ".MC": "EWP",
    ".T": "EWJ",
}

country_etf_ticker_override = {
    "HSBC": "EWU",
    "BP": "EWU",
}


def get_country_etf_for_ticker(ticker):
    if ticker in country_etf_ticker_override:
        return country_etf_ticker_override[ticker]
    for suf, etf in country_etf_by_suffix.items():
        if ticker.endswith(suf):
            return etf
    return None


asset_country_etf = {}
if use_country_factor:
    asset_country_etf = {t: get_country_etf_for_ticker(t) for t in selected_tickers}

paises_presentes = sorted(set(v for v in asset_country_etf.values() if v is not None))
paises_sin_precio = [p for p in paises_presentes if p not in log_returns.columns]
if paises_sin_precio:
    print(f"  ADVERTENCIA ETFs de pais sin precios descargados (se ignoran): {', '.join(paises_sin_precio)}")
    asset_country_etf = {t: (None if v in paises_sin_precio else v) for t, v in asset_country_etf.items()}

paises_unicos = sorted(set(v for v in asset_country_etf.values() if v is not None))
k_paises = len(paises_unicos)

n_con_pais = sum(1 for v in asset_country_etf.values() if v is not None)
print(f"  OK Pais asignado: {n_con_pais} de {n_sel} activos | {k_paises} paises distintos en el pool"
      + (f" ({', '.join(paises_unicos)})" if k_paises > 0 else ""))

# ==============================================================================
# MAPEO DE MONEDA PARA TICKERS INTERNACIONALES (FACTOR FX)
# ==============================================================================

asset_currency = {}
if use_fx_factor:
    asset_currency = {t: get_currency_for_ticker(t) for t in selected_tickers}

currencies_presentes = sorted(set(
    v for v in asset_currency.values() if v != "USD" and v in fx_weekly_returns.columns
))
k_currencies = len(currencies_presentes)

n_con_fx = sum(1 for v in asset_currency.values() if v != "USD" and v in fx_weekly_returns.columns)
print(f"  OK Moneda no-USD con factor FX: {n_con_fx} de {n_sel} activos | {k_currencies} monedas distintas en el pool"
      + (f" ({', '.join(currencies_presentes)})" if k_currencies > 0 else ""))

# --- Betas de factores por activo: r_i ~ r_MKT + r_SECTOR + r_PAIS + r_FX ---
beta_hist_implied = {}
beta_sector_implied = {}
beta_country_implied = {}
beta_fx_implied = {}

r_mkt_full = benchmark_returns.iloc[:, 0]
var_spy_w = r_mkt_full.var()

for t in selected_tickers:
    r_i = log_returns_selected[t]
    sec_etf = asset_sector_etf.get(t)
    pais_etf = asset_country_etf.get(t)
    cur = asset_currency.get(t)
    fit_ok = False

    covariate_cols = {}
    if sec_etf is not None:
        covariate_cols["sec"] = log_returns[sec_etf].rename("sec")
    if pais_etf is not None:
        covariate_cols["pais"] = log_returns[pais_etf].rename("pais")
    if use_fx_factor and cur is not None and cur != "USD" and cur in fx_weekly_returns.columns:
        covariate_cols["fx"] = fx_weekly_returns[cur].rename("fx")

    if covariate_cols:
        aligned = pd.concat([r_i.rename(t), r_mkt_full.rename("mkt")] + list(covariate_cols.values()), axis=1).dropna()
        if len(aligned) >= 10:
            try:
                X = sm.add_constant(aligned[["mkt"] + list(covariate_cols.keys())])
                y = aligned[t]
                model = sm.OLS(y, X).fit()
                cf = model.params
                if not cf.isna().any() and len(cf) == len(covariate_cols) + 2:
                    beta_hist_implied[t] = cf["mkt"]
                    beta_sector_implied[t] = cf["sec"] if "sec" in covariate_cols else 0.0
                    beta_country_implied[t] = cf["pais"] if "pais" in covariate_cols else 0.0
                    beta_fx_implied[t] = cf["fx"] if "fx" in covariate_cols else 0.0
                    fit_ok = True
            except Exception:
                fit_ok = False

    if not fit_ok:
        aligned2 = pd.concat([r_i, r_mkt_full.rename("mkt")], axis=1).dropna()
        if len(aligned2) >= 6:
            beta_hist_implied[t] = aligned2[t].cov(aligned2["mkt"]) / var_spy_w
        else:
            beta_hist_implied[t] = 1.0
        beta_sector_implied[t] = 0.0
        beta_country_implied[t] = 0.0
        beta_fx_implied[t] = 0.0

beta_hist_arr = np.array([beta_hist_implied[t] for t in selected_tickers])
beta_sector_arr = np.array([beta_sector_implied[t] for t in selected_tickers])
beta_country_arr = np.array([beta_country_implied[t] for t in selected_tickers])
beta_fx_arr = np.array([beta_fx_implied[t] for t in selected_tickers])

print(f"     beta_mercado (primeros 3):        {beta_hist_arr[0]:.4f} | {beta_hist_arr[1]:.4f} | {beta_hist_arr[2]:.4f}")
print(f"     beta_sector (primeros 3):          {beta_sector_arr[0]:.4f} | {beta_sector_arr[1]:.4f} | {beta_sector_arr[2]:.4f}")
print(f"     beta_pais (primeros 3):            {beta_country_arr[0]:.4f} | {beta_country_arr[1]:.4f} | {beta_country_arr[2]:.4f}")
print(f"     beta_fx (primeros 3):              {beta_fx_arr[0]:.4f} | {beta_fx_arr[1]:.4f} | {beta_fx_arr[2]:.4f}")

# --- Matriz de factores F: MKT + cada sector presente + cada pais presente + cada moneda presente ---
factor_names = ["MKT"] + sectores_unicos + paises_unicos + currencies_presentes

factor_returns_mat = pd.DataFrame(index=log_returns.index, columns=factor_names, dtype=float)
factor_returns_mat["MKT"] = r_mkt_full.reindex(log_returns.index)
for s in sectores_unicos:
    factor_returns_mat[s] = log_returns[s]
for p in paises_unicos:
    factor_returns_mat[p] = log_returns[p]
for c in currencies_presentes:
    factor_returns_mat[c] = fx_weekly_returns[c].reindex(log_returns.index)

corr_factors = factor_returns_mat.corr()

iv_factor = {f: np.nan for f in factor_names}
iv_factor["MKT"] = iv_spy_implied
for s in sectores_unicos:
    iv_s = bkm_annual_vol(s) if use_iv_for_horizon else np.nan
    if pd.isna(iv_s) or iv_s <= 0:
        iv_s = log_returns[s].std() * math.sqrt(annualization_factor)
    iv_factor[s] = iv_s
for p in paises_unicos:
    iv_p = bkm_annual_vol(p) if use_iv_for_horizon else np.nan
    if pd.isna(iv_p) or iv_p <= 0:
        iv_p = log_returns[p].std() * math.sqrt(annualization_factor)
    iv_factor[p] = iv_p
for c in currencies_presentes:
    # Polygon no cubre opciones sobre pares FX - se usa volatilidad historica del par
    iv_c = fx_weekly_returns[c].std() * math.sqrt(annualization_factor)
    iv_factor[c] = iv_c

iv_factor_weekly = np.array([iv_factor[f] / math.sqrt(annualization_factor) for f in factor_names])
D_factor = np.diag(iv_factor_weekly)
F_mat = D_factor @ corr_factors.loc[factor_names, factor_names].values @ D_factor

# --- Matriz de cargas B y covarianza explicada por factores ------------------
B_load = np.zeros((n_sel, len(factor_names)))
mkt_idx = factor_names.index("MKT")
B_load[:, mkt_idx] = beta_hist_arr
for i, t in enumerate(selected_tickers):
    sec_etf = asset_sector_etf.get(t)
    if sec_etf is not None:
        j = factor_names.index(sec_etf)
        B_load[i, j] = beta_sector_implied[t]
    pais_etf = asset_country_etf.get(t)
    if pais_etf is not None:
        j = factor_names.index(pais_etf)
        B_load[i, j] = beta_country_implied[t]
    cur = asset_currency.get(t)
    if use_fx_factor and cur is not None and cur in currencies_presentes:
        j = factor_names.index(cur)
        B_load[i, j] = beta_fx_implied[t]

Cov_explained = B_load @ F_mat @ B_load.T

print(f"     diag(Cov_explained) rango (semanal): {np.diag(Cov_explained).min():.6f} - {np.diag(Cov_explained).max():.6f}")

# ==============================================================================
# VOLATILIDAD OBJETIVO POR ACTIVO: IV vs HISTORICA (shrinkage), o IV pura
# ==============================================================================
if use_iv_shrinkage:
    sd_hist_annual_weekly = sd_hist_annual / math.sqrt(annualization_factor)
    iv_hist_ratio = iv_final / sd_hist_annual

    dist_from_1 = np.abs(iv_hist_ratio - 1)
    shrink_frac = np.tanh(dist_from_1 / ratio_band)
    lambda_i = shrinkage_min + (shrinkage_max - shrinkage_min) * shrink_frac

    iv_final_weekly = iv_final / math.sqrt(annualization_factor)
    vol_final_weekly = (1 - lambda_i) * iv_final_weekly + lambda_i * sd_hist_annual_weekly
    vol_final_weekly = np.where(pd.isna(vol_final_weekly), sd_hist_annual_weekly, vol_final_weekly)

    print("\n  SHRINKAGE POR ACTIVO (vol implied vs historica, ancla = ratio IV/Hist):")
    print(f"     ratio IV/Hist (primeros 3):        {iv_hist_ratio[0]:.3f} | {iv_hist_ratio[1]:.3f} | {iv_hist_ratio[2]:.3f}")
    print(f"     lambda_i rango:                    {lambda_i.min():.3f} - {lambda_i.max():.3f}")
    print(f"     lambda_i promedio (peso historica): {lambda_i.mean():.3f}")
else:
    vol_final_weekly = iv_final / math.sqrt(annualization_factor)

print(f"     vol_final semanal rango:           {vol_final_weekly.min() * 100:.4f}% - {vol_final_weekly.max() * 100:.4f}%")

# --- Varianza idiosincratica --------------------------------------------------
explained_var = np.diag(Cov_explained)
idio_var = np.maximum(vol_final_weekly ** 2 - explained_var, 1e-8)

cov_mat = Cov_explained + np.diag(idio_var)
cov_mat = pd.DataFrame(cov_mat, index=selected_tickers, columns=selected_tickers)

print(f"     varianza explicada por mercado+sector+pais+fx (rango): {explained_var.min():.6f} - {explained_var.max():.6f}")
print(f"     varianza idiosincratica anadida (rango):       {idio_var.min():.6f} - {idio_var.max():.6f}")
print(f"     diag(cov_mat) final rango:         {np.diag(cov_mat.values).min():.6f} - {np.diag(cov_mat.values).max():.6f}")
print(f"     sqrt(diag(cov_mat)) = vol semanal: {np.sqrt(np.diag(cov_mat.values)).min() * 100:.4f}% - "
      f"{np.sqrt(np.diag(cov_mat.values)).max() * 100:.4f}%")

# ==============================================================================
# BLEND CON COVARIANZA HISTORICA SIMPLE + PENALIZACION DE RIESGO DE COLA (BKM)
# Sigma_modificada = Sigma_final + diag(alpha*MFIK - beta*MFIS)
# ==============================================================================
cov_hist_simple = log_returns_selected.cov().values

hist_shrink_alpha = 0.35

off_diag_factor = cov_mat.values - np.diag(np.diag(cov_mat.values))
off_diag_hist = cov_hist_simple - np.diag(np.diag(cov_hist_simple))
diff_offdiag = np.abs(off_diag_hist - off_diag_factor)
print(f"\n  BLEND CON COVARIANZA HISTORICA (alpha={hist_shrink_alpha:.2f}):")
print(f"     |cov_hist - cov_factor| fuera de diagonal, promedio: {diff_offdiag[np.triu_indices_from(diff_offdiag, k=1)].mean():.6f}")
print(f"     |cov_hist - cov_factor| fuera de diagonal, max:      {diff_offdiag[np.triu_indices_from(diff_offdiag, k=1)].max():.6f}")

cov_mat_values = (1 - hist_shrink_alpha) * cov_mat.values + hist_shrink_alpha * cov_hist_simple

mfik_diag = np.where(np.isnan(mfik_arr), 3.0, mfik_arr)
mfis_diag = np.where(np.isnan(mfis_arr), 0.0, mfis_arr)
typical_var = np.median(np.diag(cov_mat_values))
tail_penalty_diag = typical_var * (tail_risk_alpha * (mfik_diag - 3.0) - tail_risk_beta * mfis_diag)
cov_mat_values = cov_mat_values + np.diag(tail_penalty_diag)

cov_mat = pd.DataFrame(cov_mat_values, index=selected_tickers, columns=selected_tickers)

print(f"     diag(cov_mat) tras blend + penalizacion de cola, rango: "
      f"{np.diag(cov_mat.values).min():.6f} - {np.diag(cov_mat.values).max():.6f}")
print(f"     Penalizacion de cola (alpha={tail_risk_alpha:.3f}, beta={tail_risk_beta:.3f}, "
      f"typical_var={typical_var:.6f}) - diag(alpha*(MFIK-3)-beta*MFIS)*typical_var rango: "
      f"{tail_penalty_diag.min():.6f} - {tail_penalty_diag.max():.6f}")

eig_vals = np.linalg.eigvalsh(cov_mat.values)
print(f"  Eigenvalue minimo (Sigma modificada): {eig_vals.min():.6f}")
if not np.all(eig_vals >= -1e-8):
    print("  ADVERTENCIA Sigma modificada no es PSD - aplicando correccion Higham...")
    eig_full_vals, eig_full_vecs = np.linalg.eigh(cov_mat.values)
    eps_floor = 1e-6 * np.mean(np.diag(cov_mat.values))
    eig_full_vals = np.maximum(eig_full_vals, eps_floor)
    cov_mat = pd.DataFrame(
        eig_full_vecs @ np.diag(eig_full_vals) @ eig_full_vecs.T,
        index=selected_tickers, columns=selected_tickers
    )

print(f"  OK cov_mat final (Sigma_modificada = factores + blend historico + penalizacion de cola): "
      f"{cov_mat.shape[0]} x {cov_mat.shape[1]} activos")
print(f"  Rango MFIV anualizada: {iv_final.min() * 100:.1f}% - {iv_final.max() * 100:.1f}%")
print(f"  Rango beta_mercado: {beta_hist_arr.min():.3f} - {beta_hist_arr.max():.3f} | "
      f"Rango beta_sector: {beta_sector_arr.min():.3f} - {beta_sector_arr.max():.3f} | "
      f"Rango beta_pais: {beta_country_arr.min():.3f} - {beta_country_arr.max():.3f} | "
      f"Rango beta_fx: {beta_fx_arr.min():.3f} - {beta_fx_arr.max():.3f}")

# === OPTIMIZACION MINIMA VARIANZA (quadprog) =================================
n_etf_en_pool = sum(1 for t in selected_tickers if t in etf_universe_tickers)
if use_etf_constraint:
    print(f"[INFO] Restriccion de participacion ETF activa: {etf_min_weight * 100:.0f}%-{etf_max_weight * 100:.0f}% "
          f"del capital invertido")
    print(f"[INFO] ETFs/commodities en el pool tras filtros previos: {n_etf_en_pool} de {len(selected_tickers)} activos")
    if n_etf_en_pool == 0:
        print("  ADVERTENCIA Ningun ETF sobrevivio a los filtros de score/volatilidad/correlacion/")
        print("     estacionalidad/Delta - la restriccion de % ETF no tendra efecto.")
        print("     Revisa volatility_percentile, correlation_percentile o delta_min si")
        print("     quieres que mas ETFs lleguen a esta etapa.")

n_fx_en_pool = sum(1 for t in selected_tickers if get_currency_for_ticker(t) != "USD")
if use_fx_factor:
    print(f"[INFO] Restriccion de exposicion cambiaria activa: maximo {max_fx_exposure * 100:.0f}% "
          f"del capital invertido en tickers no-USD")
    print(f"[INFO] Tickers no-USD en el pool tras filtros previos: {n_fx_en_pool} de {len(selected_tickers)} activos")
    if n_fx_en_pool == 0:
        print("  ADVERTENCIA Ningun ticker no-USD sobrevivio a los filtros previos - la restriccion")
        print("     de exposicion cambiaria no tendra efecto.")

print(f"\n[INFO] Ejecutando optimizacion iterativa - maximo {max_assets_in_portfolio} activos...")
print("[INFO] Poda ETF/FX-aware activa (constraints_feasible)\n")


# chequeo de factibilidad estructural de las bandas ETF/FX: dado max_weight_per_asset,
# evita que la poda deje un subconjunto donde la banda de % ETF o el limite de exposicion
# cambiaria ya no puedan cumplirse matematicamente.
def constraints_feasible(tickers_list):
    if len(tickers_list) == 0:
        return False
    if use_etf_constraint:
        n_etf = sum(1 for t in tickers_list if t in etf_universe_tickers)
        n_stock = len(tickers_list) - n_etf
        if n_stock * max_weight_per_asset < (1 - etf_max_weight) - 1e-9:
            return False
        if n_etf * max_weight_per_asset < etf_min_weight - 1e-9:
            return False
    if use_fx_factor:
        n_fx = sum(1 for t in tickers_list if get_currency_for_ticker(t) != "USD")
        n_usd = len(tickers_list) - n_fx
        if n_usd * max_weight_per_asset < (1 - max_fx_exposure) - 1e-9:
            return False
    return True


def run_minvar_qp(tickers_subset):
    cm = cov_mat.loc[tickers_subset, tickers_subset].values
    n = len(tickers_subset)
    reg = 1e-6 * np.mean(np.diag(cm))
    Dm = 2 * cm + np.eye(n) * reg
    dv = np.zeros(n)

    if require_full_investment or math.isclose(min_total_weight, max_total_weight):
        target_total = 1.0 if require_full_investment else min_total_weight
        A_eq = np.ones((n, 1))
        b_eq = np.array([target_total])
        meq = 1
    else:
        A_eq = np.column_stack([np.ones(n), -np.ones(n)])
        b_eq = np.array([min_total_weight, -max_total_weight])
        meq = 0

    A_box = np.column_stack([np.eye(n), -np.eye(n)])
    b_box = np.concatenate([np.full(n, min_weight_per_asset), np.full(n, -max_weight_per_asset)])

    Amat_base = np.column_stack([A_eq, A_box])
    bvec_base = np.concatenate([b_eq, b_box])

    is_etf_vec = np.array([1.0 if t in etf_universe_tickers else 0.0 for t in tickers_subset])
    has_etf = is_etf_vec.sum() > 0

    is_fx_vec = np.array([1.0 if get_currency_for_ticker(t) != "USD" else 0.0 for t in tickers_subset])
    has_fx = is_fx_vec.sum() > 0

    extra_cols = []
    extra_b = []
    if use_etf_constraint and has_etf:
        a_etf_min = is_etf_vec - etf_min_weight
        a_etf_max = etf_max_weight - is_etf_vec
        extra_cols += [a_etf_min, a_etf_max]
        extra_b += [0.0, 0.0]
    if use_fx_factor and has_fx:
        a_fx_max = max_fx_exposure - is_fx_vec
        extra_cols += [a_fx_max]
        extra_b += [0.0]

    Amat_full = Amat_base
    bvec_full = bvec_base
    if extra_cols:
        Amat_full = np.column_stack([Amat_base] + extra_cols)
        bvec_full = np.concatenate([bvec_base, extra_b])

    def solve_attempt(Amat_try, bvec_try):
        try:
            sol = quadprog.solve_qp(Dm, dv, Amat_try, bvec_try, meq)
            w = np.maximum(sol[0], 0)
            return pd.Series(w, index=tickers_subset)
        except Exception as e:
            print(f"    ADVERTENCIA quadprog.solve_qp error: {e}")
            return None

    w = solve_attempt(Amat_full, bvec_full)

    if w is None and extra_cols:
        print("  ADVERTENCIA Restriccion de % ETF/exposicion cambiaria infactible para este subset - "
              "optimizando sin restricciones adicionales")
        w = solve_attempt(Amat_base, bvec_base)

    if w is None:
        print("  ADVERTENCIA quadprog fallo para subset")

    return w


# Contribucion Marginal al CVaR (MTR) via Cornish-Fisher: deriva analiticamente el CVaR_CF
# del portafolio (misma formula de Boudt-Peterson-Croux usada mas abajo en el reporte final)
# respecto a cada peso w_i, ponderada por w_i (estilo Euler), para podar por aporte marginal
# al riesgo de cola real en vez de solo la contribucion marginal a la varianza.
alpha_final = 1 - cornish_fisher_confidence
z_a_final = norm.ppf(alpha_final)

mfis_by_ticker = dict(zip(selected_tickers, mfis_diag))
mfik_by_ticker = dict(zip(selected_tickers, mfik_diag))


def compute_marginal_cvar_contrib(tickers_subset, w_vec, cm_sub):
    mu_vec = mean_ret.loc[tickers_subset].values
    mfis_vec = np.array([mfis_by_ticker[t] for t in tickers_subset])
    mfik_vec = np.array([mfik_by_ticker[t] for t in tickers_subset])

    w_sum = w_vec.sum()
    var_p = float(w_vec @ cm_sub @ w_vec)
    if w_sum <= 0 or var_p <= 1e-12:
        return pd.Series(w_vec * (cm_sub @ w_vec), index=tickers_subset)

    sigma_p = math.sqrt(var_p)
    S_p = float(np.sum(w_vec * mfis_vec) / w_sum)
    K_exc_p = float(np.sum(w_vec * mfik_vec) / w_sum) - 3.0

    S_p_clip = float(np.clip(S_p, -cornish_fisher_mfis_clip, cornish_fisher_mfis_clip))
    K_exc_p_clip = float(np.clip(K_exc_p, 0.0, cornish_fisher_mfik_clip))

    phi_za = norm.pdf(z_a_final)
    mes_alpha = (phi_za / alpha_final) * (
        1 + S_p_clip / 6 * z_a_final ** 2
        + K_exc_p_clip / 24 * (z_a_final ** 3 - 3 * z_a_final)
        - S_p_clip ** 2 / 36 * (2 * z_a_final ** 3 - 5 * z_a_final)
    )

    d_mes_dS = (phi_za / alpha_final) * (
        z_a_final ** 2 / 6 - S_p_clip * (2 * z_a_final ** 3 - 5 * z_a_final) / 18
    )
    d_mes_dK = (phi_za / alpha_final) * (z_a_final ** 3 - 3 * z_a_final) / 24

    d_sigma_dw = (cm_sub @ w_vec) / sigma_p
    d_S_dw = (mfis_vec - S_p) / w_sum
    d_Kexc_dw = (mfik_vec - (K_exc_p + 3.0)) / w_sum
    d_mes_dw = d_mes_dS * d_S_dw + d_mes_dK * d_Kexc_dw

    # Riesgo_p = -CVaR_p = -mu_p + MES_alpha * sigma_p
    d_risk_dw = -mu_vec + mes_alpha * d_sigma_dw + sigma_p * d_mes_dw

    return pd.Series(w_vec * d_risk_dw, index=tickers_subset)


current_tickers = list(selected_tickers)
iteration = 0
w_last_valid = None
tickers_last_valid = None

while True:
    iteration += 1
    w_iter = run_minvar_qp(current_tickers)

    if w_iter is None:
        print("  ADVERTENCIA Optimizacion fallo para este subset - usando la ultima solucion valida")
        break

    w_last_valid = w_iter
    tickers_last_valid = current_tickers

    active = w_iter[w_iter > 0.001]
    n_active = len(active)

    cm_iter = cov_mat.loc[current_tickers, current_tickers].values
    w_vec = w_iter.values
    vol_iter = math.sqrt(float(w_vec @ cm_iter @ w_vec))

    print(f"  Iteracion {iteration}: {n_active} activos activos | Vol semanal (tail-adj): {vol_iter * 100:.4f}%")

    if n_active <= max_assets_in_portfolio:
        break

    # Poda por Marginal Tail Risk (MTR) via CVaR, sin romper la banda ETF/FX
    mtr = compute_marginal_cvar_contrib(current_tickers, w_vec, cm_iter)
    mtr_active = mtr.loc[active.index].sort_values(ascending=False)

    drop_ticker = None
    for candidate in mtr_active.index:
        remaining = [t for t in current_tickers if t != candidate]
        if constraints_feasible(remaining):
            drop_ticker = candidate
            break
        else:
            print(f"    (saltando '{candidate}' como candidato a eliminar - "
                  f"dejaria la banda ETF/FX estructuralmente infactible)")

    if drop_ticker is None:
        drop_ticker = mtr_active.index[0]
        print("    ADVERTENCIA: ninguna eliminacion preserva la banda ETF/FX factible - "
              "eliminando por mayor MTR de todas formas")

    current_tickers = [t for t in current_tickers if t != drop_ticker]
    print(f"    -> Eliminando '{drop_ticker}' (MTR={mtr_active[drop_ticker]:.6f}, peso {active[drop_ticker] * 100:.3f}%)")

    if len(current_tickers) < 3:
        print("  ADVERTENCIA Quedan menos de 3 activos - deteniendo iteracion")
        break

if w_last_valid is None:
    raise RuntimeError(
        "Error: la optimizacion de minima varianza no encontro ninguna solucion factible.\n"
        "   Revisa: (1) que cov_mat sea PSD, (2) min_weight_per_asset * n_activos <= max_total_weight,\n"
        "   (3) max_weight_per_asset * n_activos >= min_total_weight, (4) la restriccion ETF\n"
        "   (etf_min_weight/etf_max_weight) si use_etf_constraint = True, y (5) la restriccion\n"
        "   de exposicion cambiaria (max_fx_exposure) si use_fx_factor = True."
    )

w_iter = w_last_valid
selected_tickers = w_iter[w_iter > 0.001].index.tolist()
w_final_implied = w_iter[selected_tickers]

print(f"\nOK Portafolio final: {len(selected_tickers)} activos (limite: {max_assets_in_portfolio})")
print(f"  Activos: {', '.join(selected_tickers)}\n")

log_returns_selected = log_returns[selected_tickers]
mean_ret = log_returns_selected.mean()
cov_mat = cov_mat.loc[selected_tickers, selected_tickers]
sd_ret = log_returns_selected.std()

opt_min_var = dict(weights=w_final_implied, selected=selected_tickers)

# ==============================================================================
# SECCION 9: FRONTERA EFICIENTE
# ==============================================================================

print("[INFO] Calculando Frontera Eficiente...")

min_ret = mean_ret.min() * horizon_weeks
max_ret = mean_ret.max() * horizon_weeks
target_returns = np.linspace(min_ret, max_ret, 50)

n = len(selected_tickers)
Dmat_ef = 2 * cov_mat.values
dvec_ef = np.zeros(n)

efficient_frontier_rows = []
for target_ret in target_returns:
    try:
        target_weekly = target_ret / horizon_weeks

        Amat = np.column_stack([
            np.eye(n),
            np.ones(n),
            -np.ones(n),
            mean_ret.values,
            -np.eye(n),
        ])
        bvec = np.concatenate([
            np.zeros(n),
            [min_total_weight],
            [-max_total_weight],
            [target_weekly],
            np.full(n, -max_weight_per_asset),
        ])

        sol = quadprog.solve_qp(Dmat_ef, dvec_ef, Amat, bvec, 0)
        w_sol = sol[0].copy()
        w_sol[w_sol < 1e-6] = 0

        ret = float(np.sum(w_sol * mean_ret.values)) * horizon_weeks
        risk = math.sqrt(float(w_sol @ cov_mat.values @ w_sol)) * horizon_sqrt

        efficient_frontier_rows.append(dict(Return=ret, Risk=risk))
    except Exception:
        pass

efficient_frontier = pd.DataFrame(efficient_frontier_rows)
print(f"[INFO] Frontera eficiente calculada con {len(efficient_frontier)} puntos.")

# ==============================================================================
# SECCION 10: EXTRACCION DE METRICAS
# ==============================================================================


def extract_metrics(opt_obj, label):
    w_raw = opt_obj["weights"]
    common = [t for t in w_raw.index if t in log_returns_selected.columns]
    w = w_raw[common]
    mr = mean_ret[common]
    cm = cov_mat.loc[common, common]
    ret_mat = log_returns_selected[common]

    total_weight = w.sum()
    cash_position = 1 - total_weight

    w_vec = w.values
    ret_horizon = float(np.sum(w_vec * mr.values)) * horizon_weeks
    risk_horizon = math.sqrt(float(w_vec @ cm.values @ w_vec)) * horizon_sqrt
    sharpe_horizon = (ret_horizon - rf_horizon) / risk_horizon

    ret_annual = float(np.sum(w_vec * mr.values)) * annualization_factor
    risk_annual = math.sqrt(float(w_vec @ cm.values @ w_vec)) * math.sqrt(annualization_factor)
    sharpe_annual = (ret_annual - risk_free_rate) / risk_annual

    ret_weekly = float(np.sum(w_vec * mr.values))
    risk_weekly = math.sqrt(float(w_vec @ cm.values @ w_vec))

    port_returns = ret_mat.dot(w)
    mdd = max_drawdown_from_returns(port_returns)

    port_excess = port_returns.values - risk_free_rate_weekly
    downside_neg = port_excess[port_excess < 0]
    downside_dev = math.sqrt(np.mean(downside_neg ** 2)) if len(downside_neg) else np.nan
    sortino_horizon = (ret_horizon - rf_horizon) / (downside_dev * horizon_sqrt)
    sortino_annual = (ret_annual - risk_free_rate) / (downside_dev * math.sqrt(annualization_factor))

    return dict(
        Label=label, Return_Horizon=ret_horizon, Risk_Horizon=risk_horizon,
        Sharpe_Horizon=sharpe_horizon, Sortino_Horizon=sortino_horizon, RF_Horizon=rf_horizon,
        Return_Annual=ret_annual, Risk_Annual=risk_annual, Sharpe_Annual=sharpe_annual,
        Sortino_Annual=sortino_annual, Return_Weekly=ret_weekly, Risk_Weekly=risk_weekly,
        MaxDrawdown=mdd, Weights=w, Total_Weight=total_weight, Cash_Position=cash_position,
    )


metrics_minvar = extract_metrics(opt_min_var, "Minima Varianza")

# ==============================================================================
# VaR/CVaR (Expected Shortfall) prospectivo global, Cornish-Fisher (BKM)
# Cornish-Fisher (1937): z_cf = z_a + (z_a^2-1)/6*S + (z_a^3-3z_a)/24*K_exc - (2z_a^3-5z_a)/36*S^2
# ES modificado (Boudt, Peterson & Croux, 2008):
#   MES_a = -phi(z_a)/a * (1 + S/6*z_a^2 + K_exc/24*(z_a^3-3z_a) - S^2/36*(2z_a^3-5z_a))
# ==============================================================================
alpha_final = 1 - cornish_fisher_confidence
z_a_final = norm.ppf(alpha_final)

w_final_bkm = metrics_minvar["Weights"]
mfis_final = np.array([bkm_moments_cache.get(t, {}).get("mfis", np.nan) for t in w_final_bkm.index])
mfik_final = np.array([bkm_moments_cache.get(t, {}).get("mfik", np.nan) for t in w_final_bkm.index])

mask_final_ok = np.isfinite(mfis_final) & np.isfinite(mfik_final)
w_final_vals = w_final_bkm.values

if mask_final_ok.sum() > 0 and w_final_vals[mask_final_ok].sum() > 0:
    w_norm_bkm = w_final_vals[mask_final_ok] / w_final_vals[mask_final_ok].sum()
    port_skew_final = float(np.sum(w_norm_bkm * mfis_final[mask_final_ok]))
    port_kurt_exc_final = float(np.sum(w_norm_bkm * mfik_final[mask_final_ok])) - 3.0

    # La expansion de Cornish-Fisher solo es valida (monotona) para desviaciones
    # moderadas de la normalidad; se acotan los momentos antes de usarlos en la formula.
    port_skew_cf = float(np.clip(port_skew_final, -cornish_fisher_mfis_clip, cornish_fisher_mfis_clip))
    port_kurt_exc_cf = float(np.clip(port_kurt_exc_final, 0.0, cornish_fisher_mfik_clip))

    port_sd_horizon = metrics_minvar["Risk_Horizon"]  # sigma diversificada (Sigma_modificada), no promedio de MFIV individuales
    port_mu_horizon = metrics_minvar["Return_Horizon"]

    z_cf_final = (z_a_final + (z_a_final ** 2 - 1) / 6 * port_skew_cf
                  + (z_a_final ** 3 - 3 * z_a_final) / 24 * port_kurt_exc_cf
                  - (2 * z_a_final ** 3 - 5 * z_a_final) / 36 * port_skew_cf ** 2)
    var_cf_final = port_mu_horizon + z_cf_final * port_sd_horizon

    mes_a_final = (norm.pdf(z_a_final) / alpha_final) * (
        1 + port_skew_cf / 6 * z_a_final ** 2
        + port_kurt_exc_cf / 24 * (z_a_final ** 3 - 3 * z_a_final)
        - port_skew_cf ** 2 / 36 * (2 * z_a_final ** 3 - 5 * z_a_final)
    )
    cvar_cf_final = port_mu_horizon - mes_a_final * port_sd_horizon
    cvar_cf_final = min(cvar_cf_final, var_cf_final)  # el CVaR nunca puede ser menos negativo que el VaR
else:
    var_cf_final, cvar_cf_final = np.nan, np.nan
    print("  ADVERTENCIA: sin cobertura BKM suficiente en el portafolio final - VaR/CVaR Cornish-Fisher = NaN")

# ==============================================================================
# SECCION 11: VISUALIZACIONES
# ==============================================================================

print("\n[INFO] Generando visualizaciones...")

df_points = pd.DataFrame({
    "Label": ["Minima Varianza"],
    "Risk": [metrics_minvar["Risk_Horizon"]],
    "Return": [metrics_minvar["Return_Horizon"]],
})

asset_metrics = pd.DataFrame({
    "Symbol": log_returns_selected.columns,
    "Return": mean_ret.values * horizon_weeks,
    "Risk": sd_ret.values * horizon_sqrt,
})

ef_sorted = efficient_frontier.sort_values("Risk") if len(efficient_frontier) > 0 else efficient_frontier
frontier_plot_df = pd.DataFrame({
    "Nombre": "Frontera eficiente", "Categoria": "Frontera eficiente",
    "Risk": ef_sorted["Risk"], "Return": ef_sorted["Return"],
}) if len(ef_sorted) > 0 else pd.DataFrame(columns=["Nombre", "Categoria", "Risk", "Return"])
activos_plot_df = pd.DataFrame({
    "Nombre": asset_metrics["Symbol"], "Categoria": "Activos individuales",
    "Risk": asset_metrics["Risk"], "Return": asset_metrics["Return"],
})
optimo_plot_df = pd.DataFrame({
    "Nombre": df_points["Label"], "Categoria": "Portafolio optimo",
    "Risk": df_points["Risk"], "Return": df_points["Return"],
})

fig = px.scatter(
    pd.concat([frontier_plot_df, activos_plot_df, optimo_plot_df], ignore_index=True),
    x="Risk", y="Return", color="Categoria", hover_name="Nombre",
    color_discrete_map={
        "Frontera eficiente": "#08519c",
        "Activos individuales": "#ff7f00",
        "Portafolio optimo": "#2ca25f",
    },
    labels={"Risk": f"Riesgo acumulado horizonte ({horizon_label})",
            "Return": f"Retorno esperado acumulado ({horizon_label})", "Categoria": ""},
)
if len(ef_sorted) > 0:
    fig.add_trace(go.Scatter(x=ef_sorted["Risk"], y=ef_sorted["Return"], mode="lines",
                              line=dict(color="#08519c", width=2), opacity=0.6,
                              name="Frontera eficiente", showlegend=False, hoverinfo="skip"))
fig.update_traces(marker=dict(size=6, opacity=0.5), selector=dict(name="Frontera eficiente"))
fig.update_traces(marker=dict(size=9, opacity=0.75), selector=dict(name="Activos individuales"))
fig.update_traces(marker=dict(size=16, line=dict(width=1.5, color="black")), selector=dict(name="Portafolio optimo"))
fig.update_layout(
    title=dict(text="Frontera Eficiente - Minima Varianza<br>"
                     f"<sup>Horizonte: {horizon_label} (~{horizon_weeks:.1f} sem) | Periodo datos: {start_date} -> "
                     f"{end_date:%Y-%m-%d} | {len(selected_tickers)} activos</sup>"),
    xaxis_tickformat=".1%", yaxis_tickformat=".1%",
    template="plotly_white",
)
fig.show()


def prepare_weights_with_cash(weights, total_weight, label):
    df = pd.DataFrame({"Symbol": weights.index, "Weight": weights.values, "Portfolio": label})
    if not require_full_investment and abs(1 - total_weight) > 0.001:
        df = pd.concat([df, pd.DataFrame({"Symbol": ["EFECTIVO"], "Weight": [1 - total_weight],
                                           "Portfolio": [label]})], ignore_index=True)
    return df


df_weights = prepare_weights_with_cash(metrics_minvar["Weights"], metrics_minvar["Total_Weight"], "Minima Varianza")

df_weights_plot = df_weights.sort_values("Weight", ascending=True)
fig = px.bar(
    df_weights_plot, x="Weight", y="Symbol", orientation="h",
    text=df_weights_plot["Weight"].map(lambda x: f"{x * 100:.1f}%"),
    labels={"Weight": "Peso del portafolio", "Symbol": ""},
    color_discrete_sequence=["#2ca25f"],
)
fig.update_traces(textposition="outside", marker_line_width=0,
                   hovertemplate="%{y}: %{x:.2%}<extra></extra>")
fig.update_layout(
    title=dict(text="Asignacion Optima de Activos - Minima Varianza<br>"
                     f"<sup>Horizonte: {horizon_label} | Peso max por activo: {max_weight_per_asset * 100:.0f}%</sup>"),
    xaxis_tickformat=".0%",
    template="plotly_white",
    showlegend=False,
    height=max(400, 24 * len(df_weights_plot)),
)
fig.show()

corr_matrix = log_returns_selected.corr()
try:
    dist = 1 - corr_matrix.abs()
    condensed = squareform(dist.values, checks=False)
    Z = linkage(condensed, method="average")
    order = leaves_list(Z)
    corr_ordered = corr_matrix.iloc[order, order]
except Exception:
    corr_ordered = corr_matrix

fig = px.imshow(
    corr_ordered.values, x=corr_ordered.columns, y=corr_ordered.columns,
    color_continuous_scale="RdBu", zmin=-1, zmax=1, text_auto=".2f", aspect="auto",
    labels=dict(color="Correlacion"),
)
fig.update_xaxes(tickangle=90)
fig.update_traces(textfont_size=8, hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>")
fig.update_layout(
    title=f"Matriz de Correlacion - {len(selected_tickers)} activos seleccionados",
    template="plotly_white",
)
fig.show()

# ==============================================================================
# SECCION 12: RESUMEN FINAL
# ==============================================================================

print("\n" + "=" * 65)
print("PORTAFOLIO OPTIMO - MINIMA VARIANZA")
print("=" * 65)
print(f"    Entrenado con : {n_weeks} semanas ({start_date} -> {end_date:%Y-%m-%d})")
print(f"    Horizonte     : {horizon_label} (~{horizon_weeks:.1f} semanas)")
print(f"    RF horizonte  : {rf_horizon * 100:.4f}%\n")

w_final = metrics_minvar["Weights"]
w_final = w_final[w_final > 0.001].sort_values(ascending=False)

print(f"  {'Ticker':<8}  {'Peso':>8}")
print("  " + "-" * 52)
for nm, val in w_final.items():
    barra = "#" * round(val * 100 / 2)
    print(f"  {nm:<8}  {val * 100:6.2f}%   {barra}")
print("  " + "-" * 52)
print(f"  {'TOTAL':<8}  {w_final.sum() * 100:6.2f}%\n")

if abs(metrics_minvar["Cash_Position"]) > 0.001:
    if metrics_minvar["Cash_Position"] > 0:
        print(f"  Posicion en efectivo: {metrics_minvar['Cash_Position'] * 100:.2f}%\n")
    else:
        print(f"  ADVERTENCIA Apalancamiento: {abs(metrics_minvar['Cash_Position']) * 100:.2f}%\n")

print(f"  Activos en portafolio : {len(w_final)}")
print(f"  Peso maximo           : {w_final.max() * 100:.2f}% ({w_final.index[0]})")
print(f"  Concentracion top 3   : {w_final.head(3).sum() * 100:.2f}%")

fx_weight_final = sum(w_final.get(t, 0.0) for t in w_final.index if get_currency_for_ticker(t) != "USD")
print(f"  Exposicion cambiaria  : {fx_weight_final * 100:.2f}% (limite: {max_fx_exposure * 100:.0f}%)")
print("=" * 65 + "\n")

# ==============================================================================
# ATRIBUCION DE RIESGO SINTETICA (GRIEGAS BLACK-SCHOLES) - DIAGNOSTICO
# ==============================================================================
print("ATRIBUCION DE RIESGO SINTETICA (GRIEGAS BSM) - DIAGNOSTICO")
print("=" * 65)

T_greeks = horizon_months / 12
mu_weekly_g = log_returns[list(w_final.index)].mean()

greeks_rows = []
for tk in w_final.index:
    S = spot_cache.get(tk, np.nan)
    iv = iv_cache.get(tk, np.nan)
    if iv is None or pd.isna(iv) or iv <= 0:
        iv = log_returns[tk].std() * math.sqrt(annualization_factor)
    if S is None or pd.isna(S) or S <= 0 or pd.isna(iv) or iv <= 0:
        greeks_rows.append(dict(Ticker=tk, Weight=w_final[tk], Delta=np.nan, Gamma=np.nan, Vega=np.nan, Theta=np.nan))
        continue

    mu_i = mu_weekly_g.get(tk, 0.0)
    mu_i = 0.0 if pd.isna(mu_i) else mu_i

    if delta_strike_mode == "atm":
        K = S
    elif delta_strike_mode == "rf":
        K = S * (1 + risk_free_rate * T_greeks)
    elif delta_strike_mode == "mu":
        K = S * (1 + mu_i * horizon_weeks)
    else:
        K = S

    d1 = (math.log(S / K) + (risk_free_rate + iv ** 2 / 2) * T_greeks) / (iv * math.sqrt(T_greeks))
    d2 = d1 - iv * math.sqrt(T_greeks)

    delta_i = norm.cdf(d1)
    gamma_raw = norm.pdf(d1) / (S * iv * math.sqrt(T_greeks))
    vega_raw = S * norm.pdf(d1) * math.sqrt(T_greeks) / 100
    theta_raw = (-(S * norm.pdf(d1) * iv) / (2 * math.sqrt(T_greeks))
                 - risk_free_rate * K * math.exp(-risk_free_rate * T_greeks) * norm.cdf(d2)) / 365

    gamma_i = gamma_raw / S
    vega_i = vega_raw / S
    theta_i = theta_raw / S

    greeks_rows.append(dict(Ticker=tk, Weight=w_final[tk], Delta=delta_i, Gamma=gamma_i, Vega=vega_i, Theta=theta_i))

greeks_df = pd.DataFrame(greeks_rows)

print(f"  {'Ticker':<8} {'Peso':>7} {'Delta':>8} {'Gamma/$':>10} {'Vega/$':>8} {'Theta/$':>9}")
print("  " + "-" * 58)
for _, row in greeks_df.iterrows():
    delta_str = "N/D" if pd.isna(row["Delta"]) else f"{row['Delta']:.3f}"
    gamma_str = "N/D" if pd.isna(row["Gamma"]) else f"{row['Gamma']:.6f}"
    vega_str = "N/D" if pd.isna(row["Vega"]) else f"{row['Vega']:.6f}"
    theta_str = "N/D" if pd.isna(row["Theta"]) else f"{row['Theta']:.6f}"
    print(f"  {row['Ticker']:<8} {row['Weight'] * 100:6.2f}% "
          f"{delta_str:>8} {gamma_str:>10} {vega_str:>8} {theta_str:>9}")
print("  " + "-" * 58)

delta_port = np.nansum(greeks_df["Weight"] * greeks_df["Delta"])
gamma_port = np.nansum(greeks_df["Weight"] * greeks_df["Gamma"])
vega_port = np.nansum(greeks_df["Weight"] * greeks_df["Vega"])
theta_port = np.nansum(greeks_df["Weight"] * greeks_df["Theta"])

print(f"  {'PORTF.':<8} {'':>7} {delta_port:8.3f} {gamma_port:10.6f} {vega_port:8.6f} {theta_port:9.6f}")
print("\n  Gamma/Vega/Theta normalizadas por precio del subyacente (por $1 de")
print("  exposicion, no por accion) para ser comparables entre tickers.")
print("  Vega por 1% de cambio en IV | Theta por dia | Delta sin normalizar (adimensional)")
print("=" * 65 + "\n")

# ==============================================================================
# ATRIBUCION DE RIESGO DE COLA (BKM)
# ==============================================================================
print("ATRIBUCION DE RIESGO DE COLA (BKM) - DIAGNOSTICO")
print("=" * 65)

tail_attr_rows = []
for tk in w_final.index:
    mom_tk = bkm_moments_cache.get(tk, {})
    mfis_tk = mom_tk.get("mfis", np.nan)
    mfik_tk = mom_tk.get("mfik", np.nan)
    diag_penalty_tk = typical_var * (tail_risk_alpha * ((mfik_tk if not pd.isna(mfik_tk) else 3.0) - 3.0)
                                      - tail_risk_beta * (mfis_tk if not pd.isna(mfis_tk) else 0.0))
    tail_attr_rows.append(dict(Ticker=tk, Weight=w_final[tk], MFIS=mfis_tk, MFIK=mfik_tk,
                                Diag_Penalty=diag_penalty_tk,
                                Weighted_Penalty=w_final[tk] * diag_penalty_tk))

tail_attr_df = pd.DataFrame(tail_attr_rows)
print(f"  {'Ticker':<8} {'Peso':>7} {'MFIS':>8} {'MFIK':>8} {'DiagPen':>10} {'Pen*Peso':>10}")
print("  " + "-" * 58)
for _, row in tail_attr_df.iterrows():
    mfis_str = "N/D" if pd.isna(row["MFIS"]) else f"{row['MFIS']:.3f}"
    mfik_str = "N/D" if pd.isna(row["MFIK"]) else f"{row['MFIK']:.3f}"
    print(f"  {row['Ticker']:<8} {row['Weight'] * 100:6.2f}% {mfis_str:>8} {mfik_str:>8} "
          f"{row['Diag_Penalty']:10.6f} {row['Weighted_Penalty']:10.6f}")
print("  " + "-" * 58)
print(f"  Penalizacion de cola ponderada del portafolio: {tail_attr_df['Weighted_Penalty'].sum():.6f}")
print(f"  VaR Cornish-Fisher ({cornish_fisher_confidence * 100:.0f}%, horizonte):  {var_cf_final * 100:.4f}%")
print(f"  CVaR Cornish-Fisher ({cornish_fisher_confidence * 100:.0f}%, horizonte): {cvar_cf_final * 100:.4f}%")
print("=" * 65 + "\n")

print("=" * 68)
print("              METRICAS DEL PORTAFOLIO OPTIMO")
print("=" * 68 + "\n")

summary_table = pd.DataFrame({
    "Metrica": [
        f"Retorno Esperado  ({horizon_label})",
        f"Volatilidad       ({horizon_label})",
        f"Sharpe Ratio      ({horizon_label})",
        f"Sortino Ratio     ({horizon_label})",
        "Retorno Esperado  (Anual, ref.)",
        "Volatilidad       (Anual, ref.)",
        "Sharpe Ratio      (Anual, ref.)",
        "Sortino Ratio     (Anual, ref.)",
        "Maximum Drawdown",
        "Retorno Semanal (raw)",
        "Volatilidad Semanal (raw)",
        "Sharpe Semanal (raw)",
        f"VaR Cornish-Fisher ({cornish_fisher_confidence * 100:.0f}%, horizonte)",
        f"CVaR Cornish-Fisher ({cornish_fisher_confidence * 100:.0f}%, horizonte)",
        "Peso Total Invertido",
        "Posicion en Efectivo",
        "Exposicion Cambiaria",
    ],
    "Valor": [
        f"{metrics_minvar['Return_Horizon'] * 100:.2f}%",
        f"{metrics_minvar['Risk_Horizon'] * 100:.2f}%",
        round(metrics_minvar["Sharpe_Horizon"], 3),
        round(metrics_minvar["Sortino_Horizon"], 3),
        f"{metrics_minvar['Return_Annual'] * 100:.2f}%",
        f"{metrics_minvar['Risk_Annual'] * 100:.2f}%",
        round(metrics_minvar["Sharpe_Annual"], 3),
        round(metrics_minvar["Sortino_Annual"], 3),
        f"{metrics_minvar['MaxDrawdown'] * 100:.2f}%",
        f"{metrics_minvar['Return_Weekly'] * 100:.2f}%",
        f"{metrics_minvar['Risk_Weekly'] * 100:.2f}%",
        round((metrics_minvar["Return_Weekly"] - risk_free_rate_weekly) / metrics_minvar["Risk_Weekly"], 3),
        f"{var_cf_final * 100:.4f}%" if pd.notna(var_cf_final) else "N/D",
        f"{cvar_cf_final * 100:.4f}%" if pd.notna(cvar_cf_final) else "N/D",
        f"{metrics_minvar['Total_Weight'] * 100:.2f}%",
        f"{metrics_minvar['Cash_Position'] * 100:.2f}%",
        f"{fx_weight_final * 100:.2f}%",
    ],
})

print(summary_table.to_string(index=False))

print("\n" + "=" * 65)
print(f"RESUMEN EJECUTIVO - EJECUTAR EN: {execution_label.upper()}")
print("=" * 65 + "\n")

for nm, val in w_final.items():
    print(f"   {nm:<8}  {val * 100:.2f}%")

print(f"\n   -- Metricas del horizonte ({horizon_label}) --")
print(f"   Retorno esperado  : {metrics_minvar['Return_Horizon'] * 100:.2f}%")
print(f"   Volatilidad       : {metrics_minvar['Risk_Horizon'] * 100:.2f}%")
print(f"   Sharpe Ratio      : {metrics_minvar['Sharpe_Horizon']:.3f}")
print(f"   Sortino Ratio     : {metrics_minvar['Sortino_Horizon']:.3f}")
print(f"   VaR Cornish-Fisher (95%)  : {var_cf_final * 100:.4f}%" if pd.notna(var_cf_final) else "   VaR Cornish-Fisher (95%)  : N/D")
print(f"   CVaR Cornish-Fisher (95%) : {cvar_cf_final * 100:.4f}%" if pd.notna(cvar_cf_final) else "   CVaR Cornish-Fisher (95%) : N/D")
print("\n   -- Referencia anual --")
print(f"   Retorno esperado  : {metrics_minvar['Return_Annual'] * 100:.2f}%")
print(f"   Volatilidad       : {metrics_minvar['Risk_Annual'] * 100:.2f}%")
print(f"   Sharpe Ratio      : {metrics_minvar['Sharpe_Annual']:.3f}")
print(f"   Sortino Ratio     : {metrics_minvar['Sortino_Annual']:.3f}")
print("=" * 65)

print("\nScript completado con exito!")
