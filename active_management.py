# ============================================================================
# BLOQUE 0: LIBRERIAS
# ============================================================================

import os
import time
import math
import warnings
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# BLOQUE 1: PARAMETROS CONFIGURABLES
# ============================================================================

portfolio = {"GLD": 0.18,
    "DHR": 0.18,
    "CASY": 0.18,
    "IBKR": 0.1443,
    "CDNS": 0.1014,
    "SLV": 0.096,
    "PDBC": 0.0722,
    "CME": 0.0385}

investment_horizon_days = 30

# ------------------------------------------------
# API KEY - Polygon.io
# ------------------------------------------------
from dotenv import load_dotenv
load_dotenv()
polygon_api_key = os.environ.get("POLYGON_API_KEY")
cash_reserve_limit      = 0.30

risk_free_rate           = 0.046
dividend_yield           = 0.00
otm_uoa_vol_oi_threshold = 3.0
strike_window_pct        = 0.15

POLYGON_BASE_URL = "https://api.polygon.io"

polygon_max_retries    = 2
polygon_retry_wait_sec = 15
polygon_min_interval_sec = 13.0  # espaciado global entre llamadas: el plan actual limita ~5/min
polygon_max_pages      = 40
polygon_page_limit     = 250

expiration_search_window_days = 20

uoa_min_volume             = 50
sweep_dominance_multiplier = 1.3

straddle_to_em_factor = 0.85

score_flip_scale            = 250
score_flip_cap               = 35
score_gex_weight             = 25
score_gex_intensity_divisor  = 1e6
score_gex_intensity_base     = 0.4
score_gex_intensity_scale    = 0.6
score_flow_sweep             = 25
score_flow_balanced          = 5
score_pcr_scale              = 15
score_pcr_cap                = 15
score_vanna_weight              = 10
score_vanna_intensity_divisor   = 1e6
score_vanna_intensity_base      = 0.4
score_vanna_intensity_scale     = 0.6
score_charm_weight              = 10
score_charm_intensity_divisor   = 1e6
score_charm_intensity_base      = 0.4
score_charm_intensity_scale     = 0.6
score_override_cap           = -51

gex_liquidity_min_notional   = 1_000_000
score_low_liquidity_damping  = 0.5

recorte_score_range = (-50, -1)
recorte_pct_range   = (0.50, 0.25)

score_threshold_aumentar = 50
score_threshold_mantener = 0
score_threshold_recortar = -50

# --- capa de riesgo de portafolio ---
corr_method             = "sample"       # "ewma" | "rmt" | "sample"
# Con N=8 activos y T~275 dias, la muestra cruda esta bien condicionada (T >> N):
# recupera relaciones conocidas (GLD-SLV ~0.8) que el shrinkage de EWMA/RMT, calibrado
# a la memoria corta del EWMA (T efectivo ~17 obs con lambda=0.94), aplana o inventa.
# Revisar si la cartera crece mucho en numero de activos.
corr_lookback_days      = 400
corr_ewma_lambda        = 0.94
corr_eig_floor          = 1e-8
corr_min_observations   = 20
atm_iv_n_strikes        = 6
history_request_pause_sec = 1.0
history_max_retries       = 4

ctr_cap_multiple        = 1.5            # techo de CTR = multiplo del reparto equiponderado
guardrail_max_iter      = 25
guardrail_damping       = 0.6
guardrail_tolerance     = 1e-4
guardrail_forced_passes = 10
guardrail_acciones_receptoras = ("AUMENTAR", "MANTENER")
guardrail_max_weight_growth   = 1.35   # un receptor no crece mas de 35% sobre su peso tactico

# --- historial de riesgo y regimen (persistencia entre corridas) ---
risk_history_path      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio_risk_history.csv")
risk_regime_min_history = 20   # corridas minimas antes de confiar en un percentil
risk_regime_alto_pct    = 80   # percentil de vol_portafolio por encima del cual se marca estres
risk_regime_bajo_pct    = 20   # percentil de ratio_diversificacion por debajo del cual se marca colapso

gamma_profile_colors = {"positivo": "#2E7D32", "negativo": "#C62828", "flip": "#1565C0"}
allocation_colors    = {"Actual": "#78909C", "Sugerido": "#1565C0"}
risk_layer_colors    = {"antes": "#B0BEC5", "despues": "#00695C", "techo": "#C62828"}

# ============================================================================
# BLOQUE 2: EXTRACCION Y PROCESAMIENTO DE OPCIONES (POLYGON API v3)
# ============================================================================

_last_polygon_call_ts = 0.0  # espaciado global entre llamadas, cualquier endpoint


def polygon_get(url, params=None, api_key=polygon_api_key, max_retries=polygon_max_retries):
    global _last_polygon_call_ts
    if params is None:
        params = {}
    params = dict(params)
    params["apiKey"] = api_key
    attempt = 0
    while True:
        attempt += 1
        espera = polygon_min_interval_sec - (time.time() - _last_polygon_call_ts)
        if espera > 0:
            time.sleep(espera)
        try:
            resp = requests.get(url, params=params, timeout=20)
        except requests.exceptions.RequestException as e:
            _last_polygon_call_ts = time.time()
            warnings.warn(f"Fallo de red en {url}: {e}")
            return None
        _last_polygon_call_ts = time.time()
        status = resp.status_code
        if status == 200:
            try:
                return resp.json()
            except ValueError:
                return None
        elif status == 429 and attempt <= max_retries:
            warnings.warn("Rate limit alcanzado (429). Esperando antes de reintentar...")
            time.sleep(polygon_retry_wait_sec)
            continue
        else:
            warnings.warn(f"Polygon API error [{status}] en {url}")
            return None


def get_options_snapshot(ticker, exp_date_from, exp_date_to, api_key=polygon_api_key):
    url = f"{POLYGON_BASE_URL}/v3/snapshot/options/{ticker}"
    params = {
        "expiration_date.gte": str(exp_date_from),
        "expiration_date.lte": str(exp_date_to),
        "limit": polygon_page_limit,
        "order": "asc",
        "sort": "strike_price",
    }

    all_results = []
    next_url = url
    next_params = params
    page_guard = 0

    while True:
        page_guard += 1
        if page_guard > polygon_max_pages:
            warnings.warn(f"Se alcanzo el limite de paginas de seguridad para {ticker}")
            break

        resp = polygon_get(next_url, params=next_params, api_key=api_key)
        if not resp or not resp.get("results"):
            break

        all_results.extend(resp["results"])

        next_url_val = resp.get("next_url")
        if next_url_val:
            next_url = next_url_val
            next_params = {"apiKey": api_key}
        else:
            break

    if not all_results:
        warnings.warn(f"Sin cadena de opciones disponible para {ticker} en el rango solicitado.")
        return None

    def get_path(d, path, default=np.nan):
        cur = d
        for p in path.split("."):
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
        return cur

    rows = []
    for r in all_results:
        rows.append({
            "ticker": ticker,
            "contract": get_path(r, "details.ticker", None),
            "strike": get_path(r, "details.strike_price"),
            "type": get_path(r, "details.contract_type", None),
            "expiration": get_path(r, "details.expiration_date", None),
            "volume": get_path(r, "day.volume", 0),
            "open_interest": get_path(r, "open_interest", 0),
            "iv": get_path(r, "implied_volatility"),
            "delta": get_path(r, "greeks.delta"),
            "gamma": get_path(r, "greeks.gamma"),
            "theta": get_path(r, "greeks.theta"),
            "vega": get_path(r, "greeks.vega"),
            "last_price": get_path(r, "day.close"),
            "bid": get_path(r, "last_quote.bid"),
            "ask": get_path(r, "last_quote.ask"),
            "underlying_price": get_path(r, "underlying_asset.price"),
        })

    chain = pd.DataFrame(rows)
    if chain.empty:
        return None

    for col in ["strike", "volume", "open_interest", "iv", "delta", "gamma", "theta",
                "vega", "last_price", "bid", "ask", "underlying_price"]:
        chain[col] = pd.to_numeric(chain[col], errors="coerce")

    chain["volume"] = chain["volume"].fillna(0)
    chain["open_interest"] = chain["open_interest"].fillna(0)
    chain = chain.dropna(subset=["strike", "type"])

    if chain.empty:
        return None

    return chain.reset_index(drop=True)


def get_spot_price(ticker, api_key=polygon_api_key):
    url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/prev"
    resp = polygon_get(url, api_key=api_key)
    if not resp or not resp.get("results"):
        warnings.warn(f"No se pudo obtener precio spot para {ticker}")
        return np.nan
    return float(resp["results"][0]["c"])


def select_target_expiration(ticker, horizon_days, api_key=polygon_api_key):
    target_date = date.today() + timedelta(days=math.ceil(horizon_days * 7 / 5))
    url = f"{POLYGON_BASE_URL}/v3/reference/options/contracts"
    params = {
        "underlying_ticker": ticker,
        "expiration_date.gte": str(date.today()),
        "expiration_date.lte": str(target_date + timedelta(days=expiration_search_window_days)),
        "limit": 1000,
        "order": "asc",
        "sort": "expiration_date",
    }
    resp = polygon_get(url, params=params, api_key=api_key)
    if not resp or not resp.get("results"):
        warnings.warn(f"No se encontraron contratos de opciones para {ticker}")
        return None

    expirations = sorted({r["expiration_date"] for r in resp["results"] if r.get("expiration_date")})
    if not expirations:
        return None

    expirations = [datetime.strptime(e, "%Y-%m-%d").date() for e in expirations]
    return min(expirations, key=lambda d: abs((d - target_date).days))

# ============================================================================
# BLOQUE 3: METRICAS DE MICROESTRUCTURA (GEX / ORDER FLOW / VANNA-CHARM)
# ============================================================================

def calculate_gex(chain, spot_price):
    if chain is None or chain.empty or pd.isna(spot_price):
        return None

    df = chain.dropna(subset=["gamma", "open_interest"]).copy()
    df["dealer_gex"] = np.where(
        df["type"] == "call",
        df["gamma"] * df["open_interest"] * 100 * spot_price**2 * 0.01,
        -df["gamma"] * df["open_interest"] * 100 * spot_price**2 * 0.01,
    )

    gex_by_strike = (
        df.groupby("strike", as_index=False)["dealer_gex"].sum()
        .rename(columns={"dealer_gex": "net_gex"})
        .sort_values("strike")
        .reset_index(drop=True)
    )
    gex_by_strike["cum_gex"] = gex_by_strike["net_gex"].cumsum()

    flip_level = np.nan
    # Se excluyen strikes sin exposicion real (net_gex == 0, tipicamente OI=0)
    # de la busqueda del cruce de signo. Sin este filtro, el cumsum se queda
    # en signo 0 durante los strikes sin OI y np.sign() detecta un "cambio"
    # espurio justo donde empiezan los primeros datos reales de la ventana
    # (comun en tickers de poca liquidez de opciones), en vez de un nivel
    # de flip genuino. Los valores de cum_gex usados si son los reales.
    gex_sig = gex_by_strike[gex_by_strike["net_gex"] != 0].reset_index(drop=True)
    if len(gex_sig) >= 2:
        signs = np.sign(gex_sig["cum_gex"].values)
        change_idx = np.where(np.diff(signs) != 0)[0]
        if len(change_idx) > 0:
            i = change_idx[0]
            x0, y0 = gex_sig["strike"].iloc[i], gex_sig["cum_gex"].iloc[i]
            x1, y1 = gex_sig["strike"].iloc[i + 1], gex_sig["cum_gex"].iloc[i + 1]
            flip_level = x0 - y0 * (x1 - x0) / (y1 - y0)

    total_gex = gex_by_strike["net_gex"].sum()
    total_abs_gex = gex_by_strike["net_gex"].abs().sum()

    return {
        "gex_by_strike": gex_by_strike,
        "total_gex": total_gex,
        "total_abs_gex": total_abs_gex,
        "gex_flip_level": flip_level,
        "gex_regime": "POSITIVO" if total_gex >= 0 else "NEGATIVO",
        "liquidity_confidence": "BAJA" if total_abs_gex < gex_liquidity_min_notional else "NORMAL",
    }


def calculate_order_flow(chain, spot_price, uoa_threshold=otm_uoa_vol_oi_threshold):
    if chain is None or chain.empty or pd.isna(spot_price):
        return None

    vol_calls = chain.loc[chain["type"] == "call", "volume"].sum()
    vol_puts  = chain.loc[chain["type"] == "put", "volume"].sum()
    oi_calls  = chain.loc[chain["type"] == "call", "open_interest"].sum()
    oi_puts   = chain.loc[chain["type"] == "put", "open_interest"].sum()

    pcr_volume = vol_puts / vol_calls if vol_calls > 0 else np.nan
    pcr_oi     = oi_puts / oi_calls if oi_calls > 0 else np.nan

    chain_otm = chain.copy()
    chain_otm["is_otm"] = np.where(
        chain_otm["type"] == "call",
        chain_otm["strike"] > spot_price,
        chain_otm["strike"] < spot_price,
    )
    chain_otm["vol_oi_ratio"] = np.where(
        chain_otm["open_interest"] > 0,
        chain_otm["volume"] / chain_otm["open_interest"],
        np.nan,
    )

    uoa_contracts = chain_otm[
        chain_otm["is_otm"]
        & chain_otm["vol_oi_ratio"].notna()
        & (chain_otm["vol_oi_ratio"] > uoa_threshold)
        & (chain_otm["volume"] >= uoa_min_volume)
    ].sort_values("vol_oi_ratio", ascending=False)

    fe = chain[
        chain["bid"].notna()
        & chain["ask"].notna()
        & (chain["ask"] > chain["bid"])
        & chain["last_price"].notna()
        & (chain["volume"] > 0)
    ].copy()
    fe["aggressiveness"] = (fe["last_price"] - fe["bid"]) / (fe["ask"] - fe["bid"])
    fe["buy_pressure_premium"] = np.where(fe["aggressiveness"] > 0.5, fe["volume"] * fe["last_price"] * 100, 0)
    fe["sell_pressure_premium"] = np.where(fe["aggressiveness"] <= 0.5, fe["volume"] * fe["last_price"] * 100, 0)

    call_sweep_premium = fe.loc[fe["type"] == "call", "buy_pressure_premium"].sum()
    put_sweep_premium  = fe.loc[fe["type"] == "put", "buy_pressure_premium"].sum()

    if (call_sweep_premium + put_sweep_premium) == 0:
        sweep_bias = "NEUTRAL"
    elif call_sweep_premium > put_sweep_premium * sweep_dominance_multiplier:
        sweep_bias = "CALL_SWEEP_DOMINANTE"
    elif put_sweep_premium > call_sweep_premium * sweep_dominance_multiplier:
        sweep_bias = "PUT_SWEEP_DOMINANTE"
    else:
        sweep_bias = "BALANCEADO"

    return {
        "pcr_volume": pcr_volume,
        "pcr_oi": pcr_oi,
        "uoa_contracts": uoa_contracts,
        "n_uoa_calls": int((uoa_contracts["type"] == "call").sum()),
        "n_uoa_puts": int((uoa_contracts["type"] == "put").sum()),
        "call_sweep_premium": call_sweep_premium,
        "put_sweep_premium": put_sweep_premium,
        "sweep_bias": sweep_bias,
    }


def _bsm_d1_d2(S, K, T_years, r, q, sigma):
    if any(pd.isna(x) for x in [S, K, T_years, r, q, sigma]) or sigma <= 0 or T_years <= 0 or S <= 0 or K <= 0:
        return np.nan, np.nan
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T_years) / (sigma * np.sqrt(T_years))
    d2 = d1 - sigma * np.sqrt(T_years)
    return d1, d2


def calculate_vanna_charm(chain, spot_price, r=risk_free_rate, q=dividend_yield):
    if chain is None or chain.empty or pd.isna(spot_price):
        return None

    today = date.today()
    df = chain[chain["iv"].notna() & (chain["iv"] > 0) & chain["expiration"].notna()].copy()
    if df.empty:
        return None

    df["T_years"] = df["expiration"].apply(
        lambda e: max((datetime.strptime(e, "%Y-%m-%d").date() - today).days, 1) / 365
    )

    d1_list, d2_list = [], []
    for _, row in df.iterrows():
        d1, d2 = _bsm_d1_d2(spot_price, row["strike"], row["T_years"], r, q, row["iv"])
        d1_list.append(d1)
        d2_list.append(d2)
    df["d1"] = d1_list
    df["d2"] = d2_list
    df["phi_d1"] = norm.pdf(df["d1"])

    df["vanna_unit"] = -np.exp(-q * df["T_years"]) * df["phi_d1"] * df["d2"] / df["iv"]
    df["vanna_dollar"] = df["vanna_unit"] * df["open_interest"] * 100 * spot_price * 0.01

    df["charm_unit"] = (
        -np.exp(-q * df["T_years"]) * df["phi_d1"]
        * ((2 * (r - q) * df["T_years"] - df["d2"] * df["iv"] * np.sqrt(df["T_years"]))
           / (2 * df["T_years"] * df["iv"] * np.sqrt(df["T_years"])))
        / 365
    )
    df["charm_dollar"] = df["charm_unit"] * df["open_interest"] * 100 * spot_price

    df["vanna_dollar"] = np.where(df["type"] == "call", df["vanna_dollar"], -df["vanna_dollar"])
    df["charm_dollar"] = np.where(df["type"] == "call", df["charm_dollar"], -df["charm_dollar"])

    net_vanna = df["vanna_dollar"].sum()
    net_charm = df["charm_dollar"].sum()

    return {
        "net_vanna_exposure": net_vanna,
        "net_charm_exposure": net_charm,
        "vanna_regime": "COBERTURA_PRO_VOLATILIDAD" if net_vanna >= 0 else "COBERTURA_ANTI_VOLATILIDAD",
        "charm_regime": "SOPORTE_POR_DECAIMIENTO" if net_charm >= 0 else "PRESION_BAJISTA_POR_DECAIMIENTO",
    }


def calculate_expected_move(chain, spot_price):
    if chain is None or chain.empty or pd.isna(spot_price):
        return None

    atm_strike = chain.loc[(chain["strike"] - spot_price).abs().idxmin(), "strike"]
    atm_chain = chain[chain["strike"] == atm_strike]

    atm_call = atm_chain[atm_chain["type"] == "call"].head(1)
    atm_put = atm_chain[atm_chain["type"] == "put"].head(1)

    em_dollars = np.nan
    method = None

    call_px = atm_call["last_price"].iloc[0] if len(atm_call) == 1 and pd.notna(atm_call["last_price"].iloc[0]) else np.nan
    put_px = atm_put["last_price"].iloc[0] if len(atm_put) == 1 and pd.notna(atm_put["last_price"].iloc[0]) else np.nan

    if pd.notna(call_px) and pd.notna(put_px):
        em_dollars = (call_px + put_px) * straddle_to_em_factor
        method = "STRADDLE_ATM"
    else:
        iv_vals = [v for v in [
            atm_call["iv"].iloc[0] if len(atm_call) else np.nan,
            atm_put["iv"].iloc[0] if len(atm_put) else np.nan,
        ] if pd.notna(v)]
        iv_atm = np.mean(iv_vals) if iv_vals else np.nan
        exp_date_str = atm_chain["expiration"].iloc[0]
        exp_date = datetime.strptime(exp_date_str, "%Y-%m-%d").date() if pd.notna(exp_date_str) else None
        T_years = (exp_date - date.today()).days / 365 if exp_date else np.nan
        if pd.notna(iv_atm) and iv_atm > 0 and pd.notna(T_years) and T_years > 0:
            em_dollars = spot_price * iv_atm * np.sqrt(T_years)
            method = "IV_ATM_FALLBACK"

    if pd.isna(em_dollars):
        return None

    return {
        "expected_move_dollars": em_dollars,
        "expected_move_pct": em_dollars / spot_price,
        "upper_bound": spot_price + em_dollars,
        "lower_bound": spot_price - em_dollars,
        "method": method,
    }


def analyze_ticker_options(ticker, horizon_days, api_key=polygon_api_key):
    print(f"Analizando flujo de opciones para {ticker}...")

    try:
        spot = get_spot_price(ticker, api_key)
        target_exp = select_target_expiration(ticker, horizon_days, api_key)
        if pd.isna(spot) or target_exp is None:
            raise ValueError("Datos insuficientes de spot o expiracion.")

        chain = get_options_snapshot(ticker, exp_date_from=target_exp, exp_date_to=target_exp, api_key=api_key)
        if chain is None or chain.empty:
            raise ValueError("Cadena de opciones vacia para la expiracion objetivo.")

        if chain["underlying_price"].isna().any():
            chain["underlying_price"] = chain["underlying_price"].fillna(spot)
        if pd.isna(spot):
            spot = chain["underlying_price"].mean()

        chain_windowed = chain[
            (chain["strike"] >= spot * (1 - strike_window_pct))
            & (chain["strike"] <= spot * (1 + strike_window_pct))
        ]

        gex_res = calculate_gex(chain_windowed, spot)
        flow_res = calculate_order_flow(chain, spot)
        vc_res = calculate_vanna_charm(chain, spot)
        em_res = calculate_expected_move(chain, spot)

        return {
            "ticker": ticker,
            "spot_price": spot,
            "expiration": target_exp,
            "chain": chain,
            "chain_windowed": chain_windowed,
            "gex": gex_res,
            "flow": flow_res,
            "vanna_charm": vc_res,
            "expected_move": em_res,
            "status": "OK",
        }
    except Exception as e:
        warnings.warn(f"Fallo el analisis de {ticker}: {e}")
        return {"ticker": ticker, "status": "ERROR", "error_message": str(e)}

# ============================================================================
# BLOQUE 4: ALGORITMO DE DECISION Y SCORING TACTICO
# ============================================================================

def calculate_tactical_score(analysis):
    if analysis is None or analysis.get("status") != "OK":
        return {
            "ticker": analysis.get("ticker") if analysis else None,
            "score": np.nan,
            "action": "SIN_DATOS",
            "rationale": analysis.get("error_message", "Datos no disponibles") if analysis else "Datos no disponibles",
        }

    spot = analysis["spot_price"]
    gex = analysis["gex"]
    flow = analysis["flow"]
    em = analysis["expected_move"]
    vc = analysis["vanna_charm"]

    score = 0.0
    reasons = []

    if gex is not None and pd.notna(gex["gex_flip_level"]):
        dist_pct = (spot - gex["gex_flip_level"]) / spot
        flip_points = max(min(dist_pct * score_flip_scale, score_flip_cap), -score_flip_cap)
        score += flip_points
        reasons.append(f"Spot {spot:.2f} vs Flip {gex['gex_flip_level']:.2f} ({dist_pct*100:.1f}% dist.)")

    if gex is not None and pd.notna(gex["total_gex"]):
        gex_points = score_gex_weight if gex["total_gex"] > 0 else -score_gex_weight
        intensity = min(abs(gex["total_gex"]) / (spot * score_gex_intensity_divisor), 1)
        gex_points = gex_points * (score_gex_intensity_base + score_gex_intensity_scale * intensity)
        score += gex_points
        reasons.append(f"GEX total {gex['gex_regime']} ({gex['total_gex']/1e6:.2f}MM)")

    if flow is not None:
        if flow["sweep_bias"] == "CALL_SWEEP_DOMINANTE":
            flow_points = score_flow_sweep
        elif flow["sweep_bias"] == "PUT_SWEEP_DOMINANTE":
            flow_points = -score_flow_sweep
        elif flow["sweep_bias"] == "BALANCEADO":
            flow_points = score_flow_balanced
        else:
            flow_points = 0
        score += flow_points
        reasons.append(f"Flujo: {flow['sweep_bias']} (UOA calls={flow['n_uoa_calls']}, puts={flow['n_uoa_puts']})")

    if flow is not None and pd.notna(flow["pcr_volume"]):
        pcr_points = max(min((1 - flow["pcr_volume"]) * score_pcr_scale, score_pcr_cap), -score_pcr_cap)
        score += pcr_points
        reasons.append(f"PCR volumen = {flow['pcr_volume']:.2f}")

    if vc is not None and pd.notna(vc["net_vanna_exposure"]):
        vanna_points = score_vanna_weight if vc["net_vanna_exposure"] >= 0 else -score_vanna_weight
        intensity = min(abs(vc["net_vanna_exposure"]) / (spot * score_vanna_intensity_divisor), 1)
        vanna_points = vanna_points * (score_vanna_intensity_base + score_vanna_intensity_scale * intensity)
        score += vanna_points
        reasons.append(f"Vanna: {vc['vanna_regime']} ({vc['net_vanna_exposure']/1e6:.2f}MM)")

    if vc is not None and pd.notna(vc["net_charm_exposure"]):
        charm_points = score_charm_weight if vc["net_charm_exposure"] >= 0 else -score_charm_weight
        intensity = min(abs(vc["net_charm_exposure"]) / (spot * score_charm_intensity_divisor), 1)
        charm_points = charm_points * (score_charm_intensity_base + score_charm_intensity_scale * intensity)
        score += charm_points
        reasons.append(f"Charm: {vc['charm_regime']} ({vc['net_charm_exposure']/1e6:.2f}MM)")

    liquidity_confidence = gex["liquidity_confidence"] if gex is not None else "NORMAL"
    if liquidity_confidence == "BAJA":
        score = score * score_low_liquidity_damping
        reasons.append(
            f"Confianza BAJA: notional GEX ${gex['total_abs_gex']:,.0f} < umbral "
            f"${gex_liquidity_min_notional:,.0f} (poca liquidez de opciones) - score amortiguado x{score_low_liquidity_damping}"
        )

    score = max(min(score, 100), -100)

    max_pain_break = gex is not None and pd.notna(gex["gex_flip_level"]) and spot < gex["gex_flip_level"]
    aggressive_put_sweep = flow is not None and flow["sweep_bias"] == "PUT_SWEEP_DOMINANTE"
    if max_pain_break and aggressive_put_sweep and gex is not None and gex["total_gex"] < 0:
        score = min(score, score_override_cap)
        reasons.append("OVERRIDE: precio bajo Flip Level + GEX negativo + Put Sweep agresivo")

    if score >= score_threshold_aumentar:
        action = "AUMENTAR"
    elif score >= score_threshold_mantener:
        action = "MANTENER"
    elif score >= score_threshold_recortar:
        action = "RECORTAR"
    else:
        action = "LIQUIDAR"

    if action == "RECORTAR":
        lo_s, hi_s = recorte_score_range
        lo_p, hi_p = recorte_pct_range
        recorte_sugerido = lo_p + (score - lo_s) * (hi_p - lo_p) / (hi_s - lo_s)
    else:
        recorte_sugerido = np.nan

    return {
        "ticker": analysis["ticker"],
        "spot_price": spot,
        "score": round(score, 1),
        "action": action,
        "recorte_pct": recorte_sugerido,
        "gex_flip_level": gex["gex_flip_level"] if gex is not None else np.nan,
        "total_gex": gex["total_gex"] if gex is not None else np.nan,
        "liquidity_confidence": liquidity_confidence,
        "sweep_bias": flow["sweep_bias"] if flow is not None else None,
        "expected_move_lower": em["lower_bound"] if em is not None else np.nan,
        "expected_move_upper": em["upper_bound"] if em is not None else np.nan,
        "expected_move_pct": em["expected_move_pct"] if em is not None else np.nan,
        "rationale": " | ".join(reasons),
    }

# ============================================================================
# BLOQUE 5: CAPA DE RIESGO DE PORTAFOLIO (CORRELACION Y ATRIBUCION DE EULER)
# Convierte los analisis por ticker en una vista de portafolio: volatilidad
# implicita por activo desde la cadena, matriz de correlacion historica
# (EWMA / RMT), covarianza y atribucion de riesgo por descomposicion de Euler.
# La correlacion entra por una sola matriz para poder sustituirla mas adelante
# por una correlacion implicita despejada de opciones sobre indice.
# ============================================================================

def get_price_history(tickers, lookback_days=corr_lookback_days, api_key=polygon_api_key):
    end = date.today()
    start = end - timedelta(days=lookback_days)
    series = {}

    for i, tk in enumerate(tickers):
        if i > 0:
            time.sleep(history_request_pause_sec)  # el plan gratuito de Polygon limita por minuto
        url = (f"{POLYGON_BASE_URL}/v2/aggs/ticker/{tk}/range/1/day/"
               f"{start.isoformat()}/{end.isoformat()}")
        resp = polygon_get(url, params={"adjusted": "true", "sort": "asc", "limit": 50000},
                           api_key=api_key, max_retries=history_max_retries)
        rows = (resp or {}).get("results") or []
        if not rows:
            warnings.warn(f"Sin historico de precios para {tk}: queda fuera de la capa de riesgo.")
            continue
        idx = pd.to_datetime([r["t"] for r in rows], unit="ms").normalize()
        series[tk] = pd.Series([float(r["c"]) for r in rows], index=idx, name=tk)

    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).dropna()


def _cov_to_corr(cov):
    d = np.sqrt(np.clip(np.diag(cov), 1e-16, None))
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def _nearest_pd_corr(corr, eig_floor=corr_eig_floor):
    corr = 0.5 * (corr + corr.T)
    vals, vecs = np.linalg.eigh(corr)
    vals = np.clip(vals, eig_floor, None)
    return _cov_to_corr(vecs @ np.diag(vals) @ vecs.T)


def _ewma_cov(rets, lam=corr_ewma_lambda):
    t = rets.shape[0]
    w = (1.0 - lam) * lam ** np.arange(t - 1, -1, -1)
    w /= w.sum()
    x = rets - np.average(rets, axis=0, weights=w)
    return (x * w[:, None]).T @ x


def _rmt_filter(corr, t):
    # Random Matrix Theory: los autovalores por debajo del borde de
    # Marchenko-Pastur son ruido muestral y se colapsan a su promedio. El "t"
    # debe ser el tamano muestral EFECTIVO de la matriz que se filtra, no el
    # numero de filas crudo: una matriz EWMA con lambda alto viene de muchas
    # menos observaciones independientes de lo que sugiere su historico.
    n = corr.shape[0]
    if t <= n + 1:
        warnings.warn(
            f"T efectivo = {t:.1f} <= N + 1 = {n+1}: con este numero de activos y esta "
            "vida media del EWMA no hay observaciones independientes suficientes para "
            "separar senal de ruido. Solo se proyecta a definida positiva (sin filtrar)."
        )
        return _nearest_pd_corr(corr)
    vals, vecs = np.linalg.eigh(corr)
    lam_plus = (1.0 + math.sqrt(n / t)) ** 2
    noise = vals < lam_plus
    noise[-1] = False  # el modo de mercado siempre se conserva
    if noise.sum() > 0:
        vals = vals.copy()
        vals[noise] = vals[noise].sum() / noise.sum()
    return _nearest_pd_corr(vecs @ np.diag(vals) @ vecs.T)


def _ewma_effective_n(lam):
    # Vida media efectiva de un EWMA: cuantas observaciones independientes
    # equivalen a su ventana de memoria (RiskMetrics, misma convencion que
    # se usa para fijar el tamano de ventana de una vol EWMA).
    return 1.0 / (1.0 - lam)


def estimate_correlation(prices, method=corr_method):
    n = prices.shape[1]
    if n <= 1:
        return np.ones((max(n, 1), max(n, 1)))

    rets = np.log(prices.astype(float)).diff().dropna().values
    if rets.shape[0] < corr_min_observations:
        warnings.warn(f"Menos de {corr_min_observations} retornos: se asume correlacion nula (identidad).")
        return np.eye(n)

    if method == "sample":
        corr = _cov_to_corr(np.cov(rets, rowvar=False))
    elif method == "ewma":
        corr = _cov_to_corr(_ewma_cov(rets))
    elif method == "rmt":
        t_efectivo = min(_ewma_effective_n(corr_ewma_lambda), rets.shape[0])
        corr = _rmt_filter(_cov_to_corr(_ewma_cov(rets)), t_efectivo)
    else:
        raise ValueError(f"Metodo de correlacion desconocido: {method}")

    return _nearest_pd_corr(corr)


def implied_sigma_annual(analysis, n_strikes=atm_iv_n_strikes):
    # Volatilidad implicita anualizada ATM, ponderada por open interest sobre
    # los strikes mas cercanos al spot. Es la pieza forward-looking del modelo.
    if analysis is None or analysis.get("status") != "OK":
        return np.nan

    chain = analysis.get("chain")
    spot = analysis.get("spot_price")
    if chain is None or chain.empty or pd.isna(spot):
        return np.nan

    valid = chain[(chain["iv"] > 0) & chain["iv"].notna()].copy()
    if valid.empty:
        return np.nan

    valid["dist"] = (valid["strike"] - spot).abs()
    near = valid.nsmallest(n_strikes, "dist")
    if near.empty:
        return np.nan

    pesos = near["open_interest"].fillna(0).to_numpy(dtype=float) + 1.0
    iv = near["iv"].to_numpy(dtype=float)
    sigma = float(np.average(iv, weights=pesos))

    if not np.isfinite(sigma) or sigma <= 0:
        return np.nan
    return sigma


def historical_sigma_annual(prices, ticker, trading_days_per_year=252):
    if prices.empty or ticker not in prices.columns:
        return np.nan
    rets = np.log(prices[ticker].astype(float)).diff().dropna()
    if len(rets) < corr_min_observations:
        return np.nan
    return float(rets.std(ddof=1) * math.sqrt(trading_days_per_year))


def euler_risk_attribution(tickers, weights, sigma, corr):
    # Descomposicion de Euler: sigma_p es homogenea de grado 1 en los pesos,
    # asi que MCR_i = d sigma_p / d w_i y los CTR_i suman sigma_p exacto.
    w = np.asarray(weights, dtype=float)
    sig = np.asarray(sigma, dtype=float)

    cov = np.outer(sig, sig) * corr
    var_p = float(w @ cov @ w)
    sig_p = math.sqrt(max(var_p, 1e-16))

    mcr = (cov @ w) / sig_p
    ctr = w * mcr
    ctr_pct = ctr / sig_p if sig_p > 0 else np.full_like(ctr, np.nan)

    vol_media_ponderada = float(np.abs(w) @ sig)
    ratio_div = vol_media_ponderada / sig_p if sig_p > 0 else np.nan

    num = var_p - float(np.sum(w**2 * sig**2))
    den = 2.0 * float(np.sum(np.triu(np.outer(w, w) * np.outer(sig, sig), k=1)))
    corr_implicita = num / den if abs(den) > 1e-14 else np.nan

    tabla = pd.DataFrame({
        "Ticker": list(tickers),
        "Peso": w,
        "Vol_Individual": 100.0 * sig,
        "MCR": mcr,
        "CTR_pts": 100.0 * ctr,
        "CTR_Pct": 100.0 * ctr_pct,
        "Rho_vs_Portafolio": np.divide(mcr, sig, out=np.full_like(mcr, np.nan), where=sig > 0),
    })
    tabla["Intensidad_Riesgo"] = np.divide(
        tabla["CTR_Pct"].to_numpy(), 100.0 * w,
        out=np.full(len(w), np.nan), where=w > 1e-12)

    metricas = {
        "vol_portafolio": 100.0 * sig_p,
        "vol_media_ponderada": 100.0 * vol_media_ponderada,
        "ratio_diversificacion": ratio_div,
        "beneficio_diversificacion_pts": 100.0 * (vol_media_ponderada - sig_p),
        "correlacion_implicita_media": corr_implicita,
        "exposicion_riesgosa": float(w.sum()),
    }
    return tabla, metricas, cov


def build_risk_inputs(analyses, tickers, api_key=polygon_api_key):
    prices = get_price_history(tickers, api_key=api_key)

    # Un activo entra en la capa de riesgo solo si tiene volatilidad Y
    # historico: sin historico no hay fila ni columna en la correlacion. Se
    # excluye ese activo y el resto conserva su matriz real; su peso queda
    # intacto porque el guardrail solo escribe sobre los tickers que evalua.
    sigmas, fuentes, validos, excluidos = [], [], [], []
    for tk in tickers:
        if tk not in prices.columns:
            excluidos.append((tk, "sin historico de precios"))
            continue
        s = implied_sigma_annual(analyses.get(tk))
        fuente = "IMPLICITA_ATM"
        if pd.isna(s):
            s = historical_sigma_annual(prices, tk)
            fuente = "HISTORICA_FALLBACK"
        if pd.isna(s):
            excluidos.append((tk, "sin volatilidad implicita ni historica"))
            continue
        validos.append(tk)
        sigmas.append(s)
        fuentes.append(fuente)

    for tk, motivo in excluidos:
        warnings.warn(f"{tk} queda fuera de la capa de riesgo ({motivo}): su peso no se ajusta.")

    if len(validos) < 2:
        warnings.warn("Menos de 2 activos utilizables: se omite la capa de riesgo de portafolio.")
        return None

    return {
        "tickers": validos,
        "sigma": np.asarray(sigmas, dtype=float),
        "fuente_vol": fuentes,
        "corr": estimate_correlation(prices[validos]),
        "prices": prices,
        "excluidos": excluidos,
    }


def apply_diversification_guardrail(tabla_rebalanceo, risk_inputs,
                                     cap_multiple=ctr_cap_multiple,
                                     max_iter=guardrail_max_iter,
                                     damping=guardrail_damping):
    # Recalcula la atribucion de riesgo con los pesos que propuso el scoring
    # tactico y recorta cualquier nombre que supere el techo de CTR. El peso
    # liberado va primero a los activos con holgura de riesgo y solo el
    # remanente a caja.
    if risk_inputs is None:
        return tabla_rebalanceo, None

    tickers = risk_inputs["tickers"]
    sigma = risk_inputs["sigma"]
    corr = risk_inputs["corr"]

    tabla = tabla_rebalanceo.copy()
    fila_cash = tabla["Ticker"] == "CASH"
    idx = {tk: tabla.index[tabla["Ticker"] == tk][0]
           for tk in tickers if (tabla["Ticker"] == tk).any()}
    tickers = [tk for tk in tickers if tk in idx]
    if len(tickers) < 2:
        return tabla_rebalanceo, None

    pos = [risk_inputs["tickers"].index(tk) for tk in tickers]
    sigma = sigma[pos]
    corr = corr[np.ix_(pos, pos)]

    w0 = np.array([float(tabla.at[idx[tk], "Nuevo_Peso"]) for tk in tickers])
    acciones = np.array([str(tabla.at[idx[tk], "Accion"]) for tk in tickers])

    tabla_antes, metricas_antes, cov = euler_risk_attribution(tickers, w0, sigma, corr)

    def ctr_share(pesos):
        sig_p = math.sqrt(max(float(pesos @ cov @ pesos), 1e-16))
        return (pesos * ((cov @ pesos) / sig_p)) / sig_p

    cap = cap_multiple / len(tickers)
    ctr_inicial = ctr_share(w0)
    w = w0.copy()
    caja_extra = 0.0
    forzados = []

    for _ in range(max_iter):
        ctr_pct = ctr_share(w)
        exceso = ctr_pct > cap + guardrail_tolerance
        if not exceso.any():
            break

        w_new = w.copy()
        w_new[exceso] = w[exceso] * (cap / ctr_pct[exceso]) ** damping
        liberado = float((w - w_new).sum())

        # El peso liberado va a los activos con holgura de riesgo que el
        # scoring tactico no queria reducir. Un activo con CTR negativo (una
        # cobertura genuina) tiene la holgura mas alta y absorberia todo el
        # reparto, asi que ademas se limita cuanto puede crecer cada receptor
        # sobre su peso tactico. Lo que no cabe termina en caja.
        receptor = (~exceso) & np.isin(acciones, guardrail_acciones_receptoras) & (w_new > 1e-6)
        holgura = np.where(receptor, np.clip(cap - ctr_pct, 0.0, None), 0.0)
        capacidad = np.where(receptor, np.clip(w0 * guardrail_max_weight_growth - w_new, 0.0, None), 0.0)

        if holgura.sum() > 1e-12:
            asignacion = np.minimum(liberado * holgura / holgura.sum(), capacidad)
            w_new = w_new + asignacion
            caja_extra += liberado - float(asignacion.sum())
        else:
            caja_extra += liberado
        w = w_new

    # Pase final: lo que siga por encima del techo se recorta contra caja. El
    # escalado es lineal y el CTR no lo es, asi que se repite hasta cerrar.
    for _ in range(guardrail_forced_passes):
        ctr_pct = ctr_share(w)
        exceso = ctr_pct > cap + guardrail_tolerance
        if not exceso.any():
            break
        w_forzado = w.copy()
        w_forzado[exceso] = w[exceso] * (cap / ctr_pct[exceso])
        caja_extra += float((w - w_forzado).sum())
        forzados = sorted(set(forzados) | {tickers[i] for i in np.flatnonzero(exceso)})
        w = w_forzado

    ctr_final = ctr_share(w)
    bitacora = []
    for i, tk in enumerate(tickers):
        if abs(w[i] - w0[i]) < 1e-6:
            continue
        marca = " (recorte forzado)" if tk in forzados else ""
        bitacora.append(
            f"{tk}: CTR {100*ctr_inicial[i]:.1f}% -> {100*ctr_final[i]:.1f}% "
            f"| peso {100*w0[i]:.2f}% -> {100*w[i]:.2f}%{marca}")

    for tk, peso in zip(tickers, w):
        tabla.at[idx[tk], "Nuevo_Peso"] = peso
    if caja_extra > 0:
        if fila_cash.any():
            tabla.loc[fila_cash, "Nuevo_Peso"] = tabla.loc[fila_cash, "Nuevo_Peso"] + caja_extra
        else:
            warnings.warn("Sin fila CASH donde aparcar el peso liberado: la renormalizacion "
                          "revierte parte del recorte del guardrail.")

    suma = tabla["Nuevo_Peso"].sum(skipna=True)
    if abs(suma - 1) > 1e-6 and suma > 0:
        tabla["Nuevo_Peso"] = tabla["Nuevo_Peso"] / suma

    w_final = np.array([float(tabla.at[idx[tk], "Nuevo_Peso"]) for tk in tickers])
    tabla_despues, metricas_despues, _ = euler_risk_attribution(tickers, w_final, sigma, corr)

    reporte = {
        "cap_ctr": 100.0 * cap,
        "atribucion_antes": tabla_antes,
        "atribucion_despues": tabla_despues,
        "metricas_antes": metricas_antes,
        "metricas_despues": metricas_despues,
        "correlacion": pd.DataFrame(corr, index=tickers, columns=tickers),
        "fuente_vol": dict(zip(risk_inputs["tickers"], risk_inputs["fuente_vol"])),
        "excluidos": risk_inputs.get("excluidos", []),
        "caja_por_guardrail": caja_extra,
        "bitacora": bitacora,
    }
    return tabla, reporte


def cargar_historial_riesgo():
    if os.path.exists(risk_history_path):
        return pd.read_csv(risk_history_path, parse_dates=["fecha"])
    return pd.DataFrame()


def evaluar_regimen_riesgo(risk_report, hist_df):
    # Compara la corrida de HOY contra el historial previo (sin incluir la fila
    # de hoy): un VIX_port o un ratio de diversificacion solo dicen algo si se
    # leen contra su propia historia, no como nivel aislado.
    if risk_report is None:
        return None

    md = risk_report["metricas_despues"]
    n_obs = len(hist_df)
    resultado = {
        "n_observaciones": n_obs,
        "vol_portafolio": md["vol_portafolio"],
        "ratio_diversificacion": md["ratio_diversificacion"],
        "alertas": [],
    }

    if n_obs < risk_regime_min_history:
        resultado["estado"] = "HISTORIAL_INSUFICIENTE"
        resultado["mensaje"] = (
            f"{n_obs} corrida(s) registrada(s); se necesitan al menos "
            f"{risk_regime_min_history} para que un percentil sea confiable."
        )
        return resultado

    serie_vol = hist_df["vol_portafolio_despues"].dropna()
    serie_ratio = hist_df["ratio_diversificacion_despues"].dropna()
    pct_vol = float((serie_vol < md["vol_portafolio"]).mean() * 100)
    pct_ratio = float((serie_ratio < md["ratio_diversificacion"]).mean() * 100)

    resultado["estado"] = "OK"
    resultado["percentil_vol_portafolio"] = pct_vol
    resultado["percentil_ratio_diversificacion"] = pct_ratio

    if pct_vol >= risk_regime_alto_pct:
        resultado["alertas"].append(
            f"REGIMEN DE ESTRES: vol_portafolio ({md['vol_portafolio']:.2f}%) en el percentil "
            f"{pct_vol:.0f} de sus ultimas {n_obs} corridas."
        )
    if pct_ratio <= risk_regime_bajo_pct:
        resultado["alertas"].append(
            f"DIVERSIFICACION COMPRIMIDA: ratio_diversificacion ({md['ratio_diversificacion']:.2f}) "
            f"en el percentil {pct_ratio:.0f} (mas bajo de lo usual en sus ultimas {n_obs} corridas)."
        )
    return resultado


def registrar_historial_riesgo(risk_report, hist_df):
    if risk_report is None:
        warnings.warn("Sin capa de riesgo en esta corrida: no se registra en el historial.")
        return hist_df

    ma, md = risk_report["metricas_antes"], risk_report["metricas_despues"]
    fila = pd.DataFrame([{
        "fecha": datetime.now(),
        "corr_method": corr_method,
        "cash_reserve_limit": cash_reserve_limit,
        "vol_portafolio_antes": ma["vol_portafolio"],
        "vol_portafolio_despues": md["vol_portafolio"],
        "vol_media_ponderada_despues": md["vol_media_ponderada"],
        "ratio_diversificacion_antes": ma["ratio_diversificacion"],
        "ratio_diversificacion_despues": md["ratio_diversificacion"],
        "beneficio_diversificacion_pts_despues": md["beneficio_diversificacion_pts"],
        "correlacion_implicita_media_despues": md["correlacion_implicita_media"],
        "exposicion_riesgosa_despues": md["exposicion_riesgosa"],
    }])

    hist_actualizado = pd.concat([hist_df, fila], ignore_index=True)
    # Si ya se corrio hoy, se queda solo la fila mas reciente en vez de duplicarla
    hist_actualizado["fecha_dia"] = pd.to_datetime(hist_actualizado["fecha"]).dt.date
    hist_actualizado = hist_actualizado.drop_duplicates(subset=["fecha_dia"], keep="last")
    hist_actualizado = hist_actualizado.drop(columns=["fecha_dia"]).sort_values("fecha")
    hist_actualizado.to_csv(risk_history_path, index=False)
    return hist_actualizado


def print_risk_regime(regimen):
    if regimen is None:
        return
    print("\n-- Regimen de riesgo (contra historial propio) --")
    if regimen["estado"] == "HISTORIAL_INSUFICIENTE":
        print(f"  {regimen['mensaje']}")
        return
    print(f"  vol_portafolio: {regimen['vol_portafolio']:.2f}% (percentil {regimen['percentil_vol_portafolio']:.0f} "
          f"de {regimen['n_observaciones']} corridas)")
    print(f"  ratio_diversificacion: {regimen['ratio_diversificacion']:.2f} (percentil "
          f"{regimen['percentil_ratio_diversificacion']:.0f})")
    if regimen["alertas"]:
        for a in regimen["alertas"]:
            print(f"  [!] {a}")
    else:
        print("  Sin alertas: ambas metricas dentro de su rango historico normal.")


# ============================================================================
# BLOQUE 6: REBALANCEO DEL PORTAFOLIO Y ASIGNACION DE CASH
# ============================================================================

def rebalance_portfolio(portfolio, scores_df, cash_reserve_limit):
    base = scores_df.copy()
    base["peso_inicial"] = base["ticker"].map(portfolio)

    def peso_liberado_row(row):
        if row["action"] == "LIQUIDAR":
            return row["peso_inicial"]
        elif row["action"] == "RECORTAR":
            return row["peso_inicial"] * row["recorte_pct"]
        else:
            return 0.0

    base["peso_liberado"] = base.apply(peso_liberado_row, axis=1)
    base["peso_retenido"] = base["peso_inicial"] - base["peso_liberado"]

    capital_liberado = base["peso_liberado"].sum(skipna=True)

    cash_asignado = min(capital_liberado, cash_reserve_limit)
    remanente_para_aumentar = capital_liberado - cash_asignado

    aumentar_idx = base.index[base["action"] == "AUMENTAR"]
    if len(aumentar_idx) > 0 and remanente_para_aumentar > 0:
        pesos_score = base.loc[aumentar_idx, "score"].clip(lower=1)
        distrib = remanente_para_aumentar * (pesos_score / pesos_score.sum())
        base.loc[aumentar_idx, "peso_retenido"] = base.loc[aumentar_idx, "peso_retenido"] + distrib
    else:
        cash_asignado = cash_asignado + remanente_para_aumentar
        remanente_para_aumentar = 0

    tabla_rebalanceo = base.rename(columns={
        "ticker": "Ticker",
        "peso_inicial": "Peso_Inicial",
        "score": "Score_Tactico",
        "action": "Accion",
        "peso_retenido": "Nuevo_Peso",
        "expected_move_lower": "Rango_Bajo_USD",
        "expected_move_upper": "Rango_Alto_USD",
        "expected_move_pct": "Movimiento_Esperado_Pct",
        "rationale": "Racional",
    })[["Ticker", "Peso_Inicial", "Score_Tactico", "Accion", "Nuevo_Peso",
        "Rango_Bajo_USD", "Rango_Alto_USD", "Movimiento_Esperado_Pct", "Racional"]]

    fila_cash = pd.DataFrame([{
        "Ticker": "CASH",
        "Peso_Inicial": 0,
        "Score_Tactico": np.nan,
        "Accion": "RESERVA_TACTICA",
        "Nuevo_Peso": cash_asignado,
        "Rango_Bajo_USD": np.nan,
        "Rango_Alto_USD": np.nan,
        "Movimiento_Esperado_Pct": np.nan,
        "Racional": f"Limite configurado: {cash_reserve_limit*100:.0f}%",
    }])

    tabla_final = pd.concat([tabla_rebalanceo, fila_cash], ignore_index=True)

    suma_final = tabla_final["Nuevo_Peso"].sum(skipna=True)
    if abs(suma_final - 1) > 1e-6 and suma_final > 0:
        tabla_final["Nuevo_Peso"] = tabla_final["Nuevo_Peso"] / suma_final

    return tabla_final

# ============================================================================
# BLOQUE 7: SALIDA GRAFICA Y REPORTE
# ============================================================================

def generate_executive_dashboard(tabla_rebalanceo):
    tabla_fmt = tabla_rebalanceo.copy()
    tabla_fmt["Peso_Inicial"] = tabla_fmt["Peso_Inicial"].apply(lambda x: f"{x*100:.1f}%")
    tabla_fmt["Nuevo_Peso"] = tabla_fmt["Nuevo_Peso"].apply(lambda x: f"{x*100:.1f}%")
    tabla_fmt["Movimiento_Esperado_Pct"] = tabla_rebalanceo["Movimiento_Esperado_Pct"].apply(
        lambda x: "-" if pd.isna(x) else f"{x*100:.1f}%")
    tabla_fmt["Rango_USD"] = tabla_rebalanceo.apply(
        lambda r: "-" if pd.isna(r["Rango_Bajo_USD"]) else f"${r['Rango_Bajo_USD']:.2f} - ${r['Rango_Alto_USD']:.2f}",
        axis=1)
    tabla_fmt["Score_Tactico"] = tabla_rebalanceo["Score_Tactico"].apply(
        lambda x: "-" if pd.isna(x) else f"{x:.1f}")

    tabla_fmt = tabla_fmt[["Ticker", "Peso_Inicial", "Score_Tactico", "Accion",
                            "Nuevo_Peso", "Movimiento_Esperado_Pct", "Rango_USD", "Racional"]]

    print("\n================ MOTOR DE GESTION ACTIVA - RESUMEN EJECUTIVO ================")
    print(f"Horizonte: {investment_horizon_days} dias habiles | Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    display_cols = ["Ticker", "Peso_Inicial", "Score_Tactico", "Accion",
                     "Nuevo_Peso", "Movimiento_Esperado_Pct", "Rango_USD"]
    rename_display = {
        "Peso_Inicial": "Peso Inicial", "Score_Tactico": "Score",
        "Nuevo_Peso": "Nuevo Peso", "Movimiento_Esperado_Pct": "Mov. Esperado", "Rango_USD": "Rango ($)"
    }
    print(tabla_fmt[display_cols].rename(columns=rename_display).to_string(index=False))

    print("\n-- Racional por activo --")
    for _, row in tabla_fmt.iterrows():
        print(f"[{row['Ticker']}] {row['Racional']}")
    print("===============================================================================")

    return tabla_fmt


def plot_gamma_profiles(analyses):
    valid = {tk: an for tk, an in analyses.items()
             if an is not None and an.get("status") == "OK" and an.get("gex") is not None}
    if not valid:
        return None

    tickers_ = list(valid.keys())
    n = len(tickers_)
    cols = min(3, n)
    rows = math.ceil(n / cols)

    subplot_titles = [
        f"{tk} | Exp: {valid[tk]['expiration']} | {valid[tk]['gex']['gex_regime']}"
        for tk in tickers_
    ]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)

    for i, tk in enumerate(tickers_):
        analysis = valid[tk]
        r, c = i // cols + 1, i % cols + 1

        gex_df = analysis["gex"]["gex_by_strike"]
        spot = analysis["spot_price"]
        flip = analysis["gex"]["gex_flip_level"]
        em = analysis["expected_move"]

        colors = [gamma_profile_colors["positivo"] if v > 0 else gamma_profile_colors["negativo"]
                  for v in gex_df["net_gex"]]

        fig.add_trace(go.Bar(x=gex_df["strike"], y=gex_df["net_gex"] / 1e6, marker_color=colors,
                              showlegend=False, hovertemplate="Strike %{x}: %{y:.2f} $MM<extra></extra>"),
                      row=r, col=c)

        fig.add_vline(x=spot, line_color="black", annotation_text=f"Spot: {spot:.2f}",
                      annotation_position="top left", row=r, col=c)

        if pd.notna(flip):
            fig.add_vline(x=flip, line_color=gamma_profile_colors["flip"], line_dash="dash",
                          annotation_text=f"Flip: {flip:.2f}", annotation_position="bottom left",
                          annotation_font_color=gamma_profile_colors["flip"], row=r, col=c)

        if em is not None:
            fig.add_vrect(x0=em["lower_bound"], x1=em["upper_bound"], fillcolor="orange",
                          opacity=0.08, line_width=0, row=r, col=c)

        fig.update_xaxes(title_text="Strike", row=r, col=c)
        fig.update_yaxes(title_text="Net GEX ($MM)", row=r, col=c)

    fig.update_layout(
        title="Perfil de Exposicion Gamma (GEX) por Activo",
        template="plotly_white",
        height=380 * rows,
        showlegend=False,
    )
    return fig


def plot_allocation_comparison(tabla_rebalanceo):
    df = tabla_rebalanceo[["Ticker", "Peso_Inicial", "Nuevo_Peso"]].copy()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Ticker"], y=df["Peso_Inicial"], name="Actual",
                          marker_color=allocation_colors["Actual"],
                          text=df["Peso_Inicial"].map(lambda x: f"{x * 100:.1f}%"), textposition="outside",
                          hovertemplate="%{x} Actual: %{y:.2%}<extra></extra>"))
    fig.add_trace(go.Bar(x=df["Ticker"], y=df["Nuevo_Peso"], name="Sugerido",
                          marker_color=allocation_colors["Sugerido"],
                          text=df["Nuevo_Peso"].map(lambda x: f"{x * 100:.1f}%"), textposition="outside",
                          hovertemplate="%{x} Sugerido: %{y:.2%}<extra></extra>"))
    fig.update_layout(
        title="Cambios en la Asignacion del Portafolio",
        yaxis_title="Peso del Portafolio", yaxis_tickformat=".0%",
        barmode="group", template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig

def print_risk_report(reporte):
    if reporte is None:
        print("\n[Capa de riesgo de portafolio no disponible: datos insuficientes.]")
        return

    ma, md = reporte["metricas_antes"], reporte["metricas_despues"]

    print("\n================ CAPA DE RIESGO DE PORTAFOLIO ================")
    print(f"Techo de CTR por activo: {reporte['cap_ctr']:.1f}% del riesgo total "
          f"({ctr_cap_multiple:.1f}x el reparto equiponderado)\n")

    resumen = pd.DataFrame({
        "Metrica": ["Vol portafolio (anual, %)", "Vol media ponderada (%)",
                     "Ratio de diversificacion", "Beneficio diversificacion (pts)",
                     "Correlacion implicita media", "Exposicion riesgosa (%)"],
        "Antes": [ma["vol_portafolio"], ma["vol_media_ponderada"], ma["ratio_diversificacion"],
                   ma["beneficio_diversificacion_pts"], ma["correlacion_implicita_media"],
                   100 * ma["exposicion_riesgosa"]],
        "Despues": [md["vol_portafolio"], md["vol_media_ponderada"], md["ratio_diversificacion"],
                     md["beneficio_diversificacion_pts"], md["correlacion_implicita_media"],
                     100 * md["exposicion_riesgosa"]],
    })
    print(resumen.round(4).to_string(index=False))

    print("\n-- Atribucion de riesgo con los pesos sugeridos --")
    cols = ["Ticker", "Peso", "Vol_Individual", "MCR", "CTR_pts", "CTR_Pct",
            "Rho_vs_Portafolio", "Intensidad_Riesgo"]
    print(reporte["atribucion_despues"][cols].sort_values("CTR_Pct", ascending=False)
          .round(4).to_string(index=False))

    print("\n-- Fuente de la volatilidad por activo --")
    print(" | ".join(f"{tk}: {src}" for tk, src in reporte["fuente_vol"].items()))

    if reporte["excluidos"]:
        print("\n-- Activos FUERA de la capa de riesgo (peso sin ajustar) --")
        for tk, motivo in reporte["excluidos"]:
            print(f"  {tk}: {motivo}")
        print("  Los CTR% de arriba se reparten solo entre los activos evaluados.")

    if reporte["bitacora"]:
        print("\n-- Ajustes aplicados por el guardrail de concentracion --")
        for linea in reporte["bitacora"]:
            print(f"  {linea}")
        print(f"  Caja adicional generada por el guardrail: {100*reporte['caja_por_guardrail']:.2f}%")
    else:
        print("\n-- Guardrail: ningun activo supero el techo de CTR; no hubo ajustes. --")
    print("==============================================================")


def plot_risk_attribution(reporte):
    if reporte is None:
        return None

    antes = reporte["atribucion_antes"].set_index("Ticker")
    despues = reporte["atribucion_despues"].set_index("Ticker")
    orden = despues["CTR_Pct"].sort_values(ascending=False).index
    labels = list(orden)

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.58, 0.42],
        subplot_titles=("Contribucion al riesgo total (CTR %)",
                        f"Correlacion ({corr_method.upper()}) entre activos"),
    )

    fig.add_trace(go.Bar(x=labels, y=antes.loc[orden, "CTR_Pct"], name="Antes del guardrail",
                          marker_color=risk_layer_colors["antes"],
                          hovertemplate="%{x} antes: %{y:.2f}% del riesgo<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=despues.loc[orden, "CTR_Pct"], name="Despues del guardrail",
                          marker_color=risk_layer_colors["despues"],
                          customdata=np.stack([100 * despues.loc[orden, "Peso"],
                                                despues.loc[orden, "Intensidad_Riesgo"]], axis=-1),
                          hovertemplate="%{x} despues: %{y:.2f}% del riesgo"
                                        "<br>peso = %{customdata[0]:.2f}%"
                                        "<br>intensidad = %{customdata[1]:.2f}x<extra></extra>"),
                  row=1, col=1)
    fig.add_hline(y=reporte["cap_ctr"], line_color=risk_layer_colors["techo"], line_dash="dash",
                  annotation_text=f"Techo {reporte['cap_ctr']:.1f}%", annotation_position="top right",
                  row=1, col=1)
    fig.add_hline(y=100.0 / len(labels), line_color="#9E9E9E", line_dash="dot",
                  annotation_text="Risk parity", annotation_position="bottom right",
                  row=1, col=1)

    corr = reporte["correlacion"].loc[labels, labels]
    fig.add_trace(go.Heatmap(z=corr.values, x=labels, y=labels, zmid=0, zmin=-1, zmax=1,
                              colorscale="RdBu_r", showscale=True,
                              hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>"),
                  row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_yaxes(title_text="% del riesgo total", row=1, col=1)

    fig.update_layout(
        title="Riesgo de Portafolio: Concentracion y Correlacion",
        template="plotly_white", barmode="group", height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.29),
    )
    return fig

# ============================================================================
# BLOQUE 8: ORQUESTACION PRINCIPAL
# ============================================================================

def run_active_management_engine(portfolio, horizon_days, api_key, cash_limit):
    tickers = list(portfolio.keys())

    analyses = {tk: analyze_ticker_options(tk, horizon_days, api_key) for tk in tickers}

    scores_list = [calculate_tactical_score(analyses[tk]) for tk in tickers]
    scores_df = pd.DataFrame(scores_list)

    tabla_tactica = rebalance_portfolio(portfolio, scores_df, cash_limit)

    # El scoring tactico es transversal y ciego a la correlacion: la capa de
    # riesgo revisa su propuesta y recorta las concentraciones de riesgo.
    print("\nConstruyendo la capa de riesgo de portafolio...")
    risk_inputs = build_risk_inputs(analyses, tickers, api_key)
    tabla_rebalanceo, risk_report = apply_diversification_guardrail(tabla_tactica, risk_inputs)

    dashboard = generate_executive_dashboard(tabla_rebalanceo)
    print_risk_report(risk_report)

    # El regimen se evalua contra el historial PREVIO (sin la corrida de hoy)
    # y solo despues se agrega la fila de hoy al archivo.
    hist_riesgo_previo = cargar_historial_riesgo()
    regimen_riesgo = evaluar_regimen_riesgo(risk_report, hist_riesgo_previo)
    print_risk_regime(regimen_riesgo)
    hist_riesgo = registrar_historial_riesgo(risk_report, hist_riesgo_previo)

    gamma_plot = plot_gamma_profiles(analyses)
    allocation_plot = plot_allocation_comparison(tabla_rebalanceo)
    risk_plot = plot_risk_attribution(risk_report)

    return {
        "analyses": analyses,
        "scores": scores_df,
        "tabla_tactica": tabla_tactica,
        "tabla_rebalanceo": tabla_rebalanceo,
        "risk_inputs": risk_inputs,
        "risk_report": risk_report,
        "regimen_riesgo": regimen_riesgo,
        "historial_riesgo": hist_riesgo,
        "dashboard": dashboard,
        "gamma_plot": gamma_plot,
        "allocation_plot": allocation_plot,
        "risk_plot": risk_plot,
    }

# ============================================================================
# BLOQUE 9: EJECUCION
# ============================================================================

resultado = run_active_management_engine(portfolio, investment_horizon_days, polygon_api_key, cash_reserve_limit)

if resultado["gamma_plot"] is not None:
    resultado["gamma_plot"].show()

resultado["allocation_plot"].show()

if resultado["risk_plot"] is not None:
    resultado["risk_plot"].show()

print("\n-- Tabla de Rebalanceo (data.frame crudo) --")
print(resultado["tabla_rebalanceo"])
