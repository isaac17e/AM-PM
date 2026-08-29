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

gamma_profile_colors = {"positivo": "#2E7D32", "negativo": "#C62828", "flip": "#1565C0"}
allocation_colors    = {"Actual": "#78909C", "Sugerido": "#1565C0"}

# ============================================================================
# BLOQUE 2: EXTRACCION Y PROCESAMIENTO DE OPCIONES (POLYGON API v3)
# ============================================================================

def polygon_get(url, params=None, api_key=polygon_api_key, max_retries=polygon_max_retries):
    if params is None:
        params = {}
    params = dict(params)
    params["apiKey"] = api_key
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = requests.get(url, params=params, timeout=20)
        except requests.exceptions.RequestException as e:
            warnings.warn(f"Fallo de red en {url}: {e}")
            return None
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
# BLOQUE 5: REBALANCEO DEL PORTAFOLIO Y ASIGNACION DE CASH
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
# BLOQUE 6: SALIDA GRAFICA Y REPORTE
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

# ============================================================================
# BLOQUE 7: ORQUESTACION PRINCIPAL
# ============================================================================

def run_active_management_engine(portfolio, horizon_days, api_key, cash_limit):
    tickers = list(portfolio.keys())

    analyses = {tk: analyze_ticker_options(tk, horizon_days, api_key) for tk in tickers}

    scores_list = [calculate_tactical_score(analyses[tk]) for tk in tickers]
    scores_df = pd.DataFrame(scores_list)

    tabla_rebalanceo = rebalance_portfolio(portfolio, scores_df, cash_limit)

    dashboard = generate_executive_dashboard(tabla_rebalanceo)

    gamma_plot = plot_gamma_profiles(analyses)
    allocation_plot = plot_allocation_comparison(tabla_rebalanceo)

    return {
        "analyses": analyses,
        "scores": scores_df,
        "tabla_rebalanceo": tabla_rebalanceo,
        "dashboard": dashboard,
        "gamma_plot": gamma_plot,
        "allocation_plot": allocation_plot,
    }

# ============================================================================
# BLOQUE 8: EJECUCION
# ============================================================================

resultado = run_active_management_engine(portfolio, investment_horizon_days, polygon_api_key, cash_reserve_limit)

if resultado["gamma_plot"] is not None:
    resultado["gamma_plot"].show()

resultado["allocation_plot"].show()

print("\n-- Tabla de Rebalanceo (data.frame crudo) --")
print(resultado["tabla_rebalanceo"])
