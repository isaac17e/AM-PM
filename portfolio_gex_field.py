# ============================================================
# BLOQUE 1: CONFIGURACIÓN
# ============================================================

import os
import time
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from datetime import datetime, timedelta
import requests
import plotly.graph_objects as go
import yfinance as yf
import warnings
from IPython.display import display, HTML, clear_output

warnings.filterwarnings("ignore")

# ------------------------------------------------
# API KEY - Polygon.io
# ------------------------------------------------
# >>> PEGA TU POLYGON API KEY AQUI (entre las comillas) <<<
POLYGON_API_KEY = "***REMOVED-POLYGON-API-KEY***"

# Si prefieres no dejar la key escrita en el archivo, define la variable de
# entorno POLYGON_API_KEY (en la terminal: export POLYGON_API_KEY="tu_key")
# y deja la linea de arriba en blanco (""); se usara automaticamente.
if not POLYGON_API_KEY:
    POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY")

BASE_URL = "https://api.polygon.io"

TIME_HORIZON_MONTHS = 2
MAX_EXPIRATIONS_CAP = 60
BASE_STRIKE_RANGE_PCT = 0.08
RISK_FREE_RATE = 0.05
NEAR_ATM_BAND_PCT = 0.03
IV_SANITY_MIN, IV_SANITY_MAX = 0.01, 2.50
NEAR_TERM_DAYS_CUTOFF = 7

MACRO_LOOKBACK_PERIOD = "1y"
MACRO_TICKERS = {
    "VIX": ["^VIX"],
    "TNX": ["^TNX"],
    "DXY": ["DX-Y.NYB", "DX=F"],
}
MACRO_WEIGHTS = {"VIX": 0.5, "TNX": 0.25, "DXY": 0.25}

SIGMA_RANGE = 3.0
SIGMA_RESOLUTION = 60
GEX_SMOOTHING_SIGMA = 2.0
ALPHA_GEX = 1.0
BETA_MACRO_AMPLIFICATION = 0.8
GAMMA_MACRO_TILT = 0.6

GRADIENT_EPSILON = 1e-4
ARROW_SCALE = 0.35
SPHERE_SIZE = 12

PORTFOLIO_HOLDINGS = {
    "9432.T": 0.1647,
    "DUK": 0.1381,
    "GLD": 0.1251,
    "VZ": 0.1020,
    "RSG": 0.0921,
    "MCD": 0.0865,
    "WM": 0.0796,
    "PDBC": 0.0749,
    "CME": 0.0707,
    "T": 0.0662
}

REFRESH_SECONDS = 60
MAX_ITERATIONS = None
ROTATE_CAMERA = True

MIN_SIGMA_COVERAGE = 2.0
MAX_STRIKE_RANGE_PCT = 0.35
EDGE_TAPER_ENABLED = True
NORM_PERCENTILE = 90

if not POLYGON_API_KEY or POLYGON_API_KEY == "TU_API_KEY":
    print("ADVERTENCIA: No se detectó una API Key válida de Polygon.")


# ============================================================
# BLOQUE 2: UTILIDADES BLACK-SCHOLES
# ============================================================

def get_dynamic_strike_range(horizon_months, base_pct=BASE_STRIKE_RANGE_PCT):
    return base_pct * np.sqrt(max(horizon_months, 1))


def clamp_iv(iv):
    if iv is None or (isinstance(iv, float) and np.isnan(iv)):
        return np.nan
    return iv if (IV_SANITY_MIN <= iv <= IV_SANITY_MAX) else np.nan


def bs_price(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if option_type == "call" else (K - S))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_delta(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        return 1.0 if (option_type == "call" and S > K) else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1.0


def implied_vol_bisection(market_price, S, K, T, r, option_type="call"):
    if market_price <= 0 or T <= 0:
        return np.nan
    try:
        f = lambda sigma: bs_price(S, K, T, r, sigma, option_type) - market_price
        return brentq(f, 1e-4, 5.0, maxiter=200)
    except (ValueError, RuntimeError):
        return np.nan


# ============================================================
# BLOQUE 3: EXTRACCIÓN DE DATOS
# ============================================================

def get_current_price(ticker):
    try:
        t = yf.Ticker(ticker)
        price = t.fast_info.get("last_price")
        if price:
            return float(price)
        raise ValueError("fast_info sin 'last_price'")
    except Exception as e:
        print(f"⚠️  [{ticker}] yfinance fast_info falló ({e}). Intentando history()...")
        try:
            hist = yf.Ticker(ticker).history(period="1d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
            raise ValueError("history() devolvió DataFrame vacío")
        except Exception as e2:
            print(f"⚠️  [{ticker}] yfinance history() también falló ({e2}). Usando Polygon /prev...")
            try:
                url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/prev"
                resp = requests.get(url, params={"apiKey": POLYGON_API_KEY}, timeout=10)
                resp.raise_for_status()
                return float(resp.json()["results"][0]["c"])
            except Exception as e3:
                print(f"❌ [{ticker}] ERROR CRÍTICO: no se pudo obtener el precio spot por ninguna fuente: {e3}")
                return None


def get_polygon_options_data(ticker, current_price, horizon_months=TIME_HORIZON_MONTHS,
                              strike_range_pct=None):
    if current_price is None:
        print(f"❌ [{ticker}] No se puede extraer la cadena de opciones sin precio spot.")
        return pd.DataFrame()

    if strike_range_pct is None:
        strike_range_pct = get_dynamic_strike_range(horizon_months)

    today = datetime.utcnow().date()
    cutoff_date = today + timedelta(days=int(horizon_months * 30.44))

    all_contracts = []
    url = f"{BASE_URL}/v3/snapshot/options/{ticker}"
    params = {
        "apiKey": POLYGON_API_KEY,
        "limit": 250,
        "strike_price.gte": round(current_price * (1 - strike_range_pct), 2),
        "strike_price.lte": round(current_price * (1 + strike_range_pct), 2),
        "expiration_date.gte": today.isoformat(),
        "expiration_date.lte": cutoff_date.isoformat(),
    }

    try:
        next_url = url
        page_count = 0
        while next_url and page_count < 40:
            resp = requests.get(next_url, params=params if page_count == 0 else None, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
            results = payload.get("results", [])
            if not results:
                break
            all_contracts.extend(results)
            next_url = payload.get("next_url")
            if next_url:
                next_url = f"{next_url}&apiKey={POLYGON_API_KEY}"
            page_count += 1
    except requests.exceptions.RequestException as e:
        print(f"❌ [{ticker}] ERROR de conexión con Polygon API: {e}")
        return pd.DataFrame()
    except ValueError as e:
        print(f"❌ [{ticker}] ERROR parseando respuesta JSON de Polygon: {e}")
        return pd.DataFrame()

    if not all_contracts:
        print(f"⚠️  [{ticker}] Polygon no devolvió contratos para horizonte={horizon_months}m "
              f"/ ventana strikes=±{strike_range_pct*100:.1f}%.")
        return pd.DataFrame()

    exp_dates_all = sorted({c.get("details", {}).get("expiration_date")
                             for c in all_contracts if c.get("details", {}).get("expiration_date")})
    if len(exp_dates_all) > MAX_EXPIRATIONS_CAP:
        print(f"⚠️  [{ticker}] {len(exp_dates_all)} expiraciones encontradas, truncando a las "
              f"{MAX_EXPIRATIONS_CAP} más cercanas.")
    target_expirations = set(exp_dates_all[:MAX_EXPIRATIONS_CAP])

    raw_rows = []
    for c in all_contracts:
        details = c.get("details", {})
        exp_date_str = details.get("expiration_date")
        if exp_date_str not in target_expirations:
            continue
        strike = details.get("strike_price")
        opt_type = details.get("contract_type")
        if strike is None or opt_type not in ("call", "put"):
            continue

        greeks = c.get("greeks", {}) or {}
        last_quote = c.get("last_quote", {}) or {}
        bid, ask = last_quote.get("bid"), last_quote.get("ask")
        ref_price = None
        if bid and ask and bid > 0 and ask > 0:
            ref_price = (bid + ask) / 2.0
        else:
            last_trade_px = (c.get("last_trade", {}) or {}).get("price")
            if last_trade_px:
                ref_price = last_trade_px
            else:
                day_close = (c.get("day", {}) or {}).get("close")
                if day_close:
                    ref_price = day_close

        raw_rows.append({
            "strike": strike, "expiration": exp_date_str, "type": opt_type,
            "open_interest": c.get("open_interest", 0) or 0,
            "iv_polygon": clamp_iv(c.get("implied_volatility")),
            "gamma_polygon": greeks.get("gamma") or np.nan,
            "delta_polygon": greeks.get("delta") or np.nan,
            "ref_price": ref_price,
        })

    if not raw_rows:
        print(f"⚠️  [{ticker}] Tras el filtrado de expiraciones no quedaron contratos.")
        return pd.DataFrame()

    df_raw = pd.DataFrame(raw_rows)
    df_raw["moneyness"] = (df_raw["strike"] - current_price) / current_price

    atm_iv_by_exp = {}
    for exp, grp in df_raw.groupby("expiration"):
        near_atm = grp[grp["moneyness"].abs() <= NEAR_ATM_BAND_PCT]
        reliable = near_atm["iv_polygon"].dropna()
        atm_iv_by_exp[exp] = float(reliable.median()) if len(reliable) > 0 else np.nan

    rows = []
    n_far_proxy = 0
    for _, r in df_raw.iterrows():
        try:
            exp_date = datetime.strptime(r["expiration"], "%Y-%m-%d").date()
            T = max((exp_date - today).days, 0) / 365.0
            T_eff = max(T, 1 / 365.0 / 24)

            iv = r["iv_polygon"]
            gamma = r["gamma_polygon"]
            delta = r["delta_polygon"]
            is_near_atm = abs(r["moneyness"]) <= NEAR_ATM_BAND_PCT

            if pd.isna(iv):
                if is_near_atm and r["ref_price"]:
                    iv_solved = clamp_iv(implied_vol_bisection(r["ref_price"], current_price, r["strike"],
                                                                 T_eff, RISK_FREE_RATE, r["type"]))
                    if not np.isnan(iv_solved):
                        iv = iv_solved
                if pd.isna(iv):
                    proxy = atm_iv_by_exp.get(r["expiration"])
                    if proxy and not np.isnan(proxy):
                        iv = proxy
                        n_far_proxy += 1

            if (pd.isna(gamma) or gamma == 0) and iv and not np.isnan(iv):
                gamma = bs_gamma(current_price, r["strike"], T_eff, RISK_FREE_RATE, iv)

            if pd.isna(delta) and iv and not np.isnan(iv):
                delta = bs_delta(current_price, r["strike"], T_eff, RISK_FREE_RATE, iv, r["type"])

            rows.append({
                "strike": r["strike"], "expiration": r["expiration"], "type": r["type"],
                "open_interest": r["open_interest"],
                "implied_volatility": iv if iv is not None else np.nan,
                "gamma": gamma if gamma is not None else np.nan,
                "delta": delta if delta is not None else np.nan,
                "moneyness": r["moneyness"],
                "days_to_exp": (exp_date - today).days,
            })
        except Exception as e:
            print(f"⚠️  [{ticker}] Contrato omitido por error de parseo: {e}")
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"⚠️  [{ticker}] No quedaron contratos válidos tras el procesamiento.")
        return df

    n_before = len(df)
    df = df.dropna(subset=["gamma"])
    n_dropped_gamma = n_before - len(df)

    print(f"✅ [{ticker}] Cadena cargada: {len(df)} contratos válidos | Horizonte: {horizon_months} mes(es) "
          f"(hasta {cutoff_date}) | Ventana strikes: ±{strike_range_pct*100:.1f}% | "
          f"Expiraciones: {len(df['expiration'].unique())}")
    if n_dropped_gamma:
        print(f"ℹ️  [{ticker}] {n_dropped_gamma} contratos descartados por falta de gamma.")
    if n_far_proxy:
        print(f"ℹ️  [{ticker}] {n_far_proxy} contratos fuera de la banda ATM usaron IV proxy plano de su expiración.")

    return df


# ============================================================
# BLOQUE 4: GEX NETO
# ============================================================

def calculate_gex_and_surface_forces(df_options, current_price, label=""):
    if df_options is None or df_options.empty:
        print(f"❌ [{label}] No hay datos de opciones para calcular GEX.")
        return pd.DataFrame(), None, {}

    df = df_options.copy()
    df["gex_raw"] = df["gamma"] * df["open_interest"] * 100 * (current_price ** 2) * 0.01
    df["gex_signed"] = np.where(df["type"] == "call", df["gex_raw"], -df["gex_raw"])

    df_gex = (df.groupby("strike", as_index=False)["gex_signed"]
                .sum().sort_values("strike").reset_index(drop=True))
    df_gex.rename(columns={"gex_signed": "net_gex"}, inplace=True)
    df_gex["cumulative_gex"] = df_gex["net_gex"].cumsum()

    gamma_flip = None
    try:
        sign_changes = np.where(np.diff(np.sign(df_gex["cumulative_gex"])) != 0)[0]
        if len(sign_changes) > 0:
            idx = sign_changes[0]
            x0, x1 = df_gex["strike"].iloc[idx], df_gex["strike"].iloc[idx + 1]
            y0, y1 = df_gex["cumulative_gex"].iloc[idx], df_gex["cumulative_gex"].iloc[idx + 1]
            gamma_flip = x0 + (0 - y0) * (x1 - x0) / (y1 - y0) if (y1 - y0) != 0 else x0
        else:
            print(f"ℹ️  [{label}] No se detectó cruce de signo en el GEX acumulado en esta ventana.")
    except Exception as e:
        print(f"⚠️  [{label}] Error calculando Gamma Flip Point: {e}")

    try:
        call_wall = df_gex.loc[df_gex["net_gex"].idxmax(), "strike"] if df_gex["net_gex"].max() > 0 else None
        put_wall = df_gex.loc[df_gex["net_gex"].idxmin(), "strike"] if df_gex["net_gex"].min() < 0 else None
    except Exception as e:
        print(f"⚠️  [{label}] Error calculando Call/Put Wall: {e}")
        call_wall, put_wall = None, None

    metrics = {
        "total_gex": df_gex["net_gex"].sum(),
        "gamma_flip": gamma_flip,
        "call_wall": call_wall,
        "put_wall": put_wall,
        "regime": "Positive Gamma (estable)" if df_gex["net_gex"].sum() > 0 else "Negative Gamma (inestable)",
    }

    print(f"📐 [{label}] GEX Total: {metrics['total_gex']:,.0f} | "
          f"Gamma Flip: {gamma_flip:.2f}" if gamma_flip else f"📐 [{label}] GEX Total: {metrics['total_gex']:,.0f} | Gamma Flip: no detectado")
    print(f"🧱 [{label}] Call Wall: {call_wall} | Put Wall: {put_wall} | Régimen: {metrics['regime']}")

    return df_gex, gamma_flip, metrics


def calculate_gex_split(df_options, current_price, near_term_days_cutoff=NEAR_TERM_DAYS_CUTOFF):
    if df_options is None or df_options.empty:
        print("❌ No hay datos de opciones para separar por horizonte.")
        return {}

    df_short = df_options[df_options["days_to_exp"] <= near_term_days_cutoff].copy()
    df_structural = df_options[df_options["days_to_exp"] > near_term_days_cutoff].copy()

    gex_short, flip_short, metrics_short = calculate_gex_and_surface_forces(
        df_short, current_price, label="corto plazo")

    gex_structural, flip_structural, metrics_structural = calculate_gex_and_surface_forces(
        df_structural, current_price, label="estructural")

    return {
        "short_term": {"df_gex": gex_short, "gamma_flip": flip_short, "metrics": metrics_short},
        "structural": {"df_gex": gex_structural, "gamma_flip": flip_structural, "metrics": metrics_structural},
    }


# ============================================================
# BLOQUE 5: EJE Y — MACRO / LIQUIDEZ / VOLATILIDAD
# ============================================================

def _fetch_macro_series(ticker_candidates, period=MACRO_LOOKBACK_PERIOD):
    for tk in ticker_candidates:
        try:
            hist = yf.Ticker(tk).history(period=period)
            if not hist.empty and "Close" in hist.columns and hist["Close"].dropna().shape[0] > 5:
                return hist["Close"].dropna(), tk
        except Exception as e:
            print(f"⚠️  Falló descarga de {tk}: {e}")
            continue
    return None, None


def _percentile_rank(series, current_value):
    if series is None or series.empty or current_value is None or np.isnan(current_value):
        return np.nan
    return float((series < current_value).sum() / len(series))


def get_macro_indicators(lookback=MACRO_LOOKBACK_PERIOD):
    results = {}
    for name, candidates in MACRO_TICKERS.items():
        series, used_ticker = _fetch_macro_series(candidates, lookback)
        if series is None:
            print(f"❌ No se pudo obtener {name} de ninguno de los tickers candidatos {candidates}.")
            results[name] = {"value": np.nan, "percentile": np.nan, "ticker": None, "ok": False}
            continue

        current_value = float(series.iloc[-1])
        percentile = _percentile_rank(series, current_value)
        results[name] = {"value": current_value, "percentile": percentile,
                          "ticker": used_ticker, "ok": True}
        print(f"✅ {name} ({used_ticker}): valor actual={current_value:.2f} | "
              f"percentil {lookback}={percentile*100:.0f}%")

    return results


def calculate_macro_y_axis(macro_indicators, weights=MACRO_WEIGHTS):
    available = {k: v for k, v in macro_indicators.items()
                 if v["ok"] and not np.isnan(v["percentile"])}

    if not available:
        print("❌ Ningún indicador macro disponible. No se puede calcular el eje Y.")
        return {"y_score": np.nan, "components_used": []}

    total_weight = sum(weights[k] for k in available)
    y_score = sum(available[k]["percentile"] * weights[k] for k in available) / total_weight

    missing = set(macro_indicators.keys()) - set(available.keys())
    if missing:
        print(f"⚠️  Componentes macro no disponibles, pesos redistribuidos entre el resto: {missing}")

    print(f"🌐 Score macro combinado (Eje Y): {y_score:.3f} "
          f"(0=relajado, 1=tenso) | componentes usados: {list(available.keys())}")

    return {"y_score": y_score, "components_used": list(available.keys())}


# ============================================================
# BLOQUE 6: MOVIMIENTO ESPERADO (1σ)
# ============================================================

def calculate_expected_move(df_options, current_price, horizon_months=TIME_HORIZON_MONTHS,
                             near_atm_band=NEAR_ATM_BAND_PCT, near_term_days_cutoff=NEAR_TERM_DAYS_CUTOFF):
    if df_options is None or df_options.empty:
        print("❌ No hay datos de opciones para estimar el movimiento esperado.")
        return np.nan

    df_atm_structural = df_options[
        (df_options["moneyness"].abs() <= near_atm_band) &
        (df_options["days_to_exp"] > near_term_days_cutoff)
    ]
    iv_atm = df_atm_structural["implied_volatility"].mean()

    if np.isnan(iv_atm):
        print("⚠️  No hay IV ATM estructural disponible; usando IV ATM total como fallback.")
        iv_atm = df_options[df_options["moneyness"].abs() <= near_atm_band]["implied_volatility"].mean()
        if np.isnan(iv_atm):
            print("❌ No se pudo estimar IV ATM por ningún método.")
            return np.nan

    T_horizon = horizon_months / 12.0
    expected_move_pct = iv_atm * np.sqrt(T_horizon)

    return expected_move_pct


# ============================================================
# BLOQUE 7: DATOS POR HOLDING Y CAMPO DE FUERZA COMPUESTO
# ============================================================

def get_portfolio_chains(holdings, horizon_months=TIME_HORIZON_MONTHS):
    portfolio_data = {}
    failed = []

    for ticker, weight in holdings.items():
        print(f"\n{'='*60}\nProcesando {ticker} (peso original: {weight:.2%})\n{'='*60}")

        price = get_current_price(ticker)
        if price is None:
            print(f"❌ {ticker}: sin precio spot, se excluye del portafolio.")
            failed.append(ticker)
            continue

        df_opts = get_polygon_options_data(ticker, price, horizon_months=horizon_months)
        if df_opts.empty:
            print(f"❌ {ticker}: sin cadena de opciones válida, se excluye.")
            failed.append(ticker)
            continue

        split = calculate_gex_split(df_opts, price)
        df_gex_structural = split.get("structural", {}).get("df_gex")
        metrics_structural = split.get("structural", {}).get("metrics")

        if df_gex_structural is None or df_gex_structural.empty:
            print(f"❌ {ticker}: sin GEX estructural (>{NEAR_TERM_DAYS_CUTOFF}d), se excluye.")
            failed.append(ticker)
            continue

        expected_move = calculate_expected_move(df_opts, price, horizon_months=horizon_months)
        if expected_move is None or np.isnan(expected_move) or expected_move <= 0:
            print(f"❌ {ticker}: no se pudo estimar el movimiento esperado (1σ), se excluye.")
            failed.append(ticker)
            continue

        current_range_pct = get_dynamic_strike_range(horizon_months)
        coverage_sigma = current_range_pct / expected_move
        required_range_pct = min(MIN_SIGMA_COVERAGE * expected_move, MAX_STRIKE_RANGE_PCT)

        if coverage_sigma < MIN_SIGMA_COVERAGE and required_range_pct > current_range_pct * 1.05:
            print(f"ℹ️  [{ticker}] Cobertura real ±{coverage_sigma:.2f}σ < objetivo ±{MIN_SIGMA_COVERAGE:.1f}σ "
                  f"(mov. esperado ±{expected_move*100:.1f}%). Re-descargando con ventana "
                  f"±{required_range_pct*100:.1f}% (antes ±{current_range_pct*100:.1f}%)...")

            df_opts_wide = get_polygon_options_data(ticker, price, horizon_months=horizon_months,
                                                      strike_range_pct=required_range_pct)
            if not df_opts_wide.empty:
                split_wide = calculate_gex_split(df_opts_wide, price)
                df_gex_structural_wide = split_wide.get("structural", {}).get("df_gex")
                metrics_structural_wide = split_wide.get("structural", {}).get("metrics")
                expected_move_wide = calculate_expected_move(df_opts_wide, price, horizon_months=horizon_months)

                if (df_gex_structural_wide is not None and not df_gex_structural_wide.empty
                        and expected_move_wide and not np.isnan(expected_move_wide) and expected_move_wide > 0):
                    df_opts = df_opts_wide
                    df_gex_structural = df_gex_structural_wide
                    metrics_structural = metrics_structural_wide
                    expected_move = expected_move_wide
                    new_coverage = required_range_pct / expected_move
                    print(f"✅ [{ticker}] Cobertura ampliada a ±{new_coverage:.2f}σ.")
                else:
                    print(f"⚠️  [{ticker}] El re-fetch no produjo datos utilizables; se conserva la ventana original.")
            else:
                print(f"⚠️  [{ticker}] El re-fetch con ventana ampliada no devolvió contratos; se conserva la ventana original.")

        portfolio_data[ticker] = {
            "weight": weight,
            "price": price,
            "df_options": df_opts,
            "df_gex_structural": df_gex_structural,
            "metrics_structural": metrics_structural,
            "expected_move": expected_move,
        }

    if not portfolio_data:
        print("❌ ERROR CRÍTICO: no quedó ningún holding válido en el portafolio.")
        return {}

    if failed:
        used_weight = sum(v["weight"] for v in portfolio_data.values())
        print(f"\n⚠️  Tickers excluidos: {failed}. Pesos re-normalizados sobre el "
              f"{used_weight:.2%} de peso restante.")
        for v in portfolio_data.values():
            v["weight_normalized"] = v["weight"] / used_weight
    else:
        for v in portfolio_data.values():
            v["weight_normalized"] = v["weight"]

    print(f"\n✅ Portafolio final: {len(portfolio_data)}/{len(holdings)} holdings procesados.")
    for t, v in portfolio_data.items():
        try:
            achieved_range_pct = float(v["df_gex_structural"]["strike"].sub(v["price"]).abs().max() / v["price"])
            achieved_sigma = achieved_range_pct / v["expected_move"]
            coverage_note = f"| cobertura real ±{achieved_sigma:.2f}σ"
        except Exception:
            coverage_note = ""
        print(f"   {t}: peso={v['weight_normalized']:.2%} | "
              f"movimiento esperado (1σ, {horizon_months}m)=±{v['expected_move']*100:.2f}% {coverage_note}")

    return portfolio_data


def _build_gex_force_function_sigma(df_gex_structural, current_price, expected_move,
                                     sigma_range=SIGMA_RANGE, resolution=SIGMA_RESOLUTION,
                                     smoothing_sigma=GEX_SMOOTHING_SIGMA):
    if df_gex_structural is None or df_gex_structural.empty or expected_move is None or expected_move <= 0:
        return None, None

    df = df_gex_structural.copy()
    df["moneyness"] = (df["strike"] - current_price) / current_price
    df["sigma_x"] = df["moneyness"] / expected_move
    df = df[df["sigma_x"].abs() <= sigma_range].sort_values("sigma_x")

    if len(df) < 2:
        print(f"⚠️  Solo {len(df)} strikes estructurales dentro de ±{sigma_range}σ. Insuficiente.")
        return None, None

    grid_sigma = np.linspace(-sigma_range, sigma_range, resolution)

    try:
        interp_func = interp1d(df["sigma_x"], df["net_gex"], kind="linear",
                                bounds_error=False,
                                fill_value=(df["net_gex"].iloc[0], df["net_gex"].iloc[-1]))
        gex_on_grid = interp_func(grid_sigma)
    except Exception as e:
        print(f"❌ Error interpolando perfil de GEX en sigma-units: {e}")
        return None, None

    if EDGE_TAPER_ENABLED:
        real_min, real_max = df["sigma_x"].iloc[0], df["sigma_x"].iloc[-1]
        left_band = real_min - (-sigma_range)
        right_band = sigma_range - real_max
        taper = np.ones_like(grid_sigma)
        for i, gx in enumerate(grid_sigma):
            if gx < real_min and left_band > 1e-6:
                taper[i] = np.clip((gx - (-sigma_range)) / left_band, 0.0, 1.0)
            elif gx > real_max and right_band > 1e-6:
                taper[i] = np.clip((sigma_range - gx) / right_band, 0.0, 1.0)
        gex_on_grid = gex_on_grid * taper

    gex_smoothed = gaussian_filter1d(gex_on_grid, sigma=smoothing_sigma)
    norm_reference = np.percentile(np.abs(gex_smoothed), NORM_PERCENTILE)
    if norm_reference == 0 or np.isnan(norm_reference):
        return grid_sigma, np.zeros_like(grid_sigma)

    return grid_sigma, np.clip(gex_smoothed / norm_reference, -1.0, 1.0)


def build_composite_portfolio_force(portfolio_data, sigma_range=SIGMA_RANGE,
                                     resolution=SIGMA_RESOLUTION):
    grid_sigma = np.linspace(-sigma_range, sigma_range, resolution)
    composite = np.zeros_like(grid_sigma)
    weight_used = 0.0
    contributions = {}

    for ticker, data in portfolio_data.items():
        g, force = _build_gex_force_function_sigma(
            data["df_gex_structural"], data["price"], data["expected_move"],
            sigma_range, resolution)

        if g is None:
            print(f"⚠️  {ticker}: no aporta curva de GEX válida al compuesto, se excluye de esta etapa.")
            continue

        w = data["weight_normalized"]
        composite += w * force
        weight_used += w
        contributions[ticker] = force

    if weight_used == 0:
        print("❌ Ningún holding aportó curva de GEX válida. No se puede construir el campo compuesto.")
        return None, None, {}

    if weight_used < 0.999:
        print(f"⚠️  Solo {weight_used:.2%} del peso del portafolio aportó al campo compuesto; "
              f"se re-normaliza sobre ese subconjunto.")
        composite = composite / weight_used

    print(f"\n✅ Campo de fuerza compuesto construido con {len(contributions)}/{len(portfolio_data)} "
          f"holdings | rango: ±{sigma_range}σ")

    return grid_sigma, composite, contributions


# ============================================================
# BLOQUE 8: SUPERFICIE DE POTENCIAL Z(X,Y)
# ============================================================

def generate_portfolio_potential_surface(grid_sigma, composite_force, macro_y_axis,
                                          sigma_range=SIGMA_RANGE, resolution=SIGMA_RESOLUTION):
    if grid_sigma is None or composite_force is None:
        print("❌ No se pudo construir el eje X compuesto. Superficie no generada.")
        return None, None, None, None

    if macro_y_axis is None or np.isnan(macro_y_axis.get("y_score", np.nan)):
        print("⚠️  Score macro no disponible; se usa Y=0.5 (neutral) como fallback.")
        y_current = 0.5
    else:
        y_current = macro_y_axis["y_score"]

    grid_y = np.linspace(0.0, 1.0, resolution)
    X, Y = np.meshgrid(grid_sigma, grid_y)

    P_x_grid = np.tile(-composite_force, (resolution, 1))
    Z = ALPHA_GEX * P_x_grid * (1 + BETA_MACRO_AMPLIFICATION * Y) + GAMMA_MACRO_TILT * Y

    print(f"✅ Superficie de potencial del portafolio generada: grid {resolution}x{resolution} | "
          f"±{sigma_range}σ | Y actual (macro): {y_current:.3f}")
    print(f"   Rango de Z: [{Z.min():.3f}, {Z.max():.3f}]")

    surface_data = {
        "grid_sigma": grid_sigma, "grid_y": grid_y, "composite_force": composite_force,
        "y_current": y_current, "sigma_range": sigma_range,
    }

    return X, Y, Z, surface_data


def potential_function(x, y, grid_sigma, composite_force):
    force_at_x = np.interp(x, grid_sigma, composite_force)
    P_x = -force_at_x
    return ALPHA_GEX * P_x * (1 + BETA_MACRO_AMPLIFICATION * y) + GAMMA_MACRO_TILT * y


def calculate_gradient(x0, y0, grid_sigma, composite_force, h=GRADIENT_EPSILON):
    try:
        dzdx = (potential_function(x0 + h, y0, grid_sigma, composite_force)
                 - potential_function(x0 - h, y0, grid_sigma, composite_force)) / (2 * h)
        dzdy = (potential_function(x0, y0 + h, grid_sigma, composite_force)
                 - potential_function(x0, y0 - h, grid_sigma, composite_force)) / (2 * h)
        return dzdx, dzdy
    except Exception as e:
        print(f"⚠️  Error calculando el gradiente: {e}")
        return 0.0, 0.0


def get_portfolio_current_state(surface_data):
    if surface_data is None:
        print("❌ No hay datos de superficie para ubicar el estado del portafolio.")
        return None

    grid_sigma = surface_data["grid_sigma"]
    composite_force = surface_data["composite_force"]
    y_current = surface_data["y_current"]
    x_current = 0.0

    z_current = potential_function(x_current, y_current, grid_sigma, composite_force)
    dzdx, dzdy = calculate_gradient(x_current, y_current, grid_sigma, composite_force)

    grad_norm = np.sqrt(dzdx**2 + dzdy**2)
    if grad_norm > 1e-9:
        direction_x = -dzdx / grad_norm * ARROW_SCALE
        direction_y = -dzdy / grad_norm * ARROW_SCALE
    else:
        direction_x, direction_y = 0.0, 0.0
        print("ℹ️  Gradiente ~0 en el punto actual: zona plana del campo de potencial.")

    z_target = potential_function(x_current + direction_x, y_current + direction_y, grid_sigma, composite_force)
    direction_z = z_target - z_current

    print(f"\n📍 Estado del portafolio: X=0σ | Y(macro)={y_current:.3f} | Z={z_current:.3f}")
    print(f"🧭 Gradiente ∇Z: (dZ/dx={dzdx:.3f}, dZ/dy={dzdy:.3f}) | magnitud={grad_norm:.3f}")

    return {
        "x": x_current, "y": y_current, "z": z_current,
        "grad_x": dzdx, "grad_y": dzdy, "grad_magnitude": grad_norm,
        "arrow_dx": direction_x, "arrow_dy": direction_y, "arrow_dz": direction_z,
    }


def get_holdings_reference_lines(portfolio_data, sigma_range=SIGMA_RANGE):
    lines = []
    if not portfolio_data:
        return lines

    for ticker, data in portfolio_data.items():
        metrics = data.get("metrics_structural") or {}
        price = data.get("price")
        expected_move = data.get("expected_move")
        if not price or not expected_move or expected_move <= 0:
            continue

        for key, label in (("gamma_flip", "Gamma Flip"), ("call_wall", "Call Wall"), ("put_wall", "Put Wall")):
            strike_val = metrics.get(key)
            if strike_val is None:
                continue
            try:
                sigma_x = ((strike_val - price) / price) / expected_move
            except ZeroDivisionError:
                continue
            if abs(sigma_x) > sigma_range:
                continue
            lines.append({
                "ticker": ticker, "label": label, "sigma_x": sigma_x,
                "strike": strike_val, "weight": data.get("weight_normalized", data.get("weight")),
            })

    print(f"\nℹ️  {len(lines)} líneas de referencia (Gamma Flip/Call Wall/Put Wall) dentro de ±{sigma_range}σ.")
    return lines


# ============================================================
# BLOQUE 9: VISUALIZACIÓN 3D
# ============================================================

def plot_3d_portfolio_field(X, Y, Z, current_state, reference_lines, portfolio_data,
                             horizon_months=TIME_HORIZON_MONTHS, camera_eye=None):
    if X is None or Y is None or Z is None:
        print("❌ Superficie no disponible. No se puede graficar.")
        return None

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z, colorscale="RdYlBu_r", opacity=0.85,
        colorbar=dict(title="Potencial Z<br>(riesgo)", x=1.02),
        contours={"z": {"show": True, "usecolormap": True, "project_z": False}},
        name="Campo de Potencial del Portafolio",
    ))

    if current_state is not None:
        tickers_label = ", ".join(portfolio_data.keys()) if portfolio_data else "Portafolio"
        fig.add_trace(go.Scatter3d(
            x=[current_state["x"]], y=[current_state["y"]], z=[current_state["z"]],
            mode="markers+text",
            marker=dict(size=SPHERE_SIZE, color="black", symbol="diamond",
                        line=dict(color="lime", width=2)),
            text=[f"Portafolio<br>({tickers_label})"], textposition="top center",
            name="Portafolio (agregado, X=0σ)",
        ))

        ax, ay, az = current_state["arrow_dx"], current_state["arrow_dy"], current_state["arrow_dz"]
        if abs(ax) > 1e-9 or abs(ay) > 1e-9:
            fig.add_trace(go.Cone(
                x=[current_state["x"]], y=[current_state["y"]], z=[current_state["z"]],
                u=[ax], v=[ay], w=[az],
                colorscale=[[0, "green"], [1, "green"]], showscale=False,
                sizemode="absolute", sizeref=0.15, anchor="tail",
                name="Gradiente -∇Z del portafolio",
            ))

    line_colors = {"Gamma Flip": "orange", "Call Wall": "blue", "Put Wall": "red"}
    for ref in reference_lines:
        color = line_colors.get(ref["label"], "gray")
        try:
            fig.add_trace(go.Scatter3d(
                x=[ref["sigma_x"], ref["sigma_x"]], y=[0, 1], z=[Z.min(), Z.min()],
                mode="lines", line=dict(color=color, width=3, dash="dash"),
                name=f"{ref['ticker']} {ref['label']} ({ref['strike']:.1f})",
            ))
        except Exception as e:
            print(f"⚠️  No se pudo dibujar la línea de referencia de {ref['ticker']} '{ref['label']}': {e}")

    if camera_eye is None:
        camera_eye = dict(x=1.6, y=-1.6, z=1.2)

    fig.update_layout(
        title=f"Portfolio GEX Potential Field — {len(portfolio_data)} holdings "
              f"(horizonte {horizon_months}m) | actualizado {datetime.utcnow().strftime('%H:%M:%S')} UTC",
        scene=dict(
            domain=dict(x=[0.24, 1.0], y=[0.0, 1.0]),
            xaxis_title="X: Posición en σ del movimiento esperado agregado",
            yaxis_title="Y: Tensión Macro (0=relajado, 1=tenso)",
            zaxis_title="Z: Potencial de Riesgo del Portafolio",
            camera=dict(eye=camera_eye),
        ),
        width=1100, height=750,
        legend=dict(
            x=0.0, y=0.98, xanchor="left", yanchor="top",
            font=dict(size=10),
            itemsizing="constant",
            tracegroupgap=1,
            bgcolor="rgba(11,13,16,0.55)",
            bordercolor="rgba(255,255,255,0.15)",
            borderwidth=1,
        ),
        paper_bgcolor="#0b0d10",
        plot_bgcolor="#0b0d10",
        font=dict(color="#d8dee5"),
    )
    return fig


# ============================================================
# BLOQUE 10: EJECUCIÓN
# ============================================================

def run_once():
    macro_indicators = get_macro_indicators()
    macro_y_axis = calculate_macro_y_axis(macro_indicators)

    portfolio_data = get_portfolio_chains(PORTFOLIO_HOLDINGS)

    if not portfolio_data:
        print("❌ No se pudo construir el campo de potencial: ningún holding tiene datos válidos.")
        return None

    grid_sigma, composite_force, contributions = build_composite_portfolio_force(portfolio_data)
    X, Y, Z, surface_data = generate_portfolio_potential_surface(grid_sigma, composite_force, macro_y_axis)
    current_state = get_portfolio_current_state(surface_data)
    reference_lines = get_holdings_reference_lines(portfolio_data)

    return {
        "X": X, "Y": Y, "Z": Z, "current_state": current_state,
        "reference_lines": reference_lines, "portfolio_data": portfolio_data,
    }


def run_live(refresh_seconds=REFRESH_SECONDS, max_iterations=MAX_ITERATIONS,
             rotate_camera=ROTATE_CAMERA):
    handle = None
    angle = 0.0
    iteration = 0

    while max_iterations is None or iteration < max_iterations:
        clear_output(wait=True)
        print(f"🔄 Iteración {iteration + 1} | refrescando cada {refresh_seconds}s | "
              f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n")

        result = run_once()

        if result is not None:
            camera_eye = None
            if rotate_camera:
                angle += 0.35
                camera_eye = dict(
                    x=1.6 * np.cos(angle), y=-1.6 + 0.4 * np.sin(angle * 0.6), z=1.2
                )

            fig = plot_3d_portfolio_field(
                result["X"], result["Y"], result["Z"],
                result["current_state"], result["reference_lines"],
                result["portfolio_data"], camera_eye=camera_eye,
            )
            if fig is not None:
                fig.show()
        else:
            print("⚠️  No se generó gráfico en esta iteración; se reintentará en el próximo refresh.")

        iteration += 1
        if max_iterations is None or iteration < max_iterations:
            time.sleep(refresh_seconds)

    print("\n✅ Loop finalizado.")


run_live()