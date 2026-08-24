# =============================================================================
# BLACK-LITTERMAN PORTFOLIO OPTIMIZATION
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import math
import os
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

import yfinance as yf
from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator
import quadprog

import matplotlib.pyplot as plt

# ------------------------------------------------
# API KEY - Polygon.io
# ------------------------------------------------
POLYGON_API_KEY = "***REMOVED-POLYGON-API-KEY***"

if not POLYGON_API_KEY:
    POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY")

# -----------------------------------------------------------------------------
# 1. UNIVERSO DE TICKERS
# -----------------------------------------------------------------------------
TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "BRK.B", "LLY", "JPM", "V",
    "XOM", "UNH", "MA", "JNJ", "PG",
    "HD", "MRK", "AVGO", "CVX", "COST",
]

# -----------------------------------------------------------------------------
# 2. HORIZONTE TEMPORAL
# -----------------------------------------------------------------------------
MESES_HORIZONTE = 1
DIAS_HABILES_MES = 21
SEMANAS_MES = 4.33

# -----------------------------------------------------------------------------
# 3. PERFIL DE RIESGO
# -----------------------------------------------------------------------------
PERFIL_RIESGO = "moderado"

# -----------------------------------------------------------------------------
# 4. PARAMETROS DE OPTIMIZACION (por perfil)
# -----------------------------------------------------------------------------
PERFILES = {
    "conservador": dict(delta=4.0, omega_scale=5.0, tau=0.025),
    "moderado": dict(delta=2.5, omega_scale=1.0, tau=0.05),
    "agresivo": dict(delta=1.0, omega_scale=0.25, tau=0.10),
}

# -----------------------------------------------------------------------------
# 5. LIMITES DEL PORTAFOLIO FINAL
# -----------------------------------------------------------------------------
UMBRAL_PESO_MIN = 0.005
MAX_TICKERS_FINAL = 10

# -----------------------------------------------------------------------------
# 6. TASA LIBRE DE RIESGO - FALLBACK
# -----------------------------------------------------------------------------
Rf = 0.044

# -----------------------------------------------------------------------------
# 7. ANALISIS DE MAXIMUM DRAWDOWN (MDD)
# -----------------------------------------------------------------------------
MDD_START_YEAR = date.today().year - 2

# -----------------------------------------------------------------------------
# 8. VOLATILIDAD IMPLICITA VIA POLYGON - SSVI
# -----------------------------------------------------------------------------
USAR_IV_POLYGON = True
MIN_STRIKES_SLICE = 5
MIN_DIAS_VENCIMIENTO = 5

if USAR_IV_POLYGON and not POLYGON_API_KEY:
    raise ValueError(
        "USAR_IV_POLYGON = True pero POLYGON_API_KEY no esta definida. "
        "Configura el secreto 'PolygonAPI' en Colab, o pon USAR_IV_POLYGON = False "
        "para usar el metodo historico."
    )

# =============================================================================
# FIN BLOQUE 0
# =============================================================================

horizonte_dias = MESES_HORIZONTE * DIAS_HABILES_MES
horizonte_semanas = MESES_HORIZONTE * SEMANAS_MES
factor_anualizacion = round(52 * (MESES_HORIZONTE / 12))

print("=== HORIZONTE TEMPORAL ===")
print(f"Meses: {MESES_HORIZONTE}")
print(f"Dias habiles: {horizonte_dias}")
print(f"Factor de escala (semanas): {factor_anualizacion}\n")

print("=== UNIVERSO ===")
print(f"Tickers: {len(TICKERS)}")
print(TICKERS)
print()

# =============================================================================
# BLOQUE 1: DESCARGA DE PRECIOS
# =============================================================================

fecha_fin = date.today()
fecha_inicio = fecha_fin - timedelta(days=365 * 2)

print("=== Descargando precios ===")
print(f"Desde: {fecha_inicio} | Hasta: {fecha_fin}\n")


def descargar_precio(ticker, start, end, max_retries=3):
    for _ in range(max_retries):
        try:
            hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
            if hist is not None and len(hist) > 0:
                s = hist["Close"].copy()
                s.index = pd.to_datetime(s.index).tz_localize(None)
                return s
        except Exception:
            time.sleep(1)
    return None


tickers_ok = []
precios_dict = {}
for tk in TICKERS:
    serie = descargar_precio(tk, fecha_inicio, fecha_fin)
    if serie is None or len(serie) == 0:
        print(f"  Error descargando {tk}")
        continue
    precios_dict[tk] = serie
    tickers_ok.append(tk)

print(f"Tickers descargados: {len(tickers_ok)} / {len(TICKERS)}")

precios_diarios = pd.DataFrame(precios_dict)[tickers_ok]
precios_semanales = precios_diarios.resample("W").last()

retornos_sem = np.log(precios_semanales / precios_semanales.shift(1)).dropna(how="all")

pct_na = retornos_sem.isna().mean()
tickers_limpios = pct_na[pct_na < 0.05].index.tolist()
retornos_sem = retornos_sem[tickers_limpios].dropna()

tickers = tickers_limpios
n = len(tickers)

print(f"Tickers en universo BL: {n}")
if n < len(TICKERS):
    excluidos = [t for t in TICKERS if t not in tickers]
    print(f"  Excluidos: {', '.join(excluidos)}")

Sigma_sem = retornos_sem.cov().values
Sigma_hist = Sigma_sem * factor_anualizacion
D_hist_inv = np.diag(1 / np.sqrt(np.diag(Sigma_hist)))
Corr_hist = D_hist_inv @ Sigma_hist @ D_hist_inv
Sigma_hist_df = pd.DataFrame(Sigma_hist, index=tickers, columns=tickers)
mu_historico = retornos_sem.mean().values * factor_anualizacion

print("\n=== Retornos historicos escalados al horizonte ===")
print(pd.Series(np.round(mu_historico, 4), index=tickers))

# =============================================================================
# BLOQUE 1B: VOLATILIDAD IMPLICITA VIA POLYGON (SSVI)
# =============================================================================

if USAR_IV_POLYGON:

    tau_horizonte = MESES_HORIZONTE / 12

    print("\n=== Extrayendo volatilidad implicita ATM via Polygon (SSVI) ===")
    print(f"Horizonte objetivo (tau): {tau_horizonte:.4f} anios\n")

    # --- 1. Descarga de la cadena de opciones completa (paginado) ----------
    def polygon_fetch_chain(ticker, api_key, max_pages=40):
        url = f"https://api.polygon.io/v3/snapshot/options/{ticker}?limit=250&apiKey={api_key}"
        out = []
        page = 0
        while True:
            try:
                resp = requests.get(url, timeout=20)
            except Exception:
                break
            if resp.status_code != 200:
                print(f"HTTP {resp.status_code} para {ticker}")
                break
            data = resp.json()
            results = data.get("results")
            if results:
                out.extend(results)
            page += 1
            next_url = data.get("next_url")
            if next_url is None or page >= max_pages:
                break
            url = f"{next_url}&apiKey={api_key}"
            time.sleep(0.05)
        return out

    # --- 2. Parseo de la cadena cruda a DataFrame ---------------------------
    def parse_chain(chain_raw):
        filas = []
        for c in chain_raw:
            try:
                details = c.get("details", {})
                last_quote = c.get("last_quote", {}) or {}
                day = c.get("day", {}) or {}
                underlying = c.get("underlying_asset", {}) or {}

                precio_opcion = np.nan
                if last_quote.get("midpoint") is not None:
                    precio_opcion = last_quote["midpoint"]
                elif day.get("close") not in (None, 0):
                    precio_opcion = day["close"]

                filas.append(dict(
                    strike=details.get("strike_price"),
                    expiracion=pd.to_datetime(details.get("expiration_date")),
                    tipo=details.get("contract_type"),
                    iv=c.get("implied_volatility", np.nan),
                    precio=precio_opcion,
                    oi=c.get("open_interest", np.nan),
                    spot=underlying.get("price", np.nan),
                ))
            except Exception:
                continue
        return pd.DataFrame(filas)

    # --- 3. Forward por vencimiento via regresion de paridad put-call -------
    def estimar_forward(df_exp):
        anchos = df_exp[["strike", "tipo", "precio"]].pivot_table(
            index="strike", columns="tipo", values="precio", aggfunc="mean"
        ).reset_index()
        if not {"call", "put"}.issubset(anchos.columns):
            return np.nan
        anchos = anchos.dropna(subset=["call", "put"])
        if len(anchos) < 4:
            return np.nan
        y = (anchos["call"] - anchos["put"]).values
        x = anchos["strike"].values
        try:
            b1, b0 = np.polyfit(x, y, 1)
        except Exception:
            return np.nan
        if b1 >= 0:
            return np.nan
        F_est = -b0 / b1
        if not np.isfinite(F_est) or F_est <= 0:
            return np.nan
        return F_est

    # --- 4. SSVI: funcion de varianza total y power-law phi(theta) ---------
    def phi_powerlaw(theta, eta, gamma):
        return eta * theta ** (-gamma)

    def ssvi_w(k, theta, rho, eta, gamma):
        phi = phi_powerlaw(theta, eta, gamma)
        return theta / 2 * (1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + (1 - rho ** 2)))

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def qlogis(p):
        return math.log(p / (1 - p))

    # --- 5. Calibracion conjunta de la superficie SSVI para un ticker ------
    def calibrar_ssvi_ticker(ticker, api_key, tau_obj,
                              min_strikes=MIN_STRIKES_SLICE,
                              min_dias=MIN_DIAS_VENCIMIENTO):

        chain_raw = polygon_fetch_chain(ticker, api_key)
        if len(chain_raw) == 0:
            raise ValueError("Cadena vacia")

        df = parse_chain(chain_raw)
        df = df[df["iv"].notna() & (df["iv"] > 0)]
        if len(df) == 0:
            raise ValueError("Sin IVs validas")

        hoy = pd.Timestamp(date.today())
        df["dias"] = (df["expiracion"] - hoy).dt.days
        df = df[df["dias"] >= min_dias]

        vencimientos = sorted(df["expiracion"].unique())
        puntos = []
        theta_guess = []
        t_years = []

        for venc in vencimientos:
            df_exp = df[df["expiracion"] == venc]
            T_anios = (pd.Timestamp(venc) - hoy).days / 365

            F_est = estimar_forward(df_exp)
            if pd.isna(F_est):
                continue

            df_exp = df_exp.copy()
            df_exp["k"] = np.log(df_exp["strike"] / F_est)
            otm_put = df_exp[(df_exp["tipo"] == "put") & (df_exp["k"] < 0)]
            otm_call = df_exp[(df_exp["tipo"] == "call") & (df_exp["k"] >= 0)]
            otm = pd.concat([otm_put, otm_call])
            otm = otm.drop_duplicates(subset="strike")
            if len(otm) < min_strikes:
                continue

            otm = otm.copy()
            otm["w"] = otm["iv"] ** 2 * T_anios
            otm = otm.sort_values("k")

            try:
                theta0 = np.interp(0, otm["k"].values, otm["w"].values)
            except Exception:
                theta0 = np.nan
            if pd.isna(theta0) or theta0 <= 0:
                continue

            idx = len(puntos)
            puntos.append(pd.DataFrame({"k": otm["k"].values, "w": otm["w"].values, "slice": idx}))
            theta_guess.append(theta0)
            t_years.append(T_anios)

        if len(puntos) < 2:
            raise ValueError("Menos de 2 vencimientos utilizables")

        datos = pd.concat(puntos, ignore_index=True)
        m = len(theta_guess)
        theta_guess = np.array(theta_guess)
        t_years = np.array(t_years)

        # --- ETAPA 1: theta_j se FIJA en su valor model-free -----------
        theta_fijo = theta_guess

        order = np.argsort(t_years)
        t_sorted = t_years[order]
        theta_sorted = theta_fijo[order]
        theta_interp = PchipInterpolator(t_sorted, theta_sorted, extrapolate=False)

        if tau_obj < t_years.min():
            i_min = np.argmin(t_years)
            theta_tau = theta_fijo[i_min] * (tau_obj / t_years[i_min])
        elif tau_obj > t_years.max():
            i_max = np.argmax(t_years)
            theta_tau = theta_fijo[i_max] * (tau_obj / t_years[i_max])
        else:
            theta_tau = float(theta_interp(tau_obj))

        # --- ETAPA 2: calibracion conjunta de rho, eta, gamma -----------
        theta_por_fila = theta_fijo[datos["slice"].values]

        u0 = np.array([math.atanh(0.0), math.log(1.0), qlogis((0.3 - 0.05) / 0.9)])

        def objetivo(u):
            rho = math.tanh(u[0])
            eta = math.exp(u[1])
            gamma = sigmoid(u[2]) * 0.9 + 0.05

            w_modelo = ssvi_w(datos["k"].values, theta_por_fila, rho, eta, gamma)
            error_ajuste = np.sum((w_modelo - datos["w"].values) ** 2)

            gj = theta_fijo * phi_powerlaw(theta_fijo, eta, gamma) * (1 + abs(rho))
            penalizacion = np.sum(np.maximum(0, gj - 4) ** 2) * 1e3

            return error_ajuste + penalizacion

        opt = minimize(objetivo, u0, method="BFGS", options=dict(maxiter=2000, gtol=1e-10))

        rho = math.tanh(opt.x[0])
        eta = math.exp(opt.x[1])
        gamma = sigmoid(opt.x[2]) * 0.9 + 0.05

        gj_max = np.max(theta_fijo * phi_powerlaw(theta_fijo, eta, gamma) * (1 + abs(rho)))
        sin_arbitraje = gj_max <= 4 + 1e-6

        ssvi_confiable = opt.success and sin_arbitraje
        metodo = "ssvi_conjunto" if ssvi_confiable else "ssvi_no_convergio"

        sigma_atm = math.sqrt(theta_tau / tau_obj)

        return dict(sigma_atm=sigma_atm, theta_j=theta_fijo, t_years=t_years,
                    rho=rho, eta=eta, gamma=gamma, n_vencimientos=m,
                    metodo=metodo, gj_max=gj_max)

    # --- 6. Loop por ticker con fallback individual -------------------------
    sigma_iv = {t: np.nan for t in tickers}
    detalle_ssvi = {}

    for tk in tickers:
        print(f"  Calibrando SSVI: {tk} ... ", end="")
        try:
            resultado = calibrar_ssvi_ticker(tk, POLYGON_API_KEY, tau_horizonte)
        except Exception as e:
            print(f"FALLBACK ({e}) ", end="")
            resultado = None

        if resultado is not None:
            sigma_iv[tk] = resultado["sigma_atm"]
            detalle_ssvi[tk] = resultado
            print(f"OK ({resultado['metodo']}) - sigma_ATM = {resultado['sigma_atm']:.4f} "
                  f"| vencimientos: {resultado['n_vencimientos']} | rho = {resultado['rho']:.3f} "
                  f"| GJ_max = {resultado['gj_max']:.3f} (<=4 sin arbitraje)")
        else:
            sigma_iv[tk] = math.sqrt(Sigma_hist_df.loc[tk, tk])
            print(f"-> vol historica = {sigma_iv[tk]:.4f}")

    print(f"\n=== Volatilidades ATM implicitas (SSVI, horizonte {MESES_HORIZONTE} meses) ===")
    print(pd.Series({t: round(sigma_iv[t], 4) for t in tickers}))

    # --- 7. Sigma final: D_IV . Corr_hist . D_IV ----------------------------
    sigma_iv_vec = np.array([sigma_iv[t] for t in tickers])
    D_IV = np.diag(sigma_iv_vec)
    Sigma = D_IV @ Corr_hist @ D_IV
    Sigma = pd.DataFrame(Sigma, index=tickers, columns=tickers)

else:
    print("\n=== USAR_IV_POLYGON = False - usando Sigma 100% historica ===")
    Sigma = Sigma_hist_df.copy()


# =============================================================================
# BLOQUE 1C: MODULO BKM - MOMENTOS RISK-NEUTRAL DE ORDEN SUPERIOR
# =============================================================================

from scipy.stats import norm

Rf_anual_bkm = Rf


def bs_price(S, K, T, r, sigma, tipo="call"):
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if tipo == "call" else (K - S))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if tipo == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def sigma_desde_ssvi(K, F, T, rho, eta, gamma, theta_tau):
    k = np.log(K / F)
    w = ssvi_w(k, theta_tau, rho, eta, gamma)
    w = max(w, 1e-8)
    return np.sqrt(w / T)


def otm_price_ssvi(K, S, F, T, r, rho, eta, gamma, theta_tau):
    sigma_k = sigma_desde_ssvi(K, F, T, rho, eta, gamma, theta_tau)
    tipo = "put" if K < F else "call"
    return bs_price(S, K, T, r, sigma_k, tipo=tipo)


def calcular_bkm_moments(S, F, T, r, rho, eta, gamma, theta_tau,
                          n_std=6, n_puntos=400):
    sigma_atm = sigma_desde_ssvi(F, F, T, rho, eta, gamma, theta_tau)
    K_min = F * np.exp(-n_std * sigma_atm * np.sqrt(T))
    K_max = F * np.exp(n_std * sigma_atm * np.sqrt(T))
    strikes = np.linspace(K_min, K_max, n_puntos)

    precios = np.array([
        otm_price_ssvi(K, S, F, T, r, rho, eta, gamma, theta_tau)
        for K in strikes
    ])

    lnKF = np.log(strikes / S)

    peso_V = 2.0 * (1 - lnKF) / strikes ** 2
    peso_W = (6.0 * lnKF - 3.0 * lnKF ** 2) / strikes ** 2
    peso_X = (12.0 * lnKF ** 2 - 4.0 * lnKF ** 3) / strikes ** 2

    factor = np.exp(r * T)
    V_T = factor * np.trapz(peso_V * precios, strikes)
    W_T = factor * np.trapz(peso_W * precios, strikes)
    X_T = factor * np.trapz(peso_X * precios, strikes)

    mu_T = (np.exp(r * T) - 1
            - np.exp(r * T) / 2 * V_T
            - np.exp(r * T) / 6 * W_T
            - np.exp(r * T) / 24 * X_T)

    MFIV = np.exp(r * T) * V_T - mu_T ** 2
    if MFIV <= 0 or not np.isfinite(MFIV):
        return dict(MFIV=np.nan, MFIS=np.nan, MFIK=np.nan,
                     V_T=V_T, W_T=W_T, X_T=X_T)

    MFIS = (np.exp(r * T) * W_T - 3 * mu_T * np.exp(r * T) * V_T + 2 * mu_T ** 3) / MFIV ** 1.5
    MFIK = (np.exp(r * T) * X_T - 4 * mu_T * np.exp(r * T) * W_T
            + 6 * mu_T ** 2 * np.exp(r * T) * V_T - 3 * mu_T ** 4) / MFIV ** 2

    return dict(MFIV=MFIV, MFIS=MFIS, MFIK=MFIK, V_T=V_T, W_T=W_T, X_T=X_T)


print("\n=== Calculando momentos BKM (MFIV, MFIS, MFIK) por ticker ===")

bkm_moments = {}
r_bkm = Rf_anual_bkm

for tk in tickers:
    if tk not in detalle_ssvi:
        print(f"  {tk}: sin superficie SSVI valida, se omite BKM (MFIS=0, MFIK=3 neutro)")
        bkm_moments[tk] = dict(MFIV=np.nan, MFIS=0.0, MFIK=3.0, V_T=np.nan, W_T=np.nan, X_T=np.nan)
        continue

    det = detalle_ssvi[tk]
    try:
        S_tk = precios_diarios[tk].iloc[-1]
        F_tk = S_tk * np.exp(r_bkm * tau_horizonte)
        theta_tau_tk = det["sigma_atm"] ** 2 * tau_horizonte
        resultado_bkm = calcular_bkm_moments(
            S=S_tk, F=F_tk, T=tau_horizonte, r=r_bkm,
            rho=det["rho"], eta=det["eta"], gamma=det["gamma"],
            theta_tau=theta_tau_tk,
        )
        bkm_moments[tk] = resultado_bkm
        print(f"  {tk}: MFIV={resultado_bkm['MFIV']:.4f} | MFIS={resultado_bkm['MFIS']:.3f} "
              f"| MFIK={resultado_bkm['MFIK']:.3f}")
    except Exception as e:
        print(f"  {tk}: fallback neutro ({e})")
        bkm_moments[tk] = dict(MFIV=np.nan, MFIS=0.0, MFIK=3.0, V_T=np.nan, W_T=np.nan, X_T=np.nan)

MFIS_vec = np.array([bkm_moments[t]["MFIS"] for t in tickers])
MFIK_vec = np.array([bkm_moments[t]["MFIK"] for t in tickers])

print("\n=== Resumen momentos implicitos (skew=0, kurt=3 => distribucion normal) ===")
print(pd.DataFrame({"MFIS": np.round(MFIS_vec, 3), "MFIK": np.round(MFIK_vec, 3)}, index=tickers))

# =============================================================================
# BLOQUE 2: MARKET CAPS VIA YAHOO FINANCE -> w_mkt
# =============================================================================

print("\n=== Extrayendo market caps via Yahoo Finance ===")


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


market_caps_raw = pd.Series({t: get_market_cap(t) for t in tickers})

print("Market caps extraidos (USD):")
print(market_caps_raw.map(lambda x: f"{x:,.0f}" if not pd.isna(x) else "NA"))

if market_caps_raw.isna().all():
    print("  Yahoo Finance no devolvio datos - usando pesos iguales")
    market_caps_raw = pd.Series(1.0, index=tickers)
else:
    n_na = market_caps_raw.isna().sum()
    if n_na > 0:
        print(f"  Tickers sin market cap ({n_na}) - imputados con mediana del universo")
        market_caps_raw = market_caps_raw.fillna(market_caps_raw.median())

w_mkt = (market_caps_raw / market_caps_raw.sum()).values

print("\nPesos de mercado (w_mkt):")
print(pd.Series(np.round(w_mkt, 4), index=tickers))

# =============================================================================
# BLOQUE 3: PERFIL DE RIESGO -> DELTA, TAU, OMEGA_SCALE
# =============================================================================

perfil = PERFILES[PERFIL_RIESGO]
delta = perfil["delta"]
tau = perfil["tau"]

desc_perfiles = dict(
    conservador="Portafolio cercano al benchmark. Views con poco peso.",
    moderado="Balance entre consenso de mercado y vision activa.",
    agresivo="Portafolio con fuerte inclinacion hacia los views del gestor.",
)

print(f"\n=== PERFIL: {PERFIL_RIESGO.upper()} ===")
print(f"Descripcion: {desc_perfiles[PERFIL_RIESGO]}")
print(f"Delta (d): {delta} | Tau (t): {tau} | Omega scale: {perfil['omega_scale']}")

# =============================================================================
# BLOQUE 4: RETORNOS DE EQUILIBRIO pi - CAPM INVERTIDO
# =============================================================================

Sigma_mat = Sigma.values
pi_eq = delta * (Sigma_mat @ w_mkt)

print("\n=== Retornos de equilibrio pi (prior) ===")
print(pd.Series(np.round(pi_eq, 4), index=tickers))

# =============================================================================
# BLOQUE 5: TABLA DE REFERENCIA PARA VIEWS
# =============================================================================

Rf_h = Rf * (MESES_HORIZONTE / 12)

referencia_views = pd.DataFrame({
    "Ticker": tickers,
    "Pi_eq": np.round(pi_eq, 4),
    "Mu_historico": np.round(mu_historico, 4),
})
referencia_views["Diff_Hist_Pi"] = np.round(referencia_views["Mu_historico"] - referencia_views["Pi_eq"], 4)

print("\n=== REFERENCIA PARA FORMULAR VIEWS (Historico vs pi) ===")
print("(Diff > 0: retorno historico supera el equilibrio de mercado)")
print(referencia_views.sort_values("Diff_Hist_Pi", ascending=False).to_string(index=False))

# =============================================================================
# BLOQUE 6: VIEWS DEL GESTOR <- EDITA AQUI
# =============================================================================

# --- PASO 1: Define cuantos views tienes ------------------------------------
N_VIEWS = 3

# --- PASO 2: Construye la matriz P ------------------------------------------
P = pd.DataFrame(0.0, index=[f"View_{i+1}" for i in range(N_VIEWS)], columns=tickers)

# VIEW 1 - RELATIVO: NVDA outperforma a XOM
P.loc["View_1", "NVDA"] = 1
P.loc["View_1", "XOM"] = -1

# VIEW 2 - RELATIVO: JPM y MA outperforman a JNJ y MRK
P.loc["View_2", "JPM"] = 0.5
P.loc["View_2", "MA"] = 0.5
P.loc["View_2", "JNJ"] = -0.5
P.loc["View_2", "MRK"] = -0.5

# VIEW 3 - ABSOLUTO: MSFT retorna al menos X% en el horizonte
P.loc["View_3", "MSFT"] = 1

print("\n=== Matriz P (views del gestor) ===")
print(P.round(4))

# --- PASO 3: Define el vector Q ---------------------------------------------
Q = pd.Series({
    "View_1": 0.10,
    "View_2": 0.06,
    "View_3": 0.08,
})

print("\n=== Vector Q (magnitudes del gestor) ===")
print(Q.round(4))

# --- PASO 4: Omega (automatico - no editar) ---------------------------------
P_mat = P.values
Q_vec = Q.values.reshape(-1, 1)

Omega_base = np.diag(np.diag(tau * P_mat @ Sigma_mat @ P_mat.T))
Omega = Omega_base * perfil["omega_scale"]

print(f"\n=== Matriz Omega (incertidumbre de views - perfil: {PERFIL_RIESGO}) ===")
print(pd.DataFrame(Omega, index=P.index, columns=P.index).round(6))

# =============================================================================
# BLOQUE 6B: BRIDGE BKM -> BLACK-LITTERMAN (Q, Omega, mu_BL, Sigma_BL)
# =============================================================================

KAPPA_SKEW = 0.15
KAPPA_KURT = 0.20
KURT_BASELINE = 3.0

pi_eq_col = pi_eq.reshape(-1, 1)
tauSigma = tau * Sigma_mat

# --- Ajuste de Q por asimetria negativa -------------------------------------
exposicion_skew_view = P_mat @ MFIS_vec
penalizacion_skew = KAPPA_SKEW * np.minimum(0.0, exposicion_skew_view)
Q_adjusted = Q.values + penalizacion_skew

print("\n=== Ajuste de Q por asimetria implicita negativa ===")
print(pd.DataFrame({
    "Q_original": Q.values,
    "Exposicion_MFIS": np.round(exposicion_skew_view, 4),
    "Penalizacion": np.round(penalizacion_skew, 4),
    "Q_adjusted": np.round(Q_adjusted, 4),
}, index=Q.index))

# --- Ajuste de Omega por exceso de curtosis ---------------------------------
exceso_kurt_vec = np.maximum(0.0, MFIK_vec - KURT_BASELINE)
exposicion_kurt_view = np.abs(P_mat) @ exceso_kurt_vec
factor_omega = 1.0 + KAPPA_KURT * exposicion_kurt_view
Omega_adjusted = np.diag(np.diag(Omega) * factor_omega)

print("\n=== Ajuste de Omega por exceso de curtosis implicita ===")
print(pd.DataFrame({
    "Omega_original": np.diag(Omega),
    "Exceso_MFIK_view": np.round(exposicion_kurt_view, 4),
    "Factor_escala": np.round(factor_omega, 4),
    "Omega_adjusted": np.diag(Omega_adjusted),
}, index=Q.index))

# --- Actualizacion bayesiana con Q_adjusted / Omega_adjusted ----------------
Q_adj_col = Q_adjusted.reshape(-1, 1)
sorpresa_adj = Q_adj_col - P_mat @ pi_eq_col
M_adj = P_mat @ tauSigma @ P_mat.T + Omega_adjusted
M_adj_inv = np.linalg.inv(M_adj)
ajuste_adj = tauSigma @ P_mat.T @ M_adj_inv @ sorpresa_adj
mu_BL_adj = pi_eq_col + ajuste_adj

# --- Sigma_BL posterior formal -----------------------------------------------
Sigma_BL = Sigma_mat + tauSigma - tauSigma @ P_mat.T @ M_adj_inv @ P_mat @ tauSigma
Sigma_BL = pd.DataFrame(Sigma_BL, index=tickers, columns=tickers)

comparacion = pd.DataFrame({
    "Ticker": tickers,
    "Pi_eq": np.round(pi_eq, 4),
    "Mu_hist": np.round(mu_historico, 4),
    "Mu_BL_adj": np.round(mu_BL_adj.flatten(), 4),
})
comparacion["Ajuste_BL"] = np.round(comparacion["Mu_BL_adj"] - comparacion["Pi_eq"], 4)

print("\n=== Comparacion: pi vs Historico vs mu_BL_adj (con ajuste BKM) ===")
print(comparacion.to_string(index=False))

print(f"\nSigma_BL calculada. Norma Frobenius |Sigma_BL - Sigma|: "
      f"{np.linalg.norm(Sigma_BL.values - Sigma_mat):.6f}")

# =============================================================================
# BLOQUE 8B: OPTIMIZACION MEAN-VARIANCE-SKEWNESS-KURTOSIS (MVSK)
# =============================================================================

from scipy.optimize import minimize as scipy_minimize

GAMMA_RA = 3.0
LAMBDA3 = 1.0
LAMBDA4 = 1.0


def construir_tensores_comomento(retornos, sigma_diag_bkm, mfis_bkm, mfik_bkm):
    X = retornos.values - retornos.values.mean(axis=0)
    T_obs, n_ = X.shape

    M3 = np.zeros((n_, n_ * n_))
    M4 = np.zeros((n_, n_ * n_ * n_))

    for i in range(n_):
        for j in range(n_):
            for k in range(n_):
                M3[i, j * n_ + k] = np.mean(X[:, i] * X[:, j] * X[:, k])

    for i in range(n_):
        for j in range(n_):
            for k in range(n_):
                for l in range(n_):
                    M4[i, j * n_ * n_ + k * n_ + l] = np.mean(X[:, i] * X[:, j] * X[:, k] * X[:, l])

    for i in range(n_):
        if np.isfinite(mfis_bkm[i]):
            M3[i, i * n_ + i] = mfis_bkm[i] * sigma_diag_bkm[i] ** 3
        if np.isfinite(mfik_bkm[i]):
            M4[i, i * n_ * n_ + i * n_ + i] = mfik_bkm[i] * sigma_diag_bkm[i] ** 4

    return M3, M4


def portfolio_skew(w, M3):
    kron_ww = np.kron(w, w)
    return float(w @ M3 @ kron_ww)


def portfolio_kurt(w, M4):
    kron_www = np.kron(np.kron(w, w), w)
    return float(w @ M4 @ kron_www)


def utilidad_mvsk_negativa(w, mu, Sigma_arr, M3, M4, gamma, lam3, lam4):
    ret = w @ mu
    var = w @ Sigma_arr @ w
    skew = portfolio_skew(w, M3)
    kurt = portfolio_kurt(w, M4)
    utilidad = ret - (gamma / 2) * var + (lam3 / 3) * skew - (lam4 / 4) * kurt
    return -utilidad


sigma_diag_para_bkm = np.sqrt(np.diag(Sigma_BL.values))

print("\n=== Construyendo tensores de co-skewness / co-kurtosis ===")
M3, M4 = construir_tensores_comomento(retornos_sem[tickers], sigma_diag_para_bkm, MFIS_vec, MFIK_vec)
print(f"M3 shape: {M3.shape} | M4 shape: {M4.shape}")

mu_opt = mu_BL_adj.flatten()
Sigma_opt = Sigma_BL.values

w0 = np.ones(n) / n
bounds = [(0.0, 1.0)] * n
constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

resultado_mvsk = scipy_minimize(
    utilidad_mvsk_negativa, w0,
    args=(mu_opt, Sigma_opt, M3, M4, GAMMA_RA, LAMBDA3, LAMBDA4),
    method="SLSQP", bounds=bounds, constraints=constraints,
    options=dict(maxiter=500, ftol=1e-10),
)

if not resultado_mvsk.success:
    print(f"  Aviso: el optimizador SLSQP no convergio limpiamente ({resultado_mvsk.message})")

w_mvsk = resultado_mvsk.x.copy()
w_mvsk[w_mvsk < UMBRAL_PESO_MIN] = 0
w_mvsk = w_mvsk / w_mvsk.sum()

if np.sum(w_mvsk > 0) > MAX_TICKERS_FINAL:
    idx_top = np.argsort(w_mvsk)[::-1][:MAX_TICKERS_FINAL]
    w_final = np.zeros(n)
    w_final[idx_top] = w_mvsk[idx_top]
    w_mvsk = w_final / w_final.sum()

w_mvsk = pd.Series(w_mvsk, index=tickers)

print("\n=== Pesos optimos - Portafolio MVSK (BL + BKM) ===")
print(w_mvsk[w_mvsk > 0].round(4))

# =============================================================================
# BLOQUE 9: METRICAS DEL PORTAFOLIO
# =============================================================================

etiqueta_horizonte = f"{MESES_HORIZONTE} mes(es)"

w_mvsk_vec = w_mvsk.values
mu_BL_vec = mu_BL_adj.flatten()

ret_port = float(w_mvsk_vec @ mu_BL_vec)
var_port = float(w_mvsk_vec @ Sigma_opt @ w_mvsk_vec)
vol_port = math.sqrt(var_port)
skew_port = portfolio_skew(w_mvsk_vec, M3)
kurt_port = portfolio_kurt(w_mvsk_vec, M4)
Rf_h = Rf * (MESES_HORIZONTE / 12)
sharpe_BL = (ret_port - Rf_h) / vol_port

ret_mkt = float(w_mkt @ mu_BL_vec)
var_mkt = float(w_mkt @ Sigma_opt @ w_mkt)
vol_mkt = math.sqrt(var_mkt)
sharpe_mkt = (ret_mkt - Rf_h) / vol_mkt

print(f"\n=== Metricas - Portafolio MVSK ({etiqueta_horizonte}) ===")
print(f"Retorno esperado:   {ret_port:.4f}")
print(f"Volatilidad:        {vol_port:.4f}")
print(f"Skewness:            {skew_port:.6f}")
print(f"Kurtosis:            {kurt_port:.6f}")
print(f"Sharpe Ratio:        {sharpe_BL:.4f}")
print(f"Rf horizonte:        {Rf_h:.4f}")

print(f"\n=== Metricas - Portafolio de mercado ({etiqueta_horizonte}) ===")
print(f"Retorno esperado:   {ret_mkt:.4f}")
print(f"Volatilidad:        {vol_mkt:.4f}")
print(f"Sharpe Ratio:       {sharpe_mkt:.4f}")

# =============================================================================
# BLOQUE 10: VISUALIZACION
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 11))

ax = axes[0, 0]
x = np.arange(n)
width = 0.27
ax.bar(x - width, comparacion["Pi_eq"], width, color="#4472C4", label="pi (Equilibrio)")
ax.bar(x, comparacion["Mu_hist"], width, color="#FFC000", label="Historico")
ax.bar(x + width, comparacion["Mu_BL_adj"], width, color="#ED7D31", label="mu_BL_adj (Posterior BKM)")
ax.set_xticks(x)
ax.set_xticklabels(comparacion["Ticker"], rotation=90, fontsize=8)
ax.set_title(f"Retornos esperados - {etiqueta_horizonte}", fontweight="bold")
ax.set_ylabel("Retorno esperado")
ax.legend(fontsize=8)

ax = axes[0, 1]
ax.bar(x - width / 2, w_mkt, width, color="#4472C4", label="Mercado")
ax.bar(x + width / 2, w_mvsk_vec, width, color="#70AD47", label=f"MVSK: {PERFIL_RIESGO.upper()}")
ax.set_xticks(x)
ax.set_xticklabels(tickers, rotation=90, fontsize=8)
ax.set_title("Pesos: Mercado vs MVSK", fontweight="bold")
ax.set_ylabel("Peso")
ax.legend(fontsize=8)

ax = axes[1, 0]
sharpes_perfiles = {}
for nm, pf in PERFILES.items():
    Om_sim = np.diag(np.diag(pf["tau"] * P_mat @ Sigma_mat @ P_mat.T)) * pf["omega_scale"]
    M_sim = P_mat @ (pf["tau"] * Sigma_mat) @ P_mat.T + Om_sim
    try:
        aj_sim = (pf["tau"] * Sigma_mat) @ P_mat.T @ np.linalg.inv(M_sim) @ (Q_adj_col - P_mat @ pi_eq_col)
        mu_sim = pi_eq_col + aj_sim
        Dm_sim = pf["delta"] * Sigma_mat * 2 + np.eye(n) * 1e-10
        dv_sim = mu_sim.flatten()
        Amat_sim = np.column_stack([np.ones(n), np.eye(n)])
        bvec_sim = np.concatenate([[1.0], np.zeros(n)])
        op_sim = quadprog.solve_qp(Dm_sim, dv_sim, Amat_sim, bvec_sim, meq=1)[0]
    except Exception:
        op_sim = w_mkt.copy()
        mu_sim = mu_BL_vec.reshape(-1, 1)
    op_sim = np.where(op_sim < UMBRAL_PESO_MIN, 0, op_sim)
    if op_sim.sum() > 0:
        op_sim = op_sim / op_sim.sum()
    r_sim = float(op_sim @ mu_sim.flatten())
    v_sim = math.sqrt(float(op_sim @ Sigma_mat @ op_sim))
    sharpes_perfiles[nm] = (r_sim - Rf_h) / v_sim

nombres_perfiles = ["conservador", "moderado", "agresivo"]
valores_sharpe = [sharpes_perfiles[nm] for nm in nombres_perfiles]
ax.bar(["Conservador", "Moderado", "Agresivo"], valores_sharpe,
       color=["#70AD47", "#FFC000", "#ED7D31"])
ax.axhline(sharpe_mkt, color="#4472C4", linestyle="--", linewidth=2, label="Mercado")
ax.set_title("Sharpe por perfil de riesgo (MV, referencia)", fontweight="bold")
ax.set_ylabel("Sharpe Ratio")
ax.set_ylim(0, max(valores_sharpe) * 1.3)
ax.legend(fontsize=8)

ax = axes[1, 1]
comp_sorted = comparacion.sort_values("Ajuste_BL")
colores_ajuste = ["#70AD47" if v > 0 else "#ED7D31" for v in comp_sorted["Ajuste_BL"]]
ax.bar(comp_sorted["Ticker"], comp_sorted["Ajuste_BL"], color=colores_ajuste)
ax.axhline(0, color="black", linewidth=1.2)
ax.set_xticklabels(comp_sorted["Ticker"], rotation=90, fontsize=8)
ax.set_title("Ajuste mu_BL_adj vs pi (impacto de views + BKM)", fontweight="bold")
ax.set_ylabel("mu_BL_adj - pi")

plt.tight_layout()
plt.show()

# =============================================================================
# BLOQUE 11: MAXIMUM DRAWDOWN (MDD) DEL PORTAFOLIO MVSK
# =============================================================================

print("\n=== ANALISIS DE MAXIMUM DRAWDOWN DEL PORTAFOLIO ===")
print(f"Periodo de analisis MDD: desde {MDD_START_YEAR}")


def calc_mdd(r):
    r = pd.Series(r).dropna()
    if len(r) < 2:
        return np.nan
    cv = (1 + r).cumprod()
    dd = (cv - cv.cummax()) / cv.cummax()
    mdd = dd.min()
    if not np.isfinite(mdd):
        return np.nan
    return mdd


tickers_bl = w_mvsk[w_mvsk > 0].index.tolist()
weights_bl = w_mvsk[tickers_bl]

fecha_mdd_inicio = date(MDD_START_YEAR, 1, 1)

series_ok = {}
for tk in tickers_bl:
    serie = descargar_precio(tk, fecha_mdd_inicio, date.today())
    if serie is None or len(serie) == 0:
        alt = tk.replace(".", "-") if "." in tk else tk.replace("-", ".")
        serie = descargar_precio(alt, fecha_mdd_inicio, date.today())
        if serie is not None and len(serie) > 0:
            print(f"  Nota: {tk} recuperado como {alt}")
        else:
            print(f"  Aviso: no se pudo obtener precio de {tk} para MDD - se excluye de este analisis")
            continue
    series_ok[tk] = serie

if len(series_ok) == 0:
    precios_mdd = None
    print("  Error preparando precios para MDD: ningun ticker del portafolio tiene precios disponibles")
else:
    precios_mdd = pd.DataFrame(series_ok)
    tickers_bl = [t for t in tickers_bl if t in precios_mdd.columns]
    weights_bl = w_mvsk[tickers_bl]
    weights_bl = weights_bl / weights_bl.sum()

if precios_mdd is not None and len(precios_mdd) >= 10:

    retornos_diarios_mdd = np.log(precios_mdd / precios_mdd.shift(1)).dropna(how="all")

    def port_ret_row(fila):
        validos = fila.notna()
        if validos.sum() == 0:
            return np.nan
        w_norm = weights_bl[validos.index[validos]] / weights_bl[validos.index[validos]].sum()
        return float((fila[validos] * w_norm).sum())

    retornos_diarios_mdd = retornos_diarios_mdd[tickers_bl]
    port_ret = retornos_diarios_mdd.apply(port_ret_row, axis=1)
    port_ret = port_ret[port_ret.notna() & np.isfinite(port_ret)]

    print(f"  Observaciones diarias validas: {len(port_ret)}")
    print(f"  Cobertura: {port_ret.index.min():%Y-%m-%d} a {port_ret.index.max():%Y-%m-%d}")

    mdd_global = calc_mdd(port_ret)
    print(f"\n  MDD Global ({MDD_START_YEAR} - hoy): {mdd_global * 100:.2f}%")

    years_series = port_ret.index.year
    anios_disponibles = sorted(years_series.unique())

    mdd_anual_rows = []
    for y in anios_disponibles:
        r_y = port_ret[years_series == y]
        mdd_anual_rows.append(dict(year=y, mdd=calc_mdd(r_y), n_obs=len(r_y)))
    mdd_anual = pd.DataFrame(mdd_anual_rows)
    mdd_anual = mdd_anual[np.isfinite(mdd_anual["mdd"])]

    print("\n  MDD por ano:")
    disp = mdd_anual.copy()
    disp["mdd_pct"] = (disp["mdd"] * 100).round(2)
    print(disp[["year", "mdd_pct", "n_obs"]].to_string(index=False))

    print("\n=== ESTADISTICAS DE MDD ===")

    if len(mdd_anual) >= 3:
        q1 = mdd_anual["mdd"].quantile(0.25)
        q3 = mdd_anual["mdd"].quantile(0.75)
        iqr = q3 - q1
        mdd_clean = mdd_anual[(mdd_anual["mdd"] >= q1 - 1.5 * iqr) & (mdd_anual["mdd"] <= q3 + 1.5 * iqr)]
        if len(mdd_clean) < 2:
            mdd_clean = mdd_anual
    else:
        mdd_clean = mdd_anual

    mdd_mediana = mdd_clean["mdd"].median()
    mdd_p90 = mdd_clean["mdd"].quantile(0.90)
    mdd_media = mdd_clean["mdd"].mean()
    mdd_peor = mdd_clean["mdd"].min()
    mdd_mejor = mdd_clean["mdd"].max()

    print(f"  Peor escenario historico:      {mdd_peor * 100:.2f}%")
    print(f"  Escenario conservador (P90):   {mdd_p90 * 100:.2f}%")
    print(f"  Escenario tipico (mediana):    {mdd_mediana * 100:.2f}%")
    print(f"  Promedio:                      {mdd_media * 100:.2f}%")
    print(f"  Mejor escenario historico:     {mdd_mejor * 100:.2f}%")

    ultimo_anio = mdd_anual.iloc[-1]
    print(f"  MDD mas reciente ({int(ultimo_anio['year'])}):         {ultimo_anio['mdd'] * 100:.2f}%")

    anios_completos = pd.DataFrame({"year": range(MDD_START_YEAR, date.today().year + 1)})
    plot_data = anios_completos.merge(mdd_anual, on="year", how="left")
    plot_data["mdd_pct"] = plot_data["mdd"] * 100
    plot_data["con_dato"] = plot_data["mdd_pct"].notna()

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(plot_data["year"], plot_data["mdd_pct"], color="darkred", linewidth=1.5)
    colors = plot_data["con_dato"].map({True: "darkred", False: "gray"})
    ax.scatter(plot_data["year"], plot_data["mdd_pct"], c=colors, s=40, zorder=3)
    ax.axhline(mdd_mediana * 100, linestyle="--", color="steelblue", linewidth=1.2)
    ax.axhline(mdd_p90 * 100, linestyle="--", color="darkorange", linewidth=1.2)
    ax.text(plot_data["year"].min() + 0.2, mdd_mediana * 100 + 0.3,
            f"Mediana: {mdd_mediana * 100:.2f}%", color="steelblue", fontsize=9)
    ax.text(plot_data["year"].min() + 0.2, mdd_p90 * 100 + 0.3,
            f"P90: {mdd_p90 * 100:.2f}%", color="darkorange", fontsize=9)
    ax.set_xticks(range(MDD_START_YEAR, date.today().year + 1))
    plt.xticks(rotation=45)
    ax.set_title("Maximum Drawdown Historico - Portafolio MVSK (BL + BKM)", fontweight="bold", fontsize=13)
    ax.set_xlabel("Ano")
    ax.set_ylabel("MDD (%)")
    plt.suptitle(f"Perfil: {PERFIL_RIESGO.upper()} | Horizonte: {etiqueta_horizonte} | "
                 f"MDD global ({MDD_START_YEAR}-hoy): {mdd_global * 100:.2f}% | "
                 f"{int(plot_data['con_dato'].sum())} anos con datos", fontsize=9, y=0.96)
    plt.tight_layout()
    plt.show()
    print("  Grafico de MDD generado.")

else:
    print("  Datos insuficientes para calcular MDD (minimo 10 observaciones).")
    print("  Verifica que MDD_START_YEAR este dentro de la ventana de 2 anios.")

print("\n=== FIN - BLACK-LITTERMAN + BKM + MVSK ===")
print(f"Perfil: {PERFIL_RIESGO.upper()} | Horizonte: {etiqueta_horizonte} | "
      f"Universo: {n} tickers | Views: {N_VIEWS} | MDD desde: {MDD_START_YEAR}")