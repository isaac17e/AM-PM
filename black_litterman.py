# =============================================================================
# BLACK-LITTERMAN EXTENDIDO POR MOMENTOS DE ORDEN SUPERIOR MODEL-FREE (BKM)
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
from scipy.optimize import minimize, linprog
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm
from scipy import sparse
import quadprog

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- statsmodels es opcional: si no esta, se usa un HAC propio --------------
try:
    import statsmodels.api as sm
    HAY_STATSMODELS = True
except ImportError:
    HAY_STATSMODELS = False
    print("Aviso: statsmodels no disponible. Se usara un estimador HAC interno "
          "(Newey-West con kernel de Bartlett) para las primas de riesgo.")

# ------------------------------------------------
# API KEY - Polygon.io
# ------------------------------------------------
from dotenv import load_dotenv
load_dotenv()
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY")

# -----------------------------------------------------------------------------
# 1. UNIVERSO DE TICKERS
# -----------------------------------------------------------------------------
TICKERS = [
    "META", "GOOGL", "ORCL", "DELL", "MSFT",
    "BLK", "CRM", "CMCSA", "GS", "REGN",
    "ABNB", "ARES", "LVS", "BXP", "CYBR",
    "YELP", "EBAY", "IT", "EL"
]

# -----------------------------------------------------------------------------
# 2. HORIZONTE TEMPORAL
# -----------------------------------------------------------------------------
MESES_HORIZONTE = 4
DIAS_HABILES_MES = 21
SEMANAS_MES = 4.33

# -----------------------------------------------------------------------------
# 3. PERFIL DE RIESGO
# -----------------------------------------------------------------------------
PERFIL_RIESGO = "agresivo"

# -----------------------------------------------------------------------------
# 4. PARAMETROS DE OPTIMIZACION (por perfil)
# -----------------------------------------------------------------------------
PERFILES = {
    "conservador": dict(omega_scale=5.0, tau=0.025, gamma_ra=6.0),
    "moderado": dict(omega_scale=1.0, tau=0.05, gamma_ra=3.0),
    "agresivo": dict(omega_scale=0.25, tau=0.10, gamma_ra=1.5),
}

# -----------------------------------------------------------------------------
# 5. LIMITES DEL PORTAFOLIO FINAL
# -----------------------------------------------------------------------------
UMBRAL_PESO_MIN = 0.005
MAX_TICKERS_FINAL = 10
PESO_MAX_ACTIVO = 0.35          # cota superior por activo en la optimizacion

# --- Validacion de factibilidad de cardinalidad -----------------------------
# Con sum(w) = 1 y 0 <= w_i <= PESO_MAX_ACTIVO sobre un soporte de a lo sumo
# MAX_TICKERS_FINAL activos, el problema es infactible si
# MAX_TICKERS_FINAL * PESO_MAX_ACTIVO < 1.0 (no alcanza para sumar 1). Se
# corrige automaticamente subiendo PESO_MAX_ACTIVO al minimo necesario, con
# un margen pequeno para evitar quedar justo en el borde numerico.
_cap_cardinalidad = MAX_TICKERS_FINAL * PESO_MAX_ACTIVO
if _cap_cardinalidad < 1.0:
    _peso_max_previo = PESO_MAX_ACTIVO
    PESO_MAX_ACTIVO = min(1.0, (1.0 / MAX_TICKERS_FINAL) * 1.05)
    print(f"  AVISO: MAX_TICKERS_FINAL({MAX_TICKERS_FINAL}) x "
          f"PESO_MAX_ACTIVO({_peso_max_previo:.4f}) = {_cap_cardinalidad:.4f} < 1.0 "
          f"=> el optimizador quedaria infactible en sum(w)=1. "
          f"Se ajusta PESO_MAX_ACTIVO a {PESO_MAX_ACTIVO:.4f}.")

# -----------------------------------------------------------------------------
# 6. TASA LIBRE DE RIESGO - FALLBACK
# -----------------------------------------------------------------------------
Rf = 0.046

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

# -----------------------------------------------------------------------------
# 9. MODULO ECONOMETRICO Q -> P (BLOQUE 1D)
# -----------------------------------------------------------------------------
PASO_VENTANA_ROLLING = 5        # paso (en dias habiles) entre ventanas rodantes
MIN_VENTANAS_ROLLING = 12       # ventanas minimas para aceptar la estimacion
NW_LAGS_AUTO = True             # rezagos NW = floor(4*(T/100)^(2/9)) si True
NW_LAGS_FIJOS = 6               # usado solo si NW_LAGS_AUTO = False
WINSOR_MOMENTOS = 0.05          # winsorizacion de colas de momentos realizados

# Bootstrap por bloques: estimador primario de los momentos fisicos (ver 1D.1b).
N_REP_BOOTSTRAP_MOM = 20        # replicas del bootstrap para los momentos P
J_POR_REPLICA_MOM = 2000        # trayectorias por replica
N_MC_DELTA = 200                # replicas del delta-method de la prima no gaussiana

# Rango admisible del parametro de Esscher. Sobre log-retornos coincide con la
# aversion relativa al riesgo del agente representativo: por eso se restringe a
# valores no negativos (un theta < 0 describiria un agente amante del riesgo).
THETA_ESSCHER_COTA = (0.0, 25.0)
# Difusion relativa del ancla del GMM de theta (ver Bloque 4B).
THETA_PRIOR_CV = 1.0
# Guardarrail economico: la prima no gaussiana no puede exceder esta fraccion
# de la volatilidad fisica del activo. Se informa cuando actua.
MAX_PRIMA_HM_SIGMA = 0.35

# Cotas de sensatez sobre los momentos proyectados a P (evitan que un outlier
# de la superficie de opciones contamine todo el posterior).
COTA_SKEW_P = (-2.5, 1.5)
COTA_KURT_P = (1.8, 12.0)
# Piso/techo del ratio sigma_P / sigma_Q (la vol implicita sobreestima la
# realizada, pero la correccion no puede ser arbitrariamente grande).
COTA_RATIO_VOL_P = (0.55, 1.25)

# -----------------------------------------------------------------------------
# 10. [NUEVO] INTEGRACION BAYESIANA NO GAUSSIANA (BLOQUE 7)
# -----------------------------------------------------------------------------
METODO_POSTERIOR = "entropy_pooling"   # "entropy_pooling" | "gram_charlier"
N_ESCENARIOS = 12000                   # tamano del panel de escenarios
# Longitud media del bloque. Debe ser una fraccion apreciable del horizonte: si
# es muy corta, el retorno a H dias es suma de muchos bloques casi independientes
# y el TLC borra artificialmente la asimetria y la curtosis del agregado.
BOOTSTRAP_BLOQUE = max(21, (MESES_HORIZONTE * DIAS_HABILES_MES) // 4)
HALF_LIFE_PRIOR = 252                  # vida media (dias) del decay del prior
SEMILLA = 20260904
EP_IMPONER_CURTOSIS = True             # incluir restricciones de 4.o momento
EP_TOL_ENS = 0.10                      # ENS minimo aceptable (fraccion de J)

# -----------------------------------------------------------------------------
# 11. [NUEVO] RIESGO DE COLA Y MODO DE OPTIMIZACION (BLOQUES 7B / 8B)
# -----------------------------------------------------------------------------
NIVEL_CONFIANZA_VAR = 0.95
NIVELES_CVAR = (0.95, 0.99)
UMBRAL_OMEGA_RATIO = 0.0        # umbral tau del Omega ratio (en exceso de 0)

MODO_OPTIMIZACION = "mvsk"      # "mvsk" | "cvar"
# (a) "mvsk": maximiza la utilidad esperada por expansion de Taylor de 4.o orden
#             U = E[r] - (gamma/2) Var + (lambda3/3) Skew - (lambda4/4) Kurt
# (b) "cvar": minimiza CVaR_alpha sujeto a E[r] >= RETORNO_MIN_CVAR
LAMBDA3 = 1.0
LAMBDA4 = 1.0
ALPHA_CVAR_OBJETIVO = 0.95
# Retorno minimo exigido en el modo CVaR. None => se usa el retorno esperado
# del portafolio de mercado bajo el posterior (restriccion "al menos como el
# benchmark").
RETORNO_MIN_CVAR = None
MAX_ESCENARIOS_LP = 4000        # submuestreo de escenarios para el LP de CVaR

if USAR_IV_POLYGON and not POLYGON_API_KEY:
    raise ValueError(
        "USAR_IV_POLYGON = True pero POLYGON_API_KEY no esta definida. "
        "Configura el secreto 'PolygonAPI' en Colab, o pon USAR_IV_POLYGON = False "
        "para usar el metodo historico."
    )

rng_global = np.random.default_rng(SEMILLA)


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
    # Definiciones minimas para que el Bloque 1C (BKM) y el 1D (Q -> P) operen
    # en modo neutro: sin superficie SSVI no hay momentos risk-neutral, y las
    # primas de riesgo quedan implicitamente en cero.
    tau_horizonte = MESES_HORIZONTE / 12
    detalle_ssvi = {}


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


_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


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
    V_T = factor * _trapz(peso_V * precios, strikes)
    W_T = factor * _trapz(peso_W * precios, strikes)
    X_T = factor * _trapz(peso_X * precios, strikes)

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
# BLOQUE 1D: MODULO ECONOMETRICO Q -> P
# -----------------------------------------------------------------------------
# Los momentos BKM del Bloque 1C viven bajo la medida risk-neutral Q, mientras
# que Black-Litterman opera bajo la fisica P. Aqui se estiman los momentos
# fisicos al horizonte, se calibran contra los implicitos por regresion de
# Mincer-Zarnowitz y se obtienen las primas VRP, SRP y KRP y la covarianza
# Sigma_P. La proyeccion Q -> P se cierra en el Bloque 4B.
# =============================================================================

print("\n" + "=" * 79)
print("BLOQUE 1D: AJUSTE ECONOMETRICO Q -> P (VRP / SRP / KRP)")
print("=" * 79)

retornos_dia = np.log(precios_diarios / precios_diarios.shift(1)).dropna(how="all")
retornos_dia = retornos_dia[tickers]

H_VENTANA = horizonte_dias   # ventana / agregacion = horizonte de inversion

R_dia = retornos_dia.dropna().values
T_dia = R_dia.shape[0]

# Ponderacion temporal del prior: decaimiento exponencial con vida media fija.
peso_tiempo = np.exp(-np.log(2.0) * (T_dia - 1 - np.arange(T_dia)) / HALF_LIFE_PRIOR)
peso_tiempo = peso_tiempo / peso_tiempo.sum()


# -----------------------------------------------------------------------------
# 1D.0  Bootstrap estacionario por bloques (compartido con el Bloque 7)
# -----------------------------------------------------------------------------

def bootstrap_estacionario(R, J, H, L_bloque, pesos_inicio, rng, chunk=2000):
    """Panel (J x n) de log-retornos agregados a H dias.

    Se remuestrean FILAS COMPLETAS, de modo que la dependencia transversal
    (correlaciones y colas conjuntas) se preserva sin imponer copula alguna.
    Con probabilidad 1/L se salta a un nuevo indice inicial (muestreado con
    decaimiento exponencial en el tiempo); si no, se avanza un dia con
    envoltura circular. Longitudes de bloque geometricas => estacionariedad
    del esquema de remuestreo (Politis-Romano, 1994).
    """
    T_, n_ = R.shape
    p_salto = 1.0 / max(L_bloque, 1)
    salida = np.empty((J, n_))
    hecho = 0
    while hecho < J:
        m = min(chunk, J - hecho)
        idx = np.empty((m, H), dtype=np.int64)
        idx[:, 0] = rng.choice(T_, size=m, p=pesos_inicio)
        saltos = rng.random((m, H - 1)) < p_salto
        nuevos = rng.choice(T_, size=(m, H - 1), p=pesos_inicio)
        for h in range(1, H):
            avance = (idx[:, h - 1] + 1) % T_
            idx[:, h] = np.where(saltos[:, h - 1], nuevos[:, h - 1], avance)
        salida[hecho:hecho + m] = R[idx].sum(axis=1)
        hecho += m
    return salida


# -----------------------------------------------------------------------------
# 1D.1a  Estimador de ventanas rodantes (diagnostico)
# -----------------------------------------------------------------------------

def momentos_realizados_rolling(serie_diaria, H=H_VENTANA, paso=PASO_VENTANA_ROLLING):
    """Momentos realizados del retorno agregado a H dias, en ventanas rodantes.

        RV_t    = sum_{d in t} r_d^2                    (exacto, sin supuestos)
        RSkew_t = sum r_d^3 / RV_t^{3/2}                (agregacion iid)
        RKurt_t = 3 + (H*sum r_d^4/RV_t^2 - 3)/H        (agregacion iid)

    RV_t es la varianza realizada del retorno a H dias. Las otras dos suponen
    incrementos iid (Skew_H = Skew_d/sqrt(H), ExKurt_H = ExKurt_d/H), lo que
    hace que converjan MECANICAMENTE a 0 y 3 al crecer H: a 4 meses ese
    estimador es vacio y por eso solo se conserva como diagnostico frente al
    bootstrap por bloques, que es el estimador primario.
    """
    r = pd.Series(serie_diaria).dropna().values
    if len(r) < H + paso:
        return np.array([]), np.array([]), np.array([])

    rv_l, rs_l, rk_l = [], [], []
    for fin in range(H, len(r) + 1, paso):
        w = r[fin - H:fin]
        rv = float(np.sum(w ** 2))
        if rv <= 0 or not np.isfinite(rv):
            continue
        rs = float(np.sum(w ** 3) / rv ** 1.5)
        kurt_d = float(H * np.sum(w ** 4) / rv ** 2)
        rk = 3.0 + (kurt_d - 3.0) / H
        if not (np.isfinite(rs) and np.isfinite(rk)):
            continue
        rv_l.append(rv); rs_l.append(rs); rk_l.append(rk)
    return np.array(rv_l), np.array(rs_l), np.array(rk_l)


def winsorizar(x, p=WINSOR_MOMENTOS):
    """Winsorizacion simetrica: acota outliers sin descartar observaciones."""
    if len(x) == 0 or p <= 0:
        return x
    lo, hi = np.quantile(x, [p, 1 - p])
    return np.clip(x, lo, hi)


def _nw_lags(T, H=H_VENTANA, paso=PASO_VENTANA_ROLLING):
    """Rezagos Newey-West. Como minimo el solapamiento mecanico H/paso."""
    if NW_LAGS_AUTO:
        L = int(np.floor(4 * (max(T, 2) / 100.0) ** (2.0 / 9.0)))
    else:
        L = NW_LAGS_FIJOS
    L_solape = int(np.ceil(H / max(paso, 1))) - 1
    return int(max(1, min(max(L, L_solape), max(1, T - 2))))


def media_hac(x):
    """Media muestral y error estandar robusto a heterocedasticidad y
    autocorrelacion (Newey-West / Bartlett). Necesario porque las ventanas
    rodantes se solapan y por tanto estan fuertemente autocorrelacionadas."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    T = len(x)
    if T == 0:
        return np.nan, np.nan, 0
    if T == 1:
        return float(x[0]), np.nan, 1

    L = _nw_lags(T)
    if HAY_STATSMODELS:
        try:
            mod = sm.OLS(x, np.ones(T)).fit(cov_type="HAC",
                                            cov_kwds=dict(maxlags=L, use_correction=True))
            return float(mod.params[0]), float(mod.bse[0]), T
        except Exception:
            pass

    mu = float(np.mean(x))
    e = x - mu
    S = float(np.dot(e, e) / T)
    for l in range(1, L + 1):
        S += 2.0 * (1.0 - l / (L + 1.0)) * float(np.dot(e[l:], e[:-l]) / T)
    return mu, float(np.sqrt(max(S, 1e-18) / T)), T


filas_roll = []
for tk in tickers:
    rv, rs, rk = momentos_realizados_rolling(retornos_dia[tk])
    if len(rv) < MIN_VENTANAS_ROLLING:
        filas_roll.append(dict(ticker=tk, n_ventanas=len(rv),
                               RV_med=float(Sigma_hist_df.loc[tk, tk]), RV_se=np.nan,
                               RS_roll=0.0, RK_roll=3.0))
        continue
    rv, rs, rk = winsorizar(rv), winsorizar(rs), winsorizar(rk)
    rv_m, rv_se, _ = media_hac(rv)
    rs_m, _, _ = media_hac(rs)
    rk_m, _, _ = media_hac(rk)
    filas_roll.append(dict(ticker=tk, n_ventanas=len(rv),
                           RV_med=rv_m, RV_se=rv_se, RS_roll=rs_m, RK_roll=rk_m))

rolling = pd.DataFrame(filas_roll).set_index("ticker").loc[tickers]

# -----------------------------------------------------------------------------
# 1D.1b  Estimador bootstrap de la distribucion a H dias (primario)
# -----------------------------------------------------------------------------

def momentos_horizonte_bootstrap(R, H, n_rep=N_REP_BOOTSTRAP_MOM,
                                 J_rep=J_POR_REPLICA_MOM, rng=None):
    """Momentos fisicos del retorno a H dias y su error estandar.

    Estimador PRIMARIO de la asimetria y la curtosis fisicas: los bloques
    preservan la agrupacion de volatilidad y los saltos, de modo que la
    distribucion a H dias no converge artificialmente a la normal como si
    ocurre bajo la agregacion iid de las ventanas rodantes.

    Cada replica genera J_rep trayectorias por bootstrap estacionario y calcula
    varianza, asimetria y curtosis de la distribucion agregada. La media entre
    replicas es el estimador puntual; la desviacion estandar entre replicas es
    el error estandar (variabilidad de remuestreo).
    """
    rng = rng_global if rng is None else rng
    n_ = R.shape[1]
    var_r = np.empty((n_rep, n_)); sk_r = np.empty((n_rep, n_)); ku_r = np.empty((n_rep, n_))
    for m in range(n_rep):
        Xb = bootstrap_estacionario(R, J_rep, H, BOOTSTRAP_BLOQUE, peso_tiempo, rng)
        c = Xb - Xb.mean(axis=0)
        v = (c ** 2).mean(axis=0)
        sd = np.sqrt(np.maximum(v, 1e-18))
        var_r[m] = v
        sk_r[m] = (c ** 3).mean(axis=0) / sd ** 3
        ku_r[m] = (c ** 4).mean(axis=0) / sd ** 4
    return (var_r.mean(0), var_r.std(0, ddof=1),
            sk_r.mean(0), sk_r.std(0, ddof=1),
            ku_r.mean(0), ku_r.std(0, ddof=1))


print(f"\nEstimando momentos fisicos a {H_VENTANA} dias por bootstrap estacionario "
      f"({N_REP_BOOTSTRAP_MOM} replicas x {J_POR_REPLICA_MOM} trayectorias)...")

(var_bs, var_bs_se, skew_bs, skew_bs_se,
 kurt_bs, kurt_bs_se) = momentos_horizonte_bootstrap(R_dia, H_VENTANA)

# Varianza fisica: se usa la varianza realizada rodante (estimador exacto y sin
# supuestos) y su error estandar HAC; el bootstrap sirve de contraste.
var_P_est = rolling["RV_med"].values.copy()
var_P_se = np.where(np.isfinite(rolling["RV_se"].values), rolling["RV_se"].values,
                    var_bs_se)
var_P_se = np.where(var_P_se > 0, var_P_se, 0.10 * var_P_est + 1e-10)

skew_P_est, skew_P_se = skew_bs, np.maximum(skew_bs_se, 1e-4)
kurt_P_est, kurt_P_se = kurt_bs, np.maximum(kurt_bs_se, 1e-4)

print("\n=== Momentos fisicos estimados al horizonte ===")
print(pd.DataFrame({
    "Var_P": np.round(var_P_est, 5), "se": np.round(var_P_se, 5),
    "Var_P_bootstrap": np.round(var_bs, 5),
    "Skew_P": np.round(skew_P_est, 3), "se_S": np.round(skew_P_se, 3),
    "Skew_iid_roll": np.round(rolling["RS_roll"].values, 3),
    "Kurt_P": np.round(kurt_P_est, 3), "se_K": np.round(kurt_P_se, 3),
    "Kurt_iid_roll": np.round(rolling["RK_roll"].values, 3),
}, index=tickers).to_string())
print("  (las columnas *_iid_roll son el estimador de ventanas rodantes bajo "
      "agregacion iid;\n   convergen mecanicamente a 0 y 3 al crecer H y por eso "
      "no se usan como estimador primario)")

# -----------------------------------------------------------------------------
# 1D.2  Regresion de calibracion Mincer-Zarnowitz -> primas de riesgo
# -----------------------------------------------------------------------------

MFIV_vec = np.array([bkm_moments[t]["MFIV"] for t in tickers], dtype=float)
for i, tk in enumerate(tickers):
    if not np.isfinite(MFIV_vec[i]) or MFIV_vec[i] <= 0:
        MFIV_vec[i] = float(Sigma.iloc[i, i])   # varianza implicita ATM como sustituto


def mincer_zarnowitz(y, x, se_y, nombre, dominio=None, piso=None):
    """WLS de y (momento fisico) sobre x (momento implicito), pesos 1/se_y^2.

    Regresion de calibracion en la seccion transversal de los n activos:

        Momento_fisico_i = a + b * Momento_implicito_i + u_i,   w_i = 1/se_i^2

    El pronostico fisico es el valor ajustado y la prima de riesgo del momento
    es el residuo estructural entre el implicito y ese pronostico:

        VRP_i = MFIV_i - (a_V + b_V * MFIV_i),   idem SRP y KRP

    No es circular, a diferencia de restar la media realizada -- que devolveria
    el propio momento realizado y destruiria la informacion prospectiva de la
    superficie: b conserva la dispersion transversal de las opciones y a y
    (1 - b) corrigen el sesgo.

    `piso` activa la version MULTIPLICATIVA del modelo. La varianza y la
    curtosis estan acotadas por abajo (var > 0, kurt >= 1), de modo que una
    prima ADITIVA puede empujar el pronostico fuera del dominio: restar una
    prima media de 2.5 a una curtosis implicita de 3.5 daria 1.0, inadmisible.
    Con `piso` la regresion se corre en logaritmos del exceso sobre esa cota,

        log(y - piso) = a + b * log(x - piso) + u

    y se retransforma con el estimador de smearing de Duan, E[exp(u)], que
    corrige el sesgo de Jensen al volver a niveles. El pronostico resultante
    respeta el dominio por construccion. La asimetria, que no esta acotada, se
    estima en niveles.

    Si la pendiente no es informativa se degrada al modelo restringido b = 1
    (prima constante, de nivel o de escala segun el caso); si el momento
    implicito no tiene dispersion, a la media ponderada.
    """
    y_orig = np.asarray(y, float); x_orig = np.asarray(x, float)
    se_orig = np.asarray(se_y, float)
    w = 1.0 / np.maximum(se_orig, 1e-12) ** 2
    w = w / w.mean()

    log_mode = piso is not None
    if log_mode:
        eps = 1e-6
        y = np.log(np.maximum(y_orig - piso, eps))
        x = np.log(np.maximum(x_orig - piso, eps))
        # delta method: se(log y) = se(y) / (y - piso)
        se_y = se_orig / np.maximum(y_orig - piso, eps)
    else:
        y, x, se_y = y_orig, x_orig, se_orig

    def a_nivel(fit_log, resid=None):
        """Retransformacion con smearing de Duan."""
        smear = float(np.mean(np.exp(resid))) if resid is not None else 1.0
        return piso + np.exp(fit_log) * smear

    if np.std(x) < 1e-12 or len(y) < 4:
        ajuste = np.full_like(y, float(np.average(y, weights=w)))
        if log_mode:
            ajuste = a_nivel(ajuste, y - np.average(y, weights=w))
        if dominio is not None:
            ajuste = np.clip(ajuste, dominio[0], dominio[1])
        return dict(fit=ajuste, se_fit=se_orig, a=np.nan, b=np.nan,
                    t_b=np.nan, r2=np.nan, modelo="media_ponderada", nombre=nombre)

    Xr = np.column_stack([np.ones_like(x), x])
    if HAY_STATSMODELS:
        mod = sm.WLS(y, Xr, weights=w).fit(cov_type="HC3")
        a, b = float(mod.params[0]), float(mod.params[1])
        se_b = float(mod.bse[1])
        r2 = float(mod.rsquared)
        se_media = np.sqrt(np.maximum(
            np.sum((Xr @ mod.cov_params()) * Xr, axis=1), 0.0))
    else:
        W = np.diag(w)
        XtWX_inv = np.linalg.pinv(Xr.T @ W @ Xr)
        beta = XtWX_inv @ (Xr.T @ W @ y)
        a, b = float(beta[0]), float(beta[1])
        resid = y - Xr @ beta
        s2 = float(resid @ (w * resid) / max(len(y) - 2, 1))
        V = s2 * XtWX_inv
        se_b = float(np.sqrt(max(V[1, 1], 0.0)))
        sst = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
        r2 = 1.0 - float(np.sum(w * resid ** 2)) / max(sst, 1e-18)
        se_media = np.sqrt(np.maximum(np.sum((Xr @ V) * Xr, axis=1), 0.0))

    t_b = b / se_b if se_b > 0 else np.nan

    # Criterio de degradacion: la pendiente debe ser informativa y de signo y
    # magnitud economicamente admisibles.
    usable = (np.isfinite(t_b) and abs(t_b) >= 1.0 and 0.02 <= b <= 3.0)
    if usable:
        ajuste_lin = a + b * x
        resid = y - ajuste_lin
        modelo = "mincer_zarnowitz" + ("_log" if log_mode else "")
    else:
        # Restriccion b = 1: prima constante (de nivel, o de escala en logs)
        prima_nivel = float(np.average(x - y, weights=w))
        ajuste_lin = x - prima_nivel
        resid = y - ajuste_lin
        se_media = np.full_like(x, float(np.sqrt(
            np.average(resid ** 2, weights=w) / len(x))))
        modelo = ("escala_constante(b=1)" if log_mode else "nivel_constante(b=1)")

    # Error estandar en la escala de estimacion, antes de retransformar
    se_pred_esc = np.sqrt(se_media ** 2 + np.asarray(se_y, float) ** 2)

    if log_mode:
        ajuste = a_nivel(ajuste_lin, resid)
        # delta method inverso: se(nivel) = (nivel - piso) * se(log)
        se_fit = np.maximum(ajuste - piso, 1e-12) * se_pred_esc
    else:
        ajuste = ajuste_lin
        se_fit = se_pred_esc

    # Guardarrail final de dominio (normalmente inactivo en modo logaritmico)
    if dominio is not None:
        ajuste = np.clip(ajuste, dominio[0], dominio[1])

    return dict(fit=ajuste, se_fit=se_fit, a=a, b=b, t_b=t_b, r2=r2,
                modelo=modelo, nombre=nombre)


# Varianza y curtosis: modelo multiplicativo (dominio acotado por abajo).
# Asimetria: modelo aditivo en niveles (no acotada, cambia de signo).
reg_V = mincer_zarnowitz(var_P_est, MFIV_vec, var_P_se, "Varianza",
                         dominio=(1e-8, np.inf), piso=0.0)
reg_S = mincer_zarnowitz(skew_P_est, MFIS_vec, skew_P_se, "Asimetria",
                         dominio=COTA_SKEW_P)
reg_K = mincer_zarnowitz(kurt_P_est, MFIK_vec, kurt_P_se, "Curtosis",
                         dominio=COTA_KURT_P, piso=1.0)

print("\n=== Regresiones de calibracion Mincer-Zarnowitz (seccion transversal, "
      f"n = {n} activos, WLS) ===")
print(pd.DataFrame([
    dict(Momento=r["nombre"], Modelo=r["modelo"],
         a=round(r["a"], 4) if np.isfinite(r["a"]) else np.nan,
         b=round(r["b"], 4) if np.isfinite(r["b"]) else np.nan,
         t_b=round(r["t_b"], 2) if np.isfinite(r["t_b"]) else np.nan,
         R2=round(r["r2"], 3) if np.isfinite(r["r2"]) else np.nan)
    for r in (reg_V, reg_S, reg_K)
]).to_string(index=False))
print("  (b < 1 => el momento implicito sobre-reacciona respecto del fisico, "
      "que es el patron documentado)")

# --- Pronosticos fisicos y primas de riesgo ---------------------------------
var_P = reg_V["fit"].copy()
skew_P = reg_S["fit"].copy()
kurt_P = reg_K["fit"].copy()

se_var_P, se_skew_P, se_kurt_P = reg_V["se_fit"], reg_S["se_fit"], reg_K["se_fit"]

VRP = MFIV_vec - var_P          # prima de riesgo de varianza
SRP = MFIS_vec - skew_P         # prima de riesgo de asimetria
KRP = MFIK_vec - kurt_P         # prima de riesgo de curtosis

t_VRP = VRP / np.maximum(se_var_P, 1e-12)
t_SRP = SRP / np.maximum(se_skew_P, 1e-12)
t_KRP = KRP / np.maximum(se_kurt_P, 1e-12)

print("\n=== Primas de riesgo de momentos (Q - P) al horizonte de "
      f"{MESES_HORIZONTE} meses ===")
print(pd.DataFrame({
    "MFIV_Q": np.round(MFIV_vec, 5), "Var_P_fit": np.round(var_P, 5),
    "VRP": np.round(VRP, 5), "t_VRP": np.round(t_VRP, 2),
    "MFIS_Q": np.round(MFIS_vec, 3), "Skew_P_fit": np.round(skew_P, 3),
    "SRP": np.round(SRP, 3), "t_SRP": np.round(t_SRP, 2),
    "MFIK_Q": np.round(MFIK_vec, 3), "Kurt_P_fit": np.round(kurt_P, 3),
    "KRP": np.round(KRP, 3), "t_KRP": np.round(t_KRP, 2),
}, index=tickers).to_string())

print(f"\n  VRP medio: {np.nanmean(VRP):+.5f}  "
      "(> 0 = la varianza implicita excede a la fisica, el signo esperado)")
print(f"  SRP medio: {np.nanmean(SRP):+.4f}  "
      "(< 0 = la asimetria implicita es mas negativa que la fisica)")
print(f"  KRP medio: {np.nanmean(KRP):+.4f}  "
      "(> 0 = la curtosis implicita excede a la fisica)")

# -----------------------------------------------------------------------------
# 1D.3  Cotas de sensatez y cumulantes fisicos
# -----------------------------------------------------------------------------

ratio_vol = np.sqrt(np.maximum(var_P, 1e-12) / np.maximum(MFIV_vec, 1e-12))
n_clip_vol = int(np.sum((ratio_vol < COTA_RATIO_VOL_P[0]) | (ratio_vol > COTA_RATIO_VOL_P[1])))
ratio_vol = np.clip(ratio_vol, *COTA_RATIO_VOL_P)
var_P = ratio_vol ** 2 * MFIV_vec

n_clip_skew = int(np.sum((skew_P < COTA_SKEW_P[0]) | (skew_P > COTA_SKEW_P[1])))
skew_P = np.clip(skew_P, *COTA_SKEW_P)
n_clip_kurt = int(np.sum((kurt_P < COTA_KURT_P[0]) | (kurt_P > COTA_KURT_P[1])))
kurt_P = np.clip(kurt_P, *COTA_KURT_P)
# Cota de factibilidad de Pearson: toda distribucion cumple kurt >= skew^2 + 1
kurt_P = np.maximum(kurt_P, skew_P ** 2 + 1.05)

if n_clip_vol or n_clip_skew or n_clip_kurt:
    print(f"\n  Cotas de sensatez activadas -> vol: {n_clip_vol} | "
          f"skew: {n_clip_skew} | kurt: {n_clip_kurt} activos")

sigma_P_vec = np.sqrt(var_P)

# Cumulantes centrados del log-retorno al horizonte
k2_P = var_P
k3_P = skew_P * sigma_P_vec ** 3
k4_P = (kurt_P - 3.0) * sigma_P_vec ** 4

k2_Q_obs = MFIV_vec
k3_Q_obs = MFIS_vec * MFIV_vec ** 1.5
k4_Q_obs = (MFIK_vec - 3.0) * MFIV_vec ** 2

# Errores estandar de los cumulantes (delta method de primer orden)
se_k2_vec = np.maximum(se_var_P, 1e-10)
se_k3_vec = np.maximum(se_skew_P * sigma_P_vec ** 3, 1e-12)
se_k4_vec = np.maximum(se_kurt_P * sigma_P_vec ** 4, 1e-14)


# -----------------------------------------------------------------------------
# 1D.4  Covarianza bajo la medida fisica P
# -----------------------------------------------------------------------------

D_P = np.diag(sigma_P_vec)
Sigma_P = pd.DataFrame(D_P @ Corr_hist @ D_P, index=tickers, columns=tickers)

print(f"\n  Sigma_P construida. Vol media Q: {np.mean(np.sqrt(MFIV_vec)):.4f} | "
      f"vol media P: {np.mean(sigma_P_vec):.4f} | "
      f"reduccion: {(1 - np.mean(sigma_P_vec) / np.mean(np.sqrt(MFIV_vec))) * 100:.1f}%")


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
# BLOQUE 3: PERFIL DE RIESGO -> TAU, OMEGA_SCALE, GAMMA_RA
# =============================================================================

perfil = PERFILES[PERFIL_RIESGO]
tau = perfil["tau"]
gamma_ra = perfil["gamma_ra"]

desc_perfiles = dict(
    conservador="Portafolio cercano al benchmark. Views con poco peso.",
    moderado="Balance entre consenso de mercado y vision activa.",
    agresivo="Portafolio con fuerte inclinacion hacia los views del gestor.",
)

# --- Delta de mercado (fijo, no depende del perfil) -------------------------
# delta_mkt es la aversion al riesgo IMPLICITA del mercado (CAPM invertido):
# cuanto retorno en exceso exige el mercado por unidad de varianza. Se calcula
# con datos historicos, no con el perfil del gestor, para que pi_eq (y por lo
# tanto el Sharpe de referencia) no cambie solo por elegir otro perfil.
Rf_h = Rf * (MESES_HORIZONTE / 12)
ret_mkt_hist = float(w_mkt @ mu_historico)
var_mkt_hist = float(w_mkt @ Sigma_hist @ w_mkt)
delta_mkt = (ret_mkt_hist - Rf_h) / var_mkt_hist

print(f"\n=== PERFIL: {PERFIL_RIESGO.upper()} ===")
print(f"Descripcion: {desc_perfiles[PERFIL_RIESGO]}")
print(f"Delta de mercado (fijo): {delta_mkt:.4f} | Tau (t): {tau} | "
      f"Omega scale: {perfil['omega_scale']} | Gamma_RA: {gamma_ra}")

# =============================================================================
# BLOQUE 4: RETORNOS DE EQUILIBRIO pi - CAPM INVERTIDO
# =============================================================================

# Se usa Sigma_P (medida fisica), no Sigma (risk-neutral, sobreestima el riesgo).
Sigma_mat = Sigma_P.values
pi_eq = delta_mkt * (Sigma_mat @ w_mkt)

print("\n=== Retornos de equilibrio pi (prior) ===")
print(pd.Series(np.round(pi_eq, 4), index=tickers))


# =============================================================================
# BLOQUE 4B: PROYECCION Q -> P POR TRANSFORMADA DE ESSCHER
# -----------------------------------------------------------------------------
# Cierra el paso Q -> P: estima el parametro de Esscher theta por GMM anclado a
# la aversion al riesgo implicita del mercado, y descompone la prima de riesgo
# del activo en su parte gaussiana (ya recogida por pi_eq) y la no gaussiana
# (prima_HM), que es la que ajustara Q y Omega en el Bloque 6B.
# =============================================================================

print("\n" + "=" * 79)
print("BLOQUE 4B: PROYECCION Q -> P (TRANSFORMADA DE ESSCHER)")
print("=" * 79)

# Si delta_mkt sale negativo (mercado por debajo de Rf en la ventana) no informa
# sobre la aversion al riesgo: el ancla pasa a 0, la hipotesis nula Q = P.
theta_ancla = float(np.clip(delta_mkt, THETA_ESSCHER_COTA[0], THETA_ESSCHER_COTA[1]))
print(f"\n  delta_mkt (aversion al riesgo implicita del mercado): {delta_mkt:.3f}")
if not (THETA_ESSCHER_COTA[0] <= delta_mkt <= THETA_ESSCHER_COTA[1]):
    print(f"  Aviso: delta_mkt fuera del rango admisible {THETA_ESSCHER_COTA}; "
          f"el ancla se fija en {theta_ancla:.3f}"
          + (" (hipotesis nula Q = P)" if theta_ancla == 0.0 else ""))
print(f"  Ancla de theta: {theta_ancla:.3f} | rango admisible: "
      f"{THETA_ESSCHER_COTA} | difusion del ancla: {THETA_PRIOR_CV:.0%}")
# -----------------------------------------------------------------------------
# 4B.1  Transformada de Esscher: estimacion de theta por GMM anclado
# -----------------------------------------------------------------------------

def cumulantes_Q_desde_P(theta, k2p, k3p, k4p):
    """k_n^Q = sum_m k_{n+m}^P (-theta)^m / m!, truncado en el 4.o cumulante."""
    return (k2p - theta * k3p + 0.5 * theta ** 2 * k4p,
            k3p - theta * k4p)


def estimar_theta_esscher(k2p, k3p, k4p, k2q, k3q, se_k2, se_k3,
                          ancla=None, cv_ancla=THETA_PRIOR_CV):
    """GMM ponderado y anclado de un parametro sobre dos condiciones de momento.

        min_theta  w2*(k2_Q(th) - k2q)^2 + w3*(k3_Q(th) - k3q)^2
                   + wa*(th - ancla)^2

    w2, w3 = inversa de la varianza estimada de cada prima (GMM eficiente de
    dos etapas simplificado). wa = 1/(cv_ancla*ancla)^2 reescalado a la misma
    unidad que las otras condiciones; su papel es identificar theta cuando
    k4_P ~ 0 vuelve plana la condicion del 3.er cumulante. Devuelve
    (theta, en_cota, J) donde J es el residuo GMM normalizado de las dos
    condiciones de momento (diagnostico de sobre-identificacion).
    """
    ancla = theta_ancla if ancla is None else ancla
    w2 = 1.0 / max(se_k2 ** 2, 1e-20)
    w3 = 1.0 / max(se_k3 ** 2, 1e-20)
    esc = w2 + w3
    w2, w3 = w2 / esc, w3 / esc

    # Escala del ancla, en las mismas unidades que el objetivo GMM (k2q^2
    # normaliza). Con ancla ~ 0 el prior debe ser DIFUSO, no estrecho: de ahi
    # el piso de medio rango admisible.
    rango = THETA_ESSCHER_COTA[1] - THETA_ESSCHER_COTA[0]
    sd_ancla = cv_ancla * max(abs(ancla), 0.5 * rango)
    wa = (k2q ** 2) / sd_ancla ** 2

    def objetivo(th):
        th = float(np.atleast_1d(th)[0])
        c2, c3 = cumulantes_Q_desde_P(th, k2p, k3p, k4p)
        return (w2 * (c2 - k2q) ** 2 + w3 * (c3 - k3q) ** 2
                + wa * (th - ancla) ** 2)

    rejilla = np.linspace(THETA_ESSCHER_COTA[0], THETA_ESSCHER_COTA[1], 251)
    th0 = float(rejilla[int(np.argmin([objetivo(t) for t in rejilla]))])
    res = minimize(objetivo, np.array([th0]), method="L-BFGS-B",
                   bounds=[THETA_ESSCHER_COTA])
    th = float(res.x[0]) if res.success else th0

    c2, c3 = cumulantes_Q_desde_P(th, k2p, k3p, k4p)
    J = float(w2 * (c2 - k2q) ** 2 + w3 * (c3 - k3q) ** 2)
    en_cota = bool(abs(th - THETA_ESSCHER_COTA[0]) < 1e-6
                   or abs(th - THETA_ESSCHER_COTA[1]) < 1e-6)
    return th, en_cota, J


def primas_esscher(theta, k2p, k3p, k4p):
    """Descomposicion de la prima total k1_P - k1_Q.

        total    = theta*k2 - theta^2/2*k3 + theta^3/6*k4
        gaussian = theta*k2                        <- ya recogida por pi_eq
        HM       = -theta^2/2*k3 + theta^3/6*k4    <- componente no gaussiana
    """
    gauss = theta * k2p
    hm = -0.5 * theta ** 2 * k3p + (theta ** 3 / 6.0) * k4p
    return gauss + hm, gauss, hm


theta_esscher = np.zeros(n); en_cota = np.zeros(n, dtype=bool); J_gmm = np.zeros(n)
prima_total = np.zeros(n); prima_gauss = np.zeros(n); prima_hm = np.zeros(n)

for i in range(n):
    theta_esscher[i], en_cota[i], J_gmm[i] = estimar_theta_esscher(
        k2_P[i], k3_P[i], k4_P[i], k2_Q_obs[i], k3_Q_obs[i],
        se_k2_vec[i], se_k3_vec[i])
    prima_total[i], prima_gauss[i], prima_hm[i] = primas_esscher(
        theta_esscher[i], k2_P[i], k3_P[i], k4_P[i])

# Cota economica sobre la prima no gaussiana: no puede exceder una fraccion
# razonable de la volatilidad fisica del activo. Es un guardarrail (se informa
# cuando actua), no un parametro de calibracion.
tope_hm = MAX_PRIMA_HM_SIGMA * sigma_P_vec
n_topados = int(np.sum(np.abs(prima_hm) > tope_hm))
prima_hm = np.clip(prima_hm, -tope_hm, tope_hm)

# -----------------------------------------------------------------------------
# 4B.2  Incertidumbre de la prima no gaussiana (alimenta Omega)
# -----------------------------------------------------------------------------
# prima_HM es funcion no lineal de los momentos estimados: su error estandar se
# propaga por delta-method de Monte Carlo.

se_prima_hm = np.zeros(n)
for i in range(n):
    v_d = var_P[i] + se_var_P[i] * rng_global.standard_normal(N_MC_DELTA)
    s_d = skew_P[i] + se_skew_P[i] * rng_global.standard_normal(N_MC_DELTA)
    k_d = kurt_P[i] + se_kurt_P[i] * rng_global.standard_normal(N_MC_DELTA)
    draws = np.empty(N_MC_DELTA)
    for m in range(N_MC_DELTA):
        v_p = float(np.clip(v_d[m], (COTA_RATIO_VOL_P[0] ** 2) * MFIV_vec[i],
                            (COTA_RATIO_VOL_P[1] ** 2) * MFIV_vec[i]))
        s_p = float(np.clip(s_d[m], *COTA_SKEW_P))
        k_p = max(float(np.clip(k_d[m], *COTA_KURT_P)), s_p ** 2 + 1.05)
        sd = math.sqrt(v_p)
        c2, c3, c4 = v_p, s_p * sd ** 3, (k_p - 3.0) * sd ** 4
        th, _, _ = estimar_theta_esscher(c2, c3, c4, MFIV_vec[i], k3_Q_obs[i],
                                         se_k2_vec[i], se_k3_vec[i])
        draws[m] = np.clip(primas_esscher(th, c2, c3, c4)[2],
                           -tope_hm[i], tope_hm[i])
    se_prima_hm[i] = float(np.std(draws, ddof=1))

print("\n=== Proyeccion Q -> P y descomposicion de la prima de riesgo ===")
print("(prima_gauss ya esta recogida en pi_eq via el CAPM invertido; "
      "prima_HM es el termino no gaussiano que ajusta Q)")
print(pd.DataFrame({
    "sigma_Q": np.round(np.sqrt(MFIV_vec), 4),
    "sigma_P": np.round(sigma_P_vec, 4),
    "skew_Q": np.round(MFIS_vec, 3), "skew_P": np.round(skew_P, 3),
    "kurt_Q": np.round(MFIK_vec, 3), "kurt_P": np.round(kurt_P, 3),
    "theta": np.round(theta_esscher, 3),
    "prima_gauss": np.round(prima_gauss, 4),
    "prima_HM": np.round(prima_hm, 4),
    "se_HM": np.round(se_prima_hm, 4),
    "t_HM": np.round(prima_hm / np.maximum(se_prima_hm, 1e-10), 2),
    "J_gmm": np.round(J_gmm, 4),
}, index=tickers).to_string())
print("  (J_gmm = residuo de sobre-identificacion: valores altos indican que el "
      "truncamiento\n   en el 4.o cumulante no logra reconciliar Q y P para ese "
      "activo)")

n_cota_sup = int(np.sum(theta_esscher >= THETA_ESSCHER_COTA[1] - 1e-6))
if n_cota_sup:
    print(f"  Aviso: theta alcanzo su cota SUPERIOR en {n_cota_sup} activos "
          f"(rango admitido {THETA_ESSCHER_COTA}); bajo truncamiento en el 4.o "
          "cumulante la reconciliacion Q-P de esos activos es solo parcial.")
if float(np.mean(theta_esscher)) < 1e-3:
    print("  Nota: theta ~ 0 en todo el universo => no se detecta prima de "
          "momentos superiores y el modelo se reduce, correctamente, al "
          "Black-Litterman gaussiano estandar.")
if n_topados:
    print(f"  Aviso: prima_HM topada en {n_topados} activos por la cota economica "
          f"de {MAX_PRIMA_HM_SIGMA:.0%} de sigma_P.")


# =============================================================================
# BLOQUE 5: TABLA DE REFERENCIA PARA VIEWS
# =============================================================================

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
P.loc["View_1", "DELL"] = 1
P.loc["View_1", "META"] = -1

# VIEW 2 - RELATIVO: JPM y MA outperforman a JNJ y MRK
P.loc["View_2", "GS"] = 1
P.loc["View_2", "REGN"] = -1

# VIEW 3 - ABSOLUTO: MSFT retorna al menos X% en el horizonte
P.loc["View_3", "EBAY"] = 1
P.loc["View_3", "ARES"] = -1

print("\n=== Matriz P (views del gestor) ===")
print(P.round(4))

# --- PASO 3: Define el vector Q ---------------------------------------------
Q = pd.Series({
    "View_1": 0.15,
    "View_2": 0.10,
    "View_3": 0.08,
})

print("\n=== Vector Q (magnitudes del gestor) ===")
print(Q.round(4))

# =============================================================================
# BLOQUE 6B: PUENTE Q -> P SOBRE (Q, Omega) + NUCLEO GAUSSIANO BL
# -----------------------------------------------------------------------------
# Sustituye el parche de KAPPA_SKEW / KAPPA_KURT. Q se corrige con la prima no
# gaussiana del Bloque 4B y Omega suma dos fuentes ortogonales: el riesgo de
# mercado de las vistas y la incertidumbre de estimacion de esa prima. Con
# ambos se calcula el posterior gaussiano (mu_BL, Sigma_BL).
# =============================================================================

P_mat = P.values
Q_vec = Q.values.reshape(-1, 1)
pi_eq_col = pi_eq.reshape(-1, 1)
tauSigma = tau * Sigma_mat

# --- (i) Q bajo la medida fisica --------------------------------------------
ajuste_Q_hm = P_mat @ prima_hm
Q_P = Q.values + ajuste_Q_hm

print("\n" + "=" * 79)
print("BLOQUE 6B: PUENTE ECONOMETRICO Q -> P")
print("=" * 79)
print("\n=== Ajuste vectorial de Q por prima de riesgo no gaussiana ===")
print(pd.DataFrame({
    "Q_riesgo_neutral": np.round(Q.values, 4),
    "P.prima_HM": np.round(ajuste_Q_hm, 4),
    "Q_fisico": np.round(Q_P, 4),
}, index=Q.index).to_string())

# --- (ii) Omega bajo la medida fisica ---------------------------------------
Omega_mercado = np.diag(np.diag(tau * P_mat @ Sigma_mat @ P_mat.T)) * perfil["omega_scale"]
# Varianza de estimacion de la prima no gaussiana proyectada al espacio de views
Var_prima = np.diag(se_prima_hm ** 2)
Omega_estimacion = np.diag(np.diag(P_mat @ Var_prima @ P_mat.T))
Omega_P = Omega_mercado + Omega_estimacion

print("\n=== Descomposicion de Omega (varianzas, no desviaciones) ===")
print(pd.DataFrame({
    "Riesgo_mercado": np.diag(Omega_mercado),
    "Riesgo_estimacion": np.diag(Omega_estimacion),
    "Omega_total": np.diag(Omega_P),
    "%_estimacion": np.round(100 * np.diag(Omega_estimacion) /
                             np.maximum(np.diag(Omega_P), 1e-18), 1),
}, index=Q.index).to_string())

# --- Nucleo gaussiano de Black-Litterman ------------------------------------
# mu_BL    = pi + tau*Sigma*P' (P tau Sigma P' + Omega)^-1 (Q_P - P pi)
# Sigma_BL = Sigma + [tau*Sigma - tau*Sigma P' M^-1 P tau*Sigma]
# Solo aporta los dos primeros momentos; el Bloque 7 anade el 3.o y el 4.o.
Q_P_col = Q_P.reshape(-1, 1)
sorpresa = Q_P_col - P_mat @ pi_eq_col
M_bl = P_mat @ tauSigma @ P_mat.T + Omega_P
M_bl_inv = np.linalg.inv(M_bl)
mu_BL = (pi_eq_col + tauSigma @ P_mat.T @ M_bl_inv @ sorpresa).flatten()
Sigma_BL = Sigma_mat + tauSigma - tauSigma @ P_mat.T @ M_bl_inv @ P_mat @ tauSigma
Sigma_BL = pd.DataFrame(Sigma_BL, index=tickers, columns=tickers)

comparacion = pd.DataFrame({
    "Ticker": tickers,
    "Pi_eq": np.round(pi_eq, 4),
    "Mu_hist": np.round(mu_historico, 4),
    "Prima_HM": np.round(prima_hm, 4),
    "Mu_BL": np.round(mu_BL, 4),
})
comparacion["Ajuste_BL"] = np.round(comparacion["Mu_BL"] - comparacion["Pi_eq"], 4)

print("\n=== Posterior gaussiano: pi vs historico vs mu_BL ===")
print(comparacion.to_string(index=False))
print(f"\nNorma Frobenius |Sigma_BL - Sigma_P|: "
      f"{np.linalg.norm(Sigma_BL.values - Sigma_mat):.6f}")

# =============================================================================
# BLOQUE 7: INTEGRACION BAYESIANA NO GAUSSIANA
# -----------------------------------------------------------------------------
# El posterior del Bloque 6B es gaussiano: solo tiene media y covarianza. Aqui
# se sustituye por una distribucion DISCRETA sobre un panel de escenarios cuyos
# momentos 1 a 4 coinciden con los del modelo, por Entropy Pooling de Meucci
# (opcion A) o por inclinacion de Gram-Charlier / Edgeworth (opcion B, que
# ademas actua de fallback).
# =============================================================================

from scipy.special import logsumexp

print("\n" + "=" * 79)
print("BLOQUE 7: POSTERIOR NO GAUSSIANO")
print("=" * 79)

# -----------------------------------------------------------------------------
# 7.1  Panel de escenarios prior (bootstrap estacionario, funcion del Bloque 1D)
# -----------------------------------------------------------------------------

print(f"\nGenerando panel de {N_ESCENARIOS} escenarios "
      f"(bootstrap estacionario, bloque medio {BOOTSTRAP_BLOQUE} dias, "
      f"half-life {HALF_LIFE_PRIOR} dias)...")

X_raw = bootstrap_estacionario(R_dia, N_ESCENARIOS, horizonte_dias,
                               BOOTSTRAP_BLOQUE, peso_tiempo, rng_global)

# Reescalado afin por columna hacia (pi_eq, sigma_P): no altera la correlacion
# del panel ni la asimetria/curtosis que aporta el bootstrap.
mu_raw = X_raw.mean(axis=0)
sd_raw = X_raw.std(axis=0, ddof=1)
sd_raw = np.where(sd_raw > 0, sd_raw, 1.0)
X_prior = (X_raw - mu_raw) / sd_raw * sigma_P_vec + pi_eq

f_prior = np.full(N_ESCENARIOS, 1.0 / N_ESCENARIOS)

print(f"  Panel listo: {X_prior.shape[0]} x {X_prior.shape[1]}")
print(f"  Asimetria media del prior:  {pd.DataFrame(X_prior).skew().mean():+.3f}")
print(f"  Curtosis media del prior:   {pd.DataFrame(X_prior).kurt().mean() + 3:.3f}")


# -----------------------------------------------------------------------------
# 7.2  Utilidades de momentos ponderados
# -----------------------------------------------------------------------------

def momentos_ponderados(X, p):
    """Media, covarianza, asimetria y curtosis estandarizadas por activo."""
    mu = p @ X
    Xc = X - mu
    Sig = (Xc * p[:, None]).T @ Xc
    sd = np.sqrt(np.maximum(np.diag(Sig), 1e-18))
    sk = (p @ (Xc ** 3)) / sd ** 3
    ku = (p @ (Xc ** 4)) / sd ** 4
    return mu, Sig, sk, ku


def momentos_portafolio(w, X, p):
    """Momentos centrales del retorno del portafolio bajo (X, p).

    Equivale a w'mu, w'Sigma w, w'M3(w x w) y w'M4(w x w x w) pero se calcula
    en O(J*n) en vez de construir tensores de co-momentos de tamano n^3 y n^4.
    """
    r = X @ w
    mu = float(p @ r)
    d = r - mu
    m2 = float(p @ d ** 2)
    m3 = float(p @ d ** 3)
    m4 = float(p @ d ** 4)
    return mu, m2, m3, m4


# -----------------------------------------------------------------------------
# 7.3  OPCION A: Entropy Pooling
# -----------------------------------------------------------------------------

def entropy_pooling(f, A, b, tol_residuo=1e-6, maxiter=800):
    """Posterior de minima entropia relativa sujeto a E_p[A] = b.

    Parametros
    ----------
    f : (J,)   probabilidades prior
    A : (K, J) matriz de features (una fila por restriccion de momento)
    b : (K,)   valores objetivo

    Devuelve (p, info). El problema dual es
        min_lambda  log sum_j f_j exp(-lambda' V_j),  V = (A - b) / escala
    convexo y sin restricciones; su gradiente es -E_p[V], de modo que el optimo
    satisface exactamente las restricciones cuando son factibles.
    """
    K, J = A.shape
    escala = A.std(axis=1, ddof=1)
    escala = np.where(escala > 1e-14, escala, 1.0)
    V = (A - b[:, None]) / escala[:, None]
    log_f = np.log(np.maximum(f, 1e-300))

    def dual(lam):
        z = log_f - lam @ V
        lse = logsumexp(z)
        p = np.exp(z - lse)
        return float(lse), -(V @ p)

    res = minimize(dual, np.zeros(K), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=maxiter, ftol=1e-14, gtol=1e-10))

    lam = res.x
    z = log_f - lam @ V
    p = np.exp(z - logsumexp(z))
    p = np.maximum(p, 0.0)
    p = p / p.sum()

    residuo = np.abs(V @ p)
    kl = float(np.sum(p * (np.log(np.maximum(p, 1e-300)) - log_f)))
    ens = float(np.exp(-np.sum(p * np.log(np.maximum(p, 1e-300)))) / J)

    info = dict(exito=bool(res.success), kl=kl, ens=ens,
                residuo_max=float(residuo.max()),
                factible=bool(residuo.max() < max(tol_residuo, 1e-4)),
                lam=lam, mensaje=str(res.message))
    return p, info


def construir_restricciones(X, mu_obj, var_obj, skew_obj, kurt_obj, nivel):
    """Bloques de restricciones por nivel de momento.

    nivel = 2 -> media y varianza
    nivel = 3 -> + asimetria
    nivel = 4 -> + curtosis
    """
    filas, objetivos, etiquetas = [], [], []
    sd_obj = np.sqrt(var_obj)

    for i in range(X.shape[1]):
        filas.append(X[:, i]); objetivos.append(mu_obj[i]); etiquetas.append(f"mean_{i}")
    for i in range(X.shape[1]):
        filas.append((X[:, i] - mu_obj[i]) ** 2); objetivos.append(var_obj[i])
        etiquetas.append(f"var_{i}")
    if nivel >= 3:
        for i in range(X.shape[1]):
            filas.append((X[:, i] - mu_obj[i]) ** 3)
            objetivos.append(skew_obj[i] * sd_obj[i] ** 3)
            etiquetas.append(f"skew_{i}")
    if nivel >= 4:
        for i in range(X.shape[1]):
            filas.append((X[:, i] - mu_obj[i]) ** 4)
            objetivos.append(kurt_obj[i] * sd_obj[i] ** 4)
            etiquetas.append(f"kurt_{i}")

    return np.array(filas), np.array(objetivos), etiquetas


# --- Objetivos de momento extraidos del modelo -------------------------------
# Momentos 1 y 2: posterior gaussiano de Black-Litterman (Bloque 6B).
# Momentos 3 y 4: proyeccion a P de los momentos BKM (Bloque 1D).
mu_obj = mu_BL.copy()
var_obj = np.diag(Sigma_BL.values).copy()
skew_obj = skew_P.copy()
kurt_obj = kurt_P.copy()


# -----------------------------------------------------------------------------
# 7.4  OPCION B: inclinacion de Gram-Charlier / Edgeworth
# -----------------------------------------------------------------------------

def he3(z):
    return z ** 3 - 3 * z


def he4(z):
    return z ** 4 - 6 * z ** 2 + 3


def gram_charlier_pdf_ratio(z, s, k):
    """g(z)/phi(z) = 1 + (S/6) He_3(z) + ((K-3)/24) He_4(z).

    Serie tipo A truncada en el 4.o momento. Puede volverse negativa fuera de
    la region de validez de Barton-Dennis, por lo que se trunca por abajo.
    """
    return 1.0 + (s / 6.0) * he3(z) + ((k - 3.0) / 24.0) * he4(z)


def posterior_gram_charlier(X, f, mu_obj, var_obj, skew_obj, kurt_obj):
    """Reponderacion del panel por el factor de Gram-Charlier activo a activo.

    Es el analogo por muestreo de importancia de expandir la densidad posterior
    del vector de retornos en serie de Edgeworth alrededor de la normal
    N(mu_BL, Sigma_BL) sumando las contribuciones del 3.er y 4.o momento BKM.
    """
    Z = (X - mu_obj) / np.sqrt(var_obj)
    log_w = np.zeros(X.shape[0])
    for i in range(X.shape[1]):
        ratio = gram_charlier_pdf_ratio(Z[:, i], skew_obj[i], kurt_obj[i])
        log_w += np.log(np.maximum(ratio, 1e-6))
    log_p = np.log(np.maximum(f, 1e-300)) + log_w
    p = np.exp(log_p - logsumexp(log_p))
    p = p / p.sum()
    ens = float(np.exp(-np.sum(p * np.log(np.maximum(p, 1e-300)))) / len(p))
    kl = float(np.sum(p * (np.log(np.maximum(p, 1e-300)) - np.log(np.maximum(f, 1e-300)))))
    negativos = int(np.sum(gram_charlier_pdf_ratio(Z, skew_obj, kurt_obj) < 0))
    return p, dict(exito=True, kl=kl, ens=ens, residuo_max=np.nan,
                   factible=True, densidades_negativas=negativos,
                   mensaje="gram_charlier")


# -----------------------------------------------------------------------------
# 7.5  Resolucion con degradacion controlada
# -----------------------------------------------------------------------------

metodo_posterior_usado = None
info_post = None

if METODO_POSTERIOR == "entropy_pooling":
    niveles = [4, 3, 2] if EP_IMPONER_CURTOSIS else [3, 2]
    nombres_nivel = {4: "media+var+skew+kurt", 3: "media+var+skew", 2: "media+var"}

    for niv in niveles:
        A_ep, b_ep, _ = construir_restricciones(
            X_prior, mu_obj, var_obj, skew_obj, kurt_obj, niv)
        print(f"\n  Entropy Pooling nivel {niv} ({nombres_nivel[niv]}): "
              f"{A_ep.shape[0]} restricciones sobre {N_ESCENARIOS} escenarios...")
        p_post, info = entropy_pooling(f_prior, A_ep, b_ep)
        print(f"    exito={info['exito']} | residuo_max={info['residuo_max']:.2e} | "
              f"KL={info['kl']:.4f} | ENS={info['ens'] * 100:.1f}% de J")
        if info["factible"] and info["ens"] >= EP_TOL_ENS:
            metodo_posterior_usado = f"entropy_pooling_n{niv}"
            info_post = info
            break
        print("    -> rechazado (infactible o ENS demasiado bajo); se relaja un nivel")

    if metodo_posterior_usado is None:
        print("\n  Entropy Pooling no alcanzo una solucion aceptable. "
              "Se recurre a la expansion de Gram-Charlier (Opcion B).")
        p_post, info_post = posterior_gram_charlier(
            X_prior, f_prior, mu_obj, var_obj, skew_obj, kurt_obj)
        metodo_posterior_usado = "gram_charlier_fallback"
else:
    p_post, info_post = posterior_gram_charlier(
        X_prior, f_prior, mu_obj, var_obj, skew_obj, kurt_obj)
    metodo_posterior_usado = "gram_charlier"
    print(f"\n  Gram-Charlier: KL={info_post['kl']:.4f} | "
          f"ENS={info_post['ens'] * 100:.1f}% de J | "
          f"celdas con densidad negativa truncadas: "
          f"{info_post.get('densidades_negativas', 0)}")

# -----------------------------------------------------------------------------
# 7.6  Momentos del posterior no gaussiano
# -----------------------------------------------------------------------------

mu_post, Sigma_post_arr, skew_post, kurt_post = momentos_ponderados(X_prior, p_post)
Sigma_post = pd.DataFrame(Sigma_post_arr, index=tickers, columns=tickers)

print(f"\n=== Posterior obtenido por: {metodo_posterior_usado} ===")
print(f"  Entropia relativa D(p||f): {info_post['kl']:.4f} nats")
print(f"  Numero efectivo de escenarios: {info_post['ens'] * 100:.1f}% de {N_ESCENARIOS}")

tabla_post = pd.DataFrame({
    "mu_BL_obj": np.round(mu_obj, 4),
    "mu_post": np.round(mu_post, 4),
    "sd_obj": np.round(np.sqrt(var_obj), 4),
    "sd_post": np.round(np.sqrt(np.diag(Sigma_post_arr)), 4),
    "skew_obj": np.round(skew_obj, 3),
    "skew_post": np.round(skew_post, 3),
    "kurt_obj": np.round(kurt_obj, 3),
    "kurt_post": np.round(kurt_post, 3),
}, index=tickers)

print("\n=== Ajuste del posterior a los momentos objetivo ===")
print(tabla_post.to_string())

# =============================================================================
# BLOQUE 7B: MODULO DE RIESGO DE COLA (VaR, CVaR / EXPECTED SHORTFALL)
# -----------------------------------------------------------------------------
# VaR historico, gaussiano y ajustado por Cornish-Fisher; CVaR / Expected
# Shortfall en version historica exacta y Cornish-Fisher; y ratios ajustados
# por cola (Sortino y Omega). Todo se evalua sobre distribuciones ponderadas,
# para poder aplicarlo tal cual al posterior del Bloque 7.
# =============================================================================

def _pesos_normalizados(m, probabilidades):
    if probabilidades is None:
        return np.full(m, 1.0 / m)
    p = np.asarray(probabilidades, dtype=float)
    return p / p.sum()


def var_cvar_historico(perdidas, p, alpha):
    """VaR y CVaR exactos de una distribucion discreta ponderada de perdidas.

    ES_alpha = 1/(1-alpha) * [ sum_{L>q} p_i L_i + (1-alpha - sum_{L>q} p_i) q ]
    (formula exacta que reparte correctamente la masa del propio cuantil).
    """
    orden = np.argsort(perdidas)
    L = perdidas[orden]
    w = p[orden]
    acum = np.cumsum(w)
    idx = int(np.searchsorted(acum, alpha, side="left"))
    idx = min(idx, len(L) - 1)
    var = float(L[idx])

    cola = L > var
    masa_cola = float(w[cola].sum())
    suma_cola = float(np.sum(w[cola] * L[cola]))
    resto = max(1.0 - alpha - masa_cola, 0.0)
    cvar = (suma_cola + resto * var) / (1.0 - alpha)
    return var, float(cvar)


def cornish_fisher_z(alpha_cola, s, k):
    """Cuantil de Cornish-Fisher para la probabilidad de cola alpha_cola.

    Inversa de la expansion de Gram-Charlier / Edgeworth: corrige el cuantil
    normal z con la asimetria S y la curtosis K del portafolio,

        z_CF = z + (z^2-1)S/6 + (z^3-3z)(K-3)/24 - (2z^3-5z)S^2/36

    de modo que VaR = -(mu + sigma*z_CF) recoge la forma real de la cola y no
    la normal implicita del VaR parametrico clasico.
    """
    z = norm.ppf(alpha_cola)
    ex = k - 3.0
    return (z
            + (z ** 2 - 1.0) * s / 6.0
            + (z ** 3 - 3.0 * z) * ex / 24.0
            - (2.0 * z ** 3 - 5.0 * z) * s ** 2 / 36.0)


def var_cvar_cornish_fisher(mu, sigma, s, k, alpha, n_grid=2000):
    """VaR y CVaR bajo la distribucion implicita por Cornish-Fisher.

    El CVaR se obtiene integrando el cuantil CF sobre la cola izquierda:
        ES = -( mu + sigma * (1/(1-alpha)) * integral_0^{1-alpha} z_CF(u) du )
    """
    z_cf = cornish_fisher_z(1.0 - alpha, s, k)
    var = -(mu + sigma * z_cf)

    u = np.linspace(1e-6, 1.0 - alpha, n_grid)
    z_u = cornish_fisher_z(u, s, k)
    media_cola = float(_trapz(z_u, u) / (1.0 - alpha))
    cvar = -(mu + sigma * media_cola)
    return float(var), float(cvar)


def calcular_metricas_riesgo_cola(w, retornos_df, nivel_confianza=0.95,
                                  probabilidades=None, rf_periodo=0.0,
                                  umbral_omega=None, etiqueta=""):
    """Metricas de riesgo de cola de un portafolio.

    Parametros
    ----------
    w : array (n,) o pd.Series
        Pesos del portafolio.
    retornos_df : pd.DataFrame (m x n) o np.ndarray
        Panel de retornos por activo al horizonte de analisis. Puede ser la
        serie historica realizada o el panel de escenarios del posterior.
    nivel_confianza : float
        Nivel del VaR principal (por defecto 0.95).
    probabilidades : array (m,), opcional
        Probabilidades de cada fila. None => equiponderadas. Permite evaluar
        directamente el posterior de Entropy Pooling.
    rf_periodo : float
        Tasa libre de riesgo del periodo; sirve de MAR para el Sortino.
    umbral_omega : float, opcional
        Umbral tau del Omega ratio. None => UMBRAL_OMEGA_RATIO.

    Devuelve
    --------
    pd.Series con VaR historico y Cornish-Fisher, CVaR historico y CF a los
    niveles de NIVELES_CVAR, Sortino, Omega y estadisticos de apoyo.

        Sortino = (E[r] - MAR) / sqrt(E[min(r - MAR, 0)^2])
        Omega   = E[(r - tau)+] / E[(tau - r)+]
    """
    if isinstance(w, pd.Series):
        w_arr = w.values.astype(float)
    else:
        w_arr = np.asarray(w, dtype=float)

    X = retornos_df.values if isinstance(retornos_df, pd.DataFrame) else np.asarray(retornos_df)
    if X.shape[1] != len(w_arr):
        raise ValueError(f"Dimension incompatible: X tiene {X.shape[1]} activos "
                         f"y w tiene {len(w_arr)}")

    p = _pesos_normalizados(X.shape[0], probabilidades)
    r = X @ w_arr

    mu = float(p @ r)
    d = r - mu
    m2 = float(p @ d ** 2)
    sigma = math.sqrt(max(m2, 1e-18))
    s_std = float(p @ d ** 3) / sigma ** 3
    k_std = float(p @ d ** 4) / sigma ** 4

    perdidas = -r
    tau_omega = UMBRAL_OMEGA_RATIO if umbral_omega is None else umbral_omega

    out = {"Retorno_esperado": mu, "Volatilidad": sigma,
           "Skewness": s_std, "Kurtosis": k_std}

    # --- VaR al nivel principal ---------------------------------------------
    var_h, cvar_h = var_cvar_historico(perdidas, p, nivel_confianza)
    var_cf, cvar_cf = var_cvar_cornish_fisher(mu, sigma, s_std, k_std, nivel_confianza)
    var_gauss = -(mu + sigma * norm.ppf(1.0 - nivel_confianza))
    nc = int(round(nivel_confianza * 100))
    out[f"VaR{nc}_historico"] = var_h
    out[f"VaR{nc}_gaussiano"] = var_gauss
    out[f"VaR{nc}_CornishFisher"] = var_cf

    # --- CVaR / Expected Shortfall a los niveles solicitados ----------------
    for a in NIVELES_CVAR:
        v_h, c_h = var_cvar_historico(perdidas, p, a)
        _, c_cf = var_cvar_cornish_fisher(mu, sigma, s_std, k_std, a)
        na = int(round(a * 100))
        out[f"CVaR{na}_historico"] = c_h
        out[f"CVaR{na}_CornishFisher"] = c_cf

    # --- Ratios ajustados por riesgo de cola --------------------------------
    exceso_mar = r - rf_periodo
    downside = np.sqrt(float(p @ np.minimum(exceso_mar, 0.0) ** 2))
    out["Sortino"] = float((mu - rf_periodo) / downside) if downside > 1e-12 else np.nan

    ganancia = float(p @ np.maximum(r - tau_omega, 0.0))
    perdida = float(p @ np.maximum(tau_omega - r, 0.0))
    out["Omega"] = float(ganancia / perdida) if perdida > 1e-12 else np.inf

    # --- Apoyo ---------------------------------------------------------------
    out["Sharpe"] = float((mu - rf_periodo) / sigma) if sigma > 1e-12 else np.nan
    out["Prob_perdida"] = float(p[r < 0].sum())
    out["Peor_escenario"] = float(r.min())

    return pd.Series(out, name=etiqueta if etiqueta else None)

# =============================================================================
# BLOQUE 8B: OPTIMIZACION - MODO MVSK O MODO CVaR
# -----------------------------------------------------------------------------
# Dos modos seleccionables con MODO_OPTIMIZACION, ambos sobre el posterior no
# gaussiano del Bloque 7:
#   "mvsk"  maximiza la utilidad esperada por expansion de Taylor de 4.o orden.
#   "cvar"  minimiza CVaR_alpha con un retorno esperado minimo, por el programa
#           lineal de Rockafellar-Uryasev (optimo global).
# =============================================================================

print("\n" + "=" * 79)
print(f"BLOQUE 8B: OPTIMIZACION (modo = {MODO_OPTIMIZACION.upper()})")
print("=" * 79)

mu_opt = mu_post.copy()


def utilidad_mvsk_negativa(w, X, p, gamma, lam3, lam4):
    """-U(w) con U = E[r] - (g/2)m2 + (l3/3)m3 - (l4/4)m4."""
    mu_w, m2, m3, m4 = momentos_portafolio(w, X, p)
    return -(mu_w - (gamma / 2.0) * m2 + (lam3 / 3.0) * m3 - (lam4 / 4.0) * m4)


def optimizar_mvsk(X, p, gamma, lam3, lam4, activos_permitidos=None, w_ini=None):
    n_ = X.shape[1]
    if activos_permitidos is None:
        activos_permitidos = np.ones(n_, dtype=bool)
    bounds = [(0.0, PESO_MAX_ACTIVO) if activos_permitidos[i] else (0.0, 0.0)
              for i in range(n_)]
    if w_ini is None:
        w_ini = activos_permitidos.astype(float)
        w_ini = w_ini / w_ini.sum()
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    res = minimize(utilidad_mvsk_negativa, w_ini, args=(X, p, gamma, lam3, lam4),
                   method="SLSQP", bounds=bounds, constraints=cons,
                   options=dict(maxiter=600, ftol=1e-11))
    if not res.success:
        print(f"  Aviso SLSQP: {res.message}")
    w = np.clip(res.x, 0.0, None)
    return w / w.sum(), res


def optimizar_min_cvar(X, p, alpha, retorno_min, mu_vec,
                       activos_permitidos=None, max_escenarios=MAX_ESCENARIOS_LP,
                       rng=None):
    """LP de Rockafellar-Uryasev. Devuelve (w, info).

        min_{w, zeta, u}   zeta + 1/((1-alpha) M) sum_j u_j
        s.a.  u_j >= -x_j'w - zeta,  u_j >= 0
              mu'w >= retorno_min,  sum w = 1,  0 <= w <= w_max

    El optimo en zeta es el propio VaR_alpha y el valor objetivo es el CVaR.
    Al ser un programa lineal el optimo es global, sin la convergencia local
    de SLSQP.

    El panel se remuestrea por importancia con probabilidades p para que el LP
    trabaje con escenarios equiponderados de tamano manejable sin sesgar la
    distribucion posterior.
    """
    rng = rng_global if rng is None else rng
    J_, n_ = X.shape
    if activos_permitidos is None:
        activos_permitidos = np.ones(n_, dtype=bool)

    if J_ > max_escenarios:
        idx = rng.choice(J_, size=max_escenarios, replace=True, p=p)
        Xs = X[idx]
    else:
        # Remuestreo tambien aqui para poder usar pesos uniformes en el LP
        idx = rng.choice(J_, size=J_, replace=True, p=p)
        Xs = X[idx]
    M = Xs.shape[0]

    # z = [w (n_), zeta (1), u (M)]
    c = np.concatenate([np.zeros(n_), [1.0], np.full(M, 1.0 / ((1.0 - alpha) * M))])

    # -x_j'w - zeta - u_j <= 0
    A1 = sparse.hstack([
        sparse.csr_matrix(-Xs),
        sparse.csr_matrix(-np.ones((M, 1))),
        -sparse.identity(M, format="csr"),
    ], format="csr")
    b1 = np.zeros(M)

    # -mu'w <= -retorno_min
    A2 = sparse.csr_matrix(
        np.concatenate([-mu_vec, [0.0], np.zeros(M)]).reshape(1, -1))
    b2 = np.array([-retorno_min])

    A_ub = sparse.vstack([A1, A2], format="csr")
    b_ub = np.concatenate([b1, b2])

    A_eq = sparse.csr_matrix(
        np.concatenate([np.ones(n_), [0.0], np.zeros(M)]).reshape(1, -1))
    b_eq = np.array([1.0])

    bounds = ([(0.0, PESO_MAX_ACTIVO) if activos_permitidos[i] else (0.0, 0.0)
               for i in range(n_)]
              + [(None, None)]
              + [(0.0, None)] * M)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")

    if not res.success:
        return None, dict(exito=False, mensaje=res.message)

    w = np.clip(res.x[:n_], 0.0, None)
    w = w / w.sum()
    return w, dict(exito=True, cvar=float(res.fun), var=float(res.x[n_]),
                   mensaje=res.message, n_escenarios=M)


def aplicar_limites_cartera(w, umbral=UMBRAL_PESO_MIN, max_activos=MAX_TICKERS_FINAL):
    """Umbral de peso minimo + cardinalidad maxima. Devuelve la mascara final."""
    mask = w >= umbral
    if mask.sum() == 0:
        mask = np.zeros_like(w, dtype=bool)
        mask[int(np.argmax(w))] = True
    if mask.sum() > max_activos:
        idx_top = np.argsort(np.where(mask, w, -np.inf))[::-1][:max_activos]
        mask = np.zeros_like(w, dtype=bool)
        mask[idx_top] = True
    return mask


def _retorno_min_factible(retorno_deseado, mu_vec, activos_permitidos=None, etiqueta=""):
    """Recorta el retorno objetivo del LP de CVaR al maximo alcanzable dentro
    del conjunto de activos permitidos (mu_max = max(mu_vec) sobre el
    soporte). Un retorno_min por encima de ese maximo vuelve el LP infactible
    solo por la restriccion de retorno, sin que eso refleje ningun problema
    real de riesgo: se ajusta dinamicamente en vez de fallar."""
    mu_disp = mu_vec if activos_permitidos is None else mu_vec[activos_permitidos]
    r_max = float(np.max(mu_disp))
    if retorno_deseado > r_max:
        print(f"  Aviso: retorno objetivo ({retorno_deseado:.4f}) supera el maximo "
              f"retorno alcanzable{etiqueta} ({r_max:.4f}); se ajusta al maximo.")
        return r_max
    return retorno_deseado


# --- Retorno minimo exigido en modo CVaR ------------------------------------
retorno_min_deseado = (float(w_mkt @ mu_opt) if RETORNO_MIN_CVAR is None
                       else float(RETORNO_MIN_CVAR))
retorno_min_efectivo = _retorno_min_factible(retorno_min_deseado, mu_opt,
                                             etiqueta=" en el universo completo")

# --- Primera pasada ---------------------------------------------------------
if MODO_OPTIMIZACION == "cvar":
    print(f"\n  Minimizando CVaR_{ALPHA_CVAR_OBJETIVO:.0%} con "
          f"E[r] >= {retorno_min_efectivo:.4f} "
          f"({'benchmark de mercado' if RETORNO_MIN_CVAR is None else 'fijado por el usuario'})")
    w_bruto, info_opt = optimizar_min_cvar(
        X_prior, p_post, ALPHA_CVAR_OBJETIVO, retorno_min_efectivo, mu_opt)
    if w_bruto is None:
        print(f"  LP infactible ({info_opt['mensaje']}). Se recurre al modo MVSK.")
        w_bruto, info_opt = optimizar_mvsk(X_prior, p_post, gamma_ra, LAMBDA3, LAMBDA4)
        modo_efectivo = "mvsk (fallback)"
    else:
        print(f"  LP resuelto sobre {info_opt['n_escenarios']} escenarios | "
              f"VaR={info_opt['var']:.4f} | CVaR={info_opt['cvar']:.4f}")
        modo_efectivo = "cvar"
else:
    print(f"\n  Maximizando utilidad MVSK | gamma={gamma_ra} | "
          f"lambda3={LAMBDA3} | lambda4={LAMBDA4}")
    w_bruto, info_opt = optimizar_mvsk(X_prior, p_post, gamma_ra, LAMBDA3, LAMBDA4)
    modo_efectivo = "mvsk"

# --- Segunda pasada: re-optimizacion sobre el soporte final -----------------
# Truncar y renormalizar destruiria la optimalidad: se fija el soporte y se
# resuelve otra vez el mismo problema restringido a el.
mask_final = aplicar_limites_cartera(w_bruto)
print(f"  Soporte final: {int(mask_final.sum())} activos "
      f"(umbral {UMBRAL_PESO_MIN:.1%}, maximo {MAX_TICKERS_FINAL})")

if modo_efectivo == "cvar":
    # El soporte reducido (MAX_TICKERS_FINAL activos) puede no alcanzar el
    # retorno minimo del universo completo: se recorta el target al maximo
    # retorno posible del subconjunto ANTES de resolver, en vez de esperar a
    # que el LP falle por infactibilidad.
    retorno_min_soporte = _retorno_min_factible(
        retorno_min_efectivo, mu_opt, activos_permitidos=mask_final,
        etiqueta=" en el soporte reducido")
    w_re, info_re = optimizar_min_cvar(X_prior, p_post, ALPHA_CVAR_OBJETIVO,
                                       retorno_min_soporte, mu_opt,
                                       activos_permitidos=mask_final)
    if w_re is None:
        # Infactibilidad residual por otra causa (no el retorno minimo, ya
        # acotado arriba); se usa el bruto truncado y renormalizado.
        print(f"  Re-optimizacion CVaR infactible en el soporte reducido "
              f"({info_re['mensaje']}); se usa el bruto truncado y renormalizado.")
    w_opt = w_re if w_re is not None else (w_bruto * mask_final) / (w_bruto * mask_final).sum()
else:
    w_ini = np.where(mask_final, w_bruto, 0.0)
    w_ini = w_ini / w_ini.sum()
    w_opt, _ = optimizar_mvsk(X_prior, p_post, gamma_ra, LAMBDA3, LAMBDA4,
                              activos_permitidos=mask_final, w_ini=w_ini)

w_opt = np.where(w_opt < 1e-10, 0.0, w_opt)
w_opt = w_opt / w_opt.sum()
w_mvsk = pd.Series(w_opt, index=tickers)

print(f"\n=== Pesos optimos - Portafolio BL+BKM ({modo_efectivo.upper()}) ===")
print(w_mvsk[w_mvsk > 0].sort_values(ascending=False).round(4).to_string())

# =============================================================================
# BLOQUE 8C: PORTAFOLIO MARKOWITZ TRADICIONAL (CONTROL)
# -----------------------------------------------------------------------------
# Media-varianza clasica con insumos 100% historicos: sin Black-Litterman, sin
# informacion de opciones y sin momentos superiores. Es la referencia contra la
# que se mide todo lo anterior.
# =============================================================================

print("\n" + "=" * 79)
print("BLOQUE 8C: PORTAFOLIO MARKOWITZ TRADICIONAL (CONTROL)")
print("=" * 79)


def markowitz_clasico(mu_vec, Sigma_arr, gamma, w_max=PESO_MAX_ACTIVO):
    """QP de media-varianza long-only con el MISMO tope por activo que el
    optimizador del Bloque 8B, para que la comparacion sea justa.

        max_w  mu'w - (gamma/2) w'Sigma w   s.a.  sum w = 1,  0 <= w <= w_max

    quadprog resuelve el QP con restricciones de desigualdad; si falla (matriz
    no definida positiva, por ejemplo) se recurre a SLSQP.
    """
    n_ = len(mu_vec)
    w_max = min(max(w_max, 1.0 / n_), 1.0)   # el tope debe permitir sum w = 1
    try:
        G = gamma * (Sigma_arr + Sigma_arr.T) / 2.0 + np.eye(n_) * 1e-8
        # Restricciones: sum w = 1 (igualdad), w >= 0, -w >= -w_max
        Amat = np.column_stack([np.ones(n_), np.eye(n_), -np.eye(n_)])
        bvec = np.concatenate([[1.0], np.zeros(n_), np.full(n_, -w_max)])
        w = quadprog.solve_qp(G, mu_vec, Amat, bvec, meq=1)[0]
        w = np.clip(w, 0.0, None)
        return w / w.sum()
    except Exception as e:
        print(f"  quadprog fallo ({e}); se usa SLSQP")
        obj = lambda w: -(w @ mu_vec - (gamma / 2.0) * w @ Sigma_arr @ w)
        res = minimize(obj, np.full(n_, 1.0 / n_), method="SLSQP",
                       bounds=[(0.0, w_max)] * n_,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}])
        w = np.clip(res.x, 0.0, None)
        return w / w.sum()


w_mkw_bruto = markowitz_clasico(mu_historico, Sigma_hist, gamma_ra)
mask_mkw = aplicar_limites_cartera(w_mkw_bruto)

# Re-optimizacion sobre el subespacio seleccionado (mismo criterio que en 8B:
# se resuelve el QP restringido, no se trunca y renormaliza).
idx_mkw = np.where(mask_mkw)[0]
w_sub = markowitz_clasico(mu_historico[idx_mkw],
                          Sigma_hist[np.ix_(idx_mkw, idx_mkw)], gamma_ra)
w_mkw = np.zeros(n)
w_mkw[idx_mkw] = w_sub
w_markowitz = pd.Series(w_mkw, index=tickers)

print("\n=== Pesos - Markowitz tradicional (mu_hist, Sigma_hist) ===")
print(w_markowitz[w_markowitz > 0].sort_values(ascending=False).round(4).to_string())

# =============================================================================
# BLOQUE 9: METRICAS COMPARATIVAS DE RIESGO DE COLA
# -----------------------------------------------------------------------------
# Compara mercado, Markowitz (control) y BL+BKM sobre dos distribuciones: el
# posterior del modelo (forward-looking, sobre el que se optimizo) y los
# retornos historicos realizados al horizonte (validacion fuera del modelo).
# =============================================================================

etiqueta_horizonte = f"{MESES_HORIZONTE} mes(es)"

print("\n" + "=" * 79)
print(f"BLOQUE 9: METRICAS DE RIESGO DE COLA ({etiqueta_horizonte})")
print("=" * 79)

# --- Panel historico de retornos al horizonte (ventanas solapadas) ----------
ret_hist_horizonte = (retornos_dia[tickers]
                      .rolling(horizonte_dias)
                      .sum()
                      .dropna())
print(f"\nPanel historico: {len(ret_hist_horizonte)} ventanas solapadas de "
      f"{horizonte_dias} dias | Panel posterior: {N_ESCENARIOS} escenarios")

portafolios = {
    "Mercado (w_mkt)": pd.Series(w_mkt, index=tickers),
    "Markowitz (control)": w_markowitz,
    f"BL+BKM {modo_efectivo.upper()}": w_mvsk,
}

# --- (1) Bajo el posterior del modelo ---------------------------------------
metricas_post = pd.DataFrame({
    nombre: calcular_metricas_riesgo_cola(
        w, X_prior, nivel_confianza=NIVEL_CONFIANZA_VAR,
        probabilidades=p_post, rf_periodo=Rf_h, etiqueta=nombre)
    for nombre, w in portafolios.items()
})

# --- (2) Bajo los retornos historicos realizados ----------------------------
metricas_hist = pd.DataFrame({
    nombre: calcular_metricas_riesgo_cola(
        w, ret_hist_horizonte, nivel_confianza=NIVEL_CONFIANZA_VAR,
        rf_periodo=Rf_h, etiqueta=nombre)
    for nombre, w in portafolios.items()
})

nc = int(round(NIVEL_CONFIANZA_VAR * 100))
orden_filas = [
    "Retorno_esperado", "Volatilidad", "Skewness", "Kurtosis",
    f"VaR{nc}_historico", f"VaR{nc}_gaussiano", f"VaR{nc}_CornishFisher",
] + [f"CVaR{int(round(a * 100))}_historico" for a in NIVELES_CVAR] \
  + [f"CVaR{int(round(a * 100))}_CornishFisher" for a in NIVELES_CVAR] \
  + ["Sharpe", "Sortino", "Omega", "Prob_perdida", "Peor_escenario"]

print(f"\n--- (1) Bajo el POSTERIOR del modelo ({metodo_posterior_usado}) ---")
print(metricas_post.loc[orden_filas].round(4).to_string())

print("\n--- (2) Bajo los RETORNOS HISTORICOS realizados (validacion) ---")
print(metricas_hist.loc[orden_filas].round(4).to_string())

# --- Lectura del efecto de los momentos superiores --------------------------
print("\n--- Efecto de los momentos de orden superior sobre el VaR ---")
print("(VaR_CF - VaR_gaussiano > 0 => la normalidad SUBESTIMA la perdida)")
brecha = (metricas_post.loc[f"VaR{nc}_CornishFisher"]
          - metricas_post.loc[f"VaR{nc}_gaussiano"])
print(pd.DataFrame({
    f"VaR{nc}_gauss": metricas_post.loc[f"VaR{nc}_gaussiano"].round(4),
    f"VaR{nc}_CF": metricas_post.loc[f"VaR{nc}_CornishFisher"].round(4),
    "Brecha": brecha.round(4),
    "Brecha_%": (100 * brecha / metricas_post.loc[f"VaR{nc}_gaussiano"]).round(1),
}).to_string())

# --- Variables de compatibilidad para los bloques siguientes ----------------
w_mvsk_vec = w_mvsk.values
mu_BL_vec = mu_post.copy()
ret_port, m2_port, m3_port, m4_port = momentos_portafolio(w_mvsk_vec, X_prior, p_post)
vol_port = math.sqrt(max(m2_port, 1e-18))
sharpe_BL = float(metricas_post.loc["Sharpe", f"BL+BKM {modo_efectivo.upper()}"])
sharpe_mkt = float(metricas_post.loc["Sharpe", "Mercado (w_mkt)"])
sharpe_mkw = float(metricas_post.loc["Sharpe", "Markowitz (control)"])

print(f"\n=== Resumen del portafolio BL+BKM ({modo_efectivo.upper()}) ===")
print(f"  Retorno esperado:  {ret_port:.4f}")
print(f"  Volatilidad:       {vol_port:.4f}")
print(f"  Skewness (m3):     {m3_port:.6f}")
print(f"  Kurtosis (m4):     {m4_port:.6f}")
print(f"  Sharpe:            {sharpe_BL:.4f}   (mercado: {sharpe_mkt:.4f} | "
      f"Markowitz: {sharpe_mkw:.4f})")
print(f"  Rf al horizonte:   {Rf_h:.4f}")

# =============================================================================
# BLOQUE 10: VISUALIZACION
# =============================================================================

comp_sorted = comparacion.sort_values("Ajuste_BL")
colores_ajuste = ["#70AD47" if v > 0 else "#ED7D31" for v in comp_sorted["Ajuste_BL"]]
nombres_port = list(portafolios.keys())
colores_port = ["#4472C4", "#FFC000", "#70AD47"]

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=(
        f"Retornos esperados - {etiqueta_horizonte}",
        "Pesos: Mercado vs Markowitz vs BL+BKM",
        f"VaR{nc} y CVaR: gaussiano vs Cornish-Fisher",
        "Ratios ajustados por riesgo de cola",
        "Primas de riesgo de momentos (Q - P)",
        "Ajuste mu_BL vs pi (views + prima no gaussiana)",
    ),
    vertical_spacing=0.10,
)

# (1,1) Retornos esperados
fig.add_trace(go.Bar(x=comparacion["Ticker"], y=comparacion["Pi_eq"], name="pi (equilibrio)",
                     marker_color="#4472C4"), row=1, col=1)
fig.add_trace(go.Bar(x=comparacion["Ticker"], y=comparacion["Mu_hist"], name="Historico",
                     marker_color="#FFC000"), row=1, col=1)
fig.add_trace(go.Bar(x=tickers, y=mu_post, name="mu posterior (no gaussiano)",
                     marker_color="#ED7D31"), row=1, col=1)

# (1,2) Pesos
for nombre, color in zip(nombres_port, colores_port):
    fig.add_trace(go.Bar(x=tickers, y=portafolios[nombre].values, name=nombre,
                         marker_color=color, showlegend=True), row=1, col=2)

# (2,1) VaR / CVaR comparativo
filas_riesgo = [f"VaR{nc}_gaussiano", f"VaR{nc}_CornishFisher",
                f"CVaR{int(round(NIVELES_CVAR[0] * 100))}_historico",
                f"CVaR{int(round(NIVELES_CVAR[-1] * 100))}_historico"]
etq_riesgo = [f"VaR{nc} gauss", f"VaR{nc} CF",
              f"CVaR{int(round(NIVELES_CVAR[0] * 100))}",
              f"CVaR{int(round(NIVELES_CVAR[-1] * 100))}"]
for nombre, color in zip(nombres_port, colores_port):
    fig.add_trace(go.Bar(x=etq_riesgo, y=[metricas_post.loc[f, nombre] for f in filas_riesgo],
                         name=nombre, marker_color=color, showlegend=False), row=2, col=1)

# (2,2) Sharpe / Sortino / Omega
for nombre, color in zip(nombres_port, colores_port):
    fig.add_trace(go.Bar(x=["Sharpe", "Sortino", "Omega"],
                         y=[metricas_post.loc["Sharpe", nombre],
                            metricas_post.loc["Sortino", nombre],
                            metricas_post.loc["Omega", nombre]],
                         name=nombre, marker_color=color, showlegend=False), row=2, col=2)

# (3,1) Primas de riesgo de momentos
fig.add_trace(go.Bar(x=tickers, y=VRP, name="VRP", marker_color="#4472C4",
                     showlegend=False), row=3, col=1)
fig.add_trace(go.Bar(x=tickers, y=SRP, name="SRP", marker_color="#ED7D31",
                     showlegend=False), row=3, col=1)
fig.add_trace(go.Bar(x=tickers, y=KRP / 10.0, name="KRP/10", marker_color="#A5A5A5",
                     showlegend=False), row=3, col=1)

# (3,2) Ajuste BL
fig.add_trace(go.Bar(x=comp_sorted["Ticker"], y=comp_sorted["Ajuste_BL"],
                     marker_color=colores_ajuste, showlegend=False,
                     name="Ajuste BL"), row=3, col=2)
fig.add_hline(y=0, line_color="black", row=3, col=2)

for r, c in [(1, 1), (1, 2), (3, 1), (3, 2)]:
    fig.update_xaxes(tickangle=90, row=r, col=c)
fig.update_yaxes(title_text="Retorno esperado", row=1, col=1)
fig.update_yaxes(title_text="Peso", row=1, col=2)
fig.update_yaxes(title_text="Perdida", row=2, col=1)
fig.update_yaxes(title_text="Ratio", row=2, col=2)
fig.update_yaxes(title_text="Prima (Q - P)", row=3, col=1)
fig.update_yaxes(title_text="mu_post - pi", row=3, col=2)

fig.update_layout(
    title=(f"Black-Litterman + BKM extendido - {etiqueta_horizonte}<br>"
           f"<sup>Perfil: {PERFIL_RIESGO.upper()} | Posterior: {metodo_posterior_usado} | "
           f"Optimizacion: {modo_efectivo.upper()}</sup>"),
    barmode="group",
    template="plotly_white",
    height=1250,
    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
)
fig.show()

# --- Distribucion posterior del portafolio vs normal ------------------------
r_port_esc = X_prior @ w_mvsk_vec
orden_esc = np.argsort(r_port_esc)
r_ord = r_port_esc[orden_esc]
p_ord = p_post[orden_esc]

fig2 = go.Figure()
fig2.add_trace(go.Histogram(x=r_port_esc, histnorm="probability density",
                            nbinsx=90, name="Posterior (no gaussiano)",
                            marker_color="#4472C4", opacity=0.55))
grid = np.linspace(r_ord.min(), r_ord.max(), 400)
fig2.add_trace(go.Scatter(x=grid,
                          y=norm.pdf(grid, ret_port, vol_port),
                          mode="lines", name="Normal(mu, sigma) equivalente",
                          line=dict(color="#ED7D31", width=2)))
var_cf_port = metricas_post.loc[f"VaR{nc}_CornishFisher", f"BL+BKM {modo_efectivo.upper()}"]
cvar_port = metricas_post.loc[f"CVaR{nc}_historico", f"BL+BKM {modo_efectivo.upper()}"]
fig2.add_vline(x=-var_cf_port, line_dash="dash", line_color="darkorange",
               annotation_text=f"VaR{nc} CF: {var_cf_port * 100:.2f}%",
               annotation_position="top left")
fig2.add_vline(x=-cvar_port, line_dash="dash", line_color="darkred",
               annotation_text=f"CVaR{nc}: {cvar_port * 100:.2f}%",
               annotation_position="bottom left")
fig2.update_layout(
    title=("Distribucion posterior del retorno del portafolio BL+BKM<br>"
           f"<sup>Asimetria: {metricas_post.loc['Skewness', f'BL+BKM {modo_efectivo.upper()}']:.3f} | "
           f"Curtosis: {metricas_post.loc['Kurtosis', f'BL+BKM {modo_efectivo.upper()}']:.3f} | "
           f"Horizonte: {etiqueta_horizonte}</sup>"),
    xaxis_title=f"Retorno a {etiqueta_horizonte}", yaxis_title="Densidad",
    template="plotly_white", height=520,
)
fig2.show()

# =============================================================================
# BLOQUE 11: MAXIMUM DRAWDOWN (MDD) DEL PORTAFOLIO OPTIMIZADO
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

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_data["year"], y=plot_data["mdd_pct"], mode="lines+markers",
        line=dict(color="darkred", width=1.5),
        marker=dict(color=plot_data["con_dato"].map({True: "darkred", False: "gray"}), size=8),
        hovertemplate="Ano %{x}: %{y:.2f}%<extra></extra>", showlegend=False,
    ))
    fig.add_hline(y=mdd_mediana * 100, line_dash="dash", line_color="steelblue",
                  annotation_text=f"Mediana: {mdd_mediana * 100:.2f}%", annotation_position="top left",
                  annotation_font_color="steelblue")
    fig.add_hline(y=mdd_p90 * 100, line_dash="dash", line_color="darkorange",
                  annotation_text=f"P90: {mdd_p90 * 100:.2f}%", annotation_position="bottom left",
                  annotation_font_color="darkorange")
    fig.update_layout(
        title=dict(text="Maximum Drawdown Historico - Portafolio BL + BKM<br>"
                         f"<sup>Perfil: {PERFIL_RIESGO.upper()} | Horizonte: {etiqueta_horizonte} | "
                         f"MDD global ({MDD_START_YEAR}-hoy): {mdd_global * 100:.2f}% | "
                         f"{int(plot_data['con_dato'].sum())} anos con datos</sup>"),
        xaxis_title="Ano", yaxis_title="MDD (%)",
        xaxis=dict(tickmode="linear", tick0=MDD_START_YEAR, dtick=1, tickangle=45),
        template="plotly_white",
    )
    fig.show()
    print("  Grafico de MDD generado.")

else:
    print("  Datos insuficientes para calcular MDD (minimo 10 observaciones).")
    print("  Verifica que MDD_START_YEAR este dentro de la ventana de 2 anios.")

_pf = f"BL+BKM {modo_efectivo.upper()}"
print("\n" + "=" * 79)
print("FIN - BLACK-LITTERMAN EXTENDIDO POR MOMENTOS DE ORDEN SUPERIOR (BKM)")
print("=" * 79)
print(f"  Perfil de riesgo:       {PERFIL_RIESGO.upper()}")
print(f"  Horizonte:              {etiqueta_horizonte} ({horizonte_dias} dias habiles)")
print(f"  Universo:               {n} tickers | Views: {N_VIEWS}")
print(f"  Ajuste Q -> P:          Mincer-Zarnowitz (VRP/SRP/KRP) + Esscher por GMM")
print(f"  Posterior:              {metodo_posterior_usado} "
      f"(ENS {info_post['ens'] * 100:.1f}% de {N_ESCENARIOS} escenarios)")
print(f"  Optimizacion:           {modo_efectivo.upper()}")
print(f"  VaR{nc} Cornish-Fisher:   {metricas_post.loc[f'VaR{nc}_CornishFisher', _pf]:.4f}")
print(f"  CVaR{nc} historico:       {metricas_post.loc[f'CVaR{nc}_historico', _pf]:.4f}")
print(f"  Sortino / Omega:        {metricas_post.loc['Sortino', _pf]:.3f} / "
      f"{metricas_post.loc['Omega', _pf]:.3f}")
print(f"  MDD analizado desde:    {MDD_START_YEAR}")
