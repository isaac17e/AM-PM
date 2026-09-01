# ============================================================
# ENTRY SIGNAL TOOL - Score de Conviccion para Entrada en Portafolio
# ============================================================

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import time
import os
import json
import plotly.graph_objects as go

# ---------------- CONFIGURACION ----------------
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.environ.get("POLYGON_API_KEY")

BASE_URL = "https://api.polygon.io"

# Peso objetivo de cada activo dentro del portafolio total (debe coincidir con
# `portfolio` en portfolio_risk_score_leverage.py). Se usa para traducir el %
# ya invertido / % pendiente de cada activo a puntos porcentuales del portafolio.
PESOS_OBJETIVO = {
    "XLU": 0.12, "GLD": 0.12, "T": 0.1031, "GILD": 0.0925, "FXI": 0.0779,
    "MRK": 0.0734, "ADP": 0.0595, "KO": 0.0578, "ABT": 0.0562, "VRTX": 0.0541,
    "AMGN": 0.0509, "UNP": 0.0485, "NEE": 0.0484, "ABBV": 0.0235, "TMO": 0.0142,
}

TICKERS = list(PESOS_OBJETIVO.keys())

PESOS = {
    "gex_regime": 0.18,
    "zero_gamma_dist": 0.15,
    "wall_space": 0.12,
    "iv_rank": 0.15,
    "skew": 0.1,
    "expected_move": 0.1,
    "smart_money": 0.1,
    "volumen_relativo": 0.1,
}

UMBRAL_ALTO = 75
UMBRAL_MEDIO = 40

# Dias habiles consecutivos con score por debajo de UMBRAL_MEDIO antes de forzar
# una decision definitiva (ENTRAR/ESPERAR) en vez de seguir goteando la entrada.
UMBRAL_DIAS_DECISION_FORZADA = 5

HORIZON_DIAS_OBJETIVO = 30
VENTANA_BUSQUEDA_VENCIMIENTO_DIAS = 20

SEGUNDOS_ENTRE_LLAMADAS_STOCKS = 13  # respeta el limite de 5/min del tier gratuito Stocks Basic

HIST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "entry_signal_history.csv")
STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "entry_state.json")

# Si ya corriste el script hoy con el mismo portafolio, se reemplaza esa fila en vez de duplicarla
def cargar_historial():
    if os.path.exists(HIST_PATH):
        return pd.read_csv(HIST_PATH, parse_dates=["fecha"])
    return pd.DataFrame()

# ---------------- ESTADO DE ENTRADAS (POSICIONES YA TOMADAS) ----------------

def cargar_estado():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    return None

def guardar_estado(estado):
    with open(STATE_PATH, "w") as f:
        json.dump(estado, f, indent=2)

def pedir_pct(mensaje):
    while True:
        crudo = input(mensaje).strip().replace("%", "")
        if crudo == "":
            return 0.0
        try:
            valor = float(crudo)
        except ValueError:
            print("  Ingresa un numero valido (ej: 40 para 40%).")
            continue
        if valor < 0 or valor > 100:
            print("  El porcentaje debe estar entre 0 y 100.")
            continue
        return valor / 100

def inicializar_estado():
    hoy = datetime.now(timezone.utc).date().isoformat()
    respuesta = input("\n¿Ya tienes posiciones abiertas en este portafolio? [s/n]: ").strip().lower()
    estado = {}
    if respuesta.startswith("s"):
        print("\nIndica el % ya invertido en cada activo, respecto a SU peso objetivo (0-100). Enter = 0%.\n")
        for ticker in TICKERS:
            peso = PESOS_OBJETIVO.get(ticker, np.nan)
            pct = pedir_pct(f"  {ticker} (peso objetivo {peso*100:.2f}% del portafolio): ")
            estado[ticker] = {"pct_ya_invertido": pct, "dias_score_bajo": 0, "ultima_actualizacion": hoy}
    else:
        print("\nEntendido: dia 1 del portafolio, se parte de 0% invertido en todos los activos.\n")
        for ticker in TICKERS:
            estado[ticker] = {"pct_ya_invertido": 0.0, "dias_score_bajo": 0, "ultima_actualizacion": hoy}
    return estado

def permitir_correccion_manual(estado):
    respuesta = input(
        "\n¿Quieres corregir manualmente el % ya invertido de algun activo antes de continuar? [s/n]: "
    ).strip().lower()
    if not respuesta.startswith("s"):
        return estado
    print("Escribe el ticker a corregir (Enter vacio para terminar).")
    while True:
        ticker = input("  Ticker: ").strip().upper()
        if ticker == "":
            break
        if ticker not in estado:
            print(f"  '{ticker}' no esta en el portafolio, se ignora.")
            continue
        estado[ticker]["pct_ya_invertido"] = pedir_pct(f"  Nuevo % ya invertido en {ticker} (0-100): ")
    return estado

def evaluar_decision_forzada(ticker, hist_df):
    hist_ticker = hist_df[hist_df["ticker"] == ticker].sort_values("fecha").tail(5)
    if len(hist_ticker) < 3:
        return "ESPERAR", "historial insuficiente para una decision forzada confiable"

    señales_favorables = 0
    señales_totales = 0
    detalles = []

    smart_money = hist_ticker["smart_money"].dropna()
    if len(smart_money) >= 2:
        señales_totales += 1
        if smart_money.tail(3).mean() > 0:
            señales_favorables += 1
            detalles.append("smart money favorable (put/call < 1)")
        else:
            detalles.append("smart money desfavorable (put/call >= 1)")

    dist_zg = hist_ticker["dist_zero_gamma"].dropna().abs()
    if len(dist_zg) >= 2:
        señales_totales += 1
        if dist_zg.iloc[-1] < dist_zg.iloc[0]:
            señales_favorables += 1
            detalles.append("precio acercandose al zero-gamma (mayor estabilidad esperada)")
        else:
            detalles.append("precio alejandose del zero-gamma")

    if señales_totales == 0:
        return "ESPERAR", "sin señales de flujo de opciones suficientes"

    decision = "ENTRAR" if señales_favorables / señales_totales >= 0.5 else "ESPERAR"
    return decision, "; ".join(detalles)

# ---------------- FUNCIONES DE EXTRACCION ----------------

def get_daily_history(ticker, dias=40):
    fecha_fin = datetime.now(timezone.utc).date()
    fecha_inicio = fecha_fin - timedelta(days=dias * 2)  # buffer por fines de semana
    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{fecha_inicio}/{fecha_fin}"
    params = {"adjusted": "true", "sort": "asc", "limit": 5000, "apiKey": API_KEY}
    r = requests.get(url, params=params).json()
    if r.get("status") not in ("OK", "DELAYED") or "results" not in r:
        raise RuntimeError(f"Fallo consultando historico diario de {ticker}: {r}")
    df = pd.DataFrame(r["results"])
    if df.empty:
        return df
    df["fecha"] = pd.to_datetime(df["t"], unit="ms")
    return df[["fecha", "v", "vw", "c"]].rename(columns={"v": "volumen", "vw": "vwap", "c": "cierre"})

def get_spot_y_volumen_relativo(ticker):
    df = get_daily_history(ticker, dias=30)
    if df.empty or len(df) < 5:
        return np.nan, np.nan
    spot = df["cierre"].iloc[-1]
    adv_20 = df["volumen"].tail(20).mean()
    volumen_hoy = df["volumen"].iloc[-1]
    volumen_relativo = volumen_hoy / adv_20
    return spot, volumen_relativo

def seleccionar_vencimiento_objetivo(ticker, horizon_days):
    hoy = datetime.now(timezone.utc).date()
    fecha_objetivo = hoy + timedelta(days=horizon_days)
    url = f"{BASE_URL}/v3/reference/options/contracts"
    params = {
        "underlying_ticker": ticker,
        "expiration_date.gte": str(hoy),
        "expiration_date.lte": str(fecha_objetivo + timedelta(days=VENTANA_BUSQUEDA_VENCIMIENTO_DIAS)),
        "limit": 1000,
        "order": "asc",
        "sort": "expiration_date",
        "apiKey": API_KEY,
    }
    r = requests.get(url, params=params).json()
    resultados = r.get("results", [])
    if not resultados:
        return None
    vencimientos = sorted({c["expiration_date"] for c in resultados if c.get("expiration_date")})
    if not vencimientos:
        return None
    vencimientos = [datetime.strptime(v, "%Y-%m-%d").date() for v in vencimientos]
    return min(vencimientos, key=lambda d: abs((d - fecha_objetivo).days))

def get_options_chain(ticker, spot, vencimiento):
    url = f"{BASE_URL}/v3/snapshot/options/{ticker}"
    params = {
        "strike_price.gte": round(spot * 0.85, 2),
        "strike_price.lte": round(spot * 1.15, 2),
        "expiration_date": vencimiento.strftime("%Y-%m-%d"),
        "limit": 250,
        "apiKey": API_KEY
    }
    resultados = []
    while url:
        r = requests.get(url, params=params).json()
        resultados.extend(r.get("results", []))
        next_url = r.get("next_url")
        if next_url:
            url = next_url
            params = {"apiKey": API_KEY}
        else:
            url = None
    return resultados

def parse_chain(chain):
    filas = []
    for c in chain:
        details = c.get("details", {})
        greeks = c.get("greeks", {})
        day = c.get("day", {})
        filas.append({
            "strike": details.get("strike_price"),
            "tipo": details.get("contract_type"),
            "vencimiento": details.get("expiration_date"),
            "oi": c.get("open_interest", 0),
            "volumen": day.get("volume", 0),
            "iv": c.get("implied_volatility", None),
            "delta": greeks.get("delta"),
            "gamma": greeks.get("gamma"),
            "bid": c.get("last_quote", {}).get("bid"),
            "ask": c.get("last_quote", {}).get("ask"),
        })
    return pd.DataFrame(filas)

# ---------------- CALCULO DE INDICADORES ----------------

def calcular_gex_y_zero_gamma(df_chain, spot):
    df = df_chain.dropna(subset=["gamma", "oi"]).copy()
    if df.empty:
        return np.nan, np.nan
    signo = np.where(df["tipo"] == "call", 1, -1)
    df["gex_strike"] = signo * df["gamma"] * df["oi"] * 100 * spot**2 * 0.01
    gex_por_strike = df.groupby("strike")["gex_strike"].sum().sort_index()
    gex_total = gex_por_strike.sum()

    # Se excluyen strikes sin exposicion real (gex_strike == 0, tipicamente sin
    # OI) de la busqueda del cruce, y se detecta el cruce en cualquier
    # direccion (no solo negativo->positivo) para no perder flips genuinos.
    gex_sig = gex_por_strike[gex_por_strike != 0]
    acumulado_sig = gex_sig.cumsum()
    zero_gamma = np.nan
    if len(acumulado_sig) >= 2:
        signs = np.sign(acumulado_sig.values)
        change_idx = np.where(np.diff(signs) != 0)[0]
        if len(change_idx) > 0:
            i = change_idx[0]
            x0, y0 = acumulado_sig.index[i], acumulado_sig.values[i]
            x1, y1 = acumulado_sig.index[i + 1], acumulado_sig.values[i + 1]
            zero_gamma = x0 - y0 * (x1 - x0) / (y1 - y0)

    if pd.isna(zero_gamma):
        # Sin cruce de signo real dentro de la ventana: no hay nivel de
        # zero-gamma confiable que reportar (antes esto devolvia por error
        # la strike de mayor concentracion de gamma, un "wall", como si
        # fuera el flip).
        return gex_total, np.nan

    distancia_zero_gamma = (spot - zero_gamma) / spot
    return gex_total, distancia_zero_gamma

def calcular_walls(df_chain):
    df = df_chain.dropna(subset=["oi"])
    calls = df[df["tipo"] == "call"]
    puts = df[df["tipo"] == "put"]
    call_wall = calls.loc[calls["oi"].idxmax(), "strike"] if not calls.empty else np.nan
    put_wall = puts.loc[puts["oi"].idxmax(), "strike"] if not puts.empty else np.nan
    return call_wall, put_wall

def calcular_espacio_walls(spot, call_wall, put_wall):
    if pd.isna(call_wall) or pd.isna(put_wall) or call_wall == put_wall:
        return np.nan
    rango = call_wall - put_wall
    posicion = (spot - put_wall) / rango
    return 1 - abs(posicion - 0.5) * 2

def calcular_iv_atm(df_chain, spot):
    df = df_chain.dropna(subset=["iv", "strike"]).copy()
    if df.empty:
        return np.nan
    df["dist_spot"] = (df["strike"] - spot).abs()
    atm = df.sort_values("dist_spot").head(6)
    return atm["iv"].mean()

def calcular_skew(df_chain):
    df = df_chain.dropna(subset=["delta", "iv"]).copy()
    if df.empty:
        return np.nan
    calls = df[df["tipo"] == "call"].copy()
    puts = df[df["tipo"] == "put"].copy()
    if calls.empty or puts.empty:
        return np.nan
    calls["dist_delta"] = (calls["delta"] - 0.25).abs()
    puts["dist_delta"] = (puts["delta"] - (-0.25)).abs()
    iv_call_25 = calls.sort_values("dist_delta").iloc[0]["iv"]
    iv_put_25 = puts.sort_values("dist_delta").iloc[0]["iv"]
    return iv_put_25 - iv_call_25

def calcular_expected_move(df_chain, spot):
    df = df_chain.dropna(subset=["delta", "bid", "ask"]).copy()
    if df.empty:
        return np.nan
    df["dist_delta_call"] = (df["delta"] - 0.5).abs()
    df["dist_delta_put"] = (df["delta"] - (-0.5)).abs()
    call_atm = df[df["tipo"] == "call"].sort_values("dist_delta_call").head(1)
    put_atm = df[df["tipo"] == "put"].sort_values("dist_delta_put").head(1)
    if call_atm.empty or put_atm.empty:
        return np.nan
    precio_call = (call_atm["bid"].values[0] + call_atm["ask"].values[0]) / 2
    precio_put = (put_atm["bid"].values[0] + put_atm["ask"].values[0]) / 2
    return (precio_call + precio_put) / spot

def calcular_smart_money(df_chain):
    df = df_chain.dropna(subset=["volumen", "oi"])
    if df.empty:
        return np.nan
    calls_vol = df[df["tipo"] == "call"]["volumen"].sum()
    puts_vol = df[df["tipo"] == "put"]["volumen"].sum()
    if (calls_vol + puts_vol) == 0:
        return np.nan
    put_call_ratio = puts_vol / (calls_vol + 1e-9)
    vol_oi_ratio = df["volumen"].sum() / (df["oi"].sum() + 1e-9)
    return vol_oi_ratio * (1 if put_call_ratio < 1 else -1)

def dias_a_opex(hoy=None):
    hoy = hoy or datetime.now(timezone.utc).date()
    primer_dia = hoy.replace(day=1)
    primer_viernes = primer_dia + timedelta(days=(4 - primer_dia.weekday()) % 7)
    tercer_viernes = primer_viernes + timedelta(days=14)
    if tercer_viernes < hoy:
        if hoy.month == 12:
            primer_dia = hoy.replace(year=hoy.year + 1, month=1, day=1)
        else:
            primer_dia = hoy.replace(month=hoy.month + 1, day=1)
        primer_viernes = primer_dia + timedelta(days=(4 - primer_dia.weekday()) % 7)
        tercer_viernes = primer_viernes + timedelta(days=14)
    return (tercer_viernes - hoy).days

def calcular_vanna_charm_factor():
    dias = dias_a_opex()
    if dias <= 5:
        return (5 - dias) / 5
    return 0.0

# ---------------- NORMALIZACION Y SCORE ----------------

def percentile_historico(hist_df, ticker, columna, valor_actual):
    if hist_df.empty or valor_actual is None or pd.isna(valor_actual):
        return 50.0
    serie = hist_df[hist_df["ticker"] == ticker][columna].dropna()
    if len(serie) < 5:
        return 50.0
    return (serie < valor_actual).mean() * 100

def calcular_indicadores_ticker(ticker, hist_df):
    spot, vol_relativo = get_spot_y_volumen_relativo(ticker)
    vencimiento = seleccionar_vencimiento_objetivo(ticker, HORIZON_DIAS_OBJETIVO)
    if vencimiento is None:
        chain = pd.DataFrame()
    else:
        chain = parse_chain(get_options_chain(ticker, spot, vencimiento))

    gex_total, dist_zero_gamma = calcular_gex_y_zero_gamma(chain, spot)
    call_wall, put_wall = calcular_walls(chain)
    espacio_walls = calcular_espacio_walls(spot, call_wall, put_wall)
    iv_atm = calcular_iv_atm(chain, spot)
    skew = calcular_skew(chain)
    expected_move = calcular_expected_move(chain, spot)
    smart_money = calcular_smart_money(chain)
    vanna_charm = calcular_vanna_charm_factor()

    crudos = {
        "spot": spot,
        "vencimiento": vencimiento,
        "gex_total": gex_total,
        "dist_zero_gamma": dist_zero_gamma,
        "call_wall": call_wall,
        "put_wall": put_wall,
        "espacio_walls": espacio_walls,
        "iv_atm": iv_atm,
        "skew": skew,
        "expected_move": expected_move,
        "smart_money": smart_money,
        "volumen_relativo": vol_relativo,
        "vanna_charm": vanna_charm,
    }

    normalizados = {
        "gex_regime": 100 - abs(percentile_historico(hist_df, ticker, "gex_total", gex_total) - 50) * 2,
        "zero_gamma_dist": percentile_historico(
            hist_df, ticker, "dist_zero_gamma",
            abs(dist_zero_gamma) if not pd.isna(dist_zero_gamma) else np.nan
        ),
        "wall_space": (espacio_walls * 100) if not pd.isna(espacio_walls) else 50.0,
        "iv_rank": percentile_historico(hist_df, ticker, "iv_atm", iv_atm),
        "skew": 100 - percentile_historico(hist_df, ticker, "skew", abs(skew) if not pd.isna(skew) else np.nan),
        "expected_move": 100 - percentile_historico(hist_df, ticker, "expected_move", expected_move),
        "smart_money": percentile_historico(hist_df, ticker, "smart_money", smart_money),
        "volumen_relativo": (min(vol_relativo, 2.0) / 2.0 * 100) if not pd.isna(vol_relativo) else 50.0,
    }

    score = sum(normalizados[k] * PESOS[k] for k in PESOS)

    if score >= UMBRAL_ALTO:
        pct_entrada = 1.0
    elif score >= UMBRAL_MEDIO:
        pct_entrada = 0.4 + (score - UMBRAL_MEDIO) / (UMBRAL_ALTO - UMBRAL_MEDIO) * 0.4
    else:
        pct_entrada = max(score / UMBRAL_MEDIO * 0.3, 0.0)

    fila = {
        "fecha": pd.Timestamp.now(timezone.utc).normalize(),
        "ticker": ticker,
        **crudos,
        **{f"norm_{k}": v for k, v in normalizados.items()},
        "score_conviccion": score,
        "pct_entrada_sugerido": round(pct_entrada, 3),
    }
    return fila

# ---------------- VISUALIZACION ----------------

def _reco_corta(texto):
    if texto.startswith("DECISION FORZADA -> ENTRAR"):
        return "<b>FORZADO: ENTRAR</b>"
    if texto.startswith("DECISION FORZADA -> ESPERAR"):
        return "<b>FORZADO: ESPERAR</b>"
    if texto.startswith("MANTENER"):
        return "MANTENER"
    return texto

def graficar_resumen(resumen):
    if resumen.empty:
        return None

    df = resumen.sort_values("score_conviccion", ascending=True).reset_index(drop=True)
    despues = (df["pct_ya_invertido_previo"] + df["delta_sugerido_hoy_pct"]).clip(upper=100)
    pendiente = 100 - despues

    color_invertido = "#2a78d6"
    color_hoy = "#1baf7a"
    color_pendiente = "#e1e0d9"
    color_forzado_entrar = "#0ca30c"
    color_forzado_esperar = "#d03b3b"
    color_texto_normal = "#52514e"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["ticker"], x=df["pct_ya_invertido_previo"], name="Ya invertido", orientation="h",
        marker_color=color_invertido,
        customdata=df["recomendacion"],
        hovertemplate="<b>%{y}</b><br>Ya invertido: %{x:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=df["ticker"], x=df["delta_sugerido_hoy_pct"], name="Sugerido hoy", orientation="h",
        marker_color=color_hoy,
        customdata=df["recomendacion"],
        hovertemplate="<b>%{y}</b><br>Sugerido hoy: %{x:.1f}%<br>%{customdata}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=df["ticker"], x=pendiente, name="Pendiente", orientation="h",
        marker_color=color_pendiente,
        customdata=df["recomendacion"],
        hovertemplate="<b>%{y}</b><br>Pendiente: %{x:.1f}%<br>%{customdata}<extra></extra>",
    ))

    anotaciones = []
    for _, fila in df.iterrows():
        texto = fila["recomendacion"]
        if "DECISION FORZADA -> ENTRAR" in texto:
            color_txt = color_forzado_entrar
        elif "DECISION FORZADA -> ESPERAR" in texto:
            color_txt = color_forzado_esperar
        else:
            color_txt = color_texto_normal
        peso_obj = fila["peso_objetivo_pct"]
        peso_obj_txt = f"{peso_obj:.1f}%" if not pd.isna(peso_obj) else "?"
        linea1 = f"<b>{_reco_corta(texto)}</b>"
        linea2 = (
            f"<span style='font-size:10px;color:{color_texto_normal}'>"
            f"peso obj. {peso_obj_txt} · hoy +{fila['delta_puntos_portafolio']:.2f} pts portafolio</span>"
        )
        anotaciones.append(dict(
            x=1.02, y=fila["ticker"], xref="paper", yref="y",
            text=f"{linea1}<br>{linea2}",
            showarrow=False, xanchor="left", align="left",
            font=dict(size=12, color=color_txt),
        ))

    fig.update_layout(
        barmode="stack",
        title="Estado de entrada por activo — % del PESO OBJETIVO propio de cada activo (no del capital total)",
        xaxis=dict(title="% del peso objetivo del activo ya invertido", range=[0, 100], ticksuffix="%"),
        yaxis=dict(title=None, categoryorder="array", categoryarray=df["ticker"].tolist()),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="top", y=-0.12, x=0),
        margin=dict(r=340, l=80, t=70, b=90),
        annotations=anotaciones,
        height=max(420, 58 * len(df) + 170),
        width=1200,
    )
    return fig

# ---------------- EJECUCION PRINCIPAL ----------------

def correr_entry_signal():
    hist_df = cargar_historial()

    estado = cargar_estado()
    if estado is None:
        estado = inicializar_estado()
    else:
        estado = permitir_correccion_manual(estado)

    filas_nuevas = []
    for i, ticker in enumerate(TICKERS):
        try:
            fila = calcular_indicadores_ticker(ticker, hist_df)
            filas_nuevas.append(fila)
        except Exception as e:
            print(f"Error con {ticker}: {e}")
        if i < len(TICKERS) - 1:
            time.sleep(SEGUNDOS_ENTRE_LLAMADAS_STOCKS)

    df_nuevo = pd.DataFrame(filas_nuevas)

    hist_actualizado = pd.concat([hist_df, df_nuevo], ignore_index=True)
    # Si el mismo ticker ya tiene una fila con la fecha de hoy, se queda solo la mas reciente
    hist_actualizado["fecha_dia"] = pd.to_datetime(hist_actualizado["fecha"]).dt.date
    hist_actualizado = hist_actualizado.drop_duplicates(subset=["ticker", "fecha_dia"], keep="last")
    hist_actualizado = hist_actualizado.drop(columns=["fecha_dia"]).sort_values(["ticker", "fecha"])
    hist_actualizado.to_csv(HIST_PATH, index=False)

    # ---- Combina el score de hoy con el estado de entradas ya tomadas ----
    hoy = datetime.now(timezone.utc).date().isoformat()
    filas_estado = []
    for _, fila in df_nuevo.iterrows():
        ticker = fila["ticker"]
        info = estado.setdefault(
            ticker, {"pct_ya_invertido": 0.0, "dias_score_bajo": 0, "ultima_actualizacion": hoy}
        )
        pct_previo = info["pct_ya_invertido"]
        pct_objetivo_hoy = fila["pct_entrada_sugerido"]
        peso_objetivo = PESOS_OBJETIVO.get(ticker, np.nan)

        if fila["score_conviccion"] < UMBRAL_MEDIO:
            info["dias_score_bajo"] += 1
        else:
            info["dias_score_bajo"] = 0

        if pct_previo >= 0.999:
            delta = 0.0
            recomendacion = "COMPLETO (100%)"
        elif info["dias_score_bajo"] >= UMBRAL_DIAS_DECISION_FORZADA:
            decision, motivo = evaluar_decision_forzada(ticker, hist_actualizado)
            if decision == "ENTRAR":
                delta = 1.0 - pct_previo
                info["dias_score_bajo"] = 0
                recomendacion = f"DECISION FORZADA -> ENTRAR AL 100% ({motivo})"
            else:
                delta = 0.0
                recomendacion = f"DECISION FORZADA -> ESPERAR ({motivo})"
        else:
            delta = max(0.0, pct_objetivo_hoy - pct_previo)
            if delta > 0:
                recomendacion = "COMPLETAR A 100%" if pct_objetivo_hoy >= 1.0 else f"SUMAR +{delta*100:.1f}%"
            else:
                recomendacion = "MANTENER (ya en el nivel objetivo de hoy)"

        info["pct_ya_invertido"] = min(1.0, pct_previo + delta)
        info["ultima_actualizacion"] = hoy

        filas_estado.append({
            "ticker": ticker,
            "peso_objetivo_pct": round(peso_objetivo * 100, 2) if not pd.isna(peso_objetivo) else np.nan,
            "pct_ya_invertido_previo": round(pct_previo * 100, 1),
            "pct_objetivo_hoy": round(pct_objetivo_hoy * 100, 1),
            "delta_sugerido_hoy_pct": round(delta * 100, 1),
            "delta_puntos_portafolio": (
                round(delta * peso_objetivo * 100, 2) if not pd.isna(peso_objetivo) else np.nan
            ),
            "dias_score_bajo": info["dias_score_bajo"],
            "recomendacion": recomendacion,
        })

    guardar_estado(estado)

    df_estado = pd.DataFrame(filas_estado)
    resumen = df_nuevo[["ticker", "spot", "score_conviccion", "pct_entrada_sugerido"]].merge(
        df_estado, on="ticker"
    ).sort_values("score_conviccion", ascending=False)

    return resumen, hist_actualizado

resumen, historial = correr_entry_signal()
print(resumen.to_string(index=False))

fig_resumen = graficar_resumen(resumen)
if fig_resumen is not None:
    fig_resumen.show()
