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

# Peso objetivo de cada activo dentro del portafolio total 
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

# Duracion del ciclo de entrada en dias habiles. En el ultimo dia se decide en
# firme la fraccion no invertida: comprar el 100% restante o consolidarla en cash.
DIAS_CICLO = 5

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

# ---------------- ESTADO DEL CICLO Y DE LAS ENTRADAS ----------------

def _activo_vacio():
    return {"pct_ya_invertido": 0.0, "pct_cash_consolidado": 0.0, "decision_final": None}

def _estado_vacio():
    return {
        "ciclo": 1,
        "dia_ciclo": 1,
        "ciclo_cerrado": False,
        "ultima_actualizacion": None,
        "activos": {ticker: _activo_vacio() for ticker in TICKERS},
    }

# El dia del ciclo vive una sola vez a nivel global (antes cada activo llevaba su
# propio contador dias_score_bajo). Los estados en formato antiguo se migran aqui.
def normalizar_estado(estado):
    if not estado:
        return _estado_vacio()
    if "activos" not in estado:
        antiguo = estado
        estado = _estado_vacio()
        for ticker, info in antiguo.items():
            if isinstance(info, dict) and "pct_ya_invertido" in info:
                estado["activos"].setdefault(ticker, _activo_vacio())
                estado["activos"][ticker]["pct_ya_invertido"] = float(info["pct_ya_invertido"])
    for ticker in TICKERS:
        estado["activos"].setdefault(ticker, _activo_vacio())
    return estado

def cargar_estado():
    if not os.path.exists(STATE_PATH):
        return _estado_vacio()
    with open(STATE_PATH, "r") as f:
        return normalizar_estado(json.load(f))

def guardar_estado(estado):
    with open(STATE_PATH, "w") as f:
        json.dump(estado, f, indent=2)

# ---------------- FLUJO INTERACTIVO DE INICIO ----------------

def pedir_pct(mensaje, default=0.0):
    while True:
        crudo = input(mensaje).strip().replace("%", "")
        if crudo == "":
            return default
        try:
            valor = float(crudo)
        except ValueError:
            print("  Ingresa un numero valido (ej: 40 para 40%).")
            continue
        if valor < 0 or valor > 100:
            print("  El porcentaje debe estar entre 0 y 100.")
            continue
        return valor / 100

def pedir_dia_ciclo(sugerido):
    while True:
        crudo = input(
            f"\n¿En que dia del ciclo de {DIAS_CICLO} dias habiles se encuentra la estrategia? "
            f"[1-{DIAS_CICLO}] (Enter = {sugerido}): "
        ).strip()
        if crudo == "":
            return sugerido
        try:
            dia = int(crudo)
        except ValueError:
            print(f"  Ingresa un entero entre 1 y {DIAS_CICLO}.")
            continue
        if not 1 <= dia <= DIAS_CICLO:
            print(f"  El dia debe estar entre 1 y {DIAS_CICLO}.")
            continue
        return dia

def pedir_pesos_actuales(estado):
    print(
        "\n% YA INVERTIDO en cada activo respecto a SU peso objetivo (0-100)."
        "\nEnter = mantener el valor guardado que se muestra entre parentesis.\n"
    )
    for ticker in TICKERS:
        activo = estado["activos"][ticker]
        peso = PESOS_OBJETIVO.get(ticker, np.nan)
        actual = activo["pct_ya_invertido"]
        activo["pct_ya_invertido"] = pedir_pct(
            f"  {ticker} (peso objetivo {peso*100:.2f}% del portafolio | guardado {actual*100:.0f}%): ",
            default=actual,
        )
    return estado

def flujo_inicio(estado):
    hoy = datetime.now(timezone.utc).date().isoformat()

    if estado["ultima_actualizacion"] is None or estado["ciclo_cerrado"]:
        sugerido = 1
    else:
        sugerido = min(estado["dia_ciclo"] + 1, DIAS_CICLO)

    dia = pedir_dia_ciclo(sugerido)

    # Retroceder en el numero de dia significa que arranco un ciclo nuevo:
    # se limpian las decisiones de cierre y el cash consolidado del anterior.
    if estado["ultima_actualizacion"] is not None and dia < estado["dia_ciclo"]:
        estado["ciclo"] += 1
        estado["ciclo_cerrado"] = False
        for activo in estado["activos"].values():
            activo["pct_cash_consolidado"] = 0.0
            activo["decision_final"] = None
        print(f"\nArranca el ciclo {estado['ciclo']}: se limpian las decisiones del ciclo anterior.")

    estado["dia_ciclo"] = dia
    estado["ultima_actualizacion"] = hoy

    pedir_pesos_actuales(estado)

    if dia >= DIAS_CICLO:
        print(
            f"\nDIA {DIAS_CICLO} (ultimo del ciclo): hoy se decide en firme la fraccion no invertida."
            "\nSegun el flujo de opciones acumulado se compra el 100% restante o se consolida en cash.\n"
        )
    else:
        print(f"\nDia {dia} de {DIAS_CICLO}: entrada por goteo segun el score de conviccion de hoy.\n")
    return estado

# ---------------- DECISION DE CIERRE (DIA 5) ----------------

def evaluar_flujo_opciones(ticker, hist_df):
    hist_ticker = hist_df[hist_df["ticker"] == ticker].sort_values("fecha").tail(DIAS_CICLO)
    if len(hist_ticker) < 3:
        return "CASH", "historial insuficiente para forzar la compra del restante"

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
            detalles.append("acortamiento a zero-gamma (mayor estabilidad esperada)")
        else:
            detalles.append("precio alejandose del zero-gamma")

    if señales_totales == 0:
        return "CASH", "sin señales de flujo de opciones suficientes"

    decision = "ENTRAR" if señales_favorables / señales_totales >= 0.5 else "CASH"
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

COLORES_ACCION = {
    "CIERRE: ENTRAR": "#0ca30c",
    "CIERRE: CASH": "#d03b3b",
    "COMPLETO": "#2a78d6",
}
COLOR_TEXTO_NORMAL = "#52514e"

def graficar_resumen(resumen):
    if resumen.empty:
        return None

    df = resumen.sort_values("score_conviccion", ascending=True).reset_index(drop=True)
    dia = int(df["dia_ciclo"].iloc[0])
    ciclo = int(df["ciclo"].iloc[0])
    es_ultimo_dia = dia >= DIAS_CICLO

    cash = df["cash_definitivo_pct"]
    pendiente = (100 - df["pct_invertido_final_pct"] - cash).clip(lower=0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["ticker"], x=df["pct_ya_invertido_previo"], name="Ya invertido", orientation="h",
        marker_color="#2a78d6",
        hovertemplate="<b>%{y}</b><br>Ya invertido: %{x:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=df["ticker"], x=df["delta_sugerido_hoy_pct"], name="Sugerido hoy", orientation="h",
        marker_color="#1baf7a",
        customdata=df["recomendacion"],
        hovertemplate="<b>%{y}</b><br>Sugerido hoy: %{x:.1f}%<br>%{customdata}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=df["ticker"], x=pendiente, name="Pendiente (goteo)", orientation="h",
        marker_color="#e1e0d9",
        customdata=df["recomendacion"],
        hovertemplate="<b>%{y}</b><br>Pendiente: %{x:.1f}%<br>%{customdata}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=df["ticker"], x=cash, name=f"Cash definitivo (cierre dia {DIAS_CICLO})", orientation="h",
        marker_color="#f0b7b7",
        marker_line=dict(color="#d03b3b", width=1),
        customdata=df["recomendacion"],
        hovertemplate="<b>%{y}</b><br>Cash definitivo: %{x:.1f}%<br>%{customdata}<extra></extra>",
    ))

    anotaciones = []
    for _, fila in df.iterrows():
        accion = fila["accion"]
        color_txt = COLORES_ACCION.get(accion, COLOR_TEXTO_NORMAL)
        peso_obj = fila["peso_objetivo_pct"]
        peso_obj_txt = f"{peso_obj:.1f}%" if not pd.isna(peso_obj) else "?"
        if accion == "CIERRE: CASH":
            detalle = f"{fila['cash_definitivo_pct']:.0f}% a cash"
        elif fila["delta_sugerido_hoy_pct"] > 0:
            detalle = f"+{fila['delta_sugerido_hoy_pct']:.0f}%"
        else:
            detalle = "sin cambios"
        linea1 = f"<b>{accion} · {detalle}</b>"
        linea2 = (
            f"<span style='font-size:10px;color:{COLOR_TEXTO_NORMAL}'>"
            f"peso obj. {peso_obj_txt} · hoy +{fila['delta_puntos_portafolio']:.2f} pts portafolio</span>"
        )
        anotaciones.append(dict(
            x=1.02, y=fila["ticker"], xref="paper", yref="y",
            text=f"{linea1}<br>{linea2}",
            showarrow=False, xanchor="left", align="left",
            font=dict(size=12, color=color_txt),
        ))

    subtitulo = (
        "cierre del ciclo: la fraccion no invertida se compra al 100% o queda en cash"
        if es_ultimo_dia else "entrada por goteo segun el score de conviccion"
    )
    fig.update_layout(
        barmode="stack",
        title=(
            f"CICLO {ciclo} · DIA {dia} DE {DIAS_CICLO} — {subtitulo}"
            "<br><span style='font-size:12px'>% del PESO OBJETIVO propio de cada activo (no del capital total)</span>"
        ),
        xaxis=dict(title="% del peso objetivo del activo", range=[0, 100], ticksuffix="%"),
        yaxis=dict(title=None, categoryorder="array", categoryarray=df["ticker"].tolist()),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="top", y=-0.12, x=0),
        margin=dict(r=340, l=80, t=90, b=90),
        annotations=anotaciones,
        height=max(420, 58 * len(df) + 170),
        width=1200,
    )
    return fig

# ---------------- EJECUCION PRINCIPAL ----------------

def correr_entry_signal():
    hist_df = cargar_historial()

    estado = flujo_inicio(cargar_estado())
    dia_ciclo = estado["dia_ciclo"]
    ciclo = estado["ciclo"]
    es_ultimo_dia = dia_ciclo >= DIAS_CICLO

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
    if df_nuevo.empty:
        guardar_estado(estado)
        return pd.DataFrame(), hist_df

    hist_actualizado = pd.concat([hist_df, df_nuevo], ignore_index=True)
    # Si el mismo ticker ya tiene una fila con la fecha de hoy, se queda solo la mas reciente
    hist_actualizado["fecha_dia"] = pd.to_datetime(hist_actualizado["fecha"]).dt.date
    hist_actualizado = hist_actualizado.drop_duplicates(subset=["ticker", "fecha_dia"], keep="last")
    hist_actualizado = hist_actualizado.drop(columns=["fecha_dia"]).sort_values(["ticker", "fecha"])
    hist_actualizado.to_csv(HIST_PATH, index=False)

    # ---- Combina el score de hoy con el estado de entradas ya tomadas ----
    filas_estado = []
    for _, fila in df_nuevo.iterrows():
        ticker = fila["ticker"]
        activo = estado["activos"].setdefault(ticker, _activo_vacio())
        pct_previo = activo["pct_ya_invertido"]
        pct_objetivo_hoy = fila["pct_entrada_sugerido"]
        peso_objetivo = PESOS_OBJETIVO.get(ticker, np.nan)
        pct_cash = 0.0

        if pct_previo >= 0.999:
            delta = 0.0
            accion = "COMPLETO"
            recomendacion = "COMPLETO (100% del peso objetivo)"
            activo["decision_final"] = "ENTRAR" if es_ultimo_dia else activo["decision_final"]
        elif es_ultimo_dia:
            # Dia 5: no hay goteo adicional, se cierra el ciclo en firme.
            decision, motivo = evaluar_flujo_opciones(ticker, hist_actualizado)
            if decision == "ENTRAR":
                delta = 1.0 - pct_previo
                accion = "CIERRE: ENTRAR"
                recomendacion = f"CIERRE DIA {DIAS_CICLO} -> COMPRAR EL {delta*100:.1f}% RESTANTE ({motivo})"
            else:
                delta = 0.0
                pct_cash = 1.0 - pct_previo
                accion = "CIERRE: CASH"
                recomendacion = f"CIERRE DIA {DIAS_CICLO} -> DEJAR {pct_cash*100:.1f}% EN CASH ({motivo})"
            activo["decision_final"] = decision
        else:
            delta = max(0.0, pct_objetivo_hoy - pct_previo)
            if delta > 0:
                accion = "SUMAR"
                recomendacion = "COMPLETAR A 100%" if pct_objetivo_hoy >= 1.0 else f"SUMAR +{delta*100:.1f}%"
            else:
                accion = "MANTENER"
                recomendacion = "MANTENER (ya en el nivel objetivo de hoy)"

        pct_final = min(1.0, pct_previo + delta)
        activo["pct_ya_invertido"] = pct_final
        activo["pct_cash_consolidado"] = pct_cash

        filas_estado.append({
            "ciclo": ciclo,
            "dia_ciclo": dia_ciclo,
            "ticker": ticker,
            "peso_objetivo_pct": round(peso_objetivo * 100, 2) if not pd.isna(peso_objetivo) else np.nan,
            "pct_ya_invertido_previo": round(pct_previo * 100, 1),
            "pct_objetivo_hoy": round(pct_objetivo_hoy * 100, 1),
            "delta_sugerido_hoy_pct": round(delta * 100, 1),
            "pct_invertido_final_pct": round(pct_final * 100, 1),
            "cash_definitivo_pct": round(pct_cash * 100, 1),
            "delta_puntos_portafolio": (
                round(delta * peso_objetivo * 100, 2) if not pd.isna(peso_objetivo) else np.nan
            ),
            "accion": accion,
            "recomendacion": recomendacion,
        })

    estado["ciclo_cerrado"] = es_ultimo_dia
    guardar_estado(estado)

    df_estado = pd.DataFrame(filas_estado)
    resumen = df_nuevo[["ticker", "spot", "score_conviccion", "pct_entrada_sugerido"]].merge(
        df_estado, on="ticker"
    ).sort_values("score_conviccion", ascending=False)

    return resumen, hist_actualizado

def imprimir_resumen(resumen):
    if resumen.empty:
        print("\nSin datos: ningun ticker devolvio indicadores.")
        return

    dia = int(resumen["dia_ciclo"].iloc[0])
    ciclo = int(resumen["ciclo"].iloc[0])
    print(f"\n=== CICLO {ciclo} · DIA {dia} DE {DIAS_CICLO} ===")

    columnas = [
        "dia_ciclo", "ticker", "spot", "score_conviccion", "peso_objetivo_pct",
        "pct_ya_invertido_previo", "delta_sugerido_hoy_pct", "pct_invertido_final_pct",
        "cash_definitivo_pct", "delta_puntos_portafolio", "accion", "recomendacion",
    ]
    print(resumen[columnas].to_string(index=False))

    pts_invertidos = (resumen["pct_invertido_final_pct"] / 100 * resumen["peso_objetivo_pct"]).sum()
    pts_cash = (resumen["cash_definitivo_pct"] / 100 * resumen["peso_objetivo_pct"]).sum()
    if dia >= DIAS_CICLO:
        entrar = resumen[resumen["accion"] == "CIERRE: ENTRAR"]["ticker"].tolist()
        a_cash = resumen[resumen["accion"] == "CIERRE: CASH"]["ticker"].tolist()
        print(f"\nCIERRE DE CICLO (dia {DIAS_CICLO}) — decision definitiva, sin goteo adicional:")
        print(f"  Compra del restante al 100%: {', '.join(entrar) if entrar else '(ninguno)'}")
        print(f"  Consolidado en cash / no entrar: {', '.join(a_cash) if a_cash else '(ninguno)'}")
        print(f"  Asignacion final invertida: {pts_invertidos:.2f} pts de portafolio")
        print(f"  Reserva final en cash: {pts_cash:.2f} pts de portafolio")
    else:
        pendiente = resumen["peso_objetivo_pct"].sum() - pts_invertidos
        print(f"\nDia {dia} de {DIAS_CICLO} — quedan {DIAS_CICLO - dia} dia(s) de goteo antes del cierre forzado.")
        print(f"  Invertido tras hoy: {pts_invertidos:.2f} pts de portafolio")
        print(f"  Pendiente por asignar: {pendiente:.2f} pts de portafolio")

resumen, historial = correr_entry_signal()
imprimir_resumen(resumen)

fig_resumen = graficar_resumen(resumen)
if fig_resumen is not None:
    fig_resumen.show()
