"""
Motor de analisis de la cadena de opciones.

Incluye:
- Limpieza de datos poco fiables (clave: la IV de Yahoo trae mucha basura)
- Metricas objetivas del mercado de opciones
- Probabilidad implicita por strike (estilo "curva de probabilidad")
- Deteccion de strikes clave por open interest
- Motor de condiciones alcista/bajista

IMPORTANTE: esto mide lo que el mercado descuenta AHORA a traves de los
precios de las opciones. No es una prediccion ni una recomendacion.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


# =================================================================
# LIMPIEZA DE DATOS  (lo mas importante para que la IV tenga sentido)
# =================================================================
IV_MINIMA = 0.01    # 1%   - por debajo casi siempre es un placeholder de Yahoo
IV_MAXIMA = 3.00    # 300% - por encima suele ser ruido de opciones sin liquidez


def limpiar_cadena(tabla, spot, estricto=True):
    """Filtra las filas de una cadena de opciones que no son fiables.

    Yahoo Finance devuelve la volatilidad implicita para TODAS las
    opciones, incluidas las que llevan dias sin negociarse o que tienen
    una horquilla bid/ask enorme. En esos casos la IV que calcula es
    basura y puede salir 200%, 400% o 0.001%.

    Filtros aplicados:
      1. IV dentro de un rango razonable (1% - 300%)
      2. Que la opcion tenga algo de vida: volumen o interes abierto
      3. Que tenga precio y una horquilla bid/ask coherente
      4. (estricto) Se descartan opciones muy dentro de dinero (ITM),
         porque su IV es la menos fiable de todas
    """
    if tabla is None or tabla.empty:
        return tabla

    df = tabla.copy()

    if "impliedVolatility" not in df.columns:
        return df

    df = df[df["impliedVolatility"].notna()]
    df = df[(df["impliedVolatility"] >= IV_MINIMA) & (df["impliedVolatility"] <= IV_MAXIMA)]

    if not estricto:
        return df

    # que la opcion se haya negociado o tenga posiciones abiertas
    volumen = df["volume"].fillna(0) if "volume" in df.columns else 0
    interes = df["openInterest"].fillna(0) if "openInterest" in df.columns else 0
    df = df[(volumen > 0) | (interes > 0)]

    # que tenga precio real
    if "lastPrice" in df.columns:
        df = df[df["lastPrice"].fillna(0) > 0]

    # horquilla bid/ask no absurda (mas del 200% del punto medio = ilíquida)
    if "bid" in df.columns and "ask" in df.columns:
        bid = df["bid"].fillna(0)
        ask = df["ask"].fillna(0)
        medio = (bid + ask) / 2
        horquilla_ok = (medio <= 0) | ((ask - bid) / medio.replace(0, np.nan) < 2.0)
        df = df[horquilla_ok.fillna(True)]

    return df


def calidad_datos(original, limpio):
    """Devuelve que porcentaje de filas ha sobrevivido al filtro."""
    if original is None or len(original) == 0:
        return 0.0, 0, 0
    return len(limpio) / len(original) * 100, len(limpio), len(original)


# =================================================================
# BLACK-SCHOLES: probabilidad implicita y delta
# =================================================================
def _d1_d2(spot, strike, iv, dias, tasa=0.045):
    """Componentes de Black-Scholes. iv en decimal, dias en dias naturales."""
    if spot <= 0 or strike <= 0 or iv <= 0 or dias <= 0:
        return None, None
    t = dias / 365.0
    sigma_raiz_t = iv * np.sqrt(t)
    if sigma_raiz_t == 0:
        return None, None
    d1 = (np.log(spot / strike) + (tasa + 0.5 * iv ** 2) * t) / sigma_raiz_t
    d2 = d1 - sigma_raiz_t
    return d1, d2


def delta_call(spot, strike, iv, dias, tasa=0.045):
    """Delta de una call. Aproxima la sensibilidad al precio del subyacente."""
    d1, _ = _d1_d2(spot, strike, iv, dias, tasa)
    return None if d1 is None else float(norm.cdf(d1))


def prob_acabar_encima(spot, strike, iv, dias, tasa=0.045):
    """Probabilidad implicita de que el precio acabe POR ENCIMA del strike.

    Es N(d2) de Black-Scholes: la probabilidad neutral al riesgo de que
    la call acabe dentro de dinero. Esto es lo mas parecido a la "curva
    de probabilidad" que muestran los mercados de prediccion.

    Ojo: es probabilidad NEUTRAL AL RIESGO, no la probabilidad real del
    mundo. Incorpora la prima que la gente paga por cubrirse, asi que
    tiende a sobreestimar la probabilidad de caidas fuertes.
    """
    _, d2 = _d1_d2(spot, strike, iv, dias, tasa)
    return None if d2 is None else float(norm.cdf(d2))


def curva_probabilidad(calls_limpio, spot, dias, tasa=0.045):
    """Construye la curva de probabilidad implicita para un vencimiento.

    Devuelve un DataFrame con strike, probabilidad de acabar por encima,
    probabilidad de acabar por debajo, y delta.
    """
    if calls_limpio is None or calls_limpio.empty:
        return pd.DataFrame()

    filas = []
    for _, f in calls_limpio.iterrows():
        strike = f.get("strike")
        iv = f.get("impliedVolatility")
        if strike is None or iv is None:
            continue
        p_encima = prob_acabar_encima(spot, strike, float(iv), dias, tasa)
        d = delta_call(spot, strike, float(iv), dias, tasa)
        if p_encima is None:
            continue
        filas.append({
            "strike": float(strike),
            "prob_encima": p_encima * 100,
            "prob_debajo": (1 - p_encima) * 100,
            "delta": d,
            "iv": float(iv) * 100,
        })

    if not filas:
        return pd.DataFrame()

    return pd.DataFrame(filas).sort_values("strike").reset_index(drop=True)


def rango_probable(curva, confianza=0.68):
    """Rango de precios donde el mercado ve que acabara, a X% de confianza.

    Con 68% equivale aproximadamente a una desviacion tipica.
    """
    if curva.empty:
        return None, None

    cola = (1 - confianza) / 2 * 100  # ej: 16% por cada lado con 68%

    # strike donde la probabilidad de acabar encima es (100 - cola)
    inferior = curva.iloc[(curva["prob_encima"] - (100 - cola)).abs().argsort()[:1]]
    superior = curva.iloc[(curva["prob_encima"] - cola).abs().argsort()[:1]]

    if inferior.empty or superior.empty:
        return None, None

    return float(inferior["strike"].iloc[0]), float(superior["strike"].iloc[0])


# =================================================================
# STRIKES CLAVE (donde esta el dinero de verdad)
# =================================================================
def strikes_clave(calls, puts, top=3):
    """Los strikes con mas interes abierto, que actuan como referencias.

    Mucho open interest en un strike significa que hay muchas posiciones
    abiertas ahi. Suelen funcionar como zonas de soporte/resistencia
    porque los creadores de mercado ajustan sus coberturas alrededor.
    """
    resultado = {"calls": [], "puts": []}

    if calls is not None and not calls.empty and "openInterest" in calls.columns:
        top_c = calls.nlargest(top, "openInterest")[["strike", "openInterest", "volume"]]
        resultado["calls"] = top_c.to_dict("records")

    if puts is not None and not puts.empty and "openInterest" in puts.columns:
        top_p = puts.nlargest(top, "openInterest")[["strike", "openInterest", "volume"]]
        resultado["puts"] = top_p.to_dict("records")

    return resultado


# =================================================================
# METRICAS
# =================================================================
def put_call_ratio(calls, puts, columna):
    """Ratio put/call de volumen o de open interest."""
    if calls is None or puts is None or columna not in calls.columns:
        return None
    total_calls = calls[columna].fillna(0).sum()
    total_puts = puts[columna].fillna(0).sum()
    if total_calls <= 0:
        return None
    return total_puts / total_calls


def atm_iv(calls, puts, spot):
    """IV del strike mas cercano al precio actual, promediando call y put.

    Promediar las dos es mas robusto que fiarse de una sola, porque si
    una de las dos tiene un precio raro el promedio lo suaviza.
    """
    valores = []
    for tabla in (calls, puts):
        if tabla is None or tabla.empty:
            continue
        idx = (tabla["strike"] - spot).abs().idxmin()
        iv = tabla.loc[idx, "impliedVolatility"]
        if pd.notna(iv):
            valores.append(float(iv) * 100)
    return float(np.mean(valores)) if valores else None


def skew_25(calls, puts, spot):
    """Skew: IV de puts OTM (90% del spot) menos IV de calls OTM (110%)."""
    puts_otm = puts[puts["strike"] <= spot] if puts is not None else pd.DataFrame()
    calls_otm = calls[calls["strike"] >= spot] if calls is not None else pd.DataFrame()

    if puts_otm.empty or calls_otm.empty:
        return None

    idx_p = (puts_otm["strike"] - spot * 0.90).abs().idxmin()
    idx_c = (calls_otm["strike"] - spot * 1.10).abs().idxmin()

    iv_p = puts_otm.loc[idx_p, "impliedVolatility"]
    iv_c = calls_otm.loc[idx_c, "impliedVolatility"]

    if pd.isna(iv_p) or pd.isna(iv_c):
        return None

    return (float(iv_p) - float(iv_c)) * 100


def estructura_temporal(df_superficie, spot):
    """IV del vencimiento cercano vs uno lejano (>=45 dias si existe)."""
    if df_superficie is None or df_superficie.empty:
        return None, None, None

    dias_unicos = sorted(df_superficie["dias"].unique())
    if len(dias_unicos) < 2:
        return None, None, None

    cercano = dias_unicos[0]
    lejanos = [d for d in dias_unicos if d >= 45]
    lejano = lejanos[0] if lejanos else dias_unicos[-1]

    def iv_atm_en(dias):
        sub = df_superficie[df_superficie["dias"] == dias]
        if sub.empty:
            return None
        idx = (sub["strike"] - spot).abs().idxmin()
        return float(sub.loc[idx, "iv"])

    iv_c, iv_l = iv_atm_en(cercano), iv_atm_en(lejano)
    if iv_c is None or iv_l is None:
        return None, None, None

    return iv_c, iv_l, iv_c - iv_l


def volatilidad_realizada(historico, dias=30):
    """Volatilidad realizada anualizada de los ultimos N dias de cotizacion."""
    if historico is None or len(historico) < dias + 1:
        return None
    retornos = np.log(historico["Close"] / historico["Close"].shift(1)).dropna()
    if len(retornos) < dias:
        return None
    return float(retornos.tail(dias).std() * np.sqrt(252) * 100)


def movimiento_esperado(calls, puts, spot):
    """Movimiento implicito segun el precio del straddle ATM."""
    if calls is None or puts is None or calls.empty or puts.empty:
        return None, None

    idx_c = (calls["strike"] - spot).abs().idxmin()
    strike = calls.loc[idx_c, "strike"]

    put_match = puts[puts["strike"] == strike]
    if put_match.empty:
        return None, None

    precio_call = calls.loc[idx_c, "lastPrice"]
    precio_put = put_match.iloc[0]["lastPrice"]

    if pd.isna(precio_call) or pd.isna(precio_put):
        return None, None

    coste = float(precio_call) + float(precio_put)
    return coste, (coste / spot) * 100 if spot else None


def max_pain(calls, puts):
    """Strike donde el conjunto de opciones tiene menor valor intrinseco."""
    if calls is None or puts is None or calls.empty or puts.empty:
        return None

    strikes = sorted(set(calls["strike"]).union(set(puts["strike"])))
    if not strikes:
        return None

    mejor, menor = None, None
    for s in strikes:
        dolor_c = ((s - calls["strike"]).clip(lower=0) * calls["openInterest"].fillna(0)).sum()
        dolor_p = ((puts["strike"] - s).clip(lower=0) * puts["openInterest"].fillna(0)).sum()
        total = dolor_c + dolor_p
        if menor is None or total < menor:
            menor, mejor = total, s
    return mejor


# =================================================================
# MOTOR DE CONDICIONES
# =================================================================
def evaluar_condiciones(m):
    """Traduce las metricas a condiciones con sesgo y horizonte."""
    condiciones = []

    pcr_vol = m.get("pcr_volumen")
    if pcr_vol is not None:
        if pcr_vol > 1.2:
            sesgo, cumple = "bajista", True
            expl = "Mas volumen en puts que en calls: flujo defensivo o apuestas a la baja."
        elif pcr_vol < 0.7:
            sesgo, cumple = "alcista", True
            expl = "Volumen cargado en calls: apetito por subidas."
        else:
            sesgo, cumple = "neutral", False
            expl = "Volumen equilibrado, sin sesgo claro."
        condiciones.append({
            "nombre": "Put/Call Ratio (volumen)", "valor": f"{pcr_vol:.2f}",
            "se_cumple": cumple, "sesgo": sesgo, "horizonte": "corto", "explicacion": expl,
        })

    pcr_oi = m.get("pcr_oi")
    if pcr_oi is not None:
        if pcr_oi > 1.2:
            sesgo, cumple, expl = "bajista", True, "Posicionamiento acumulado cargado en puts."
        elif pcr_oi < 0.7:
            sesgo, cumple, expl = "alcista", True, "Posicionamiento acumulado cargado en calls."
        else:
            sesgo, cumple, expl = "neutral", False, "Posicionamiento repartido, sin extremo."
        condiciones.append({
            "nombre": "Put/Call Ratio (open interest)", "valor": f"{pcr_oi:.2f}",
            "se_cumple": cumple, "sesgo": sesgo, "horizonte": "medio", "explicacion": expl,
        })

    skew = m.get("skew")
    if skew is not None:
        if skew > 5:
            sesgo, cumple = "bajista", True
            expl = "Los puts OTM cotizan muy por encima de los calls: se paga caro cubrirse de caidas."
        elif skew < -2:
            sesgo, cumple = "alcista", True
            expl = "Los calls OTM cotizan por encima de los puts: se persigue el movimiento al alza."
        else:
            sesgo, cumple = "neutral", False
            expl = "Skew dentro de lo normal."
        condiciones.append({
            "nombre": "Skew (IV put 90% - IV call 110%)", "valor": f"{skew:+.1f} pts",
            "se_cumple": cumple, "sesgo": sesgo, "horizonte": "medio", "explicacion": expl,
        })

    pend = m.get("pendiente_temporal")
    if pend is not None:
        if pend > 2:
            sesgo, cumple = "bajista", True
            expl = "Backwardation: el vencimiento cercano paga mas IV que el lejano. Estres o evento inminente."
        elif pend < -2:
            sesgo, cumple = "alcista", True
            expl = "Contango normal: calma a corto plazo."
        else:
            sesgo, cumple = "neutral", False
            expl = "Curva de IV plana entre vencimientos."
        condiciones.append({
            "nombre": "Estructura temporal (IV cercana - lejana)", "valor": f"{pend:+.1f} pts",
            "se_cumple": cumple, "sesgo": sesgo, "horizonte": "corto", "explicacion": expl,
        })

    iv, rv = m.get("iv_atm"), m.get("vol_realizada")
    if iv is not None and rv is not None and rv > 0:
        prima = iv - rv
        if prima > 5:
            sesgo, cumple = "neutral", True
            expl = ("La IV supera claramente al movimiento real reciente: opciones caras, "
                    "el mercado descuenta mas movimiento del que ha habido.")
        elif prima < -5:
            sesgo, cumple = "neutral", True
            expl = ("La IV esta por debajo del movimiento real: opciones baratas respecto "
                    "a lo que el activo se esta moviendo.")
        else:
            sesgo, cumple = "neutral", False
            expl = "IV alineada con la volatilidad realizada."
        condiciones.append({
            "nombre": "Prima de IV sobre vol. realizada 30d",
            "valor": f"{prima:+.1f} pts (IV {iv:.1f}% vs RV {rv:.1f}%)",
            "se_cumple": cumple, "sesgo": sesgo, "horizonte": "corto", "explicacion": expl,
        })

    mp, spot = m.get("max_pain"), m.get("spot")
    if mp is not None and spot:
        dif = (mp - spot) / spot * 100
        if dif < -2:
            sesgo, cumple, expl = "bajista", True, "El max pain queda por debajo del precio actual."
        elif dif > 2:
            sesgo, cumple, expl = "alcista", True, "El max pain queda por encima del precio actual."
        else:
            sesgo, cumple, expl = "neutral", False, "Max pain practicamente en el precio actual."
        condiciones.append({
            "nombre": "Max pain vs precio actual", "valor": f"{mp:.2f} ({dif:+.1f}%)",
            "se_cumple": cumple, "sesgo": sesgo, "horizonte": "corto",
            "explicacion": expl + " Heuristica de referencia, no una ley del mercado.",
        })

    return condiciones


def lectura_agregada(condiciones, horizonte):
    """Cuenta condiciones alcistas vs bajistas para un horizonte dado."""
    rel = [c for c in condiciones if c["horizonte"] == horizonte and c["se_cumple"]]
    alc = [c for c in rel if c["sesgo"] == "alcista"]
    baj = [c for c in rel if c["sesgo"] == "bajista"]
    n_a, n_b = len(alc), len(baj)

    if n_a == 0 and n_b == 0:
        etiqueta, color = "SIN SESGO CLARO", "neutral"
    elif n_a > n_b:
        etiqueta = "SESGO ALCISTA" if n_a - n_b >= 2 else "LIGERO SESGO ALCISTA"
        color = "alcista"
    elif n_b > n_a:
        etiqueta = "SESGO BAJISTA" if n_b - n_a >= 2 else "LIGERO SESGO BAJISTA"
        color = "bajista"
    else:
        etiqueta, color = "SENALES CONTRADICTORIAS", "neutral"

    return {"etiqueta": etiqueta, "color": color, "n_alcistas": n_a, "n_bajistas": n_b}
