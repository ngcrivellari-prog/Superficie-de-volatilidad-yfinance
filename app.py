"""
OPTIONS TERMINAL - Superficie de volatilidad, probabilidades y analisis

Ejecutar con:
    streamlit run app.py

AVISO: datos de Yahoo Finance con retraso tipico de 15-20 minutos.
El analisis mide lo que el mercado descuenta ahora; no es prediccion
ni recomendacion de inversion.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from scipy.interpolate import griddata

import analisis as an

st.set_page_config(page_title="OPTIONS TERMINAL", layout="wide", page_icon="📡")

# =================================================================
# ESTILO TERMINAL
# =================================================================
st.markdown("""
<style>
    .stApp, [data-testid="stSidebar"] { background-color: #000000; }
    html, body, [class*="css"], .stMarkdown, p, span, div, label {
        font-family: 'Consolas','Courier New',monospace !important; color: #FF9E00 !important;
    }
    h1,h2,h3,h4 { color:#FFB84D !important; font-family:'Consolas',monospace !important; letter-spacing:1px; }
    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div,
    .stNumberInput input {
        background-color:#0D0D0D !important; color:#FF9E00 !important;
        border:1px solid #FF9E00 !important; font-family:'Consolas',monospace !important;
    }
    .stButton button {
        background-color:#1A1A1A !important; color:#FF9E00 !important;
        border:1px solid #FF9E00 !important; font-family:'Consolas',monospace !important;
        width:100%; font-size:12px !important;
    }
    .stButton button:hover { background-color:#FF9E00 !important; color:#000000 !important; }
    [data-testid="stMetricValue"] { color:#FFB84D !important; font-family:'Consolas',monospace !important; }
    [data-testid="stMetricLabel"] { color:#B36F00 !important; }
    hr { border-color:#333333 !important; }
    .terminal-header {
        background-color:#1A1A1A; border:1px solid #FF9E00; padding:10px 15px; margin-bottom:15px;
    }
    .gris  { color:#888888 !important; }
    .caja-senal { border:2px solid; padding:12px 18px; margin:8px 0; background-color:#0D0D0D; }
    .caja-info { border-left:3px solid #FF9E00; padding:8px 14px; margin:6px 0; background-color:#0D0D0D; }
    .alerta { border:1px solid #FF3B3B; padding:10px 14px; margin:8px 0; background-color:#1A0000; }
</style>
""", unsafe_allow_html=True)


# =================================================================
# BUSCADOR
# =================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def buscar_tickers(texto):
    if not texto:
        return []
    try:
        r = requests.get(
            "https://query2.finance.yahoo.com/v1/finance/search",
            params={"q": texto, "quotesCount": 12, "newsCount": 0},
            headers={"User-Agent": "Mozilla/5.0"}, timeout=10,
        )
        r.raise_for_status()
        out = []
        for q in r.json().get("quotes", []):
            s = q.get("symbol")
            if s:
                nombre = (q.get("shortname") or q.get("longname") or "")[:36]
                out.append({"etiqueta": f"{s:<8} | {nombre:<36} | {q.get('quoteType','')} {q.get('exchange','')}"})
        return out
    except Exception:
        return []


st.sidebar.markdown("### 🔍 BUSCADOR")
texto_busqueda = st.sidebar.text_input(
    "Escribe letras o nombre", value="", placeholder="ej: N, NVDA, apple, gold...",
)

ticker = None
if texto_busqueda:
    coincidencias = buscar_tickers(texto_busqueda)
    if coincidencias:
        sel = st.sidebar.selectbox(f"{len(coincidencias)} coincidencias",
                                   options=[c["etiqueta"] for c in coincidencias])
        ticker = sel.split("|")[0].strip()
    else:
        st.sidebar.warning("Sin coincidencias.")
else:
    ticker = st.sidebar.text_input("O ticker directo", value="AAPL").upper().strip()

st.sidebar.markdown("---")

# =================================================================
# PRESETS DE RANGO DE STRIKES
# =================================================================
PRESETS = {
    "estrecho": {
        "rango": (90, 110), "etiqueta": "🎯 ESTRECHO  90-110%",
        "desc": "Solo la zona con liquidez real, donde se concentra casi todo el volumen. "
                "La superficie sale limpia y el nivel de IV que ves es el fiable. "
                "**Empieza siempre por aqui.**",
    },
    "normal": {
        "rango": (75, 125), "etiqueta": "⚖️ NORMAL  75-125%",
        "desc": "Equilibrio. Se aprecia la forma completa del smile/skew sin demasiado ruido "
                "de opciones ilíquidas. Buen punto medio para el dia a dia.",
    },
    "amplio": {
        "rango": (50, 150), "etiqueta": "🌐 AMPLIO  50-150%",
        "desc": "Incluye coberturas lejanas y apuestas de cola (tail risk). Util para ver si "
                "alguien esta comprando proteccion extrema, pero con mas datos poco fiables.",
    },
    "todo": {
        "rango": (20, 250), "etiqueta": "🔭 TODO  20-250%",
        "desc": "Absolutamente todo lo que haya listado. La superficie se deforma bastante "
                "porque los extremos apenas se negocian. Solo para explorar posiciones raras.",
    },
}

if "rango_strikes" not in st.session_state:
    st.session_state.rango_strikes = PRESETS["normal"]["rango"]
if "preset_activo" not in st.session_state:
    st.session_state.preset_activo = "normal"

st.sidebar.markdown("### 📐 RANGO DE STRIKES")
st.sidebar.caption("Qué strikes se muestran, en % respecto al precio actual.")

for clave, datos_p in PRESETS.items():
    if st.sidebar.button(datos_p["etiqueta"], key=f"btn_{clave}"):
        st.session_state.rango_strikes = datos_p["rango"]
        st.session_state.preset_activo = clave
        st.rerun()

rango = st.sidebar.slider(
    "Ajuste manual", 20, 250, st.session_state.rango_strikes, key="slider_rango",
    help="Mueve los extremos si quieres un rango distinto a los presets.",
)
if rango != st.session_state.rango_strikes:
    st.session_state.rango_strikes = rango
    st.session_state.preset_activo = "manual"

preset_actual = st.session_state.preset_activo
if preset_actual in PRESETS:
    st.sidebar.markdown(f"<div class='caja-info'>{PRESETS[preset_actual]['desc']}</div>",
                        unsafe_allow_html=True)
else:
    st.sidebar.markdown("<div class='caja-info'>Rango manual personalizado.</div>",
                        unsafe_allow_html=True)

st.sidebar.markdown("---")

# =================================================================
# OPCIONES DE VISUALIZACION
# =================================================================
st.sidebar.markdown("### 👁️ VISUALIZACION")
tipo_vista = st.sidebar.radio("Superficie con", ["Calls", "Puts"], horizontal=True)
ver_lineas_venc = st.sidebar.checkbox(
    "Lineas por vencimiento", value=True,
    help="Dibuja una linea de color distinto por cada fecha de vencimiento, para diferenciarlas.",
)
ver_linea_spot = st.sidebar.checkbox(
    "Linea del precio actual", value=True,
    help="Plano vertical que marca donde esta el precio ahora mismo sobre la superficie.",
)
ver_superficie = st.sidebar.checkbox("Superficie interpolada", value=True)
ver_puntos = st.sidebar.checkbox("Puntos de datos reales", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧹 CALIDAD DE DATOS")
filtrar = st.sidebar.checkbox(
    "Filtrar datos poco fiables", value=True,
    help="MUY RECOMENDADO. Yahoo devuelve IV basura para opciones sin liquidez, "
         "y eso infla los numeros muchisimo.",
)
tasa = st.sidebar.number_input(
    "Tasa libre de riesgo (%)", 0.0, 15.0, 4.5, 0.25,
    help="Se usa para calcular las probabilidades implicitas. Aproximado al bono a 1 año.",
) / 100

if st.sidebar.button("🔄 REFRESCAR DATOS"):
    st.cache_data.clear()
    st.rerun()


# =================================================================
# DESCARGA
# =================================================================
@st.cache_data(ttl=180, show_spinner=False)
def cargar_datos(tk, aplicar_filtro):
    activo = yf.Ticker(tk)

    spot, historico = None, None
    try:
        spot = float(activo.fast_info["last_price"])
    except Exception:
        pass
    try:
        historico = activo.history(period="3mo")
        if spot is None and not historico.empty:
            spot = float(historico["Close"].iloc[-1])
    except Exception:
        pass
    if spot is None:
        return None, "No se pudo obtener el precio. Revisa el ticker."

    try:
        vencimientos = activo.options
    except Exception:
        vencimientos = []
    if not vencimientos:
        return None, f"{tk} no tiene opciones listadas en Yahoo Finance."

    hoy = datetime.now()
    cadenas, filas, n_orig, n_limp = {}, [], 0, 0

    for fecha in vencimientos:
        try:
            cad = activo.option_chain(fecha)
        except Exception:
            continue
        dias = (datetime.strptime(fecha, "%Y-%m-%d") - hoy).days
        if dias <= 0:
            continue

        calls_raw, puts_raw = cad.calls, cad.puts
        n_orig += len(calls_raw) + len(puts_raw)

        calls = an.limpiar_cadena(calls_raw, spot, estricto=aplicar_filtro)
        puts = an.limpiar_cadena(puts_raw, spot, estricto=aplicar_filtro)
        n_limp += len(calls) + len(puts)

        if calls.empty and puts.empty:
            continue

        cadenas[fecha] = {"calls": calls, "puts": puts,
                          "calls_raw": calls_raw, "puts_raw": puts_raw, "dias": dias}

        for tipo, tabla in (("Calls", calls), ("Puts", puts)):
            for _, f in tabla.iterrows():
                filas.append({"tipo": tipo, "strike": f["strike"], "dias": dias,
                              "iv": f["impliedVolatility"] * 100, "vencimiento": fecha})

    if not cadenas:
        return None, "Tras filtrar no quedaron datos fiables. Prueba a desactivar el filtro."

    return {
        "spot": spot, "historico": historico, "cadenas": cadenas,
        "superficie": pd.DataFrame(filas), "vencimientos": list(cadenas.keys()),
        "n_original": n_orig, "n_limpio": n_limp,
    }, None


if not ticker:
    st.info("Escribe algo en el buscador de la barra lateral para empezar.")
    st.stop()

with st.spinner(f"CARGANDO CADENA DE OPCIONES DE {ticker}..."):
    datos, error = cargar_datos(ticker, filtrar)

if error:
    st.error(f"⚠️ {error}")
    st.stop()

spot = datos["spot"]
s_min, s_max = spot * rango[0] / 100, spot * rango[1] / 100

# =================================================================
# CABECERA
# =================================================================
pct_calidad = datos["n_limpio"] / datos["n_original"] * 100 if datos["n_original"] else 0
st.markdown(
    f"""<div class="terminal-header">
    <span style="font-size:22px;">📡 OPTIONS TERMINAL</span> &nbsp;|&nbsp;
    <b>{ticker}</b> &nbsp; LAST <b>{spot:,.2f}</b> &nbsp;|&nbsp;
    {len(datos['vencimientos'])} VENCIMIENTOS &nbsp;|&nbsp;
    DATOS UTILES {datos['n_limpio']}/{datos['n_original']} ({pct_calidad:.0f}%) &nbsp;|&nbsp;
    <span class="gris">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · RETRASO ~15-20 MIN</span>
    </div>""", unsafe_allow_html=True)

# =================================================================
# METRICAS
# =================================================================
venc_front = datos["vencimientos"][0]
calls_front = datos["cadenas"][venc_front]["calls"]
puts_front = datos["cadenas"][venc_front]["puts"]
dias_front = datos["cadenas"][venc_front]["dias"]

iv_c, iv_l, pendiente = an.estructura_temporal(datos["superficie"], spot)
coste_straddle, mov_pct = an.movimiento_esperado(calls_front, puts_front, spot)

metricas = {
    "spot": spot,
    "pcr_volumen": an.put_call_ratio(calls_front, puts_front, "volume"),
    "pcr_oi": an.put_call_ratio(calls_front, puts_front, "openInterest"),
    "iv_atm": an.atm_iv(calls_front, puts_front, spot),
    "skew": an.skew_25(calls_front, puts_front, spot),
    "pendiente_temporal": pendiente,
    "vol_realizada": an.volatilidad_realizada(datos["historico"]),
    "max_pain": an.max_pain(calls_front, puts_front),
}

condiciones = an.evaluar_condiciones(metricas)
lect_corto = an.lectura_agregada(condiciones, "corto")
lect_medio = an.lectura_agregada(condiciones, "medio")

st.markdown("#### ▌ METRICAS CLAVE")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("IV ATM", f"{metricas['iv_atm']:.1f}%" if metricas["iv_atm"] else "n/d",
          help="Volatilidad implicita del strike mas cercano al precio actual. Es la referencia limpia.")
c2.metric("VOL REAL 30D", f"{metricas['vol_realizada']:.1f}%" if metricas["vol_realizada"] else "n/d",
          help="Cuanto se ha movido de verdad el activo en los ultimos 30 dias, anualizado.")
c3.metric("P/C VOLUMEN", f"{metricas['pcr_volumen']:.2f}" if metricas["pcr_volumen"] else "n/d")
c4.metric("SKEW", f"{metricas['skew']:+.1f}" if metricas["skew"] is not None else "n/d",
          help="IV de puts OTM menos IV de calls OTM. Positivo alto = miedo a caidas.")
c5.metric("MOV. ESPERADO", f"±{mov_pct:.1f}%" if mov_pct else "n/d",
          help=f"Movimiento implicito segun el straddle ATM, a {dias_front} dias.")

# --- Aviso si la IV parece desproporcionada ---
if metricas["iv_atm"] and metricas["vol_realizada"]:
    ratio = metricas["iv_atm"] / metricas["vol_realizada"]
    if ratio > 2.5:
        st.markdown(
            f"""<div class="alerta">
            ⚠️ <b>LA IV PARECE MUY ALTA</b>: {metricas['iv_atm']:.0f}% implicita frente a
            {metricas['vol_realizada']:.0f}% realizada ({ratio:.1f}x).<br>
            <span class="gris">Puede ser real (evento inminente: resultados, dato macro, noticia) o
            un problema de datos. Comprueba: 1) que el filtro de calidad este activado,
            2) que estes en preset ESTRECHO, 3) si el activo es de por si muy volatil.</span>
            </div>""", unsafe_allow_html=True)

# =================================================================
# LECTURA DEL MOTOR
# =================================================================
COLORES = {"alcista": "#00FF41", "bajista": "#FF3B3B", "neutral": "#888888"}

st.markdown("#### ▌ LECTURA DEL MOTOR DE CONDICIONES")
ca, cb = st.columns(2)
for col, lect, titulo in ((ca, lect_corto, "CORTO PLAZO"), (cb, lect_medio, "MEDIO PLAZO")):
    color = COLORES[lect["color"]]
    col.markdown(
        f"""<div class="caja-senal" style="border-color:{color};">
        <span class="gris">{titulo}</span><br>
        <span style="color:{color};font-size:20px;"><b>{lect['etiqueta']}</b></span><br>
        <span class="gris">alcistas: {lect['n_alcistas']} · bajistas: {lect['n_bajistas']}</span>
        </div>""", unsafe_allow_html=True)

st.caption("Refleja que condiciones objetivas se cumplen ahora en la cadena de opciones. "
           "Es una foto del posicionamiento actual, no una prediccion. No es asesoramiento financiero.")

with st.expander("▌ DESGLOSE DE CONDICIONES", expanded=False):
    for c in condiciones:
        color = COLORES[c["sesgo"]]
        marca = "✓" if c["se_cumple"] else "·"
        st.markdown(
            f"""<div style="border-left:3px solid {color};padding-left:12px;margin-bottom:12px;">
            <b>{marca} {c['nombre']}</b> <span style="color:{color};">[{c['sesgo'].upper()} · {c['horizonte']}]</span><br>
            <span style="font-size:18px;">{c['valor']}</span><br>
            <span class="gris">{c['explicacion']}</span></div>""", unsafe_allow_html=True)

# =================================================================
# SUPERFICIE 3D
# =================================================================
st.markdown("#### ▌ SUPERFICIE DE VOLATILIDAD 3D")

df_sup = datos["superficie"]
df_sup = df_sup[(df_sup["tipo"] == tipo_vista) &
                (df_sup["strike"] >= s_min) & (df_sup["strike"] <= s_max)]

if len(df_sup) < 8:
    st.warning("Pocos datos en este rango. Prueba un preset mas amplio o desactiva el filtro.")
else:
    fig = go.Figure()

    if ver_superficie and len(df_sup) >= 12:
        gx = np.linspace(df_sup["strike"].min(), df_sup["strike"].max(), 60)
        gy = np.linspace(df_sup["dias"].min(), df_sup["dias"].max(), 60)
        GX, GY = np.meshgrid(gx, gy)
        pts = (df_sup["strike"].values, df_sup["dias"].values)
        GZ = griddata(pts, df_sup["iv"].values, (GX, GY), method="cubic")
        GZ_lin = griddata(pts, df_sup["iv"].values, (GX, GY), method="linear")
        GZ = np.where(np.isnan(GZ), GZ_lin, GZ)
        fig.add_trace(go.Surface(
            x=GX, y=GY, z=GZ, colorscale="Inferno", opacity=0.72,
            colorbar=dict(title="IV %", tickfont=dict(color="#FF9E00"), x=1.02),
            name="Superficie", showscale=True, hoverinfo="skip",
        ))

    # --- lineas por vencimiento, cada una de un color ---
    PALETA = ["#00FF41", "#00E5FF", "#FF3B3B", "#FFD500", "#FF6EC7", "#7CFF00",
              "#FF9E00", "#B388FF", "#00FFB3", "#FF5722", "#4FC3F7", "#F48FB1"]

    if ver_lineas_venc:
        for i, venc in enumerate(sorted(df_sup["vencimiento"].unique())):
            sub = df_sup[df_sup["vencimiento"] == venc].sort_values("strike")
            if len(sub) < 2:
                continue
            dias_v = int(sub["dias"].iloc[0])
            fig.add_trace(go.Scatter3d(
                x=sub["strike"], y=sub["dias"], z=sub["iv"],
                mode="lines+markers",
                line=dict(color=PALETA[i % len(PALETA)], width=5),
                marker=dict(size=3, color=PALETA[i % len(PALETA)]),
                name=f"{venc} ({dias_v}d)",
                hovertemplate="Strike %{x:.1f}<br>IV %{z:.1f}%<br>" + f"{venc}<extra></extra>",
            ))

    # --- plano vertical del precio actual ---
    if ver_linea_spot and s_min <= spot <= s_max:
        z_min, z_max = df_sup["iv"].min(), df_sup["iv"].max()
        y_vals = np.array([df_sup["dias"].min(), df_sup["dias"].max()])
        z_vals = np.array([z_min * 0.95, z_max * 1.05])
        YY, ZZ = np.meshgrid(y_vals, z_vals)
        XX = np.full_like(YY, spot, dtype=float)
        fig.add_trace(go.Surface(
            x=XX, y=YY, z=ZZ, showscale=False, opacity=0.28,
            colorscale=[[0, "#FFFFFF"], [1, "#FFFFFF"]],
            name=f"Spot {spot:.2f}", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter3d(
            x=[spot, spot], y=[y_vals[0], y_vals[1]], z=[z_max * 1.05, z_max * 1.05],
            mode="lines+text", line=dict(color="#FFFFFF", width=6),
            text=["", f"  SPOT {spot:.2f}"], textposition="middle right",
            textfont=dict(color="#FFFFFF", size=11), name=f"SPOT {spot:.2f}",
        ))

    if ver_puntos:
        fig.add_trace(go.Scatter3d(
            x=df_sup["strike"], y=df_sup["dias"], z=df_sup["iv"],
            mode="markers", marker=dict(size=2, color="#666666"), name="Datos crudos",
        ))

    ejes = dict(gridcolor="#333333", zerolinecolor="#555555", color="#FF9E00",
                backgroundcolor="#000000", showbackground=True)
    fig.update_layout(
        paper_bgcolor="#000000", plot_bgcolor="#000000",
        font=dict(family="Consolas, monospace", color="#FF9E00", size=11),
        scene=dict(xaxis=dict(title="STRIKE", **ejes), yaxis=dict(title="DIAS A VENC.", **ejes),
                   zaxis=dict(title="IV %", **ejes),
                   camera=dict(eye=dict(x=1.7, y=-1.5, z=0.9))),
        height=680, margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(font=dict(color="#FF9E00", size=10), bgcolor="rgba(13,13,13,0.85)",
                    bordercolor="#FF9E00", borderwidth=1),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Cada linea de color es un vencimiento distinto (mira la leyenda). "
               "El plano blanco vertical marca el precio actual. "
               "Arrastra para girar · scroll para zoom · cursor encima para valores exactos. "
               "Puedes activar y desactivar cada capa en la barra lateral.")

# =================================================================
# CURVA DE PROBABILIDAD IMPLICITA
# =================================================================
st.markdown("#### ▌ CURVA DE PROBABILIDAD IMPLICITA")
st.caption("Lo mas parecido a un mercado de predicciones: para cada strike, que probabilidad "
           "le asigna el mercado a que el precio acabe por encima o por debajo al vencimiento.")

venc_prob = st.selectbox("Vencimiento para la curva", options=datos["vencimientos"], key="venc_prob")
cad_prob = datos["cadenas"][venc_prob]
curva = an.curva_probabilidad(cad_prob["calls"], spot, cad_prob["dias"], tasa)

if curva.empty:
    st.warning("No hay datos suficientes para la curva de probabilidad en este vencimiento.")
else:
    curva_f = curva[(curva["strike"] >= s_min) & (curva["strike"] <= s_max)]
    if curva_f.empty:
        curva_f = curva

    p_low, p_high = an.rango_probable(curva, 0.68)

    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(
        x=curva_f["strike"], y=curva_f["prob_encima"], mode="lines+markers",
        line=dict(color="#00FF41", width=3), marker=dict(size=5),
        name="Prob. acabar POR ENCIMA",
        hovertemplate="Strike %{x:.2f}<br>Prob. encima: %{y:.1f}%<extra></extra>",
    ))
    fig_p.add_trace(go.Scatter(
        x=curva_f["strike"], y=curva_f["prob_debajo"], mode="lines",
        line=dict(color="#FF3B3B", width=2, dash="dot"),
        name="Prob. acabar POR DEBAJO",
        hovertemplate="Strike %{x:.2f}<br>Prob. debajo: %{y:.1f}%<extra></extra>",
    ))
    fig_p.add_vline(x=spot, line=dict(color="#FFFFFF", width=2, dash="dash"),
                    annotation_text=f"SPOT {spot:.2f}", annotation_position="top",
                    annotation_font_color="#FFFFFF")
    fig_p.add_hline(y=50, line=dict(color="#555555", width=1))

    if p_low and p_high:
        fig_p.add_vrect(x0=p_low, x1=p_high, fillcolor="#FF9E00", opacity=0.10,
                        line_width=0, annotation_text="zona 68%", annotation_position="bottom left",
                        annotation_font_color="#FF9E00")

    fig_p.update_layout(
        paper_bgcolor="#000000", plot_bgcolor="#0D0D0D",
        font=dict(family="Consolas, monospace", color="#FF9E00", size=11),
        xaxis=dict(title="STRIKE", gridcolor="#222222", color="#FF9E00"),
        yaxis=dict(title="PROBABILIDAD IMPLICITA (%)", gridcolor="#222222",
                   color="#FF9E00", range=[0, 100]),
        height=420, margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified",
        legend=dict(font=dict(color="#FF9E00", size=10), bgcolor="rgba(13,13,13,0.85)",
                    bordercolor="#FF9E00", borderwidth=1, orientation="h", y=1.12),
    )
    st.plotly_chart(fig_p, use_container_width=True)

    cp1, cp2, cp3 = st.columns(3)
    cp1.metric("VENCE EN", f"{cad_prob['dias']} dias")
    if p_low and p_high:
        cp2.metric("ZONA 68% CONFIANZA", f"{p_low:.1f} — {p_high:.1f}")
        cp3.metric("AMPLITUD", f"±{(p_high - p_low) / 2 / spot * 100:.1f}%")

    st.markdown(
        """<div class="caja-info">
        <b>Como leerla:</b> la linea verde es la probabilidad de que el precio acabe
        <b>por encima</b> de cada strike. En el precio actual (linea blanca) ronda el 50%.
        A la izquierda sube hacia 100% (casi seguro que acaba por encima de un strike muy bajo),
        a la derecha baja hacia 0%.<br><br>
        <span class="gris">Matiz importante: es probabilidad <b>neutral al riesgo</b>, extraida de
        Black-Scholes. Incluye la prima que la gente paga por cubrirse, asi que exagera algo la
        probabilidad de caidas fuertes respecto a la probabilidad real. No es una bola de cristal.</span>
        </div>""", unsafe_allow_html=True)

# =================================================================
# STRIKES CLAVE POR OPEN INTEREST
# =================================================================
st.markdown("#### ▌ STRIKES CLAVE · DONDE ESTA EL DINERO")

claves = an.strikes_clave(cad_prob["calls"], cad_prob["puts"], top=4)
k1, k2 = st.columns(2)

with k1:
    st.markdown("**📈 CALLS con mas interes abierto**")
    if claves["calls"]:
        for r in claves["calls"]:
            dist = (r["strike"] - spot) / spot * 100
            st.markdown(
                f"<span style='color:#00FF41;'>▲ {r['strike']:>9,.2f}</span> "
                f"<span class='gris'>({dist:+.1f}%)</span> · "
                f"OI <b>{int(r['openInterest'] or 0):,}</b> · "
                f"vol {int(r['volume'] or 0):,}", unsafe_allow_html=True)
    else:
        st.caption("Sin datos.")

with k2:
    st.markdown("**📉 PUTS con mas interes abierto**")
    if claves["puts"]:
        for r in claves["puts"]:
            dist = (r["strike"] - spot) / spot * 100
            st.markdown(
                f"<span style='color:#FF3B3B;'>▼ {r['strike']:>9,.2f}</span> "
                f"<span class='gris'>({dist:+.1f}%)</span> · "
                f"OI <b>{int(r['openInterest'] or 0):,}</b> · "
                f"vol {int(r['volume'] or 0):,}", unsafe_allow_html=True)
    else:
        st.caption("Sin datos.")

st.caption("Mucho interes abierto en un strike = muchas posiciones vivas ahi. Suelen actuar como "
           "referencias de soporte/resistencia porque los creadores de mercado ajustan sus "
           "coberturas alrededor de esos niveles.")

# =================================================================
# CADENA EN FORMATO STRADDLE
# =================================================================
st.markdown("#### ▌ CADENA DE OPCIONES · FORMATO STRADDLE")

venc_tabla = st.selectbox("Vencimiento", options=datos["vencimientos"], key="venc_tabla")
cad_t = datos["cadenas"][venc_tabla]

COLS = ["lastPrice", "change", "percentChange", "volume", "openInterest"]
NOMBRES = ["Last Price", "Change", "% Change", "Volume", "Open Interest"]


def preparar(tabla, sufijo):
    disp = [c for c in COLS if c in tabla.columns]
    t = tabla[["strike"] + disp].copy()
    t.columns = ["strike"] + [f"{NOMBRES[COLS.index(c)]}{sufijo}" for c in disp]
    return t


straddle = pd.merge(preparar(cad_t["calls"], " (C)"), preparar(cad_t["puts"], " (P)"),
                    on="strike", how="outer").sort_values("strike")
straddle = straddle[(straddle["strike"] >= s_min) & (straddle["strike"] <= s_max)]

cols_c = [c for c in straddle.columns if c.endswith("(C)")]
cols_p = [c for c in straddle.columns if c.endswith("(P)")]
straddle = straddle[cols_c + ["strike"] + cols_p].rename(columns={"strike": "STRIKE"})

st.dataframe(straddle, use_container_width=True, height=400, hide_index=True)

pie = f"◄ CALLS │ STRIKE │ PUTS ► · Vencimiento {venc_tabla} ({cad_t['dias']} dias)."
if coste_straddle and mov_pct:
    pie += (f" El straddle ATM del vencimiento mas cercano cuesta {coste_straddle:.2f}, "
            f"lo que implica ±{mov_pct:.1f}% de movimiento esperado.")
st.caption(pie)

st.markdown("---")
st.caption("⚠️ Herramienta de analisis informativo. Datos con retraso y posibles errores. "
           "Las metricas describen el posicionamiento actual del mercado de opciones, no predicciones. "
           "No constituye asesoramiento financiero ni recomendacion de compra o venta.")
