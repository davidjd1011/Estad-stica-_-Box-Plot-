"""
Medidas de Posición y Diagrama de Cajas y Alambres (Box-Plot)
Estadística · Ingeniería de Sistemas
Ejecutar:  streamlit run app.py
"""
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Box-Plot y Medidas de Posición", page_icon="📦", layout="wide")

# ---------------------------------------------------------------- utilidades
METODOS = {
    "Lineal (Excel CUARTIL.INC / R tipo 7)": "linear",
    "Weibull (Excel CUARTIL.EXC / R tipo 6)": "weibull",
    "Hazen (R tipo 5)": "hazen",
    "Mediana insesgada (R tipo 8)": "median_unbiased",
}


def ejemplo() -> pd.DataFrame:
    lat = [120, 125, 130, 132, 135, 138, 140, 142, 145, 148, 150, 152,
           155, 158, 160, 165, 170, 175, 182, 190, 420, 510]
    a = [10, 12, 13, 14, 15, 16, 17, 18, 20]
    b = [8, 9, 10, 11, 12, 13, 14, 15, 40]
    return pd.DataFrame({"Latencia_API_ms": pd.Series(lat),
                         "Algoritmo_A_ms": pd.Series(a),
                         "Algoritmo_B_ms": pd.Series(b)})


def leer_archivo(f) -> pd.DataFrame:
    if f.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(f)
    return pd.read_csv(f, sep=None, engine="python", decimal=",") if b"," in f.getvalue()[:2000] and b";" in f.getvalue()[:2000] \
        else pd.read_csv(io.BytesIO(f.getvalue()), sep=None, engine="python")


def fmt(x, d=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{d}f}"


def estadisticos(x: np.ndarray, metodo: str, ddof: int, k: float, p: float) -> dict:
    x = np.sort(x[~np.isnan(x)])
    n = len(x)
    q1, q2, q3 = np.percentile(x, [25, 50, 75], method=metodo)
    ric = q3 - q1
    li, ls = q1 - k * ric, q3 + k * ric
    dentro = x[(x >= li) & (x <= ls)]
    at = x[(x < li) | (x > ls)]
    media = x.mean()
    sd = x.std(ddof=ddof) if n > ddof else np.nan
    vals, cnt = np.unique(x, return_counts=True)
    moda = ", ".join(fmt(v, 3) for v in vals[cnt == cnt.max()]) if cnt.max() > 1 else "Sin moda"
    return dict(n=n, x=x, media=media, mediana=q2, moda=moda, min=x[0], max=x[-1],
                rango=x[-1] - x[0], varianza=sd ** 2 if n > ddof else np.nan, sd=sd,
                cv=sd / media * 100 if media else np.nan, q1=q1, q2=q2, q3=q3, ric=ric,
                li=li, ls=ls, wl=dentro[0], wh=dentro[-1], atipicos=at,
                pk=np.percentile(x, p, method=metodo),
                asim=pd.Series(x).skew(), curt=pd.Series(x).kurt())


# ---------------------------------------------------------------- barra lateral
st.sidebar.title("⚙️ Configuración")
fuente = st.sidebar.radio("1. Fuente de datos", ["Datos de ejemplo", "Subir archivo (CSV/Excel)", "Pegar datos"])
df = None
if fuente == "Datos de ejemplo":
    df = ejemplo()
elif fuente.startswith("Subir"):
    f = st.sidebar.file_uploader("Archivo CSV o Excel", type=["csv", "txt", "xlsx", "xls"])
    if f is not None:
        try:
            df = leer_archivo(f)
        except Exception as e:
            st.sidebar.error(f"No se pudo leer el archivo: {e}")
else:
    txt = st.sidebar.text_area("Números separados por espacio, coma o salto de línea", height=120)
    try:
        v = [float(t.replace(",", ".")) for t in txt.replace(";", " ").split()] if txt.strip() else []
        df = pd.DataFrame({"Datos": v}) if v else None
    except ValueError:
        st.sidebar.error("Hay valores no numéricos.")

metodo_nombre = st.sidebar.selectbox("2. Método de cuartiles", list(METODOS))
muestral = st.sidebar.checkbox("Desviación muestral (n − 1)", value=True)
k = st.sidebar.number_input("Factor k de las cercas", 0.5, 5.0, 1.5, 0.5,
                            help="1,5 = atípicos; 3,0 = atípicos extremos")
p = st.sidebar.slider("3. Percentil Pₚ dinámico", 0, 100, 90)
etiquetas = st.sidebar.checkbox("Mostrar etiquetas en el gráfico", value=True)
puntos = st.sidebar.checkbox("Mostrar todos los datos", value=False)

# ---------------------------------------------------------------- encabezado
st.title("📦 Medidas de Posición y Diagrama de Cajas y Alambres")
st.caption("Estadística · Ingeniería de Sistemas — carga de datos, tendencia central, dispersión, posición y Box-Plot interactivo")

if df is None or df.empty:
    st.info("Cargue o pegue datos en la barra lateral para comenzar.")
    st.stop()

num = df.select_dtypes(include=np.number).columns.tolist()
if not num:
    st.error("El conjunto de datos no contiene columnas numéricas.")
    st.stop()
cols = st.multiselect("Variables a analizar (puede comparar varias)", num, default=num[:1] if fuente != "Datos de ejemplo" else num)
if not cols:
    st.stop()

metodo, ddof = METODOS[metodo_nombre], int(muestral)
res = {c: estadisticos(df[c].to_numpy(dtype=float), metodo, ddof, k, p) for c in cols}

tabs = st.tabs(["📄 Datos", "📊 Tendencia central y dispersión", "📐 Medidas de posición", "📦 Box-Plot", "🧠 Interpretación"])

# ---------------------------------------------------------------- 1. datos
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.metric("Filas", len(df))
    c2.metric("Variables numéricas", len(num))
    c3.metric("Valores faltantes", int(df[cols].isna().sum().sum()))
    st.dataframe(df, use_container_width=True, height=300)

# ---------------------------------------------------------------- 2. tendencia y dispersión
with tabs[1]:
    sim = "s" if muestral else "σ"
    filas = {
        "n": lambda r: r["n"], "Media (x̄)": lambda r: fmt(r["media"]), "Mediana (Me)": lambda r: fmt(r["mediana"]),
        "Moda (Mo)": lambda r: r["moda"], "Mínimo": lambda r: fmt(r["min"]), "Máximo": lambda r: fmt(r["max"]),
        "Rango (R)": lambda r: fmt(r["rango"]), f"Varianza ({sim}²)": lambda r: fmt(r["varianza"]),
        f"Desviación estándar ({sim})": lambda r: fmt(r["sd"]), "Coef. de variación (CV %)": lambda r: fmt(r["cv"], 2),
        "Asimetría (g₁)": lambda r: fmt(r["asim"], 3), "Curtosis (g₂)": lambda r: fmt(r["curt"], 3),
    }
    tabla = pd.DataFrame({c: [f(res[c]) for f in filas.values()] for c in cols}, index=list(filas)).astype(str)
    st.dataframe(tabla, use_container_width=True, height=460)
    st.latex(r"\bar{x}=\frac{\sum x_i}{n}\qquad s^2=\frac{\sum (x_i-\bar{x})^2}{n-1}\qquad CV=\frac{s}{\bar{x}}\cdot 100")

# ---------------------------------------------------------------- 3. posición
with tabs[2]:
    pos = {"Q₁ (P25)": "q1", "Q₂ (P50)": "q2", "Q₃ (P75)": "q3", "RIC = Q₃ − Q₁": "ric",
           f"Cerca inferior (Q₁ − {k}·RIC)": "li", f"Cerca superior (Q₃ + {k}·RIC)": "ls",
           "Bigote inferior": "wl", "Bigote superior": "wh", f"Percentil P{p}": "pk"}
    tp = pd.DataFrame({c: [fmt(res[c][v]) for v in pos.values()] for c in cols}, index=list(pos)).astype(str)
    tp.loc["N.º de atípicos"] = [str(len(res[c]["atipicos"])) for c in cols]
    st.dataframe(tp, use_container_width=True, height=400)
    st.latex(r"h=(n-1)\cdot\frac{p}{100}\;;\quad P_p=x_{(\lfloor h\rfloor)}+(h-\lfloor h\rfloor)\,(x_{(\lfloor h\rfloor+1)}-x_{(\lfloor h\rfloor)})")
    st.latex(r"RIC=Q_3-Q_1\;;\quad LI=Q_1-k\cdot RIC\;;\quad LS=Q_3+k\cdot RIC")
    st.markdown("**Tabla de percentiles y deciles**")
    pcts = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]
    tpc = pd.DataFrame({c: [fmt(np.percentile(res[c]["x"], q, method=metodo)) for q in pcts] for c in cols},
                       index=[f"P{q}" for q in pcts]).astype(str)
    st.dataframe(tpc, use_container_width=True, height=300)

# ---------------------------------------------------------------- 4. boxplot
with tabs[3]:
    def separar(items, gap):
        items = sorted(items, key=lambda t: t[1])
        adj, prev = [], -np.inf
        for _, v_ in items:
            prev = max(v_, prev + gap)
            adj.append(prev)
        desp = np.mean(adj) - np.mean([v_ for _, v_ in items])
        return [(n_, v_, a_ - desp) for (n_, v_), a_ in zip(items, adj)]

    nc = len(cols)
    fig = make_subplots(rows=1, cols=nc, subplot_titles=cols, horizontal_spacing=0.06)
    rng_j = np.random.default_rng(1)
    for i, c in enumerate(cols, start=1):
        r = res[c]
        lo, hi = min(r["min"], r["li"]), max(r["max"], r["ls"])
        pad = (hi - lo) * 0.08 or 1.0
        rng = [lo - pad, hi + pad]
        gap = (rng[1] - rng[0]) * 0.05
        first = i == 1

        fig.add_trace(go.Box(x=[0], q1=[r["q1"]], median=[r["q2"]], q3=[r["q3"]],
                             lowerfence=[r["wl"]], upperfence=[r["wh"]],
                             fillcolor="rgba(31,111,235,0.25)", line=dict(color="#1f6feb", width=2),
                             width=0.5, whiskerwidth=0.6, boxpoints=False, showlegend=False,
                             hoverinfo="y"), row=1, col=i)
        if puntos:
            fig.add_trace(go.Scatter(x=rng_j.uniform(-0.15, 0.15, r["n"]), y=r["x"], mode="markers",
                                     name="Datos", legendgroup="d", showlegend=first,
                                     marker=dict(size=5, color="rgba(120,120,120,0.5)"),
                                     hovertemplate="%{y:.3f}<extra></extra>"), row=1, col=i)
        fig.add_trace(go.Scatter(x=[-0.3, 0.3, None, -0.3, 0.3], y=[r["li"], r["li"], None, r["ls"], r["ls"]],
                                 mode="lines", name="Cercas (LI / LS)", legendgroup="c", showlegend=first,
                                 line=dict(color="#d93025", dash="dash", width=1.5),
                                 hoverinfo="skip"), row=1, col=i)
        fig.add_trace(go.Scatter(x=[0], y=[r["media"]], mode="markers", name="Media (x̄)",
                                 legendgroup="m", showlegend=first,
                                 marker=dict(symbol="diamond", size=12, color="#188038",
                                             line=dict(color="white", width=1)),
                                 hovertemplate=f"Media: {fmt(r['media'])}<extra></extra>"), row=1, col=i)
        if len(r["atipicos"]):
            fig.add_trace(go.Scatter(x=[0] * len(r["atipicos"]), y=r["atipicos"], mode="markers",
                                     name="Valores atípicos", legendgroup="o", showlegend=first,
                                     marker=dict(symbol="circle-open", size=11, color="#d93025",
                                                 line=dict(width=2)),
                                     hovertemplate="Atípico: %{y:.3f}<extra></extra>"), row=1, col=i)
        if etiquetas:
            items = [("Q₃", r["q3"]), ("Mediana", r["q2"]), ("Q₁", r["q1"]), ("Media", r["media"]),
                     ("Bigote sup.", r["wh"]), ("Bigote inf.", r["wl"]),
                     ("Cerca sup.", r["ls"]), ("Cerca inf.", r["li"])]
            lab = separar(items, gap)
            for n_, v_, a_ in lab:
                fig.add_trace(go.Scatter(x=[0.3, 0.42], y=[v_, a_], mode="lines", hoverinfo="skip",
                                         showlegend=False, line=dict(color="rgba(150,150,150,0.6)", width=1)),
                              row=1, col=i)
            fig.add_trace(go.Scatter(x=[0.44] * len(lab), y=[a_ for _, _, a_ in lab], mode="text",
                                     text=[f"{n_} {fmt(v_, 2)}" for n_, v_, _ in lab],
                                     textposition="middle right", textfont=dict(size=11),
                                     showlegend=False, hoverinfo="skip"), row=1, col=i)
        fig.add_trace(go.Scatter(x=[-0.6, 0.3], y=[r["pk"], r["pk"]], mode="lines+text",
                                 text=[f"P{p} = {fmt(r['pk'], 2)}", ""], textposition="top right",
                                 textfont=dict(size=11, color="gray"),
                                 line=dict(color="gray", dash="dot", width=1),
                                 showlegend=False, hoverinfo="skip"), row=1, col=i)
        fig.update_xaxes(visible=False, range=[-0.6, 1.6], fixedrange=True, row=1, col=i)
        fig.update_yaxes(range=rng, showgrid=True, gridcolor="rgba(150,150,150,0.2)", zeroline=False,
                         title_text="Valor" if first else None, row=1, col=i)
    fig.update_layout(height=640, margin=dict(t=50, b=70, l=60, r=20),
                      legend=dict(orientation="h", y=-0.06, x=0.5, xanchor="center"))
    st.plotly_chart(fig, use_container_width=True)
    resumen = pd.DataFrame({c: {"Q1": res[c]["q1"], "Mediana": res[c]["q2"], "Media": res[c]["media"], "Q3": res[c]["q3"],
                                "RIC": res[c]["ric"], "Cerca inf.": res[c]["li"], "Cerca sup.": res[c]["ls"],
                                "Bigote inf.": res[c]["wl"], "Bigote sup.": res[c]["wh"],
                                "Atípicos": "; ".join(fmt(a, 2) for a in res[c]["atipicos"]) or "ninguno"} for c in cols})
    st.download_button("⬇️ Descargar resumen (CSV)", resumen.to_csv().encode("utf-8-sig"), "resumen_boxplot.csv", "text/csv")

# ---------------------------------------------------------------- 5. interpretación
with tabs[4]:
    for c in cols:
        r = res[c]
        st.subheader(c)
        dif = r["media"] - r["mediana"]
        forma = ("simétrica" if abs(dif) < 0.05 * (r["ric"] or 1)
                 else "asimétrica positiva (cola a la derecha, x̄ > Me)" if dif > 0
                 else "asimétrica negativa (cola a la izquierda, x̄ < Me)")
        cv = r["cv"]
        disp = "baja" if cv < 15 else "moderada" if cv < 30 else "alta"
        st.markdown(f"- **Forma:** distribución {forma}.\n- **Dispersión:** CV = {fmt(cv, 1)} % → variabilidad {disp}. "
                    f"El 50 % central de los datos se concentra en [{fmt(r['q1'])}; {fmt(r['q3'])}] (RIC = {fmt(r['ric'])}).")
        if len(r["atipicos"]):
            st.warning(f"**{len(r['atipicos'])} valor(es) atípico(s):** {', '.join(fmt(a, 2) for a in r['atipicos'])}, "
                       f"fuera de [{fmt(r['li'])}; {fmt(r['ls'])}]. Verifique si son errores de captura o eventos reales "
                       f"(picos de carga, timeouts). La mediana ({fmt(r['mediana'])}) es más representativa que la media ({fmt(r['media'])}).")
        else:
            st.success("No se detectan valores atípicos con el factor k seleccionado.")