import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st

st.set_page_config(page_title="BoxLab | Medidas de Posición", page_icon="📦", layout="wide")
st.markdown("""<style>
.stApp{background:#FFFFFF}
h1,h2,h3{color:#1B2A41;font-weight:700}
.kpi{background:#F3F6FB;border:1px solid #E1E8F3;border-radius:10px;padding:12px 14px}
.kpi small{color:#5B6B85;display:block}.kpi b{font-size:1.35rem;color:#1F4E9C}
</style>""", unsafe_allow_html=True)

EJEMPLOS = {
 "Caso 1 · Tiempo de respuesta API (ms)": [112,118,121,125,127,130,132,135,138,140,143,146,150,155,158,162,170,185,420,510],
 "Caso 2 · Tiempo de resolución de tickets (h)": [1.5,2,2.2,2.5,2.8,3,3.1,3.4,3.6,4,4.2,4.5,5,5.5,6,6.5,7,8,26,30],
}

def stats(x, method):
    x = np.sort(np.asarray(x, float))
    q1, q2, q3 = np.percentile(x, [25, 50, 75], method=method)
    iqr = q3 - q1
    li, ls = q1 - 1.5*iqr, q3 + 1.5*iqr
    dentro = x[(x >= li) & (x <= ls)]
    return dict(x=x, q1=q1, q2=q2, q3=q3, iqr=iqr, li=li, ls=ls,
                wl=dentro.min(), wu=dentro.max(), out=x[(x < li) | (x > ls)],
                ext=x[(x < q1-3*iqr) | (x > q3+3*iqr)])

st.title("📦 BoxLab: Medidas de Posición y Diagrama de Cajas")
st.caption("Ingeniería de Sistemas · Análisis estadístico interactivo con detección de valores atípicos")

# ---------- 1. Carga de datos ----------
with st.sidebar:
    st.header("1. Carga de datos")
    fuente = st.radio("Fuente", ["Ejemplo", "Archivo CSV/Excel", "Escribir datos"])
    datos, nombre, grupo, dfg = None, "Variable", None, None
    if fuente == "Ejemplo":
        nombre = st.selectbox("Conjunto", list(EJEMPLOS))
        datos = pd.Series(EJEMPLOS[nombre], name=nombre)
    elif fuente == "Archivo CSV/Excel":
        f = st.file_uploader("Sube tu archivo", type=["csv", "xlsx"])
        if f:
            df = pd.read_csv(f) if f.name.endswith("csv") else pd.read_excel(f)
            num = df.select_dtypes("number").columns.tolist()
            if num:
                nombre = st.selectbox("Columna numérica", num, index=next((i for i, c in enumerate(num) if "umedad" in c), 0))
                datos = df[nombre].dropna()
                cat = [c for c in df.columns if c not in num]
                opc = ["Ninguno"] + cat
                grupo = st.selectbox("Agrupar por (ej. Turno)", opc, index=opc.index("Turno") if "Turno" in opc else 0)
                if grupo != "Ninguno":
                    dfg = df[[nombre, grupo]].dropna()
            else:
                st.error("El archivo no tiene columnas numéricas.")
    else:
        txt = st.text_area("Valores separados por coma, espacio o salto de línea", "12, 15, 14, 10, 18, 20, 16, 55")
        try:
            datos = pd.Series(pd.to_numeric(txt.replace(",", " ").split()), name="Variable")
        except Exception:
            st.error("Hay valores no numéricos.")
    st.header("2. Parámetros")
    metodo = st.selectbox("Método de cuantiles", ["linear", "weibull", "hazen", "median_unbiased", "lower", "higher", "midpoint"],
                          help="'linear' equivale a Excel INC.; 'weibull' equivale a Excel EXC.")
    k = st.slider("Percentil k a calcular", 1, 99, 90)
    k_iqr = st.slider("Factor de los bigotes (×IQR)", 1.0, 3.0, 1.5, 0.5)

if datos is None or len(datos) < 4:
    st.info("Carga al menos 4 datos para comenzar."); st.stop()

x = datos.to_numpy(float); n = len(x)
s = stats(x, metodo)
li, ls = s["q1"] - k_iqr*s["iqr"], s["q3"] + k_iqr*s["iqr"]
dentro = s["x"][(s["x"] >= li) & (s["x"] <= ls)]
wl, wu = dentro.min(), dentro.max()
out = s["x"][(s["x"] < li) | (s["x"] > ls)]
media, mediana = x.mean(), s["q2"]
sd = x.std(ddof=1)

t1, t2, t3, t4 = st.tabs(["📊 Resumen", "📦 Box-Plot", "📐 Medidas de posición", "📘 Fórmulas"])

with t1:
    st.subheader("Medidas de tendencia central y dispersión")
    moda = pd.Series(x).mode()
    kp = [("n", n), ("Media x̄", media), ("Mediana Me", mediana),
          ("Moda Mo", moda.iloc[0] if len(moda) < n else "Sin moda"),
          ("Mínimo", x.min()), ("Máximo", x.max()), ("Rango R", np.ptp(x)),
          ("Varianza s²", x.var(ddof=1)), ("Desv. estándar s", sd),
          ("Coef. variación CV", f"{sd/media*100:.2f}%" if media else "—"),
          ("Asimetría g₁", pd.Series(x).skew()), ("Curtosis g₂", pd.Series(x).kurt())]
    cols = st.columns(4)
    for i, (l, v) in enumerate(kp):
        v = f"{v:,.3f}" if isinstance(v, (int, float, np.floating)) else v
        cols[i % 4].markdown(f'<div class="kpi"><small>{l}</small><b>{v}</b></div>', unsafe_allow_html=True)
        if i % 4 == 3: cols = st.columns(4)
    st.markdown("**Ordenación de los datos (ascendente)**")
    st.dataframe(pd.DataFrame({"Posición i": range(1, n+1), "x(i)": s["x"]}).set_index("Posición i").T, width="stretch")
    if media > mediana*1.05: st.warning("Media > Mediana: distribución sesgada a la derecha (posible influencia de atípicos altos).")
    elif media < mediana*0.95: st.warning("Media < Mediana: distribución sesgada a la izquierda.")
    else: st.success("Media ≈ Mediana: distribución aproximadamente simétrica.")

def box_por_grupo():
    st.subheader("Box-Plot por grupo (turno)")
    fig = go.Figure(); filas = []
    for g, d in dfg.groupby(grupo):
        v = d[nombre].to_numpy(float); gx = str(g)
        a, b, c = np.percentile(v, [25, 50, 75], method=metodo); r = c - a
        lo, hi = a - k_iqr*r, c + k_iqr*r
        ins = v[(v >= lo) & (v <= hi)]; oi, os_ = v[v < lo], v[v > hi]
        fig.add_trace(go.Box(x=[gx], q1=[a], median=[b], q3=[c], lowerfence=[ins.min()], upperfence=[ins.max()],
                             fillcolor="rgba(31,78,156,.15)", line=dict(color="#1F4E9C", width=2), boxpoints=False,
                             showlegend=False, width=0.4))
        fig.add_trace(go.Scatter(x=[gx], y=[v.mean()], mode="markers+text", text=[f"x̄={v.mean():.2f}"], textposition="middle right",
                                 marker=dict(color="#F2A900", size=12, symbol="diamond", line=dict(color="#1B2A41", width=1)),
                                 name="Promedio", legendgroup="m", showlegend=(g == sorted(dfg[grupo].unique())[0])))
        fig.add_trace(go.Scatter(x=[gx, gx], y=[lo, hi], mode="markers", marker=dict(color="#7A8AA3", size=14, symbol="line-ew-open"),
                                 name="Límites LI / LS", legendgroup="l", showlegend=(g == sorted(dfg[grupo].unique())[0])))
        if len(oi): fig.add_trace(go.Scatter(x=[gx]*len(oi), y=oi, mode="markers+text", text=[f"{t:g}" for t in oi], textposition="middle right",
                                 marker=dict(color="#2A7DE1", size=10, symbol="circle-open", line=dict(width=2)), name="Atípico inferior", legendgroup="i", showlegend=False))
        if len(os_): fig.add_trace(go.Scatter(x=[gx]*len(os_), y=os_, mode="markers+text", text=[f"{t:g}" for t in os_], textposition="middle right",
                                 marker=dict(color="#D64545", size=10, symbol="circle-open", line=dict(width=2)), name="Atípico superior", legendgroup="s", showlegend=False))
        filas.append({grupo: gx, "n": len(v), "Media x̄": round(v.mean(), 3), "Mediana": round(b, 3), "Desv. s": round(v.std(ddof=1), 3),
                      "Q1": round(a, 3), "Q3": round(c, 3), "IQR": round(r, 3), "LI": round(lo, 3), "LS": round(hi, 3),
                      "Atípicos inferiores": ", ".join(f"{t:g}" for t in oi) or "—", "Atípicos superiores": ", ".join(f"{t:g}" for t in os_) or "—"})
    fig.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white", height=560, yaxis_title=nombre,
                      xaxis_title=grupo, legend=dict(orientation="h", y=-0.15), margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, width="stretch")
    tb = pd.DataFrame(filas); st.dataframe(tb, hide_index=True, width="stretch")
    ni = sum(1 for f in filas if f["Atípicos inferiores"] != "—"); ns = sum(1 for f in filas if f["Atípicos superiores"] != "—")
    st.info(f"{ni} grupo(s) con atípicos inferiores y {ns} grupo(s) con atípicos superiores.")



with t2:
    if dfg is not None:
        box_por_grupo()
    else:
        fig = go.Figure()
        fig.add_trace(go.Box(q1=[s["q1"]], median=[mediana], q3=[s["q3"]], lowerfence=[wl], upperfence=[wu],
                             x=[nombre[:40]], name="Caja", fillcolor="rgba(31,78,156,.15)",
                             line=dict(color="#1F4E9C", width=2), boxpoints=False, width=0.35))
        fig.add_trace(go.Scatter(x=[nombre[:40]]*len(out), y=out, mode="markers+text", name="Atípicos",
                                 text=[f"{v:g}" for v in out], textposition="middle right",
                                 marker=dict(color="#D64545", size=11, symbol="circle-open", line=dict(width=2))))
        fig.add_trace(go.Scatter(x=[nombre[:40]], y=[media], mode="markers", name=f"Promedio x̄ = {media:.2f}",
                                 marker=dict(color="#F2A900", size=13, symbol="diamond", line=dict(color="#1B2A41", width=1))))
        for y, lbl, c in [(li, "Límite inferior", "#7A8AA3"), (ls, "Límite superior", "#7A8AA3"),
                          (mediana, f"Mediana = {mediana:.2f}", "#1F4E9C")]:
            fig.add_hline(y=y, line_dash="dot", line_color=c, annotation_text=f"{lbl}: {y:.2f}",
                          annotation_position="top left", annotation_font_color=c)
        fig.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white", height=560,
                          yaxis_title=nombre, legend=dict(orientation="h", y=-0.1), margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, width="stretch")
        c = st.columns(5)
        for col, (l, v) in zip(c, [("Q₁", s["q1"]), ("Q₃", s["q3"]), ("IQR", s["iqr"]), ("Bigote inf.", wl), ("Bigote sup.", wu)]):
            col.metric(l, f"{v:,.3f}")
        if len(out):
            st.error(f"Se detectaron {len(out)} valor(es) atípico(s): {', '.join(f'{v:g}' for v in out)}")
            st.dataframe(pd.DataFrame({"Valor": out, "Tipo": ["Extremo" if (v < s['q1']-3*s['iqr'] or v > s['q3']+3*s['iqr']) else "Leve" for v in out]}), hide_index=True)
        else:
            st.success("No se detectaron valores atípicos con el criterio actual.")


with t3:
    st.subheader("Cálculo dinámico de medidas de posición")
    qs = [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, k]
    qs = sorted(set(qs))
    tabla = pd.DataFrame({"Percentil": [f"P{p}" for p in qs],
                          "Equivale a": [{25: "Q1", 50: "Q2 (Mediana)", 75: "Q3"}.get(p, f"D{p//10}" if p % 10 == 0 else "") for p in qs],
                          "Valor": [np.percentile(x, p, method=metodo) for p in qs]})
    st.dataframe(tabla, hide_index=True, width="stretch")
    pos = (n + 1) * k / 100
    st.info(f"Posición de P{k}: L = (n+1)·k/100 = ({n}+1)·{k}/100 = **{pos:.2f}** → valor = **{np.percentile(x, k, method=metodo):.3f}**")
    v = st.number_input("Rango percentil de un valor x₀", value=float(mediana))
    st.metric(f"PR(x₀={v:g})", f"{(x < v).sum()/n*100 + 0.5*(x == v).sum()/n*100:.1f} %")

with t4:
    st.subheader("Fórmulas utilizadas")
    st.latex(r"\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i \qquad s^2=\frac{\sum (x_i-\bar{x})^2}{n-1} \qquad s=\sqrt{s^2} \qquad CV=\frac{s}{\bar{x}}\cdot 100")
    st.latex(r"L_k=\frac{k\,(n+1)}{100} \qquad P_k = x_{(\lfloor L_k\rfloor)}+(L_k-\lfloor L_k\rfloor)\,\big(x_{(\lfloor L_k\rfloor+1)}-x_{(\lfloor L_k\rfloor)}\big)")
    st.latex(r"Q_1=P_{25},\; Q_2=P_{50}=Me,\; Q_3=P_{75} \qquad IQR=Q_3-Q_1")
    st.latex(r"LI=Q_1-1.5\cdot IQR \qquad LS=Q_3+1.5\cdot IQR")
    st.markdown("Un dato es **atípico leve** si cae fuera de [LI, LS] y **extremo** si excede Q₁−3·IQR o Q₃+3·IQR.")
    st.caption("Los bigotes llegan al dato más extremo que aún está dentro de [LI, LS].")