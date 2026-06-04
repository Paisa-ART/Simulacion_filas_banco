"""
Simulacion de Filas en un Banco - Modelo M/M/c
Materia: Modelacion y Simulacion
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import poisson, expon
import math

st.set_page_config(page_title="Filas en un Banco", layout="wide")

# ---------------------------------------------------------------------------
# Titulo
# ---------------------------------------------------------------------------
st.title("Simulacion de Filas en un Banco")
st.markdown("**Modelo M/M/c** | Distribuciones: Poisson y Exponencial")
st.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar - parametros
# ---------------------------------------------------------------------------
st.sidebar.header("Parametros del sistema")
lam = st.sidebar.slider("Tasa de llegada (clientes/hora)", 5, 80, 30)
mu  = st.sidebar.slider("Tasa de servicio (clientes/hora/cajero)", 5, 60, 20)
c   = st.sidebar.slider("Numero de cajeros", 1, 6, 2)
n   = st.sidebar.slider("Clientes a simular", 50, 500, 200)

rho = lam / (c * mu)
st.sidebar.markdown("---")
st.sidebar.metric("Utilizacion rho", f"{rho:.3f}")
if rho >= 1:
    st.sidebar.error("Sistema inestable: rho >= 1")
else:
    st.sidebar.success(f"Sistema estable")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Fenomeno Aleatorio",
    "2. Distribuciones",
    "3. Simulacion",
    "4. Resultados"
])

# TAB 1 - FENOMENO ALEATORIO
with tab1:
    st.header("Identificacion del Fenomeno Aleatorio")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Descripcion del sistema")
        st.markdown("""
Los clientes llegan a un banco, toman un turno y esperan
hasta ser atendidos por uno de los cajeros disponibles.

**Entradas:**
- Clientes que llegan al banco
- Solicitudes de servicio (retiros, depositos, consultas)
- Numero de cajeros disponibles

**Procesos:**
- Asignacion de turno
- Espera en la fila
- Atencion en ventanilla

**Salidas:**
- Cliente atendido
- Transaccion completada
- Registro del tiempo de espera
        """)

    with col2:
        st.subheader("Variables aleatorias")
        st.markdown("""
**Variable 1 - Numero de llegadas** `(Discreta)`
- Cuenta cuantos clientes llegan por hora
- Valores: 0, 1, 2, 3, ...
- Distribucion: **Poisson**

---

**Variable 2 - Tiempo entre llegadas** `(Continua)`
- Minutos entre dos clientes consecutivos
- Valores: cualquier numero real positivo
- Distribucion: **Exponencial**

---

**Variable 3 - Tiempo de servicio** `(Continua)`
- Duracion de la atencion en ventanilla
- Varia segun tipo de transaccion
- Distribucion: **Exponencial**
        """)

    st.markdown("---")
    st.subheader("Por que es un sistema estocastico?")
    st.info("""
Los clientes deciden de forma **independiente** cuando ir al banco.
No existe un patron fijo que permita predecir con exactitud cuantos
llegaran ni cuanto tardara cada transaccion. Por eso necesitamos
modelos probabilisticos para analizar el sistema.

**Clasificacion:** Estocastico y Discreto (modelo M/M/c de Kendall)
    """)

# TAB 2 - DISTRIBUCIONES
with tab2:
    st.header("Distribuciones Probabilisticas Seleccionadas")

    # --- Poisson ---
    st.subheader("Distribucion de Poisson - Llegadas por hora")
    st.markdown(f"""
**Por que Poisson?** Las llegadas son independientes entre si y
la tasa promedio **lambda = {lam} clientes/hora** es constante.
Estas son exactamente las condiciones de la distribucion de Poisson.

Formula: &nbsp; `P(X = k) = (lambda^k * e^-lambda) / k!`
    """)

    k_vals = np.arange(0, int(lam * 2.5) + 1)
    p_vals = poisson.pmf(k_vals, lam)

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    colores = ["#1B6CA8" if abs(k - lam) <= 3 else "#B5D4F4" for k in k_vals]
    ax1.bar(k_vals, p_vals, color=colores, edgecolor="white")
    ax1.axvline(lam, color="#F0B429", linestyle="--", linewidth=2, label=f"Media = {lam}")
    ax1.set_xlabel("Numero de clientes (k)")
    ax1.set_ylabel("Probabilidad P(X=k)")
    ax1.set_title(f"Poisson con lambda = {lam}")
    ax1.legend()
    ax1.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig1)
    plt.close()

    st.markdown(f"La probabilidad de que lleguen exactamente **{lam} clientes** en una hora es: `{poisson.pmf(lam, lam):.4f}`")

    st.markdown("---")

    # --- Exponencial ---
    st.subheader("Distribucion Exponencial - Tiempo entre llegadas")
    st.markdown(f"""
**Por que Exponencial?** Si las llegadas siguen una Poisson,
el tiempo entre ellas sigue una Exponencial. Ademas tiene la
propiedad de **falta de memoria**: no importa cuanto tiempo
lleva esperando el siguiente cliente, la probabilidad es la misma.

Formula: &nbsp; `f(x) = lambda * e^(-lambda*x)`,  x >= 0
    """)

    lam_min = lam / 60.0
    mu_min  = mu  / 60.0
    x_max   = max(20.0, 4 / lam_min)
    xs      = np.linspace(0.001, x_max, 300)

    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(10, 4))

    ax2.fill_between(xs, expon.pdf(xs, scale=1/lam_min), alpha=0.3, color="#1D9E75")
    ax2.plot(xs, expon.pdf(xs, scale=1/lam_min), color="#1D9E75", linewidth=2)
    ax2.axvline(1/lam_min, color="#F0B429", linestyle="--", linewidth=2,
                label=f"Media = {1/lam_min:.1f} min")
    ax2.set_title("Tiempo entre llegadas")
    ax2.set_xlabel("Tiempo (min)")
    ax2.set_ylabel("f(x)")
    ax2.legend()
    ax2.spines[["top", "right"]].set_visible(False)

    ax3.fill_between(xs, expon.pdf(xs, scale=1/mu_min), alpha=0.3, color="#378ADD")
    ax3.plot(xs, expon.pdf(xs, scale=1/mu_min), color="#378ADD", linewidth=2)
    ax3.axvline(1/mu_min, color="#F0B429", linestyle="--", linewidth=2,
                label=f"Media = {1/mu_min:.1f} min")
    ax3.set_title("Tiempo de servicio")
    ax3.set_xlabel("Tiempo (min)")
    ax3.set_ylabel("f(x)")
    ax3.legend()
    ax3.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()


# TAB 3 - SIMULACION

with tab3:
    st.header("Construccion de la Simulacion")

    st.subheader("Como funciona la simulacion?")
    st.markdown("""
Se usa una tecnica llamada **Simulacion de Eventos Discretos**.
Los pasos son:

1. Generar tiempos de llegada con distribucion Exponencial
2. Generar tiempos de servicio con distribucion Exponencial
3. Asignar cada cliente al cajero disponible mas proximo
4. Registrar cuanto espero cada cliente
5. Calcular metricas del sistema
    """)

    st.code("""
# Pseudocodigo de la simulacion
Para cada cliente i:
    llegada[i]  = llegada[i-1] + Exponencial(lambda)
    servicio[i] = Exponencial(mu)
    
    cajero_libre = min(tiempo_libre_cajeros)
    inicio[i]   = max(llegada[i], cajero_libre)
    espera[i]   = inicio[i] - llegada[i]
    fin[i]      = inicio[i] + servicio[i]
    """, language="text")

    # Ejecutar simulacion
    np.random.seed(42)
    inter = -np.log(np.random.uniform(0, 1, n)) / (lam / 60)
    svcs  = -np.log(np.random.uniform(0, 1, n)) / (mu  / 60)
    arrs  = np.cumsum(inter)

    server_free = np.zeros(c)
    queue_len   = []
    records     = []

    for i in range(n):
        arr = arrs[i]
        svc = svcs[i]
        busy = int(np.sum(server_free > arr))
        queue_len.append(max(0, busy - c + 1))
        j     = int(np.argmin(server_free))
        start = max(arr, server_free[j])
        end   = start + svc
        server_free[j] = end
        records.append({
            "Cliente":        i + 1,
            "Llegada (min)":  round(arr,       2),
            "Inicio (min)":   round(start,     2),
            "Fin (min)":      round(end,        2),
            "Espera (min)":   round(start - arr,2),
            "Servicio (min)": round(svc,        2),
        })

    df = pd.DataFrame(records)
    df["queue"] = queue_len
    st.session_state["df"] = df

    # Metricas
    avg_wait = df["Espera (min)"].mean()
    avg_svc  = df["Servicio (min)"].mean()

    st.markdown("---")
    st.subheader("Metricas del sistema")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Utilizacion rho",     f"{rho:.3f}")
    c2.metric("Espera promedio (min)", f"{avg_wait:.2f}")
    c3.metric("Servicio prom. (min)", f"{avg_svc:.2f}")
    c4.metric("Tiempo en sistema (min)", f"{avg_wait + avg_svc:.2f}")

    if rho >= 1:
        st.error("Sistema inestable: la fila crece sin limite. Agrega cajeros.")
    elif rho > 0.85:
        st.warning(f"Alta utilizacion ({rho:.2f}). Considera un cajero mas en horas pico.")
    else:
        st.success(f"Sistema estable con utilizacion {rho:.2f}")

    st.markdown("---")
    st.subheader("Primeros 15 clientes simulados")

    def color_espera(val):
        if val > 10:   return "background-color: #FCEBEB"
        elif val > 5:  return "background-color: #FAEEDA"
        else:          return "background-color: #E1F5EE"

    styled = df.head(15)[["Cliente","Llegada (min)","Inicio (min)",
                           "Fin (min)","Espera (min)","Servicio (min)"]]\
               .style.map(color_espera, subset=["Espera (min)"])
    st.dataframe(styled, hide_index=True)
    st.caption("Verde: espera <= 5 min  |  Amarillo: 5-10 min  |  Rojo: > 10 min")

# TAB 4 - RESULTADOS
with tab4:
    st.header("Resultados y Analisis")

    df = st.session_state.get("df")
    if df is None:
        st.warning("Primero ve a la pestana Simulacion para generar los datos.")
        st.stop()

    wait_times = df["Espera (min)"].values
    svc_times  = df["Servicio (min)"].values

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Resultados de la Simulacion  |  lambda={lam}, mu={mu}, c={c} cajeros, n={n}",
                 fontsize=12, fontweight="bold")

    # Histograma espera
    ax = axes[0, 0]
    ax.hist(wait_times, bins=20, color="#378ADD", edgecolor="white")
    ax.axvline(np.mean(wait_times), color="#F0B429", linestyle="--",
               linewidth=2, label=f"Media {np.mean(wait_times):.1f} min")
    ax.set_title("Distribucion de Tiempos de Espera")
    ax.set_xlabel("Espera (min)")
    ax.set_ylabel("Frecuencia")
    ax.legend()
    ax.spines[["top","right"]].set_visible(False)

    # Histograma servicio
    ax = axes[0, 1]
    ax.hist(svc_times, bins=20, color="#1D9E75", edgecolor="white")
    ax.axvline(np.mean(svc_times), color="#F0B429", linestyle="--",
               linewidth=2, label=f"Media {np.mean(svc_times):.1f} min")
    ax.set_title("Distribucion de Tiempos de Servicio")
    ax.set_xlabel("Servicio (min)")
    ax.set_ylabel("Frecuencia")
    ax.legend()
    ax.spines[["top","right"]].set_visible(False)

    # Evolucion de la fila
    ax = axes[1, 0]
    ax.fill_between(range(len(df)), df["queue"], alpha=0.3, color="#1D9E75")
    ax.plot(df["queue"], color="#1D9E75", linewidth=1)
    ax.axhline(df["queue"].mean(), color="#F0B429", linestyle="--",
               linewidth=2, label=f"Media {df['queue'].mean():.2f}")
    ax.set_title("Longitud de la Fila a lo largo del Tiempo")
    ax.set_xlabel("Cliente #")
    ax.set_ylabel("Clientes en espera")
    ax.legend()
    ax.spines[["top","right"]].set_visible(False)

    # Boxplot
    ax = axes[1, 1]
    bp = ax.boxplot([wait_times, svc_times], patch_artist=True,
                    labels=["Espera", "Servicio"],
                    medianprops={"color": "#F0B429", "linewidth": 2})
    bp["boxes"][0].set_facecolor("#378ADD")
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor("#1D9E75")
    bp["boxes"][1].set_alpha(0.6)
    ax.set_title("Boxplot Comparativo")
    ax.set_ylabel("Tiempo (min)")
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Estadisticas
    st.markdown("---")
    st.subheader("Estadisticas descriptivas")

    stats = pd.DataFrame({
        "Metrica": ["Media", "Mediana", "Desv. Estandar", "Maximo",
                    "P(espera > 5 min)", "P(espera > 10 min)"],
        "Espera (min)": [
            f"{np.mean(wait_times):.2f}",
            f"{np.median(wait_times):.2f}",
            f"{np.std(wait_times):.2f}",
            f"{np.max(wait_times):.2f}",
            f"{np.mean(wait_times > 5)*100:.1f}%",
            f"{np.mean(wait_times > 10)*100:.1f}%",
        ],
        "Servicio (min)": [
            f"{np.mean(svc_times):.2f}",
            f"{np.median(svc_times):.2f}",
            f"{np.std(svc_times):.2f}",
            f"{np.max(svc_times):.2f}",
            "—", "—",
        ]
    })
    st.dataframe(stats, hide_index=True)

    # Conclusion
    st.markdown("---")
    st.subheader("Conclusion")
    p5  = np.mean(wait_times > 5)  * 100
    p10 = np.mean(wait_times > 10) * 100

    if rho >= 1:
        st.error(f"Sistema inestable (rho={rho:.2f}). La fila crece sin limite.")
    elif rho > 0.85:
        st.warning(f"Sistema con alta utilizacion (rho={rho:.2f}). "
                   f"El {p5:.1f}% espera mas de 5 min. "
                   f"Considerar un cajero adicional en horas pico.")
    else:
        st.success(f"Sistema estable (rho={rho:.2f}). "
                   f"Espera promedio: {np.mean(wait_times):.1f} min. "
                   f"Solo el {p5:.1f}% de los clientes espera mas de 5 min.")

    # Descarga
    csv = df[["Cliente","Llegada (min)","Inicio (min)",
              "Fin (min)","Espera (min)","Servicio (min)"]]\
            .to_csv(index=False).encode("utf-8")
    st.download_button("Descargar datos (.csv)", csv,
                       f"simulacion_lam{lam}_mu{mu}_c{c}.csv", "text/csv")
