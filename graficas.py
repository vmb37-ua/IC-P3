# graficas de la tarea 4 a partir de resultados/pc/medidas.csv y del gprof
# python3 graficas.py   ->  deja los png en resultados/pc/graficas
import csv
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados", "pc")
OUT = os.path.join(RES, "graficas")
os.makedirs(OUT, exist_ok=True)

NOMBRES = {"A": "A (reparto de K)", "B": "B (reparto de fotos)", "E": "Especulativo"}
COLORES = {"A": "tab:blue", "B": "tab:orange", "E": "tab:green"}


def leer():
    filas = []
    with open(os.path.join(RES, "medidas.csv")) as f:
        for r in csv.DictReader(f, delimiter=";"):
            for c in ("procesos", "muestras", "ntest", "evaluadores", "k", "reinicios", "descartadas"):
                r[c] = int(r[c])
            for c in ("acierto", "carga", "ajuste", "test", "total", "comunicacion"):
                r[c] = float(r[c])
            filas.append(r)
    return filas


def talla(r):
    return (r["muestras"], r["ntest"])


def nombre_talla(t):
    return f"{t[0]} muestras / {t[1]} test"


def guardar(nombre):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, nombre), dpi=150)
    plt.close()
    print("hecha", nombre)


filas = leer()
# los del especulativo con evaluadores cambiados van aparte
normales = [r for r in filas if not (r["estrategia"] == "E" and r["evaluadores"] != max(1, r["procesos"] // 4))]
tallas = sorted(set(talla(r) for r in normales))

# tiempo secuencial de cada talla = estrategia A con 1 proceso
seq = {talla(r): r["total"] for r in normales if r["procesos"] == 1}

# tabla resumen para la memoria
with open(os.path.join(OUT, "tabla.csv"), "w") as f:
    f.write("estrategia;muestras;ntest;procesos;tiempo;speedup;eficiencia;comunicacion_%;k;acierto;reinicios\n")
    for r in sorted(normales, key=lambda r: (talla(r), r["estrategia"], r["procesos"])):
        t = talla(r)
        if t not in seq:
            continue
        s = seq[t] / r["total"]
        e = "Secuencial" if r["procesos"] == 1 else r["estrategia"]
        f.write(f"{e};{t[0]};{t[1]};{r['procesos']};{r['total']:.2f};{s:.2f};{s / r['procesos']:.2f};"
                f"{100 * r['comunicacion'] / r['total']:.1f};{r['k']};{r['acierto']:.4f};{r['reinicios']}\n")


def por_estrategia(t):
    d = defaultdict(list)
    for r in normales:
        if talla(r) == t:
            d[r["estrategia"]].append(r)
    for e in d:
        d[e].sort(key=lambda r: r["procesos"])
    # el punto de 1 proceso es el mismo para todas
    uno = [r for r in d["A"] if r["procesos"] == 1]
    for e in ("B", "E"):
        if e in d and uno and e != "E":
            d[e] = uno + d[e]
    return d


# 1) tiempos, speed-up y eficiencia (una columna por talla)
for tipo in ("tiempo", "speedup", "eficiencia"):
    fig, ejes = plt.subplots(1, len(tallas), figsize=(5 * len(tallas), 4), squeeze=False)
    for ax, t in zip(ejes[0], tallas):
        d = por_estrategia(t)
        maxp = 1
        for e in ("A", "B", "E"):
            if e not in d:
                continue
            p = [r["procesos"] for r in d[e]]
            maxp = max(maxp, max(p))
            if tipo == "tiempo":
                y = [r["total"] for r in d[e]]
            elif tipo == "speedup":
                y = [seq[t] / r["total"] for r in d[e]]
            else:
                y = [seq[t] / r["total"] / r["procesos"] for r in d[e]]
            ax.plot(p, y, "o-", label=NOMBRES[e], color=COLORES[e])
        if tipo == "speedup":
            ax.plot([1, maxp], [1, maxp], "k--", lw=1, label="ideal")
        if tipo == "eficiencia":
            ax.axhline(1, color="k", ls="--", lw=1)
            ax.set_ylim(0, 1.15)
        ax.set_title(nombre_talla(t))
        ax.set_xlabel("procesos MPI")
        ax.set_ylabel({"tiempo": "tiempo total (s)", "speedup": "speed-up", "eficiencia": "eficiencia"}[tipo])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    guardar(f"{tipo}.png")

# 2) speed-up de cada estrategia segun la talla (para ver como influye el tamaño)
fig, ejes = plt.subplots(1, 3, figsize=(15, 4))
for ax, e in zip(ejes, ("A", "B", "E")):
    for t in tallas:
        d = por_estrategia(t)
        if e not in d:
            continue
        ax.plot([r["procesos"] for r in d[e]], [seq[t] / r["total"] for r in d[e]], "o-", label=nombre_talla(t))
    ax.plot([1, 14], [1, 14], "k--", lw=1, label="ideal")
    ax.set_title(NOMBRES[e])
    ax.set_xlabel("procesos MPI")
    ax.set_ylabel("speed-up")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
guardar("speedup_por_talla.png")

# 3) computo frente a comunicacion+sincronizacion (talla mas grande)
t = tallas[-1]
d = por_estrategia(t)
fig, ejes = plt.subplots(1, 3, figsize=(15, 4))
for ax, e in zip(ejes, ("A", "B", "E")):
    if e not in d:
        continue
    rs = [r for r in d[e] if r["procesos"] > 1]
    p = [str(r["procesos"]) for r in rs]
    com = [r["comunicacion"] for r in rs]
    calc = [r["total"] - r["comunicacion"] - r["carga"] for r in rs]
    carga = [r["carga"] for r in rs]
    ax.bar(p, carga, label="carga de datos", color="tab:gray")
    ax.bar(p, calc, bottom=carga, label="cálculo", color=COLORES[e])
    ax.bar(p, com, bottom=[a + b for a, b in zip(carga, calc)], label="comunicación + espera", color="tab:red")
    for i, r in enumerate(rs):
        ax.text(i, r["total"], f"{100 * r['comunicacion'] / r['total']:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_title(f"{NOMBRES[e]} - {nombre_talla(t)}")
    ax.set_xlabel("procesos MPI")
    ax.set_ylabel("tiempo (s)")
    ax.legend(fontsize=8)
guardar("comunicacion.png")

# 4) especulativo cambiando los evaluadores (8 procesos)
ev = sorted([r for r in filas if r["estrategia"] == "E" and r["procesos"] == 8 and talla(r) == (10000, 2500)],
            key=lambda r: r["evaluadores"])
vistos = {}
for r in ev:
    vistos[r["evaluadores"]] = r       # si hay repetidos me quedo con el ultimo
ev = [vistos[k] for k in sorted(vistos)]
if ev:
    fig, ax = plt.subplots(figsize=(6, 4))
    x = [str(r["evaluadores"]) for r in ev]
    ax.bar(x, [r["total"] for r in ev], color="tab:green")
    for i, r in enumerate(ev):
        ax.text(i, r["total"], f"{r['reinicios']} reinicios", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("procesos evaluando el test (de 8)")
    ax.set_ylabel("tiempo total (s)")
    ax.set_title("Especulativo: reparto entre ajuste y evaluación")
    guardar("especulativo_evaluadores.png")


# 5) gprof: % de tiempo de las funciones con mas peso, -O0 frente a -O2
def leer_gprof(fich):
    funcs = {}
    if not os.path.exists(fich):
        return funcs
    with open(fich) as f:
        dentro = False
        for linea in f:
            if linea.strip().startswith("time   seconds"):
                dentro = True
                continue
            if dentro:
                if not linea.strip():
                    break
                m = re.match(r"\s*([\d.]+)\s+[\d.]+\s+([\d.]+)\s+(\d+)?\s*[\d.]*\s*[\d.]*\s+(.*)", linea)
                if m:
                    funcs[m.group(4).strip()] = (float(m.group(1)), int(m.group(3) or 0))
    return funcs


def corto(nombre):
    nombre = re.sub(r"\(.*\)( const)?$", "", nombre)
    nombre = nombre.replace("std::", "").replace("__gnu_cxx::", "").replace("knn::", "")
    nombre = re.sub(r"<.*>", "<...>", nombre)
    return nombre[:45]


g0 = leer_gprof(os.path.join(RES, "gprof", "perfil_O0.txt"))
g2 = leer_gprof(os.path.join(RES, "gprof", "perfil_O2.txt"))
if g0 or g2:
    top = sorted(set(list(g0)[:8] + list(g2)[:8]), key=lambda n: -max(g0.get(n, (0, 0))[0], g2.get(n, (0, 0))[0]))[:10]
    fig, ax = plt.subplots(figsize=(10, 5))
    y = range(len(top))
    ax.barh([i + 0.2 for i in y], [g0.get(n, (0, 0))[0] for n in top], height=0.4, label="-O0")
    ax.barh([i - 0.2 for i in y], [g2.get(n, (0, 0))[0] for n in top], height=0.4, label="-O2")
    ax.set_yticks(list(y))
    ax.set_yticklabels([corto(n) for n in top], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("% del tiempo de ejecución")
    ax.set_title("gprof del kNN secuencial")
    ax.legend()
    guardar("gprof.png")
    with open(os.path.join(OUT, "gprof_tabla.csv"), "w") as f:
        f.write("funcion;%O0;llamadas_O0;%O2;llamadas_O2\n")
        for n in top:
            a, b = g0.get(n, (0, 0)), g2.get(n, (0, 0))
            f.write(f"{corto(n)};{a[0]};{a[1]};{b[0]};{b[1]}\n")
