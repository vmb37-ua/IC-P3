#!/bin/bash
# todas las medidas de la P3 que van en la memoria (T0 matrices, gprof del kNN y T4)
# se lanza desde la carpeta del repo:   ./medir_todo.sh
# tarda unas 2-3 horas, no usar el pc mientras (cambia los tiempos)
# al final deja resultados_<usuario>.tar.gz, que es lo que hay que pasar

set -u
RAIZ="$(cd "$(dirname "$0")" && pwd)"
cd "$RAIZ"
export RES="$RAIZ/resultados/$(whoami)"
mkdir -p "$RES"

# --- comprobar que esta todo instalado ---
falta=""
for p in mpic++ mpirun g++ gprof cmake wget make; do
    command -v $p > /dev/null || falta="$falta $p"
done
if [ -n "$falta" ]; then
    echo "falta instalar:$falta"
    exit 1
fi

# --- procesos segun los nucleos fisicos del pc ---
nucleos=$(lscpu -p=core,socket | grep -v '^#' | sort -u | wc -l)
PROCS=""
for p in 2 3 4 6 8 10 12 14 16; do
    [ $p -le $nucleos ] && PROCS="$PROCS $p"
done
export PROCS
echo "nucleos fisicos: $nucleos -> procesos:$PROCS"

# --- dataset MNIST (solo la primera vez) ---
if [ ! -s knn2/build/test/t10k-images-idx3-ubyte ]; then
    echo "descargando MNIST..."
    ./knn2/get_dataset.sh > "$RES/dataset.txt" 2>&1 || { echo "fallo al bajar el dataset, mirar $RES/dataset.txt"; exit 1; }
fi

# --- T0 matrices: -O0, 3 veces cada talla ---
echo "T0 matrices..."
ulimit -s unlimited      # para 4000 y 8000 (8000x8000 int = 256 MB en la pila)
M="$RES/matrices.txt"
echo "talla;ejecucion;tiempo_A;tiempo_B;ganancia" > "$M"
for n in 500 1000 2000 4000 8000; do
    sed "s/const int SIZE = [0-9]*;/const int SIZE = $n;/" tarea0-matrices.cpp > /tmp/matrices_$n.cpp
    g++ -O0 -o /tmp/matrices_$n /tmp/matrices_$n.cpp || exit 1
    for i in 1 2 3; do
        salida=$(/tmp/matrices_$n)
        ta=$(echo "$salida" | sed -n 1p | grep -o '[0-9.e-]*s$' | tr -d s)
        tb=$(echo "$salida" | sed -n 2p | grep -o '[0-9.e-]*s$' | tr -d s)
        g=$(echo "$salida" | sed -n 3p | awk '{print $2}')
        echo "$n;$i;$ta;$tb;$g" >> "$M"
        echo "  $n x $n ($i): ganancia $g"
    done
    rm -f /tmp/matrices_$n /tmp/matrices_$n.cpp
done

# --- kNN: entorno, secuencial, gprof -O0/-O2, barridos y evaluadores ---
echo "kNN (esto es lo largo)..."
./medir_pc.sh todo

# --- empaquetar ---
cd "$RAIZ/resultados"
tar czf "$RAIZ/resultados_$(whoami).tar.gz" "$(whoami)"
echo
echo "HECHO -> $RAIZ/resultados_$(whoami).tar.gz"
