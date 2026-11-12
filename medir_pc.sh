#!/bin/bash
# medidas de la tarea 4 en un solo pc (portatil, i7-12700H 14 nucleos)
# ./medir_pc.sh            todo (gprof + secuencial + barridos), unas 2 horas
# ./medir_pc.sh gprof|barrido|evaluadores    solo esa parte

set -u
RAIZ="$(cd "$(dirname "$0")" && pwd)"
B="$RAIZ/knn2/build"            # aqui estan train/ y test/
RES="${RES:-$RAIZ/resultados/pc}"   # RES=... ./medir_pc.sh para otra carpeta
PROCS="${PROCS:-2 3 4 6 8 10 12 14}"   # PROCS="2 3 4 6 8" en un pc de 8 nucleos
CSV="$RES/medidas.csv"
CABECERA="estrategia;procesos;muestras;maxK;folds;ntest;evaluadores;k;acierto;carga;ajuste;test;total;comunicacion;reinicios;descartadas"

mkdir -p "$RES/salidas" "$RES/gprof"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RES/log.txt"; }

compilar() {
    make -C "$RAIZ/knn2" knn_mpi cv gprof > /dev/null || exit 1
    cp "$RAIZ"/knn2/{knn_mpi,cv,cv_pg_O0,cv_pg_O2} "$B/"
}

entorno() {
    { lscpu; echo; lscpu -C; echo; free -h; echo; head -2 /etc/os-release; uname -r
      mpirun --version | head -1; g++ --version | head -1; } > "$RES/entorno.txt" 2>&1
}

gprof_y_secuencial() {
    log "secuencial original (-O2)"
    t=$(date +%s.%N)
    (cd "$B" && timeout -s KILL 1800 ./cv > "$RES/secuencial_original.txt" 2>&1)
    echo "tiempo real: $(awk -v a="$t" -v b="$(date +%s.%N)" 'BEGIN{print b-a}') s" >> "$RES/secuencial_original.txt"
    log "  $(tail -1 "$RES/secuencial_original.txt")"

    for v in O2 O0; do
        log "gprof -$v"
        (cd "$B" && rm -f gmon.out && timeout -s KILL 5400 ./cv_pg_$v > "$RES/gprof/salida_$v.txt" 2>&1 \
            && gprof -b ./cv_pg_$v gmon.out > "$RES/gprof/perfil_$v.txt")
        log "  hecho"
    done
}

# estrategia procesos muestras ntest [evaluadores]
ejecutar() {
    nombre="$1_np$2_$3_$4${5:+_ev$5}"
    salida=$(cd "$B" && timeout -s KILL 3600 mpirun -np "$2" ./knn_mpi "$1" "$3" 10 5 "$4" ${5:-} 2>&1)
    echo "$salida" > "$RES/salidas/$nombre.txt"
    linea=$(echo "$salida" | grep '^CSV;' | sed 's/^CSV;//')
    if [ -n "$linea" ]; then
        echo "$linea" >> "$CSV"
        log "  $nombre -> $(echo "$linea" | cut -d';' -f13) s"
    else
        log "  $nombre FALLO"
    fi
}

barrido() {
    log "talla $1 muestras / $2 test"
    ejecutar A 1 "$1" "$2"          # con 1 proceso = secuencial
    for p in $PROCS; do
        ejecutar A "$p" "$1" "$2"
        ejecutar B "$p" "$1" "$2"
        ejecutar E "$p" "$1" "$2"
    done
}

# cuantos procesos dedicar a evaluar en el especulativo
evaluadores() {
    log "especulativo con 8 procesos cambiando los evaluadores"
    for e in 1 2 3 4 6; do ejecutar E 8 10000 2500 "$e"; done
}

[ -f "$CSV" ] || echo "$CABECERA" > "$CSV"
compilar
case "${1:-todo}" in
    gprof)       gprof_y_secuencial ;;
    barrido)     barrido 5000 1000; barrido 10000 2500; barrido 20000 10000 ;;
    evaluadores) evaluadores ;;
    todo)        entorno; gprof_y_secuencial
                 barrido 5000 1000; barrido 10000 2500; evaluadores; barrido 20000 10000 ;;
esac
log "acabado"
