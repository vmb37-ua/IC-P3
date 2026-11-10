#!/bin/bash
# script para el lab (practica 3)
# en el pc principal:
#   ./lab.sh todo IP2 IP3     prepara todo, hola mundo, info de los pcs y una ronda de medidas
#   ./lab.sh medir            otra ronda de medidas
#   ./lab.sh grande           medidas con 20000 muestras (tarda mucho)
#   ./lab.sh pen              comprime resultados/ y lo copia al pen
# en otro pc que no este en el cluster:
#   ./lab.sh aparte           tiempo del programa del profe y gprof
# tambien se puede: ./lab.sh preparar IP2 IP3 | hola | info

set -u
RAIZ="$(cd "$(dirname "$0")" && pwd)"
W="$HOME/ic_p3_lab"              # misma carpeta en todos los nodos
RES="$RAIZ/resultados"
HOSTS="$W/hosts"
MPIRUN_OPC="-mca plm_rsh_no_tree_spawn 1 --map-by node"
SSH_OPC="-o StrictHostKeyChecking=accept-new -o ConnectTimeout=5"
CABECERA="estrategia;procesos;muestras;maxK;folds;ntest;evaluadores;k;acierto;carga;ajuste;test;total;comunicacion;reinicios;descartadas"

mkdir -p "$RES"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RES/log.txt"; }

datos() {
    if [ ! -s "$RAIZ/knn2/build/train/train-images-idx3-ubyte" ]; then
        log "descargando MNIST"
        (cd "$RAIZ/knn2" && bash get_dataset.sh > /dev/null) || exit 1
    fi
    mkdir -p "$1/train" "$1/test"
    cp "$RAIZ"/knn2/build/train/*-ubyte "$1/train/"
    cp "$RAIZ"/knn2/build/test/*-ubyte "$1/test/"
}

preparar() {
    if [ $# -lt 2 ]; then
        echo "uso: ./lab.sh preparar IP2 IP3 ...  (las IPs de los OTROS PCs, minimo 2)"
        exit 1
    fi
    mkdir -p "$W"
    yo=$(hostname -I | awk '{print $1}')
    log "este PC: $yo, los otros: $*"

    # clave ssh y copiarla a los demas
    [ -f "$HOME/.ssh/id_rsa" ] || ssh-keygen -t rsa -N "" -f "$HOME/.ssh/id_rsa"
    # tambien a este mismo pc
    mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
    grep -qf "$HOME/.ssh/id_rsa.pub" "$HOME/.ssh/authorized_keys" 2>/dev/null || cat "$HOME/.ssh/id_rsa.pub" >> "$HOME/.ssh/authorized_keys"
    chmod 600 "$HOME/.ssh/authorized_keys"
    for ip in "$@"; do
        log "copiando clave a $ip"
        ssh-copy-id $SSH_OPC "$USER@$ip" || { echo "fallo con $ip"; exit 1; }
    done

    # fichero de hosts, slots = nucleos de cada pc
    : > "$HOSTS"
    for ip in "$yo" "$@"; do
        n=$(ssh $SSH_OPC "$ip" "lscpu -p=core | grep -v '^#' | sort -u | wc -l")
        echo "$ip slots=$n" >> "$HOSTS"
    done
    chmod 600 "$HOSTS"
    cp "$HOSTS" "$RES/hosts.txt"
    log "hosts:"; cat "$HOSTS"

    log "compilando"
    make -C "$RAIZ/knn2" knn_mpi > /dev/null || exit 1
    mpicc "$RAIZ/tarea2/hola.c" -o "$W/hola" || exit 1
    cp "$RAIZ/knn2/knn_mpi" "$W/"
    chmod o+x "$W/hola" "$W/knn_mpi"
    datos "$W"

    # misma carpeta en los otros nodos
    for ip in "$@"; do
        log "copiando a $ip"
        ssh $SSH_OPC "$ip" "mkdir -p $W" && scp -q -r $SSH_OPC "$W"/* "$ip:$W/" || { echo "fallo copiando a $ip"; exit 1; }
        ssh $SSH_OPC "$ip" "which mpirun >/dev/null" || echo "OJO: $ip no tiene mpirun"
    done
    log "preparado"
}

total_slots() { awk -F'slots=' '{s+=$2} END {print s}' "$HOSTS"; }

hola() {
    n=$(total_slots)
    log "hola mundo con $n procesos"
    (cd "$W" && mpirun $MPIRUN_OPC -hostfile hosts -n "$n" ./hola) | tee "$RES/hola.txt"
    log "guardado en resultados/hola.txt"
}

info() {
    f="$RES/entorno.txt"
    : > "$f"
    for ip in $(awk '{print $1}' "$HOSTS"); do
        log "info de $ip"
        {
            echo "===================== $ip"
            ssh $SSH_OPC "$ip" 'hostname; lscpu; echo; lscpu -C 2>/dev/null; echo; free -h; echo; head -2 /etc/os-release; mpirun --version | head -1; g++ --version | head -1; echo; ip -br addr; for i in /sys/class/net/*/speed; do echo "$i $(cat $i 2>/dev/null)"; done'
        } >> "$f" 2>&1
    done
    log "guardado en resultados/entorno.txt"
}

# una ejecucion: estrategia procesos muestras ntest
ejecutar() {
    nombre="$1_np$2_$3_$4_$(date +%H%M%S)"
    salida=$(cd "$W" && timeout -s KILL 5400 mpirun $MPIRUN_OPC -hostfile hosts -np "$2" ./knn_mpi "$1" "$3" 10 5 "$4" 2>&1)
    echo "$salida" > "$RES/salidas/$nombre.txt"
    linea=$(echo "$salida" | grep '^CSV;' | sed 's/^CSV;//')
    if [ -n "$linea" ]; then
        echo "$linea" >> "$RES/medidas.csv"
        log "  $1 np=$2 -> $(echo "$linea" | cut -d';' -f13) s (k=$(echo "$linea" | cut -d';' -f8))"
    else
        log "  $1 np=$2 FALLO (mira resultados/salidas/$nombre.txt)"
    fi
}

# todas las pruebas para una talla
barrido() {
    n=$(total_slots)
    log "talla $1 muestras / $2 de test"
    ejecutar A 1 "$1" "$2"          # con 1 proceso es la version secuencial
    for p in 2 3 4 6 8 12 16 24; do
        [ "$p" -le "$n" ] || continue
        ejecutar A "$p" "$1" "$2"
        ejecutar B "$p" "$1" "$2"
        ejecutar E "$p" "$1" "$2"
    done
}

medir() {
    mkdir -p "$RES/salidas"
    [ -f "$RES/medidas.csv" ] || echo "$CABECERA" > "$RES/medidas.csv"
    barrido 5000 1000
    barrido 10000 2500
    log "ronda acabada, todo en resultados/medidas.csv"
}

grande() {
    mkdir -p "$RES/salidas"
    [ -f "$RES/medidas.csv" ] || echo "$CABECERA" > "$RES/medidas.csv"
    barrido 20000 10000
    log "talla grande acabada"
}

# programa original del profe + gprof
aparte() {
    A="$HOME/ic_p3_aparte"
    mkdir -p "$A" "$RES/gprof"
    make -C "$RAIZ/knn2" cv gprof > /dev/null || exit 1
    datos "$A"
    (lscpu; echo; free -h) > "$RES/gprof/pc_aparte.txt"

    log "programa original (-O2), tarda unos 10 min"
    t=$(date +%s.%N)
    (cd "$A" && "$RAIZ/knn2/cv" > salida_cv.txt 2>&1)
    echo "tiempo real: $(awk -v a="$t" -v b="$(date +%s.%N)" 'BEGIN{print b-a}') s" >> "$A/salida_cv.txt"
    tr '\r' '\n' < "$A/salida_cv.txt" | grep -v '^[0-9]*/[0-9]*$' > "$RES/secuencial_original.txt"
    log "guardado en resultados/secuencial_original.txt"

    for v in O2 O0; do
        log "gprof -$v (O2 unos 10 min, O0 unos 40)"
        (cd "$A" && rm -f gmon.out && "$RAIZ/knn2/cv_pg_$v" > salida_$v.txt 2>&1 \
            && gprof -b "$RAIZ/knn2/cv_pg_$v" gmon.out > "$RES/gprof/perfil_$v.txt")
        tr '\r' '\n' < "$A/salida_$v.txt" | grep -v '^[0-9]*/[0-9]*$' > "$RES/gprof/salida_$v.txt"
        log "guardado en resultados/gprof/perfil_$v.txt"
    done
}

pen() {
    f="$RAIZ/resultados_$(hostname)_$(date +%Y%m%d_%H%M).tar.gz"
    tar -czf "$f" -C "$RAIZ" resultados
    log "empaquetado: $f"
    for d in /media/$USER/* /run/media/$USER/*; do
        [ -d "$d" ] && [ -w "$d" ] || continue
        cp "$f" "$d/" && sync && log "copiado al pen: $d/$(basename "$f")"
        return
    done
    echo "no encuentro el pen, copia a mano el archivo: $f"
}

case "${1:-}" in
    preparar) shift; preparar "$@" ;;
    hola)     hola ;;
    info)     info ;;
    medir)    medir ;;
    grande)   grande ;;
    aparte)   aparte ;;
    pen)      pen ;;
    todo)     shift; preparar "$@" && hola && info && medir ;;
    *)        sed -n 2,9p "$0" ;;
esac
