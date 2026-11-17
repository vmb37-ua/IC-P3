#!/bin/bash
# P3 de IC: todas las medidas de la memoria en el aula, con 3 PCs (MPICH, usuario usuariocp)
#
# En el PC "jefe", como usuariocp (desde alumno: su - usuariocp):
#   git clone https://github.com/vmb37-ua/IC-P3 ~/IC-P3 && cd ~/IC-P3
#   ./clase.sh IP2 IP3 [IP4]   IP2 e IP3 = los otros 2 PCs del cluster
#                              IP4 (opcional) = un 4o PC libre para gprof y secuenciales
#   ./clase.sh seguir          si se corta, sigue por donde iba (lo hecho no se repite)
#   ./clase.sh recoger         trae lo que hayan acabado los otros PCs (gprof...)
#   ./clase.sh pen             empaqueta resultados/clase en un .tar.gz (y lo copia al pen si puede)
#   ./clase.sh grande          extra: kNN con 20000 muestras (tarda mucho, solo si sobra tiempo)
#
# Que saca (todo en resultados/clase/):
#   entorno.txt, hola.txt          T2 y datos de los PCs y la red
#   knn/medidas.csv                T4: A, B y E con 1..3 nodos, 2 tallas, evaluadores del especulativo
#   campo/medidas.csv              T3: campo, campo_mpi y campo_mpi_func, series A y B, 1..3 nodos
#   aparte/*                       T0 (vectores y matrices), secuencial del profe, gprof kNN -O2/-O0, gprof campo

set -u
RAIZ="$(cd "$(dirname "$0")" && pwd)"
R="$RAIZ/resultados/clase"
B="$RAIZ/knn2/build"
CONF="$R/config"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes"
CABECERA="nodos;estrategia;procesos;muestras;maxK;folds;ntest;evaluadores;k;acierto;carga;ajuste;test;total;comunicacion;reinicios;descartadas"
PRUEBA="${PRUEBA:-0}"     # PRUEBA=1 ./clase.sh x  -> todo en este pc y con tallas pequeñas (para probar el script)

mkdir -p "$R"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$R/log.txt"; }
hecho() { [ -f "$R/.hecho_$1" ]; }
marcar() { touch "$R/.hecho_$1"; }

# --- lanzar mpirun igual con MPICH (aula) que con Open MPI (casa) ---
# prefijo FICHERO_HOSTS -> "mpirun ..." sin el -np
prefijo() {
    if [ "$MPI" = mpich ]; then
        echo "mpirun -f $1"
    else
        echo "mpirun --hostfile $1.ompi --map-by node -mca plm_rsh_no_tree_spawn 1"
    fi
}

# ejecutar algo en un nodo (en este pc sin ssh)
en_nodo() {
    local ip="$1"; shift
    if [ "$ip" = "$YO" ]; then bash -c "$*"; else $SSH "$ip" "$*"; fi
}

cargar_config() {
    if [ ! -f "$CONF" ]; then
        echo "no hay config, lanza primero: ./clase.sh IP2 IP3 [IP4]"
        exit 1
    fi
    . "$CONF"
}

# ------------------------------------------------------------------ preparar
preparar() {
    if [ "$PRUEBA" = 1 ]; then
        YO=localhost; OTROS=""; APARTE=""
    else
        if [ $# -lt 2 ]; then
            sed -n 2,12p "$0"; exit 1
        fi
        [ "$(whoami)" = usuariocp ] || echo "OJO: no eres usuariocp (en la T2 solo funcionaba con usuariocp: su - usuariocp)"
        YO=$(ip route get "$1" 2>/dev/null | grep -o 'src [0-9.]*' | awk '{print $2}')
        [ -n "$YO" ] || YO=$(hostname -I | awk '{print $1}')
        OTROS="$1 $2"
        APARTE="${3:-}"
    fi
    TODOS="$YO $OTROS"
    log "este PC: $YO   cluster: $TODOS   aparte: ${APARTE:-ninguno}"

    # ssh sin contraseña a todos (incluido este, mpirun tambien entra aqui por ssh)
    if [ "$PRUEBA" != 1 ]; then
        mkdir -p ~/.ssh && chmod 700 ~/.ssh
        for ip in $TODOS $APARTE; do
            ssh-keyscan -T 5 "$ip" >> ~/.ssh/known_hosts 2>/dev/null
            if ! $SSH "$ip" true; then
                echo "no entra por ssh sin contraseña a $ip"
                echo "  - mira que la IP esta bien (hostname -I en ese pc) y que alli tambien eres usuariocp"
                echo "  - si pide contraseña: ssh-keygen -t rsa (Enter a todo) y ssh-copy-id usuariocp@$ip"
                exit 1
            fi
            log "ssh ok con $ip ($($SSH "$ip" hostname))"
        done
    fi

    # que haya de todo
    for p in mpirun mpicc mpic++ g++ gprof python3 make; do
        command -v $p > /dev/null || { echo "falta $p en este pc"; exit 1; }
    done
    for ip in $OTROS; do
        $SSH "$ip" "command -v mpirun > /dev/null" || { echo "$ip no tiene mpirun"; exit 1; }
    done
    if mpirun --version 2>&1 | grep -qi "open mpi"; then MPI=openmpi; else MPI=mpich; fi
    log "MPI: $MPI"

    # nucleos fisicos (el minimo de los 3 pcs)
    C=99
    for ip in $TODOS; do
        n=$(en_nodo "$ip" "lscpu -p=core,socket | grep -v '^#' | sort -u | wc -l")
        log "  $ip: $n nucleos fisicos"
        [ "$n" -lt "$C" ] && C=$n
    done
    [ "$PRUEBA" = 1 ] && C=2

    # ficheros de hosts con 1, 2 y 3 nodos (MPICH reparte en orden: 1 proceso por pc y vuelta a empezar)
    i=0; : > "$R/hosts1"; : > "$R/hosts2"; : > "$R/hosts3"
    for ip in $TODOS; do
        i=$((i + 1))
        for n in 1 2 3; do [ $i -le $n ] && echo "$ip" >> "$R/hosts$n"; done
    done
    if [ "$PRUEBA" = 1 ]; then
        for n in 1 2 3; do echo localhost > "$R/hosts$n"; done
    fi
    for n in 1 2 3; do
        sed "s/\$/ slots=$C/" "$R/hosts$n" > "$R/hosts$n.ompi"
        [ "$PRUEBA" = 1 ] && echo "localhost slots=$((3 * C))" > "$R/hosts$n.ompi"
        chmod 600 "$R/hosts$n" "$R/hosts$n.ompi"
    done

    { echo "YO=\"$YO\""; echo "OTROS=\"$OTROS\""; echo "TODOS=\"$TODOS\""
      echo "APARTE=\"$APARTE\""; echo "MPI=$MPI"; echo "C=$C"; echo "PRUEBA=$PRUEBA"; } > "$CONF"

    # compilar todo aqui
    log "compilando"
    make -C "$RAIZ/knn2" knn_mpi cv gprof > "$R/compilar.txt" 2>&1 || { echo "fallo compilando el knn, mira $R/compilar.txt"; exit 1; }
    make -C "$RAIZ/campo" campo campo_mpi campo_mpi_func >> "$R/compilar.txt" 2>&1 || { echo "fallo compilando campo"; exit 1; }
    g++ -O2 -g -pg -std=c++11 -o "$RAIZ/campo/campo_pg_O2" "$RAIZ/campo/campo.cpp" &&
    g++ -O0 -g -pg -std=c++11 -o "$RAIZ/campo/campo_pg_O0" "$RAIZ/campo/campo.cpp" &&
    mpicc "$RAIZ/tarea2/hola.c" -o "$RAIZ/tarea2/hola" || { echo "fallo compilando"; exit 1; }

    # MNIST (sin cmake, que en el aula igual no esta)
    mkdir -p "$B/train" "$B/test"
    for f in train/train-images-idx3-ubyte train/train-labels-idx1-ubyte test/t10k-images-idx3-ubyte test/t10k-labels-idx1-ubyte; do
        [ -s "$B/$f" ] && continue
        log "bajando $(basename $f)"
        url="https://storage.googleapis.com/cvdf-datasets/mnist/$(basename $f).gz"
        if command -v wget > /dev/null; then wget -q -O "$B/$f.gz" "$url"; else curl -sfL -o "$B/$f.gz" "$url"; fi
        gzip -df "$B/$f.gz" || { echo "no se pudo bajar el MNIST (hay internet?)"; exit 1; }
    done
    cp "$RAIZ"/knn2/{knn_mpi,cv,cv_pg_O0,cv_pg_O2} "$B/"
    chmod o+x "$B/knn_mpi" "$RAIZ"/campo/campo* "$RAIZ/tarea2/hola"

    # misma carpeta en los otros pcs
    for ip in $OTROS $APARTE; do
        log "copiando la carpeta a $ip"
        tar czf - -C "$RAIZ" --exclude=./resultados --exclude=./.git . |
            $SSH "$ip" "mkdir -p '$RAIZ' && tar xzf - -C '$RAIZ' && mkdir -p '$R'" || { echo "fallo copiando a $ip"; exit 1; }
    done
    log "preparado (C=$C nucleos por pc, hasta $((3 * C)) procesos)"
}

# ------------------------------------------------------------------ entorno y T2
entorno() {
    f="$R/entorno.txt"; : > "$f"
    for ip in $TODOS $APARTE; do
        { echo "===================== $ip"
          en_nodo "$ip" 'hostname; lscpu; echo; free -h; echo; head -2 /etc/os-release; uname -r; mpirun --version 2>&1 | head -3; g++ --version | head -1; echo; ip -br addr; for i in /sys/class/net/*/speed; do echo "$i $(cat $i 2>/dev/null)"; done'
          echo; echo "--- ping desde el jefe"; ping -c 5 -q "$ip" 2>&1 | tail -2
        } >> "$f" 2>&1
    done
    log "entorno guardado"
    log "hola mundo (T2) con $((3 * C)) procesos"
    { echo "\$ $(prefijo "$R/hosts3") -np $((3 * C)) ./hola"
      cd "$RAIZ/tarea2" && timeout -s KILL 60 $(prefijo "$R/hosts3") -np $((3 * C)) ./hola; } 2>&1 | tee "$R/hola.txt"
}

# ------------------------------------------------------------------ kNN (T4)
lista_np() {   # procesos a probar segun los nucleos
    for p in 2 3 4 6 8 9 12 16 18 24 32 36 48 $C $((2 * C)) $((3 * C)); do
        [ "$p" -le $((3 * C)) ] && echo "$p"
    done | sort -nu | tr '\n' ' '
}

# estrategia procesos nodos muestras ntest [evaluadores]
knn() {
    mkdir -p "$R/knn/salidas"
    [ -f "$R/knn/medidas.csv" ] || echo "$CABECERA" > "$R/knn/medidas.csv"
    nombre="$1_np$2_nodos$3_$4_$5${6:+_ev$6}"
    f="$R/knn/salidas/$nombre.txt"
    if grep -q '^CSV;' "$f" 2>/dev/null; then return; fi   # ya hecho
    if [ "$2" = 1 ]; then pre="mpirun -np"; else pre="$(prefijo "$R/hosts$3") -np"; fi
    salida=$(cd "$B" && timeout -s KILL 1800 $pre "$2" ./knn_mpi "$1" "$4" 10 5 "$5" ${6:-} 2>&1)
    echo "$salida" > "$f"
    linea=$(echo "$salida" | grep '^CSV;' | sed 's/^CSV;//')
    if [ -n "$linea" ]; then
        echo "$3;$linea" >> "$R/knn/medidas.csv"
        log "  knn $nombre -> $(echo "$linea" | cut -d';' -f13) s"
    else
        rm -f "$f.fallo"; mv "$f" "$f.fallo"
        log "  knn $nombre FALLO (mira $f.fallo)"
    fi
}

knn_barrido() {   # muestras ntest
    log "kNN $1 muestras / $2 de test, 3 nodos"
    knn A 1 1 "$1" "$2"            # 1 proceso = secuencial
    for p in $(lista_np); do
        n=$((p < 3 ? p : 3))
        for e in A B E; do knn $e "$p" "$n" "$1" "$2"; done
    done
}

knn_nodos() {   # mismos procesos repartidos en 1, 2 y 3 pcs
    log "kNN cambiando el numero de nodos"
    for e in A B E; do
        knn $e "$C" 1 "$TM" "$TT"
        knn $e "$C" 2 "$TM" "$TT"
        knn $e $((2 * C)) 2 "$TM" "$TT"
    done
}

knn_evaluadores() {
    log "especulativo: cuantos procesos evaluan"
    for ev in 1 2 3 4 6; do
        [ "$ev" -lt $((3 * C)) ] && knn E $((3 * C)) 3 "$TM" "$TT" "$ev"
    done
}

# ------------------------------------------------------------------ campo (T3)
campo() {   # etiqueta nodos np... -- casos...
    et="$1"; n="$2"; shift 2
    nps=""; while [ "$1" != "--" ]; do nps="$nps $1"; shift; done; shift
    mkdir -p "$R/campo"
    (cd "$RAIZ/campo" && python3 scripts/medir.py --casos "$@" --np $nps --reps 2 \
        --hostfile "$R/hosts$n" --lanzador "$(prefijo "$R/hosts$n")" \
        --etiqueta "$et" --salida "$R/campo/medidas.csv") 2>&1 | tee -a "$R/campo/log.txt"
}

campo_todo() {
    log "campo (T3) series A y B con 3 nodos"
    nps=$(for p in 2 3 $C $((2 * C)) $((3 * C)); do echo $p; done | sort -nu | tr '\n' ' ')
    if [ "$PRUEBA" = 1 ]; then campo prueba 3 2 4 -- casos/espacio_2000.txt; return; fi
    campo clase-3nodos 3 $nps -- serieA serieB
}

campo_nodos() {
    log "campo cambiando el numero de nodos (espacio_9000)"
    [ "$PRUEBA" = 1 ] && return
    (cd "$RAIZ/campo" && python3 scripts/medir.py --programas campo_mpi campo_mpi_func --casos casos/espacio_9000.txt \
        --np "$C" --reps 2 --hostfile "$R/hosts1" --lanzador "$(prefijo "$R/hosts1")" --etiqueta clase-1nodo --salida "$R/campo/medidas.csv"
     python3 scripts/medir.py --programas campo_mpi campo_mpi_func --casos casos/espacio_9000.txt \
        --np "$C" $((2 * C)) --reps 2 --hostfile "$R/hosts2" --lanzador "$(prefijo "$R/hosts2")" --etiqueta clase-2nodos --salida "$R/campo/medidas.csv") \
        2>&1 | tee -a "$R/campo/log.txt"
}

# ------------------------------------------------------------------ cosas de un solo pc (se lanzan en otro pc, sin molestar al cluster)
trabajo() {
    d="$R/aparte/$1"; mkdir -p "$d"; cd "$d" || exit 1
    rm -f .fin
    { lscpu | grep -E 'Model name|^CPU\(s\)'; hostname; } > pc.txt
    case "$1" in
        t0)
            g++ -O0 -o vectores "$RAIZ/tarea0.cc"
            for i in 1 2 3; do echo "--- ejecucion $i"; ./vectores; done > vectores.txt 2>&1
            ulimit -s unlimited
            echo "talla;ejecucion;tiempo_A;tiempo_B;ganancia" > matrices.csv
            for n in 100 500 1000 2000 4000 8000; do
                sed "s/const int SIZE = [0-9]*;/const int SIZE = $n;/" "$RAIZ/tarea0-matrices.cpp" > m.cpp
                g++ -O0 -o m m.cpp
                for i in 1 2 3; do
                    s=$(./m)
                    echo "$n;$i;$(echo "$s" | sed -n 1p | grep -o '[0-9.e-]*s$' | tr -d s);$(echo "$s" | sed -n 2p | grep -o '[0-9.e-]*s$' | tr -d s);$(echo "$s" | sed -n 3p | awk '{print $2}')" >> matrices.csv
                done
            done
            rm -f m m.cpp vectores ;;
        secuencial)
            ln -sfn "$B/train" train; ln -sfn "$B/test" test
            t=$(date +%s.%N)
            timeout -s KILL 2400 "$B/cv" > salida.txt 2>&1
            echo "tiempo real: $(awk -v a="$t" -v b="$(date +%s.%N)" 'BEGIN{print b-a}') s" >> salida.txt
            tr '\r' '\n' < salida.txt | grep -v '^[0-9]*/[0-9]*$' > secuencial_original.txt; rm -f salida.txt ;;
        gprof_knn_O2|gprof_knn_O0)
            v=${1##*_}
            ln -sfn "$B/train" train; ln -sfn "$B/test" test
            rm -f gmon.out
            t=$(date +%s.%N)
            timeout -s KILL 7200 "$B/cv_pg_$v" > salida.txt 2>&1
            echo "tiempo real: $(awk -v a="$t" -v b="$(date +%s.%N)" 'BEGIN{print b-a}') s" >> salida.txt
            gprof -b "$B/cv_pg_$v" gmon.out > perfil_$v.txt 2>&1
            gprof -b -p "$B/cv_pg_$v" gmon.out > plano_$v.txt 2>&1
            tr '\r' '\n' < salida.txt | grep -v '^[0-9]*/[0-9]*$' > salida_$v.txt; rm -f salida.txt gmon.out ;;
        gprof_campo)
            for v in O2 O0; do
                rm -f gmon.out
                "$RAIZ/campo/campo_pg_$v" "$RAIZ/campo/espacio.txt" | grep -i tiempo > salida_$v.txt
                gprof -b "$RAIZ/campo/campo_pg_$v" gmon.out > perfil_$v.txt 2>&1
            done
            rm -f gmon.out ;;
    esac
    touch .fin
}

# lanza una lista de trabajos en un pc, en segundo plano
lanzar_trabajos() {
    ip="$1"; shift
    log "lanzando en $ip: $*"
    en_nodo "$ip" "cd '$RAIZ' && nohup ./clase.sh trabajos $* > '$R/trabajos_$ip.txt' 2>&1 < /dev/null &"
}

recoger() {
    for ip in $OTROS $APARTE; do
        $SSH "$ip" "[ -d '$R/aparte' ] && tar czf - -C '$R' aparte" 2>/dev/null | tar xzf - -C "$R" 2>/dev/null
    done
    for t in t0 secuencial gprof_campo gprof_knn_O2 gprof_knn_O0; do
        if [ -f "$R/aparte/$t/.fin" ]; then echo "  $t: hecho"; else echo "  $t: falta"; fi
    done
}

esperar_trabajos() {
    log "esperando a los trabajos de un solo pc (gprof -O0 es lo que mas tarda; Ctrl+C y luego ./clase.sh pen si no da tiempo)"
    while true; do
        sal=$(recoger)
        echo "$sal" | grep -q falta || break
        echo "[$(date +%H:%M)] $(echo "$sal" | grep -c hecho)/5 hechos"
        sleep 60
    done
    log "trabajos de un solo pc acabados"
}

pen() {
    [ -f "$CONF" ] && { . "$CONF"; recoger; }
    f="/tmp/resultados_clase_$(date +%Y%m%d_%H%M).tar.gz"
    tar czf "$f" -C "$RAIZ/resultados" clase && chmod 644 "$f"
    cp "$f" "$RAIZ/"
    log "empaquetado: $f (y copia en $RAIZ)"
    for d in /media/*/* /run/media/*/*; do
        [ -d "$d" ] && [ -w "$d" ] || continue
        cp "$f" "$d/" && sync && { log "copiado al pen: $d"; return; }
    done
    echo "no puedo escribir en el pen siendo $(whoami). Sal a la sesion de alumno (exit) y copia:"
    echo "   cp $f /media/alumno/<PEN>/"
}

# ------------------------------------------------------------------ todo seguido
fase() {   # nombre funcion args...
    local _fase="$1"; shift
    hecho "$_fase" && return
    "$@" && marcar "$_fase"
}

todo() {
    cargar_config
    TM=10000; TT=2500
    [ "$PRUEBA" = 1 ] && { TM=1000; TT=200; }
    inicio=$(date +%s)

    # con 4o pc: lo de un solo pc va alli desde ya
    if [ -n "$APARTE" ] && ! hecho aparte; then
        lanzar_trabajos "$APARTE" t0 secuencial gprof_campo gprof_knn_O2 gprof_knn_O0 && marcar aparte
    fi

    fase entorno entorno
    if [ "$PRUEBA" = 1 ]; then
        fase knn_peque knn_barrido 500 100
    else
        fase knn_5000 knn_barrido 5000 1000
    fi
    fase knn_10000 knn_barrido "$TM" "$TT"
    fase campo campo_todo
    fase knn_nodos knn_nodos
    fase campo_nodos campo_nodos
    fase knn_eval knn_evaluadores
    log "cluster acabado en $(( ($(date +%s) - inicio) / 60 )) min"

    # sin 4o pc: ahora que el cluster esta libre, uno en cada pc a la vez
    if [ -z "$APARTE" ] && ! hecho aparte; then
        if [ "$PRUEBA" = 1 ]; then
            lanzar_trabajos "$YO" t0 gprof_campo
        else
            set -- $OTROS
            lanzar_trabajos "$YO" t0 secuencial gprof_campo
            lanzar_trabajos "$1" gprof_knn_O2
            lanzar_trabajos "$2" gprof_knn_O0
        fi
        marcar aparte
    fi
    [ "$PRUEBA" = 1 ] && { sleep 20; recoger; } || esperar_trabajos
    pen
    log "TODO ACABADO en $(( ($(date +%s) - inicio) / 60 )) min"
}

case "${1:-}" in
    ""|-h|--help) sed -n 2,18p "$0" ;;
    seguir)   todo ;;
    recoger)  cargar_config; recoger ;;
    pen)      pen ;;
    grande)   cargar_config; knn_barrido 20000 10000 ;;
    trabajos) shift; cargar_config; for t in "$@"; do (trabajo "$t"); done ;;
    *)        preparar "$@" && todo ;;
esac
