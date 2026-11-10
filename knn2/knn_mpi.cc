// knn con MPI - tarea 4
// uso: mpirun -np P ./knn_mpi A|B|E [muestras] [maxK] [folds] [ntest] [evaluadores]
// A = repartir los K entre procesos, B = repartir las fotos
// E = especulativo: un grupo busca el K y otro va evaluando el test con el mejor K que conoce

#include "sample.h"
#include "knn.h"
#include "mnist.h"

#include <mpi.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace knn;
using namespace mnist;

#define TAG_K 1
#define BLOQUE 50   // cada cuantas fotos miran los evaluadores si ha llegado otro K

int rank, nprocs;
double tiempoComunicacion = 0.0;

// igual que splitDataset del profe
void partirFold(const std::vector<TSample> &datos, int folds, int fold,
                std::vector<TSample> &train, std::vector<TSample> &validacion)
{
    int ini = datos.size() * fold / folds;
    int fin = datos.size() * (fold + 1) / folds;
    train.reserve(datos.size() - (fin - ini));
    validacion.reserve(fin - ini);
    for (int i = 0; i < (int)datos.size(); i++) {
        if (i < ini || i >= fin) train.push_back(datos[i]);
        else validacion.push_back(datos[i]);
    }
}

int aciertosEnRango(const KNN &clasificador, const std::vector<TSample> &fotos, int ini, int fin)
{
    int aciertos = 0;
    for (int i = ini; i < fin; i++)
        if (clasificador.classify(fotos[i]) == fotos[i].label)
            aciertos++;
    return aciertos;
}

int sumarAciertos(int misAciertos, MPI_Comm comm)
{
    int total = 0;
    double t = MPI_Wtime();
    MPI_Allreduce(&misAciertos, &total, 1, MPI_INT, MPI_SUM, comm);
    tiempoComunicacion += MPI_Wtime() - t;
    return total;
}

// trozo de fotos que le toca a cada proceso del comunicador
void miBloque(int n, MPI_Comm comm, int &ini, int &fin)
{
    int r, s;
    MPI_Comm_rank(comm, &r);
    MPI_Comm_size(comm, &s);
    ini = (long)n * r / s;
    fin = (long)n * (r + 1) / s;
}

// estrategia A: cada proceso hace los k = rank+1, rank+1+P, ...
int ajusteEstrategiaA(const std::vector<TSample> &datos, int maxK, int folds)
{
    struct { double acc; int k; } mio = { -1.0, 0 }, mejor;

    for (int k = 1 + rank; k <= maxK; k += nprocs) {
        double acc = 0.0;
        for (int f = 0; f < folds; f++) {
            std::vector<TSample> train, val;
            partirFold(datos, folds, f, train, val);
            KNN clasificador(train, k);
            acc += aciertosEnRango(clasificador, val, 0, val.size()) / (double)val.size();
        }
        acc /= folds;
        printf("[proceso %d] K: %d Accuracy: %.2f%%\n", rank, k, acc * 100);
        if (acc > mio.acc) { mio.acc = acc; mio.k = k; }
    }

    // maxloc devuelve el mejor acierto y su k
    double t = MPI_Wtime();
    MPI_Allreduce(&mio, &mejor, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
    tiempoComunicacion += MPI_Wtime() - t;
    return mejor.k;
}

// estrategia B: todos los del comunicador hacen todos los k, cada uno con su trozo de fotos
// si evaluador >= 0 (especulativo) el jefe del grupo le manda cada k que mejora y al final el definitivo
int ajusteEstrategiaB(const std::vector<TSample> &datos, int maxK, int folds, MPI_Comm comm, int evaluador)
{
    int r;
    MPI_Comm_rank(comm, &r);
    int mejorK = 1;
    double mejorAcc = 0.0;

    // buffers de los Isend, no se pueden tocar hasta el Waitall
    std::vector<int> mensajes(2 * (maxK + 1));
    std::vector<MPI_Request> peticiones;
    int nmsg = 0;

    for (int k = 1; k <= maxK; k++) {
        double acc = 0.0;
        for (int f = 0; f < folds; f++) {
            std::vector<TSample> train, val;
            partirFold(datos, folds, f, train, val);
            KNN clasificador(train, k);
            int ini, fin;
            miBloque(val.size(), comm, ini, fin);
            int total = sumarAciertos(aciertosEnRango(clasificador, val, ini, fin), comm);
            acc += total / (double)val.size();
        }
        acc /= folds;
        if (r == 0) printf("K: %d Accuracy: %.2f%%\n", k, acc * 100);
        if (acc > mejorAcc) {
            mejorAcc = acc;
            mejorK = k;
            if (evaluador >= 0 && r == 0) {
                int *m = &mensajes[2 * nmsg++];
                m[0] = k; m[1] = 0;            // {k, no es el final}
                MPI_Request p;
                MPI_Isend(m, 2, MPI_INT, evaluador, TAG_K, MPI_COMM_WORLD, &p);
                peticiones.push_back(p);
            }
        }
    }

    if (evaluador >= 0 && r == 0) {
        int *m = &mensajes[2 * nmsg++];
        m[0] = mejorK; m[1] = 1;               // señal de fin con el k definitivo
        MPI_Request p;
        MPI_Isend(m, 2, MPI_INT, evaluador, TAG_K, MPI_COMM_WORLD, &p);
        peticiones.push_back(p);
        double t = MPI_Wtime();
        MPI_Waitall(peticiones.size(), peticiones.data(), MPI_STATUSES_IGNORE);
        tiempoComunicacion += MPI_Wtime() - t;
    }
    return mejorK;
}

// especulativo, grupo de evaluacion: empieza con k = 1 y si llega un k mejor tira lo hecho y empieza otra vez
// el jefe del grupo es el que recibe los mensajes y se lo pasa a los demas con un Bcast cada BLOQUE fotos
int evaluacionEspeculativa(const std::vector<TSample> &train, const std::vector<TSample> &test, int ntest,
                           MPI_Comm comm, int jefeAjuste, int &k, int &reinicios, long &descartadas)
{
    int r, s;
    MPI_Comm_rank(comm, &r);
    MPI_Comm_size(comm, &s);
    int estado[2] = { 1, 0 };   // {mejor k conocido, ha llegado el fin}
    k = 1;
    reinicios = 0;
    descartadas = 0;

    // todos tienen que hacer el mismo numero de pasos para que cuadren los Bcast
    int maxPorProceso = (ntest + s - 1) / s;
    int pasos = (maxPorProceso + BLOQUE - 1) / BLOQUE;

    while (true) {
        KNN clasificador(train, k);
        int ini, fin;
        miBloque(ntest, comm, ini, fin);
        int aciertos = 0, hechas = 0;
        bool cambia = false;

        for (int p = 0; p < pasos && !cambia; p++) {
            int a = std::min(ini + p * BLOQUE, fin), b = std::min(a + BLOQUE, fin);
            aciertos += aciertosEnRango(clasificador, test, a, b);
            hechas += b - a;

            double t = MPI_Wtime();
            if (r == 0) {
                int hay = 1;
                while (hay) {
                    MPI_Status st;
                    MPI_Iprobe(jefeAjuste, TAG_K, MPI_COMM_WORLD, &hay, &st);
                    if (hay) MPI_Recv(estado, 2, MPI_INT, jefeAjuste, TAG_K, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
            MPI_Bcast(estado, 2, MPI_INT, 0, comm);
            tiempoComunicacion += MPI_Wtime() - t;
            if (estado[0] != k) cambia = true;
        }

        if (!cambia) {
            int total = sumarAciertos(aciertos, comm);
            // acabado el test; si el ajuste no ha terminado hay que esperar a ver si cambia el k
            double t = MPI_Wtime();
            while (!estado[1] && estado[0] == k) {
                if (r == 0) MPI_Recv(estado, 2, MPI_INT, jefeAjuste, TAG_K, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Bcast(estado, 2, MPI_INT, 0, comm);
            }
            tiempoComunicacion += MPI_Wtime() - t;
            if (estado[0] == k) return total;   // el k con el que hemos evaluado es el bueno
        }

        // hay otro k mejor: se tira lo hecho
        descartadas += hechas;
        reinicios++;
        if (r == 0) printf("[evaluacion] reinicio: k %d -> %d\n", k, estado[0]);
        k = estado[0];
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    char estrategia = argc > 1 ? argv[1][0] : ' ';
    if (estrategia != 'A' && estrategia != 'B' && estrategia != 'E') {
        if (rank == 0) printf("uso: mpirun -np P ./knn_mpi A|B|E [muestras] [maxK] [folds] [ntest] [evaluadores]\n");
        MPI_Finalize();
        return 1;
    }
    int muestras = argc > 2 ? atoi(argv[2]) : 20000;
    int maxK     = argc > 3 ? atoi(argv[3]) : 10;
    int folds    = argc > 4 ? atoi(argv[4]) : 5;
    int ntest    = argc > 5 ? atoi(argv[5]) : 10000;
    int nEval    = argc > 6 ? atoi(argv[6]) : std::max(1, nprocs / 4);
    if (estrategia == 'E' && (nprocs < 2 || nEval < 1 || nEval >= nprocs)) {
        if (rank == 0) printf("el especulativo necesita al menos 2 procesos y 1 <= evaluadores < procesos\n");
        MPI_Finalize();
        return 1;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double tInicio = MPI_Wtime();

    // cada proceso lee los datos de su disco
    std::vector<TSample> train, test;
    if (!loadMnist("train/train-images-idx3-ubyte", "train/train-labels-idx1-ubyte", train) ||
        !loadMnist("test/t10k-images-idx3-ubyte", "test/t10k-labels-idx1-ubyte", test)) {
        printf("[proceso %d] no se pueden leer los datos\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    muestras = std::min<int>(muestras, train.size());
    ntest = std::min<int>(ntest, test.size());
    std::vector<TSample> datosAjuste(train.begin(), train.begin() + muestras);
    double tCarga = MPI_Wtime();

    int k = 0, aciertos = 0, reinicios = 0;
    long descartadas = 0;
    double tAjuste = 0, tTest = 0;

    if (estrategia != 'E') {
        k = estrategia == 'A' ? ajusteEstrategiaA(datosAjuste, maxK, folds)
                              : ajusteEstrategiaB(datosAjuste, maxK, folds, MPI_COMM_WORLD, -1);
        tAjuste = MPI_Wtime() - tCarga;
        if (rank == 0) printf("Best k: %d\n", k);

        // test final repartido por fotos
        double t0 = MPI_Wtime();
        KNN clasificador(train, k);
        int ini, fin;
        miBloque(ntest, MPI_COMM_WORLD, ini, fin);
        aciertos = sumarAciertos(aciertosEnRango(clasificador, test, ini, fin), MPI_COMM_WORLD);
        tTest = MPI_Wtime() - t0;
    } else {
        // los primeros nprocs-nEval hacen el ajuste, el resto la evaluacion
        int nAjuste = nprocs - nEval;
        int grupo = rank < nAjuste ? 0 : 1;
        MPI_Comm comm;
        MPI_Comm_split(MPI_COMM_WORLD, grupo, rank, &comm);

        if (grupo == 0) {
            k = ajusteEstrategiaB(datosAjuste, maxK, folds, comm, nAjuste);
            tAjuste = MPI_Wtime() - tCarga;
        } else {
            aciertos = evaluacionEspeculativa(train, test, ntest, comm, 0, k, reinicios, descartadas);
            tTest = MPI_Wtime() - tCarga;
        }

        // el jefe de la evaluacion le pasa el resultado al proceso 0 para que lo imprima
        long res[4] = { k, aciertos, reinicios, 0 };
        long descTotal = 0;
        if (grupo == 1) MPI_Reduce(&descartadas, &descTotal, 1, MPI_LONG, MPI_SUM, 0, comm);
        res[3] = descTotal;
        if (rank == nAjuste) MPI_Send(res, 4, MPI_LONG, 0, 2, MPI_COMM_WORLD);
        if (rank == 0) {
            MPI_Recv(res, 4, MPI_LONG, nAjuste, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Best k: %d (ajuste %d, evaluacion %ld) reinicios: %ld fotos descartadas: %ld\n",
                   k, nAjuste, (long)nEval, res[2], res[3]);
            if (res[0] != k) printf("OJO: la evaluacion ha usado k=%ld\n", res[0]);
            aciertos = res[1]; reinicios = res[2]; descartadas = res[3];
        }
        MPI_Comm_free(&comm);
    }
    double tFin = MPI_Wtime();

    // nos quedamos con el tiempo del proceso mas lento
    double mios[5] = { tCarga - tInicio, tAjuste, tTest, tFin - tInicio, tiempoComunicacion };
    double peores[5];
    MPI_Reduce(mios, peores, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Test accuracy: %.4f\n", aciertos / (double)ntest);
        printf("carga %.3f s | ajuste %.3f s | test %.3f s | total %.3f s | comunicacion %.3f s\n",
               peores[0], peores[1], peores[2], peores[3], peores[4]);
        // estrategia;procesos;muestras;maxK;folds;ntest;evaluadores;k;acierto;carga;ajuste;test;total;comunicacion;reinicios;descartadas
        printf("CSV;%c;%d;%d;%d;%d;%d;%d;%d;%.4f;%.3f;%.3f;%.3f;%.3f;%.3f;%d;%ld\n", estrategia, nprocs, muestras, maxK, folds,
               ntest, estrategia == 'E' ? nEval : 0, k, aciertos / (double)ntest,
               peores[0], peores[1], peores[2], peores[3], peores[4], reinicios, descartadas);
    }

    MPI_Finalize();
    return 0;
}
