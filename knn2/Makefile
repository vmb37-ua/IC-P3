CXX = mpic++
CXXFLAGS = -O2 -std=c++17 -I.
COMUNES = knn.cc mnist.cc sample.cc
SECUENCIAL = main.cc crossValidation.cc $(COMUNES)

all: knn_mpi cv

# version paralela
knn_mpi: knn_mpi.cc $(COMUNES)
	$(CXX) $(CXXFLAGS) -o $@ knn_mpi.cc $(COMUNES)

# version secuencial del profe
cv: $(SECUENCIAL)
	g++ -O2 -std=c++17 -I. -o $@ $(SECUENCIAL)

# para gprof
gprof: cv_pg_O0 cv_pg_O2
cv_pg_O0: $(SECUENCIAL)
	g++ -O0 -g -pg -std=c++17 -I. -o $@ $(SECUENCIAL)
cv_pg_O2: $(SECUENCIAL)
	g++ -O2 -g -pg -std=c++17 -I. -o $@ $(SECUENCIAL)

clean:
	rm -f knn_mpi cv cv_pg_O0 cv_pg_O2 gmon.out
