# make            -> versions CPU (OpenMP), double precision
# make simple     -> version CPU en simple precision
# make gpu        -> versions GPU double et simple (NVIDIA HPC SDK : nvfortran, OpenMP offload)
# make gpu_test   -> etude de convergence executee sur GPU
FC      = gfortran
FFLAGS  ?= -O3 -march=native -fopenmp -cpp
DEBUG   = -O0 -g -fcheck=all -Wall -fopenmp -cpp
SRC     = src/precision.F90 src/solveur.F90

all: convergence performance

convergence: $(SRC) src/convergence.F90
	$(FC) $(FFLAGS) -o $@ $^

performance: $(SRC) src/performance.F90
	$(FC) $(FFLAGS) -o $@ $^

simple: $(SRC) src/performance.F90
	$(FC) $(FFLAGS) -DSINGLE -o performance_simple $^

debug: $(SRC) src/convergence.F90
	$(FC) $(DEBUG) -o convergence_debug $^

GPUFLAGS = -O3 -mp=gpu -Minfo=mp -cpp -DUSE_GPU

gpu: $(SRC) src/performance.F90
	nvfortran $(GPUFLAGS) -o performance_gpu $^
	nvfortran $(GPUFLAGS) -DSINGLE -o performance_gpu_simple $^

gpu_test: $(SRC) src/convergence.F90
	nvfortran $(GPUFLAGS) -o convergence_gpu $^
	./convergence_gpu

test: convergence
	./convergence

clean:
	rm -f convergence performance performance_simple performance_gpu performance_gpu_simple convergence_gpu convergence_debug *.mod

.PHONY: all simple debug gpu gpu_test test clean
