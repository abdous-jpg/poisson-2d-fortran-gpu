#!/bin/bash
# Lance les mesures CPU : 1 thread vs tous les coeurs, double vs simple precision.
# Usage : bash scripts/mesures.sh   (depuis la racine du depot, apres make && make simple)
set -e
ITERS=${ITERS:-500}
NMAX=$(nproc)
echo "precision,threads,n,iterations,ms_par_iteration,mpoints_par_s,octets_par_reel" > results/performance.csv
for prog in performance performance_simple; do
  prec=$([ "$prog" = performance ] && echo double || echo simple)
  for t in 1 $NMAX; do
    for n in 513 1025 2049; do
      ligne=$(OMP_NUM_THREADS=$t OMP_PROC_BIND=true ./$prog $n $ITERS)
      echo "$prec,$t,$ligne" | tee -a results/performance.csv
    done
  done
done
