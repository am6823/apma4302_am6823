#include <petsc.h>

int main(int argc, char **argv) {
  PetscMPIInt    rank, nP;
  PetscInt       N = 20, q, r, j, k, k_start, k_end; // defini entiers
  PetscReal      x = 1.0, y, localsum, globalsum, term; // defini nombre reels 

  PetscCall(PetscInitialize(&argc,&argv,NULL,
      "Compute exp(x) in parallel with PETSc.\n\n"));
  PetscCall(MPI_Comm_rank(PETSC_COMM_WORLD,&rank)); // & permet de modifier la variable
  PetscCall(MPI_Comm_size(PETSC_COMM_WORLD,&nP)); // stocke le nombre de processus

  // read option
  PetscOptionsBegin(PETSC_COMM_WORLD,"","options for expx","");
  PetscCall(PetscOptionsReal("-x","input to exp(x) function",NULL,x,&x,NULL));
  PetscCall(PetscOptionsInt("-N","number of terms in series",NULL,N,&N,NULL));
  PetscOptionsEnd();

  // determine work distribution
  q = N / nP; // nombre de termes par processus
  r = N % nP; // reste
  y = PetscAbsReal(x);

  k_start = rank*q + (rank < r ? rank : r); // indice de debut
  k_end   = k_start + q + (rank < r ? 1 : 0); // indice de fin

  // compute local sum
  localsum = 0.0;
  term = 1.0; // first term is always 1
  for (j = 0; j < k_start; j++) term *= y/(j+1);
    for (k = k_start; k < k_end; k++) {
      localsum += term;
      term *= y / (k + 1); 
    }

  // sum the contributions over all processes
  PetscCall(MPI_Reduce(&localsum,&globalsum,1,MPIU_REAL,MPIU_SUM, 0, PETSC_COMM_WORLD));

  if (!rank) {
  // 1) approx final (inversion si x < 0)
    PetscReal approx = globalsum;
    if (x < 0) approx = 1.0 / approx;
  // 2) exact = exp(x)
    PetscReal exact = PetscExpReal(x);
  // 3) relerr = |approx - exact|/|exact|
    PetscReal relerr = PetscAbsReal(approx - exact) / PetscAbsReal(exact);
  // 4) relerr_eps = relerr / PETSC_MACHINE_EPSILON
    PetscReal relerr_eps = relerr / PETSC_MACHINE_EPSILON;
  // 5) print
    PetscCall(PetscPrintf(PETSC_COMM_SELF,
        "Using %" PetscInt_FMT " terms to approximate exp(%.15e):\n",N,x));
    PetscCall(PetscPrintf(PETSC_COMM_SELF,
        "  exact = %.15e\n",exact));
    PetscCall(PetscPrintf(PETSC_COMM_SELF,
        "  approx = %.15e\n",approx));
    PetscCall(PetscPrintf(PETSC_COMM_SELF,
        "  relerr = %.15e\n",relerr));
    PetscCall(PetscPrintf(PETSC_COMM_SELF,
        "  relerr / EPS = %.15e\n",relerr_eps));
}


  PetscCall(PetscFinalize());
  return 0;
}
