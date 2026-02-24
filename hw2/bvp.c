#include <petsc.h>
#include <petscviewerhdf5.h>

static char help[] = "Solve 1D BVP -u'' + gamma u = f with Dirichlet BCs.\n";

int main(int argc, char **argv)
{
  Vec         u, f, uexact, e;
  Mat         A;
  KSP         ksp;
  PetscInt    m = 32;
  PetscInt    n;              
  PetscReal   gamma = 0.0;
  PetscInt    k = 1;
  PetscReal   c = 0.0;

  PetscInt    Istart, Iend, i;
  PetscReal   h, xi;

  PetscViewer viewer;
  PetscBool   write_hdf5 = PETSC_FALSE;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscOptionsBegin(PETSC_COMM_WORLD, "bvp_", "BVP options", "");
  PetscCall(PetscOptionsInt("-m", "number of subintervals (h=1/m)", "", m, &m, NULL));
  PetscCall(PetscOptionsReal("-gamma", "reaction coefficient gamma", "", gamma, &gamma, NULL));
  PetscCall(PetscOptionsInt("-k", "integer k in sin(k*pi*x)", "", k, &k, NULL));
  PetscCall(PetscOptionsReal("-c", "constant c in cubic term", "", c, &c, NULL));
  PetscCall(PetscOptionsBool("-write_hdf5", "write u, uexact, f to HDF5", "", write_hdf5, &write_hdf5, NULL));
  PetscOptionsEnd();

  n = m + 1;
  h = 1.0 / (PetscReal)m;

  //vectors
  PetscCall(VecCreate(PETSC_COMM_WORLD, &u));
  PetscCall(VecSetSizes(u, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(u));
  PetscCall(VecDuplicate(u, &f));
  PetscCall(VecDuplicate(u, &uexact));
  PetscCall(VecDuplicate(u, &e));

  //matrix
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  
  PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));


  for (i = Istart; i < Iend; i++) {
    xi = h * (PetscReal)i;

    // exact solution at all grid points
    PetscReal uex = PetscSinReal((PetscReal)k * PETSC_PI * xi)
                  + c * PetscPowReal(xi - 0.5, 3.0);
    PetscCall(VecSetValues(uexact, 1, &i, &uex, INSERT_VALUES));

    if (i == 0 || i == m) {
      PetscReal fi = 0.0;
      PetscCall(VecSetValues(f, 1, &i, &fi, INSERT_VALUES));
      continue;
    }

    PetscInt  cols[3] = {i-1, i, i+1};
    PetscReal vals[3];
    vals[0] = -1.0/(h*h);
    vals[1] =  2.0/(h*h) + gamma;
    vals[2] = -1.0/(h*h);
    PetscCall(MatSetValues(A, 1, &i, 3, cols, vals, INSERT_VALUES));

    PetscReal fi = (((PetscReal)k*PETSC_PI)*((PetscReal)k*PETSC_PI) + gamma)
                    * PetscSinReal((PetscReal)k * PETSC_PI * xi)
                 - 6.0*c*(xi - 0.5)
                 + gamma*c*PetscPowReal(xi - 0.5, 3.0);
    PetscCall(VecSetValues(f, 1, &i, &fi, INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(VecAssemblyBegin(uexact));
  PetscCall(VecAssemblyEnd(uexact));
  PetscCall(VecAssemblyBegin(f));
  PetscCall(VecAssemblyEnd(f));

  PetscInt bc[2] = {0, m};
  PetscCall(MatZeroRowsColumns(A, 2, bc, 1.0, uexact, f));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, f, u));

  PetscCall(VecCopy(u, e));
  PetscCall(VecAXPY(e, -1.0, uexact)); // e = u - uexact
  PetscReal nerr, nex;
  PetscCall(VecNorm(e, NORM_2, &nerr));
  PetscCall(VecNorm(uexact, NORM_2, &nex));
  PetscReal relerr = (nex > 0.0) ? (nerr / nex) : nerr;

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "m = %d (n=%d), h = %.3e, gamma=%.6g, k=%d, c=%.6g,  relative error = %.6e\n",
    (int)m, (int)n, (double)h, (double)gamma, (int)k, (double)c, (double)relerr));

  PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, "bvp_solution.h5", FILE_MODE_WRITE, &viewer));
  PetscCall(PetscObjectSetName((PetscObject) uexact, "uexact"));
  PetscCall(PetscObjectSetName((PetscObject) f, "f"));
  PetscCall(PetscObjectSetName((PetscObject) u, "u"));
  PetscCall(VecView(f, viewer));
  PetscCall(VecView(u, viewer));
  PetscCall(VecView(uexact, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  // cleanup
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&f));
  PetscCall(VecDestroy(&uexact));
  PetscCall(VecDestroy(&e));

  PetscCall(PetscFinalize());
  return 0;
}
