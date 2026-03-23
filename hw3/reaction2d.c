static char help[] = "2D nonlinear reaction-diffusion solver using SNES";

#include <petsc.h>

typedef struct {
    PetscReal gamma;
    PetscInt  p;
    PetscBool linear_f;
} AppCtx;

extern PetscReal     ufunction(PetscReal, PetscReal);
extern PetscReal     d2ufunction(PetscReal, PetscReal);
extern PetscErrorCode formExact(DM, Vec);

// Nonlinear residual and Jacobian
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*, PetscScalar**,
                                        PetscScalar**, AppCtx*);
extern PetscErrorCode FormJacobianLocal(DMDALocalInfo*, PetscScalar**,
                                        Mat, Mat, AppCtx*);

//STARTMAIN
int main(int argc, char **args) {
    DM            da;
    SNES          snes;         // replaces KSP from poisson2d.c
    AppCtx        user;
    Vec           u, uexact;
    PetscReal     errnorm, uexactnorm;
    DMDALocalInfo info;
    PetscViewer   viewer;

    PetscCall(PetscInitialize(&argc, &args, NULL, help));

    user.gamma    = 1.0;
    user.p        = 2;
    user.linear_f = PETSC_FALSE;

    PetscOptionsBegin(PETSC_COMM_WORLD, "rct_", "options for reaction2d", "");
    PetscCall(PetscOptionsReal("-gamma",    "reaction coefficient", "reaction2d.c", user.gamma,    &user.gamma,    NULL));
    PetscCall(PetscOptionsInt ("-p",        "reaction exponent",    "reaction2d.c", user.p,        &user.p,        NULL));
    PetscCall(PetscOptionsBool("-linear_f", "use linear RHS only",  "reaction2d.c", user.linear_f, &user.linear_f, NULL));
    PetscOptionsEnd();

    PetscCall(DMDACreate2d(PETSC_COMM_WORLD,
                 DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
                 9, 9, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
    PetscCall(DMSetFromOptions(da));
    PetscCall(DMSetUp(da));
    PetscCall(DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));
    PetscCall(DMSetApplicationContext(da, &user));

    // same vectors as poisson2d.c
    PetscCall(DMCreateGlobalVector(da, &u));
    PetscCall(VecDuplicate(u, &uexact));
    PetscCall(formExact(da, uexact));

    // initial guess: u = uexact
    PetscCall(VecCopy(uexact, u));

    // SNES replaces KSP from poisson2d.c
    PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
    PetscCall(SNESSetDM(snes, da));
    PetscCall(DMDASNESSetFunctionLocal(da, INSERT_VALUES,
                (DMDASNESFunctionFn *)FormFunctionLocal, &user));
    PetscCall(DMDASNESSetJacobianLocal(da,
                (DMDASNESJacobianFn *)FormJacobianLocal, &user));
    PetscCall(SNESSetFromOptions(snes));
    PetscCall(SNESSolve(snes, NULL, u));

    PetscCall(PetscViewerVTKOpen(PETSC_COMM_WORLD, "reaction2d.vtr", FILE_MODE_WRITE, &viewer));
    PetscCall(PetscObjectSetName((PetscObject)uexact, "uexact"));
    PetscCall(PetscObjectSetName((PetscObject)u,      "u"));
    PetscCall(VecView(uexact, viewer));
    PetscCall(VecView(u,      viewer));
    PetscCall(DMView(da,      viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    PetscCall(DMDAGetLocalInfo(da, &info));
    PetscCall(VecNorm(uexact, NORM_2, &uexactnorm));
    PetscCall(VecAXPY(u, -1.0, uexact));
    PetscCall(VecNorm(u, NORM_2, &errnorm));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "on %d x %d grid:  rel_error |u-uexact|_2/|uexact|_2 = %g\n",
        info.mx, info.my, (double)(errnorm/uexactnorm)));

    PetscCall(VecDestroy(&u));
    PetscCall(VecDestroy(&uexact));
    PetscCall(SNESDestroy(&snes));
    PetscCall(DMDestroy(&da));
    PetscCall(PetscFinalize());
    return 0;
}
//ENDMAIN

PetscReal ufunction(PetscReal x, PetscReal y) {
    PetscReal sigma = 0.3, x0 = 0.65, y0 = 0.65, amp = 1.0;
    PetscReal r2 = (x-x0)*(x-x0) + (y-y0)*(y-y0);
    return amp * PetscExpReal(-r2 / (sigma*sigma));
}

PetscReal d2ufunction(PetscReal x, PetscReal y) {
    PetscReal sigma = 0.3, x0 = 0.65, y0 = 0.65, amp = 1.0;
    PetscReal r2 = (x-x0)*(x-x0) + (y-y0)*(y-y0);
    PetscReal expterm = PetscExpReal(-r2 / (sigma*sigma));
    return amp * expterm * 4.0/(sigma*sigma) * (r2/(sigma*sigma) - 1.0);
}

//STARTEXACT
PetscErrorCode formExact(DM da, Vec uexact) {
    PetscInt      i, j;
    PetscReal     hx, hy, x, y, **auexact;
    DMDALocalInfo info;

    PetscCall(DMDAGetLocalInfo(da, &info));
    hx = 1.0/(info.mx-1);  hy = 1.0/(info.my-1);
    PetscCall(DMDAVecGetArray(da, uexact, &auexact));
    for (j = info.ys; j < info.ys+info.ym; j++) {
        y = j * hy;
        for (i = info.xs; i < info.xs+info.xm; i++) {
            x = i * hx;
            auexact[j][i] = ufunction(x, y);
        }
    }
    PetscCall(DMDAVecRestoreArray(da, uexact, &auexact));
    return 0;
}
//ENDEXACT

// replaces formMatrix + formRHS from poisson2d.c
// F(u) = 0 :
//   boundary : F = u - uexact(x,y)
//   interior : F = -lap(u) + gamma*u^p - f(x,y)
//STARTFUNCTION
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscScalar **u,
                                  PetscScalar **FF, AppCtx *user) {
    PetscInt  i, j;
    PetscReal hx, hy, x, y, f_rhs;
    PetscReal left, right, down, up, lap_u;

    hx = 1.0/(info->mx-1);  hy = 1.0/(info->my-1);

    for (j = info->ys; j < info->ys+info->ym; j++) {
        y = j * hy;
        for (i = info->xs; i < info->xs+info->xm; i++) {
            x = i * hx;
            if (i == 0 || i == info->mx-1 || j == 0 || j == info->my-1) {
                FF[j][i] = u[j][i] - ufunction(x, y);
            } else {
                // lift Dirichlet BCs into neighbor values
                left  = (i == 1)          ? ufunction(x-hx, y) : u[j][i-1];
                right = (i == info->mx-2) ? ufunction(x+hx, y) : u[j][i+1];
                down  = (j == 1)          ? ufunction(x, y-hy) : u[j-1][i];
                up    = (j == info->my-2) ? ufunction(x, y+hy) : u[j+1][i];

                lap_u = (left  - 2.0*u[j][i] + right) / (hx*hx)
                      + (down  - 2.0*u[j][i] + up)    / (hy*hy);

                f_rhs = -d2ufunction(x, y);
                if (!user->linear_f)
                    f_rhs += user->gamma * PetscPowReal(ufunction(x,y), (PetscReal)user->p);

                FF[j][i] = -lap_u + user->gamma * PetscPowReal(u[j][i], (PetscReal)user->p) - f_rhs;
            }
        }
    }
    return 0;
}
//ENDFUNCTION

// Jacobian of F(u)
//STARTJACOBIAN
PetscErrorCode FormJacobianLocal(DMDALocalInfo *info, PetscScalar **u,
                                  Mat J, Mat P, AppCtx *user) {
    PetscInt   i, j, ncols;
    PetscReal  hx, hy, dRdu;
    MatStencil row, col[5];
    PetscReal  v[5];

    hx = 1.0/(info->mx-1);  hy = 1.0/(info->my-1);

    for (j = info->ys; j < info->ys+info->ym; j++) {
        for (i = info->xs; i < info->xs+info->xm; i++) {
            row.j = j;  row.i = i;
            if (i == 0 || i == info->mx-1 || j == 0 || j == info->my-1) {
                col[0].j = j;  col[0].i = i;  v[0] = 1.0;
                PetscCall(MatSetValuesStencil(P, 1, &row, 1, col, v, INSERT_VALUES));
            } else {
                // same 5-point stencil as formMatrix in poisson2d.c
                // + reaction derivative on diagonal
                dRdu = user->gamma * (PetscReal)user->p
                     * PetscPowReal(u[j][i], (PetscReal)(user->p-1));

                ncols = 0;
                col[ncols].j = j;  col[ncols].i = i;
                v[ncols++] = 2.0/(hx*hx) + 2.0/(hy*hy) + dRdu;

                if (i-1 > 0)           { col[ncols].j = j;   col[ncols].i = i-1;  v[ncols++] = -1.0/(hx*hx); }
                if (i+1 < info->mx-1) { col[ncols].j = j;   col[ncols].i = i+1;  v[ncols++] = -1.0/(hx*hx); }
                if (j-1 > 0)           { col[ncols].j = j-1; col[ncols].i = i;    v[ncols++] = -1.0/(hy*hy); }
                if (j+1 < info->my-1) { col[ncols].j = j+1; col[ncols].i = i;    v[ncols++] = -1.0/(hy*hy); }

                PetscCall(MatSetValuesStencil(P, 1, &row, ncols, col, v, INSERT_VALUES));
            }
        }
    }
    PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
    if (J != P) {
        PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
    }
    return 0;
}
//ENDJACOBIAN