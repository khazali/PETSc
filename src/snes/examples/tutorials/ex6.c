static char help[] = "Darcy Problem with simplicial finite elements.\n\
We solve the mixed-form Laplacian problem in a rectangular\n\
domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\n\n";

/*
The Darcy flow problem, which we discretize using the finite
element method on an unstructured mesh. The weak form equations are

  < v, u > + < \nabla\cdot v, p > = 0
  < q, \nabla\cdot u > + < q, f > = 0

We start with homogeneous Dirichlet conditions. We will expand this as the set
of test problems is developed.
*/

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>

typedef enum {NEUMANN, DIRICHLET} BCType;

typedef struct {
  /* Domain and mesh definition */
  PetscInt  dim;      /* The topological mesh dimension */
  PetscBool simplex;  /* Use simplices or tensor product cells */
  PetscInt  cells[3]; /* The number of faces in each direction */
  /* Problem definition */
  BCType    bcType;   /* Type of boundary conditions */
} AppCtx;

static PetscErrorCode zero_vector(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  PetscInt d;
  for (d = 0; d < dim; ++d) u[d] = 0.0;
  return 0;
}

static PetscErrorCode constant_p(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *p, void *ctx)
{
  *p = 1.0;
  return 0;
}

static void pressure(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar p[])
{
  p[0] = u[uOff[1]];
}

/*
  In 2D we use exact solution:

    u = 2x + y
    v = 2y + x
    p = x^2 + xy + y^2 - 11/12
    f = -4

  so that

    u - \nabla p       = <2x + y, 2y + x> - <2x + y, 2y + x> = 0
    \nabla \cdot u + f = 2 + 2 - 4                           = 0
*/
static PetscErrorCode linear_u_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 2.0*x[0] + x[1];
  u[1] = 2.0*x[1] + x[0];
  return 0;
}

static PetscErrorCode quadratic_p_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *p, void *ctx)
{
  *p = x[0]*x[0] + x[0]*x[1] + x[1]*x[1] - 11.0/12.0;
  return 0;
}

static void f0_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt c;
  for (c = 0; c < dim; ++c) f0[c] = u[c];
}

static void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt c;
  for (c = 0; c < dim*dim; ++c) f1[c] = 0.0;
  for (c = 0; c < dim; ++c) f1[c*dim+c] += u[uOff[1]];
}

static void f0_p(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  f0[0] = -4.0;
  for (d = 0; d < dim; ++d) f0[0] += u_x[d*dim+d];
}

static void f1_p(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = 0.0;
}

/* < q, \nabla\cdot u >
   NcompI = 1, NcompJ = dim */
static void g1_pu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g1[d*dim+d] = 1.0; /* \frac{\partial\phi^{u_d}}{\partial x_d} */
}

/* < \nabla\cdot v, p >
    NcompI = dim, NcompJ = 1 */
static void g2_up(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g2[d*dim+d] = 1.0; /* \frac{\partial\psi^{u_d}}{\partial x_d} */
}

/* < v, u > */
static void g0_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  PetscInt c;
  for (c = 0; c < dim; ++c) g0[c*dim+c] = 1.0;
}

/*
  In 3D we use exact solution:

    u = x^2 + y^2
    v = y^2 + z^2
    w = x^2 + y^2 - 2(x+y)z
    p = x + y + z - 3/2
    f_x = f_y = f_z = 3

  so that

    -\Delta u + \nabla p + f = <-4, -4, -4> + <1, 1, 1> + <3, 3, 3> = 0
    \nabla \cdot u           = 2x + 2y - 2(x + y)                   = 0
*/
static PetscErrorCode quadratic_u_3d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = x[0]*x[0] + x[1]*x[1];
  u[1] = x[1]*x[1] + x[2]*x[2];
  u[2] = x[0]*x[0] + x[1]*x[1] - 2.0*(x[0] + x[1])*x[2];
  return 0;
}

static PetscErrorCode linear_p_3d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *p, void *ctx)
{
  *p = x[0] + x[1] + x[2] - 1.5;
  return 0;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  const char    *bcTypes[2]  = {"neumann", "dirichlet"};
  PetscInt       n, bc;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->dim      = 2;
  options->simplex  = PETSC_TRUE;
  options->cells[0] = 3;
  options->cells[1] = 3;
  options->cells[2] = 3;
  options->bcType   = DIRICHLET;

  ierr = PetscOptionsBegin(comm, "", "Darcy Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex6.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Use simplices or tensor product cells", "ex6.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  n    = 3;
  ierr = PetscOptionsIntArray("-cells", "The number of faces in each direction", "ex12.c", options->cells, &n, NULL);CHKERRQ(ierr);
  bc   = options->bcType;
  ierr = PetscOptionsEList("-bc_type", "Type of boundary condition", "ex6.c", bcTypes, 2, bcTypes[options->bcType], &bc, NULL);CHKERRQ(ierr);
  options->bcType = (BCType) bc;
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexCreateBoxMesh(comm, user->dim, user->simplex, user->cells, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
  {
    DM               pdm = NULL;
    PetscPartitioner part;

    ierr = DMPlexGetPartitioner(*dm, &part);CHKERRQ(ierr);
    ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
    ierr = DMPlexDistribute(*dm, 0, NULL, &pdm);CHKERRQ(ierr);
    if (pdm) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = pdm;
    }
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupProblem(PetscDS prob, AppCtx *user)
{
  const PetscInt id = 1;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscDSSetResidual(prob, 0, f0_u, f1_u);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(prob, 1, f0_p, f1_p);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 0, g0_uu, NULL,  NULL,  NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 1, NULL,  NULL,  g2_up, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 1, 0, NULL,  g1_pu, NULL,  NULL);CHKERRQ(ierr);
  switch (user->dim) {
  case 2:
    ierr = PetscDSSetExactSolution(prob, 0, linear_u_2d);CHKERRQ(ierr);
    ierr = PetscDSSetExactSolution(prob, 1, quadratic_p_2d);CHKERRQ(ierr);
    ierr = PetscDSAddBoundary(prob, user->bcType == DIRICHLET ? DM_BC_ESSENTIAL : DM_BC_NATURAL, "wall", user->bcType == NEUMANN ? "boundary" : "marker", 0, 0, NULL, (void (*)(void)) linear_u_2d, 1, &id, user);CHKERRQ(ierr);
    break;
  case 3:
    ierr = PetscDSSetExactSolution(prob, 0, quadratic_u_3d);CHKERRQ(ierr);
    ierr = PetscDSSetExactSolution(prob, 1, linear_p_3d);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension %d", user->dim);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  DM              cdm   = dm;
  const PetscInt  dim   = user->dim;
  PetscFE         fe[2];
  PetscQuadrature q;
  PetscDS         prob;
  MPI_Comm        comm;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  /* Create finite element */
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(comm, dim, dim, user->simplex, "vel_", PETSC_DEFAULT, &fe[0]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[0], "velocity");CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe[0], &q);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(comm, dim, 1, user->simplex, "pres_", PETSC_DEFAULT, &fe[1]);CHKERRQ(ierr);
  ierr = PetscFESetQuadrature(fe[1], q);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[1], "pressure");CHKERRQ(ierr);
  /* Set discretization and boundary conditions for each mesh */
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) fe[0]);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 1, (PetscObject) fe[1]);CHKERRQ(ierr);
  ierr = SetupProblem(prob, user);CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject) prob, NULL, "-petscds_view");CHKERRQ(ierr);
  while (cdm) {
    ierr = DMSetDS(cdm, prob);CHKERRQ(ierr);
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe[0]);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe[1]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreatePressureNullSpace(DM dm, PetscInt dummy, MatNullSpace *nullSpace)
{
  Vec              vec;
  PetscErrorCode (*funcs[2])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void* ctx) = {zero_vector, constant_p};
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  ierr = DMCreateGlobalVector(dm, &vec);CHKERRQ(ierr);
  ierr = DMProjectFunction(dm, 0.0, funcs, NULL, INSERT_ALL_VALUES, vec);CHKERRQ(ierr);
  ierr = VecNormalize(vec, NULL);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) vec, "Pressure Null Space");CHKERRQ(ierr);
  ierr = VecViewFromOptions(vec, NULL, "-pressure_nullspace_view");CHKERRQ(ierr);
  ierr = MatNullSpaceCreate(PetscObjectComm((PetscObject)dm), PETSC_FALSE, 1, &vec, nullSpace);CHKERRQ(ierr);
  ierr = VecDestroy(&vec);CHKERRQ(ierr);
  /* New style for field null spaces */
  {
    PetscObject  pressure;
    MatNullSpace nullSpacePres;

    ierr = DMGetField(dm, 1, &pressure);CHKERRQ(ierr);
    ierr = MatNullSpaceCreate(PetscObjectComm(pressure), PETSC_TRUE, 0, NULL, &nullSpacePres);CHKERRQ(ierr);
    ierr = PetscObjectCompose(pressure, "nullspace", (PetscObject) nullSpacePres);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullSpacePres);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Add a vector in the nullspace to make the continuum integral 0.

   If int(u) = a and int(n) = b, then int(u - a/b n) = a - a/b b = 0
*/
static PetscErrorCode CorrectDiscretePressure(DM dm, MatNullSpace nullspace, Vec u, AppCtx *user)
{
  PetscDS        prob;
  const Vec     *nullvecs;
  PetscScalar    pintd, intc[2], intn[2];
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetObjective(prob, 1, pressure);CHKERRQ(ierr);
  ierr = MatNullSpaceGetVecs(nullspace, NULL, NULL, &nullvecs);CHKERRQ(ierr);
  ierr = VecDot(nullvecs[0], u, &pintd);CHKERRQ(ierr);
  if (PetscAbsScalar(pintd) > 1.0e-10) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Discrete integral of pressure: %g\n", (double) PetscRealPart(pintd));
  ierr = DMPlexComputeIntegralFEM(dm, nullvecs[0], intn, user);CHKERRQ(ierr);
  ierr = DMPlexComputeIntegralFEM(dm, u, intc, user);CHKERRQ(ierr);
  ierr = VecAXPY(u, -intc[1]/intn[1], nullvecs[0]);CHKERRQ(ierr);
  ierr = DMPlexComputeIntegralFEM(dm, u, intc, user);CHKERRQ(ierr);
  if (PetscAbsScalar(intc[1]) > 1.0e-10) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Continuum integral of pressure after correction: %g\n", (double) PetscRealPart(intc[1]));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESConvergenceCorrectPressure(SNES snes, PetscInt it, PetscReal xnorm, PetscReal gnorm, PetscReal f, SNESConvergedReason *reason, void *user)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = SNESConvergedDefault(snes, it, xnorm, gnorm, f, reason, user);CHKERRQ(ierr);
  if (*reason > 0) {
    DM           dm;
    Mat          J;
    Vec          u;
    MatNullSpace nullspace;

    ierr = SNESGetDM(snes, &dm);CHKERRQ(ierr);
    ierr = SNESGetSolution(snes, &u);CHKERRQ(ierr);
    ierr = SNESGetJacobian(snes, &J, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = MatGetNullSpace(J, &nullspace);CHKERRQ(ierr);
    ierr = CorrectDiscretePressure(dm, nullspace, u, user);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  SNES           snes;
  DM             dm;
  Vec            u;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, &user);CHKERRQ(ierr);

  ierr = SetupDiscretization(dm, &user);CHKERRQ(ierr);
  ierr = DMPlexCreateClosureIndex(dm, NULL);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);

  ierr = DMSetNullSpaceConstructor(dm, 2, CreatePressureNullSpace);CHKERRQ(ierr);
  ierr = SNESSetConvergenceTest(snes, SNESConvergenceCorrectPressure, &user, NULL);CHKERRQ(ierr);
  ierr = DMPlexSetSNESLocalFEM(dm,&user,&user,&user);CHKERRQ(ierr);

  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = DMSNESCheckFromOptions(snes, u, NULL, NULL);CHKERRQ(ierr);

  ierr = VecSet(u, 0.0);CHKERRQ(ierr);
  ierr = SNESSolve(snes, NULL, u);CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-sol_vec_view");CHKERRQ(ierr);
  {
    PetscErrorCode (*exactFuncs[2])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *);
    PetscDS   prob;
    PetscReal error = 0.0;
    PetscReal ferrors[2];

    ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
    ierr = PetscDSGetExactSolution(prob, 0, &exactFuncs[0]);CHKERRQ(ierr);
    ierr = PetscDSGetExactSolution(prob, 1, &exactFuncs[1]);CHKERRQ(ierr);
    ierr = DMComputeL2Diff(dm, 0.0, exactFuncs, NULL, u, &error);CHKERRQ(ierr);
    ierr = DMComputeL2FieldDiff(dm, 0.0, exactFuncs, NULL, u, ferrors);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: %g [%g, %g]\n", error, ferrors[0], ferrors[1]);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 2d_p2_p1
    requires: triangle
    args: -cells 2,2 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -dmsnes_check -ksp_error_if_not_converged -ksp_rtol 1e-10 -pc_type jacobi -petscds_view -snes_converged_reason
  test:
    suffix: 2d_p2_p1_conv
    requires: triangle
    args: -cells 2,2 -dm_refine 1 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -snes_convergence_estimate -num_refine 3 -ksp_error_if_not_converged -ksp_rtol 1e-10 -pc_type jacobi -petscds_view

TEST*/
