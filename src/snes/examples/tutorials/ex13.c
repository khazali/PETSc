static char help[] = "One-Shot Multigrid for an Optimal Control Problem for the Poisson Equation.\n\
http://www.math.cmu.edu/~shlomo/VKI-Lectures/lecture4/\n\
We use DMPlex, in 2d and 3d, with P1 Lagrange finite elements.\n\n\n";

/*F
  Let $\phi$ be the state variable, $\alpha$ the design variable and consider the minimization problem,
\begin{equation}
  \min_\alpha \frac{1}{2} \int_{\partial\Omega} (\frac{\partial\phi}{\partial n} - d)^2 ds
\end{equation}
where $\phi$ satisfies
\begin{align}
  \Delta \phi &= 0      & \Omega \\
  \phi        &= \alpha & \partial\Omega
\end{align}
and where  $\Omega = [0,1]\times[0,1]$. A simple calculation shows that the necessary (optimality) conditions are given by
\begin{align}
  \Delta \phi    &= 0      & \Omega \\
  \phi           &= \alpha & \partial\Omega \\
  \Delta \lambda &= 0      & \Omega \\
  \lambda        &= \alpha - \frac{\partial\phi}{\partial n} & \partial\Omega \\
  \frac{\partial\lambda}{\partial n} &= 0 & \partial\Omega.
\end{align}
F*/

#include <petscdmplex.h>
#include <petscsnes.h>

#define NUM_FIELDS 3
static PetscInt spatialDim = 0;

typedef struct {
  PetscFEM      fem;               /* REQUIRED to use DMPlexComputeResidualFEM() */
  PetscInt      debug;             /* The debugging level */
  PetscMPIInt   rank;              /* The process rank */
  PetscMPIInt   numProcs;          /* The number of processes */
  /* Domain and mesh definition */
  PetscInt      dim;               /* The topological mesh dimension */
  /* Element definition */
  PetscFE       fe[NUM_FIELDS];
  PetscFE       feBd[NUM_FIELDS];
  /* Problem definition */
  void (*f0Funcs[NUM_FIELDS])(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar f0[]); /* f0_u(x,y,z), and f0_p(x,y,z) */
  void (*f1Funcs[NUM_FIELDS])(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar f1[]); /* f1_u(x,y,z), and f1_p(x,y,z) */
  void (*g0Funcs[NUM_FIELDS*NUM_FIELDS])(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g0[]); /* g0_uu(x,y,z), g0_up(x,y,z), g0_pu(x,y,z), and g0_pp(x,y,z) */
  void (*g1Funcs[NUM_FIELDS*NUM_FIELDS])(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g1[]); /* g1_uu(x,y,z), g1_up(x,y,z), g1_pu(x,y,z), and g1_pp(x,y,z) */
  void (*g2Funcs[NUM_FIELDS*NUM_FIELDS])(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g2[]); /* g2_uu(x,y,z), g2_up(x,y,z), g2_pu(x,y,z), and g2_pp(x,y,z) */
  void (*g3Funcs[NUM_FIELDS*NUM_FIELDS])(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], PetscScalar g3[]); /* g3_uu(x,y,z), g3_up(x,y,z), g3_pu(x,y,z), and g3_pp(x,y,z) */
  void (**exactFuncs)(const PetscReal x[], PetscScalar *u, void *ctx); /* The exact solution function u(x,y,z), v(x,y,z), and p(x,y,z) */
  void (*f0BdFuncs[NUM_FIELDS])(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], const PetscReal n[], PetscScalar f0[]); /* f0_u(x,y,z), and f0_p(x,y,z) */
  void (*f1BdFuncs[NUM_FIELDS])(const PetscScalar u[], const PetscScalar gradU[], const PetscScalar a[], const PetscScalar gradA[], const PetscReal x[], const PetscReal n[], PetscScalar f1[]); /* f1_u(x,y,z), and f1_p(x,y,z) */
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->debug               = 0;
  options->dim                 = 2;

  options->fem.f0Funcs = (void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[])) &options->f0Funcs;
  options->fem.f1Funcs = (void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[])) &options->f1Funcs;
  options->fem.g0Funcs = (void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[])) &options->g0Funcs;
  options->fem.g1Funcs = (void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[])) &options->g1Funcs;
  options->fem.g2Funcs = (void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[])) &options->g2Funcs;
  options->fem.g3Funcs = (void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[])) &options->g3Funcs;
  options->fem.f0BdFuncs = (void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[])) &options->f0BdFuncs;
  options->fem.f1BdFuncs = (void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[])) &options->f1BdFuncs;

  ierr = MPI_Comm_size(comm, &options->numProcs);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &options->rank);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(comm, "", "Poisson Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex12.c", options->debug, &options->debug, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex12.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  spatialDim = options->dim;
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  DM             distributedMesh = NULL;
  DMLabel        label;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexCreateBoxMesh(comm, user->dim, PETSC_TRUE, dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  ierr = DMPlexGetLabel(*dm, "marker", &label);CHKERRQ(ierr);
  ierr = DMPlexLabelComplete(*dm, label);CHKERRQ(ierr);
  ierr = DMPlexDistribute(*dm, NULL, 0, NULL, &distributedMesh);CHKERRQ(ierr);
  if (distributedMesh) {
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = distributedMesh;
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupSection"
PetscErrorCode SetupSection(DM dm, AppCtx *user)
{
  PetscSection   section;
  DMLabel        label;
  PetscInt       dim         = user->dim;
  const char    *bdLabel     = "marker";
  PetscInt       numBC       = 3;
  PetscInt       bcFields[3] = {0, 1, 2};
  IS             bcPoints[3] = {NULL, NULL, NULL}, tmpIS;
  PetscInt       numComp[NUM_FIELDS];
  PetscInt      *numDof, pStart, pEnd, d, f;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscMalloc1(NUM_FIELDS*(dim+1),&numDof);CHKERRQ(ierr);
  for (f = 0; f < NUM_FIELDS; ++f) {
    const PetscInt *numFDof;
    ierr = PetscFEGetNumComponents(user->fe[f], &numComp[f]);CHKERRQ(ierr);
    ierr = PetscFEGetNumDof(user->fe[f], &numFDof);CHKERRQ(ierr);
    for (d = 0; d <= dim; ++d) numDof[f*(dim+1)+d] = numFDof[d];
  }
  ierr = DMPlexGetLabel(dm, bdLabel, &label);CHKERRQ(ierr);
  ierr = DMPlexLabelComplete(dm, label);CHKERRQ(ierr);
  ierr = DMPlexGetStratumIS(dm, bdLabel, 1, &bcPoints[0]);CHKERRQ(ierr);
  bcPoints[1] = bcPoints[0];
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF, pEnd-pStart, pStart, 1, &tmpIS);CHKERRQ(ierr);
  ierr = ISDifference(tmpIS, bcPoints[0], &bcPoints[2]);CHKERRQ(ierr);
  ierr = ISDestroy(&tmpIS);CHKERRQ(ierr);
  ierr = DMPlexCreateSection(dm, dim, NUM_FIELDS, numComp, numDof, numBC, bcFields, bcPoints, &section);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(section, 0, "state");CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(section, 1, "adjoint");CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(section, 2, "control");CHKERRQ(ierr);
  ierr = DMSetDefaultSection(dm, section);CHKERRQ(ierr);
  ierr = PetscSectionView(section, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  ierr = ISDestroy(&bcPoints[0]);CHKERRQ(ierr);
  ierr = ISDestroy(&bcPoints[2]);CHKERRQ(ierr);
  ierr = PetscFree(numDof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  DM             dm;          /* Problem specification */
  SNES           snes;        /* Nonlinear solver */
  AppCtx         user;        /* user-defined work context */
  PetscInt       f;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);

  ierr = PetscFECreateDefault(dm, user.dim, "state_",   -1, &user.fe[0]);
  ierr = PetscFECreateDefault(dm, user.dim, "adjoint_", -1, &user.fe[1]);
  ierr = PetscFECreateDefault(dm, user.dim, "control_", -1, &user.fe[2]);
  user.fem.fe = user.fe;
  ierr = SetupSection(dm, &user);CHKERRQ(ierr);

  for (f = 0; f < NUM_FIELDS; ++f) {ierr = PetscFEDestroy(&user.fe[f]);CHKERRQ(ierr);}
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
