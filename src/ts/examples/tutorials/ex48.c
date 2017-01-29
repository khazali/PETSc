static char help[] = "Evolution of magnetic islands.\n\
The aim of this model is to self-consistently study the interaction between the tearing mode and small scale drift-wave turbulence.\n\n\n";

/*F
This is a three field model for the density $\tilde n$, vorticity $\tilde\Omega$, and magnetic flux $\tilde\psi$, using auxiliary variables potential $\tilde\phi$ and current $j_z$.
\begin{align}
  \frac{\partial \tilde n}{\partial t}      &= \left\{ \tilde n, \tilde\phi \right\} + \beta \left\{ j, \tilde\psi \right\} + \left\{ \ln n_0, \tilde\phi \right\} + \mu \nabla^2_\perp \tilde n \\
  \frac{\partial \tilde\Omega}{\partial t}   &= \left\{ \tilde\Omega, \tilde\phi \right\} + \beta \left\{ j, \tilde\psi \right\} + \mu \nabla^2_\perp \tilde\Omega \\
  \frac{\partial \tilde\psi}{\partial t}   &= \left\{ \psi, \tilde\phi - \tilde n \right\} - \left\{ \ln n_0, \tilde\psi \right\} + \frac{\eta}{\beta} \nabla^2_\perp \tilde\psi \\
  \nabla^2_\perp\phi        &= \Omega \\
  \nabla^2_\perp \tilde\psi &= -j_z
\end{align}
F*/

#include <petscdmplex.h>
#include <petscts.h>
#include <petscds.h>
#include <assert.h>

typedef struct {
  PetscInt       debug;             /* The debugging level */
  /* Domain and mesh definition */
  PetscInt       dim;               /* The topological mesh dimension */
  char           filename[2048];    /* The optional ExodusII file */
  PetscBool      simplex;           /* Simplicial mesh */
  /* Problem definition */
  PetscErrorCode (**exactFuncs)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
} AppCtx;

static PetscScalar poissonBracket(const PetscScalar df[], const PetscScalar dg[])
{
  return df[0]*dg[1] - df[1]*dg[0];
}

static void f0_n(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscScalar f0[])
{
  const PetscScalar *dn   = &u_x[uOff_x[0]];
  const PetscScalar *dpsi = &u_x[uOff_x[2]];
  const PetscScalar *dphi = &u_x[uOff_x[3]];
  const PetscScalar *djz  = &u_x[uOff_x[4]];
  const PetscScalar  beta = 1.0;
  PetscScalar        logRefDensityDer[2];

  f0[0] = u_t[0] - poissonBracket(dn, dphi) - beta*poissonBracket(djz, dpsi) - poissonBracket(logRefDensityDer, dphi);
}

static void f1_n(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscScalar f1[])
{
  const PetscScalar *dn = &u_x[uOff_x[0]];
  const PetscScalar  mu = 1.0;
  PetscInt           d;

  for (d = 0; d < dim-1; ++d) f1[d] = mu*dn[d];
}

static void f0_Omega(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, const PetscReal x[], PetscScalar f0[])
{
  const PetscScalar *dOmega = &u_x[uOff_x[1]];
  const PetscScalar *dpsi   = &u_x[uOff_x[2]];
  const PetscScalar *dphi   = &u_x[uOff_x[3]];
  const PetscScalar *djz    = &u_x[uOff_x[4]];
  const PetscScalar  beta   = 1.0;

  f0[0] = u_t[1] - poissonBracket(dOmega, dphi) - beta*poissonBracket(djz, dpsi);
}

static void f1_Omega(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, const PetscReal x[], PetscScalar f1[])
{
  const PetscScalar *dOmega = &u_x[uOff_x[1]];
  const PetscScalar  mu     = 1.0;
  PetscInt           d;

  for (d = 0; d < dim-1; ++d) f1[d] = mu*dOmega[d];
}

static void f0_psi(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                   const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                   PetscReal t, const PetscReal x[], PetscScalar f0[])
{
  const PetscScalar *dn   = &u_x[uOff_x[0]];
  const PetscScalar *dpsi = &u_x[uOff_x[2]];
  const PetscScalar *dphi = &u_x[uOff_x[3]];
  PetscScalar        logRefDensityDer[2];

  f0[0] = u_t[2] - poissonBracket(dpsi, dphi) + poissonBracket(dpsi, dn) - poissonBracket(logRefDensityDer, dpsi);
}

static void f1_psi(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                   const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                   PetscReal t, const PetscReal x[], PetscScalar f1[])
{
  const PetscScalar *dpsi = &u_x[uOff_x[2]];
  const PetscScalar  eta  = 1.0;
  const PetscScalar  beta = 1.0;
  PetscInt           d;

  for (d = 0; d < dim-1; ++d) f1[d] = (eta/beta)*dpsi[d];
}

static void f0_phi(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                   const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                   PetscReal t, const PetscReal x[], PetscScalar f0[])
{
  f0[0] = u[1];
}

static void f1_phi(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                   const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                   PetscReal t, const PetscReal x[], PetscScalar f1[])
{
  const PetscScalar *dphi = &u_x[uOff_x[3]];
  PetscInt           d;

  for (d = 0; d < dim-1; ++d) f1[d] = dphi[d];
}

static void f0_jz(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, const PetscReal x[], PetscScalar f0[])
{
  f0[0] = -u[4];
}

static void f1_jz(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, const PetscReal x[], PetscScalar f1[])
{
  const PetscScalar *dpsi = &u_x[uOff_x[2]];
  PetscInt           d;

  for (d = 0; d < dim-1; ++d) f1[d] = dpsi[d];
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->debug               = 0;
  options->dim                 = 2;
  options->filename[0]         = '\0';
  options->simplex             = PETSC_TRUE;

  ierr = PetscOptionsBegin(comm, "", "Poisson Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex12.c", options->debug, &options->debug, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex12.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-f", "Exodus.II filename to read", "ex12.c", options->filename, options->filename, sizeof(options->filename), &flg);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Simplicial (true) or tensor (false) mesh", "ex12.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PostStep"
static PetscErrorCode PostStep(TS ts)
{
  PetscErrorCode    ierr;
  DM                dm;
  AppCtx            *ctx;
  PetscInt          stepi;
  Vec               X;
  const             char prefix[] = "ex48";
  PetscViewer       viewer = NULL;
  char              buf[256];
  PetscBool         isHDF5,isVTK;
  PetscFunctionBegin;
  ierr = TSGetApplicationContext(ts, &ctx);CHKERRQ(ierr); assert(ctx);
  if (ctx->debug<1) PetscFunctionReturn(0);
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = TSGetSolution(ts, &X);CHKERRQ(ierr);
  ierr = TSGetTimeStepNumber(ts, &stepi);CHKERRQ(ierr);
  ierr = PetscViewerCreate(PetscObjectComm((PetscObject)dm),&viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer,PETSCVIEWERVTK);CHKERRQ(ierr);
  ierr = PetscViewerSetFromOptions(viewer);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,&isHDF5);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERVTK,&isVTK);CHKERRQ(ierr);
  if (isHDF5) {
    ierr = PetscSNPrintf(buf, 256, "%s-%d.h5", prefix, stepi);CHKERRQ(ierr);
  } else if (isVTK) {
    ierr = PetscSNPrintf(buf, 256, "%s-%d.vtu", prefix, stepi);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_VTK_VTU);CHKERRQ(ierr);
  }
  ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer,buf);CHKERRQ(ierr);
  if (isHDF5) {
    ierr = DMView(dm,viewer);CHKERRQ(ierr);
    ierr = PetscViewerFileSetMode(viewer,FILE_MODE_UPDATE);CHKERRQ(ierr);
  }
  /* view */
  ierr = VecView(X,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateBCLabel(DM dm, const char name[])
{
  DMLabel        label;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCreateLabel(dm, name);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, name, &label);CHKERRQ(ierr);
  ierr = DMPlexMarkBoundaryFaces(dm, label);CHKERRQ(ierr);
  ierr = DMPlexLabelComplete(dm, label);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim      = user->dim;
  const char    *filename = user->filename;
  size_t         len;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
  if (!len) {
    if (user->simplex) {
      ierr = DMPlexCreateBoxMesh(comm, dim, dim == 2 ? 2 : 1, PETSC_TRUE, dm);CHKERRQ(ierr);
    } else {
      PetscInt cells[3] = {1, 1, 1}; /* coarse mesh is one cell; refine from there */

      ierr = DMPlexCreateHexBoxMesh(comm, dim, cells,  DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, dm);CHKERRQ(ierr);
    }
    ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateFromFile(comm, filename, PETSC_TRUE, dm);CHKERRQ(ierr);
  }
  {
    DM distributedMesh = NULL;

    /* Distribute mesh over processes */
    ierr = DMPlexDistribute(*dm, 0, NULL, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = distributedMesh;
    }
  }
 {
    PetscBool hasLabel;

    ierr = DMHasLabel(*dm, "marker", &hasLabel);CHKERRQ(ierr);
    if (!hasLabel) {ierr = CreateBCLabel(*dm, "marker");CHKERRQ(ierr);}
  }
  {
    char      convType[256];
    PetscBool flg;

    ierr = PetscOptionsBegin(comm, "", "Mesh conversion options", "DMPLEX");CHKERRQ(ierr);
    ierr = PetscOptionsFList("-dm_plex_convert_type","Convert DMPlex to another format","ex12",DMList,DMPLEX,convType,256,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();
    if (flg) {
      DM dmConv;

      ierr = DMConvert(*dm,convType,&dmConv);CHKERRQ(ierr);
      if (dmConv) {
        ierr = DMDestroy(dm);CHKERRQ(ierr);
        *dm  = dmConv;
      }
    }
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode n_0(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return 0;
}

static PetscErrorCode Omega_0(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return 0;
}

static PetscErrorCode psi_0(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return 0;
}

static PetscErrorCode exactSolution_n(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return 0;
}

static PetscErrorCode exactSolution_Omega(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return 0;
}

static PetscErrorCode exactSolution_psi(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return 0;
}

static PetscErrorCode exactSolution_phi(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return 0;
}

static PetscErrorCode exactSolution_jz(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return 0;
}

static PetscErrorCode SetupProblem(PetscDS prob, AppCtx *user)
{
  const PetscInt id = 1;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscDSSetResidual(prob, 0, f0_n,     f1_n);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(prob, 1, f0_Omega, f1_Omega);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(prob, 2, f0_psi,   f1_psi);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(prob, 3, f0_phi,   f1_phi);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(prob, 4, f0_jz,    f1_jz);CHKERRQ(ierr);
  user->exactFuncs[0] = exactSolution_n;
  user->exactFuncs[1] = exactSolution_Omega;
  user->exactFuncs[2] = exactSolution_psi;
  user->exactFuncs[3] = exactSolution_phi;
  user->exactFuncs[4] = exactSolution_jz;
  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "wall", "marker", 0, 0, NULL, (void (*)()) user->exactFuncs[0], 1, &id, user);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupEquilibriumFields(DM dm, DM dmAux, AppCtx *user)
{
  PetscErrorCode (*eqFuncs[3])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx) = {n_0, Omega_0, psi_0};
  Vec            eq;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreateLocalVector(dmAux, &eq);CHKERRQ(ierr);
  ierr = DMProjectFunctionLocal(dmAux, 0.0, eqFuncs, NULL, INSERT_ALL_VALUES, eq);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) dm, "A", (PetscObject) eq);CHKERRQ(ierr);
  ierr = VecDestroy(&eq);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  DM              cdm = dm;
  const PetscInt  dim = user->dim;
  PetscQuadrature q;
  PetscFE         fe[5], feAux[3];
  PetscDS         prob, probAux;
  PetscInt        Nf = 5, NfAux = 3, f;
  PetscBool       simplex = user->simplex;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  /* Create finite element */
  ierr = PetscFECreateDefault(dm, dim, 1, simplex, "density_", -1, &fe[0]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[0], "density");CHKERRQ(ierr);
  ierr = PetscFECreateDefault(dm, dim, 1, simplex, "vorticity_", -1, &fe[1]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[1], "vorticity");CHKERRQ(ierr);
  ierr = PetscFECreateDefault(dm, dim, 1, simplex, "flux_", -1, &fe[2]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[2], "flux");CHKERRQ(ierr);
  ierr = PetscFECreateDefault(dm, dim, 1, simplex, "potential_", -1, &fe[3]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[3], "potential");CHKERRQ(ierr);
  ierr = PetscFECreateDefault(dm, dim, 1, simplex, "current_", -1, &fe[4]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[4], "current");CHKERRQ(ierr);

  ierr = PetscFECreateDefault(dm, dim, 1, simplex, "density_eq_", -1, &feAux[0]);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe[0], &q);CHKERRQ(ierr);
  ierr = PetscFESetQuadrature(feAux[0], q);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(dm, dim, 1, simplex, "vorticity_eq_", -1, &feAux[1]);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe[1], &q);CHKERRQ(ierr);
  ierr = PetscFESetQuadrature(feAux[1], q);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(dm, dim, 1, simplex, "flux_eq_", -1, &feAux[2]);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe[2], &q);CHKERRQ(ierr);
  ierr = PetscFESetQuadrature(feAux[2], q);CHKERRQ(ierr);
  /* Set discretization and boundary conditions for each mesh */
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {ierr = PetscDSSetDiscretization(prob, f, (PetscObject) fe[f]);CHKERRQ(ierr);}
  ierr = PetscDSCreate(PetscObjectComm((PetscObject) dm), &probAux);CHKERRQ(ierr);
  for (f = 0; f < NfAux; ++f) {ierr = PetscDSSetDiscretization(probAux, f, (PetscObject) feAux[f]);CHKERRQ(ierr);}
  ierr = SetupProblem(prob, user);CHKERRQ(ierr);
  while (cdm) {
    DM coordDM, dmAux;

    ierr = DMSetDS(cdm,prob);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(cdm,&coordDM);CHKERRQ(ierr);
    {
      PetscBool hasLabel;

      ierr = DMHasLabel(cdm, "marker", &hasLabel);CHKERRQ(ierr);
      if (!hasLabel) {ierr = CreateBCLabel(cdm, "marker");CHKERRQ(ierr);}
    }

    ierr = DMClone(cdm, &dmAux);CHKERRQ(ierr);
    ierr = DMSetCoordinateDM(dmAux, coordDM);CHKERRQ(ierr);
    ierr = DMSetDS(dmAux, probAux);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) dm, "dmAux", (PetscObject) dmAux);CHKERRQ(ierr);
    ierr = SetupEquilibriumFields(cdm, dmAux, user);CHKERRQ(ierr);
    ierr = DMDestroy(&dmAux);CHKERRQ(ierr);

    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  for (f = 0; f < Nf; ++f) {ierr = PetscFEDestroy(&fe[f]);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  TS             ts;
  Vec            u, r;
  AppCtx         ctx;
  PetscReal      t       = 0.0;
  PetscReal      L2error = 0.0;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &ctx);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &ctx, &dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, &ctx);CHKERRQ(ierr);
  ierr = PetscMalloc1(5, &ctx.exactFuncs);CHKERRQ(ierr);
  ierr = SetupDiscretization(dm, &ctx);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "mhd-");CHKERRQ(ierr);
  ierr = VecDuplicate(u, &r);CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD, &ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts, dm);CHKERRQ(ierr);
  ierr = TSSetApplicationContext(ts, &ctx);CHKERRQ(ierr);
  ierr = DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, &ctx);CHKERRQ(ierr);
  ierr = DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM, &ctx);CHKERRQ(ierr);
  ierr = DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, &ctx);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetPostStep(ts, PostStep);CHKERRQ(ierr);

  ierr = DMProjectFunction(dm, t, ctx.exactFuncs, NULL, INSERT_ALL_VALUES, u);CHKERRQ(ierr);
  ierr = TSSolve(ts, u);CHKERRQ(ierr);

  ierr = TSGetTime(ts, &t);CHKERRQ(ierr);
  ierr = DMComputeL2Diff(dm, t, ctx.exactFuncs, NULL, u, &L2error);CHKERRQ(ierr);
  if (L2error < 1.0e-11) {ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: < 1.0e-11\n");CHKERRQ(ierr);}
  else                   {ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: %g\n", L2error);CHKERRQ(ierr);}
  ierr = VecViewFromOptions(u, NULL, "-sol_vec_view");CHKERRQ(ierr);
#if 0
  {
    PetscReal res = 0.0;

    /* Check discretization error */
    ierr = VecViewFromOptions(u, NULL, "-initial_guess_view");CHKERRQ(ierr);
    ierr = DMComputeL2Diff(dm, 0.0, ctx.exactFuncs, NULL, u, &error);CHKERRQ(ierr);
    if (error < 1.0e-11) {ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: < 1.0e-11\n");CHKERRQ(ierr);}
    else                 {ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: %g\n", error);CHKERRQ(ierr);}
    /* Check residual */
    ierr = SNESComputeFunction(snes, u, r);CHKERRQ(ierr);
    ierr = VecChop(r, 1.0e-10);CHKERRQ(ierr);
    ierr = VecViewFromOptions(r, NULL, "-initial_residual_view");CHKERRQ(ierr);
    ierr = VecNorm(r, NORM_2, &res);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Residual: %g\n", res);CHKERRQ(ierr);
    /* Check Jacobian */
    {
      Mat A;
      Vec b;

      ierr = SNESGetJacobian(snes, &A, NULL, NULL, NULL);CHKERRQ(ierr);
      ierr = SNESComputeJacobian(snes, u, A, A);CHKERRQ(ierr);
      ierr = VecDuplicate(u, &b);CHKERRQ(ierr);
      ierr = VecSet(r, 0.0);CHKERRQ(ierr);
      ierr = SNESComputeFunction(snes, r, b);CHKERRQ(ierr);
      ierr = MatMult(A, u, r);CHKERRQ(ierr);
      ierr = VecAXPY(r, 1.0, b);CHKERRQ(ierr);
      ierr = VecDestroy(&b);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Au - b = Au + F(0)\n");CHKERRQ(ierr);
      ierr = VecChop(r, 1.0e-10);CHKERRQ(ierr);
      ierr = VecViewFromOptions(r, NULL, "-linear_residual_view");CHKERRQ(ierr);
      ierr = VecNorm(r, NORM_2, &res);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD, "Linear L_2 Residual: %g\n", res);CHKERRQ(ierr);
    }
  }
#endif

  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFree(ctx.exactFuncs);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0

TEST*/
