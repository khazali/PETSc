static char help[] = "Evolution of magnetic islands.\n\
The aim of this model is to self-consistently study the interaction between the tearing mode and small scale drift-wave turbulence.\n\n\n";

/*F
This is a three field model for the density $\tilde n$, vorticity $\tilde\Omega$, and magnetic flux $\tilde\psi$, using auxiliary variables potential $\tilde\phi$ and current $j_z$.
\begin{equation}
  \begin{aligned}
    \partial_t \tilde n       &= \left\{ \tilde n, \tilde\phi \right\} + \beta \left\{ j_z, \tilde\psi \right\} + \left\{ \ln n_0, \tilde\phi \right\} + \mu \nabla^2_\perp \tilde n \\
  \partial_t \tilde\Omega   &= \left\{ \tilde\Omega, \tilde\phi \right\} + \beta \left\{ j_z, \tilde\psi \right\} + \mu \nabla^2_\perp \tilde\Omega \\
  \partial_t \tilde\psi     &= \left\{ \psi_0 + \tilde\psi, \tilde\phi - \tilde n \right\} - \left\{ \ln n_0, \tilde\psi \right\} + \frac{\eta}{\beta} \nabla^2_\perp \tilde\psi \\
  \nabla^2_\perp\tilde\phi        &= \tilde\Omega \\
  j_z  &= -\nabla^2_\perp  \left(\tilde\psi + \psi_0  \right)\\
  \end{aligned}
\end{equation}
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
  PetscBool      cell_simplex;           /* Simplicial mesh */
  DMBoundaryType boundary_types[3];
  PetscInt       cells[3];
  PetscInt       refine;
  /* geometry  */
  PetscReal      domain_lo[3], domain_hi[3];
  DMBoundaryType periodicity[3];              /* The domain periodicity */
  PetscReal      b0[3]; /* not used */
  /* Problem definition */
  PetscErrorCode (**initialFuncs)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
  PetscReal      mu, eta, beta;
} AppCtx;
static AppCtx *s_ctx;
static PetscScalar poissonBracket(const PetscScalar df[], const PetscScalar dg[])
{
  return df[0]*dg[1] - df[1]*dg[0];
}

enum field_idx {DENSITY,OMEGA,PSI,PHI,JZ};

static void f0_n(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscScalar f0[])
{
  const PetscScalar *pnDer   = &u_x[uOff_x[DENSITY]];
  const PetscScalar *ppsiDer = &u_x[uOff_x[PSI]];
  const PetscScalar *pphiDer = &u_x[uOff_x[PHI]];
  const PetscScalar *jzDer   = &u_x[uOff_x[JZ]];
  const PetscScalar *logRefDenDer = &a_x[aOff_x[DENSITY]];

  f0[0] = u_t[DENSITY] - poissonBracket(pnDer, pphiDer) - s_ctx->beta*poissonBracket(jzDer, ppsiDer) - poissonBracket(logRefDenDer, pphiDer);
}

static void f1_n(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscScalar f1[])
{
  const PetscScalar *pnDer = &u_x[uOff_x[DENSITY]];
  PetscInt           d;

  for (d = 0; d < dim-1; ++d) f1[d] = -s_ctx->mu*pnDer[d];
}

static void f0_Omega(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, const PetscReal x[], PetscScalar f0[])
{
  const PetscScalar *pOmegaDer = &u_x[uOff_x[OMEGA]];
  const PetscScalar *ppsiDer   = &u_x[uOff_x[PSI]];
  const PetscScalar *pphiDer   = &u_x[uOff_x[PHI]];
  const PetscScalar *jzDer     = &u_x[uOff_x[JZ]];

  f0[0] = u_t[OMEGA] - poissonBracket(pOmegaDer, pphiDer) - s_ctx->beta*poissonBracket(jzDer, ppsiDer);
}

static void f1_Omega(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, const PetscReal x[], PetscScalar f1[])
{
  const PetscScalar *pOmegaDer = &u_x[uOff_x[OMEGA]];
  PetscInt           d;

  for (d = 0; d < dim-1; ++d) f1[d] = -s_ctx->mu*pOmegaDer[d];
}

static void f0_psi(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                   const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                   PetscReal t, const PetscReal x[], PetscScalar f0[])
{
  const PetscScalar *pnDer     = &u_x[uOff_x[DENSITY]];
  const PetscScalar *ppsiDer   = &u_x[uOff_x[PSI]];
  const PetscScalar *pphiDer   = &u_x[uOff_x[PHI]];
  const PetscScalar *refPsiDer = &a_x[aOff_x[PSI]];
  const PetscScalar *logRefDenDer= &a_x[aOff_x[DENSITY]];
  PetscScalar psiDer[] = {refPsiDer[0]+ppsiDer[0],refPsiDer[1]+ppsiDer[1]};
  PetscScalar phi_n_Der[] = {pphiDer[0]-pnDer[0],pphiDer[1]-pnDer[1]};

  f0[0] = u_t[PSI] - poissonBracket(psiDer, phi_n_Der) + poissonBracket(logRefDenDer, ppsiDer);
}

static void f1_psi(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                   const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                   PetscReal t, const PetscReal x[], PetscScalar f1[])
{
  const PetscScalar *ppsi = &u_x[uOff_x[PSI]];
  PetscInt           d;

  for (d = 0; d < dim-1; ++d) f1[d] = -(s_ctx->eta/s_ctx->beta)*ppsi[d];
}

static void f0_phi(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                   const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                   PetscReal t, const PetscReal x[], PetscScalar f0[])
{
  f0[0] = -u[uOff[OMEGA]];
}

static void f1_phi(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                   const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                   PetscReal t, const PetscReal x[], PetscScalar f1[])
{
  const PetscScalar *pphi = &u_x[uOff_x[PHI]];
  PetscInt           d;

  for (d = 0; d < dim-1; ++d) f1[d] = pphi[d];
}

static void f0_jz(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, const PetscReal x[], PetscScalar f0[])
{
  f0[0] = u[uOff[JZ]];
}

static void f1_jz(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, const PetscReal x[], PetscScalar f1[])
{
  const PetscScalar *ppsi = &u_x[uOff_x[PSI]];
  const PetscScalar *refPsiDer = &a_x[aOff_x[PSI]]; /* aOff_x[PSI] == 2*PSI */
  PetscInt           d;

  for (d = 0; d < dim-1; ++d) f1[d] = ppsi[d] + refPsiDer[d];
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscBool      flg;
  PetscErrorCode ierr;
  PetscInt       ii, bd;
  PetscFunctionBeginUser;
  options->debug               = 0;
  options->dim                 = 2;
  options->filename[0]         = '\0';
  options->cell_simplex        = PETSC_FALSE;
  options->refine              = 2;
  options->domain_hi[0]  = 1;
  options->domain_hi[1]  = 1;
  options->domain_hi[2]  = 1;
  options->domain_lo[0]  = -1;
  options->domain_lo[1]  = -1;
  options->domain_lo[2]  = -1;
  options->periodicity[0]    = DM_BOUNDARY_NONE;
  options->periodicity[1]    = DM_BOUNDARY_NONE;
  options->periodicity[2]    = DM_BOUNDARY_NONE;
  options->mu   = 0;
  options->eta  = 0;
  options->beta = 1;
  for (ii = 0; ii < options->dim; ++ii) options->cells[ii] = 2;
  ierr = PetscOptionsBegin(comm, "", "Poisson Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex48.c", options->debug, &options->debug, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex48.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dm_refine", "Hack to get refinement level for cylinder", "ex48.c", options->refine, &options->refine, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mu", "mu", "ex48.c", options->mu, &options->mu, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-eta", "eta", "ex48.c", options->eta, &options->eta, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-beta", "beta", "ex48.c", options->beta, &options->beta, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-f", "Exodus.II filename to read", "ex48.c", options->filename, options->filename, sizeof(options->filename), &flg);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-cell_simplex", "Simplicial (true) or tensor (false) mesh", "ex48.c", options->cell_simplex, &options->cell_simplex, NULL);CHKERRQ(ierr);
  ii = options->dim;
  ierr = PetscOptionsRealArray("-domain_hi", "Domain size", "ex48.c", options->domain_hi, &ii, NULL);CHKERRQ(ierr);
  ii = options->dim;
  ierr = PetscOptionsRealArray("-domain_lo", "Domain size", "ex48.c", options->domain_lo, &ii, NULL);CHKERRQ(ierr);
  ii = options->dim;
  bd = options->periodicity[0];
  ierr = PetscOptionsEList("-x_periodicity", "The x-boundary periodicity", "ex48.c", DMBoundaryTypes, 5, DMBoundaryTypes[options->periodicity[0]], &bd, NULL);CHKERRQ(ierr);
  options->periodicity[0] = (DMBoundaryType) bd;
  bd = options->periodicity[1];
  ierr = PetscOptionsEList("-y_periodicity", "The y-boundary periodicity", "ex48.c", DMBoundaryTypes, 5, DMBoundaryTypes[options->periodicity[1]], &bd, NULL);CHKERRQ(ierr);
  options->periodicity[1] = (DMBoundaryType) bd;
  bd = options->periodicity[2];
  ierr = PetscOptionsEList("-z_periodicity", "The z-boundary periodicity", "ex48.c", DMBoundaryTypes, 5, DMBoundaryTypes[options->periodicity[2]], &bd, NULL);CHKERRQ(ierr);
  options->periodicity[2] = (DMBoundaryType) bd;
  ii = options->dim;
  ierr = PetscOptionsIntArray("-cells", "Number of cells in each dimension", "ex48.c", options->cells, &ii, NULL);CHKERRQ(ierr);
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
  /* ierr = PetscViewerSetType(viewer,PETSCVIEWERVTK);CHKERRQ(ierr); */
  ierr = PetscViewerSetType(viewer,PETSCVIEWERHDF5);CHKERRQ(ierr);
  ierr = PetscViewerSetFromOptions(viewer);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,&isHDF5);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERVTK,&isVTK);CHKERRQ(ierr);
  if (isHDF5) {
    ierr = PetscSNPrintf(buf, 256, "%s-%dD-%05d.h5", prefix, ctx->dim, stepi);CHKERRQ(ierr);
  } else if (isVTK) {
    ierr = PetscSNPrintf(buf, 256, "%s-%dD-%d.vtu", prefix, ctx->dim, stepi);CHKERRQ(ierr);
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
  PetscMPIInt    numProcs;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MPI_Comm_size(comm, &numProcs);CHKERRQ(ierr);
  ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
  if (len) {
    ierr = DMPlexCreateFromFile(comm, filename, PETSC_TRUE, dm);CHKERRQ(ierr);
  } else {
    Vec          coordinates;
    PetscScalar *coords;
    PetscInt     nCoords, n, dimEmbed, d;
    PetscReal    L[3];
    /* create DM */
    if (dim==2) {
      PetscInt refineRatio, totCells = 1;
      if (user->cell_simplex) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Cannot mesh 2D with simplices");
      refineRatio = PetscMax((PetscInt) (PetscPowReal(numProcs, 1.0/dim) + 0.1) - 1, 1);
      for (d = 0; d < dim; ++d) {
        if (user->periodicity[d]==DM_BOUNDARY_PERIODIC && user->cells[d]*refineRatio <= 2) refineRatio += (3 - user->cells[d]);
      }
      for (d = 0; d < dim; ++d) {
        user->cells[d] *= refineRatio;
        totCells *= user->cells[d];
      }
      if (totCells % numProcs) SETERRQ2(comm,PETSC_ERR_ARG_WRONG,"Total cells %D not divisible by processes %D", totCells, numProcs);
      ierr = DMPlexCreateHexBoxMesh(comm, dim, user->cells, user->periodicity[0], user->periodicity[1], user->periodicity[2], dm);CHKERRQ(ierr);
    } else if (user->cell_simplex && dim==3) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Cannot mesh a cylinder with simplices");
    else {
      /* we stole dm_refine so clear it */
      ierr = PetscOptionsClearValue(NULL,"-dm_refine");CHKERRQ(ierr);
      ierr = DMPlexCreateHexCylinderMesh(comm, user->refine, user->periodicity[2], dm);CHKERRQ(ierr);
      if (user->periodicity[0]==DM_BOUNDARY_PERIODIC || user->periodicity[1]==DM_BOUNDARY_PERIODIC) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Cannot do periodic in x or y in a cylinder");
    }
    ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
    /* Rescale to physical domain */
    ierr = DMGetCoordinateDim(*dm, &dimEmbed);CHKERRQ(ierr);
    for (d = 0; d < dimEmbed; ++d) L[d] = user->domain_hi[d] - user->domain_lo[d];
    ierr = DMGetCoordinatesLocal(*dm, &coordinates);CHKERRQ(ierr);
    ierr = VecGetLocalSize(coordinates, &nCoords);CHKERRQ(ierr);
    if (nCoords % dimEmbed) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Dimension %D does not divide the coordinate vector size %D", dimEmbed, nCoords);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
    for (n = 0; n < nCoords; n += dimEmbed) {
      PetscScalar *coord = &coords[n];
      /* for (d = 0; d < dimEmbed; ++d) coord[d] = user->domain_lo[d] + coord[d] * L[d]; */
    }
    ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
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
    ierr = PetscOptionsFList("-dm_plex_convert_type","Convert DMPlex to another format","ex48",DMList,DMPLEX,convType,256,&flg);CHKERRQ(ierr);
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
  /* ierr = DMLocalizeCoordinates(*dm);CHKERRQ(ierr); ???? */
  PetscFunctionReturn(0);
}

static PetscErrorCode log_n_0(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = log(1.0);
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

static PetscErrorCode initialSolution_n(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0; /* perturbation? */
  return 0;
}

static PetscErrorCode initialSolution_Omega(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return 0;
}

static PetscErrorCode initialSolution_psi(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return 0;
}

static PetscErrorCode initialSolution_phi(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return 0;
}

static PetscErrorCode initialSolution_jz(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
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
  user->initialFuncs[0] = initialSolution_n;
  user->initialFuncs[1] = initialSolution_Omega;
  user->initialFuncs[2] = initialSolution_psi;
  user->initialFuncs[3] = initialSolution_phi;
  user->initialFuncs[4] = initialSolution_jz;
  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "wall", "marker", 0, 0, NULL, (void (*)()) user->initialFuncs[0], 1, &id, user);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupEquilibriumFields(DM dm, DM dmAux, AppCtx *user)
{
  PetscErrorCode (*eqFuncs[3])(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx) = {log_n_0, Omega_0, psi_0};
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
  PetscBool       cell_simplex = user->cell_simplex;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  /* Create finite element */
  ierr = PetscFECreateDefault(dm, dim, 1, cell_simplex, "density_", -1, &fe[0]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[0], "density");CHKERRQ(ierr);
  ierr = PetscFECreateDefault(dm, dim, 1, cell_simplex, "vorticity_", -1, &fe[1]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[1], "vorticity");CHKERRQ(ierr);
  ierr = PetscFECreateDefault(dm, dim, 1, cell_simplex, "flux_", -1, &fe[2]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[2], "flux");CHKERRQ(ierr);
  ierr = PetscFECreateDefault(dm, dim, 1, cell_simplex, "potential_", -1, &fe[3]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[3], "potential");CHKERRQ(ierr);
  ierr = PetscFECreateDefault(dm, dim, 1, cell_simplex, "current_", -1, &fe[4]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[4], "current");CHKERRQ(ierr);

  ierr = PetscFECreateDefault(dm, dim, 1, cell_simplex, "density_eq_", -1, &feAux[0]);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe[0], &q);CHKERRQ(ierr);
  ierr = PetscFESetQuadrature(feAux[0], q);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(dm, dim, 1, cell_simplex, "vorticity_eq_", -1, &feAux[1]);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe[1], &q);CHKERRQ(ierr);
  ierr = PetscFESetQuadrature(feAux[1], q);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(dm, dim, 1, cell_simplex, "flux_eq_", -1, &feAux[2]);CHKERRQ(ierr);
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
  for (f = 0; f < NfAux; ++f) {ierr = PetscFEDestroy(&feAux[f]);CHKERRQ(ierr);}
  ierr = PetscDSDestroy(&probAux);CHKERRQ(ierr);
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
  s_ctx = &ctx;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &ctx);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &ctx, &dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, &ctx);CHKERRQ(ierr);
  ierr = PetscMalloc1(5, &ctx.initialFuncs);CHKERRQ(ierr);
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

  ierr = DMProjectFunction(dm, t, ctx.initialFuncs, NULL, INSERT_ALL_VALUES, u);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,u);CHKERRQ(ierr);
  ierr = PostStep(ts);CHKERRQ(ierr); /* get the initial state */
  ierr = TSSolve(ts, u);CHKERRQ(ierr);

  ierr = TSGetTime(ts, &t);CHKERRQ(ierr);
  ierr = DMComputeL2Diff(dm, t, ctx.initialFuncs, NULL, u, &L2error);CHKERRQ(ierr);
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
  ierr = PetscFree(ctx.initialFuncs);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    args: -debug 1 -dim 2 -dm_refine 1 -cells 4,4 -x_periodicity PERIODIC -ts_max_steps 1 -ts_final_time 10. -ts_dt 1.0
  test:
    suffix: 0
    args: -debug 1 -dim 3 -dm_refine 1 -z_periodicity PERIODIC -ts_max_steps 1 -ts_final_time 10. -ts_dt 1.0

TEST*/
