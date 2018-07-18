static char help[] = "Nonlinear driven cavity using finite elements with multigrid in 2d.\n \
  \n\
The 2D driven cavity problem is solved in a velocity-vorticity formulation.\n\
The flow can be driven with the lid or with bouyancy or both:\n\
  -lidvelocity &ltlid&gt, where &ltlid&gt = dimensionless velocity of lid\n\
  -grashof &ltgr&gt, where &ltgr&gt = dimensionless temperature gradent\n\
  -prandtl &ltpr&gt, where &ltpr&gt = dimensionless thermal/momentum diffusity ratio\n\
 -contours : draw contour plots of solution\n\n";
/* in HTML, '&lt' = '<' and '&gt' = '>' */

/*
      See src/snes/examples/tutorials/ex19.c
*/

/*T
   Concepts: SNES^solving a system of nonlinear equations (parallel multicomponent example);
   Concepts: DMDA^using distributed arrays;
   Concepts: multicomponent
   Processors: n
T*/


/*F-----------------------------------------------------------------------

    This problem is modeled by the partial differential equation system in the unit square.

\begin{eqnarray}
        - \triangle U - \nabla_y \Omega & = & 0  \\
        - \triangle V + \nabla_x\Omega & = & 0  \\
        - \triangle \Omega + \nabla \cdot ([U*\Omega,V*\Omega]) - GR* \nabla_x T & = & 0  \\
        - \triangle T + PR* \nabla \cdot ([U*T,V*T]) & = & 0
\end{eqnarray}
    where we note that
\begin{align}
      \nabla\cdot (U \Omega, V \Omega) &= \frac{\partial}{\partial x} (U \Omega) + \frac{\partial}{\partial y} (V \Omega) \\
        &= \frac{\partial U}{\partial x} Omega + U \frac{\partial\Omega}{\partial x} + \frac{\partial V}{\partial y} Omega + V \frac{\partial\Omega}{\partial y} \\
        &= \frac{\partial U}{\partial x} Omega + \frac{\partial V}{\partial y} Omega + U \frac{\partial\Omega}{\partial x} + V \frac{\partial\Omega}{\partial y} \\
        &= (\nabla\cdot{\mathbf U}) \Omega + {\mathbf U} \cdot \nabla\Omega \\
        &= {\mathbf U} \cdot \nabla\Omega
\end{align}

    No-slip, rigid-wall Dirichlet conditions are used for $ [U,V]$.
    Dirichlet conditions are used for Omega, based on the definition of
    vorticity: $ \Omega = - \nabla_y U + \nabla_x V$, where along each
    constant coordinate boundary, the tangential derivative is zero.
    Dirichlet conditions are used for T on the left and right walls,
    and insulation homogeneous Neumann conditions are used for T on
    the top and bottom walls.

    A finite element approximation is used to discretize the boundary
    value problem to obtain a nonlinear system of equations.  Entropy
    viscosity is used to stabilize the divergence (convective) terms.

\alpha = 1 or 2
\bata = 0.03
k = \frac{1}{4} \frac{h_K}{\|\|u\|\|_{\infty(K)}}
c_R = 2^\frac{4 - 2\alpha}{d}
var(T) = max_\Omega T - min_\Omega T
diam(\Omega) = 2^{1/d} for the unit cube
c(u,T) = c_R \|\|u\|\|_{\infty(\Omega)} var(T) |diam(\Omega)|^{\alpha−2}

\nu_\alpha(T)_K = \beta \|\|u\|\|_{\infty(K)} \mathrm{min} \left( h_K, h^\alpha_K \frac{\|\|R_\alpha(T)\|\|_{\infty(K)}}{c(u,T)} \right)

  The entropy viscosity will be $P_0$ auxiliary field which we update at each SNES iteration.

  ------------------------------------------------------------------------F*/

/*
   Include "petscdmplex.h" so that we can use unstructured meshes (DMDAs).
   Include "petscsnes.h" so that we can use SNES solvers.  Note that this
   file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers
*/
#include <petscdmplex.h>
#include <petscds.h>
#include <petscsnes.h>
#include <petscbag.h>

typedef struct {
  PetscReal prandtl;     /* Prandtl number, ratio of momentum diffusivity to thermal diffusivity */
  PetscReal grashof;     /* Grashof number, ratio of bouyancy to viscous forces */
  PetscReal lidvelocity; /* Shear velocity of top boundary */
} Parameter;

typedef struct {
  PetscInt  dim;            /* Topological dimension */
  char      filename[2048]; /* The optional mesh file */
  PetscBool simplex;        /* Simplicial mesh */
  PetscBag  params;         /* Problem parameters */
  PetscInt  mms;            /* Number of the MMS solution, or -1 */
} AppCtx;

static PetscErrorCode coordX(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = x[0];
  return 0;
}

static PetscErrorCode zerovec(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;
  for (d = 0; d < dim; ++d) u[d] = 0.0;
  return 0;
}

static PetscErrorCode lidshear(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = ((Parameter *) ctx)->lidvelocity;
  u[1] = 0.0;
  return 0;
}

static PetscErrorCode tempbc(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = ((Parameter *) ctx)->grashof > 0 ? 1.0 : 0.0;
  return 0;
}

/* MMS 0

  u = x^2 + y^2
  v = 2 x^2 - 2xy
  O = 4x - 4y = curl u
  T = x
  f_O = 4 (x^2 - 2 x y - y^2) + GR
  f_T = -PR (x^2 + y^2)

so that

  -\Delta U - \nabla_y \Omega = -4 - -4 = 0
  -\Delta V + \nabla_x\Omega  = -4 +  4 = 0
  -\Delta \Omega + \nabla \cdot <U \Omega,V \Omega> - GR \nabla_x T = 0 + div <4 x^3 - 4 x^2 y + 4 x y^2 - 4 y^3, 8 x^3 - 16 x^2 y + 8 x y^2> - GR
    = (12 x^2 - 8 x y + 4 y^2 - 16 x^2 + 16 x y) - GR = -4 (x^2 - 2 x y - y^2) - GR
  -\Delta T + PR \nabla \cdot <U*T,V*T> = 0 + PR div <x^3 + x y^2, 2 x^3 - 2 x^2 y> = PR (3 x^2 + y^2 - 2 x^2) = PR (x^2 + y^2)

and we check that

    \nabla \cdot u = 2x - 2x = 0
    \hat n \cdot \nabla T = < 0, \pm 1> \cdot <1, 0> = 0
*/

static PetscErrorCode mms_0_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = x[0]*x[0] + x[1]*x[1];
  u[1] = 2.0*x[0]*x[0] - 2.0*x[0]*x[1];
  return 0;
}

static PetscErrorCode mms_0_O(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = 4.0*x[0] - 4.0*x[1];
  return 0;
}

static PetscErrorCode mms_0_T(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = x[0];
  return 0;
}

/* curl u = -u_y */
static void omegabc_horiz(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                          PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar uexact[])
{
  uexact[0] = -u_x[0*dim+1];
}

/* curl u = v_x */
static void omegabc_vert(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                         const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                         const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                         PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar uexact[])
{
  uexact[0] = u_x[1*dim+0];
}

/* <v, -curl Omega> */
static void f0_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = -u_x[uOff_x[1]+1];
  f0[1] =  u_x[uOff_x[1]+0];
}

/* <grad v, grad u> */
static void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt c, d;

  for (c = 0; c < dim; ++c) {
    for (d = 0; d < dim; ++d) {
      f1[c*dim+d] = u_x[c*dim+d];
    }
  }
}

/* -Gr T_x + U . grad Omega */
static void f0_O(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = -constants[1] * u_x[uOff_x[2]+0] + u[0] * u_x[uOff_x[1]+0] + u[1] * u_x[uOff_x[1]+1];
}

/* <grad Tau, grad Omega> */
static void f1_O(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u_x[uOff_x[1]+d];
}

/* -Gr T_x + U . grad Omega + 4 (x^2 - 2 x y - y^2) + GR */
static void f0_mms_0_O(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                       const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                       const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                       PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal Gr = constants[1];

  f0[0] = -Gr * u_x[uOff_x[2]+0] + u[0] * u_x[uOff_x[1]+0] + u[1] * u_x[uOff_x[1]+1]
        + 4.0*(x[0]*x[0] - 2.0*x[0]*x[1] - x[1]*x[1]) + Gr;
}

/* Pr (U . grad T) */
static void f0_T(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal Pr = constants[0];

  f0[0] = Pr * (u[0] * u_x[uOff_x[2]+0] + u[1] * u_x[uOff_x[2]+1]);
}

/* <grad S, grad T> */
static void f1_T(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u_x[uOff_x[2]+d];
}

/* Pr (U . grad T) - PR (x^2 + y^2) */
static void f0_mms_0_T(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                       const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                       const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                       PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal Pr = constants[0];

  f0[0] = Pr * (u[0] * u_x[uOff_x[2]+0] + u[1] * u_x[uOff_x[2]+1]) - Pr * (x[0]*x[0] + x[1]*x[1]);
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->dim         = 2;
  options->filename[0] = '\0';
  options->simplex     = PETSC_TRUE;
  options->mms         = -1;

  ierr = PetscOptionsBegin(comm, "", "Driven Cavity Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex4.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-f", "Mesh filename to read", "ex4.c", options->filename, options->filename, sizeof(options->filename), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Simplicial (true) or tensor (false) mesh", "ex4.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mms", "The MMS solution number", "ex4.c", options->mms, &options->mms, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupParameters(AppCtx *user)
{
  Parameter     *p;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscBagGetData(user->params, (void **) &p);CHKERRQ(ierr);
  ierr = PetscBagSetName(user->params, "par", "Problem parameters");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user->params, &p->prandtl,     1.0, "prandtl",     "Prandtl number, ratio of momentum diffusivity to thermal diffusivity");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user->params, &p->grashof,     1.0, "grashof",     "Grashof number, ratio of bouyancy to viscous forces");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(user->params, &p->lidvelocity, 1.0, "lidvelocity", "Shear velocity of top boundary");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim      = user->dim;
  const char    *filename = user->filename;
  PetscInt       cells[3] = {3, 3, 3};
  size_t         len;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
  if (!len) {
    ierr = DMPlexCreateBoxMesh(comm, dim, user->simplex, cells, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateFromFile(comm, filename, PETSC_TRUE, dm);CHKERRQ(ierr);
    ierr = DMPlexSetRefinementUniform(*dm, PETSC_FALSE);CHKERRQ(ierr);
  }
  {
    PetscPartitioner part;
    DM               pdm = NULL;

    /* Distribute mesh over processes */
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
  Parameter     *param;
  PetscScalar    constants[3];
  PetscInt       ids[4] = {1, 2, 3, 4};
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscBagGetData(user->params, (void **) &param);CHKERRQ(ierr);
  if (user->mms >= 0) {ierr = PetscPrintf(PetscObjectComm((PetscObject) prob), "Using MMS solution %D\n", user->mms);CHKERRQ(ierr);}
  switch(user->mms) {
  case 0:
    /* Equations */
    ierr = PetscDSSetResidual(prob, 0, f0_u,       f1_u);CHKERRQ(ierr);
    ierr = PetscDSSetResidual(prob, 1, f0_mms_0_O, f1_O);CHKERRQ(ierr);
    ierr = PetscDSSetResidual(prob, 2, f0_mms_0_T, f1_T);CHKERRQ(ierr);
    /* Boundary conditions */
    ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "U wall",       "marker", 0, 0, NULL, (void (*)()) mms_0_u, 4, ids, user);CHKERRQ(ierr);
    /*   Replace with standard BC */
    ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "Omega wall",   "marker", 1, 0, NULL, (void (*)()) mms_0_O, 4, ids, user);CHKERRQ(ierr);
    /*   Replace with standard BC */
    ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "T right wall", "marker", 2, 0, NULL, (void (*)()) mms_0_T, 1, &ids[1], user);CHKERRQ(ierr);
    ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "T left wall",  "marker", 2, 0, NULL, (void (*)()) mms_0_T, 1, &ids[3], user);CHKERRQ(ierr);
    /* MMS solutions */
    ierr = PetscDSSetExactSolution(prob, 0, mms_0_u);CHKERRQ(ierr);
    ierr = PetscDSSetExactSolution(prob, 1, mms_0_O);CHKERRQ(ierr);
    ierr = PetscDSSetExactSolution(prob, 2, mms_0_T);CHKERRQ(ierr);
    break;
  default:
    /* Equations */
    ierr = PetscDSSetResidual(prob, 0, f0_u, f1_u);CHKERRQ(ierr);
    ierr = PetscDSSetResidual(prob, 1, f0_O, f1_O);CHKERRQ(ierr);
    ierr = PetscDSSetResidual(prob, 2, f0_T, f1_T);CHKERRQ(ierr);
    /* Boundary conditions */
    ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL,       "U bottom wall",     "marker", 0, 0, NULL, (void (*)()) zerovec,       1, &ids[0], param);CHKERRQ(ierr);
    ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL,       "U right wall",      "marker", 0, 0, NULL, (void (*)()) zerovec,       1, &ids[1], param);CHKERRQ(ierr);
    ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL,       "U top wall",        "marker", 0, 0, NULL, (void (*)()) lidshear,      1, &ids[2], param);CHKERRQ(ierr);
    ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL,       "U left wall",       "marker", 0, 0, NULL, (void (*)()) zerovec,       1, &ids[3], param);CHKERRQ(ierr);
    ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL_FIELD, "Omega bottom wall", "marker", 1, 0, NULL, (void (*)()) omegabc_horiz, 1, &ids[0], param);CHKERRQ(ierr);
    ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL_FIELD, "Omega right wall",  "marker", 1, 0, NULL, (void (*)()) omegabc_vert,  1, &ids[1], param);CHKERRQ(ierr);
    ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL_FIELD, "Omega top wall",    "marker", 1, 0, NULL, (void (*)()) omegabc_horiz, 1, &ids[2], param);CHKERRQ(ierr);
    ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL_FIELD, "Omega left wall",   "marker", 1, 0, NULL, (void (*)()) omegabc_vert,  1, &ids[3], param);CHKERRQ(ierr);
    ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL,       "T right wall",      "marker", 2, 0, NULL, (void (*)()) zerovec,       1, &ids[1], param);CHKERRQ(ierr);
    ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL,       "T left wall",       "marker", 2, 0, NULL, (void (*)()) tempbc,        1, &ids[3], param);CHKERRQ(ierr);
    break;
  }
  /* Physical Constants */
  constants[0] = param->prandtl;
  constants[1] = param->grashof;
  constants[2] = param->lidvelocity;
  ierr = PetscDSSetConstants(prob, 3, constants);CHKERRQ(ierr);
  ierr = PetscDSSetFromOptions(prob);CHKERRQ(ierr);
  ierr = PetscPrintf(PetscObjectComm((PetscObject) prob), "lid velocity = %g, prandtl # = %g, grashof # = %g\n", (double) param->lidvelocity, (double) param->prandtl, (double) param->grashof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  MPI_Comm        comm;
  const PetscInt  dim     = user->dim;
  PetscBool       simplex = user->simplex;
  PetscDS         prob;
  PetscFE         feU, feO, feT;
  PetscQuadrature q;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  /* Create finite element */
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(comm, dim, dim, simplex, "vel_", -1, &feU);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(feU, &q);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) feU, "velocity");CHKERRQ(ierr);

  ierr = PetscFECreateDefault(comm, dim, 1, simplex, "vort_", -1, &feO);CHKERRQ(ierr);
  ierr = PetscFESetQuadrature(feO, q);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) feO, "vorticity");CHKERRQ(ierr);

  ierr = PetscFECreateDefault(comm, dim, 1, simplex, "temp_", -1, &feT);CHKERRQ(ierr);
  ierr = PetscFESetQuadrature(feT, q);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) feT, "temperature");CHKERRQ(ierr);

  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) feU);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 1, (PetscObject) feO);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 2, (PetscObject) feT);CHKERRQ(ierr);
  ierr = SetupProblem(prob, user);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&feU);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&feO);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&feT);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  MPI_Comm       comm;
  DM             dm;
  Vec            x;
  SNES           snes;
  AppCtx         user; /* user-defined work context */
  PetscErrorCode (*initialGuesses[3])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar*, void*) = {NULL, NULL, coordX};
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);
  ierr = CreateMesh(comm, &user, &dm);CHKERRQ(ierr);
  ierr = SNESCreate(comm, &snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, &user);CHKERRQ(ierr);

  ierr = PetscBagCreate(comm, sizeof(Parameter), &user.params);CHKERRQ(ierr);
  ierr = SetupParameters(&user);CHKERRQ(ierr);
  ierr = SetupDiscretization(dm, &user);CHKERRQ(ierr);
  ierr = DMPlexSetSNESLocalFEM(dm, &user, &user, &user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm, &x);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) x, "solution");CHKERRQ(ierr);
  ierr = DMProjectFunction(dm, 0.0, initialGuesses, NULL, INSERT_ALL_VALUES, x);CHKERRQ(ierr);
  ierr = DMSNESCheckFromOptions(snes, x, NULL, NULL);CHKERRQ(ierr);
  ierr = SNESSolve(snes, NULL, x);CHKERRQ(ierr);
  ierr = VecViewFromOptions(x, NULL, "-sol_view");CHKERRQ(ierr);

  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscBagDestroy(&user.params);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   # Use -snes_monitor_lg_residualnorm -draw_save_final_image $PWD/conv.ppm to get an image of the convergence
   # https://www.online-utility.org/image/convert/to/PNG for conversion
   test:
     suffix: 0
     args: -mms 0 -lidvelocity 100 -simplex 0 -dm_refine 0 -dm_plex_separate_marker -dm_view \
       -vel_petscspace_order 1 -vort_petscspace_order 1 -temp_petscspace_order 1 -petscds_view -dmsnes_check \
       -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -snes_monitor_short -snes_converged_reason -snes_view \
       -ksp_rtol 1e-10 -ksp_error_if_not_converged -pc_type lu

   test:
     suffix: 1
     args: -mms 0 -lidvelocity 100 -simplex 0 -dm_refine 0 -dm_plex_separate_marker -dm_view \
       -vel_petscspace_order 2 -vort_petscspace_order 2 -temp_petscspace_order 2 -petscds_view -dmsnes_check \
       -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -snes_monitor_short -snes_converged_reason -snes_view \
       -ksp_rtol 1e-10 -ksp_error_if_not_converged -pc_type lu

   test:
     suffix: matt
     args: -lidvelocity 100 -grashof 1.3372e4 -da_grid_x 16 -da_grid_y 16 -da_refine 2 \
       -snes_monitor_short -snes_converged_reason -snes_view -pc_type lu

   test:
     suffix: matt_chord
     args: -lidvelocity 100 -grashof 1.3372e2 -da_grid_x 16 -da_grid_y 16 -da_refine 2 \
       -snes_lag_jacobian -3 -snes_linesearch_type cp -snes_max_it 100 -snes_monitor_short -snes_converged_reason -snes_view -pc_type lu

   test:
     suffix: matt_nrichardson
     args: -lidvelocity 100 -grashof 1.3372e2 -da_grid_x 16 -da_grid_y 16 -da_refine 2 \
       -snes_type nrichardson -snes_linesearch_type cp -snes_max_it 10000 -snes_monitor_short -snes_converged_reason -snes_view -pc_type lu

   test:
     suffix: matt_bad
     args: -lidvelocity 100 -grashof 1.3373e4 -da_grid_x 16 -da_grid_y 16 -da_refine 2 \
       -snes_max_it 100 -snes_monitor_short -snes_converged_reason -snes_view -pc_type lu

   test:
     suffix: matt_bad_fas
     args: -lidvelocity 100 -grashof 1.3373e4 -da_grid_x 16 -da_grid_y 16 -da_refine 2 \
       -snes_type fas -snes_max_it 100 -snes_monitor_short -snes_converged_reason -snes_view \
         -fas_levels_snes_type ngs -fas_levels_snes_max_it 6

   test:
     suffix: matt_bad_fas_big
     args: -lidvelocity 100 -grashof 5e4 -da_refine 4 \
       -snes_type fas -snes_monitor_short -snes_converged_reason -snes_view \
         -fas_levels_snes_type ngs -fas_levels_snes_max_it 6 \
         -fas_coarse_snes_linesearch_type basic -fas_coarse_snes_converged_reason

   test:
     suffix: matt_bad_nrichardson
     args: -lidvelocity 100 -grashof 1.3373e4 -da_grid_x 16 -da_grid_y 16 -da_refine 2 \
       -snes_type nrichardson -snes_max_it 1000 -snes_view

   test:
     suffix: matt_bad_nrich_newton_stag
     args: -lidvelocity 100 -grashof 1.3373e4 -da_grid_x 16 -da_grid_y 16 -da_refine 2 \
       -snes_type nrichardson -snes_max_it 200 -snes_monitor_short -snes_converged_reason -snes_view \
       -npc_snes_type newtonls -npc_snes_max_it 3 -npc_snes_converged_reason -npc_pc_type lu

   test:
     suffix: matt_bad_nrich_newton
     args: -lidvelocity 100 -grashof 1.3373e4 -da_grid_x 16 -da_grid_y 16 -da_refine 2 \
       -snes_type nrichardson -snes_monitor_short -snes_converged_reason -snes_view \
       -npc_snes_type newtonls -npc_snes_max_it 4 -npc_snes_converged_reason -npc_pc_type lu

   test:
     suffix: matt_bad_newton_nrich_it1
     args: -lidvelocity 100 -grashof 1.3373e4 -da_grid_x 16 -da_grid_y 16 -da_refine 2 \
       -snes_type newtonls -pc_type lu -snes_max_it 1000 -snes_monitor_short -snes_converged_reason -snes_view \
       -npc_snes_type nrichardson -npc_snes_max_it 1

   test:
     suffix: matt_bad_newton_nrich_it3
     args: -lidvelocity 100 -grashof 1.3373e4 -da_grid_x 16 -da_grid_y 16 -da_refine 2 \
       -snes_type newtonls -pc_type lu -snes_max_it 1000 -snes_monitor_short -snes_converged_reason -snes_view \
       -npc_snes_type nrichardson -npc_snes_max_it 3

   test:
     suffix: matt_bad_newton_nrich_it5
     args: -lidvelocity 100 -grashof 1.3373e4 -da_grid_x 16 -da_grid_y 16 -da_refine 2 \
       -snes_type newtonls -pc_type lu -snes_max_it 1000 -snes_monitor_short -snes_converged_reason -snes_view \
       -npc_snes_type nrichardson -npc_snes_max_it 5

   test:
     suffix: matt_bad_newton_nrich_it6
     args: -lidvelocity 100 -grashof 1.3373e4 -da_grid_x 16 -da_grid_y 16 -da_refine 2 \
       -snes_type newtonls -pc_type lu -snes_max_it 1000 -snes_monitor_short -snes_converged_reason -snes_view \
       -npc_snes_type nrichardson -npc_snes_max_it 6
TEST*/
