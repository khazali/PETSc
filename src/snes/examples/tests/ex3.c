static char help[] = "Poisson Problem using finite elements with curved boundaries.\n\
The point is to check that higher order elements for geometry work correctly.\n\n\n";

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include <petscconvest.h>

#include <petsc/private/petscfeimpl.h> /* For PetscFECreate_Internal() */

typedef struct {
  /* Domain and mesh definition */
  PetscInt  dim;       /* The topological mesh dimension */
  PetscBool simplex;   /* Simplicial mesh */
  PetscInt  cells[3];  /* The initial domain division */
  PetscBool geomNA;    /* Use a non-affine geometry */
  PetscBool curved;    /* Use a curved boundary */
} AppCtx;

static PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  *u = 0.0;
  return 0;
}

/*
  Our domain will be the unit square deformed by the mapping

    phi(x, y) = <x, y + 2x * (1-x)>

The exact solution to the Poisson equation on this domain is given by

  u = (y - 2x * (1-x)) * (y-1 - 2x * (1-x))
  f = 48 x (x - 1) + 8 y + 6

so that

  -Delta u + f = -(48 x^2 - 48 x + 8 y + 4) - (2) + 48 x (x - 1) + 8 y + 6 = 0

On the top and bottom boundary the solution is zero. On the sides it is

  u |_side  = y * (y-1)
*/

static PetscErrorCode phi(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = x[0];
  u[1] = x[1] + 2.0*x[0] * (1.0 - x[0]);
  return 0;
}

static PetscErrorCode quartic_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  *u = (x[1] - 2.0*x[0] * (1.0 - x[0])) * (x[1] - 1.0 - 2.0*x[0] * (1.0 - x[0]));
  return 0;
}

static PetscErrorCode bc_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  *u = x[1] * (x[1] - 1.0);
  return 0;
}

static void f0_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = 48.0*x[0]*(x[0] - 1.0) + 8.0*x[1] + 6.0;
}

static void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u_x[d];
}

static void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d*dim+d] = 1.0;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt       n = 3;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->dim      = 2;
  options->cells[0] = 1;
  options->cells[1] = 1;
  options->cells[2] = 1;
  options->simplex  = PETSC_TRUE;
  options->geomNA   = PETSC_TRUE;
  options->curved   = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "Poisson Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex3.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-cells", "The initial mesh division", "ex3.c", options->cells, &n, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Simplicial (true) or tensor (false) mesh", "ex3.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-geom_na", "Use a non-affine geometry", "ex3.c", options->geomNA, &options->geomNA, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-curved", "Use a curved boundary", "ex3.c", options->curved, &options->curved, NULL);CHKERRQ(ierr);
  if (!options->geomNA && options->curved) SETERRQ(comm, PETSC_ERR_ARG_INCOMP, "Cannot have a curved boundary without non-affine geometry");
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static void old_coords(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                       const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                       const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                       PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f[d] = a[d];
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* Create box mesh */
  ierr = DMPlexCreateBoxMesh(comm, user->dim, user->simplex, user->cells, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
  /* Create non-affine coordinates and remap */
  if (user->geomNA) {
    void (*funcs[1])(PetscInt, PetscInt, PetscInt,
                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                     const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                     PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]) = {old_coords};
    DM           cdm,  cdmTmp;
    PetscDS      prob, probTmp;
    PetscFE      fe;
    Vec          coordinates, newCoordinates;
    PetscSection csection;
    PetscScalar *a;
    PetscInt     pStart, pEnd, p;

    ierr = DMGetCoordinateDM(*dm, &cdm);CHKERRQ(ierr);
    ierr = DMGetDS(cdm, &prob);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(*dm, &coordinates);CHKERRQ(ierr);
    ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);

    ierr = DMClone(cdm, &cdmTmp);CHKERRQ(ierr);
    ierr = DMGetDS(cdmTmp, &probTmp);CHKERRQ(ierr);
    ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) cdm), user->dim, user->dim, user->simplex, "coords_", -1, &fe);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fe, "coordinates");CHKERRQ(ierr);
    ierr = PetscDSSetDiscretization(probTmp, 0, (PetscObject) fe);CHKERRQ(ierr);
    ierr = DMGetLocalVector(cdmTmp, &newCoordinates);CHKERRQ(ierr);
    /* Setup element for default */
    {
      PetscFE         oldfe;
      PetscQuadrature q;

      ierr = PetscFECreate_Internal(PetscObjectComm((PetscObject) cdm), user->dim, user->dim, user->simplex, 1, -1, &oldfe);CHKERRQ(ierr);
      ierr = PetscFEGetQuadrature(fe, &q);CHKERRQ(ierr);
      ierr = PetscFESetQuadrature(oldfe, q);CHKERRQ(ierr);
      ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) oldfe);CHKERRQ(ierr);
      ierr = PetscFEDestroy(&oldfe);CHKERRQ(ierr);
    }
    /* Project old coordinates */
    ierr = VecViewFromOptions(coordinates, NULL, "-dm_view");CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) cdmTmp, "dmAux", (PetscObject) cdm);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) cdmTmp, "A",     (PetscObject) coordinates);CHKERRQ(ierr);
    ierr = DMProjectFieldLocal(cdmTmp, 0.0, newCoordinates, funcs, INSERT_VALUES, newCoordinates);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) cdmTmp, "dmAux", NULL);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) cdmTmp, "A",     NULL);CHKERRQ(ierr);
    ierr = VecViewFromOptions(newCoordinates, NULL, "-dm_view");CHKERRQ(ierr);
    /* Kill old coordinates */
    ierr = DMSetCoordinates(*dm, NULL);CHKERRQ(ierr);
    ierr = DMSetDefaultSection(cdm, NULL);CHKERRQ(ierr);
    /* Set new discretization and coordinates */
    ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) fe);CHKERRQ(ierr);
    ierr = PetscDSSetFromOptions(prob);CHKERRQ(ierr);
    ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
    ierr = DMCreateLocalVector(cdm, &coordinates);CHKERRQ(ierr);
    ierr = VecCopy(newCoordinates, coordinates);CHKERRQ(ierr);
    ierr = DMSetCoordinatesLocal(*dm, coordinates);CHKERRQ(ierr);
    ierr = VecDestroy(&coordinates);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(cdmTmp, &newCoordinates);CHKERRQ(ierr);
    ierr = DMDestroy(&cdmTmp);CHKERRQ(ierr);
    /* Warp grid */
    if (user->curved) {
      ierr = DMGetCoordinatesLocal(*dm, &coordinates);CHKERRQ(ierr);
      ierr = DMGetDefaultSection(cdm, &csection);CHKERRQ(ierr);
      ierr = PetscSectionGetChart(csection, &pStart, &pEnd);CHKERRQ(ierr);
      ierr = VecGetArray(coordinates, &a);CHKERRQ(ierr);
      for (p = pStart; p < pEnd; ++p) {
        PetscScalar *coords;
        PetscInt     dof;

        ierr = PetscSectionGetDof(csection, p, &dof);CHKERRQ(ierr);
        if (!dof) continue;
        if (dof != user->dim) SETERRQ3(PetscObjectComm((PetscObject) cdm), PETSC_ERR_ARG_SIZ, "Coodinate dof for point %D is %D != %D", p, dof, user->dim);
        ierr = DMPlexPointLocalRef(cdm, p, a, &coords);CHKERRQ(ierr);
        ierr = phi(user->dim, 0.0, coords, dof, coords, NULL);CHKERRQ(ierr);
      }
      ierr = VecRestoreArray(coordinates, &a);CHKERRQ(ierr);
      ierr = VecViewFromOptions(coordinates, NULL, "-dm_view");CHKERRQ(ierr);
    }
  }
  /* Distribute mesh over processes */
  {
    DM               dmDist = NULL;
    PetscPartitioner part;

    ierr = DMPlexGetPartitioner(*dm, &part);CHKERRQ(ierr);
    ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
    ierr = DMPlexDistribute(*dm, 0, NULL, &dmDist);CHKERRQ(ierr);
    if (dmDist) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = dmDist;
    }
  }
  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  ierr = DMSetApplicationContext(*dm, user);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupPrimalProblem(PetscDS prob, AppCtx *user)
{
  PetscInt       id;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscDSSetResidual(prob, 0, f0_u, f1_u);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_uu);CHKERRQ(ierr);
  id   = 1;
  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "bottom wall", "marker", 0, 0, NULL, (void (*)(void)) zero, 1, &id, user);CHKERRQ(ierr);
  id   = 2;
  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "right wall",  "marker", 0, 0, NULL, (void (*)(void)) bc_u, 1, &id, user);CHKERRQ(ierr);
  id   = 3;
  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "top wall",    "marker", 0, 0, NULL, (void (*)(void)) zero, 1, &id, user);CHKERRQ(ierr);
  id   = 4;
  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "left wall",   "marker", 0, 0, NULL, (void (*)(void)) bc_u, 1, &id, user);CHKERRQ(ierr);
  ierr = PetscDSSetExactSolution(prob, 0, quartic_u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, const char name[], PetscErrorCode (*setup)(PetscDS, AppCtx *), AppCtx *user)
{
  DM             cdm = dm;
  PetscFE        fe;
  PetscDS        prob;
  char           prefix[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* Create finite element */
  ierr = PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", name);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) dm), user->dim, 1, user->simplex, name ? prefix : NULL, -1, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, name);CHKERRQ(ierr);
  /* Set discretization and boundary conditions for each mesh */
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) fe);CHKERRQ(ierr);
  ierr = (*setup)(prob, user);CHKERRQ(ierr);
  while (cdm) {
    ierr = DMSetDS(cdm, prob);CHKERRQ(ierr);
    /* TODO: Check whether the boundary of coarse meshes is marked */
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscDSSetFromOptions(prob);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;   /* Problem specification */
  SNES           snes; /* Nonlinear solver */
  Vec            u;    /* Solutions */
  AppCtx         user; /* User-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  /* Primal system */
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);
  ierr = SetupDiscretization(dm, "potential", SetupPrimalProblem, &user);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = VecSet(u, 0.0);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "potential");CHKERRQ(ierr);
  ierr = DMPlexSetSNESLocalFEM(dm, &user, &user, &user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = DMSNESCheckFromOptions(snes, u, NULL, NULL);CHKERRQ(ierr);
  ierr = SNESSolve(snes, NULL, u);CHKERRQ(ierr);
  ierr = SNESGetSolution(snes, &u);CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-potential_view");CHKERRQ(ierr);
  /* Cleanup */
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 2d_p4_0
    requires: triangle
    args: -cells 2,2 -dm_plex_separate_marker -curved -potential_petscspace_order 4 -coords_petscspace_order 2 -dmsnes_check -snes_converged_reason -snes_error_if_not_converged -ksp_rtol 1e-10

TEST*/
