static char help[] = "2d and 3d particle-in-cell code.\n\
We solve a Poisson problem (eg, gravity or electrostatics) a\n\
rectangular domain, using a parallel unstructured mesh (DMPLEX) \n\
and particles (DMSWARM).\n\n\n";

/*
The Laplacian, which we discretize using the finite element method on an unstructured mesh. The weak form equations are

  < \nabla v, \nabla u > + < v, f > = 0

where the rhs function f depends on the particles. We start with a delta function discretization of the rhs function

  f = \sum_j f_j \delta(x - x_j)
*/

#include <petscdmplex.h>
#include <petsc/private/petscfeimpl.h>
#include <petscdmswarm.h>
#include <petscsnes.h>
#include <petscds.h>

typedef enum {NEUMANN, DIRICHLET} BCType;

typedef struct {
  /* Domain and mesh definition */
  PetscInt          dim;        /* The topological mesh dimension */
  PetscBool         simplex;    /* Use simplices or tensor product cells */
  /* Particle definition */
  PetscInt          Npc;        /* The nubmer of particles per cell */
  /* Problem definition */
  BCType            bcType;
  PetscBool         usePartRhs; /* Use the particle definition for the rhs */
  PetscErrorCode (**exactFuncs)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
} AppCtx;

/*
  In 2D and 3D we use exact solution:

    u =  sin(2\pi x)
    f = -4\pi^2 sin(2\pi x)

  so that

    -\Delta u + f = 4\pi^2 sin(2\pi x) - 4\pi^2 sin(2\pi x)
*/
#if 0
PetscErrorCode exact_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = sin(2*PETSC_PI*x[0]);
  return 0;
}

void f0_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = -4.0*PetscSqr(PETSC_PI)*sin(2*PETSC_PI*x[0]);
}
#else
PetscErrorCode exact_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = x[0]*x[0] + x[1]*x[1];
  return 0;
}

void f0_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = 4.0;
}
#endif

/* gradU[comp*dim+d] = {u_x, u_y} or {u_x, u_y, u_z} */
void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u_x[d];
}


static void identity(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = 1.0;
}

/* < \nabla v, \nabla u > This just gives \nabla u */
void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d*dim+d] = 1.0;
}

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  const char    *bcTypes[2]  = {"neumann", "dirichlet"};
  PetscInt       bc;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->dim        = 2;
  options->simplex    = PETSC_TRUE;
  options->Npc        = 1;
  options->bcType     = DIRICHLET;
  options->usePartRhs = PETSC_TRUE;

  ierr = PetscOptionsBegin(comm, "", "PIC Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex64.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Use simplices or tensor product cells", "ex64.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-npc", "The number of particle per cell", "ex64.c", options->Npc, &options->Npc, NULL);CHKERRQ(ierr);
  bc   = options->bcType;
  ierr = PetscOptionsEList("-bc_type","Type of boundary condition","ex64.c",bcTypes,2,bcTypes[options->bcType],&bc,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-use_part_rhs", "Use the particle definition of the rhs", "ex64.c", options->usePartRhs, &options->usePartRhs, NULL);CHKERRQ(ierr);
  options->bcType = (BCType) bc;
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  DM             pdm      = NULL;
  const PetscInt cells[3] = {2, 2, 2};
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexCreateBoxMesh(comm, user->dim, user->simplex, cells, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
  ierr = DMPlexDistribute(*dm, 0, NULL, &pdm);CHKERRQ(ierr);
  if (pdm) {ierr = DMDestroy(dm);CHKERRQ(ierr); *dm  = pdm;}
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CreateParticles(DM dm, AppCtx *user, DM *pdm)
{
  DM               sdm;
  PetscDS          prob;
  PetscFE          fe;
  PetscQuadrature  quad;
  const PetscReal *qpoints;
  PetscReal       *v0, *J, *invJ, detJ, *coords, *weight;
  PetscInt        *cellid;
  PetscInt         dim, Nq, q, cStart, cEnd, c, Nc;
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  ierr = DMCreate(PetscObjectComm((PetscObject) dm), &sdm);CHKERRQ(ierr);
  ierr = DMSetType(sdm, DMSWARM);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMSetDimension(sdm, dim);CHKERRQ(ierr);
  ierr = DMSwarmSetCellDM(sdm, dm);CHKERRQ(ierr);
  /* Setup particle information */
  ierr = DMSwarmSetType(sdm, DMSWARM_PIC);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(sdm, "weight", 1, PETSC_SCALAR);CHKERRQ(ierr);
  ierr = DMSwarmFinalizeFieldRegister(sdm);CHKERRQ(ierr);
  /* Setup number of particles and coordinates */
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetDiscretization(prob, 0, (PetscObject *) &fe);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe, &quad);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quad, NULL, NULL, &Nq, &qpoints, NULL);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  Nc   = cEnd - cStart;
  ierr = DMSwarmSetLocalSizes(sdm, Nc * Nq, 0);CHKERRQ(ierr);
  ierr = DMSetFromOptions(sdm);CHKERRQ(ierr);

  ierr = PetscMalloc3(dim, &v0, dim*dim, &J, dim*dim, &invJ);CHKERRQ(ierr);
  ierr = DMSwarmGetField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = DMSwarmGetField(sdm, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid);CHKERRQ(ierr);
  ierr = DMSwarmGetField(sdm, "weight", NULL, NULL, (void **) &weight);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    for (q = 0; q < Nq; ++q) {
      ierr = DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr);
      cellid[c*Nq + q] = c;
      CoordinatesRefToReal(dim, dim, v0, J, &qpoints[q*dim], &coords[(c*Nq + q)*dim]);
      f0_u(dim, 1, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0.0, &coords[(c*Nq + q)*dim], 0, NULL, &weight[c*Nq + q]);
      weight[c*Nq + q] *= -1.0;
    }
  }
  ierr = DMSwarmRestoreField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(sdm, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(sdm, "weight", NULL, NULL, (void **) &weight);CHKERRQ(ierr);
  ierr = DMViewFromOptions(sdm, NULL, "-part_dm_view");CHKERRQ(ierr);
  ierr = PetscFree3(v0, J, invJ);CHKERRQ(ierr);
  *pdm = sdm;
  PetscFunctionReturn(0);
}

PetscErrorCode SetupProblem(DM dm, AppCtx *user)
{
  PetscDS        prob;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  if (user->usePartRhs) {ierr = PetscDSSetResidual(prob, 0, 0,    f1_u);CHKERRQ(ierr);}
  else                  {ierr = PetscDSSetResidual(prob, 0, f0_u, f1_u);CHKERRQ(ierr);}
  ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_uu);CHKERRQ(ierr);
  switch (user->dim) {
  case 2:
    user->exactFuncs[0] = exact_u;
    break;
  case 3:
    user->exactFuncs[0] = exact_u;
    break;
  default:
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension %d", user->dim);
  }
  ierr = PetscDSSetExactSolution(prob, 0, user->exactFuncs[0]);CHKERRQ(ierr);
  ierr = PetscDSSetImplicit(prob, 0, PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscDSSetContext(prob, 0, user);CHKERRQ(ierr);
  ierr = PetscDSSetFromOptions(prob);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  DM              cdm = dm;
  const PetscInt  id  = 1;
  PetscFE         fe;
  PetscDS         prob;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  /* Create finite element */
  ierr = PetscFECreateDefault(dm, user->dim, 1, user->simplex, "potential_", PETSC_DEFAULT, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, "potential");CHKERRQ(ierr);
  /* Set discretization and boundary conditions for each mesh */
  while (cdm) {
    ierr = DMGetDS(cdm, &prob);CHKERRQ(ierr);
    ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) fe);CHKERRQ(ierr);
    ierr = SetupProblem(cdm, user);CHKERRQ(ierr);
    ierr = DMAddBoundary(cdm, user->bcType == DIRICHLET ? DM_BC_ESSENTIAL : DM_BC_NATURAL, "wall", user->bcType == NEUMANN ? "boundary" : "marker", 0, 0, NULL, (void (*)()) user->exactFuncs[0], 1, &id, user);CHKERRQ(ierr);
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  SNES           snes;        /* nonlinear solver */
  DM             dm, sdm;     /* problem definition */
  Vec            u, b = NULL; /* solution and particle rhs vectors */
  AppCtx         user;        /* user-defined work context */
  PetscReal      error;       /* L_2 error in the solution */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = PetscMalloc(1 * sizeof(void (*)(const PetscReal[], PetscScalar *, void *)), &user.exactFuncs);CHKERRQ(ierr);
  /* Setup PDE mesh and solver */
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = CreateMesh(PetscObjectComm((PetscObject) snes), &user, &dm);CHKERRQ(ierr);
  ierr = SetupDiscretization(dm, &user);CHKERRQ(ierr);
  ierr = CreateParticles(dm, &user, &sdm);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);
  ierr = DMPlexSetSNESLocalFEM(dm, &user, &user, &user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, &user);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = VecSet(u, 0.0);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "Solution");CHKERRQ(ierr);
  if (user.usePartRhs) {
    Mat mass;
    Vec up;

    ierr = DMSwarmCreateGlobalVectorFromField(sdm, "weight", &up);CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(dm, &b);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) b, "Particle Rhs");CHKERRQ(ierr);
    ierr = DMCreateMassMatrix(sdm, dm, &mass);CHKERRQ(ierr);
    ierr = MatMult(mass, up, b);CHKERRQ(ierr);
    ierr = MatDestroy(&mass);CHKERRQ(ierr);
    ierr = VecDestroy(&up);CHKERRQ(ierr);
    ierr = VecViewFromOptions(b, NULL, "-rhs_vec_view");CHKERRQ(ierr);
    {
      PetscDS prob;
      KSP     ksp;
      Vec     fproj;

      ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
      ierr = DMGetGlobalVector(dm, &fproj);CHKERRQ(ierr);
      ierr = DMCreateMatrix(dm, &mass);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 0, 0, identity, NULL, NULL, NULL);CHKERRQ(ierr);
      ierr = DMPlexSNESComputeJacobianFEM(dm, fproj, mass, mass, &user);CHKERRQ(ierr);
      ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_uu);CHKERRQ(ierr);
      //ierr = MatViewFromOptions(mass, NULL, "-mass_mat_view");CHKERRQ(ierr);
      ierr = KSPCreate(PetscObjectComm((PetscObject) dm), &ksp);CHKERRQ(ierr);
      ierr = KSPSetOperators(ksp, mass, mass);CHKERRQ(ierr);
      ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
      ierr = KSPSolve(ksp, b, fproj);CHKERRQ(ierr);
      ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) fproj, "Rhs function projection");CHKERRQ(ierr);
      ierr = VecViewFromOptions(fproj, NULL, "-rhs_vec_view");CHKERRQ(ierr);
      ierr = MatDestroy(&mass);CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(dm, &fproj);CHKERRQ(ierr);
    }
  }
  ierr = SNESSolve(snes, b, u);CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-solution_vec_view");CHKERRQ(ierr);
  /* Now that we have DS exact sol, we can have an L2 error and error vec monitor */
  {
    Vec r;

    ierr = DMComputeL2Diff(dm, 0.0, user.exactFuncs, NULL, u, &error);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: %.3g\n", error);CHKERRQ(ierr);

    ierr = DMGetGlobalVector(dm, &r);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) r, "Solution Error");CHKERRQ(ierr);
    ierr = DMProjectFunction(dm, 0.0, user.exactFuncs, NULL, INSERT_ALL_VALUES, r);CHKERRQ(ierr);
    ierr = VecAXPY(r, -1.0, u);CHKERRQ(ierr);
    ierr = VecViewFromOptions(r, NULL, "-error_vec_view");CHKERRQ(ierr);
    if (b) {
      PetscDS prob;
      Vec     rhs;

      ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
      ierr = VecSet(r, 0.0);CHKERRQ(ierr);
      ierr = DMGetGlobalVector(dm, &rhs);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) rhs, "Analytic Rhs");CHKERRQ(ierr);
      ierr = PetscDSSetResidual(prob, 0, f0_u, f1_u);CHKERRQ(ierr);
      ierr = SNESComputeFunction(snes, r, rhs);CHKERRQ(ierr);
      ierr = PetscDSSetResidual(prob, 0, 0,    f1_u);CHKERRQ(ierr);
      ierr = VecViewFromOptions(rhs, NULL, "-rhs_vec_view");CHKERRQ(ierr);
      ierr = VecAXPY(rhs, -1.0, b);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) rhs, "Rhs Error");CHKERRQ(ierr);
      ierr = VecViewFromOptions(rhs, NULL, "-rhs_error_vec_view");CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(dm, &rhs);CHKERRQ(ierr);
    }
    ierr = DMRestoreGlobalVector(dm, &r);CHKERRQ(ierr);
  }

  /* Move particles */
  if (0) {
    Vec       lu;
    PetscReal dt = 0.01;
    PetscInt  dim, numSteps = 30, tn;

    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = DMGetLocalVector(dm, &lu);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dm, u, INSERT_VALUES, lu);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dm, u, INSERT_VALUES, lu);CHKERRQ(ierr);
    for (tn = 0; tn < numSteps; ++tn) {
      DMInterpolationInfo ictx;
      Vec                 pu;
      const PetscScalar  *pv;
      PetscReal          *coords;
      PetscInt            Np, p, d;

      ierr = DMCreateGlobalVector(sdm, &pu);CHKERRQ(ierr);
      ierr = DMSwarmVectorDefineField(sdm, DMSwarmPICField_coor);CHKERRQ(ierr);
      ierr = DMSwarmGetLocalSize(sdm, &Np);CHKERRQ(ierr);
      ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Timestep: %D Np: %D\n", tn, Np);CHKERRQ(ierr);
      ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD, NULL);CHKERRQ(ierr);
      /* Interpolate velocity */
      ierr = DMInterpolationCreate(PETSC_COMM_SELF, &ictx);CHKERRQ(ierr);
      ierr = DMInterpolationSetDim(ictx, dim);CHKERRQ(ierr);
      ierr = DMInterpolationSetDof(ictx, dim);CHKERRQ(ierr);
      ierr = DMSwarmGetField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
      ierr = DMInterpolationAddPoints(ictx, Np, coords);CHKERRQ(ierr);
      ierr = DMSwarmRestoreField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
      ierr = DMInterpolationSetUp(ictx, dm, PETSC_FALSE);CHKERRQ(ierr);
      ierr = DMInterpolationEvaluate(ictx, dm, lu, pu);CHKERRQ(ierr);
      ierr = DMInterpolationDestroy(&ictx);CHKERRQ(ierr);
      /* Push particles */
      ierr = DMSwarmGetField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
      ierr = VecGetArrayRead(pu, &pv);CHKERRQ(ierr);
      for (p = 0; p < Np; ++p) {
        for (d = 0; d < dim; ++d) coords[p*dim+d] += dt*pv[p*dim+d];
      }
      ierr = VecRestoreArrayRead(pu, &pv);CHKERRQ(ierr);
      ierr = DMSwarmRestoreField(sdm, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
      /* Migrate particles */
      ierr = DMSwarmMigrate(sdm, PETSC_TRUE);CHKERRQ(ierr);
      ierr = DMViewFromOptions(sdm, NULL, "-part_dm_view");CHKERRQ(ierr);
      ierr = VecDestroy(&pu);CHKERRQ(ierr);
    }
    ierr = DMRestoreLocalVector(dm, &lu);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&sdm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFree(user.exactFuncs);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 2d_p1_0
    args: -potential_petscspace_order 1 -dm_refine 1 -ksp_rtol 1e-9

TEST*/
