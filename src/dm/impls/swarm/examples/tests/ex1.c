static char help[] = "Tests projection with DMSwarm.\n";

#include <petsc/private/dmswarmimpl.h>
#include <petsc/private/petscfeimpl.h>

#include <petscdmplex.h>
#include <petscds.h>
#include <petscksp.h>

typedef struct {
  PetscInt  dim;                        /* The topological mesh dimension */
  PetscBool simplex;                    /* Flag for simplices or tensor cells */
  char      mshNam[PETSC_MAX_PATH_LEN]; /* Name of the mesh filename if any */
  /* meshes */
  PetscInt  nbrVerEdge;                 /* Number of vertices per edge if unit square/cube generated */
  PetscInt  particles_cell;
  /* geometry  */
  PetscReal      domain_lo[3], domain_hi[3];
  DMBoundaryType boundary[3];        /* The domain boundary */
  /* test */
  PetscInt       k;
  PetscReal      factor;                /* cache for scaling */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;
  PetscInt       ii, bd;
  PetscFunctionBeginUser;
  options->dim     = 2;
  options->simplex = PETSC_TRUE;
  options->domain_lo[0]  = 0.0;
  options->domain_lo[1]  = 0.0;
  options->domain_lo[2]  = 0.0;
  options->domain_hi[0]  = 2*PETSC_PI;
  options->domain_hi[1]  = 1.0;
  options->domain_hi[2]  = 1.0;
  options->boundary[0]= DM_BOUNDARY_NONE; /* PERIODIC (plotting does not work in paralle) */
  options->boundary[1]= DM_BOUNDARY_NONE; /* Neumann */
  options->boundary[2]= DM_BOUNDARY_NONE;
  options->particles_cell = 0; /* > 0 for grid of particles, 0 for quadrature points */
  options->k = 1;
  options->factor = 1;
  ierr = PetscStrcpy(options->mshNam, "");CHKERRQ(ierr);

  ierr = PetscOptionsBegin(comm, "", "Meshing Adaptation Options", "DMPLEX");CHKERRQ(ierr);

  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex1.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "The flag for simplices or tensor cells", "ex1.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-msh", "Name of the mesh filename if any", "ex1.c", options->mshNam, options->mshNam, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-nbrVerEdge", "Number of vertices per edge if unit square/cube generated", "ex1.c", options->nbrVerEdge, &options->nbrVerEdge, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-k", "Mode number of test", "ex1.c", options->k, &options->k, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-particles_cell", "Number of particles per cell", "ex1.c", options->particles_cell, &options->particles_cell, NULL);CHKERRQ(ierr);
  ii = options->dim;
  ierr = PetscOptionsRealArray("-domain_hi", "Domain size", "ex48.c", options->domain_hi, &ii, NULL);CHKERRQ(ierr);
  ii = options->dim;
  ierr = PetscOptionsRealArray("-domain_lo", "Domain size", "ex48.c", options->domain_lo, &ii, NULL);CHKERRQ(ierr);
  ii = options->dim;
  bd = options->boundary[0];
  ierr = PetscOptionsEList("-x_boundary", "The x-boundary", "ex48.c", DMBoundaryTypes, 5, DMBoundaryTypes[options->boundary[0]], &bd, NULL);CHKERRQ(ierr);
  options->boundary[0] = (DMBoundaryType) bd;
  bd = options->boundary[1];
  ierr = PetscOptionsEList("-y_boundary", "The y-boundary", "ex48.c", DMBoundaryTypes, 5, DMBoundaryTypes[options->boundary[1]], &bd, NULL);CHKERRQ(ierr);
  options->boundary[1] = (DMBoundaryType) bd;
  bd = options->boundary[2];
  ierr = PetscOptionsEList("-z_boundary", "The z-boundary", "ex48.c", DMBoundaryTypes, 5, DMBoundaryTypes[options->boundary[2]], &bd, NULL);CHKERRQ(ierr);
  options->boundary[2] = (DMBoundaryType) bd;

  ierr = PetscOptionsEnd();

  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *user)
{
  PetscBool      flag;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscStrcmp(user->mshNam, "", &flag);CHKERRQ(ierr);
  if (flag) {
    PetscInt faces[3];
    faces[0] = user->nbrVerEdge-1; faces[1] = user->nbrVerEdge-1; faces[2] = user->nbrVerEdge-1;
    ierr = DMPlexCreateBoxMesh(comm, user->dim, user->simplex, faces, user->domain_lo, user->domain_hi, user->boundary, PETSC_TRUE, dm);CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateFromFile(comm, user->mshNam, PETSC_TRUE, dm);CHKERRQ(ierr);
    ierr = DMGetDimension(*dm, &user->dim);CHKERRQ(ierr);
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
  ierr = DMLocalizeCoordinates(*dm);CHKERRQ(ierr); /* needed for periodic */
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode linear(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;

  u[0] = 0.0;
  for (d = 0; d < dim; ++d) u[0] += x[d];
  return 0;
}

static PetscErrorCode sinx(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *a_ctx)
{
  AppCtx *ctx = (AppCtx*)a_ctx;
  u[0] = sin(x[0]*ctx->k);
/* PetscPrintf(PETSC_COMM_SELF, "[%D]sinx: x = %12.5e,%12.5e, val=%12.5e\n",-1,x[0],x[1],u[0]); */
  return 0;
}

static void g3_1(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) {
    g3[d*dim+d] = 1;
  }
}

static void identity(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = 1.0;
}

static PetscErrorCode CreateParticles(DM dm, DM *sw, AppCtx *user)
{
  PetscDS          prob;
  PetscFE          fe;
  PetscInt         Nq=0, Np=0, q, dim, N=0, c, cell, p, cStart, cEnd;
  PetscReal       *v0, *J, *invJ, detJ, *coords, *xi0;
  PetscInt        *cellid;
  PetscErrorCode   ierr;
  PetscMPIInt      rank;
  const PetscReal *qpoints=0;
  PetscFunctionBeginUser;
  MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, 1, user->simplex, NULL, -1, &fe);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) fe);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 0, identity, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMCreate(PetscObjectComm((PetscObject) dm), sw);CHKERRQ(ierr);
  ierr = DMSetType(*sw, DMSWARM);CHKERRQ(ierr);
  ierr = DMSetDimension(*sw, dim);CHKERRQ(ierr);
  ierr = DMSwarmSetType(*sw, DMSWARM_PIC);CHKERRQ(ierr);
  ierr = DMSwarmSetCellDM(*sw, dm);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(*sw, "f_q", 1, PETSC_SCALAR);CHKERRQ(ierr);
  ierr = DMSwarmFinalizeFieldRegister(*sw);CHKERRQ(ierr);
  if (user->particles_cell == 0) { /* make particles at quadrature points */
    PetscQuadrature  quad;
    ierr = PetscDSGetDiscretization(prob, 0, (PetscObject *) &fe);CHKERRQ(ierr);
    ierr = PetscFEGetQuadrature(fe, &quad);CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(quad, NULL, NULL, &Nq, &qpoints, NULL);CHKERRQ(ierr);
    ierr = DMSwarmSetLocalSizes(*sw, (cEnd-cStart) * Nq, 0);CHKERRQ(ierr);
    Np = Nq;
  } else {
    N = PetscCeilReal(PetscPowReal((PetscReal)user->particles_cell,1./(double)dim));
    Np = user->particles_cell = PetscPowReal((PetscReal)N,(PetscReal)dim); /* change p/c to make fit */
    if (user->particles_cell<1) SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "user->particles_cell<1 ???????");
    q = (cEnd-cStart) * user->particles_cell;
    ierr = DMSwarmSetLocalSizes(*sw, q, 0);CHKERRQ(ierr);
PetscPrintf(PetscObjectComm((PetscObject) dm),"CreateParticles: particles/cell=%D, N cells_x = %D, number local particels=%D, weight factor %g\n",user->particles_cell,user->nbrVerEdge-1,q,user->factor);
  }
  ierr = DMSetFromOptions(*sw);CHKERRQ(ierr);
  q = (cEnd-cStart) * Np;
  ierr = MPI_Allreduce(&q,&c,1,MPI_INT,MPI_SUM,PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
  user->factor = PetscPowReal((PetscReal)2,(PetscReal)dim)/(double)Np;
  /* create each particle: set cellid and coord */
  ierr = PetscMalloc4(dim, &xi0, dim, &v0, dim*dim, &J, dim*dim, &invJ);CHKERRQ(ierr);
  for (c = 0; c < dim; c++) xi0[c] = -1.;
  ierr = DMSwarmGetField(*sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = DMSwarmGetField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid);CHKERRQ(ierr);
  for (cell = cStart, c = 0; cell < cEnd; ++cell, ++c) {
    ierr = DMPlexComputeCellGeometryFEM(dm, cell, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr); /* affine */
    if (user->particles_cell == 0) { /* make particles at quadrature points */
      for (q = 0; q < Nq; ++q) {
        cellid[c*Nq + q] = cell;
        CoordinatesRefToReal(dim, dim, xi0, v0, J, &qpoints[q*dim], &coords[(c*Nq + q)*dim]);
      }
    } else { /* make particles in cells with regular grid, assumes tensor elements (-1,1)^D*/
      PetscInt  ii,jj,kk;
      PetscReal ecoord[3];
      const PetscReal dx = 2./(PetscReal)N, dx_2 = dx/2;
      for ( p = kk = 0; kk < (dim==3 ? N : 1) ; kk++) {
        ecoord[2] = kk*dx - 1 + dx_2; /* regular grid on [-1,-1] */
        for ( ii = 0; ii < N ; ii++) {
          ecoord[0] = ii*dx - 1 + dx_2; /* regular grid on [-1,-1] */
          for ( jj = 0; jj < N ; jj++, p++) {
            ecoord[1] = jj*dx - 1 + dx_2; /* regular grid on [-1,-1] */
            cellid[c*Np + p] = cell;
            CoordinatesRefToReal(dim, dim, xi0, v0, J, ecoord, &coords[(c*Np + p)*dim]);
          }
        }
      }
      if (p!=Np) SETERRQ2(PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "p %D != Np %D",p,Np);
    }
  }
  ierr = DMSwarmRestoreField(*sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid);CHKERRQ(ierr);
  ierr = PetscFree4(xi0, v0, J, invJ);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe,"fe");CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode TestL2Projection(DM dm, DM sw, AppCtx *user)
{
  PetscErrorCode (*funcs[1])(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *);
  AppCtx          *ctxs[1];
  KSP              ksp;
  Mat              mass;
  Vec              f_q, rhs, uproj, uexact;
  PetscReal        error,normerr,norm;
  PetscErrorCode   ierr;
  PetscScalar      none = -1.0;
  PetscInt         p, dim, Np, cStart, rStart, cEnd, cell, maxC, *idxbuf;
  const PetscReal *coords, *c2;
  PetscMPIInt      rank;
  PetscLayout      rLayout;
  PetscScalar     *vbuf;
  PetscFunctionBeginUser;
  funcs[0] = (user->particles_cell == 0) ? linear : sinx;
  ctxs[0] = user;
  MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMSwarmGetLocalSize(sw,&Np);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  /* create particle mass matrix */
  ierr = DMCreateMassMatrix(sw, dm, &mass);CHKERRQ(ierr);
  /* create RHS vector data */
  ierr = PetscLayoutCreate(PetscObjectComm((PetscObject) dm), &rLayout);CHKERRQ(ierr);
  ierr = PetscLayoutSetLocalSize(rLayout, Np);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(rLayout, 1);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(rLayout);CHKERRQ(ierr);
  ierr = PetscLayoutGetRange(rLayout, &rStart, NULL);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&rLayout);CHKERRQ(ierr);
  ierr = DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);

  ierr = VecCreate(PetscObjectComm((PetscObject) dm),&f_q);CHKERRQ(ierr);
  ierr = VecSetSizes(f_q,Np,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(f_q);CHKERRQ(ierr);
  ierr = DMSwarmSortGetAccess(sw);CHKERRQ(ierr);
  for (cell = cStart, maxC = 0; cell < cEnd; ++cell) {
    PetscInt *cindices;
    PetscInt  numCIndices;
    ierr = DMSwarmSortGetPointsPerCell(sw, cell, &numCIndices, &cindices);CHKERRQ(ierr);
    ierr = PetscFree(cindices);CHKERRQ(ierr);
    maxC = PetscMax(maxC, numCIndices);
  }
  ierr = PetscMalloc1(maxC, &idxbuf);CHKERRQ(ierr);
  ierr = PetscMalloc1(maxC, &vbuf);CHKERRQ(ierr);
  for (cell = cStart, c2 = coords; cell < cEnd; ++cell) {
    PetscInt *cindices;
    PetscInt  numCIndices;
    ierr = DMSwarmSortGetPointsPerCell(sw, cell, &numCIndices, &cindices);CHKERRQ(ierr);
    for (p = 0; p < numCIndices; ++p) {
      funcs[0](dim, 0.0, &c2[p*dim], 1, &vbuf[p], user);
      vbuf[p] *= user->factor;
      idxbuf[p] = cindices[p] + rStart;
/* PetscPrintf(PETSC_COMM_SELF, "[%D]TestL2Projection: %D/%D) real coord[%4D]:%12.5e,%12.5e, rhs[%D]=%12.5e\n",rank,c2-coords+1,Np,p*dim,c2[p*dim],c2[p*dim + 1],p,buf[p]); */
    }
    ierr = VecSetValues(f_q,numCIndices,idxbuf,vbuf,INSERT_VALUES);CHKERRQ(ierr);
    c2 += numCIndices*dim;
    ierr = PetscFree(cindices);CHKERRQ(ierr);
  }
  if ((c2-coords)!=Np*dim) SETERRQ2(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Indexing error iteration: count %D, %D local particles",c2-coords,Np*dim);
  ierr = DMSwarmSortRestoreAccess(sw);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(f_q);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(f_q);CHKERRQ(ierr);
  ierr = PetscFree(vbuf);CHKERRQ(ierr);
  ierr = PetscFree(idxbuf);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject)f_q, NULL, "-f_view");CHKERRQ(ierr);
  /* make RHS */
  ierr = DMGetGlobalVector(dm, &rhs);CHKERRQ(ierr);
  ierr = MatMultTranspose(mass, f_q, rhs);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) rhs,"rhs");CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject)rhs, NULL, "-vec_view");CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject)rhs, NULL, "-rhs_view");CHKERRQ(ierr);
  ierr = VecDestroy(&f_q);CHKERRQ(ierr);
  ierr = MatViewFromOptions(mass, NULL, "-particle_mass_mat_view");CHKERRQ(ierr);
  ierr = MatDestroy(&mass);CHKERRQ(ierr);

  /* make FE mass, solve, project, compute error */
  ierr = DMGetGlobalVector(dm, &uproj);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm, &mass);CHKERRQ(ierr);
  ierr = DMPlexSNESComputeJacobianFEM(dm, uproj, mass, mass, user);CHKERRQ(ierr);
  ierr = MatViewFromOptions(mass, NULL, "-fe_mass_mat_view");CHKERRQ(ierr);
  ierr = KSPCreate(PetscObjectComm((PetscObject)dm), &ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, mass, mass);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp, rhs, uproj);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) uproj,"u");CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject)uproj, NULL, "-vec_view");CHKERRQ(ierr);
  ierr = DMComputeL2Diff(dm, 0.0, funcs, (void**)ctxs, uproj, &error);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm, &uexact);CHKERRQ(ierr);
  ierr = DMProjectFunction(dm, 0.0, funcs, (void**)ctxs, INSERT_ALL_VALUES, uexact);CHKERRQ(ierr);
  ierr = VecNorm(uexact, NORM_2, &norm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) uexact, "exact");CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject)uexact, NULL, "-vec_view");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) uproj,"u");CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject)uproj, NULL, "-vec_view");CHKERRQ(ierr);
  ierr = VecAYPX(uexact,none,uproj);CHKERRQ(ierr); /* uexact = error function */
  ierr = PetscObjectSetName((PetscObject) uexact, "error");CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject)uexact, NULL, "-vec_view");CHKERRQ(ierr);
  ierr = VecNorm(uexact, NORM_2, &normerr);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "relative discrete error = %g, |exact| = %g, Projected L2 error = %g\n", normerr/norm, norm, error);CHKERRQ(ierr);

  /* clean up */
  ierr = MatDestroy(&mass);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &rhs);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &uproj);CHKERRQ(ierr);
  ierr = VecDestroy(&uexact);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main (int argc, char * argv[]) {
  MPI_Comm       comm;
  DM             dm, sw;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);
  ierr = CreateMesh(comm, &dm, &user);CHKERRQ(ierr);
  ierr = CreateParticles(dm, &sw, &user);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dm, "Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = DMViewFromOptions(sw, NULL, "-sw_view");CHKERRQ(ierr);
  ierr = TestL2Projection(dm, sw, &user);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&sw);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}

/*TEST

  test:
    suffix: proj_0
    requires: pragmatic
    args: -dim 2 -nbrVerEdge 3 -dm_plex_separate_marker 0 -dm_view -sw_view -petscspace_order 1 -petscfe_default_quadrature_order 1 -pc_type lu
  test:
    suffix: proj_1
    requires: pragmatic
    args: -dim 2 -simplex 0 -nbrVerEdge 3 -dm_plex_separate_marker 0 -dm_view -sw_view -petscspace_order 1 -petscfe_default_quadrature_order 1 -pc_type lu
  test:
    suffix: proj_2
    requires: pragmatic
    args: -dim 3 -nbrVerEdge 3 -dm_view -sw_view -petscspace_order 1 -petscfe_default_quadrature_order 1 -pc_type lu
  test:
    suffix: proj_3
    requires: pragmatic
    args: -dim 2 -simplex 0 -nbrVerEdge 3 -dm_plex_separate_marker 0 -dm_view -sw_view -petscspace_order 1 -petscfe_default_quadrature_order 1 -pc_type lu

TEST*/
