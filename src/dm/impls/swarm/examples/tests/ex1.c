static char help[] = "Tests projection with DMSwarm.\n";

#include <petsc/private/dmswarmimpl.h>
#include <petsc/private/petscfeimpl.h>

#include <petscdmplex.h>
#include <petscds.h>
#include <petscksp.h>

typedef struct {
  PetscInt  dim;                        /* The topological mesh dimension */
  PetscBool simplex;                    /* Flag for simplices or tensor cells */
  char      meshName[PETSC_MAX_PATH_LEN]; /* Name of the mesh filename if any */
  /* meshes */
  PetscInt  nbrVerEdge;                 /* Number of vertices per edge if unit square/cube generated */
  PetscInt  particles_cell;
  /* geometry  */
  PetscReal      domain_lo[3], domain_hi[3];
  DMBoundaryType boundary[3];        /* The domain boundary */
  PetscReal      particle_perturbation;
  PetscReal      mesh_perturbation;
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
  options->boundary[0]= DM_BOUNDARY_NONE; /* PERIODIC (plotting does not work in parallel, moments not conserved) */
  options->boundary[1]= DM_BOUNDARY_NONE; /* Neumann */
  options->boundary[2]= DM_BOUNDARY_NONE;
  options->particles_cell = 0; /* > 0 for grid of particles, 0 for quadrature points */
  options->k = 1;
  options->factor = 1;
  options->particle_perturbation = 1.e-20;
  options->mesh_perturbation = 1.e-20;
  ierr = PetscStrcpy(options->meshName, "");CHKERRQ(ierr);

  ierr = PetscOptionsBegin(comm, "", "Meshing Adaptation Options", "DMPLEX");CHKERRQ(ierr);

  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex1.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "The flag for simplices or tensor cells", "ex1.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-msh", "Name of the mesh filename if any", "ex1.c", options->meshName, options->meshName, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-nbrVerEdge", "Number of vertices per edge if unit square/cube generated", "ex1.c", options->nbrVerEdge, &options->nbrVerEdge, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-k", "Mode number of test", "ex1.c", options->k, &options->k, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-particles_cell", "Number of particles per cell", "ex1.c", options->particles_cell, &options->particles_cell, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-particle_perturbation", "Relative perturbation of particles (0,1)", "ex1.c", options->particle_perturbation, &options->particle_perturbation, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mesh_perturbation", "Relative perturbation of mesh points (0,1)", "ex1.c", options->mesh_perturbation, &options->mesh_perturbation, NULL);CHKERRQ(ierr);
  ii = options->dim;
  ierr = PetscOptionsRealArray("-domain_hi", "Domain size", "ex1.c", options->domain_hi, &ii, NULL);CHKERRQ(ierr);
  ii = options->dim;
  ierr = PetscOptionsRealArray("-domain_lo", "Domain size", "ex1.c", options->domain_lo, &ii, NULL);CHKERRQ(ierr);
  ii = options->dim;
  bd = options->boundary[0];
  ierr = PetscOptionsEList("-x_boundary", "The x-boundary", "ex1.c", DMBoundaryTypes, 5, DMBoundaryTypes[options->boundary[0]], &bd, NULL);CHKERRQ(ierr);
  options->boundary[0] = (DMBoundaryType) bd;
  bd = options->boundary[1];
  ierr = PetscOptionsEList("-y_boundary", "The y-boundary", "ex1.c", DMBoundaryTypes, 5, DMBoundaryTypes[options->boundary[1]], &bd, NULL);CHKERRQ(ierr);
  options->boundary[1] = (DMBoundaryType) bd;
  bd = options->boundary[2];
  ierr = PetscOptionsEList("-z_boundary", "The z-boundary", "ex1.c", DMBoundaryTypes, 5, DMBoundaryTypes[options->boundary[2]], &bd, NULL);CHKERRQ(ierr);
  options->boundary[2] = (DMBoundaryType) bd;

  ierr = PetscOptionsEnd();

  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *user)
{
  PetscBool      flag;
  PetscErrorCode ierr,i;
  PetscReal      hh[3];
  PetscFunctionBeginUser;
  ierr = PetscStrcmp(user->meshName, "", &flag);CHKERRQ(ierr);
  if (flag) {
    PetscInt faces[3];
    faces[0] = user->nbrVerEdge-1; faces[1] = user->nbrVerEdge-1; faces[2] = user->nbrVerEdge-1;
    ierr = DMPlexCreateBoxMesh(comm, user->dim, user->simplex, faces, user->domain_lo, user->domain_hi, user->boundary, PETSC_TRUE, dm);CHKERRQ(ierr);
    for (i=0;i<user->dim;i++) hh[i] = (user->domain_hi[i]-user->domain_lo[i])/faces[i];
  } else {
    ierr = DMPlexCreateFromFile(comm, user->meshName, PETSC_TRUE, dm);CHKERRQ(ierr);
    ierr = DMGetDimension(*dm, &user->dim);CHKERRQ(ierr);
    for (i=0;i<user->dim;i++) hh[i] = 1;
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
  /* pertub coordinates */
  {
    Vec         coordinates;
    PetscScalar *coords;
    PetscInt    i, dimEmbed, nCoords;
    PetscRandom rnd;
    PetscReal   interval = user->mesh_perturbation;
    ierr = PetscRandomCreate(PetscObjectComm((PetscObject)*dm),&rnd);CHKERRQ(ierr);
    ierr = PetscRandomSetInterval(rnd,-interval,interval);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(rnd);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(*dm,&coordinates);CHKERRQ(ierr);
    ierr = DMGetCoordinateDim(*dm,&dimEmbed);CHKERRQ(ierr);
    ierr = VecGetLocalSize(coordinates,&nCoords);CHKERRQ(ierr);
    if (nCoords % dimEmbed) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Coordinate vector the wrong size");
    ierr = VecGetArray(coordinates,&coords);CHKERRQ(ierr);
    for (i = 0; i < nCoords; i += dimEmbed) {
      PetscInt j, pert = 1;
      PetscScalar *coord = &coords[i];
      for (j = 0; j < dimEmbed ; j++) {
        if (user->domain_hi[j] == coord[j] || coord[j] == user->domain_lo[j]) pert = 0;
      }
      if (pert) {
        PetscScalar value;
        for (j = 0; j < dimEmbed ; j++) {
          ierr = PetscRandomGetValue(rnd,&value);CHKERRQ(ierr);
          coord[j] += value*hh[j];
        }
      }
    }
    ierr = VecRestoreArray(coordinates,&coords);CHKERRQ(ierr);
    ierr = DMSetCoordinatesLocal(*dm,coordinates);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) *dm, "half-plane");CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&rnd);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode linear(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *a_ctx)
{
  AppCtx *ctx = (AppCtx*)a_ctx;
  if (x[0] < ctx->domain_hi[0]/2) u[0] = x[0]*2/ctx->domain_hi[0];
  else                            u[0] = 2 - x[0]*2/ctx->domain_hi[0];
  return 0;
}

static PetscErrorCode x2_x4(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *a_ctx)
{
  AppCtx *ctx = (AppCtx*)a_ctx;
  int     i = 0;
  u[0] = 1;
  for(i=0;i<dim;i++){
    double L2 = ctx->domain_hi[i]*ctx->domain_hi[i], x2 = x[i]*x[i];
    u[0] *= x2*L2 - x2*x2;
  }

  return 0;
}

static PetscErrorCode sinx(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *a_ctx)
{
  AppCtx *ctx = (AppCtx*)a_ctx;
  u[0] = sin(x[0]*ctx->k);
/* PetscPrintf(PETSC_COMM_SELF, "[%D]sinx: x = %12.5e,%12.5e, val=%12.5e\n",-1,x[0],x[1],u[0]); */
  return 0;
}

static PetscErrorCode one(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *a_ctx)
{
  u[0] = 1;
  return 0;
}

static PetscErrorCode x0(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *a_ctx)
{
  u[0] = x[0];
  return 0;
}

static PetscErrorCode x0_2(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *a_ctx)
{
  u[0] = x[0]*x[0];
  return 0;
}

/* static void g3_1(PetscInt dim, PetscInt Nf, PetscInt NfAux, */
/*                   const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], */
/*                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], */
/*                   PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[]) */
/* { */
/*   PetscInt d; */
/*   for (d = 0; d < dim; ++d) { */
/*     g3[d*dim+d] = 1; */
/*   } */
/* } */

static void g0_1(PetscInt dim, PetscInt Nf, PetscInt NfAux,
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
  PetscInt         Nq=0, Np=0, q, dim, N=0, c, cell, p, cStart, cEnd, order = -1;
  PetscReal       *v0, *J, *invJ, detJ, *coords, *xi0, interval = user->particle_perturbation;
  PetscInt        *cellid;
  PetscErrorCode   ierr;
  PetscMPIInt      rank;
  const PetscReal *qpoints=0;
  PetscRandom      rnd;
  PetscQuadrature  quad;
  PetscDualSpace   QF;
  PetscFunctionBeginUser;
  MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);
  ierr = PetscRandomCreate(PetscObjectComm((PetscObject)dm),&rnd);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rnd,-interval,interval);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rnd);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, 1, user->simplex, NULL, -1, &fe);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) fe);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 0, g0_1, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = PetscDSGetDiscretization(prob, 0, (PetscObject *) &fe);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe, &quad);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quad, NULL, NULL, &Nq, &qpoints, NULL);CHKERRQ(ierr);
  ierr = PetscFEGetDualSpace(fe, &QF);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetOrder(QF, &order);CHKERRQ(ierr);
  ierr = DMCreate(PetscObjectComm((PetscObject) dm), sw);CHKERRQ(ierr);
  ierr = DMSetType(*sw, DMSWARM);CHKERRQ(ierr);
  ierr = DMSetDimension(*sw, dim);CHKERRQ(ierr);
  ierr = DMSwarmSetType(*sw, DMSWARM_PIC);CHKERRQ(ierr);
  ierr = DMSwarmSetCellDM(*sw, dm);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(*sw, "f_q", 1, PETSC_SCALAR);CHKERRQ(ierr);
  ierr = DMSwarmFinalizeFieldRegister(*sw);CHKERRQ(ierr);
  if (user->particles_cell == 0) { /* make particles at quadrature points */
    ierr = DMSwarmSetLocalSizes(*sw, (cEnd-cStart) * Nq, 0);CHKERRQ(ierr);
    Np = Nq;
    q = (cEnd-cStart) * Nq;
  } else {
    N = PetscCeilReal(PetscPowReal((PetscReal)user->particles_cell,1./(PetscReal)dim));
    Np = user->particles_cell = PetscPowReal((PetscReal)N,(PetscReal)dim); /* change p/c to make fit */
    if (user->particles_cell<1) SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "user->particles_cell<1 ???????");
    q = (cEnd-cStart) * user->particles_cell;
    ierr = DMSwarmSetLocalSizes(*sw, q, 0);CHKERRQ(ierr);
  }
  user->factor = 4/(double)Np; /* where does 4 come from? not integral of u(x), # of elements on vertex? */
  PetscPrintf(PetscObjectComm((PetscObject) dm),"CreateParticles: particles/cell=%D, N cells_x = %D, number local particels=%D scaling factor %g, %D quadrature points, %D order elements\n",user->particles_cell,user->nbrVerEdge-1,q,user->factor,Nq,order);
  ierr = DMSetFromOptions(*sw);CHKERRQ(ierr);
  q = (cEnd-cStart) * Np;
  ierr = MPI_Allreduce(&q,&c,1,MPI_INT,MPI_SUM,PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
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
      PetscInt    ii,jj,kk;
      PetscReal   ecoord[3];
      PetscReal   dx = 2./(PetscReal)N, dx_2 = dx/2;
      PetscScalar value;
      if (interval>dx_2) SETERRQ2(PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "Perturbation %g > dx/2 %g",interval,dx_2);
      for ( p = kk = 0; kk < (dim==3 ? N : 1) ; kk++) {
        ierr = PetscRandomGetValue(rnd,&value);CHKERRQ(ierr);
        ecoord[2] = kk*dx - 1 + dx_2 + value; /* regular grid on [-1,-1] */
        for ( ii = 0; ii < N ; ii++) {
          ierr = PetscRandomGetValue(rnd,&value);CHKERRQ(ierr);
          ecoord[0] = ii*dx - 1 + dx_2 + value; /* regular grid on [-1,-1] */
          for ( jj = 0; jj < N ; jj++, p++) {
            ierr = PetscRandomGetValue(rnd,&value);CHKERRQ(ierr);
            ecoord[1] = jj*dx - 1 + dx_2 + value; /* regular grid on [-1,-1] */
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
  ierr = PetscRandomDestroy(&rnd);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "f0_1"
/* < v, ru > */
static void f0_1(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                   const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                   PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  *f0 = u[0];
  /* PetscPrintf(PETSC_COMM_SELF,"f0_ex: rho=%12.5e\n",u[0]); */
}

#undef __FUNCT__
#define __FUNCT__ "f0_momx"
/* < v, u > */
static void f0_momx(PetscInt dim, PetscInt Nf, PetscInt NfAux,
		    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
		    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
		    PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  *f0 = x[0]*u[0];
}

#undef __FUNCT__
#define __FUNCT__ "f0_ex"
static void f0_ex(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  *f0 = x[0]*x[0]*u[0];
}

/* compute particle moments */
#undef __FUNCT__
#define __FUNCT__ "computeParticleMoments"
static PetscErrorCode computeParticleMoments(DM sw, DM dm, Vec f_q, AppCtx *user, double *den0tot, double *mom0tot, double *energy0tot)
{
  PetscErrorCode   ierr;
  double           den0,mom0,energy0;
  PetscScalar      *f0;
  PetscInt         cell,cStart,cEnd,dim,p;
  const PetscReal *coords, *c2;
  ierr = DMGetDimension(sw, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMSwarmSortGetAccess(sw);CHKERRQ(ierr);
  ierr = DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = VecGetArray(f_q,&f0);CHKERRQ(ierr);
  den0 = 0; mom0 = 0; energy0 = 0;
  for (cell = cStart, c2 = coords ; cell < cEnd ; ++cell) {
    PetscInt *cindices;

    PetscInt  numCIndices;
    ierr = DMSwarmSortGetPointsPerCell(sw, cell, &numCIndices, &cindices);CHKERRQ(ierr);
    for (p = 0; p < numCIndices; ++p, c2 += dim) {
      const PetscInt idx = cindices[p];
      den0    +=             f0[idx];
      mom0    += c2[0]      *f0[idx];
      energy0 += c2[0]*c2[0]*f0[idx];
    }
    ierr = PetscFree(cindices);CHKERRQ(ierr);
  }
  ierr = DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = DMSwarmSortRestoreAccess(sw);CHKERRQ(ierr);
  ierr = VecRestoreArray(f_q,&f0);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&den0,den0tot,1,MPI_DOUBLE,MPI_SUM,PetscObjectComm((PetscObject) sw));CHKERRQ(ierr);
  ierr = MPI_Allreduce(&mom0,mom0tot,1,MPI_DOUBLE,MPI_SUM,PetscObjectComm((PetscObject) sw));CHKERRQ(ierr);
  ierr = MPI_Allreduce(&energy0,energy0tot,1,MPI_DOUBLE,MPI_SUM,PetscObjectComm((PetscObject) sw));CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "getParticleVector"
static PetscErrorCode getParticleVector(DM dm, DM sw,
                                        PetscErrorCode (*funcs[])(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *),
                                        AppCtx *user, Vec f_q)
{
  PetscErrorCode   ierr;
  PetscInt         p, dim, rStart, cStart, cEnd, cell, maxC, *idxbuf;
  const PetscReal *coords, *c2;
  PetscScalar     *vbuf;
  PetscFunctionBeginUser;
  ierr = VecGetOwnershipRange(f_q,&rStart,NULL);CHKERRQ(ierr);
  ierr = DMGetDimension(sw, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMSwarmSortGetAccess(sw);CHKERRQ(ierr);
  ierr = DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  for (cell = cStart, maxC = 0; cell < cEnd; ++cell) {
    PetscInt *cindices;
    PetscInt  numCIndices;
    ierr = DMSwarmSortGetPointsPerCell(sw, cell, &numCIndices, &cindices);CHKERRQ(ierr);
    ierr = PetscFree(cindices);CHKERRQ(ierr);
    maxC = PetscMax(maxC, numCIndices);
  }
  /* create particle vector */
  ierr = PetscMalloc1(maxC, &idxbuf);CHKERRQ(ierr);
  ierr = PetscMalloc1(maxC, &vbuf);CHKERRQ(ierr);
  for (cell = cStart, c2 = coords; cell < cEnd; ++cell) {
    PetscInt *cindices;
    PetscInt  numCIndices;
    ierr = DMSwarmSortGetPointsPerCell(sw, cell, &numCIndices, &cindices);CHKERRQ(ierr);
    for (p = 0; p < numCIndices; ++p, c2 += dim) {
      funcs[0](dim, 0.0, c2, 1, &vbuf[p], user);
      vbuf[p] *= user->factor;
      idxbuf[p] = cindices[p] + rStart;
/* PetscPrintf(PETSC_COMM_SELF, "[%D]TestL2Projection: %D/%D) real coord[%4D]:%12.5e,%12.5e, rhs[%D]=%12.5e\n",rank,c2-coords+1,Np,p*dim,c2[0],c2[1],p,buf[p]); */
    }
    ierr = VecSetValues(f_q,numCIndices,idxbuf,vbuf,INSERT_VALUES);CHKERRQ(ierr);
    ierr = PetscFree(cindices);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(f_q);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(f_q);CHKERRQ(ierr);
  ierr = PetscFree(vbuf);CHKERRQ(ierr);
  ierr = PetscFree(idxbuf);CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject)f_q, NULL, "-f_view");CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = DMSwarmSortRestoreAccess(sw);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "computeMomentVectors"
/* compute moment vectors */
static PetscErrorCode computeMomentVectors(DM dm, Vec vecs[], AppCtx *user)
{
  PetscErrorCode  ierr;
  PetscInt        ii;
  PetscErrorCode (*funcs[3])(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *) = {one,x0,x0_2};
  AppCtx          *ctxs[1];
  PetscFunctionBeginUser;
  ctxs[0] = user;
  for (ii = 0; ii < 3; ii++) {
    ierr = DMCreateGlobalVector(dm,&vecs[ii]);CHKERRQ(ierr);
    ierr = DMProjectFunction(dm, 0.0, &funcs[ii], (void**)ctxs, INSERT_ALL_VALUES, vecs[ii]);CHKERRQ(ierr);
    ierr = PetscObjectViewFromOptions((PetscObject)vecs[ii], NULL, "-coord_vec_view");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "projectToParticles"
/* project u with a post psuedo inverse to f_q */
static PetscErrorCode projectToParticles(DM sw, DM dm, Vec u, Mat QinterpT, AppCtx *user, Vec f_q)
{
  PetscErrorCode   ierr;
  KSP              ksp;
  PC               pc;
  PetscInt         p, c, cStart, cEnd, cell, dim, *rowIDXs, rStart;
  PetscSection     fsection, globalFSection;
  IS               *rowis, *colis;
  Mat              *QelemT,QQt;
  PetscFunctionBeginUser;
  ierr = DMGetCoordinateDim(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &fsection);CHKERRQ(ierr);
  ierr = DMGetDefaultGlobalSection(dm, &globalFSection);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(f_q,&rStart,NULL);CHKERRQ(ierr);
  ierr = VecSet(f_q,1.e100);CHKERRQ(ierr);
  ierr = DMSwarmSortGetAccess(sw);CHKERRQ(ierr);
  ierr = PetscMalloc1(cEnd-cStart, &rowis);CHKERRQ(ierr); /* particles */
  ierr = PetscMalloc1(cEnd-cStart, &colis);CHKERRQ(ierr); /* vertices */
 /* For each cell */
  for (cell = cStart, c = 0; cell < cEnd; ++cell, c++) {
    PetscInt *findices,   *cindices; /* fine is vertices, coarse is particles */
    PetscInt  numFIndices, numCIndices;
    ierr = DMPlexGetClosureIndices(dm, fsection, globalFSection, cell, &numFIndices, &findices, NULL);CHKERRQ(ierr);
    ierr = DMSwarmSortGetPointsPerCell(sw, cell, &numCIndices, &cindices);CHKERRQ(ierr);
    ierr = PetscMalloc1(numCIndices, &rowIDXs);CHKERRQ(ierr); /* particles */
    for (p = 0; p < numCIndices; ++p) rowIDXs[p] = cindices[p] + rStart; /* globalize particle (row) index */
    ierr = ISCreateGeneral(PETSC_COMM_SELF,numCIndices,rowIDXs,PETSC_COPY_VALUES,&rowis[c]);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,numFIndices,findices,PETSC_COPY_VALUES,&colis[c]);CHKERRQ(ierr);
    ierr = PetscFree(rowIDXs);CHKERRQ(ierr);
    ierr = PetscFree(cindices);CHKERRQ(ierr);
    ierr = DMPlexRestoreClosureIndices(dm, fsection, globalFSection, cell, &numFIndices, &findices, NULL);CHKERRQ(ierr);
  }
  ierr = DMSwarmSortRestoreAccess(sw);CHKERRQ(ierr);
  ierr = MatCreateSubMatrices(QinterpT,cEnd-cStart,rowis,colis,MAT_INITIAL_MATRIX,&QelemT);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_SELF,&ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPSetInitialGuessNonzero(ksp,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
  ierr = DMSwarmSortGetAccess(sw);CHKERRQ(ierr);
  for (cell = cStart, c = 0; cell < cEnd; ++cell, c++) {
    PetscScalar *f0;
    PetscInt    *findices, *cindices; /* fine is vertices, coarse is particles */
    PetscInt    numFIndices, numCIndices;
    Vec         loc_part, loc_part2, loc_vert, loc_vert2;
    /* TODO -- this requires off processor/ghost vertices, need to get local */
    ierr = VecGetSubVector(u,colis[c],&loc_vert);CHKERRQ(ierr);
    ierr = MatCreateVecs(QelemT[c],NULL,&loc_part);CHKERRQ(ierr);
    ierr = DMSwarmSortGetPointsPerCell(sw, cell, &numCIndices, &cindices);CHKERRQ(ierr);
    ierr = DMPlexGetClosureIndices(dm, fsection, globalFSection, cell, &numFIndices, &findices, NULL);CHKERRQ(ierr);
    ierr = DMPlexRestoreClosureIndices(dm, fsection, globalFSection, cell, &numFIndices, &findices, NULL);CHKERRQ(ierr);
    /* get RHS for pseudo inverse solve */
    if (numCIndices >= numFIndices) { /* Q' (QQ')^-1 */
      ierr = MatTransposeMatMult(QelemT[c],QelemT[c],MAT_INITIAL_MATRIX,PETSC_DEFAULT,&QQt);CHKERRQ(ierr);
    } else { /* (Q'Q)^-1 Q' */
      ierr = MatMatTransposeMult(QelemT[c],QelemT[c],MAT_INITIAL_MATRIX,PETSC_DEFAULT,&QQt);CHKERRQ(ierr);
      ierr = MatCreateVecs(QelemT[c],NULL,&loc_part2);CHKERRQ(ierr);
      ierr = MatMult(QelemT[c], loc_vert, loc_part2);CHKERRQ(ierr);
    }
    /* create KSP and solve */
    ierr = KSPSetOperators(ksp,QQt,QQt);CHKERRQ(ierr);
    ierr = KSPSetUp(ksp);CHKERRQ(ierr);
    if (numCIndices >= numFIndices) { /* Q' (QQ')^-1 */
      ierr = MatCreateVecs(QelemT[c],&loc_vert2,NULL);CHKERRQ(ierr);
      ierr = KSPSolve(ksp,loc_vert,loc_vert2);CHKERRQ(ierr);
      ierr = MatMult(QelemT[c], loc_vert2, loc_part);CHKERRQ(ierr);
      ierr = VecDestroy(&loc_vert2);CHKERRQ(ierr);
    } else { /* (Q'Q)^-1 Q' */
      ierr = KSPSolve(ksp,loc_part2,loc_part);CHKERRQ(ierr);
      ierr = VecDestroy(&loc_part2);CHKERRQ(ierr);
    }
    /* Assemble loc_part back to f_q */
    ierr = VecGetArray(loc_part,&f0);CHKERRQ(ierr);
    ierr = PetscMalloc1(numCIndices, &rowIDXs);CHKERRQ(ierr); /* particles */
    for (p = 0; p < numCIndices; ++p) rowIDXs[p] = cindices[p] + rStart; /* globalize particle (row) index */
    ierr = VecSetValues(f_q,numCIndices,rowIDXs,f0,INSERT_VALUES);CHKERRQ(ierr);
    ierr = PetscFree(rowIDXs);CHKERRQ(ierr);
    ierr = VecRestoreArray(loc_part,&f0);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(u,colis[c],&loc_vert);CHKERRQ(ierr);
    ierr = MatDestroy(&QQt);CHKERRQ(ierr);
    ierr = VecDestroy(&loc_part);CHKERRQ(ierr);
    ierr = VecDestroy(&loc_vert);CHKERRQ(ierr);
    ierr = ISDestroy(&rowis[c]);CHKERRQ(ierr);
    ierr = ISDestroy(&colis[c]);CHKERRQ(ierr);
    ierr = PetscFree(cindices);CHKERRQ(ierr);
    ierr = MatDestroy(&QelemT[c]);CHKERRQ(ierr);
  }
  ierr = DMSwarmSortRestoreAccess(sw);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(f_q);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(f_q);CHKERRQ(ierr);
  ierr = PetscFree(rowis);CHKERRQ(ierr);
  ierr = PetscFree(colis);CHKERRQ(ierr);
  ierr = PetscFree(QelemT);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestL2Projection(DM dm, DM sw, AppCtx *user)
{
  PetscErrorCode (*funcs[1])(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *);
  AppCtx          *ctxs[1];
  KSP              ksp;
  Mat              mass, QinterpT;
  Vec              f_q, rhs, uproj, uexact;
  PetscReal        error,normerr,norm;
  PetscErrorCode   ierr;
  const PetscScalar none = -1.0;
  PetscInt         dim, Np;
  PetscMPIInt      rank,size;
  MPI_Comm         comm;
  double           den0tot,mom0tot,energy0tot;
  PetscFunctionBeginUser;
  funcs[0] = (user->particles_cell == 0) ? linear : x2_x4;
  ctxs[0] = user;
  comm = PetscObjectComm((PetscObject)dm);
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&size);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMSwarmGetLocalSize(sw,&Np);CHKERRQ(ierr);
  ierr = KSPCreate(comm, &ksp);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm, &uproj);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm, &rhs);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm, &uexact);CHKERRQ(ierr);
  /* create RHS vector data */
  ierr = VecCreate(PetscObjectComm((PetscObject) sw),&f_q);CHKERRQ(ierr);
  ierr = VecSetSizes(f_q,Np,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(f_q);CHKERRQ(ierr);

  /* create particles and compute moments */
  ierr = getParticleVector( dm, sw, funcs, user, f_q);CHKERRQ(ierr);

  /* create particle mass matrix */
  ierr = DMCreateMassMatrix(sw, dm, &mass);CHKERRQ(ierr);
  ierr = DMSwarmCreateInterpolationMatrix(sw, dm, &QinterpT);CHKERRQ(ierr);
  /* make RHS - apply pseudo inverse (in post apply) */
  ierr = MatMultTranspose(mass, f_q, rhs);CHKERRQ(ierr); /* part I of post pseudo inverse (trivial) */
  ierr = PetscObjectSetName((PetscObject) rhs,"rhs");CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject)rhs, NULL, "-vec_view");CHKERRQ(ierr);
  ierr = MatViewFromOptions(QinterpT, NULL, "-particle_interp_mat_view");CHKERRQ(ierr);
  ierr = MatDestroy(&mass);CHKERRQ(ierr);

  /* make FE mass, solve, project, compute error */
  ierr = DMCreateMatrix(dm, &mass);CHKERRQ(ierr);
  ierr = DMPlexSNESComputeJacobianFEM(dm, uproj, mass, mass, user);CHKERRQ(ierr);
  ierr = MatViewFromOptions(mass, NULL, "-fe_mass_mat_view");CHKERRQ(ierr);
  /* solve for operator */
  ierr = KSPSetOperators(ksp, mass, mass);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp, rhs, uproj);CHKERRQ(ierr);

  /* view */
  ierr = PetscObjectSetName((PetscObject) uproj,"u");CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject)uproj, NULL, "-vec_view");CHKERRQ(ierr);
  ierr = DMComputeL2Diff(dm, 0.0, funcs, (void**)ctxs, uproj, &error);CHKERRQ(ierr);
  ierr = DMProjectFunction(dm, 0.0, funcs, (void**)ctxs, INSERT_ALL_VALUES, uexact);CHKERRQ(ierr);
  ierr = VecNorm(uexact, NORM_2, &norm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) uexact, "exact");CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject)uexact, NULL, "-vec_view");CHKERRQ(ierr);
  ierr = VecAYPX(uexact,none,uproj);CHKERRQ(ierr); /* uexact = error function */
  ierr = PetscObjectSetName((PetscObject) uexact, "error");CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject)uexact, NULL, "-vec_view");CHKERRQ(ierr);
  ierr = VecNorm(uexact, NORM_2, &normerr);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "relative discrete error = %g, |exact| = %g, Projected L2 error = %g\n", normerr/norm, norm, error);CHKERRQ(ierr);
  /* compute particle moments */
  ierr = computeParticleMoments(sw, dm, f_q, user, &den0tot, &mom0tot, &energy0tot);CHKERRQ(ierr);
  /* get FE moments */
  {
    PetscScalar momentum, energy, density, tt[0];
    PetscDS     prob;
    Vec         vecs[3];
    ierr = MatMultTranspose(QinterpT, f_q, rhs);CHKERRQ(ierr); /* interpolate particles to grid: Q * w_p */
    ierr = KSPSolve(ksp, rhs, uproj);CHKERRQ(ierr);
    ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
    ierr = PetscDSSetObjective(prob, 0, &f0_1);CHKERRQ(ierr);
    ierr = DMPlexComputeIntegralFEM(dm,uproj,tt,user);CHKERRQ(ierr);
    density = tt[0];
    ierr = PetscDSSetObjective(prob, 0, &f0_momx);CHKERRQ(ierr);
    ierr = DMPlexComputeIntegralFEM(dm,uproj,tt,user);CHKERRQ(ierr);
    momentum = tt[0];
    ierr = PetscDSSetObjective(prob, 0, &f0_ex);CHKERRQ(ierr);
    ierr = DMPlexComputeIntegralFEM(dm,uproj,tt,user);CHKERRQ(ierr);
    energy = tt[0];
    PetscPrintf(comm, "\t[%D] L2 projection x_m ([x_m-x_p]/x_m) rho: %20.13e (%11.4e), momentum_x: %20.13e (%11.4e), energy: %20.13e (%11.4e).\n",rank,density,(density-den0tot)/density,momentum,(momentum-mom0tot)/momentum,energy,(energy-energy0tot)/energy);
    /* compute coordinate vectos x and moments x' * Q * w_p */
    ierr = computeMomentVectors(dm, vecs, user);CHKERRQ(ierr);
    ierr = VecDot(vecs[0],rhs,&density);CHKERRQ(ierr);
    ierr = VecDot(vecs[1],rhs,&momentum);CHKERRQ(ierr);
    ierr = VecDot(vecs[2],rhs,&energy);CHKERRQ(ierr);
    PetscPrintf(comm, "\t[%D] x' * Q * w_p: x_m ([x_m-x_p]/x_m) rho: %20.13e (%11.4e), momentum_x: %20.13e (%11.4e), energy: %20.13e (%11.4e).\n",rank,density,(density-den0tot)/density,momentum,(momentum-mom0tot)/momentum,energy,(energy-energy0tot)/energy);
    ierr = VecDestroy(&vecs[0]);CHKERRQ(ierr);
    ierr = VecDestroy(&vecs[1]);CHKERRQ(ierr);
    ierr = VecDestroy(&vecs[2]);CHKERRQ(ierr);
  }
  /* project solution back to particles (post pseudo inverse) */
  if (size==1) {
    ierr = projectToParticles(sw, dm, uproj, QinterpT, user, f_q);CHKERRQ(ierr);
    /* compute particle moments */
    ierr = computeParticleMoments(sw, dm, f_q, user, &den0tot, &mom0tot, &energy0tot);CHKERRQ(ierr);
    PetscPrintf(comm, "\t[%D] Pseudo-inverse to particles: x_p  rho: %20.13e,\t       momentum_x: %20.13e,\t       energy: %20.13e\n",rank,den0tot,mom0tot,energy0tot);
  }
  /* clean up */
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&f_q);CHKERRQ(ierr);
  ierr = MatDestroy(&mass);CHKERRQ(ierr);
  ierr = MatDestroy(&QinterpT);CHKERRQ(ierr);
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
