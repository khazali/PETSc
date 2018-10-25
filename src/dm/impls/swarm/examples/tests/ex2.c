static char help[] = "Tests basic operations and fields on a DMSwarm.\n";

#include <petscdmda.h>
#include <petscdmplex.h>
#include <petscdmswarm.h>

#define DIM             (3)
#define NFIELDS         (2)
#define GAUSS_POINTS    (8)
#define PRESSUREQ       "pressure_q"
#define DENSITYQ        "density_q"
#define PRESSURE        "pressure"
#define DENSITY         "density"

int main(int argc, char *argv[])
{
  Vec *pfields;
  const char *fieldnames[] = { PRESSURE, DENSITY };
  PetscRandom rng;
  PetscScalar *array_coor,*array_press,*array_rho;
  PetscInt nel, dim = 3,cStart,cEnd,ppcell = 0,refine = 1;
  PetscInt npoints = 1000,npoints_local,idx,p;
  DM dmp_mesh,dmp_mesh_dist,dms_quadrature, dms_mpoint;
  PetscViewer viewer;
  PetscErrorCode ierr;
  PetscInt numComp[NFIELDS],numDof[NFIELDS*(DIM+1)] = { 0 };
  PetscSection section;
  PetscReal r, phi, z1, zmax = 0.25, *array_xyz;
  PetscScalar x, y, z2;

  ierr = PetscInitialize(&argc,&argv,NULL,help);CHKERRQ(ierr);

  /* Get the 'refine' option */
  ierr = PetscOptionsGetInt(NULL,NULL,"-refine",&refine,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL);CHKERRQ(ierr);

  /* Create the cylinder mesh */
  ierr = DMPlexCreateHexCylinderMesh(PETSC_COMM_WORLD,refine,DM_BOUNDARY_NONE,&dmp_mesh);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)dmp_mesh, "MySwarm");CHKERRQ(ierr);
  ierr = DMPlexDistribute(dmp_mesh,0,NULL,&dmp_mesh_dist);CHKERRQ(ierr);
  if (dmp_mesh_dist) {
    ierr = DMDestroy(&dmp_mesh);CHKERRQ(ierr);
    dmp_mesh = dmp_mesh_dist;
  }

  numComp[0] = 1;
  numDof[0*(DIM+1)+DIM] = 1;
  ierr = DMPlexCreateSection(dmp_mesh,dim,1,numComp,numDof,0,NULL,NULL,NULL,NULL,&section);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(section,0,"_mesh");CHKERRQ(ierr);
  ierr = DMSetDefaultSection(dmp_mesh,section);CHKERRQ(ierr);

  /* Get number of cells */
  ierr = DMPlexGetHeightStratum(dmp_mesh,0,&cStart,&cEnd);CHKERRQ(ierr);
  nel = cEnd - cStart;

  ierr = DMCreate(PETSC_COMM_WORLD,&dms_quadrature);CHKERRQ(ierr);
  ierr = DMSetType(dms_quadrature,DMSWARM);CHKERRQ(ierr);
  ierr = DMSetDimension(dms_quadrature,dim);CHKERRQ(ierr);

  /* Register fields for viscosity and density on the quadrature points */
  ierr = DMSwarmRegisterPetscDatatypeField(dms_quadrature,PRESSUREQ,1,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(dms_quadrature,DENSITYQ,1,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmFinalizeFieldRegister(dms_quadrature);CHKERRQ(ierr);
  ierr = DMSwarmSetLocalSizes(dms_quadrature,nel * GAUSS_POINTS,0);CHKERRQ(ierr);

  /* Create the material point swarm */
  ierr = DMCreate(PETSC_COMM_WORLD,&dms_mpoint);CHKERRQ(ierr);
  ierr = DMSetType(dms_mpoint,DMSWARM);CHKERRQ(ierr);
  ierr = DMSetDimension(dms_mpoint,dim);CHKERRQ(ierr);

  /* Configure the material point swarm to be of type Particle-In-Cell */
  ierr = DMSwarmSetType(dms_mpoint,DMSWARM_PIC);CHKERRQ(ierr);

  /* Specify the DM to use for point location and projections
   * within the context of a PIC scheme */
  ierr = DMSwarmSetCellDM(dms_mpoint,dmp_mesh);CHKERRQ(ierr);

  /* Register fields for viscosity and density */
  ierr = DMSwarmRegisterPetscDatatypeField(dms_mpoint,PRESSURE,1,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(dms_mpoint,DENSITY,1,PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmFinalizeFieldRegister(dms_mpoint);CHKERRQ(ierr);

  /* Get the 'ppcell' option */
  ierr = PetscOptionsGetInt(NULL,NULL,"-ppcell",&ppcell,NULL);CHKERRQ(ierr);
  ierr = DMSwarmSetLocalSizes(dms_mpoint,nel * ppcell,0);CHKERRQ(ierr);

  /* Layout the material points in space using the cell DM.
   * Particle coordinates are defined by cell wise using different methods.
   * - DMSWARMPIC_LAYOUT_GAUSS defines particles coordinates at the positions
   *                           corresponding to a Gauss quadrature rule with
   *                           ppcell points in each direction.
   * - DMSWARMPIC_LAYOUT_REGULAR defines particle coordinates at the centoid
   *                             of ppcell x ppcell quadralaterals defined
   *                             within the reference element.
   * - DMSWARMPIC_LAYOUT_SUBDIVISION defines particles coordinates at the
   *                                 centroid of each quadralateral obtained
   *                                 by sub-dividing the reference element
   *                                 cell ppcell times. */
  ierr = DMSwarmInsertPointsUsingCellDM(dms_mpoint,DMSWARMPIC_LAYOUT_SUBDIVISION,ppcell);CHKERRQ(ierr);

  /* Set swarm coordinates */
  ierr = PetscRandomCreate(PETSC_COMM_SELF,&rng);CHKERRQ(ierr);
  ierr = PetscRandomSetType(rng,PETSCRAND48);CHKERRQ(ierr);
  ierr = PetscMalloc1(dim * npoints,&array_xyz);CHKERRQ(ierr);

  for (idx = 0; idx < dim * npoints; idx += dim) {
    ierr = PetscRandomGetValue(rng, &z2);CHKERRQ(ierr);
    r = PetscRealPart(z2);
    ierr = PetscRandomGetValue(rng, &z2);CHKERRQ(ierr);
    phi  = 2. * PETSC_PI * PetscRealPart(z2);
    ierr = PetscRandomGetValue(rng, &z2);CHKERRQ(ierr);
    z1 = 2. * zmax * (PetscRealPart(z2) - 0.5);

    array_xyz[idx+0] = r * PetscCosReal(phi);
    array_xyz[idx+1] = r * PetscSinReal(phi);
    array_xyz[idx+2] = z1;
  }

  ierr = DMSwarmSetPointCoordinates(dms_mpoint,npoints,array_xyz,PETSC_TRUE,ADD_VALUES);CHKERRQ(ierr);
  ierr = PetscFree(array_xyz);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rng);CHKERRQ(ierr);

  /* View for debugging */
  ierr = DMView(dms_mpoint,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMView(dms_quadrature,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Get swarm fields */
  ierr = DMSwarmGetField(dms_mpoint,DMSwarmPICField_coor,NULL,NULL,(void **)&array_coor);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dms_mpoint,PRESSURE,NULL,NULL,(void **)&array_press);CHKERRQ(ierr);
  ierr = DMSwarmGetField(dms_mpoint,DENSITY,NULL,NULL,(void **)&array_rho);CHKERRQ(ierr);

  ierr = DMSwarmGetLocalSize(dms_mpoint,&npoints_local);CHKERRQ(ierr);
  for (p = 0; p < npoints_local; ++p) {
    x  = array_coor[DIM*p+0];
    y  = array_coor[DIM*p+1];
    z2 = array_coor[DIM*p+2];

    array_press[p] = x * x + y * y * y - 2. * z2;
    array_rho[p]   = PetscSqrtReal(x * x + y * y);
  }

  /* Restore swarm fields */
  ierr = DMSwarmRestoreField(dms_mpoint,DENSITY,NULL,NULL,(void **)&array_rho);
  ierr = DMSwarmRestoreField(dms_mpoint,PRESSURE,NULL,NULL,(void **)&array_press);
  ierr = DMSwarmRestoreField(dms_mpoint,DMSwarmPICField_coor,NULL,NULL,(void **)&array_coor);

  ierr = DMSwarmProjectFields(dms_mpoint,NFIELDS,fieldnames,&pfields,PETSC_FALSE);CHKERRQ(ierr);

  ierr = PetscViewerCreate(PETSC_COMM_WORLD,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer,PETSCVIEWERVTK);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_VTK_VTU);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer,"ex2.vtu");CHKERRQ(ierr);
  ierr = VecView(pfields[0],viewer);CHKERRQ(ierr);
  ierr = VecView(pfields[1],viewer);CHKERRQ(ierr);
  ierr = DMPlexVTKWriteAll((PetscObject)dmp_mesh,viewer);CHKERRQ(ierr);

  ierr = DMView(dmp_mesh,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = DMDestroy(&dmp_mesh);CHKERRQ(ierr);
  ierr = DMDestroy(&dms_quadrature);CHKERRQ(ierr);
  ierr = DMDestroy(&dms_mpoint);CHKERRQ(ierr);
  PetscFinalize();

  return 0;
}

/*TEST

  test:
    suffix: 3d_1
    args: -dim 3 -refine 2
  test:
    suffix: 3d_2
    nsize: 4
    args: -dim 3 -refine 2

TEST*/
