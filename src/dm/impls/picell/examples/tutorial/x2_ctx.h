/* M. Adams, August 2016 */
/*
  General parameters and context
*/
typedef struct {
  PetscLogEvent *events;
  PetscInt      bsp_chunksize;
  PetscInt      chunksize;
  runType       run_type;
  PetscBool     plot;
  /* MPI parallel data */
  MPI_Comm      particlePlaneComm,wComm;
  PetscMPIInt   rank,npe,npe_particlePlane,particlePlaneRank,ParticlePlaneIdx;
  /* grids & solver */
  DM             dm;
  X2GridParticle particleGrid;
  PetscBool      inflate_torus;
  /* time */
  PetscInt  msteps;
  PetscReal maxTime;
  PetscReal dt;
  /* physics */
  PetscErrorCode (**BCFuncs)(PetscInt dim, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
  PetscReal massAu; /* =2D0  !mass ratio to proton */
  /* PetscReal eMassAu; /\* =2D-2 *\/ */
  PetscReal chargeEu; /* =1D0  ! charge number */
  PetscReal eChargeEu; /* =-1D0 */
  /* particles */
  PetscInt  npart_proc;
  PetscBool useElectrons;
  PetscInt  collisionPeriod;
  PetscReal max_vpar;
  PetscInt  nElems; /* size of array of particle lists */
  X2PList  *partlists[X2_NION+1]; /* 0: electron, 1:N ions */
  X2Species species[X2_NION+1]; /* 0: electron, 1:N ions */
  PetscInt  tablesize,tablecount; /* hash table meta-data for proc-send list table */
  X2PSendList *sendListTable;
} X2Ctx;
