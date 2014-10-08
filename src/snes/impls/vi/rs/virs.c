#include <petsctao.h>
#include <../src/snes/impls/vi/rs/virsimpl.h> /*I "petscsnes.h" I*/
#include <petsc-private/kspimpl.h>
#include <petsc-private/matimpl.h>
#include <petsc-private/dmimpl.h>
#include <petsc-private/vecimpl.h>
const char *SubSetTypes[] = {  "subvec","mask","matrixfree","TaoSubSetType","TAO_SUBSET_",0};

#undef __FUNCT__
#define __FUNCT__ "SNESVIGetInactiveSet"
/*
   SNESVIGetInactiveSet - Gets the global indices for the inactive set variables (these correspond to the degrees of freedom the linear
     system is solved on)

   Input parameter
.  snes - the SNES context

   Output parameter
.  ISact - active set index set

 */
PetscErrorCode SNESVIGetInactiveSet(SNES snes,IS *inact)
{
  SNES_VINEWTONRSLS *vi = (SNES_VINEWTONRSLS*)snes->data;

  PetscFunctionBegin;
  *inact = vi->IS_inact;
  PetscFunctionReturn(0);
}

/*
    Provides a wrapper to a DM to allow it to be used to generated the interpolation/restriction from the DM for the smaller matrices and vectors
  defined by the reduced space method.

    Simple calls the regular DM interpolation and restricts it to operation on the variables not associated with active constraints.

<*/
typedef struct {
  IS       inactive;
  PetscInt ninactive;                  /* size of vectors in the reduced DM space */
  PetscInt nfull;
  /* DM's original routines */
  PetscErrorCode (*createinterpolation)(DM,DM,Mat*,Vec*);
  PetscErrorCode (*coarsen)(DM, MPI_Comm, DM*);
  PetscErrorCode (*createglobalvector)(DM,Vec*);
  DM dm;   /* when destroying this object we need to reset the above function into the base DM */
  TaoSubsetType subset_type;
  Vec diag; /* work vector (full space) used for submatrix */
} DM_SNESVI;

#undef __FUNCT__
#define __FUNCT__ "DMCreateGlobalVector_SNESVI"
/*
     DMCreateGlobalVector_SNESVI - Creates global vector of the size of the reduced space
*/
PetscErrorCode  DMCreateGlobalVector_SNESVI(DM dm,Vec *vec)
{
  PetscErrorCode ierr;
  PetscContainer isnes;
  DM_SNESVI      *dmsnesvi;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)dm,"VI",(PetscObject*)&isnes);CHKERRQ(ierr);
  if (!isnes) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Composed SNES is missing");
  ierr = PetscContainerGetPointer(isnes,(void**)&dmsnesvi);CHKERRQ(ierr);
  if (0) {printf("creating dm global vector. nfull=%d, nreduced=%d\n",dmsnesvi->nfull,dmsnesvi->ninactive);}
  if (dmsnesvi->subset_type == TAO_SUBSET_SUBVEC) {
    ierr = VecCreateMPI(PetscObjectComm((PetscObject)dm),dmsnesvi->ninactive,PETSC_DETERMINE,vec);CHKERRQ(ierr);
  } else {
    ierr = (*dmsnesvi->createglobalvector)(dm,vec);
    if (0) {
      PetscInt n;
      ierr = VecGetSize(*vec,&n);CHKERRQ(ierr);
      printf("created vector of size %d\n",n);
    }

  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateInterpolation_SNESVI"
/*
     DMCreateInterpolation_SNESVI - Modifieds the interpolation obtained from the DM by removing all rows and columns associated with active constraints.

*/
PetscErrorCode  DMCreateInterpolation_SNESVI(DM dm1,DM dm2,Mat *mat,Vec *scaling)
{
  PetscErrorCode ierr;
  PetscContainer isnes;
  Vec            vec,workvec;
  DM_SNESVI      *dmsnesvi1,*dmsnesvi2;
  Mat            interp;

  PetscFunctionBegin;
  printf("INTERPOLATE\n");
  ierr = PetscObjectQuery((PetscObject)dm1,"VI",(PetscObject*)&isnes);CHKERRQ(ierr);
  if (!isnes) SETERRQ(PetscObjectComm((PetscObject)dm1),PETSC_ERR_PLIB,"Composed VI data structure is missing");
  ierr = PetscContainerGetPointer(isnes,(void**)&dmsnesvi1);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)dm2,"VI",(PetscObject*)&isnes);CHKERRQ(ierr);
  if (!isnes) SETERRQ(PetscObjectComm((PetscObject)dm2),PETSC_ERR_PLIB,"Composed VI data structure is missing");
  ierr = PetscContainerGetPointer(isnes,(void**)&dmsnesvi2);CHKERRQ(ierr);

  ierr = (*dmsnesvi1->createinterpolation)(dm1,dm2,&interp,NULL);CHKERRQ(ierr);
  ierr = (*dmsnesvi2->createglobalvector)(dm2,&vec);CHKERRQ(ierr);
  ierr = (*dmsnesvi2->createglobalvector)(dm2,&workvec);CHKERRQ(ierr);
  ierr = TaoMatGetSubMat(interp,dmsnesvi2->inactive,workvec,dmsnesvi2->subset_type,mat);CHKERRQ(ierr);
  // old  ierr = MatGetSubMatrix(interp,dmsnesvi2->inactive,dmsnesvi1->inactive,MAT_INITIAL_MATRIX,mat);CHKERRQ(ierr);
  ierr = MatDestroy(&interp);CHKERRQ(ierr);
  ierr = VecDestroy(&vec);CHKERRQ(ierr);
  *scaling = 0;
  PetscFunctionReturn(0);
}

extern PetscErrorCode  DMSetVI(DM,IS);

#undef __FUNCT__
#define __FUNCT__ "DMCoarsen_SNESVI"
/*
     DMCoarsen_SNESVI - Computes the regular coarsened DM then computes additional information about its inactive set

*/
PetscErrorCode  DMCoarsen_SNESVI(DM dm1,MPI_Comm comm,DM *dm2)
{
  PetscErrorCode ierr;
  PetscContainer isnes;
  DM_SNESVI      *dmsnesvi1;
  Vec            finemarked,coarsemarked;
  IS             inactive;
  VecScatter     inject;
  const PetscInt *index;
  PetscInt       n,k,cnt = 0,rstart,*coarseindex;
  PetscScalar    *marked;

  PetscFunctionBegin;
  printf("COARSEN\n");
  ierr = PetscObjectQuery((PetscObject)dm1,"VI",(PetscObject*)&isnes);CHKERRQ(ierr);
  if (!isnes) SETERRQ(PetscObjectComm((PetscObject)dm1),PETSC_ERR_PLIB,"Composed VI data structure is missing");
  ierr = PetscContainerGetPointer(isnes,(void**)&dmsnesvi1);CHKERRQ(ierr);

  /* call the original coarsen */
  ierr = (*dmsnesvi1->coarsen)(dm1,comm,dm2);CHKERRQ(ierr);

  /* not sure why this extra reference is needed, but without the dm2 disappears too early */
  ierr = PetscObjectReference((PetscObject)*dm2);CHKERRQ(ierr);

  /* need to set back global vectors in order to use the original injection */
  ierr = DMClearGlobalVectors(dm1);CHKERRQ(ierr);

  dm1->ops->createglobalvector = dmsnesvi1->createglobalvector;


  /* create fine-mesh vector of zeros. If a fine-mesh vector value is inactive, then mark the
     vector location with a 1.0 value. Then inject this matrix into the coarse-mesh space, create
     a new inactive IS on the coarse-mesh dm by noting which values are nonzero */
  ierr = DMCreateGlobalVector(dm1,&finemarked);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(*dm2,&coarsemarked);CHKERRQ(ierr);

  /*
     fill finemarked with locations of inactive points
  */
  ierr = ISGetIndices(dmsnesvi1->inactive,&index);CHKERRQ(ierr);
  ierr = ISGetLocalSize(dmsnesvi1->inactive,&n);CHKERRQ(ierr);
  ierr = VecSet(finemarked,0.0);CHKERRQ(ierr);
  for (k=0; k<n; k++) {
    ierr = VecSetValue(finemarked,index[k],1.0,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(finemarked);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(finemarked);CHKERRQ(ierr);

  ierr = DMCreateInjection(*dm2,dm1,&inject);CHKERRQ(ierr);
  ierr = VecScatterBegin(inject,finemarked,coarsemarked,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(inject,finemarked,coarsemarked,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&inject);CHKERRQ(ierr);

  /*
     create index set list of coarse inactive points from coarsemarked
  */
  ierr = VecGetLocalSize(coarsemarked,&n);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(coarsemarked,&rstart,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(coarsemarked,&marked);CHKERRQ(ierr);
  for (k=0; k<n; k++) {
    if (marked[k] != 0.0) cnt++;
  }
  ierr = PetscMalloc1(cnt,&coarseindex);CHKERRQ(ierr);
  cnt  = 0;
  for (k=0; k<n; k++) {
    if (marked[k] != 0.0) coarseindex[cnt++] = k + rstart;
  }
  ierr = VecRestoreArray(coarsemarked,&marked);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)coarsemarked),cnt,coarseindex,PETSC_OWN_POINTER,&inactive);CHKERRQ(ierr);

  ierr = DMClearGlobalVectors(dm1);CHKERRQ(ierr);

  dm1->ops->createglobalvector = DMCreateGlobalVector_SNESVI;

  ierr = DMSetVI(*dm2,inactive);CHKERRQ(ierr);

  ierr = VecDestroy(&finemarked);CHKERRQ(ierr);
  ierr = VecDestroy(&coarsemarked);CHKERRQ(ierr);
  ierr = ISDestroy(&inactive);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDestroy_SNESVI"
PetscErrorCode DMDestroy_SNESVI(DM_SNESVI *dmsnesvi)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* reset the base methods in the DM object that were changed when the DM_SNESVI was reset */
  dmsnesvi->dm->ops->createinterpolation = dmsnesvi->createinterpolation;
  dmsnesvi->dm->ops->coarsen             = dmsnesvi->coarsen;
  dmsnesvi->dm->ops->createglobalvector  = dmsnesvi->createglobalvector;
  /* need to clear out this vectors because some of them may not have a reference to the DM
    but they are counted as having references to the DM in DMDestroy() */
  ierr = DMClearGlobalVectors(dmsnesvi->dm);CHKERRQ(ierr);

  ierr = ISDestroy(&dmsnesvi->inactive);CHKERRQ(ierr);
  ierr = PetscFree(dmsnesvi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSetVI"
/*
     DMSetVI - Marks a DM as associated with a VI problem. This causes the
               interpolation/restriction operators to
               be restricted to only those variables NOT associated with active constraints.

     Stashes a DM_SNESVI object "VI" in the DM object
     Redirects the dm->ops through DMCreateInterpolation_SNESVI
                                   DMCoarsen_SNESVI;
                                   DMCreateGlobalVector_SNESVI;


*/
PetscErrorCode  DMSetVI(DM dm,IS inactive)
{
  PetscErrorCode ierr;
  PetscContainer container;
  Vec            fullspace_vec;
  DM_SNESVI      *dmsnesvi;

  PetscFunctionBegin;
  if (!dm) PetscFunctionReturn(0);

  /* Check if VI already exists in dm */
  ierr = PetscObjectReference((PetscObject)inactive);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)dm,"VI",(PetscObject*)&container);CHKERRQ(ierr);

  if (container) {
    ierr = PetscContainerGetPointer(container,(void**)&dmsnesvi);CHKERRQ(ierr);
    ierr = ISDestroy(&dmsnesvi->inactive);CHKERRQ(ierr);
  } else {
    ierr = PetscContainerCreate(PetscObjectComm((PetscObject)dm),&container);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(container,(PetscErrorCode (*)(void*))DMDestroy_SNESVI);CHKERRQ(ierr);
    ierr = PetscNew(&dmsnesvi);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(container,(void*)dmsnesvi);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)dm,"VI",(PetscObject)container);CHKERRQ(ierr);
    ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);

    dmsnesvi->subset_type = TAO_SUBSET_SUBVEC;
    ierr = PetscOptionsEnum("-tao_subset_type","subset type","",SubSetTypes,(PetscEnum)dmsnesvi->subset_type,(PetscEnum*)&dmsnesvi->subset_type,0);CHKERRQ(ierr);

    /* start hack */
    ierr = DMCreateGlobalVector(dm,&fullspace_vec);CHKERRQ(ierr);
    ierr = VecGetLocalSize(fullspace_vec,&dmsnesvi->nfull);CHKERRQ(ierr);
    ierr = VecDestroy(&fullspace_vec);CHKERRQ(ierr);
    /* end hack */


    dmsnesvi->createinterpolation = dm->ops->createinterpolation;
    dmsnesvi->coarsen             = dm->ops->coarsen;
    dmsnesvi->createglobalvector  = dm->ops->createglobalvector;

    dm->ops->createinterpolation  = DMCreateInterpolation_SNESVI;
    dm->ops->coarsen              = DMCoarsen_SNESVI;
    dm->ops->createglobalvector   = DMCreateGlobalVector_SNESVI;

  }
  ierr = PetscOptionsEnum("-tao_subset_type","subset type","",SubSetTypes,(PetscEnum)dmsnesvi->subset_type,(PetscEnum*)&dmsnesvi->subset_type,0);CHKERRQ(ierr);

  ierr = DMClearGlobalVectors(dm);CHKERRQ(ierr);
  ierr = ISGetLocalSize(inactive,&dmsnesvi->ninactive);CHKERRQ(ierr);

  dmsnesvi->inactive = inactive;
  dmsnesvi->dm       = dm;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDestroyVI"
/*
     DMDestroyVI - Frees the DM_SNESVI object contained in the DM
         - also resets the function pointers in the DM for createinterpolation() etc to use the original DM
*/
PetscErrorCode  DMDestroyVI(DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!dm) PetscFunctionReturn(0);
  ierr = PetscObjectCompose((PetscObject)dm,"VI",(PetscObject)NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------------------------*/




#undef __FUNCT__
#define __FUNCT__ "SNESCreateIndexSets_VINEWTONRSLS"
PetscErrorCode SNESCreateIndexSets_VINEWTONRSLS(SNES snes,Vec X,Vec F,IS *ISact,IS *ISinact)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESVIGetActiveSetIS(snes,X,F,ISact);CHKERRQ(ierr);
  ierr = ISComplement(*ISact,X->map->rstart,X->map->rend,ISinact);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Create active and inactive set vectors. The local size of this vector is set and petsc computes the global size */
#undef __FUNCT__
#define __FUNCT__ "SNESCreateSubVectors_VINEWTONRSLS"
PetscErrorCode SNESCreateSubVectors_VINEWTONRSLS(SNES snes,PetscInt n,Vec *newv)
{
  PetscErrorCode ierr;
  Vec            v;

  PetscFunctionBegin;
  ierr  = VecCreate(PetscObjectComm((PetscObject)snes),&v);CHKERRQ(ierr);
  ierr  = VecSetSizes(v,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr  = VecSetType(v,VECSTANDARD);CHKERRQ(ierr);
  *newv = v;
  PetscFunctionReturn(0);
}

/* Variational Inequality solver using reduce space method. No semismooth algorithm is
   implemented in this algorithm. It basically identifies the active constraints and does
   a linear solve on the other variables (those not associated with the active constraints). */
#undef __FUNCT__
#define __FUNCT__ "SNESSolve_VINEWTONRSLS"
PetscErrorCode SNESSolve_VINEWTONRSLS(SNES snes)
{
  SNES_VINEWTONRSLS  *vi = (SNES_VINEWTONRSLS*)snes->data;
  PetscErrorCode     ierr;
  PetscInt           maxits,i,lits;
  PetscBool          lssucceed;
  PetscReal          fnorm,gnorm,xnorm=0,ynorm;
  Vec                Y,X,F,wx;
  KSPConvergedReason kspreason;

  PetscFunctionBegin;
  snes->numFailures            = 0;
  snes->numLinearSolveFailures = 0;
  snes->reason                 = SNES_CONVERGED_ITERATING;

  maxits = snes->max_its;               /* maximum number of iterations */
  X      = snes->vec_sol;               /* solution vector */
  F      = snes->vec_func;              /* residual vector */
  Y      = snes->work[0];               /* work vectors */
  wx     = snes->work[1];
  //ierr = VecDuplicate(X,&Y);CHKERRQ(ierr);

  ierr = SNESLineSearchSetVIFunctions(snes->linesearch, SNESVIProjectOntoBounds, SNESVIComputeInactiveSetFnorm);CHKERRQ(ierr);
  ierr = SNESLineSearchSetVecs(snes->linesearch, X, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = SNESLineSearchSetUp(snes->linesearch);CHKERRQ(ierr);

  ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->iter = 0;
  snes->norm = 0.0;
  ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);

  /* project X onto bounds */
  ierr = VecMedian(snes->xl,X,snes->xu,X);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
  if (0) {
    Vec temp;
    ierr = VecDuplicate(F,&temp); CHKERRQ(ierr);CHKMEMQ;
    ierr = VecSet(temp,0.0);CHKERRQ(ierr);CHKMEMQ;
  }

  if (snes->domainerror) {
    snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
    PetscFunctionReturn(0);
  }
  ierr = SNESVIComputeInactiveSetFnorm(snes,F,X,&fnorm);CHKERRQ(ierr);
  ierr = VecNormBegin(X,NORM_2,&xnorm);CHKERRQ(ierr);        /* xnorm <- ||x||  */
  ierr = VecNormEnd(X,NORM_2,&xnorm);CHKERRQ(ierr);
  if (PetscIsInfOrNanReal(fnorm)) SETERRQ(PetscObjectComm((PetscObject)X),PETSC_ERR_FP,"User provided compute function generated a Not-a-Number");

  ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->norm = fnorm;
  ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
  ierr       = SNESLogConvergenceHistory(snes,fnorm,0);CHKERRQ(ierr);
  ierr       = SNESMonitor(snes,0,fnorm);CHKERRQ(ierr);

  /* test convergence */
  ierr = (*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
  if (snes->reason) PetscFunctionReturn(0);


  for (i=0; i<maxits; i++) {

    IS         IS_act; /* _act -> active set */
    IS         IS_redact; /* redundant active set */
    PetscInt   nis_act,nis_inact;
    Vec        Y_act=0,Y_inact=0,F_inact=0;
    Mat        jac_inact_inact=0,prejac_inact_inact=0;
    PetscBool  isequal;

    /* Call general purpose update function */
    if (snes->ops->update) {
      ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
    }
    ierr = SNESComputeJacobian(snes,X,snes->jacobian,snes->jacobian_pre);CHKERRQ(ierr);
    if (0) {
      Vec temp;
      ierr = VecDuplicate(F,&temp); CHKERRQ(ierr);CHKMEMQ;
      ierr = VecSet(temp,0.0);CHKERRQ(ierr);CHKMEMQ;
    }


    /* Create active and inactive index sets */
    ierr = SNESVIGetActiveSetIS(snes,X,F,&IS_act);CHKERRQ(ierr);

    if (vi->checkredundancy) {
      (*vi->checkredundancy)(snes,IS_act,&IS_redact,vi->ctxP);CHKERRQ(ierr);
      if (IS_redact) {
        ierr = ISSort(IS_redact);CHKERRQ(ierr);
        ierr = ISComplement(IS_redact,X->map->rstart,X->map->rend,&vi->IS_inact);CHKERRQ(ierr);
        ierr = ISDestroy(&IS_redact);CHKERRQ(ierr);
      } else {
        ierr = ISComplement(IS_act,X->map->rstart,X->map->rend,&vi->IS_inact);CHKERRQ(ierr);
      }
    } else {
      ierr = ISComplement(IS_act,X->map->rstart,X->map->rend,&vi->IS_inact);CHKERRQ(ierr);
    }
    if (0) {
      Vec temp;
      ierr = VecDuplicate(F,&temp); CHKERRQ(ierr);CHKMEMQ;
      ierr = VecSet(temp,0.0);CHKERRQ(ierr);CHKMEMQ;
    }


    /* Create inactive set submatrix */
    ierr = TaoMatGetSubMat(snes->jacobian,vi->IS_inact,wx,vi->subset_type,&jac_inact_inact);CHKERRQ(ierr);
    if (snes->jacobian == snes->jacobian_pre) {
      ierr = MatDestroy(&prejac_inact_inact);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)(prejac_inact_inact));
      prejac_inact_inact = jac_inact_inact;
    } else {
      ierr = TaoMatGetSubMat(snes->jacobian_pre,vi->IS_inact,wx,vi->subset_type,&prejac_inact_inact);CHKERRQ(ierr);
    }

    // old ierr = MatGetSubMatrix(snes->jacobian,vi->IS_inact,vi->IS_inact,MAT_INITIAL_MATRIX,&jac_inact_inact);CHKERRQ(ierr);
    if (0) {
      Vec temp;
      ierr = VecDuplicate(F,&temp); CHKERRQ(ierr);CHKMEMQ;
      ierr = VecSet(temp,0.0);CHKERRQ(ierr);CHKMEMQ;
    }

    ierr = DMSetVI(snes->dm,vi->IS_inact);CHKERRQ(ierr);
    if (0) {
      Vec temp;
      ierr = VecDuplicate(F,&temp); CHKERRQ(ierr);CHKMEMQ;
      ierr = VecSet(temp,0.0);CHKERRQ(ierr);CHKMEMQ;
    }

    /* Get sizes of active and inactive sets */
    ierr = ISGetLocalSize(IS_act,&nis_act);CHKERRQ(ierr);
    if (0) {
      Vec temp;
      ierr = VecDuplicate(F,&temp); CHKERRQ(ierr);CHKMEMQ;
      ierr = VecSet(temp,0.0);CHKERRQ(ierr);CHKMEMQ;
    }

    ierr = ISGetLocalSize(vi->IS_inact,&nis_inact);CHKERRQ(ierr);

    /* Create active and inactive set vectors */
    if (0) {
      Vec temp;
      ierr = VecDuplicate(F,&temp); CHKERRQ(ierr);CHKMEMQ;
      ierr = VecSet(temp,0.0);CHKERRQ(ierr);CHKMEMQ;
    }
    CHKMEMQ;
    ierr = TaoVecGetSubVec(F,vi->IS_inact,vi->subset_type,0.0,&F_inact);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = TaoVecGetSubVec(Y,vi->IS_inact,vi->subset_type,0.0,&Y_inact);CHKERRQ(ierr);
    CHKMEMQ;
    ierr = TaoVecGetSubVec(Y,IS_act,vi->subset_type,0.0,&Y_act);CHKERRQ(ierr);
    CHKMEMQ;
    /* If active set has changed, then reset KSP (and PC) */
    ierr = ISEqual(vi->IS_inact_prev,vi->IS_inact,&isequal);CHKERRQ(ierr);
    if (!isequal) {
      ierr = KSPReset(snes->ksp);CHKERRQ(ierr);
    }

    /* Active set direction = 0 */
    ierr = VecSet(Y_act,0);CHKERRQ(ierr);
    if (0) {
      PetscInt ny,nf;
      ierr = VecGetLocalSize(F_inact,&nf); CHKERRQ(ierr);
      ierr = VecGetLocalSize(Y_inact,&ny); CHKERRQ(ierr);
      printf("nF = %d, nY = %d\n",nf,ny);
    }
    ierr = KSPSetOperators(snes->ksp,jac_inact_inact,prejac_inact_inact);CHKERRQ(ierr);
    ierr = KSPSetUp(snes->ksp);CHKERRQ(ierr);
    ierr = KSPSolve(snes->ksp,F_inact,Y_inact);CHKERRQ(ierr);
    ierr = KSPGetConvergedReason(snes->ksp,&kspreason);CHKERRQ(ierr);
    if (kspreason < 0) {
      if (++snes->numLinearSolveFailures >= snes->maxLinearSolveFailures) {
        ierr         = PetscInfo2(snes,"iter=%D, number linear solve failures %D greater than current SNES allowed, stopping solve\n",snes->iter,snes->numLinearSolveFailures);CHKERRQ(ierr);
        snes->reason = SNES_DIVERGED_LINEAR_SOLVE;
        break;
      }
     }
    ierr = VecSet(Y,0.0);
    ierr = VecISAXPY(Y,IS_act,1.0,Y_act);CHKERRQ(ierr);
    ierr = VecISAXPY(Y,vi->IS_inact,1.0,Y_inact);CHKERRQ(ierr);


    ierr = VecDestroy(&F_inact);CHKERRQ(ierr);
    ierr = VecDestroy(&Y_act);CHKERRQ(ierr);
    ierr = VecDestroy(&Y_inact);CHKERRQ(ierr);
    ierr = ISDestroy(&IS_act);CHKERRQ(ierr);
    if (!isequal) {
      ierr = ISDestroy(&vi->IS_inact_prev);CHKERRQ(ierr);
      ierr = ISDuplicate(vi->IS_inact,&vi->IS_inact_prev);CHKERRQ(ierr);
    }
    ierr = ISDestroy(&vi->IS_inact);CHKERRQ(ierr);
    ierr = MatDestroy(&jac_inact_inact);CHKERRQ(ierr);
    if (snes->jacobian != snes->jacobian_pre) {
      ierr = MatDestroy(&prejac_inact_inact);CHKERRQ(ierr);
    }
    ierr              = KSPGetIterationNumber(snes->ksp,&lits);CHKERRQ(ierr);
    snes->linear_its += lits;
    ierr              = PetscInfo2(snes,"iter=%D, linear solve iterations=%D\n",snes->iter,lits);CHKERRQ(ierr);

    /* Compute a (scaled) negative update in the line search routine:
         Y <- X - lambda*Y
       and evaluate G = function(Y) (depends on the line search).
    */
    ierr  = VecCopy(Y,snes->vec_sol_update);CHKERRQ(ierr);
    ynorm = 1; gnorm = fnorm;
    ierr  = SNESLineSearchApply(snes->linesearch, X, F, &gnorm, Y);CHKERRQ(ierr);
    ierr  = SNESLineSearchGetSuccess(snes->linesearch, &lssucceed);CHKERRQ(ierr);
    ierr  = SNESLineSearchGetNorms(snes->linesearch, &xnorm, &gnorm, &ynorm);CHKERRQ(ierr);
    ierr  = PetscInfo4(snes,"fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, lssucceed=%d\n",(double)fnorm,(double)gnorm,(double)ynorm,(int)lssucceed);CHKERRQ(ierr);
    if (snes->reason == SNES_DIVERGED_FUNCTION_COUNT) break;
    if (snes->domainerror) {
      snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
      ierr         = DMDestroyVI(snes->dm);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
    if (!lssucceed) {
      if (++snes->numFailures >= snes->maxFailures) {
        PetscBool ismin;
        snes->reason = SNES_DIVERGED_LINE_SEARCH;
        ierr         = SNESVICheckLocalMin_Private(snes,snes->jacobian,F,X,gnorm,&ismin);CHKERRQ(ierr);
        if (ismin) snes->reason = SNES_DIVERGED_LOCAL_MIN;
        break;
      }
    }
    /* Update function and solution vectors */
    fnorm = gnorm;
    /* Monitor convergence */
    ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
    snes->iter = i+1;
    snes->norm = fnorm;
    ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
    ierr       = SNESLogConvergenceHistory(snes,snes->norm,lits);CHKERRQ(ierr);
    ierr       = SNESMonitor(snes,snes->iter,snes->norm);CHKERRQ(ierr);
    /* Test for convergence, xnorm = || X || */
    if (snes->ops->converged != SNESConvergedSkip) { ierr = VecNorm(X,NORM_2,&xnorm);CHKERRQ(ierr); }
    ierr = (*snes->ops->converged)(snes,snes->iter,xnorm,ynorm,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) break;
  }
  ierr = DMDestroyVI(snes->dm);CHKERRQ(ierr);
  if (i == maxits) {
    ierr = PetscInfo1(snes,"Maximum number of iterations has been reached: %D\n",maxits);CHKERRQ(ierr);
    if (!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESVISetRedundancyCheck"
PetscErrorCode SNESVISetRedundancyCheck(SNES snes,PetscErrorCode (*func)(SNES,IS,IS*,void*),void *ctx)
{
  SNES_VINEWTONRSLS *vi = (SNES_VINEWTONRSLS*)snes->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  vi->checkredundancy = func;
  vi->ctxP            = ctx;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MATLAB_ENGINE)
#include <engine.h>
#include <mex.h>
typedef struct {char *funcname; mxArray *ctx;} SNESMatlabContext;

#undef __FUNCT__
#define __FUNCT__ "SNESVIRedundancyCheck_Matlab"
PetscErrorCode SNESVIRedundancyCheck_Matlab(SNES snes,IS is_act,IS *is_redact,void *ctx)
{
  PetscErrorCode    ierr;
  SNESMatlabContext *sctx = (SNESMatlabContext*)ctx;
  int               nlhs  = 1, nrhs = 5;
  mxArray           *plhs[1], *prhs[5];
  long long int     l1      = 0, l2 = 0, ls = 0;
  PetscInt          *indices=NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidHeaderSpecific(is_act,IS_CLASSID,2);
  PetscValidPointer(is_redact,3);
  PetscCheckSameComm(snes,1,is_act,2);

  /* Create IS for reduced active set of size 0, its size and indices will
   bet set by the Matlab function */
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)snes),0,indices,PETSC_OWN_POINTER,is_redact);CHKERRQ(ierr);
  /* call Matlab function in ctx */
  ierr    = PetscMemcpy(&ls,&snes,sizeof(snes));CHKERRQ(ierr);
  ierr    = PetscMemcpy(&l1,&is_act,sizeof(is_act));CHKERRQ(ierr);
  ierr    = PetscMemcpy(&l2,is_redact,sizeof(is_act));CHKERRQ(ierr);
  prhs[0] = mxCreateDoubleScalar((double)ls);
  prhs[1] = mxCreateDoubleScalar((double)l1);
  prhs[2] = mxCreateDoubleScalar((double)l2);
  prhs[3] = mxCreateString(sctx->funcname);
  prhs[4] = sctx->ctx;
  ierr    = mexCallMATLAB(nlhs,plhs,nrhs,prhs,"PetscSNESVIRedundancyCheckInternal");CHKERRQ(ierr);
  ierr    = mxGetScalar(plhs[0]);CHKERRQ(ierr);
  mxDestroyArray(prhs[0]);
  mxDestroyArray(prhs[1]);
  mxDestroyArray(prhs[2]);
  mxDestroyArray(prhs[3]);
  mxDestroyArray(plhs[0]);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESVISetRedundancyCheckMatlab"
PetscErrorCode SNESVISetRedundancyCheckMatlab(SNES snes,const char *func,mxArray *ctx)
{
  PetscErrorCode    ierr;
  SNESMatlabContext *sctx;

  PetscFunctionBegin;
  /* currently sctx is memory bleed */
  ierr      = PetscMalloc(sizeof(SNESMatlabContext),&sctx);CHKERRQ(ierr);
  ierr      = PetscStrallocpy(func,&sctx->funcname);CHKERRQ(ierr);
  sctx->ctx = mxDuplicateArray(ctx);
  ierr      = SNESVISetRedundancyCheck(snes,SNESVIRedundancyCheck_Matlab,sctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif

/* -------------------------------------------------------------------------- */
/*
   SNESSetUp_VINEWTONRSLS - Sets up the internal data structures for the later use
   of the SNESVI nonlinear solver.

   Input Parameter:
.  snes - the SNES context
.  x - the solution vector

   Application Interface Routine: SNESSetUp()

   Notes:
   For basic use of the SNES solvers, the user need not explicitly call
   SNESSetUp(), since these actions will automatically occur during
   the call to SNESSolve().
 */
#undef __FUNCT__
#define __FUNCT__ "SNESSetUp_VINEWTONRSLS"
PetscErrorCode SNESSetUp_VINEWTONRSLS(SNES snes)
{
  PetscErrorCode    ierr;
  SNES_VINEWTONRSLS *vi = (SNES_VINEWTONRSLS*) snes->data;
  PetscInt          *indices;
  PetscInt          i,n,rstart,rend;
  SNESLineSearch    linesearch;

  PetscFunctionBegin;
  ierr = SNESSetUp_VI(snes);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-tao_subset_type","subset type","",SubSetTypes,(PetscEnum)vi->subset_type,(PetscEnum*)&vi->subset_type,0);CHKERRQ(ierr);

  /* Set up previous active index set for the first snes solve
   vi->IS_inact_prev = 0,1,2,....N */

  ierr = VecGetOwnershipRange(snes->vec_sol,&rstart,&rend);CHKERRQ(ierr);
  ierr = VecGetLocalSize(snes->vec_sol,&n);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&indices);CHKERRQ(ierr);
  for (i=0; i < n; i++) indices[i] = rstart + i;
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)snes),n,indices,PETSC_OWN_POINTER,&vi->IS_inact_prev);CHKERRQ(ierr);

  /* set the line search functions */
  if (!snes->linesearch) {
    ierr = SNESGetLineSearch(snes, &linesearch);CHKERRQ(ierr);
    ierr = SNESLineSearchSetType(linesearch, SNESLINESEARCHBT);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "SNESReset_VINEWTONRSLS"
PetscErrorCode SNESReset_VINEWTONRSLS(SNES snes)
{
  SNES_VINEWTONRSLS *vi = (SNES_VINEWTONRSLS*) snes->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = SNESReset_VI(snes);CHKERRQ(ierr);
  ierr = ISDestroy(&vi->IS_inact_prev);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESView_VINEWTONRSLS"
PetscErrorCode SNESView_VINEWTONRSLS(SNES snes,PetscViewer viewer)
{
  SNES_VINEWTONRSLS *vi = (SNES_VINEWTONRSLS*) snes->data;
  PetscBool         isascii;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Active Set subset type: %s\n",SubSetTypes[vi->subset_type]);
  }
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*MC
      SNESVINEWTONRSLS - Reduced space active set solvers for variational inequalities based on Newton's method

   Options Database:
+   -snes_vi_type <ss,rs,rsaug> a semi-smooth solver, a reduced space active set method, and a reduced space active set method that does not eliminate the active constraints from the Jacobian instead augments the Jacobian with additional variables that enforce the constraints
-   -snes_vi_monitor - prints the number of active constraints at each iteration.

   Level: beginner

   References:
   - T. S. Munson, and S. Benson. Flexible Complementarity Solvers for Large-Scale
     Applications, Optimization Methods and Software, 21 (2006).

.seealso:  SNESVISetVariableBounds(), SNESVISetComputeVariableBounds(), SNESCreate(), SNES, SNESSetType(), SNESVINEWTONRSLS, SNESVINEWTONSSLS, SNESNEWTONTR, SNESLineSearchSet(),
           SNESLineSearchSetPostCheck(), SNESLineSearchNo(), SNESLineSearchCubic(), SNESLineSearchQuadratic(),
           SNESLineSearchSet(), SNESLineSearchNoNorms(), SNESLineSearchSetPreCheck(), SNESLineSearchSetParams(), SNESLineSearchGetParams()

M*/
#undef __FUNCT__
#define __FUNCT__ "SNESCreate_VINEWTONRSLS"
PETSC_EXTERN PetscErrorCode SNESCreate_VINEWTONRSLS(SNES snes)
{
  PetscErrorCode    ierr;
  SNES_VINEWTONRSLS *vi;

  PetscFunctionBegin;
  snes->ops->reset          = SNESReset_VINEWTONRSLS;
  snes->ops->setup          = SNESSetUp_VINEWTONRSLS;
  snes->ops->solve          = SNESSolve_VINEWTONRSLS;
  snes->ops->destroy        = SNESDestroy_VI;
  snes->ops->setfromoptions = SNESSetFromOptions_VI;
  snes->ops->view           = SNESView_VINEWTONRSLS;
  snes->ops->converged      = SNESConvergedDefault_VI;

  snes->usesksp = PETSC_TRUE;
  snes->usespc  = PETSC_FALSE;

  ierr                = PetscNewLog(snes,&vi);CHKERRQ(ierr);
  snes->data          = (void*)vi;
  vi->checkredundancy = NULL;
  ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESVISetVariableBounds_C",SNESVISetVariableBounds_VI);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESVISetComputeVariableBounds_C",SNESVISetComputeVariableBounds_VI);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

