#include <../src/ksp/pc/impls/gamg/gamg.h>        /*I "petscpc.h" I*/
#include <petsc-private/kspimpl.h>
#include <petscblaslapack.h>

typedef struct {
  PetscInt nv;   /* number of test vectors to use in the construction of the prolongator */
} PC_GAMG_Bootstrap;


/* the bootstrap test space lives on the operator and must be improved by the operator */
typedef struct {
  PetscInt  nv;      /* the number of vectors in the space */
  PetscInt  nvset;   /* the number of vectors in the space that have been set */
  Vec       *v;      /* the subspace */
} BootstrapTestSpace;

/* the boostrap prolong info lives on the prolongator and maps the coarse indices to a fine unknown */
typedef struct {
  PetscInt *idxmap;
} BootstrapProlongInfo;

#undef __FUNCT__
#define __FUNCT__ "PCGAMGBootstrapTestSpaceCreate"
PetscErrorCode PCGAMGBootstrapTestSpaceCreate(BootstrapTestSpace **bsts,Mat G,PetscInt n)
{
  BootstrapTestSpace *ts;
  PetscErrorCode ierr;
  Vec            v;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(BootstrapTestSpace),&ts);CHKERRQ(ierr);
  ierr = MatGetVecs(G,&v,NULL);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(v,n,&ts->v);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ts->nv = n;
  *bsts = ts;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGBootstrapTestSpaceDestroy"
PetscErrorCode PCGAMGBootstrapTestSpaceDestroy(BootstrapTestSpace *bsts)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = VecDestroyVecs(bsts->nv,&bsts->v);CHKERRQ(ierr);
  ierr = PetscFree(bsts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PCGAMGBootstrapCreateTestSpace_Private"
PetscErrorCode PCGAMGBootstrapCreateTestSpace_Private(Mat G,PetscInt nv,Vec **v) {
  PetscErrorCode     ierr;
  PetscContainer     c;
  BootstrapTestSpace *bsts;

  PetscFunctionBegin;
  ierr = PetscContainerCreate(PetscObjectComm((PetscObject)G),&c);CHKERRQ(ierr);
  ierr = PCGAMGBootstrapTestSpaceCreate(&bsts,G,nv);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(c,bsts);CHKERRQ(ierr);
  ierr = PetscContainerSetUserDestroy(c,(PetscErrorCode (*)(void *))PCGAMGBootstrapTestSpaceDestroy);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)G,"BootstrapTestSpace",(PetscObject)c);CHKERRQ(ierr);
  ierr = PetscContainerDestroy(&c);CHKERRQ(ierr);
  *v = bsts->v;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGBootstrapGetTestSpace_Private"
PetscErrorCode PCGAMGBootstrapGetTestSpace_Private(Mat G,PetscInt *nv,Vec **v)
{
  PetscErrorCode     ierr;
  PetscContainer     c;
  BootstrapTestSpace *bsts;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)G,"BootstrapTestSpace",(PetscObject *)&c);
  if (!c) {
    if (nv) *nv = 0;
    if (v)  *v = NULL;
    PetscFunctionReturn(0);
  }
  ierr = PetscContainerGetPointer(c,(void**)&bsts);CHKERRQ(ierr);
  if (nv) *nv = bsts->nv;
  if (v) *v = bsts->v;
  PetscFunctionReturn(0);
}

/* prolong info functions */
#undef __FUNCT__
#define __FUNCT__ "PCGAMGBootstrapProlongInfoCreate"
PetscErrorCode PCGAMGBootstrapProlongInfoCreate(BootstrapProlongInfo **bspi,PetscInt *idxmap)
{
  BootstrapProlongInfo *pi;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(BootstrapTestSpace),&pi);CHKERRQ(ierr);
  pi->idxmap=idxmap;
  *bspi = pi;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGBootstrapProlongInfoDestroy"
PetscErrorCode PCGAMGBootstrapProlongInfoDestroy(BootstrapProlongInfo *bspi)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscFree(bspi->idxmap);CHKERRQ(ierr);
  ierr = PetscFree(bspi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PCGAMGBootstrapProlongInfoInitialize_Private"
PetscErrorCode PCGAMGBootstrapProlongInfoInitialize_Private(Mat P,PetscInt *idxmap) {
  PetscErrorCode     ierr;
  PetscContainer     c;
  BootstrapProlongInfo *bspi;

  PetscFunctionBegin;
  ierr = PetscContainerCreate(PetscObjectComm((PetscObject)P),&c);CHKERRQ(ierr);
  ierr = PCGAMGBootstrapProlongInfoCreate(&bspi,idxmap);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(c,bspi);CHKERRQ(ierr);
  ierr = PetscContainerSetUserDestroy(c,(PetscErrorCode (*)(void *))PCGAMGBootstrapProlongInfoDestroy);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)P,"BootstrapProlongInfo",(PetscObject)c);CHKERRQ(ierr);
  ierr = PetscContainerDestroy(&c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGBootstrapProlongInfoCopy_Private"
PetscErrorCode PCGAMGBootstrapProlongInfoCopy_Private(Mat P,Mat Pnew) {
  PetscErrorCode     ierr;
  PetscContainer     c;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)P,"BootstrapProlongInfo",(PetscObject *)&c);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)Pnew,"BootstrapProlongInfo",(PetscObject)c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGBootstrapGetProlongInfo_Private"
PetscErrorCode PCGAMGBootstrapGetProlongInfo_Private(Mat G,BootstrapProlongInfo **bspi)
{
  PetscErrorCode       ierr;
  PetscContainer       c;
  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)G,"BootstrapProlongInfo",(PetscObject *)&c);CHKERRQ(ierr);
  if (!c) {
    if (bspi)  *bspi = NULL;
    PetscFunctionReturn(0);
  }
  if (bspi) {ierr = PetscContainerGetPointer(c,(void**)bspi);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* -- */


#undef __FUNCT__
#define __FUNCT__ "PCGAMGSetupLevel_Bootstrap"
PetscErrorCode PCGAMGSetupLevel_Bootstrap(PC pc,Mat Gf,Mat P, Mat Gc)
{
  PetscErrorCode ierr;
  PC_MG             *mg          = (PC_MG*)pc->data;
  PC_GAMG           *pc_gamg     = (PC_GAMG*)mg->innerctx;
  PC_GAMG_Bootstrap *pc_gamg_bs  = (PC_GAMG_Bootstrap*)pc_gamg->subctx;
  Vec               *vf;
  Vec               *vc;
  PetscInt          nv = pc_gamg_bs->nv,i;

  PetscFunctionBegin;
  ierr = PCGAMGBootstrapGetTestSpace_Private(Gf,NULL,&vf);CHKERRQ(ierr);
  if (!vf) {
    ierr = PCGAMGBootstrapCreateTestSpace_Private(Gf,nv,&vf);CHKERRQ(ierr);
  }
  ierr = PCGAMGBootstrapCreateTestSpace_Private(Gc,nv,&vc);CHKERRQ(ierr);
  for (i=0;i<nv;i++) {
    ierr = MatRestrict(P,vf[i],vc[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGBootstrapCreateGhostVector_Private"
PetscErrorCode PCGAMGBootstrapCreateGhostVector_Private(Mat G,Vec *gvec,PetscInt **global)
{
  Mat_MPIAIJ     *aij = (Mat_MPIAIJ*)G->data;
  PetscErrorCode ierr;
  PetscBool      isMPIAIJ;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)G, MATMPIAIJ, &isMPIAIJ); CHKERRQ(ierr);
  if (isMPIAIJ) {
    if (gvec)ierr = VecDuplicate(aij->lvec,gvec);CHKERRQ(ierr);
    if (global)*global = aij->garray;
  } else {
    /* no off-processor nodes */
    if (gvec)*gvec = NULL;
    if (global)*global = NULL;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGBootstrapGraphSplitting_Private"
/*
 Split the relevant graph into diagonal and off-diagonal parts in local numbering; for now this
 a roundabout private interface to the mats' internal diag and offdiag mats.
 */
PetscErrorCode PCGAMGBootstrapGraphSplitting_Private(Mat G,Mat *Gd, Mat *Go)
{
  Mat_MPIAIJ     *aij = (Mat_MPIAIJ*)G->data;
  PetscErrorCode ierr;
  PetscBool      isMPIAIJ;
  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)G, MATMPIAIJ, &isMPIAIJ ); CHKERRQ(ierr);
  if (isMPIAIJ) {
    *Gd = aij->A;
    *Go = aij->B;
  } else {
    *Gd = G;
    *Go = NULL;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGGraph_Bootstrap"
PetscErrorCode PCGAMGGraph_Bootstrap(PC pc,const Mat A,Mat *G)
{
  PetscErrorCode            ierr;
  PC_MG                     *mg          = (PC_MG*)pc->data;
  PC_GAMG                   *pc_gamg     = (PC_GAMG*)mg->innerctx;
  PetscInt                  verbose      = pc_gamg->verbose;
  PetscReal                 vfilter      = pc_gamg->threshold;
  Mat                       Gmat;

  PetscFunctionBegin;
  ierr = PCGAMGCreateGraph(A,&Gmat);CHKERRQ(ierr);
  ierr = PCGAMGFilterGraph(&Gmat,vfilter,PETSC_FALSE,verbose);CHKERRQ(ierr);
  *G = Gmat;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGCoarsen_Bootstrap"
PetscErrorCode PCGAMGCoarsen_Bootstrap(PC pc,Mat *G,PetscCoarsenData **agg_lists)
{
  PetscErrorCode   ierr;
  MatCoarsen       crs;
  MPI_Comm         fcomm = ((PetscObject)pc)->comm;

  PetscFunctionBegin;

  /* construct the graph if necessary */
  if (!G) {
    SETERRQ(fcomm,PETSC_ERR_ARG_WRONGSTATE,"Must set Graph in PC in PCGAMG before coarsening");
  }

  ierr = MatCoarsenCreate(fcomm,&crs);CHKERRQ(ierr);
  ierr = MatCoarsenSetFromOptions(crs);CHKERRQ(ierr);
  ierr = MatCoarsenSetAdjacency(crs,*G);CHKERRQ(ierr);
  ierr = MatCoarsenSetStrictAggs(crs,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatCoarsenApply(crs);CHKERRQ(ierr);
  ierr = MatCoarsenGetData(crs,agg_lists);CHKERRQ(ierr);
  ierr = MatCoarsenDestroy(&crs);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGBootstrapGhost_Private"
/*
 Find all ghost nodes that are coarse and output the fine/coarse splitting for those as well

 Input:
 G - graph;
 gvec - Global Vector
 avec - Local part of the scattered vec
 bvec - Global part of the scattered vec

 Output:
 findx - indirection t

 */
PetscErrorCode PCGAMGBootstrapGhost_Private(Mat G,Vec v,Vec gv)
{
  PetscErrorCode ierr;
  Mat_MPIAIJ     *aij = (Mat_MPIAIJ*)G->data;
  PetscBool      isMPIAIJ;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)G, MATMPIAIJ, &isMPIAIJ ); CHKERRQ(ierr);
  if (isMPIAIJ) {
    ierr = VecScatterBegin(aij->Mvctx,v,gv,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(aij->Mvctx,v,gv,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGOptprol_Bootstrap"
PetscErrorCode PCGAMGOptprol_Bootstrap(PC pc,const Mat A,Mat *aP)
{
  PetscErrorCode    ierr;
  MPI_Comm          comm;
  Mat               P=*aP;
  Mat               Pnew;
  Mat               lP,gP;
  PetscInt          rs,re,rn;                  /* fine (row) ownership range*/
  PetscInt          i,j,k,l,iidx;              /* indices! */
  Vec               *vs,cgv,*cvs,*cgvs;        /* vec of off-diagonal fine size */
  const PetscScalar *vcols,*gvcols;
  const PetscInt    *icols,*gicols;
  PetscScalar       *wvcols;
  PetscInt          *wicols;
  PC_MG             *mg          = (PC_MG*)pc->data;
  PC_GAMG           *pc_gamg     = (PC_GAMG*)mg->innerctx;
  PC_GAMG_Bootstrap *bs          = (PC_GAMG_Bootstrap*)pc_gamg->subctx;
  PetscInt          ncols,gncols,ncolstotal,nv=bs->nv,ncolsloc;
  PetscScalar       *a,*b;
  PetscScalar       *wts;
  Vec               wf,wc;                    /* fine and coarse work vectors */
  PetscInt          cn,cs,ce;
  PetscRandom       rand;
  PetscInt          *gidx;
  PetscScalar       *vsarray,*cvsarray,*cgvsarray;
  PetscScalar       denom;
  PetscInt          sz,ncolsmax=0;
  PetscBLASInt      nls,mls,nrhs,lda,ldb,lwork,info,rank;
  PetscScalar       *work;
  PetscReal         *s,rcond;
  PetscInt          *idxmap;
  IS                is;
  VecScatter        inj;
  KSP               smoothksp;
  PC                smoothpc;
  BootstrapProlongInfo *bspi;
#if defined(PETSC_USE_COMPLEX)
  PetscReal         *rwork;
#endif
  PetscFunctionBegin;
  comm = ((PetscObject)pc)->comm;
  const char *prefix;

  /* split the prolongator into two */
  ierr = PCGAMGBootstrapGraphSplitting_Private(P,&lP,&gP);CHKERRQ(ierr);
  ierr = MatDuplicate(P,MAT_SHARE_NONZERO_PATTERN,&Pnew);CHKERRQ(ierr);

  /* build the scatter injection into the coarse space */
  ierr = MatGetVecs(P,&wc,&wf);CHKERRQ(ierr);
  ierr = VecGetLocalSize(wc,&cn);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(wc,&cs,&ce);CHKERRQ(ierr);
  ierr = PCGAMGBootstrapGetProlongInfo_Private(P,&bspi);CHKERRQ(ierr);
  idxmap=bspi->idxmap;
  ierr = ISCreateGeneral(comm,cn,idxmap,PETSC_USE_POINTER,&is);CHKERRQ(ierr);
  ierr = VecScatterCreate(wf,is,wc,NULL,&inj);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rs,&re);CHKERRQ(ierr);
  rn = (re - rs);
  /* scatter to the ghost vector */
  ierr = PCGAMGBootstrapGetTestSpace_Private(A,NULL,&vs);
  if (!vs) {
    ierr = PCGAMGBootstrapCreateTestSpace_Private(A,nv,&vs);CHKERRQ(ierr);
    ierr = PetscRandomCreate(comm,&rand);CHKERRQ(ierr);
    ierr = PetscRandomSetType(rand,PETSCRAND48);CHKERRQ(ierr);
    for (i=0;i<nv;i++) {
      if (i==0) {
        ierr = VecSet(vs[i],1.);CHKERRQ(ierr);
      } else {
        ierr = VecSetRandom(vs[i],rand);CHKERRQ(ierr);
      }
    }
    ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
    ierr = KSPCreate(comm,&smoothksp);CHKERRQ(ierr);
    ierr = KSPSetType(smoothksp,KSPRICHARDSON);CHKERRQ(ierr);
    ierr = KSPGetPC(smoothksp,&smoothpc);CHKERRQ(ierr);
    ierr = PCSetType(smoothpc,PCSOR);CHKERRQ(ierr);
    ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
    ierr = KSPAppendOptionsPrefix(smoothksp,"smooth_");CHKERRQ(ierr);
    ierr = KSPAppendOptionsPrefix(smoothksp,prefix);CHKERRQ(ierr);
    ierr = KSPSetOperators(smoothksp,A,A,SAME_PRECONDITIONER);CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(smoothksp,PETSC_TRUE);CHKERRQ(ierr);
    ierr = KSPSetTolerances(smoothksp,1e-10,1e-10,1e15,3);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(smoothksp);CHKERRQ(ierr);
    for (i=0;i<nv;i++) {
      ierr = VecSet(wf,0.);CHKERRQ(ierr);
      /* ierr = KSPSolve(smoothksp,wf,vs[i]);CHKERRQ(ierr); */
    }
    ierr = KSPDestroy(&smoothksp);CHKERRQ(ierr);
  }

  ierr = PetscMalloc(sizeof(PetscScalar)*nv,&wts);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(wc,nv,&cvs);CHKERRQ(ierr);
  if (gP) {
    ierr = PCGAMGBootstrapCreateGhostVector_Private(P,&cgv,&gidx);CHKERRQ(ierr);
  }
  if (gP) {
    ierr = VecDuplicateVecs(cgv,nv,&cgvs);CHKERRQ(ierr);
  }

  /* find the biggest column and allocate */
  for (i=rs;i<re;i++) {
    ierr = MatGetRow(P,i,&ncols,NULL,NULL);CHKERRQ(ierr);
    if (ncols > ncolsmax) ncolsmax=ncols;
    ierr = MatRestoreRow(P,i,&ncols,NULL,NULL);CHKERRQ(ierr);
  }
  ierr = PetscMalloc6(ncolsmax*ncolsmax,PetscScalar,&a,
                      ncolsmax,PetscScalar,&b,
                      12*ncolsmax,PetscScalar,&work,
                      ncolsmax,PetscScalar,&s,
                      ncolsmax,PetscScalar,&wvcols,
                      ncolsmax,PetscScalar,&wicols);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc(ncolsmax*sizeof(PetscScalar),&rwork);CHKERRQ(ierr);
#endif

  /* construct the weights by the interpolation Rayleigh quotient <Av,v>^-1<Pv,Pv> */
  for (j=0;j<nv;j++) {
    ierr = MatMult(A,vs[j],wf);CHKERRQ(ierr);
    ierr = VecScatterBegin(inj,vs[j],cvs[j],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(inj,vs[j],cvs[j],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecDotBegin(wf,vs[j],&wts[j]);CHKERRQ(ierr);
    ierr = VecDotBegin(cvs[j],cvs[j],&denom);CHKERRQ(ierr);
    ierr = VecDotEnd(wf,vs[j],&wts[j]);CHKERRQ(ierr);
    ierr = VecDotEnd(cvs[j],cvs[j],&denom);CHKERRQ(ierr);
    if (PetscRealPart(wts[j]) > 0.) {
      wts[j] = denom / PetscRealPart(wts[j]);
    } else { /* nullspace */
      wts[j] = 1.;
    }
    ierr = VecGetSize(vs[j],&sz);CHKERRQ(ierr);
    if (gP) {
      ierr = PCGAMGBootstrapGhost_Private(P,cvs[j],cgvs[j]);CHKERRQ(ierr);
    }
  }
  for (i=0;i<rn;i++) {
    iidx=i+rs;
    /* set up the least squares minimization problem */
    ierr = MatGetRow(lP,i,&ncols,&icols,&vcols);CHKERRQ(ierr);
    ncolsloc = ncols;
    ncolstotal = ncols;
    ierr = MatRestoreRow(lP,i,&ncols,&icols,&vcols);CHKERRQ(ierr);
    if (gP) {
      ierr = MatGetRow(gP,i,&ncols,&icols,&vcols);CHKERRQ(ierr);
      ncolstotal += ncols;
      ierr = MatRestoreRow(gP,i,&ncols,&icols,&vcols);CHKERRQ(ierr);
    }
    if (ncolstotal > 1) {
      for (j=0;j<ncolstotal;j++) {
        b[j] = 0.;
        for (k=0;k<ncolstotal;k++) {
          a[k+ncolstotal*j] = 0.;
        }
      }
      ierr = MatGetRow(lP,i,&ncols,&icols,&vcols);CHKERRQ(ierr);
      if (gP) {
        ierr = MatGetRow(gP,i,&gncols,&gicols,&gvcols);CHKERRQ(ierr);
      }
      for (l=0;l<nv;l++) {
        ierr = VecGetArray(vs[l],&vsarray);CHKERRQ(ierr);
        ierr = VecGetArray(cvs[l],&cvsarray);CHKERRQ(ierr);
        if (gP) {
          ierr = VecGetArray(cgvs[l],&cgvsarray);CHKERRQ(ierr);
        }
        /* addition for on-processor entries */
        for (j=0;j<ncols;j++) {
          b[j] += cvsarray[icols[j]]*wts[l]*vsarray[i];
          for (k=0;k<ncols;k++) {
            a[k+j*ncolstotal] += cvsarray[icols[j]]*cvsarray[icols[k]]*wts[l];
          }
          if (gP) {
            for (k=0;k<gncols;k++) {
              a[k+ncolsloc+j*ncolstotal] += cvsarray[icols[j]]*cgvsarray[gicols[k]]*wts[l];
            }
          }
        }
        /* addition for off-processor entries */
        if (gP) {
          for (j=0;j<gncols;j++) {
            b[j+ncolsloc] += cgvsarray[icols[j]]*wts[l]*vsarray[i];
            for (k=0;k<ncols;k++) {
              a[k+(j+ncolsloc)*ncolstotal] += cgvsarray[gicols[j]]*cvsarray[icols[k]]*wts[l];
            }
            for (k=0;k<gncols;k++) {
              a[k+ncolsloc+(j+ncolsloc)*ncolstotal] += cgvsarray[gicols[j]]*cgvsarray[gicols[k]]*wts[l];
            }
          }
        }
        ierr = VecRestoreArray(vs[l],&vsarray);CHKERRQ(ierr);
        ierr = VecRestoreArray(cvs[l],&cvsarray);CHKERRQ(ierr);
        if (gP) {
          ierr = VecRestoreArray(cgvs[l],&cgvsarray);CHKERRQ(ierr);
        }
      }
      ierr = MatRestoreRow(lP,i,&ncols,&icols,&vcols);CHKERRQ(ierr);
      if (gP) {
        ierr = MatRestoreRow(gP,i,&gncols,&gicols,&gvcols);CHKERRQ(ierr);
      }
      nls=ncolstotal;mls=ncolstotal;nrhs=1;lda=ncolstotal;ldb=ncolstotal;lwork=12*ncolstotal;info=0;
      /* solve the problem */
#if defined(PETSC_MISSING_LAPACK_GELSS)
      SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"LS solve in PCGAMGBootstrap requires the LAPACK GELSS routine.");
#else
      rcond         = -1.;
      ierr          = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
      PetscStackCallBLAS("LAPACKgelss",LAPACKgelss_(&nls,&mls,&nrhs,a,&lda,b,&ldb,s,&rcond,&rank,work,&lwork,rwork,&info));
#else
      PetscStackCallBLAS("LAPACKgelss",LAPACKgelss_(&nls,&mls,&nrhs,a,&lda,b,&ldb,s,&rcond,&rank,work,&lwork,&info));
#endif
      ierr = PetscFPTrapPop();CHKERRQ(ierr);
      if (info < 0) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_LIB,"Bad argument to GELSS");
      if (info > 0) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_LIB,"SVD failed to converge");
#endif
      /* set the row to be the solution */
      ierr = MatGetRow(lP,i,&ncols,&icols,&vcols);CHKERRQ(ierr);
      for (j=0;j<ncols;j++) {
        wvcols[j] = b[j];
        wicols[j] = icols[j]+cs;
      }
      ierr = MatRestoreRow(lP,i,&ncols,&icols,&vcols);CHKERRQ(ierr);
      if (gP) {
        ierr = MatGetRow(gP,i,&gncols,&gicols,&gvcols);CHKERRQ(ierr);
        for (j=0;j<gncols;j++) {
          wvcols[j+ncolsloc] = b[j+ncolsloc];
          wicols[j+ncolsloc] = gidx[gicols[j]];
        }
        ierr = MatRestoreRow(gP,i,&gncols,&gicols,&gvcols);CHKERRQ(ierr);
      }
      ierr = MatSetValues(Pnew,1,&iidx,ncolstotal,wicols,wvcols,INSERT_VALUES);CHKERRQ(ierr);
    } else {
      /* set the row to be the solution */
      ierr = MatGetRow(lP,i,&ncols,&icols,&vcols);CHKERRQ(ierr);
      for (j=0;j<ncols;j++) {
        wvcols[j] = vcols[j];
        wicols[j] = icols[j]+cs;
      }
      ncolsloc = ncols;
      ierr = MatRestoreRow(lP,i,&ncols,&icols,&vcols);CHKERRQ(ierr);
      if (gP) {
        ierr = MatGetRow(gP,i,&gncols,&gicols,&gvcols);CHKERRQ(ierr);
        for (j=0;j<gncols;j++) {
          wvcols[j+ncolsloc] = gvcols[j];
          wicols[j+ncolsloc] = gidx[gicols[j]];
        }
        ncolsloc = gncols;
        ierr = MatRestoreRow(gP,i,&gncols,&gicols,&gvcols);CHKERRQ(ierr);
      }
      ierr = MatSetValues(Pnew,1,&iidx,ncolsloc,wicols,wvcols,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree6(a,b,work,s,wvcols,wicols);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscFree(rwork);CHKERRQ(ierr);
#endif

  ierr = PCGAMGBootstrapProlongInfoCopy_Private(P,Pnew);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(Pnew, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Pnew, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = VecDestroy(&wf);CHKERRQ(ierr);
  ierr = VecDestroy(&wc);CHKERRQ(ierr);
  ierr = VecDestroyVecs(nv,&cvs);CHKERRQ(ierr);
  if (gP) {
    ierr = VecDestroy(&cgv);CHKERRQ(ierr);
    ierr = VecDestroyVecs(nv,&cgvs);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&P);CHKERRQ(ierr);
  *aP = Pnew;
  ierr = VecScatterDestroy(&inj);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = PetscFree(wts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGProlongator_Bootstrap"
PetscErrorCode PCGAMGProlongator_Bootstrap(PC pc, const Mat A, const Mat G, PetscCoarsenData *agg_lists,Mat *P)
{
  PetscErrorCode    ierr;
  MPI_Comm          comm;
  Mat               lG,gG,lA,gA;     /* on and off diagonal matrices */
  PetscInt          fn;                        /* fine local blocked sizes */
  PetscInt          cn;                        /* coarse local blocked sizes */
  PetscInt          gn;                        /* size of the off-diagonal fine vector */
  PetscInt          fs,fe;                     /* fine (row) ownership range*/
  PetscInt          cs,ce;                     /* coarse (column) ownership range */
  PetscInt          i,j,k;                     /* indices! */
  PetscBool         iscoarse;                  /* flag for determining if a node is coarse */
  PetscInt          *lcid,*gcid;               /* on and off-processor coarse unknown IDs */
  PetscInt          *lsparse,*gsparse;         /* on and off-processor sparsity patterns for prolongator */
  PetscScalar       pij;
  const PetscScalar *rval;
  const PetscInt    *rcol;
  Vec               F;   /* vec of coarse size */
  Vec               C;   /* vec of fine size */
  Vec               gF;  /* vec of off-diagonal fine size */
  MatType           mtype;
  PetscInt          c_indx;
  const PetscScalar *vcols;
  const PetscInt    *icols;
  PetscScalar       c_scalar;
  PetscInt          ncols,col;
  PetscInt          row_f,row_c;
  PetscInt          cmax=0,ncolstotal,idx;
  PetscScalar       *pvals;
  PetscInt          *pcols;
  PetscInt          *idxmap;
  PetscInt          bs;
  PetscInt          a_n,a_m,g_n,g_m;

  PetscFunctionBegin;
  comm = ((PetscObject)pc)->comm;
  ierr = MatGetOwnershipRange(A,&fs,&fe); CHKERRQ(ierr);
  ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
  fn = (fe - fs);

  ierr = MatGetSize(A,&a_n,&a_m);CHKERRQ(ierr);
  ierr = MatGetSize(G,&g_n,&g_m);CHKERRQ(ierr);
  ierr = MatGetVecs(A,&F,NULL);CHKERRQ(ierr);

  /* get the number of local unknowns and the indices of the local unknowns */

  ierr = PetscMalloc(sizeof(PetscInt)*fn,&lsparse);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*fn,&gsparse);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*fn,&lcid);CHKERRQ(ierr);

  /* count the number of coarse unknowns */
  cn = 0;
  for (i=0;i<fn;i++) {
    /* filter out singletons */
    ierr = PetscCDEmptyAt(agg_lists,i/bs,&iscoarse); CHKERRQ(ierr);
    lcid[i] = -1;
    if (!iscoarse) {
      cn++;
    }
  }

   /* create the coarse vector */
  ierr = VecCreateMPI(comm,cn,PETSC_DECIDE,&C);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(C,&cs,&ce);CHKERRQ(ierr);
  ierr = PetscMalloc(cn*sizeof(PetscInt),&idxmap);CHKERRQ(ierr);

  /* construct a global vector indicating the global indices of the coarse unknowns */
  cn = 0;
  for (i=0;i<fn;i++) {
    ierr = PetscCDEmptyAt(agg_lists,i/bs,&iscoarse); CHKERRQ(ierr);
    if (!iscoarse) {
      lcid[i] = cs+cn;
      idxmap[cn] = cs+cn;
      cn++;
    } else {
      lcid[i] = -1;
    }
    c_scalar = (PetscScalar)lcid[i];
    c_indx = fs+i;
    ierr = VecSetValues(F,1,&c_indx,&c_scalar,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = VecAssemblyBegin(F);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(F);CHKERRQ(ierr);

  /* split the graph into two */
  ierr = PCGAMGBootstrapGraphSplitting_Private(G,&lG,&gG);CHKERRQ(ierr);
  ierr = PCGAMGBootstrapGraphSplitting_Private(A,&lA,&gA);CHKERRQ(ierr);

  /* scatter to the ghost vector */
  ierr = PCGAMGBootstrapCreateGhostVector_Private(G,&gF,NULL);CHKERRQ(ierr);
  ierr = PCGAMGBootstrapGhost_Private(G,F,gF);CHKERRQ(ierr);

  if (gG) {
    ierr = VecGetSize(gF,&gn);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscInt)*gn,&gcid);CHKERRQ(ierr);
    for (i=0;i<gn;i++) {
      ierr = VecGetValues(gF,1,&i,&c_scalar);CHKERRQ(ierr);
      gcid[i] = (PetscInt)PetscRealPart(c_scalar);
    }
  }

  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = VecDestroy(&gF);CHKERRQ(ierr);
  ierr = VecDestroy(&C);CHKERRQ(ierr);

  /* count the on and off processor sparsity patterns for the prolongator */

  for (i=0;i<fn;i++) {
    /* on */
    ierr = MatGetRow(lG,i/bs,&ncols,&icols,&vcols);CHKERRQ(ierr);
    ncolstotal = ncols;
    lsparse[i] = 0;
    gsparse[i] = 0;
    if (lcid[i] >= 0) {
      lsparse[i] = 1;
      gsparse[i] = 0;
    } else {
      for (j = 0;j < ncols;j++) {
        col = icols[j];
        if (lcid[col] >= 0 && vcols[j] != 0.) {
          lsparse[i] += 1;
        }
      }
      ierr = MatRestoreRow(lG,i/bs,&ncols,&icols,&vcols);CHKERRQ(ierr);
      ncolstotal += ncols;
      /* off */
      if (gG) {
        ierr = MatGetRow(gG,i/bs,&ncols,&icols,&vcols);CHKERRQ(ierr);
        for (j = 0; j < ncols; j++) {
          col = icols[j];
          if (gcid[col] >= 0 && vcols[j] != 0.) {
            gsparse[i] += 1;
          }
        }
        ierr = MatRestoreRow(gG,i/bs,&ncols,NULL,NULL);CHKERRQ(ierr);
      }
      if (ncolstotal > cmax) cmax = ncolstotal;
    }
  }

  ierr = PetscMalloc(sizeof(PetscInt)*cmax,&pcols);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*cmax,&pvals);CHKERRQ(ierr);

  /* preallocate and create the prolongator */
  ierr = MatCreate(comm,P); CHKERRQ(ierr);
  ierr = MatGetType(G,&mtype);CHKERRQ(ierr);
  ierr = MatSetType(*P,mtype);CHKERRQ(ierr);

  ierr = MatSetSizes(*P,fn,cn,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(*P,0,lsparse,0,gsparse);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(*P,0,lsparse);CHKERRQ(ierr);

  for (i = 0;i < fn;i++) {
    /* determine on or off */
    row_f = i + fs;
    row_c = lcid[i];
    if (row_c >= 0) {
      pij = 1.;
      ierr = MatSetValues(*P,1,&row_f,1,&row_c,&pij,INSERT_VALUES);CHKERRQ(ierr);
    } else {
      PetscInt ntotal=0;
      /* local connections */
      ierr = MatGetRow(lG,i/bs,&ncols,&rcol,&rval);CHKERRQ(ierr);
      for (k = 0; k < ncols; k++) {
        if (lcid[rcol[k]] >= 0) {
          ntotal++;
        }
      }
      ierr = MatRestoreRow(lG,i/bs,&ncols,&rcol,&rval);CHKERRQ(ierr);

      /* ghosted connections */
      if (gG) {
        ierr = MatGetRow(gG,i/bs,&ncols,&rcol,&rval);CHKERRQ(ierr);
        for (k = 0; k < ncols; k++) {
          if (gcid[rcol[k]] >= 0) {
            ntotal++;
          }
        }
        ierr = MatRestoreRow(gG,i/bs,&ncols,&rcol,&rval);CHKERRQ(ierr);
      }

      /* on */
      ierr = MatGetRow(lG,i/bs,&ncols,&icols,&vcols);CHKERRQ(ierr);
      idx = 0;
      for (j = 0;j < ncols;j++) {
        col = icols[j];
        if (lcid[col] >= 0 && vcols[j] != 0.) {
          row_f = i + fs;
          row_c = lcid[col];
          pij = 1. / ntotal;
          pvals[idx] = pij;
          pcols[idx] = row_c;
          idx++;
        }
      }
      ierr = MatRestoreRow(lG,i/bs,&ncols,&icols,&vcols);CHKERRQ(ierr);
      /* off */
      if (gG) {
        ierr = MatGetRow(gG,i/bs,&ncols,&icols,&vcols);CHKERRQ(ierr);
        for (j = 0; j < ncols; j++) {
          col = icols[j];
          if (gcid[col] >= 0 && vcols[j] != 0.) {
            row_f = i + fs;
            row_c = gcid[col];
            pij = 1./ntotal;
            pvals[idx] = pij;
            pcols[idx] = row_c;
            idx++;
          }
        }
        ierr = MatRestoreRow(gG,i/bs,&ncols,&icols,&vcols);CHKERRQ(ierr);
      }
      ierr = MatSetValues(*P,1,&row_f,idx,pcols,pvals,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(*P, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*P, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PCGAMGBootstrapProlongInfoInitialize_Private(*P,idxmap);CHKERRQ(ierr);

  ierr = PetscFree(lsparse);CHKERRQ(ierr);
  ierr = PetscFree(gsparse);CHKERRQ(ierr);
  ierr = PetscFree(pcols);CHKERRQ(ierr);
  ierr = PetscFree(pvals);CHKERRQ(ierr);
  ierr = PetscFree(lcid);CHKERRQ(ierr);
  if (gG) {ierr = PetscFree(gcid);CHKERRQ(ierr);}

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGDestroy_Bootstrap"
PetscErrorCode PCGAMGDestroy_Bootstrap(PC pc)
{
  PetscErrorCode ierr;
  PC_MG          *mg          = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg     = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  ierr = PetscFree(pc_gamg->subctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGSetFromOptions_Bootstrap"
PetscErrorCode PCGAMGSetFromOptions_Bootstrap(PC pc)
{
  PC_MG             *mg      = (PC_MG*)pc->data;
  PC_GAMG           *pc_gamg = (PC_GAMG*)mg->innerctx;
  PC_GAMG_Bootstrap *bs      = (PC_GAMG_Bootstrap*)pc_gamg->subctx;
  PetscErrorCode    ierr;
  PetscBool         flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("GAMGBootstrap options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_gamg_bootstrap_nv","Number of test vectors forming the bootstrap space","",bs->nv,&bs->nv,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGSetData_Bootstrap"
PetscErrorCode PCGAMGSetData_Bootstrap(PC pc, Mat A)
{
  PC_MG          *mg      = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;

  PetscFunctionBegin;
  /* no data for bootstrap AMG */
  pc_gamg->data           = NULL;
  pc_gamg->data_cell_cols = 1;
  pc_gamg->data_cell_rows = 1;
  pc_gamg->data_sz = 0;
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "PCCreateGAMG_Bootstrap"
PetscErrorCode  PCCreateGAMG_Bootstrap(PC pc)
{
  PetscErrorCode ierr;
  PC_MG             *mg      = (PC_MG*)pc->data;
  PC_GAMG           *pc_gamg = (PC_GAMG*)mg->innerctx;
  PC_GAMG_Bootstrap *pc_gamg_bootstrap;

  PetscFunctionBegin;

  /* create sub context for SA */
  ierr = PetscNewLog(pc, PC_GAMG_Bootstrap, &pc_gamg_bootstrap);CHKERRQ(ierr);
  pc_gamg_bootstrap->nv     = 1;
  pc_gamg->subctx           = pc_gamg_bootstrap;

  /* set internal function pointers */
  pc_gamg->ops->setfromoptions = PCGAMGSetFromOptions_Bootstrap;
  pc_gamg->ops->destroy        = PCGAMGDestroy_Bootstrap;
  pc_gamg->ops->graph          = PCGAMGGraph_Bootstrap;
  pc_gamg->ops->coarsen        = PCGAMGCoarsen_Bootstrap;
  pc_gamg->ops->prolongator    = PCGAMGProlongator_Bootstrap;
  pc_gamg->ops->optprol        = PCGAMGOptprol_Bootstrap;
  pc_gamg->ops->setuplevel     = PCGAMGSetupLevel_Bootstrap;

  pc_gamg->ops->createdefaultdata = PCGAMGSetData_Bootstrap;
  PetscFunctionReturn(0);
}
