/*

      Contributed by Arvid Bessen, Columbia University, June 2007

       Extension of DA object to any number of dimensions.

*/
#include "../src/dm/impls/adda/addaimpl.h"                          /*I "petscda.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "ADDACreate"
/*@C
  ADDACreate - Creates and ADDA object that translate between coordinates
  in a geometric grid of arbitrary dimension and data in a PETSc vector
  distributed on several processors.

  Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator
.  dim - the dimension of the grid
.  nodes - array with d entries that give the number of nodes in each dimension
.  procs - array with d entries that give the number of processors in each dimension
          (or PETSC_NULL if to be determined automatically)
.  dof - number of degrees of freedom per node
-  periodic - array with d entries that, i-th entry is set to  true iff dimension i is periodic

   Output Parameters:
.  adda - pointer to ADDA data structure that is created

  Level: intermediate

@*/
PetscErrorCode PETSCDM_DLLEXPORT ADDACreate(MPI_Comm comm, PetscInt dim, PetscInt *nodes,PetscInt *procs,PetscInt dof, PetscBool  *periodic,DM *dm_p)
{
  PetscErrorCode ierr;
  DM             dm;
  PetscInt       s=1; /* stencil width, fixed to 1 at the moment */
  PetscMPIInt    rank,size;
  PetscInt       i;
  PetscInt       nodes_total;
  PetscInt       nodesleft;
  PetscInt       procsleft;
  PetscInt       procsdimi;
  PetscInt       ranki;
  PetscInt       rpq;
  DM_ADDA        *dd;

  PetscFunctionBegin;
  PetscValidPointer(nodes,3);
  PetscValidPointer(dm_p,6);
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = DMInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(*dm_p,_p_DM,struct _DMOps,DM_CLASSID,0,"DM",comm,ADDADestroy,0);CHKERRQ(ierr);
  ierr = PetscNewLog(*dm_p,DM_ADDA,&dd);CHKERRQ(ierr);
  dm = *dm_p;
  dm->ops->view = ADDAView;
  dm->ops->createglobalvector = ADDACreateGlobalVector;
  dm->ops->getcoloring = ADDAGetColoring;
  dm->ops->getmatrix = ADDAGetMatrix;
  dm->ops->getinterpolation = ADDAGetInterpolation;
  dm->ops->refine = ADDARefine;
  dm->ops->coarsen = ADDACoarsen;
  dm->ops->getinjection = ADDAGetInjection;
  dm->ops->getaggregates = ADDAGetAggregates;
  
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr); 
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr); 
  
  dd->dim = dim;
  dd->dof = dof;

  /* nodes */
  ierr = PetscMalloc(dim*sizeof(PetscInt), &(dd->nodes));CHKERRQ(ierr);
  ierr = PetscMemcpy(dd->nodes, nodes, dim*sizeof(PetscInt));CHKERRQ(ierr);
  /* total number of nodes */
  nodes_total = 1;
  for(i=0; i<dim; i++) nodes_total *= nodes[i];

  /* procs */
  ierr = PetscMalloc(dim*sizeof(PetscInt), &(dd->procs));CHKERRQ(ierr);
  /* create distribution of nodes to processors */
  if(procs == PETSC_NULL) {
    procs = dd->procs;
    nodesleft = nodes_total;
    procsleft = size;
    /* figure out a good way to split the array to several processors */
    for(i=0; i<dim; i++) {
      if(i==dim-1) {
	procs[i] = procsleft;
      } else {
	/* calculate best partition */
	procs[i] = (PetscInt)(((double) nodes[i])*pow(((double) procsleft)/((double) nodesleft),1./((double)(dim-i)))+0.5);
	if(procs[i]<1) procs[i]=1;
	while( procs[i] > 0 ) {
	  if( procsleft % procs[i] )
	    procs[i]--;
	  else
	    break;
	}
	nodesleft /= nodes[i];
	procsleft /= procs[i];
      }
    }
  } else {
    /* user provided the number of processors */
    ierr = PetscMemcpy(dd->procs, procs, dim*sizeof(PetscInt));CHKERRQ(ierr);
  }
  /* check for validity */
  procsleft = 1;
  for(i=0; i<dim; i++) {
    if (nodes[i] < procs[i]) {
      SETERRQ3(comm,PETSC_ERR_ARG_OUTOFRANGE,"Partition in direction %d is too fine! %D nodes, %D processors", i, nodes[i], procs[i]);
    }
    procsleft *= procs[i];
  }
  if (procsleft != size) SETERRQ(comm,PETSC_ERR_PLIB, "Created or was provided with inconsistent distribution of processors");

  /* periodicity */
  dd->periodic = periodic;
  
  /* find out local region */
  ierr = PetscMalloc(dim*sizeof(PetscInt), &(dd->lcs));CHKERRQ(ierr);
  ierr = PetscMalloc(dim*sizeof(PetscInt), &(dd->lce));CHKERRQ(ierr);
  procsdimi=size;
  ranki=rank;
  for(i=0; i<dim; i++) {
    /* What is the number of processor for dimensions i+1, ..., dim-1? */
    procsdimi /= procs[i];
    /* these are all nodes that come before our region */
    rpq = ranki / procsdimi;
    dd->lcs[i] = rpq * (nodes[i]/procs[i]);
    if( rpq + 1 < procs[i] ) {
      dd->lce[i] = (rpq + 1) * (nodes[i]/procs[i]);
    } else {
      /* last one gets all the rest */
      dd->lce[i] = nodes[i];
    }
    ranki = ranki - rpq*procsdimi;
  }
  
  /* compute local size */
  dd->lsize=1;
  for(i=0; i<dim; i++) {
    dd->lsize *= (dd->lce[i]-dd->lcs[i]);
  }
  dd->lsize *= dof;

  /* find out ghost points */
  ierr = PetscMalloc(dim*sizeof(PetscInt), &(dd->lgs));CHKERRQ(ierr);
  ierr = PetscMalloc(dim*sizeof(PetscInt), &(dd->lge));CHKERRQ(ierr);
  for(i=0; i<dim; i++) {
    if( periodic[i] ) {
      dd->lgs[i] = dd->lcs[i] - s;
      dd->lge[i] = dd->lce[i] + s;
    } else {
      dd->lgs[i] = PetscMax(dd->lcs[i] - s, 0);
      dd->lge[i] = PetscMin(dd->lce[i] + s, nodes[i]);
    }
  }
  
  /* compute local size with ghost points */
  dd->lgsize=1;
  for(i=0; i<dim; i++) {
    dd->lgsize *= (dd->lge[i]-dd->lgs[i]);
  }
  dd->lgsize *= dof;

  /* create global and local prototype vector */
  ierr = VecCreateMPIWithArray(comm,dd->lsize,PETSC_DECIDE,0,&(dd->global));CHKERRQ(ierr);
  ierr = VecSetBlockSize(dd->global,dd->dof);CHKERRQ(ierr);
#if ADDA_NEEDS_LOCAL_VECTOR
  /* local includes ghost points */
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,dd->lgsize,0,&(dd->local));CHKERRQ(ierr);
  ierr = VecSetBlockSize(dd->local,dof);CHKERRQ(ierr);
#endif

  ierr = PetscMalloc(dim*sizeof(PetscInt), &(dd->refine));CHKERRQ(ierr);
  for(i=0; i<dim; i++) dd->refine[i] = 3;
  dd->dofrefine = 1;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ADDADestroy"
/*@
   ADDADestroy - Destroys a distributed array.

   Collective on ADDA

   Input Parameter:
.  adda - the distributed array to destroy 

   Level: beginner

.keywords: distributed array, destroy

.seealso: ADDACreate()
@*/
PetscErrorCode PETSCDM_DLLEXPORT ADDADestroy(DM dm)
{
  PetscErrorCode ierr;
  DM_ADDA        *dd = (DM_ADDA*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);

  /* check reference count */
  if(--((PetscObject)dm)->refct > 0) PetscFunctionReturn(0);

  /* destroy the allocated data */
  ierr = PetscFree(dd->nodes);CHKERRQ(ierr);
  ierr = PetscFree(dd->procs);CHKERRQ(ierr);
  ierr = PetscFree(dd->lcs);CHKERRQ(ierr);
  ierr = PetscFree(dd->lce);CHKERRQ(ierr);
  ierr = PetscFree(dd->lgs);CHKERRQ(ierr);
  ierr = PetscFree(dd->lge);CHKERRQ(ierr);
  ierr = PetscFree(dd->refine);CHKERRQ(ierr);

  ierr = VecDestroy(dd->global);CHKERRQ(ierr);

  ierr = PetscHeaderDestroy(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ADDAView"
/*@
   ADDAView - Views a distributed array.

   Collective on ADDA

    Input Parameter:
+   adda - the ADDA object to view
-   v - the viewer

    Level: developer

.keywords: distributed array, view

.seealso: DMView()
@*/
PetscErrorCode PETSCDM_DLLEXPORT ADDAView(DM dm, PetscViewer v) 
{
  PetscFunctionBegin;
  SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP, "Not implemented yet");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ADDACreateGlobalVector"
/*@
   ADDACreateGlobalVector - Creates global vector for distributed array.

   Collective on ADDA

   Input Parameter:
.  adda - the distributed array for which we create a global vector

   Output Parameter:
.  vec - the global vector

   Level: beginner

.keywords: distributed array, vector

.seealso: DMCreateGlobalVector()
@*/
PetscErrorCode PETSCDM_DLLEXPORT ADDACreateGlobalVector(DM dm, Vec *vec) 
{
  PetscErrorCode ierr;
  DM_ADDA        *dd = (DM_ADDA*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(vec,2);
  ierr = VecDuplicate(dd->global, vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ADDAGetColoring"
/*@
   ADDAGetColoring - Creates coloring for distributed array.

   Collective on ADDA

   Input Parameter:
+  adda - the distributed array for which we create a global vector
-  ctype - IS_COLORING_GHOSTED or IS_COLORING_LOCAL

   Output Parameter:
.  coloring - the coloring

   Level: developer

.keywords: distributed array, coloring

.seealso: DMGetColoring()
@*/
PetscErrorCode PETSCDM_DLLEXPORT ADDAGetColoring(DM dm, ISColoringType ctype,const MatType mtype,ISColoring *coloring) 
{
  PetscFunctionBegin;
  SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP, "Not implemented yet");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ADDAGetMatrix"
/*@
   ADDAGetMatrix - Creates matrix compatible with distributed array.

   Collective on ADDA

   Input Parameter:
.  adda - the distributed array for which we create the matrix
-  mtype - Supported types are MATSEQAIJ, MATMPIAIJ, MATSEQBAIJ, MATMPIBAIJ, or
           any type which inherits from one of these (such as MATAIJ, MATLUSOL, etc.).

   Output Parameter:
.  mat - the empty Jacobian 

   Level: beginner

.keywords: distributed array, matrix

.seealso: DMGetMatrix()
@*/
PetscErrorCode PETSCDM_DLLEXPORT ADDAGetMatrix(DM dm, const MatType mtype, Mat *mat) 
{
  PetscErrorCode ierr;
  DM_ADDA        *dd = (DM_ADDA*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = MatCreate(((PetscObject)dm)->comm, mat);CHKERRQ(ierr);
  ierr = MatSetSizes(*mat, dd->lsize, dd->lsize, PETSC_DECIDE, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(*mat, mtype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ADDAGetMatrixNS"
/*@
   ADDAGetMatrixNS - Creates matrix compatiable with two distributed arrays

   Collective on ADDA

   Input Parameter:
.  addar - the distributed array for which we create the matrix, which indexes the rows
.  addac - the distributed array for which we create the matrix, which indexes the columns
-  mtype - Supported types are MATSEQAIJ, MATMPIAIJ, MATSEQBAIJ, MATMPIBAIJ, or
           any type which inherits from one of these (such as MATAIJ, MATLUSOL, etc.).

   Output Parameter:
.  mat - the empty Jacobian 

   Level: beginner

.keywords: distributed array, matrix

.seealso: DMGetMatrix()
@*/
PetscErrorCode PETSCDM_DLLEXPORT ADDAGetMatrixNS(DM dm, DM dmc, const MatType mtype, Mat *mat) 
{
  PetscErrorCode ierr;
  DM_ADDA        *dd = (DM_ADDA*)dm->data;
  DM_ADDA        *ddc = (DM_ADDA*)dmc->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmc, DM_CLASSID, 2);
  PetscCheckSameComm(dm, 1, dmc, 2);
  ierr = MatCreate(((PetscObject)dm)->comm, mat);CHKERRQ(ierr);
  ierr = MatSetSizes(*mat, dd->lsize, ddc->lsize, PETSC_DECIDE, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(*mat, mtype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ADDAGetInterpolation"
/*@
   ADDAGetInterpolation - Gets interpolation matrix between two ADDA objects

   Collective on ADDA

   Input Parameter:
+  adda1 - the fine ADDA object
-  adda2 - the second, coarser ADDA object

    Output Parameter:
+  mat - the interpolation matrix
-  vec - the scaling (optional)

   Level: developer

.keywords: distributed array, interpolation

.seealso: DMGetInterpolation()
@*/
PetscErrorCode PETSCDM_DLLEXPORT ADDAGetInterpolation(DM dm1,DM dm2,Mat *mat,Vec *vec) 
{
  PetscFunctionBegin;
  SETERRQ(((PetscObject)dm1)->comm,PETSC_ERR_SUP, "Not implemented yet");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ADDARefine"
/*@
   ADDARefine - Refines a distributed array.

   Collective on ADDA

   Input Parameter:
+  adda - the distributed array to refine
-  comm - the communicator to contain the new ADDA object (or PETSC_NULL)

   Output Parameter:
.  addaf - the refined ADDA

   Level: developer

.keywords: distributed array, refine

.seealso: DMRefine()
@*/
PetscErrorCode PETSCDM_DLLEXPORT ADDARefine(DM dm, MPI_Comm comm, DM *dmf) 
{
  PetscFunctionBegin;
  SETERRQ(((PetscObject)dm)->comm,PETSC_ERR_SUP, "Not implemented yet");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ADDACoarsen"
/*@
   ADDACoarsen - Coarsens a distributed array.

   Collective on ADDA

   Input Parameter:
+  adda - the distributed array to coarsen
-  comm - the communicator to contain the new ADDA object (or PETSC_NULL)

   Output Parameter:
.  addac - the coarsened ADDA

   Level: developer

.keywords: distributed array, coarsen

.seealso: DMCoarsen()
@*/
PetscErrorCode PETSCDM_DLLEXPORT ADDACoarsen(DM dm, MPI_Comm comm,DM *dmc)
{
  PetscErrorCode ierr;
  PetscInt       *nodesc;
  PetscInt       dofc;
  PetscInt       i;
  DM_ADDA        *dd = (DM_ADDA*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(dmc, 3);
  ierr = PetscMalloc(dd->dim*sizeof(PetscInt), &nodesc);CHKERRQ(ierr);
  for(i=0; i<dd->dim; i++) {
    nodesc[i] = (dd->nodes[i] % dd->refine[i]) ? dd->nodes[i] / dd->refine[i] + 1 : dd->nodes[i] / dd->refine[i];
  }
  dofc = (dd->dof % dd->dofrefine) ? dd->dof / dd->dofrefine + 1 : dd->dof / dd->dofrefine;
  ierr = ADDACreate(((PetscObject)dm)->comm, dd->dim, nodesc, dd->procs, dofc, dd->periodic, dmc);CHKERRQ(ierr);
  ierr = PetscFree(nodesc);CHKERRQ(ierr);
  /* copy refinement factors */
  ierr = ADDASetRefinement(*dmc, dd->refine, dd->dofrefine);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ADDAGetInjection"
/*@
   ADDAGetInjection - Gets injection between distributed arrays.

   Collective on ADDA

   Input Parameter:
+  adda1 - the fine ADDA object
-  adda2 - the second, coarser ADDA object

    Output Parameter:
.  ctx - the injection

   Level: developer

.keywords: distributed array, injection

.seealso: DMGetInjection()
@*/
PetscErrorCode PETSCDM_DLLEXPORT ADDAGetInjection(DM dm1,DM dm2, VecScatter *ctx)
{
  PetscFunctionBegin;
  SETERRQ(((PetscObject)dm1)->comm,PETSC_ERR_SUP, "Not implemented yet");
  PetscFunctionReturn(0);
}

/*@C
  ADDAHCiterStartup - performs the first check for an iteration through a hypercube
  lc, uc, idx all have to be valid arrays of size dim
  This function sets idx to lc and then checks, whether the lower corner (lc) is less
  than thre upper corner (uc). If lc "<=" uc in all coordinates, it returns PETSC_TRUE,
  and PETSC_FALSE otherwise.
  
  Input Parameters:
+ dim - the number of dimension
. lc - the "lower" corner
- uc - the "upper" corner

  Output Parameters:
. idx - the index that this function increases

  Developer Notes: This code is crap! You cannot return a value and NO ERROR code in PETSc!

  Level: developer
@*/
PetscBool  ADDAHCiterStartup(const PetscInt dim, const PetscInt *const lc, const PetscInt *const uc, PetscInt *const idx) {
  PetscErrorCode ierr;
  PetscInt i;

  ierr = PetscMemcpy(idx, lc, sizeof(PetscInt)*dim);
  if(ierr) {
    PetscError(PETSC_COMM_SELF,__LINE__,__FUNCT__,__FILE__,__SDIR__,ierr,PETSC_ERROR_REPEAT," ");
    return PETSC_FALSE;
  }
  for(i=0; i<dim; i++) {
    if( lc[i] > uc[i] ) {
      return PETSC_FALSE;
    }
  }
  return PETSC_TRUE;
}

/*@C
  ADDAHCiter - iterates through a hypercube
  lc, uc, idx all have to be valid arrays of size dim
  This function return PETSC_FALSE, if idx exceeds uc, PETSC_TRUE otherwise.
  There are no guarantees on what happens if idx is not in the hypercube
  spanned by lc, uc, this should be checked with ADDAHCiterStartup.
  
  Use this code as follows:
  if( ADDAHCiterStartup(dim, lc, uc, idx) ) {
    do {
      ...
    } while( ADDAHCiter(dim, lc, uc, idx) );
  }
  
  Input Parameters:
+ dim - the number of dimension
. lc - the "lower" corner
- uc - the "upper" corner

  Output Parameters:
. idx - the index that this function increases

  Level: developer
@*/
PetscBool  ADDAHCiter(const PetscInt dim, const PetscInt *const lc, const PetscInt *const uc, PetscInt *const idx) {
  PetscInt i;
  for(i=dim-1; i>=0; i--) {
    idx[i] += 1;
    if( uc[i] > idx[i] ) {
      return PETSC_TRUE;
    } else {
      idx[i] -= uc[i] - lc[i];
    }
  }
  return PETSC_FALSE;
}

#undef __FUNCT__  
#define __FUNCT__ "ADDAGetAggregates"
/*@C
   ADDAGetAggregates - Gets the aggregates that map between 
   grids associated with two ADDAs.

   Collective on ADDA

   Input Parameters:
+  addac - the coarse grid ADDA
-  addaf - the fine grid ADDA

   Output Parameters:
.  rest - the restriction matrix (transpose of the projection matrix)

   Level: intermediate

.keywords: interpolation, restriction, multigrid 

.seealso: ADDARefine(), ADDAGetInjection(), ADDAGetInterpolation()
@*/
PetscErrorCode PETSCDM_DLLEXPORT ADDAGetAggregates(DM dmc,DM dmf,Mat *rest)
{
  PetscErrorCode ierr=0;
  PetscInt       i;
  PetscInt       dim;
  PetscInt       dofc, doff;
  PetscInt       *lcs_c, *lce_c;
  PetscInt       *lcs_f, *lce_f;
  PetscInt       *fgs, *fge;
  PetscInt       fgdofs, fgdofe;
  ADDAIdx        iter_c, iter_f;
  PetscInt       max_agg_size;
  PetscMPIInt    comm_size;
  ADDAIdx        *fine_nodes;
  PetscInt       fn_idx;
  PetscScalar    *one_vec;
  DM_ADDA        *ddc = (DM_ADDA*)dmc->data;
  DM_ADDA        *ddf = (DM_ADDA*)dmf->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmc, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dmf, DM_CLASSID, 2);
  PetscValidPointer(rest,3);
  if (ddc->dim != ddf->dim) SETERRQ2(((PetscObject)dmf)->comm,PETSC_ERR_ARG_INCOMP,"Dimensions of ADDA do not match %D %D", ddc->dim, ddf->dim);CHKERRQ(ierr);
/*   if (dmc->dof != dmf->dof) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"DOF of ADDA do not match %D %D", dmc->dof, dmf->dof);CHKERRQ(ierr); */
  dim = ddc->dim;
  dofc = ddc->dof;
  doff = ddf->dof;

  ierr = ADDAGetCorners(dmc, &lcs_c, &lce_c);CHKERRQ(ierr);
  ierr = ADDAGetCorners(dmf, &lcs_f, &lce_f);CHKERRQ(ierr);
  
  /* compute maximum size of aggregate */
  max_agg_size = 1;
  for(i=0; i<dim; i++) {
    max_agg_size *= ddf->nodes[i] / ddc->nodes[i] + 1;
  }
  max_agg_size *= doff / dofc + 1;

  /* create the matrix that will contain the restriction operator */
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&comm_size);CHKERRQ(ierr);

  /* construct matrix */
  if( comm_size == 1 ) {
    ierr = ADDAGetMatrixNS(dmc, dmf, MATSEQAIJ, rest);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(*rest, max_agg_size, PETSC_NULL);CHKERRQ(ierr);
  } else {
    ierr = ADDAGetMatrixNS(dmc, dmf, MATMPIAIJ, rest);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(*rest, max_agg_size, PETSC_NULL, max_agg_size, PETSC_NULL);CHKERRQ(ierr);
  }
  /* store nodes in the fine grid here */
  ierr = PetscMalloc(sizeof(ADDAIdx)*max_agg_size, &fine_nodes);CHKERRQ(ierr);
  /* these are the values to set to, a collection of 1's */
  ierr = PetscMalloc(sizeof(PetscScalar)*max_agg_size, &one_vec);CHKERRQ(ierr);
  /* initialize */
  for(i=0; i<max_agg_size; i++) {
    ierr = PetscMalloc(sizeof(PetscInt)*dim, &(fine_nodes[i].x));CHKERRQ(ierr);
    one_vec[i] = 1.0;
  }

  /* get iterators */
  ierr = PetscMalloc(sizeof(PetscInt)*dim, &(iter_c.x));CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*dim, &(iter_f.x));CHKERRQ(ierr);

  /* the fine grid node corner for each coarse grid node */
  ierr = PetscMalloc(sizeof(PetscInt)*dim, &fgs);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*dim, &fge);CHKERRQ(ierr);

  /* loop over all coarse nodes */
  ierr = PetscMemcpy(iter_c.x, lcs_c, sizeof(PetscInt)*dim);CHKERRQ(ierr);
  if( ADDAHCiterStartup(dim, lcs_c, lce_c, iter_c.x) ) {
    do {
      /* find corresponding fine grid nodes */
      for(i=0; i<dim; i++) {
	fgs[i] = iter_c.x[i]*ddf->nodes[i]/ddc->nodes[i];
	fge[i] = PetscMin((iter_c.x[i]+1)*ddf->nodes[i]/ddc->nodes[i], ddf->nodes[i]);
      }
      /* treat all dof of the coarse grid */
      for(iter_c.d=0; iter_c.d<dofc; iter_c.d++) {
	/* find corresponding fine grid dof's */
	fgdofs = iter_c.d*doff/dofc;
	fgdofe = PetscMin((iter_c.d+1)*doff/dofc, doff);
	/* we now know the "box" of all the fine grid nodes that are mapped to one coarse grid node */
	fn_idx = 0;
	/* loop over those corresponding fine grid nodes */
	if( ADDAHCiterStartup(dim, fgs, fge, iter_f.x) ) {
	  do {
	    /* loop over all corresponding fine grid dof */
	    for(iter_f.d=fgdofs; iter_f.d<fgdofe; iter_f.d++) {
	      ierr = PetscMemcpy(fine_nodes[fn_idx].x, iter_f.x, sizeof(PetscInt)*dim);CHKERRQ(ierr);
	      fine_nodes[fn_idx].d = iter_f.d;
	      fn_idx++;
	    }
	  } while( ADDAHCiter(dim, fgs, fge, iter_f.x) );
	}
	/* add all these points to one aggregate */
	ierr = ADDAMatSetValues(*rest, dmc, 1, &iter_c, dmf, fn_idx, fine_nodes, one_vec, INSERT_VALUES);CHKERRQ(ierr);
      }
    } while( ADDAHCiter(dim, lcs_c, lce_c, iter_c.x) );
  }

  /* free memory */
  ierr = PetscFree(fgs);CHKERRQ(ierr);
  ierr = PetscFree(fge);CHKERRQ(ierr);
  ierr = PetscFree(iter_c.x);CHKERRQ(ierr);
  ierr = PetscFree(iter_f.x);CHKERRQ(ierr);
  ierr = PetscFree(lcs_c);CHKERRQ(ierr);
  ierr = PetscFree(lce_c);CHKERRQ(ierr);
  ierr = PetscFree(lcs_f);CHKERRQ(ierr);
  ierr = PetscFree(lce_f);CHKERRQ(ierr);
  ierr = PetscFree(one_vec);CHKERRQ(ierr);
  for(i=0; i<max_agg_size; i++) {
    ierr = PetscFree(fine_nodes[i].x);CHKERRQ(ierr);
  }
  ierr = PetscFree(fine_nodes);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*rest, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*rest, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ADDASetRefinement"
/*@
   ADDASetRefinement - Sets the refinement factors of the distributed arrays.

   Collective on ADDA

   Input Parameter:
+  adda - the ADDA object
.  refine - array of refinement factors
-  dofrefine - the refinement factor for the dof, usually just 1

   Level: developer

.keywords: distributed array, refinement
@*/
PetscErrorCode PETSCDM_DLLEXPORT ADDASetRefinement(DM dm, PetscInt *refine, PetscInt dofrefine) 
{
  DM_ADDA        *dd = (DM_ADDA*)dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(refine,3);
  ierr = PetscMemcpy(dd->refine, refine, dd->dim*sizeof(PetscInt));CHKERRQ(ierr);
  dd->dofrefine = dofrefine;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ADDAGetCorners"
/*@
   ADDAGetCorners - Gets the corners of the local area

   Not Collective

   Input Parameter:
.  adda - the ADDA object

   Output Parameter:
+  lcorner - the "lower" corner
-  ucorner - the "upper" corner

   Both lcorner and ucorner are allocated by this procedure and will point to an
   array of size dd->dim.

   Level: beginner

.keywords: distributed array, refinement
@*/
PetscErrorCode PETSCDM_DLLEXPORT ADDAGetCorners(DM dm, PetscInt **lcorner, PetscInt **ucorner) 
{
  DM_ADDA        *dd = (DM_ADDA*)dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(lcorner,2);
  PetscValidPointer(ucorner,3);
  ierr = PetscMalloc(dd->dim*sizeof(PetscInt), lcorner);CHKERRQ(ierr);
  ierr = PetscMalloc(dd->dim*sizeof(PetscInt), ucorner);CHKERRQ(ierr);
  ierr = PetscMemcpy(*lcorner, dd->lcs, dd->dim*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(*ucorner, dd->lce, dd->dim*sizeof(PetscInt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ADDAGetGhostCorners"
/*@
   ADDAGetGhostCorners - Gets the ghost corners of the local area

   Note Collective

   Input Parameter:
.  adda - the ADDA object

   Output Parameter:
+  lcorner - the "lower" corner of the ghosted area
-  ucorner - the "upper" corner of the ghosted area

   Both lcorner and ucorner are allocated by this procedure and will point to an
   array of size dd->dim.

   Level: beginner

.keywords: distributed array, refinement
@*/
PetscErrorCode PETSCDM_DLLEXPORT ADDAGetGhostCorners(DM dm, PetscInt **lcorner, PetscInt **ucorner) 
{
  DM_ADDA        *dd = (DM_ADDA*)dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(lcorner,2);
  PetscValidPointer(ucorner,3);
  ierr = PetscMalloc(dd->dim*sizeof(PetscInt), lcorner);CHKERRQ(ierr);
  ierr = PetscMalloc(dd->dim*sizeof(PetscInt), ucorner);CHKERRQ(ierr);
  ierr = PetscMemcpy(*lcorner, dd->lgs, dd->dim*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(*ucorner, dd->lge, dd->dim*sizeof(PetscInt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



#undef __FUNCT__  
#define __FUNCT__ "ADDAMatSetValues"
/*@C 
   ADDAMatSetValues - Inserts or adds a block of values into a matrix. The values
   are indexed geometrically with the help of the ADDA data structure.
   These values may be cached, so MatAssemblyBegin() and MatAssemblyEnd() 
   MUST be called after all calls to ADDAMatSetValues() have been completed.

   Not Collective

   Input Parameters:
+  mat - the matrix
.  addam - the ADDA geometry information for the rows
.  m - the number of rows
.  idxm - the row indices, each of the a proper ADDAIdx
+  addan - the ADDA geometry information for the columns
.  n - the number of columns
.  idxn - the column indices, each of the a proper ADDAIdx
.  v - a logically two-dimensional array of values of size m*n
-  addv - either ADD_VALUES or INSERT_VALUES, where
   ADD_VALUES adds values to any existing entries, and
   INSERT_VALUES replaces existing entries with new values

   Notes:
   By default the values, v, are row-oriented and unsorted.
   See MatSetOption() for other options.

   Calls to ADDAMatSetValues() (and MatSetValues()) with the INSERT_VALUES and ADD_VALUES 
   options cannot be mixed without intervening calls to the assembly
   routines.

   Efficiency Alert:
   The routine ADDAMatSetValuesBlocked() may offer much better efficiency
   for users of block sparse formats (MATSEQBAIJ and MATMPIBAIJ).

   Level: beginner

   Concepts: matrices^putting entries in

.seealso: MatSetOption(), MatAssemblyBegin(), MatAssemblyEnd(), MatSetValues(), ADDAMatSetValuesBlocked(),
          InsertMode, INSERT_VALUES, ADD_VALUES
@*/
PetscErrorCode PETSCDM_DLLEXPORT ADDAMatSetValues(Mat mat, DM dmm, PetscInt m, const ADDAIdx idxm[],DM dmn, PetscInt n, const ADDAIdx idxn[],
						  const PetscScalar v[], InsertMode addv) 
{
  DM_ADDA        *ddm = (DM_ADDA*)dmm->data;
  DM_ADDA        *ddn = (DM_ADDA*)dmn->data;
  PetscErrorCode ierr;
  PetscInt       *nodemult;
  PetscInt       i, j;
  PetscInt       *matidxm, *matidxn;
  PetscInt       *x, d;
  PetscInt       idx;

  PetscFunctionBegin;
  /* find correct multiplying factors */
  ierr = PetscMalloc(ddm->dim*sizeof(PetscInt), &nodemult);CHKERRQ(ierr);
  nodemult[ddm->dim-1] = 1;
  for(j=ddm->dim-2; j>=0; j--) {
    nodemult[j] = nodemult[j+1]*(ddm->nodes[j+1]);
  }
  /* convert each coordinate in idxm to the matrix row index */
  ierr = PetscMalloc(m*sizeof(PetscInt), &matidxm);CHKERRQ(ierr);
  for(i=0; i<m; i++) {
    x = idxm[i].x; d = idxm[i].d;
    idx = 0;
    for(j=ddm->dim-1; j>=0; j--) {
      if( x[j] < 0 ) { /* "left", "below", etc. of boundary */
	if( ddm->periodic[j] ) { /* periodic wraps around */
	  x[j] += ddm->nodes[j];
	} else { /* non-periodic get discarded */
	  matidxm[i] = -1; /* entries with -1 are ignored by MatSetValues() */
	  goto endofloop_m;
	}
      }
      if( x[j] >= ddm->nodes[j] ) { /* "right", "above", etc. of boundary */
	if( ddm->periodic[j] ) { /* periodic wraps around */
	  x[j] -= ddm->nodes[j];
	} else { /* non-periodic get discarded */
	  matidxm[i] = -1; /* entries with -1 are ignored by MatSetValues() */
	  goto endofloop_m;
	}
      }
      idx += x[j]*nodemult[j];
    }
    matidxm[i] = idx*(ddm->dof) + d;
  endofloop_m:
    ;
  }
  ierr = PetscFree(nodemult);CHKERRQ(ierr);

  /* find correct multiplying factors */
  ierr = PetscMalloc(ddn->dim*sizeof(PetscInt), &nodemult);CHKERRQ(ierr);
  nodemult[ddn->dim-1] = 1;
  for(j=ddn->dim-2; j>=0; j--) {
    nodemult[j] = nodemult[j+1]*(ddn->nodes[j+1]);
  }
  /* convert each coordinate in idxn to the matrix colum index */
  ierr = PetscMalloc(n*sizeof(PetscInt), &matidxn);CHKERRQ(ierr);
  for(i=0; i<n; i++) {
    x = idxn[i].x; d = idxn[i].d;
    idx = 0;
    for(j=ddn->dim-1; j>=0; j--) {
      if( x[j] < 0 ) { /* "left", "below", etc. of boundary */
	if( ddn->periodic[j] ) { /* periodic wraps around */
	  x[j] += ddn->nodes[j];
	} else { /* non-periodic get discarded */
	  matidxn[i] = -1; /* entries with -1 are ignored by MatSetValues() */
	  goto endofloop_n;
	}
      }
      if( x[j] >= ddn->nodes[j] ) { /* "right", "above", etc. of boundary */
	if( ddn->periodic[j] ) { /* periodic wraps around */
	  x[j] -= ddn->nodes[j];
	} else { /* non-periodic get discarded */
	  matidxn[i] = -1; /* entries with -1 are ignored by MatSetValues() */
	  goto endofloop_n;
	}
      }
      idx += x[j]*nodemult[j];
    }
    matidxn[i] = idx*(ddn->dof) + d;
  endofloop_n:
    ;
  }
  /* call original MatSetValues() */
  ierr = MatSetValues(mat, m, matidxm, n, matidxn, v, addv);CHKERRQ(ierr);
  /* clean up */
  ierr = PetscFree(nodemult);CHKERRQ(ierr);
  ierr = PetscFree(matidxm);CHKERRQ(ierr);
  ierr = PetscFree(matidxn);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

