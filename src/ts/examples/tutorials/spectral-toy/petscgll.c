
#include "petscgll.h"
#include "petscmat.h"
#include <petscviewer.h>
#include <petscblaslapack.h>
#include <petsc/private/petscimpl.h>

static void qAndLEvaluation(PetscInt n, PetscReal x, PetscReal *q, PetscReal *qp, PetscReal *Ln)
/*
  Compute the polynomial q(x) = L_{N+1}(x) - L_{n-1}(x) and its derivative in
  addition to L_N(x) as these are needed for computing the GLL points via Newton's method.
  Reference: "Implementing Spectral Methods for Partial Differential Equations: Algorithms
  for Scientists and Engineers" by David A. Kopriva.
*/
{
  PetscInt k;

  PetscReal Lnp;
  PetscReal Lnp1, Lnp1p;
  PetscReal Lnm1, Lnm1p;
  PetscReal Lnm2, Lnm2p;

  Lnm1  = 1.0;
  *Ln   = x;
  Lnm1p = 0.0;
  Lnp   = 1.0;

  for (k=2; k<=n; ++k) {
    Lnm2  = Lnm1;
    Lnm1  = *Ln;
    Lnm2p = Lnm1p;
    Lnm1p = Lnp;
    *Ln   = (2.*((PetscReal)k)-1.)/(1.0*((PetscReal)k))*x*Lnm1 - (((PetscReal)k)-1.)/((PetscReal)k)*Lnm2;
    Lnp   = Lnm2p + (2.0*((PetscReal)k)-1.)*Lnm1;
  }
  k     = n+1;
  Lnp1  = (2.*((PetscReal)k)-1.)/(((PetscReal)k))*x*(*Ln) - (((PetscReal)k)-1.)/((PetscReal)k)*Lnm1;
  Lnp1p = Lnm1p + (2.0*((PetscReal)k)-1.)*(*Ln);
  *q    = Lnp1 - Lnm1;
  *qp   = Lnp1p - Lnm1p;
}

#undef __FUNCT__
#define __FUNCT__ "PetscGLLIPCreate"
/*@C
   PetscGLLIPCreate - creates a set of the locations and weights of the Gauss-Lobatto-Legendre (GLL) nodes of a given size
                      on the domain [-1,1]

   Not Collective

   Input Parameter:
+  n - number of grid nodes
-  type - PETSCGLLIP_VIA_LINEARALGEBRA or PETSCGLLIP_VIA_NEWTON

   Output Parameter:
.  gll - the nodes

   Notes: For n > 30  the Newton approach computes duplicate (incorrect) values for some nodes because the initial guess is apparently not
          close enough to the desired solution

   See  http://epubs.siam.org/doi/abs/10.1137/110855442  http://epubs.siam.org/doi/abs/10.1137/120889873 for better ways to compute GLL nodes

   Level: beginner

.seealso: PetscGLLIP, PetscGLLIPDestroy(), PetscGLLIPView()

@*/
PetscErrorCode PetscGLLIPCreate(PetscInt n,PetscGLLIPCreateType type,PetscGLLIP *gll)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc2(n,&gll->nodes,n,&gll->weights);CHKERRQ(ierr);

  if (type == PETSCGLLIP_VIA_LINEARALGEBRA) {
    PetscReal      *M,si;
    PetscBLASInt   bn,lierr;
    PetscScalar    x,z0,z1,z2;
    PetscInt       i,p = n - 1,nn;

    gll->nodes[0]   =-1.0;
    gll->nodes[n-1] = 1.0;
    if (n-2 > 0){
      ierr = PetscMalloc1(n-1,&M);CHKERRQ(ierr);
      for (i=0; i<n-2; i++) {
        si  = ((PetscReal)i)+1.0;
        M[i]=0.5*PetscSqrtReal(si*(si+2.0)/((si+0.5)*(si+1.5)));
      }
      ierr = PetscBLASIntCast(n-2,&bn);CHKERRQ(ierr);
      ierr = PetscMemzero(&gll->nodes[1],bn*sizeof(gll->nodes[1]));CHKERRQ(ierr);
      ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
      PetscRealView(n-2,M,0);
      x=0;
      printf(" bn %d\n", bn);
      PetscStackCallBLAS("LAPACKsteqr",LAPACKsteqr_("N",&bn,&gll->nodes[1],M,&x,&bn,M,&lierr));
      if (lierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in STERF Lapack routine %d",(int)lierr);
      ierr = PetscFPTrapPop();CHKERRQ(ierr);
      ierr = PetscFree(M);CHKERRQ(ierr);
    }
    if ((n-1)%2==0) {
      gll->nodes[(n-1)/2]   = 0.0; /* hard wire to exactly 0.0 since linear algebra produces nonzero */
    }

    gll->weights[0] = gll->weights[p] = 2.0/(((PetscReal)(p))*(((PetscReal)p)+1.0));
    z2 = -1.;                      /* Dummy value to avoid -Wmaybe-initialized */
    for (i=1; i<p; i++) {
      x  = gll->nodes[i];
      z0 = 1.0;
      z1 = x;
      for (nn=1; nn<p; nn++) {
        z2 = x*z1*(2.0*((PetscReal)nn)+1.0)/(((PetscReal)nn)+1.0)-z0*(((PetscReal)nn)/(((PetscReal)nn)+1.0));
        z0 = z1;
        z1 = z2;
      }
      gll->weights[i]=2.0/(((PetscReal)p)*(((PetscReal)p)+1.0)*z2*z2);
    }
  } else {
    PetscInt  j,m;
    PetscReal z1,z,q,qp,Ln;

    if (n > 30) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"PETSCGLLIP_VIA_NEWTON produces incorrect answers for n > 30");
    gll->nodes[0]     = -1.0;
    gll->nodes[n-1]   = 1.0;
    gll->weights[0]   = gll->weights[n-1] = 2./(((PetscReal)n)*(((PetscReal)n)-1.0));;
    m  = (n-1)/2; /* The roots are symmetric, so we only find half of them. */
    for (j=1; j<=m; j++) { /* Loop over the desired roots. */
      z = -1.0*PetscCosReal((PETSC_PI*((PetscReal)j)+0.25)/(((PetscReal)n)-1.0))-(3.0/(8.0*(((PetscReal)n)-1.0)*PETSC_PI))*(1.0/(((PetscReal)j)+0.25));
      /* Starting with the above approximation to the ith root, we enter */
      /* the main loop of refinement by Newton's method.                 */
      do {
        qAndLEvaluation(n-1,z,&q,&qp,&Ln);
        z1 = z;
        z  = z1-q/qp; /* Newton's method. */
      } while (PetscAbs(z-z1) > 10.*PETSC_MACHINE_EPSILON);
      qAndLEvaluation(n-1,z,&q,&qp,&Ln);
      gll->nodes[j]       = z;
      gll->nodes[n-1-j]   = -z;      /* and put in its symmetric counterpart.   */
      gll->weights[j]     = 2.0/(((PetscReal)n)*(((PetscReal)n)-1.)*Ln*Ln);  /* Compute the weight */
      gll->weights[n-1-j] = gll->weights[j];                 /* and its symmetric counterpart. */
    }
    if ((n-1)%2==0) {
      qAndLEvaluation(n-1,0.0,&q,&qp,&Ln);
      gll->nodes[(n-1)/2]   = 0.0;
      gll->weights[(n-1)/2] = 2.0/(((PetscReal)n)*(((PetscReal)n)-1.)*Ln*Ln);
    }
  }
  gll->n = n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscGLLIPDestroy"
/*@C
   PetscGLLIPDestroy - destroys a set of GLL nodes and weights

   Not Collective

   Input Parameter:
.  gll - the nodes

   Level: beginner

.seealso: PetscGLLIP, PetscGLLIPCreate(), PetscGLLIPView()

@*/
PetscErrorCode PetscGLLIPDestroy(PetscGLLIP *gll)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr   = PetscFree2(gll->nodes,gll->weights);CHKERRQ(ierr);
  gll->n = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscGLLIPView"
/*@C
   PetscGLLIPView - views a set of GLL nodes

   Not Collective

   Input Parameter:
+  gll - the nodes
.  viewer - the viewer

   Level: beginner

.seealso: PetscGLLIP, PetscGLLIPCreate(), PetscGLLIPDestroy()

@*/
PetscErrorCode PetscGLLIPView(PetscGLLIP *gll,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscBool         iascii;

  PetscInt          i;

  PetscFunctionBegin;
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(PETSC_COMM_SELF,&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"%D Gauss-Lobatto-Legendre (GLL) nodes and weights\n",gll->n);CHKERRQ(ierr);
    for (i=0; i<gll->n; i++) {
      ierr = PetscViewerASCIIPrintf(viewer,"  %D %16.14e %16.14e\n",i,(double)gll->nodes[i],(double)gll->weights[i]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscGLLIPElementStiffnessCreate"
/*@C
   PetscGLLIPElementStiffnessCreate - computes the stiffness for a single 1d GLL element for the Laplacian

   Not Collective

   Input Parameter:
.  gll - the nodes

   Output Parameter:
.  A - the stiffness element

   Level: beginner

   Notes: Destroy this with PetscGLLIPElementStiffnessDestroy()

   You can access entries in this matrix with AA[i][j] but in memory it is stored in contiguous memory, row oriented (the matrix is symmetric)

.seealso: PetscGLLIP, PetscGLLIPDestroy(), PetscGLLIPView(), PetscGLLIPElementStiffnessDestroy()

@*/
PetscErrorCode PetscGLLIPElementStiffnessCreate(PetscGLLIP *gll,PetscReal ***AA)
{
  PetscReal        **A;
  PetscErrorCode  ierr;
  const PetscReal  *nodes = gll->nodes;
  const PetscInt   n = gll->n, p = gll->n-1;
  PetscReal        z0,z1,z2,x,Lpj,Lpr;
  PetscInt         i,j,nn,r;

  PetscFunctionBegin;
  ierr = PetscMalloc1(n,&A);CHKERRQ(ierr);
  ierr = PetscMalloc1(n*n,&A[0]);CHKERRQ(ierr);
  for (i=1; i<n; i++) A[i] = A[i-1]+n;

  for (j=1; j<p; j++) {
    x  = nodes[j];
    z0 = 1.;
    z1 = x;
    for (nn=1; nn<p; nn++) {
      z2 = x*z1*(2.*((PetscReal)nn)+1.)/(((PetscReal)nn)+1.)-z0*(((PetscReal)nn)/(((PetscReal)nn)+1.));
      z0 = z1;
      z1 = z2;
    }
    Lpj=z2;
    for (r=1; r<p; r++) {
      if (r == j) {
        A[j][j]=2./(3.*(1.-nodes[j]*nodes[j])*Lpj*Lpj);
      } else {
        x  = nodes[r];
        z0 = 1.;
        z1 = x;
        for (nn=1; nn<p; nn++) {
          z2 = x*z1*(2.*((PetscReal)nn)+1.)/(((PetscReal)nn)+1.)-z0*(((PetscReal)nn)/(((PetscReal)nn)+1.));
          z0 = z1;
          z1 = z2;
        }
        Lpr     = z2;
        A[r][j] = 4./(((PetscReal)p)*(((PetscReal)p)+1.)*Lpj*Lpr*(nodes[j]-nodes[r])*(nodes[j]-nodes[r]));
      }
    }
  }
  for (j=1; j<p+1; j++) {
    x  = nodes[j];
    z0 = 1.;
    z1 = x;
    for (nn=1; nn<p; nn++) {
      z2 = x*z1*(2.*((PetscReal)nn)+1.)/(((PetscReal)nn)+1.)-z0*(((PetscReal)nn)/(((PetscReal)nn)+1.));
      z0 = z1;
      z1 = z2;
    }
    Lpj     = z2;
    A[j][0] = 4.*PetscPowRealInt(-1.,p)/(((PetscReal)p)*(((PetscReal)p)+1.)*Lpj*(1.+nodes[j])*(1.+nodes[j]));
    A[0][j] = A[j][0];
  }
  for (j=0; j<p; j++) {
    x  = nodes[j];
    z0 = 1.;
    z1 = x;
    for (nn=1; nn<p; nn++) {
      z2 = x*z1*(2.*((PetscReal)nn)+1.)/(((PetscReal)nn)+1.)-z0*(((PetscReal)nn)/(((PetscReal)nn)+1.));
      z0 = z1;
      z1 = z2;
    }
    Lpj=z2;

    A[p][j] = 4./(((PetscReal)p)*(((PetscReal)p)+1.)*Lpj*(1.-nodes[j])*(1.-nodes[j]));
    A[j][p] = A[p][j];
  }
  A[0][0]=0.5+(((PetscReal)p)*(((PetscReal)p)+1.)-2.)/6.;
  A[p][p]=A[0][0];
  *AA = A;
  PetscFunctionReturn(0);
}
/* 
  Create mass matrix
*/


#undef __FUNCT__
#define __FUNCT__ "PetscGLLIPElementStiffnessDestroy"
/*@C
   PetscGLLIPElementStiffnessDestroy - frees the stiffness for a single 1d GLL element

   Not Collective

   Input Parameter:
+  gll - the nodes
-  A - the stiffness element

   Level: beginner

.seealso: PetscGLLIP, PetscGLLIPDestroy(), PetscGLLIPView(), PetscGLLIPElementStiffnessCreate()

@*/
PetscErrorCode PetscGLLIPElementStiffnessDestroy(PetscGLLIP *gll,PetscReal ***AA)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree((*AA)[0]);CHKERRQ(ierr);
  ierr = PetscFree(*AA);CHKERRQ(ierr);
  *AA  = NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscGLLIPIntegrate"
/*@C
   PetscGLLIPIntegrate - Integrate a function on the GLL points

   Not Collective

   Input Parameter:
+  gll - the nodes
.  f - the function values at the nodes

   Output Parameter:
.  in - the value of the integral

   Level: beginner

.seealso: PetscGLLIP, PetscGLLIPCreate(), PetscGLLIPDestroy()

@*/
PetscErrorCode PetscGLLIPIntegrate(PetscGLLIP *gll,PetscReal *f,PetscReal *in)
{
  PetscInt          i;

  PetscFunctionBegin;
  *in = 0.;
  for (i=0; i<gll->n; i++) {
    *in += f[i]*f[i]*gll->weights[i];
  }
  PetscFunctionReturn(0);
}
