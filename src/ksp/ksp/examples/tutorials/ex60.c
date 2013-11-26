static char help[] = "\
Solves the constant-coefficient 1D Heat equation with an Implicit   \n\
Runge-Kutta method using MatTAIJ.                                   \n\
                                                                    \n\
    du      d^2 u                                                   \n\
    --  = a ----- ; 0 <= x <= 1;                                    \n\
    dt      dx^2                                                    \n\
                                                                    \n\
  with periodic boundary conditions                                 \n\
                                                                    \n\
2nd order central discretization in space:                          \n\
                                                                    \n\
   [ d^2 u ]     u_{i+1} - 2u_i + u_{i-1}                           \n\
   [ ----- ]  =  ------------------------                           \n\
   [ dx^2  ]i              h^2                                      \n\
                                                                    \n\
    i = grid index;    h = x_{i+1}-x_i (Uniform)                    \n\
    0 <= i < n         h = 1.0/n                                    \n\
                                                                    \n\
Thus,                                                               \n\
                                                                    \n\
   du                                                               \n\
   --  = Ju;  J = (a/h^2) tridiagonal(1,-2,1)_n                     \n\
   dt                                                               \n\
                                                                    \n\
Implicit Runge-Kutta method:                                        \n\
                                                                    \n\
  U^(k)   = u^n + dt \\sum_i a_{ki} JU^{i}                          \n\
  u^{n+1} = u^n + dt \\sum_i b_i JU^{i}                             \n\
                                                                    \n\
  i = 1,...,s (s -> number of stages)                               \n\
                                                                    \n\
At each time step, we solve                                         \n\
                                                                    \n\
 [  1                                  ]     1                      \n\
 [ -- I \\otimes A^{-1} - J \\otimes I ] U = -- u^n \\otimes A^{-1} \n\
 [ dt                                  ]     dt                     \n\
                                                                    \n\
  where A is the Butcher tableaux of the implicit                   \n\
  Runge-Kutta method,                                               \n\
                                                                    \n\
with MATTAIJ and KSP.                                               \n\
                                                                    \n\
Available IRK Methods:                                              \n\
  2       4th-order, 2-stage Gauss method                           \n\
                                                                    \n";

/*T
  Concepts: MATTAIJ
  Concepts: MAT
  Concepts: KSP
T*/

/*
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
  petscsys.h      - base PETSc routines   
  petscvec.h      - vectors
  petscmat.h      - matrices
  petscis.h       - index sets            
  petscviewer.h   - viewers               
  petscpc.h       - preconditioners
*/
#include <petscksp.h>
#include <petscdt.h>

/* define the IRK methods available */
#define IRKGAUSS24    "gauss24"
#define IRKGAUSS      "gauss"

typedef enum {
  PHYSICS_DIFFUSION,
  PHYSICS_ADVECTION
} PhysicsType;
const char *const PhysicsTypes[] = {"DIFFUSION","ADVECTION","PhysicsType","PHYSICS_",NULL};

typedef struct __context__ {
  PetscReal     a;              /* diffusion coefficient      */
  PetscReal     xmin,xmax;      /* domain bounds              */
  PetscInt      imax;           /* number of grid points      */
  PetscInt      niter;          /* number of time iterations  */
  PetscReal     dt;             /* time step size             */
  PhysicsType   physics_type;
} UserContext;

static PetscErrorCode ExactSolution(Vec,void*,PetscReal);
static PetscErrorCode RKCreate_Gauss24(PetscInt,PetscScalar**,PetscScalar**,PetscReal**);
static PetscErrorCode RKCreate_Gauss(PetscInt,PetscScalar**,PetscScalar**,PetscReal**);
static PetscErrorCode Assemble_AdvDiff(MPI_Comm,UserContext*,Mat*);

#include <petsc-private/kernels/blockinvert.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  PetscErrorCode    ierr;
  Vec               u,uex,rhs,z;
  UserContext       ctxt;
  PetscInt          nstages,is,ie,matis,matie,*ix,*ix2;
  PetscInt          n,i,s,t,total_its;
  PetscScalar       *A,*B,*At,*b,*zvals,one = 1.0;
  PetscReal         *c,err,time;
  Mat               Identity,J,TA,SC,R;
  KSP               ksp;
  PetscFunctionList IRKList = NULL;
  char              irktype[256] = IRKGAUSS;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscFunctionListAdd(&IRKList,IRKGAUSS24,RKCreate_Gauss24);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&IRKList,IRKGAUSS,RKCreate_Gauss);CHKERRQ(ierr);

  /* default value */
  ctxt.a       = 1.0;
  ctxt.xmin    = 0.0;
  ctxt.xmax    = 1.0;
  ctxt.imax    = 20;
  ctxt.niter   = 0;
  ctxt.dt      = 0.0;
  ctxt.physics_type = PHYSICS_DIFFUSION;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"IRK options","");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-a","diffusion coefficient","<1.0>",ctxt.a,
                          &ctxt.a,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt ("-imax","grid size","<20>",ctxt.imax,
                          &ctxt.imax,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-xmin","xmin","<0.0>",ctxt.xmin,
                          &ctxt.xmin,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-xmax","xmax","<1.0>",ctxt.xmax,
                          &ctxt.xmax,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt ("-niter","number of time steps","<0>",ctxt.niter,
                          &ctxt.niter,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dt","time step size","<0.0>",ctxt.dt,
                          &ctxt.dt,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsList("-irk_type","IRK method family","",
                          IRKList,irktype,irktype,sizeof(irktype),NULL);CHKERRQ(ierr);
  nstages = 2;
  ierr = PetscOptionsInt ("-irk_nstages","Number of stages in IRK method","",
                          nstages,&nstages,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-physics_type","Type of process to discretize","",
                          PhysicsTypes,(PetscEnum)ctxt.physics_type,(PetscEnum*)&ctxt.physics_type,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* allocate and initialize solution vector and exact solution */
  ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
  ierr = VecSetSizes(u,PETSC_DECIDE,ctxt.imax);CHKERRQ(ierr);
  ierr = VecSetFromOptions(u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&uex);CHKERRQ(ierr);
  /* initial solution */
  ierr = ExactSolution(u  ,&ctxt,0.0);CHKERRQ(ierr);                 
  /* exact   solution */
  ierr = ExactSolution(uex,&ctxt,ctxt.dt*ctxt.niter);CHKERRQ(ierr);

  {                             /* Create A,b,c */
    PetscErrorCode (*irkcreate)(PetscInt,PetscScalar**,PetscScalar**,PetscReal**);
    ierr = PetscFunctionListFind(IRKList,irktype,&irkcreate);CHKERRQ(ierr);
    ierr = (*irkcreate)(nstages,&A,&b,&c);CHKERRQ(ierr);
  }
  {                             /* Invert A */
    PetscInt *pivots;
    PetscScalar *work;
    ierr = PetscMalloc2(nstages,PetscInt,&pivots,nstages,PetscScalar,&work);CHKERRQ(ierr);
    ierr = PetscKernel_A_gets_inverse_A(nstages,A,pivots,work);CHKERRQ(ierr);
    ierr = PetscFree2(pivots,work);CHKERRQ(ierr);
  }
  /* Scale (1/dt)*A^{-1} and (1/dt)*b */
  for (s=0; s<nstages*nstages; s++) A[s] *= 1.0/ctxt.dt;
  for (s=0; s<nstages; s++) b[s] *= (-ctxt.dt);

  /* Compute row sums At and identity B */
  ierr = PetscMalloc2(nstages,PetscScalar,&At,PetscSqr(nstages),PetscScalar,&B);CHKERRQ(ierr);
  for (s=0; s<nstages; s++) {
    At[s] = 0;
    for (t=0; t<nstages; t++) {
      At[s] += A[s+nstages*t];      /* Row sums of  */
      B[s+nstages*t] = 1.*(s == t); /* identity */
    }
  }

  /* allocate and calculate the (-J) matrix */
  switch (ctxt.physics_type) {
  case PHYSICS_ADVECTION:
  case PHYSICS_DIFFUSION:
    ierr = Assemble_AdvDiff(PETSC_COMM_WORLD,&ctxt,&J);CHKERRQ(ierr);
  }
  ierr = MatCreate(PETSC_COMM_WORLD,&Identity);CHKERRQ(ierr);
  ierr = MatSetType(Identity,MATAIJ);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(J,&matis,&matie);CHKERRQ(ierr);
  ierr = MatSetSizes(Identity,matie-matis,matie-matis,ctxt.imax,ctxt.imax);CHKERRQ(ierr);
  ierr = MatSetUp(Identity);CHKERRQ(ierr);
  for (i=matis; i<matie; i++) {
    ierr= MatSetValues(Identity,1,&i,1,&i,&one,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(Identity,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (Identity,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Create the TAIJ matrix for solving the stages */
  ierr = MatCreateTAIJ(J,nstages,nstages,A,B,&TA);CHKERRQ(ierr);

  /* Create the TAIJ matrix for step completion */
  ierr = MatCreateTAIJ(J,1,nstages,NULL,b,&SC);CHKERRQ(ierr);

  /* Create the TAIJ matrix to create the R for solving the stages */
  ierr = MatCreateTAIJ(Identity,nstages,1,NULL,At,&R);CHKERRQ(ierr);

  /* Create and set options for KSP */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,TA,TA,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /* Allocate work and right-hand-side vectors */
  ierr = VecCreate(PETSC_COMM_WORLD,&z);CHKERRQ(ierr);
  ierr = VecSetFromOptions(z);CHKERRQ(ierr);
  ierr = VecSetSizes(z,PETSC_DECIDE,ctxt.imax*nstages);CHKERRQ(ierr);
  ierr = VecDuplicate(z,&rhs);

  ierr = VecGetOwnershipRange(u,&is,&ie);CHKERRQ(ierr);
  ierr = PetscMalloc3(nstages,PetscInt,&ix,
                      nstages,PetscScalar,&zvals,
                      ie-is,PetscInt,&ix2);CHKERRQ(ierr);
  /* iterate in time */
  for (n=0,time=0.,total_its=0; n<ctxt.niter; n++) {
    PetscInt its;

    /* compute and set the right hand side */
    ierr = MatMult(R,u,rhs);CHKERRQ(ierr);

    /* Solve the system */
    ierr = KSPSolve(ksp,rhs,z);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
    total_its += its;

    /* Update the solution */
    ierr = MatMultAdd(SC,z,u,u);CHKERRQ(ierr);

    /* time step complete */
    time += ctxt.dt;
  }
  PetscFree3(ix,ix2,zvals);

  /* Deallocate work and right-hand-side vectors */
  ierr = VecDestroy(&z);CHKERRQ(ierr);
  ierr = VecDestroy(&rhs);CHKERRQ(ierr);

  /* Calculate error in final solution */
  ierr = VecAYPX(uex,-1.0,u);
  ierr = VecNorm(uex,NORM_2,&err);
  err  = PetscSqrtReal(err*err/((PetscReal)ctxt.imax));
  ierr = PetscPrintf(PETSC_COMM_WORLD,"L2 norm of the numerical error = %G (time=%G)\n",err,time);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of time steps: %D (%D Krylov iterations)\n",ctxt.niter,total_its);CHKERRQ(ierr);

  /* Free up memory */
  ierr = KSPDestroy(&ksp);      CHKERRQ(ierr);
  ierr = MatDestroy(&TA);       CHKERRQ(ierr);
  ierr = MatDestroy(&SC);       CHKERRQ(ierr);
  ierr = MatDestroy(&R);        CHKERRQ(ierr);
  ierr = MatDestroy(&J);        CHKERRQ(ierr);
  ierr = MatDestroy(&Identity); CHKERRQ(ierr);
  ierr = PetscFree3(A,b,c);     CHKERRQ(ierr);
  ierr = PetscFree2(At,B);      CHKERRQ(ierr);
  ierr = VecDestroy(&uex);      CHKERRQ(ierr);
  ierr = VecDestroy(&u);        CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&IRKList);CHKERRQ(ierr);

  PetscFinalize();
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "ExactSolution"
PetscErrorCode ExactSolution(Vec u,void *c,PetscReal t)
{
  UserContext     *ctxt = (UserContext*) c;
  PetscErrorCode  ierr;
  PetscInt        i,is,ie;
  PetscScalar     *uarr;
  PetscReal       x,dx,a=ctxt->a,pi=PETSC_PI;

  PetscFunctionBegin;
  dx = (ctxt->xmax - ctxt->xmin)/((PetscReal) ctxt->imax);
  ierr = VecGetOwnershipRange(u,&is,&ie);CHKERRQ(ierr);
  ierr = VecGetArray(u,&uarr);CHKERRQ(ierr);
  for(i=is; i<ie; i++) {
    x          = i * dx;
    switch (ctxt->physics_type) {
    case PHYSICS_DIFFUSION:
      uarr[i-is] = PetscExpScalar(-4.0*pi*pi*a*t)*PetscSinScalar(2*pi*x);
      break;
    case PHYSICS_ADVECTION:
      uarr[i-is] = PetscSinScalar(2*pi*(x - a*t));
      break;
    default: SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for physics type %s",PhysicsTypes[ctxt->physics_type]);
    }
  }
  ierr = VecRestoreArray(u,&uarr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RKCreate_Gauss24"
/* Arrays should be freed with PetscFree3(A,b,c) */
static PetscErrorCode RKCreate_Gauss24(PetscInt nstages,PetscScalar **gauss_A,PetscScalar **gauss_b,PetscReal **gauss_c)
{
  PetscErrorCode    ierr;
  PetscScalar       *A,*b;
  PetscReal         *c;

  PetscFunctionBegin;
  if (nstages != 2) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Method 'gauss24' does not support %D stages, use -irk_nstages 2",nstages);
  ierr = PetscMalloc3(PetscSqr(nstages),PetscReal,&A,nstages,PetscReal,&b,nstages,PetscReal,&c);CHKERRQ(ierr);
  A[0] = 0.25; A[2] = (3.0-2.0*1.7320508075688772)/12.0;
  A[1] = (3.0+2.0*1.7320508075688772)/12.0; A[3] = 0.25;
  b[0] = 0.5;                        b[1] = 0.5;
  c[0] = 0.5 - PetscSqrtReal(3)/6;   c[1] = 0.5 + PetscSqrtReal(3)/6;

  *gauss_A = A;
  *gauss_b = b;
  *gauss_c = c;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RKCreate_Gauss"
/* Arrays should be freed with PetscFree3(A,b,c) */
static PetscErrorCode RKCreate_Gauss(PetscInt nstages,PetscScalar **gauss_A,PetscScalar **gauss_b,PetscReal **gauss_c)
{
  PetscErrorCode    ierr;
  PetscScalar       *A,*G0,*G1;
  PetscReal         *b,*c;
  PetscInt          i,j;
  Mat               G0mat,G1mat,Amat;

  PetscFunctionBegin;
  ierr = PetscMalloc3(PetscSqr(nstages),PetscScalar,&A,nstages,PetscScalar,gauss_b,nstages,PetscReal,&c);CHKERRQ(ierr);
  ierr = PetscMalloc3(nstages,PetscReal,&b,PetscSqr(nstages),PetscScalar,&G0,PetscSqr(nstages),PetscScalar,&G1);CHKERRQ(ierr);
  ierr = PetscDTGaussQuadrature(nstages,0.,1.,c,b);CHKERRQ(ierr);
  for (i=0; i<nstages; i++) (*gauss_b)[i] = b[i]; /* copy to possibly-complex array */

  /* A^T = G0^{-1} G1 */
  for (i=0; i<nstages; i++) {
    for (j=0; j<nstages; j++) {
      G0[i*nstages+j] = PetscPowRealInt(c[i],j);
      G1[i*nstages+j] = PetscPowRealInt(c[i],j+1)/(j+1);
    }
  }
  /* The arrays above are row-aligned, but we create dense matrices as the transpose */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,nstages,nstages,G0,&G0mat);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,nstages,nstages,G1,&G1mat);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,nstages,nstages,A,&Amat);CHKERRQ(ierr);
  ierr = MatLUFactor(G0mat,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = MatMatSolve(G0mat,G1mat,Amat);CHKERRQ(ierr);
  ierr = MatTranspose(Amat,MAT_REUSE_MATRIX,&Amat);CHKERRQ(ierr);

  ierr = MatDestroy(&G0mat);CHKERRQ(ierr);
  ierr = MatDestroy(&G1mat);CHKERRQ(ierr);
  ierr = MatDestroy(&Amat);CHKERRQ(ierr);
  ierr = PetscFree3(b,G0,G1);CHKERRQ(ierr);
  *gauss_A = A;
  *gauss_c = c;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Assemble_AdvDiff"
static PetscErrorCode Assemble_AdvDiff(MPI_Comm comm,UserContext *user,Mat *J)
{
  PetscErrorCode ierr;
  PetscInt       matis,matie,i;
  PetscReal      dx,dx2;
  PetscFunctionBegin;
  dx = (user->xmax - user->xmin)/((PetscReal)user->imax); dx2 = dx*dx;
  ierr = MatCreate(comm,J);CHKERRQ(ierr);
  ierr = MatSetType(*J,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetSizes(*J,PETSC_DECIDE,PETSC_DECIDE,user->imax,user->imax);CHKERRQ(ierr);
  ierr = MatSetUp(*J);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(*J,&matis,&matie);CHKERRQ(ierr);
  for (i=matis; i<matie; i++) {
    PetscScalar values[3];
    PetscInt    col[3];
    switch (user->physics_type) {
    case PHYSICS_DIFFUSION:
      values[0] = -user->a*1.0/dx2;
      values[1] = user->a*2.0/dx2;
      values[2] = -user->a*1.0/dx2;
      break;
    case PHYSICS_ADVECTION:
      values[0] = -user->a*.5/dx;
      values[1] = 0.;
      values[2] = user->a*.5/dx;
      break;
    default: SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for physics type %s",PhysicsTypes[user->physics_type]);
    }
    /* periodic boundaries */
    if (i == 0) {
      col[0] = user->imax-1;
      col[1] = i;
      col[2] = i+1;
    } else if (i == user->imax-1) {
      col[0] = i-1;
      col[1] = i;
      col[2] = 0;
    } else {
      col[0] = i-1;
      col[1] = i;
      col[2] = i+1;
    }
    ierr= MatSetValues(*J,1,&i,3,col,values,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
