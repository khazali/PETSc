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

/* define the IRK methods available */
#define GAUSS24    "gauss24"

typedef struct __context__ {
  PetscReal     a;              /* diffusion coefficient      */
  PetscReal     xmin,xmax;      /* domain bounds              */
  PetscInt      imax;           /* number of grid points      */
  PetscInt      niter;          /* number of time iterations  */
  PetscReal     dt;             /* time step size             */
  char          irktype[50];    /* irk method                 */
} UserContext;

static PetscErrorCode ExactSolution(Vec,void*,PetscReal);

#include <petsc-private/kernels/blockinvert.h>

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc, char **argv)
{
  PetscErrorCode    ierr;
  Vec               u,uex,rhs,z,w;
  UserContext       ctxt;
  PetscInt          nstages,is,ie,matis,matie,*ix,*ix2;
  PetscInt          n,i,s,t;
  PetscScalar       *A,*b;
  PetscReal         dx,dx2,*zvals,err;
  Mat               J,TA;
  KSP               ksp;

  PetscInitialize(&argc,&argv,(char*)0,help);
  /* default value */
  ctxt.a       = 1.0;
  ctxt.xmin    = 0.0;
  ctxt.xmax    = 1.0;
  ctxt.imax    = 20;
  ctxt.niter   = 0; 
  ctxt.dt      = 0.0;
  ierr = PetscStrcpy(ctxt.irktype,GAUSS24);CHKERRQ(ierr);

  /* Read options */
  ierr = PetscOptionsReal("-a","diffusion coefficient","<1.0>",ctxt.a,
                          &ctxt.a,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt ("-imax","grid size","<20>",ctxt.imax,
                          &ctxt.imax,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-xmin","xmin","<0.0>",ctxt.xmin,
                          &ctxt.xmin,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-xmax","xmax","<1.0>",ctxt.xmax,
                          &ctxt.xmax,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt ("-niter","number of time steps","<0>",ctxt.niter,
                          &ctxt.niter,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dt","time step size","<0.0>",ctxt.dt,
                          &ctxt.dt,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-irktype","IRK method","<2>",ctxt.irktype,
                            ctxt.irktype,50,PETSC_NULL);CHKERRQ(ierr);

  /* allocate and initialize solution vector and exact solution */
  dx = (ctxt.xmax - ctxt.xmin)/((PetscReal) ctxt.imax); dx2 = dx*dx;
  ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
  ierr = VecSetSizes(u,PETSC_DECIDE,ctxt.imax);CHKERRQ(ierr);
  ierr = VecSetFromOptions(u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&uex);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&w);CHKERRQ(ierr);
  /* initial solution */
  ierr = ExactSolution(u  ,&ctxt,0.0);CHKERRQ(ierr);                 
  /* exact   solution */
  ierr = ExactSolution(uex,&ctxt,ctxt.dt*ctxt.niter);CHKERRQ(ierr);

  /* allocate and calculate (1/dt)*A^{-1} and b */
  if (!strcmp(ctxt.irktype,GAUSS24)) {
    nstages = 2;
    ierr = PetscMalloc2(nstages*nstages,PetscScalar,&A,
                        nstages        ,PetscScalar,&b);CHKERRQ(ierr);
    A[0] = 0.25; A[2] = (3.0-2.0*1.7320508075688772)/12.0;
    A[1] = (3.0+2.0*1.7320508075688772)/12.0; A[3] = 0.25;
    b[0] = 0.5; b[1] = 0.5;
    ierr = PetscKernel_A_gets_inverse_A_2(A,0.0);CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error: %s is not a supported IRK method.\n",
                       ctxt.irktype);CHKERRQ(ierr);
    PetscFinalize();
    return(0);
  }
  for (s=0; s<nstages*nstages; s++) A[s] *= 1.0/ctxt.dt;
  for (s=0; s<nstages; s++) b[s] *= ctxt.dt;

  /* allocate and calculate the (-J) matrix */
  ierr = MatCreate(PETSC_COMM_WORLD,&J);CHKERRQ(ierr);
  ierr = MatSetType(J,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,ctxt.imax,ctxt.imax);CHKERRQ(ierr);
  ierr = MatSetUp(J);CHKERRQ(ierr);
  PetscReal values[3] = {-ctxt.a*1.0/dx2,ctxt.a*2.0/dx2,-ctxt.a*1.0/dx2};
  PetscInt  col[3];
  ierr = MatGetOwnershipRange(J,&matis,&matie);CHKERRQ(ierr);
  for (i=matis; i<matie; i++) {
    /* periodic boundaries */
    if (i == 0) {
      col[0] = ctxt.imax-1;
      col[1] = i;
      col[2] = i+1;
    } else if (i == ctxt.imax-1) {
      col[0] = i-1;
      col[1] = i;
      col[2] = 0;
    } else {
      col[0] = i-1;
      col[1] = i;
      col[2] = i+1;
    }
    ierr= MatSetValues(J,1,&i,3,col,values,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Create the TAIJ matrix */
  ierr = MatCreateTAIJ(J,nstages,A,&TA);CHKERRQ(ierr);

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
  PetscMalloc3(nstages,PetscInt,&ix,
               nstages,PetscReal,&zvals,
               ie-is,PetscInt,&ix2);CHKERRQ(ierr);

  /* iterate in time */
  for (n=0; n<ctxt.niter; n++) {

    /* compute and set the right hand side */
    PetscScalar *uarr;
    ierr = VecGetArray(u,&uarr);CHKERRQ(ierr);
    for (i=is; i<ie; i++) {
      for (s=0; s<nstages; s++) {
        ix[s]     = i*nstages+s;
        zvals[s]  = 0;
        for (t=0; t<nstages; t++) zvals[s] += uarr[i]*A[t*nstages+s];
      }
      ierr = VecSetValues(rhs,nstages,ix,zvals,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecAssemblyBegin (rhs);CHKERRQ(ierr);
    ierr = VecAssemblyEnd   (rhs);CHKERRQ(ierr);
    ierr = VecRestoreArray  (u,&uarr);CHKERRQ(ierr);

    /* Solve the system */
    ierr = KSPSolve(ksp,rhs,z);CHKERRQ(ierr);

    /* Update the solution */
    for (s=0; s<nstages; s++) {
      for (i=is; i<ie; i++) ix2[i-is] = i*nstages+s;
      /* extract stage vector */
      PetscScalar *warr;
      ierr = VecGetArray(w,&warr);CHKERRQ(ierr);
      ierr = VecGetValues(z,ie-is,ix2,warr);CHKERRQ(ierr);
      ierr = VecRestoreArray(w,&warr);CHKERRQ(ierr);

      /* Step completion */
      ierr = VecScale(w,-b[s]);CHKERRQ(ierr); 
      /* -b because J is actually -J */
      ierr = MatMultAdd(J,w,u,u);CHKERRQ(ierr);
    }

    /* time step complete */
  }
  PetscFree3(ix,ix2,zvals);

  /* Deallocate work and right-hand-side vectors */
  ierr = VecDestroy(&z);CHKERRQ(ierr);
  ierr = VecDestroy(&rhs);CHKERRQ(ierr);

  /* Calculate error in final solution */
  ierr = VecAYPX(uex,-1.0,u);
  ierr = VecNorm(uex,NORM_2,&err);
  err  = PetscSqrtScalar(err*err/((PetscReal)ctxt.imax));
  ierr = PetscPrintf(PETSC_COMM_WORLD,"L2 norm of the numerical error = %1.16E\n",err);CHKERRQ(ierr);

  /* Free up memory */
  ierr = KSPDestroy(&ksp);      CHKERRQ(ierr);
  ierr = MatDestroy(&TA);       CHKERRQ(ierr);
  ierr = MatDestroy(&J);        CHKERRQ(ierr);
  ierr = PetscFree(A);          CHKERRQ(ierr);
  ierr = PetscFree(b);          CHKERRQ(ierr);
  ierr = VecDestroy(&w);        CHKERRQ(ierr);
  ierr = VecDestroy(&uex);      CHKERRQ(ierr);
  ierr = VecDestroy(&u);        CHKERRQ(ierr);

  PetscFinalize();
  return(0);
}

#undef __FUNC__
#define __FUNC__
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
    x       = i * dx;
    uarr[i] = PetscExpScalar(-4.0*pi*pi*a*t)*PetscSinScalar(2*pi*x);
  }
  ierr = VecRestoreArray(u,&uarr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
