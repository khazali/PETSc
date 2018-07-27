/* Program usage: mpiexec -n 1 rosenbrock1 [-help] [all TAO options] */

/*  Include "petsctao.h" so we can use TAO solvers.  */
#include <petsctao.h>

static  char help[] = "This example demonstrates use of the TAO package to \n\
solve an unconstrained minimization problem on a single processor.  We \n\
minimize the extended Rosenbrock function: \n\
   sum_{i=0}^{n/2-1} ( alpha*(x_{2i+1}-x_{2i}^2)^2 + (1-x_{2i})^2 ) \n\
or the chained Rosenbrock function:\n\
   sum_{i=0}^{n-1} alpha*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2\n";

/*T
   Concepts: TAO^Solving a bound constrained minimization problem
   Routines: TaoCreate();
   Routines: TaoSetType(); TaoSetObjectiveAndGradientRoutine();
   Routines: TaoSetInitialVector();
   Routines: TaoSetFromOptions();
   Routines: TaoSolve();
   Routines: TaoDestroy();
   Processors: 1
T*/

/* -------------- User-defined routines ---------- */
PetscErrorCode FormFunctionGradient(Tao,Vec,PetscReal*,Vec,void*);
PetscErrorCode FormHessian(Tao,Vec,Mat,Mat,void*);

int main(int argc,char **argv)
{
  PetscErrorCode     ierr;                  /* used to check for functions returning nonzeros */
  PetscReal          zero=0.0;
  Vec                x, lb, ub;                     /* solution vector */
  Tao                tao;                   /* Tao solver context */
  PetscMPIInt        size,rank;                  /* number of processes running */
  PetscReal          *xx;

  /* Initialize TAO and PETSc */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  if (size >1) SETERRQ(PETSC_COMM_SELF,1,"Incorrect number of processors");

  /* Allocate vectors for the solution and gradient */
  ierr = VecCreateSeq(PETSC_COMM_SELF,2,&x);CHKERRQ(ierr);

  /* The TAO code begins here */

  /* Create TAO solver with desired solution method */
  ierr = TaoCreate(PETSC_COMM_SELF,&tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAOBLMVM);CHKERRQ(ierr);

  /* Set solution vec and an initial guess */
  ierr = VecSet(x, zero);CHKERRQ(ierr);
  ierr = VecGetArray(x, &xx);CHKERRQ(ierr);
  xx[0] = 1.1984610469441122;
  xx[1] = -1.861715120513463;
  ierr = VecRestoreArray(x, &xx);CHKERRQ(ierr);
  ierr = TaoSetInitialVector(tao,x);CHKERRQ(ierr);
  
  /* Set lower and upper bounds */
  ierr = VecDuplicate(x, &lb);CHKERRQ(ierr);
  ierr = VecSet(lb, -2.0);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &ub);CHKERRQ(ierr);
  ierr = VecSet(ub, 2.0);CHKERRQ(ierr);
  ierr = TaoSetVariableBounds(tao, lb, ub);CHKERRQ(ierr);

  /* Set routines for function and gradient evaluation */
  ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,PETSC_NULL);CHKERRQ(ierr);

  /* Check for TAO command line options */
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);
  
  /* SOLVE THE APPLICATION */
  ierr = TaoSolve(tao);CHKERRQ(ierr);

  ierr = TaoDestroy(&tao);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&lb);CHKERRQ(ierr);
  ierr = VecDestroy(&ub);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/* -------------------------------------------------------------------- */
/*
    FormFunctionGradient - Evaluates the function, f(X), and gradient, G(X).

    Input Parameters:
.   tao  - the Tao context
.   X    - input vector
.   ptr  - optional user-defined context, as set by TaoSetFunctionGradient()

    Output Parameters:
.   G - vector containing the newly evaluated gradient
.   f - function value

    Note:
    Some optimization methods ask for the function and the gradient evaluation
    at the same time.  Evaluating both at once may be more efficient that
    evaluating each separately.
*/
PetscErrorCode FormFunctionGradient(Tao tao,Vec X,PetscReal *f, Vec G,void *ptr)
{
  PetscErrorCode    ierr;
  PetscReal         t1,t2,t3,x1,x2;
  PetscScalar       *g;
  const PetscScalar *x;

  PetscFunctionBeginUser;
  /* Get pointers to vector data */
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(G,&g);CHKERRQ(ierr);
  x1 = x[0]; x2 = x[1];

  /* Compute G(X) */
  g[0] = 2.0*(PetscPowReal(x1,5.0) - 4.2*PetscPowReal(x1,3.0) + 4.0*x1 + 0.5*x2);
  g[1] = x1 + 16.0*PetscPowReal(x2,3.0) - 8.0*x2;
  
  /* Compute function value */
  t1 = (4.0 - 2.1*PetscPowReal(x1,2.0) + PetscPowReal(x1,4.0)/3.0) * PetscPowReal(x1,2.0);
  t2 = x1*x2;
  t3 = (-4.0 + 4.0*PetscPowReal(x2,2.0)) * PetscPowReal(x2,2.0);
  *f = t1 + t2 + t3;

  /* Restore vectors */
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(G,&g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*TEST

   build:
      requires: !complex

   test:
      args: -tao_monitor -tao_gatol 1.e-5 -tao_grtol 1e-5
      requires: !single

TEST*/
