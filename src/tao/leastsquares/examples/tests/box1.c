/*
   Include "petsctao.h" so that we can use TAO solvers.  Note that this
   file automatically includes libraries such as:
     petsc.h       - base PETSc routines   petscvec.h - vectors
     petscsys.h    - sysem routines        petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners

 This version tests correlated terms using both vector and listed forms 
*/

#include <petsctao.h>

/*
             exp(−0.1x1 ) − exp(−0.1x2 ) − x3(exp(−0.1) + exp(−1))
F (x) :=    exp(−0.2x1 ) − exp(−0.2x2 ) − x3(exp(−0.2) + exp(−2)) =0
             exp(−0.3x1 ) − exp(−0.3x2 ) − x3(exp(−0.3) + exp(−3))
has the two solutions [1; 10; 1] and [1; 10; −1].


 */


static char help[]="Finds the nonlinear least-squares solution to the model \n\
             exp(−0.1x1 ) − exp(−0.1x2 ) − x3(exp(−0.1) + exp(−1)) \n\
F (x) :=     exp(−0.2x1 ) − exp(−0.2x2 ) − x3(exp(−0.2) + exp(−2))  =0\n\
             exp(−0.3x1 ) − exp(−0.3x2 ) − x3(exp(−0.3) + exp(−3))\n";


/*T
   Concepts: TAO^Solving a system of nonlinear equations, nonlinear least squares
   Routines: TaoCreate();
   Routines: TaoSetType();
   Routines: TaoSetSeparableObjectiveRoutine();
   Routines: TaoSetJacobianRoutine();
   Routines: TaoSetInitialVector();
   Routines: TaoSetFromOptions();
   Routines: TaoSetConvergenceHistory(); TaoGetConvergenceHistory();
   Routines: TaoSolve();
   Routines: TaoView(); TaoDestroy();
   Processors: 1
T*/

#define NOBSERVATIONS 3
#define NPARAMETERS 3

/* User-defined application context */
typedef struct {
  /* Working space */
  PetscInt idm[NOBSERVATIONS];  /* Matrix indices for jacobian */
  PetscInt idn[NPARAMETERS];
} AppCtx;

/* User provided Routines */
PetscErrorCode FormStartingPoint(Vec);
PetscErrorCode EvaluateFunction(Tao, Vec, Vec, void *);
PetscErrorCode EvaluateJacobian(Tao, Vec, Mat, Mat, void *);


/*--------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;           /* used to check for functions returning nonzeros */
  PetscInt       wtype=0;
  Vec            x, f;               /* solution, function */
  Vec            w;                  /* weights */
  Mat            J;                  /* Jacobian matrix */
  Tao            tao;                /* Tao solver context */
  PetscInt       i;               /* iteration information */
  PetscInt       w_row[NOBSERVATIONS]; /* explicit weights */
  PetscInt       w_col[NOBSERVATIONS];
  PetscReal      w_vals[NOBSERVATIONS];
  PetscBool      flg;
  AppCtx         user;               /* user-defined work context */

  PetscInitialize(&argc,&argv,(char *)0,help);

  ierr = PetscOptionsGetInt(NULL,NULL,"-wtype",&wtype,&flg);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"wtype=%d\n",wtype);CHKERRQ(ierr);
  /* Allocate vectors */
  ierr = VecCreateSeq(MPI_COMM_SELF,NPARAMETERS,&x);CHKERRQ(ierr);
  ierr = VecCreateSeq(MPI_COMM_SELF,NOBSERVATIONS,&f);CHKERRQ(ierr);

  ierr = VecDuplicate(f,&w);CHKERRQ(ierr);

  /* no correlation, but set in different ways */
  ierr = VecSet(w,1.0);CHKERRQ(ierr);
  for (i=0;i<NOBSERVATIONS;i++) {
    w_row[i]=i; w_col[i]=i; w_vals[i]=1.0;
  }

  /* Create the Jacobian matrix. */
  ierr = MatCreateSeqDense(MPI_COMM_SELF,NOBSERVATIONS,NPARAMETERS,NULL,&J);CHKERRQ(ierr);

  for (i=0;i<NOBSERVATIONS;i++) user.idm[i] = i;

  for (i=0;i<NPARAMETERS;i++) user.idn[i] = i;

  /* Create TAO solver and set desired solution method */
  ierr = TaoCreate(PETSC_COMM_SELF,&tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAOPOUNDERS);CHKERRQ(ierr);

 /* Set the function and Jacobian routines. */
  ierr = FormStartingPoint(x);CHKERRQ(ierr);
  ierr = TaoSetInitialVector(tao,x);CHKERRQ(ierr);
  ierr = TaoSetSeparableObjectiveRoutine(tao,f,EvaluateFunction,(void*)&user);CHKERRQ(ierr);
  if (wtype == 1) {
    ierr = TaoSetSeparableObjectiveWeights(tao,w,0,NULL,NULL,NULL);CHKERRQ(ierr);
  } else if (wtype == 2) {
    ierr = TaoSetSeparableObjectiveWeights(tao,NULL,NOBSERVATIONS,w_row,w_col,w_vals);CHKERRQ(ierr);
  }

  /* Check for any TAO command line arguments */
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);

  /* Perform the Solve */
  ierr = TaoSolve(tao);CHKERRQ(ierr);
  ierr = TaoView(tao,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  /* Free TAO data structures */
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);

   /* Free PETSc data structures */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&w);CHKERRQ(ierr);
  ierr = VecDestroy(&f);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);

  PetscFinalize();
  return 0;
}

/*--------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "EvaluateFunction"
PetscErrorCode EvaluateFunction(Tao tao, Vec X, Vec F, void *ptr)
{
  PetscInt       i;
  PetscReal      *x,*f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);

  for (i=0;i<NOBSERVATIONS;i++) {
    f[0] = exp(-0.1*x[0]) - exp(-0.1*x[1]) - x[2]*(exp(-0.1) + exp(-1.0));
    f[1] = exp(-0.2*x[0]) - exp(-0.2*x[1]) - x[2]*(exp(-0.2) + exp(-2.0));
    f[2] = exp(-0.3*x[0]) - exp(-0.3*x[1]) - x[2]*(exp(-0.3) + exp(-3.0));
  }
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  ierr = VecView(F,0);

  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */
#undef __FUNCT__
#define __FUNCT__ "FormStartingPoint"
PetscErrorCode FormStartingPoint(Vec X)
{
  PetscErrorCode ierr;
  PetscReal *x;

  PetscFunctionBegin;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  x[0]=1.0;
  x[1]=10.0;
  x[2]=1.0;
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
