static char help[] = "TAO/Pounders Matlab Testing on the More'-Wild Benchmark Problems\n\
The interface calls:\n\
    TestingInitialize.m to initialize the problem set\n\
    ProblemInitialize.m to initialize each instance\n\
    ProblemFinalize.m to store the performance data for the instance solved\n\
    TestingFinalize.m to store the entire set of performance data\n\
\n\
TestingPlot.m is called outside of TAO/Pounders to produce a performance profile\n\
of the results compared to the Matlab fminsearch algorithm.\n";

#include <petsctao.h>
#include <petscmatlab.h>

typedef struct {
  PetscMatlabEngine mengine;

  double delta;           /* Initial trust region radius */

  int n;                  /* Number of inputs */
  int m;                  /* Number of outputs */
  int nfmax;              /* Maximum function evaluations */
  int npmax;              /* Maximum interpolation points */
} AppCtx;

static PetscErrorCode EvaluateResidual(Tao tao, Vec X, Vec F, void *ptr)
{
  AppCtx         *user = (AppCtx *)ptr;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectSetName((PetscObject)X,"X");CHKERRQ(ierr);
  ierr = PetscMatlabEnginePut(user->mengine,(PetscObject)X);CHKERRQ(ierr);
  ierr = PetscMatlabEngineEvaluate(user->mengine,"F = func(X);");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)F,"F");CHKERRQ(ierr);
  ierr = PetscMatlabEngineGet(user->mengine,(PetscObject)F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EvaluateJacobian(Tao tao, Vec X, Mat J, Mat JPre, void *ptr)
{
  AppCtx         *user = (AppCtx *)ptr;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectSetName((PetscObject)X,"X");CHKERRQ(ierr);
  ierr = PetscMatlabEnginePut(user->mengine,(PetscObject)X);CHKERRQ(ierr);
  ierr = PetscMatlabEngineEvaluate(user->mengine,"J = jac(X);");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)J,"J");CHKERRQ(ierr);
  ierr = PetscMatlabEngineGet(user->mengine,(PetscObject)J);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoPounders(AppCtx *user)
{
  PetscErrorCode ierr;
  Tao            tao;
  Vec            X, F;
  Mat            J;
  char           buf[1024];

  PetscFunctionBegin;

  /* Set the values for the algorithm options we want to use */
  sprintf(buf,"%d",user->nfmax);
  ierr = PetscOptionsSetValue(NULL,"-tao_max_funcs",buf);CHKERRQ(ierr);
  sprintf(buf,"%d",user->npmax);
  ierr = PetscOptionsSetValue(NULL,"-tao_pounders_npmax",buf);CHKERRQ(ierr);
  sprintf(buf,"%5.4e",user->delta);
  ierr = PetscOptionsSetValue(NULL,"-tao_pounders_delta",buf);CHKERRQ(ierr);

  /* Create the TAO objects and set the type */
  ierr = TaoCreate(PETSC_COMM_SELF,&tao);CHKERRQ(ierr);

  /* Create starting point and initialize */
  ierr = VecCreateSeq(PETSC_COMM_SELF,user->n,&X);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)X,"X0");CHKERRQ(ierr);
  ierr = PetscMatlabEngineGet(user->mengine,(PetscObject)X);CHKERRQ(ierr);
  ierr = TaoSetInitialVector(tao,X);CHKERRQ(ierr);

  /* Create residuals vector and set residual function */  
  ierr = VecCreateSeq(PETSC_COMM_SELF,user->m,&F);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)F,"F");CHKERRQ(ierr);
  ierr = TaoSetResidualRoutine(tao,F,EvaluateResidual,(void*)user);CHKERRQ(ierr);

  /* Create Jacobian matrix and set residual Jacobian routine */  
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,user->m,user->n,user->n,NULL,&J);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)J,"J");CHKERRQ(ierr);
  ierr = TaoSetResidualJacobianRoutine(tao,J,J,EvaluateJacobian,(void*)user);CHKERRQ(ierr);

  /* Solve the problem */
  ierr = TaoSetType(tao,TAOPOUNDERS);CHKERRQ(ierr);
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);
  ierr = TaoSolve(tao);CHKERRQ(ierr);

  /* Finish the problem */
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  AppCtx         user;
  PetscErrorCode ierr;
  PetscScalar    tmp;
  int            i;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscMatlabEngineCreate(PETSC_COMM_SELF,NULL,&user.mengine);CHKERRQ(ierr);
  ierr = PetscMatlabEngineEvaluate(user.mengine,"TestingInitialize");CHKERRQ(ierr);

  for (i = 1; i <= 53; ++i) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"%d\n",i);
    ierr = PetscMatlabEngineEvaluate(user.mengine,"np = %d; ProblemInitialize",i);CHKERRQ(ierr);
    ierr = PetscMatlabEngineGetArray(user.mengine,1,1,&tmp,"n");CHKERRQ(ierr);
    user.n = (int)tmp;
    ierr = PetscMatlabEngineGetArray(user.mengine,1,1,&tmp,"m");CHKERRQ(ierr);
    user.m = (int)tmp;
    ierr = PetscMatlabEngineGetArray(user.mengine,1,1,&tmp,"nfmax");CHKERRQ(ierr);
    user.nfmax = (int)tmp;
    ierr = PetscMatlabEngineGetArray(user.mengine,1,1,&tmp,"npmax");CHKERRQ(ierr);
    user.npmax = (int)tmp;
    ierr = PetscMatlabEngineGetArray(user.mengine,1,1,&tmp,"delta");CHKERRQ(ierr);
    user.delta = (double)tmp;

    /* Ignore return code for now -- do not stop testing on inf or nan errors */
    ierr = TaoPounders(&user);

    ierr = PetscMatlabEngineEvaluate(user.mengine,"ProblemFinalize");CHKERRQ(ierr);
  }

  ierr = PetscMatlabEngineEvaluate(user.mengine,"TestingFinalize");CHKERRQ(ierr);
  ierr = PetscMatlabEngineDestroy(&user.mengine);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

