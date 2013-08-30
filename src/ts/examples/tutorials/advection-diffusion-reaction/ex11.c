const static char help[] = "2D Oregonator test at one point\n";
/*
Example which runs the 2D oregonator from Jahnke, Skaggs, and Winfree 1989

This system is a toy related to the BZ system, an oscillatory chemical reaction.

As this is a small problem, -ts_monitor_solution_lg is useful

*/

#include <petscts.h>
#include <petscdmda.h>

typedef PetscScalar Field[2];

typedef struct {
  PetscScalar f,q,epsilon;
} AppCtx;

static PetscErrorCode FormRHSFunction(TS,PetscReal,Vec,Vec,void*); 
static PetscErrorCode FormIFunction(TS,PetscReal,Vec,Vec,Vec,void*); 
static PetscErrorCode FormRHSJacobian(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*);
static PetscErrorCode FormIJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat*,Mat*,MatStructure*,void*);
static PetscErrorCode FormInitialSolution(TS,Vec,void*);

/* 
   Main function 
*/
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  PetscErrorCode  ierr;
  AppCtx          appctx;
  PetscReal       ftime, T = 30;
  PetscInt        steps;
  Vec             X;
  PetscMPIInt     size;
  TS              ts;
  DM              da;

  ierr = PetscInitialize(&argc, &argv, (char*) 0,help);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is a uniprocessor example only");

  appctx.f       = 1.4;
  appctx.q       = 0.002;
  appctx.epsilon = 0.01;
  
  ierr = PetscOptionsGetReal(NULL,"-f",&appctx.f,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,"-q",&appctx.q,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,"-epsilon",&appctx.epsilon,NULL);CHKERRQ(ierr);

  /* Create a 2d grid with only one point , with zero stencil width*/

  ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_PERIODIC,DMDA_BOUNDARY_PERIODIC,DMDA_STENCIL_STAR,1,1,PETSC_DECIDE,PETSC_DECIDE,2,0,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,0,"u");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,1,"v");CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSROSW);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetDM(ts,da);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,FormRHSFunction,&appctx);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,FormIFunction,&appctx);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,NULL,NULL,FormRHSJacobian,&appctx);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,NULL,NULL,FormIJacobian,&appctx);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da,&X);CHKERRQ(ierr);
  ierr = FormInitialSolution(ts,X,&appctx);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,PETSC_MAX_INT,T);CHKERRQ(ierr);
  ierr = TSSetMaxSNESFailures(ts,-1); /* unlimited failures */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr); 

  ierr = TSSolve(ts,X);CHKERRQ(ierr);

  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
  ierr = TSGetTimeStepNumber(ts,&steps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," steps %D, ftime %G\n",steps,ftime);CHKERRQ(ierr);
  ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  PetscFinalize();
  return EXIT_SUCCESS;
}

/*
I Function
*/
#undef __FUNCT__
#define __FUNCT__ "FormIFunction"
static PetscErrorCode FormIFunction(TS ts, PetscReal t, Vec X, Vec Xdot, Vec F, void* ctx)
{
  AppCtx           *appctx = (AppCtx*) ctx; 
  DM               da;
  PetscErrorCode   ierr;
  Field            **x,**xdot,**f;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  
  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,Xdot,&xdot);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);
  { PetscInt i=0,j=0;
    PetscScalar u = x[j][i][0], v = x[j][i][1], ff = appctx->f, q = appctx->q, epsilon=appctx->epsilon;
    f[j][i][0] = xdot[j][i][0] - (u - u*u - ff*v*(u-q)/(u+q))/epsilon;
    f[j][i][1] = xdot[j][i][1];
  }
  ierr = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,Xdot,&xdot);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*
RHS Function
*/
#undef __FUNCT__
#define __FUNCT__ "FormRHSFunction"
static PetscErrorCode FormRHSFunction(TS ts, PetscReal t, Vec X, Vec F, void* ctx)
{
  DM               da;
  PetscErrorCode   ierr;
  Field            **x,**f; 

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  
  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);
  { PetscInt i=0,j=0;
    PetscScalar uc = x[j][i][0], vc = x[j][i][1];
    f[j][i][0] = 0;
    f[j][i][1] = uc - vc;
  }
  ierr = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*
Initial Conditions
*/
#undef __FUNCT__
#define __FUNCT__ "FormInitialSolution"
static PetscErrorCode FormInitialSolution(TS ts,Vec X,void *ctx)
{ 
  PetscErrorCode ierr;
  Field          **x;
  DM             da;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);
  {
    PetscInt i=0,j=0;
    x[j][i][0] = 1;
    x[j][i][1] = 0.1;
  }
  ierr = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "FormRHSJacobian"
PetscErrorCode FormRHSJacobian(TS ts,PetscReal t,Vec X,Mat *A,Mat *B,MatStructure *str,void *ctx)
{
  PetscErrorCode   ierr;
  Field            **x;
  MatStencil       stencil[2],rowstencil[2];
  PetscScalar      v[2*2];
  DM               da;

  PetscFunctionBeginUser;
  
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);

  {PetscInt i=0, j=0;
   {PetscInt a;
   for(a =0;a<2;++a){
      rowstencil[a].i = i;
      rowstencil[a].j = j;
      rowstencil[a].k = 0;
      rowstencil[a].c = a;
      stencil[a].i = i;
      stencil[a].j = j;
      stencil[a].k = 0;
      stencil[a].c = a;
    }
   }
   
  v[0] =    0; 
  v[1] =    0;
  v[2] =  1.0;
  v[3] = -1.0;

  ierr = MatSetValuesStencil(*A,2,rowstencil,2,stencil,v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (*A != *B) {
      ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  *str = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "FormIJacobian"
PetscErrorCode FormIJacobian(TS ts,PetscReal t,Vec X,Vec Xdot, PetscReal a, Mat *A,Mat *B,MatStructure *str,void *ctx)
{
  PetscErrorCode   ierr;
  AppCtx           *appctx = (AppCtx*) ctx; 
  Field            **x;
  MatStencil       stencil[2],rowstencil[2];
  PetscScalar      v[2*2];
  DM               da;

  PetscFunctionBeginUser;
  
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);

  {PetscInt i=0, j=0;
   {PetscInt a;
   for(a =0;a<2;++a){
      rowstencil[a].i = i;
      rowstencil[a].j = j;
      rowstencil[a].k = 0;
      rowstencil[a].c = a;
      stencil[a].i = i;
      stencil[a].j = j;
      stencil[a].k = 0;
      stencil[a].c = a;
    }
   }
   PetscScalar uc = x[j][i][0], vc = x[j][i][1];
   PetscScalar fac = appctx->q + uc;
  v[0] =  a - (1 - 2*uc - 2*appctx->f*vc*appctx->q/(fac*fac))/appctx->epsilon; 
  v[1] =    - (appctx->f * (uc - appctx->q)/fac)/appctx->epsilon;
  v[2] =  0 ;
  v[3] =  a;

  ierr = MatSetValuesStencil(*A,2,rowstencil,2,stencil,v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (*A != *B) {
      ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  *str = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}
