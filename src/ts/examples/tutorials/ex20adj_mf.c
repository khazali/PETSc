#define c11 1.0
#define c12 0
#define c21 2.0
#define c22 1.0
static char help[] = "Solves the van der Pol equation.\n\
Input parameters include:\n";

/*
   Concepts: TS^time-dependent nonlinear problems
   Concepts: TS^van der Pol equation DAE equivalent
   Processors: 1
*/
/* ------------------------------------------------------------------------

   This program solves the van der Pol DAE ODE equivalent
       y' = z                 (1)
       z' = mu[(1-y^2)z-y]
   on the domain 0 <= x <= 1, with the boundary conditions
       y(0) = 2, y'(0) = -6.666665432100101e-01,
   and
       mu = 10^6.
   This is a nonlinear equation.

   Notes:
   This code demonstrates the TS solver interface to a variant of
   linear problems, u_t = f(u,t), namely turning (1) into a system of
   first order differential equations,

   [ y' ] = [          z          ]
   [ z' ]   [     mu[(1-y^2)z-y]  ]

   which then we can write as a vector equation

   [ u_1' ] = [      u_2              ]  (2)
   [ u_2' ]   [ mu[(1-u_1^2)u_2-u_1]  ]

   which is now in the desired form of u_t = f(u,t). One way that we
   can split f(u,t) in (2) is to split by component,

   [ u_1' ] = [  u_2 ] + [       0              ]
   [ u_2' ]   [  0   ]   [ mu[(1-u_1^2)u_2-u_1] ]

   where

   [ F(u,t) ] = [  u_2 ]
                [  0   ]

   and

   [ G(u',u,t) ] = [ u_1' ] - [            0         ]
                   [ u_2' ]   [ mu[(1-u_1^2)u_2-u_1] ]

   Using the definition of the Jacobian of G (from the PETSc user manual),
   in the equation G(u',u,t) = F(u,t),

              dG   dG
   J(G) = a * -- + --
              du'  du

   where d is the partial derivative. In this example,

   dG   [ 1 ; 0 ]
   -- = [       ]
   du'  [ 0 ; 1 ]

   dG   [ 0                       ;         0         ]
   -- = [                                             ]
   du   [ mu*(1.0 + 2.0*u_1*u_2) ; -mu*(1-u_1*u_1)    ]

   Hence,

          [      a                 ;         0          ]
   J(G) = [                                             ]
          [ mu*(1.0 + 2.0*u_1*u_2) ; a - mu*(1-u_1*u_1) ]

  ------------------------------------------------------------------------- */
#include <petscts.h>
#include <petsctao.h>

typedef struct _n_User *User;
struct _n_User {
  PetscReal mu;
  PetscReal next_output;
 
  /* Sensitivity analysis support */ 
  PetscInt  steps;
  PetscReal ftime;
  Mat       A;                       /* Jacobian matrix shell */
  Mat       Jac;                     /* Jacobian matrix */
  Mat       Jacp;                    /* JacobianP matrix */
  Vec       x,lambda[2],lambdap[2];  /* adjoint variables */
};

/* Matrix free support */
typedef struct _mat_ctx *Mctx;
struct _mat_ctx {
  PetscReal time;
  Vec       X;
  Vec       Xdot;
  PetscReal shift;
  User      uctx;
  TS        ts;
};

/*
*  User-defined routines
*/
#undef __FUNCT__
#define __FUNCT__ "IFunction"
static PetscErrorCode IFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx)
{
  PetscErrorCode    ierr;
  User              user = (User)ctx;
  const PetscScalar *x,*xdot;
  PetscScalar       *f;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = xdot[0] - x[1];
  f[1] = c21*(xdot[0]-x[1]) + xdot[1] - user->mu*((1.0-x[0]*x[0])*x[1] - x[0]) ;
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IJacobian"
static PetscErrorCode IJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat A,Mat B,void *ctx)
{
  PetscErrorCode    ierr;
  User              user     = (User)ctx;
  PetscInt          rowcol[] = {0,1};
  PetscScalar       J[2][2];
  const PetscScalar *x;

  PetscFunctionBeginUser;
  ierr    = VecGetArrayRead(X,&x);CHKERRQ(ierr);

  J[0][0] = a;     J[0][1] =  -1.0;
  J[1][0] = c21*a + user->mu*(1.0 + 2.0*x[0]*x[1]);   J[1][1] = -c21 + a - user->mu*(1.0-x[0]*x[0]);
 
  ierr    = MatSetValues(B,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr    = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (A != B) {
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormIJacobian"
static PetscErrorCode FormIJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat A_shell,Mat B,void *ctx)
{
  PetscErrorCode    ierr;
  Mctx              mctx=NULL;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(A_shell,(void **)&mctx);CHKERRQ(ierr);

  mctx->time  = t;
  mctx->shift = a;
  if (mctx->ts != ts) mctx->ts = ts;
  if (mctx->uctx != ctx) mctx->uctx  = ctx;
  ierr = VecCopy(X,mctx->X);CHKERRQ(ierr);
  ierr = VecCopy(Xdot,mctx->Xdot);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RHSJacobianP"
static PetscErrorCode RHSJacobianP(TS ts,PetscReal t,Vec X,Mat A,void *ctx)
{
  PetscErrorCode    ierr;
  PetscInt          row[] = {0,1},col[]={0};
  PetscScalar       J[2][1];
  const PetscScalar *x;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);

  J[0][0] = 0;
  J[1][0] = (1.-x[0]*x[0])*x[1]-x[0];
  ierr    = MatSetValues(A,2,row,1,col,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Monitor"
/* Monitor timesteps and use interpolation to output at integer multiples of 0.1 */
static PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal t,Vec X,void *ctx)
{
  PetscErrorCode    ierr;
  const PetscScalar *x;
  PetscReal         tfinal, dt;
  User              user = (User)ctx;
  Vec               interpolatedX;

  PetscFunctionBeginUser;
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  ierr = TSGetDuration(ts,NULL,&tfinal);CHKERRQ(ierr);

  while (user->next_output <= t && user->next_output <= tfinal) {
    ierr = VecDuplicate(X,&interpolatedX);CHKERRQ(ierr);
    ierr = TSInterpolate(ts,user->next_output,interpolatedX);CHKERRQ(ierr);
    ierr = VecGetArrayRead(interpolatedX,&x);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"[%.1f] %D TS %.6f (dt = %.6f) X % 12.6e % 12.6e\n",
                       user->next_output,step,t,dt,(double)PetscRealPart(x[0]),
                       (double)PetscRealPart(x[1]));CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(interpolatedX,&x);CHKERRQ(ierr);
    ierr = VecDestroy(&interpolatedX);CHKERRQ(ierr);
    user->next_output += 0.1;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MyMult"
static PetscErrorCode MyMult(Mat A_shell,Vec X,Vec Y)
{
  Mctx           mctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(A_shell,(void**)&mctx);CHKERRQ(ierr);
  ierr = IJacobian(mctx->ts,mctx->time,mctx->X,mctx->Xdot,mctx->shift,mctx->uctx->Jac,mctx->uctx->Jac,mctx->uctx);CHKERRQ(ierr);
  ierr = MatMult(mctx->uctx->Jac,X,Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MyMultTranspose"
static PetscErrorCode MyMultTranspose(Mat A_shell,Vec X,Vec Y)
{
  Mctx           mctx;
  Mat            A;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(A_shell,(void**)&mctx);CHKERRQ(ierr);
  ierr = IJacobian(mctx->ts,mctx->time,mctx->X,mctx->Xdot,mctx->shift,mctx->uctx->Jac,mctx->uctx->Jac,mctx->uctx);CHKERRQ(ierr);
  ierr = MatMultTranspose(mctx->uctx->Jac,X,Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  TS              ts;            /* nonlinear solver */
  PetscBool       monitor = PETSC_FALSE;
  PetscScalar     *x_ptr,*y_ptr;
  PetscMPIInt     size;
  struct _n_User  user;
  struct _mat_ctx mctx;
  PetscErrorCode  ierr;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscInitialize(&argc,&argv,NULL,help);
  
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_SELF,1,"This is a uniprocessor example only!");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  user.next_output = 0.0;
  user.mu          = 1.0e6;
  user.steps       = 0;
  user.ftime       = 0.5;
  ierr = PetscOptionsGetBool(NULL,"-monitor",&monitor,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,"-mu",&user.mu,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors, solve same ODE on every process
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&user.Jac);CHKERRQ(ierr);
  ierr = MatSetSizes(user.Jac,PETSC_DECIDE,PETSC_DECIDE,2,2);CHKERRQ(ierr);
  ierr = MatSetUp(user.Jac);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&user.Jacp);CHKERRQ(ierr);
  ierr = MatSetSizes(user.Jacp,PETSC_DECIDE,PETSC_DECIDE,2,1);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user.Jacp);CHKERRQ(ierr);
  ierr = MatSetUp(user.Jacp);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create matrix free context
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreateShell(PETSC_COMM_WORLD,2,2,2,2,&mctx,&user.A);CHKERRQ(ierr);
  ierr = MatShellSetOperation(user.A,MATOP_MULT,(void (*)(void))MyMult);CHKERRQ(ierr);
  ierr = MatShellSetOperation(user.A,MATOP_MULT_TRANSPOSE,(void (*)(void))MyMultTranspose);CHKERRQ(ierr);
  ierr = MatCreateVecs(user.A,&user.x,NULL);CHKERRQ(ierr);
  ierr = VecDuplicate(user.x,&mctx.X);CHKERRQ(ierr);
  ierr = VecDuplicate(user.x,&mctx.Xdot);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSBEULER);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,IFunction,&user);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,user.A,user.A,FormIJacobian,&user);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,PETSC_DEFAULT,user.ftime);CHKERRQ(ierr);
  if (monitor) {
    ierr = TSMonitorSet(ts,Monitor,&user,NULL);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecGetArray(user.x,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 2.0;   x_ptr[1] = -0.66666654321;
  ierr = VecRestoreArray(user.x,&x_ptr);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    Save trajectory of solution so that TSAdjointSolve() may be used
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetSaveTrajectory(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,user.x);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&user.ftime);CHKERRQ(ierr);
  ierr = TSGetTimeStepNumber(ts,&user.steps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"mu %g, steps %D, ftime %g\n",(double)user.mu,user.steps,(double)user.ftime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n ode solution \n");CHKERRQ(ierr);
  ierr = VecView(user.x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Adjoint model starts here
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreateVecs(user.A,&user.lambda[0],NULL);CHKERRQ(ierr);
  /*   Set initial conditions for the adjoint integration */
  ierr = VecGetArray(user.lambda[0],&y_ptr);CHKERRQ(ierr);
  y_ptr[0] = 1.0; y_ptr[1] = 0.0;
  ierr = VecRestoreArray(user.lambda[0],&y_ptr);CHKERRQ(ierr);
  ierr = MatCreateVecs(user.A,&user.lambda[1],NULL);CHKERRQ(ierr);
  ierr = VecGetArray(user.lambda[1],&y_ptr);CHKERRQ(ierr); 
  y_ptr[0] = 0.0; y_ptr[1] = 1.0;
  ierr = VecRestoreArray(user.lambda[1],&y_ptr);CHKERRQ(ierr);

  ierr = MatCreateVecs(user.Jacp,&user.lambdap[0],NULL);CHKERRQ(ierr);
  ierr = VecGetArray(user.lambdap[0],&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 0.0;
  ierr = VecRestoreArray(user.lambdap[0],&x_ptr);CHKERRQ(ierr);
  ierr = MatCreateVecs(user.Jacp,&user.lambdap[1],NULL);CHKERRQ(ierr);
  ierr = VecGetArray(user.lambdap[1],&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 0.0;
  ierr = VecRestoreArray(user.lambdap[1],&x_ptr);CHKERRQ(ierr);

  ierr = TSAdjointSetGradients(ts,2,user.lambda,user.lambdap);CHKERRQ(ierr);

  /*   Set RHS JacobianP */
  ierr = TSAdjointSetRHSJacobian(ts,user.Jacp,RHSJacobianP,&user);CHKERRQ(ierr);

  ierr = TSAdjointSolve(ts);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt initial conditions: d[y(tf)]/d[y0]  d[y(tf)]/d[z0]\n");CHKERRQ(ierr);
  ierr = VecView(user.lambda[0],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt initial conditions: d[z(tf)]/d[y0]  d[z(tf)]/d[z0]\n");CHKERRQ(ierr);
  ierr = VecView(user.lambda[1],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt parameters: d[y(tf)]/d[mu]\n");CHKERRQ(ierr);
  ierr = VecView(user.lambdap[0],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n sensivitity wrt parameters: d[z(tf)]/d[mu]\n");CHKERRQ(ierr);
  ierr = VecView(user.lambdap[1],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  
  ierr = VecDestroy(&mctx.X);CHKERRQ(ierr);
  ierr = VecDestroy(&mctx.Xdot);CHKERRQ(ierr);
  ierr = MatDestroy(&user.A);CHKERRQ(ierr);
  ierr = MatDestroy(&user.Jac);CHKERRQ(ierr);
  ierr = MatDestroy(&user.Jacp);CHKERRQ(ierr);
  ierr = VecDestroy(&user.x);CHKERRQ(ierr);
  ierr = VecDestroy(&user.lambda[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&user.lambda[1]);CHKERRQ(ierr);
  ierr = VecDestroy(&user.lambdap[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&user.lambdap[1]);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);

  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}
