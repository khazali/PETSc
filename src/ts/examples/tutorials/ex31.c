static const char help[] = "Solve a Fermi-Pasta-Ulam problem, a Hamiltonian system of harmonic oscillators coupled with nonlinear springs, amendable to solution with IMEX methods"; 

/*
Use TS tools to solve the Fermi-Pasta-Ulam problem:

Consider a colinear series of 2m unit point masses, connected in series by springs which are alternately 'stiff' or 'weak',
between two fixed boundaries.
The dynamics of this system (with 2m DOF) involve rapid oscillations, and exhibit interesting energy exhange on O(\omega^{-1})
timescales.

The 'stiff' (fast) springs are linear with spring constant \omega^2, and the 'weak' (slow) springs produce force
proportional to the cube of their length. All springs are of rest length 0 and we take the the first and last point mass
 positions to be fixed. 

To isolate the fast quadratic potential, a change of variables is employed. If q_j is the position of the jth mass, consider 

x_{i,0} = (q_{2i} + q_{2i-1})/sqrt{2}   [scaled midpoint of the ith stiff spring]
x_{i,1} = (q_{2i} - q_{2i-1})/sqrt{2}   [scaled length of the ith stiff spring]

y are the conjugate momenta (change the q's to p's and x's to y's in the equations above).

These variables are ordered x_{i,0} y_{i,0} x_{i,1} y_{i,1} at each of l nodes corresponding to the stiff springs.

The system is Hamiltonian and from Hamiltonian's equations (or a Lagrangian and the Euler-Lagrange Equations)
one obtains an ODE system with a linear LHS and a nonlinear RHS which does not involve the 'fast' scale \omega^2.

The initial conditions provided place energy in the first stiff springs, and integrators can be compared based
on how well they capture the slow energy exchange between the  oscillators.

References:
   Machlachlan and O'Neal, "Comparison of Integrators for the Fermi-Pasta-Ulam Problem" 2007
   Hairer, Lubich and Wanner, "Geometric Numerical Integration, 2nd ed." 2006

Accepts special options
     -omega         the characteristic frequency of the stiff springs
     -numsprings    the number of stiff springs
     -saveascii     save an ascii file of the state, suitable for analysis of various (almost) conserved quantities
*/

#include <petscdmda.h>
#include <petscts.h>

typedef PetscScalar Field[4];

/*
Problem Data
*/
typedef struct _User *User;
struct _User
{
  MPI_Comm    comm;  
  PetscReal   omega, omega2;
  PetscViewer viewer;
};

/*
 User-defined routines
*/
static PetscErrorCode FormRHSFunction(TS,PetscReal,Vec,Vec,void*);
static PetscErrorCode FormIFunction(TS,PetscReal,Vec,Vec,Vec,void*);
static PetscErrorCode FormIJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat*,Mat*,MatStructure*,void*);
static PetscErrorCode FormInitialSolution(TS,Vec,void*);
static PetscErrorCode Monitor(TS,PetscInt,PetscReal,Vec,void*);

/*
Main function
*/
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char* argv[])
{
  PetscErrorCode ierr;
  Vec                 X;
  DM                  da;
  Mat                 J;
  TS                  ts;
  struct _User        user;
  PetscInt            m = 3; /* The number of stiff/weak spring pairs */
  PetscReal           timestep = 0.03,maxtime,ftime,omega = 50;
  PetscInt            maxsteps = 100000,steps;
  PetscBool           useViewer = PETSC_FALSE;
  TSConvergedReason   reason;

  ierr = PetscInitialize(&argc, &argv, (char*) 0,help);CHKERRQ(ierr);

  ierr = PetscOptionsGetReal(NULL,"-omega",&omega,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-numsprings",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,"-saveascii",&useViewer,NULL);CHKERRQ(ierr);

  /* Set parameters */
  user.comm =     PETSC_COMM_WORLD;
  user.omega  =   omega;
  user.omega2 =   omega * omega;
  maxtime =       4*omega;

  /*  Create DMDA to manage our system,  a 1d grid with 4dof at each  of m points */
  ierr = DMDACreate1d(PETSC_COMM_WORLD, DMDA_BOUNDARY_NONE, m, 4, 1, NULL, &da);CHKERRQ(ierr);

  /* Create Global Vector */ 
  ierr = DMCreateGlobalVector(da, &X);CHKERRQ(ierr);

  /* Set up TS */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts,da);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSTHETA);
  ierr = TSSetRHSFunction(ts,NULL,FormRHSFunction,&user);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,FormIFunction,&user);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,MATAIJ,&J);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,J,J,FormIJacobian,&user);CHKERRQ(ierr);
  ierr = TSSetDuration(ts, maxsteps, maxtime);CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts, 0, timestep);CHKERRQ(ierr);

  /* Set Initial Conditions */
  ierr = FormInitialSolution(ts, X, &user);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,X);CHKERRQ(ierr);

  /* Viewer */
  if(useViewer){
    ierr = PetscViewerASCIIOpen(user.comm,"ex31_output.txt",&user.viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(user.viewer,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr); 
    ierr = PetscViewerASCIISynchronizedAllow(user.viewer,PETSC_TRUE);CHKERRQ(ierr);
    ierr = TSMonitorSet(ts,Monitor,&user,NULL);CHKERRQ(ierr);
  }
  
/* Get runtime TS options */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* Solve */
  ierr = TSSolve(ts,X); CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
  ierr = TSGetTimeStepNumber(ts,&steps);CHKERRQ(ierr);
  ierr = TSGetConvergedReason(ts,&reason);CHKERRQ(ierr);
  
  /* Print */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%s at time %G after %D steps\n",TSConvergedReasons[reason],ftime,steps);CHKERRQ(ierr);

  /* Clean Up */
  if(useViewer) {ierr = PetscViewerDestroy(&user.viewer);CHKERRQ(ierr);}
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr); 
  ierr = PetscFinalize();

  return EXIT_SUCCESS;
}

/*
RHS (slow, nonlinear) function
*/
#undef __FUNCT__
#define __FUNCT__ "FormRHSFunction"
static PetscErrorCode FormRHSFunction(TS ts, PetscReal t, Vec X, Vec F, void* ctx)
{

  DM             da; 
  Vec            Xloc;
  DMDALocalInfo  info;
  PetscInt       i;
  Field          *x,*f;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);

  ierr = DMGetLocalVector(da,&Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);

  /* Get pointers to vector data */
  ierr = DMDAVecGetArray(da,Xloc,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);

  for (i=info.xs; i<info.xs+info.xm; i++) {
    PetscReal tL, tR;
    if(!i){
      tL = x[i][0]    - x[i][2];
      tR = x[i+1][0]  - x[i+1][2] - x[i][0]   - x[i][2];
    } else if (i==info.mx-1){
      tL =   x[i][0]   - x[i][2]  - x[i-1][0] - x[i-1][2];
      tR =                        - x[i][0]   - x[i][2];
    } else {
      tL = x[i][0]    - x[i][2]   - x[i-1][0] - x[i-1][2];
      tR = x[i+1][0]  - x[i+1][2] - x[i][0]   - x[i][2];
    }
  
    PetscReal flocL = tL*tL*tL, flocR = tR*tR*tR;     
    f[i][0] = 0;
    f[i][1] = - flocL + flocR; 
    f[i][2] = 0;
    f[i][3] =   flocL + flocR; 
  }

  /* Restore vectors */
  ierr = DMDAVecRestoreArray(da,Xloc,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&Xloc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*  
LHS (Implicit, fast, linear) Function
*/
#undef __FUNCT__
#define __FUNCT__ "FormIFunction"
static PetscErrorCode FormIFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ptr)
{
 User           user = (User)ptr;
  DM             da; 
  DMDALocalInfo  info;
  PetscInt       i;  
  Field          *x,*xdot,*f;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);

  /* Get pointers to vector data */
  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,Xdot,&xdot);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for (i=info.xs; i<info.xs+info.xm; i++) {
    f[i][0] = xdot[i][0] - x[i][1]; 
    f[i][1] = xdot[i][1]; 
    f[i][2] = xdot[i][2] - x[i][3];
    f[i][3] = xdot[i][3] + user->omega2 * x[i][2]; 
  }

  /* Restore vectors */
  ierr = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,Xdot,&xdot);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
LHS Jacobian function
*/
#undef __FUNCT__
#define __FUNCT__ "FormIJacobian"
static PetscErrorCode FormIJacobian(TS ts,PetscReal t,Vec X,Vec Xdot, PetscReal a,Mat *J,Mat *Jpre,MatStructure *str,void *ptr)
{
 
  User           user = (User)ptr;
  PetscErrorCode ierr;
  DMDALocalInfo  info;
  PetscInt       i;
  DM             da;
  Field          *x,*xdot;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);

  /* Get pointers to vector data */
  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,Xdot,&xdot);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for (i=info.xs; i<info.xs+info.xm; i++) {

    PetscScalar     v[4][4];

    v[0][0] = a;            v[0][1] =  -1; v[0][2] = 0;            v[0][3] =  0;
    v[1][0] = 0;            v[1][1] =   a; v[1][2] = 0;            v[1][3] =  0;
    v[2][0] = 0;            v[2][1] =   0; v[2][2] = a;            v[2][3] = -1;
    v[3][0] = 0;            v[3][1] =   0; v[3][2] = user->omega2; v[3][3] =  a;
    ierr    = MatSetValuesBlocked(*Jpre,1,&i,1,&i,&v[0][0],INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Restore vectors */
  ierr = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,Xdot,&xdot);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (*J != *Jpre) {
    ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
Initial Conditions
*/
#undef __FUNCT__
#define __FUNCT__ "FormInitialSolution"
static PetscErrorCode FormInitialSolution(TS ts,Vec X,void *ctx)
{ 
  User           user = (User)ctx;
  DM             da;
  DMDALocalInfo   info;
  Field          *x;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&da);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);

  /* Get pointers to vector data */
  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);

  for (i=info.xs; i<info.xs+info.xm; i++) {
    if(i==0){
      x[i][0] = 1; 
      x[i][1] = 1;
      x[i][2] = 1/user->omega;
      x[i][3] = 1;
    }else{
      x[i][0] = 0; 
      x[i][1] = 0;
      x[i][2] = 0;
      x[i][3] = 0;
    }
  } 

  ierr = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Monitor"
static PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal time,Vec u,void *ctx)
{
  User            user = (User)ctx;
  PetscErrorCode  ierr;
  Field           *uarr;
  DMDALocalInfo   info;
  DM              da;
  PetscInt        i,j;
  PetscMPIInt     rank,size;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&da);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,u,&uarr);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(user->comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(user->comm,&size);CHKERRQ(ierr);
  for (i=info.xs; i<info.xs+info.xm; i++) {
    for(j=0;j<info.dof;++j){
      ierr = PetscViewerASCIISynchronizedPrintf(user->viewer," %15.15g ",uarr[i][j]);CHKERRQ(ierr);
    }
  }
  if(rank==size-1){
    ierr = PetscViewerASCIISynchronizedPrintf(user->viewer,"\n");CHKERRQ(ierr);
  }
  ierr = PetscViewerFlush(user->viewer);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,u,&uarr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
