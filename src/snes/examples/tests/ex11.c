
static char help[] ="Solves the 2-dim Bratu (SFI - solid fuel ignition) test problem, where\n\
analytic formation of the Jacobian is the default.  \n\
\n\
  Solves the linear systems via 2 level additive Schwarz \n\
\n\
The command line\n\
options are:\n\
  -par <parameter>, where <parameter> indicates the problem's nonlinearity\n\
     problem SFI:  <parameter> = Bratu parameter (0 <= par <= 6.81)\n\
  -Mx <xg>, where <xg> = number of grid points in the x-direction on coarse grid\n\
  -My <yg>, where <yg> = number of grid points in the y-direction on coarse grid\n\n";

/*
    1) Solid Fuel Ignition (SFI) problem.  This problem is modeled by
    the partial differential equation

            -Laplacian u - lambda*exp(u) = 0,  0 < x,y < 1 ,

    with boundary conditions

             u = 0  for  x = 0, x = 1, y = 0, y = 1.

    A finite difference approximation with the usual 5-point stencil
    is used to discretize the boundary value problem to obtain a nonlinear
    system of equations.

   The code has two cases for multilevel solver
     I. the coarse grid Jacobian is computed in parallel
     II. the coarse grid Jacobian is computed sequentially on each processor
   in both cases the coarse problem is SOLVED redundantly.

*/

#include <petscsnes.h>
#include <petscdm.h>
#include <petscdmda.h>

/* User-defined application contexts */

typedef struct {
  PetscInt mx,my;               /* number grid points in x and y direction */
  Vec      localX,localF;       /* local vectors with ghost region */
  DM       da;
  Vec      x,b,r;               /* global vectors */
  Mat      J;                   /* Jacobian on grid */
} GridCtx;

typedef struct {
  PetscReal  param;             /* test problem parameter */
  GridCtx    fine;
  GridCtx    coarse;
  KSP        ksp_coarse;
  KSP        ksp_fine;
  PetscInt   ratio;
  Mat        R;                 /* restriction fine to coarse */
  Vec        Rscale;
  PetscBool  redundant_build;   /* build coarse matrix redundantly */
  Vec        localall;          /* contains entire coarse vector on each processor in NATURAL order*/
  VecScatter tolocalall;        /* maps from parallel "global" coarse vector to localall */
  VecScatter fromlocalall;      /* maps from localall vector back to global coarse vector */
} AppCtx;

#define COARSE_LEVEL 0
#define FINE_LEVEL   1

extern PetscErrorCode FormFunction(SNES,Vec,Vec,void*), FormInitialGuess1(AppCtx*,Vec);
extern PetscErrorCode FormJacobian(SNES,Vec,Mat,Mat,void*);
extern PetscErrorCode FormInterpolation(AppCtx*);

/*
      Mm_ratio - ration of grid lines between fine and coarse grids.
*/
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  SNES           snes;
  AppCtx         user;
  PetscErrorCode ierr;
  PetscInt       its, N, n, Nx = PETSC_DECIDE, Ny = PETSC_DECIDE, nlocal,Nlocal;
  PetscMPIInt    size;
  PetscReal      bratu_lambda_max = 6.81, bratu_lambda_min = 0.;
  KSP            ksp;
  PC             pc;

  /*
      Initialize PETSc, note that default options in ex11options can be
      overridden at the command line
  */
  PetscInitialize(&argc, &argv,"ex11options",help);

  user.ratio     = 2;
  user.coarse.mx = 5; user.coarse.my = 5; user.param = 6.0;
  ierr           = PetscOptionsGetInt(NULL,"-Mx",&user.coarse.mx,NULL);CHKERRQ(ierr);
  ierr           = PetscOptionsGetInt(NULL,"-My",&user.coarse.my,NULL);CHKERRQ(ierr);
  ierr           = PetscOptionsGetInt(NULL,"-ratio",&user.ratio,NULL);CHKERRQ(ierr);
  user.fine.mx   = user.ratio*(user.coarse.mx-1)+1; user.fine.my = user.ratio*(user.coarse.my-1)+1;

  ierr = PetscOptionsHasName(NULL,"-redundant_build",&user.redundant_build);CHKERRQ(ierr);
  if (user.redundant_build) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Building coarse Jacobian redundantly\n");CHKERRQ(ierr);
  }

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Coarse grid size %D by %D\n",user.coarse.mx,user.coarse.my);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Fine grid size %D by %D\n",user.fine.mx,user.fine.my);CHKERRQ(ierr);

  ierr = PetscOptionsGetReal(NULL,"-par",&user.param,NULL);CHKERRQ(ierr);
  if (user.param >= bratu_lambda_max || user.param < bratu_lambda_min) SETERRQ(PETSC_COMM_SELF,1,"Lambda is out of range");
  n = user.fine.mx*user.fine.my; N = user.coarse.mx*user.coarse.my;

  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-Nx",&Nx,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-Ny",&Ny,NULL);CHKERRQ(ierr);

  /* Set up distributed array for fine grid */
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,user.fine.mx,
                      user.fine.my,Nx,Ny,1,1,NULL,NULL,&user.fine.da);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(user.fine.da,&user.fine.x);CHKERRQ(ierr);
  ierr = VecDuplicate(user.fine.x,&user.fine.r);CHKERRQ(ierr);
  ierr = VecDuplicate(user.fine.x,&user.fine.b);CHKERRQ(ierr);
  ierr = VecGetLocalSize(user.fine.x,&nlocal);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(user.fine.da,&user.fine.localX);CHKERRQ(ierr);
  ierr = VecDuplicate(user.fine.localX,&user.fine.localF);CHKERRQ(ierr);
  ierr = MatCreateAIJ(PETSC_COMM_WORLD,nlocal,nlocal,n,n,5,NULL,3,NULL,&user.fine.J);CHKERRQ(ierr);

  /* Set up distributed array for coarse grid */
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,user.coarse.mx,
                      user.coarse.my,Nx,Ny,1,1,NULL,NULL,&user.coarse.da);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(user.coarse.da,&user.coarse.x);CHKERRQ(ierr);
  ierr = VecDuplicate(user.coarse.x,&user.coarse.b);CHKERRQ(ierr);
  if (user.redundant_build) {
    /* Create scatter from parallel global numbering to redundant with natural ordering */
    ierr = DMDAGlobalToNaturalAllCreate(user.coarse.da,&user.tolocalall);CHKERRQ(ierr);
    ierr = DMDANaturalAllToGlobalCreate(user.coarse.da,&user.fromlocalall);CHKERRQ(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF,N,&user.localall);CHKERRQ(ierr);
    /* Create sequential matrix to hold entire coarse grid Jacobian on each processor */
    ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,N,N,5,NULL,&user.coarse.J);CHKERRQ(ierr);
  } else {
    ierr = VecGetLocalSize(user.coarse.x,&Nlocal);CHKERRQ(ierr);
    ierr = DMCreateLocalVector(user.coarse.da,&user.coarse.localX);CHKERRQ(ierr);
    ierr = VecDuplicate(user.coarse.localX,&user.coarse.localF);CHKERRQ(ierr);
    /* We will compute the coarse Jacobian in parallel */
    ierr = MatCreateAIJ(PETSC_COMM_WORLD,Nlocal,Nlocal,N,N,5,NULL,3,NULL,&user.coarse.J);CHKERRQ(ierr);
  }

  /* Create nonlinear solver */
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);

  /* provide user function and Jacobian */
  ierr = SNESSetFunction(snes,user.fine.b,FormFunction,&user);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,user.fine.J,user.fine.J,FormJacobian,&user);CHKERRQ(ierr);

  /* set two level additive Schwarz preconditioner */
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCMG);CHKERRQ(ierr);
  ierr = PCMGSetLevels(pc,2,NULL);CHKERRQ(ierr);
  ierr = PCMGSetType(pc,PC_MG_ADDITIVE);CHKERRQ(ierr);

  /* always solve the coarse problem redundantly with direct LU solver */
  ierr = PetscOptionsSetValue("-coarse_pc_type","redundant");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-coarse_redundant_pc_type","lu");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-coarse_redundant_ksp_type","preonly");CHKERRQ(ierr);

  /* Create coarse level */
  ierr = PCMGGetCoarseSolve(pc,&user.ksp_coarse);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(user.ksp_coarse,"coarse_");CHKERRQ(ierr);
  ierr = KSPSetFromOptions(user.ksp_coarse);CHKERRQ(ierr);
  ierr = KSPSetOperators(user.ksp_coarse,user.coarse.J,user.coarse.J);CHKERRQ(ierr);
  ierr = PCMGSetX(pc,COARSE_LEVEL,user.coarse.x);CHKERRQ(ierr);
  ierr = PCMGSetRhs(pc,COARSE_LEVEL,user.coarse.b);CHKERRQ(ierr);
  if (user.redundant_build) {
    PC rpc;
    ierr = KSPGetPC(user.ksp_coarse,&rpc);CHKERRQ(ierr);
    ierr = PCRedundantSetScatter(rpc,user.tolocalall,user.fromlocalall);CHKERRQ(ierr);
  }

  /* Create fine level */
  ierr = PCMGGetSmoother(pc,FINE_LEVEL,&user.ksp_fine);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(user.ksp_fine,"fine_");CHKERRQ(ierr);
  ierr = KSPSetFromOptions(user.ksp_fine);CHKERRQ(ierr);
  ierr = KSPSetOperators(user.ksp_fine,user.fine.J,user.fine.J);CHKERRQ(ierr);
  ierr = PCMGSetR(pc,FINE_LEVEL,user.fine.r);CHKERRQ(ierr);
  ierr = PCMGSetResidual(pc,FINE_LEVEL,NULL,user.fine.J);CHKERRQ(ierr);

  /* Create interpolation between the levels */
  ierr = FormInterpolation(&user);CHKERRQ(ierr);
  ierr = PCMGSetInterpolation(pc,FINE_LEVEL,user.R);CHKERRQ(ierr);
  ierr = PCMGSetRestriction(pc,FINE_LEVEL,user.R);CHKERRQ(ierr);

  /* Set options, then solve nonlinear system */
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = FormInitialGuess1(&user,user.fine.x);CHKERRQ(ierr);
  ierr = SNESSolve(snes,NULL,user.fine.x);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of SNES iterations = %D\n", its);CHKERRQ(ierr);

  /* Free data structures */
  if (user.redundant_build) {
    ierr = VecScatterDestroy(&user.tolocalall);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&user.fromlocalall);CHKERRQ(ierr);
    ierr = VecDestroy(&user.localall);CHKERRQ(ierr);
  } else {
    ierr = VecDestroy(&user.coarse.localX);CHKERRQ(ierr);
    ierr = VecDestroy(&user.coarse.localF);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&user.fine.J);CHKERRQ(ierr);
  ierr = VecDestroy(&user.fine.x);CHKERRQ(ierr);
  ierr = VecDestroy(&user.fine.r);CHKERRQ(ierr);
  ierr = VecDestroy(&user.fine.b);CHKERRQ(ierr);
  ierr = DMDestroy(&user.fine.da);CHKERRQ(ierr);
  ierr = VecDestroy(&user.fine.localX);CHKERRQ(ierr);
  ierr = VecDestroy(&user.fine.localF);CHKERRQ(ierr);

  ierr = MatDestroy(&user.coarse.J);CHKERRQ(ierr);
  ierr = VecDestroy(&user.coarse.x);CHKERRQ(ierr);
  ierr = VecDestroy(&user.coarse.b);CHKERRQ(ierr);
  ierr = DMDestroy(&user.coarse.da);CHKERRQ(ierr);

  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = MatDestroy(&user.R);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Rscale);CHKERRQ(ierr);
  PetscFinalize();

  return 0;
} /* --------------------  Form initial approximation ----------------- */
#undef __FUNCT__
#define __FUNCT__ "FormInitialGuess1"
PetscErrorCode FormInitialGuess1(AppCtx *user,Vec X)
{
  PetscInt       i, j, row, mx, my, xs, ys, xm, ym, Xm, Ym, Xs, Ys;
  PetscErrorCode ierr;
  PetscReal      one = 1.0, lambda, temp1, temp, hx, hy;
  PetscScalar    *x;
  Vec            localX = user->fine.localX;

  mx = user->fine.mx;       my = user->fine.my;            lambda = user->param;
  hx = one/(PetscReal)(mx-1);  hy = one/(PetscReal)(my-1);
  /* sc = hx*hy*lambda;        hxdhy = hx/hy;            hydhx = hy/hx; */

  temp1 = lambda/(lambda + one);

  /* Get ghost points */
  ierr = DMDAGetCorners(user->fine.da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(user->fine.da,&Xs,&Ys,0,&Xm,&Ym,0);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&x);CHKERRQ(ierr);

  /* Compute initial guess */
  for (j=ys; j<ys+ym; j++) {
    temp = (PetscReal)(PetscMin(j,my-j-1))*hy;
    for (i=xs; i<xs+xm; i++) {
      row = i - Xs + (j - Ys)*Xm;
      if (i == 0 || j == 0 || i == mx-1 || j == my-1) {
        x[row] = 0.0;
        continue;
      }
      x[row] = temp1*PetscSqrtReal(PetscMin((PetscReal)(PetscMin(i,mx-i-1))*hx,temp));
    }
  }
  ierr = VecRestoreArray(localX,&x);CHKERRQ(ierr);

  /* Insert values into global vector */
  ierr = DMLocalToGlobalBegin(user->fine.da,localX,INSERT_VALUES,X);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(user->fine.da,localX,INSERT_VALUES,X);CHKERRQ(ierr);
  return 0;
}

/* --------------------  Evaluate Function F(x) --------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormFunction"
PetscErrorCode FormFunction(SNES snes,Vec X,Vec F,void *ptr)
{
  AppCtx         *user = (AppCtx*) ptr;
  PetscInt       i, j, row, mx, my, xs, ys, xm, ym, Xs, Ys, Xm, Ym;
  PetscErrorCode ierr;
  PetscReal      two = 2.0, one = 1.0, lambda,hx, hy, hxdhy, hydhx,sc;
  PetscScalar    u, uxx, uyy, *x,*f;
  Vec            localX = user->fine.localX, localF = user->fine.localF;

  mx = user->fine.mx;       my = user->fine.my;       lambda = user->param;
  hx = one/(PetscReal)(mx-1);  hy = one/(PetscReal)(my-1);
  sc = hx*hy*lambda;        hxdhy = hx/hy;            hydhx = hy/hx;

  /* Get ghost points */
  ierr = DMGlobalToLocalBegin(user->fine.da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(user->fine.da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMDAGetCorners(user->fine.da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(user->fine.da,&Xs,&Ys,0,&Xm,&Ym,0);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&x);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f);CHKERRQ(ierr);

  /* Evaluate function */
  for (j=ys; j<ys+ym; j++) {
    row = (j - Ys)*Xm + xs - Xs - 1;
    for (i=xs; i<xs+xm; i++) {
      row++;
      if (i > 0 && i < mx-1 && j > 0 && j < my-1) {
        u      = x[row];
        uxx    = (two*u - x[row-1] - x[row+1])*hydhx;
        uyy    = (two*u - x[row-Xm] - x[row+Xm])*hxdhy;
        f[row] = uxx + uyy - sc*PetscExpScalar(u);
      } else if ((i > 0 && i < mx-1) || (j > 0 && j < my-1)) {
        f[row] = .5*two*(hydhx + hxdhy)*x[row];
      } else {
        f[row] = .25*two*(hydhx + hxdhy)*x[row];
      }
    }
  }
  ierr = VecRestoreArray(localX,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&f);CHKERRQ(ierr);

  /* Insert values into global vector */
  ierr = DMLocalToGlobalBegin(user->fine.da,localF,INSERT_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(user->fine.da,localF,INSERT_VALUES,F);CHKERRQ(ierr);
  ierr = PetscLogFlops(11.0*ym*xm);CHKERRQ(ierr);
  return 0;
}

/*
        Computes the part of the Jacobian associated with this processor
*/
#undef __FUNCT__
#define __FUNCT__ "FormJacobian_Grid"
PetscErrorCode FormJacobian_Grid(AppCtx *user,GridCtx *grid,Vec X, Mat J,Mat B)
{
  Mat            jac = J;
  PetscErrorCode ierr;
  PetscInt       i, j, row, mx, my, xs, ys, xm, ym, Xs, Ys, Xm, Ym, col[5], nloc, grow;
  const PetscInt *ltog;
  PetscScalar    two    = 2.0, one = 1.0, lambda, v[5], hx, hy, hxdhy, hydhx, sc, *x, value;
  Vec            localX = grid->localX;

  mx = grid->mx;            my = grid->my;            lambda = user->param;
  hx = one/(PetscReal)(mx-1);  hy = one/(PetscReal)(my-1);
  sc = hx*hy;               hxdhy = hx/hy;            hydhx = hy/hx;

  /* Get ghost points */
  ierr = DMGlobalToLocalBegin(grid->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(grid->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMDAGetCorners(grid->da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(grid->da,&Xs,&Ys,0,&Xm,&Ym,0);CHKERRQ(ierr);
  ierr = DMDAGetGlobalIndices(grid->da,&nloc,&ltog);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&x);CHKERRQ(ierr);

  /* Evaluate Jacobian of function */
  for (j=ys; j<ys+ym; j++) {
    row = (j - Ys)*Xm + xs - Xs - 1;
    for (i=xs; i<xs+xm; i++) {
      row++;
      grow = ltog[row];
      if (i > 0 && i < mx-1 && j > 0 && j < my-1) {
        v[0] = -hxdhy; col[0] = ltog[row - Xm];
        v[1] = -hydhx; col[1] = ltog[row - 1];
        v[2] = two*(hydhx + hxdhy) - sc*lambda*PetscExpScalar(x[row]); col[2] = grow;
        v[3] = -hydhx; col[3] = ltog[row + 1];
        v[4] = -hxdhy; col[4] = ltog[row + Xm];
        ierr = MatSetValues(jac,1,&grow,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
      } else if ((i > 0 && i < mx-1) || (j > 0 && j < my-1)) {
        value = .5*two*(hydhx + hxdhy);
        ierr  = MatSetValues(jac,1,&grow,1,&grow,&value,INSERT_VALUES);CHKERRQ(ierr);
      } else {
        value = .25*two*(hydhx + hxdhy);
        ierr  = MatSetValues(jac,1,&grow,1,&grow,&value,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = DMDARestoreGlobalIndices(grid->da,&nloc,&ltog);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecRestoreArray(localX,&x);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  return 0;
}

/*
        Computes the ENTIRE Jacobian associated with the ENTIRE grid sequentially
    This is for generating the coarse grid redundantly.

          This is BAD code duplication, since the bulk of this routine is the
    same as the routine above

       Note the numbering of the rows/columns is the NATURAL numbering
*/
#undef __FUNCT__
#define __FUNCT__ "FormJacobian_Coarse"
PetscErrorCode FormJacobian_Coarse(AppCtx *user,GridCtx *grid,Vec X, Mat J,Mat B)
{
  Mat            jac = J;
  PetscErrorCode ierr;
  PetscInt       i, j, row, mx, my, col[5];
  PetscScalar    two = 2.0, one = 1.0, lambda, v[5], hx, hy, hxdhy, hydhx, sc, *x, value;

  mx = grid->mx;            my = grid->my;            lambda = user->param;
  hx = one/(PetscReal)(mx-1);  hy = one/(PetscReal)(my-1);
  sc = hx*hy;               hxdhy = hx/hy;            hydhx = hy/hx;

  ierr = VecGetArray(X,&x);CHKERRQ(ierr);

  /* Evaluate Jacobian of function */
  for (j=0; j<my; j++) {
    row = j*mx - 1;
    for (i=0; i<mx; i++) {
      row++;
      if (i > 0 && i < mx-1 && j > 0 && j < my-1) {
        v[0] = -hxdhy; col[0] = row - mx;
        v[1] = -hydhx; col[1] = row - 1;
        v[2] = two*(hydhx + hxdhy) - sc*lambda*PetscExpScalar(x[row]); col[2] = row;
        v[3] = -hydhx; col[3] = row + 1;
        v[4] = -hxdhy; col[4] = row + mx;
        ierr = MatSetValues(jac,1,&row,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
      } else if ((i > 0 && i < mx-1) || (j > 0 && j < my-1)) {
        value = .5*two*(hydhx + hxdhy);
        ierr  = MatSetValues(jac,1,&row,1,&row,&value,INSERT_VALUES);CHKERRQ(ierr);
      } else {
        value = .25*two*(hydhx + hxdhy);
        ierr  = MatSetValues(jac,1,&row,1,&row,&value,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  return 0;
}

/* --------------------  Evaluate Jacobian F'(x) --------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormJacobian"
PetscErrorCode FormJacobian(SNES snes,Vec X,Mat J,Mat B,void *ptr)
{
  AppCtx         *user = (AppCtx*) ptr;
  PetscErrorCode ierr;
  KSP            ksp;
  PC             pc;
  PetscBool      ismg;

  ierr  = FormJacobian_Grid(user,&user->fine,X,J,B);CHKERRQ(ierr);

  /* create coarse grid jacobian for preconditioner */
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);

  ierr = PetscObjectTypeCompare((PetscObject)pc,PCMG,&ismg);CHKERRQ(ierr);
  if (ismg) {

    ierr = KSPSetOperators(user->ksp_fine,user->fine.J,user->fine.J);CHKERRQ(ierr);

    /* restrict X to coarse grid */
    ierr = MatMult(user->R,X,user->coarse.x);CHKERRQ(ierr);
    ierr = VecPointwiseMult(user->coarse.x,user->coarse.x,user->Rscale);CHKERRQ(ierr);

    /* form Jacobian on coarse grid */
    if (user->redundant_build) {
      /* get copy of coarse X onto each processor */
      ierr = VecScatterBegin(user->tolocalall,user->coarse.x,user->localall,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(user->tolocalall,user->coarse.x,user->localall,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = FormJacobian_Coarse(user,&user->coarse,user->localall,user->coarse.J,user->coarse.J);CHKERRQ(ierr);

    } else {
      /* coarse grid Jacobian computed in parallel */
      ierr = FormJacobian_Grid(user,&user->coarse,user->coarse.x,user->coarse.J,user->coarse.J);CHKERRQ(ierr);
    }
    ierr = KSPSetOperators(user->ksp_coarse,user->coarse.J,user->coarse.J);CHKERRQ(ierr);
  }

  return 0;
}


#undef __FUNCT__
#define __FUNCT__ "FormInterpolation"
/*
      Forms the interpolation (and restriction) operator from
coarse grid to fine.
*/
PetscErrorCode FormInterpolation(AppCtx *user)
{
  PetscErrorCode ierr;
  PetscInt       i,j,i_start,m_fine,j_start,m,n;
  const PetscInt *idx,*idx_c;
  PetscInt       m_ghost,n_ghost,m_ghost_c,n_ghost_c,m_coarse;
  PetscInt       row,i_start_ghost,j_start_ghost,cols[4], m_c;
  PetscInt       nc,ratio = user->ratio,m_c_local,m_fine_local;
  PetscInt       i_c,j_c,i_start_c,j_start_c,n_c,i_start_ghost_c,j_start_ghost_c,col;
  PetscScalar    v[4],x,y, one = 1.0;
  Mat            mat;
  Vec            Rscale;

  ierr = DMDAGetCorners(user->fine.da,&i_start,&j_start,0,&m,&n,0);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(user->fine.da,&i_start_ghost,&j_start_ghost,0,&m_ghost,&n_ghost,0);CHKERRQ(ierr);
  ierr = DMDAGetGlobalIndices(user->fine.da,NULL,&idx);CHKERRQ(ierr);

  ierr = DMDAGetCorners(user->coarse.da,&i_start_c,&j_start_c,0,&m_c,&n_c,0);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(user->coarse.da,&i_start_ghost_c,&j_start_ghost_c,0,&m_ghost_c,&n_ghost_c,0);CHKERRQ(ierr);
  ierr = DMDAGetGlobalIndices(user->coarse.da,NULL,&idx_c);CHKERRQ(ierr);

  /* create interpolation matrix */
  ierr = VecGetLocalSize(user->fine.x,&m_fine_local);CHKERRQ(ierr);
  ierr = VecGetLocalSize(user->coarse.x,&m_c_local);CHKERRQ(ierr);
  ierr = VecGetSize(user->fine.x,&m_fine);CHKERRQ(ierr);
  ierr = VecGetSize(user->coarse.x,&m_coarse);CHKERRQ(ierr);
  ierr = MatCreateAIJ(PETSC_COMM_WORLD,m_fine_local,m_c_local,m_fine,m_coarse,
                      5,0,3,0,&mat);CHKERRQ(ierr);

  /* loop over local fine grid nodes setting interpolation for those*/
  for (j=j_start; j<j_start+n; j++) {
    for (i=i_start; i<i_start+m; i++) {
      /* convert to local "natural" numbering and then to PETSc global numbering */
      row = idx[m_ghost*(j-j_start_ghost) + (i-i_start_ghost)];

      i_c = (i/ratio);    /* coarse grid node to left of fine grid node */
      j_c = (j/ratio);    /* coarse grid node below fine grid node */

      /*
         Only include those interpolation points that are truly
         nonzero. Note this is very important for final grid lines
         in x and y directions; since they have no right/top neighbors
      */
      x  = ((PetscReal)(i - i_c*ratio))/((PetscReal)ratio);
      y  = ((PetscReal)(j - j_c*ratio))/((PetscReal)ratio);
      nc = 0;
      /* one left and below; or we are right on it */
      if (j_c < j_start_ghost_c || j_c > j_start_ghost_c+n_ghost_c) SETERRQ3(PETSC_COMM_SELF,1,"Sorry j %D %D %D",j_c,j_start_ghost_c,j_start_ghost_c+n_ghost_c);
      if (i_c < i_start_ghost_c || i_c > i_start_ghost_c+m_ghost_c) SETERRQ3(PETSC_COMM_SELF,1,"Sorry i %D %D %D",i_c,i_start_ghost_c,i_start_ghost_c+m_ghost_c);
      col      = m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c);
      cols[nc] = idx_c[col];
      v[nc++]  = x*y - x - y + 1.0;
      /* one right and below */
      if (i_c*ratio != i) {
        cols[nc] = idx_c[col+1];
        v[nc++]  = -x*y + x;
      }
      /* one left and above */
      if (j_c*ratio != j) {
        cols[nc] = idx_c[col+m_ghost_c];
        v[nc++]  = -x*y + y;
      }
      /* one right and above */
      if (j_c*ratio != j && i_c*ratio != i) {
        cols[nc] = idx_c[col+m_ghost_c+1];
        v[nc++]  = x*y;
      }
      ierr = MatSetValues(mat,1,&row,nc,cols,v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = DMDARestoreGlobalIndices(user->fine.da,NULL,&idx);CHKERRQ(ierr);
  ierr = DMDARestoreGlobalIndices(user->fine.da,NULL,&idx_c);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr         = VecDuplicate(user->coarse.x,&Rscale);CHKERRQ(ierr);
  ierr         = VecSet(user->fine.x,one);CHKERRQ(ierr);
  ierr         = MatMultTranspose(mat,user->fine.x,Rscale);CHKERRQ(ierr);
  ierr         = VecReciprocal(Rscale);CHKERRQ(ierr);
  user->Rscale = Rscale;
  user->R      = mat;
  return 0;
}




