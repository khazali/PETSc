static char help[] = "Oregonator Reaction-Diffusion Equation.\n";

/*

  Here, we modify ex5.c, replacing the chemical reaction with the 2D Oregonator

  Reference: Jahnke, Skaggs, and Winfree 1989
  "Chemical Vortex Dynamics in the Belousov-Zhabotinsky Reaction and the Two-Variable Oregonator Model"

   Helpful runtime monitor options:
         -ts_monitor_draw_solution
         -ts_monitor
         -da_grid_x 
         -da_grid_y 
*/

#include <petscdmda.h>
#include <petscts.h>

#define NUMSPECIES 2

typedef struct {
  PetscScalar u,v;
} Field;

typedef struct {
  PetscScalar epsilon,q,f,D1,D2,boxSize;
} AppCtx;

/*
   User-defined routines
*/
extern PetscErrorCode RHSFunction(TS,PetscReal,Vec,Vec,void*);
extern PetscErrorCode InitialConditions(DM,Vec,void*);
extern PetscErrorCode IFunction(TS,PetscReal,Vec,Vec,Vec,void*);
extern PetscErrorCode RHSJacobian(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*);
extern PetscErrorCode IJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat*,Mat*,MatStructure*,void*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  TS             ts;                  /* ODE integrator */
  Vec            x;                   /* solution */
  PetscErrorCode ierr;
  DM             da;
  AppCtx         appctx;
  
  PetscBool      setJac = PETSC_TRUE;

  PetscInitialize(&argc,&argv,(char*)0,help);

  appctx.boxSize = 2.50;

  /* arbitrarily chosen diffusion constants */
  appctx.D1    = 0.01; 
  appctx.D2    = 0.01;
  appctx.f       = 1.4;
  appctx.q       = 0.002;
  appctx.epsilon = 0.01;
  
  ierr = PetscOptionsGetReal(NULL,"-f",&appctx.f,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,"-q",&appctx.q,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,"-epsilon",&appctx.epsilon,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,"-setjac",&setJac,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,"-D1",&appctx.D1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,"-D2",&appctx.D2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,"-boxsize",&appctx.boxSize,NULL);CHKERRQ(ierr);


  ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_PERIODIC,DMDA_BOUNDARY_PERIODIC,DMDA_STENCIL_STAR,-10,-10,PETSC_DECIDE,PETSC_DECIDE,NUMSPECIES,1,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,0,"u");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,1,"v");CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSROSW);CHKERRQ(ierr);
  ierr = TSSetDM(ts,da);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,RHSFunction,&appctx);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,IFunction,&appctx);CHKERRQ(ierr);
  if(setJac){
    ierr = TSSetIJacobian(ts,NULL,NULL,IJacobian,&appctx);CHKERRQ(ierr);
    ierr = TSSetRHSJacobian(ts,NULL,NULL,RHSJacobian,&appctx);CHKERRQ(ierr);
  }
  ierr = TSSetMaxSNESFailures(ts,-1); /* unlimited failures, as this is a long, stiff solve */

  ierr = InitialConditions(da,x,&appctx);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,x);CHKERRQ(ierr);

  ierr = TSSetDuration(ts,PETSC_MAX_INT,6.0);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = TSSolve(ts,x);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);

  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IFunction"
PetscErrorCode IFunction(TS ts,PetscReal ftime,Vec U,Vec Udot, Vec F,void *ptr)
{
  AppCtx         *appctx = (AppCtx*)ptr;
  DM             da;
  PetscErrorCode ierr;
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  PetscReal      hx,hy,sx,sy;
  Field          **u,**udot,**f;
  PetscScalar    uc,vc;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  hx = appctx->boxSize/(PetscReal)(Mx); sx = 1.0/(hx*hx);
  hy = appctx->boxSize/(PetscReal)(My); sy = 1.0/(hy*hy);

  ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,Udot,&udot);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);

  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
       uc = u[j][i].u; 
       vc = u[j][i].v;    
      f[j][i].u = udot[j][i].u - (uc - uc*uc - appctx->f*vc*(uc - appctx->q)/(uc + appctx->q))/appctx->epsilon;
      f[j][i].v = udot[j][i].v - (uc-vc);
    }
  }

  ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,Udot,&udot);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RHSFunction"
PetscErrorCode RHSFunction(TS ts,PetscReal ftime,Vec U,Vec F,void *ptr)
{
  AppCtx         *appctx = (AppCtx*)ptr;
  DM             da;
  PetscErrorCode ierr;
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  PetscReal      hx,hy,sx,sy;
  PetscScalar    uc,uxx,uyy,vc,vxx,vyy; 
  Field          **u,**f;
  Vec            localU;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  hx = appctx->boxSize/(PetscReal)(Mx); sx = 1.0/(hx*hx);
  hy = appctx->boxSize/(PetscReal)(My); sy = 1.0/(hy*hy);

  ierr = DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(da,localU,&u);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);

  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      uc        = u[j][i].u;
      uxx       = (-2.0*uc + u[j][i-1].u + u[j][i+1].u)*sx;
      uyy       = (-2.0*uc + u[j-1][i].u + u[j+1][i].u)*sy;
      vc        = u[j][i].v;
      vxx       = (-2.0*vc + u[j][i-1].v + u[j][i+1].v)*sx;
      vyy       = (-2.0*vc + u[j-1][i].v + u[j+1][i].v)*sy;
      
      f[j][i].u = appctx->D1*(uxx + uyy) ; 
      f[j][i].v = appctx->D2*(vxx + vyy) ;
    }
  }

  ierr = DMDAVecRestoreArray(da,localU,&u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localU);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RHSJacobian"
PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec U,Mat *AA,Mat *BB,MatStructure *str,void *ctx)
{
  Mat            A       = *AA;
  AppCtx         *appctx = (AppCtx*)ctx;
  DM             da;
  PetscErrorCode ierr;
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  PetscReal      hx,hy,sx,sy;
  Field          **u;
  Vec            localU;
  MatStencil     stencil[6],rowstencil;
  PetscScalar    entries[6];

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  hx = 2.50/(PetscReal)(Mx); sx = 1.0/(hx*hx);
  hy = 2.50/(PetscReal)(My); sy = 1.0/(hy*hy);

  ierr = DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(da,localU,&u);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  stencil[0].k = 0;
  stencil[1].k = 0;
  stencil[2].k = 0;
  stencil[3].k = 0;
  stencil[4].k = 0;
  stencil[5].k = 0;
  rowstencil.k = 0;
  for (j=ys; j<ys+ym; j++) {

    stencil[0].j = j-1;
    stencil[1].j = j+1;
    stencil[2].j = j;
    stencil[3].j = j;
    stencil[4].j = j;
    stencil[5].j = j;
    rowstencil.k = 0; rowstencil.j = j;
    for (i=xs; i<xs+xm; i++) {

      stencil[0].i = i;   stencil[0].c = 0; entries[0] = appctx->D1*sy;
      stencil[1].i = i;   stencil[1].c = 0; entries[1] = appctx->D1*sy;
      stencil[2].i = i-1; stencil[2].c = 0; entries[2] = appctx->D1*sx;
      stencil[3].i = i+1; stencil[3].c = 0; entries[3] = appctx->D1*sx;
      stencil[4].i = i;   stencil[4].c = 0; entries[4] = -2.0*appctx->D1*(sx + sy) ; 
      stencil[5].i = i;   stencil[5].c = 1; entries[5] = 0; 
      rowstencil.i = i; rowstencil.c = 0;

      ierr = MatSetValuesStencil(A,1,&rowstencil,6,stencil,entries,INSERT_VALUES);CHKERRQ(ierr);

      stencil[0].c = 1; entries[0] = appctx->D2*sy;
      stencil[1].c = 1; entries[1] = appctx->D2*sy;
      stencil[2].c = 1; entries[2] = appctx->D2*sx;
      stencil[3].c = 1; entries[3] = appctx->D2*sx;
      stencil[4].c = 0; entries[4] = 0; 
      stencil[5].c = 1; entries[5] = -2.0*appctx->D2*(sx + sy) ; 
      rowstencil.c = 1;

      ierr = MatSetValuesStencil(A,1,&rowstencil,6,stencil,entries,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = DMDAVecRestoreArray(da,localU,&u);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *str = SAME_NONZERO_PATTERN;
  ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "IJacobian"
PetscErrorCode IJacobian(TS ts,PetscReal t,Vec U,Vec Udot, PetscReal a, Mat *AA,Mat *BB,MatStructure *str,void *ctx)
{
  Mat            A       = *AA;             
  AppCtx         *appctx = (AppCtx*)ctx;   
  DM             da;
  PetscErrorCode ierr;
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  PetscReal      hx,hy,sx,sy;
  Field          **u;
  MatStencil     stencil[6],rowstencil;
  PetscScalar    entries[6],uc,vc,fac;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  hx = 2.50/(PetscReal)(Mx); sx = 1.0/(hx*hx);
  hy = 2.50/(PetscReal)(My); sy = 1.0/(hy*hy);

  ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  stencil[0].k = 0;
  stencil[1].k = 0;
  stencil[2].k = 0;
  stencil[3].k = 0;
  stencil[4].k = 0;
  stencil[5].k = 0;
  rowstencil.k = 0;
  for (j=ys; j<ys+ym; j++) {

    stencil[0].j = j-1;
    stencil[1].j = j+1;
    stencil[2].j = j;
    stencil[3].j = j;
    stencil[4].j = j;
    stencil[5].j = j;
    rowstencil.k = 0; rowstencil.j = j;
    for (i=xs; i<xs+xm; i++) {
      uc = u[j][i].u;
      vc = u[j][i].v;
      fac = uc + appctx->q;

      stencil[0].i = i;   stencil[0].c = 0; entries[0] = 0;
      stencil[1].i = i;   stencil[1].c = 0; entries[1] = 0;
      stencil[2].i = i-1; stencil[2].c = 0; entries[2] = 0;
      stencil[3].i = i+1; stencil[3].c = 0; entries[3] = 0;
      stencil[4].i = i;   stencil[4].c = 0; entries[4] = a - (1 - 2*uc - 2*appctx->f*vc*appctx->q/(fac*fac))/appctx->epsilon;
      stencil[5].i = i;   stencil[5].c = 1; entries[5] =   - (appctx->f * (uc - appctx->q)/fac)/appctx->epsilon;

      rowstencil.i = i; rowstencil.c = 0;

      ierr = MatSetValuesStencil(A,1,&rowstencil,6,stencil,entries,INSERT_VALUES);CHKERRQ(ierr);

      stencil[0].c = 1; entries[0] = 0;
      stencil[1].c = 1; entries[1] = 0;
      stencil[2].c = 1; entries[2] = 0;
      stencil[3].c = 1; entries[3] = 0;
      stencil[4].c = 0; entries[4] = -1.0; 
      stencil[5].c = 1; entries[5] = a + 1.0; 
      rowstencil.c = 1;

      ierr = MatSetValuesStencil(A,1,&rowstencil,6,stencil,entries,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *str = SAME_NONZERO_PATTERN;
  ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "InitialConditions"
PetscErrorCode InitialConditions(DM da,Vec U,void *ptr)
{
  PetscErrorCode ierr;
  AppCtx         *appctx = (AppCtx*)ptr;
  PetscInt       i,j,xs,ys,xm,ym,Mx,My;
  Field          **u;
  PetscReal      hx,hy;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

  hx = appctx->boxSize/(PetscReal)(Mx);
  hy = appctx->boxSize/(PetscReal)(My);

  ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

 //!! Might be better to have a smother initial distribution

  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      if(!rank && j==(ys+ym)/2 && i==(xs+xm)/2){ /* note int division */
        u[j][i].u =  1; 
        u[j][i].v =  0.1;
      }else{
        u[j][i].u =  0; 
        u[j][i].v =  0;
      }
    }
  }

  ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
