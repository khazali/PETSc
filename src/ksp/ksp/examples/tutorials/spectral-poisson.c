
static char help[] ="Solves a 2d Poisson problem using the spectral element method\n\n";

#include <petscksp.h>
#include <petscgll.h>
#include <petscdraw.h>
#include <petscdmda.h>
#include <petscblaslapack.h>
#include <petsc/private/petscimpl.h>

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines.
*/

typedef struct {
  PetscInt    N;             /* grid points per elements*/
  PetscInt    Ex;              /* number of elements */
  PetscInt    Ey;              /* number of elements */
  PetscReal   Lx;              /* total length of domain */ 
  PetscReal   Ly;              /* total length of domain */     
  PetscReal   Lex; 
  PetscReal   Ley; 
  PetscInt    lenx;
  PetscInt    leny;
  PetscReal   mu;
} PetscParam;


typedef struct {
  Vec         grid;              /* total grid */   
  Vec         mass;              /* mass matrix for total integration */
  Mat         stiff;             /* stifness matrix */
  PetscGLL    gll;
} PetscSEMOperators;

typedef struct {
  DM                da;                /* distributed array data structure */
  PetscSEMOperators SEMop;
  PetscParam        param;
} AppCtx;

/*
   User-defined routines
*/
extern PetscErrorCode MyMatMult(Mat,Vec,Vec);
extern PetscErrorCode MyMatCreateSubMatrices(Mat,PetscInt,const IS[],const IS[],MatReuse,Mat*[]);
extern PetscErrorCode TestMult(Mat);

int main(int argc,char **argv)
{
  AppCtx         appctx;                 /* user-defined application context */
  Vec            u,b;                      /* approximate solution vector */
  PetscErrorCode ierr;
  PetscInt       xs, xm, ys,ym, ix,iy;
  PetscInt       indx,indy,m, nn;
  PetscReal      x,y;
  PetscScalar    **bmass;
  DMDACoor2d     **coors;
  Vec            global,loc;
  DM             cda;
  PetscInt       jx,jy,num_domains,*indices,cnt,dcnt;
  Mat            H;
  MatNullSpace   nsp;
  KSP            ksp;
  PC             pc;
  IS             *domains;
  AO             ao;

   /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program and set problem parameters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscFunctionBegin;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /*initialize parameters */
  appctx.param.N     = 8;  /* order of the spectral element */
  appctx.param.Ex    = 6;  /* number of elements */
  appctx.param.Ey    = 6;  /* number of elements */
  appctx.param.Lx    = 4.0;  /* length of the domain */
  appctx.param.Ly    = 4.0;  /* length of the domain */
  appctx.param.mu   = 0.005; /* diffusion coefficient */

  ierr = PetscOptionsGetInt(NULL,NULL,"-N",&appctx.param.N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-Ex",&appctx.param.Ex,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-Ey",&appctx.param.Ey,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-mu",&appctx.param.mu,NULL);CHKERRQ(ierr);
  appctx.param.Lex = appctx.param.Lx/appctx.param.Ex;
  appctx.param.Ley = appctx.param.Ly/appctx.param.Ey;


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create GLL data structures
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscGLLCreate(appctx.param.N,PETSCGLL_VIA_LINEARALGEBRA,&appctx.SEMop.gll);CHKERRQ(ierr);
  
  appctx.param.lenx = appctx.param.Ex*(appctx.param.N-1);
  appctx.param.leny = appctx.param.Ey*(appctx.param.N-1);

  /*
     Create distributed array (DMDA) to manage parallel grid and vectors
     and to set up the ghost point communication pattern.  There are E*(Nl-1)+1
     total grid values spread equally among all the processors, except first and last
  */

  
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DMDA_STENCIL_BOX,appctx.param.lenx,appctx.param.leny,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&appctx.da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(appctx.da);CHKERRQ(ierr);
  ierr = DMSetUp(appctx.da);CHKERRQ(ierr);
  
  /*
     Extract global and local vectors from DMDA; we use these to store the
     approximate solution.  Then duplicate these for remaining vectors that
     have the same types.
  */

  ierr = DMCreateGlobalVector(appctx.da,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.SEMop.mass);CHKERRQ(ierr);
 
  ierr = DMDAGetCorners(appctx.da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  /* Compute function over the locally owned part of the grid */
  /* scale the corners so they represent ELEMENTS not grid points */
  if (xs % (appctx.param.N-1)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"xs is not divisible by element size minus one");
  xs=xs/(appctx.param.N-1);
  if (xm % (appctx.param.N-1)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"xm is not divisible by element size minus one");
  xm=xm/(appctx.param.N-1);
  if (ys % (appctx.param.N-1)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"ys is not divisible by element size minus one");
  ys=ys/(appctx.param.N-1);
  if (ym % (appctx.param.N-1)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"ym is not divisible by element size minus one");
  ym=ym/(appctx.param.N-1);

  VecSet(appctx.SEMop.mass,0.0);

  DMGetLocalVector(appctx.da,&loc);
  ierr = DMDAVecGetArray(appctx.da,loc,&bmass);CHKERRQ(ierr);

  /*
     Build mass over entire mesh (multi-elemental) 
  */ 

  for (ix=xs; ix<xs+xm; ix++) {
    for (jx=0; jx<appctx.param.N; jx++) {
      for (iy=ys; iy<ys+ym; iy++) {
        for (jy=0; jy<appctx.param.N; jy++){   
          x = (appctx.param.Lex/2.0)*(appctx.SEMop.gll.nodes[jx]+1.0)+appctx.param.Lex*ix; 
          y = (appctx.param.Ley/2.0)*(appctx.SEMop.gll.nodes[jy]+1.0)+appctx.param.Ley*iy; 
          indx=ix*(appctx.param.N-1)+jx;
          indy=iy*(appctx.param.N-1)+jy;
          bmass[indy][indx] +=appctx.SEMop.gll.weights[jx]*appctx.SEMop.gll.weights[jy]*.25*appctx.param.Ley*appctx.param.Lex;
        } 
      }
    }
  }

  DMDAVecRestoreArray(appctx.da,loc,&bmass);CHKERRQ(ierr);
  DMLocalToGlobalBegin(appctx.da,loc,ADD_VALUES,appctx.SEMop.mass);
  DMLocalToGlobalEnd(appctx.da,loc,ADD_VALUES,appctx.SEMop.mass); 
  DMRestoreLocalVector(appctx.da,&loc);
  DMDASetUniformCoordinates(appctx.da,0.0,appctx.param.Lx,0.0,appctx.param.Ly,0.0,0.0);
  DMGetCoordinateDM(appctx.da,&cda);
  DMGetCoordinates(appctx.da,&global);
  VecSet(global,0.0);
  DMDAVecGetArray(cda,global,&coors);
  for (ix=xs; ix<xs+xm; ix++) {
    for (jx=0; jx<appctx.param.N-1; jx++) {
      for (iy=ys; iy<ys+ym; iy++) {
        for (jy=0; jy<appctx.param.N-1; jy++)   {
          x = (appctx.param.Lex/2.0)*(appctx.SEMop.gll.nodes[jx]+1.0)+appctx.param.Lex*ix-2.0; 
          y = (appctx.param.Ley/2.0)*(appctx.SEMop.gll.nodes[jy]+1.0)+appctx.param.Ley*iy-2.0;
          indx=ix*(appctx.param.N-1)+jx;
          indy=iy*(appctx.param.N-1)+jy;
          coors[indy][indx].x=x;
          coors[indy][indx].y=y;
        } 
      }
    }
  }
  DMDAVecRestoreArray(cda,global,&coors);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Create matrix data structure; set matrix evaluation routine.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  VecGetLocalSize(u,&m);
  VecGetSize(u,&nn);

  MatCreateShell(PETSC_COMM_WORLD,m,m,nn,nn,&appctx,&H);
  MatShellSetOperation(H,MATOP_MULT,(void(*)(void))MyMatMult);
  MatShellSetOperation(H,MATOP_CREATE_SUBMATRICES,(void(*)(void))MyMatCreateSubMatrices);  
  
 /* attach the null space to the matrix, this probably is not needed but does no harm */
  
  ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,NULL,&nsp);CHKERRQ(ierr);
  ierr = MatSetNullSpace(H,nsp);CHKERRQ(ierr);
  ierr = MatNullSpaceTest(nsp,H,NULL);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nsp);CHKERRQ(ierr);

  ierr = TestMult(H);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetDM(ksp,appctx.da);CHKERRQ(ierr);
  ierr = KSPSetDMActive(ksp,PETSC_FALSE);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&b);CHKERRQ(ierr);
  ierr = VecSetRandom(b,PETSC_RANDOM_(PETSC_COMM_WORLD));CHKERRQ(ierr);
  ierr = VecSet(b,1.0);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,H,H);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);


  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = DMDAGetAO(appctx.da,&ao);CHKERRQ(ierr);
  /*  Each element is its own subdomain for additive Schwarz */
  num_domains = xm*ym;
  ierr = PetscMalloc1(num_domains,&domains);CHKERRQ(ierr);
  ierr = PetscMalloc1(appctx.param.N*appctx.param.N,&indices);CHKERRQ(ierr);
  dcnt = 0;
  for (ix=xs; ix<xs+xm; ix++) {
    for (iy=ys; iy<ys+ym; iy++) {
      cnt = 0;
      for (jx=0; jx<appctx.param.N; jx++) {
        for (jy=0; jy<appctx.param.N; jy++)   {
          indx=ix*(appctx.param.N-1)+jx;
          indy=iy*(appctx.param.N-1)+jy;
          if (indx == appctx.param.lenx) indx = 0;
          if (indy == appctx.param.leny) indy = 0;
          indices[cnt++] = indx + indy*appctx.param.lenx;
        }
      }
      ierr = AOApplicationToPetsc(ao,cnt,indices);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PETSC_COMM_SELF,cnt,indices,PETSC_COPY_VALUES,&domains[dcnt++]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(indices);CHKERRQ(ierr);
  ierr = PCASMSetLocalSubdomains(pc, num_domains, domains,domains);CHKERRQ(ierr);
  for (dcnt=0; dcnt<num_domains; dcnt++) {
    ierr = ISDestroy(&domains[dcnt]);CHKERRQ(ierr);
  }
  ierr = PetscFree(domains);CHKERRQ(ierr);

  ierr = PCASMSetType(pc,PC_ASM_BASIC);CHKERRQ(ierr);
  ierr = PCASMSetOverlap(pc,0);CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);

  ierr = KSPSolve(ksp,b,u);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&H);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.SEMop.mass);CHKERRQ(ierr);
  ierr = PetscGLLDestroy(&appctx.SEMop.gll);CHKERRQ(ierr);
  ierr = DMDestroy(&appctx.da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode TestMult(Mat H)
{
  PetscErrorCode ierr;
  Vec            in,out,global;
  AppCtx         *appctx;
  DMDACoor2d     **coors;
  PetscScalar    **invalues,x,y;
  DM             cda;
  PetscInt       xs,ys,xm,ym,ix,iy;

  ierr = MatShellGetContext(H,&appctx);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(appctx->da,&in);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(appctx->da,&out);CHKERRQ(ierr);  
  ierr = DMGetCoordinates(appctx->da,&global);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(appctx->da,&cda);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cda,global,&coors);CHKERRQ(ierr);
  ierr = DMDAGetCorners(appctx->da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(appctx->da,in,&invalues);CHKERRQ(ierr);
  for (ix=xs; ix<xs+xm; ix++) {
    for (iy=ys; iy<ys+ym; iy++) {
      x = coors[iy][ix].x;
      y = coors[iy][ix].y;
      invalues[iy][ix] = PetscSinScalar(PETSC_PI*x);
    }
  }
  ierr = DMDAVecRestoreArray(appctx->da,in,&invalues);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(cda,global,&coors);CHKERRQ(ierr);
  ierr = VecView(in,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  ierr = MatMult(H,in,out);CHKERRQ(ierr);
  ierr = VecView(out,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  ierr = VecDestroy(&in);CHKERRQ(ierr);
  ierr = VecDestroy(&out);CHKERRQ(ierr);  
  return 0;
}

PetscErrorCode MyMatMult(Mat H, Vec in, Vec out)
 {
   AppCtx         *appctx;
   PetscScalar    **outl;
   PetscScalar     **stiff, **mass;
   PetscScalar     **wrk1, **wrk2, **wrk3;
   PetscScalar     **ulb, **ul;
   PetscInt        Nl, Nl2, inc;
   PetscInt        xs,ys,xm,ym,ix,iy,jx,jy, indx,indy, i;
   PetscErrorCode  ierr;
   Vec             uloc, outloc;
   PetscScalar     alpha, beta;
 
  MatShellGetContext(H,&appctx);

  ierr = PetscGLLElementLaplacianCreate(&appctx->SEMop.gll,&stiff);CHKERRQ(ierr);
  ierr = PetscGLLElementMassCreate(&appctx->SEMop.gll,&mass);CHKERRQ(ierr); 

  /* unwrap local vector for the input solution */
  DMGetLocalVector(appctx->da,&uloc);

  DMGlobalToLocalBegin(appctx->da,in,INSERT_VALUES,uloc);
  DMGlobalToLocalEnd(appctx->da,in,INSERT_VALUES,uloc);

  DMDAVecGetArrayRead(appctx->da,uloc,&ul);CHKERRQ(ierr);

  /* unwrap local vector for the output solution */
  DMGetLocalVector(appctx->da,&outloc);
  VecSet(outloc,0.0);

  ierr = DMDAVecGetArray(appctx->da,outloc,&outl);CHKERRQ(ierr);
 
  ierr = DMDAGetCorners(appctx->da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  Nl    = appctx->param.N; 
    
  xs=xs/(Nl-1);
  xm=xm/(Nl-1);
  ys=ys/(Nl-1);
  ym=ym/(Nl-1);
  
  /*
     Initialize work arrays
  */ 

  ierr = PetscMalloc1(appctx->param.N,&ulb);CHKERRQ(ierr);
  ierr = PetscMalloc1(appctx->param.N*appctx->param.N,&ulb[0]);CHKERRQ(ierr);
  for (i=1; i<Nl; i++) ulb[i] = ulb[i-1]+Nl;

  ierr = PetscMalloc1(appctx->param.N,&wrk1);CHKERRQ(ierr);
  ierr = PetscMalloc1(appctx->param.N*appctx->param.N,&wrk1[0]);CHKERRQ(ierr);
  for (i=1; i<Nl; i++) wrk1[i] = wrk1[i-1]+Nl;

  ierr = PetscMalloc1(appctx->param.N,&wrk2);CHKERRQ(ierr);
  ierr = PetscMalloc1(appctx->param.N*appctx->param.N,&wrk2[0]);CHKERRQ(ierr);
  for (i=1; i<Nl; i++) wrk2[i] = wrk2[i-1]+Nl;

  ierr = PetscMalloc1(appctx->param.N,&wrk3);CHKERRQ(ierr);
  ierr = PetscMalloc1(appctx->param.N*appctx->param.N,&wrk3[0]);CHKERRQ(ierr);
  for (i=1; i<Nl; i++) wrk3[i] = wrk3[i-1]+Nl;

  alpha = 1.0;
  beta  = 0.0;
  Nl2= Nl*Nl;
  inc=1;
  for (ix=xs; ix<xs+xm; ix++) {
    for (iy=ys; iy<ys+ym; iy++) {
      for (jx=0; jx<appctx->param.N; jx++) {
        for (jy=0; jy<appctx->param.N; jy++){
          indx=ix*(Nl-1)+jx;
          indy=iy*(Nl-1)+jy;
          ulb[jy][jx]=ul[indy][indx];
        }
      }

      //here the stifness matrix in 2d
      //first product (B x K_yy) u=W2 (u_yy)
      alpha=appctx->param.Lex/2.0;
      BLASgemm_("N","N",&Nl,&Nl,&Nl,&alpha,&mass[0][0],&Nl,&ulb[0][0],&Nl,&beta,&wrk1[0][0],&Nl);
      alpha=2./appctx->param.Ley;
      BLASgemm_("N","T",&Nl,&Nl,&Nl,&alpha,&wrk1[0][0],&Nl,&stiff[0][0],&Nl,&beta,&wrk2[0][0],&Nl);
      
      //second product (K_xx x B) u=W3 (u_xx)
      alpha=2.0/appctx->param.Lex;
      BLASgemm_("N","N",&Nl,&Nl,&Nl,&alpha,&stiff[0][0],&Nl,&ulb[0][0],&Nl,&beta,&wrk1[0][0],&Nl);
      alpha=appctx->param.Ley/2.0;
      BLASgemm_("N","T",&Nl,&Nl,&Nl,&alpha,&wrk1[0][0],&Nl,&mass[0][0],&Nl,&beta,&wrk3[0][0],&Nl);

      alpha=1.0;
      BLASaxpy_(&Nl2,&alpha, &wrk3[0][0],&inc,&wrk2[0][0],&inc); //I freed wrk3 and saved the lalplacian in wrk2

      for (jx=0; jx<appctx->param.N; jx++) {
        for (jy=0; jy<appctx->param.N; jy++)   {
          indx=ix*(appctx->param.N-1)+jx;
          indy=iy*(appctx->param.N-1)+jy;
          
          outl[indy][indx] += appctx->param.mu*(wrk2[jy][jx]);
        }
      }
    }
  }
  
  ierr = DMDAVecRestoreArray(appctx->da,outloc,&outl);CHKERRQ(ierr);
  DMDAVecRestoreArrayRead(appctx->da,uloc,&ul);CHKERRQ(ierr);

  VecSet(out,0.0);

  DMLocalToGlobalBegin(appctx->da,outloc,ADD_VALUES,out);
  DMLocalToGlobalEnd(appctx->da,outloc,ADD_VALUES,out);

  VecScale(out, -1.0);
  ierr = VecPointwiseDivide(out,out,appctx->SEMop.mass);CHKERRQ(ierr);

  DMRestoreLocalVector(appctx->da,&uloc);
  DMRestoreLocalVector(appctx->da,&outloc);
  
  ierr = PetscGLLElementLaplacianDestroy(&appctx->SEMop.gll,&stiff);CHKERRQ(ierr);
  ierr = PetscGLLElementMassDestroy(&appctx->SEMop.gll,&mass);CHKERRQ(ierr);
 
  ierr = PetscFree((ulb)[0]);CHKERRQ(ierr);
  ierr = PetscFree(ulb);CHKERRQ(ierr);
  ierr = PetscFree((wrk1)[0]);CHKERRQ(ierr);
  ierr = PetscFree(wrk1);CHKERRQ(ierr);
  ierr = PetscFree((wrk2)[0]);CHKERRQ(ierr);
  ierr = PetscFree(wrk2);CHKERRQ(ierr);
  ierr = PetscFree((wrk3)[0]);CHKERRQ(ierr);
  ierr = PetscFree(wrk3);CHKERRQ(ierr);
  return 0;
}

PetscErrorCode MyMatCreateSubMatrices(Mat mat,PetscInt n,const IS irow[],const IS icol[],MatReuse scall,Mat *submat[])
{
  Mat            *subs;
  PetscErrorCode ierr;
  PetscInt       i,m;

  ierr = PetscMalloc1(n,&subs);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = MatCreate(PETSC_COMM_SELF,&subs[i]);CHKERRQ(ierr);
    ierr = ISGetSize(irow[i],&m);CHKERRQ(ierr);
    ierr = MatSetSizes(subs[i],m,m,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = MatSetType(subs[i],MATSEQAIJ);CHKERRQ(ierr);
    ierr = MatSetUp(subs[i]);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(subs[i],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(subs[i],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);    
    ierr = MatShift(subs[i],1.0);CHKERRQ(ierr);
  }
  *submat = subs;
  return 0;
}

/*TEST

   build:
     requires: !complex

   test:
     requires: !single
     args: -tao_monitor  -ts_adapt_dt_max 3.e-3 -E 10 -N 8 -ncoeff 5 
     TODO: example needs work

   test:
     suffix: cn
     requires: !single
     args: -tao_monitor -ts_type cn -ts_dt .003 -pc_type lu -E 10 -N 8 -ncoeff 5 
     TODO: example needs work

TEST*/
