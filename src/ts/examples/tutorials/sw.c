#include <petsctao.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>

#define sbc_uv(a) ((a)<0 ? (-(a)-1) : (a))
#define nbc_uv(a) ((a)>My-1 ? (2*My-(a)-1) : (a))
#define sbc_h(a) ((a)<0 ? 0 : (a))
#define nbc_h(a) ((a)>My-1 ? (My-1) : (a))

typedef struct {
  PetscScalar u,v,h;
} Field;

typedef struct {
  PetscReal EarthRadius;
  PetscReal Gravity;
  PetscReal AngularSpeed;
  PetscReal alpha,phi;
  PetscInt  p;
  PetscInt  q;
  PetscReal sqrtru,sqrtrv,sqrtrh;
  PetscInt  nobs;
  TS        ts;
  PetscBool be; /* flag to add background error */
  Vec       Uf; /* forecast state */
  Vec       U;  /* working vector */
  Mat       corX2d,corY2d;
} Model_SW;


/*
  Apply the operator to transform simulation results to obvervations. Normally interpolation is needed.
 */
PetscErrorCode ApplyObservation(DM da,Vec Uob)
{
  PetscInt       i,j,xs,ys,xm,ym;
  Field          **uarr;
  PetscErrorCode ierr;

  ierr = DMDAVecGetArray(da,Uob,&uarr);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++) { /* latitude */
    for (i=xs; i<xs+xm; i++) { /* longitude */
      if ((i%10!=0 && j%10!=0)) {
        uarr[j][i].u = 0;
        uarr[j][i].v = 0;
        uarr[j][i].h = 0;
      }
    }
  }
  ierr = DMDAVecRestoreArray(da,Uob,&uarr);CHKERRQ(ierr);
  return 0;
}

/*
  Build the background error covariance
  B = Sigma*CorX2d*CorY2d*Sigma
  where
    CorX2d = CorX1d X I
    CorY2d = I X CorY1d
    Sigma is a diagonal scaling matrix

*/
PetscErrorCode BuildCovariance(DM da,Model_SW *sw)
{
  MPI_Comm       comm;
  Mat            corX1d,corY1d,corX1dsc,corY1dsc;
  PetscInt       i,j,xs,ys,nx,ny,Mx,My,maxlen;
  PetscScalar    *y1dscarr,*x1dscarr;
  PetscReal      dist,value,theta;
  PetscInt       *colidxs,*rowidxs;
  PetscErrorCode ierr;

  theta = 0.2;
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,0,0,0,&nx,&ny,NULL);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)da,&comm);CHKERRQ(ierr);

  /* Create 1d correlation matrix (dense) */
  ierr = MatCreateSeqDense(comm,Mx,Mx,NULL,&corX1d);CHKERRQ(ierr);
  ierr = MatSetUp(corX1d);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(comm,My,My,NULL,&corY1d);CHKERRQ(ierr);
  ierr = MatSetUp(corY1d);CHKERRQ(ierr);

  /* Create 2d correlation matrix (sparse) */
  ierr = MatCreate(comm,&sw->corX2d);CHKERRQ(ierr);
  ierr = MatSetSizes(sw->corX2d,PETSC_DETERMINE,PETSC_DETERMINE,3*Mx*My,3*Mx*My);CHKERRQ(ierr);
  ierr = MatSetType(sw->corX2d,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(sw->corX2d);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(sw->corX2d,Mx,NULL);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(sw->corX2d,"X2d_");CHKERRQ(ierr);
  ierr = MatSetUp(sw->corX2d);CHKERRQ(ierr);
  ierr = MatCreate(comm,&sw->corY2d);CHKERRQ(ierr);
  ierr = MatSetSizes(sw->corY2d,PETSC_DETERMINE,PETSC_DETERMINE,3*Mx*My,3*Mx*My);CHKERRQ(ierr);
  ierr = MatSetType(sw->corY2d,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(sw->corY2d);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(sw->corY2d,My,NULL);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(sw->corY2d,"Y2d_");CHKERRQ(ierr);
  ierr = MatSetUp(sw->corY2d);CHKERRQ(ierr);

  maxlen = PetscMax(Mx,My);
  ierr = PetscCalloc2(maxlen,&rowidxs,maxlen,&colidxs);CHKERRQ(ierr);

  for (i=0;i<Mx;i++) {
    for (j=0;j<Mx;j++) {
      dist = PetscMin(PetscAbs(i-j),Mx-PetscAbs(i-j));
      value = (1.-theta)*PetscExpReal(-dist*dist);
      if (i==j) value += theta;
      ierr = MatSetValues(corX1d,1,&i,1,&j,&value,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  for (i=0;i<My;i++) {
    for (j=0;j<My;j++) {
      value = (1.-theta)*PetscExpReal(-(i-j)*(i-j));
      if (i==j) value += theta;
      ierr = MatSetValues(corY1d,1,&i,1,&j,&value,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(corX1d,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(corX1d,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(corY1d,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(corY1d,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Kronecker product of corX1d and I(My,My), duplicated for the three fields  */
  ierr = MatDuplicate(corX1d,MAT_COPY_VALUES,&corX1dsc);CHKERRQ(ierr);
  ierr = MatDenseGetArray(corX1dsc,&x1dscarr);CHKERRQ(ierr);
  for (i=0;i<Mx*Mx;i++) x1dscarr[i] *=  sw->sqrtru;
  for (ys=0;ys<My;ys++) { /* u */
    for (i=0;i<Mx;i++) rowidxs[i] = (i*My+ys)*3;
    for (i=0;i<Mx;i++) colidxs[i] = (i*My+ys)*3;
    ierr = MatSetValues(sw->corX2d,Mx,rowidxs,Mx,colidxs,x1dscarr,INSERT_VALUES);CHKERRQ(ierr);
  }
  for (i=0;i<Mx*Mx;i++) x1dscarr[i] *=  sw->sqrtrv/sw->sqrtru;
  for (ys=0;ys<My;ys++) { /* v */
    for (i=0;i<Mx;i++) rowidxs[i] = (i*My+ys)*3+1;
    for (i=0;i<Mx;i++) colidxs[i] = (i*My+ys)*3+1;
    ierr = MatSetValues(sw->corX2d,Mx,rowidxs,Mx,colidxs,x1dscarr,INSERT_VALUES);CHKERRQ(ierr);
  }
  for (i=0;i<Mx*Mx;i++) x1dscarr[i] *=  sw->sqrtrh/sw->sqrtrv;
  for (ys=0;ys<My;ys++) { /* h */
    for (i=0;i<Mx;i++) rowidxs[i] = (i*My+ys)*3+2;
    for (i=0;i<Mx;i++) colidxs[i] = (i*My+ys)*3+2;
    ierr = MatSetValues(sw->corX2d,Mx,rowidxs,Mx,colidxs,x1dscarr,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatDenseRestoreArray(corX1dsc,&x1dscarr);CHKERRQ(ierr);
  ierr = MatDestroy(&corX1dsc);CHKERRQ(ierr);
  ierr = MatDestroy(&corX1d);CHKERRQ(ierr);

  /* Kronecker product of I(Mx,Mx) and corY1d, duplicated for the three fields */
  ierr = MatDuplicate(corY1d,MAT_COPY_VALUES,&corY1dsc);CHKERRQ(ierr);
  ierr = MatDenseGetArray(corY1dsc,&y1dscarr);CHKERRQ(ierr);
  for (i=0;i<My*My;i++) y1dscarr[i] *=  sw->sqrtru;
  for (xs=0;xs<Mx;xs++) { /* u */
    for (i=0;i<My;i++) rowidxs[i] = (xs*My+i)*3;
    for (i=0;i<My;i++) colidxs[i] = (xs*My+i)*3;
    ierr = MatSetValues(sw->corY2d,My,rowidxs,My,colidxs,y1dscarr,INSERT_VALUES);CHKERRQ(ierr);
  }
  for (i=0;i<My*My;i++) y1dscarr[i] *=  sw->sqrtrv/sw->sqrtru;
  for (xs=0;xs<Mx;xs++) { /* u and v */
    for (i=0;i<My;i++) rowidxs[i] = (xs*My+i)*3+1;
    for (i=0;i<My;i++) colidxs[i] = (xs*My+i)*3+1;
    ierr = MatSetValues(sw->corY2d,My,rowidxs,My,colidxs,y1dscarr,INSERT_VALUES);CHKERRQ(ierr);
  }
  for (i=0;i<My*My;i++) y1dscarr[i] *=  sw->sqrtrh/sw->sqrtrv;
  for (xs=0;xs<Mx;xs++) { /* h */
    for (i=0;i<My;i++) rowidxs[i] = (xs*My+i)*3+2;
    for (i=0;i<My;i++) colidxs[i] = (xs*My+i)*3+2;
    ierr = MatSetValues(sw->corY2d,My,rowidxs,My,colidxs,y1dscarr,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatDenseRestoreArray(corY1dsc,&y1dscarr);CHKERRQ(ierr);
  ierr = MatDestroy(&corY1dsc);CHKERRQ(ierr);
  ierr = MatDestroy(&corY1d);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(sw->corX2d,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(sw->corX2d,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(sw->corY2d,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(sw->corY2d,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscFree2(rowidxs,colidxs);CHKERRQ(ierr);

  return 0;
}

/*
  Compute the regulization term
  (u-u_b)^T B^-1 (u-u_b)
  where B is the background error covariance and u is the initial conditoin.
  Its gradients is 2*B^-1 (u-u_b).
*/
PetscErrorCode BackgroundError(DM da,Vec U,Model_SW *sw,PetscReal *r,Vec Gr)
{
  KSP            ksp;
  Vec            b0,b1;
  PetscErrorCode ierr;

  /* Solve for (U-Uf)^T (corX2d*corY2d)^-1 (U-Uf) */
  ierr = VecDuplicate(U,&b0);CHKERRQ(ierr);
  ierr = VecDuplicate(U,&b1);CHKERRQ(ierr);
  ierr = VecCopy(U,b0);CHKERRQ(ierr);
  ierr = VecAXPY(b0,-1.,sw->Uf);CHKERRQ(ierr);
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,sw->corX2d,sw->corX2d);CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b0,b1);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,sw->corY2d,sw->corY2d);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b1,Gr);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDot(b0,Gr,r);CHKERRQ(ierr);
  *r = *r/2.;
  ierr = VecDestroy(&b0);CHKERRQ(ierr);
  ierr = VecDestroy(&b1);CHKERRQ(ierr);
  //printf("Regularization term = %lf\n",*r);
  return 0;
}

PetscErrorCode ReInitializeLambda(DM da,Vec lambda,Vec U,PetscInt iob,Model_SW *sw)
{
  PetscInt       i,j,xs,ys,xm,ym;
  Field          **uarr,**larr;
  char           filename[PETSC_MAX_PATH_LEN]="";
  Vec            Uob;
  PetscViewer    viewer;
  PetscErrorCode ierr;

  ierr = VecDuplicate(U,&Uob);CHKERRQ(ierr);

  ierr = PetscSNPrintf(filename,sizeof filename,"sw-%03d.obs",iob);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = VecLoad(Uob,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = VecAYPX(Uob,-1.,U);CHKERRQ(ierr);
  ierr = ApplyObservation(da,Uob);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(da,lambda,&larr);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,Uob,&uarr);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++) { /* latitude */
    for (i=xs; i<xs+xm; i++) { /* longitude */
      larr[j][i].u += uarr[j][i].u/(sw->sqrtru*sw->sqrtru);
      larr[j][i].v += uarr[j][i].v/(sw->sqrtrv*sw->sqrtrv);
      larr[j][i].h += uarr[j][i].h/(sw->sqrtrh*sw->sqrtrh);
    }
  }
  ierr = DMDAVecRestoreArray(da,Uob,&uarr);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,lambda,&larr);CHKERRQ(ierr);

  ierr = VecDestroy(&Uob);CHKERRQ(ierr);
  return 0;
}

PetscErrorCode InitialConditions(DM da,Vec U,Model_SW *sw)
{
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  PetscReal      a,omega,g,phi,u0,dlat,lat,lon;
  Field          **uarr;
  PetscErrorCode ierr;

  a     = sw->EarthRadius;
  omega = sw->AngularSpeed;
  g     = sw->Gravity;
  phi   = sw->phi;
  u0    = 20.0;

  ierr = DMDAVecGetArray(da,U,&uarr);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
  dlat = PETSC_PI/(PetscReal)(My);
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  for (j=ys; j<ys+ym; j++) { /* latitude */
    lat = -PETSC_PI/2.+j*dlat+dlat/2.; /* shift half grid size to avoid North pole and South pole */
    for (i=xs; i<xs+xm; i++) { /* longitude */
      lon = i*dlat; /* dlon = dlat */
      uarr[j][i].u = -3.*u0*PetscSinReal(lat)*PetscCosReal(lat)*PetscCosReal(lat)*PetscSinReal(lon)+u0*PetscSinReal(lat)*PetscSinReal(lat)*PetscSinReal(lat)*PetscSinReal(lon);
      uarr[j][i].v = u0*PetscSinReal(lat)*PetscSinReal(lat)*PetscCosReal(lon);
      uarr[j][i].h = (phi+2.*omega*a*u0*PetscSinReal(lat)*PetscSinReal(lat)*PetscSinReal(lat)*PetscCosReal(lat)*PetscSinReal(lon))/g;
    }
  }
  ierr = DMDAVecRestoreArray(da,U,&uarr);CHKERRQ(ierr);
  return 0;
}

PetscErrorCode RHSFunction(TS ts,PetscReal ftime,Vec U,Vec F,void *ptr)
{
  Model_SW      *sw = (Model_SW*)ptr;
  DM             da;
  PetscInt       i,j,Mx,My,xs,ys,xm,ym,p,q,ph,qh;
  PetscReal      a,g,alpha,omega,lat,dlat,dlon;
  PetscScalar    fc,fnq,fsq,uc,ue,uw,uep,uwp,ueph,uwph,un,us,unq,usq,uephnqh,uwphnqh,uephsqh,uwphsqh,vc,ve,vw,vep,vwp,vn,vs,vnqh,vsqh,vephnqh,vwphnqh,vephsqh,vwphsqh,hc,he,hw,hn,hs,heph,hwph,hnqh,hsqh;
  Field          **uarr,**farr;
  Vec            localU;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  a     = sw->EarthRadius;
  omega = sw->AngularSpeed;
  g     = sw->Gravity;
  alpha = sw->alpha;
  p     = sw->p;
  q     = sw->q;
  ph    = sw->p/2; /* staggered */
  qh    = sw->q/2; /* staggered */
  dlon  = 2.*PETSC_PI/(PetscReal)(Mx); /* longitude */
  dlat  = PETSC_PI/(PetscReal)(My); /* latitude */

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);

  /* Get pointers to vector data */
  ierr = DMDAVecGetArray(da,localU,&uarr);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&farr);CHKERRQ(ierr);

  /* Get local grid boundaries */
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  /*
     Place a solid wall at north and south boundaries, velocities are reflected, e.g. v(-1) = v(+1).
     Height is assumed to be constant, e.g. h(-1) = h(0).
     The forced velocity at the boundaries is a source of instability which causes the height of the boundary to rapidly increase.
     Copying over the next-row values can prevent the instability.
  */
  if (ys == 0) {
    for (i=xs-2; i<xs+xm+2; i++) {
      for (j=1; j<=q; j++) uarr[-j][i].u = -uarr[j-1][i].u;
      for (j=1; j<=qh; j++) {
        uarr[-j][i].h = uarr[0][i].h;
        uarr[-j][i].v = -uarr[j-1][i].v;
      }
    }
  }
  if (ys+ym == My) {
    for (i=xs-2; i<xs+xm+2; i++) {
      for (j=1; j<=q; j++) uarr[My+j-1][i].u = -uarr[My-j][i].u;
      for (j=1; j<=qh; j++) {
        uarr[My+j-1][i].h = uarr[My-1][i].h;
        uarr[My+j-1][i].v = -uarr[My-j][i].v;
      }
    }
  }

  /* Compute function over the locally owned part of the grid */
  for (j=ys; j<ys+ym; j++) { /* latitude */
    lat = -PETSC_PI/2.+j*dlat+dlat/2.; /* shift half dlat to avoid singularity */
    fc  = 2.*omega*PetscSinReal(lat);
    fnq = 2.*omega*PetscSinReal(lat+q*dlat);
    fsq = 2.*omega*PetscSinReal(lat-q*dlat);
    for (i=xs; i<xs+xm; i++) { /* longitude */
      uc      = uarr[j][i].u;
      uep     = uarr[j][i+p].u;
      uwp     = uarr[j][i-p].u;
      ueph    = uarr[j][i+ph].u;
      uwph    = uarr[j][i-ph].u;
      ue      = uarr[j][i+1].u;
      uw      = uarr[j][i-1].u;
      un      = uarr[j+1][i].u;
      us      = uarr[j-1][i].u;
      unq     = uarr[j+q][i].u;
      usq     = uarr[j-q][i].u;
      uephnqh = uarr[j+qh][i+ph].u;
      uwphnqh = uarr[j+qh][i-ph].u;
      uephsqh = uarr[j-qh][i+ph].u;
      uwphsqh = uarr[j-qh][i-ph].u;
      vc      = uarr[j][i].v;
      ve      = uarr[j][i+1].v;
      vw      = uarr[j][i-1].v;
      vep     = uarr[j][i+p].v;
      vwp     = uarr[j][i-p].v;
      vn      = uarr[j+1][i].v;
      vs      = uarr[j-1][i].v;
      vnqh    = uarr[j+qh][i].v;
      vsqh    = uarr[j-qh][i].v;
      vephnqh = uarr[j+qh][i+ph].v;
      vwphnqh = uarr[j+qh][i-ph].v;
      vephsqh = uarr[j-qh][i+ph].v;
      vwphsqh = uarr[j-qh][i-ph].v;
      hc      = uarr[j][i].h;
      he      = uarr[j][i+1].h;
      hw      = uarr[j][i-1].h;
      hn      = uarr[j+1][i].h;
      hs      = uarr[j-1][i].h;
      heph    = uarr[j][i+ph].h;
      hwph    = uarr[j][i-ph].h;
      hnqh    = uarr[j+qh][i].h;
      hsqh    = uarr[j-qh][i].h;

      farr[j][i].u = -1./(2.*a*dlat)*(uc/PetscCosReal(lat)*(ue-uw)+vc*(un-us)+2.*g/(p*PetscCosReal(lat))*(heph-hwph))
                    +(1.-alpha)*(fc+uc/a*PetscTanReal(lat))*vc
                    +alpha/2.*(fc+uep/a*PetscTanReal(lat))*vep
                    +alpha/2.*(fc+uwp/a*PetscTanReal(lat))*vwp;
      farr[j][i].v = -1./(2.*a*dlat)*(uc/PetscCosReal(lat)*(ve-vw)+vc*(vn-vs)+2.*g/q*(hnqh-hsqh))
                    -(1.-alpha)*(fc+uc/a*PetscTanReal(lat))*uc
                    -alpha/2.*(fnq+unq/a*PetscTanReal(lat+q*dlat))*unq
                    -alpha/2.*(fsq+usq/a*PetscTanReal(lat-q*dlat))*usq;
      farr[j][i].h = -1./(2.*a*dlat)*(
                     uc/PetscCosReal(lat)*(he-hw)
                    +vc*(hn-hs)
                    +2.*hc/PetscCosReal(lat)*((1.-alpha)*(ueph-uwph)+alpha/2.*(uephnqh-uwphnqh+uephsqh-uwphsqh))/p
                    +2.*hc/PetscCosReal(lat)*((1.-alpha)*(vnqh*PetscCosReal(lat+qh*dlat)-vsqh*PetscCosReal(lat-qh*dlat))+alpha/2.*(vephnqh*PetscCosReal(lat+qh*dlat)-vephsqh*PetscCosReal(lat-qh*dlat)+vwphnqh*PetscCosReal(lat+qh*dlat)-vwphsqh*PetscCosReal(lat-qh*dlat)))/q
                     );
    }
  }

  /* Restore vectors */
  ierr = DMDAVecRestoreArray(da,localU,&uarr);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&farr);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localU);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec U,Mat A,Mat BB,void *ptx)
{
  Model_SW       *sw=(Model_SW*)ptx;
  DM             da;
  PetscInt       i,j,Mx,My,xs,ys,xm,ym,p,q,ph,qh;
  PetscReal      a,g,alpha,omega,lat,dlat,dlon;
  Field          **uarr;
  Vec            localU;
  MatStencil     stencil[19],rowstencil;
  PetscScalar    entries[19];
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  a     = sw->EarthRadius;
  omega = sw->AngularSpeed;
  g     = sw->Gravity;
  alpha = sw->alpha;
  p     = sw->p;
  q     = sw->q;
  ph    = sw->p/2; /* staggered */
  qh    = sw->q/2; /* staggered */
  dlon  = 2.*PETSC_PI/(PetscReal)(Mx); /* longitude */
  dlat  = PETSC_PI/(PetscReal)(My); /* latitude */

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);

  /* Get pointers to vector data */
  ierr = DMDAVecGetArray(da,localU,&uarr);CHKERRQ(ierr);

  /* Get local grid boundaries */
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  /*
     Place a solid wall at north and south boundaries, velocities are reflected, e.g. v(-1) = v(+1).
     Height is assumed to be constant, e.g. h(-1) = h(0).
  */
  if (ys == 0) {
    for (i=xs-2; i<xs+xm+2; i++) {
      for (j=1; j<=q; j++) uarr[-j][i].u = -uarr[j-1][i].u;
      for (j=1; j<=qh; j++) {
        uarr[-j][i].h = uarr[0][i].h;
        uarr[-j][i].v = -uarr[j-1][i].v;
      }
    }
  }
  if (ys+ym == My) {
    for (i=xs-2; i<xs+xm+2; i++) {
      for (j=1; j<=q; j++) uarr[My+j-1][i].u = -uarr[My-j][i].u;
      for (j=1; j<=qh; j++) {
        uarr[My+j-1][i].h = uarr[My-1][i].h;
        uarr[My+j-1][i].v = -uarr[My-j][i].v;
      }
    }
  }

  for (i=0; i<19; i++) stencil[i].k = 0;
  rowstencil.k = 0;
  rowstencil.c = 0;
  for (j=ys; j<ys+ym; j++) { /* checked */
    PetscReal fc;
    lat = -PETSC_PI/2.+j*dlat+dlat/2.; /* shift half dlat to avoid singularity */
    fc  = 2.*omega*PetscSinReal(lat);

    /* Relocate the ghost points at north and south boundaries */
    stencil[0].j  = nbc_uv(j+1);
    stencil[1].j  = j;
    stencil[2].j  = j;
    stencil[3].j  = j;
    stencil[4].j  = j;
    stencil[5].j  = j;
    stencil[6].j  = sbc_uv(j-1);
    stencil[7].j  = j;
    stencil[8].j  = j;
    stencil[9].j  = j;
    stencil[10].j = j;
    stencil[11].j = j;

    rowstencil.j = j;
    for (i=xs; i<xs+xm; i++) {
      PetscReal vc,vep,vwp,uc,ue,uw,un,us,uep,uwp;
      uc  = uarr[j][i].u;
      ue  = uarr[j][i+1].u;
      uw  = uarr[j][i-1].u;
      un  = uarr[j+1][i].u;
      us  = uarr[j-1][i].u;
      uep = uarr[j][i+p].u;
      uwp = uarr[j][i-p].u;
      vc  = uarr[j][i].v;
      vep = uarr[j][i+p].v;
      vwp = uarr[j][i-p].v;

      stencil[0].i  = i;    stencil[0].c  = 0; entries[0]  = -1./(2.*a*dlat)*vc; /* un */
      stencil[1].i  = i-p; stencil[1].c  = 0; entries[1]  = alpha/2./a*PetscTanReal(lat)*vwp; /* uwp */
      stencil[2].i  = i-1;  stencil[2].c  = 0; entries[2]  = 1./(2.*a*dlat)*uc/PetscCosReal(lat); /* uw */
      stencil[3].i  = i;    stencil[3].c  = 0; entries[3]  = -1./(2.*a*dlat)/PetscCosReal(lat)*(ue-uw)+(1.-alpha)/a*PetscTanReal(lat)*vc; /* uc */
      stencil[4].i  = i+1;  stencil[4].c  = 0; entries[4]  = -entries[2]; /* ue */
      stencil[5].i  = i+p;  stencil[5].c  = 0; entries[5]  = alpha/2./a*PetscTanReal(lat)*vep; /* uep */
      stencil[6].i  = i;    stencil[6].c  = 0; entries[6]  = -entries[0]; /* us */
      stencil[7].i  = i-p;  stencil[7].c  = 1; entries[7]  = alpha/2.*(fc+uwp/a*PetscTanReal(lat)); /* vwp */
      stencil[8].i  = i;    stencil[8].c  = 1; entries[8]  = -1./(2.*a*dlat)*(un-us)+(1.-alpha)*(fc+uc/a*PetscTanReal(lat)); /* vc */
      stencil[9].i  = i+p; stencil[9].c  = 1; entries[9]  = alpha/2.*(fc+uep/a*PetscTanReal(lat)); /* vep */
      stencil[10].i = i-ph; stencil[10].c = 2; entries[10] = 1./(2.*a*dlat)*2.*g/(p*PetscCosReal(lat));/* hwph */
      stencil[11].i = i+ph; stencil[11].c = 2; entries[11] = -entries[10]; /* heph */

      /* flip the sign */
      if (j==0) entries[6] = -entries[6];
      if (j==My-1) entries[0] = -entries[0];

      rowstencil.i = i;
      /* for (int k=0;k<19;k++) entries[k] += 30000+k+10*j+1000*i; for debugging */
      ierr = MatSetValuesStencil(A,1,&rowstencil,12,stencil,entries,ADD_VALUES);CHKERRQ(ierr);
    }
  }

  rowstencil.c = 1;
  for (j=ys; j<ys+ym; j++) {
    PetscReal fc,fnq,fsq;
    lat = -PETSC_PI/2.+j*dlat+dlat/2.; /* shift half dlat to avoid singularity */
    fc  = 2.*omega*PetscSinReal(lat);
    fnq = 2.*omega*PetscSinReal(lat+q*dlat);
    fsq = 2.*omega*PetscSinReal(lat-q*dlat);

    /* Relocate the ghost points at north and south boundaries */
    stencil[0].j = nbc_uv(j+q);
    stencil[1].j = j;
    stencil[2].j = sbc_uv(j-q);
    stencil[3].j = nbc_uv(j+1);
    stencil[4].j = j;
    stencil[5].j = j;
    stencil[6].j = j;
    stencil[7].j = sbc_uv(j-1);
    stencil[8].j = nbc_h(j+qh);
    stencil[9].j = sbc_h(j-qh);

    rowstencil.j = j;
    for (i=xs; i<xs+xm; i++) {
      PetscReal uc,unq,usq,ve,vs,vw,vn,vc;
      uc  = uarr[j][i].u;
      unq = uarr[j+q][i].u;
      usq = uarr[j-q][i].u;
      ve  = uarr[j][i+1].v;
      vs  = uarr[j-1][i].v;
      vw  = uarr[j][i-1].v;
      vn  = uarr[j+1][i].v;
      vc  = uarr[j][i].v;

      stencil[0].i = i;   stencil[0].c = 0; entries[0] = -alpha/2.*(fnq+2.*unq/a*PetscTanReal(lat+q*dlat)); /* unq */
      stencil[1].i = i;   stencil[1].c = 0; entries[1] = -1./(2.*a*dlat*PetscCosReal(lat))*(ve-vw)-(1.-alpha)*(fc+2.*uc/a*PetscTanReal(lat)); /* uc */
      stencil[2].i = i;   stencil[2].c = 0; entries[2] = -alpha/2.*(fsq+2.*usq/a*PetscTanReal(lat-q*dlat)); /* usq */
      stencil[3].i = i;   stencil[3].c = 1; entries[3] = -1./(2.*a*dlat)*vc; /* vn */
      stencil[4].i = i-1; stencil[4].c = 1; entries[4] = 1./(2.*a*dlat*PetscCosReal(lat))*uc; /* vw */
      stencil[5].i = i;   stencil[5].c = 1; entries[5] = -1./(2.*a*dlat)*(vn-vs); /* vc */
      stencil[6].i = i+1; stencil[6].c = 1; entries[6] = -entries[4]; /* ve */
      stencil[7].i = i;   stencil[7].c = 1; entries[7] = -entries[3]; /* vs */
      stencil[8].i = i;   stencil[8].c = 2; entries[8] = -g/(a*dlat*q); /* hnqh */
      stencil[9].i = i;   stencil[9].c = 2; entries[9] = -entries[8]; /* hsqh */

      /* flip the sign */
      if (j < q) entries[2] = -entries[2];
      if (j > My-q-1) entries[0] = -entries[0];
      if (j == 0) entries[7] = -entries[7];
      if (j == My-1) entries[3] = -entries[3];
      rowstencil.i = i;
      /* for (int k=0;k<19;k++) entries[k] += 50000+k+10*j+1000*i; for debugging */
      ierr = MatSetValuesStencil(A,1,&rowstencil,10,stencil,entries,ADD_VALUES);CHKERRQ(ierr);
    }
  }

  rowstencil.c = 2;
  for (j=ys; j<ys+ym; j++) {
    lat = -PETSC_PI/2.+j*dlat+dlat/2.; /* shift half dlat to avoid singularity */

    /* Relocate the ghost points at north and south boundaries */
    stencil[0].j  = nbc_uv(j+qh);
    stencil[1].j  = nbc_uv(j+qh);
    stencil[2].j  = j;
    stencil[3].j  = j;
    stencil[4].j  = j;
    stencil[5].j  = sbc_uv(j-qh);
    stencil[6].j  = sbc_uv(j-qh);
    stencil[7].j  = nbc_uv(j+qh);
    stencil[8].j  = nbc_uv(j+qh);
    stencil[9].j  = nbc_uv(j+qh);
    stencil[10].j = j;
    stencil[11].j = sbc_uv(j-qh);
    stencil[12].j = sbc_uv(j-qh);
    stencil[13].j = sbc_uv(j-qh);
    stencil[14].j = nbc_h(j+1);
    stencil[15].j = j;
    stencil[16].j = j;
    stencil[17].j = j;
    stencil[18].j = sbc_h(j-1);

    rowstencil.j  = j;
    for (i=xs; i<xs+xm; i++) {
      PetscReal uc,ueph,uwph,uephnqh,uwphnqh,uephsqh,uwphsqh,vc,vnqh,vsqh,vephnqh,vwphnqh,vephsqh,vwphsqh,hc,he,hw,hs,hn;
      uc      = uarr[j][i].u;
      ueph    = uarr[j][i+ph].u;
      uwph    = uarr[j][i-ph].u;
      uephnqh = uarr[j+qh][i+ph].u;
      uwphnqh = uarr[j+qh][i-ph].u;
      uephsqh = uarr[j-qh][i+ph].u;
      uwphsqh = uarr[j-qh][i-ph].u;
      vc      = uarr[j][i].v;
      vnqh    = uarr[j+qh][i].v;
      vsqh    = uarr[j-qh][i].v;
      vephnqh = uarr[j+qh][i+ph].v;
      vwphnqh = uarr[j+qh][i-ph].v;
      vephsqh = uarr[j-qh][i+ph].v;
      vwphsqh = uarr[j-qh][i-ph].v;
      hc      = uarr[j][i].h;
      he      = uarr[j][i+1].h;
      hw      = uarr[j][i-1].h;
      hs      = uarr[j-1][i].h;
      hn      = uarr[j+1][i].h;

      stencil[0].i  = i-ph; stencil[0].c  = 0; entries[0]  = 1./(2.*a*dlat*PetscCosReal(lat)*2.*p)*2.*hc*alpha; /* uwphnqh */
      stencil[1].i  = i+ph; stencil[1].c  = 0; entries[1]  = -entries[0]; /* uephnqh */
      stencil[2].i  = i-ph; stencil[2].c  = 0; entries[2]  = 1./(2.*a*dlat*PetscCosReal(lat)*p)*2.*hc*(1.-alpha); /* uwph */
      stencil[3].i  = i;    stencil[3].c  = 0; entries[3]  = -1./(2.*a*dlat*PetscCosReal(lat))*(he-hw); /* uc */
      stencil[4].i  = i+ph; stencil[4].c  = 0; entries[4]  = -entries[2]; /* ueph */
      stencil[5].i  = i-ph; stencil[5].c  = 0; entries[5]  = entries[0]; /* uwphsqh */
      stencil[6].i  = i+ph; stencil[6].c  = 0; entries[6]  = -entries[0]; /* uephsqh */
      stencil[7].i  = i-ph; stencil[7].c  = 1; entries[7]  = -1./(2.*a*dlat*PetscCosReal(lat)*2.*q)*2.*hc*alpha*PetscCosReal(lat+qh*dlat); /* vwphnqh */
      stencil[8].i  = i;    stencil[8].c  = 1; entries[8]  = -1./(2.*a*dlat*PetscCosReal(lat)*q)*2.*hc*(1.-alpha)*PetscCosReal(lat+qh*dlat); /* vnqh */
      stencil[9].i  = i+ph; stencil[9].c  = 1; entries[9]  = entries[7]; /* vephnqh */
      stencil[10].i = i;    stencil[10].c = 1; entries[10] = -1./(2.*a*dlat)*(hn-hs); /* vc */
      stencil[11].i = i-ph; stencil[11].c = 1; entries[11] = 1./(2.*a*dlat*PetscCosReal(lat)*2.*q)*(2.*hc*alpha*PetscCosReal(lat-qh*dlat)); /* vwphsqh */
      stencil[12].i = i;    stencil[12].c = 1; entries[12] = 1./(2.*a*dlat*PetscCosReal(lat)*q)*2.*hc*(1.-alpha)*PetscCosReal(lat-qh*dlat); /* vsqh */
      stencil[13].i = i+ph; stencil[13].c = 1; entries[13] = entries[11]; /* vephsqh */
      stencil[14].i = i;    stencil[14].c = 2; entries[14] = -1./(2.*a*dlat)*vc; /* hn */
      stencil[15].i = i-1;  stencil[15].c = 2; entries[15] = 1./(2.*a*dlat*PetscCosReal(lat))*uc; /* hw */
      stencil[16].i = i;    stencil[16].c = 2; entries[16] = -1./(2.*a*dlat)*(2./PetscCosReal(lat)*((1.-alpha)*(ueph-uwph)+alpha/2.*(uephnqh-uwphnqh+uephsqh-uwphsqh))/p + 2./PetscCosReal(lat)*((1.-alpha)*(vnqh*PetscCosReal(lat+qh*dlat)-vsqh*PetscCosReal(lat-qh*dlat))+alpha/2.*(vephnqh*PetscCosReal(lat+qh*dlat)-vephsqh*PetscCosReal(lat-qh*dlat)+vwphnqh*PetscCosReal(lat+qh*dlat)-vwphsqh*PetscCosReal(lat-qh*dlat)))/q); /* hc */
      stencil[17].i = i+1;  stencil[17].c = 2; entries[17] = -entries[15]; /* he */
      stencil[18].i = i;    stencil[18].c = 2; entries[18] = -entries[14]; /* hs */

      /* flip the sign */
      if (j < qh) {
        entries[5]  = -entries[5];
        entries[6]  = -entries[6];
        entries[11] = -entries[11];
        entries[12] = -entries[12];
        entries[13] = -entries[13];
      }
      if (j > My-qh-1) {
        entries[0] = -entries[0];
        entries[1] = -entries[1];
        entries[7] = -entries[7];
        entries[8] = -entries[8];
        entries[9] = -entries[9];
      }
      rowstencil.i = i;
      /* for (int k=0;k<19;k++) entries[k] += 70000+k+10*j+1000*i; for debugging */
      ierr = MatSetValuesStencil(A,1,&rowstencil,19,stencil,entries,ADD_VALUES);CHKERRQ(ierr);
    }
  }

  /* Restore vectors */
  ierr = DMDAVecRestoreArray(da,localU,&uarr);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode OutputBIN(DM dm, const char *filename, PetscViewer *viewer)
{
  PetscErrorCode ierr;

  ierr = PetscViewerCreate(PetscObjectComm((PetscObject)dm),viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(*viewer,PETSCVIEWERBINARY);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(*viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(*viewer,filename);CHKERRQ(ierr);
  return 0;
}

PetscErrorCode GenerateGaussianNoise(PetscRandom rand,PetscReal mu,PetscReal sigma,PetscReal *noise)
{
  PetscReal      u1,u2,z;
  PetscErrorCode ierr;

  ierr = PetscRandomGetValue(rand,&u1);CHKERRQ(ierr);
  ierr = PetscRandomGetValue(rand,&u2);CHKERRQ(ierr);
  z = PetscSqrtReal(-2.*PetscLogReal(u1))*PetscCosReal(2.*PETSC_PI*u2);
  *noise = z*sigma+mu;
  return 0;;
}

/*
   Generate observations
 */
PetscErrorCode GenerateOBs(TS ts,Vec U,Model_SW *sw)
{
  PetscInt       i,j,iob,xs,ys,xm,ym;
  PetscReal      randn;
  Field          **uarr;
  PetscInt       maxsteps;
  Vec            Uob;
  PetscRandom    rand;
  char           filename[PETSC_MAX_PATH_LEN] = "";
  PetscViewer    viewer;
  DM             dm;
  PetscErrorCode ierr;

  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PetscObjectComm((PetscObject)dm),&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rand,0,1.);CHKERRQ(ierr);
  ierr = VecDuplicate(U,&Uob);CHKERRQ(ierr);
  ierr = TSGetMaxSteps(ts,&maxsteps);CHKERRQ(ierr);
  for (iob=1; iob<=sw->nobs; iob++) {
    ierr = TSSetMaxSteps(ts,iob*maxsteps/sw->nobs);CHKERRQ(ierr);
    ierr = TSSolve(ts,U);CHKERRQ(ierr);
    ierr = VecCopy(U,Uob);CHKERRQ(ierr);

    ierr = DMDAVecGetArray(dm,Uob,&uarr);CHKERRQ(ierr);
    ierr = DMDAGetCorners(dm,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
    for (j=ys; j<ys+ym; j+=10) { /* latitude */
      for (i=xs; i<xs+xm; i+=10) { /* longitude */
        ierr = GenerateGaussianNoise(rand,0,1.,&randn);CHKERRQ(ierr);
        uarr[j][i].u += sw->sqrtru*randn;
        ierr = GenerateGaussianNoise(rand,0,1.,&randn);CHKERRQ(ierr);
        uarr[j][i].v += sw->sqrtrv*randn;
        ierr = GenerateGaussianNoise(rand,0,1.,&randn);CHKERRQ(ierr);
        uarr[j][i].h += sw->sqrtrh*randn;
      }
    }
    ierr = DMDAVecRestoreArray(dm,Uob,&uarr);CHKERRQ(ierr);

    ierr = PetscSNPrintf(filename,sizeof filename,"sw-%03D.obs",iob);CHKERRQ(ierr);
    ierr = OutputBIN(dm,filename,&viewer);CHKERRQ(ierr);
    ierr = VecView(Uob,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&Uob);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  return 0;
}

/*
   Generate forecast state
 */
PetscErrorCode GenerateFS(DM dm,Vec U,Model_SW *sw)
{
  PetscInt       i,j,xs,ys,xm,ym;
  PetscReal      randn;
  Field          **xfarr;
  PetscRandom    rand;
  PetscErrorCode ierr;

  ierr = PetscRandomCreate(PetscObjectComm((PetscObject)dm),&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(rand,0,1.);CHKERRQ(ierr);
  ierr = VecDuplicate(U,&sw->Uf);CHKERRQ(ierr);
  ierr = VecCopy(U,sw->Uf);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(dm,sw->Uf,&xfarr);CHKERRQ(ierr);
  ierr = DMDAGetCorners(dm,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j+=10) { /* latitude */
    for (i=xs; i<xs+xm; i+=10) { /* longitude */
      ierr = GenerateGaussianNoise(rand,0,0.1,&randn);CHKERRQ(ierr);
      xfarr[j][i].u += sw->sqrtru*randn;
      ierr = GenerateGaussianNoise(rand,0,0.1,&randn);CHKERRQ(ierr);
      xfarr[j][i].v += sw->sqrtrv*randn;
      ierr = GenerateGaussianNoise(rand,0,0.1,&randn);CHKERRQ(ierr);
      xfarr[j][i].h += sw->sqrtrh*randn;
    }
  }
  ierr = DMDAVecRestoreArray(dm,sw->Uf,&xfarr);CHKERRQ(ierr);

  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  return 0;
}

/*
   FormFunctionAndGradient - Evaluates the function and corresponding gradient.

   Input Parameters:
   tao - the Tao context
   X   - the input vector
   ctx - optional user-defined context, as set by TaoSetObjectiveAndGradientRoutine()

   Output Parameters:
   f   - the newly evaluated function
   G   - the newly evaluated gradient
*/
PetscErrorCode FormFunctionAndGradient(Tao tao,Vec P,PetscReal *f,Vec G,void *ctx)
{
  PetscInt       i,j,iob,xs,ys,xm,ym;
  PetscInt       maxsteps,numcost;
  PetscReal      soberr,timestep,r;
  Vec            *lambda;
  Vec            SDiff;
  DM             da;
  Field          **sdarr;
  Model_SW       *sw = (Model_SW*)ctx;
  char           filename[PETSC_MAX_PATH_LEN]="";
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSSetTime(sw->ts,0.0);CHKERRQ(ierr);
  ierr = TSGetTimeStep(sw->ts,&timestep);CHKERRQ(ierr);
  if (timestep<0) {
    ierr = TSSetTimeStep(sw->ts,-timestep);CHKERRQ(ierr);
  }
  ierr = TSSetStepNumber(sw->ts,0);CHKERRQ(ierr);

  ierr = VecDuplicate(P,&SDiff);CHKERRQ(ierr);
  ierr = VecCopy(P,sw->U);CHKERRQ(ierr);
  ierr = TSGetDM(sw->ts,&da);CHKERRQ(ierr);
  ierr = TSGetMaxSteps(sw->ts,&maxsteps);CHKERRQ(ierr);
  *f = 0;
  for (iob=1; iob<=sw->nobs; iob++) {
    ierr = TSSetMaxSteps(sw->ts,iob*maxsteps/sw->nobs);CHKERRQ(ierr);
    ierr = TSSolve(sw->ts,sw->U);CHKERRQ(ierr);
    ierr = PetscSNPrintf(filename,sizeof filename,"sw-%03d.obs",iob);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
    ierr = VecLoad(SDiff,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = VecAYPX(SDiff,-1.,sw->U);CHKERRQ(ierr);

    ierr = DMDAVecGetArray(da,SDiff,&sdarr);CHKERRQ(ierr);
    ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
    for (j=ys; j<ys+ym; j++) { /* latitude */
      for (i=xs; i<xs+xm; i++) { /* longitude */
        sdarr[j][i].u /= sw->sqrtru;
        sdarr[j][i].v /= sw->sqrtrv;
        sdarr[j][i].h /= sw->sqrtrh;
      }
    }
    ierr = DMDAVecRestoreArray(da,SDiff,&sdarr);CHKERRQ(ierr);
    ierr = ApplyObservation(da,SDiff);CHKERRQ(ierr);
    ierr = VecDot(SDiff,SDiff,&soberr);CHKERRQ(ierr);
    *f += soberr/2.;
  }

  ierr = TSGetCostGradients(sw->ts,&numcost,&lambda,NULL);CHKERRQ(ierr);
  ierr = VecSet(lambda[0],0.0);CHKERRQ(ierr);
  for (iob=sw->nobs; iob>=1; iob--) {
    TSTrajectory tj;
    PetscReal    time;
    ierr = TSGetTrajectory(sw->ts,&tj);CHKERRQ(ierr);
    ierr = TSTrajectoryGet(tj,sw->ts,iob*maxsteps/sw->nobs,&time);CHKERRQ(ierr);
    ierr = ReInitializeLambda(da,lambda[0],sw->U,iob,sw);CHKERRQ(ierr);
    ierr = TSAdjointSetSteps(sw->ts,maxsteps/sw->nobs);CHKERRQ(ierr);
    ierr = TSAdjointSolve(sw->ts);CHKERRQ(ierr);
  }
  ierr = VecCopy(lambda[0],G);CHKERRQ(ierr);

  if (sw->be) {
    ierr = BackgroundError(da,sw->U,sw,&r,SDiff);CHKERRQ(ierr);
    *f  += r;
    ierr = VecAXPY(G,1.,SDiff);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&SDiff);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscBool      forwardonly;
  Vec            P;
  DM             da;
  Model_SW       sw;
  Vec            lambda[1];
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,NULL);if (ierr) return ierr;
  sw.Gravity      = 9.8;
  sw.EarthRadius  = 6.37e6;
  sw.alpha        = 1./3.;
  sw.phi          = 5.768e4;
  sw.AngularSpeed = 7.292e-5;
  sw.p            = 4;
  sw.q            = 2;
  sw.nobs         = 2;
  sw.sqrtru       = 1.; /* 5% error */
  sw.sqrtrv       = 1.; /* 5% error */
  sw.sqrtrh       = 700.; /* 1.5% error */
  forwardonly     = PETSC_FALSE;
  sw.be           = PETSC_FALSE;

  ierr = PetscOptionsGetBool(NULL,NULL,"-forwardonly",&forwardonly,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-num_obs",&sw.nobs,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-background_error",&sw.be,NULL);CHKERRQ(ierr);

  ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_GHOSTED,DMDA_STENCIL_BOX,150,75,PETSC_DECIDE,PETSC_DECIDE,3,4,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,0,"u");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,1,"v");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,2,"h");CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&sw.U);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&P);CHKERRQ(ierr);
  /*
  {// testing
    PetscReal r;
    Vec Gr;
    ierr = VecDuplicate(U,&Gr);CHKERRQ(ierr);
    ierr = BackgroundError(da,U,&sw,&r,Gr);CHKERRQ(ierr);
    ierr = VecDestroy(&Gr);CHKERRQ(ierr);
  }
  return 0;
  */
  ierr = TSCreate(PETSC_COMM_WORLD,&sw.ts);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(sw.ts,NULL,RHSFunction,&sw);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(sw.ts,NULL,NULL,RHSJacobian,&sw);CHKERRQ(ierr);
  ierr = TSSetType(sw.ts,TSRK);CHKERRQ(ierr);
  ierr = TSSetDM(sw.ts,da);CHKERRQ(ierr);

  ierr = InitialConditions(da,sw.U,&sw);CHKERRQ(ierr);
  ierr = TSSetSolution(sw.ts,sw.U);CHKERRQ(ierr);

  /* Build Covariance matrix and generate forecast state */
  ierr = BuildCovariance(da,&sw);CHKERRQ(ierr);
  ierr = GenerateFS(da,sw.U,&sw);CHKERRQ(ierr);

  if (!forwardonly) {
    /* Let TS save its trajectory for TSAdjointSolve() */
    ierr = TSSetSaveTrajectory(sw.ts);CHKERRQ(ierr);
    ierr = VecDuplicate(P,&lambda[0]);CHKERRQ(ierr);
    ierr = TSSetCostGradients(sw.ts,1,lambda,NULL);CHKERRQ(ierr);
  }

  ierr = TSSetTime(sw.ts,0.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(sw.ts,225.0);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(sw.ts,16);CHKERRQ(ierr); /* 3600 s */
  ierr = TSSetExactFinalTime(sw.ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetFromOptions(sw.ts);CHKERRQ(ierr);
  ierr = GenerateOBs(sw.ts,sw.U,&sw);CHKERRQ(ierr);

  if (!forwardonly) {
    Tao         tao;

    /* Create TAO solver and set desired solution method */
    ierr = TaoCreate(PETSC_COMM_WORLD,&tao);CHKERRQ(ierr);
    ierr = TaoSetType(tao,TAOBLMVM);CHKERRQ(ierr);

    /* Set initial guess for TAO */
    ierr = VecCopy(sw.Uf,P);CHKERRQ(ierr);
    ierr = TaoSetInitialVector(tao,P);CHKERRQ(ierr);

    /* Set routine for function and gradient evaluation */
    ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionAndGradient,&sw);CHKERRQ(ierr);

    /* Check for any TAO command line options */
    ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);

    /*
    ierr = TaoGetKSP(tao,&ksp);CHKERRQ(ierr);
    if (ksp) {
      ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
      ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
    }
    */

    ierr = TaoSolve(tao);CHKERRQ(ierr);
    ierr = TaoDestroy(&tao);CHKERRQ(ierr);
    ierr = VecDestroy(&lambda[0]);CHKERRQ(ierr);
  }

  ierr = TSDestroy(&sw.ts);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = VecDestroy(&P);CHKERRQ(ierr);
  ierr = VecDestroy(&sw.U);CHKERRQ(ierr);
  ierr = VecDestroy(&sw.Uf);CHKERRQ(ierr);
  ierr = MatDestroy(&sw.corX2d);CHKERRQ(ierr);
  ierr = MatDestroy(&sw.corY2d);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
