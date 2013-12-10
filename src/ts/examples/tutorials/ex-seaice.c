
static char help[] = "Sea ice solver in 2D.\n";
/*T
   Concepts: SNES^solving a system of nonlinear equations;
   Concepts: DMDA^using distributed arrays;
   Processors: n
T*/

/*F-----------------------------------------------------------------------

    This example solves the 2D sea ice problem for velocity u

\begin{equation}
    \alpha m \dot u + m f \times u - \tau_w + m g \nabla_H p(0) - div(\sigma) = 0
\end{equation}
where
\begin{gather}
  m = \rho h                                                      & ice mass per area \\
  \alpha = 1/\Delta t                                             & inverse time step \\
  f = 1.46e-4 s^{-1}                                              & Coriolis \\
  \sigma = 2 \eta D u + ((\zeta - \eta) tr(D u) - P/2) 1          & stress \\
  D u = 1/2 (\nabla u + (\nabla u)^T)                             & strain rate tensor \\
  \zeta(Du,P) = P/2 sqrt[(Du_11^2+Du_22^2)(1+e^{-2})
                + 4e^{-2}Du_12^2 + 2 Du_11 Du_22 (1-e^{-2})]      & bulk viscosity \\
  \eta = \zeta / e^2                                              & shear viscosity \\
  \tau_w = \rho_w C_w|U_w - u|[(U_w-u) \cos\theta + k\times (U_w-u) \sin\theta]
\end{gather}

  ------------------------------------------------------------------------F*/

#include <petscts.h>
#include <petscdmda.h>
#include <petscfe.h>

typedef struct {
  PetscScalar u[2];
} Field;
typedef struct {
  PetscScalar x[2];
} CoordField;

typedef struct _User *User;
struct _User {
  PetscFE fe;
  PetscQuadrature q;
  PetscReal   *De;              /* Derivative matrix with respect to physical coordinates */
  PetscReal   *weights;         /* Real-space quadrature weights */
  Field       *xq;              /* Workspace for solution at quadrature points */
  Field       *Dxq;             /* Workspace for solution gradient at quadrature points Dxq[2*q+i].u */
  Field       *xdotq;           /* Workspace for time derivatives of solution at quadrature points */
  Field       *fq;              /* Workspace for coefficient of test function at quadrature points */
  Field       *Dfq;             /* Workspace for coefficient of test function gradient at quadrature points */
  CoordField  *cq;

  /* Options */
  CoordField  L;
  PetscReal   Coriolis_f;
  PetscReal   rho_ice;
  PetscReal   rho_air;
  PetscReal   rho_water;
  PetscReal   Cdrag_air;
  PetscReal   Cdrag_water;
  PetscReal   theta_drag_air;
  PetscReal   theta_drag_water;
  PetscReal   Pstar;
  PetscReal   ellipse_ratio;
  PetscReal   Concentration;
  PetscReal   zeta_max_coeff;

  PetscBool   view_initial;

  PetscReal   range_zeta[2];
  PetscBool   monitor_range;
};

/*****************************  Finite element support  ******************************/
#undef __FUNCT__
#define __FUNCT__ "DMDAComputeCellGeometry_2D"
static PetscErrorCode DMDAComputeCellGeometry_2D(DM dm, const PetscScalar vertices[], const PetscReal refPoint[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  const PetscScalar x0   = vertices[0];
  const PetscScalar y0   = vertices[1];
  const PetscScalar x1   = vertices[2];
  const PetscScalar y1   = vertices[3];
  const PetscScalar x2   = vertices[4];
  const PetscScalar y2   = vertices[5];
  const PetscScalar x3   = vertices[6];
  const PetscScalar y3   = vertices[7];
  const PetscScalar f_01 = x2 - x1 - x3 + x0;
  const PetscScalar g_01 = y2 - y1 - y3 + y0;
  const PetscScalar x    = refPoint[0];
  const PetscScalar y    = refPoint[1];
  PetscReal         invDet;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (0) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "Cell (%g,%g)--(%g,%g)--(%g,%g)--(%g,%g)\n",
                       PetscRealPart(x0),PetscRealPart(y0),PetscRealPart(x1),PetscRealPart(y1),PetscRealPart(x2),PetscRealPart(y2),PetscRealPart(x3),PetscRealPart(y3));CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, "Ref Point (%g,%g)\n", PetscRealPart(x), PetscRealPart(y));CHKERRQ(ierr);
  }
  J[0]    = PetscRealPart(x1 - x0 + f_01*y) * 0.5; J[1] = PetscRealPart(x3 - x0 + f_01*x) * 0.5;
  J[2]    = PetscRealPart(y1 - y0 + g_01*y) * 0.5; J[3] = PetscRealPart(y3 - y0 + g_01*x) * 0.5;
  *detJ   = J[0]*J[3] - J[1]*J[2];
  invDet  = 1.0/(*detJ);
  invJ[0] =  invDet*J[3]; invJ[1] = -invDet*J[1];
  invJ[2] = -invDet*J[2]; invJ[3] =  invDet*J[0];
  ierr    = PetscLogFlops(30);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static void QuadExtract(Field **x,PetscInt i,PetscInt j,Field ex[]) {
  ex[0] = x[j][i];
  ex[1] = x[j][i+1];
  ex[2] = x[j+1][i+1];
  ex[3] = x[j+1][i];
}

static void QuadExtractCoord(CoordField **x,PetscInt i,PetscInt j,CoordField ex[]) {
  ex[0] = x[j][i];
  ex[1] = x[j][i+1];
  ex[2] = x[j+1][i+1];
  ex[3] = x[j+1][i];
}

/* Map reference derivatives to real space and use more sane matrix ordering: D[q*2+i][j] */
static void QuadRealSpace(User user,const CoordField ce[],const PetscReal D[],const PetscReal **weights,const PetscReal **De) {
  PetscInt i,j;
  const PetscReal refpt[2] = {0,0}; /* Elements are affine, so just evaluate once */
  PetscReal J[4],invJ[4],detJ;

  DMDAComputeCellGeometry_2D(NULL,(const PetscScalar*)ce,refpt,J,invJ,&detJ);
  for (i=0; i<user->q.numPoints; i++) {
    for (j=0; j<4; j++) {
      user->De[(i*2+0)*4+j] = D[(i*4+j)*2+0]*invJ[0] + D[(i*4+j)*2+1]*invJ[1];
      user->De[(i*2+1)*4+j] = D[(i*4+j)*2+0]*invJ[2] + D[(i*4+j)*2+1]*invJ[3];
    }
    user->weights[i] = user->q.weights[i]*detJ;
  }
  *weights = user->weights;
  *De = user->De;
}

static PetscBool SubdomainInterior(DMDALocalInfo *info,PetscInt i,PetscInt j) {
  if (info->xs <= i && i < info->xs+info->xm && info->ys <= j && j < info->ys+info->ym) return PETSC_TRUE;
  return PETSC_FALSE;
}

PETSC_UNUSED
static PetscBool DomainInterior(DMDALocalInfo *info,PetscInt i,PetscInt j) {
  if (0 < i && i < info->mx-1 && 0 < j && j < info->my-1) return PETSC_TRUE;
  return PETSC_FALSE;
}

static void QuadExtractRef(DMDALocalInfo *info,Field **x,PetscInt i,PetscInt j,Field *ef[]) {
  ef[0] = SubdomainInterior(info,i  ,j  ) ? &x[j  ][i  ] : NULL;
  ef[1] = SubdomainInterior(info,i+1,j  ) ? &x[j  ][i+1] : NULL;
  ef[2] = SubdomainInterior(info,i+1,j+1) ? &x[j+1][i+1] : NULL;
  ef[3] = SubdomainInterior(info,i  ,j+1) ? &x[j+1][i  ] : NULL;
}

static void QuadMult(PetscQuadrature *q,PetscInt dim,const PetscReal B[],const Field xe[],Field xq[]) {
  PetscInt i,j;
  for (i=0; i<dim*q->numPoints; i++) {
    xq[i].u[0] = 0;
    xq[i].u[1] = 0;
    for (j=0; j<4; j++) {
      xq[i].u[0] += B[i*4+j] * xe[j].u[0];
      xq[i].u[1] += B[i*4+j] * xe[j].u[1];
    }
  }
}

static void QuadMultCoord(PetscQuadrature *q,PetscInt dim,const PetscReal B[],const CoordField ce[],CoordField cq[]) {
  QuadMult(q,dim,B,(const Field*)ce,(Field*)cq);
}

static void QuadMultTransposeAdd(PetscQuadrature *q,PetscInt dim,const PetscReal B[],const Field fq[],Field fe[]) {
  PetscInt i,j;
  for (j=0; j<4; j++) {
    for (i=0; i<dim*q->numPoints; i++) {
      fe[j].u[0] += B[i*4+j] * fq[i].u[0];
      fe[j].u[1] += B[i*4+j] * fq[i].u[1];
    }
  }
}

static void GetElementRange(DMDALocalInfo *info,PetscInt *xes,PetscInt *xee,PetscInt *yes,PetscInt *yee) {
  *xes = PetscMax(info->xs-1,0);
  *yes = PetscMax(info->ys-1,0);
  *xee = PetscMin(info->xs+info->xm,info->mx-1);
  *yee = PetscMin(info->ys+info->ym,info->my-1);
}

/*****************************  Physics  ******************************/
static void StrainRate(const Field dx[2],PetscScalar e[2][2]) {
  e[0][0] = dx[0].u[0];
  e[0][1] = e[1][0] = 0.5*(dx[0].u[1] + dx[1].u[0]);
  e[1][1] = dx[1].u[1];
}

static PetscScalar IceThickness(User user,CoordField c) {
  return 1.0;
}

static Field VelocityWater(User user,CoordField c) {
  Field v;
  PetscReal L = PetscMin(user->L.x[0],user->L.x[1]) / 6;
  PetscScalar x = c.x[0] / L,y = c.x[1]/L;
  PetscScalar xa = x + 2.,ya = y + 2.,r2a = PetscSqr(xa) + PetscSqr(ya);
  PetscScalar xb = x - 2.,yb = y - 2.,r2b = PetscSqr(xb) + PetscSqr(yb);

  v.u[0] = -PetscExpScalar(1-r2a) * ya - PetscExpScalar(1-r2b) * yb;
  v.u[1] =  PetscExpScalar(1-r2a) * xa + PetscExpScalar(1-r2b) * xb;
  v.u[0] *= 0.1;
  v.u[1] *= 0.1;

  if (0) {
    v.u[0] = 0.;
    v.u[1] = (x < 0) ? -1. : 1.;
  }
  return v;
}

static PetscScalar IceStrength(User user,PetscScalar h) {
  PetscReal A = 0.98;                                                       /* Ice concentration */
  return user->Pstar * h * PetscExpScalar(-user->Concentration * (1 - A)); /* LKT+12 Eq. 5 */
}

#undef __FUNCT__
#define __FUNCT__ "EffectiveViscosity"
static PetscErrorCode EffectiveViscosity(User user,PetscScalar h,PetscScalar P,const PetscScalar e[2][2],PetscScalar *zeta,PetscScalar *eta)
{
  PetscScalar Delta,zeta_max;

  PetscFunctionBegin;
  Delta = PetscSqrtScalar((PetscSqr(e[0][0]) + PetscSqr(e[1][1]))*(1 + 1/PetscSqr(user->ellipse_ratio))
                          + 4*PetscSqr(e[0][1]/user->ellipse_ratio)
                          + 2*e[0][0]*e[1][1]*(1 - 1/PetscSqr(user->ellipse_ratio)));
  zeta_max = user->zeta_max_coeff * P;
  *zeta = zeta_max * PetscTanhScalar(P / (2 * Delta * zeta_max)); /* bulk modulus */
  *eta = *zeta / PetscSqr(user->ellipse_ratio);                   /* shear modulus */

  user->range_zeta[0] = PetscMin(user->range_zeta[0],PetscRealPart(*zeta));
  user->range_zeta[1] = PetscMax(user->range_zeta[1],PetscRealPart(*zeta));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "WaterDrag"
static PetscErrorCode WaterDrag(User user,CoordField c,Field x,Field *tau_w)
{
  Field uwater,u;
  PetscScalar umag;
  PetscInt i;

  PetscFunctionBegin;
  uwater = VelocityWater(user,c);
  for (i=0; i<2; i++) u.u[i] = uwater.u[i] - x.u[i];
  umag = PetscSqrtScalar(PetscSqr(u.u[0]) + PetscSqr(u.u[1]));
  for (i=0; i<2; i++) {
    tau_w->u[i] = user->rho_ice * user->Cdrag_water * umag
      * (u.u[i] * PetscCosReal(user->theta_drag_water)
         + (i?1:-1) * u.u[(i+1)%2] * PetscSinReal(user->theta_drag_water));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PointwiseResidual"
static PetscErrorCode PointwiseResidual(User user,CoordField c,PetscReal time,Field x,const Field dx[],Field xdot,Field *f,Field df[])
{
  PetscErrorCode ierr;
  PetscScalar h,m,edot[2][2],P,zeta,eta;
  Field tau_w;
  PetscInt i,j;

  PetscFunctionBegin;
  h = IceThickness(user,c);
  m = user->rho_ice * h;
  P = IceStrength(user,h);

  StrainRate(dx,edot);
  ierr = EffectiveViscosity(user,h,P,edot,&zeta,&eta);CHKERRQ(ierr);
  ierr = WaterDrag(user,c,x,&tau_w);CHKERRQ(ierr);
  for (i=0; i<2; i++) {
    f->u[i] = m * xdot.u[i]
      + m * user->Coriolis_f * (i?1:-1) * x.u[(i+1)%2]
      - tau_w.u[i];
    for (j=0; j<2; j++) {       /* 2 \eta \dot\epsilon + (\zeta-\eta) \trace(\dot\epsilon) I - P/2 I */
      df[j].u[i] = 2 * eta * edot[i][j]
        + (i==j) * ((zeta - eta) * (edot[0][0] + edot[1][1]) - P/2);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PointwiseJacobian"
static PetscErrorCode PointwiseJacobian(User user,CoordField c,PetscReal time,Field x,const Field dx[],Field xdot,PetscReal shift,Field y,const Field dy[],Field *g,Field dg[])
{

  PetscFunctionBegin;
  g->u[0] = shift*y.u[0];
  g->u[1] = shift*y.u[1];
  dg[0].u[0] = dy[0].u[0];
  dg[1].u[0] = dy[1].u[0];
  dg[0].u[1] = dy[0].u[1];
  dg[1].u[1] = dy[1].u[1];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal"
static PetscErrorCode FormFunctionLocal(DMDALocalInfo *info,PetscReal time,Field **x,Field **xdot,Field **f,void *ptr)
{
  User        user = (User)ptr;
  DM          cda;
  CoordField  **c;
  Vec         C;
  PetscInt    i,j,k,l,xes,yes,xee,yee;
  PetscScalar *B,*D;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  user->range_zeta[0] = PETSC_MAX_REAL;
  user->range_zeta[1] = PETSC_MIN_REAL;

  ierr = DMGetCoordinateDM(info->da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(info->da,&C);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cda,C,&c);CHKERRQ(ierr);

  ierr = PetscFEGetDefaultTabulation(user->fe,&B,&D,NULL);CHKERRQ(ierr);
  /* loop over interior nodes */
  for (j=info->ys; j<info->ys+info->ym; j++) {
    for (i=info->xs; i<info->xs+info->xm; i++) {
      f[j][i].u[0] = 0.;
      f[j][i].u[1] = 0.;
    }
  }
  GetElementRange(info,&xes,&xee,&yes,&yee);
  for (j=yes; j<yee; j++) {
    for (i=xes; i<xee; i++) {
      Field xe[4],xdote[4],fe[4],*feref[4];
      Field *xq = user->xq,*dxq = user->Dxq,*xdotq = user->xdotq,*fq = user->fq,*dfq = user->Dfq;
      const PetscReal *weights,*De;
      CoordField ce[4],*cq = user->cq;
      QuadExtract(x,i,j,xe);
      QuadExtract(xdot,i,j,xdote);
      QuadExtractRef(info,f,i,j,feref);
      QuadExtractCoord(c,i,j,ce);
      QuadRealSpace(user,ce,D,&weights,&De);
      QuadMult(&user->q,1,B,xe,xq);
      QuadMult(&user->q,1,B,xdote,xdotq);
      QuadMult(&user->q,2,De,xe,dxq);
      QuadMultCoord(&user->q,1,B,ce,cq);

      for (k=0; k<user->q.numPoints; k++) {
        ierr = PointwiseResidual(user,cq[k],time,xq[k],&dxq[k*2],xdotq[k],&fq[k],&dfq[k*2]);CHKERRQ(ierr);
        fq[k].u[0] *= weights[k];
        fq[k].u[1] *= weights[k];
        for (l=0; l<2; l++) {
          dfq[k*2+l].u[0] *= weights[k];
          dfq[k*2+l].u[1] *= weights[k];
        }
      }
      ierr = PetscMemzero(fe,sizeof fe);CHKERRQ(ierr);
      QuadMultTransposeAdd(&user->q,1,B,fq,fe);
      QuadMultTransposeAdd(&user->q,2,De,dfq,fe);
      for (k=0; k<4; k++) {
        if (feref[k]) {
          feref[k]->u[0] += fe[k].u[0];
          feref[k]->u[1] += fe[k].u[1];
        }
      }
    }
  }
  ierr = DMDAVecRestoreArray(cda,C,&c);CHKERRQ(ierr);

  if (user->monitor_range) {
    user->range_zeta[1] *= -1;
    ierr = MPI_Allreduce(MPI_IN_PLACE,user->range_zeta,2,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)info->da));CHKERRQ(ierr);
    user->range_zeta[1] *= -1;
    ierr = PetscPrintf(PetscObjectComm((PetscObject)info->da),"Ranges: zeta [%G,%G]\n",user->range_zeta[0],user->range_zeta[1]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormJacobianLocal"
static PetscErrorCode FormJacobianLocal(DMDALocalInfo *info,PetscReal time,Field **x,Field **xdot,PetscReal shift,Mat Amat,Mat Pmat,MatStructure *mstructure,void *ptr)
{
  User        user = (User)ptr;
  DM          cda;
  CoordField  **c;
  Vec         C;
  PetscInt    i,j,k,xes,yes,xee,yee;
  PetscScalar *B,*D;
  MatNullSpace nullsp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinateDM(info->da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(info->da,&C);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cda,C,&c);CHKERRQ(ierr);

  ierr = PetscFEGetDefaultTabulation(user->fe,&B,&D,NULL);CHKERRQ(ierr);

  ierr = MatSetOption(Pmat,MAT_IGNORE_OFF_PROC_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatZeroEntries(Pmat);CHKERRQ(ierr);
  GetElementRange(info,&xes,&xee,&yes,&yee);
  for (j=yes; j<yee; j++) {
    for (i=xes; i<xee; i++) {
      const MatStencil rowcol[4] = {{0,j,i,0},{0,j,i+1,0},{0,j+1,i+1,0},{0,j+1,i,0}};
      const PetscReal *weights,*De;
      Field xe[4],xdote[4],*xq = user->xq,*dxq = user->Dxq,*xdotq = user->xdotq;
      CoordField ce[4],*cq = user->cq;
      PetscScalar K[4*2][4*2];
      PetscInt p,q,qf;
      QuadExtract(x,i,j,xe);
      QuadExtract(xdot,i,j,xdote);
      QuadExtractCoord(c,i,j,ce);
      QuadRealSpace(user,ce,D,&weights,&De);
      QuadMult(&user->q,1,B,xe,xq);
      QuadMult(&user->q,1,B,xdote,xdotq);
      QuadMult(&user->q,2,De,xe,dxq);
      QuadMultCoord(&user->q,1,B,ce,cq);

      ierr = PetscMemzero(K,sizeof K);CHKERRQ(ierr);
      for (k=0; k<user->q.numPoints; k++) {
        for (p=0; p<4*2; p++) {
          PetscInt pfield = p%2 ? 1 : 0,ofield = (pfield+1)%2;
          Field yp,dyp[2],g,dg[2];
          yp.u[pfield] = B[k*4+p/2];
          yp.u[ofield] = 0;
          dyp[0].u[pfield] = De[(k*2+0)*4+p/2];
          dyp[1].u[pfield] = De[(k*2+1)*4+p/2];
          dyp[0].u[ofield] = 0;
          dyp[1].u[ofield] = 0;
          ierr = PointwiseJacobian(user,cq[k],time,xq[k],&dxq[k*2],xdotq[k],shift,yp,dyp,&g,dg);CHKERRQ(ierr);
          for (q=0; q<4; q++) {
            for (qf=0; qf<2; qf++) {
              K[q*2+qf][p] += B[k*4+q] * weights[k] * g.u[qf]
                + De[(k*2+0)*4+q] * weights[k] * dg[0].u[qf]
                + De[(k*2+1)*4+q] * weights[k] * dg[1].u[qf];
            }
          }
        }
      }
      ierr = MatSetValuesBlockedStencil(Pmat,4,rowcol,4,rowcol,&K[0][0],ADD_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = DMDAVecRestoreArray(cda,C,&c);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(Pmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Pmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatGetNearNullSpace(Pmat,&nullsp);CHKERRQ(ierr);
  if (!nullsp) {
    Vec coords;
    ierr = DMGetCoordinates(info->da,&coords);CHKERRQ(ierr);
    ierr = MatNullSpaceCreateRigidBody(coords,&nullsp);CHKERRQ(ierr);
    ierr = MatSetNearNullSpace(Pmat,nullsp);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullsp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "InitialGuess"
static PetscErrorCode InitialGuess(DM da,User user,Vec X)
{
  DMDALocalInfo  info;
  PetscInt       i,j;
  PetscErrorCode ierr;
  DM             cda;
  Vec            Coord;
  Field          **x;
  CoordField     **c;

  PetscFunctionBegin;
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(da,&cda);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(da,&Coord);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cda,Coord,&c);CHKERRQ(ierr);

  for (j=info.ys; j<info.ys+info.ym; j++) {
    for (i=info.xs; i<info.xs+info.xm; i++) {
      x[j][i] = VelocityWater(user,c[j][i]);
    }
  }
  ierr = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(cda,Coord,&c);CHKERRQ(ierr);

  if (user->view_initial) {
    PetscViewer viewer;
    ierr = PetscViewerVTKOpen(PetscObjectComm((PetscObject)da),"si-initial.vts",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = VecView(X,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetupElement"
static PetscErrorCode SetupElement(MPI_Comm comm,User user)
{
  const PetscInt  dim = 2;
  PetscFE         fem;
  PetscQuadrature q;
  DM              K;
  PetscSpace      P;
  PetscDualSpace  Q;
  PetscInt        order;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* Create space */
  ierr = PetscSpaceCreate(comm,&P);CHKERRQ(ierr);
  ierr = PetscSpaceSetOrder(P,1);CHKERRQ(ierr);
  ierr = PetscSpaceSetFromOptions(P);CHKERRQ(ierr);
  ierr = PetscSpacePolynomialSetTensor(P,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscSpacePolynomialSetNumVariables(P,dim);CHKERRQ(ierr);
  ierr = PetscSpaceSetUp(P);CHKERRQ(ierr);
  ierr = PetscSpaceGetOrder(P,&order);CHKERRQ(ierr);
  /* Create dual space */
  ierr = PetscDualSpaceCreate(comm,&Q);CHKERRQ(ierr);
  ierr = PetscDualSpaceCreateReferenceCell(Q,dim,PETSC_FALSE,&K);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetDM(Q,K);CHKERRQ(ierr);
  ierr = DMDestroy(&K);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetOrder(Q,order);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetFromOptions(Q);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetUp(Q);CHKERRQ(ierr);
  /* Create element */
  ierr = PetscFECreate(comm,&fem);CHKERRQ(ierr);
  ierr = PetscFESetFromOptions(fem);CHKERRQ(ierr);
  ierr = PetscFESetBasisSpace(fem,P);CHKERRQ(ierr);
  ierr = PetscFESetDualSpace(fem,Q);CHKERRQ(ierr);
  ierr = PetscFESetNumComponents(fem,1);CHKERRQ(ierr);
  ierr = PetscSpaceDestroy(&P);CHKERRQ(ierr);
  ierr = PetscDualSpaceDestroy(&Q);CHKERRQ(ierr);
  {                             /* Create tensor product Gauss quadrature */
    PetscReal x[2],w[2],*points,*weights;
    PetscInt i,j;
    ierr = PetscDTGaussQuadrature(2,-1,1,x,w);CHKERRQ(ierr);
    q.dim = 2;
    q.numPoints = 4;
    ierr = PetscMalloc1(q.dim*q.numPoints,&points);CHKERRQ(ierr);
    ierr = PetscMalloc1(q.numPoints,&weights);CHKERRQ(ierr);
    for (i=0; i<2; i++) {
      for (j=0; j<2; j++) {
        points[(i*2+j)*2+0] = x[i];
        points[(i*2+j)*2+1] = x[j];
        weights[i*2+j]      = w[i]*w[j];
      }
    }
    q.points = points;
    q.weights = weights;
  }
  ierr = PetscFESetQuadrature(fem,q);CHKERRQ(ierr);
  user->fe = fem;
  user->q  = q;

  ierr = PetscMalloc5(q.numPoints*2*4,&user->De,q.numPoints,&user->weights,q.numPoints,&user->xq,q.numPoints*2,&user->Dxq,q.numPoints,&user->xdotq);CHKERRQ(ierr);
  ierr = PetscMalloc3(q.numPoints,&user->fq,q.numPoints*2,&user->Dfq,q.numPoints,&user->cq);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "UserSetFromOptions"
static PetscErrorCode UserSetFromOptions(User user)
{
  PetscErrorCode ierr;
  PetscInt n;
  PetscReal L[2] = {1e4,1e4};

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Sea ice solver options",NULL);CHKERRQ(ierr);
  n = 2;
  ierr = PetscOptionsRealArray("-L","Length of domain in x and y directions [m]","",L,&n,NULL);CHKERRQ(ierr);
  user->L.x[0] = L[0];
  user->L.x[1] = L[1];

  user->Coriolis_f = 1.46e-4;
  ierr = PetscOptionsReal("-coriolis_f","Coriolis parameter [s^-1]","",user->Coriolis_f,&user->Coriolis_f,NULL);CHKERRQ(ierr);

  user->rho_ice = 900;
  ierr = PetscOptionsReal("-rho_ice","Desity of sea ice [kg m^-3]","",user->rho_ice,&user->rho_ice,NULL);CHKERRQ(ierr);

  user->rho_air = 1.3;
  ierr = PetscOptionsReal("-rho_air","Desity of air [kg m^-3]","",user->rho_air,&user->rho_air,NULL);CHKERRQ(ierr);

  user->rho_water = 1026;
  ierr = PetscOptionsReal("-rho_water","Desity of ocean water [kg m^-3]","",user->rho_water,&user->rho_water,NULL);CHKERRQ(ierr);

  user->Cdrag_air = 1.2e-3;
  ierr = PetscOptionsReal("-Cdrag_air","Air drag coefficient []","",user->Cdrag_air,&user->Cdrag_air,NULL);CHKERRQ(ierr);

  user->Cdrag_water = 5.5-3;
  ierr = PetscOptionsReal("-Cdrag_water","Water drag coefficient []","",user->Cdrag_water,&user->Cdrag_water,NULL);CHKERRQ(ierr);

  user->theta_drag_air = 25;
  ierr = PetscOptionsReal("-theta_drag_air","Air drag turning angle [degrees]","",user->theta_drag_air,&user->theta_drag_air,NULL);CHKERRQ(ierr);
  user->theta_drag_air *= PETSC_PI/180;

  user->theta_drag_water = 25;
  ierr = PetscOptionsReal("-theta_drag_water","Water drag turning angle [degrees]","",user->theta_drag_water,&user->theta_drag_water,NULL);CHKERRQ(ierr);
  user->theta_drag_water *= PETSC_PI/180;

  user->Pstar = 27.5e3;
  ierr = PetscOptionsReal("-Pstar","Ice strength parameter [N m^-2]","",user->Pstar,&user->Pstar,NULL);CHKERRQ(ierr);

  user->ellipse_ratio = 2;
  ierr = PetscOptionsReal("-ellipse_ratio","Ellipse ratio in constitutive model []","",user->ellipse_ratio,&user->ellipse_ratio,NULL);CHKERRQ(ierr);

  user->Concentration = 20;
  ierr = PetscOptionsReal("-Concentration","Ice concentration parameter []","",user->Concentration,&user->Concentration,NULL);CHKERRQ(ierr);

  user->zeta_max_coeff = 2.5e8;
  ierr = PetscOptionsReal("-zeta_max_coeff","Coefficient for limiting zeta []","",user->zeta_max_coeff,&user->zeta_max_coeff,NULL);CHKERRQ(ierr);

  user->view_initial = PETSC_FALSE;
  ierr = PetscOptionsBool("-view_initial","View initial velocity solution","",user->view_initial,&user->view_initial,NULL);CHKERRQ(ierr);

  user->monitor_range = PETSC_FALSE;
  ierr = PetscOptionsBool("-monitor_range","Monitor range of effective viscosity","",user->monitor_range,&user->monitor_range,NULL);CHKERRQ(ierr);

  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  struct _User   user;
  PetscInt       its,stepno;
  PetscErrorCode ierr;
  MPI_Comm       comm;
  TS             ts;
  DM             da;
  Vec            X;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return(1);
  comm = PETSC_COMM_WORLD;
  ierr = UserSetFromOptions(&user);CHKERRQ(ierr);
  ierr = SetupElement(comm,&user);CHKERRQ(ierr);

  ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_STENCIL_BOX,-8,-8,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,-user.L.x[0],user.L.x[0],-user.L.x[1],user.L.x[1],0.0,0.0);CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,0,"ux");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,1,"uy");CHKERRQ(ierr);

  ierr = TSCreate(comm,&ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts,da);CHKERRQ(ierr);
  ierr = DMDATSSetIFunctionLocal(da,INSERT_VALUES,(DMDATSIFunctionLocal)FormFunctionLocal,&user);CHKERRQ(ierr);
  ierr = DMDATSSetIJacobianLocal(da,(DMDATSIJacobianLocal)FormJacobianLocal,&user);CHKERRQ(ierr);
  ierr = TSSetEquationType(ts,TS_EQ_IMPLICIT);CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,0.,30.);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,100,24*3600);CHKERRQ(ierr);
  ierr = TSSetMaxSNESFailures(ts,-1);CHKERRQ(ierr); /* Retry forever */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da,&X);CHKERRQ(ierr);
  ierr = InitialGuess(da,&user,X);CHKERRQ(ierr);

  ierr = TSSolve(ts,X);CHKERRQ(ierr);

  ierr = TSGetTimeStepNumber(ts,&stepno);CHKERRQ(ierr);
  ierr = TSGetSNESIterations(ts,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"TS steps %D, SNES iterations %D\n",stepno,its);CHKERRQ(ierr);

  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&user.fe);CHKERRQ(ierr);
  ierr = PetscFree5(user.De,user.weights,user.xq,user.Dxq,user.xdotq);CHKERRQ(ierr);
  ierr = PetscFree3(user.fq,user.Dfq,user.cq);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
