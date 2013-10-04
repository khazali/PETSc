static char help[] = "Second order lapacian finite volume convergance test.\n";
/*F

  A convergance test to verify that the equation solver (ie, one full multigrid iteration) is accuarte enough to provide second order accuracy. This is intended a coarse grid solver and comparison solver for a block structured segmental refinement (SR) solver [a la Brandt & Diskin 1995].

 * 9/27 point finite volume discretization of the Laplacian (L) with constant material coeficient.

 * Solve Lu=f, with homogenous Diriclet boundary conditions on [0,1]^D, where u = (x^2-x^4)*(y^4-y^2) in 2D, found on page 64 of
    `A MULTIGRID TUTORIAL by Briggs, Henson & McCormick'

 * Refine mesh and print order of convergence.

 This is a simplified derivative of ex11 by Matt Knepley.

F*/

#include <petscts.h>
#include <petscdmplex.h>
#include <petscsf.h>
#include <petscblaslapack.h>
#if defined(PETSC_HAVE_EXODUSII)
#include <exodusII.h>
#else
//#error This example requires ExodusII support. Reconfigure using --download-exodusii
#endif

#define DIM 2 /* Geometric dimension */
#define ALEN(a) (sizeof(a)/sizeof((a)[0]))

static PetscFunctionList PhysicsList;

/* Represents continuum physical equations. */
typedef struct _n_Physics *Physics;

/* Physical model includes boundary conditions, initial conditions, and functionals of interest. It is
 * discretization-independent, but its members depend on the scenario being solved. */
typedef struct _n_Model *Model;

/* 'User' implements a discretization of a continuous model. */
typedef struct _n_User *User;

typedef PetscErrorCode (*RiemannFunction)(Physics,const PetscReal*,const PetscReal*,const PetscScalar*,const PetscScalar*,PetscScalar*);
typedef PetscErrorCode (*SolutionFunction)(Model,PetscReal,const PetscReal*,PetscScalar*,void*);
typedef PetscErrorCode (*BoundaryFunction)(Model,PetscReal,const PetscReal*,const PetscReal*,const PetscScalar*,PetscScalar*,void*);
typedef PetscErrorCode (*FunctionalFunction)(Model,PetscReal,const PetscReal*,const PetscScalar*,PetscReal*,void*);
typedef PetscErrorCode (*SetupFields)(Physics,PetscSection);
static PetscErrorCode ModelBoundaryRegister(Model,const char*,BoundaryFunction,void*,PetscInt,const PetscInt*);
static PetscErrorCode ModelSolutionSetDefault(Model,SolutionFunction,void*);
static PetscErrorCode ModelFunctionalRegister(Model,const char*,PetscInt*,FunctionalFunction,void*);
static PetscErrorCode OutputVTK(DM,const char*,PetscViewer*);

struct FieldDescription {
  const char *name;
  PetscInt dof;
};

typedef struct _n_BoundaryLink *BoundaryLink;
struct _n_BoundaryLink {
  char             *name;
  BoundaryFunction func;
  void             *ctx;
  PetscInt         numids;
  PetscInt         *ids;
  BoundaryLink     next;
};

typedef struct _n_FunctionalLink *FunctionalLink;
struct _n_FunctionalLink {
  char               *name;
  FunctionalFunction func;
  void               *ctx;
  PetscInt           offset;
  FunctionalLink     next;
};

struct _n_Physics {
  RiemannFunction riemann;
  PetscInt        dof;          /* number of degrees of freedom per cell */
  void            *data;
  PetscInt        nfields;
  const struct FieldDescription *field_desc;
};

struct _n_Model {
  MPI_Comm         comm;        /* Does not do collective communicaton, but some error conditions can be collective */
  Physics          physics;
  BoundaryLink     boundary;
  FunctionalLink   functionalRegistry;
  PetscInt         maxComputed;
  PetscInt         numMonitored;
  FunctionalLink   *functionalMonitored;
  PetscInt         numCall;
  FunctionalLink   *functionalCall;
  SolutionFunction solution;
  void             *solutionctx;
};

struct _n_User {
  PetscErrorCode (*RHSFunctionLocal)(DM,DM,DM,PetscReal,Vec,Vec,User);
  PetscReal      (*Limit)(PetscReal);
  PetscInt       numGhostCells, numSplitFaces;
  PetscInt       cEndInterior; /* First boundary ghost cell */
  Vec            cellgeom, facegeom;
  DM             dmGrad;
  PetscReal      minradius;
  PetscInt       vtkInterval;   /* For monitor */
  Model          model;
  struct {
    PetscScalar *flux;
    PetscScalar *state0;
    PetscScalar *state1;
  } work;
};

typedef struct {
  PetscScalar normal[DIM];              /* Area-scaled normals */
  PetscScalar centroid[DIM];            /* Location of centroid (quadrature point) */
  PetscScalar grad[2][DIM];             /* Face contribution to gradient in left and right cell */
} FaceGeom;

typedef struct {
  PetscScalar centroid[DIM];
  PetscScalar volume;
} CellGeom;


PETSC_STATIC_INLINE PetscScalar DotDIM(const PetscScalar *x,const PetscScalar *y)
{
  PetscInt    i;
  PetscScalar prod=0.0;

  for (i=0; i<DIM; i++) prod += x[i]*y[i];
  return prod;
}
PETSC_STATIC_INLINE PetscReal NormDIM(const PetscScalar *x) { return PetscSqrtReal(PetscAbsScalar(DotDIM(x,x))); }
PETSC_STATIC_INLINE void axDIM(const PetscScalar a,PetscScalar *x)
{
  PetscInt i;
  for (i=0; i<DIM; i++) x[i] *= a;
}
PETSC_STATIC_INLINE void waxDIM(const PetscScalar a,const PetscScalar *x, PetscScalar *w)
{
  PetscInt i;
  for (i=0; i<DIM; i++) w[i] = x[i]*a;
}
PETSC_STATIC_INLINE void NormalSplitDIM(const PetscReal *n,const PetscScalar *x,PetscScalar *xn,PetscScalar *xt)
{                               /* Split x into normal and tangential components */
  PetscInt    i;
  PetscScalar c;
  c = DotDIM(x,n)/DotDIM(n,n);
  for (i=0; i<DIM; i++) {
    xn[i] = c*n[i];
    xt[i] = x[i]-xn[i];
  }
}

PETSC_STATIC_INLINE PetscScalar Dot2(const PetscScalar *x,const PetscScalar *y) { return x[0]*y[0] + x[1]*y[1];}
PETSC_STATIC_INLINE PetscReal Norm2(const PetscScalar *x) { return PetscSqrtReal(PetscAbsScalar(Dot2(x,x)));}
PETSC_STATIC_INLINE void Normalize2(PetscScalar *x) { PetscReal a = 1./Norm2(x); x[0] *= a; x[1] *= a; }
PETSC_STATIC_INLINE void Waxpy2(PetscScalar a,const PetscScalar *x,const PetscScalar *y,PetscScalar *w) { w[0] = a*x[0] + y[0]; w[1] = a*x[1] + y[1]; }
PETSC_STATIC_INLINE void Scale2(PetscScalar a,const PetscScalar *x,PetscScalar *y) { y[0] = a*x[0]; y[1] = a*x[1]; }

PETSC_STATIC_INLINE void WaxpyD(PetscInt dim, PetscScalar a, const PetscScalar *x, const PetscScalar *y, PetscScalar *w) {PetscInt d; for (d = 0; d < dim; ++d) w[d] = a*x[d] + y[d];}
PETSC_STATIC_INLINE PetscScalar DotD(PetscInt dim, const PetscScalar *x, const PetscScalar *y) {PetscScalar sum = 0.0; PetscInt d; for (d = 0; d < dim; ++d) sum += x[d]*y[d]; return sum;}
PETSC_STATIC_INLINE PetscReal NormD(PetscInt dim, const PetscScalar *x) {return PetscSqrtReal(PetscAbsScalar(DotD(dim,x,x)));}

PETSC_STATIC_INLINE void NormalSplit(const PetscReal *n,const PetscScalar *x,PetscScalar *xn,PetscScalar *xt)
{                               /* Split x into normal and tangential components */
  Scale2(Dot2(x,n)/Dot2(n,n),n,xn);
  Waxpy2(-1,xn,x,xt);
}

 /* RHS */
PetscScalar rhsFunc(PetscScalar x[DIM])
{
  PetscScalar r;
  PetscScalar x2[DIM]; /* = x*x; */
  PetscScalar a[DIM]; /* = x2*(1-x2) */
  PetscScalar b[DIM]; /* = 2-12*x2; */
  PetscInt d;

  for (d = 0; d < DIM; ++d) x2[d] = x[d]*x[d];
  for (d = 0; d < DIM; ++d) a[d] = x2[d]*(1-x2[d]);
  for (d = 0; d < DIM; ++d) b[d] = 2-12*x2[d];

  /* cross products of coordinates prevents the use of D_TERM */
  if (DIM==1)
    r = b[0];
  else if (DIM==2)
    r = b[0]*a[1] + a[0]*b[1];
  else if (DIM==3)
    r = b[0]*a[1]*a[2] + a[0]*b[1]*a[2] + a[0]*a[1]*b[2];
  else
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for solution this DIM %d",DIM);
  return -r; // lets keep soluion positive
}

// exact u
PetscScalar exactSolution(PetscScalar x[DIM])
{
  PetscScalar r;
  PetscScalar e[DIM];
  PetscInt d;
  /* RealVect e = a_x*a_x; */
  /* e *= 1. - e; */
  /* return e.product(); */
  for (d = 0; d < DIM; ++d) e[d] = x[d]*x[d];
  for (d = 0; d < DIM; ++d) e[d] *= 1 - e[d];
  for (r = 1.0, d = 0; d < DIM; ++d) r *= e[d];
  return r;
}

/******************* Diff ********************/
typedef enum {DIFF_X4_X2} DiffSolType;
static const char *const DiffSolTypes[] = {"x**4-x**2",0};

typedef struct {
  PetscReal dummy;
} Physics_Diff_x4_x2;

typedef struct {
  DiffSolType soltype;
  union {
    Physics_Diff_x4_x2 x4_x2;
  } sol;
  struct {
    PetscInt Error;
  } functional;
} Physics_Diff;

static const struct FieldDescription PhysicsFields_Diff[] = {{"U",1},{NULL,0}};

#undef __FUNCT__
#define __FUNCT__ "PhysicsFlux_Diff"
static PetscErrorCode PhysicsFlux_Diff(Physics phys, const PetscReal *qp, const PetscReal *n, const PetscScalar *xL, const PetscScalar *xR, PetscScalar *flux)
{
  Physics_Diff *diff = (Physics_Diff*)phys->data;
  PetscFunctionBeginUser;
  switch (diff->soltype) {
  case DIFF_X4_X2: {
    flux[0] = 0.5*(xL[0] + xR[0]);
  } break;
  default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for solution type");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PhysicsSolution_Diff"
static PetscErrorCode PhysicsSolution_Diff(Model mod,PetscReal time,const PetscReal *x,PetscScalar *u,void *ctx)
{
  Physics        phys    = (Physics)ctx;
  Physics_Diff *diff = (Physics_Diff*)phys->data;

  PetscFunctionBeginUser;
  switch (advect->soltype) {
  case DIFF_X4_X2: {
    u[0] = exactSolution(x);
  } break;
  default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unknown solution type");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PhysicsFunctional_Diff"
static PetscErrorCode PhysicsFunctional_Diff(Model mod,PetscReal time,const PetscScalar *x,const PetscScalar *y,PetscReal *f,void *ctx)
{
  Physics        phys    = (Physics)ctx;
  Physics_Diff *diff = (Physics_Diff*)phys->data;
  PetscScalar    yexact[1];
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PhysicsSolution_Diff(mod,time,x,yexact,phys);CHKERRQ(ierr);
  f[diff->functional.Error] = PetscAbsScalar(y[0]-yexact[0]);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PhysicsCreate_Diff"
static PetscErrorCode PhysicsCreate_Diff(Model mod,Physics phys)
{
  Physics_Diff *diff = (Physics_Diff*)phys->data;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  phys->field_desc = PhysicsFields_Diff;
  phys->riemann = PhysicsFlux_Diff;
  ierr = PetscNew(Physics_Diff,&phys->data);CHKERRQ(ierr);
  diff = phys->data;
  ierr = PetscOptionsHead("Diff options");CHKERRQ(ierr);
  {
    diff->soltype = DIFF_X4_X2;
    ierr = PetscOptionsEnum("-diff_sol_type","solution type","",DiffSolTypes,(PetscEnum)diff->soltype,(PetscEnum*)&diff->soltype,NULL);CHKERRQ(ierr);
    switch (diff->soltype) {
    case DIFF_X4_X2: {

    } break;
    default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unknown solution type");
    }
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);

  {
    /* Register "canned" boundary conditions and defaults for where to apply. */
    ierr = ModelBoundaryRegister(mod,"diri",PhysicsBoundary_Diff_Diri,phys,ALEN(inflowids),inflowids);CHKERRQ(ierr);
    /* Initial/transient solution with default boundary conditions */
    ierr = ModelSolutionSetDefault(mod,PhysicsSolution_Diff,phys);CHKERRQ(ierr);
    /* Register "canned" functionals */
    ierr = ModelFunctionalRegister(mod,"Error",&advect->functional.Error,PhysicsFunctional_Diff,phys);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ConstructCellBoundary"
PetscErrorCode ConstructCellBoundary(DM dm, User user)
{
  const char     *name   = "Cell Sets";
  const char     *bdname = "split faces";
  IS             regionIS, innerIS;
  const PetscInt *regions, *cells;
  PetscInt       numRegions, innerRegion, numCells, c;

  PetscInt cStart, cEnd, fStart, fEnd;

  PetscBool      hasLabel;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);

  ierr = DMPlexHasLabel(dm, name, &hasLabel);CHKERRQ(ierr);
  if (!hasLabel) PetscFunctionReturn(0);
  ierr = DMPlexGetLabelSize(dm, name, &numRegions);CHKERRQ(ierr);
  if (numRegions != 2) PetscFunctionReturn(0);
  /* Get the inner id */
  ierr = DMPlexGetLabelIdIS(dm, name, &regionIS);CHKERRQ(ierr);
  ierr = ISGetIndices(regionIS, &regions);CHKERRQ(ierr);
  innerRegion = regions[0];
  ierr = ISRestoreIndices(regionIS, &regions);CHKERRQ(ierr);
  ierr = ISDestroy(&regionIS);CHKERRQ(ierr);
  /* Find the faces between cells in different regions, could call DMPlexCreateNeighborCSR() */
  ierr = DMPlexGetStratumIS(dm, name, innerRegion, &innerIS);CHKERRQ(ierr);
  ierr = ISGetLocalSize(innerIS, &numCells);CHKERRQ(ierr);
  ierr = ISGetIndices(innerIS, &cells);CHKERRQ(ierr);
  ierr = DMPlexCreateLabel(dm, bdname);CHKERRQ(ierr);
  for (c = 0; c < numCells; ++c) {
    const PetscInt cell = cells[c];
    const PetscInt *faces;
    PetscInt       numFaces, f;

    if ((cell < cStart) || (cell >= cEnd)) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "Got invalid point %d which is not a cell", cell);
    ierr = DMPlexGetConeSize(dm, cell, &numFaces);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, cell, &faces);CHKERRQ(ierr);
    for (f = 0; f < numFaces; ++f) {
      const PetscInt face = faces[f];
      const PetscInt *neighbors;
      PetscInt       nC, regionA, regionB;

      if ((face < fStart) || (face >= fEnd)) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "Got invalid point %d which is not a face", face);
      ierr = DMPlexGetSupportSize(dm, face, &nC);CHKERRQ(ierr);
      if (nC != 2) continue;
      ierr = DMPlexGetSupport(dm, face, &neighbors);CHKERRQ(ierr);
      if ((neighbors[0] >= user->cEndInterior) || (neighbors[1] >= user->cEndInterior)) continue;
      if ((neighbors[0] < cStart) || (neighbors[0] >= cEnd)) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "Got invalid point %d which is not a cell", neighbors[0]);
      if ((neighbors[1] < cStart) || (neighbors[1] >= cEnd)) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "Got invalid point %d which is not a cell", neighbors[1]);
      ierr = DMPlexGetLabelValue(dm, name, neighbors[0], &regionA);CHKERRQ(ierr);
      ierr = DMPlexGetLabelValue(dm, name, neighbors[1], &regionB);CHKERRQ(ierr);
      if (regionA < 0) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Invalid label %s: Cell %d has no value", name, neighbors[0]);
      if (regionB < 0) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Invalid label %s: Cell %d has no value", name, neighbors[1]);
      if (regionA != regionB) {
        ierr = DMPlexSetLabelValue(dm, bdname, faces[f], 1);CHKERRQ(ierr);
      }
    }
  }
  ierr = ISRestoreIndices(innerIS, &cells);CHKERRQ(ierr);
  ierr = ISDestroy(&innerIS);CHKERRQ(ierr);
  {
    DMLabel label;

    ierr = PetscViewerASCIISynchronizedAllow(PETSC_VIEWER_STDOUT_WORLD, PETSC_TRUE);CHKERRQ(ierr);
    ierr = DMPlexGetLabel(dm, bdname, &label);CHKERRQ(ierr);
    ierr = DMLabelView(label, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SplitFaces"
/* Right now, I have just added duplicate faces, which see both cells. We can
- Add duplicate vertices and decouple the face cones
- Disconnect faces from cells across the rotation gap
*/
PetscErrorCode SplitFaces(DM *dmSplit, const char labelName[], User user)
{
  DM             dm = *dmSplit, sdm;
  PetscSF        sfPoint, gsfPoint;
  PetscSection   coordSection, newCoordSection;
  Vec            coordinates;
  IS             idIS;
  const PetscInt *ids;
  PetscInt       *newpoints;
  PetscInt       dim, depth, maxConeSize, maxSupportSize, numLabels;
  PetscInt       numFS, fs, pStart, pEnd, p, vStart, vEnd, v, fStart, fEnd, newf, d, l;
  PetscBool      hasLabel;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexHasLabel(dm, labelName, &hasLabel);CHKERRQ(ierr);
  if (!hasLabel) PetscFunctionReturn(0);
  ierr = DMCreate(PetscObjectComm((PetscObject)dm), &sdm);CHKERRQ(ierr);
  ierr = DMSetType(sdm, DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexSetDimension(sdm, dim);CHKERRQ(ierr);

  ierr = DMPlexGetLabelIdIS(dm, labelName, &idIS);CHKERRQ(ierr);
  ierr = ISGetLocalSize(idIS, &numFS);CHKERRQ(ierr);
  ierr = ISGetIndices(idIS, &ids);CHKERRQ(ierr);

  user->numSplitFaces = 0;
  for (fs = 0; fs < numFS; ++fs) {
    PetscInt numBdFaces;

    ierr = DMPlexGetStratumSize(dm, labelName, ids[fs], &numBdFaces);CHKERRQ(ierr);
    user->numSplitFaces += numBdFaces;
  }
  ierr  = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  pEnd += user->numSplitFaces;
  ierr  = DMPlexSetChart(sdm, pStart, pEnd);CHKERRQ(ierr);
  /* Set cone and support sizes */
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  for (d = 0; d <= depth; ++d) {
    ierr = DMPlexGetDepthStratum(dm, d, &pStart, &pEnd);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      PetscInt newp = p;
      PetscInt size;

      ierr = DMPlexGetConeSize(dm, p, &size);CHKERRQ(ierr);
      ierr = DMPlexSetConeSize(sdm, newp, size);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, p, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(sdm, newp, size);CHKERRQ(ierr);
    }
  }
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  for (fs = 0, newf = fEnd; fs < numFS; ++fs) {
    IS             faceIS;
    const PetscInt *faces;
    PetscInt       numFaces, f;

    ierr = DMPlexGetStratumIS(dm, labelName, ids[fs], &faceIS);CHKERRQ(ierr);
    ierr = ISGetLocalSize(faceIS, &numFaces);CHKERRQ(ierr);
    ierr = ISGetIndices(faceIS, &faces);CHKERRQ(ierr);
    for (f = 0; f < numFaces; ++f, ++newf) {
      PetscInt size;

      /* Right now I think that both faces should see both cells */
      ierr = DMPlexGetConeSize(dm, faces[f], &size);CHKERRQ(ierr);
      ierr = DMPlexSetConeSize(sdm, newf, size);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, faces[f], &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(sdm, newf, size);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(faceIS, &faces);CHKERRQ(ierr);
    ierr = ISDestroy(&faceIS);CHKERRQ(ierr);
  }
  ierr = DMSetUp(sdm);CHKERRQ(ierr);
  /* Set cones and supports */
  ierr = DMPlexGetMaxSizes(dm, &maxConeSize, &maxSupportSize);CHKERRQ(ierr);
  ierr = PetscMalloc(PetscMax(maxConeSize, maxSupportSize) * sizeof(PetscInt), &newpoints);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    const PetscInt *points, *orientations;
    PetscInt       size, i, newp = p;

    ierr = DMPlexGetConeSize(dm, p, &size);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, p, &points);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientation(dm, p, &orientations);CHKERRQ(ierr);
    for (i = 0; i < size; ++i) newpoints[i] = points[i];
    ierr = DMPlexSetCone(sdm, newp, newpoints);CHKERRQ(ierr);
    ierr = DMPlexSetConeOrientation(sdm, newp, orientations);CHKERRQ(ierr);
    ierr = DMPlexGetSupportSize(dm, p, &size);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dm, p, &points);CHKERRQ(ierr);
    for (i = 0; i < size; ++i) newpoints[i] = points[i];
    ierr = DMPlexSetSupport(sdm, newp, newpoints);CHKERRQ(ierr);
  }
  ierr = PetscFree(newpoints);CHKERRQ(ierr);
  for (fs = 0, newf = fEnd; fs < numFS; ++fs) {
    IS             faceIS;
    const PetscInt *faces;
    PetscInt       numFaces, f;

    ierr = DMPlexGetStratumIS(dm, labelName, ids[fs], &faceIS);CHKERRQ(ierr);
    ierr = ISGetLocalSize(faceIS, &numFaces);CHKERRQ(ierr);
    ierr = ISGetIndices(faceIS, &faces);CHKERRQ(ierr);
    for (f = 0; f < numFaces; ++f, ++newf) {
      const PetscInt *points;

      ierr = DMPlexGetCone(dm, faces[f], &points);CHKERRQ(ierr);
      ierr = DMPlexSetCone(sdm, newf, points);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, faces[f], &points);CHKERRQ(ierr);
      ierr = DMPlexSetSupport(sdm, newf, points);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(faceIS, &faces);CHKERRQ(ierr);
    ierr = ISDestroy(&faceIS);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(idIS, &ids);CHKERRQ(ierr);
  ierr = ISDestroy(&idIS);CHKERRQ(ierr);
  ierr = DMPlexStratify(sdm);CHKERRQ(ierr);
  /* Convert coordinates */
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm), &newCoordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(newCoordSection, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(newCoordSection, 0, dim);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(newCoordSection, vStart, vEnd);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {
    ierr = PetscSectionSetDof(newCoordSection, v, dim);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(newCoordSection, v, 0, dim);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(newCoordSection);CHKERRQ(ierr);
  ierr = DMPlexSetCoordinateSection(sdm, newCoordSection);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&newCoordSection);CHKERRQ(ierr); /* relinquish our reference */
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(sdm, coordinates);CHKERRQ(ierr);
  /* Convert labels */
  ierr = DMPlexGetNumLabels(dm, &numLabels);CHKERRQ(ierr);
  for (l = 0; l < numLabels; ++l) {
    const char *lname;
    PetscBool  isDepth;

    ierr = DMPlexGetLabelName(dm, l, &lname);CHKERRQ(ierr);
    ierr = PetscStrcmp(lname, "depth", &isDepth);CHKERRQ(ierr);
    if (isDepth) continue;
    ierr = DMPlexCreateLabel(sdm, lname);CHKERRQ(ierr);
    ierr = DMPlexGetLabelIdIS(dm, lname, &idIS);CHKERRQ(ierr);
    ierr = ISGetLocalSize(idIS, &numFS);CHKERRQ(ierr);
    ierr = ISGetIndices(idIS, &ids);CHKERRQ(ierr);
    for (fs = 0; fs < numFS; ++fs) {
      IS             pointIS;
      const PetscInt *points;
      PetscInt       numPoints;

      ierr = DMPlexGetStratumIS(dm, lname, ids[fs], &pointIS);CHKERRQ(ierr);
      ierr = ISGetLocalSize(pointIS, &numPoints);CHKERRQ(ierr);
      ierr = ISGetIndices(pointIS, &points);CHKERRQ(ierr);
      for (p = 0; p < numPoints; ++p) {
        PetscInt newpoint = points[p];

        ierr = DMPlexSetLabelValue(sdm, lname, newpoint, ids[fs]);CHKERRQ(ierr);
      }
      ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
      ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(idIS, &ids);CHKERRQ(ierr);
    ierr = ISDestroy(&idIS);CHKERRQ(ierr);
  }
  /* Convert pointSF */
  const PetscSFNode *remotePoints;
  PetscSFNode       *gremotePoints;
  const PetscInt    *localPoints;
  PetscInt          *glocalPoints,*newLocation,*newRemoteLocation;
  PetscInt          numRoots, numLeaves;
  PetscMPIInt       numProcs;

  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dm), &numProcs);CHKERRQ(ierr);
  ierr = DMGetPointSF(dm, &sfPoint);CHKERRQ(ierr);
  ierr = DMGetPointSF(sdm, &gsfPoint);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm,&pStart,&pEnd);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sfPoint, &numRoots, &numLeaves, &localPoints, &remotePoints);CHKERRQ(ierr);
  if (numRoots >= 0) {
    ierr = PetscMalloc2(numRoots,PetscInt,&newLocation,pEnd-pStart,PetscInt,&newRemoteLocation);CHKERRQ(ierr);
    for (l=0; l<numRoots; l++) newLocation[l] = l; /* + (l >= cEnd ? user->numGhostCells : 0); */
    ierr = PetscSFBcastBegin(sfPoint, MPIU_INT, newLocation, newRemoteLocation);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sfPoint, MPIU_INT, newLocation, newRemoteLocation);CHKERRQ(ierr);
    ierr = PetscMalloc(numLeaves * sizeof(PetscInt),    &glocalPoints);CHKERRQ(ierr);
    ierr = PetscMalloc(numLeaves * sizeof(PetscSFNode), &gremotePoints);CHKERRQ(ierr);
    for (l = 0; l < numLeaves; ++l) {
      glocalPoints[l]        = localPoints[l]; /* localPoints[l] >= cEnd ? localPoints[l] + user->numGhostCells : localPoints[l]; */
      gremotePoints[l].rank  = remotePoints[l].rank;
      gremotePoints[l].index = newRemoteLocation[localPoints[l]];
    }
    ierr = PetscFree2(newLocation,newRemoteLocation);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(gsfPoint, numRoots+user->numGhostCells, numLeaves, glocalPoints, PETSC_OWN_POINTER, gremotePoints, PETSC_OWN_POINTER);CHKERRQ(ierr);
  }
  ierr     = DMDestroy(dmSplit);CHKERRQ(ierr);
  *dmSplit = sdm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IsExteriorOrGhostFace"
static PetscErrorCode IsExteriorOrGhostFace(DM dm,PetscInt face,PetscBool *isghost)
{
  PetscErrorCode ierr;
  PetscInt       ghost,boundary;

  PetscFunctionBeginUser;
  *isghost = PETSC_FALSE;
  ierr = DMPlexGetLabelValue(dm, "ghost", face, &ghost);CHKERRQ(ierr);
  ierr = DMPlexGetLabelValue(dm, "Face Sets", face, &boundary);CHKERRQ(ierr);
  if (ghost >= 0 || boundary >= 0) *isghost = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ConstructGeometry"
/* Set up face data and cell data */
PetscErrorCode ConstructGeometry(DM dm, Vec *facegeom, Vec *cellgeom, User user)
{
  DM             dmFace, dmCell;
  DMLabel        ghostLabel;
  PetscSection   sectionFace, sectionCell;
  PetscSection   coordSection;
  Vec            coordinates;
  PetscReal      minradius;
  PetscScalar    *fgeom, *cgeom;
  PetscInt       dim, cStart, cEnd, c, fStart, fEnd, f;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  if (dim != DIM) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"No support for dim %D != DIM %D",dim,DIM);
  ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);

  /* Make cell centroids and volumes */
  ierr = DMClone(dm, &dmCell);CHKERRQ(ierr);
  ierr = DMPlexSetCoordinateSection(dmCell, coordSection);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dmCell, coordinates);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm), &sectionCell);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sectionCell, cStart, cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    ierr = PetscSectionSetDof(sectionCell, c, sizeof(CellGeom)/sizeof(PetscScalar));CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(sectionCell);CHKERRQ(ierr);
  ierr = DMSetDefaultSection(dmCell, sectionCell);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sectionCell);CHKERRQ(ierr); /* relinquish our reference */

  ierr = DMCreateLocalVector(dmCell, cellgeom);CHKERRQ(ierr);
  ierr = VecGetArray(*cellgeom, &cgeom);CHKERRQ(ierr);
  for (c = cStart; c < user->cEndInterior; ++c) {
    CellGeom *cg;

    ierr = DMPlexPointLocalRef(dmCell, c, cgeom, &cg);CHKERRQ(ierr);
    ierr = PetscMemzero(cg,sizeof(*cg));CHKERRQ(ierr);
    ierr = DMPlexComputeCellGeometryFVM(dmCell, c, &cg->volume, cg->centroid, NULL);CHKERRQ(ierr);
  }
  /* Compute face normals and minimum cell radius */
  ierr = DMClone(dm, &dmFace);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm), &sectionFace);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sectionFace, fStart, fEnd);CHKERRQ(ierr);
  for (f = fStart; f < fEnd; ++f) {
    ierr = PetscSectionSetDof(sectionFace, f, sizeof(FaceGeom)/sizeof(PetscScalar));CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(sectionFace);CHKERRQ(ierr);
  ierr = DMSetDefaultSection(dmFace, sectionFace);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sectionFace);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(dmFace, facegeom);CHKERRQ(ierr);
  ierr = VecGetArray(*facegeom, &fgeom);CHKERRQ(ierr);
  ierr = DMPlexGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);
  minradius = PETSC_MAX_REAL;
  for (f = fStart; f < fEnd; ++f) {
    FaceGeom *fg;
    PetscInt  ghost;

    ierr = DMLabelGetValue(ghostLabel, f, &ghost);CHKERRQ(ierr);
    if (ghost >= 0) continue;
    ierr = DMPlexPointLocalRef(dmFace, f, fgeom, &fg);CHKERRQ(ierr);
    ierr = DMPlexComputeCellGeometryFVM(dm, f, NULL, fg->centroid, fg->normal);CHKERRQ(ierr);
    /* Flip face orientation if necessary to match ordering in support, and Update minimum radius */
    {
      CellGeom       *cL, *cR;
      const PetscInt *cells;
      PetscReal      *lcentroid, *rcentroid;
      PetscScalar     v[3];
      PetscInt        d;

      ierr = DMPlexGetSupport(dm, f, &cells);CHKERRQ(ierr);
      ierr = DMPlexPointLocalRead(dmCell, cells[0], cgeom, &cL);CHKERRQ(ierr);
      ierr = DMPlexPointLocalRead(dmCell, cells[1], cgeom, &cR);CHKERRQ(ierr);
      lcentroid = cells[0] >= user->cEndInterior ? fg->centroid : cL->centroid;
      rcentroid = cells[1] >= user->cEndInterior ? fg->centroid : cR->centroid;
      WaxpyD(dim, -1, lcentroid, rcentroid, v);
      if (DotD(dim, fg->normal, v) < 0) {
        for (d = 0; d < dim; ++d) fg->normal[d] = -fg->normal[d];
      }
      if (DotD(dim, fg->normal, v) <= 0) {
#if DIM == 2
        SETERRQ5(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Direction for face %d could not be fixed, normal (%g,%g) v (%g,%g)", f, fg->normal[0], fg->normal[1], v[0], v[1]);
#elif DIM == 3
        SETERRQ7(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Direction for face %d could not be fixed, normal (%g,%g,%g) v (%g,%g,%g)", f, fg->normal[0], fg->normal[1], fg->normal[2], v[0], v[1], v[2]);
#else
#  error DIM not supported
#endif
      }
      if (cells[0] < user->cEndInterior) {
        WaxpyD(dim, -1, fg->centroid, cL->centroid, v);
        minradius = PetscMin(minradius, NormD(dim, v));
      }
      if (cells[1] < user->cEndInterior) {
        WaxpyD(dim, -1, fg->centroid, cR->centroid, v);
        minradius = PetscMin(minradius, NormD(dim, v));
      }
    }
  }
  /* Compute centroids of ghost cells */
  for (c = user->cEndInterior; c < cEnd; ++c) {
    FaceGeom       *fg;
    const PetscInt *cone,    *support;
    PetscInt        coneSize, supportSize, s;

    ierr = DMPlexGetConeSize(dmCell, c, &coneSize);CHKERRQ(ierr);
    if (coneSize != 1) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Ghost cell %d has cone size %d != 1", c, coneSize);
    ierr = DMPlexGetCone(dmCell, c, &cone);CHKERRQ(ierr);
    ierr = DMPlexGetSupportSize(dmCell, cone[0], &supportSize);CHKERRQ(ierr);
    if (supportSize != 2) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %d has support size %d != 1", cone[0], supportSize);
    ierr = DMPlexGetSupport(dmCell, cone[0], &support);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRef(dmFace, cone[0], fgeom, &fg);CHKERRQ(ierr);
    for (s = 0; s < 2; ++s) {
      /* Reflect ghost centroid across plane of face */
      if (support[s] == c) {
        const CellGeom *ci;
        CellGeom       *cg;
        PetscScalar     c2f[3], a;

        ierr = DMPlexPointLocalRead(dmCell, support[(s+1)%2], cgeom, &ci);CHKERRQ(ierr);
        WaxpyD(dim, -1, ci->centroid, fg->centroid, c2f); /* cell to face centroid */
        a    = DotD(dim, c2f, fg->normal)/DotD(dim, fg->normal, fg->normal);
        ierr = DMPlexPointLocalRef(dmCell, support[s], cgeom, &cg);CHKERRQ(ierr);
        WaxpyD(dim, 2*a, fg->normal, ci->centroid, cg->centroid);
        cg->volume = ci->volume;
      }
    }
  }
  ierr = VecRestoreArray(*facegeom, &fgeom);CHKERRQ(ierr);
  ierr = VecRestoreArray(*cellgeom, &cgeom);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&minradius, &user->minradius, 1, MPIU_SCALAR, MPI_MIN, PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
  ierr = DMDestroy(&dmCell);CHKERRQ(ierr);
  ierr = DMDestroy(&dmFace);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreatePartitionVec"
PetscErrorCode CreatePartitionVec(DM dm, DM *dmCell, Vec *partition)
{
  PetscSF        sfPoint;
  PetscSection   coordSection;
  Vec            coordinates;
  PetscSection   sectionCell;
  PetscScalar    *part;
  PetscInt       cStart, cEnd, c;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMClone(dm, dmCell);CHKERRQ(ierr);
  ierr = DMGetPointSF(dm, &sfPoint);CHKERRQ(ierr);
  ierr = DMSetPointSF(*dmCell, sfPoint);CHKERRQ(ierr);
  ierr = DMPlexSetCoordinateSection(*dmCell, coordSection);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(*dmCell, coordinates);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm), &sectionCell);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(*dmCell, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sectionCell, cStart, cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    ierr = PetscSectionSetDof(sectionCell, c, 1);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(sectionCell);CHKERRQ(ierr);
  ierr = DMSetDefaultSection(*dmCell, sectionCell);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sectionCell);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(*dmCell, partition);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)*partition, "partition");CHKERRQ(ierr);
  ierr = VecGetArray(*partition, &part);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *p;

    ierr = DMPlexPointLocalRef(*dmCell, c, part, &p);CHKERRQ(ierr);
    p[0] = rank;
  }
  ierr = VecRestoreArray(*partition, &part);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMassMatrix"
PetscErrorCode CreateMassMatrix(DM dm, Vec *massMatrix, User user)
{
  DM                dmMass, dmFace, dmCell, dmCoord;
  PetscSection      coordSection;
  Vec               coordinates;
  PetscSection      sectionMass;
  PetscScalar       *m;
  const PetscScalar *facegeom, *cellgeom, *coords;
  PetscInt          vStart, vEnd, v;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMClone(dm, &dmMass);CHKERRQ(ierr);
  ierr = DMPlexSetCoordinateSection(dmMass, coordSection);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dmMass, coordinates);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm), &sectionMass);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sectionMass, vStart, vEnd);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {
    PetscInt numFaces;

    ierr = DMPlexGetSupportSize(dmMass, v, &numFaces);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(sectionMass, v, numFaces*numFaces);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(sectionMass);CHKERRQ(ierr);
  ierr = DMSetDefaultSection(dmMass, sectionMass);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&sectionMass);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dmMass, massMatrix);CHKERRQ(ierr);
  ierr = VecGetArray(*massMatrix, &m);CHKERRQ(ierr);
  ierr = VecGetDM(user->facegeom, &dmFace);CHKERRQ(ierr);
  ierr = VecGetArrayRead(user->facegeom, &facegeom);CHKERRQ(ierr);
  ierr = VecGetDM(user->cellgeom, &dmCell);CHKERRQ(ierr);
  ierr = VecGetArrayRead(user->cellgeom, &cellgeom);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm, &dmCoord);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coordinates, &coords);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {
    const PetscInt    *faces;
    const FaceGeom    *fgA, *fgB, *cg;
    const PetscScalar *vertex;
    PetscInt          numFaces, sides[2], f, g;

    ierr = DMPlexPointLocalRead(dmCoord, v, coords, &vertex);CHKERRQ(ierr);
    ierr = DMPlexGetSupportSize(dmMass, v, &numFaces);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dmMass, v, &faces);CHKERRQ(ierr);
    for (f = 0; f < numFaces; ++f) {
      sides[0] = faces[f];
      ierr = DMPlexPointLocalRead(dmFace, faces[f], facegeom, &fgA);CHKERRQ(ierr);
      for (g = 0; g < numFaces; ++g) {
        const PetscInt *cells = NULL;;
        PetscReal      area   = 0.0;
        PetscInt       numCells;

        sides[1] = faces[g];
        ierr = DMPlexPointLocalRead(dmFace, faces[g], facegeom, &fgB);CHKERRQ(ierr);
        ierr = DMPlexGetJoin(dmMass, 2, sides, &numCells, &cells);CHKERRQ(ierr);
        if (numCells != 1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Invalid join for faces");
        ierr = DMPlexPointLocalRead(dmCell, cells[0], cellgeom, &cg);CHKERRQ(ierr);
        area += PetscAbsScalar((vertex[0] - cg->centroid[0])*(fgA->centroid[1] - cg->centroid[1]) - (vertex[1] - cg->centroid[1])*(fgA->centroid[0] - cg->centroid[0]));
        area += PetscAbsScalar((vertex[0] - cg->centroid[0])*(fgB->centroid[1] - cg->centroid[1]) - (vertex[1] - cg->centroid[1])*(fgB->centroid[0] - cg->centroid[0]));
        m[f*numFaces+g] = Dot2(fgA->normal, fgB->normal)*area*0.5;
        ierr = DMPlexRestoreJoin(dmMass, 2, sides, &numCells, &cells);CHKERRQ(ierr);
      }
    }
  }
  ierr = VecRestoreArrayRead(user->facegeom, &facegeom);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(user->facegeom, &facegeom);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(coordinates, &coords);CHKERRQ(ierr);
  ierr = VecRestoreArray(*massMatrix, &m);CHKERRQ(ierr);
  ierr = DMDestroy(&dmMass);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetUpLocalSpace"
PetscErrorCode SetUpLocalSpace(DM dm, User user)
{
  PetscSection   stateSection;
  Physics        phys;
  PetscInt       dof = user->model->physics->dof, *cind, d, stateSize, cStart, cEnd, c, i;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm), &stateSection);CHKERRQ(ierr);
  phys = user->model->physics;
  ierr = PetscSectionSetNumFields(stateSection,phys->nfields);CHKERRQ(ierr);
  for (i=0; i<phys->nfields; i++) {
    ierr = PetscSectionSetFieldName(stateSection,i,phys->field_desc[i].name);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldComponents(stateSection,i,phys->field_desc[i].dof);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetChart(stateSection, cStart, cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    for (i=0; i<phys->nfields; i++) {
      ierr = PetscSectionSetFieldDof(stateSection,c,i,phys->field_desc[i].dof);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetDof(stateSection, c, dof);CHKERRQ(ierr);
  }
  for (c = user->cEndInterior; c < cEnd; ++c) {
    ierr = PetscSectionSetConstraintDof(stateSection, c, dof);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(stateSection);CHKERRQ(ierr);
  ierr = PetscMalloc(dof * sizeof(PetscInt), &cind);CHKERRQ(ierr);
  for (d = 0; d < dof; ++d) cind[d] = d;
#if 0
  for (c = cStart; c < cEnd; ++c) {
    PetscInt val;

    ierr = DMPlexGetLabelValue(dm, "vtk", c, &val);CHKERRQ(ierr);
    if (val < 0) {ierr = PetscSectionSetConstraintIndices(stateSection, c, cind);CHKERRQ(ierr);}
  }
#endif
  for (c = user->cEndInterior; c < cEnd; ++c) {
    ierr = PetscSectionSetConstraintIndices(stateSection, c, cind);CHKERRQ(ierr);
  }
  ierr = PetscFree(cind);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(stateSection, &stateSize);CHKERRQ(ierr);
  ierr = DMSetDefaultSection(dm,stateSection);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&stateSection);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetUpBoundaries"
PetscErrorCode SetUpBoundaries(DM dm, User user)
{
  Model          mod = user->model;
  PetscErrorCode ierr;
  BoundaryLink   b;

  PetscFunctionBeginUser;
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)dm),NULL,"Boundary condition options","");CHKERRQ(ierr);
  for (b = mod->boundary; b; b=b->next) {
    char      optname[512];
    PetscInt  ids[512],len = 512;
    PetscBool flg;
    ierr = PetscSNPrintf(optname,sizeof optname,"-bc_%s",b->name);CHKERRQ(ierr);
    ierr = PetscMemzero(ids,sizeof(ids));CHKERRQ(ierr);
    ierr = PetscOptionsIntArray(optname,"List of boundary IDs","",ids,&len,&flg);CHKERRQ(ierr);
    if (flg) {
      /* TODO: check all IDs to make sure they exist in the mesh */
      ierr      = PetscFree(b->ids);CHKERRQ(ierr);
      b->numids = len;
      ierr      = PetscMalloc(len*sizeof(PetscInt),&b->ids);CHKERRQ(ierr);
      ierr      = PetscMemcpy(b->ids,ids,len*sizeof(PetscInt));CHKERRQ(ierr);
    }
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ModelBoundaryRegister"
/* The ids are just defaults, can be overridden at command line. I expect to set defaults based on names in the future. */
static PetscErrorCode ModelBoundaryRegister(Model mod,const char *name,BoundaryFunction bcFunc,void *ctx,PetscInt numids,const PetscInt *ids)
{
  PetscErrorCode ierr;
  BoundaryLink   link;

  PetscFunctionBeginUser;
  ierr          = PetscNew(struct _n_BoundaryLink,&link);CHKERRQ(ierr);
  ierr          = PetscStrallocpy(name,&link->name);CHKERRQ(ierr);
  link->numids  = numids;
  ierr          = PetscMalloc(numids*sizeof(PetscInt),&link->ids);CHKERRQ(ierr);
  ierr          = PetscMemcpy(link->ids,ids,numids*sizeof(PetscInt));CHKERRQ(ierr);
  link->func    = bcFunc;
  link->ctx     = ctx;
  link->next    = mod->boundary;
  mod->boundary = link;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BoundaryLinkDestroy"
static PetscErrorCode BoundaryLinkDestroy(BoundaryLink *link)
{
  PetscErrorCode ierr;
  BoundaryLink   l,next;

  PetscFunctionBeginUser;
  if (!link) PetscFunctionReturn(0);
  l     = *link;
  *link = NULL;
  for (; l; l=next) {
    next = l->next;
    ierr = PetscFree(l->ids);CHKERRQ(ierr);
    ierr = PetscFree(l->name);CHKERRQ(ierr);
    ierr = PetscFree(l);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ModelBoundaryFind"
static PetscErrorCode ModelBoundaryFind(Model mod,PetscInt id,BoundaryFunction *bcFunc,void **ctx)
{
  BoundaryLink link;
  PetscInt     i;

  PetscFunctionBeginUser;
  *bcFunc = NULL;
  for (link=mod->boundary; link; link=link->next) {
    for (i=0; i<link->numids; i++) {
      if (link->ids[i] == id) {
        *bcFunc = link->func;
        *ctx    = link->ctx;
        PetscFunctionReturn(0);
      }
    }
  }
  SETERRQ1(mod->comm,PETSC_ERR_USER,"Boundary ID %D not associated with any registered boundary condition",id);
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "ModelSolutionSetDefault"
/* Behavior will be different for multi-physics or when using non-default boundary conditions */
static PetscErrorCode ModelSolutionSetDefault(Model mod,SolutionFunction func,void *ctx)
{
  PetscFunctionBeginUser;
  mod->solution    = func;
  mod->solutionctx = ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ModelFunctionalRegister"
static PetscErrorCode ModelFunctionalRegister(Model mod,const char *name,PetscInt *offset,FunctionalFunction func,void *ctx)
{
  PetscErrorCode ierr;
  FunctionalLink link,*ptr;
  PetscInt       lastoffset = -1;

  PetscFunctionBeginUser;
  for (ptr=&mod->functionalRegistry; *ptr; ptr = &(*ptr)->next) lastoffset = (*ptr)->offset;
  ierr         = PetscNew(struct _n_FunctionalLink,&link);CHKERRQ(ierr);
  ierr         = PetscStrallocpy(name,&link->name);CHKERRQ(ierr);
  link->offset = lastoffset + 1;
  link->func   = func;
  link->ctx    = ctx;
  link->next   = NULL;
  *ptr         = link;
  *offset      = link->offset;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ModelFunctionalSetFromOptions"
static PetscErrorCode ModelFunctionalSetFromOptions(Model mod)
{
  PetscErrorCode ierr;
  PetscInt       i,j;
  FunctionalLink link;
  char           *names[256];

  PetscFunctionBeginUser;
  mod->numMonitored = ALEN(names);
  ierr = PetscOptionsStringArray("-monitor","list of functionals to monitor","",names,&mod->numMonitored,NULL);CHKERRQ(ierr);
  /* Create list of functionals that will be computed somehow */
  ierr = PetscMalloc(mod->numMonitored*sizeof(FunctionalLink),&mod->functionalMonitored);CHKERRQ(ierr);
  /* Create index of calls that we will have to make to compute these functionals (over-allocation in general). */
  ierr = PetscMalloc(mod->numMonitored*sizeof(FunctionalLink),&mod->functionalCall);CHKERRQ(ierr);
  mod->numCall = 0;
  for (i=0; i<mod->numMonitored; i++) {
    for (link=mod->functionalRegistry; link; link=link->next) {
      PetscBool match;
      ierr = PetscStrcasecmp(names[i],link->name,&match);CHKERRQ(ierr);
      if (match) break;
    }
    if (!link) SETERRQ1(mod->comm,PETSC_ERR_USER,"No known functional '%s'",names[i]);
    mod->functionalMonitored[i] = link;
    for (j=0; j<i; j++) {
      if (mod->functionalCall[j]->func == link->func && mod->functionalCall[j]->ctx == link->ctx) goto next_name;
    }
    mod->functionalCall[mod->numCall++] = link; /* Just points to the first link using the result. There may be more results. */
next_name:
    ierr = PetscFree(names[i]);CHKERRQ(ierr);
  }

  /* Find out the maximum index of any functional computed by a function we will be calling (even if we are not using it) */
  mod->maxComputed = -1;
  for (link=mod->functionalRegistry; link; link=link->next) {
    for (i=0; i<mod->numCall; i++) {
      FunctionalLink call = mod->functionalCall[i];
      if (link->func == call->func && link->ctx == call->ctx) {
        mod->maxComputed = PetscMax(mod->maxComputed,link->offset);
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FunctionalLinkDestroy"
static PetscErrorCode FunctionalLinkDestroy(FunctionalLink *link)
{
  PetscErrorCode ierr;
  FunctionalLink l,next;

  PetscFunctionBeginUser;
  if (!link) PetscFunctionReturn(0);
  l     = *link;
  *link = NULL;
  for (; l; l=next) {
    next = l->next;
    ierr = PetscFree(l->name);CHKERRQ(ierr);
    ierr = PetscFree(l);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetInitialCondition"
PetscErrorCode SetInitialCondition(DM dm, Vec X, User user)
{
  DM                dmCell;
  Model             mod = user->model;
  const PetscScalar *cellgeom;
  PetscScalar       *x;
  PetscInt          cStart, cEnd, cEndInterior = user->cEndInterior, c;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetDM(user->cellgeom, &dmCell);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = VecGetArrayRead(user->cellgeom, &cellgeom);CHKERRQ(ierr);
  ierr = VecGetArray(X, &x);CHKERRQ(ierr);
  for (c = cStart; c < cEndInterior; ++c) {
    const CellGeom *cg;
    PetscScalar    *xc;

    ierr = DMPlexPointLocalRead(dmCell,c,cellgeom,&cg);CHKERRQ(ierr);
    ierr = DMPlexPointGlobalRef(dm,c,x,&xc);CHKERRQ(ierr);
    if (xc) {ierr = (*mod->solution)(mod,0.0,cg->centroid,xc,mod->solutionctx);CHKERRQ(ierr);}
  }
  ierr = VecRestoreArrayRead(user->cellgeom, &cellgeom);CHKERRQ(ierr);
  ierr = VecRestoreArray(X, &x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ApplyBC"
static PetscErrorCode ApplyBC(DM dm, PetscReal time, Vec locX, User user)
{
  Model             mod   = user->model;
  const char        *name = "Face Sets";
  DM                dmFace;
  IS                idIS;
  const PetscInt    *ids;
  PetscScalar       *x;
  const PetscScalar *facegeom;
  PetscInt          numFS, fs;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetDM(user->facegeom,&dmFace);CHKERRQ(ierr);
  ierr = DMPlexGetLabelIdIS(dm, name, &idIS);CHKERRQ(ierr);
  if (!idIS) PetscFunctionReturn(0);
  ierr = ISGetLocalSize(idIS, &numFS);CHKERRQ(ierr);
  ierr = ISGetIndices(idIS, &ids);CHKERRQ(ierr);
  ierr = VecGetArrayRead(user->facegeom, &facegeom);CHKERRQ(ierr);
  ierr = VecGetArray(locX, &x);CHKERRQ(ierr);
  for (fs = 0; fs < numFS; ++fs) {
    BoundaryFunction bcFunc;
    void             *bcCtx;
    IS               faceIS;
    const PetscInt   *faces;
    PetscInt         numFaces, f;

    ierr = ModelBoundaryFind(mod,ids[fs],&bcFunc,&bcCtx);CHKERRQ(ierr);
    ierr = DMPlexGetStratumIS(dm, name, ids[fs], &faceIS);CHKERRQ(ierr);
    ierr = ISGetLocalSize(faceIS, &numFaces);CHKERRQ(ierr);
    ierr = ISGetIndices(faceIS, &faces);CHKERRQ(ierr);
    for (f = 0; f < numFaces; ++f) {
      const PetscInt    face = faces[f], *cells;
      const PetscScalar *xI;
      PetscScalar       *xG;
      const FaceGeom    *fg;

      ierr = DMPlexPointLocalRead(dmFace, face, facegeom, &fg);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, face, &cells);CHKERRQ(ierr);
      ierr = DMPlexPointLocalRead(dm, cells[0], x, &xI);CHKERRQ(ierr);
      ierr = DMPlexPointLocalRef(dm, cells[1], x, &xG);CHKERRQ(ierr);
      ierr = (*bcFunc)(mod, time, fg->centroid, fg->normal, xI, xG, bcCtx);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(faceIS, &faces);CHKERRQ(ierr);
    ierr = ISDestroy(&faceIS);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(locX, &x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(user->facegeom,&facegeom);CHKERRQ(ierr);
  ierr = ISRestoreIndices(idIS, &ids);CHKERRQ(ierr);
  ierr = ISDestroy(&idIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RHSFunctionLocal_Upwind"
static PetscErrorCode RHSFunctionLocal_Upwind(DM dm,DM dmFace,DM dmCell,PetscReal time,Vec locX,Vec F,User user)
{
  Physics           phys = user->model->physics;
  DMLabel           ghostLabel;
  PetscErrorCode    ierr;
  const PetscScalar *facegeom, *cellgeom, *x;
  PetscScalar       *f;
  PetscInt          fStart, fEnd, face;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(user->facegeom,&facegeom);CHKERRQ(ierr);
  ierr = VecGetArrayRead(user->cellgeom,&cellgeom);CHKERRQ(ierr);
  ierr = VecGetArrayRead(locX,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  ierr = DMPlexGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  for (face = fStart; face < fEnd; ++face) {
    const PetscInt    *cells;
    PetscInt          i,ghost;
    PetscScalar       *flux = user->work.flux,*fL,*fR;
    const FaceGeom    *fg;
    const CellGeom    *cgL,*cgR;
    const PetscScalar *xL,*xR;

    ierr = DMLabelGetValue(ghostLabel, face, &ghost);CHKERRQ(ierr);
    if (ghost >= 0) continue;
    ierr = DMPlexGetSupport(dm, face, &cells);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dmFace,face,facegeom,&fg);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dmCell,cells[0],cellgeom,&cgL);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dmCell,cells[1],cellgeom,&cgR);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dm,cells[0],x,&xL);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dm,cells[1],x,&xR);CHKERRQ(ierr);
    ierr = DMPlexPointGlobalRef(dm,cells[0],f,&fL);CHKERRQ(ierr);
    ierr = DMPlexPointGlobalRef(dm,cells[1],f,&fR);CHKERRQ(ierr);
    ierr = (*phys->riemann)(phys, fg->centroid, fg->normal, xL, xR, flux);CHKERRQ(ierr);
    for (i=0; i<phys->dof; i++) {
      if (fL) fL[i] -= flux[i] / cgL->volume;
      if (fR) fR[i] += flux[i] / cgR->volume;
    }
  }
  ierr = VecRestoreArrayRead(user->facegeom,&facegeom);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(user->cellgeom,&cellgeom);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(locX,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RHSFunction"
static PetscErrorCode RHSFunction(TS ts,PetscReal time,Vec X,Vec F,void *ctx)
{
  User           user = (User)ctx;
  DM             dm, dmFace, dmCell;
  PetscSection   section;
  Vec            locX;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = VecGetDM(user->facegeom,&dmFace);CHKERRQ(ierr);
  ierr = VecGetDM(user->cellgeom,&dmCell);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&locX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, X, INSERT_VALUES, locX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, X, INSERT_VALUES, locX);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);

  ierr = ApplyBC(dm, time, locX, user);CHKERRQ(ierr);

  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  ierr = (*user->RHSFunctionLocal)(dm,dmFace,dmCell,time,locX,F,user);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&locX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "OutputVTK"
static PetscErrorCode OutputVTK(DM dm, const char *filename, PetscViewer *viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscViewerCreate(PetscObjectComm((PetscObject)dm), viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(*viewer, PETSCVIEWERVTK);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(*viewer, filename);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MonitorVTK"
static PetscErrorCode MonitorVTK(TS ts,PetscInt stepnum,PetscReal time,Vec X,void *ctx)
{
  User           user = (User)ctx;
  DM             dm;
  PetscViewer    viewer;
  char           filename[PETSC_MAX_PATH_LEN],*ftable = NULL;
  PetscReal      xnorm;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectSetName((PetscObject) X, "solution");CHKERRQ(ierr);
  ierr = VecGetDM(X,&dm);CHKERRQ(ierr);
  ierr = VecNorm(X,NORM_INFINITY,&xnorm);CHKERRQ(ierr);
  if (stepnum >= 0) {           /* No summary for final time */
    Model             mod = user->model;
    PetscInt          c,cStart,cEnd,fcount,i;
    size_t            ftableused,ftablealloc;
    const PetscScalar *cellgeom,*x;
    DM                dmCell;
    PetscReal         *fmin,*fmax,*fintegral,*ftmp;
    fcount = mod->maxComputed+1;
    ierr   = PetscMalloc4(fcount,PetscReal,&fmin,fcount,PetscReal,&fmax,fcount,PetscReal,&fintegral,fcount,PetscReal,&ftmp);CHKERRQ(ierr);
    for (i=0; i<fcount; i++) {
      fmin[i]      = PETSC_MAX_REAL;
      fmax[i]      = PETSC_MIN_REAL;
      fintegral[i] = 0;
    }
    ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
    ierr = VecGetDM(user->cellgeom,&dmCell);CHKERRQ(ierr);
    ierr = VecGetArrayRead(user->cellgeom,&cellgeom);CHKERRQ(ierr);
    ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
    for (c=cStart; c<user->cEndInterior; c++) {
      const CellGeom    *cg;
      const PetscScalar *cx;
      ierr = DMPlexPointLocalRead(dmCell,c,cellgeom,&cg);CHKERRQ(ierr);
      ierr = DMPlexPointGlobalRead(dm,c,x,&cx);CHKERRQ(ierr);
      if (!cx) continue;        /* not a global cell */
      for (i=0; i<mod->numCall; i++) {
        FunctionalLink flink = mod->functionalCall[i];
        ierr = (*flink->func)(mod,time,cg->centroid,cx,ftmp,flink->ctx);CHKERRQ(ierr);
      }
      for (i=0; i<fcount; i++) {
        fmin[i]       = PetscMin(fmin[i],ftmp[i]);
        fmax[i]       = PetscMax(fmax[i],ftmp[i]);
        fintegral[i] += cg->volume * ftmp[i];
      }
    }
    ierr = VecRestoreArrayRead(user->cellgeom,&cellgeom);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
    ierr = MPI_Allreduce(MPI_IN_PLACE,fmin,fcount,MPIU_REAL,MPI_MIN,PetscObjectComm((PetscObject)ts));CHKERRQ(ierr);
    ierr = MPI_Allreduce(MPI_IN_PLACE,fmax,fcount,MPIU_REAL,MPI_MAX,PetscObjectComm((PetscObject)ts));CHKERRQ(ierr);
    ierr = MPI_Allreduce(MPI_IN_PLACE,fintegral,fcount,MPIU_REAL,MPI_SUM,PetscObjectComm((PetscObject)ts));CHKERRQ(ierr);

    ftablealloc = fcount * 100;
    ftableused  = 0;
    ierr        = PetscMalloc(ftablealloc,&ftable);CHKERRQ(ierr);
    for (i=0; i<mod->numMonitored; i++) {
      size_t         countused;
      char           buffer[256],*p;
      FunctionalLink flink = mod->functionalMonitored[i];
      PetscInt       id    = flink->offset;
      if (i % 3) {
        ierr = PetscMemcpy(buffer,"  ",2);CHKERRQ(ierr);
        p    = buffer + 2;
      } else if (i) {
        char newline[] = "\n";
        ierr = PetscMemcpy(buffer,newline,sizeof newline-1);CHKERRQ(ierr);
        p    = buffer + sizeof newline - 1;
      } else {
        p = buffer;
      }
      ierr = PetscSNPrintfCount(p,sizeof buffer-(p-buffer),"%12s [%10.7G,%10.7G] int %10.7G",&countused,flink->name,fmin[id],fmax[id],fintegral[id]);CHKERRQ(ierr);
      countused += p - buffer;
      if (countused > ftablealloc-ftableused-1) { /* reallocate */
        char *ftablenew;
        ftablealloc = 2*ftablealloc + countused;
        ierr = PetscMalloc(ftablealloc,&ftablenew);CHKERRQ(ierr);
        ierr = PetscMemcpy(ftablenew,ftable,ftableused);CHKERRQ(ierr);
        ierr = PetscFree(ftable);CHKERRQ(ierr);
        ftable = ftablenew;
      }
      ierr = PetscMemcpy(ftable+ftableused,buffer,countused);CHKERRQ(ierr);
      ftableused += countused;
      ftable[ftableused] = 0;
    }
    ierr = PetscFree4(fmin,fmax,fintegral,ftmp);CHKERRQ(ierr);

    ierr = PetscPrintf(PetscObjectComm((PetscObject)ts),"% 3D  time %8.4G  |x| %8.4G  %s\n",stepnum,time,xnorm,ftable ? ftable : "");CHKERRQ(ierr);
    ierr = PetscFree(ftable);CHKERRQ(ierr);
  }
  if (user->vtkInterval < 1) PetscFunctionReturn(0);
  if ((stepnum == -1) ^ (stepnum % user->vtkInterval == 0)) {
    if (stepnum == -1) {        /* Final time is not multiple of normal time interval, write it anyway */
      ierr = TSGetTimeStepNumber(ts,&stepnum);CHKERRQ(ierr);
    }
    ierr = PetscSNPrintf(filename,sizeof filename,"ex11-%03D.vtu",stepnum);CHKERRQ(ierr);
    ierr = OutputVTK(dm,filename,&viewer);CHKERRQ(ierr);
    ierr = VecView(X,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  MPI_Comm          comm;
  User              user;
  Model             mod;
  Physics           phys;
  DM                dm, dmDist;
  PetscReal         ftime,cfl,dt;
  PetscInt          dim, overlap, nsteps, i, nGrids = 3;
  int               CPU_word_size = 0, IO_word_size = 0, exoid;
  float             version;
  TS                ts;
  TSConvergedReason reason;
  Vec               X;
  PetscViewer       viewer;
  PetscMPIInt       rank;
  char              filename[PETSC_MAX_PATH_LEN] = "sevenside.exo";
  PetscBool         vtkCellGeom, splitFaces;
  PetscErrorCode    ierr;

  ierr = PetscInitialize(&argc, &argv, (char*) 0, help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);

  ierr = PetscNew(struct _n_User,&user);CHKERRQ(ierr);
  ierr = PetscNew(struct _n_Model,&user->model);CHKERRQ(ierr);
  ierr = PetscNew(struct _n_Physics,&user->model->physics);CHKERRQ(ierr);
  mod  = user->model;
  phys = mod->physics;
  mod->comm = comm;

  /* Register physical models to be available on the command line */
  ierr = PetscFunctionListAdd(&PhysicsList,"diff"          ,PhysicsCreate_Diff);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(comm,NULL,"Unstructured Finite Volume Options","");CHKERRQ(ierr);
  {
    char           physname[256] = "diff";
    PetscErrorCode (*physcreate)(Model,Physics);
    ierr              = PetscOptionsString("-f","Exodus.II filename to read","",filename,filename,sizeof(filename),NULL);CHKERRQ(ierr);
    user->vtkInterval = 1;
    ierr = PetscOptionsInt("-ufv_vtk_interval","VTK output interval (0 to disable)","",user->vtkInterval,&user->vtkInterval,NULL);CHKERRQ(ierr);
    overlap = 1;
    ierr = PetscOptionsInt("-ufv_mesh_overlap","Number of cells to overlap partitions","",overlap,&overlap,NULL);CHKERRQ(ierr);
    vtkCellGeom = PETSC_FALSE;

    ierr = PetscOptionsBool("-ufv_vtk_cellgeom","Write cell geometry (for debugging)","",vtkCellGeom,&vtkCellGeom,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsList("-physics","Physics module to solve","",PhysicsList,physname,physname,sizeof physname,NULL);CHKERRQ(ierr);
    ierr = PetscFunctionListFind(PhysicsList,physname,&physcreate);CHKERRQ(ierr);
    ierr = PetscMemzero(phys,sizeof(struct _n_Physics));CHKERRQ(ierr);
    ierr = (*physcreate)(mod,phys);CHKERRQ(ierr);
    /* Count number of fields and dofs */
    for (phys->nfields=0,phys->dof=0; phys->field_desc[phys->nfields].name; phys->nfields++) phys->dof += phys->field_desc[phys->nfields].dof;

    if (phys->dof <= 0) SETERRQ1(comm,PETSC_ERR_ARG_WRONGSTATE,"Physics '%s' did not set dof",physname);
    ierr = PetscMalloc3(phys->dof,PetscScalar,&user->work.flux,phys->dof,PetscScalar,&user->work.state0,phys->dof,PetscScalar,&user->work.state1);CHKERRQ(ierr);
    user->RHSFunctionLocal = RHSFunctionLocal_X4_X2;
    splitFaces = PETSC_FALSE; // ?????
    ierr = PetscOptionsBool("-ufv_split_faces","Split faces between cell sets","",splitFaces,&splitFaces,NULL);CHKERRQ(ierr);
    ierr = ModelFunctionalSetFromOptions(mod);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if (!rank) {
    //exoid = ex_open(filename, EX_READ, &CPU_word_size, &IO_word_size, &version);
    if (exoid <= 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ex_open(\"%s\",...) did not return a valid file ID",filename);
  } else exoid = -1;                 /* Not used */
  ierr = DMPlexCreateExodus(comm, exoid, PETSC_TRUE, &dm);CHKERRQ(ierr);
  //if (!rank) {ierr = ex_close(exoid);CHKERRQ(ierr);}
  /* Distribute mesh */
  ierr = DMPlexDistribute(dm, "chaco", overlap, &dmDist);CHKERRQ(ierr);
  if (dmDist) {
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
    dm   = dmDist;
  }
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  {
    DM gdm;

    ierr = DMPlexGetHeightStratum(dm, 0, NULL, &user->cEndInterior);CHKERRQ(ierr);
    ierr = DMPlexConstructGhostCells(dm, NULL, &user->numGhostCells, &gdm);CHKERRQ(ierr);
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
    dm   = gdm;
  }
  if (splitFaces) {ierr = ConstructCellBoundary(dm, user);CHKERRQ(ierr);}
  ierr = SplitFaces(&dm, "split faces", user);CHKERRQ(ierr);
  ierr = ConstructGeometry(dm, &user->facegeom, &user->cellgeom, user);CHKERRQ(ierr);
  if (0) {ierr = VecView(user->cellgeom, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexSetPreallocationCenterDimension(dm, 0);CHKERRQ(ierr);

  /* Set up DM with section describing local vector and configure local vector. */
  ierr = SetUpLocalSpace(dm, user);CHKERRQ(ierr);
  ierr = SetUpBoundaries(dm, user);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm, &X);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) X, "solution");CHKERRQ(ierr);
  ierr = SetInitialCondition(dm, X, user);CHKERRQ(ierr);
  if (vtkCellGeom) {
    DM  dmCell;
    Vec partition;

    ierr = OutputVTK(dm, "ex11-cellgeom.vtk", &viewer);CHKERRQ(ierr);
    ierr = VecView(user->cellgeom, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = CreatePartitionVec(dm, &dmCell, &partition);CHKERRQ(ierr);
    ierr = OutputVTK(dmCell, "ex11-partition.vtk", &viewer);CHKERRQ(ierr);
    ierr = VecView(partition, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = VecDestroy(&partition);CHKERRQ(ierr);
    ierr = DMDestroy(&dmCell);CHKERRQ(ierr);
  }

  ierr = TSCreate(comm, &ts);CHKERRQ(ierr);
  ierr = TSSetType(ts, TSSSP);CHKERRQ(ierr);
  ierr = TSSetDM(ts, dm);CHKERRQ(ierr);
  ierr = TSMonitorSet(ts,MonitorVTK,user,NULL);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,RHSFunction,user);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,1000,2.0);CHKERRQ(ierr);
  dt   = cfl * user->minradius / user->model->maxspeed;
  ierr = TSSetInitialTimeStep(ts,0.0,dt);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* convergance test loop */
  for (i=0;i<nGrids;++i) {
    ierr = TSSolve(ts,X);CHKERRQ(ierr);
    ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
    ierr = TSGetTimeStepNumber(ts,&nsteps);CHKERRQ(ierr);
    ierr = TSGetConvergedReason(ts,&reason);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%s at time %G after %D steps\n",TSConvergedReasons[reason],ftime,nsteps);CHKERRQ(ierr);
  }

  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = VecDestroy(&user->cellgeom);CHKERRQ(ierr);
  ierr = VecDestroy(&user->facegeom);CHKERRQ(ierr);
  ierr = DMDestroy(&user->dmGrad);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&PhysicsList);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&LimitList);CHKERRQ(ierr);
  ierr = BoundaryLinkDestroy(&user->model->boundary);CHKERRQ(ierr);
  ierr = FunctionalLinkDestroy(&user->model->functionalRegistry);CHKERRQ(ierr);
  ierr = PetscFree(user->model->functionalMonitored);CHKERRQ(ierr);
  ierr = PetscFree(user->model->functionalCall);CHKERRQ(ierr);
  ierr = PetscFree(user->model->physics->data);CHKERRQ(ierr);
  ierr = PetscFree(user->model->physics);CHKERRQ(ierr);
  ierr = PetscFree(user->model);CHKERRQ(ierr);
  ierr = PetscFree3(user->work.flux,user->work.state0,user->work.state1);CHKERRQ(ierr);
  ierr = PetscFree(user);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return(0);
}
