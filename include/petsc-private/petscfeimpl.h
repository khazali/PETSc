#if !defined(_PETSCFEIMPL_H)
#define _PETSCFEIMPL_H

#include <petscfe.h>
#include <petsc-private/petscimpl.h>

typedef struct _PetscSpaceOps *PetscSpaceOps;
struct _PetscSpaceOps {
  PetscErrorCode (*setfromoptions)(PetscSpace);
  PetscErrorCode (*setup)(PetscSpace);
  PetscErrorCode (*view)(PetscSpace,PetscViewer);
  PetscErrorCode (*destroy)(PetscSpace);

  PetscErrorCode (*getdimension)(PetscSpace,PetscInt*);
  PetscErrorCode (*evaluate)(PetscSpace,PetscInt,const PetscReal*,PetscReal*,PetscReal*,PetscReal*);
};

struct _p_PetscSpace {
  PETSCHEADER(struct _PetscSpaceOps);
  void    *data;  /* Implementation object */
  PetscInt order; /* The approximation order of the space */
  DM       dm;    /* Shell to use for temp allocation */
};

typedef struct {
  PetscInt   numVariables; /* The number of variables in the space, e.g. x and y */
  PetscBool  symmetric;    /* Use only symmetric polynomials */
  PetscBool  tensor;       /* Flag for tensor product */
  PetscInt  *degrees;      /* Degrees of single variable which we need to compute */
} PetscSpace_Poly;

typedef struct {
  PetscInt        numVariables; /* The spatial dimension */
  PetscQuadrature quad;         /* The points defining the space */
} PetscSpace_DG;

typedef struct _PetscDualSpaceOps *PetscDualSpaceOps;
struct _PetscDualSpaceOps {
  PetscErrorCode (*setfromoptions)(PetscDualSpace);
  PetscErrorCode (*setup)(PetscDualSpace);
  PetscErrorCode (*view)(PetscDualSpace,PetscViewer);
  PetscErrorCode (*destroy)(PetscDualSpace);

  PetscErrorCode (*duplicate)(PetscDualSpace,PetscDualSpace*);
  PetscErrorCode (*getdimension)(PetscDualSpace,PetscInt*);
  PetscErrorCode (*getnumdof)(PetscDualSpace,const PetscInt**);
};

struct _p_PetscDualSpace {
  PETSCHEADER(struct _PetscDualSpaceOps);
  void            *data;       /* Implementation object */
  DM               dm;         /* The integration region K */
  PetscInt         order;      /* The approximation order of the space */
  PetscQuadrature *functional; /* The basis of functionals for this space */
};

typedef struct {
  PetscInt *numDof;
  PetscBool simplex;
  PetscBool continuous;
} PetscDualSpace_Lag;

typedef struct _PetscFEOps *PetscFEOps;
struct _PetscFEOps {
  PetscErrorCode (*setfromoptions)(PetscFE);
  PetscErrorCode (*setup)(PetscFE);
  PetscErrorCode (*view)(PetscFE,PetscViewer);
  PetscErrorCode (*destroy)(PetscFE);
  PetscErrorCode (*getdimension)(PetscFE,PetscInt*);
  PetscErrorCode (*gettabulation)(PetscFE,PetscInt,const PetscReal*,PetscReal*,PetscReal*,PetscReal*);
  /* Element integration */
  PetscErrorCode (*integrate)(PetscFE, PetscInt, PetscInt, PetscFE[], PetscInt, PetscCellGeometry, const PetscScalar[],
                              PetscInt, PetscFE[], const PetscScalar[],
                              void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                              PetscReal[]);
  PetscErrorCode (*integrateresidual)(PetscFE, PetscInt, PetscInt, PetscFE[], PetscInt, PetscCellGeometry, const PetscScalar[],
                                      PetscInt, PetscFE[], const PetscScalar[],
                                      void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                      void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                      PetscScalar[]);
  PetscErrorCode (*integratebdresidual)(PetscFE, PetscInt, PetscInt, PetscFE[], PetscInt, PetscCellGeometry, const PetscScalar[],
                                        PetscInt, PetscFE[], const PetscScalar[],
                                        void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]),
                                        void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]),
                                        PetscScalar[]);
  PetscErrorCode (*integratejacobianaction)(PetscFE, PetscInt, PetscInt, PetscFE[], PetscInt, PetscCellGeometry, const PetscScalar[], const PetscScalar[],
                                            void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                            void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                            void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                            void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                            PetscScalar[]);
  PetscErrorCode (*integratejacobian)(PetscFE, PetscInt, PetscInt, PetscFE[], PetscInt, PetscInt, PetscCellGeometry, const PetscScalar[],
                                      PetscInt, PetscFE[], const PetscScalar[],
                                      void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                      void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                      void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                      void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                      PetscScalar[]);
  PetscErrorCode (*integratebdjacobian)(PetscFE, PetscInt, PetscInt, PetscFE[], PetscInt, PetscInt, PetscCellGeometry, const PetscScalar[],
                                        PetscInt, PetscFE[], const PetscScalar[],
                                        void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]),
                                        void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]),
                                        void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]),
                                        void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]),
                                        PetscScalar[]);
  PetscErrorCode (*integrateifunction)(PetscFE, PetscInt, PetscInt, PetscFE[], PetscInt, PetscCellGeometry, const PetscScalar[], const PetscScalar[],
                                       PetscInt, PetscFE[], const PetscScalar[],
                                       void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                       void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                      PetscScalar[]);
  PetscErrorCode (*integratebdifunction)(PetscFE, PetscInt, PetscInt, PetscFE[], PetscInt, PetscCellGeometry, const PetscScalar[], const PetscScalar[],
                                         PetscInt, PetscFE[], const PetscScalar[],
                                         void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]),
                                         void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]),
                                         PetscScalar[]);
};

struct _p_PetscFE {
  PETSCHEADER(struct _PetscFEOps);
  void           *data;          /* Implementation object */
  PetscSpace      basisSpace;    /* The basis space P */
  PetscDualSpace  dualSpace;     /* The dual space P' */
  PetscInt        numComponents; /* The number of field components */
  PetscQuadrature quadrature;    /* Suitable quadrature on K */
  PetscInt       *numDof;        /* The number of dof on mesh points of each depth */
  PetscReal      *invV;          /* Change of basis matrix, from prime to nodal basis set */
  PetscReal      *B, *D, *H;     /* Tabulation of basis and derivatives at quadrature points */
  PetscInt        blockSize, numBlocks;  /* Blocks are processed concurrently */
  PetscInt        batchSize, numBatches; /* A batch is made up of blocks, Batches are processed in serial */
};

typedef struct {
  PetscInt cellType;
} PetscFE_Basic;

typedef struct {
  PetscInt dummy;
} PetscFE_Nonaffine;

#ifdef PETSC_HAVE_OPENCL

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

typedef struct {
  cl_platform_id   pf_id;
  cl_device_id     dev_id;
  cl_context       ctx_id;
  cl_command_queue queue_id;
  PetscDataType    realType;
  PetscLogEvent    residualEvent;
  PetscInt         op; /* ANDY: Stand-in for real equation code generation */
} PetscFE_OpenCL;
#endif

typedef struct {
  PetscInt   cellRefiner;    /* The cell refiner defining the cell division */
  PetscInt   numSubelements; /* The number of subelements */
  PetscReal *v0;             /* The affine transformation for each subelement */
  PetscReal *jac, *invjac;
  PetscInt  *embedding;      /* Map from subelements dofs to element dofs */
} PetscFE_Composite;

#endif
