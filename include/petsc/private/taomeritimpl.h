#ifndef __TAOMERIT_IMPL_H
#define __TAOMEIRT_IMPL_H
#include <petscvec.h>
#include <petsc/private/petscimpl.h>
#include <petsctaomerit.h>

typedef struct _TaoMeritOps *TaoMeritOps;
struct _TaoMeritOps {
    PetscErrorCode (*computeobjective)(TaoMerit, Vec, PetscReal*, void*);
    PetscErrorCode (*computegradient)(TaoMerit, Vec, Vec, void*);
    PetscErrorCode (*computeobjectiveandgradient)(TaoMerit, Vec, PetscReal *, Vec, void*);
    PetscErrorCode (*computeobjectiveandgts)(TaoMerit, Vec, Vec, PetscReal*, PetscReal*, void*);
    PetscErrorCode (*setup)(TaoMerit);
    PetscErrorCode (*computevalue)(TaoMerit,PetscReal,PetscReal*);
    PetscErrorCode (*computedirderiv)(TaoMerit,PetscReal,PetscReal*);
    PetscErrorCode (*computeall)(TaoMerit,PetscReal,PetscReal*,PetscReal*);
    PetscErrorCode (*view)(TaoMerit,PetscViewer);
    PetscErrorCode (*setfromoptions)(PetscOptionItems*,TaoMerit);
    PetscErrorCode (*reset)(TaoMerit,Vec,Vec);
    PetscErrorCode (*destroy)(TaoMerit);
};

struct _p_TaoMerit {
    PETSCHEADER(struct _TaoMerit);
    void *userctx_func;
    void *userctx_grad;
    void *userctx_funcgrad;
    void *userctx_funcgts;

    PetscBool setupcalled;
    PetscBool usegts;
    PetscBool usetaoroutines;
    PetscBool hasobjective;
    PetscBool hasgradient;
    PetscBool hasobjectiveandgradient;
    void *data;
    
    Vec x0, x_alpha, s;
    PetscReal f0, f_alpha, alpha;

    PetscInt nfeval;
    PetscInt ngeval;
    PetscInt nfgeval;

    Tao tao;
};

extern PetscLogEvent;
#endif
