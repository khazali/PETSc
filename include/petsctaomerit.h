#ifndef __TAOMERIT_H
#define __TAOMERIT_H
#include <petscvec.h>

#define TaoMeritType  char*
#define TAOMERITOBJ         "obj"

PETSC_EXTERN PetscClassId TAOMERIT_CLASSID;
PETSC_EXTERN PetscFunctionList TaoMeritList;

#include <petsctao.h>

PETSC_EXTERN PetscErrorCode TaoMeritCreate(MPI_Comm,TaoMerit*);
PETSC_EXTERN PetscErrorCode TaoMeritSetFromOptions(TaoMerit);
PETSC_EXTERN PetscErrorCode TaoMeritSetUp(TaoMerit);
PETSC_EXTERN PetscErrorCode TaoMeritDestroy(TaoMerit*);
PETSC_EXTERN PetscErrorCode TaoMeritView(TaoMerit,PetscViewer);
PETSC_STATIC_INLINE PetscErrorCode TaoMeritViewFromOptions(TaoMerit A,PetscObject obj,const char name[]) {return PetscObjectViewFromOptions((PetscObject)A,obj,name);}

PETSC_EXTERN PetscErrorCode TaoMeritSetOptionsPrefix(TaoMerit,const char prefix[]);
PETSC_EXTERN PetscErrorCode TaoMeritReset(TaoMerit,Vec);
PETSC_EXTERN PetscErrorCode TaoMeritAppendOptionsPrefix(TaoMerit,const char prefix[]);
PETSC_EXTERN PetscErrorCode TaoMeritGetOptionsPrefix(TaoMerit,const char *prefix[]);
PETSC_EXTERN PetscErrorCode TaoMeritComputeValue(TaoMerit,PetscReal,PetscReal*);
PETSC_EXTERN PetscErrorCode TaoMeritComputeDirDeriv(TaoMerit,PetscReal,PetscReal*);
PETSC_EXTERN PetscErrorCode TaoMeritComputeValueAndDeriv(TaoMerit,PetscReal,PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode TaoMeritGetStartingVector(TaoMerit,Vec*);
PETSC_EXTERN PetscErrorCode TaoMeritGetNumberFunctionEvaluations(TaoMerit, PetscInt*, PetscInt*, PetscInt*);

PETSC_EXTERN PetscErrorCode TaoMeritGetType(TaoMerit, const TaoMeritType *);
PETSC_EXTERN PetscErrorCode TaoMeritSetType(TaoMerit, const TaoMeritType);

PETSC_EXTERN PetscErrorCode TaoMeritIsUsingTaoRoutines(TaoMerit, PetscBool *);
PETSC_EXTERN PetscErrorCode TaoMeritSetObjectiveAndGTSRoutine(TaoMerit, PetscErrorCode(*)(TaoMerit, Vec, Vec, PetscReal*, PetscReal*, void*), void*);
PETSC_EXTERN PetscErrorCode TaoMeritSetObjectiveRoutine(TaoMerit, PetscErrorCode(*)(TaoMerit, Vec, PetscReal*,void*), void*);
PETSC_EXTERN PetscErrorCode TaoMeritSetGradientRoutine(TaoMerit, PetscErrorCode(*)(TaoMerit, Vec, Vec, void*), void*);
PETSC_EXTERN PetscErrorCode TaoMeritSetObjectiveAndGradientRoutine(TaoMerit, PetscErrorCode(*)(TaoMerit, Vec, PetscReal*, Vec, void*), void*);

PETSC_EXTERN PetscErrorCode TaoMeritComputeObjective(TaoMerit, Vec, PetscReal*);
PETSC_EXTERN PetscErrorCode TaoMeritComputeGradient(TaoMerit, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoMeritComputeObjectiveAndGradient(TaoMerit, Vec, PetscReal*, Vec);
PETSC_EXTERN PetscErrorCode TaoMeritComputeObjectiveAndGTS(TaoMerit, Vec, PetscReal*, PetscReal*);

PETSC_EXTERN PetscErrorCode TaoMeritInitializePackage(void);
PETSC_EXTERN PetscErrorCode TaoMeritFinalizePackage(void);

PETSC_EXTERN PetscErrorCode TaoMeritRegister(const char[], PetscErrorCode (*)(TaoMerit));

#endif
