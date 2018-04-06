#include <petsctaomerit.h> /*I "petscTaoMerit.h" I*/
#include <petsc/private/taomeritimpl.h>

/* TODO LIST BELOW:
PetscErrorCode TaoMeritCreate(MPI_Comm,TaoMerit*);
PetscErrorCode TaoMeritSetFromOptions(TaoMerit);
PetscErrorCode TaoMeritSetUp(TaoMerit);
PetscErrorCode TaoMeritDestroy(TaoMerit*);
PetscErrorCode TaoMeritView(TaoMerit,PetscViewer);
PetscErrorCode TaoMeritViewFromOptions(TaoMerit A,PetscObject obj,const char name[]) {return PetscObjectViewFromOptions((PetscObject)A,obj,name);}

PetscErrorCode TaoMeritSetOptionsPrefix(TaoMerit,const char prefix[]);
PetscErrorCode TaoMeritReset(TaoMerit,Vec);
PetscErrorCode TaoMeritAppendOptionsPrefix(TaoMerit,const char prefix[]);
PetscErrorCode TaoMeritGetOptionsPrefix(TaoMerit,const char *prefix[]);
PetscErrorCode TaoMeritComputeValue(TaoMerit,PetscReal,PetscReal*);
PetscErrorCode TaoMeritComputeDirDeriv(TaoMerit,PetscReal,PetscReal*);
PetscErrorCode TaoMeritComputeValueAndDeriv(TaoMerit,PetscReal,PetscReal*,PetscReal*);
PetscErrorCode TaoMeritGetStartingVector(TaoMerit,Vec*);
PetscErrorCode TaoMeritGetNumberFunctionEvaluations(TaoMerit, PetscInt*, PetscInt*, PetscInt*);

PetscErrorCode TaoMeritGetType(TaoMerit, const TaoMeritType *);
PetscErrorCode TaoMeritSetType(TaoMerit, const TaoMeritType);

PetscErrorCode TaoMeritIsUsingTaoRoutines(TaoMerit, PetscBool *);
PetscErrorCode TaoMeritSetObjectiveAndGTSRoutine(TaoMerit, PetscErrorCode(*)(TaoMerit, Vec, Vec, PetscReal*, PetscReal*, void*), void*);
PetscErrorCode TaoMeritSetObjectiveRoutine(TaoMerit, PetscErrorCode(*)(TaoMerit, Vec, PetscReal*,void*), void*);
PetscErrorCode TaoMeritSetGradientRoutine(TaoMerit, PetscErrorCode(*)(TaoMerit, Vec, Vec, void*), void*);
PetscErrorCode TaoMeritSetObjectiveAndGradientRoutine(TaoMerit, PetscErrorCode(*)(TaoMerit, Vec, PetscReal*, Vec, void*), void*);

PetscErrorCode TaoMeritComputeObjective(TaoMerit, Vec, PetscReal*);
PetscErrorCode TaoMeritComputeGradient(TaoMerit, Vec, Vec);
PetscErrorCode TaoMeritComputeObjectiveAndGradient(TaoMerit, Vec, PetscReal*, Vec);
PetscErrorCode TaoMeritComputeObjectiveAndGTS(TaoMerit, Vec, PetscReal*, PetscReal*);

PetscErrorCode TaoMeritInitializePackage(void);
PetscErrorCode TaoMeritFinalizePackage(void);

PetscErrorCode TaoMeritRegister(const char[], PetscErrorCode (*)(TaoMerit));
*/