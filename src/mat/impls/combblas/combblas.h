
#define PETSC_HAVE_BROKEN_RECURSIVE_MACRO 1
#include <petsc/private/matimpl.h>
#undef NOFILE
#include <CombBLAS/CombBLAS.h>

#include "CombBLAS/SpParMat.h"

typedef struct {
  combblas::SpParMat<PetscInt,PetscScalar,combblas::SpDCCols<PetscInt, PetscScalar>> *combmat;
} Mat_CombBLAS;
