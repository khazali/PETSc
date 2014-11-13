#include <../src/mat/impls/elemental/matelemimpl.h> /*I "petscmat.h" I*/

/*
    The variable Petsc_Elemental_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a Mat_Elemental_Grid
*/

#undef __FUNCT__
#define __FUNCT__ "PetscElementalInitializePackage"
/*@C
   PetscElementalInitializePackage - Initialize Elemental package

   Logically Collective

   Level: developer

.seealso: MATELEMENTAL, PetscElementalFinalizePackage()
@*/
PetscErrorCode PetscElementalInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (El::Initialized()) PetscFunctionReturn(0);
  { /* We have already initialized MPI, so this song and dance is just to pass these variables (which won't be used by Elemental) through the interface that needs references */
    int zero = 0;
    char **nothing = 0;
    El::Initialize(zero,nothing);
  }
  ierr = PetscRegisterFinalize(PetscElementalFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscElementalFinalizePackage"
/*@C
   PetscElementalFinalizePackage - Finalize Elemental package

   Logically Collective

   Level: developer

.seealso: MATELEMENTAL, PetscElementalInitializePackage()
@*/
PetscErrorCode PetscElementalFinalizePackage(void)
{

  PetscFunctionBegin;
  El::Finalize();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatFactorGetSolverPackage_elemental_elemental"
PetscErrorCode MatFactorGetSolverPackage_elemental_elemental(Mat A,const MatSolverPackage *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERELEMENTAL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolverPackageRegister_Elemental"
PETSC_EXTERN PetscErrorCode MatSolverPackageRegister_Elemental(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolverPackageRegister(MATSOLVERELEMENTAL,MATELEMDENSE,MAT_FACTOR_LU,MatGetFactor_elemdense_elemdense);CHKERRQ(ierr);
  ierr = MatSolverPackageRegister(MATSOLVERELEMENTAL,MATELEMDENSE,MAT_FACTOR_CHOLESKY,MatGetFactor_elemdense_elemdense);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
