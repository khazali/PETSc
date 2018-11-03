
static char help[] = "Tests inclusion of petscsystypes.h.\n\n";

#include <petscsystypes.h>
#include <stddef.h> /* for NULL */

int main(int argc,char **argv)
{
  /* numeric types */
  PetscScalar  svalue;
  PetscReal    rvalue;
#if defined(PETSC_HAVE_COMPLEX)
  PetscComplex cvalue;
#endif

  /* integer types */
  PetscInt64   i64;
  PetscInt     i;
  PetscBLASInt bi;
  PetscMPIInt  rank;

  /* PETSc types */
  PetscBool        b;
  PetscErrorCode   ierr;
  PetscClassId     cid;
  PetscEnum        e;
  PetscShort       s;
  PetscChar        c;
  PetscFloat       f;
  PetscLogDouble   ld;
  PetscObjectId    oid;
  PetscObjectState ost;

  /* Enums */
  PetscCopyMode          cp;
  PetscDataType          dt;
  PetscFileMode          fm;
  PetscDLMode            dlm;
  PetscBinarySeekType    bsk;
  PetscBuildTwoSidedType b2s;
  InsertMode             im;
  PetscSubcommType       subct;

  /* Sys objects */
  PetscObject             obj;
  PetscRandom             rand;
  PetscToken              token;
  PetscFunctionList       flist;
  PetscDLHandle           dlh;
  PetscObjectList         olist;
  PetscDLLibrary          dlist;
  PetscContainer          cont;
  PetscSubcomm            subc;
  PetscHeap               pheap;
  PetscShmComm            scomm;
  PetscOmpCtrl            octrl;
  PetscSegBuffer          sbuff;
  PetscOptionsHelpPrinted oh;

  svalue = 0.0;
  rvalue = 0.0;
#if defined(PETSC_HAVE_COMPLEX)
  cvalue = 0.0;
#endif

  i64  = 0;
  i    = 0;
  bi   = 0;
  rank = 0;

  b    = PETSC_FALSE;
  ierr = 0;
  cid  = 0;
  e    = ENUM_DUMMY;
  s    = 0;
  c    = '\0';
  f    = 0;
  ld   = 0.0;
  oid  = 0;
  ost  = 0;

  cp    = PETSC_COPY_VALUES;
  dt    = PETSC_DATATYPE_UNKNOWN;
  fm    = FILE_MODE_READ;
  dlm   = PETSC_DL_DECIDE;
  bsk   = PETSC_BINARY_SEEK_SET;
  b2s   = PETSC_BUILDTWOSIDED_NOTSET;
  im    = INSERT_VALUES;
  subct = PETSC_SUBCOMM_GENERAL;

  obj   = NULL;
  rand  = NULL;
  token = NULL;
  flist = NULL;
  dlh   = NULL;
  olist = NULL;
  dlist = NULL;
  cont  = NULL;
  subc  = NULL;
  pheap = NULL;
  scomm = NULL;
  octrl = NULL;
  sbuff = NULL;
  oh    = NULL;

  /* prevent to issue warning about set-but-not-used variables */
  (void)help;

  (void)svalue;
  (void)rvalue;
#if defined(PETSC_HAVE_COMPLEX)
  (void)cvalue;
#endif
  (void)i64;
  (void)i;
  (void)bi;
  (void)rank;

  (void)b;
  (void)ierr;
  (void)cid;
  (void)e;
  (void)s;
  (void)c;
  (void)f;
  (void)ld;
  (void)oid;
  (void)ost;

  (void)cp;
  (void)dt;
  (void)fm;
  (void)dlm;
  (void)bsk;
  (void)b2s;
  (void)im;
  (void)subct;

  (void)obj;
  (void)rand;
  (void)token;
  (void)flist;
  (void)dlh;
  (void)olist;
  (void)dlist;
  (void)cont;
  (void)subc;
  (void)pheap;
  (void)scomm;
  (void)octrl;
  (void)sbuff;
  (void)oh;
  return ierr;
}


/*TEST

  test:

TEST*/
