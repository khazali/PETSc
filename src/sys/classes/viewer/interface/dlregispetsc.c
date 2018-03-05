
#include <petscdraw.h>
#include <petscviewer.h>
#include <petsc/private/viewerimpl.h>

extern PetscLogEvent PETSC_Barrier,PETSC_BuildTwoSided,PETSC_BuildTwoSidedF;
static PetscBool PetscSysPackageInitialized = PETSC_FALSE;

PetscViewerFormat PETSC_VIEWER_DEFAULT;
PetscViewerFormat PETSC_VIEWER_ASCII_MATLAB;
PetscViewerFormat PETSC_VIEWER_ASCII_IMPL;
PetscViewerFormat PETSC_VIEWER_ASCII_INFO;
PetscViewerFormat PETSC_VIEWER_ASCII_INFO_DETAIL;
PetscViewerFormat PETSC_VIEWER_ASCII_COMMON;
PetscViewerFormat PETSC_VIEWER_ASCII_INDEX;
PetscViewerFormat PETSC_VIEWER_ASCII_MATRIXMARKET;
PetscViewerFormat PETSC_VIEWER_ASCII_VTK;
PetscViewerFormat PETSC_VIEWER_ASCII_VTK_CELL;
PetscViewerFormat PETSC_VIEWER_ASCII_VTK_COORDS;
PetscViewerFormat PETSC_VIEWER_ASCII_PYTHON;
PetscViewerFormat PETSC_VIEWER_ASCII_LATEX;
PetscViewerFormat PETSC_VIEWER_ASCII_XML;
PetscViewerFormat PETSC_VIEWER_ASCII_GLVIS;
PetscViewerFormat PETSC_VIEWER_DRAW_BASIC;
PetscViewerFormat PETSC_VIEWER_DRAW_LG;
PetscViewerFormat PETSC_VIEWER_DRAW_CONTOUR;
PetscViewerFormat PETSC_VIEWER_DRAW_PORTS;
PetscViewerFormat PETSC_VIEWER_VTK_VTS;
PetscViewerFormat PETSC_VIEWER_VTK_VTR;
PetscViewerFormat PETSC_VIEWER_VTK_VTU;
PetscViewerFormat PETSC_VIEWER_BINARY_MATLAB;
PetscViewerFormat PETSC_VIEWER_NATIVE;
PetscViewerFormat PETSC_VIEWER_NOFORMAT;
PetscViewerFormat PETSC_VIEWER_LOAD_BALANCE;

/*@C
  PetscSysFinalizePackage - This function destroys everything in the PETSc created internally in the system library portion of PETSc.
  It is called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package
.seealso: PetscFinalize()
@*/
PetscErrorCode  PetscSysFinalizePackage(void)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0; i<PetscViewerFormatNumber; i++) {
    ierr = PetscFree(PetscViewerFormats[i]);CHKERRQ(ierr);
  }
  PetscViewerFormatNumber = 0;
  if (Petsc_Seq_keyval != MPI_KEYVAL_INVALID) {
    ierr = MPI_Comm_free_keyval(&Petsc_Seq_keyval);CHKERRQ(ierr);
  }
  PetscSysPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscSysInitializePackage - This function initializes everything in the main Petsc package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the call to PetscInitialize()
  when using static libraries.

  Level: developer

.keywords: Petsc, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode  PetscSysInitializePackage(void)
{
  char           logList[256];
  char           *className;
  PetscBool      opt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscSysPackageInitialized) PetscFunctionReturn(0);
  PetscSysPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("Object",&PETSC_OBJECT_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("Container",&PETSC_CONTAINER_CLASSID);CHKERRQ(ierr);

  /* Register Events */
  ierr = PetscLogEventRegister("PetscBarrier", PETSC_SMALLEST_CLASSID,&PETSC_Barrier);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("BuildTwoSided",PETSC_SMALLEST_CLASSID,&PETSC_BuildTwoSided);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("BuildTwoSidedF",PETSC_SMALLEST_CLASSID,&PETSC_BuildTwoSidedF);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(NULL,NULL, "-info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "null", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(0);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL, "-log_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "null", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(0);CHKERRQ(ierr);
    }
  }

  ierr = PetscViewerFormatRegister("DEFAULT",&PETSC_VIEWER_DEFAULT);CHKERRQ(ierr);
  ierr = PetscViewerFormatRegister("ASCII_MATRIXMARKET",&PETSC_VIEWER_ASCII_MATRIXMARKET);CHKERRQ(ierr);
  ierr = PetscViewerFormatRegister("ASCII_MATLAB",&PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
  ierr = PetscViewerFormatRegister("ASCII_IMPL",&PETSC_VIEWER_ASCII_IMPL);CHKERRQ(ierr);
  ierr = PetscViewerFormatRegister("ASCII_INFO",&PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
  ierr = PetscViewerFormatRegister("ASCII_INFO_DETAIL",&PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
  ierr = PetscViewerFormatRegister("ASCII_COMMON",&PETSC_VIEWER_ASCII_COMMON);CHKERRQ(ierr);
  ierr = PetscViewerFormatRegister("ASCII_INDEX",&PETSC_VIEWER_ASCII_INDEX);CHKERRQ(ierr);
  ierr = PetscViewerFormatRegister("ASCII_VTK",&PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
  ierr = PetscViewerFormatRegister("ASCII_VTK_CELL",&PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);
  ierr = PetscViewerFormatRegister("ASCII_VTK_COORDS",&PETSC_VIEWER_ASCII_VTK_COORDS);CHKERRQ(ierr);
  ierr = PetscViewerFormatRegister("ASCII_PYTHON",&PETSC_VIEWER_ASCII_PYTHON);CHKERRQ(ierr);
  ierr = PetscViewerFormatRegister("ASCII_LATEX",&PETSC_VIEWER_ASCII_LATEX);CHKERRQ(ierr);
  ierr = PetscViewerFormatRegister("ASCII_XML",&PETSC_VIEWER_ASCII_XML);CHKERRQ(ierr);
  ierr = PetscViewerFormatRegister("ASCII_GLVIS",&PETSC_VIEWER_ASCII_GLVIS);CHKERRQ(ierr);
  ierr = PetscViewerFormatRegister("DRAW_BASIC",&PETSC_VIEWER_DRAW_BASIC);CHKERRQ(ierr);
  ierr = PetscViewerFormatRegister("DRAW_LG",&PETSC_VIEWER_DRAW_LG);CHKERRQ(ierr);
  ierr = PetscViewerFormatRegister("DRAW_CONTOUR",&PETSC_VIEWER_DRAW_CONTOUR);CHKERRQ(ierr);
  ierr = PetscViewerFormatRegister("DRAW_PORTS",&PETSC_VIEWER_DRAW_PORTS);CHKERRQ(ierr);
  ierr = PetscViewerFormatRegister("VTK_VTS",&PETSC_VIEWER_VTK_VTS);CHKERRQ(ierr);
  ierr = PetscViewerFormatRegister("VTK_VTR",&PETSC_VIEWER_VTK_VTR);CHKERRQ(ierr);
  ierr = PetscViewerFormatRegister("VTK_VTU",&PETSC_VIEWER_VTK_VTU);CHKERRQ(ierr);
  ierr = PetscViewerFormatRegister("BINARY_MATLAB",&PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
  ierr = PetscViewerFormatRegister("NATIVE",&PETSC_VIEWER_NATIVE);CHKERRQ(ierr);
  ierr = PetscViewerFormatRegister("NOFORMAT",&PETSC_VIEWER_NOFORMAT);CHKERRQ(ierr);
  ierr = PetscViewerFormatRegister("LOAD_BALANCE",&PETSC_VIEWER_LOAD_BALANCE);CHKERRQ(ierr);  

  ierr = PetscRegisterFinalize(PetscSysFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)

#if defined(PETSC_USE_SINGLE_LIBRARY)
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscvec(void);
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscmat(void);
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscdm(void);
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscksp(void);
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscsnes(void);
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscts(void);
#endif

#if defined(PETSC_USE_SINGLE_LIBRARY)
#else
#endif
/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers all the draw and PetscViewer objects.

 */
#if defined(PETSC_USE_SINGLE_LIBRARY)
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petsc(void)
#else
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscsys(void)
#endif
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*
      If we got here then PETSc was properly loaded
  */
  ierr = PetscSysInitializePackage();CHKERRQ(ierr);
  ierr = PetscDrawInitializePackage();CHKERRQ(ierr);
  ierr = PetscViewerInitializePackage();CHKERRQ(ierr);
  ierr = PetscRandomInitializePackage();CHKERRQ(ierr);

#if defined(PETSC_USE_SINGLE_LIBRARY)
  ierr = PetscDLLibraryRegister_petscvec();CHKERRQ(ierr);
  ierr = PetscDLLibraryRegister_petscmat();CHKERRQ(ierr);
  ierr = PetscDLLibraryRegister_petscdm();CHKERRQ(ierr);
  ierr = PetscDLLibraryRegister_petscksp();CHKERRQ(ierr);
  ierr = PetscDLLibraryRegister_petscsnes();CHKERRQ(ierr);
  ierr = PetscDLLibraryRegister_petscts();CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}
#endif  /* PETSC_HAVE_DYNAMIC_LIBRARIES */
