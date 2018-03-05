
#include <petsc/private/viewerimpl.h>  /*I "petscsys.h" I*/

#define PETSC_VIEWER_FORMAT_MAX 100
char *PetscViewerFormats[PETSC_VIEWER_FORMAT_MAX];
PetscViewerFormat PetscViewerFormatNumber = 0;

/*@C
   PetscViewerFormatRegister - Register a new viewer format

   Logically Collective

   Input Parameter:
.  name - a string name for the format

   Output Parameter:
.  format - the format

   Level: intermediate

.seealso: PetscViwerFormat, PetscViewerPushFormat(), PetscViewerPopFormat(), PetscViewerFormatGet()

@*/
PetscErrorCode PetscViewerFormatRegister(const char *name,PetscViewerFormat *value)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscViewerFormatNumber > PETSC_VIEWER_FORMAT_MAX-1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Rebuild with a larger PETSC_VIEWER_FORMAT_MAX");
  ierr = PetscStrallocpy(name,&PetscViewerFormats[PetscViewerFormatNumber]);CHKERRQ(ierr);
  *value = PetscViewerFormatNumber++;
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerFormatGet - gets the PetscViewerFormat value for a string name of a format

   Logically Collective

   Input Parameter:
.  name - a string name for the format

   Output Parameter:
.  format - the format

   Notes: This routine is most useful from Fortran since in C the various format values are available as variables

   Level: intermediate

.seealso: PetscViwerFormat, PetscViewerPushFormat(), PetscViewerPopFormat(), PetscViewerFormatRegister()

@*/
PetscErrorCode PetscViewerFormatGet(const char *name,PetscViewerFormat *value)
{
  PetscErrorCode ierr;
  PetscBool      flag;
  
  PetscFunctionBegin;
  ierr = PetscEListFind(PetscViewerFormatNumber,(const char *const *)PetscViewerFormats,name,value,&flag);CHKERRQ(ierr);
  if (!flag) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unknown viewer format %s",name);
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerSetFormat - Sets the format for PetscViewers.

   Logically Collective on PetscViewer

   This routine is deprecated, you should use PetscViewerPushFormat()/PetscViewerPopFormat()

   Input Parameters:
+  viewer - the PetscViewer
-  format - the format

   Level: intermediate

   Notes:
   Available formats include
+    PETSC_VIEWER_DEFAULT - default format
.    PETSC_VIEWER_ASCII_MATLAB - MATLAB format
.    PETSC_VIEWER_ASCII_DENSE - print matrix as dense
.    PETSC_VIEWER_ASCII_IMPL - implementation-specific format
      (which is in many cases the same as the default)
.    PETSC_VIEWER_ASCII_INFO - basic information about object
.    PETSC_VIEWER_ASCII_INFO_DETAIL - more detailed info
       about object
.    PETSC_VIEWER_ASCII_COMMON - identical output format for
       all objects of a particular type
.    PETSC_VIEWER_ASCII_INDEX - (for vectors) prints the vector
       element number next to each vector entry
.    PETSC_VIEWER_ASCII_SYMMODU - print parallel vectors without
       indicating the processor ranges
.    PETSC_VIEWER_ASCII_VTK - outputs the object to a VTK file
.    PETSC_VIEWER_NATIVE - store the object to the binary
       file in its native format (for example, dense
       matrices are stored as dense), DMDA vectors are dumped directly to the
       file instead of being first put in the natural ordering
.    PETSC_VIEWER_DRAW_BASIC - views the vector with a simple 1d plot
.    PETSC_VIEWER_DRAW_LG - views the vector with a line graph
-    PETSC_VIEWER_DRAW_CONTOUR - views the vector with a contour plot

   These formats are most often used for viewing matrices and vectors.

   If a format (for example PETSC_VIEWER_DRAW_CONTOUR) was applied to a viewer
  where it didn't apply (PETSC_VIEWER_STDOUT_WORLD) it cause the default behavior
  for that viewer to be used.

    Note: This supports passing in a NULL for the viewer for use in the debugger, but it should never be called in the code with a NULL viewer

   Concepts: PetscViewer^setting format

.seealso: PetscViewerASCIIOpen(), PetscViewerBinaryOpen(), MatView(), VecView(), PetscViewerType,
          PetscViewerPushFormat(), PetscViewerPopFormat(), PetscViewerDrawOpen(),PetscViewerSocketOpen()
@*/
PetscErrorCode  PetscViewerSetFormat(PetscViewer viewer,PetscViewerFormat format)
{
  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_SELF;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidLogicalCollectiveEnum(viewer,format,2);
  viewer->format = format;
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerPushFormat - Sets the format for file PetscViewers.

   Logically Collective on PetscViewer

   Input Parameters:
+  viewer - the PetscViewer
-  format - the format

   Level: intermediate

   Notes:
   Available formats include
+    PETSC_VIEWER_DEFAULT - default format
.    PETSC_VIEWER_ASCII_MATLAB - MATLAB format
.    PETSC_VIEWER_ASCII_IMPL - implementation-specific format
      (which is in many cases the same as the default)
.    PETSC_VIEWER_ASCII_INFO - basic information about object
.    PETSC_VIEWER_ASCII_INFO_DETAIL - more detailed info
       about object
.    PETSC_VIEWER_ASCII_COMMON - identical output format for
       all objects of a particular type
.    PETSC_VIEWER_ASCII_INDEX - (for vectors) prints the vector
       element number next to each vector entry
.    PETSC_VIEWER_NATIVE - store the object to the binary
       file in its native format (for example, dense
       matrices are stored as dense), for DMDA vectors displays vectors in DMDA ordering, not natural
.    PETSC_VIEWER_DRAW_BASIC - views the vector with a simple 1d plot
.    PETSC_VIEWER_DRAW_LG - views the vector with a line graph
-    PETSC_VIEWER_DRAW_CONTOUR - views the vector with a contour plot

   These formats are most often used for viewing matrices and vectors.
   Currently, the object name is used only in the MATLAB format.

   Concepts: PetscViewer^setting format

.seealso: PetscViewerASCIIOpen(), PetscViewerBinaryOpen(), MatView(), VecView(),
          PetscViewerSetFormat(), PetscViewerPopFormat()
@*/
PetscErrorCode  PetscViewerPushFormat(PetscViewer viewer,PetscViewerFormat format)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidLogicalCollectiveEnum(viewer,format,2);
  if (viewer->iformat > PETSCVIEWERFORMATPUSHESMAX-1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many PetscViewerPushFormat(), perhaps you forgot PetscViewerPopFormat()?");

  viewer->formats[viewer->iformat++] = viewer->format;
  viewer->format                     = format;
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerPopFormat - Resets the format for file PetscViewers.

   Logically Collective on PetscViewer

   Input Parameters:
.  viewer - the PetscViewer

   Level: intermediate

   Concepts: PetscViewer^setting format

.seealso: PetscViewerASCIIOpen(), PetscViewerBinaryOpen(), MatView(), VecView(),
          PetscViewerSetFormat(), PetscViewerPushFormat()
@*/
PetscErrorCode  PetscViewerPopFormat(PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  if (viewer->iformat <= 0) PetscFunctionReturn(0);

  viewer->format = viewer->formats[--viewer->iformat];
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscViewerGetFormat(PetscViewer viewer,PetscViewerFormat *format)
{
  PetscFunctionBegin;
  *format =  viewer->format;
  PetscFunctionReturn(0);
}



