
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: index.c,v 1.54 1998/04/21 17:32:11 curfman Exp curfman $";
#endif
/*  
   Defines the abstract operations on index sets, i.e. the public interface. 
*/
#include "src/is/isimpl.h" 
     /*I "is.h" I*/

#undef __FUNC__  
#define __FUNC__ "ISIdentity" 
/*@C
   ISIdentity - Determines whether index set is the identity mapping.

   Collective on IS

   Input Parmeters:
.  is - the index set

   Output Parameters:
.  ident - PETSC_TRUE if an identity, else PETSC_FALSE

.keywords: IS, index set, identity

.seealso: ISSetIdentity()
@*/
int ISIdentity(IS is,PetscTruth *ident)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE);
  PetscValidIntPointer(ident);
  *ident = (PetscTruth) is->isidentity;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISSetIdentity" 
/*@
   ISSetIdentity - Informs the index set that it is an identity.

   Collective on IS

   Input Parmeters:
.  is - the index set

.keywords: IS, index set, identity

.seealso: ISIdentity()
@*/
int ISSetIdentity(IS is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE);
  is->isidentity = 1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISPermutation" 
/*@C
   ISPermutation - PETSC_TRUE or PETSC_FALSE depending on whether the 
   index set has been declared to be a permutation.

   Collective on IS

   Input Parmeters:
.  is - the index set

   Output Parameters:
.  perm - PETSC_TRUE if a permutation, else PETSC_FALSE

.keywords: IS, index set, permutation

.seealso: ISSetPermutation()
@*/
int ISPermutation(IS is,PetscTruth *perm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE);
  PetscValidIntPointer(perm);
  *perm = (PetscTruth) is->isperm;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISSetPermutation" 
/*@
   ISSetPermutation - Informs the index set that it is a permutation.

   Collective on IS

   Input Parmeters:
.  is - the index set

.keywords: IS, index set, permutation

.seealso: ISPermutation()
@*/
int ISSetPermutation(IS is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE);
  is->isperm = 1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISDestroy" 
/*@C
   ISDestroy - Destroys an index set.

   Collective on IS

   Input Parameters:
.  is - the index set

.keywords: IS, index set, destroy

.seealso: ISCreateSeq(), ISCreateStride()
@*/
int ISDestroy(IS is)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE);
  if (--is->refct > 0) PetscFunctionReturn(0);
  ierr = (*is->ops->destroy)(is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISInvertPermutation" 
/*@C
   ISInvertPermutation - Creates a new permutation that is the inverse of 
                         a given permutation.

   Collective on IS

   Input Parameter:
.  is - the index set

   Output Parameter:
.  isout - the inverse permutation

.keywords: IS, index set, invert, inverse, permutation
@*/
int ISInvertPermutation(IS is,IS *isout)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE);
  if (!is->isperm) SETERRQ(PETSC_ERR_ARG_WRONG,0,"not a permutation");
  ierr = (*is->ops->invertpermutation)(is,isout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISGetSize" 
/*@
   ISGetSize - Returns the global length of an index set. 

   Not Collective

   Input Parameter:
.  is - the index set

   Output Parameter:
.  size - the global size

.keywords: IS, index set, get, global, size

@*/
int ISGetSize(IS is,int *size)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE);
  PetscValidIntPointer(size);
  ierr = (*is->ops->getsize)(is,size);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISGetIndices" 
/*@C
   ISGetIndices - Returns a pointer to the indices.  The user should call 
   ISRestoreIndices() after having looked at the indices.  The user should 
   NOT change the indices.

   Not Collective

   Input Parameter:
.  is - the index set

   Output Parameter:
.  ptr - the location to put the pointer to the indices

   Fortran Note:
   The Fortran interface is slightly different from that given below.
   See the Fortran chapter of the users manual and 
   petsc/src/is/examples for details.  

.keywords: IS, index set, get, indices

.seealso: ISRestoreIndices(), ISGetIndicesF90()
@*/
int ISGetIndices(IS is,int **ptr)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE);
  PetscValidPointer(ptr);
  ierr = (*is->ops->getindices)(is,ptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef __FUNC__  
#define __FUNC__ "ISRestoreIndices" 
/*@C
   ISRestoreIndices - Restores an index set to a usable state after a call 
                      to ISGetIndices().

   Not Collective

   Input Parameters:
+  is - the index set
-  ptr - the pointer obtained by ISGetIndices()

   Fortran Note:
   The Fortran interface is slightly different from that given below.
   See the users manual and petsc/src/is/examples for details.

.keywords: IS, index set, restore, indices

.seealso: ISGetIndices(), ISRestoreIndicesF90()
@*/
int ISRestoreIndices(IS is,int **ptr)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE);
  PetscValidPointer(ptr);
  if (is->ops->restoreindices) {
    ierr = (*is->ops->restoreindices)(is,ptr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISView" 
/*@C
   ISView - Displays an index set.

   Collective on IS unless Viewer is sequential

   Input Parameters:
+  is - the index set
-  viewer - viewer used to display the set, for example VIEWER_STDOUT_SELF.

.keywords: IS, index set, indices

.seealso: ViewerFileOpenASCII()
@*/
int ISView(IS is, Viewer viewer)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE);
  if (viewer) {PetscValidHeader(viewer);}
  else { viewer = VIEWER_STDOUT_SELF; }
  ierr = (*is->ops->view)(is,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISSort" 
/*@
   ISSort - Sorts the indices of an index set.

   Collective on IS

   Input Parameters:
.  is - the index set

.keywords: IS, index set, sort, indices

.seealso: ISSorted()
@*/
int ISSort(IS is)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE);
  ierr = (*is->ops->sortindices)(is); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISSorted" 
/*@C
   ISSorted - Checks the indices to determine whether they have been sorted.

  Collective on IS

   Input Parameter:
.  is - the index set

   Output Parameter:
.  flg - output flag, either PETSC_TRUE if the index set is sorted, 
         or PETSC_FALSE otherwise.

.keywords: IS, index set, sort, indices

.seealso: ISSort()
@*/
int ISSorted(IS is, PetscTruth *flg)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE);
  PetscValidIntPointer(flg);
  ierr = (*is->ops->sorted)(is, flg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISDuplicate" 
/*@C
   ISDuplicate - Determines whether index set is the identity mapping.

   Collective on IS

   Input Parmeters:
.  is - the index set

   Output Parameters:
.  ident - PETSC_TRUE if an identity, otherwise PETSC_FALSE

.keywords: IS, index set, identity

.seealso: ISCreateGeneral()
@*/
int ISDuplicate(IS is, IS *newIS)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is, IS_COOKIE);
  PetscValidPointer(newIS);
  ierr = (*is->ops->duplicate)(is, newIS); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*MC
    ISGetIndicesF90 - Accesses the elements of an index set from Fortran90.
    The users should call ISRestoreIndicesF90() after having looked at the
    indices.  The user should NOT change the indices.

    Synopsis:
    ISGetIndicesF90(IS x,{Scalar, pointer :: xx_v(:)},integer ierr)

    Not collective

    Input Parameter:
.   x - index set

    Output Parameters:
+   xx_v - the Fortran90 pointer to the array
-   ierr - error code

    Example of Usage: 
.vb
    Scalar, pointer xx_v(:)
    ....
    call ISGetIndicesF90(x,xx_v,ierr)
    a = xx_v(3)
    call ISRestoreIndicesF90(x,xx_v,ierr)
.ve

    Notes:
    Currently only supported using the NAG F90 compiler.

.seealso:  ISRestoreIndicesF90(), ISGetIndices(), ISRestoreIndices()

.keywords:  IS, index set, get, indices, f90
M*/

/*MC
    ISRestoreIndicesF90 - Restores an index set to a usable state after
    a call to ISGetIndicesF90().

    Input Parameters:
.   x - index set
.   xx_v - the Fortran90 pointer to the array

    Output Parameter:
.   ierr - error code

    Synopsis:
    ISRestoreIndicesF90(IS x,{Scalar, pointer :: xx_v(:)},integer ierr)

    Example of Usage: 
.vb
    Scalar, pointer xx_v(:)
    ....
    call ISGetIndicesF90(x,xx_v,ierr)
    a = xx_v(3)
    call ISRestoreIndicesF90(x,xx_v,ierr)
.ve
   
    Notes:
    Currently only supported using the NAG F90 compiler.

.seealso:  ISGetIndicesF90(), ISGetIndices(), ISRestoreIndices()

.keywords:  IS, index set, restore, indices, f90
M*/

/*MC
    ISBlockGetIndicesF90 - Accesses the elements of an index set from Fortran90.
    The users should call ISBlockRestoreIndicesF90() after having looked at the
    indices.  The user should NOT change the indices.

    Synopsis:
    ISBlockGetIndicesF90(IS x,{Scalar, pointer :: xx_v(:)},integer ierr)

    Not collective

    Input Parameter:
.   x - index set

    Output Parameters:
+   xx_v - the Fortran90 pointer to the array
-   ierr - error code
    Example of Usage: 
.vb
    Scalar, pointer xx_v(:)
    ....
    call ISBlockGetIndicesF90(x,xx_v,ierr)
    a = xx_v(3)
    call ISBlockRestoreIndicesF90(x,xx_v,ierr)
.ve

    Notes:
    Currently only supported using the NAG F90 compiler.

.seealso:  ISBlockRestoreIndicesF90(), ISGetIndices(), ISRestoreIndices(),
           ISRestoreIndices()

.keywords:  IS, index set, get, indices, f90
M*/

/*MC
    ISBlockRestoreIndicesF90 - Restores an index set to a usable state after
    a call to ISBlockGetIndicesF90().

    Input Parameters:
.   x - index set
.   xx_v - the Fortran90 pointer to the array

    Output Parameter:
.   ierr - error code

    Synopsis:
    ISBlockRestoreIndicesF90(IS x,{Scalar, pointer :: xx_v(:)},integer ierr)

    Example of Usage: 
.vb
    Scalar, pointer xx_v(:)
    ....
    call ISBlockGetIndicesF90(x,xx_v,ierr)
    a = xx_v(3)
    call ISBlockRestoreIndicesF90(x,xx_v,ierr)
.ve
   
    Notes:
    Currently only supported using the NAG F90 compiler.

.seealso:  ISBlockGetIndicesF90(), ISGetIndices(), ISRestoreIndices(), ISRestoreIndicesF90()

.keywords:  IS, index set, restore, indices, f90
M*/


