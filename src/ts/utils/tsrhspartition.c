#include <petsc-private/tsimpl.h> /*I "petscts.h" I*/

/*
 Functions for working with linked lists of Partition/Slot indexed pointers
*/
#undef __FUNCT__
#define __FUNCT__ "DMTSRHSPartitionDataGet"
PetscErrorCode DMTSRHSPartitionDataGet(DMTSRHSPartitionLink start, TSRHSPartitionType type, TSRHSPartitionSlotType slot, void **ptr)
{
  DMTSRHSPartitionLink       lnk;
  DMTSRHSPartitionSlotLink   lnk2;  
  PetscBool               success = PETSC_FALSE;

  PetscFunctionBegin;  
  for(lnk = start; lnk && !success; lnk = lnk->next)
  {
    if(lnk->type == type){
      for(lnk2 = lnk->data; lnk2 && !success; lnk2 = lnk2->next){
        if(lnk2->type == slot){
          *ptr = lnk2->ptr;
          success = PETSC_TRUE;
        }
      }
    }
  }  
  /* If the linked list doesn't have the appropriate entry, return a null pointer */ 
  if(!success){
    *ptr = NULL; 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMTSRHSPartitionDataSet"
PetscErrorCode DMTSRHSPartitionDataSet(DMTSRHSPartitionLink *start, TSRHSPartitionType type, TSRHSPartitionSlotType slot, void *ptr)
{
  PetscErrorCode          ierr;
  PetscBool               found = PETSC_FALSE, found2 = PETSC_FALSE;
  DMTSRHSPartitionLink       lnk;
  DMTSRHSPartitionSlotLink   lnk2; 

  PetscFunctionBegin;
  for(lnk = *start; lnk; lnk = lnk->next){
      if(lnk->type == type){
        found = PETSC_TRUE; 
        for(lnk2 = lnk->data; lnk2; lnk2 = lnk2->next){
          if(lnk2->type == slot){
            found2 = PETSC_TRUE;
            break; 
          }
        }
        break;
      }
  }
  if (!found){
    ierr = PetscMalloc(sizeof(struct _DMTSRHSPartitionLink),&lnk);CHKERRQ(ierr);
    lnk->data = NULL;
    lnk->type = type;
    lnk->next = *start;
    *start = lnk;
  } 
  if(!found2){ 
    ierr = PetscMalloc(sizeof(struct _DMTSRHSPartitionSlotLink),&lnk2);CHKERRQ(ierr);
    lnk2->next = lnk->data;
    lnk->data = lnk2;
    lnk2->type = slot;
  }
  lnk2->ptr = ptr;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMTSRHSPartitionDataDestroy"
PetscErrorCode DMTSRHSPartitionDataDestroy(DMTSRHSPartitionLink start)
{
    DMTSRHSPartitionLink      lnk,  tmp;
    DMTSRHSPartitionSlotLink  lnk2, tmp2;
    
    PetscFunctionBegin;
    lnk = start;
    while(lnk){
      lnk2 = lnk->data;
      while(lnk2){
        tmp2 = lnk2;
        lnk2 = lnk2->next;
        PetscFree(tmp2);
      }
      tmp = lnk;
      lnk = lnk->next;  
      PetscFree(tmp);
    }
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMTSRHSPartitionDataCopy"
/* !! Completely untested !! */
PetscErrorCode DMTSRHSPartitionDataCopy(DMTSRHSPartitionLink startSource, DMTSRHSPartitionLink *startSink)
{
  PetscErrorCode          ierr;
  DMTSRHSPartitionLink       lnk, newlnk, tmp;
  DMTSRHSPartitionSlotLink   lnk2, newlnk2, tmp2;

  PetscFunctionBegin;
  DMTSRHSPartitionDataDestroy(*startSink);
  newlnk = NULL;
  for(lnk = startSource; lnk; lnk=lnk->next)
  {
    tmp = newlnk;
    ierr = PetscMalloc(sizeof(struct _DMTSRHSPartitionLink),&newlnk);
    newlnk->next = tmp;
    newlnk->type = lnk->type;
    newlnk2 = NULL;
    for(lnk2 = lnk->data; lnk2; lnk2=lnk2->next){
      tmp2 = newlnk2;
      ierr = PetscMalloc(sizeof(struct _DMTSRHSPartitionSlotLink),&newlnk2);CHKERRQ(ierr);
      newlnk2->next = tmp2;
      newlnk2->ptr  = lnk2->ptr;
      newlnk2->type = lnk2->type;
    }
    newlnk->data = newlnk2;
  }
  *startSink = newlnk;
  PetscFunctionReturn(0);
}
