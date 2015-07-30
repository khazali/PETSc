#include <petscdmda.h>
#include <petscsnes.h>
#include <petscviewerhdf5.h>

static  char help[] ="";


#undef __FUNCT__
#define __FUNCT__ "main"
int main( int argc, char **argv )
{
  Mat              A,Ba;          /* matrix */
  Vec              rhs;
  IS               isa_full;
  IS               isa;
  IS               isb_full;
  IS               isb;
  MPI_Comm         comm;
  PetscViewer      fd,hdf;
  char             Afilename[PETSC_MAX_PATH_LEN-80]={0};
  char             Bfilename[PETSC_MAX_PATH_LEN-80]={0};
  char             RHSfilename[PETSC_MAX_PATH_LEN-80]={0};
  char             ISafilename[PETSC_MAX_PATH_LEN-80]={0};
  char             ISbfilename[PETSC_MAX_PATH_LEN-80]={0};
  char             prefix[80];
  PetscBool        flg;

  PetscInt         lo,hi;
  PetscInt         first,last,size;
  const PetscInt   *indices;
  PetscErrorCode   ierr;

  PetscInitialize(&argc,&argv,(char*)0,help);
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsGetString(NULL,"-prefix",prefix,80,&flg);CHKERRQ(ierr);
  if (flg) {
    size_t plen;
    ierr = PetscStrlen(prefix,&plen);CHKERRQ(ierr);
    ierr = PetscStrncpy(Afilename,prefix,plen+1);CHKERRQ(ierr);
    ierr = PetscStrncpy(Bfilename,prefix,plen+1);CHKERRQ(ierr);
    ierr = PetscStrncpy(RHSfilename,prefix,plen+1);CHKERRQ(ierr);
    ierr = PetscStrncpy(ISafilename,prefix,plen+1);CHKERRQ(ierr);
    ierr = PetscStrncpy(ISbfilename,prefix,plen+1);CHKERRQ(ierr);
  }

  ierr = PetscStrncat(Afilename,"A",1);CHKERRQ(ierr);
  ierr = PetscStrncat(Bfilename,"Ba",2);CHKERRQ(ierr);
  ierr = PetscStrncat(RHSfilename,"RHS",3);CHKERRQ(ierr);
  ierr = PetscStrncat(ISafilename,"ISa",3);CHKERRQ(ierr);
  ierr = PetscStrncat(ISbfilename,"ISb",3);CHKERRQ(ierr);



  /*
    Open binary files.  Note that we use FILE_MODE_READ to indicate
    reading from this file.
  */
  ierr = PetscViewerBinaryOpen(comm,RHSfilename,FILE_MODE_READ,&fd);CHKERRQ(ierr);

  /* create and load a vector */
  ierr = VecCreate(comm,&rhs);CHKERRQ(ierr);
  ierr = VecLoad(rhs,fd);CHKERRQ(ierr);
  ierr = VecView(rhs,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(rhs,&lo,&hi);CHKERRQ(ierr);
  /*Destroy it */
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);


  /* Read in active set IS */
  ierr = PetscViewerHDF5Open(PETSC_COMM_SELF,ISafilename,FILE_MODE_READ,&hdf);CHKERRQ(ierr);
  ierr = ISCreate(PETSC_COMM_SELF,&isa_full);CHKERRQ(ierr);
  PetscObjectSetName((PetscObject)isa_full,"ISa");
  ierr = ISLoad(isa_full,hdf);CHKERRQ(ierr);
  ierr = ISGetIndices(isa_full,&indices);CHKERRQ(ierr);
  ierr = ISGetSize(isa_full,&size);CHKERRQ(ierr);
  first = 0;
  while (indices[first]<lo) first++;
  last=first;
  while (indices[last]<hi && last<size) last++;
  ierr = ISCreateGeneral(comm,last-first,&indices[first],PETSC_COPY_VALUES,&isa);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isa_full,&indices);CHKERRQ(ierr);
  ierr = ISView(isa_full,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = ISView(isa,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&hdf);CHKERRQ(ierr);

  /* Read in basis IS of Ba  (ISb) */
  ierr = PetscViewerHDF5Open(PETSC_COMM_SELF,ISbfilename,FILE_MODE_READ,&hdf);CHKERRQ(ierr);
  ierr = ISCreate(PETSC_COMM_SELF,&isb_full);CHKERRQ(ierr);
  PetscObjectSetName((PetscObject)isb_full,"ISa");
  ierr = ISLoad(isb_full,hdf);CHKERRQ(ierr);
  ierr = ISGetIndices(isb_full,&indices);CHKERRQ(ierr);
  ierr = ISGetSize(isb_full,&size);CHKERRQ(ierr);
  first = 0;
  while (indices[first]<lo) first++;
  last=first;
  while (indices[last]<hi && last<size) last++;
  ierr = ISCreateGeneral(comm,last-first,&indices[first],PETSC_COPY_VALUES,&isb);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isb_full,&indices);CHKERRQ(ierr);
  ierr = ISView(isb_full,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = ISView(isb,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&hdf);CHKERRQ(ierr);



  ierr = PetscViewerBinaryOpen(comm,Afilename,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  /* create and load a matrix */
  ierr = MatCreate(comm,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,hi-lo,hi-lo,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatLoad(A,fd);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(comm,Bfilename,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  /* create and load a matrix */
  ierr = MatCreate(comm,&Ba);CHKERRQ(ierr);
  ierr = MatSetSizes(Ba,last-first,hi-lo,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatLoad(Ba,fd);CHKERRQ(ierr);
  ierr = MatView(Ba,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);


  ierr = VecDestroy(&rhs);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&Ba);CHKERRQ(ierr);
  ierr = ISDestroy(&isa);CHKERRQ(ierr);
  ierr = ISDestroy(&isb);CHKERRQ(ierr);
  ierr = ISDestroy(&isa_full);CHKERRQ(ierr);
  ierr = ISDestroy(&isb_full);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}
