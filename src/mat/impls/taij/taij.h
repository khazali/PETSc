#if !defined(_TAIJ_H)
#define _TAIJ_H

#include <../src/mat/impls/aij/mpi/mpiaij.h>

#define TAIJHEADER                                \
  PetscInt    p,q;                                \
  Mat         AIJ;                                \
  PetscScalar *S;                                 \
  PetscScalar *T;                                 \
  PetscScalar *ibdiag;                            \
  PetscBool   ibdiagvalid,getrowactive,isTI;      \
  struct {                                        \
    PetscBool setup;                              \
    PetscScalar *w,*work,*t,*arr,*y;              \
  } sor;

typedef struct {
  TAIJHEADER;
} Mat_SeqTAIJ;

typedef struct {
  TAIJHEADER;
  Mat        OAIJ;    
  Mat        A;
  VecScatter ctx;     /* update ghost points for parallel case */
  Vec        w;       /* work space for ghost values for parallel case */
} Mat_MPITAIJ;

#endif
