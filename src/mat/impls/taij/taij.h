#if !defined(_TAIJ_H)
#define _TAIJ_H

#include <../src/mat/impls/aij/mpi/mpiaij.h>

#define TAIJHEADER          \
  PetscInt    dof;          \
  Mat         AIJ;          \
  PetscScalar *S;           \
  PetscScalar *ibdiag;      \
  PetscBool   ibdiagvalid;  \

typedef struct {
  TAIJHEADER;
} Mat_SeqTAIJ;

typedef struct {
  TAIJHEADER;
  Mat        OAIJ;    /* representation of interpolation for one component */
  Mat        A;
  VecScatter ctx;     /* update ghost points for parallel case */
  Vec        w;       /* work space for ghost values for parallel case */
} Mat_MPITAIJ;

#endif
