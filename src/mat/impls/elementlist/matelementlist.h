

#if !defined(__matelementlist_h)
#define __matelementlist_h

#include <petscmat.h>

typedef struct _Mat_ElementlistLink *Mat_ElementlistLink;
struct _Mat_ElementlistLink {
  Mat                 elem;          /* element matrix */
  IS                  row,col;       /* global row and columns for the element */
  Vec                 rv,cv;         /* element work vectors */
  Mat_ElementlistLink next;
  Mat_ElementlistLink previous;
};

typedef struct {
  Mat_ElementlistLink head;          /* current head for MatSetValues */
  Mat_ElementlistLink root;          /* root node for linked list of elements */
  PetscBool           unsym;         /* non-square matrix or non-square elements */
  PetscBool           roworiented;   /* handle row- column-major ordering for MatSetValues */
  PetscBool           hasneg[3];     /* handle MatSetValues with negative indices */
  Vec                 rv,cv;         /* local work vectors for row and columns */
  VecScatter          rscctx,cscctx; /* scatters from assembled vector to local work vectors */
  IS                  rgis,cgis;     /* aggregated global indices from local elements */
  IS                  ris,cis;       /* aggregated local indices from local elements */
} Mat_Elementlist;

#endif
