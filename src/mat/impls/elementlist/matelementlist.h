

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
  Mat_ElementlistLink head;
  Mat_ElementlistLink root;
  PetscBool           roworiented;
  Vec                 rv,cv;         /* local work vectors for row and columns */
  VecScatter          rscctx,cscctx; /* scatters from assembled vector to local work vectors */
} Mat_Elementlist;

#endif
