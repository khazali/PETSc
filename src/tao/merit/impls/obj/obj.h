#ifndef __TAOMERIT_OBJ_H
#define __TAOMERIT_OBJ_H

#include <petsc/private/taomeritimpl.h>
typedef struct {
  Vec g;
  PetscReal gdx0, gdx_alpha;
} TaoMerit_OBJ;

#endif