#if !defined(__TAODEF_H)
#define __TAODEF_H

#include "petsc/finclude/petscts.h"

#define Tao PetscFortranAddr
#define TaoLineSearch PetscFortranAddr
#define TaoConvergedReason integer
#define TaoType character*(80)
#define TaoLineSearchType character*(80)

#define TaoProblemType PetscEnum
#define TAO_PROBLEM_UNKNOWN         1
#define TAO_PROBLEM_LINEAR          2
#define TAO_PROBLEM_QUADRATIC       3
#define TAO_PROBLEM_NONLINEAR       4
#define TAO_PROBLEM_COMPLEMENTARITY 5
#define TAO_PROBLEM_FULLSPACEPDE    6

#define TAOLMVM     "lmvm"
#define TAONLS      "nls"
#define TAONTR      "ntr"
#define TAONTL      "ntl"
#define TAOCG       "cg"
#define TAOTRON     "tron"
#define TAOOWLQN    "owlqn"
#define TAOBMRM     "bmrm"
#define TAOBLMVM    "blmvm"
#define TAOBQPIP    "bqpip"
#define TAOGPCG     "gpcg"
#define TAONM       "nm"
#define TAOPOUNDERS "pounders"
#define TAOLCL      "lcl"
#define TAOSSILS    "ssils"
#define TAOSSFLS    "ssfls"
#define TAOASILS    "asils"
#define TAOASFLS    "asfls"
#define TAOIPM      "ipm"
#define TAOFDTEST   "test"

#endif
