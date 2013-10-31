#ifndef __MATORDERIMPL_H
#define __MATORDERIMPL_H

#include <petscmat.h>
#include <petsc-private/petscimpl.h>

/*
   Defines the interface to the SparsePack routines, translated into C.
*/
PETSC_EXTERN PetscErrorCode SPARSEPACKgen1wd(const PetscInt*,const PetscInt*,const PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode SPARSEPACKgennd(const PetscInt*,const PetscInt*,const PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode SPARSEPACKgenrcm(const PetscInt*,const PetscInt*,const PetscInt*,PetscInt*,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode SPARSEPACKgenqmd(const PetscInt*,const PetscInt*,const PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);

PETSC_EXTERN PetscErrorCode SPARSEPACKqmdrch(const PetscInt*,const PetscInt*,const PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode SPARSEPACKqmdmrg(const PetscInt*,const PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode SPARSEPACKqmdqt(const PetscInt*,const PetscInt*,const PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode SPARSEPACKqmdupd(const PetscInt*,const PetscInt*,const PetscInt*,const PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode SPARSEPACKfnroot(PetscInt*,const PetscInt*,const PetscInt*, PetscInt*, PetscInt*, PetscInt*, PetscInt*);
PETSC_EXTERN PetscErrorCode SPARSEPACKrootls(const PetscInt*,const PetscInt*,const PetscInt*, PetscInt*, PetscInt*, PetscInt*, PetscInt*);
PETSC_EXTERN PetscErrorCode SPARSEPACKfn1wd(PetscInt*,const PetscInt*,const PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode SPARSEPACKrevrse(const PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode SPARSEPACKrootls(const PetscInt*,const PetscInt*,const PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode SPARSEPACKfndsep(PetscInt*,const PetscInt*,const PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode SPARSEPACKdegree(const PetscInt*,const PetscInt*,const PetscInt*, PetscInt*, PetscInt*, PetscInt*, PetscInt*);
PETSC_EXTERN PetscErrorCode SPARSEPACKrcm(const PetscInt*,const PetscInt*,const PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);

PETSC_EXTERN PetscErrorCode HSLmc64AD(const PetscInt *job, PetscInt *m, PetscInt *n, PetscInt *ne, const PetscInt *ip, const PetscInt *irn, PetscScalar *a, PetscInt *num,
                                      PetscInt *perm, PetscInt *liw, PetscInt *iw, PetscInt *ldw, PetscScalar *dw, PetscInt *icntl, PetscScalar *cntl, PetscInt *info);
PETSC_EXTERN PetscErrorCode HSLmc64BD(PetscInt *m, PetscInt *n, PetscInt *ne, const PetscInt *ip, const PetscInt *irn, PetscScalar *a, PetscInt *iperm, PetscInt *num,
                                      PetscInt *jperm, PetscInt *pr, PetscInt *q, PetscInt *l, PetscScalar *d__, PetscScalar *rinf);
PETSC_EXTERN PetscErrorCode HSLmc64SD(PetscInt *m, PetscInt *n, PetscInt *ne, const PetscInt *ip, const PetscInt *irn, PetscScalar *a, PetscInt *iperm, PetscInt *numx,
                                      PetscInt *w, PetscInt *len, PetscInt *lenl, PetscInt *lenh, PetscInt *fc, PetscInt *iw, PetscInt *iw4, PetscScalar *rlx, PetscScalar *rinf);
PETSC_EXTERN PetscErrorCode HSLmc64UD(PetscInt *id, PetscInt *mod, PetscInt *m, PetscInt *n, const PetscInt *irn, PetscInt *lirn, const PetscInt *ip, PetscInt *lenc,
                                      PetscInt *fc, PetscInt *iperm, PetscInt *num, PetscInt *numx, PetscInt *pr, PetscInt *arp, PetscInt *cv, PetscInt *out);
PETSC_EXTERN PetscErrorCode HSLmc64WD(PetscInt *m, PetscInt *n, PetscInt *ne, const PetscInt *ip, const PetscInt *irn, PetscScalar *a, PetscInt *iperm, PetscInt *num,
                                      PetscInt *jperm, PetscInt *out, PetscInt *pr, PetscInt *q, PetscInt *l, PetscScalar *u, PetscScalar *d__, PetscScalar *rinf);
PETSC_EXTERN PetscErrorCode HSLmc64ZD(PetscInt *m, PetscInt *n, const PetscInt *irn, PetscInt *lirn, const PetscInt *ip, PetscInt *lenc, PetscInt *iperm, PetscInt *num,
                                      PetscInt *pr, PetscInt *arp, PetscInt *cv, PetscInt *out);
PETSC_EXTERN PetscErrorCode HSLmc64XD(PetscInt *m, PetscInt *n, PetscInt *iperm, PetscInt *rw, PetscInt *cw);
PETSC_EXTERN PetscErrorCode mc64ID(PetscInt *icntl);
PETSC_EXTERN PetscErrorCode mc64AD(PetscInt *job, PetscInt *n, PetscInt *ne, PetscInt *ip, PetscInt *irn, PetscScalar *a, PetscInt *num, PetscInt *cperm, PetscInt *liw, PetscInt *iw, PetscInt *ldw, PetscScalar *dw, PetscInt *icntl, PetscInt *info);
PETSC_EXTERN PetscErrorCode mc64BD(PetscInt *n, PetscInt *ne, PetscInt *ip, PetscInt *irn, PetscScalar *a, PetscInt *iperm, PetscInt *num, PetscInt *jperm, PetscInt *pr, PetscInt *q, PetscInt *l, PetscScalar *d__);
PETSC_EXTERN PetscErrorCode mc64DD(PetscInt *i__, PetscInt *n, PetscInt *q, PetscScalar *d__, PetscInt *l, PetscInt *iway);
PETSC_EXTERN PetscErrorCode mc64ED(PetscInt *qlen, PetscInt *n, PetscInt *q, PetscScalar *d__, PetscInt *l, PetscInt *iway);
PETSC_EXTERN PetscErrorCode mc64FD(PetscInt *pos0, PetscInt *qlen, PetscInt *n, PetscInt *q, PetscScalar *d__, PetscInt *l, PetscInt *iway);
PETSC_EXTERN PetscErrorCode mc64RD(PetscInt *n, PetscInt *ne, const PetscInt *ip, PetscInt *irn, PetscScalar *a);
PETSC_EXTERN PetscErrorCode mc64SD(PetscInt *n, PetscInt *ne, PetscInt *ip, PetscInt *irn, PetscScalar *a, PetscInt *iperm, PetscInt *numx, PetscInt *w, PetscInt *len, PetscInt *lenl, PetscInt *lenh, PetscInt *fc, PetscInt *iw, PetscInt *iw4);
PETSC_EXTERN PetscErrorCode mc64QD(const PetscInt *ip, PetscInt *lenl, PetscInt *lenh, PetscInt *w, PetscInt *wlen, PetscScalar *a, PetscInt *nval, PetscScalar *val);
PETSC_EXTERN PetscErrorCode mc64UD(PetscInt *id, PetscInt *mod, PetscInt *n, PetscInt *irn, PetscInt *lirn, PetscInt *ip, PetscInt *lenc, PetscInt *fc, PetscInt *iperm,
                                   PetscInt *num, PetscInt *numx, PetscInt *pr, PetscInt *arp, PetscInt *cv, PetscInt *out);
PETSC_EXTERN PetscErrorCode mc64WD(PetscInt *n, PetscInt *ne, PetscInt *ip, PetscInt *irn, PetscScalar *a, PetscInt *iperm, PetscInt *num, PetscInt *jperm, PetscInt *out, PetscInt *pr, PetscInt *q, PetscInt *l, PetscScalar *u, PetscScalar *d__);
PETSC_EXTERN PetscErrorCode mc21AD(PetscInt *n, PetscInt *icn, PetscInt *licn, PetscInt *ip, PetscInt *lenr, PetscInt *iperm, PetscInt *numnz, PetscInt *iw);
PETSC_EXTERN PetscErrorCode mc21BD(PetscInt *n, PetscInt *icn, PetscInt *licn, PetscInt *ip, PetscInt *lenr, PetscInt *iperm, PetscInt *numnz, PetscInt *pr, PetscInt *arp, PetscInt *cv, PetscInt *out);
PETSC_EXTERN PetscScalar fd15AD(char *t, int t_len);
#endif
