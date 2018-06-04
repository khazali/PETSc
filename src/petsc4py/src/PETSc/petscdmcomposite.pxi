# --------------------------------------------------------------------

cdef extern from * nogil:

    int DMCompositeCreate(MPI_Comm,PetscDM*)
    int DMCompositeAddDM(PetscDM,PetscDM)
    int DMCompositeGetNumberDM(PetscDM,PetscInt*)
    int DMCompositeScatterArray(PetscDM,PetscVec,PetscVec*)
    int DMCompositeGatherArray(PetscDM,PetscInsertMode,PetscVec,PetscVec*)
    int DMCompositeGetEntriesArray(PetscDM,PetscDM*)
    int DMCompositeGetAccessArray(PetscDM,PetscVec,PetscInt,const_PetscInt*,PetscVec*)
    int DMCompositeRestoreAccessArray(PetscDM,PetscVec,PetscInt,const_PetscInt*,PetscVec*)
    int DMCompositeGetGlobalISs(PetscDM,PetscIS**)
    int DMCompositeGetLocalISs(PetscDM,PetscIS**)
    int DMCompositeGetISLocalToGlobalMappings(PetscDM,PetscLGMap**)

cdef class _DMComposite_access:
    cdef PetscDM  dm
    cdef PetscVec gvec
    cdef PetscInt nlocs
    cdef PetscInt *locs
    cdef PetscVec *vecs
    cdef object locs_mem
    cdef object vecs_mem
    cdef object access

    def __cinit__(self, DM dm, Vec gvec, locs=None):
        self.dm = dm.dm
        CHKERR( PetscINCREF(<PetscObject*>&self.dm) )
        self.gvec = gvec.vec
        CHKERR( PetscINCREF(<PetscObject*>&self.gvec) )
        if locs is None:
            CHKERR( DMCompositeGetNumberDM(self.dm, &self.nlocs) )
            locs = arange(0, <long>self.nlocs, 1)
        self.locs_mem = iarray_i(locs, &self.nlocs, &self.locs)
        self.vecs_mem = oarray_p(empty_p(self.nlocs), NULL, <void**>&self.vecs)
        self.access   = None

    def __dealloc__(self):
        CHKERR( DMDestroy(&self.dm) )
        CHKERR( VecDestroy(&self.gvec) )

    def __enter__(self):
        cdef Py_ssize_t i, n = self.nlocs
        CHKERR( DMCompositeGetAccessArray(self.dm, self.gvec, self.nlocs, self.locs, self.vecs) )
        self.access = [ref_Vec(self.vecs[i]) for i from 0 <= i < n]
        return tuple(self.access)

    def __exit__(self, *exc):
        cdef Py_ssize_t i, n = self.nlocs
        for i from 0 <= i < n: (<Vec>self.access[i]).vec = NULL
        CHKERR( DMCompositeRestoreAccessArray(self.dm, self.gvec, self.nlocs, self.locs, self.vecs) )
        self.access   = None
