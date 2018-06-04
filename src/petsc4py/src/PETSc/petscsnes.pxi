cdef extern from * nogil:

    ctypedef char* PetscSNESType "const char*"
    PetscSNESType SNESNEWTONLS
    PetscSNESType SNESNEWTONTR
    #PetscSNESType SNESPYTHON
    PetscSNESType SNESTEST
    PetscSNESType SNESNRICHARDSON
    PetscSNESType SNESKSPONLY
    PetscSNESType SNESVINEWTONRSLS
    PetscSNESType SNESVINEWTONSSLS
    PetscSNESType SNESNGMRES
    PetscSNESType SNESQN
    PetscSNESType SNESSHELL
    PetscSNESType SNESNGS
    PetscSNESType SNESNCG
    PetscSNESType SNESFAS
    PetscSNESType SNESMS
    PetscSNESType SNESNASM
    PetscSNESType SNESANDERSON
    PetscSNESType SNESASPIN
    PetscSNESType SNESCOMPOSITE

    ctypedef enum PetscSNESNormSchedule "SNESNormSchedule":
      SNES_NORM_DEFAULT
      SNES_NORM_NONE
      SNES_NORM_ALWAYS
      SNES_NORM_INITIAL_ONLY
      SNES_NORM_FINAL_ONLY
      SNES_NORM_INITIAL_FINAL_ONLY

    ctypedef enum PetscSNESConvergedReason "SNESConvergedReason":
      # iterating
      SNES_CONVERGED_ITERATING
      # converged
      SNES_CONVERGED_FNORM_ABS
      SNES_CONVERGED_FNORM_RELATIVE
      SNES_CONVERGED_SNORM_RELATIVE
      SNES_CONVERGED_ITS
      SNES_CONVERGED_TR_DELTA
      # diverged
      SNES_DIVERGED_FUNCTION_DOMAIN
      SNES_DIVERGED_FUNCTION_COUNT
      SNES_DIVERGED_LINEAR_SOLVE
      SNES_DIVERGED_FNORM_NAN
      SNES_DIVERGED_MAX_IT
      SNES_DIVERGED_LINE_SEARCH
      SNES_DIVERGED_INNER
      SNES_DIVERGED_LOCAL_MIN


    ctypedef int (*PetscSNESCtxDel)(void*)

    ctypedef int (*PetscSNESInitialGuessFunction)(PetscSNES,
                                                  PetscVec,
                                                  void*) except PETSC_ERR_PYTHON
    ctypedef int (*PetscSNESFunctionFunction)(PetscSNES,
                                              PetscVec,
                                              PetscVec,
                                              void*) except PETSC_ERR_PYTHON

    ctypedef int (*PetscSNESUpdateFunction)(PetscSNES,
                                            PetscInt) except PETSC_ERR_PYTHON

    ctypedef int (*PetscSNESJacobianFunction)(PetscSNES,
                                              PetscVec,
                                              PetscMat,
                                              PetscMat,
                                              void*) except PETSC_ERR_PYTHON

    ctypedef int (*PetscSNESObjectiveFunction)(PetscSNES,
                                               PetscVec,
                                               PetscReal*,
                                               void*) except PETSC_ERR_PYTHON

    ctypedef int (*PetscSNESConvergedFunction)(PetscSNES,
                                               PetscInt,
                                               PetscReal,
                                               PetscReal,
                                               PetscReal,
                                               PetscSNESConvergedReason*,
                                               void*) except PETSC_ERR_PYTHON

    ctypedef int (*PetscSNESMonitorFunction)(PetscSNES,
                                             PetscInt,
                                             PetscReal,
                                             void*) except PETSC_ERR_PYTHON

    int SNESCreate(MPI_Comm,PetscSNES*)
    int SNESDestroy(PetscSNES*)
    int SNESView(PetscSNES,PetscViewer)

    int SNESSetType(PetscSNES,PetscSNESType)
    int SNESGetType(PetscSNES,PetscSNESType*)
    int SNESSetOptionsPrefix(PetscSNES,char[])
    int SNESAppendOptionsPrefix(PetscSNES,char[])
    int SNESGetOptionsPrefix(PetscSNES,char*[])
    int SNESSetFromOptions(PetscSNES)

    int SNESGetKSP(PetscSNES,PetscKSP*)
    int SNESSetKSP(PetscSNES,PetscKSP)

    int SNESGetDM(PetscSNES,PetscDM*)
    int SNESSetDM(PetscSNES,PetscDM)

    # --- FAS ---
    int SNESFASSetInterpolation(PetscSNES,PetscInt,PetscMat)
    int SNESFASGetInterpolation(PetscSNES,PetscInt,PetscMat*)
    int SNESFASSetRestriction(PetscSNES,PetscInt,PetscMat)
    int SNESFASGetRestriction(PetscSNES,PetscInt,PetscMat*)
    int SNESFASSetInjection(PetscSNES,PetscInt,PetscMat)
    int SNESFASGetInjection(PetscSNES,PetscInt,PetscMat*)
    int SNESFASSetRScale(PetscSNES,PetscInt,PetscVec)
    int SNESFASSetLevels(PetscSNES,PetscInt,MPI_Comm[])
    int SNESFASGetLevels(PetscSNES,PetscInt*)
    int SNESFASGetCycleSNES(PetscSNES,PetscInt,PetscSNES*)
    int SNESFASGetCoarseSolve(PetscSNES,PetscSNES*)
    int SNESFASGetSmoother(PetscSNES,PetscInt,PetscSNES*)
    int SNESFASGetSmootherDown(PetscSNES,PetscInt,PetscSNES*)
    int SNESFASGetSmootherUp(PetscSNES,PetscInt,PetscSNES*)

    int SNESGetNPC(PetscSNES,PetscSNES*)
    int SNESHasNPC(PetscSNES,PetscBool*)
    int SNESSetNPC(PetscSNES,PetscSNES)

    int SNESGetRhs(PetscSNES,PetscVec*)
    int SNESGetSolution(PetscSNES,PetscVec*)
    int SNESSetSolution(PetscSNES,PetscVec)
    int SNESGetSolutionUpdate(PetscSNES,PetscVec*)

    int SNESSetInitialGuess"SNESSetComputeInitialGuess"(PetscSNES,PetscSNESInitialGuessFunction,void*)
    int SNESSetFunction(PetscSNES,PetscVec,PetscSNESFunctionFunction,void*)
    int SNESGetFunction(PetscSNES,PetscVec*,PetscSNESFunctionFunction*,void**)
    int SNESSetUpdate(PetscSNES,PetscSNESUpdateFunction)
    int SNESSetJacobian(PetscSNES,PetscMat,PetscMat,PetscSNESJacobianFunction,void*)
    int SNESGetJacobian(PetscSNES,PetscMat*,PetscMat*,PetscSNESJacobianFunction*,void**)
    int SNESSetObjective(PetscSNES,PetscSNESObjectiveFunction,void*)
    int SNESGetObjective(PetscSNES,PetscSNESObjectiveFunction*,void**)

    int SNESComputeFunction(PetscSNES,PetscVec,PetscVec)
    int SNESComputeJacobian(PetscSNES,PetscVec,PetscMat,PetscMat)
    int SNESComputeObjective(PetscSNES,PetscVec,PetscReal*)

    ctypedef int (*PetscSNESNGSFunction)(PetscSNES,
                                         PetscVec,
                                         PetscVec,
                                         void*) except PETSC_ERR_PYTHON
    int SNESSetNGS(PetscSNES,PetscSNESNGSFunction,void*)
    int SNESGetNGS(PetscSNES,PetscSNESNGSFunction*,void**)
    int SNESComputeNGS(PetscSNES,PetscVec,PetscVec)

    int SNESSetNormSchedule(PetscSNES,PetscSNESNormSchedule)
    int SNESGetNormSchedule(PetscSNES,PetscSNESNormSchedule*)

    int SNESSetTolerances(PetscSNES,PetscReal,PetscReal,PetscReal,PetscInt,PetscInt)
    int SNESGetTolerances(PetscSNES,PetscReal*,PetscReal*,PetscReal*,PetscInt*,PetscInt*)

    int SNESSetConvergenceTest(PetscSNES,PetscSNESConvergedFunction,void*,PetscSNESCtxDel*)
    int SNESConvergedDefault(PetscSNES,PetscInt,PetscReal,PetscReal,PetscReal,
                             PetscSNESConvergedReason*,void*) except PETSC_ERR_PYTHON
    int SNESConvergedSkip(PetscSNES,PetscInt,PetscReal,PetscReal,PetscReal,
                          PetscSNESConvergedReason*,void*) except PETSC_ERR_PYTHON
    int SNESSetConvergenceHistory(PetscSNES,PetscReal[],PetscInt[],PetscInt,PetscBool)
    int SNESGetConvergenceHistory(PetscSNES,PetscReal*[],PetscInt*[],PetscInt*)
    int SNESLogConvergenceHistory(PetscSNES,PetscReal,PetscInt)

    int SNESMonitorSet(PetscSNES,PetscSNESMonitorFunction,void*,PetscSNESCtxDel)
    int SNESMonitorCancel(PetscSNES)
    int SNESMonitor(PetscSNES,PetscInt,PetscReal)

    int SNESSetUp(PetscSNES)
    int SNESReset(PetscSNES)
    int SNESSolve(PetscSNES,PetscVec,PetscVec)

    int SNESSetConvergedReason(PetscSNES,PetscSNESConvergedReason)
    int SNESGetConvergedReason(PetscSNES,PetscSNESConvergedReason*)
    int SNESSetIterationNumber(PetscSNES,PetscInt)
    int SNESGetIterationNumber(PetscSNES,PetscInt*)
    int SNESSetFunctionNorm(PetscSNES,PetscReal)
    int SNESGetFunctionNorm(PetscSNES,PetscReal*)
    int SNESGetLinearSolveIterations(PetscSNES,PetscInt*)
    int SNESSetCountersReset(PetscSNES,PetscBool)

    int SNESGetNumberFunctionEvals(PetscSNES,PetscInt*)
    int SNESSetMaxNonlinearStepFailures(PetscSNES,PetscInt)
    int SNESGetMaxNonlinearStepFailures(PetscSNES,PetscInt*)
    int SNESGetNonlinearStepFailures(PetscSNES,PetscInt*)
    int SNESSetMaxLinearSolveFailures(PetscSNES,PetscInt)
    int SNESGetMaxLinearSolveFailures(PetscSNES,PetscInt*)
    int SNESGetLinearSolveFailures(PetscSNES,PetscInt*)

    int SNESKSPSetUseEW(PetscSNES,PetscBool)
    int SNESKSPGetUseEW(PetscSNES,PetscBool*)
    int SNESKSPSetParametersEW(PetscSNES,PetscInt,PetscReal,PetscReal,
                               PetscReal,PetscReal,PetscReal,PetscReal)
    int SNESKSPGetParametersEW(PetscSNES,PetscInt*,PetscReal*,PetscReal*,
                               PetscReal*,PetscReal*,PetscReal*,PetscReal*)

    int SNESVISetVariableBounds(PetscSNES,PetscVec,PetscVec)
    #ctypedef int (*PetscSNESVariableBoundsFunction)(PetscSNES,PetscVec,PetscVec)
    #int SNESVISetComputeVariableBounds(PetscSNES,PetscSNESVariableBoundsFunction)
    int SNESVIGetInactiveSet(PetscSNES, PetscIS*)

    int SNESCompositeGetSNES(PetscSNES,PetscInt,PetscSNES*)
    int SNESCompositeGetNumber(PetscSNES,PetscInt*)
    int SNESNASMGetSNES(PetscSNES,PetscInt,PetscSNES*)
    int SNESNASMGetNumber(PetscSNES,PetscInt*)

cdef extern from "custom.h" nogil:
    int SNESSetUseMFFD(PetscSNES,PetscBool)
    int SNESGetUseMFFD(PetscSNES,PetscBool*)

    int SNESSetUseFDColoring(PetscSNES,PetscBool)
    int SNESGetUseFDColoring(PetscSNES,PetscBool*)

    int SNESConvergenceTestCall(PetscSNES,PetscInt,
                                PetscReal,PetscReal,PetscReal,
                                PetscSNESConvergedReason*)

cdef extern from "libpetsc4py.h":
    PetscSNESType SNESPYTHON
    int SNESPythonSetContext(PetscSNES,void*)
    int SNESPythonGetContext(PetscSNES,void**)
    int SNESPythonSetType(PetscSNES,char[])

# -----------------------------------------------------------------------------

cdef inline SNES ref_SNES(PetscSNES snes):
    cdef SNES ob = <SNES> SNES()
    ob.snes = snes
    PetscINCREF(ob.obj)
    return ob

# -----------------------------------------------------------------------------

cdef int SNES_InitialGuess(
    PetscSNES snes,
    PetscVec  x,
    void*     ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef SNES Snes = ref_SNES(snes)
    cdef Vec  Xvec = ref_Vec(x)
    cdef object context = Snes.get_attr('__initialguess__')
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (initialguess, args, kargs) = context
    initialguess(Snes, Xvec, *args, **kargs)
    return 0

# -----------------------------------------------------------------------------

cdef int SNES_Function(
    PetscSNES snes,
    PetscVec  x,
    PetscVec  f,
    void*     ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef SNES Snes = ref_SNES(snes)
    cdef Vec  Xvec = ref_Vec(x)
    cdef Vec  Fvec = ref_Vec(f)
    cdef object context = Snes.get_attr('__function__')
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (function, args, kargs) = context
    function(Snes, Xvec, Fvec, *args, **kargs)
    return 0

# -----------------------------------------------------------------------------

cdef int SNES_Update(
    PetscSNES snes,
    PetscInt  its,
    ) except PETSC_ERR_PYTHON with gil:
    cdef SNES Snes = ref_SNES(snes)
    cdef object context = Snes.get_attr('__update__')
    assert context is not None and type(context) is tuple # sanity check
    (update, args, kargs) = context
    update(Snes, toInt(its), *args, **kargs)
    return 0

# -----------------------------------------------------------------------------

cdef int SNES_Jacobian(
    PetscSNES snes,
    PetscVec  x,
    PetscMat  J,
    PetscMat  P,
    void*     ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef SNES Snes = ref_SNES(snes)
    cdef Vec  Xvec = ref_Vec(x)
    cdef Mat  Jmat = ref_Mat(J)
    cdef Mat  Pmat = ref_Mat(P)
    cdef object context = Snes.get_attr('__jacobian__')
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (jacobian, args, kargs) = context
    jacobian(Snes, Xvec, Jmat, Pmat, *args, **kargs)
    return 0

# -----------------------------------------------------------------------------

cdef int SNES_Objective(
    PetscSNES  snes,
    PetscVec   x,
    PetscReal *o,
    void*      ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef SNES Snes = ref_SNES(snes)
    cdef Vec  Xvec = ref_Vec(x)
    cdef object context = Snes.get_attr('__objective__')
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (objective, args, kargs) = context
    obj = objective(Snes, Xvec, *args, **kargs)
    o[0] = asReal(obj)
    return 0

# -----------------------------------------------------------------------------

cdef int SNES_NGS(
    PetscSNES snes,
    PetscVec  x,
    PetscVec  b,
    void*     ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef SNES Snes = ref_SNES(snes)
    cdef Vec  Xvec = ref_Vec(x)
    cdef Vec  Bvec = ref_Vec(b)
    cdef object context = Snes.get_attr('__ngs__')
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (ngs, args, kargs) = context
    ngs(Snes, Xvec, Bvec, *args, **kargs)
    return 0

# -----------------------------------------------------------------------------

cdef int SNES_Converged(
    PetscSNES  snes,
    PetscInt   iters,
    PetscReal  xnorm,
    PetscReal  gnorm,
    PetscReal  fnorm,
    PetscSNESConvergedReason *r,
    void*      ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef SNES Snes = ref_SNES(snes)
    cdef object it = toInt(iters)
    cdef object xn = toReal(xnorm)
    cdef object gn = toReal(gnorm)
    cdef object fn = toReal(fnorm)
    cdef object context = Snes.get_attr('__converged__')
    if context is None and ctx != NULL: context = <object>ctx
    assert context is not None and type(context) is tuple # sanity check
    (converged, args, kargs) = context
    reason = converged(Snes, it, (xn, gn, fn), *args, **kargs)
    if   reason is None:  r[0] = SNES_CONVERGED_ITERATING
    elif reason is False: r[0] = SNES_CONVERGED_ITERATING
    elif reason is True:  r[0] = SNES_CONVERGED_ITS # XXX ?
    else:                 r[0] = reason
    return 0

# -----------------------------------------------------------------------------

cdef int SNES_Monitor(
    PetscSNES  snes,
    PetscInt   iters,
    PetscReal  rnorm,
    void*      ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef SNES Snes = ref_SNES(snes)
    cdef object monitorlist = Snes.get_attr('__monitor__')
    if monitorlist is None: return 0
    cdef object it = toInt(iters)
    cdef object rn = toReal(rnorm)
    for (monitor, args, kargs) in monitorlist:
        monitor(Snes, it, rn, *args, **kargs)
    return 0

# -----------------------------------------------------------------------------
