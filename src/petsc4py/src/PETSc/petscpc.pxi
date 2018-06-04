cdef extern from * nogil:

    ctypedef char* PetscPCType "const char*"
    PetscPCType PCNONE
    PetscPCType PCJACOBI
    PetscPCType PCSOR
    PetscPCType PCLU
    PetscPCType PCSHELL
    PetscPCType PCBJACOBI
    PetscPCType PCMG
    PetscPCType PCEISENSTAT
    PetscPCType PCILU
    PetscPCType PCICC
    PetscPCType PCASM
    PetscPCType PCGASM
    PetscPCType PCKSP
    PetscPCType PCCOMPOSITE
    PetscPCType PCREDUNDANT
    PetscPCType PCSPAI
    PetscPCType PCNN
    PetscPCType PCCHOLESKY
    PetscPCType PCPBJACOBI
    PetscPCType PCMAT
    PetscPCType PCHYPRE
    PetscPCType PCPARMS
    PetscPCType PCFIELDSPLIT
    PetscPCType PCTFS
    PetscPCType PCML
    PetscPCType PCGALERKIN
    PetscPCType PCEXOTIC
    PetscPCType PCCP
    PetscPCType PCBFBT
    PetscPCType PCLSC
    #PetscPCType PCPYTHON
    PetscPCType PCPFMG
    PetscPCType PCSYSPFMG
    PetscPCType PCREDISTRIBUTE
    PetscPCType PCSVD
    PetscPCType PCGAMG
    PetscPCType PCCHOWILUVIENNACL
    PetscPCType PCROWSCALINGVIENNACL
    PetscPCType PCSAVIENNACL
    PetscPCType PCBDDC
    PetscPCType PCKACZMARZ
    PetscPCType PCTELESCOPE

    ctypedef enum PetscPCSide "PCSide":
        PC_SIDE_DEFAULT
        PC_LEFT
        PC_RIGHT
        PC_SYMMETRIC

    ctypedef enum PetscPCASMType "PCASMType":
        PC_ASM_BASIC
        PC_ASM_RESTRICT
        PC_ASM_INTERPOLATE
        PC_ASM_NONE

    ctypedef enum PetscPCGASMType "PCGASMType":
        PC_GASM_BASIC
        PC_GASM_RESTRICT
        PC_GASM_INTERPOLATE
        PC_GASM_NONE

    ctypedef enum PetscPCMGType "PCMGType":
        PC_MG_MULTIPLICATIVE
        PC_MG_ADDITIVE
        PC_MG_FULL
        PC_MG_KASKADE

    ctypedef enum PetscPCMGCycleType "PCMGCycleType":
        PC_MG_CYCLE_V
        PC_MG_CYCLE_W

    ctypedef char* PetscPCGAMGType "const char*"
    PetscPCGAMGType PCGAMGAGG
    PetscPCGAMGType PCGAMGGEO
    PetscPCGAMGType PCGAMGCLASSICAL

    ctypedef char* PetscPCHYPREType "const char*"

    ctypedef enum PetscPCCompositeType "PCCompositeType":
        PC_COMPOSITE_ADDITIVE
        PC_COMPOSITE_MULTIPLICATIVE
        PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE
        PC_COMPOSITE_SPECIAL
        PC_COMPOSITE_SCHUR

    ctypedef enum PetscPCFieldSplitSchurPreType "PCFieldSplitSchurPreType":
        PC_FIELDSPLIT_SCHUR_PRE_SELF
        PC_FIELDSPLIT_SCHUR_PRE_SELFP
        PC_FIELDSPLIT_SCHUR_PRE_A11
        PC_FIELDSPLIT_SCHUR_PRE_USER
        PC_FIELDSPLIT_SCHUR_PRE_FULL

    ctypedef enum PetscPCFieldSplitSchurFactType "PCFieldSplitSchurFactType":
        PC_FIELDSPLIT_SCHUR_FACT_DIAG
        PC_FIELDSPLIT_SCHUR_FACT_LOWER
        PC_FIELDSPLIT_SCHUR_FACT_UPPER
        PC_FIELDSPLIT_SCHUR_FACT_FULL

    int PCCreate(MPI_Comm,PetscPC*)
    int PCDestroy(PetscPC*)
    int PCView(PetscPC,PetscViewer)

    int PCSetType(PetscPC,PetscPCType)
    int PCGetType(PetscPC,PetscPCType*)

    int PCSetOptionsPrefix(PetscPC,char[])
    int PCAppendOptionsPrefix(PetscPC,char[])
    int PCGetOptionsPrefix(PetscPC,char*[])
    int PCSetFromOptions(PetscPC)

    int PCSetUp(PetscPC)
    int PCReset(PetscPC)
    int PCSetUpOnBlocks(PetscPC)

    int PCApply(PetscPC,PetscVec,PetscVec)
    int PCApplyTranspose(PetscPC,PetscVec,PetscVec)
    int PCApplySymmetricLeft(PetscPC,PetscVec,PetscVec)
    int PCApplySymmetricRight(PetscPC,PetscVec,PetscVec)
    int PCApplyRichardson(PetscPC,PetscVec,PetscVec,PetscVec,PetscReal,PetscReal,PetscReal,PetscInt)
    int PCApplyBAorAB(PetscPC,PetscPCSide,PetscVec,PetscVec,PetscVec)
    int PCApplyBAorABTranspose(PetscPC,PetscPCSide,PetscVec,PetscVec,PetscVec)

    #int PCApplyTransposeExists(PetscPC,PetscBool*)
    #int PCApplyRichardsonExists(PetscPC,PetscBool*)

    int PCGetDM(PetscPC,PetscDM*)
    int PCSetDM(PetscPC,PetscDM)

    int PCSetOperators(PetscPC,PetscMat,PetscMat)
    int PCGetOperators(PetscPC,PetscMat*,PetscMat*)
    int PCGetOperatorsSet(PetscPC,PetscBool*,PetscBool*)
    int PCSetCoordinates(PetscPC,PetscInt,PetscInt,PetscReal[])
    int PCSetUseAmat(PetscPC,PetscBool)

    int PCComputeExplicitOperator(PetscPC,PetscMat*)

    int PCDiagonalScale(PetscPC,PetscBool*)
    int PCDiagonalScaleLeft(PetscPC,PetscVec,PetscVec)
    int PCDiagonalScaleRight(PetscPC,PetscVec,PetscVec)
    int PCDiagonalScaleSet(PetscPC,PetscVec)

    int PCASMSetType(PetscPC,PetscPCASMType)
    int PCASMSetOverlap(PetscPC,PetscInt)
    int PCASMSetLocalSubdomains(PetscPC,PetscInt,PetscIS[],PetscIS[])
    int PCASMSetTotalSubdomains(PetscPC,PetscInt,PetscIS[],PetscIS[])
    int PCASMGetSubKSP(PetscPC,PetscInt*,PetscInt*,PetscKSP*[])

    int PCGASMSetType(PetscPC,PetscPCGASMType)
    int PCGASMSetOverlap(PetscPC,PetscInt)

    int PCGAMGSetType(PetscPC,PetscPCGAMGType)
    int PCGAMGSetNlevels(PetscPC,PetscInt)
    int PCGAMGSetNSmooths(PetscPC,PetscInt)

    int PCHYPREGetType(PetscPC,PetscPCHYPREType*)
    int PCHYPRESetType(PetscPC,PetscPCHYPREType)
    int PCHYPRESetDiscreteCurl(PetscPC,PetscMat);
    int PCHYPRESetDiscreteGradient(PetscPC,PetscMat);
    int PCHYPRESetAlphaPoissonMatrix(PetscPC,PetscMat);
    int PCHYPRESetBetaPoissonMatrix(PetscPC,PetscMat);
    int PCHYPRESetEdgeConstantVectors(PetscPC,PetscVec,PetscVec,PetscVec);

    int PCFactorGetMatrix(PetscPC,PetscMat*)
    int PCFactorSetZeroPivot(PetscPC,PetscReal)
    int PCFactorSetShiftType(PetscPC,PetscMatFactorShiftType)
    int PCFactorSetShiftAmount(PetscPC,PetscReal)
    int PCFactorSetMatSolverType(PetscPC,PetscMatSolverType)
    int PCFactorGetMatSolverType(PetscPC,PetscMatSolverType*)
    int PCFactorSetUpMatSolverType(PetscPC)
    int PCFactorSetFill(PetscPC,PetscReal)
    int PCFactorSetColumnPivot(PetscPC,PetscReal)
    int PCFactorReorderForNonzeroDiagonal(PetscPC,PetscReal)
    int PCFactorSetMatOrderingType(PetscPC,PetscMatOrderingType)
    int PCFactorSetReuseOrdering(PetscPC,PetscBool )
    int PCFactorSetReuseFill(PetscPC,PetscBool )
    int PCFactorSetUseInPlace(PetscPC)
    int PCFactorSetAllowDiagonalFill(PetscPC)
    int PCFactorSetPivotInBlocks(PetscPC,PetscBool )
    int PCFactorSetLevels(PetscPC,PetscInt)
    int PCFactorSetDropTolerance(PetscPC,PetscReal,PetscReal,PetscInt)

    int PCFieldSplitSetType(PetscPC,PetscPCCompositeType)
    int PCFieldSplitSetBlockSize(PetscPC,PetscInt)
    int PCFieldSplitSetFields(PetscPC,char[],PetscInt,PetscInt*,PetscInt*)
    int PCFieldSplitSetIS(PetscPC,char[],PetscIS)
    int PCFieldSplitGetSubKSP(PetscPC,PetscInt*,PetscKSP*[])
    int PCFieldSplitSetSchurPre(PetscPC,PetscPCFieldSplitSchurPreType,PetscMat)
    int PCFieldSplitSetSchurFactType(PetscPC,PetscPCFieldSplitSchurFactType)
    #int PCFieldSplitGetSchurBlocks(PetscPC,PetscMat*,PetscMat*,PetscMat*,PetscMat*)

    int PCCompositeSetType(PetscPC,PetscPCCompositeType)
    int PCCompositeGetPC(PetscPC,PetscInt,PetscPC*)
    int PCCompositeAddPC(PetscPC,PetscPCType)

    int PCKSPGetKSP(PetscPC,PetscKSP*)

    int PCSetReusePreconditioner(PetscPC,PetscBool)

    # --- MG ---
    int PCMGSetType(PetscPC,PetscPCMGType)
    int PCMGGetType(PetscPC,PetscPCMGType*)
    int PCMGSetInterpolation(PetscPC,PetscInt,PetscMat)
    int PCMGGetInterpolation(PetscPC,PetscInt,PetscMat*)
    int PCMGSetRestriction(PetscPC,PetscInt,PetscMat)
    int PCMGGetRestriction(PetscPC,PetscInt,PetscMat*)
    int PCMGSetRScale(PetscPC,PetscInt,PetscVec)
    int PCMGGetRScale(PetscPC,PetscInt,PetscVec*)
    int PCMGGetSmoother(PetscPC,PetscInt,PetscKSP*)
    int PCMGGetSmootherUp(PetscPC,PetscInt,PetscKSP*)
    int PCMGGetSmootherDown(PetscPC,PetscInt,PetscKSP*)
    int PCMGGetCoarseSolve(PetscPC,PetscKSP*)
    int PCMGSetRhs(PetscPC,PetscInt,PetscVec)
    int PCMGSetX(PetscPC,PetscInt,PetscVec)
    int PCMGSetR(PetscPC,PetscInt,PetscVec)
    int PCMGGetLevels(PetscPC,PetscInt*)
    int PCMGSetCycleType(PetscPC,PetscPCMGCycleType)
    int PCMGSetCycleTypeOnLevel(PetscPC,PetscInt,PetscPCMGCycleType)
    int PCBDDCSetDiscreteGradient(PetscPC,PetscMat,PetscInt,PetscInt,PetscBool,PetscBool)
    int PCBDDCSetDivergenceMat(PetscPC,PetscMat,PetscBool,PetscIS)
    int PCBDDCSetChangeOfBasisMat(PetscPC,PetscMat,PetscBool)
    int PCBDDCSetPrimalVerticesIS(PetscPC,PetscIS)
    int PCBDDCSetPrimalVerticesLocalIS(PetscPC,PetscIS)
    int PCBDDCSetCoarseningRatio(PetscPC,PetscInt)
    int PCBDDCSetLevels(PetscPC,PetscInt)
    int PCBDDCSetDirichletBoundaries(PetscPC,PetscIS)
    int PCBDDCSetDirichletBoundariesLocal(PetscPC,PetscIS)
    int PCBDDCSetNeumannBoundaries(PetscPC,PetscIS)
    int PCBDDCSetNeumannBoundariesLocal(PetscPC,PetscIS)
    int PCBDDCSetDofsSplitting(PetscPC,PetscInt,PetscIS[])
    int PCBDDCSetDofsSplittingLocal(PetscPC,PetscInt,PetscIS[])

# --------------------------------------------------------------------

cdef extern from "libpetsc4py.h":
    PetscPCType PCPYTHON
    int PCPythonSetContext(PetscPC,void*)
    int PCPythonGetContext(PetscPC,void**)
    int PCPythonSetType(PetscPC,char[])

# --------------------------------------------------------------------
