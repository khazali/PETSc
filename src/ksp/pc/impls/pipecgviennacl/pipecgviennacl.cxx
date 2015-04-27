
/*  -------------------------------------------------------------------- */

/*
   Include files needed for the ViennaCL pipelined CG solver (implemented as a preconditioner):
*/

#include <petsc/private/pcimpl.h>   /*I "petscpc.h" I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/vec/vec/impls/dvecimpl.h>
#include <../src/mat/impls/aij/seq/seqviennacl/viennaclmatimpl.h>

#include <viennacl/vector.hpp>
#include <viennacl/matrix.hpp>
#include <viennacl/linalg/cg.hpp>

/*
   Private context (data structure) for the ViennaCL pipelined CG preconditioner.
 */
typedef struct {
  PetscInt   maxits;
  PetscReal  rtol;
  PetscBool  monitorverbose;
  ViennaCLAIJMatrix *mat;
} PC_PipeCGViennaCL;

#undef __FUNCT__
#define __FUNCT__ "PCPipeCGViennaCLSetTolerance_PipeCGViennaCL"
static PetscErrorCode PCPipeCGViennaCLSetTolerance_PipeCGViennaCL(PC pc,PetscReal rtol)
{
  PC_PipeCGViennaCL *cg = (PC_PipeCGViennaCL*)pc->data;

  PetscFunctionBegin;
  cg->rtol = rtol;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCPipeCGViennaCLSetUseVerboseMonitor_PipeCGViennaCL"
static PetscErrorCode PCPipeCGViennaCLSetUseVerboseMonitor_PipeCGViennaCL(PC pc, PetscBool useverbose)
{
  PC_PipeCGViennaCL *cg = (PC_PipeCGViennaCL*)pc->data;

  PetscFunctionBegin;
  cg->monitorverbose = useverbose;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCPipeCGViennaCLSetUseVerboseMonitor"
PetscErrorCode PCPipeCGViennaCLSetUseVerboseMonitor(PC pc, PetscBool useverbose)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID,1);
  ierr = PetscTryMethod(pc, "PCPipeCGViennaCLSetUseVerboseMonitors_C",(PC,PetscBool),(pc,useverbose));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCPipeCGViennaCLSetIterations_PipeCGViennaCL"
static PetscErrorCode PCPipeCGViennaCLSetIterations_PipeCGViennaCL(PC pc, PetscInt its)
{
  PC_PipeCGViennaCL *cg = (PC_PipeCGViennaCL*)pc->data;

  PetscFunctionBegin;
  cg->maxits = its;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCPipeCGViennaCLSetIterations"
PetscErrorCode PCPipeCGViennaCLSetITerations(PC pc, PetscInt its)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc, "PCPipeCGViennaCLSetIterations_C",(PC,PetscInt),(pc,its));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCPipeCGViennaCLSetTolerance"
PetscErrorCode PCPipeCGViennaCLSetTolerance(PC pc, PetscReal rtol)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc, "PCPipeCGViennaCLSetTolerance_C",(PC,PetscReal),(pc,rtol));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCSetUp_PipeCGViennaCL - Prepares for the use of the ViennaCL pipelined CG preconditioner
                    by setting data structures and options.

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCSetUp()

   Notes:
   The interface routine PCSetUp() is not usually called directly by
   the user, but instead is called by PCApply() if necessary.
*/
#undef __FUNCT__
#define __FUNCT__ "PCSetUp_PipeCGViennaCL"
static PetscErrorCode PCSetUp_PipeCGViennaCL(PC pc)
{
  PC_PipeCGViennaCL  *cg = (PC_PipeCGViennaCL*)pc->data;
  PetscBool          flg   = PETSC_FALSE;
  Mat_SeqAIJViennaCL *gpustruct;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)pc->pmat,MATSEQAIJVIENNACL,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Currently only handles ViennaCL matrices");
  try {
    ierr      = MatViennaCLCopyToGPU(pc->pmat);CHKERRQ(ierr);
    gpustruct = (Mat_SeqAIJViennaCL*)(pc->pmat->spptr);
    cg->mat   = (ViennaCLAIJMatrix*)gpustruct->mat;
  } catch(char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s",ex);
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCApply_PipeCGViennaCL - Applies the pipelined CG preconditioner in ViennaCL to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  x - input vector

   Output Parameter:
.  y - output vector

   Application Interface Routine: PCApply()
 */
#undef __FUNCT__
#define __FUNCT__ "PCApply_PipeCGViennaCL"
static PetscErrorCode PCApply_PipeCGViennaCL(PC pc,Vec x,Vec y)
{
  PC_PipeCGViennaCL     *cg = (PC_PipeCGViennaCL*)pc->data;
  PetscErrorCode        ierr;
  PetscBool             flg1,flg2;
  ViennaCLVector const *x_vcl=NULL;
  ViennaCLVector        *y_vcl=NULL;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)x,VECSEQVIENNACL,&flg1);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)y,VECSEQVIENNACL,&flg2);CHKERRQ(ierr);
  if (!(flg1 && flg2)) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP, "Currently only handles ViennaCL vectors");
  if (!cg->mat) {
    ierr = PCSetUp_PipeCGViennaCL(pc);CHKERRQ(ierr);
  }
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  ierr = VecViennaCLGetArrayRead(x,&x_vcl);CHKERRQ(ierr);
  ierr = VecViennaCLGetArrayWrite(y,&y_vcl);CHKERRQ(ierr);
  try {
    viennacl::linalg::cg_tag solver_tag(cg->rtol, cg->maxits);
    *y_vcl = viennacl::linalg::solve(*cg->mat,*x_vcl, solver_tag);
    if (cg->monitorverbose) {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "PipeCGViennaCL iterations: %d\n", solver_tag.iters());CHKERRQ(ierr);
    }
  } catch(char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex);
  }
  ierr = VecViennaCLRestoreArrayRead(x,&x_vcl);CHKERRQ(ierr);
  ierr = VecViennaCLRestoreArrayWrite(y,&y_vcl);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*
   PCDestroy_PipeCGViennaCL - Destroys the private context for the PipeCGViennaCL preconditioner
   that was created with PCCreate_PipeCGViennaCL().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCDestroy()
*/
#undef __FUNCT__
#define __FUNCT__ "PCDestroy_PipeCGViennaCL"
static PetscErrorCode PCDestroy_PipeCGViennaCL(PC pc)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /*
      Free the private data structure that was hanging off the PC
  */
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_PipeCGViennaCL"
static PetscErrorCode PCSetFromOptions_PipeCGViennaCL(PetscOptions *PetscOptionsObject,PC pc)
{
  PC_PipeCGViennaCL *cg = (PC_PipeCGViennaCL*)pc->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"PipeCGViennaCL options");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pc_pipecgviennacl_rtol","relative tolerance for PipeCGViennaCL preconditioner","PCPipeCGViennaCLSetTolerance",cg->rtol,&cg->rtol,0);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_pipecgviennacl_max_it","maximum iterations for PipeCGViennaCL preconditioner","PCPipeCGViennaCLSetIterations",cg->maxits,&cg->maxits,0);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_pipecgviennacl_monitor_verbose","Print information about GPU PipeCGViennaCL iterations","PCPipeCGViennaCLSetUseVerboseMonitor",cg->monitorverbose,&cg->monitorverbose,0);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "PCCreate_PipeCGViennaCL"
PETSC_EXTERN PetscErrorCode PCCreate_PipeCGViennaCL(PC pc)
{
  PC_PipeCGViennaCL *cg;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /*
     Creates the private data structure for this preconditioner and
     attach it to the PC object.
   */
  ierr = PetscNewLog(pc,&cg);CHKERRQ(ierr);
  /*
     Set reasonable default values.
   */
  cg->maxits         = 10000;
  cg->rtol           = 1.e-3;
  cg->monitorverbose = PETSC_FALSE;
  pc->data           = (void*)cg;
  /*
      Set the pointers for the functions that are provided above.
      Now when the user-level routines (such as PCApply(), PCDestroy(), etc.)
      are called, they will automatically call these functions.  Note we
      choose not to provide a couple of these functions since they are
      not needed.
  */
  pc->ops->apply               = PCApply_PipeCGViennaCL;
  pc->ops->applytranspose      = 0;
  pc->ops->setup               = PCSetUp_PipeCGViennaCL;
  pc->ops->destroy             = PCDestroy_PipeCGViennaCL;
  pc->ops->setfromoptions      = PCSetFromOptions_PipeCGViennaCL;
  pc->ops->view                = 0;
  pc->ops->applyrichardson     = 0;
  pc->ops->applysymmetricleft  = 0;
  pc->ops->applysymmetricright = 0;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCPipeCGViennaCLSetTolerance_C",PCPipeCGViennaCLSetTolerance_PipeCGViennaCL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc, "PCPipeCGViennaCLSetIterations_C",PCPipeCGViennaCLSetIterations_PipeCGViennaCL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc, "PCPipeCGViennaCLSetUseVerboseMonitor_C", PCPipeCGViennaCLSetUseVerboseMonitor_PipeCGViennaCL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

