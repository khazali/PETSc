static const char help[] = "Integrate chemistry using PyJac.\n";

/*
    Pyjac require knowing the chemistry and thermo data at compile time so must be compiled with
    make CHEMFILE=chemistryfile THERMFILE=thermofile expyjac

*/
#include <petscts.h>
#include <dydt.h>
#include <jacob.h>
#include <mass_mole.h>

const char * const Species[] = {"Tempature",
#include "out/species.h"
                                0};

PetscErrorCode FindSpecies(const char *sp,PetscInt *cnt)
{
  PetscBool      found;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *cnt = 0;
  while (Species[1 + *cnt]) {
    ierr = PetscStrcasecmp(Species[*cnt],sp,&found);CHKERRQ(ierr);
    if (found) PetscFunctionReturn(0);
    *cnt += 1;
  }
  ierr = PetscStrcasecmp(Species[*cnt],sp,&found);CHKERRQ(ierr);
  if (found) PetscFunctionReturn(0);
  SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER, "Unable to find species %s\n",sp);
  PetscFunctionReturn(0);
}

typedef struct _User *User;
struct _User {
  PetscReal pressure;
  int       Nspec;
  PetscReal Tini;
  double    *pyjacwork;
  double    *Jdense;        /* Dense array workspace where pyJac computes the Jacobian */ 
  PetscInt  *rows;
  char      **snames;
};


static PetscErrorCode PrintSpecies(User,Vec);
static PetscErrorCode MassFractionToMoleFraction(User,Vec,Vec*);
static PetscErrorCode MoleFractionToMassFraction(User,Vec,Vec*);
static PetscErrorCode FormRHSFunction(TS,PetscReal,Vec,Vec,void*);
static PetscErrorCode FormRHSJacobian(TS,PetscReal,Vec,Mat,Mat,void*);
static PetscErrorCode FormInitialSolution(TS,Vec,void*);
static PetscErrorCode ComputeMassConservation(Vec,PetscReal*,void*);
static PetscErrorCode MonitorMassConservation(TS,PetscInt,PetscReal,Vec,void*);
static PetscErrorCode MonitorTempature(TS,PetscInt,PetscReal,Vec,void*);


int main(int argc,char **argv)
{
  TS                ts;         /* time integrator */
  TSAdapt           adapt;
  Vec               X,lambda;          /* solution vector */
  Mat               J;          /* Jacobian matrix */
  PetscInt          steps;
  PetscErrorCode    ierr;
  PetscReal         ftime,dt;
  struct _User      user;       /* user-defined work context */
  TSConvergedReason reason;
  TSTrajectory      tj;
  PetscBool         flg = PETSC_FALSE,tflg = PETSC_FALSE;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_SELF,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Chemistry solver options","");CHKERRQ(ierr);
  user.pressure = 1.01325e5;    /* Pascal */
  ierr = PetscOptionsReal("-pressure","Pressure of reaction [Pa]","",user.pressure,&user.pressure,NULL);CHKERRQ(ierr);
  user.Tini = 1000;             /* Kelvin */
  ierr = PetscOptionsReal("-Tini","Initial temperature [K]","",user.Tini,&user.Tini,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-monitor_mass","Monitor the total mass at each timestep","",flg,&flg,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-monitor_temp","Monitor the tempature each timestep","",tflg,&tflg,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  user.Nspec = NSP;
  user.snames = (char**) Species;

  ierr = PetscMalloc3(user.Nspec,&user.pyjacwork,PetscSqr(user.Nspec),&user.Jdense,user.Nspec,&user.rows);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,user.Nspec,&X);CHKERRQ(ierr);

  ierr = MatCreateSeqDense(PETSC_COMM_SELF,user.Nspec,user.Nspec,NULL,&J);CHKERRQ(ierr);
  ierr = MatSetFromOptions(J);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSARKIMEX);CHKERRQ(ierr);
  ierr = TSARKIMEXSetFullyImplicit(ts,PETSC_TRUE);CHKERRQ(ierr);
  ierr = TSARKIMEXSetType(ts,TSARKIMEX4);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,FormRHSFunction,&user);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,J,J,FormRHSJacobian,&user);CHKERRQ(ierr);

  if (flg) {
    ierr = TSMonitorSet(ts,MonitorMassConservation,NULL,NULL);CHKERRQ(ierr);
  }
  if (tflg) {
    ierr = TSMonitorSet(ts,MonitorTempature,&user,NULL);CHKERRQ(ierr);
  }

  ftime = 1.0;
  ierr = TSSetMaxTime(ts,ftime);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = FormInitialSolution(ts,X,&user);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,X);CHKERRQ(ierr);
  dt   = 1e-10;                 /* Initial time step */
  ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
  ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
  ierr = TSAdaptSetStepLimits(adapt,1e-12,1e-4);CHKERRQ(ierr); /* Also available with -ts_adapt_dt_min/-ts_adapt_dt_max */
  ierr = TSSetMaxSNESFailures(ts,-1);CHKERRQ(ierr);            /* Retry step an unlimited number of times */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set final conditions for sensitivities
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDuplicate(X,&lambda);CHKERRQ(ierr);
  ierr = TSSetCostGradients(ts,1,&lambda,NULL);CHKERRQ(ierr);
  ierr = VecSetValue(lambda,0,1.0,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(lambda);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(lambda);CHKERRQ(ierr);

  ierr = TSGetTrajectory(ts,&tj);CHKERRQ(ierr);
  if (tj) {
    ierr = TSTrajectorySetVariableNames(tj,(const char * const *)user.snames);CHKERRQ(ierr);
    ierr = TSTrajectorySetTransform(tj,(PetscErrorCode (*)(void*,Vec,Vec*))MassFractionToMoleFraction,NULL,&user);CHKERRQ(ierr);
  }


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Pass information to graphical monitoring routine
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSMonitorLGSetVariableNames(ts,(const char * const *)user.snames);CHKERRQ(ierr);
  ierr = TSMonitorLGSetTransform(ts,(PetscErrorCode (*)(void*,Vec,Vec*))MassFractionToMoleFraction,NULL,&user);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve ODE
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,X);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&steps);CHKERRQ(ierr);
  ierr = TSGetConvergedReason(ts,&reason);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%s at time %g after %D steps\n",TSConvergedReasons[reason],(double)ftime,steps);CHKERRQ(ierr);

  /* {
    Vec                max;
    PetscInt           i;
    const PetscReal    *bmax;

    ierr = TSMonitorEnvelopeGetBounds(ts,&max,NULL);CHKERRQ(ierr);
    if (max) {
      ierr = VecGetArrayRead(max,&bmax);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF,"Species - maximum mass fraction\n");CHKERRQ(ierr);
      for (i=1; i<user.Nspec; i++) {
        if (bmax[i] > .01) {ierr = PetscPrintf(PETSC_COMM_SELF,"%s %g\n",user.snames[i],(double)bmax[i]);CHKERRQ(ierr);}
      }
      ierr = VecRestoreArrayRead(max,&bmax);CHKERRQ(ierr);
    }
  }

  Vec y;
  ierr = MassFractionToMoleFraction(&user,X,&y);CHKERRQ(ierr);
  ierr = PrintSpecies(&user,y);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr); */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&lambda);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFree3(user.pyjacwork,user.Jdense,user.rows);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

static PetscErrorCode FormRHSFunction(TS ts,PetscReal t,Vec X,Vec F,void *ptr)
{
  User              user = (User)ptr;
  PetscErrorCode    ierr;
  PetscScalar       *f;
  const PetscScalar *x;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);

  ierr = PetscMemcpy(user->pyjacwork,x,(user->Nspec)*sizeof(x[0]));CHKERRQ(ierr);
  user->pyjacwork[0] *= user->Tini; /* Dimensionalize */
  dydt(t, user->pressure,user->pyjacwork,f);
  f[0] /= user->Tini;           /* Non-dimensionalize */

  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormRHSJacobian(TS ts,PetscReal t,Vec X,Mat Amat,Mat Pmat,void *ptr)
{
  User              user = (User)ptr;
  PetscErrorCode    ierr;
  const PetscScalar *x;
  PetscInt          M = user->Nspec,i;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = PetscMemcpy(user->pyjacwork,x,M*sizeof(x[0]));CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  user->pyjacwork[0] *= user->Tini;  /* Dimensionalize temperature (first row) because that is what Tchem wants */
  eval_jacob (t, user->pressure,user->pyjacwork,user->Jdense);
  for (i=0; i<M; i++) user->Jdense[i + 0*M] /= user->Tini; /* Non-dimensionalize first column */
  for (i=0; i<M; i++) user->Jdense[0 + i*M] /= user->Tini; /* Non-dimensionalize first row */
  for (i=0; i<M; i++) user->rows[i] = i;
  ierr = MatSetOption(Pmat,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatSetOption(Pmat,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatZeroEntries(Pmat);CHKERRQ(ierr);
  ierr = MatSetValues(Pmat,M,user->rows,M,user->rows,user->Jdense,INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(Pmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Pmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (Amat != Pmat) {
    ierr = MatAssemblyBegin(Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FormInitialSolution(TS ts,Vec X,void *ctx)
{
  PetscScalar    *x;
  PetscErrorCode ierr;
  PetscInt       i;
  Vec            y;
  const PetscInt maxspecies = 10;
  PetscInt       smax = maxspecies,mmax = maxspecies;
  char           *names[maxspecies];
  PetscReal      molefracs[maxspecies],sum;
  PetscBool      flg;

  PetscFunctionBeginUser;
  ierr = VecZeroEntries(X);CHKERRQ(ierr);
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  x[0] = 1.0;  /* Non-dimensionalized by user->Tini */

  ierr = PetscOptionsGetStringArray(NULL,NULL,"-initial_species",names,&smax,&flg);CHKERRQ(ierr);
  if (smax < 2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Must provide at least two initial species");
  ierr = PetscOptionsGetRealArray(NULL,NULL,"-initial_mole",molefracs,&mmax,&flg);CHKERRQ(ierr);
  if (smax != mmax) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Must provide same number of initial species %D as initial moles %D",smax,mmax);
  sum = 0;
  for (i=0; i<smax; i++) sum += molefracs[i];
  for (i=0; i<smax; i++) molefracs[i] = molefracs[i]/sum;
  for (i=0; i<smax; i++) {
    PetscInt ispec;
    ierr = FindSpecies(names[i],&ispec);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"Species %d: %s %g\n",i,names[i],molefracs[i]);CHKERRQ(ierr);
    x[ispec] = molefracs[i];
  }
  for (i=0; i<smax; i++) {
    ierr = PetscFree(names[i]);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  /* PrintSpecies((User)ctx,X);CHKERRQ(ierr); */
  ierr = MoleFractionToMassFraction((User)ctx,X,&y);CHKERRQ(ierr);
  ierr = VecCopy(y,X);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Converts the input vector which is in mass fractions (used by tchem) to mole fractions
*/
PetscErrorCode MassFractionToMoleFraction(User user,Vec massf,Vec *molef)
{
  PetscErrorCode    ierr;
  PetscScalar       *mof;
  const PetscScalar *maf;

  PetscFunctionBegin;
  ierr = VecDuplicate(massf,molef);CHKERRQ(ierr);
  ierr = VecGetArrayRead(massf,&maf);CHKERRQ(ierr);
  ierr = VecGetArray(*molef,&mof);CHKERRQ(ierr);
  mof[0] = maf[0]; /* copy over temperature */
  mass2mole(maf+1,mof+1);
  ierr = VecRestoreArray(*molef,&mof);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(massf,&maf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Converts the input vector which is in mole fractions to mass fractions (used by tchem)
*/
PetscErrorCode MoleFractionToMassFraction(User user,Vec molef,Vec *massf)
{
  PetscErrorCode    ierr;
  const PetscScalar *mof;
  PetscScalar       *maf;

  PetscFunctionBegin;
  ierr = VecDuplicate(molef,massf);CHKERRQ(ierr);
  ierr = VecGetArrayRead(molef,&mof);CHKERRQ(ierr);
  ierr = VecGetArray(*massf,&maf);CHKERRQ(ierr);
  maf[0] = mof[0]; /* copy over temperature */
  mole2mass(mof+1,maf+1);
  ierr = VecRestoreArrayRead(molef,&mof);CHKERRQ(ierr);
  ierr = VecRestoreArray(*massf,&maf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeMassConservation(Vec x,PetscReal *mass,void* ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSum(x,mass);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MonitorMassConservation(TS ts,PetscInt step,PetscReal time,Vec x,void* ctx)
{
  const PetscScalar  *T;
  PetscReal          mass;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = ComputeMassConservation(x,&mass,ctx);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&T);CHKERRQ(ierr);
  mass -= PetscAbsScalar(T[0]);
  ierr = VecRestoreArrayRead(x,&T);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Timestep %D time %g percent mass lost or gained %g\n",step,(double)time,(double)100.*(1.0 - mass));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MonitorTempature(TS ts,PetscInt step,PetscReal time,Vec x,void* ctx)
{
  User               user = (User) ctx;
  const PetscScalar  *T;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(x,&T);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Timestep %D time %g tempature %g\n",step,(double)time,(double)T[0]*user->Tini);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x,&T);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Prints out each species with its name
*/
PETSC_UNUSED PetscErrorCode PrintSpecies(User user,Vec molef)
{
  PetscErrorCode    ierr;
  const PetscScalar *mof;
  PetscInt          i,*idx,n = user->Nspec;

  PetscFunctionBegin;
  ierr = PetscMalloc1(n,&idx);CHKERRQ(ierr);
  for (i=0; i<n;i++) idx[i] = i;
  ierr = VecGetArrayRead(molef,&mof);CHKERRQ(ierr);
  ierr = PetscSortRealWithPermutation(n,mof,idx);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"%6s %g\n",user->snames[idx[n-i-1]],mof[idx[n-i-1]]);CHKERRQ(ierr);
  }
  ierr = PetscFree(idx);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(molef,&mof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
