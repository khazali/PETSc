static char help[] = "Poisson Problem in 2d using a spectral solution.\n\
We solve the Poisson problem in a rectangular\n\
domain, using eignefunctions of the Laplacian to discretize it.\n\n\n";

#include <petscsnes.h>
#include <petscdmda.h>

typedef struct {
  PetscInt L;       /* Max wavenumber for testing */
  PetscInt l, m, n; /* Max spectral modes in each dimension */
  PetscInt k;       /* Wavenumber for perturbation */
} SpectralCtx;

/*
  In 2D for Dirichlet conditions, we use exact solution:

    u = x (1 - x) y (1 - y)
    f = -2 x (1 - x) - 2 y (1 - y)

  so that

    -\Delta u + f = 2 y (1 - y) + 2 x (1 - x) - 2 x (1 - x) - 2 y (1 - y) = 0
*/
PetscErrorCode quartic_u_2d(PetscInt dim, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  *u = x[0]*(1.0 - x[0])*x[1]*(1.0 - x[1]);
  return 0;
}

PetscErrorCode quartic_f_2d(PetscInt dim, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  *u = -2.0*x[0]*(1.0 - x[0]) - 2.0*x[1]*(1.0 - x[1]);
  return 0;
}

/*
  The Green function is given by (Jackson, p.89)

G(x, x') = \frac{16}{\pi ab} \sum^\infty_{l, m = 1} \frac{\sin(\frac{l \pi x}{a}) \sin(\frac{l \pi x'}{a}) \sin(\frac{m \pi y}{b}) \sin(\frac{m \pi y'}{b})}{\frac{l^2}{a^2} + \frac{m^2}{b^2}}

For your problem a = b = 1, we flip the sign, and we use a different normalization (no 4 \pi), so we have

G(x, x') = -\frac{4}{\pi^2} \sum^\infty_{l, m = 1} \frac{\sin(l \pi x) \sin(l \pi x') \sin(m \pi y) \sin(m \pi y')}{l^2 + m^2}

We will also need the gradient of the Green function

\nabla_{x'} G(x, x') = / -\frac{4 l \pi}{\pi^2} \sum^\infty_{l, m = 1} \frac{\sin(l \pi x) \cos(l \pi x') \sin(m \pi y) \sin(m \pi y')}{l^2 + m^2} \
                       \ -\frac{4 m \pi}{\pi^2} \sum^\infty_{l, m = 1} \frac{\sin(l \pi x) \sin(l \pi x') \sin(m \pi y) \cos(m \pi y')}{l^2 + m^2} /

*/

/*
  Lets check that our Green function reproduces a single mode solution. In 2D for Dirichlet conditions, we use exact solution:

    u = \sin(j \pi x) \sin(k \pi y)
    f = -\pi^2 (j^2 + k^2) \sin(j \pi x) \sin(k \pi y)

  so that

    -\Delta u + f = \pi^2 (j^2 + k^2) \sin(j \pi x) \sin(k \pi y) - \pi^2 (j^2 + k^2) \sin(j \pi x) \sin(k \pi y) = 0

  and

    \int_\Omega G(\vx, \vx') f(\vx')
    = \frac{4}{\pi^2} \sum^\infty_{l, m = 1} \int^1_0 dx' \int^1_0 dy' \frac{\sin(l \pi x) \sin(l \pi x') \sin(m \pi y) \sin(m \pi y')}{l^2 + m^2} \pi^2 (j^2 + k^2) \sin(j \pi x') \sin(k \pi y')
    = 4 \sum^\infty_{l, m = 1} \sin(l \pi x) \sin(m \pi y) \int^1_0 dx' \int^1_0 dy' \frac{\sin(l \pi x') \sin(m \pi y')}{l^2 + m^2} (j^2 + k^2) \sin(j \pi x') \sin(k \pi y')
    = 4 \sum^\infty_{l, m = 1} \delta_{jl} \delta_{km} \sin(l \pi x) \sin(m \pi y) \frac{1}{4} \frac{j^2 + k^2}{l^2 + m^2}
    = \sin(j \pi x) \sin(k \pi y)
*/

#undef __FUNCT__
#define __FUNCT__ "quartic_sol_2d"
/*
    \int_\Omega G(\vx, \vx') f(\vx')
    = -\frac{4}{\pi^2} \sum^\infty_{l, m = 1} \int^1_0 dx' \int^1_0 dy' \frac{\sin(l \pi x) \sin(l \pi x') \sin(m \pi y) \sin(m \pi y')}{l^2 + m^2} (-2 x' (1 - x') - 2 y' (1 - y'))
    = \frac{8}{\pi^2} \sum^\infty_{l, m = 1} \frac{\sin(l \pi x) \sin(m \pi y)}{l^2 + m^2} (\frac{1 - cos(m\pi)}{m \pi} \int^1_0 dx' \sin(l \pi x') x' (1 - x') + \frac{1-cos(l\pi)}{l \pi} \int^1_0 dy' \sin(m \pi y') y' (1 - y'))
    = \frac{8}{\pi^2} \sum^\infty_{l, m = 1} \frac{\sin(l \pi x) \sin(m \pi y)}{l^2 + m^2} (\frac{1 - cos(m\pi)}{m \pi} \frac{2 (1 - cos(l\pi))}{l^3 \pi^3} + \frac{1-cos(l\pi)}{l \pi} \frac{2 (1-cos(m\pi))}{m^3 \pi^3})
    = \frac{8}{\pi^2} \sum^\infty_{l, m = 1, odd} \frac{\sin(l \pi x) \sin(m \pi y)}{l^2 + m^2} (\frac{8}{m l^3 \pi^4} + \frac{8}{l m^3 \pi^4})
    = \frac{64}{\pi^6} \sum^\infty_{l, m = 1, odd} \sin(l \pi x) \sin(m \pi y) \frac{m^2 + l^2}{l^3 m^3 (l^2 + m^2)}
    = \frac{64}{\pi^6} \sum^\infty_{l, m = 1, odd} \sin(l \pi x) \sin(m \pi y) \frac{1}{l^3 m^3}
*/
static PetscErrorCode quartic_sol_2d(PetscInt dim, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  SpectralCtx    *spc = (SpectralCtx *) ctx;
  const PetscReal fac = 64.0/PetscPowRealInt(PETSC_PI, 6);
  const PetscInt  l   = spc->l;
  const PetscInt  m   = spc->m;
  PetscReal       sum = 0.0;
  PetscInt        j, k;

  PetscFunctionBeginUser;
  for (j = 1; j <= l; j += 2) {
    const PetscReal jcube = PetscPowRealInt(j, 3);

    for (k = 1; k <= m; k += 2) {
      const PetscReal kcube = PetscPowRealInt(k, 3);
      const PetscReal g     = 1.0/(jcube * kcube);

      sum += g*PetscSinReal(j*PETSC_PI*x[0])*PetscSinReal(k*PETSC_PI*x[1]);
    }
  }
  *u = fac*sum;
  PetscFunctionReturn(0);
}

/*
  Let us perturb the homogeneous Dirichlet condition on the right boundary by a harmonic of wave number k, so that

  u(1, y) = sin(k \pi y)

We are then led to use an exact solution

  u = x sin(k \pi y)
  f = k^2 \pi^2 x sin(k \pi y)

so that

  -\Delta u + f = -k^2 \pi^2 x sin(k \pi y) + k^2 \pi^2 x sin(k \pi y) = 0
*/
PetscErrorCode linear_u_2d(PetscInt dim, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  SpectralCtx    *spc = (SpectralCtx *) ctx;
  const PetscInt  k   = spc->k;
  *u = x[0]*PetscSinReal(k*PETSC_PI*x[1]);
  return 0;
}

PetscErrorCode linear_f_2d(PetscInt dim, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  SpectralCtx    *spc = (SpectralCtx *) ctx;
  const PetscInt  k   = spc->k;
  *u = PetscSqr(k*PETSC_PI)*x[0]*PetscSinReal(k*PETSC_PI*x[1]);
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "linear_sol_h_2d"
/*
  The homogeneous solution for the manufactured solution above is given by

    \int_\Omega G(\vx, \vx') f(\vx')
    = -\frac{4}{\pi^2} \sum^\infty_{l, m = 1} \int^1_0 dx' \int^1_0 dy' \frac{\sin(l \pi x) \sin(l \pi x') \sin(m \pi y) \sin(m \pi y')}{l^2 + m^2} k^2 \pi^2 x' sin(k \pi y')
    = -4 k^2 \sum^\infty_{l, m = 1} \frac{\sin(l \pi x) \sin(m \pi y)}{l^2 + m^2} \int^1_0 dx' x' \sin(l \pi x') \int^1_0 dy' \sin(m \pi y') sin(k \pi y')
    = -4 k^2 \sum^\infty_{l, m = 1} \frac{\sin(l \pi x) \sin(m \pi y)}{l^2 + m^2} \frac{(-1)^{l+1}}{l \pi} \frac{\delta_{km}}{2}
    = \frac{2 k^2}{\pi} \sin(k \pi y) \sum^\infty_{l=1} \frac{(-1)^l \sin(l \pi x)}{l^3 + l k^2}
*/
static PetscErrorCode linear_sol_h_2d(PetscInt dim, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  SpectralCtx    *spc = (SpectralCtx *) ctx;
  const PetscInt  l   = spc->l;
  const PetscInt  k   = spc->k;
  const PetscReal fac = (2.0*PetscSqr(k)*PetscSinReal(k*PETSC_PI*x[1]))/PETSC_PI;
  PetscReal       sum = 0.0;
  PetscInt        j;

  PetscFunctionBeginUser;
  for (j = 1; j <= l; ++j) {
    PetscReal denom = -(PetscPowRealInt(j, 3) + j*k*k);

    if (j%2) denom = -denom;
    sum += PetscSinReal(j*PETSC_PI*x[0])/denom;
  }
  *u += fac*sum;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "linear_sol_p_2d"
/*
  The particular solution for the manufactured solution above is given by the series

  -\sum^\infty_{l = 1} c_l \sinh(l \pi x) \sin(l \pi y)

where the coefficient is given by

  c_l = \frac{-2}{\sinh(l \pi)} \int^1_0 dy' g(y') \sin(l \pi y')
      = \frac{-2}{\sinh(l \pi)} \int^1_0 dy' sin(k \pi y') \sin(l \pi y')
      = \frac{-1}{\sinh(l \pi)} \delta_{kl}

meaning that our particular solution becomes

  \frac{\sinh(k \pi x) \sin(k \pi y)}{\sinh(k \pi)}
*/
static PetscErrorCode linear_sol_p_2d(PetscInt dim, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  SpectralCtx   *spc = (SpectralCtx *) ctx;
  const PetscInt k   = spc->k;

  PetscFunctionBeginUser;
  *u += (PetscSinhReal(k*PETSC_PI*x[0])*PetscSinReal(k*PETSC_PI*x[1]))/PetscSinhReal(k*PETSC_PI);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "linear_sol_2d"
static PetscErrorCode linear_sol_2d(PetscInt dim, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  *u = 0.0;
  ierr = linear_sol_h_2d(dim, x, Nf, u, ctx);CHKERRQ(ierr);
  ierr = linear_sol_p_2d(dim, x, Nf, u, ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, SpectralCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->k = 5;
  options->L = 1;

  ierr = PetscOptionsBegin(comm, "", "Spectral Solver Options", "DMDA");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-k", "Wavenumber for perturbation", "ex76.c", options->k, &options->k, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-l", "Max wavenumber for tests", "ex76.c", options->L, &options->L, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeError"
static PetscErrorCode ComputeError(PetscReal h, PetscErrorCode (*f)(PetscInt, const PetscReal[], PetscInt, PetscScalar *, void *), void *fctx, PetscErrorCode (*g)(PetscInt, const PetscReal[], PetscInt, PetscScalar *, void *), void *gctx, NormType ntype, PetscReal *error)
{
  DM              dm, cdm;
  DMDALocalInfo   info;
  const PetscInt  M  = PetscCeilReal(1.0/h);
  const PetscReal hx = 1.0/M;
  const PetscReal dx = PetscSqr(hx);
  Vec             coordinates, errv;
  PetscScalar  ***x, **e;
  PetscScalar     u1, u2;
  PetscInt        i, j;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, M, M, PETSC_DETERMINE, PETSC_DETERMINE, 1, 1, NULL, NULL, &dm);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(dm, 0.5*hx, 1.0 - 0.5*hx, 0.5*hx, 1.0 - 0.5*hx, 0.5*hx, 1.0 - 0.5*hx);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm, &errv);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(dm, &info);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetCoordinates(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(dm, errv, &e);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(cdm, coordinates, &x);CHKERRQ(ierr);
  switch (ntype) {
  case NORM_1:
  case NORM_INFINITY:
    for (j = info.ys; j < info.ys+info.ym; ++j) {
      for (i = info.xs; i < info.xs+info.xm; ++i) {
        ierr = (*f)(2, x[j][i], 1, &u1, fctx);CHKERRQ(ierr);
        ierr = (*g)(2, x[j][i], 1, &u2, gctx);CHKERRQ(ierr);
        e[j][i] = dx*PetscAbsReal(u1 - u2);
      }
    }
    break;
  case NORM_2:
    for (j = info.ys; j < info.ys+info.ym; ++j) {
      for (i = info.xs; i < info.xs+info.xm; ++i) {
        ierr = (*f)(2, x[j][i], 1, &u1, fctx);CHKERRQ(ierr);
        ierr = (*g)(2, x[j][i], 1, &u2, gctx);CHKERRQ(ierr);
        e[j][i] = dx*PetscSqr(u1 - u2);
      }
    }
    break;
  default:
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Unsupported norm type %d", (int) ntype);
  }
  ierr = DMDAVecRestoreArray(dm, errv, &e);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOF(cdm, coordinates, &x);CHKERRQ(ierr);
  ierr = VecViewFromOptions(errv, NULL, "-error_view");CHKERRQ(ierr);
  switch (ntype) {
  case NORM_1:
    ierr = VecSum(errv, error);CHKERRQ(ierr);break;
  case NORM_2:
    ierr = VecSum(errv, error);CHKERRQ(ierr);
    *error = PetscSqrtReal(*error);break;
  case NORM_INFINITY:
    ierr = VecMax(errv, NULL, error);CHKERRQ(ierr);break;
  default:
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Unsupported norm type %d", (int) ntype);
  }
  ierr = DMRestoreGlobalVector(dm, &errv);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  DM              da;
  SpectralCtx     spc;
  const PetscReal x[2] = {0.5, 0.5};
  PetscScalar     u, uexact;
  PetscInt        l;
  PetscErrorCode  ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &spc);CHKERRQ(ierr);
  /* Convergence of the bulk spectral solution at the center of our domain */
  ierr = quartic_u_2d(2, x, 1, &uexact, NULL);CHKERRQ(ierr);
  for (l = 1; l <= spc.L; l += 2) {
    spc.l = spc.m = l;
    ierr = quartic_sol_2d(2, x, 1, &u, &spc);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "At (%g, %g), u(%D, %D) %g error %g\n", x[0], x[1], l, l, u, PetscAbsScalar(u - uexact));CHKERRQ(ierr);
  }
  for (l = 1; l <= 4*spc.L; l *= 2) {
    PetscInt    pointsPerHalfPeriod = 5;
    PetscReal   h = 1.0/(l*pointsPerHalfPeriod);
    PetscScalar error;

    spc.l = spc.m = l;
    ierr = ComputeError(h, quartic_u_2d, NULL, quartic_sol_2d, &spc, NORM_2, &error);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "(%D, %D) error %g\n", l, l, error);CHKERRQ(ierr);
  }
  /* Plotting the error in the bulk spectral solution using a DMDA */
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, -4, -4, PETSC_DETERMINE, PETSC_DETERMINE, 1, 1, NULL, NULL, &da);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);CHKERRQ(ierr);
  {
    DM            cdm;
    DMDALocalInfo info;
    Vec           coordinates, error;
    PetscScalar***x, **e;
    PetscInt      i, j;

    ierr = DMGetGlobalVector(da, &error);CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da, &info);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(da, &cdm);CHKERRQ(ierr);
    ierr = DMGetCoordinates(da, &coordinates);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, error, &e);CHKERRQ(ierr);
    ierr = DMDAVecGetArrayDOF(cdm, coordinates, &x);CHKERRQ(ierr);
    for (j = info.ys; j < info.ys+info.ym; ++j) {
      for (i = info.xs; i < info.xs+info.xm; ++i) {
        spc.l = spc.m = spc.L;
        ierr = quartic_u_2d(2, x[j][i], 1, &uexact, NULL);CHKERRQ(ierr);
        ierr = quartic_sol_2d(2, x[j][i], 1, &u, &spc);CHKERRQ(ierr);
        e[j][i] = PetscAbsScalar(u - uexact);
      }
    }
    ierr = DMDAVecRestoreArray(da, error, &e);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArrayDOF(cdm, coordinates, &x);CHKERRQ(ierr);
    ierr = VecViewFromOptions(error, NULL, "-sol_view");CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(da, &error);CHKERRQ(ierr);
  }
  /* Convergence of the perturbed spectral solution at the center of our domain */
  ierr = linear_u_2d(2, x, 1, &uexact, &spc);CHKERRQ(ierr);
  for (l = 1; l <= spc.L; l += 2) {
    PetscReal u;

    spc.l = spc.m = l;
    ierr = linear_sol_2d(2, x, 1, &u, &spc);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "At (%g, %g), u(%D, %D) %g error %g\n", x[0], x[1], spc.k, l, u, PetscAbsScalar(u - uexact));CHKERRQ(ierr);
  }
  for (l = 1; l <= 4*spc.L; l *= 2) {
    PetscInt    pointsPerHalfPeriod = 5;
    PetscReal   h = 1.0/(l*pointsPerHalfPeriod);
    PetscScalar error;

    spc.l = spc.m = l;
    ierr = ComputeError(h, linear_u_2d, &spc, linear_sol_2d, &spc, NORM_2, &error);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "(%D, %D) error %g\n", l, l, error);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
