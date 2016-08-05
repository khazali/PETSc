/* M. Adams, August 2016 */
/* X2Species */
#define X2_NION 1
typedef struct {
  PetscReal mass;
  PetscReal charge;
} X2Species;
/* static const PetscReal x2ECharge=1.6022e-19;  /\* electron charge (MKS) *\/ */
/* static const PetscReal x2Epsilon0=8.8542e-12; /\* permittivity of free space (MKS) *\/ */
/* static const PetscReal x2ProtMass=1.6720e-27; /\* proton mass (MKS) *\/ */
/* static const PetscReal x2ElecMass=9.1094e-31; /\* electron mass (MKS) *\/ */

static const PetscReal x2ECharge=1.;  /* electron charge */
static const PetscReal x2ProtMass=1.; /* proton mass */
static const PetscReal x2ElecMass=0.01; /* electron mass */
