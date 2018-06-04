#ifndef PETSC4PY_SCALAR_H
#define PETSC4PY_SCALAR_H

#include "Python.h"
#include "petsc.h"

PETSC_STATIC_INLINE
PyObject *PyPetscScalar_FromPetscScalar(PetscScalar s)
{
#if defined(PETSC_USE_SCALAR_COMPLEX)
  double a = (double)PetscRealPart(s);
  double b = (double)PetscImaginaryPart(s);
  return PyComplex_FromDoubles(a, b);
#else
  return PyFloat_FromDouble((double)s);
#endif
}

PETSC_STATIC_INLINE
PetscScalar PyPetscScalar_AsPetscScalar(PyObject *o)
{
#if defined(PETSC_USE_SCALAR_COMPLEX)
  Py_complex cval = PyComplex_AsCComplex(o);
  PetscReal a = (PetscReal)cval.real;
  PetscReal b = (PetscReal)cval.imag;
  return a + b * PETSC_i;
#else
  return (PetscScalar)PyFloat_AsDouble(o);
#endif
}

#endif/*PETSC4PY_SCALAR_H*/
