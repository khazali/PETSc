#ifndef PETSC4PY_NUMPY_H
#define PETSC4PY_NUMPY_H

#include "Python.h"

/*
#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#endif
*/
#include "numpy/arrayobject.h"

#ifndef NPY_ARRAY_ALIGNED
#define NPY_ARRAY_ALIGNED NPY_ALIGNED
#endif
#ifndef NPY_ARRAY_WRITEABLE
#define NPY_ARRAY_WRITEABLE NPY_WRITEABLE
#endif
#ifndef NPY_ARRAY_NOTSWAPPED
#define NPY_ARRAY_NOTSWAPPED NPY_NOTSWAPPED
#endif
#ifndef NPY_ARRAY_CARRAY
#define NPY_ARRAY_CARRAY NPY_CARRAY
#endif
#ifndef NPY_ARRAY_FARRAY
#define NPY_ARRAY_FARRAY NPY_FARRAY
#endif

#include "petsc.h"

#if defined(PETSC_USE_64BIT_INDICES)
#  define NPY_PETSC_INT  NPY_LONGLONG
#else
#  define NPY_PETSC_INT  NPY_INT
#endif

#if   defined(PETSC_USE_REAL_SINGLE)
#  define NPY_PETSC_REAL    NPY_FLOAT
#  define NPY_PETSC_COMPLEX NPY_CFLOAT
#elif defined(PETSC_USE_REAL_DOUBLE)
#  define NPY_PETSC_REAL    NPY_DOUBLE
#  define NPY_PETSC_COMPLEX NPY_CDOUBLE
#elif defined(PETSC_USE_REAL_LONG_DOUBLE)
#  define NPY_PETSC_REAL    NPY_LONGDOUBLE
#  define NPY_PETSC_COMPLEX NPY_CLONGDOUBLE
#elif defined(PETSC_USE_REAL___FLOAT128)
#  define NPY_PETSC_REAL    NPY_FLOAT128
#  define NPY_PETSC_COMPLEX NPY_COMPLEX256
#else
#  error "unsupported real precision"
#endif

#if   defined(PETSC_USE_SCALAR_COMPLEX)
#  define NPY_PETSC_SCALAR  NPY_PETSC_COMPLEX
#elif defined(PETSC_USE_SCALAR_REAL)
#  define NPY_PETSC_SCALAR  NPY_PETSC_REAL
#else
#  error "unsupported scalar type"
#endif

#endif /* !PETSC4PY_NUMPY_H */
