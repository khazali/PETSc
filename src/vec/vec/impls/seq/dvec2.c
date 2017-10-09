
/*
   Defines some vector operation functions that are shared by
  sequential and parallel vectors.
*/
#include <../src/vec/vec/impls/dvecimpl.h>
#include <petsc/private/kernels/petscaxpy.h>

#if defined(PETSC_HAVE_IMMINTRIN_H)
#include <immintrin.h>
#if !defined(_MM_SCALE_8)
#define _MM_SCALE_8    8
#endif
#define ZERO_SUM_REP_2 \
        sum0 = 0; \
        sum1 = 0
#define ZERO_SUM_REP_3 \
        ZERO_SUM_REP_2; \
        sum2 = 0
#define ZERO_SUM_REP_4 \
        ZERO_SUM_REP_3; \
        sum3 = 0
#define ZERO_SUM_REP_5 \
        ZERO_SUM_REP_4; \
        sum4 = 0
#define ZERO_SUM_REP_6 \
        ZERO_SUM_REP_5; \
        sum5 = 0
#define ZERO_SUM_REP_7 \
        ZERO_SUM_REP_6; \
        sum6 = 0
#define ZERO_SUM_REP_8 \
        ZERO_SUM_REP_7; \
        sum7 = 0
#define ZERO_SUM_REP_9 \
        ZERO_SUM_REP_8; \
        sum8 = 0
#define ZERO_SUM_REP_10 \
        ZERO_SUM_REP_9; \
        sum9 = 0
#define ZERO_SUM_REP_11 \
        ZERO_SUM_REP_10; \
        sum10 = 0
#define ZERO_SUM_REP_12 \
        ZERO_SUM_REP_11; \
        sum11 = 0
#define ZERO_SUM_REP_13 \
        ZERO_SUM_REP_12; \
        sum12 = 0
#define ZERO_SUM_REP_14 \
        ZERO_SUM_REP_13; \
        sum13 = 0
#define ZERO_SUM_REP_15 \
        ZERO_SUM_REP_14; \
        sum14 = 0
#define ZERO_SUM_REP_16 \
        ZERO_SUM_REP_15; \
        sum15 = 0

#define ZERO_VECZ_REP_2 \
        vec_z0 = _mm512_setzero_pd(); \
        vec_z1 = _mm512_setzero_pd()
#define ZERO_VECZ_REP_3 \
        ZERO_VECZ_REP_2; \
        vec_z2 = _mm512_setzero_pd()
#define ZERO_VECZ_REP_4 \
        ZERO_VECZ_REP_3; \
        vec_z3 = _mm512_setzero_pd()
#define ZERO_VECZ_REP_5 \
        ZERO_VECZ_REP_4; \
        vec_z4 = _mm512_setzero_pd()
#define ZERO_VECZ_REP_6 \
        ZERO_VECZ_REP_5; \
        vec_z5 = _mm512_setzero_pd()
#define ZERO_VECZ_REP_7 \
        ZERO_VECZ_REP_6; \
        vec_z6 = _mm512_setzero_pd()
#define ZERO_VECZ_REP_8 \
        ZERO_VECZ_REP_7; \
        vec_z7 = _mm512_setzero_pd()
#define ZERO_VECZ_REP_9 \
        ZERO_VECZ_REP_8; \
        vec_z8 = _mm512_setzero_pd()
#define ZERO_VECZ_REP_10 \
        ZERO_VECZ_REP_9; \
        vec_z9 = _mm512_setzero_pd()
#define ZERO_VECZ_REP_11 \
        ZERO_VECZ_REP_10; \
        vec_z10 = _mm512_setzero_pd()
#define ZERO_VECZ_REP_12 \
        ZERO_VECZ_REP_11; \
        vec_z11 = _mm512_setzero_pd()
#define ZERO_VECZ_REP_13 \
        ZERO_VECZ_REP_12; \
        vec_z12 = _mm512_setzero_pd()
#define ZERO_VECZ_REP_14 \
        ZERO_VECZ_REP_13; \
        vec_z13 = _mm512_setzero_pd()
#define ZERO_VECZ_REP_15 \
        ZERO_VECZ_REP_14; \
        vec_z14 = _mm512_setzero_pd()
#define ZERO_VECZ_REP_16 \
        ZERO_VECZ_REP_15; \
        vec_z15 = _mm512_setzero_pd()

#define VECGET_YY_REP_2 \
        ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQ(ierr); \
        ierr = VecGetArrayRead(yy[1],&yy1);CHKERRQ(ierr)
#define VECGET_YY_REP_3 \
        VECGET_YY_REP_2; \
        ierr = VecGetArrayRead(yy[2],&yy2);CHKERRQ(ierr)
#define VECGET_YY_REP_4 \
        VECGET_YY_REP_3; \
        ierr = VecGetArrayRead(yy[3],&yy3);CHKERRQ(ierr)
#define VECGET_YY_REP_5 \
        VECGET_YY_REP_4; \
        ierr = VecGetArrayRead(yy[4],&yy4);CHKERRQ(ierr)
#define VECGET_YY_REP_6 \
        VECGET_YY_REP_5; \
        ierr = VecGetArrayRead(yy[5],&yy5);CHKERRQ(ierr)
#define VECGET_YY_REP_7 \
        VECGET_YY_REP_6; \
        ierr = VecGetArrayRead(yy[6],&yy6);CHKERRQ(ierr)
#define VECGET_YY_REP_8 \
        VECGET_YY_REP_7; \
        ierr = VecGetArrayRead(yy[7],&yy7);CHKERRQ(ierr)
#define VECGET_YY_REP_9 \
        VECGET_YY_REP_8; \
        ierr = VecGetArrayRead(yy[8],&yy8);CHKERRQ(ierr)
#define VECGET_YY_REP_10 \
        VECGET_YY_REP_9; \
        ierr = VecGetArrayRead(yy[9],&yy9);CHKERRQ(ierr)
#define VECGET_YY_REP_11 \
        VECGET_YY_REP_10; \
        ierr = VecGetArrayRead(yy[10],&yy10);CHKERRQ(ierr)
#define VECGET_YY_REP_12 \
        VECGET_YY_REP_11; \
        ierr = VecGetArrayRead(yy[11],&yy11);CHKERRQ(ierr)
#define VECGET_YY_REP_13 \
        VECGET_YY_REP_12; \
        ierr = VecGetArrayRead(yy[12],&yy12);CHKERRQ(ierr)
#define VECGET_YY_REP_14 \
        VECGET_YY_REP_13; \
        ierr = VecGetArrayRead(yy[13],&yy13);CHKERRQ(ierr)
#define VECGET_YY_REP_15 \
        VECGET_YY_REP_14; \
        ierr = VecGetArrayRead(yy[14],&yy14);CHKERRQ(ierr)
#define VECGET_YY_REP_16 \
        VECGET_YY_REP_15; \
        ierr = VecGetArrayRead(yy[15],&yy15);CHKERRQ(ierr)

#define VECRESTORE_YY_REP_2 \
        ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr); \
        ierr = VecRestoreArrayRead(yy[1],&yy1);CHKERRQ(ierr)
#define VECRESTORE_YY_REP_3 \
        VECRESTORE_YY_REP_2; \
        ierr = VecRestoreArrayRead(yy[2],&yy2);CHKERRQ(ierr)
#define VECRESTORE_YY_REP_4 \
        VECRESTORE_YY_REP_3; \
        ierr = VecRestoreArrayRead(yy[3],&yy3);CHKERRQ(ierr)
#define VECRESTORE_YY_REP_5 \
        VECRESTORE_YY_REP_4; \
        ierr = VecRestoreArrayRead(yy[4],&yy4);CHKERRQ(ierr)
#define VECRESTORE_YY_REP_6 \
        VECRESTORE_YY_REP_5; \
        ierr = VecRestoreArrayRead(yy[5],&yy5);CHKERRQ(ierr)
#define VECRESTORE_YY_REP_7 \
        VECRESTORE_YY_REP_6; \
        ierr = VecRestoreArrayRead(yy[6],&yy6);CHKERRQ(ierr)
#define VECRESTORE_YY_REP_8 \
        VECRESTORE_YY_REP_7; \
        ierr = VecRestoreArrayRead(yy[7],&yy7);CHKERRQ(ierr)
#define VECRESTORE_YY_REP_9 \
        VECRESTORE_YY_REP_8; \
        ierr = VecRestoreArrayRead(yy[8],&yy8);CHKERRQ(ierr)
#define VECRESTORE_YY_REP_10 \
        VECRESTORE_YY_REP_9; \
        ierr = VecRestoreArrayRead(yy[9],&yy9);CHKERRQ(ierr)
#define VECRESTORE_YY_REP_11 \
        VECRESTORE_YY_REP_10; \
        ierr = VecRestoreArrayRead(yy[10],&yy10);CHKERRQ(ierr)
#define VECRESTORE_YY_REP_12 \
        VECRESTORE_YY_REP_11; \
        ierr = VecRestoreArrayRead(yy[11],&yy11);CHKERRQ(ierr)
#define VECRESTORE_YY_REP_13 \
        VECRESTORE_YY_REP_12; \
        ierr = VecRestoreArrayRead(yy[12],&yy12);CHKERRQ(ierr)
#define VECRESTORE_YY_REP_14 \
        VECRESTORE_YY_REP_13; \
        ierr = VecRestoreArrayRead(yy[13],&yy13);CHKERRQ(ierr)
#define VECRESTORE_YY_REP_15 \
        VECRESTORE_YY_REP_14; \
        ierr = VecRestoreArrayRead(yy[14],&yy14);CHKERRQ(ierr)
#define VECRESTORE_YY_REP_16 \
        VECRESTORE_YY_REP_15; \
        ierr = VecRestoreArrayRead(yy[15],&yy15);CHKERRQ(ierr)

#define LOAD_VECY_REP_2 \
        vec_y0 = _mm512_load_pd(yy0); \
        vec_y1 = _mm512_load_pd(yy1)
#define LOAD_VECY_REP_3 \
        LOAD_VECY_REP_2; \
        vec_y2 = _mm512_load_pd(yy2)
#define LOAD_VECY_REP_4 \
        LOAD_VECY_REP_3; \
        vec_y3 = _mm512_load_pd(yy3)
#define LOAD_VECY_REP_5 \
        LOAD_VECY_REP_4; \
        vec_y4 = _mm512_load_pd(yy4)
#define LOAD_VECY_REP_6 \
        LOAD_VECY_REP_5; \
        vec_y5 = _mm512_load_pd(yy5)
#define LOAD_VECY_REP_7 \
        LOAD_VECY_REP_6; \
        vec_y6 = _mm512_load_pd(yy6)
#define LOAD_VECY_REP_8 \
        LOAD_VECY_REP_7; \
        vec_y7 = _mm512_load_pd(yy7)
#define LOAD_VECY_REP_9 \
        LOAD_VECY_REP_8; \
        vec_y8 = _mm512_load_pd(yy8)
#define LOAD_VECY_REP_10 \
        LOAD_VECY_REP_9; \
        vec_y9 = _mm512_load_pd(yy9)
#define LOAD_VECY_REP_11 \
        LOAD_VECY_REP_10; \
        vec_y10 = _mm512_load_pd(yy10)
#define LOAD_VECY_REP_12 \
        LOAD_VECY_REP_11; \
        vec_y11 = _mm512_load_pd(yy11)
#define LOAD_VECY_REP_13 \
        LOAD_VECY_REP_12; \
        vec_y12 = _mm512_load_pd(yy12)
#define LOAD_VECY_REP_14 \
        LOAD_VECY_REP_13; \
        vec_y13 = _mm512_load_pd(yy13)
#define LOAD_VECY_REP_15 \
        LOAD_VECY_REP_14; \
        vec_y14 = _mm512_load_pd(yy14)
#define LOAD_VECY_REP_16 \
        LOAD_VECY_REP_15; \
        vec_y15 = _mm512_load_pd(yy15)

#define FMADD_VECZ_REP_2 \
        vec_z0 = _mm512_fmadd_pd(vec_x,vec_y0,vec_z0); \
        vec_z1 = _mm512_fmadd_pd(vec_x,vec_y1,vec_z1)
#define FMADD_VECZ_REP_3 \
        FMADD_VECZ_REP_2; \
        vec_z2 = _mm512_fmadd_pd(vec_x,vec_y2,vec_z2)
#define FMADD_VECZ_REP_4 \
        FMADD_VECZ_REP_3; \
        vec_z3 = _mm512_fmadd_pd(vec_x,vec_y3,vec_z3)
#define FMADD_VECZ_REP_5 \
        FMADD_VECZ_REP_4; \
        vec_z4 = _mm512_fmadd_pd(vec_x,vec_y4,vec_z4)
#define FMADD_VECZ_REP_6 \
        FMADD_VECZ_REP_5; \
        vec_z5 = _mm512_fmadd_pd(vec_x,vec_y5,vec_z5)
#define FMADD_VECZ_REP_7 \
        FMADD_VECZ_REP_6; \
        vec_z6 = _mm512_fmadd_pd(vec_x,vec_y6,vec_z6)
#define FMADD_VECZ_REP_8 \
        FMADD_VECZ_REP_7; \
        vec_z7 = _mm512_fmadd_pd(vec_x,vec_y7,vec_z7)
#define FMADD_VECZ_REP_9 \
        FMADD_VECZ_REP_8; \
        vec_z8 = _mm512_fmadd_pd(vec_x,vec_y8,vec_z8)
#define FMADD_VECZ_REP_10 \
        FMADD_VECZ_REP_9; \
        vec_z9 = _mm512_fmadd_pd(vec_x,vec_y9,vec_z9)
#define FMADD_VECZ_REP_11 \
        FMADD_VECZ_REP_10; \
        vec_z10 = _mm512_fmadd_pd(vec_x,vec_y10,vec_z10)
#define FMADD_VECZ_REP_12 \
        FMADD_VECZ_REP_11; \
        vec_z11 = _mm512_fmadd_pd(vec_x,vec_y11,vec_z11)
#define FMADD_VECZ_REP_13 \
        FMADD_VECZ_REP_12; \
        vec_z12 = _mm512_fmadd_pd(vec_x,vec_y12,vec_z12)
#define FMADD_VECZ_REP_14 \
        FMADD_VECZ_REP_13; \
        vec_z13 = _mm512_fmadd_pd(vec_x,vec_y13,vec_z13)
#define FMADD_VECZ_REP_15 \
        FMADD_VECZ_REP_14; \
        vec_z14 = _mm512_fmadd_pd(vec_x,vec_y14,vec_z14)
#define FMADD_VECZ_REP_16 \
        FMADD_VECZ_REP_15; \
        vec_z15 = _mm512_fmadd_pd(vec_x,vec_y15,vec_z15)

#define MASK_FMADD_VECZ_REP_2 \
        vec_z0 = _mm512_mask3_fmadd_pd(vec_x,vec_y0,vec_z0,mask); \
        vec_z1 = _mm512_mask3_fmadd_pd(vec_x,vec_y1,vec_z1,mask)
#define MASK_FMADD_VECZ_REP_3 \
        MASK_FMADD_VECZ_REP_2; \
        vec_z2 = _mm512_mask3_fmadd_pd(vec_x,vec_y2,vec_z2,mask)
#define MASK_FMADD_VECZ_REP_4 \
        MASK_FMADD_VECZ_REP_3; \
        vec_z3 = _mm512_mask3_fmadd_pd(vec_x,vec_y3,vec_z3,mask)
#define MASK_FMADD_VECZ_REP_5 \
        MASK_FMADD_VECZ_REP_4; \
        vec_z4 = _mm512_mask3_fmadd_pd(vec_x,vec_y4,vec_z4,mask)
#define MASK_FMADD_VECZ_REP_6 \
        MASK_FMADD_VECZ_REP_5; \
        vec_z5 = _mm512_mask3_fmadd_pd(vec_x,vec_y5,vec_z5,mask)
#define MASK_FMADD_VECZ_REP_7 \
        MASK_FMADD_VECZ_REP_6; \
        vec_z6 = _mm512_mask3_fmadd_pd(vec_x,vec_y6,vec_z6,mask)
#define MASK_FMADD_VECZ_REP_8 \
        MASK_FMADD_VECZ_REP_7; \
        vec_z7 = _mm512_mask3_fmadd_pd(vec_x,vec_y7,vec_z7,mask)
#define MASK_FMADD_VECZ_REP_9 \
        MASK_FMADD_VECZ_REP_8; \
        vec_z8 = _mm512_mask3_fmadd_pd(vec_x,vec_y8,vec_z8,mask)
#define MASK_FMADD_VECZ_REP_10 \
        MASK_FMADD_VECZ_REP_9; \
        vec_z9 = _mm512_mask3_fmadd_pd(vec_x,vec_y9,vec_z9,mask)
#define MASK_FMADD_VECZ_REP_11 \
        MASK_FMADD_VECZ_REP_10; \
        vec_z10 = _mm512_mask3_fmadd_pd(vec_x,vec_y10,vec_z10,mask)
#define MASK_FMADD_VECZ_REP_12 \
        MASK_FMADD_VECZ_REP_11; \
        vec_z11 = _mm512_mask3_fmadd_pd(vec_x,vec_y11,vec_z11,mask)
#define MASK_FMADD_VECZ_REP_13 \
        MASK_FMADD_VECZ_REP_12; \
        vec_z12 = _mm512_mask3_fmadd_pd(vec_x,vec_y12,vec_z12,mask)
#define MASK_FMADD_VECZ_REP_14 \
        MASK_FMADD_VECZ_REP_13; \
        vec_z13 = _mm512_mask3_fmadd_pd(vec_x,vec_y13,vec_z13,mask)
#define MASK_FMADD_VECZ_REP_15 \
        MASK_FMADD_VECZ_REP_14; \
        vec_z14 = _mm512_mask3_fmadd_pd(vec_x,vec_y14,vec_z14,mask)
#define MASK_FMADD_VECZ_REP_16 \
        MASK_FMADD_VECZ_REP_15; \
        vec_z15 = _mm512_mask3_fmadd_pd(vec_x,vec_y15,vec_z15,mask)

#define ADD_TO_SUM_REP_2 \
        sum0 += x[0]*yy0[0]; \
        sum1 += x[0]*yy1[0]
#define ADD_TO_SUM_REP_3 \
        ADD_TO_SUM_REP_2; \
        sum2 += x[0]*yy2[0]
#define ADD_TO_SUM_REP_4 \
        ADD_TO_SUM_REP_3; \
        sum3 += x[0]*yy3[0]
#define ADD_TO_SUM_REP_5 \
        ADD_TO_SUM_REP_4; \
        sum4 += x[0]*yy4[0]
#define ADD_TO_SUM_REP_6 \
        ADD_TO_SUM_REP_5; \
        sum5 += x[0]*yy5[0]
#define ADD_TO_SUM_REP_7 \
        ADD_TO_SUM_REP_6; \
        sum6 += x[0]*yy6[0]
#define ADD_TO_SUM_REP_8 \
        ADD_TO_SUM_REP_7; \
        sum7 += x[0]*yy7[0]
#define ADD_TO_SUM_REP_9 \
        ADD_TO_SUM_REP_8; \
        sum8 += x[0]*yy8[0]
#define ADD_TO_SUM_REP_10 \
        ADD_TO_SUM_REP_9; \
        sum9 += x[0]*yy9[0]
#define ADD_TO_SUM_REP_11 \
        ADD_TO_SUM_REP_10; \
        sum10 += x[0]*yy10[0]
#define ADD_TO_SUM_REP_12 \
        ADD_TO_SUM_REP_11; \
        sum11 += x[0]*yy11[0]
#define ADD_TO_SUM_REP_13 \
        ADD_TO_SUM_REP_12; \
        sum12 += x[0]*yy12[0]
#define ADD_TO_SUM_REP_14 \
        ADD_TO_SUM_REP_13; \
        sum13 += x[0]*yy13[0]
#define ADD_TO_SUM_REP_15 \
        ADD_TO_SUM_REP_14; \
        sum14 += x[0]*yy14[0]
#define ADD_TO_SUM_REP_16 \
        ADD_TO_SUM_REP_15; \
        sum15 += x[0]*yy15[0]

#define ADD_TO_SUM2_REP_2 \
        sum0 += x[1]*yy0[1]; \
        sum1 += x[1]*yy1[1]
#define ADD_TO_SUM2_REP_3 \
        ADD_TO_SUM2_REP_2; \
        sum2 += x[1]*yy2[1]
#define ADD_TO_SUM2_REP_4 \
        ADD_TO_SUM2_REP_3; \
        sum3 += x[1]*yy3[1]
#define ADD_TO_SUM2_REP_5 \
        ADD_TO_SUM2_REP_4; \
        sum4 += x[1]*yy4[1]
#define ADD_TO_SUM2_REP_6 \
        ADD_TO_SUM2_REP_5; \
        sum5 += x[1]*yy5[1]
#define ADD_TO_SUM2_REP_7 \
        ADD_TO_SUM2_REP_6; \
        sum6 += x[1]*yy6[1]
#define ADD_TO_SUM2_REP_8 \
        ADD_TO_SUM2_REP_7; \
        sum7 += x[1]*yy7[1]
#define ADD_TO_SUM2_REP_9 \
        ADD_TO_SUM2_REP_8; \
        sum8 += x[1]*yy8[1]
#define ADD_TO_SUM2_REP_10 \
        ADD_TO_SUM2_REP_9; \
        sum9 += x[1]*yy9[1]
#define ADD_TO_SUM2_REP_11 \
        ADD_TO_SUM2_REP_10; \
        sum10 += x[1]*yy10[1]
#define ADD_TO_SUM2_REP_12 \
        ADD_TO_SUM2_REP_11; \
        sum11 += x[1]*yy11[1]
#define ADD_TO_SUM2_REP_13 \
        ADD_TO_SUM2_REP_12; \
        sum12 += x[1]*yy12[1]
#define ADD_TO_SUM2_REP_14 \
        ADD_TO_SUM2_REP_13; \
        sum13 += x[1]*yy13[1]
#define ADD_TO_SUM2_REP_15 \
        ADD_TO_SUM2_REP_14; \
        sum14 += x[1]*yy14[1]
#define ADD_TO_SUM2_REP_16 \
        ADD_TO_SUM2_REP_15; \
        sum15 += x[1]*yy15[1]

#define REDUCE_TO_SUM_REP_2 \
        sum0 += _mm512_reduce_add_pd(vec_z0); \
        sum1 += _mm512_reduce_add_pd(vec_z1)
#define REDUCE_TO_SUM_REP_3 \
        REDUCE_TO_SUM_REP_2; \
        sum2 += _mm512_reduce_add_pd(vec_z2)
#define REDUCE_TO_SUM_REP_4 \
        REDUCE_TO_SUM_REP_3; \
        sum3 += _mm512_reduce_add_pd(vec_z3)
#define REDUCE_TO_SUM_REP_5 \
        REDUCE_TO_SUM_REP_4; \
        sum4 += _mm512_reduce_add_pd(vec_z4)
#define REDUCE_TO_SUM_REP_6 \
        REDUCE_TO_SUM_REP_5; \
        sum5 += _mm512_reduce_add_pd(vec_z5)
#define REDUCE_TO_SUM_REP_7 \
        REDUCE_TO_SUM_REP_6; \
        sum6 += _mm512_reduce_add_pd(vec_z6)
#define REDUCE_TO_SUM_REP_8 \
        REDUCE_TO_SUM_REP_7; \
        sum7 += _mm512_reduce_add_pd(vec_z7)
#define REDUCE_TO_SUM_REP_9 \
        REDUCE_TO_SUM_REP_8; \
        sum8 += _mm512_reduce_add_pd(vec_z8)
#define REDUCE_TO_SUM_REP_10 \
        REDUCE_TO_SUM_REP_9; \
        sum9 += _mm512_reduce_add_pd(vec_z9)
#define REDUCE_TO_SUM_REP_11 \
        REDUCE_TO_SUM_REP_10; \
        sum10 += _mm512_reduce_add_pd(vec_z10)
#define REDUCE_TO_SUM_REP_12 \
        REDUCE_TO_SUM_REP_11; \
        sum11 += _mm512_reduce_add_pd(vec_z11)
#define REDUCE_TO_SUM_REP_13 \
        REDUCE_TO_SUM_REP_12; \
        sum12 += _mm512_reduce_add_pd(vec_z12)
#define REDUCE_TO_SUM_REP_14 \
        REDUCE_TO_SUM_REP_13; \
        sum13 += _mm512_reduce_add_pd(vec_z13)
#define REDUCE_TO_SUM_REP_15 \
        REDUCE_TO_SUM_REP_14; \
        sum14 += _mm512_reduce_add_pd(vec_z14)
#define REDUCE_TO_SUM_REP_16 \
        REDUCE_TO_SUM_REP_15; \
        sum15 += _mm512_reduce_add_pd(vec_z15)

#define COPY_SUM_REP_2 \
        z[0] = sum0; \
        z[1] = sum1
#define COPY_SUM_REP_3 \
        COPY_SUM_REP_2; \
        z[2] = sum2
#define COPY_SUM_REP_4 \
        COPY_SUM_REP_3; \
        z[3] = sum3
#define COPY_SUM_REP_5 \
        COPY_SUM_REP_4; \
        z[4] = sum4
#define COPY_SUM_REP_6 \
        COPY_SUM_REP_5; \
        z[5] = sum5
#define COPY_SUM_REP_7 \
        COPY_SUM_REP_6; \
        z[6] = sum6
#define COPY_SUM_REP_8 \
        COPY_SUM_REP_7; \
        z[7] = sum7
#define COPY_SUM_REP_9 \
        COPY_SUM_REP_8; \
        z[8] = sum8
#define COPY_SUM_REP_10 \
        COPY_SUM_REP_9; \
        z[9] = sum9
#define COPY_SUM_REP_11 \
        COPY_SUM_REP_10; \
        z[10] = sum10
#define COPY_SUM_REP_12 \
        COPY_SUM_REP_11; \
        z[11] = sum11
#define COPY_SUM_REP_13 \
        COPY_SUM_REP_12; \
        z[12] = sum12
#define COPY_SUM_REP_14 \
        COPY_SUM_REP_13; \
        z[13] = sum13
#define COPY_SUM_REP_15 \
        COPY_SUM_REP_14; \
        z[14] = sum14
#define COPY_SUM_REP_16 \
        COPY_SUM_REP_15; \
        z[15] = sum15

#define INCR_YY_REP_2 \
        yy0 += 8; \
        yy1 += 8
#define INCR_YY_REP_3 \
        INCR_YY_REP_2; \
        yy2 += 8
#define INCR_YY_REP_4 \
        INCR_YY_REP_3; \
        yy3 += 8
#define INCR_YY_REP_5 \
        INCR_YY_REP_4; \
        yy4 += 8
#define INCR_YY_REP_6 \
        INCR_YY_REP_5; \
        yy5 += 8
#define INCR_YY_REP_7 \
        INCR_YY_REP_6; \
        yy6 += 8
#define INCR_YY_REP_8 \
        INCR_YY_REP_7; \
        yy7 += 8
#define INCR_YY_REP_9 \
        INCR_YY_REP_8; \
        yy8 += 8
#define INCR_YY_REP_10 \
        INCR_YY_REP_9; \
        yy9 += 8
#define INCR_YY_REP_11 \
        INCR_YY_REP_10; \
        yy10 += 8
#define INCR_YY_REP_12 \
        INCR_YY_REP_11; \
        yy11 += 8
#define INCR_YY_REP_13 \
        INCR_YY_REP_12; \
        yy12 += 8
#define INCR_YY_REP_14 \
        INCR_YY_REP_13; \
        yy13 += 8
#define INCR_YY_REP_15 \
        INCR_YY_REP_14; \
        yy14 += 8
#define INCR_YY_REP_16 \
        INCR_YY_REP_15; \
        yy15 += 8

#define PetscAVX512MDot(factor) \
    VECGET_YY_REP_##factor;\
    for (j=0;j<((n>>3)<<3);j+=8) { \
      vec_x  = _mm512_load_pd(x); \
      LOAD_VECY_REP_##factor; \
      FMADD_VECZ_REP_##factor; \
      INCR_YY_REP_##factor; \
      x += 8; \
    } \
    if ((n&0x07)>2) { \
      vec_x  = _mm512_load_pd(x); \
      LOAD_VECY_REP_##factor; \
      MASK_FMADD_VECZ_REP_##factor; \
    } else if ((n&0x07)==2) { \
      ADD_TO_SUM_REP_##factor; \
      ADD_TO_SUM2_REP_##factor; \
    } else if ((n&0x07)==1) { \
      ADD_TO_SUM_REP_##factor; \
    } \
    if (n>2) { \
      REDUCE_TO_SUM_REP_##factor; \
    } \
    COPY_SUM_REP_##factor; \
    VECRESTORE_YY_REP_##factor

#endif

#if defined(PETSC_USE_FORTRAN_KERNEL_MDOT)
#include <../src/vec/vec/impls/seq/ftn-kernels/fmdot.h>
PetscErrorCode VecMDot_Seq(Vec xin,PetscInt nv,const Vec yin[],PetscScalar *z)
{
  PetscErrorCode    ierr;
  PetscInt          i,nv_rem,n = xin->map->n;
  PetscScalar       sum0,sum1,sum2,sum3;
  const PetscScalar *yy0,*yy1,*yy2,*yy3,*x;
  Vec               *yy;

  PetscFunctionBegin;
  sum0 = 0.0;
  sum1 = 0.0;
  sum2 = 0.0;

  i      = nv;
  nv_rem = nv&0x3;
  yy     = (Vec*)yin;
  ierr   = VecGetArrayRead(xin,&x);CHKERRQ(ierr);

  switch (nv_rem) {
  case 3:
    ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[2],&yy2);CHKERRQ(ierr);
    fortranmdot3_(x,yy0,yy1,yy2,&n,&sum0,&sum1,&sum2);
    ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[2],&yy2);CHKERRQ(ierr);
    z[0] = sum0;
    z[1] = sum1;
    z[2] = sum2;
    break;
  case 2:
    ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[1],&yy1);CHKERRQ(ierr);
    fortranmdot2_(x,yy0,yy1,&n,&sum0,&sum1);
    ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[1],&yy1);CHKERRQ(ierr);
    z[0] = sum0;
    z[1] = sum1;
    break;
  case 1:
    ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    fortranmdot1_(x,yy0,&n,&sum0);
    ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    z[0] = sum0;
    break;
  case 0:
    break;
  }
  z  += nv_rem;
  i  -= nv_rem;
  yy += nv_rem;

  while (i >0) {
    sum0 = 0.;
    sum1 = 0.;
    sum2 = 0.;
    sum3 = 0.;
    ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[2],&yy2);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[3],&yy3);CHKERRQ(ierr);
    fortranmdot4_(x,yy0,yy1,yy2,yy3,&n,&sum0,&sum1,&sum2,&sum3);
    ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[2],&yy2);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[3],&yy3);CHKERRQ(ierr);
    yy  += 4;
    z[0] = sum0;
    z[1] = sum1;
    z[2] = sum2;
    z[3] = sum3;
    z   += 4;
    i   -= 4;
  }
  ierr = VecRestoreArrayRead(xin,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(PetscMax(nv*(2.0*xin->map->n-1),0.0));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#else
#if defined(PETSC_HAVE_IMMINTRIN_H) && defined(__AVX512F__) && defined(PETSC_USE_REAL_DOUBLE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_64BIT_INDICES)
/*
PetscErrorCode VecMDot_Seq(Vec xin,PetscInt nv,const Vec yin[],PetscScalar *z)
{
  PetscInt          n = xin->map->n,i,j;
  PetscScalar       sum;
  const PetscScalar *y,*x,*xbase;
  Vec               *yy;
  __m512d           vec_x,vec_y,vec_z;
  __mmask8          mask;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if ((n&0x07)>2) mask   = (__mmask8)(0xff >> (8-(n&0x07)));

  yy   = (Vec*)yin;
  ierr = VecGetArrayRead(xin,&xbase);CHKERRQ(ierr);
  x    = xbase;
  for (i=0; i<nv; i++) {
    sum   = 0.;
    vec_z = _mm512_setzero_pd();
    ierr  = VecGetArrayRead(*yy,&y);CHKERRQ(ierr);
    x = xbase;
    for (j=0;j<((n>>3)<<3);j+=8) {
      vec_x = _mm512_load_pd(x);
      vec_y = _mm512_load_pd(y);
      vec_z = _mm512_fmadd_pd(vec_x,vec_y,vec_z);
      x += 8; y += 8;
    }
    if ((n&0x07)>2) {
      vec_x = _mm512_load_pd(x);
      vec_y = _mm512_load_pd(y);
      vec_z = _mm512_mask3_fmadd_pd(vec_x,vec_y,vec_z,mask);
    } else if ((n&0x07)==2) {
      sum += x[0]*y[0];
      sum += x[1]*y[1];
    } else if ((n&0x07)==1) {
      sum += x[0]*y[0];
    }
    if (n>2) {
      sum += _mm512_reduce_add_pd(vec_z);
    }
    z[i] = sum;
    ierr = VecRestoreArrayRead(*yy,&y);CHKERRQ(ierr);
    yy++;
  }
  ierr = VecRestoreArrayRead(xin,&xbase);CHKERRQ(ierr);
  ierr = PetscLogFlops(PetscMax(nv*(2.0*xin->map->n-1),0.0));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
*/
PetscErrorCode VecMDot_Seq(Vec xin,PetscInt nv,const Vec yin[],PetscScalar *z)
{
  PetscErrorCode    ierr;
  PetscInt          n = xin->map->n,i,j,nv_rem;
  PetscScalar       sum0,sum1,sum2,sum3;
  const PetscScalar *yy0,*yy1,*yy2,*yy3,*x,*xbase;
  Vec               *yy;
  __m512d           vec_x,vec_y0,vec_y1,vec_y2,vec_y3,vec_z0,vec_z1,vec_z2,vec_z3;
  PetscScalar       sum4,sum5,sum6,sum7,sum8,sum9,sum10,sum11,sum12,sum13,sum14,sum15;
  const PetscScalar *yy4,*yy5,*yy6,*yy7,*yy8,*yy9,*yy10,*yy11,*yy12,*yy13,*yy14,*yy15;;
  __m512d           vec_y4,vec_y5,vec_y6,vec_y7,vec_y8,vec_y9,vec_y10,vec_y11,vec_y12,vec_y13,vec_y14,vec_y15;
  __m512d           vec_z4,vec_z5,vec_z6,vec_z7,vec_z8,vec_z9,vec_z10,vec_z11,vec_z12,vec_z13,vec_z14,vec_z15;;
  __mmask8          mask;

  PetscFunctionBegin;
  if ((n&0x07)>2) mask   = (__mmask8)(0xff >> (8-(n&0x07)));

  i      = nv;
  nv_rem = nv&0xf;
  yy     = (Vec*)yin;
  ierr   = VecGetArrayRead(xin,&xbase);CHKERRQ(ierr);
  x      = xbase;
  ZERO_SUM_REP_15;
  ZERO_VECZ_REP_15;

  switch (nv_rem) {
  case 15:
    PetscAVX512MDot(15);
    break;
  case 14:
    PetscAVX512MDot(14);
    break;
  case 13:
    PetscAVX512MDot(13);
    break;
  case 12:
    PetscAVX512MDot(12);
    break;
  case 11:
    PetscAVX512MDot(11);
    break;
  case 10:
    PetscAVX512MDot(10);
    break;
  case 9:
    PetscAVX512MDot(9);
    break;
  case 8:
    PetscAVX512MDot(8);
    break;
  case 7:
    PetscAVX512MDot(7);
    break;
  case 6:
    PetscAVX512MDot(6);
    break;
  case 5:
    PetscAVX512MDot(5);
    break;
  case 4:
    PetscAVX512MDot(4);
    break;
  case 3:
    PetscAVX512MDot(3);
    break;
  case 2:
    PetscAVX512MDot(2);
    break;
  case 1:
    ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    for (j=0;j<((n>>3)<<3);j+=8) {
      vec_x  = _mm512_load_pd(x);
      vec_y0 = _mm512_load_pd(yy0);
      vec_z0 = _mm512_fmadd_pd(vec_x,vec_y0,vec_z0);
      x += 8; yy0 += 8;
    }
    if ((n&0x07)>2) {
      mask   = (__mmask8)(0xff >> (8-(n&0x07)));
      vec_x  = _mm512_load_pd(x);
      vec_y0 = _mm512_load_pd(yy0);
      vec_z0 = _mm512_mask3_fmadd_pd(vec_x,vec_y0,vec_z0,mask);
    } else if ((n&0x07)==2) {
      sum0 += x[0]*yy0[0];
      sum0 += x[1]*yy0[1];
    } else if ((n&0x07)==1) {
      sum0 += x[0]*yy0[0];
    }
    if (n>2) {
      sum0 += _mm512_reduce_add_pd(vec_z0);
    }
    z[0] = sum0;
    ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    break;
  case 0:
    break;
  }
  z  += nv_rem;
  i  -= nv_rem;
  yy += nv_rem;

  while (i >0) {
    x = xbase;
    ZERO_SUM_REP_16;
    ZERO_VECZ_REP_16;
    PetscAVX512MDot(16);
    z   += 16;
    i   -= 16;
    yy  += 16;
  }

  ierr = VecRestoreArrayRead(xin,&xbase);CHKERRQ(ierr);
  ierr = PetscLogFlops(PetscMax(nv*(2.0*xin->map->n-1),0.0));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#else
PetscErrorCode VecMDot_Seq(Vec xin,PetscInt nv,const Vec yin[],PetscScalar *z)
{
  PetscErrorCode    ierr;
  PetscInt          n = xin->map->n,i,j,nv_rem,j_rem;
  PetscScalar       sum0,sum1,sum2,sum3,x0,x1,x2,x3;
  const PetscScalar *yy0,*yy1,*yy2,*yy3,*x,*xbase;
  Vec               *yy;

  PetscFunctionBegin;
  sum0 = 0.;
  sum1 = 0.;
  sum2 = 0.;

  i      = nv;
  nv_rem = nv&0x3;
  yy     = (Vec*)yin;
  j      = n;
  ierr   = VecGetArrayRead(xin,&xbase);CHKERRQ(ierr);
  x      = xbase;

  switch (nv_rem) {
  case 3:
    ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[2],&yy2);CHKERRQ(ierr);
    switch (j_rem=j&0x3) {
    case 3:
      x2    = x[2];
      sum0 += x2*PetscConj(yy0[2]); sum1 += x2*PetscConj(yy1[2]);
      sum2 += x2*PetscConj(yy2[2]);
    case 2:
      x1    = x[1];
      sum0 += x1*PetscConj(yy0[1]); sum1 += x1*PetscConj(yy1[1]);
      sum2 += x1*PetscConj(yy2[1]);
    case 1:
      x0    = x[0];
      sum0 += x0*PetscConj(yy0[0]); sum1 += x0*PetscConj(yy1[0]);
      sum2 += x0*PetscConj(yy2[0]);
    case 0:
      x   += j_rem;
      yy0 += j_rem;
      yy1 += j_rem;
      yy2 += j_rem;
      j   -= j_rem;
      break;
    }
    while (j>0) {
      x0 = x[0];
      x1 = x[1];
      x2 = x[2];
      x3 = x[3];
      x += 4;

      sum0 += x0*PetscConj(yy0[0]) + x1*PetscConj(yy0[1]) + x2*PetscConj(yy0[2]) + x3*PetscConj(yy0[3]); yy0+=4;
      sum1 += x0*PetscConj(yy1[0]) + x1*PetscConj(yy1[1]) + x2*PetscConj(yy1[2]) + x3*PetscConj(yy1[3]); yy1+=4;
      sum2 += x0*PetscConj(yy2[0]) + x1*PetscConj(yy2[1]) + x2*PetscConj(yy2[2]) + x3*PetscConj(yy2[3]); yy2+=4;
      j    -= 4;
    }
    z[0] = sum0;
    z[1] = sum1;
    z[2] = sum2;
    ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[2],&yy2);CHKERRQ(ierr);
    break;
  case 2:
    ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[1],&yy1);CHKERRQ(ierr);
    switch (j_rem=j&0x3) {
    case 3:
      x2    = x[2];
      sum0 += x2*PetscConj(yy0[2]); sum1 += x2*PetscConj(yy1[2]);
    case 2:
      x1    = x[1];
      sum0 += x1*PetscConj(yy0[1]); sum1 += x1*PetscConj(yy1[1]);
    case 1:
      x0    = x[0];
      sum0 += x0*PetscConj(yy0[0]); sum1 += x0*PetscConj(yy1[0]);
    case 0:
      x   += j_rem;
      yy0 += j_rem;
      yy1 += j_rem;
      j   -= j_rem;
      break;
    }
    while (j>0) {
      x0 = x[0];
      x1 = x[1];
      x2 = x[2];
      x3 = x[3];
      x += 4;

      sum0 += x0*PetscConj(yy0[0]) + x1*PetscConj(yy0[1]) + x2*PetscConj(yy0[2]) + x3*PetscConj(yy0[3]); yy0+=4;
      sum1 += x0*PetscConj(yy1[0]) + x1*PetscConj(yy1[1]) + x2*PetscConj(yy1[2]) + x3*PetscConj(yy1[3]); yy1+=4;
      j    -= 4;
    }
    z[0] = sum0;
    z[1] = sum1;

    ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[1],&yy1);CHKERRQ(ierr);
    break;
  case 1:
    ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    switch (j_rem=j&0x3) {
    case 3:
      x2 = x[2]; sum0 += x2*PetscConj(yy0[2]);
    case 2:
      x1 = x[1]; sum0 += x1*PetscConj(yy0[1]);
    case 1:
      x0 = x[0]; sum0 += x0*PetscConj(yy0[0]);
    case 0:
      x   += j_rem;
      yy0 += j_rem;
      j   -= j_rem;
      break;
    }
    while (j>0) {
      sum0 += x[0]*PetscConj(yy0[0]) + x[1]*PetscConj(yy0[1])
            + x[2]*PetscConj(yy0[2]) + x[3]*PetscConj(yy0[3]);
      yy0  +=4;
      j    -= 4; x+=4;
    }
    z[0] = sum0;

    ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    break;
  case 0:
    break;
  }
  z  += nv_rem;
  i  -= nv_rem;
  yy += nv_rem;

  while (i >0) {
    sum0 = 0.;
    sum1 = 0.;
    sum2 = 0.;
    sum3 = 0.;
    ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[2],&yy2);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[3],&yy3);CHKERRQ(ierr);

    j = n;
    x = xbase;
    switch (j_rem=j&0x3) {
    case 3:
      x2    = x[2];
      sum0 += x2*PetscConj(yy0[2]); sum1 += x2*PetscConj(yy1[2]);
      sum2 += x2*PetscConj(yy2[2]); sum3 += x2*PetscConj(yy3[2]);
    case 2:
      x1    = x[1];
      sum0 += x1*PetscConj(yy0[1]); sum1 += x1*PetscConj(yy1[1]);
      sum2 += x1*PetscConj(yy2[1]); sum3 += x1*PetscConj(yy3[1]);
    case 1:
      x0    = x[0];
      sum0 += x0*PetscConj(yy0[0]); sum1 += x0*PetscConj(yy1[0]);
      sum2 += x0*PetscConj(yy2[0]); sum3 += x0*PetscConj(yy3[0]);
    case 0:
      x   += j_rem;
      yy0 += j_rem;
      yy1 += j_rem;
      yy2 += j_rem;
      yy3 += j_rem;
      j   -= j_rem;
      break;
    }
    while (j>0) {
      x0 = x[0];
      x1 = x[1];
      x2 = x[2];
      x3 = x[3];
      x += 4;

      sum0 += x0*PetscConj(yy0[0]) + x1*PetscConj(yy0[1]) + x2*PetscConj(yy0[2]) + x3*PetscConj(yy0[3]); yy0+=4;
      sum1 += x0*PetscConj(yy1[0]) + x1*PetscConj(yy1[1]) + x2*PetscConj(yy1[2]) + x3*PetscConj(yy1[3]); yy1+=4;
      sum2 += x0*PetscConj(yy2[0]) + x1*PetscConj(yy2[1]) + x2*PetscConj(yy2[2]) + x3*PetscConj(yy2[3]); yy2+=4;
      sum3 += x0*PetscConj(yy3[0]) + x1*PetscConj(yy3[1]) + x2*PetscConj(yy3[2]) + x3*PetscConj(yy3[3]); yy3+=4;
      j    -= 4;
    }
    z[0] = sum0;
    z[1] = sum1;
    z[2] = sum2;
    z[3] = sum3;
    z   += 4;
    i   -= 4;
    ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[2],&yy2);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[3],&yy3);CHKERRQ(ierr);
    yy  += 4;
  }
  ierr = VecRestoreArrayRead(xin,&xbase);CHKERRQ(ierr);
  ierr = PetscLogFlops(PetscMax(nv*(2.0*xin->map->n-1),0.0));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif
#endif

/* ----------------------------------------------------------------------------*/
PetscErrorCode VecMTDot_Seq(Vec xin,PetscInt nv,const Vec yin[],PetscScalar *z)
{
  PetscErrorCode    ierr;
  PetscInt          n = xin->map->n,i,j,nv_rem,j_rem;
  PetscScalar       sum0,sum1,sum2,sum3,x0,x1,x2,x3;
  const PetscScalar *yy0,*yy1,*yy2,*yy3,*x,*xbase;
  Vec               *yy;

  PetscFunctionBegin;
  sum0 = 0.;
  sum1 = 0.;
  sum2 = 0.;

  i      = nv;
  nv_rem = nv&0x3;
  yy     = (Vec*)yin;
  j      = n;
  ierr   = VecGetArrayRead(xin,&xbase);CHKERRQ(ierr);
  x      = xbase;

  switch (nv_rem) {
  case 3:
    ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[2],&yy2);CHKERRQ(ierr);
    switch (j_rem=j&0x3) {
    case 3:
      x2    = x[2];
      sum0 += x2*yy0[2]; sum1 += x2*yy1[2];
      sum2 += x2*yy2[2];
    case 2:
      x1    = x[1];
      sum0 += x1*yy0[1]; sum1 += x1*yy1[1];
      sum2 += x1*yy2[1];
    case 1:
      x0    = x[0];
      sum0 += x0*yy0[0]; sum1 += x0*yy1[0];
      sum2 += x0*yy2[0];
    case 0:
      x   += j_rem;
      yy0 += j_rem;
      yy1 += j_rem;
      yy2 += j_rem;
      j   -= j_rem;
      break;
    }
    while (j>0) {
      x0 = x[0];
      x1 = x[1];
      x2 = x[2];
      x3 = x[3];
      x += 4;

      sum0 += x0*yy0[0] + x1*yy0[1] + x2*yy0[2] + x3*yy0[3]; yy0+=4;
      sum1 += x0*yy1[0] + x1*yy1[1] + x2*yy1[2] + x3*yy1[3]; yy1+=4;
      sum2 += x0*yy2[0] + x1*yy2[1] + x2*yy2[2] + x3*yy2[3]; yy2+=4;
      j    -= 4;
    }
    z[0] = sum0;
    z[1] = sum1;
    z[2] = sum2;
    ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[2],&yy2);CHKERRQ(ierr);
    break;
  case 2:
    ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[1],&yy1);CHKERRQ(ierr);
    switch (j_rem=j&0x3) {
    case 3:
      x2    = x[2];
      sum0 += x2*yy0[2]; sum1 += x2*yy1[2];
    case 2:
      x1    = x[1];
      sum0 += x1*yy0[1]; sum1 += x1*yy1[1];
    case 1:
      x0    = x[0];
      sum0 += x0*yy0[0]; sum1 += x0*yy1[0];
    case 0:
      x   += j_rem;
      yy0 += j_rem;
      yy1 += j_rem;
      j   -= j_rem;
      break;
    }
    while (j>0) {
      x0 = x[0];
      x1 = x[1];
      x2 = x[2];
      x3 = x[3];
      x += 4;

      sum0 += x0*yy0[0] + x1*yy0[1] + x2*yy0[2] + x3*yy0[3]; yy0+=4;
      sum1 += x0*yy1[0] + x1*yy1[1] + x2*yy1[2] + x3*yy1[3]; yy1+=4;
      j    -= 4;
    }
    z[0] = sum0;
    z[1] = sum1;

    ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[1],&yy1);CHKERRQ(ierr);
    break;
  case 1:
    ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    switch (j_rem=j&0x3) {
    case 3:
      x2 = x[2]; sum0 += x2*yy0[2];
    case 2:
      x1 = x[1]; sum0 += x1*yy0[1];
    case 1:
      x0 = x[0]; sum0 += x0*yy0[0];
    case 0:
      x   += j_rem;
      yy0 += j_rem;
      j   -= j_rem;
      break;
    }
    while (j>0) {
      sum0 += x[0]*yy0[0] + x[1]*yy0[1] + x[2]*yy0[2] + x[3]*yy0[3]; yy0+=4;
      j    -= 4; x+=4;
    }
    z[0] = sum0;

    ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    break;
  case 0:
    break;
  }
  z  += nv_rem;
  i  -= nv_rem;
  yy += nv_rem;

  while (i >0) {
    sum0 = 0.;
    sum1 = 0.;
    sum2 = 0.;
    sum3 = 0.;
    ierr = VecGetArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[2],&yy2);CHKERRQ(ierr);
    ierr = VecGetArrayRead(yy[3],&yy3);CHKERRQ(ierr);
    x    = xbase;

    j = n;
    switch (j_rem=j&0x3) {
    case 3:
      x2    = x[2];
      sum0 += x2*yy0[2]; sum1 += x2*yy1[2];
      sum2 += x2*yy2[2]; sum3 += x2*yy3[2];
    case 2:
      x1    = x[1];
      sum0 += x1*yy0[1]; sum1 += x1*yy1[1];
      sum2 += x1*yy2[1]; sum3 += x1*yy3[1];
    case 1:
      x0    = x[0];
      sum0 += x0*yy0[0]; sum1 += x0*yy1[0];
      sum2 += x0*yy2[0]; sum3 += x0*yy3[0];
    case 0:
      x   += j_rem;
      yy0 += j_rem;
      yy1 += j_rem;
      yy2 += j_rem;
      yy3 += j_rem;
      j   -= j_rem;
      break;
    }
    while (j>0) {
      x0 = x[0];
      x1 = x[1];
      x2 = x[2];
      x3 = x[3];
      x += 4;

      sum0 += x0*yy0[0] + x1*yy0[1] + x2*yy0[2] + x3*yy0[3]; yy0+=4;
      sum1 += x0*yy1[0] + x1*yy1[1] + x2*yy1[2] + x3*yy1[3]; yy1+=4;
      sum2 += x0*yy2[0] + x1*yy2[1] + x2*yy2[2] + x3*yy2[3]; yy2+=4;
      sum3 += x0*yy3[0] + x1*yy3[1] + x2*yy3[2] + x3*yy3[3]; yy3+=4;
      j    -= 4;
    }
    z[0] = sum0;
    z[1] = sum1;
    z[2] = sum2;
    z[3] = sum3;
    z   += 4;
    i   -= 4;
    ierr = VecRestoreArrayRead(yy[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[1],&yy1);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[2],&yy2);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(yy[3],&yy3);CHKERRQ(ierr);
    yy  += 4;
  }
  ierr = VecRestoreArrayRead(xin,&xbase);CHKERRQ(ierr);
  ierr = PetscLogFlops(PetscMax(nv*(2.0*xin->map->n-1),0.0));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode VecMax_Seq(Vec xin,PetscInt *idx,PetscReal *z)
{
  PetscInt          i,j=0,n = xin->map->n;
  PetscReal         max,tmp;
  const PetscScalar *xx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
  if (!n) {
    max = PETSC_MIN_REAL;
    j   = -1;
  } else {
    max = PetscRealPart(*xx++); j = 0;
    for (i=1; i<n; i++) {
      if ((tmp = PetscRealPart(*xx++)) > max) { j = i; max = tmp;}
    }
  }
  *z = max;
  if (idx) *idx = j;
  ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecMin_Seq(Vec xin,PetscInt *idx,PetscReal *z)
{
  PetscInt          i,j=0,n = xin->map->n;
  PetscReal         min,tmp;
  const PetscScalar *xx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
  if (!n) {
    min = PETSC_MAX_REAL;
    j   = -1;
  } else {
    min = PetscRealPart(*xx++); j = 0;
    for (i=1; i<n; i++) {
      if ((tmp = PetscRealPart(*xx++)) < min) { j = i; min = tmp;}
    }
  }
  *z = min;
  if (idx) *idx = j;
  ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecSet_Seq(Vec xin,PetscScalar alpha)
{
  PetscInt       i,n = xin->map->n;
  PetscScalar    *xx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(xin,&xx);CHKERRQ(ierr);
  if (alpha == (PetscScalar)0.0) {
    ierr = PetscMemzero(xx,n*sizeof(PetscScalar));CHKERRQ(ierr);
  } else {
    for (i=0; i<n; i++) xx[i] = alpha;
  }
  ierr = VecRestoreArray(xin,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecMAXPY_Seq(Vec xin, PetscInt nv,const PetscScalar *alpha,Vec *y)
{
  PetscErrorCode    ierr;
  PetscInt          n = xin->map->n,j,j_rem;
  const PetscScalar *yy0,*yy1,*yy2,*yy3;
  PetscScalar       *xx,alpha0,alpha1,alpha2,alpha3;

#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*xx,*yy0,*yy1,*yy2,*yy3,*alpha)
#endif

  PetscFunctionBegin;
  ierr = PetscLogFlops(nv*2.0*n);CHKERRQ(ierr);
  ierr = VecGetArray(xin,&xx);CHKERRQ(ierr);
  switch (j_rem=nv&0x3) {
  case 3:
    ierr   = VecGetArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr   = VecGetArrayRead(y[1],&yy1);CHKERRQ(ierr);
    ierr   = VecGetArrayRead(y[2],&yy2);CHKERRQ(ierr);
    alpha0 = alpha[0];
    alpha1 = alpha[1];
    alpha2 = alpha[2];
    alpha += 3;
    PetscKernelAXPY3(xx,alpha0,alpha1,alpha2,yy0,yy1,yy2,n);
    ierr = VecRestoreArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(y[1],&yy1);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(y[2],&yy2);CHKERRQ(ierr);
    y   += 3;
    break;
  case 2:
    ierr   = VecGetArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr   = VecGetArrayRead(y[1],&yy1);CHKERRQ(ierr);
    alpha0 = alpha[0];
    alpha1 = alpha[1];
    alpha +=2;
    PetscKernelAXPY2(xx,alpha0,alpha1,yy0,yy1,n);
    ierr = VecRestoreArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(y[1],&yy1);CHKERRQ(ierr);
    y   +=2;
    break;
  case 1:
    ierr   = VecGetArrayRead(y[0],&yy0);CHKERRQ(ierr);
    alpha0 = *alpha++;
    PetscKernelAXPY(xx,alpha0,yy0,n);
    ierr = VecRestoreArrayRead(y[0],&yy0);CHKERRQ(ierr);
    y   +=1;
    break;
  }
  for (j=j_rem; j<nv; j+=4) {
    ierr   = VecGetArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr   = VecGetArrayRead(y[1],&yy1);CHKERRQ(ierr);
    ierr   = VecGetArrayRead(y[2],&yy2);CHKERRQ(ierr);
    ierr   = VecGetArrayRead(y[3],&yy3);CHKERRQ(ierr);
    alpha0 = alpha[0];
    alpha1 = alpha[1];
    alpha2 = alpha[2];
    alpha3 = alpha[3];
    alpha += 4;

    PetscKernelAXPY4(xx,alpha0,alpha1,alpha2,alpha3,yy0,yy1,yy2,yy3,n);
    ierr = VecRestoreArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(y[1],&yy1);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(y[2],&yy2);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(y[3],&yy3);CHKERRQ(ierr);
    y   += 4;
  }
  ierr = VecRestoreArray(xin,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <../src/vec/vec/impls/seq/ftn-kernels/faypx.h>

PetscErrorCode VecAYPX_Seq(Vec yin,PetscScalar alpha,Vec xin)
{
  PetscErrorCode    ierr;
  PetscInt          n = yin->map->n;
  PetscScalar       *yy;
  const PetscScalar *xx;

  PetscFunctionBegin;
  if (alpha == (PetscScalar)0.0) {
    ierr = VecCopy(xin,yin);CHKERRQ(ierr);
  } else if (alpha == (PetscScalar)1.0) {
    ierr = VecAXPY_Seq(yin,alpha,xin);CHKERRQ(ierr);
  } else if (alpha == (PetscScalar)-1.0) {
    PetscInt i;
    ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
    ierr = VecGetArray(yin,&yy);CHKERRQ(ierr);

    for (i=0; i<n; i++) yy[i] = xx[i] - yy[i];

    ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&yy);CHKERRQ(ierr);
    ierr = PetscLogFlops(1.0*n);CHKERRQ(ierr);
  } else {
    ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
    ierr = VecGetArray(yin,&yy);CHKERRQ(ierr);
#if defined(PETSC_USE_FORTRAN_KERNEL_AYPX)
    {
      PetscScalar oalpha = alpha;
      fortranaypx_(&n,&oalpha,xx,yy);
    }
#else
    {
      PetscInt i;

      for (i=0; i<n; i++) yy[i] = xx[i] + alpha*yy[i];
    }
#endif
    ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&yy);CHKERRQ(ierr);
    ierr = PetscLogFlops(2.0*n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#include <../src/vec/vec/impls/seq/ftn-kernels/fwaxpy.h>
/*
   IBM ESSL contains a routine dzaxpy() that is our WAXPY() but it appears
  to be slower than a regular C loop.  Hence,we do not include it.
  void ?zaxpy(int*,PetscScalar*,PetscScalar*,int*,PetscScalar*,int*,PetscScalar*,int*);
*/

PetscErrorCode VecWAXPY_Seq(Vec win, PetscScalar alpha,Vec xin,Vec yin)
{
  PetscErrorCode     ierr;
  PetscInt           i,n = win->map->n;
  PetscScalar        *ww;
  const PetscScalar  *yy,*xx;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
  ierr = VecGetArrayRead(yin,&yy);CHKERRQ(ierr);
  ierr = VecGetArray(win,&ww);CHKERRQ(ierr);
  if (alpha == (PetscScalar)1.0) {
    ierr = PetscLogFlops(n);CHKERRQ(ierr);
    /* could call BLAS axpy after call to memcopy, but may be slower */
    for (i=0; i<n; i++) ww[i] = yy[i] + xx[i];
  } else if (alpha == (PetscScalar)-1.0) {
    ierr = PetscLogFlops(n);CHKERRQ(ierr);
    for (i=0; i<n; i++) ww[i] = yy[i] - xx[i];
  } else if (alpha == (PetscScalar)0.0) {
    ierr = PetscMemcpy(ww,yy,n*sizeof(PetscScalar));CHKERRQ(ierr);
  } else {
    PetscScalar oalpha = alpha;
#if defined(PETSC_USE_FORTRAN_KERNEL_WAXPY)
    fortranwaxpy_(&n,&oalpha,xx,yy,ww);
#else
    for (i=0; i<n; i++) ww[i] = yy[i] + oalpha * xx[i];
#endif
    ierr = PetscLogFlops(2.0*n);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(yin,&yy);CHKERRQ(ierr);
  ierr = VecRestoreArray(win,&ww);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecMaxPointwiseDivide_Seq(Vec xin,Vec yin,PetscReal *max)
{
  PetscErrorCode    ierr;
  PetscInt          n = xin->map->n,i;
  const PetscScalar *xx,*yy;
  PetscReal         m = 0.0;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
  ierr = VecGetArrayRead(yin,&yy);CHKERRQ(ierr);
  for (i = 0; i < n; i++) {
    if (yy[i] != (PetscScalar)0.0) {
      m = PetscMax(PetscAbsScalar(xx[i]/yy[i]), m);
    } else {
      m = PetscMax(PetscAbsScalar(xx[i]), m);
    }
  }
  ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(yin,&yy);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&m,max,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  ierr = PetscLogFlops(n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecPlaceArray_Seq(Vec vin,const PetscScalar *a)
{
  Vec_Seq *v = (Vec_Seq*)vin->data;

  PetscFunctionBegin;
  if (v->unplacedarray) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"VecPlaceArray() was already called on this vector, without a call to VecResetArray()");
  v->unplacedarray = v->array;  /* save previous array so reset can bring it back */
  v->array         = (PetscScalar*)a;
  PetscFunctionReturn(0);
}

PetscErrorCode VecReplaceArray_Seq(Vec vin,const PetscScalar *a)
{
  Vec_Seq        *v = (Vec_Seq*)vin->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(v->array_allocated);CHKERRQ(ierr);
  v->array_allocated = v->array = (PetscScalar*)a;
  PetscFunctionReturn(0);
}
