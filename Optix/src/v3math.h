#ifndef v3math_h
#define v3math_h

#include "cuda_defs.h"

// CUDA_CALLABLE_MEMBER float PI = 3.141592654;

#define PI 3.141592654

CUDA_CALLABLE_MEMBER void v3_set_points(float *vector_a, float *vector_b);
CUDA_CALLABLE_MEMBER void v3_from_points(float *dst, float *a, float *b);
CUDA_CALLABLE_MEMBER void v3_add(float *dst, float *a, float *b);
CUDA_CALLABLE_MEMBER void v3_subtract(float *dst, float *a, float *b);
CUDA_CALLABLE_MEMBER float v3_dot_product(float *a, float *b);
CUDA_CALLABLE_MEMBER void v3_cross_product(float *dst, float *a, float *b);
CUDA_CALLABLE_MEMBER void v3_scale(float *dst, float s);
CUDA_CALLABLE_MEMBER float v3_angle(float *a, float *b);
CUDA_CALLABLE_MEMBER float v3_angle_quick(float *a, float *b);
CUDA_CALLABLE_MEMBER void v3_reflect(float *dst, float *v, float *n);
CUDA_CALLABLE_MEMBER float v3_length(float *a);
CUDA_CALLABLE_MEMBER void v3_normalize(float *dst, float *a);
CUDA_CALLABLE_MEMBER void v4_matrix_multiply(float *dst, float *matrix, float *a);
CUDA_CALLABLE_MEMBER bool v3_equals(float *a, float *b, float tolerance);

#endif
