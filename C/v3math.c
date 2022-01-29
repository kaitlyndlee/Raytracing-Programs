#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include "v3math.h"

const float PI = 3.141592654;

/**
 * Sets one v3 vector values to another v3 vector
 *
 * @param vector_a - v3 vector to get the values
 * @param vector_b - v3 vector giving the values
 */
void v3_set_points(float *vector_a, float *vector_b) {
  for (int index = 0; index < 3; index++) {
    vector_a[index] = vector_b[index];
  }
}

/**
 * Creates a 3D vector from point a to b.
 *
 * @param dst - result vector pointer
 * @param a - pointer of the first vector
 * @param b - pointer of the second vector 
 */
void v3_from_points(float *dst, float *a, float *b) {
  v3_subtract(dst, b, a);
}

/**
 * Adds the vectors a and b and stores the output in dst.
 *
 * @param dst - result vector pointer
 * @param a - pointer of the first vector
 * @param b - pointer of the second vector 
 */
void v3_add(float *dst, float *a, float *b) {
  dst[0] = a[0] + b[0];
  dst[1] = a[1] + b[1];
  dst[2] = a[2] + b[2];
}

/**
 * Subtracts the vectors a and b and stores the output in dst.
 *
 * @param dst - result vector pointer
 * @param a - pointer of the first vector
 * @param b - pointer of the second vector 
 */
void v3_subtract(float *dst, float *a, float *b) {
  dst[0] = a[0] - b[0];
  dst[1] = a[1] - b[1];
  dst[2] = a[2] - b[2];
}

/**
 * Calculates the dot product of vectors a and b.
 *
 * @param a - pointer of the first vector
 * @param b - pointer of the second vector 
 * @return - returns the dot product. 
 */
float v3_dot_product(float *a, float *b) {
  return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2]);
}
/**
 * Computes the cross product of the vectors given as parameters and stores the
 * computed vector in the destination varaible
 * 
 * @param dst - result vector pointer
 * @param a - pointer of the first vector
 * @param b - pointer of the second vector
 */
void v3_cross_product(float *dst, float *a, float *b) {
  float product[3];

  product[0] = (a[1]*b[2]) - (a[2]*b[1]);
  product[1] = (a[2]*b[0]) - (a[0]*b[2]);
  product[2] = (a[0]*b[1]) - (a[1]*b[0]);

  for (int i = 0; i < 3; i++) {
    dst[i] = product[i];
  }
}

/**
 * Scales a vector by a scalar.
 *
 * @param dst - result vector pointer and vector to scale.
 * @param s - scalar to scale dst by.
 */
void v3_scale(float *dst, float s) {
  dst[0] = dst[0] * s;
  dst[1] = dst[1] * s;
  dst[2] = dst[2] * s;
}

/**
 * Calculates the angle between two vectors. Returns value in radians.
 * 
 * @param a - pointer of the first vector
 * @param b - pointer of the second vector 
 * @return - returns the angle in radians. 
 */
float v3_angle( float *a, float *b) {
  float angle = v3_angle_quick(a,b);
  return acos(angle); 
}

/**
 * Calculates the angle between two vectors. Returns the raw value without
 * taking the arcosine of the value.
 * 
 * @param a - pointer of the first vector
 * @param b - pointer of the second vector 
 * @return - returns the raw value for computing the angle between two vectors. 
 */
float v3_angle_quick(float *a, float *b) {
  float length_a = v3_length(a);
  float length_b = v3_length(b);

  if (length_a == 0.0 || length_b == 0.0)
  {
    fprintf(stderr, "v3_length returned 0, exiting program\n");
    return -1;
  }

  return v3_dot_product(a,b) / (length_a * length_b);
}

/**
 * Reflects the vector given as the parameter a over n and stores the result in the destination pointer
 * 
 * @param dst - result vector pointer
 * @param v - pointer of the vector to relflect
 * @param n - pointer of the vector that is the surface
 */
void v3_reflect(float *dst, float *v, float *n) {
  float product[3];

  for (int i = 0; i < 3; i++) {
    product[i] = n[i];
  }

  // v3_normalize(n, n);

  v3_scale(product, (2 * v3_dot_product(v,n)));

  v3_subtract(product, v, product);

  for (int i = 0; i < 3; i++) {
    dst[i] = product[i];
  }
}

/**
 * Calculates the length or magnitude of a vector and returns it as a float.
 * This function will return -1 to avoid divide by zero issue in other functions.
 * 
 * @param a - pointer of the vector 
 * @return - returns the length of the given vector. 
 */
float v3_length(float *a) {
  float length  = sqrt(pow(a[0], 2)+pow(a[1], 2)+pow(a[2], 2));
  return length;
}

/**
 * Normalizes the vector given as the parameter a and stores the result in the destination pointer
 * 
 * @param dst - result vector pointer
 * @param a - pointer of the vector to normalize 
 */
void v3_normalize(float *dst, float *a) {
  float length = v3_length(a);

  if (length == 0.0)
  {
    fprintf(stderr, "v3_length returned 0, exiting program\n");
    return;
  }

  dst[0] = a[0]/length;
  dst[1] = a[1]/length;
  dst[2] = a[2]/length;
}

/**
 * Multiplies a 4D matrix by a column vector of size 4.
 *
 * @param dst - result vector pointer
 * @param matrix - pointer of the 4D matrix
 * @param a - pointer of the column vector. 
 */
void v4_matrix_multiply(float *dst, float *matrix, float *a) {
  float temp_a[4];
  for (int i = 0; i < 4; i++) {
    temp_a[i] = a[i];
  }

  float temp_matrix[16];
  for (int i = 0; i < 16; i++) {
    temp_matrix[i] = matrix[i];
  }

  for (int row = 0; row < 4; row++) {
    for (int col = 0; col < 4; col++) {
      dst[row] += temp_matrix[4 * row + col] * temp_a[col];
    }
  }
}

/**
 * Checks the equivalency of two vectors by using a tolerance.
 *
 * @param a - pointer of the first vector
 * @param b - pointer of the second vector 
 * @param tolerance - tolerance value.
 * @return - returns if the absolute value of the difference between the two vectors is 
 *           less than the tolerance.
 */
bool v3_equals(float *a, float *b, float tolerance) {
  for (int i = 0; i < 3; i++) {
    if (fabs(a[i] - b[i]) > tolerance) {
      printf("%d: %f\n", i, fabs(a[i] - b[i]));
      return false;
    }
  }
  return true;
}