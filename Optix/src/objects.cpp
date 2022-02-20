#include "objects.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * Creates and adds a new sphere to the shapes list.
 *
 * @param shape_list - list to add shape to
 * @param num_shapes - number of shapes already in the list
 * @param color - array storing the color of the quadric
 * @param position - (x, y, z) coordinate of a point on the plane
 * @param specular - variable specular to set the specular coefficient of the object
 * @param diffuse - variable diffuse to set the diffuse coefficient of the object
 * @param radius - variable radius to set the radius of the object
 * @param reflectivity - variable reflectivity to set the reflectiveness of the object
 * @param refractivity - variable refractivity to set the refractivity of the object
 * @param ior - variable ior to set the refractive index of the object
 */
void add_new_sphere(shape_t *shape_list,
                    unsigned int num_shapes,
                    float *diffuse,
                    float *specular,
                    float *position,
                    float radius,
                    float reflectivity,
                    float refractivity,
                    float ior) {

  for (int i = 0; i < 3; i++) {
    shape_list[num_shapes].diffuse_color[i] = diffuse[i];
    shape_list[num_shapes].specular_color[i] = specular[i];
    shape_list[num_shapes].position[i] = position[i];
  }
  shape_list[num_shapes].reflectivity = reflectivity;
  shape_list[num_shapes].refractivity = refractivity;
  shape_list[num_shapes].ior = ior;
  shape_list[num_shapes].radius = radius;
  shape_list[num_shapes].type = SPHERE;
}

/**
 * Creates and adds a new plane to the shapes list.
 *
 * @param shape_list - list to add shape to
 * @param num_shapes - number of shapes already in the list
 * @param specular - variable specular to set the specular coefficient of the object
 * @param diffuse - variable diffuse to set the diffuse coefficient of the object
 * @param color - array storing the color of the quadric
 * @param position - (x, y, z) coordinate of a point on the plane
 * @param normal - a v3 vector representing the normal of the plane
 * @param reflectivity - variable reflectivity to set the reflectiveness of the object
 */
void add_new_plane(shape_t *shape_list,
                   unsigned int num_shapes,
                   float *diffuse,
                   float *specular,
                   float *position,
                   float *normal,
                   float reflectivity) {

  for (int i = 0; i < 3; i++) {
    shape_list[num_shapes].diffuse_color[i] = diffuse[i];
    shape_list[num_shapes].specular_color[i] = specular[i];
    shape_list[num_shapes].position[i] = position[i];
    shape_list[num_shapes].normal[i] = normal[i];
  }
  shape_list[num_shapes].reflectivity = reflectivity;
  shape_list[num_shapes].refractivity = 0;
  shape_list[num_shapes].ior = 1;
  shape_list[num_shapes].type = PLANE;
}

/**
 * Creates and adds a new quadric to the shapes list.
 *
 * @param shape_list - list to add shape to
 * @param num_shapes - number of shapes already in the list
 * @param specular - variable specular to set the specular coefficient of the object
 * @param diffuse - variable diffuse to set the diffuse coefficient of the object
 * @param color - array storing the color of the quadric
 * @param a - variable a to be plugged into the quadric equation
 * @param b - variable b to be plugged into the quadric equation
 * @param c - variable c to be plugged into the quadric equation
 * @param d - variable d to be plugged into the quadric equation
 * @param e - variable e to be plugged into the quadric equation
 * @param f - variable f to be plugged into the quadric equation
 * @param g - variable g to be plugged into the quadric equation
 * @param h - variable h to be plugged into the quadric equation
 * @param i - variable i to be plugged into the quadric equation
 * @param j - variable j to be plugged into the quadric equation
 * @param reflectivity - variable reflectivity to set the reflectiveness of the object
 */
void add_new_quadric(shape_t *shape_list,
                     unsigned int num_shapes,
                     float *diffuse,
                     float *specular,
                     float a,
                     float b,
                     float c,
                     float d,
                     float e,
                     float f,
                     float g,
                     float h,
                     float i,
                     float j,
                     float reflectivity) {

  for (int count = 0; count < 3; count++) {
    shape_list[num_shapes].diffuse_color[count] = diffuse[count];
    shape_list[num_shapes].specular_color[count] = specular[count];
  }
  shape_list[num_shapes].reflectivity = reflectivity;
  shape_list[num_shapes].refractivity = 0;
  shape_list[num_shapes].ior = 1;
  shape_list[num_shapes].a = a;
  shape_list[num_shapes].b = b;
  shape_list[num_shapes].c = c;
  shape_list[num_shapes].d = d;
  shape_list[num_shapes].e = e;
  shape_list[num_shapes].f = f;
  shape_list[num_shapes].g = g;
  shape_list[num_shapes].h = h;
  shape_list[num_shapes].i = i;
  shape_list[num_shapes].j = j;
  shape_list[num_shapes].type = QUADRIC;
}

/**
 * Creates and adds a new point light to the lights list.
 *
 * @param light_list - list to add light to
 * @param num_shapes - number of light already in the list
 * @param color - array storing the color of the quadric
 * @param position - (x, y, z) coordinate of a point on the plane
 * @param radial_coef - a v3 vector representing the radial coefficent of the light
 */
void add_new_point_light(light_t *light_list,
                         unsigned int num_lights,
                         float *color,
                         float *position,
                         float *radial_coef) {
  for (int i = 0; i < 3; i++) {
    light_list[num_lights].color[i] = color[i];
    light_list[num_lights].position[i] = position[i];
    light_list[num_lights].radial_coef[i] = radial_coef[i];
  }
  light_list[num_lights].type = POINT;
}

/**
 * Creates and adds a new spotlight to the lights list.
 *
 * @param light_list - list to add light to
 * @param num_shapes - number of light already in the list
 * @param color - array storing the color of the quadric
 * @param position - (x, y, z) coordinate of a point on the plane
 * @param theta - float storing the theta value of the light
 * @param a0 - float storing the a0 value
 * @param direction - a v3 vector representing the direction of the light
 * @param radial_coef - a v3 vector representing the radial coefficent of the light
 */
void add_new_spot_light(light_t *light_list,
                        unsigned int num_lights,
                        float *color,
                        float *position,
                        float theta,
                        float a0,
                        float *direction,
                        float *radial_coef) {
  for (int i = 0; i < 3; i++) {
    light_list[num_lights].color[i] = color[i];
    light_list[num_lights].position[i] = position[i];
    light_list[num_lights].direction[i] = direction[i];
    light_list[num_lights].radial_coef[i] = radial_coef[i];
  }
  light_list[num_lights].a0 = a0;
  light_list[num_lights].theta = theta;
  light_list[num_lights].cos_theta = cos(theta);
  light_list[num_lights].type = SPOTLIGHT;
}