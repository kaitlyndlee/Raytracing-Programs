#include "objects.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h> 

/**
 * Creates and adds a new sphere to the shapes linked list.
 *
 * @param head - the head of the linked list
 * @param color - array storing the color of the quadric
 * @param position - (x, y, z) coordinate of a point on the plane
 * @param specular - variable specular to set the specular coefficient of the object
 * @param diffuse - variable diffuse to set the diffuse coefficient of the object
 * @param radius - variable radius to set the radius of the object
 * @param reflectivity - variable reflectivity to set the reflectiveness of the object
 * @param refractivity - variable refractivity to set the refractivity of the object
 * @param ior - variable ior to set the refractive index of the object
 */
shape_t *add_new_sphere(shape_t *head, float *diffuse, float *specular, float *position, float radius, 
                        float reflectivity, float refractivity, float ior) {

  if (head == NULL) {
    head = (shape_t *) malloc(sizeof(shape_t));
    for (int i = 0; i < 3; i++) {
      head->diffuse_color[i] = diffuse[i];
      head->specular_color[i] = specular[i];
      head->position[i] = position[i];
    }
    head->reflectivity = reflectivity;
    head->refractivity = refractivity;
    head->ior = ior;
    head->radius = radius;
    head->type = SPHERE;
    head->next = NULL;
    return head;
  }


  head->next = add_new_sphere(head->next, diffuse, specular, position, radius, reflectivity, refractivity, ior);
  return head;

}

/**
 * Creates and adds a new plane to the shapes linked list.
 *
 * @param head - the head of the linked list
 * @param specular - variable specular to set the specular coefficient of the object
 * @param diffuse - variable diffuse to set the diffuse coefficient of the object
 * @param color - array storing the color of the quadric
 * @param position - (x, y, z) coordinate of a point on the plane
 * @param normal - a v3 vector representing the normal of the plane
 * @param reflectivity - variable reflectivity to set the reflectiveness of the object
 */
shape_t *add_new_plane(shape_t *head, float *diffuse, float *specular, float *position, 
                       float *normal, float reflectivity) {
  if (head == NULL) {
    head = (shape_t *) malloc(sizeof(shape_t));
    for (int i = 0; i < 3; i++) {
      head->diffuse_color[i] = diffuse[i];
      head->specular_color[i] = specular[i];
      head->position[i] = position[i];
      head->normal[i] = normal[i];
    }
    head->reflectivity = reflectivity;
    head->refractivity = 0;
    head->ior = 1;
    head->type = PLANE;
    head->next = NULL;
    return head;
  }

  head->next = add_new_plane(head->next, diffuse, specular, position, normal, reflectivity);
  return head;
}

/**
 * Creates and adds a new quadric to the shapes linked list.
 *
 * @param head - the head of the linked list
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
shape_t *add_new_quadric(shape_t *head, float *diffuse, float *specular, float a, 
                         float b, float c, float d, float e, float f, float g, 
                         float h, float i, float j, float reflectivity) {
  if (head == NULL) {
    head = (shape_t *) malloc(sizeof(shape_t));

    for (int i = 0; i < 3; i++) {
      head->diffuse_color[i] = diffuse[i];
      head->specular_color[i] = specular[i];
    }
    head->reflectivity = reflectivity;
    head->refractivity = 0;
    head->ior = 1;
    head->a = a;
    head->b = b;
    head->c = c;
    head->d = d;
    head->e = e;
    head->f = f;
    head->g = g;
    head->h = h;
    head->i = i;
    head->j = j;
    head->type = QUADRIC;
    head->next = NULL;
    return head;
  }

  head->next = add_new_quadric(head->next, diffuse, specular, a, b, c, d, e, f, g, h, i, j, reflectivity);
  return head;
}


/**
 * Creates and adds a new point light to the lights linked list.
 *
 * @param head - the head of the linked list
 * @param color - array storing the color of the quadric
 * @param position - (x, y, z) coordinate of a point on the plane
 * @param radial_coef - a v3 vector representing the radial coefficent of the light
 */
light_t *add_new_point_light(light_t *head, float *color, float *position, float *radial_coef) {
  if (head == NULL) {
    head = (light_t *) malloc(sizeof(light_t));
    for (int i = 0; i < 3; i++) {
      head->color[i] = color[i];
      head->position[i] = position[i];
      head->radial_coef[i] = radial_coef[i];
    }
    head->type = POINT;
    head->next = NULL;
    return head;
  }

  head->next = add_new_point_light(head->next, color, position, radial_coef);
  return head;
}


/**
 * Creates and adds a new spotlight to the lights linked list.
 *
 * @param head - the head of the linked list
 * @param color - array storing the color of the quadric
 * @param position - (x, y, z) coordinate of a point on the plane
 * @param theta - float storing the theta value of the light
 * @param a0 - float storing the a0 value
 * @param direction - a v3 vector representing the direction of the light
 * @param radial_coef - a v3 vector representing the radial coefficent of the light
 */
light_t *add_new_spot_light(light_t *head, float *color, float *position, 
                             float theta, float a0, float *direction, float *radial_coef) {
  if (head == NULL) {
    head = (light_t *) malloc(sizeof(light_t));
    for (int i = 0; i < 3; i++) {
      head->color[i] = color[i];
      head->position[i] = position[i];
      head->direction[i] = direction[i];
      head->radial_coef[i] = radial_coef[i];
    }
    head->a0 = a0;
    head->theta = theta;
    head->cos_theta = cos(theta);
    head->type = SPOTLIGHT;
    head->next = NULL;
    return head;
  }

  head->next = add_new_spot_light(head->next, color, position, theta, a0, direction, radial_coef);
  return head;
}


/**
 * Frees each shape in the shapes linked list.
 *
 * @param head - the head of the linked list
 * @return - returns NULL to set the head to NULL
 */
shape_t *free_shape_list(shape_t *head) {
  if (head != NULL) {
    if (head->next != NULL) {
        free_shape_list(head->next);
    }
    free(head);
  }
  return NULL;
}


/**
 * Frees each light in the light linked list.
 *
 * @param head - the head of the linked list
 * @return - returns NULL to set the head to NULL
 */
light_t *free_light_list(light_t *head) {
  if (head != NULL) {
    if (head->next != NULL) {
        free_light_list(head->next);
    }
    free(head);
  }
  return NULL;
}