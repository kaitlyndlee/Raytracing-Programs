#ifndef raycast_h
#define raycast_h

#include "ppm.h"
#include "objects.h"
#include "parse.h"

void raycast(json_data_t *json_struct, PPMFormat photo_data);
bool ray_intersect_sphere(float *ray_o, float *ray_d, shape_t *sphere, float *distance);
bool ray_intersect_plane(float *ray_o, float *ray_d, shape_t *plane, float *distance);
bool ray_intersect_quadric(float *ray_o, float *ray_d, shape_t *quadric, float *distance);
float radial_attenuation (light_t *light ,float distance);
float angular_attenuation(light_t *light, float *object_point);
void diffuse_light(float *return_color, light_t *light, float *object_color, 
                   float *surface_normal, float *intersection);
void specular_light(float *return_color, light_t *light, float *object_color, float *surface_normal, 
                    float *intersection, float *view);
int find_nearest_object(json_data_t *json_struct, shape_t **nearest_object, 
                        float *ray_orgin, float *ray_direction, 
                        float *intsection, float *normal, int skip_index, bool shadow_test);
void shoot(json_data_t *json_struct, float *ray_orgin, float *ray_direction, float *out_color, 
          int recursion_level, int skip_index);
void iterative_shoot(json_data_t *json_struct, float *ray_orgin, float *ray_direction, float *out_color, int skip_index);
void calc_color(json_data_t *json_struct, float *out_color, int object_index, float *intersection, float *normal, float *ray_direction);

#endif