#ifndef raycast_h
#define raycast_h

#include "ppm.h"
#include "objects.h"
#include "parse.h"

__host__ void raycast(json_data_t *json_struct, PPMFormat *photo_data);
__device__  bool ray_intersect_sphere(float *ray_o, float *ray_d, shape_t *sphere, float *distance);
__device__  bool ray_intersect_plane(float *ray_o, float *ray_d, shape_t *plane, float *distance);
__device__  bool ray_intersect_quadric(float *ray_o, float *ray_d, shape_t *quadric, float *distance);
__device__  float radial_attenuation (light_t *light ,float distance);
__device__  float angular_attenuation(light_t *light, float *object_point);
__device__  void diffuse_light(float *return_color, light_t *light, float *object_color, 
                   float *surface_normal, float *intersection);
__device__  void specular_light(float *return_color, light_t *light, float *object_color, float *surface_normal, 
                    float *intersection, float *view);
__device__  int find_nearest_object(json_data_t *json_struct, shape_t **nearest_object, 
                        float *ray_origin, float *ray_direction, 
                        float *intsection, float *normal, int skip_index, bool shadow_test);
__global__ void raytrace_engine(float *pixel_height, float *pixel_width, json_data_t *json_struct, 
                                PPMFormat *photo_data);
__device__ void iterative_shoot(json_data_t *json_struct, float *ray_origin, float *ray_direction, 
                                float *out_color, int skip_index);
__device__ void calc_color(json_data_t *json_struct, float *out_color, int object_index, 
                           float *intersection, float *normal, float *ray_direction);

#endif