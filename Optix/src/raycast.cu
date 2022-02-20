#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <errno.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>

#include "raycast.h"
#include "v3math.h"

#define MAX_ITER 50
#define MAX_SHAPES 128
#define tx 128
#define ty 128

/**
 * Runs the Raytracer.
 */
int main(int argc, char **argv) {
  if (argc != 5) {
    printf("Usage: raytrace WIDTH HEIGHT INPUT_SCENE OUTPUT_IMAGE\n" );
    exit(0);
  }

  struct timeval start, end;
  gettimeofday(&start, NULL);

  int width = atoi(argv[1]);
  int height = atoi(argv[2]);

  FILE *input_json = fopen(argv[3], "r+");
  if(input_json == NULL) {
    fprintf(stderr, "Error: Unable to open the input scene file: %s\n", argv[3]);
    exit(1);
  }

  FILE *output_image = fopen(argv[4], "wb");
  if(output_image == NULL) {
    fprintf(stderr, "Error: Unable to open the output image file: %s\n", argv[4]);
    exit(1);
  }
  
  json_data_t *json_struct;
  cudaMallocManaged(&json_struct, sizeof(json_data_t));
  cudaMallocManaged(&json_struct->shapes_list, MAX_SHAPES * sizeof(shape_t));
  cudaMallocManaged(&json_struct->lights_list, MAX_SHAPES * sizeof(light_t));
  parse_json(input_json, json_struct);

  PPMFormat *photo_data;
  cudaMallocManaged(&photo_data, sizeof(photo_data));

  photo_data->maxColor = 255;
  photo_data->height = height;
  photo_data->width = width;
  photo_data->size = photo_data->width * photo_data->height * 3;
  cudaMallocManaged(&(photo_data->pixmap), photo_data->size);

  float pixel_height = json_struct->camera_height / photo_data->height;
  float pixel_width = json_struct->camera_width / photo_data->width;

  float *dev_pixel_height;
  float *dev_pixel_width;

  cudaMalloc( (void**) &dev_pixel_height, sizeof(float) );
  cudaMalloc( (void**) &dev_pixel_width, sizeof(float) );

  cudaMemcpy(dev_pixel_height, &pixel_height, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_pixel_width, &pixel_width, sizeof(float), cudaMemcpyHostToDevice);

  dim3 num_blocks(photo_data->width/tx+1, photo_data->height/ty+1);
  dim3 threads_per_block(tx, ty);
 
  raytrace_engine<<<num_blocks, threads_per_block>>>(dev_pixel_height, dev_pixel_width, json_struct, photo_data);
  cudaDeviceSynchronize();

  ppm_WriteOutP3(*photo_data, output_image);
  
  gettimeofday(&end, NULL);
  double elapsed = (((end.tv_sec*1000000.0 + end.tv_usec) -
                        (start.tv_sec*1000000.0 + start.tv_usec)) / 1000000.00);
  printf("Time (sec) to create a %dx%d image with %d shape(s) and %d light(s): %f\n", width, height, 
                                        json_struct->num_shapes, json_struct->num_lights, elapsed);
  cudaFree(json_struct);
  cudaFree(dev_pixel_height);
  cudaFree(dev_pixel_width);
  cudaFree(photo_data->pixmap);
  cudaFree(photo_data);
}

/**
 * Calculates the ray to shoot through each pixel from the camera and sets the color
 * for each pixel in the image. 1 thread == 1 pixel. Puts the json_data struct in shared
 * memory for faster access.
 *
 * @param pixel_height - height of each pixel
 * @param pixel_width - width of each pixel
 * @param json_struct - Struct holding all the shape information
 * @param photo_data - Struct holding all the image informaton
 */
__global__ void raytrace_engine(float *pixel_height, float *pixel_width, 
                                json_data_t *json_struct, PPMFormat *photo_data) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  unsigned int tid = threadIdx.y*tx + threadIdx.x;
  __shared__ json_data_t json_data;
  __shared__ shape_t shapes_list[MAX_SHAPES];
  __shared__ light_t lights_list[MAX_SHAPES];
  json_data.shapes_list = shapes_list;
  json_data.lights_list = lights_list;

  if(tid == 0) {
    json_data.camera_width = json_struct->camera_width;
  }
  else if (tid == 1) {
    json_data.camera_height = json_struct->camera_height;
  }
  else if (tid == 2) {
    json_data.num_shapes = json_struct->num_shapes;
  }
  else if (tid == 3) {
    json_data.num_lights = json_struct->num_lights;
  }
  else {
    int total = json_data.num_lights + json_data.num_shapes;
    // Offset by 4
    if (tid < json_data.num_shapes + 4) {
      json_data.shapes_list[tid - 4] = json_struct->shapes_list[tid - 4];
    }
    else if(tid >= json_data.num_shapes + 4 && tid < total + 4) {
      json_data.lights_list[tid - json_data.num_shapes - 4] = json_struct->lights_list[tid - json_data.num_shapes - 4];
    }
  }
  __syncthreads();

  if (x < photo_data->width && y < photo_data->height) {    
    int view_plane_center[3] = {0, 0, -1};
    float ray_origin[3] = {0, 0, 0};
    float ray_direction[3];
    float color[3];

    ray_direction[0] = view_plane_center[0] - (json_data.camera_width / 2.0) + *pixel_width * (x + 0.5);
    ray_direction[1] = view_plane_center[1] + (json_data.camera_height / 2.0) - *pixel_height * (y + 0.5);
    ray_direction[2] = view_plane_center[2];

    v3_normalize(ray_direction, ray_direction);

    iterative_shoot(&json_data, ray_origin, ray_direction, color, -1);

    v3_scale(color, 255);
    photo_data->pixmap[(y*photo_data->width + x)*3] = ppm_clamp(color[0], 0, 255);
    photo_data->pixmap[(y*photo_data->width + x)*3 + 1] = ppm_clamp(color[1], 0, 255);
    photo_data->pixmap[(y*photo_data->width + x)*3 + 2] = ppm_clamp(color[2], 0, 255);
  }
}

/**
 * Casts a ray with the given ray direction and ray origin and calculates
 * the given color given the nearest object intersection.
 *
 * @param json_struct - json struct holding the shape list
 * @param ray_origin - ray ray_origin vector
 * @param ray_direction - ray direction vector
 * @param out_color - The calculated output color for each pixel
 * @param skip_index - index of object we previously intersected
 */
__device__ void iterative_shoot(json_data_t *json_struct, float *ray_origin, float *ray_direction, 
                               float *out_color, int skip_index) {
  shape_t *nearest_object = NULL;
  int nearest_object_index = -1;
  float intersection[3];
  float normal[3];

  set_to_black(out_color);
  nearest_object_index = -1;
  nearest_object_index = find_nearest_object(json_struct, &nearest_object, ray_origin, 
                                           ray_direction, intersection, normal, skip_index, false);

  if (nearest_object == NULL) {
    set_to_black(out_color);
    return;
  }

  shape_t *next_nearest_object = nearest_object;
  float next_ray_direction[3];
  v3_set_points(next_ray_direction, ray_direction);
  float next_ray_origin[3];
  v3_set_points(next_ray_origin, intersection);
  int next_skip_index = nearest_object_index;
  int next_nearest_object_index;
  float next_intersecion[3];
  float next_normal[3];
  v3_set_points(next_normal, normal);

  float reflection_color[3]; 
  set_to_black(reflection_color);
  float reflection_vector[3];
  float color[3];
  float total_refl = nearest_object->reflectivity;

  int iter = 0;
  while(iter < MAX_ITER)
  {
    // calculate reflection color
    if (next_nearest_object->reflectivity > 0) {
      v3_reflect(reflection_vector, next_ray_direction, next_normal);
      v3_normalize(reflection_vector, reflection_vector);
      v3_set_points(next_ray_direction, reflection_vector);
      next_nearest_object_index = -1;
      next_nearest_object = NULL;
      next_nearest_object_index = find_nearest_object(json_struct, &next_nearest_object, next_ray_origin, 
                                             next_ray_direction, next_intersecion, next_normal, next_skip_index, false);
      if (next_nearest_object != NULL) {
        calc_color(json_struct, color, next_nearest_object_index, next_intersecion, next_normal, next_ray_direction);
        v3_scale(color, total_refl);
        total_refl *= next_nearest_object->reflectivity;
        v3_add(out_color, out_color, color);
        v3_set_points(next_ray_origin, next_intersecion);
        next_skip_index = next_nearest_object_index;
        iter++;
      }
      else {
        break;
      }     
    }
    else {
      break;
    }   
  }
  calc_color(json_struct, color, nearest_object_index, intersection, normal, ray_direction);
  v3_add(out_color, out_color, color);
}

/**
 * Calculates the color of an object at a point of intersection.
 * Considers the diffuse and specular color, radial and angular attenuation,
 * and if the shape is in the shadow of another shape.
 *
 * @param json_struct - json struct holding the shape list
 * @param out_color - The calculated output color for each pixel
 * @param object_index - index of object to find the color of
 * @param intersection_point - v3 vector to store nearest objects instersection_point value
 * @param normal - v3 vector to store nearest objects normal value
 * @param ray_direction - ray direction vector
 */
__device__ void calc_color(json_data_t *json_struct, float *out_color, int object_index, 
                           float *intersection, float *normal, float *ray_direction) {
  shape_t object = json_struct->shapes_list[object_index];
  float opacity = 1.0 - object.reflectivity - object.refractivity;
  set_to_black(out_color);
  if (opacity > 0.0) {
    shape_t *obj_intersection;
    float light_obj_dist;
    float light_direction[3];
    float rad_atten;
    float ang_atten;
    float diffuse_output[3];
    float specular_output[3];

    for (int lights = 0; lights < json_struct->num_lights; lights++) { 
      v3_from_points(light_direction, intersection, json_struct->lights_list[lights].position);
      light_obj_dist = v3_length(light_direction);
      v3_normalize(light_direction, light_direction);
  
      obj_intersection = NULL;
      find_nearest_object(json_struct, &obj_intersection, intersection, 
                      light_direction, NULL, NULL, object_index, true); 

      // Not in the shadow of another shape
      if (obj_intersection == NULL) {
        rad_atten = radial_attenuation(&json_struct->lights_list[lights], light_obj_dist);
        ang_atten = angular_attenuation(&json_struct->lights_list[lights], intersection);
        diffuse_light(diffuse_output, &json_struct->lights_list[lights], object.diffuse_color, 
                  normal, light_direction);

        specular_light(specular_output, &json_struct->lights_list[lights], object.specular_color, 
                   normal, light_direction, ray_direction);
  
        out_color[0] += (diffuse_output[0] + specular_output[0]) * rad_atten * ang_atten;
        out_color[1] += (diffuse_output[1] + specular_output[1]) * rad_atten * ang_atten;
        out_color[2] += (diffuse_output[2] + specular_output[2]) * rad_atten * ang_atten;
      }
    }
    v3_scale(out_color, opacity);
  }
}

/**
 * Finds the closest object that is insterescted by a ray. 
 * If skip_index != -1, skip the index of the object so that 
 * the object does not intersect with itself.
 * If skip_index == -1, don't set the shape's intersection or normal.
 *
 * @param json_struct - json struct holding the shape list
 * @param nearest_object - shape struct to hold the nearest intersected object
 * @param distance - a float to store nearest objects distance from ray
 * @param intersection_point - v3 vector to store nearest objects instersection_point value
 * @param normal - v3 vector to store nearest objects normal value
 * @param ray_origin - ray ray_origin vector
 * @param ray_direction - ray direction vector
 * @param skip_index - index of object we previously intersected
 * 
 * @returns - returns index of the nearest instersected object.
 */
__device__ int find_nearest_object(json_data_t *json_struct, shape_t **nearest_object, 
                        float *ray_origin, float *ray_direction, 
                        float *intersection, float *normal, int skip_index, bool shadow_test) {
  float nearest_distance = INFINITY;
  int object_index = -1;
  float distance;


  for (int count = 0; count < json_struct->num_shapes; count++) {    
    if (skip_index == count) { // Skip checking if a shape intersects with itself
      continue;
    }

    if (json_struct->shapes_list[count].type == SPHERE) {
      if (ray_intersect_sphere(ray_origin, ray_direction, &json_struct->shapes_list[count], &distance)) {
        if (nearest_distance > distance && distance > 0) {
          nearest_distance = distance;
          *nearest_object = &json_struct->shapes_list[count];
          object_index = count;
          if (!shadow_test) {
            intersection[0] = ray_origin[0] + ray_direction[0] * nearest_distance;
            intersection[1] = ray_origin[1] + ray_direction[1] * nearest_distance;
            intersection[2] = ray_origin[2] + ray_direction[2] * nearest_distance;

            float temp = 1.0 / json_struct->shapes_list[count].radius;
            normal[0] = (intersection[0] - json_struct->shapes_list[count].position[0]) * temp;
            normal[1] = (intersection[1] - json_struct->shapes_list[count].position[1]) * temp;
            normal[2] = (intersection[2] - json_struct->shapes_list[count].position[2]) * temp;
            v3_normalize(normal, normal);
          }
        }
      }
    }
    else if (json_struct->shapes_list[count].type == PLANE) {
      if (ray_intersect_plane(ray_origin, ray_direction, &json_struct->shapes_list[count], &distance)) {
        if (nearest_distance > distance && distance > 0) {
          nearest_distance = distance;
          *nearest_object = &json_struct->shapes_list[count];
          object_index = count;
          if (!shadow_test) {
            intersection[0] = ray_origin[0] + ray_direction[0] * nearest_distance;
            intersection[1] = ray_origin[1] + ray_direction[1] * nearest_distance;
            intersection[2] = ray_origin[2] + ray_direction[2] * nearest_distance;
            v3_set_points(normal, json_struct->shapes_list[count].normal);
          }
        }
      }
    }
    else if (json_struct->shapes_list[count].type == QUADRIC) {
      if (ray_intersect_quadric(ray_origin, ray_direction, &json_struct->shapes_list[count], &distance)) {
        // If the point is in front of the quadric.
        if (skip_index != -1 && (ray_origin[2] + distance * ray_direction[2]) < ray_origin[2]) {
          continue;
          }
        if (nearest_distance > distance && distance > 0) {
          nearest_distance = distance;
          *nearest_object = &json_struct->shapes_list[count];
          object_index = count;
          if (!shadow_test) {
            intersection[0] = ray_origin[0] + ray_direction[0] * nearest_distance;
            intersection[1] = ray_origin[1] + ray_direction[1] * nearest_distance;
            intersection[2] = ray_origin[2] + ray_direction[2] * nearest_distance;

            normal[0] = 2.0 * json_struct->shapes_list[count].a * intersection[0] 
                                     + json_struct->shapes_list[count].d * intersection[1] 
                                     + json_struct->shapes_list[count].e * intersection[2] 
                                     + json_struct->shapes_list[count].g;

            normal[1] = 2.0 * json_struct->shapes_list[count].b * intersection[1] 
                                     + json_struct->shapes_list[count].d * intersection[0] 
                                     + json_struct->shapes_list[count].f * intersection[2] 
                                     + json_struct->shapes_list[count].h;

            normal[2] = 2.0 * json_struct->shapes_list[count].c * intersection[2] 
                                     + json_struct->shapes_list[count].e * intersection[0] 
                                     + json_struct->shapes_list[count].f * intersection[1] 
                                     + json_struct->shapes_list[count].i;

            v3_normalize(normal, normal);

            if (v3_dot_product(normal, ray_direction) > 0) {
              v3_scale(normal, -1.0);
            }
          }
        }
      }
    }
  }

  return object_index;
}


/**
 * Checks if a ray intersects a plane shape. Returns the distance in the distance variable and 
 * True if it does intersect.
 *
 * @param ray_o - ray ray_origin vector
 * @param ray_d - ray direction vector
 * @param plane - the shape struct holding the information for the plane shape
 * @param distance - pointer to distance varaible
 * 
 * @returns - returns True if the ray intersects the shape, False otherwise
 */
__device__  bool ray_intersect_plane(float *ray_o, float *ray_d, shape_t *plane, float *distance) {
  float sub[3];
  v3_subtract(sub, ray_o, plane->position);

  float num = v3_dot_product(sub, plane->normal);
  float denom = v3_dot_product(ray_d, plane->normal);
  if (denom == 0) {
    return false;
  }
  
  float t = -1 * num / denom;
  if (t < 0) {
    return false;
  }  

  *distance = t;
  return true;
}


/**
 * Checks if a ray intersects a sphere shape. Returns the distance in the distance variable and 
 * True if it does intersect.
 *
 * @param ray_o - ray ray_origin vector
 * @param ray_d - ray direction vector
 * @param sphere - the shape struct holding the information for the sphere shape
 * @param distance - pointer to distance varaible 
 * 
 * @returns - returns True if the ray intersects the shape, False otherwise
 */
__device__  bool ray_intersect_sphere(float *ray_o, float *ray_d, shape_t *sphere, float *distance)
{
  float temp[3];

  v3_subtract(temp, ray_o, sphere->position);

  float a = ray_d[0] * ray_d[0] + ray_d[1] * ray_d[1] + ray_d[2]* ray_d[2];
  float b = 2 * v3_dot_product(ray_d, temp);
  float c = v3_dot_product(temp, temp) - powf(sphere->radius, 2);

  float discriminant = b * b - 4 * a * c;

  if (discriminant < 0) {
    return false;
  }
  else {
    *distance = (-b - powf(discriminant, 0.5)) / (2.0 * a);

    if (*distance < 0) {
      *distance = (-b + powf(discriminant, 0.5)) / (2.0 * a);
    } 
  }
  return true;
}


/**
 * Checks if a ray intersects a quadric shape. Returns the distance in the distance variable and 
 * True if it does intersect.
 *
 * @param ray_o - ray ray_origin vector
 * @param ray_d - ray direction vector
 * @param quadric - the shape struct holding the information for the quadric shape
 * @param distance - pointer to distance varaible
 * 
 * @returns - returns True if the ray intersects the shape, False otherwise
 */
__device__  bool ray_intersect_quadric(float *ray_o, float *ray_d, shape_t *quadric, float *distance) {
  float a_q = quadric->a * powf(ray_d[0], 2) + quadric->b * powf(ray_d[1], 2) 
            + quadric->c * powf(ray_d[2], 2) + quadric->d * ray_d[0] * ray_d[1] 
            + quadric->e * ray_d[0] * ray_d[2] + quadric->f * ray_d[1] * ray_d[2];

  float b_q = 2.0 * quadric->a * ray_o[0] * ray_d[0] 
            + 2.0 * quadric->b * ray_o[1] * ray_d[1]
            + 2.0 * quadric->c * ray_o[2] * ray_d[2]
            + quadric->d * (ray_o[0] * ray_d[1] + ray_o[1] * ray_d[0]) 
            + quadric->e * (ray_o[0] * ray_d[2] + ray_o[2] * ray_d[0])   
            + quadric->f * (ray_o[1] * ray_d[2] + ray_o[2] * ray_d[1]) 
            + quadric->g * ray_d[0]  
            + quadric->h * ray_d[1]
            + quadric->i * ray_d[2];

  float c_q = quadric->a * powf(ray_o[0], 2) 
            + quadric->b * powf(ray_o[1], 2) 
            + quadric->c * powf(ray_o[2], 2) 
            + quadric->d * ray_o[0] * ray_o[1]
            + quadric->e * ray_o[0] * ray_o[2]
            + quadric->f * ray_o[1] * ray_o[2]
            + quadric->g * ray_o[0]
            + quadric->h * ray_o[1]
            + quadric->i * ray_o[2]
            + quadric->j;
  
  if (a_q == 0.0) {
    *distance = -1.0 * c_q / b_q;
  }
  else {
    float discriminant = powf(b_q, 2) - 4.0 * a_q * c_q;
    if (discriminant < 0.0 ) {
      return false;
    }

    *distance = (-b_q - powf(discriminant, 0.5)) / (2.0 * a_q);
    if (*distance <= 0) {
      *distance = (-b_q + powf(discriminant, 0.5)) / (2.0 * a_q);
    }
  }
  
  return true;
}


/**
 * Calculates the radial attenuation value.
 *
 * @param light - The light object
 * @param distance - The distance from the light to the object
 * @return - radial attenuation value
 */
__device__  float radial_attenuation (light_t *light, float distance) {
  return 1.0 / (light->radial_coef[0] + light->radial_coef[1] * 
          distance + light->radial_coef[2] * powf(distance, 2));
}


/**
 * Calculates the angular attenuation value.
 *
 * @param light - The light object
 * @param object_point - The object's intersection point
 * @return - angular attenuation value
 */
__device__  float angular_attenuation(light_t *light, float *object_point) {

  if (light->type != SPOTLIGHT) {
    return 1.0;
  }

  float v_object[3];
  v3_from_points(v_object, light->position, object_point);
  v3_normalize(v_object, v_object);

  float alpha = v3_dot_product(v_object, light->direction);
  
  if (alpha < light->cos_theta) {
    return 0.0;
  }

  return powf(alpha, light->a0);
}


/**
 * Calculates the diffuse light color value.
 *
 * @param return_color - The diffuse light color value
 * @param light - The light object
 * @param object_color - The object's diffuse color
 * @param surface_normal - The object's normal value
 * @param light_vector - The lights direction vector
 */
__device__  void diffuse_light(float *return_color, light_t *light, float *object_color, 
                  float *surface_normal, float *light_vector) {
  float theta = v3_dot_product(surface_normal, light_vector);

  if (theta <= 0.0) {
    set_to_black(return_color);
  }
  else {
    return_color[0] = object_color[0] * light->color[0] * theta;
    return_color[1] = object_color[1] * light->color[1] * theta;
    return_color[2] = object_color[2] * light->color[2] * theta;
  }
}


/**
 * Calculates the specular light color value.
 *
 * @param return_color - The specular light color value
 * @param light - The light object
 * @param object_color - The object's specular color
 * @param surface_normal - The object's normal value
 * @param light_vector - The lights direction vector
 * @param view - The view vector
 */
__device__  void specular_light(float *return_color, light_t *light, float *object_color, 
                    float *surface_normal, float *light_vector, float *view) {

  float temp[3] = {view[0], view[1], view[2]};
  v3_scale(temp, -1);

  float theta = v3_dot_product(surface_normal, light_vector);
  if (theta <= 0.0) {
    set_to_black(return_color);
    return;
  }

  float reflection[3];
  v3_reflect(reflection, light_vector, surface_normal);

  double angle = v3_dot_product(temp, reflection);

  if (angle > 0) {
    set_to_black(return_color);
    return;
  }

  return_color[0] = object_color[0] * light->color[0] * powf(angle, 20);
  return_color[1] = object_color[1] * light->color[1] * powf(angle, 20);
  return_color[2] = object_color[2] * light->color[2] * powf(angle, 20);
}
