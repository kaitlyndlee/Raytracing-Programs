#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <errno.h>
#include <string.h>
#include <assert.h>
#include <math.h> 

#include "raycast.h"
#include "v3math.h"

const int MAX_RECURSION = 7;

/**
 * Runs the Raytracer.
 */
int main(int argc, char **argv) {

  if (argc != 5) {
    printf("Usage: raytrace WIDTH HEIGHT INPUT_SCENE OUTPUT_IMAGE\n" );
    exit(0);
  }

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
  
  json_data_t *json_struct = (json_data_t *) malloc(sizeof(json_data_t));
  json_struct->shapes_list = NULL;
  json_struct->lights_list = NULL;
  json_struct->num_shapes = 0;
  parse_json(input_json, json_struct);

  PPMFormat photo_data;

  photo_data.maxColor = 255;
  photo_data.height = height;
  photo_data.width = width;
  photo_data.size = photo_data.width * photo_data.height * 3;
  photo_data.pixmap = malloc(photo_data.size);

  raycast(json_struct, photo_data);

  ppm_WriteOutP3(photo_data, output_image);
  free(photo_data.pixmap);
  free(json_struct);
}


/**
 * Creates rays for each set of pixels and checks to see if any of the shapes intersects those rays.
 * The ray caster will take the closest shape to the ray and color in those pixels.
 *
 * @param json_struct - Struct holding all the shape information
 * @param photo_data - Struct holding all the image informaton
 */
void raycast(json_data_t *json_struct, PPMFormat photo_data) {
  int view_plane_center[3] = {0, 0, -1};
  float ray_orgin[3] = {0, 0, 0};
  float ray_direction[3];

  float color[3];
  uint8_t *pixel = photo_data.pixmap;

  shape_t object_array[json_struct->num_shapes];
  shape_t *current_object = json_struct->shapes_list;
  light_t light_array[json_struct->num_lights];
  light_t *current_light = json_struct->lights_list;

  for(int index = 0;index<json_struct->num_shapes; index++)
  {
    object_array[index] = *current_object;
    current_object = current_object->next;
  }

  for(int index = 0;index<json_struct->num_lights; index++)
  {
    light_array[index] = *current_light;
    current_light = current_light->next;
  }

  json_struct->lights_list = free_light_list(json_struct->lights_list);
  json_struct->shapes_list = free_shape_list(json_struct->shapes_list);
  json_struct->shapes_list = object_array;
  json_struct->lights_list = light_array;

  float pixel_height = json_struct->camera_height / photo_data.height;
  float pixel_width = json_struct->camera_width / photo_data.width;

  // starts at top left hand corner of view plane
  for (int y = 0; y < photo_data.height; y++) {
    for (int x = 0; x < photo_data.width; x++) {
      ray_direction[0] = view_plane_center[0] - (json_struct->camera_width / 2.0) + pixel_width * (x + 0.5);
      ray_direction[1] = view_plane_center[1] + (json_struct->camera_height / 2.0) - pixel_height * (y + 0.5);
      ray_direction[2] = view_plane_center[2];
      v3_normalize(ray_direction, ray_direction);

      
      iterative_shoot(json_struct, ray_orgin, ray_direction, color, -1);
      v3_scale(color, photo_data.maxColor);

      pixel[0] = ppm_clamp(color[0], 0, photo_data.maxColor);
      pixel[1] = ppm_clamp(color[1], 0, photo_data.maxColor);
      pixel[2] = ppm_clamp(color[2], 0, photo_data.maxColor);
      pixel += 3;
    } 
  }
}




/**
 * Casts a ray with the given ray direction and ray orgin and calculates
 * the given color given the nearest object intersection
 * 
 * If skip_index != -1, skip the index of the object so that 
 * the object does not intersect with itself.
 * If skip_index == -1, don't set the shape's intersection or normal.
 *
 * @param json_struct - json struct holding the shape list
 * @param ray_orgin - ray orgin vector
 * @param ray_direction - ray direction vector
 * @param out_color - The calculated output color for each pixel
 * @param recursion_level - The current level or recursion. The max level or recusion is 7.
 * @param skip_index - index of object we previously intersected
 * @param in_ior - variable in_ior to set the refractive index of the pervious object. 
 *                 If no pervious object in_ior = 1.
 */
void shoot(json_data_t *json_struct, float *ray_orgin, float *ray_direction, float *out_color, 
           int recursion_level, int skip_index) {
  float ZERO_VECTOR[3] = {0, 0, 0};

  if (recursion_level == 0) {
    set_to_black(out_color);
    return;
  }

  shape_t *nearest_object = NULL;
  int nearest_object_index = -1;
  float intersection[3];
  float normal[3];
  nearest_object_index = find_nearest_object(json_struct, &nearest_object, ray_orgin, 
                                             ray_direction, intersection, normal, skip_index, false);

  if (nearest_object == NULL) {
    set_to_black(out_color);
    return;
  }

  // calculate reflection color
  float reflection_color[3]; 
  float reflection_vector[3];
  if (nearest_object->reflectivity > 0) {
    v3_reflect(reflection_vector, ray_direction, normal);
    v3_normalize(reflection_vector, reflection_vector);
    shoot(json_struct, intersection, reflection_vector, reflection_color, recursion_level - 1, 
         nearest_object_index);
  }
  else {
    set_to_black(reflection_color);
  }

  // calculate refraction color
  float refraction_vector[3];
  float refraction_color[3];
  if (nearest_object->refractivity > 0) {
    float b[3];

    // Calculate transmission vector
    v3_cross_product(b, ray_direction, normal);
    v3_normalize(b, b);
    v3_cross_product(b, b, normal);

    float sin_phi = v3_dot_product(ray_direction, b);
    sin_phi *= 1 / nearest_object->ior;
    float cos_phi = sqrt(1 - pow(sin_phi, 2));

    v3_set_points(refraction_vector, normal);

    v3_scale(refraction_vector, -cos_phi);
    v3_scale(b, sin_phi);
    v3_add(refraction_vector, refraction_vector, b);

    // Add offset
    v3_scale(refraction_vector, 0.01);
    float offsetted_int[3];
    v3_add(offsetted_int, intersection, refraction_vector);
    v3_normalize(refraction_vector, refraction_vector);

    // Don't skip intersection with self since we want to find the intersection at the
    // other end of the sphere
    float across_sphere_int[3];
    float across_sphere_norm[3];
    shape_t *sphere = malloc(sizeof(shape_t));
    float sphere_index = find_nearest_object(json_struct, &sphere, offsetted_int, 
                          refraction_vector, across_sphere_int, across_sphere_norm, -1, false);

    // Calculate L vector
    v3_scale(across_sphere_norm, -1);
    v3_cross_product(b, refraction_vector, across_sphere_norm);
    v3_normalize(b, b);
    v3_cross_product(b, b, across_sphere_norm);

    sin_phi = v3_dot_product(refraction_vector, b);
    sin_phi *= nearest_object->ior;
    cos_phi = sqrt(1 - pow(sin_phi, 2));

    v3_set_points(refraction_vector, across_sphere_norm);

    v3_scale(refraction_vector, -cos_phi);
    v3_scale(b, sin_phi);
    v3_add(refraction_vector, refraction_vector, b);

    // Shoot L from the new sphere intersection
    shoot(json_struct, across_sphere_int, refraction_vector, refraction_color, recursion_level - 1, 
          nearest_object_index);
  }
  else {
      set_to_black(refraction_color);
  }

  assert(nearest_object->reflectivity + nearest_object->refractivity <= 1.0);

  float color[3];

  float opacity = 1.0 - nearest_object->reflectivity - nearest_object->refractivity;
  if (opacity > 0.0) {
    v3_set_points(color, ZERO_VECTOR);

    shape_t *obj_intersection;
    float light_obj_dist;
    float light_direction[3];
    float rad_atten;
    float ang_atten;
    float diffuse_output[3];
    float specular_output[3];
    float temp[3];

    for (int lights = 0; lights < json_struct->num_lights; lights++) { 
      v3_from_points(light_direction, intersection, json_struct->lights_list[lights].position);
      light_obj_dist = v3_length(light_direction);
      v3_normalize(light_direction, light_direction);
      
      obj_intersection = NULL;
      find_nearest_object(json_struct, &obj_intersection, intersection, 
                          light_direction, NULL, NULL, nearest_object_index, true); 

      // Not in the shadow of another shape
      if (obj_intersection == NULL) {
        rad_atten = radial_attenuation(&json_struct->lights_list[lights], light_obj_dist);
        ang_atten = angular_attenuation(&json_struct->lights_list[lights], intersection);
        diffuse_light(diffuse_output, &json_struct->lights_list[lights], nearest_object->diffuse_color, 
                      normal, light_direction);

        specular_light(specular_output, &json_struct->lights_list[lights], nearest_object->specular_color, 
                       normal, light_direction, ray_direction);
      
        color[0] += (diffuse_output[0] + specular_output[0]) * rad_atten * ang_atten;
        color[1] += (diffuse_output[1] + specular_output[1]) * rad_atten * ang_atten;
        color[2] += (diffuse_output[2] + specular_output[2]) * rad_atten * ang_atten;
      }
    }
  }
  else {
    set_to_black(color);
  }

  v3_scale(color, opacity);
  v3_scale(reflection_color, nearest_object->reflectivity);
  // v3_scale(refraction_color, nearest_object->refractivity);
  v3_add(out_color, color, reflection_color);
  // v3_add(out_color, out_color, refraction_color);
}

/**
 * Casts a ray with the given ray direction and ray orgin and calculates
 * the given color given the nearest object intersection
 * 
 * If skip_index != -1, skip the index of the object so that 
 * the object does not intersect with itself.
 * If skip_index == -1, don't set the shape's intersection or normal.
 *
 * @param json_struct - json struct holding the shape list
 * @param ray_orgin - ray orgin vector
 * @param ray_direction - ray direction vector
 * @param out_color - The calculated output color for each pixel
 * @param recursion_level - The current level or recursion. The max level or recusion is 7.
 * @param skip_index - index of object we previously intersected
 * @param in_ior - variable in_ior to set the refractive index of the pervious object. 
 *                 If no pervious object in_ior = 1.
 */
void iterative_shoot(json_data_t *json_struct, float *ray_orgin, float *ray_direction, float *out_color, int skip_index) {
  printf("Direction: (%f, %f, %f)\n", ray_direction[0], ray_direction[1], ray_direction[2]);
  float color[3];
  float opacity;
  shape_t *nearest_object = NULL;
  int nearest_object_index = -1;
  float intersection[3];
  float normal[3];

  set_to_black(out_color);
  nearest_object_index = -1;
  nearest_object_index = find_nearest_object(json_struct, &nearest_object, ray_orgin, 
                                           ray_direction, intersection, normal, skip_index, false);

  if (nearest_object == NULL) {
    set_to_black(out_color);
    return;
  }

  shape_t *next_nearest_object = nearest_object;
  float next_ray_direction[3];
  v3_set_points(next_ray_direction, ray_direction);
  float next_ray_orgin[3];
  v3_set_points(next_ray_orgin, intersection);
  int next_skip_index = nearest_object_index;
  int next_nearest_object_index = nearest_object_index;
  float next_intersecion[3];
  float next_normal[3];
  v3_set_points(next_normal, normal);

  float reflection_color[3]; 
  set_to_black(reflection_color);
  float reflection_vector[3];
  float temp_color[3];

  int recursion_level = 1;
  while(recursion_level < MAX_RECURSION)
  {
    printf("\tPrim ID: %d, intersection: (%f, %f, %f), normal: (%f, %f, %f)\n", 
    next_nearest_object_index, next_intersecion[0], next_intersecion[1], next_intersecion[2], next_normal[0], next_normal[1], next_normal[2]);
    
    // calculate reflection color
    if (next_nearest_object->reflectivity > 0) {
      v3_reflect(reflection_vector, next_ray_direction, next_normal);
      v3_normalize(reflection_vector, reflection_vector);
      v3_set_points(next_ray_direction, reflection_vector);
      next_nearest_object_index = -1;
      next_nearest_object_index = find_nearest_object(json_struct, &next_nearest_object, next_ray_orgin, 
                                             next_ray_direction, next_intersecion, next_normal, next_skip_index, false);
      if (next_nearest_object != NULL) {
        calc_color(json_struct, temp_color, next_nearest_object_index, next_intersecion, next_normal, next_ray_direction);
        // v3_scale(temp_color, next_nearest_object->reflectivity);
        v3_add(reflection_color, reflection_color, temp_color);
        v3_set_points(next_ray_orgin, next_intersecion);
        next_skip_index = next_nearest_object_index;
        recursion_level++;
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
  
  v3_scale(reflection_color, nearest_object->reflectivity);
  v3_add(out_color, out_color, color);
  v3_add(out_color, out_color, reflection_color); 
}

void calc_color(json_data_t *json_struct, float *out_color, int object_index, float *intersection, float *normal, float *ray_direction) {
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
    float temp[3];

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
 * @param ray_orgin - ray orgin vector
 * @param ray_direction - ray direction vector
 * @param skip_index - index of object we previously intersected
 * 
 * @returns - returns index of the nearest instersected object.
 */
int find_nearest_object(json_data_t *json_struct, shape_t **nearest_object, 
                        float *ray_orgin, float *ray_direction, 
                        float *intersection, float *normal, int skip_index, bool shadow_test) {
  float nearest_distance = INFINITY;
  int object_index = -1;
  float distance;


  for (int count = 0; count < json_struct->num_shapes; count++) {    
    if (skip_index == count) { // Skip checking if a shape intersects with itself
      continue;
    }

    if (json_struct->shapes_list[count].type == SPHERE) {
      if (ray_intersect_sphere(ray_orgin, ray_direction, &json_struct->shapes_list[count], &distance)) {
        if (nearest_distance > distance && distance > 0) {
          nearest_distance = distance;
          *nearest_object = &json_struct->shapes_list[count];
          object_index = count;
          if (!shadow_test) {
            intersection[0] = ray_orgin[0] + ray_direction[0] * nearest_distance;
            intersection[1] = ray_orgin[1] + ray_direction[1] * nearest_distance;
            intersection[2] = ray_orgin[2] + ray_direction[2] * nearest_distance;

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
      if (ray_intersect_plane(ray_orgin, ray_direction, &json_struct->shapes_list[count], &distance)) {
        if (nearest_distance > distance && distance > 0) {
          nearest_distance = distance;
          *nearest_object = &json_struct->shapes_list[count];
          object_index = count;
          if (!shadow_test) {
            intersection[0] = ray_orgin[0] + ray_direction[0] * nearest_distance;
            intersection[1] = ray_orgin[1] + ray_direction[1] * nearest_distance;
            intersection[2] = ray_orgin[2] + ray_direction[2] * nearest_distance;
            v3_set_points(normal, json_struct->shapes_list[count].normal);
          }
        }
      }
    }
    else if (json_struct->shapes_list[count].type == QUADRIC) {
      if (ray_intersect_quadric(ray_orgin, ray_direction, &json_struct->shapes_list[count], &distance)) {
        // If the point is in front of the quadric.
        if (skip_index != -1 && (ray_orgin[2] + distance * ray_direction[2]) < ray_orgin[2]) {
          continue;
          }
        if (nearest_distance > distance && distance > 0) {
          nearest_distance = distance;
          *nearest_object = &json_struct->shapes_list[count];
          object_index = count;
          if (!shadow_test) {
            intersection[0] = ray_orgin[0] + ray_direction[0] * nearest_distance;
            intersection[1] = ray_orgin[1] + ray_direction[1] * nearest_distance;
            intersection[2] = ray_orgin[2] + ray_direction[2] * nearest_distance;

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
 * @param ray_o - ray orgin vector
 * @param ray_d - ray direction vector
 * @param plane - the shape struct holding the information for the plane shape
 * @param distance - pointer to distance varaible
 * 
 * @returns - returns True if the ray intersects the shape, False otherwise
 */
bool ray_intersect_plane(float *ray_o, float *ray_d, shape_t *plane, float *distance) {
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
 * @param ray_o - ray orgin vector
 * @param ray_d - ray direction vector
 * @param sphere - the shape struct holding the information for the sphere shape
 * @param distance - pointer to distance varaible 
 * 
 * @returns - returns True if the ray intersects the shape, False otherwise
 */
bool ray_intersect_sphere(float *ray_o, float *ray_d, shape_t *sphere, float *distance)
{
  float temp[3];
  float temp_value;

  v3_subtract(temp, ray_o, sphere->position);

  float a = pow(ray_d[0], 2) + pow(ray_d[1], 2) + pow(ray_d[2], 2);
  float b = 2 * v3_dot_product(ray_d,temp);
  float c = v3_dot_product(temp, temp) - pow(sphere->radius, 2);

  float discriminant = pow(b, 2) - 4 * a * c;

  if (discriminant < 0) {
    return false;
  }
  else {
    *distance = (-b - pow(discriminant, 0.5)) / (2.0 * a);

    if (*distance < 0) {
      *distance = (-b + pow(discriminant, 0.5)) / (2.0 * a);
    } 
  }
  return true;
}


/**
 * Checks if a ray intersects a quadric shape. Returns the distance in the distance variable and 
 * True if it does intersect.
 *
 * @param ray_o - ray orgin vector
 * @param ray_d - ray direction vector
 * @param quadric - the shape struct holding the information for the quadric shape
 * @param distance - pointer to distance varaible
 * 
 * @returns - returns True if the ray intersects the shape, False otherwise
 */
bool ray_intersect_quadric(float *ray_o, float *ray_d, shape_t *quadric, float *distance) {
  float a_q = quadric->a * pow(ray_d[0], 2) + quadric->b * pow(ray_d[1], 2) 
            + quadric->c * pow(ray_d[2], 2) + quadric->d * ray_d[0] * ray_d[1] 
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

  float c_q = quadric->a * pow(ray_o[0], 2) 
            + quadric->b * pow(ray_o[1], 2) 
            + quadric->c * pow(ray_o[2], 2) 
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
    float discriminant = pow(b_q, 2) - 4.0 * a_q * c_q;
    if (discriminant < 0.0 ) {
      return false;
    }

    *distance = (-b_q - pow(discriminant, 0.5)) / (2.0 * a_q);
    if (*distance <= 0) {
      *distance = (-b_q + pow(discriminant, 0.5)) / (2.0 * a_q);
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
float radial_attenuation (light_t *light, float distance) {
  return 1.0 / (light->radial_coef[0] + light->radial_coef[1] * 
          distance + light->radial_coef[2] * pow(distance, 2));
}


/**
 * Calculates the angular attenuation value.
 *
 * @param light - The light object
 * @param object_point - The object's intersection point
 * @return - angular attenuation value
 */
float angular_attenuation(light_t *light, float *object_point) {

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

  return pow(alpha, light->a0);
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
void diffuse_light(float *return_color, light_t *light, float *object_color, 
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
void specular_light(float *return_color, light_t *light, float *object_color, 
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

  return_color[0] = object_color[0] * light->color[0] * pow(angle, 20);
  return_color[1] = object_color[1] * light->color[1] * pow(angle, 20);
  return_color[2] = object_color[2] * light->color[2] * pow(angle, 20);
}
