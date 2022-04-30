// ======================================================================== //
// Copyright 2019 Ingo Wald                                                 //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

// Ray gen shader for ll00-rayGenOnly. No actual rays are harmed in the making
// of this shader. The pixel location is simply translated into a checkerboard
// pattern.

#include "deviceCode.h"

#include <optix_device.h>

#define MAX_DEPTH 50

inline __device__ void set_to_black(vec3f *input) {
  input->x = 0;
  input->y = 0;
  input->z = 0;
}

inline __device__ float clamp(float value, float lower_bound, float upper_bound) {
  if (value > upper_bound) {
    value = upper_bound;
  }
  if (value < lower_bound) {
    value = lower_bound;
  }

  return value;
}

/**
 * Reflects the vector given as the parameter a over n and stores the result in the destination
 * pointer
 *
 * @param dst - result vector pointer
 * @param v - pointer of the vector to relflect
 * @param n - pointer of the vector that is the surface
 */
inline __device__ void reflect(vec3f *dst, vec3f v, vec3f n) {
  vec3f product = n;

  float dot_product = dot(v, n);
  product = product * 2 * dot_product;
  product = v - product;
  *dst = product;
}

/**
 * Calculates the radial attenuation value.
 *
 * @param light - The light object
 * @param distance - The distance from the light to the object
 * @return - radial attenuation value
 */
inline __device__ float radial_attenuation(Light light, float distance) {
  return 1.0 / (light.radial_coef.x + light.radial_coef.y * distance +
                light.radial_coef.z * powf(distance, 2));
}

/**
 * Calculates the angular attenuation value.
 *
 * @param light - The light object
 * @param object_point - The object's intersection point
 * @return - angular attenuation value
 */
inline __device__ float angular_attenuation(Light light, vec3f object_point) {

  if (light.type != SPOTLIGHT) {
    return 1.0;
  }

  vec3f v_object = object_point - light.position;
  v_object = normalize(v_object);

  const float alpha = dot(v_object, light.direction);

  if (alpha < light.cos_theta) {
    return 0.0;
  }

  return powf(alpha, light.a0);
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
inline __device__ void diffuse_light(vec3f *return_color,
                                     Light light,
                                     vec3f object_color,
                                     vec3f surface_normal,
                                     vec3f light_vector) {
  const float theta = dot(surface_normal, light_vector);

  if (theta <= 0.0) {
    set_to_black(return_color);
  }
  else {
    *return_color = object_color * light.color * theta;
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
inline __device__ void specular_light(vec3f *return_color,
                                      Light light,
                                      vec3f object_color,
                                      vec3f surface_normal,
                                      vec3f light_vector,
                                      vec3f view) {

  vec3f temp = view;
  temp = temp * -1;

  const float theta = dot(surface_normal, light_vector);
  if (theta <= 0.0) {
    set_to_black(return_color);
    return;
  }

  vec3f reflection;
  reflect(&reflection, light_vector, surface_normal);

  const float angle = dot(temp, reflection);

  if (angle > 0) {
    set_to_black(return_color);
    return;
  }

  *return_color = object_color * light.color * powf(angle, 20);
}

inline __device__ vec3f calc_color(const RayGenData &self, owl::Ray &ray, PerRayData &prd) {
  vec3f color = vec3f(0, 0, 0);
  float opacity = 1.0 - prd.reflectivity - prd.refractivity;
  if (opacity <= 0) {
    set_to_black(&color);
    return color;
  }

  vec3f diffuse_output;
  vec3f specular_output;
  float light_obj_dist;
  owl::Ray light_ray;
  float rad_atten;
  float ang_atten;

  for (int i = 0; i < self.num_lights; i++) {
    light_ray.direction = self.lights[i].position;
    light_ray.origin = prd.intersection;
    light_ray.direction = light_ray.direction - light_ray.origin;
    light_obj_dist = length(light_ray.direction);
    light_ray.direction = normalize(light_ray.direction);
    PerRayData new_prd;
    new_prd.distance = -1;
    new_prd.prev_intersection = &prd;

    owl::traceRay(/*accel to trace against*/ self.world,
                  /*the ray to trace*/ light_ray,
                  /*prd*/ new_prd,
                  /*only CH*/ OPTIX_RAY_FLAG_DISABLE_ANYHIT);

    // Not in the shadow of another shape
    if (new_prd.primId == -1) {
      rad_atten = radial_attenuation(self.lights[i], light_obj_dist);
      ang_atten = angular_attenuation(self.lights[i], prd.intersection);
      diffuse_light(&diffuse_output, self.lights[i], prd.diffuse_color, prd.normal,
                    light_ray.direction);
      specular_light(&specular_output, self.lights[i], prd.specular_color, prd.normal,
                     light_ray.direction, ray.direction);

      color += (diffuse_output + specular_output) * rad_atten * ang_atten;
    }
  }
  return color * opacity;
}

// TODO: FIX
inline __device__ vec3f iterative_shoot(const RayGenData &self, owl::Ray &ray) {
  printf("Direction: (%f, %f, %f)\n", ray.direction.x, ray.direction.y, ray.direction.z);
  PerRayData prd;
  prd.distance = -1;
  prd.primId = -1;
  prd.prev_intersection = NULL;
  owl::traceRay(/*accel to trace against*/ self.world,
                /*the ray to trace*/ ray,
                /*prd*/ prd);

  vec3f color;
  set_to_black(&color);
  if (prd.primId == -1) {
    printf("\tIn no intersection\n");
    set_to_black(&color);
    return color;
  }

  PerRayData next_prd = prd;
  next_prd.prev_intersection = &prd;
  owl::Ray next_ray;
  next_ray.direction = ray.direction;
  next_ray.origin = prd.intersection;

  vec3f reflection_vector;
  vec3f temp_color;
  set_to_black(&temp_color);
  float total_refl = prd.reflectivity;

  for (int i = 0; i < MAX_DEPTH; i++) {
    printf("\tPrim ID: %d, type: %d, normal: (%f, %f, %f)\n", 
    next_prd.primId, next_prd.shape_type, next_prd.normal.x, next_prd.normal.y, next_prd.normal.z);
    if (next_prd.reflectivity > 0) {
      reflect(&reflection_vector, next_ray.direction, next_prd.normal);
      reflection_vector = normalize(reflection_vector);
      next_ray.direction = reflection_vector;
      owl::traceRay(/*accel to trace against*/ self.world,
                    /*the ray to trace*/ next_ray,
                    /*prd*/ next_prd);
        
      if (next_prd.primId == -1) {
        break;
      }

      // Calculate reflection color
      temp_color = calc_color(self, next_ray, next_prd);
      temp_color *= total_refl;
      total_refl *= next_prd.reflectivity;
      color += temp_color;
      
      // Prepare next ray
      next_prd.prev_intersection = &next_prd;
      next_ray.origin = next_prd.intersection;
    }
    else {
      printf("\tReflectivity: %f\n", next_prd.reflectivity);
      break;
    }
  }

  temp_color = calc_color(self, ray, prd);
  color += temp_color;
  return color;
}

OPTIX_RAYGEN_PROGRAM(rayGen)() {
  const RayGenData &self = owl::getProgramData<RayGenData>();
  const vec2i pixelID = owl::getLaunchIndex();
  owl::Ray ray;
  ray.origin = vec3f(0.f, 0.f, 0.f);
  vec3i view_plane_center = vec3i(0, 0, -1);

  ray.direction.x =
      view_plane_center.x - (self.camera_width / 2.0) + self.pixel_width * (pixelID.x + 0.5);
  ray.direction.y =
      view_plane_center.y + (self.camera_height / 2.0) - self.pixel_height * (pixelID.y + 0.5);
  ray.direction.z = view_plane_center.z;

  ray.direction = normalize(ray.direction);

  vec3f color = iterative_shoot(self, ray);
  float color_vector[3] = {color.x, color.y, color.z};
  self.pixmap[(pixelID.y * self.width + pixelID.x) * 3] = clamp(color_vector[0] * 255, 0, 255);
  self.pixmap[(pixelID.y * self.width + pixelID.x) * 3 + 1] = clamp(color_vector[1] * 255, 0, 255);
  self.pixmap[(pixelID.y * self.width + pixelID.x) * 3 + 2] = clamp(color_vector[2] * 255, 0, 255);
}

OPTIX_RAYGEN_PROGRAM(simpleRayGen)() {
  const RayGenData &self = owl::getProgramData<RayGenData>();
  const vec2i pixelID = owl::getLaunchIndex();
  owl::Ray ray;
  ray.origin = vec3f(0.f, 0.f, 0.f);
  vec3i view_plane_center = vec3i(0, 0, -1);

  ray.direction.x =
      view_plane_center.x - (self.camera_width / 2.0) + self.pixel_width * (pixelID.x + 0.5);
  ray.direction.y =
      view_plane_center.y + (self.camera_height / 2.0) - self.pixel_height * (pixelID.y + 0.5);
  ray.direction.z = view_plane_center.z;

  ray.direction = normalize(ray.direction);

  PerRayData prd;
  prd.distance = -1;
  prd.primId = -1;
  prd.prev_intersection = NULL;
  owl::traceRay(/*accel to trace against*/ self.world,
                /*the ray to trace*/ ray,
                /*prd*/ prd);

  vec3f color = vec3f(0, 0, 0);

  if (prd.primId != -1) {
    vec3f diffuse_output;
    vec3f specular_output;
    float light_obj_dist;
    owl::Ray light_ray;
    float rad_atten;
    float ang_atten;
    vec3f temp;
    for (int i = 0; i < self.num_lights; i++) {
      light_ray.direction = self.lights[i].position;
      light_ray.origin = prd.intersection;
      light_ray.direction = light_ray.direction - light_ray.origin;
      light_obj_dist = length(light_ray.direction);
      light_ray.direction = normalize(light_ray.direction);
      PerRayData new_prd;
      new_prd.distance = -1;
      new_prd.prev_intersection = &prd;

      owl::traceRay(/*accel to trace against*/ self.world,
                    /*the ray to trace*/ light_ray,
                    /*prd*/ new_prd,
                    /*only CH*/ OPTIX_RAY_FLAG_DISABLE_ANYHIT);

      if (new_prd.primId == -1) {
        rad_atten = radial_attenuation(self.lights[i], light_obj_dist);
        ang_atten = angular_attenuation(self.lights[i], prd.intersection);
        diffuse_light(&diffuse_output, self.lights[i], prd.diffuse_color, prd.normal,
                      light_ray.direction);
        specular_light(&specular_output, self.lights[i], prd.specular_color, prd.normal,
                       light_ray.direction, ray.direction);
        color += (diffuse_output + specular_output) * rad_atten * ang_atten;
      }
    }
  }
  else {
    set_to_black(&color);
  }

  float color_vector[3] = {color.x, color.y, color.z};
  self.pixmap[(pixelID.y * self.width + pixelID.x) * 3] = clamp(color_vector[0] * 255, 0, 255);
  self.pixmap[(pixelID.y * self.width + pixelID.x) * 3 + 1] = clamp(color_vector[1] * 255, 0, 255);
  self.pixmap[(pixelID.y * self.width + pixelID.x) * 3 + 2] = clamp(color_vector[2] * 255, 0, 255);
}

OPTIX_MISS_PROGRAM(miss)() {
  const vec2i pixelID = owl::getLaunchIndex();
  PerRayData &prd = owl::getPRD<PerRayData>();
  set_to_black(&prd.diffuse_color);
  set_to_black(&prd.specular_color);
  prd.distance = -1;
  prd.primId = -1;
}

OPTIX_INTERSECT_PROGRAM(Spheres)() {
  const int primId = optixGetPrimitiveIndex();
  const auto &self = owl::getProgramData<SpheresList>().primitives[primId];
  PerRayData &prd = owl::getPRD<PerRayData>();

  // Do not intersect with self
  if ((prd.prev_intersection != NULL) && (primId == prd.prev_intersection->primId) &&
      (prd.prev_intersection->shape_type == SPHERE)) {
    return;
  }

  const vec3f origin = optixGetWorldRayOrigin();
  const vec3f direction = optixGetWorldRayDirection();
  float hit_t = 0;
  const float tmin = optixGetRayTmin();
  const float tmax = optixGetRayTmax();

  const vec3f origin_to_pos = origin - self.position;

  const float a = dot(direction, direction);
  const float b = 2 * dot(direction, origin_to_pos);
  const float c = dot(origin_to_pos, origin_to_pos) - (self.radius * self.radius);

  const float discriminant = b * b - 4 * a * c;
  if (discriminant < 0.f) {
    return;
  }
  else {
    hit_t = (-b - sqrtf(discriminant)) / (2.0 * a);

    if (hit_t < 0) {
      hit_t = (-b + sqrtf(discriminant)) / (2.0 * a);
    }
  }
  if (hit_t < tmax) {
    optixReportIntersection(hit_t, 0);
    prd.distance = hit_t;
  }
}

OPTIX_BOUNDS_PROGRAM(Spheres)(const void *geomData, box3f &primBounds, const int primId) {
  const SpheresList &self = *(const SpheresList *) geomData;
  const Sphere sphere = self.primitives[primId];
  primBounds =
      box3f().extend(sphere.position - sphere.radius).extend(sphere.position + sphere.radius);
}

OPTIX_CLOSEST_HIT_PROGRAM(Spheres)() {
  const int primId = optixGetPrimitiveIndex();
  const auto &self = owl::getProgramData<SpheresList>().primitives[primId];
  PerRayData &prd = owl::getPRD<PerRayData>();
  prd.primId = primId;
  prd.diffuse_color = self.diffuse_color;
  prd.specular_color = self.specular_color;
  prd.reflectivity = self.reflectivity;
  prd.refractivity = self.refractivity;
  const vec3f origin = optixGetWorldRayOrigin();
  const vec3f direction = optixGetWorldRayDirection();
  prd.intersection = origin + direction * prd.distance;
  const float temp = 1.0 / self.radius;
  prd.normal = (prd.intersection - self.position) * temp;
  prd.normal = normalize(prd.normal);
  prd.shape_type = SPHERE;
}

OPTIX_INTERSECT_PROGRAM(Planes)() {
  const int primId = optixGetPrimitiveIndex();
  const auto &self = owl::getProgramData<PlanesList>().primitives[primId];
  PerRayData &prd = owl::getPRD<PerRayData>();

  // Do not intersect with self
  if ((prd.prev_intersection != NULL) && (primId == prd.prev_intersection->primId) &&
      (prd.prev_intersection->shape_type == PLANE)) {
    return;
  }

  const vec3f origin = optixGetWorldRayOrigin();
  const vec3f direction = optixGetWorldRayDirection();

  float hit_t = optixGetRayTmax();
  const float tmin = optixGetRayTmin();

  const vec3f origin_to_pos = origin - self.position;
  const float numerator = dot(origin_to_pos, self.normal);
  const float denominator = dot(direction, self.normal);

  if (denominator == 0) {
    return;
  }

  float t = -1 * numerator / denominator;
  if (t < 0) {
    return;
  }

  if (t < hit_t && t > tmin) {
    hit_t = t;
  }

  if (hit_t < optixGetRayTmax()) {
    optixReportIntersection(hit_t, 0);
    prd.distance = hit_t;
  }
}

// TODO: what does this do exactly?
OPTIX_BOUNDS_PROGRAM(Planes)(const void *geomData, box3f &primBounds, const int primId) {
  const PlanesList &self = *(const PlanesList *) geomData;
  const Plane plane = self.primitives[primId];
  primBounds = box3f(vec3f(-1.f, -1.f, 0.f), vec3f(+1.f, +1.f, +1.f));
}

OPTIX_CLOSEST_HIT_PROGRAM(Planes)() {
  const int primId = optixGetPrimitiveIndex();
  const auto &self = owl::getProgramData<PlanesList>().primitives[primId];
  PerRayData &prd = owl::getPRD<PerRayData>();
  prd.primId = primId;
  prd.diffuse_color = self.diffuse_color;
  prd.specular_color = self.specular_color;
  prd.reflectivity = self.reflectivity;
  prd.refractivity = self.refractivity;
  const vec3f origin = optixGetWorldRayOrigin();
  const vec3f direction = optixGetWorldRayDirection();
  prd.intersection = origin + direction * prd.distance;
  prd.normal = self.normal;
  prd.shape_type = PLANE;
}

OPTIX_INTERSECT_PROGRAM(Quadrics)() {
  const int primId = optixGetPrimitiveIndex();
  const auto &self = owl::getProgramData<QuadricsList>().primitives[primId];
  PerRayData &prd = owl::getPRD<PerRayData>();

  // Do not intersect with self
  if ((prd.prev_intersection != NULL) && (primId == prd.prev_intersection->primId) &&
      (prd.prev_intersection->shape_type == QUADRIC)) {
    return;
  }

  const vec3f origin = optixGetWorldRayOrigin();
  const vec3f direction = optixGetWorldRayDirection();

  float hit_t = optixGetRayTmax();
  const float tmin = optixGetRayTmin();

  float a_q = self.a * powf(direction.x, 2) + self.b * powf(direction.y, 2) +
              self.c * powf(direction.z, 2) + self.d * direction.x * direction.y +
              self.e * direction.x * direction.z + self.f * direction.y * direction.z;

  float b_q = 2.0 * self.a * origin.x * direction.x + 2.0 * self.b * origin.y * direction.y +
              2.0 * self.c * origin.z * direction.z +
              self.d * (origin.x * direction.y + origin.y * direction.x) +
              self.e * (origin.x * direction.z + origin.z * direction.x) +
              self.f * (origin.y * direction.z + origin.z * direction.y) + self.g * direction.x +
              self.h * direction.y + self.i * direction.z;

  float c_q = self.a * powf(origin.x, 2) + self.b * powf(origin.y, 2) + self.c * powf(origin.z, 2) +
              self.d * origin.x * origin.y + self.e * origin.x * origin.z +
              self.f * origin.y * origin.z + self.g * origin.x + self.h * origin.y +
              self.i * origin.z + self.j;

  float t = 0;

  if (a_q == 0.0) {
    t = -1.0 * c_q / b_q;
  }
  else {
    float discriminant = powf(b_q, 2) - 4.0 * a_q * c_q;
    if (discriminant < 0.0) {
      return;
    }

    t = (-b_q - powf(discriminant, 0.5)) / (2.0 * a_q);
    if (t <= 0) {
      t = (-b_q + powf(discriminant, 0.5)) / (2.0 * a_q);
    }
  }

  if (t < hit_t && t > tmin) {
    hit_t = t;
  }

  if (hit_t < optixGetRayTmax()) {
    optixReportIntersection(hit_t, 0);
    prd.distance = hit_t;
  }
}

// TODO: what does this do exactly?
OPTIX_BOUNDS_PROGRAM(Quadrics)(const void *geomData, box3f &primBounds, const int primId) {
  const QuadricsList &self = *(const QuadricsList *) geomData;
  const Quadric quadric = self.primitives[primId];
  primBounds = box3f(vec3f(-1.f, -1.f, 0.f), vec3f(+1.f, +1.f, +1.f));
}

OPTIX_CLOSEST_HIT_PROGRAM(Quadrics)() {
  const int primId = optixGetPrimitiveIndex();
  const auto &self = owl::getProgramData<QuadricsList>().primitives[primId];
  PerRayData &prd = owl::getPRD<PerRayData>();
  prd.primId = primId;
  prd.diffuse_color = self.diffuse_color;
  prd.specular_color = self.specular_color;
  prd.reflectivity = self.reflectivity;
  prd.refractivity = self.refractivity;
  const vec3f origin = optixGetWorldRayOrigin();
  const vec3f direction = optixGetWorldRayDirection();
  prd.intersection = origin + direction * prd.distance;

  prd.normal.x = 2.0 * self.a * prd.intersection.x + self.d * prd.intersection.y +
                 self.e * prd.intersection.z + self.g;

  prd.normal.y = 2.0 * self.b * prd.intersection.y + self.d * prd.intersection.x +
                 self.f * prd.intersection.z + self.h;

  prd.normal.z = 2.0 * self.c * prd.intersection.z + self.e * prd.intersection.x +
                 self.f * prd.intersection.y + self.i;

  prd.normal = normalize(prd.normal);

  if (dot(prd.normal, direction) > 0) {
    prd.normal *= -1.0;
  }

  prd.shape_type = QUADRIC;
}