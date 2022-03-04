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

OPTIX_RAYGEN_PROGRAM(simpleRayGen)() {
  const RayGenData &self = owl::getProgramData<RayGenData>();
  const vec2i pixelID = owl::getLaunchIndex();
  if (pixelID == owl::vec2i(0)) {
    printf("%sHello OptiX From your First RayGen Program%s\n", OWL_TERMINAL_CYAN,
           OWL_TERMINAL_DEFAULT);
  }
  PerRayData prd;
  owl::Ray ray;
  ray.origin = vec3f(0.f, 0.f, 0.f);
  int view_plane_center[3] = {0, 0, -1};

  ray.direction[0] =
      view_plane_center[0] - (self.camera_width / 2.0) + self.pixel_width * (pixelID.x + 0.5);
  ray.direction[1] =
      view_plane_center[1] + (self.camera_height / 2.0) - self.pixel_height * (pixelID.y + 0.5);
  ray.direction[2] = view_plane_center[2];

  ray.direction = normalize(ray.direction);

  vec3f color;
  owl::traceRay(/*accel to trace against*/ self.world,
                /*the ray to trace*/ ray,
                /*prd*/ color);

  float color_vector[3] = {color.x, color.y, color.z};
  self.pixmap[(pixelID.y * self.width + pixelID.x) * 3] = color_vector[0] * 255;
  self.pixmap[(pixelID.y * self.width + pixelID.x) * 3 + 1] = color_vector[1] * 255;
  self.pixmap[(pixelID.y * self.width + pixelID.x) * 3 + 2] = color_vector[2] * 255;
}

OPTIX_MISS_PROGRAM(miss)() {
  const vec2i pixelID = owl::getLaunchIndex();

  const MissProgData &self = owl::getProgramData<MissProgData>();

  vec3f &prd = owl::getPRD<vec3f>();
  int pattern = (pixelID.x / 8) ^ (pixelID.y / 8);
  prd = vec3f(0, 0, 0);
}

OPTIX_INTERSECT_PROGRAM(Spheres)() {
  const int primID = optixGetPrimitiveIndex();
  const auto &self = owl::getProgramData<SpheresList>().primitives[primID];

  const vec3f origin = optixGetWorldRayOrigin();
  const vec3f direction = optixGetWorldRayDirection();
  float hit_t = optixGetRayTmax();
  const float tmin = optixGetRayTmin();

  const vec3f origin_to_pos = origin - self.position;

  const float a = dot(direction, direction);
  const float b = 2 * dot(direction, origin_to_pos);
  const float c = dot(origin_to_pos, origin_to_pos) - (self.radius * self.radius);

  const float discriminant = b * b - 4 * a * c;
  if (discriminant < 0.f) {
    return;
  }
  else {
    float temp = (-b + sqrtf(discriminant)) / (2.0 * a);
    if (temp < hit_t && temp > tmin) {
      hit_t = temp;
    }
    vec3f &prd = owl::getPRD<vec3f>();
    prd = self.diffuse_color;
  }
  if (hit_t < optixGetRayTmax()) {
    optixReportIntersection(hit_t, 0);
  }
}

OPTIX_BOUNDS_PROGRAM(Spheres)(const void *geomData, box3f &primBounds, const int primID) {
  const SpheresList &self = *(const SpheresList *) geomData;
  const Sphere sphere = self.primitives[primID];
  primBounds =
      box3f().extend(sphere.position - sphere.radius).extend(sphere.position + sphere.radius);
}

OPTIX_CLOSEST_HIT_PROGRAM(Spheres)() {
  // const int primID = optixGetPrimitiveIndex();
  // const auto &self = owl::getProgramData<SpheresList>().primitives[primID];
  // PerRayData &prd = owl::getPRD<PerRayData>();
  // prd = self.diffuse_color;
}

OPTIX_INTERSECT_PROGRAM(Planes)() {
  const int primID = optixGetPrimitiveIndex();
  const auto &self = owl::getProgramData<PlanesList>().primitives[primID];

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
  vec3f &prd = owl::getPRD<vec3f>();
  prd = self.diffuse_color;

  if (hit_t < optixGetRayTmax()) {
    optixReportIntersection(hit_t, 0);
  }
}

// TODO: what does this do exactly?
OPTIX_BOUNDS_PROGRAM(Planes)(const void *geomData, box3f &primBounds, const int primID) {
  const PlanesList &self = *(const PlanesList *) geomData;
  const Plane plane = self.primitives[primID];
  primBounds = box3f(vec3f(-1.f, -1.f, 0.f), vec3f(+1.f, +1.f, +1.f));
}

OPTIX_CLOSEST_HIT_PROGRAM(Planes)() {
}

OPTIX_INTERSECT_PROGRAM(Quadrics)() {
  const int primID = optixGetPrimitiveIndex();
  const auto &self = owl::getProgramData<QuadricsList>().primitives[primID];

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
  vec3f &prd = owl::getPRD<vec3f>();
  prd = self.diffuse_color;

  if (hit_t < optixGetRayTmax()) {
    optixReportIntersection(hit_t, 0);
  }
}

// TODO: what does this do exactly?
OPTIX_BOUNDS_PROGRAM(Quadrics)(const void *geomData, box3f &primBounds, const int primID) {
  const QuadricsList &self = *(const QuadricsList *) geomData;
  const Quadric quadric = self.primitives[primID];
  primBounds = box3f(vec3f(-1.f, -1.f, 0.f), vec3f(+1.f, +1.f, +1.f));
}

OPTIX_CLOSEST_HIT_PROGRAM(Quadrics)() {
}