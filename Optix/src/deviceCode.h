// ======================================================================== //
// Copyright 2019-2020 Ingo Wald                                            //
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

#pragma once

#include "objects.h"
#include "v3math.h"

#include <owl/common/math/vec.h>
#include <owl/owl.h>

using namespace owl;

// TODO: Combine objets.h and these structs
struct Light {
  vec3f position;
  vec3f color;
  vec3f radial_coef;
  float theta;
  float cos_theta;
  float a0;
  vec3f direction;
  light_type_t type;
};

struct Sphere {
  vec3f diffuse_color;
  vec3f specular_color;
  vec3f position;
  float reflectivity;
  float refractivity;
  float ior;
  float radius;
};

struct SpheresList {
  Sphere *primitives;
};

struct Plane {
  vec3f diffuse_color;
  vec3f specular_color;
  vec3f position;
  float reflectivity;
  float refractivity;
  float ior;
  vec3f normal;
};

struct PlanesList {
  Plane *primitives;
};

struct Quadric {
  vec3f diffuse_color;
  vec3f specular_color;
  vec3f position;
  float reflectivity;
  float refractivity;
  float ior;
  float a;
  float b;
  float c;
  float d;
  float e;
  float f;
  float g;
  float h;
  float i;
  float j;
};

struct QuadricsList {
  Quadric *primitives;
};

struct RayGenData {
  uint8_t *pixmap;
  int height;
  int width;
  float pixel_height;
  float pixel_width;
  float camera_height;
  float camera_width;
  OptixTraversableHandle world;
  Light *lights;
  int num_lights;
};

struct PerRayData {
  vec3f diffuse_color;
  vec3f specular_color;
  float reflectivity;
  float refractivity;
  float distance;
  vec3f intersection;
  vec3f normal;
  int primId;
  shape_type_t shape_type;
  PerRayData *prev_intersection;
};
