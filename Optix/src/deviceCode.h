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

#include "v3math.h"

#include <owl/common/math/vec.h>
#include <owl/owl.h>

using namespace owl;

/* variables for the triangle mesh geometry */
struct TrianglesGeomData {
  /*! base color we use for the entire mesh */
  vec3f color;
  /*! array/buffer of vertex indices */
  vec3i *index;
  /*! array/buffer of vertex positions */
  vec3f *vertex;
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
};

struct PerRayData {
  vec3f color;
  float distance;
};

/* variables for the miss program */
struct MissProgData {
  vec3f color0;
  vec3f color1;
};
