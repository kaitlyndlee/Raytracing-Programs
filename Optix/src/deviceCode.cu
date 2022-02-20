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

// // OPTIX_RAYGEN_PROGRAM() is a simple macro defined in deviceAPI.h to add
// // standard code for defining a shader method. It puts:
// //   extern "C" __global__ void __raygen__##programName
// // in front of the program name given
// OPTIX_RAYGEN_PROGRAM(simpleRayGen)() {
//   // read in the program data set by the calling program hostCode.cpp using
//   // lloSbtRayGensBuild; see RayGenData in deviceCode.h
//   const RayGenData &self = owl::getProgramData<RayGenData>();
//   // Under the hood, OptiX maps rays generated in CUDA thread blocks to a pixel
//   // ID, where the ID is a 2D vector, 0 to frame buffer width-1, 0 to height-1
//   const vec2i pixelID = owl::getLaunchIndex();
//   if (pixelID == owl::vec2i(0)) {
//     // the first thread ID is always (0,0), so we can generate a message to show
//     // things are working
//     printf("%sHello OptiX From your First RayGen Program%s\n",
//            OWL_TERMINAL_CYAN, OWL_TERMINAL_DEFAULT);
//   }

//   // Generate a simple checkerboard pattern as a test. Note that the upper left
//   // corner is pixel (0,0).
//   int pattern = (pixelID.x / 8) ^ (pixelID.y / 8);
//   // alternate pattern, showing that pixel (0,0) is in the upper left corner
//   // pattern = (pixelID.x*pixelID.x + pixelID.y*pixelID.y) / 100000;
//   const vec3f color = (pattern & 1) ? self.color1 : self.color0;

//   // // find the frame buffer location (x + width*y) and put the "computed"
//   // result
//   // // there
//   // const int fbOfs = pixelID.x + self.fbSize.x * pixelID.y;
//   // self.fbPtr[fbOfs] = owl::make_rgba(color);

//   float color_vector[3] = {color.x, color.y, color.z};
//   self.pixmap[(pixelID.y * self.width + pixelID.x) * 3] = color_vector[0] * 255;
//   self.pixmap[(pixelID.y * self.width + pixelID.x) * 3 + 1] = color_vector[1] * 255;
//   self.pixmap[(pixelID.y * self.width + pixelID.x) * 3 + 2] = color_vector[2] * 255;
// }

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
  vec3f &prd = owl::getPRD<vec3f>();

  const TrianglesGeomData &self = owl::getProgramData<TrianglesGeomData>();
  
  // compute normal:
  const int   primID = optixGetPrimitiveIndex();
  const vec3i index  = self.index[primID];
  const vec3f &A     = self.vertex[index.x];
  const vec3f &B     = self.vertex[index.y];
  const vec3f &C     = self.vertex[index.z];
  const vec3f Ng     = normalize(cross(B-A,C-A));

  const vec3f rayDir = optixGetWorldRayDirection();
  prd = (.2f + .8f*fabs(dot(rayDir,Ng)))*self.color;
}

OPTIX_RAYGEN_PROGRAM(simpleRayGen)()
{
  const RayGenData &self = owl::getProgramData<RayGenData>();
  const vec2i pixelID = owl::getLaunchIndex();
  if (pixelID == owl::vec2i(0)) {
    printf("%sHello OptiX From your First RayGen Program%s\n",
           OWL_TERMINAL_CYAN,
           OWL_TERMINAL_DEFAULT);
  }

  owl::Ray ray;
  ray.origin = vec3f(0.f, 0.f, 0.f);
  int view_plane_center[3] = {0, 0, -1};

  ray.direction[0] = view_plane_center[0] - (self.camera_width / 2.0) + self.pixel_width * (pixelID.x + 0.5);
  ray.direction[1] = view_plane_center[1] + (self.camera_height / 2.0) - self.pixel_height * (pixelID.y + 0.5);
  ray.direction[2] = view_plane_center[2];

  ray.direction = normalize(ray.direction);

  vec3f color;
  owl::traceRay(/*accel to trace against*/self.world,
                /*the ray to trace*/ray,
                /*prd*/color);
    
  float color_vector[3] = {color.x, color.y, color.z};
  self.pixmap[(pixelID.y * self.width + pixelID.x) * 3] = color_vector[0] * 255;
  self.pixmap[(pixelID.y * self.width + pixelID.x) * 3 + 1] = color_vector[1] * 255;
  self.pixmap[(pixelID.y * self.width + pixelID.x) * 3 + 2] = color_vector[2] * 255;

}

OPTIX_MISS_PROGRAM(miss)()
{
  const vec2i pixelID = owl::getLaunchIndex();

  const MissProgData &self = owl::getProgramData<MissProgData>();
  
  vec3f &prd = owl::getPRD<vec3f>();
  int pattern = (pixelID.x / 8) ^ (pixelID.y/8);
  prd = (pattern&1) ? self.color1 : self.color0;
}
