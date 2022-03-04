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

// This program shows a minimal setup: no geometry, just a ray generation
// shader that accesses the pixels and draws a checkerboard pattern to
// the output file ll00-rayGenOnly.png

// public owl API
#include "cuda_defs.h"
#include "deviceCode.h"
#include "objects.h"
#include "parse.h"
#include "ppm.h"

#include <cstring>
#include <owl/owl.h>
#include <sys/time.h>
#include <vector>

#define MAX_SHAPES 128

#define LOG(message)                                          \
  std::cout << OWL_TERMINAL_BLUE;                             \
  std::cout << "#owl.sample(main): " << message << std::endl; \
  std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                       \
  std::cout << OWL_TERMINAL_LIGHT_BLUE;                       \
  std::cout << "#owl.sample(main): " << message << std::endl; \
  std::cout << OWL_TERMINAL_DEFAULT;

extern "C" char deviceCode_ptx[];
extern "C" char v3_math_ptx[];

int main(int argc, char **argv) {

  if (argc != 5) {
    printf("Usage: raytrace WIDTH HEIGHT INPUT_SCENE OUTPUT_IMAGE\n");
    exit(0);
  }

  struct timeval start, end;
  gettimeofday(&start, NULL);

  int width = atoi(argv[1]);
  int height = atoi(argv[2]);

  FILE *input_json = fopen(argv[3], "r+");
  if (input_json == NULL) {
    fprintf(stderr, "Error: Unable to open the input scene file: %s\n", argv[3]);
    exit(1);
  }

  FILE *output_image = fopen(argv[4], "wb");
  if (output_image == NULL) {
    fprintf(stderr, "Error: Unable to open the output image file: %s\n", argv[4]);
    exit(1);
  }

  json_data_t *json_struct = (json_data_t *) malloc(sizeof(json_data_t));
  json_struct->shapes_list = (shape_t *) malloc(MAX_SHAPES * sizeof(shape_t));
  json_struct->lights_list = (light_t *) malloc(MAX_SHAPES * sizeof(light_t));
  parse_json(input_json, json_struct);

  PPMFormat photo_data;
  photo_data.maxColor = 255;
  photo_data.height = height;
  photo_data.width = width;
  photo_data.size = photo_data.width * photo_data.height * 3;
  photo_data.pixmap = (uint8_t *) malloc(photo_data.size);

  // Create vector of spheres only for OWL
  std::vector<Sphere> spheres;
  std::vector<Plane> planes;
  std::vector<Quadric> quadrics;
  for (int count = 0; count < json_struct->num_shapes; count++) {
    if (json_struct->shapes_list[count].type == SPHERE) {
      Sphere sphere;
      sphere.diffuse_color = vec3f(json_struct->shapes_list[count].diffuse_color[0],
                                   json_struct->shapes_list[count].diffuse_color[1],
                                   json_struct->shapes_list[count].diffuse_color[2]);
      sphere.specular_color = vec3f(json_struct->shapes_list[count].specular_color[0],
                                    json_struct->shapes_list[count].specular_color[1],
                                    json_struct->shapes_list[count].specular_color[2]);
      sphere.position = vec3f(json_struct->shapes_list[count].position[0],
                              json_struct->shapes_list[count].position[1],
                              json_struct->shapes_list[count].position[2]);
      sphere.reflectivity = json_struct->shapes_list[count].reflectivity;
      sphere.refractivity = json_struct->shapes_list[count].refractivity;
      sphere.ior = json_struct->shapes_list[count].ior;
      sphere.radius = json_struct->shapes_list[count].radius;
      spheres.push_back(sphere);
    }
    else if (json_struct->shapes_list[count].type == PLANE) {
      Plane plane;
      plane.diffuse_color = vec3f(json_struct->shapes_list[count].diffuse_color[0],
                                  json_struct->shapes_list[count].diffuse_color[1],
                                  json_struct->shapes_list[count].diffuse_color[2]);
      plane.specular_color = vec3f(json_struct->shapes_list[count].specular_color[0],
                                   json_struct->shapes_list[count].specular_color[1],
                                   json_struct->shapes_list[count].specular_color[2]);
      plane.position = vec3f(json_struct->shapes_list[count].position[0],
                             json_struct->shapes_list[count].position[1],
                             json_struct->shapes_list[count].position[2]);
      plane.reflectivity = json_struct->shapes_list[count].reflectivity;
      plane.refractivity = json_struct->shapes_list[count].refractivity;
      plane.ior = json_struct->shapes_list[count].ior;
      plane.normal = vec3f(json_struct->shapes_list[count].normal[0],
                           json_struct->shapes_list[count].normal[1],
                           json_struct->shapes_list[count].normal[2]);
      planes.push_back(plane);
    }
    else if (json_struct->shapes_list[count].type == QUADRIC) {
      Quadric quadric;
      quadric.diffuse_color = vec3f(json_struct->shapes_list[count].diffuse_color[0],
                                    json_struct->shapes_list[count].diffuse_color[1],
                                    json_struct->shapes_list[count].diffuse_color[2]);
      quadric.specular_color = vec3f(json_struct->shapes_list[count].specular_color[0],
                                     json_struct->shapes_list[count].specular_color[1],
                                     json_struct->shapes_list[count].specular_color[2]);
      quadric.position = vec3f(json_struct->shapes_list[count].position[0],
                               json_struct->shapes_list[count].position[1],
                               json_struct->shapes_list[count].position[2]);
      quadric.reflectivity = json_struct->shapes_list[count].reflectivity;
      quadric.refractivity = json_struct->shapes_list[count].refractivity;
      quadric.a = json_struct->shapes_list[count].a;
      quadric.b = json_struct->shapes_list[count].b;
      quadric.c = json_struct->shapes_list[count].c;
      quadric.d = json_struct->shapes_list[count].d;
      quadric.e = json_struct->shapes_list[count].e;
      quadric.f = json_struct->shapes_list[count].f;
      quadric.g = json_struct->shapes_list[count].g;
      quadric.h = json_struct->shapes_list[count].h;
      quadric.i = json_struct->shapes_list[count].i;
      quadric.j = json_struct->shapes_list[count].j;
      quadric.ior = json_struct->shapes_list[count].ior;
      quadrics.push_back(quadric);
    }
  }

  float pixel_height = json_struct->camera_height / photo_data.height;
  float pixel_width = json_struct->camera_width / photo_data.width;

  LOG("owl example '" << argv[0] << "' starting up");
  LOG("building module, programs, and pipeline");

  // Initialize CUDA and OptiX 7, and create an "owl device," a context to hold
  // the ray generation shader and output buffer. The "1" is the number of
  // devices requested.
  OWLContext owl = owlContextCreate(nullptr, 1);
  // PTX is the intermediate code that the CUDA deviceCode.cu shader program is
  // converted into. You can see the machine-centric PTX code in
  // build\samples\s00-rayGenOnly\cuda_compile_ptx_1_generated_deviceCode.cu.ptx_embedded.c
  // This PTX intermediate code representation is then compiled into an OptiX
  // module. See https://devblogs.nvidia.com/how-to-get-started-with-optix-7/
  // for more information.
  OWLModule module = owlModuleCreate(owl, deviceCode_ptx);

  // ##################################################################
  // set up all the *GEOMETRY* graph we want to render
  // ##################################################################

  OWLVarDecl spheresListVars[] = {{"primitives", OWL_BUFPTR, OWL_OFFSETOF(SpheresList, primitives)},
                                  {/* sentinel to mark end of list */}};
  OWLGeomType spheresGeomType =
      owlGeomTypeCreate(owl, OWL_GEOMETRY_USER, sizeof(SpheresList), spheresListVars, -1);
  owlGeomTypeSetClosestHit(spheresGeomType, 0, module, "Spheres");
  owlGeomTypeSetIntersectProg(spheresGeomType, 0, module, "Spheres");
  owlGeomTypeSetBoundsProg(spheresGeomType, module, "Spheres");

  OWLVarDecl planesListVars[] = {{"primitives", OWL_BUFPTR, OWL_OFFSETOF(PlanesList, primitives)},
                                 {/* sentinel to mark end of list */}};
  OWLGeomType planesGeomType =
      owlGeomTypeCreate(owl, OWL_GEOMETRY_USER, sizeof(PlanesList), planesListVars, -1);
  owlGeomTypeSetClosestHit(planesGeomType, 0, module, "Planes");
  owlGeomTypeSetIntersectProg(planesGeomType, 0, module, "Planes");
  owlGeomTypeSetBoundsProg(planesGeomType, module, "Planes");

  OWLVarDecl quadricsListVars[] = {
      {"primitives", OWL_BUFPTR, OWL_OFFSETOF(QuadricsList, primitives)},
      {/* sentinel to mark end of list */}};

  OWLGeomType quadricsGeomType =
      owlGeomTypeCreate(owl, OWL_GEOMETRY_USER, sizeof(QuadricsList), quadricsListVars, -1);
  owlGeomTypeSetClosestHit(quadricsGeomType, 0, module, "Quadrics");
  owlGeomTypeSetIntersectProg(quadricsGeomType, 0, module, "Quadrics");
  owlGeomTypeSetBoundsProg(quadricsGeomType, module, "Quadrics");

  owlBuildPrograms(owl);

  // ##################################################################
  // set up all the *GEOMS* we want to run that code on
  // ##################################################################
  OWLBuffer spheresBuffer =
      owlDeviceBufferCreate(owl, OWL_USER_TYPE(spheres[0]), spheres.size(), spheres.data());
  OWLGeom spheresGeom = owlGeomCreate(owl, spheresGeomType);
  owlGeomSetPrimCount(spheresGeom, spheres.size());
  owlGeomSetBuffer(spheresGeom, "primitives", spheresBuffer);

  OWLBuffer planesBuffer =
      owlDeviceBufferCreate(owl, OWL_USER_TYPE(planes[0]), planes.size(), planes.data());
  OWLGeom planesGeom = owlGeomCreate(owl, planesGeomType);
  owlGeomSetPrimCount(planesGeom, planes.size());
  owlGeomSetBuffer(planesGeom, "primitives", planesBuffer);

  OWLBuffer quadricsBuffer =
      owlDeviceBufferCreate(owl, OWL_USER_TYPE(quadrics[0]), quadrics.size(), quadrics.data());
  OWLGeom quadricsGeom = owlGeomCreate(owl, quadricsGeomType);
  owlGeomSetPrimCount(quadricsGeom, quadrics.size());
  owlGeomSetBuffer(quadricsGeom, "primitives", quadricsBuffer);

  // ##################################################################
  // set up all *ACCELS* we need to trace into those groups
  // ##################################################################

  OWLGeom userGeoms[] = {spheresGeom, planesGeom, quadricsGeom};
  OWLGroup userGeomGroup = owlUserGeomGroupCreate(owl, 3, userGeoms);
  owlGroupBuildAccel(userGeomGroup);

  OWLGroup world = owlInstanceGroupCreate(owl, 1);
  owlInstanceGroupSetChild(world, 0, userGeomGroup);
  owlGroupBuildAccel(world);

  // ##################################################################
  // set miss and raygen program required for SBT
  // ##################################################################

  // -------------------------------------------------------
  // set up miss prog
  // -------------------------------------------------------
  OWLVarDecl missProgVars[] = {{"color0", OWL_FLOAT3, OWL_OFFSETOF(MissProgData, color0)},
                               {"color1", OWL_FLOAT3, OWL_OFFSETOF(MissProgData, color1)},
                               {/* sentinel to mark end of list */}};
  // ----------- create object  ----------------------------
  OWLMissProg missProg =
      owlMissProgCreate(owl, module, "miss", sizeof(MissProgData), missProgVars, -1);

  // ----------- set variables  ----------------------------
  owlMissProgSet3f(missProg, "color0", owl3f {.8f, 0.f, 0.f});
  owlMissProgSet3f(missProg, "color1", owl3f {.8f, .8f, .8f});

  // Allocate room for one RayGen shader, create it, and
  // hold on to it with the "owl" context
  OWLVarDecl rayGenVars[] = {{"pixmap", OWL_BUFPTR, OWL_OFFSETOF(RayGenData, pixmap)},
                             {"width", OWL_INT, OWL_OFFSETOF(RayGenData, width)},
                             {"height", OWL_INT, OWL_OFFSETOF(RayGenData, height)},
                             {"pixel_width", OWL_FLOAT, OWL_OFFSETOF(RayGenData, pixel_width)},
                             {"pixel_height", OWL_FLOAT, OWL_OFFSETOF(RayGenData, pixel_height)},
                             {"camera_width", OWL_FLOAT, OWL_OFFSETOF(RayGenData, camera_width)},
                             {"camera_height", OWL_FLOAT, OWL_OFFSETOF(RayGenData, camera_height)},
                             {"world", OWL_GROUP, OWL_OFFSETOF(RayGenData, world)},
                             {/* sentinel: */ nullptr}};

  OWLRayGen rayGen =
      owlRayGenCreate(owl, module, "simpleRayGen", sizeof(RayGenData), rayGenVars, -1);

  // ------------------------------------------------------------------
  // alloc buffers
  // ------------------------------------------------------------------
  LOG("allocating frame buffer");
  // Create a frame buffer as page-locked, aka "pinned" memory. See CUDA
  // documentation for benefits and more info.
  OWLBuffer frameBuffer = owlHostPinnedBufferCreate(owl,
                                                    /*type:*/ OWL_UCHAR,
                                                    /*size:*/ photo_data.size);

  // ------------------------------------------------------------------
  // build Shader Binding Table (SBT) required to trace the groups
  // ------------------------------------------------------------------
  owlRayGenSetBuffer(rayGen, "pixmap", frameBuffer);
  owlRayGenSet1i(rayGen, "width", photo_data.width);
  owlRayGenSet1i(rayGen, "height", photo_data.height);
  owlRayGenSet1f(rayGen, "pixel_height", pixel_height);
  owlRayGenSet1f(rayGen, "pixel_width", pixel_width);
  owlRayGenSet1f(rayGen, "camera_height", json_struct->camera_height);
  owlRayGenSet1f(rayGen, "camera_width", json_struct->camera_width);
  owlRayGenSetGroup(rayGen, "world", world);

  // (re-)builds all optix programs, with current pipeline settings
  owlBuildPrograms(owl);
  // Create the pipeline. Note that owl will (kindly) warn there are no geometry
  // and no miss programs defined.
  owlBuildPipeline(owl);
  // Build a shader binding table entry for the ray generation record.
  owlBuildSBT(owl);

  // ##################################################################
  // now that everything is ready: launch it ....
  // ##################################################################

  LOG("executing the launch ...");
  // Normally launching without a hit or miss shader causes OptiX to trigger
  // warnings. Owl's wrapper call here will set up fake hit and miss records
  // into the SBT to avoid these.
  owlRayGenLaunch2D(rayGen, photo_data.width, photo_data.height);

  const uint8_t *fb = (const uint8_t *) owlBufferGetPointer(frameBuffer, 0);
  memcpy(photo_data.pixmap, fb, photo_data.size);
  ppm_WriteOutP3(photo_data, output_image);

  // ##################################################################
  // and finally, clean up
  // ##################################################################

  LOG("cleaning up ...");
  owlModuleRelease(module);
  owlRayGenRelease(rayGen);
  owlBufferRelease(frameBuffer);
  owlContextDestroy(owl);

  LOG_OK("seems all went OK; app is done, this should be the last output ...");
}
