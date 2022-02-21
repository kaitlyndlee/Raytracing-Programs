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
      // printf("Position: [%f, %f, %f]\n", sphere.position.x, sphere.position.y,
      // sphere.position.z);
    }
  }

  float pixel_height = json_struct->camera_height / photo_data.height;
  float pixel_width = json_struct->camera_width / photo_data.width;

  //################ From OWL####################
  const int NUM_VERTICES = 8;
  vec3f vertices[NUM_VERTICES] = {{-1.f, -1.f, -10.f}, {+1.f, -1.f, -10.f}, {-1.f, +1.f, -10.f},
                                  {+1.f, +1.f, -10.f}, {-1.f, -1.f, -10.f}, {+1.f, -1.f, -10.f},
                                  {-1.f, +1.f, -10.f}, {+1.f, +1.f, -10.f}};

  const int NUM_INDICES = 12;
  vec3i indices[NUM_INDICES] = {{0, 1, 3}, {2, 3, 0}, {5, 7, 6}, {5, 6, 4}, {0, 4, 5}, {0, 5, 1},
                                {2, 3, 7}, {2, 7, 6}, {1, 5, 7}, {1, 7, 3}, {4, 0, 2}, {4, 2, 6}};

  //################ From OWL####################

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

  // -------------------------------------------------------
  // declare geometry type
  // -------------------------------------------------------
  OWLVarDecl trianglesGeomVars[] = {{"index", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, index)},
                                    {"vertex", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, vertex)},
                                    {"color", OWL_FLOAT3, OWL_OFFSETOF(TrianglesGeomData, color)}};
  OWLGeomType trianglesGeomType =
      owlGeomTypeCreate(owl, OWL_TRIANGLES, sizeof(TrianglesGeomData), trianglesGeomVars, 3);
  owlGeomTypeSetClosestHit(trianglesGeomType, 0, module, "TriangleMesh");

  OWLBuffer vertexBuffer = owlDeviceBufferCreate(owl, OWL_FLOAT3, NUM_VERTICES, vertices);
  OWLBuffer indexBuffer = owlDeviceBufferCreate(owl, OWL_INT3, NUM_INDICES, indices);
  OWLGeom trianglesGeom = owlGeomCreate(owl, trianglesGeomType);

  owlTrianglesSetVertices(trianglesGeom, vertexBuffer, NUM_VERTICES, sizeof(vec3f), 0);
  owlTrianglesSetIndices(trianglesGeom, indexBuffer, NUM_INDICES, sizeof(vec3i), 0);

  owlGeomSetBuffer(trianglesGeom, "vertex", vertexBuffer);
  owlGeomSetBuffer(trianglesGeom, "index", indexBuffer);
  owlGeomSet3f(trianglesGeom, "color", owl3f {0, 1, 0});

  OWLGroup trianglesGroup = owlTrianglesGeomGroupCreate(owl, 1, &trianglesGeom);
  owlGroupBuildAccel(trianglesGroup);
  // OWLGroup world = owlInstanceGroupCreate(owl, 1, &trianglesGroup);
  // owlGroupBuildAccel(world);

  OWLVarDecl spheresListVars[] = {{"primitives", OWL_BUFPTR, OWL_OFFSETOF(SpheresList, primitives)},
                                  {/* sentinel to mark end of list */}};
  OWLGeomType spheresGeomType =
      owlGeomTypeCreate(owl, OWL_GEOMETRY_USER, sizeof(SpheresList), spheresListVars, -1);
  owlGeomTypeSetClosestHit(spheresGeomType, 0, module, "Spheres");
  owlGeomTypeSetIntersectProg(spheresGeomType, 0, module, "Spheres");
  owlGeomTypeSetBoundsProg(spheresGeomType, module, "Spheres");

  owlBuildPrograms(owl);

  // ##################################################################
  // set up all the *GEOMS* we want to run that code on
  // ##################################################################
  OWLBuffer spheresBuffer =
      owlDeviceBufferCreate(owl, OWL_USER_TYPE(spheres[0]), spheres.size(), spheres.data());
  OWLGeom spheresGeom = owlGeomCreate(owl, spheresGeomType);
  owlGeomSetPrimCount(spheresGeom, spheres.size());
  owlGeomSetBuffer(spheresGeom, "primitives", spheresBuffer);

  // ##################################################################
  // set up all *ACCELS* we need to trace into those groups
  // ##################################################################

  OWLGeom userGeoms[] = {spheresGeom};
  OWLGroup userGeomGroup = owlUserGeomGroupCreate(owl, 1, userGeoms);
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
