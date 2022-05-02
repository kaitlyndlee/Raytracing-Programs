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

void warm_up_gpu(int device);

int main(int argc, char **argv) {

  if (argc != 5) {
    printf("Usage: raytrace WIDTH HEIGHT INPUT_SCENE OUTPUT_IMAGE\n");
    exit(0);
  }

  struct timeval start, end;

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

  // TODO: Rrefactor to use parse.cpp
  // Separate shapes into separate vectors
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

  Light *lights = (Light *) malloc(json_struct->num_lights * sizeof(Light));
  for (int count = 0; count < json_struct->num_lights; count++) {
    Light light;
    light.position = vec3f(json_struct->lights_list[count].position[0],
                           json_struct->lights_list[count].position[1],
                           json_struct->lights_list[count].position[2]);
    light.color =
        vec3f(json_struct->lights_list[count].color[0], json_struct->lights_list[count].color[1],
              json_struct->lights_list[count].color[2]);

    light.radial_coef = vec3f(json_struct->lights_list[count].radial_coef[0],
                              json_struct->lights_list[count].radial_coef[1],
                              json_struct->lights_list[count].radial_coef[2]);

    light.direction = vec3f(json_struct->lights_list[count].direction[0],
                            json_struct->lights_list[count].direction[1],
                            json_struct->lights_list[count].direction[2]);

    light.theta = json_struct->lights_list[count].theta;
    light.cos_theta = json_struct->lights_list[count].cos_theta;
    light.a0 = json_struct->lights_list[count].a0;
    light.type = json_struct->lights_list[count].type;

    lights[count] = light;
  }

  float pixel_height = json_struct->camera_height / photo_data.height;
  float pixel_width = json_struct->camera_width / photo_data.width;

  warm_up_gpu(0);

  gettimeofday(&start, NULL);

  OWLContext owl = owlContextCreate(nullptr, 1);
  OWLModule module = owlModuleCreate(owl, deviceCode_ptx);

  // Set up the geometry
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

  // Set up all acceleration objects
  OWLGeom userGeoms[] = {spheresGeom, planesGeom, quadricsGeom};
  OWLGroup userGeomGroup = owlUserGeomGroupCreate(owl, 3, userGeoms);
  owlGroupBuildAccel(userGeomGroup);

  OWLGroup world = owlInstanceGroupCreate(owl, 1);
  owlInstanceGroupSetChild(world, 0, userGeomGroup);
  owlGroupBuildAccel(world);

  OWLMissProg missProg = owlMissProgCreate(owl, module, "miss", 0, NULL, -1);

  OWLVarDecl rayGenVars[] = {{"pixmap", OWL_BUFPTR, OWL_OFFSETOF(RayGenData, pixmap)},
                             {"width", OWL_INT, OWL_OFFSETOF(RayGenData, width)},
                             {"height", OWL_INT, OWL_OFFSETOF(RayGenData, height)},
                             {"pixel_width", OWL_FLOAT, OWL_OFFSETOF(RayGenData, pixel_width)},
                             {"pixel_height", OWL_FLOAT, OWL_OFFSETOF(RayGenData, pixel_height)},
                             {"camera_width", OWL_FLOAT, OWL_OFFSETOF(RayGenData, camera_width)},
                             {"camera_height", OWL_FLOAT, OWL_OFFSETOF(RayGenData, camera_height)},
                             {"world", OWL_GROUP, OWL_OFFSETOF(RayGenData, world)},
                             {"lights", OWL_BUFPTR, OWL_OFFSETOF(RayGenData, lights)},
                             {"num_lights", OWL_INT, OWL_OFFSETOF(RayGenData, num_lights)},
                             {/* sentinel: */ nullptr}};

  OWLRayGen rayGen = owlRayGenCreate(owl, module, "rayGen", sizeof(RayGenData), rayGenVars, -1);

  // TODO: allocating a device memory buffer instead of pinned here
  OWLBuffer frameBuffer = owlHostPinnedBufferCreate(owl,
                                                    /*type:*/ OWL_UCHAR,
                                                    /*size:*/ photo_data.size);

  OWLBuffer lightsBuffer =
      owlDeviceBufferCreate(owl, OWL_USER_TYPE(lights[0]), json_struct->num_lights, lights);

  // Build Shader Binding Table (SBT) required to trace the groups
  owlRayGenSetBuffer(rayGen, "pixmap", frameBuffer);
  owlRayGenSet1i(rayGen, "width", photo_data.width);
  owlRayGenSet1i(rayGen, "height", photo_data.height);
  owlRayGenSet1f(rayGen, "pixel_height", pixel_height);
  owlRayGenSet1f(rayGen, "pixel_width", pixel_width);
  owlRayGenSet1f(rayGen, "camera_height", json_struct->camera_height);
  owlRayGenSet1f(rayGen, "camera_width", json_struct->camera_width);
  owlRayGenSetGroup(rayGen, "world", world);
  owlRayGenSetBuffer(rayGen, "lights", lightsBuffer);
  owlRayGenSet1i(rayGen, "num_lights", json_struct->num_lights);

  owlBuildPrograms(owl);
  owlBuildPipeline(owl);
  owlBuildSBT(owl);

  owlRayGenLaunch2D(rayGen, photo_data.width, photo_data.height);

  const uint8_t *fb = (const uint8_t *) owlBufferGetPointer(frameBuffer, 0);
  memcpy(photo_data.pixmap, fb, photo_data.size);
  ppm_WriteOutP3(photo_data, output_image);

  owlModuleRelease(module);
  owlRayGenRelease(rayGen);
  owlBufferRelease(frameBuffer);
  owlBufferRelease(lightsBuffer);
  owlContextDestroy(owl);

  gettimeofday(&end, NULL);
  double elapsed =
      (((end.tv_sec * 1000000.0 + end.tv_usec) - (start.tv_sec * 1000000.0 + start.tv_usec)) /
       1000000.00);
  printf("Time (sec) to create a %dx%d image with %d shape(s) and %d light(s): %f\n", width, height,
         json_struct->num_shapes, json_struct->num_lights, elapsed);
}

__global__ void warmup(unsigned int *tmp) {
  if (threadIdx.x == 0)
    *tmp = 555;

  return;
}

void warm_up_gpu(int device) {
  cudaSetDevice(device);
  unsigned int *dev_tmp;
  unsigned int *tmp;
  tmp = (unsigned int *) malloc(sizeof(unsigned int));
  *tmp = 0;
  cudaMalloc((unsigned int **) &dev_tmp, sizeof(unsigned int));

  warmup<<<1, 256>>>(dev_tmp);

  // copy data from device to host
  cudaMemcpy(tmp, dev_tmp, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  cudaFree(dev_tmp);
  free(tmp);

  return;
}