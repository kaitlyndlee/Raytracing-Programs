set(CMAKE_VERBOSE_MAKEFILE on)
set(CUDA_NVCC_FLAGS --gpu-architecture=compute_75 -rdc=true)

# One way to compile using outside function - does not work
# NVCUDA_COMPILE_PTX( SOURCES deviceCode.cu v3math.cu
#                     DEPENDENCIES deviceCode.h v3math.h
#                     TARGET_PATH deviceCode_ptx
#                     GENERATED_FILES PTX_SOURCES
#                     NVCC_OPTIONS "--gpu-architecture=compute_75" "--use_fast_math" "--relocatable-device-code=true"
#                   )

# # Include source files for compilation
# add_executable(raytrace hostCode.cpp ppm.cpp ppm.h)

# # Include libraries to link
# target_link_libraries(raytrace 
#   PRIVATE owl::owl cuda cudadevrt cudart_static rt pthread dl 
# )

# cuda_add_library(v3math
#   v3math.cu
#   v3math.h)
# set_target_properties( v3math
#                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Alternative way to compile PTX using the owl custom command
embed_ptx(
  OUTPUT_TARGET
    raytrace-ptx
  PTX_LINK_LIBRARIES
    owl::owl
  SOURCES
    deviceCode.cu
)

add_executable(raytrace hostCode.cpp ppm.cpp ppm.h objects.cpp objects.h parse.cpp parse.h v3math.cu v3math.h)
set_target_properties(raytrace
  PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_USE_STATIC_CUDA_RUNTIME ON
)
target_link_libraries(raytrace
  PRIVATE
    raytrace-ptx
    owl::owl
)

# # Another way to compile - does not work
# cuda_compile_and_embed(deviceCode_ptx deviceCode.cu)
# add_executable(raytrace hostCode.cpp ppm.cpp ppm.h objects.cpp objects.h parse.cpp parse.h v3math.cpp v3math.h)
# target_link_libraries(raytrace
#   PRIVATE
#     owl::owl
#     ${deviceCode_ptx}
# )