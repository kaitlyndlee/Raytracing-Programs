#ifndef CUDA_DEFS_H
#define CUDA_DEFS_H

#ifdef __CUDACC__
  #define CUDA_CALLABLE_MEMBER __host__ __device__
  #define INLINE __forceinline__
  #define DEVICE_ONLY __device__
  #define HOST_ONLY __host__
#else
  #define CUDA_CALLABLE_MEMBER
  #define INLINE
  #define DEVICE_ONLY
  #define HOST_ONLY
#endif

#endif // CUDA_DEFS_H