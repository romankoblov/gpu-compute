// temporary taken from https://github.com/tbenthompson/cluda for testing purposes.
// taken from pyopencl._cluda
#define LOCAL_BARRIER barrier(CLK_LOCAL_MEM_FENCE)
// 'static' helps to avoid the "no previous prototype for function" warning
#if __OPENCL_VERSION__ >= 120
#define WITHIN_KERNEL static
#else
#define WITHIN_KERNEL
#endif
#define KERNEL __kernel
#define GLOBAL_MEM __global
#define LOCAL_MEM __local
#define LOCAL_MEM_DYNAMIC __local
#define LOCAL_MEM_ARG __local
#define CONSTANT __constant
// INLINE is already defined in Beignet driver
#ifndef INLINE
#define INLINE inline
#endif
#define SIZE_T size_t
#define VSIZE_T size_t
// used to align fields in structures
#define ALIGN(bytes) __attribute__ ((aligned(bytes)))
#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64: enable
#endif