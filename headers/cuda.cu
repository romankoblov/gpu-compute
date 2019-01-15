// temporary taken from https://github.com/tbenthompson/cluda for testing purposes.
#define CUDA
// taken from pycuda._cluda
#define LOCAL_BARRIER __syncthreads()
#define WITHIN_KERNEL __device__
#define KERNEL extern "C" __global__
#define GLOBAL_MEM /* empty */
#define LOCAL_MEM __shared__
#define LOCAL_MEM_DYNAMIC extern __shared__
#define LOCAL_MEM_ARG /* empty */
#define CONSTANT __constant__
#define INLINE __forceinline__
#define SIZE_T unsigned int
#define VSIZE_T unsigned int
// used to align fields in structures
#define ALIGN(bytes) __align__(bytes)
//IF EVER NEEDED, THESE CAN BE CONVERTED TO DEFINES SO THAT THERE IS LESS PER
// KERNEL OVERHEAD.
WITHIN_KERNEL SIZE_T get_local_id(unsigned int dim)
{
    if(dim == 0) return threadIdx.x;
    if(dim == 1) return threadIdx.y;
    if(dim == 2) return threadIdx.z;
    return 0;
}
WITHIN_KERNEL SIZE_T get_group_id(unsigned int dim)
{
    if(dim == 0) return blockIdx.x;
    if(dim == 1) return blockIdx.y;
    if(dim == 2) return blockIdx.z;
    return 0;
}
WITHIN_KERNEL SIZE_T get_local_size(unsigned int dim)
{
    if(dim == 0) return blockDim.x;
    if(dim == 1) return blockDim.y;
    if(dim == 2) return blockDim.z;
    return 1;
}
WITHIN_KERNEL SIZE_T get_num_groups(unsigned int dim)
{
    if(dim == 0) return gridDim.x;
    if(dim == 1) return gridDim.y;
    if(dim == 2) return gridDim.z;
    return 1;
}
WITHIN_KERNEL SIZE_T get_global_size(unsigned int dim)
{
    return get_num_groups(dim) * get_local_size(dim);
}
WITHIN_KERNEL SIZE_T get_global_id(unsigned int dim)
{
    return get_local_id(dim) + get_group_id(dim) * get_local_size(dim);
}