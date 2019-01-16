use gpu_compute;

fn nv_device_pair(){
    // what about systems with multiple cards?
    // let cuda_dev = // first device;
    // let ocl_dev = // ocl device with same name as cuda?
}

fn main() {
    /*
    How it can work?
    - get cuda device and opencl device with same name (maybe later vulkan or other thing)
    - define kernel names which we want to inspect
    - CUDA:
    return PTX, return allocations/cmem/shmem (stack spils?) for each kernel
    - OCL:
        1. verify that legacy and non-legacy compiler outputs has same code
        2. return cl-nv-verbose output PARSED!
    So, we will have API like this:
    get_ptx() {
        is_nv -> None
        ...
    }
    */


    let header = r#"
        KERNEL void say_hi2(GLOBAL_MEM unsigned int *a)
        {
            #ifndef CUDA
            unsigned int gid = get_global_id(0);
            barrier(CLK_LOCAL_MEM_FENCE);
            #else
            unsigned int gid = 99;
            #endif
            a[gid] = 6;
            LOCAL_MEM unsigned int zzz[100];
            for (int i=0; i<100; i++)
                zzz[i] = gid*2;
            printf("ZZZ: %d \n", zzz[5]);
        }
        WITHIN_KERNEL void hi(unsigned int gid)
        { 
            printf("I am %d.\n", gid); 
        }
    "#;
    let kernel = r#"
        #include "/gpu_compute/platform.h"
        CONSTANT int pyopencl_defeat_cache_10 = 0;
        #include "world.cl"
        KERNEL void say_hi(GLOBAL_MEM unsigned int *a)
        {
            #ifndef CUDA
            unsigned int gid = get_global_id(0);
            barrier(CLK_LOCAL_MEM_FENCE);
            #else
            unsigned int gid = 5;
            #endif
            hi(gid);
            a[gid] = 5;

        }
        #ifdef CUDA
        __global__ void say_hi3()
        {
            printf("HI!!!!");
        }
        #endif
    "#;


    let cp = gpu_compute::Compute::new();
    for (idx, d) in cp.list_devices(){
        if d.vendor() != gpu_compute::enums::Vendor::NVIDIA { continue; }
        let q = d.queue();
        let mut p = gpu_compute::program::SourceProgramBuilder::new(Some("hello_world.c"))
            .debug()
            .compiler_opt(gpu_compute::enums::CompilerOpt::NV_PTAX)
            .add_header(header, "world.cl")
            .add_fn("say_hi")
            .add_fn("say_hi2")
            .set_source(&kernel);

        if let Some(ptx_info) =  q.get_ptx_info(&p) {
            println!("OMG: {} {} {} {}", d.name(), d.version(), d.platform().name(), ptx_info.ptx.len());
        }
        // if let Some(info) = program.get_ptx_info() {
        //     for (k, v) in info {
        //         println!("PTX_INFO: {} ==\t{}", k, v);
        //     }
        // }
    }
}