gpu-compute
===========

Warning: this crate is Work In Progress, API will change a lot
--------------------------------------------------------------

Main idea of this crate to provide unified API for GPU
computation across different backends:

- CUDA
- OpenCL
- Vulkan/Metal shaders later

Currently its WIP prototype with pretty ugly code, API will change later.

Tools (WIP)
-----------

- get_ptx — get PTX for different backends for difference check, and shows registers/memory allocations for kernel via different backends. Only for NVIDIA cards.
- gc_test — simple tests via json file for kernel internal functions, supports bench via clock function
- gc_grid_bench — find best grid configuration for specific kernel

Architecture
------------

```rust
Platform
-> Device
   -> Context(Device, but can be multiple)
      -> Queue/Stream
         -> ProgramBuilder
            - build
```

- Currently I don't see reasons to have separated Context and Queue structs.
- OpenCL supports global_size which is not multiple of grid (like size: 3),
  but CUDA doesn't. There is two option to solve that:
  - GLOBAL_MEM u32 cnt as first arg and later check in kernel
  - just copy last items of input and remove them from output, but that won't
    work for atomic_output.
- I don't see reasons to support multiple gpu at same time:
  - Not sure if CUDA/shaders supports it correctly
  - It will be probably better to do thread-per-device pool wrapper
- At start we don't need separate queues for device.
- Program (piece of source code with headers) builds on specific device
- Program returns Functions, which has:
  - Inputs (named)
  - Outputs (named):
    - output: output size is same as input (like input: i_A+i_B=o_C)
    - atomic_output: output via atomic counter for fn's like 'is_prime',
      where output size is much less than input
- Pipeline is chain of Functions with intermediate buffers:
  - add_arg_input("intermedia_buffer1")
- Pipeline/Functions can be exported by some crate? Like gpuc-jpeg will provide
  pipeline for processing jpeg stuff?
- Each function in pipeline can have predefined grid structure, which can be
  redefined later.
- Need to have nice way to pass predefined const to kernel on comp time
- Device . get_version -> major, minor:
  - CUDA: returns compute capability
  - OpenCL: returns supported OpenCL version
- Stub header always added to header list, its up to user include it or not.

API
---

This API is not implemented yet, just design:
```rust
/*
Returns default device or specified by first arg or GPU_DEV env variable, like: 'cu0'
*/
let device = gpu_compute::default_device(None);

let say_hi = gpu_compute::FunctionBuilder()
   .parallel(2) // two threads process same item
   .grid_x(10, 50) // -> size=10*50
   .grid_xy((10, 20), (10, 20)) // size=(10*20)*(10*20)
   .grid_xyz...
   .input_cnt() // first input -- count of items in InputBuffer
   // input some vec of size less than grid size (or exact size if there is no 
   // input_cnt)
   .input<u32>("a")
   .input_const("b", 100) // const size (no depend on grid size buffer)
   .output("c") // same size as grid
   .atomic_output("c2")

let pipileline = gpu_compute::PipelineBuilder()
   .buffer("buf0", <u32>, 500) // sets intermediate buffer
   .buffer("buf1", <u64>, 100)
   .buffer("buf2", &struct_vec: Vec<SomeStruct>) // load buffer from mem
   // Initialize function, which will setup data in first buffer
   .init_fn(init_fn, ["buf0"])



let program = gpu_compute::SourceProgramBuilder()
   .debug() // Enable debug (lineinfo for cuda, -cl-nv-verbose for opencl?)
   .compiler_opt(gpu_compute::CompileOpts::LineInfo)
   .compiler_opt(gpu_compute::CompileOpts::Debug)
   .add_header(header, "world.cl") // add header from &str with name
   .set_source(&kernel) // Set main source from &str
   .add_fn("say_hi", say_hi)
   .add_fn("say_hi2", say_hi2)
   // Adds pipeline, no need to specify each fn of pipe.
   .add_pipeline("pipe", pipeline);

let ptx = device.ptx_info(program); // -> PTXInfo
let build = device.build_source(program); // -> ComputeProgram

let a = Vec::new();

let sh: ComputeProgramFn = build.get_fn("say_hi");
sh.set_grid(10, 20);
loop {
   sh.write_input("a", a)
   unsafe { sh.run(); }
   let z = Vec::new();
   sh.read_ouput("c", z); // need to add input/output types somehow.
}

```

TODO
----

- [*] get_ptx tool
- [+] add /gpu_compute/vendor.h with: VENDOR_XXX, PLATFORM_XXX
- [+] reduce copy/paste in opencl/cuda program builders
- [+] defeat cache mechanic
- [ ] cleanup && add headers
- [ ] check if GridSize more than device max_GridSize
- [ ] launch kernel
- [ ] memory management
- [ ] gc_test tool
- [ ] add pipeline support
- [ ] dynamic loading of backends
- [ ] pinned memory for OpenCL @ NVIDIA
- [ ] vulkan shaders
- [ ] ??? metal shaders ???
- [ ] ??? opengl shaders ???

Versions
--------

OpenCL 1.2 -> __OPENCL_VERSION__ == 120
CUDA 3.5   -> __CUDA_ARCH__ == 350
DEVICE_VER: major*100+minor*10