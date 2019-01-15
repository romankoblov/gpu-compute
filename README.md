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

Pipeline
--------

Allows to create pipeline from multiple kernels.
```rust
Pipeline.run(input_data) -> output_data
let some_function = Program.get_fn(“some_k”)
    .parallel(2) // runs two threads per one input
    .add_input(“a”)

Pipeline::new()
   .add_fn(some_function)
   .add_fn(another_function)
```

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

Currently I don't see reasons to have separated Context and Queue structs.

TODO
----

- [*] get_ptx tool
- [+] add /gpu_compute/vendor.h with: VENDOR_XXX, PLATFORM_XXX
- [ ] reduce copy/paster in opencl/cuda program builders
- [+] defeat cache mechanic
- [ ] cleanup && add headers
- [ ] launch kernel
- [ ] memory management
- [ ] gc_test tool
- [ ] add pipeline support
- [ ] dynamic loading of backends
- [ ] vulkan shaders
- [ ] ??? metal shaders ???
- [ ] ??? opengl shaders ???