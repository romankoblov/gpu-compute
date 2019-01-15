use log::*;

pub mod generic;
pub mod util;
pub mod cuda;
pub mod opencl;

use self::generic::{ComputeDevice, ComputePlatform};

pub struct Compute {
    pub backends: Vec<Box<dyn ComputePlatform>>,
}

macro_rules! platform_init {
    ($vec:ident, $modname:path) => {{
        use $modname as base;
        let platform = base::new();
        match platform.init() {
            Ok(_) => $vec.push(Box::new(platform)),
            Err(e) => warn!("Cannot initialize {}: {}", platform.name(), e),
        }
    }};
}

impl Compute {
    pub fn new() -> Compute {
        let mut backends: Vec<Box<dyn ComputePlatform>> = vec![];
        platform_init!(backends, crate::cuda::CudaPlatform);
        platform_init!(backends, crate::opencl::OclPlatform);
        Compute { backends }
    }
    fn default_platform(&self) -> &Box<dyn ComputePlatform> {
        &self.backends[0]
    }
    pub fn get_device<'a>(&'a self, dev_name: Option<&str>) -> Box<dyn ComputeDevice + 'a>
    {
        if let Some(dev_str) = std::env::var("GPU_DEV").ok().or(dev_name.map(|x| String::from(x))) {
            for platform in &self.backends {
                if dev_str.starts_with(platform.short_name())
                {
                    let idx = dev_str[platform.short_name().len()..].parse::<usize>().unwrap();
                    for (i, device) in platform.list_devices().into_iter().enumerate() {
                        if i==idx { 
                            return device
                        }
                    }
                }
            }
            panic!("Wrong GPU_DEV format: {}", dev_str);
        } else {
            let p = self.default_platform();
            if let Some(dev) = p.default_device() {
                return dev;
            }
            panic!("Empty default device for {:?}", p);
        }
    }
    pub fn list_devices<'a>(&'a self) -> Vec<(usize, Box<dyn ComputeDevice + 'a>)> {
        let mut res = vec![];
        for platform in &self.backends {
            for (i, device) in platform.list_devices().into_iter().enumerate()
            {
                res.push((i, device));
            }
        }
        res
    }
    pub fn print_devices(&self) {
        for platform in &self.backends {
            println!("=== {}: ===", platform.name());
            for (i, device) in platform.list_devices().iter().enumerate()
            {
                println!(" - {}{}: {}", platform.short_name(), i, device.name());   
            }
        }
    }
}

pub fn init(){
    let cp = Compute::new();
    cp.print_devices();
    let dev = cp.get_device(None);
    let queue = dev.queue();
    println!("Selected: {:?}", queue);

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
        }
        WITHIN_KERNEL void hi(unsigned int gid)
        { 
            printf("I am %d.\n", gid); 
        }
    "#;
    let kernel = r#"
        #include "/gpu_compute.h"
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
    let x = queue.program_builder()
        .debug()
        .add_stub_header()
        .add_header(header, "world.cl")
        .add_fn("say_hi")
        .add_fn("say_hi2")
        .add_fn("say_hi3")
        .set_source(&kernel)
        .build_source().unwrap();
    //println!("Q: {}", x);
}


#[cfg(test)]
mod tests {
    use super::*;
    use ocl::core;
    use std::ffi::CString;

    fn test_ocl()
    {
        let header = r#"
            __kernel void wtf() {
                printf("wtf ");
            }

            void world(){
                printf("world");
            }
        "#;
        let kernel = r#"
            #include "world.cl"
            __kernel void hello() {
                printf("hello ");
                world();
            }
        "#;

        let kernel2 = r#"
            __kernel void hello2() {
                printf("hello ");
            }
            __kernel void omg() {
                printf("omg ");
            }

        "#;

        let platform_id = core::default_platform().unwrap();
        let device_ids = core::get_device_ids(&platform_id, None, None).unwrap();
        let device = device_ids[0];
        let context_properties = core::ContextProperties::new().platform(platform_id);
        let context = core::create_context(Some(&context_properties),
            &[device], None, None).unwrap();
        let program = core::create_program_with_source(&context, &[CString::new(kernel).unwrap()]).unwrap();
        let program2 = core::create_program_with_source(&context, &[CString::new(kernel2).unwrap()]).unwrap();

        let header = core::create_program_with_source(&context, &[CString::new(header).unwrap()]).unwrap();

        //core::build_program(&program2, Some(&[device]), &CString::new("").unwrap(), None, None).unwrap();

        core::compile_program(&program, Some(&[device]), &CString::new("").unwrap(), &[&header], 
            &[CString::new("world.cl").unwrap()], None, None, None).unwrap();
        core::compile_program(&program2, Some(&[device]), &CString::new("").unwrap(), &[&header], 
            &[CString::new("world.cl").unwrap()], None, None, None).unwrap();



        match core::get_device_info(&device, core::DeviceInfo::Name).unwrap() {
            core::DeviceInfoResult::Name(x) => println!("W: {}", x),
            _ => {},
        }

        match core::get_program_build_info(&program, &device, core::ProgramBuildInfo::BinaryType).unwrap()
        {
            core::ProgramBuildInfoResult::BinaryType(s) => println!("TYPE: {:?}", s),
            _ => {},
        }
        // -create-library
        let linked = core::link_program(&context, Some(&[device]), &CString::new("").unwrap(), &[&program, &program2], 
            None, None, None).unwrap();

        match core::get_program_build_info(&linked, &device, core::ProgramBuildInfo::BinaryType).unwrap()
        {
            core::ProgramBuildInfoResult::BinaryType(s) => println!("TYPE: {:?}", s),
            _ => {},
        }

        match core::get_program_info(&linked, core::ProgramInfo::NumKernels).unwrap() {
            core::ProgramInfoResult::NumKernels(bin) => {
                println!("lol: {}", bin);
            },
            _ => {},
        };

        match core::get_program_info(&linked, core::ProgramInfo::KernelNames).unwrap() {
            core::ProgramInfoResult::KernelNames(bin) => {
                println!("lol2: {}", bin);
            },
            _ => {},
        };


        match core::get_program_info(&linked, core::ProgramInfo::Binaries).unwrap() {
            core::ProgramInfoResult::Binaries(bin) => {
                println!("LEN: {}", bin.len());
                println!("dct: {:?}", &bin[0]);
                println!("lol: {}", String::from_utf8_lossy(&bin[0]));
            },
            _ => {},
        };
    }
    #[test]
    fn it_works() {
 //       test_ocl();

        init();
        //cuda_basic().expect("lol");
        assert_eq!(2 + 2, 4);
    }
}
