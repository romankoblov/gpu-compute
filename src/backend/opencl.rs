use std::env;
use std::ffi::CString;
use std::error::Error;
use std::collections::HashMap;

use ocl::{self, ProQue, Platform, Device, Program, Event, EventList};
use ocl::core::{self, DeviceInfo, DeviceInfoResult, PlatformInfo, ContextInfo,
    CommandQueueInfo, MemInfo, ImageInfo, SamplerInfo, ProgramInfo,
    ProgramBuildInfo, KernelInfo, KernelArgInfo, KernelWorkGroupInfo,
    EventInfo, ProfilingInfo, Status};

use super::generic::{ComputePlatform, ComputeDevice, ComputeQueue, 
    ComputeProgramBuilder, ComputeProgram, PtxFunctionInfo};
use crate::util;
use crate::error::{ComputeResult, ComputeError};
use crate::enums::{Vendor};

// Convert the info or error to a string for printing:
macro_rules! ocl_to_string {
    ( $ expr : expr ) => {
        match $expr {
            Ok(info) => info.to_string(),
            Err(err) => {
                match err.api_status() {
                    Some(Status::CL_KERNEL_ARG_INFO_NOT_AVAILABLE) => "Not available".into(),
                    _ => err.to_string(),
                }
            },
        }
    };
}

const INFO_FORMAT_MULTILINE: bool = true;
pub fn ocl_debug(pro_que: &ProQue, device: &Device){
    // sleep(10)
    let program = pro_que.program();
    let (_begin, delim, end) = if INFO_FORMAT_MULTILINE {
        ("\n", "\n", "\n")
    } else {
        ("{ ", ", ", " }")
    };
    println!("BuildStatus: {}{d}\
            BuildOptions: {}{d}\
            BuildLog: \n\n{}{d}\n\
            BinaryType: {}{e}\
        ",
        ocl_to_string!(core::get_program_build_info(&program, device, ProgramBuildInfo::BuildStatus)),
        ocl_to_string!(core::get_program_build_info(&program, device, ProgramBuildInfo::BuildOptions)),
        ocl_to_string!(core::get_program_build_info(&program, device, ProgramBuildInfo::BuildLog)),
        ocl_to_string!(core::get_program_build_info(&program, device, ProgramBuildInfo::BinaryType)),
        d = delim, e = end,
    );
}

pub struct OclPlatform {}

impl ComputePlatform for OclPlatform {
    fn new() -> OclPlatform { OclPlatform {} }
    fn init(&self) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    fn name(&self) -> &str { "OpenCL" }
    fn short_name(&self) -> &str { "cl" }

    fn list_devices<'a>(&'a self) -> Vec<Box<dyn ComputeDevice + 'a>> {
        let mut out: Vec<Box<(dyn ComputeDevice)>> = vec![];
        for platform in Platform::list() {
            let devices = Device::list_all(platform).unwrap();
            for dev in devices {
                out.push(Box::new(OclDevice::new(&self, platform.clone(), dev)));
            }
        }
        out
    }
}

pub struct OclDevice<'a> {
    pub platform: &'a OclPlatform,
    ocl_platform: ocl::Platform,
    ocl_device: ocl::Device,
}

impl<'a> OclDevice<'a> {
    fn new(platform: &'a OclPlatform, ocl_platform: ocl::Platform, ocl_device: ocl::Device) -> OclDevice<'a> {
        OclDevice { platform, ocl_platform, ocl_device }
    }
}

impl<'a> ComputeDevice for OclDevice<'a> {
    fn name(&self) -> String { format!("{}", self.ocl_device.name().unwrap()) }
    fn platform(&self) -> &dyn ComputePlatform { self.platform }
    fn priority(&self) -> usize {
        let mut priority = 1;
        match self.ocl_device.info(DeviceInfo::Type).unwrap() {
            DeviceInfoResult::Type(ocl::DeviceType::CPU) => priority = priority*10,
            DeviceInfoResult::Type(ocl::DeviceType::GPU) => priority = priority*100,
            _ => {},
        }
        let name = self.name();
        if name.contains("NVIDIA") {
            priority += 3;
        }
        if name.contains("AMD") {
            priority += 2;
        }
        priority
    }

    fn platform_vendor(&self) -> Vendor {
        Vendor::parse(self.ocl_platform.vendor().unwrap().as_ref())
    }


    fn vendor(&self) -> Vendor {
        Vendor::parse(self.ocl_platform.vendor().unwrap().as_ref())
    }

    fn details(&self) -> String {
        // let global_mem = match self.device.info(ocl::enums::DeviceInfo::GlobalMemSize).unwrap() {
        //     ocl::enums::DeviceInfoResult::GlobalMemSize(val) => val,
        //     _ => 0,
        // };
        // let global_mem_alloc = match self.device.info(ocl::enums::DeviceInfo::MaxMemAllocSize).unwrap() {
        //     ocl::enums::DeviceInfoResult::MaxMemAllocSize(val) => val,
        //     _ => 0,
        // };
        // format!("CU: {} MEM: {} ALLOC: {}", 
        //     self.device.info(ocl::enums::DeviceInfo::MaxComputeUnits).unwrap(),
        //     junkyard::scale_bytes(global_mem),
        //     junkyard::scale_bytes(global_mem_alloc),
        // )
        //  Preferred work group size multiple              32
        //  Max work group size                             1024
        format!("Not implemented")
    }
    fn queue<'b>(&'b self) -> Box<dyn ComputeQueue + 'b> {
        Box::new(OclQueue::new(&self))
    }
}

pub struct OclQueue<'a> {
    pub device: &'a OclDevice<'a>,
    ocl_ctx: ocl::Context,
    ocl_queue: ocl::Queue,
}

impl <'a> OclQueue<'a> {
    fn new(device: &'a OclDevice) -> OclQueue<'a> {
        let ocl_ctx = ocl::Context::builder()
            .platform(device.ocl_platform)
            .devices(device.ocl_device)
            .build()
            .unwrap();
        let ocl_queue = ocl::Queue::new(&ocl_ctx, device.ocl_device, None).unwrap();
        OclQueue { device, ocl_ctx, ocl_queue }
    }
}

impl<'a> ComputeQueue for OclQueue<'a> {
    fn device(&self) -> &dyn ComputeDevice { self.device }
    fn program_builder<'b>(&'b self) -> Box<dyn ComputeProgramBuilder + 'b> {
        Box::new(OclProgramBuilder::new(&self))
    }
    fn flush(&self) {
        let _t: Event = self.ocl_queue.enqueue_marker::<EventList>(None).unwrap();
        self.ocl_queue.flush().unwrap();
        self.ocl_queue.finish().unwrap();
    }
}

pub struct OclProgramBuilder<'a> {
    pub queue: &'a OclQueue<'a>,
    debug: bool,
    headers: Vec<String>,
    headers_names: Vec<String>,
    compiler_options: Vec<String>,
    functions: Vec<String>,
    source: Option<String>,
}

impl <'a> OclProgramBuilder<'a> {
    fn new(queue: &'a OclQueue) -> OclProgramBuilder<'a> {
        let headers = Vec::new();
        let headers_names = Vec::new();
        let compiler_options = Vec::new();
        let functions = Vec::new();
        OclProgramBuilder { queue, debug: false, headers, headers_names, compiler_options, functions, source: None }
    }

    fn get_compiler_opt(&self) -> String {
        self.compiler_options.join(" ")
    }

    fn get_build_log(&self, program: &core::Program) -> String {
        let device = self.queue.device.ocl_device;
        match core::get_program_build_info(program, *device.as_core(), core::ProgramBuildInfo::BuildLog) {
            Ok(core::ProgramBuildInfoResult::BuildLog(log)) => { return log; },
            e => panic!("Cannot get build log (opencl): {:?}!", e),
        }

    }

    fn build_wrap<T>(&self, program: &core::Program, build_type: &str, res: core::Result<T>) -> ComputeResult<T> {
        match res {
            Ok(r) => { return Ok(r); },
            Err(e) => {
                println!("{}", self.build_fail(&self.get_build_log(program), Some(build_type), None));
                return Err(ComputeError::KernelBuilding);
            }
        }
    }

    fn build_legacy(&self)  -> Result<(), Box<Error>> {
        let src = match &self.source {
            Some(s) => s,
            None => panic!("Trying to build program without source."),
        };
        let device = &self.queue.device.ocl_device;
        let ctx = &self.queue.ocl_ctx;

        let mut strings = vec![];
        for i in 0..self.headers.len() {
            strings.push(util::legacy_process(&self.headers[i], &self.headers_names[i]));
        }
        strings.push(util::legacy_process(src, "KERNEL"));
        let cstrings = strings.iter().map(|h| CString::new(h.as_str())).collect::<Result<Vec<_>, _>>()?;

        let program = core::create_build_program(ctx, &cstrings[..], Some(&[device]), &CString::new(self.get_compiler_opt()).unwrap()).unwrap();
        println!("LEGACY: {:?}", self.get_build_log(&program));

        Ok(())
    }

    // Supported by OpenCL 1.2 only, fails on MacOS
    fn build_compile(&self) -> Result<Box<dyn ComputeProgram + 'a>, Box<Error>> {
        let src = match &self.source {
            Some(s) => s.as_str(),
            None => panic!("Trying to build program without source."),
        };
        let device = &self.queue.device.ocl_device;
        let ctx = &self.queue.ocl_ctx;

        // Compile headers
        let headers_cstr = self.headers.iter().map(|h| CString::new(h.as_str()) ).collect::<Result<Vec<_>, _>>()?;
        // uses some strange crate 'failure' for error reporting, which doesn't implement std::Error
        let headers_programs = headers_cstr.into_iter().map(|p| core::create_program_with_source(ctx, &[p]) ).collect::<Result<Vec<_>, _>>().unwrap();
        let h_programs_ptr: Vec<&core::Program> = headers_programs.iter().map(|p| p).collect();
        let headers_names = self.headers_names.iter().map(|h| CString::new(h.as_str()) ).collect::<Result<Vec<_>, _>>()?;
        let c_src = CString::new(src)?;
        let program = core::create_program_with_source(ctx, &[c_src]).unwrap();

        self.build_wrap::<()>(&program, "Compile", core::compile_program(&program,
            Some(&[device.as_core()]), 
            &CString::new(self.get_compiler_opt()).unwrap(), 
            &h_programs_ptr[..], 
            &headers_names, 
            None, None, None)).unwrap();
        // TODO: expose link options?                         // -cl-nv-verbose -- doesn't work
        let linked = core::link_program(ctx, Some(&[device]), &CString::new("").unwrap(), &[&program], 
            None, None, None).unwrap();
        println!("PROGRAM: {:?}", self.get_build_log(&program));
        Ok(Box::new(OclProgram::new(self.queue, ocl::Program::from(linked))))
    }
}

impl<'a> ComputeProgramBuilder<'a> for OclProgramBuilder<'a> {
    fn queue(&self) -> &dyn ComputeQueue { self.queue }
    fn compiler_opt(&mut self, opt: &str) -> &mut (dyn ComputeProgramBuilder<'a> + 'a) {
        self.compiler_options.push(opt.to_string());
        self
    }
    fn debug(&mut self) -> &mut (dyn ComputeProgramBuilder<'a> + 'a) {
        self.debug = true;
        if self.queue.device.vendor() == Vendor::NVIDIA {
            self.compiler_opt("-cl-nv-verbose");
        }
        self
    }
    fn add_header(&mut self, src: &str, name: &str) -> &mut (dyn ComputeProgramBuilder<'a> + 'a) {
        self.headers.push(String::from(src));
        self.headers_names.push(String::from(name));
        self
    }
    fn add_stub_header(&mut self) -> &mut (dyn ComputeProgramBuilder<'a> + 'a) {
        self.add_header(self.vendor_header().as_str(), "/gpu_compute/vendor.h");
        self.add_header(self.cache_header(self.debug).as_str(), "/gpu_compute/cache.h");
        self.add_header(&include_str!("../../headers/opencl.cl"), "/gpu_compute/platform.h");
        self
    }
    fn add_fn(&mut self, name: &str) -> &mut (dyn ComputeProgramBuilder<'a> + 'a) {
        self.functions.push(name.to_string());
        self
    }
    fn set_source(&mut self, src: &str) -> &mut (dyn ComputeProgramBuilder<'a> + 'a) {
        if self.source.is_some() { panic!("Source already set for kernel."); }
        self.source = Some(src.to_string());
        self
    }
    fn build_source(&self) -> Result<Box<dyn ComputeProgram + 'a>, Box<Error>> {
        // Works correctly on macos for non-empty kernels
        // TODO: check for opencl version per device for legacy stuff
        // self.build_legacy(src);
        self.build_compile()
    }
}

pub struct OclProgram<'a> {
    pub queue: &'a OclQueue<'a>,
    ocl_program: ocl::Program,
}

impl <'a> OclProgram<'a> {
    fn new(queue: &'a OclQueue, ocl_program: ocl::Program) -> OclProgram<'a> {
        OclProgram { queue, ocl_program }
    }
}

impl<'a> ComputeProgram for OclProgram<'a> {
    fn get_ptx(&self) -> Option<String> {
        None
    }
    fn get_ptx_info(&self) -> Option<HashMap<String, PtxFunctionInfo>> {
        None
    }
}