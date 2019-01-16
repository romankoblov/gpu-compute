use std::env;
use std::ffi::CString;
use std::error::Error;
use std::collections::HashMap;
use std::os::raw;

use ocl::{self, ProQue, Platform, Device, Program, Event, EventList};
use ocl::core::{self, DeviceInfo, DeviceInfoResult, PlatformInfo, ContextInfo,
    CommandQueueInfo, MemInfo, ImageInfo, SamplerInfo, ProgramInfo,
    ProgramBuildInfo, KernelInfo, KernelArgInfo, KernelWorkGroupInfo,
    EventInfo, ProfilingInfo, Status};

use super::generic::{ComputePlatform, ComputeDevice, ComputeQueue, ComputeProgram};
use crate::util;
use crate::error::{ComputeResult, ComputeError};
use crate::enums::{Vendor, CompilerOpt, PtxFunctionInfo, PtxInfo};
use crate::program::SourceProgramBuilder;

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
    fn stub_header(&self) -> &'static str { include_str!("../../headers/opencl.cl") }
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
    pub default_queue: OclQueue<'a>,
    ocl_platform: ocl::Platform,
    ocl_device: ocl::Device,
}

impl<'a> OclDevice<'a> {
    fn new(platform: &'a OclPlatform, ocl_platform: ocl::Platform, ocl_device: ocl::Device) -> OclDevice<'a> {
        let queue = OclQueue::new_default(ocl_platform, ocl_device);
        let device = OclDevice { platform, default_queue: queue, ocl_platform, ocl_device };
        device
    }
    fn get_wg_size(&self, kernel: &core::Kernel) -> usize {
       match core::get_kernel_work_group_info(&kernel, self.ocl_device, core::KernelWorkGroupInfo::WorkGroupSize).unwrap() {
            core::KernelWorkGroupInfoResult::WorkGroupSize(wg) => wg,
            other => panic!("OpenCL get_wg_size do incorrect output: {:?}", other),
        }
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
    
    fn version_tuple(&self) -> (usize, usize) {
        let ocl_version = match core::get_device_info(self.ocl_device, core::DeviceInfo::Version).unwrap()
        {
            core::DeviceInfoResult::Version(ver) => ver.to_raw(),
            other => panic!("OpenCL version_tuple incorrect result: {}", other),
        };
        (ocl_version.0 as usize, ocl_version.1 as usize)
    }

    fn vendor(&self) -> Vendor {
        Vendor::parse(self.ocl_platform.vendor().unwrap().as_ref())
    }

    fn details(&self) -> String {
        //  Preferred work group size multiple              32
        //  Max work group size                             1024
        format!("Not implemented")
    }
    fn default_queue<'b>(&'b self) -> &dyn ComputeQueue {
        &self.default_queue
    }
    fn queue<'b>(&'b self) -> Box<dyn ComputeQueue + 'b> {
        Box::new(OclQueue::new(&self))
    }
    fn compiler_opts(&self, opts: &[CompilerOpt]) -> Vec<String> {
        let mut opt_vec  = Vec::with_capacity(opts.len());
        for opt in opts {
            match opt {
                CompilerOpt::NV_PTAX => opt_vec.push("-cl-nv-verbose".to_string()),
//                CompilerOpt::AMD_TEMPS => opt_vec.push("-fbin-amdil".to_string()),
                _ => {},
            }
        }
        opt_vec
    }
}

pub struct OclQueue<'a> {
    pub device: Option<&'a OclDevice<'a>>,
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
        OclQueue { device: Some(device), ocl_ctx, ocl_queue }
    }
    fn new_default(ocl_platform: ocl::Platform, ocl_device: ocl::Device) -> OclQueue<'a> {
        let ocl_ctx = ocl::Context::builder()
            .platform(ocl_platform)
            .devices(ocl_device)
            .build()
            .unwrap();
        let ocl_queue = ocl::Queue::new(&ocl_ctx, ocl_device, None).unwrap();
        OclQueue { device: None, ocl_ctx, ocl_queue }
    }
    fn get_build_log(&self, program: &core::Program) -> String {
        let device = self.device.unwrap().ocl_device;
        match core::get_program_build_info(program, *device.as_core(), core::ProgramBuildInfo::BuildLog) {
            Ok(core::ProgramBuildInfoResult::BuildLog(log)) => { return log; },
            e => panic!("Cannot get build log (opencl): {:?}!", e),
        }
    }
    fn build_wrap<T>(&self, builder: &SourceProgramBuilder, program: &core::Program, res: core::Result<T>) -> ComputeResult<T> {
        match res {
            Ok(r) => { return Ok(r); },
            Err(e) => {
                println!("{}", builder.fmt_build_fail_log(self.device(), &self.get_build_log(program)  ));
                return Err(ComputeError::KernelBuilding);
            }
        }
    }
    fn build_legacy(&self, builder: &SourceProgramBuilder)  -> Result<core::Program, Box<Error>> {
        let device = &self.device.unwrap().ocl_device;
        let ctx = &self.ocl_ctx;
        
        let compiler_opts = CString::new(self.device().compiler_opts(&builder.compiler_options).join(" ")).unwrap();
        let source = builder.legacy_source(self.device());

        let program = core::create_build_program(
            ctx, 
            &[CString::new(source).unwrap()], 
            Some(&[device]), 
            &CString::new(compiler_opts).unwrap()
        ).unwrap();
        Ok(program)
    }
    fn build_compile(&self, builder: &SourceProgramBuilder) -> Result<core::Program, Box<Error>> {
        let device = &self.device.unwrap().ocl_device;
        let ctx = &self.ocl_ctx;

        let compiler_opts = CString::new(self.device().compiler_opts(&builder.compiler_options).join(" ")).unwrap();
        let source = builder.get_source();
        let (headers, names) = builder.generate_headers(self.device());

        let headers_programs = crate::cstring_vec!(headers).into_iter().map(|p| core::create_program_with_source(ctx, &[p]) ).collect::<Result<Vec<_>, _>>().unwrap();
        let headers_names = crate::cstring_vec!(names);

        let program = core::create_program_with_source(ctx, &[CString::new(source).unwrap()]).unwrap();

        self.build_wrap::<()>(&builder, &program, core::compile_program(&program,
            Some(&[device.as_core()]), 
            &compiler_opts, 
            &headers_programs.iter().map(|p| p).collect::<Vec<&core::Program>>(), 
            &headers_names, 
            None, None, None)).unwrap();

        let linked = core::link_program(ctx, Some(&[device]), &CString::new("").unwrap(), &[&program], 
            None, None, None).unwrap();

        Ok(linked)
    }
    fn get_ptx(&self, program: &core::Program) -> String {
        let bin = match core::get_program_info(&program, core::ProgramInfo::Binaries).unwrap() {
            core::ProgramInfoResult::Binaries(bin) => bin,
            other => panic!("OpenCL get_program_info unexpected result: {:?}", other),
        };
        if bin.len() != 1 { panic!("OpenCL get_ptx vector length more than 1: {} {:?}", bin.len(), bin); }
        String::from_utf8(bin[0].clone()).unwrap()
    }
}

impl<'a> ComputeQueue for OclQueue<'a> {
    fn device(&self) -> &dyn ComputeDevice { self.device.unwrap() }
    fn flush(&self) {
        let _t: Event = self.ocl_queue.enqueue_marker::<EventList>(None).unwrap();
        self.ocl_queue.flush().unwrap();
        self.ocl_queue.finish().unwrap();
    }
    fn build_source(&self, builder: &SourceProgramBuilder) {
        let program = if self.device().version() >= 120 {
            self.build_legacy(builder).unwrap()
        } else {
            self.build_compile(builder).unwrap()
        };
    }
    fn get_ptx_info(&self, builder: &SourceProgramBuilder) -> Option<PtxInfo> {
        // if not NVIDIA: return None
        if self.device().vendor() != Vendor::NVIDIA { return None; }
        let legacy = self.build_legacy(builder).unwrap();
        let ptx = self.get_ptx(&legacy);
        let mut info = PtxInfo::new(&ptx);
        let mut fn_info = PtxFunctionInfo::parse_ptxas(&self.get_build_log(&legacy));

        for fn_name in &builder.get_functions() {
            let kernel = core::create_kernel(&legacy, fn_name).unwrap();
            let wg_size = self.device.unwrap().get_wg_size(&kernel);
            let fn_info: &mut PtxFunctionInfo = fn_info.get_mut(fn_name).unwrap();
            fn_info.max_threads = Some(wg_size);
            info.add_fn(fn_name, fn_info.clone());
        }
        if self.device().version() >= 120 {
            let compiler = self.build_compile(builder).unwrap();
            info.add_ptx("compiler", &self.get_ptx(&compiler));
        }

        Some(info)
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

}