use std::error::Error;
use std::ffi::CString;
use std::collections::HashMap;

use rustacuda::prelude::*;
use rustacuda::memory::DeviceBox;
use rustacuda::device::{Device, DeviceAttribute};
use rustacuda::function::{Function, FunctionAttribute};
use nvrtc;

use super::generic::{ComputePlatform, ComputeDevice, ComputeQueue, ComputeProgram};
use crate::error::{ComputeError, ComputeResult};
use crate::enums::{Vendor, CompilerOpt, PtxInfo, PtxFunctionInfo};
use crate::program::SourceProgramBuilder;

pub struct CudaPlatform {}

impl ComputePlatform for CudaPlatform {
    fn new() -> CudaPlatform { CudaPlatform {} }
    fn init(&self) -> Result<(), Box<dyn Error>> {
        rustacuda::init(CudaFlags::empty())?;
        Ok(())
    }

    fn name(&self) -> &str { "CUDA" }
    fn short_name(&self) -> &str { "cu" }
    fn stub_header(&self) -> &'static str { include_str!("../../headers/cuda.cu") }

    fn list_devices<'a>(&'a self) -> Vec<Box<dyn ComputeDevice + 'a>> {
        let mut out: Vec<Box<(dyn ComputeDevice)>> = vec![];
        for device in rustacuda::device::Device::devices().unwrap() {
            out.push(Box::new(CudaDevice::new(&self, device.unwrap())));
        }
        out
    }
}

pub struct CudaDevice<'a> {
    pub platform: &'a CudaPlatform,
    pub default_queue: CudaQueue<'a>,

    cuda_device: Device,
}

impl<'a> CudaDevice<'a> {
    fn new(platform: &'a CudaPlatform, cuda_device: Device) -> CudaDevice<'a> {

        // println!("================");
        // println!("MultiprocessorCount: {}", cuda_device.get_attribute(DeviceAttribute::MultiprocessorCount).unwrap());
        // println!("MaxThreadsPerMultiprocessor: {}", cuda_device.get_attribute(DeviceAttribute::MaxThreadsPerMultiprocessor).unwrap());
        // println!("MaxRegistersPerMultiprocessor: {}", cuda_device.get_attribute(DeviceAttribute::MaxRegistersPerMultiprocessor).unwrap());
        // println!("MaxSharedMemoryPerMultiprocessor: {}", cuda_device.get_attribute(DeviceAttribute::MaxSharedMemoryPerMultiprocessor).unwrap());
        // println!("================");
        // println!("MaxThreadsPerBlock: {}", cuda_device.get_attribute(DeviceAttribute::MaxThreadsPerBlock).unwrap());
        // println!("MaxRegistersPerBlock: {}", cuda_device.get_attribute(DeviceAttribute::MaxRegistersPerBlock).unwrap());
        // println!("MaxThreadsPerBlock: {}", cuda_device.get_attribute(DeviceAttribute::MaxThreadsPerBlock).unwrap());
        // println!("MaxSharedMemoryPerBlock: {}", cuda_device.get_attribute(DeviceAttribute::MaxSharedMemoryPerBlock).unwrap());
        // println!("TotalConstantMemory: {}", cuda_device.get_attribute(DeviceAttribute::TotalConstantMemory).unwrap());
        // println!("================");
        // println!("WarpSize: {}", cuda_device.get_attribute(DeviceAttribute::WarpSize).unwrap());
        // println!("MaxBlockDimX: {}", cuda_device.get_attribute(DeviceAttribute::MaxBlockDimX).unwrap());
        // println!("MaxBlockDimY: {}", cuda_device.get_attribute(DeviceAttribute::MaxBlockDimY).unwrap());
        // println!("MaxBlockDimZ: {}", cuda_device.get_attribute(DeviceAttribute::MaxBlockDimZ).unwrap());
        // println!("MaxGridDimX: {}", cuda_device.get_attribute(DeviceAttribute::MaxGridDimX).unwrap());
        // println!("MaxGridDimY: {}", cuda_device.get_attribute(DeviceAttribute::MaxGridDimY).unwrap());
        // println!("MaxGridDimZ: {}", cuda_device.get_attribute(DeviceAttribute::MaxGridDimZ).unwrap());
        // println!("================");
        let queue = CudaQueue::new_default(cuda_device);
        let device = CudaDevice { platform, default_queue: queue, cuda_device };
        device
    }
    fn get_arch(&self) -> String {
        let (major, minor) = self.version_tuple();
        format!("compute_{}{}", major, minor)
    }
}

impl<'a> ComputeDevice for CudaDevice<'a> {
    fn name(&self) -> String { format!("{}", self.cuda_device.name().unwrap()) }
    fn vendor(&self) -> Vendor { Vendor::NVIDIA }
    fn platform_vendor(&self) -> Vendor { Vendor::NVIDIA }
    fn platform(&self) -> &dyn ComputePlatform { self.platform }
    fn version_tuple(&self) -> (usize, usize) {
        (self.cuda_device.get_attribute(DeviceAttribute::ComputeCapabilityMajor).unwrap() as usize, 
        self.cuda_device.get_attribute(DeviceAttribute::ComputeCapabilityMinor).unwrap() as usize)
    }
    fn details(&self) -> String { format!("Not implemented") }
    fn default_queue<'b>(&'b self) -> &dyn ComputeQueue {
        &self.default_queue
    }
    fn queue<'b>(&'b self) -> Box<dyn ComputeQueue + 'b> {
        Box::new(CudaQueue::new(&self))
    }
    fn compiler_opts(&self, opts: &[CompilerOpt]) -> Vec<String> {
        let mut opt_vec  = Vec::with_capacity(opts.len());
        for opt in opts {
            match opt {
                CompilerOpt::LineInfo => opt_vec.push("-lineinfo".to_string()),
                CompilerOpt::Debug => opt_vec.push("-debug".to_string()),
                _ => {},
            }
        }
        opt_vec
    }
}

struct CudaModule {
    module: Module,
    functions: HashMap<String, String>,
}

impl CudaModule {
    fn new(ptx: &str, map: HashMap<String, String>) -> CudaModule {
        let module = Module::load_from_string(&CString::new(ptx).unwrap()).unwrap();
        CudaModule { module, functions: map.clone() }
    }
    pub fn get_fn(&self, name: &str) -> Function {
        let f = self.module.get_function(&CString::new(name).unwrap());
        let fn_res = match f {
            Ok(f) => Ok(f),
            Err(e) => {
                // Try to get mangled function
                if let Some(mangled) = &self.functions.get(name) {
                    let c_name = CString::new(mangled.as_str()).unwrap();
                    self.module.get_function(&c_name)
                } else {
                    Err(e)
                }
            },
        };
        fn_res.unwrap()
    }
    fn get_ptx_info(&self, name: &str) -> PtxFunctionInfo {
        let f = self.get_fn(name);
        let registers = Some(f.get_attribute(FunctionAttribute::NumRegisters).unwrap() as usize);
        let mem_local = Some(f.get_attribute(FunctionAttribute::LocalSizeBytes).unwrap() as usize);
        let mem_shared = Some(f.get_attribute(FunctionAttribute::SharedMemorySizeBytes).unwrap() as usize);
        let mem_const = Some(f.get_attribute(FunctionAttribute::ConstSizeBytes).unwrap() as usize);
        let max_threads = Some(f.get_attribute(FunctionAttribute::MaxThreadsPerBlock).unwrap() as usize);
        PtxFunctionInfo { registers, mem_local, mem_shared, mem_const, max_threads }
    }
}

pub struct CudaQueue<'a> {
    pub device: Option<&'a CudaDevice<'a>>,
    // ctx should be after stream or it will crash with "Failed to destroy CUDA stream.: InvalidContext"
    cuda_stream: Stream,
    cuda_ctx: Context,
}

impl <'a> CudaQueue<'a> {
    fn new(device: &'a CudaDevice) -> CudaQueue<'a> {
        let cuda_ctx = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device.cuda_device).unwrap();
        let cuda_stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        CudaQueue { device: Some(device), cuda_stream, cuda_ctx }
    }
    fn new_default(device: Device) -> CudaQueue<'a> {
        let cuda_ctx = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device).unwrap();
        let cuda_stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        CudaQueue { device: None, cuda_stream, cuda_ctx }
    }
    fn get_build_log(&self, program: &nvrtc::NvrtcProgram) -> String {
        match program.get_log() {
            Ok(log) => { return log; },
            e => panic!("Cannot get build log (cuda): {:?}!", e),
        }
    }

    fn build_wrap<T>(&self, builder: &SourceProgramBuilder, program: &nvrtc::NvrtcProgram, res: Result<T, nvrtc::error::NvrtcError>) -> ComputeResult<T> {
        match res {
            Ok(r) => { return Ok(r); },
            Err(e) => {
                println!("{}", builder.fmt_build_fail_log(self.device(), &self.get_build_log(program)));
                return Err(ComputeError::KernelBuilding);
            }
        }
    }

    fn build_ptx(&self, builder: &SourceProgramBuilder) -> (HashMap<String, String>, String) {
        let compiler_opts = self.device().compiler_opts(&builder.compiler_options);
        let (headers, names) = builder.generate_headers(self.device());

        let name = match &builder.name {
            Some(n) => Some(n.as_str()),
            None => None,
        };

        let program_ptx = nvrtc::NvrtcProgram::new(
            builder.get_source(),
            name,
            &crate::str_vec!(headers),
            &crate::str_vec!(names),
        ).unwrap();
        
        for fn_name in &builder.get_functions() {
            program_ptx.add_expr(fn_name).unwrap();
        }

        self.build_wrap(builder, &program_ptx, 
            program_ptx.compile(&crate::str_vec!(compiler_opts))
        ).unwrap();
        
        let mut fn_map = HashMap::new();
        for fn_name in &builder.get_functions() {
            let ptx_fn = program_ptx.get_name(fn_name).unwrap();
            fn_map.insert(fn_name.clone(), ptx_fn);
        }
        (fn_map, program_ptx.get_ptx().unwrap())
    }
}

impl<'a> ComputeQueue for CudaQueue<'a> {
    fn device(&self) -> &dyn ComputeDevice { self.device.unwrap() }
    fn flush(&self) {
        self.cuda_stream.synchronize().unwrap();
    }
    fn build_source(&self, builder: &SourceProgramBuilder) {
        
        let (fn_map, ptx) = self.build_ptx(&builder);
        println!("PTX CUDA: {}", ptx);

        // let headers: Vec<_> = self.headers.iter().map(|s| s.as_str()).collect();
        // let headers_names: Vec<_> = self.headers_names.iter().map(|s| s.as_str()).collect();
        // let headers_names: Vec<_> = self.headers_names.iter().map(|s| s.as_str()).collect();
        // let compiler_opts: Vec<_> = self.compiler_options.iter().map(|s| s.as_str()).collect();
        // let program_ptx = nvrtc::NvrtcProgram::new(src, None, &headers, &headers_names)?;
        // for fn_name in &self.functions {
        //     program_ptx.add_expr(fn_name)?;
        // }
        // self.build_wrap(&program_ptx, program_ptx.compile(&compiler_opts))?;
        // let mut fn_map = HashMap::new();
        // for fn_name in &self.functions {
        //     let ptx_fn = program_ptx.get_name(fn_name)?;
        //     fn_map.insert(fn_name.clone(), ptx_fn);
        // }
       // println!("PTX CUDA: {}", program_ptx.get_ptx().unwrap())
        // Ok((program_ptx.get_ptx()?, fn_map))
    }
    fn get_ptx_info(&self, builder: &SourceProgramBuilder) -> Option<PtxInfo> {
        let (fn_map, ptx) = self.build_ptx(&builder);
        let mut info = PtxInfo::new(&ptx);
        let module = CudaModule::new(&ptx, fn_map);
        for fn_name in &builder.get_functions() {
            info.add_fn(fn_name, module.get_ptx_info(fn_name));
        }
        Some(info)
    }
}

pub struct CudaProgram<'a> {
    pub queue: &'a CudaQueue<'a>,
    module: Module,
    functions: Vec<String>,
    ptx_info: Option<(String, HashMap<String, String>)>,
}

impl<'a> CudaProgram<'a> {
    fn new(queue: &'a CudaQueue, module: Module, functions: Vec<String>, ptx_info: Option<(String, HashMap<String, String>)>) -> CudaProgram<'a> {
        CudaProgram { queue, module, functions, ptx_info }
    }
    fn get_fn(&self, name: &str) -> Result<Function, Box<Error>> {
        let f = self.module.get_function(&CString::new(name)?);
        match f {
            Ok(f) => Ok(f),
            Err(e) => {
                // Try to get mangled function
                if let Some(ptx_info) = &self.ptx_info {
                    if let Some(old_name) = ptx_info.1.get(name) {
                        let c_name = CString::new(old_name.as_str())?;
                        return Ok(self.module.get_function(&c_name)?);
                    }
                }
                Err(Box::new(e))
            }
        }
    }
}

impl<'a> ComputeProgram for CudaProgram<'a> {
    fn get_ptx(&self) -> Option<String> {
        match &self.ptx_info {
            Some(ptx_info) => Some(ptx_info.0.clone()),
            None => None,
        }
    }

}

// // Official example
// fn cuda_basic() -> Result<(), Box<dyn Error>> {
//         rustacuda::init(CudaFlags::empty())?;
    
//     // Get the first device
//     let device = Device::get_device(0)?;

//     // Create a context associated to this device
//     let _context = Context::create_and_push(
//         ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

//     // Load the module containing the function we want to call
//     let module_data = CString::new(include_str!("../resources/add.ptx"))?;
//     let module = Module::load_from_string(&module_data)?;

//     // Create a stream to submit work to
//     let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

//     // Allocate space on the device and copy numbers to it.
//     let mut x = DeviceBox::new(&10.0f32)?;
//     let mut y = DeviceBox::new(&20.0f32)?;
//     let mut result = DeviceBox::new(&0.0f32)?;

//     // Launching kernels is unsafe since Rust can't enforce safety - think of kernel launches
//     // as a foreign-function call. In this case, it is - this kernel is written in CUDA C.
//     unsafe {
//         // Launch the `sum` function with one block containing one thread on the given stream.
//         launch!(module.sum<<<1, 1, 0, stream>>>(
//             x.as_device_ptr(),
//             y.as_device_ptr(),
//             result.as_device_ptr(),
//             1 // Length
//         ))?;
//     }

//     // The kernel launch is asynchronous, so we wait for the kernel to finish executing
//     stream.synchronize()?;

//     // Copy the result back to the host
//     let mut result_host = 0.0f32;
//     result.copy_to(&mut result_host)?;
    
//     println!("Sum is {}", result_host);

//     Ok(())
// }
