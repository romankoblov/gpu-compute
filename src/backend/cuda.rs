use std::error::Error;
use std::ffi::CString;
use std::collections::HashMap;

use rustacuda::prelude::*;
use rustacuda::memory::DeviceBox;
use rustacuda::device::{Device, DeviceAttribute};
use rustacuda::function::{Function, FunctionAttribute};
use nvrtc;

use super::generic::{ComputePlatform, ComputeDevice, ComputeQueue, 
    ComputeProgramBuilder, ComputeProgram, PtxFunctionInfo};
use crate::error::{ComputeError, ComputeResult};
use crate::enums::{Vendor};

pub struct CudaPlatform {}

impl ComputePlatform for CudaPlatform {
    fn new() -> CudaPlatform { CudaPlatform {} }
    fn init(&self) -> Result<(), Box<dyn Error>> {
        rustacuda::init(CudaFlags::empty())?;
        Ok(())
    }

    fn name(&self) -> &str { "CUDA" }
    fn short_name(&self) -> &str { "cu" }

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

        CudaDevice { platform, cuda_device }
    }
    fn get_arch(&self) -> String {
        format!("compute_{}{}", 
            self.cuda_device.get_attribute(DeviceAttribute::ComputeCapabilityMajor).unwrap(),
            self.cuda_device.get_attribute(DeviceAttribute::ComputeCapabilityMinor).unwrap()
        )
    }
}

impl<'a> ComputeDevice for CudaDevice<'a> {
    fn name(&self) -> String { format!("{}", self.cuda_device.name().unwrap()) }
    fn vendor(&self) -> Vendor { Vendor::NVIDIA }
    fn platform_vendor(&self) -> Vendor { Vendor::NVIDIA }
    fn platform(&self) -> &dyn ComputePlatform { self.platform }
    fn details(&self) -> String { format!("Not implemented") }
    fn queue<'b>(&'b self) -> Box<dyn ComputeQueue + 'b> {
        Box::new(CudaQueue::new(&self))
    }
}


pub struct CudaQueue<'a> {
    pub device: &'a CudaDevice<'a>,
    // ctx should be after stream or it will crash with "Failed to destroy CUDA stream.: InvalidContext"
    cuda_stream: Stream,
    cuda_ctx: Context,
}

impl <'a> CudaQueue<'a> {
    fn new(device: &'a CudaDevice) -> CudaQueue<'a> {
        let cuda_ctx = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device.cuda_device).unwrap();
        let cuda_stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        CudaQueue { device, cuda_stream, cuda_ctx }
    }
}

impl<'a> ComputeQueue for CudaQueue<'a> {
    fn device(&self) -> &dyn ComputeDevice { self.device }
    fn program_builder<'b>(&'b self) -> Box<dyn ComputeProgramBuilder + 'b> {
        Box::new(CudaProgramBuilder::new(&self))
    }
    fn flush(&self) {
        self.cuda_stream.synchronize().unwrap();
    }
}

pub struct CudaProgramBuilder<'a> {
    pub queue: &'a CudaQueue<'a>,
    debug: bool,
    headers: Vec<String>,
    headers_names: Vec<String>,
    compiler_options: Vec<String>,
    functions: Vec<String>,
    source: Option<String>,
}

impl <'a> CudaProgramBuilder<'a> {
    fn new(queue: &'a CudaQueue) -> CudaProgramBuilder<'a> {
        let headers = Vec::new();
        let headers_names = Vec::new();
        let compiler_options = Vec::new();
        let functions = Vec::new();
        CudaProgramBuilder { queue, debug: false, headers, headers_names, compiler_options, functions, source: None }
    }

    fn get_build_log(&self, program: &nvrtc::NvrtcProgram) -> String {
        match program.get_log() {
            Ok(log) => { return log; },
            e => panic!("Cannot get build log (cuda): {:?}!", e),
        }
    }

    fn build_wrap<T>(&self, program: &nvrtc::NvrtcProgram, res: Result<T, nvrtc::error::NvrtcError>) -> ComputeResult<T> {
        match res {
            Ok(r) => { return Ok(r); },
            Err(e) => {
                println!("{}", self.build_fail(&self.get_build_log(program), Some("Compiling"), None));
                return Err(ComputeError::KernelBuilding);
            }
        }
    }
    // Returns (PTX, Vec<kernels>)
    fn build_ptx(&self) -> Result<(String, HashMap<String, String>), Box<Error>> {
        let src = match &self.source {
            Some(s) => s,
            None => panic!("Trying to build program without source."),
        };
        let headers: Vec<_> = self.headers.iter().map(|s| s.as_str()).collect();
        let headers_names: Vec<_> = self.headers_names.iter().map(|s| s.as_str()).collect();
        let headers_names: Vec<_> = self.headers_names.iter().map(|s| s.as_str()).collect();
        let compiler_opts: Vec<_> = self.compiler_options.iter().map(|s| s.as_str()).collect();
        let program_ptx = nvrtc::NvrtcProgram::new(src, None, &headers, &headers_names)?;
        for fn_name in &self.functions {
            program_ptx.add_expr(fn_name)?;
        }
        self.build_wrap(&program_ptx, program_ptx.compile(&compiler_opts))?;
        let mut fn_map = HashMap::new();
        for fn_name in &self.functions {
            let ptx_fn = program_ptx.get_name(fn_name)?;
            fn_map.insert(fn_name.clone(), ptx_fn);
        }
        Ok((program_ptx.get_ptx()?, fn_map))
    }

    // Returns string with PTX and 
    fn get_ptx(&mut self) -> Option<(String, HashMap<String, PtxFunctionInfo>)> {
        //let (ptx, fn_map) = self.build_ptx()?;

        None
    }
}

impl<'a> ComputeProgramBuilder<'a> for CudaProgramBuilder<'a> {
    fn queue(&self) -> &dyn ComputeQueue { self.queue }
    fn compiler_opt(&mut self, opt: &str) -> &mut (dyn ComputeProgramBuilder<'a> + 'a) {
        self.compiler_options.push(opt.to_string());
        self
    }
    fn debug(&mut self) -> &mut (dyn ComputeProgramBuilder<'a> + 'a) { 
        self.debug = true;
        self.compiler_opt("-lineinfo");
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
        self.add_header(&include_str!("../../headers/cuda.cu"), "/gpu_compute/platform.h");
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
    fn build_source(&self) -> Result<Box<dyn ComputeProgram + 'a>, Box<Error>>{
        let (ptx, fn_map) = self.build_ptx()?;
        let module = Module::load_from_string(&CString::new(ptx.as_str())?)?;
        Ok(Box::new(CudaProgram::new(self.queue, module, self.functions.clone(), Some((ptx, fn_map)) )))
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
    // TODO: very ugly, do something
    fn get_ptx_info(&self) -> Option<HashMap<String, PtxFunctionInfo>> {
        let mut map = HashMap::with_capacity(self.functions.len());
        for name in &self.functions {
            let f = self.get_fn(name).unwrap();
            let registers = Some(f.get_attribute(FunctionAttribute::NumRegisters).unwrap() as usize);
            let mem_local = Some(f.get_attribute(FunctionAttribute::LocalSizeBytes).unwrap() as usize);
            let mem_shared = Some(f.get_attribute(FunctionAttribute::SharedMemorySizeBytes).unwrap() as usize);
            let mem_const = Some(f.get_attribute(FunctionAttribute::ConstSizeBytes).unwrap() as usize);
            let max_threads = Some(f.get_attribute(FunctionAttribute::MaxThreadsPerBlock).unwrap() as usize);
            let info = PtxFunctionInfo { registers, mem_local, mem_shared, mem_const, max_threads };
            map.insert(name.to_string(), info);
        }
        Some(map)
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
