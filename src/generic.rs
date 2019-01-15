use std::collections::HashMap;
use ansi_term::Colour;

#[derive(PartialEq, Debug)]
pub enum ComputeVendor {
    NVIDIA,
    AMD,
    Intel,
    Apple,
    Generic,
}

#[derive(Debug)]
pub enum ComputeProgramType {
    Binary,
    Source,
    
}
#[derive(Debug)]
pub enum ComputeError {
    KernelBuilding,
}

impl std::error::Error for ComputeError {}

impl std::fmt::Display for ComputeError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub type ComputeResult<T> = ::std::result::Result<T, ComputeError>;


pub trait ComputePlatform {
    // Static method signature; `Self` refers to the implementor type.
    fn new() -> Self where Self: std::marker::Sized;
    fn init(&self) -> Result<(), Box<dyn std::error::Error>> where Self: std::marker::Sized;
    fn name(&self) -> &str;
    fn short_name(&self) -> &str;
    fn list_devices<'a>(&'a self) -> Vec<Box<dyn ComputeDevice + 'a>>;
    fn default_device<'a>(&'a self) -> Option<Box<dyn ComputeDevice + 'a>> {
        let mut dev = self.list_devices();
        dev.sort_by(|a, b| {
            b.priority().cmp(&a.priority())
        });
        dev.into_iter().next()
    }
}

impl<'a> std::fmt::Debug for ComputePlatform + 'a {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(formatter, "ComputePlatform<{}>", self.name())
    }
}

pub trait ComputeDevice {
    fn name(&self) -> String;
    fn priority(&self) -> usize { 0 }
    fn vendor(&self) -> ComputeVendor;
    fn platform_vendor(&self) -> ComputeVendor;
    fn details(&self) -> String;
    fn platform(&self) -> &dyn ComputePlatform;
    fn queue<'a>(&'a self) -> Box<dyn ComputeQueue + 'a>;
}

impl<'a> std::fmt::Debug for ComputeDevice + 'a {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(formatter, "ComputeDevice<{}/{}>", self.platform().name(), self.name())
    }
}

pub trait ComputeQueue {
    fn device(&self) -> &dyn ComputeDevice;
    fn program_builder<'a>(&'a self) -> Box<dyn ComputeProgramBuilder + 'a>;
    fn flush(&self);
}

impl<'a> std::fmt::Debug for ComputeQueue + 'a {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        let device = self.device();
        let platform = device.platform();
        write!(formatter, "ComputeQueue(platform={} dev={})", platform.name(), device.name())
    }
}

pub trait ComputeProgramBuilder<'a> {
    fn queue(&self) -> &dyn ComputeQueue;
    // Builder
    fn compiler_opt(&mut self, opt: &str) -> &mut (dyn ComputeProgramBuilder<'a> + 'a);
    fn debug(&mut self) -> &mut (dyn ComputeProgramBuilder<'a> + 'a);
    fn add_header(&mut self, src: &str, name: &str) -> &mut (dyn ComputeProgramBuilder<'a> + 'a);
    fn add_stub_header(&mut self) -> &mut (dyn ComputeProgramBuilder<'a> + 'a);
    fn add_fn(&mut self, name: &str) -> &mut (dyn ComputeProgramBuilder<'a> + 'a);
    fn set_source(&mut self, src: &str) -> &mut (dyn ComputeProgramBuilder<'a> + 'a);

    // Debug
    fn build_fail(&self, log: &str, build_type: Option<&str>, file: Option<&str>) -> String {
        let b_failed = Colour::Red.paint("BUILD FAILED!");
        let small_h = Colour::Yellow.paint("################################");
        let big_h = Colour::Yellow.paint("###############################################################################");
        let file_h = match file {
            Some(file) => format!(" File: {}", Colour::Yellow.paint(file)),
            None => "".to_string(),
        };
        let type_h = match build_type {
            Some(file) => format!(" Type: {}", Colour::Yellow.paint(file)),
            None => "".to_string(),
        };
        let info = format!("{} Platform: {}{}{}", 
            Colour::Yellow.paint("#"),
            Colour::Yellow.paint(self.queue().device().platform().name()),
            type_h, file_h);
        format!("{} {} {}\n{}\n{}\n{}\n{}\n", small_h, b_failed, small_h, info, big_h, log, big_h)
    }
    // Final
    fn build_source(&self) -> Result<Box<dyn ComputeProgram + 'a>, Box<std::error::Error>>;
}

// NVIDIA specific struct of kernel sizes
pub struct PtxFunctionInfo {
    pub registers: Option<usize>,
    pub mem_local: Option<usize>,
    pub mem_shared: Option<usize>,
    pub mem_const: Option<usize>,
    // MaxThreadsPerBlock
    pub max_threads: Option<usize>,
}

impl std::fmt::Display for PtxFunctionInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut s = String::new();
        if let Some(registers) = self.registers {
            s.push_str("R: ");
            s.push_str(registers.to_string().as_str())
        }
        if let Some(mem_local) = self.mem_local {
            s.push_str(" L: ");
            s.push_str(mem_local.to_string().as_str())
        }
        if let Some(mem_shared) = self.mem_shared {
            s.push_str(" S: ");
            s.push_str(mem_shared.to_string().as_str())
        }
        if let Some(mem_const) = self.mem_const {
            s.push_str(" C: ");
            s.push_str(mem_const.to_string().as_str())
        }
        if let Some(max_threads) = self.max_threads {
            s.push_str(" T: ");
            s.push_str(max_threads.to_string().as_str())
        }
        write!(f, "{}", s)
    }
}

pub trait ComputeProgram {
    fn get_ptx(&self) -> Option<String>;
    fn get_ptx_info(&self) -> Option<HashMap<String, PtxFunctionInfo>>;
//    fn queue(&self) -> &dyn ComputeQueue;
}

impl<'a> std::fmt::Debug for ComputeProgram + 'a {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(formatter, "ComputeProgram()")
    }
}
