use std::collections::HashMap;
use ansi_term::Colour;
use crate::enums::{Vendor, CompilerOpt, PtxInfo};
use crate::program::SourceProgramBuilder;

pub trait ComputePlatform {
    // Static method signature; `Self` refers to the implementor type.
    fn new() -> Self where Self: std::marker::Sized;
    fn init(&self) -> Result<(), Box<dyn std::error::Error>> where Self: std::marker::Sized;
    fn name(&self) -> &str;
    fn short_name(&self) -> &str;
    fn stub_header(&self) -> &'static str;
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
    fn vendor(&self) -> Vendor;
    fn platform_vendor(&self) -> Vendor;
    fn version_tuple(&self) -> (usize, usize);
    fn version(&self) -> usize {
        let (major, minor) = self.version_tuple();
        major*100+minor*10
    }
    fn details(&self) -> String;
    fn platform(&self) -> &dyn ComputePlatform;
    fn default_queue<'a>(&'a self) -> &'a (dyn ComputeQueue + 'a);
    fn queue<'a>(&'a self) -> Box<dyn ComputeQueue + 'a>;
    fn compiler_opts(&self, opts: &[CompilerOpt]) -> Vec<String>;
}

impl<'a> std::fmt::Debug for ComputeDevice + 'a {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(formatter, "ComputeDevice<{}/{}>", self.platform().name(), self.name())
    }
}

pub trait ComputeQueue {
    fn device(&self) -> &dyn ComputeDevice;
    fn build_source(&self, builder: &SourceProgramBuilder);
    fn get_ptx_info(&self, builder: &SourceProgramBuilder) -> Option<PtxInfo>;

    fn flush(&self);
}

impl<'a> std::fmt::Debug for ComputeQueue + 'a {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        let device = self.device();
        let platform = device.platform();
        write!(formatter, "ComputeQueue(platform={} dev={})", platform.name(), device.name())
    }
}

pub trait ComputeProgram {
    fn get_ptx(&self) -> Option<String>;
//    fn queue(&self) -> &dyn ComputeQueue;
}

impl<'a> std::fmt::Debug for ComputeProgram + 'a {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(formatter, "ComputeProgram()")
    }
}
