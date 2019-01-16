use std::collections::HashMap;

use ansi_term::Colour;

use crate::enums::{CompilerOpt};
use crate::backend::generic::{ComputeDevice};

// TODO: a lot of allocations, fix.
struct LegacyIncludeProvider {
    name: String,
    map: HashMap<String, String>,
}

impl LegacyIncludeProvider {
    fn new(headers: Vec<String>, names: Vec<String>, source: &str, name: &str) -> LegacyIncludeProvider {
        let mut map = HashMap::with_capacity(headers.len());
        for i in 0..headers.len() {
            map.insert(names[i].clone(), headers[i].clone());
        }
        map.insert(name.to_string(), source.to_string());
        LegacyIncludeProvider { name: name.to_string(), map }
    }
    fn get_source(&mut self) -> String {
        let name = self.name.clone();
        let chunks = shader_prepper::process_file(name.as_str(), self, ()).unwrap();
        chunks.into_iter().map(|chunk| chunk.source).collect::<Vec<_>>().join("")
    }
}

impl shader_prepper::IncludeProvider for LegacyIncludeProvider {
    type IncludeContext = ();

    fn get_include(&mut self, path: &str, _context: &Self::IncludeContext) -> Result<(String, Self::IncludeContext), failure::Error> {
        match self.map.get(&path.to_string()) {
            Some(source) => Ok((String::from(source.as_str()), ())),
            None => Err(failure::err_msg("Cannot find header")),
        }     
    }
}

pub struct SourceProgramBuilder {
    pub name: Option<String>,
    debug: bool,
    use_stub: bool,
    headers: Vec<(String, String)>,
    pub compiler_options: Vec<CompilerOpt>,
    functions: Vec<String>,
    source: Option<String>,
}

impl SourceProgramBuilder {
    pub fn new(program_name: Option<&str>) -> SourceProgramBuilder {
        let headers = Vec::new();
        let compiler_options = Vec::new();
        let functions = Vec::new();
        let name = match program_name {
            Some(n) => Some(n.to_string()),
            None => None,
        };
        SourceProgramBuilder { 
            name, 
            debug: false,
            use_stub: true,
            headers, compiler_options, functions, 
            source: None
        }
    }
    pub fn compiler_opt(mut self, opt: CompilerOpt) -> SourceProgramBuilder {
        self.compiler_options.push(opt);
        self
    }
    pub fn debug(mut self) -> SourceProgramBuilder {
        self.debug = true;
        self
    }
    pub fn add_header(mut self, src: &str, name: &str) -> SourceProgramBuilder {
        self.headers.push((String::from(src), String::from(name)));
        self
    }
    fn cache_header(&self) -> String {
        let uuid = if self.debug { format!("{}", uuid::Uuid::new_v4().to_simple()) } else { "".to_string() };
        format!("CONSTANT int GPU_COMPUTE_DISABLE_CACHE_{} = 0;", uuid)
    }
    fn vendor_header(&self, device: &dyn ComputeDevice) -> String {
        let vendor = device.platform_vendor().to_string();
        format!(r#"
#define PLATFROM_{} 1
#define VENDOR_{} 1
        "#, device.platform().name().to_uppercase(), vendor.to_uppercase())
    } 
    pub fn add_fn(mut self, name: &str) -> SourceProgramBuilder {
        self.functions.push(name.to_string());
        self
    }
    pub fn set_source(mut self, src: &str) -> SourceProgramBuilder {
        if self.source.is_some() { panic!("Source already set for kernel."); }
        self.source = Some(src.to_string());
        self
    }
    pub fn fmt_build_fail_log(&self, device: &dyn ComputeDevice, log: &str) -> String {
        let b_failed = Colour::Red.paint("BUILD FAILED!");
        let small_h = Colour::Yellow.paint("################################");
        let big_h = Colour::Yellow.paint("###############################################################################");
        let name_h = match &self.name {
            Some(program) => format!(" Program: {}", Colour::Yellow.paint(program)),
            None => "".to_string(),
        };
        let h_names: Vec<&str> = self.headers.iter().map(|h| h.1.as_str()).collect();
        let headers = format!(" Headers: {}", Colour::Yellow.paint(h_names.join(", ")));
        let info = format!("{} Platform: {}{}{}", 
            Colour::Yellow.paint("#"),
            Colour::Yellow.paint(device.platform().name()),
            name_h, headers);
        format!("{} {} {}\n{}\n{}\n{}\n{}\n", small_h, b_failed, small_h, info, big_h, log, big_h)
    }
    pub fn get_source(&self) -> &str {
        match &self.source {
            Some(s) => s,
            None => panic!("Trying to build program without source."),
        }
    }
    // Later: check pipelines
    pub fn get_functions(&self) -> Vec<String> {
        let mut fn_vec = Vec::with_capacity(self.functions.len());
        for f in &self.functions {
            fn_vec.push(f.clone());
        }
        fn_vec
    }
    // TODO: a lot of useless allocations here
    pub fn generate_headers(&self, device: &dyn ComputeDevice) -> (Vec<String>, Vec<String>) {
        let headers_len = self.headers.len()+3;
        let mut headers = Vec::with_capacity(headers_len);
        let mut names = Vec::with_capacity(headers_len);
        // Vendor
        let vendor_src = self.vendor_header(device);
        headers.push(self.vendor_header(device));
        names.push("/gpu_compute/vendor.h".to_string());
        // Cache
        headers.push(self.cache_header());
        names.push("/gpu_compute/cache.h".to_string());
        // Platform
        headers.push(device.platform().stub_header().to_string());
        names.push("/gpu_compute/platform.h".to_string());
        for (src, name) in &self.headers {
            headers.push(src.clone());
            names.push(name.clone());
        }
        debug_assert_eq!(headers.len(), names.len(), "program headers and headers names length mismatch");
        (headers, names)
    }
    // OpenCL < 1.2; OpenCL get_ptx;  Probably shaders.
    pub fn legacy_source(&self, device: &dyn ComputeDevice) -> String {
        let (headers, names) = self.generate_headers(device);
        let source = self.get_source();
        let mut provider = LegacyIncludeProvider::new(headers, names, source, "LEGACY_KERNEL");
        provider.get_source()
    }
}

// TODO: BinaryProgramBuilder?