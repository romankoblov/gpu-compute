use std::collections::HashMap;
use regex::Regex;

#[derive(PartialEq, Display, Debug)]
pub enum Vendor {
    NVIDIA,
    AMD,
    Intel,
    Apple,
    Generic,
}

impl Vendor {
    pub fn parse(platform: &str) -> Vendor {
        match platform {
            "NVIDIA Corporation" | "NVIDIA" => Vendor::NVIDIA,
            "Advanced Micro Devices, Inc." | "AuthenticAMD" | "AMD" => Vendor::AMD,
            "GenuineIntel" | "Intel" | "Intel(R) Corporation" => Vendor::Intel,
            "Apple" => Vendor::Apple,
            _ => Vendor::Generic,
        }
    }
}

#[derive(PartialEq, Display, Debug)]
pub enum CompilerOpt {
    LineInfo, // Lineinfo for cuda
    Debug, // debug for cuda
    NV_PTAX, // -cl-nv-verbose
    AMD_TEMPS,   //-save-temps
}

// NVIDIA specific struct of kernel sizes
#[derive(PartialEq, Debug, Clone)]
pub struct PtxFunctionInfo {
    pub registers: Option<usize>,
    pub mem_local: Option<usize>,
    pub mem_shared: Option<usize>,
    pub mem_const: Option<usize>,
    // MaxThreadsPerBlock
    pub max_threads: Option<usize>,
}

impl PtxFunctionInfo {
    pub fn parse_ptxas(out: &str) -> HashMap<String, PtxFunctionInfo> {
        lazy_static::lazy_static! {
            static ref RE_FN: Regex = Regex::new(r"Function properties for (?P<name>[A-Za-z_0-9]+)").unwrap();
            static ref RE_STACK: Regex = Regex::new(r"(?P<stack>\d+) bytes stack frame, (?P<stores>\d+) bytes spill stores, (?P<loads>\d+) bytes spill loads").unwrap();
            static ref RE_REGISTERS: Regex = Regex::new(r"Used (?P<registers>\d+) registers(, (?P<smem>\d+) bytes smem)?(, (?P<cmem>\d+) bytes cmem)?").unwrap();
        }
        let mut ptx_info_map = HashMap::new();
        // Stack -- local memory
        let mut name = None;
        let mut stack = 0;
        let mut stores = 0;
        let mut loads = 0;
        for line in out.lines() {
            let line = line.trim();
            if !line.starts_with("ptxas") { continue; }
            if let Some(c) = RE_FN.captures(line) {
                if name.is_some() { panic!("PTXAS: name already set: {:?}, got: {}", name, line); }
                name = Some(c["name"].to_string());
            }
            if let (Some(c), Some(n)) = (&RE_STACK.captures(line), &name) {
                stack = c["stack"].parse::<usize>().unwrap();
                stores = c["stores"].parse::<usize>().unwrap();
                loads = c["loads"].parse::<usize>().unwrap();

            }
            if let (Some(c), Some(n)) = (&RE_REGISTERS.captures(line), &name) {
                let registers = c["registers"].parse::<usize>().unwrap();
                let cmem = if let Some(cmem_m) = c.name("cmem") { cmem_m.as_str().parse::<usize>().unwrap() } else { 0 };
                let smem = if let Some(smem_m) = c.name("smem") { smem_m.as_str().parse::<usize>().unwrap() } else { 0 };

                let info = PtxFunctionInfo { 
                    registers: Some(registers), 
                    mem_local: Some(stack), 
                    mem_shared: Some(smem), 
                    mem_const: Some(cmem), 
                    max_threads: None 
                };
                ptx_info_map.insert(n.to_string(), info);
                name = None;
            }
        }
        ptx_info_map
    }
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

pub struct PtxInfo {
    pub ptx: String,
    pub functions: HashMap<String, PtxFunctionInfo>,
    pub other_ptx: Vec<(String, String)>, // another ptx, for OpenCL compile mode
}

impl PtxInfo {
    pub fn new(ptx: &str) -> PtxInfo {
        let functions = HashMap::new();
        let other_ptx = Vec::new();
        PtxInfo { ptx: ptx.to_string(), functions, other_ptx }
    }
    pub fn add_fn(&mut self, name: &str, info: PtxFunctionInfo) {
        self.functions.insert(name.to_string(), info);
    }
    pub fn add_ptx(&mut self, name: &str, ptx: &str) {
        self.other_ptx.push((name.to_string(), ptx.to_string()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ptxas_info() {
        let line = "ptxas info    : 10 bytes gmem, 12 bytes cmem[3]\nptxas info    : Compiling entry function \'say_hi\' for \'sm_61\'\nptxas info    : Function properties for say_hi\nptxas         .     8 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\nptxas info   : Used 8 registers, 328 bytes cmem[0]\nptxas info    : Compiling entry function \'say_hi2\' for \'sm_61\'\nptxas info    : Function properties for say_hi2\nptxas         .     0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\nptxas info    : Used 6 registers, 328 bytes cmem[0]";
        let out = PtxFunctionInfo::parse_ptxas(line);
        assert_eq!(format!("{}", out.get("say_hi").unwrap()).as_str(), "R: 8 L: 8 S: 0 C: 328");
        assert_eq!(format!("{}", out.get("say_hi2").unwrap()).as_str(), "R: 6 L: 0 S: 0 C: 328");
    }
    #[test]
    fn parse_ptxas_info_smem() {
        let line = "ptxas info    : 20 bytes gmem, 12 bytes cmem[3]\nptxas info    : Compiling entry function \'say_hi\' for \'sm_61\'\nptxas info    : Function properties for say_hi\nptxas         .     8 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\nptxas info   : Used 8 registers, 328 bytes cmem[0]\nptxas info    : Compiling entry function \'say_hi2\' for \'sm_61\'\nptxas info    : Function properties for say_hi2\nptxas         .     8 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\nptxas info    : Used 20 registers, 404 bytes smem, 328 bytes cmem[0]";
        let out = PtxFunctionInfo::parse_ptxas(line);
        assert_eq!(format!("{}", out.get("say_hi").unwrap()).as_str(), "R: 8 L: 8 S: 0 C: 328");
        assert_eq!(format!("{}", out.get("say_hi2").unwrap()).as_str(), "R: 20 L: 8 S: 404 C: 328");
    }
}
