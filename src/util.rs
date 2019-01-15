use std::error::Error;
use super::generic::{ComputeVendor};

pub fn parse_platform(platform: &str) -> ComputeVendor {
    match platform {
        "NVIDIA Corporation" | "NVIDIA" => ComputeVendor::NVIDIA,
        "Advanced Micro Devices, Inc." | "AuthenticAMD" | "AMD" => ComputeVendor::AMD,
        "GenuineIntel" | "Intel" | "Intel(R) Corporation" => ComputeVendor::Intel,
        "Apple" => ComputeVendor::Apple,
        _ => ComputeVendor::Generic,
    }
}
// Properly process dependencies
pub fn legacy_process(src: &str, name: &str) -> String {
    let mut file_string = format!("\n// FILE: {}", name);
    for line in src.lines(){
        if !line.trim().starts_with("#include") {
            //println!("Legacy:  {}", line);
            file_string.push_str(line);
            file_string.push('\n');
        }
    }
    file_string
}
