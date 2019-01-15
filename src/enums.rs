#[derive(PartialEq, Display, Debug)]
pub enum ComputeVendor {
    NVIDIA,
    AMD,
    Intel,
    Apple,
    Generic,
}

impl ComputeVendor {
    pub fn parse(platform: &str) -> ComputeVendor {
        match platform {
            "NVIDIA Corporation" | "NVIDIA" => ComputeVendor::NVIDIA,
            "Advanced Micro Devices, Inc." | "AuthenticAMD" | "AMD" => ComputeVendor::AMD,
            "GenuineIntel" | "Intel" | "Intel(R) Corporation" => ComputeVendor::Intel,
            "Apple" => ComputeVendor::Apple,
            _ => ComputeVendor::Generic,
        }
    }
}
