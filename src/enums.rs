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
