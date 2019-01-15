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
