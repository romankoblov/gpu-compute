use crate::grid::GridSize;

pub trait GenericFnBuilder {
    
}

pub struct FnBuilder {
    name: String,
    parallel: usize,
    grid: Option<GridSize>,
}
impl FnBuilder {}
impl GenericFnBuilder for FnBuilder {}

// pub struct PipelineBuilder {}
// impl PipelineBuilder {}
// impl GenericFnBuilder for PipelineBuilder {}
