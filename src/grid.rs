

pub struct GridSize {
    // <<<dimGrid, dimBlock>>>
    // <<<blocks, threads>>> -> blocks per grid; threads per block
    x: (usize, usize),
    y: Option<(usize, usize)>,
    z: Option<(usize, usize)>,
}
impl GridSize {
    pub fn new_x(blocks: usize, threads: usize) -> GridSize {
        GridSize { x: (blocks, threads), y: None, z: None }
    }
    pub fn new_xy(blocks: (usize, usize), threads: (usize, usize)) -> GridSize {

        GridSize { x: (blocks.0, threads.0), y: Some((blocks.1, threads.1)), z: None }
    }
    pub fn new_xyz(blocks: (usize, usize, usize), threads: (usize, usize, usize)) -> GridSize {
        GridSize { x: (blocks.0, threads.0), y: Some((blocks.1, threads.1)), z: Some((blocks.2, threads.2)) }
    }
    pub fn total_grid(&self) -> (usize, usize) {
        let mut blocks = self.x.0;
        let mut threads = self.x.1;
        if let Some((b, t)) = self.y {
            blocks *= b;
            threads *= t;
        }
        if let Some((b, t)) = self.z {
            blocks *= b;
            threads *= t;
        }
        (blocks, threads)

    }
    pub fn total_size(&self) -> usize {
        let (blocks, threads) = self.total_grid();
        blocks*threads
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_grid() {
        let g = GridSize::new_xyz((2, 3, 5), (7, 11, 13));
        assert_eq!(g.total_grid(), (30, 1001), "Total grid");
        assert_eq!(g.total_size(), 30_030, "Grid size");
    }
}
