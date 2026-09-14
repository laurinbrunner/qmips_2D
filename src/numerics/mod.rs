pub const MATRIX_DIMENSION: usize = 4;

pub mod matrix;
pub mod spatial;
pub mod spin_wavefunction;

pub use matrix::Matrix;
pub use spatial::{Direction, LatticeState};
pub use spin_wavefunction::{DOWN_STATE, LEFT_STATE, RIGHT_STATE, SpinWavefunction, UP_STATE};
