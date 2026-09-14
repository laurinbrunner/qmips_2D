use num_complex::Complex;
use std::ops::{Add, Mul};
use std::ops::{Index, IndexMut};

use crate::numerics::Direction;

use super::MATRIX_DIMENSION;

pub static LEFT_STATE: [Complex<f64>; MATRIX_DIMENSION] = [
    Complex::new(1.0, 0.0),
    Complex::new(0.0, 0.0),
    Complex::new(0.0, 0.0),
    Complex::new(0.0, 0.0),
];

pub static RIGHT_STATE: [Complex<f64>; MATRIX_DIMENSION] = [
    Complex::new(0.0, 0.0),
    Complex::new(1.0, 0.0),
    Complex::new(0.0, 0.0),
    Complex::new(0.0, 0.0),
];

pub static UP_STATE: [Complex<f64>; MATRIX_DIMENSION] = [
    Complex::new(0.0, 0.0),
    Complex::new(0.0, 0.0),
    Complex::new(1.0, 0.0),
    Complex::new(0.0, 0.0),
];

pub static DOWN_STATE: [Complex<f64>; MATRIX_DIMENSION] = [
    Complex::new(0.0, 0.0),
    Complex::new(0.0, 0.0),
    Complex::new(0.0, 0.0),
    Complex::new(1.0, 0.0),
];

#[derive(Debug, Clone)]
pub struct SpinWavefunction {
    pub psi: Vec<[Complex<f64>; MATRIX_DIMENSION]>,
}

impl SpinWavefunction {
    pub fn new(psi: Vec<[Complex<f64>; MATRIX_DIMENSION]>) -> Self {
        SpinWavefunction { psi }
    }

    pub fn zeros(dimension: usize) -> Self {
        SpinWavefunction {
            psi: vec![[Complex::new(0.0, 0.0); MATRIX_DIMENSION]; dimension],
        }
    }

    pub fn normalized_clone(&self) -> Self {
        let mut normalized_psi = self.clone();
        for n in 0..self.psi.len() {
            let norm: f64 = self[n].iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
            for val in normalized_psi[n].iter_mut() {
                *val /= norm;
            }
        }
        normalized_psi
    }

    pub fn norm_squared(&self) -> f64 {
        let mut total: f64 = 1.0;
        for n in 0..self.psi.len() {
            total *= self[n].iter().map(|&c| c.norm_sqr()).sum::<f64>().sqrt();
        }
        total.abs().powi(2)
    }

    /// Returns an iterator over the spin states.
    pub fn iter(&self) -> std::slice::Iter<'_, [Complex<f64>; MATRIX_DIMENSION]> {
        self.psi.iter()
    }

    /// Returns a mutable iterator over the spin states.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, [Complex<f64>; MATRIX_DIMENSION]> {
        self.psi.iter_mut()
    }

    /// Post-jump internal state for *any* jump channel: full collapse onto
    /// the direction eigenstate `|spin_dir>`.
    ///
    /// Every jump operator of the model is rank-1 in the internal space --
    /// forward `M_{r,sigma} = sqrt(gamma_sigma) b^dag_{r+e_sigma,sigma} b_{r,sigma}`
    /// and lateral `M_{r,sigma,mu} = sqrt(eps) b^dag_{r+e_mu,sigma} b_{r,sigma}`
    /// alike act internally as the projector `P_sigma` -- so the post-jump
    /// state is always `|sigma>`, independent of the displacement. For a
    /// forward jump `sigma` *is* the displacement; for a lateral jump the
    /// displacement `mu` is one of the two directions perpendicular to
    /// `sigma` (see `Direction::perpendicular` and
    /// `quantum_lateral_Heff_derivation.md` §2/§6.3).
    pub fn collapse_at_index(&mut self, index: usize, spin_dir: Direction) {
        self.psi[index] = match spin_dir {
            Direction::Left => LEFT_STATE,
            Direction::Right => RIGHT_STATE,
            Direction::Up => UP_STATE,
            Direction::Down => DOWN_STATE,
        };
    }
}

impl Index<usize> for SpinWavefunction {
    type Output = [Complex<f64>; MATRIX_DIMENSION];

    fn index(&self, index: usize) -> &Self::Output {
        &self.psi[index]
    }
}

impl IndexMut<usize> for SpinWavefunction {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.psi[index]
    }
}

impl Mul<Complex<f64>> for &SpinWavefunction {
    type Output = SpinWavefunction;

    fn mul(self, rhs: Complex<f64>) -> SpinWavefunction {
        let mut result_data = SpinWavefunction::zeros(self.psi.len());

        for i in 0..self.psi.len() {
            for j in 0..MATRIX_DIMENSION {
                result_data[i][j] = self.psi[i][j] * rhs;
            }
        }

        result_data
    }
}

impl Add<&SpinWavefunction> for &SpinWavefunction {
    type Output = SpinWavefunction;

    fn add(self, rhs: &SpinWavefunction) -> SpinWavefunction {
        let mut result_data = SpinWavefunction::zeros(self.psi.len());

        for i in 0..self.psi.len() {
            for j in 0..MATRIX_DIMENSION {
                result_data[i][j] = self.psi[i][j] + rhs.psi[i][j];
            }
        }

        result_data
    }
}

pub fn normalize_state(state: &SpinWavefunction) -> SpinWavefunction {
    let mut normalized_psi = state.clone();
    for n in 0..state.psi.len() {
        let norm: f64 = state[n].iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        for val in normalized_psi[n].iter_mut() {
            *val /= norm;
        }
    }
    normalized_psi
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A generic (non-eigenstate) 4-component state with all-distinct,
    /// nonzero real and imaginary parts so that any component mix-up in the
    /// collapse is caught.
    fn generic_state() -> SpinWavefunction {
        SpinWavefunction::new(vec![[
            Complex::new(0.1, 0.2),  // L (index 0)
            Complex::new(0.3, -0.4), // R (index 1)
            Complex::new(-0.5, 0.6), // U (index 2)
            Complex::new(0.7, -0.8), // D (index 3)
        ]])
    }

    /// Every jump channel -- forward or lateral -- collapses the internal
    /// state onto the direction eigenstate `|sigma>` (rank-1 projector
    /// `P_sigma`), leaving no residual amplitude anywhere else.
    #[test]
    fn collapse_gives_the_pure_direction_eigenstate() {
        for (dir, expected) in [
            (Direction::Left, LEFT_STATE),
            (Direction::Right, RIGHT_STATE),
            (Direction::Up, UP_STATE),
            (Direction::Down, DOWN_STATE),
        ] {
            let mut psi = generic_state();
            psi.collapse_at_index(0, dir);
            assert_eq!(psi[0], expected, "collapse onto {:?}", dir);
            // Already normalized: a rank-1 collapse needs no rescaling.
            assert_eq!(psi.normalized_clone()[0], expected);
        }
    }

    /// The lateral displacements available to a `sigma`-pointing particle
    /// are the two directions perpendicular to `sigma`: never forward
    /// (that is the `gamma` channel), never backward (the model has no
    /// 180° process at all), and the two are opposite to each other.
    #[test]
    fn lateral_displacements_are_the_two_perpendicular_directions() {
        for spin in Direction::ALL {
            let [mu_plus, mu_minus] = spin.perpendicular();
            assert_ne!(mu_plus, spin, "a lateral hop never moves along sigma");
            assert_ne!(mu_minus, spin, "a lateral hop never moves along sigma");
            assert_ne!(mu_plus, spin.opposite(), "a lateral hop is never backward");
            assert_ne!(mu_minus, spin.opposite(), "a lateral hop is never backward");
            assert_eq!(mu_plus.opposite(), mu_minus, "the two laterals are opposite");
        }
    }
}
