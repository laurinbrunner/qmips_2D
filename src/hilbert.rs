use crate::numerics::{self, MATRIX_DIMENSION};
use num_complex::Complex;
use serde::{Deserialize, Serialize};
use std::fmt;

pub const NUM_POSSIBLE_HAMILTONIANS: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize, Copy)]
pub struct SystemSize {
    pub l_x: usize,
    pub l_y: usize,
}

impl SystemSize {
    pub fn new(l_x: usize, l_y: usize) -> Self {
        SystemSize { l_x, l_y }
    }

    /// Total number of lattice sites.
    pub fn num_sites(&self) -> usize {
        self.l_x * self.l_y
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Occupation {
    Empty = 0,
    Occupied = 1,
}

impl From<Occupation> for usize {
    fn from(occ: Occupation) -> Self {
        occ as usize
    }
}

impl fmt::Debug for Occupation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Occupation::Empty => write!(f, "0"),
            Occupation::Occupied => write!(f, "1"),
        }
    }
}

/// Builds the 16 per-particle 4x4 non-Hermitian effective Hamiltonians, one
/// per neighbor-occupancy bitmask (bit order: left, right, up, down).
///
/// The diagonal anti-Hermitian decay entries include both the forward jump
/// channels (`rates`, gated on the same-direction neighbor) and the two
/// lateral jump channels of each internal direction (`lateral_rate` = ε,
/// gated on the neighbor in the lateral hop direction, i.e. summed over the
/// two neighbors perpendicular to that direction):
///
/// ```text
/// Gamma_L = rates[0] * m_left  + lateral_rate * (m_up + m_down)
/// Gamma_R = rates[1] * m_right + lateral_rate * (m_up + m_down)
/// Gamma_U = rates[2] * m_up    + lateral_rate * (m_left + m_right)
/// Gamma_D = rates[3] * m_down  + lateral_rate * (m_left + m_right)
/// ```
///
/// where `m_d = 1 - occ_d` is the target-emptiness mask. With
/// `lateral_rate == 0.0` this reduces exactly (bit-for-bit) to the
/// forward-only model.
///
/// Note that `sum_k M_k^dag M_k` -- and hence `H_eff` -- is the *same* for
/// the rank-1 lateral operators the code implements
/// (`M_{r,sigma,mu} = sqrt(eps) b^dag_{r+e_mu,sigma} b_{r,sigma}`, one per
/// (sigma, perpendicular mu)) and for the rank-2 variant that sums the two
/// sigma sharing a displacement; the two differ only in the post-jump state.
/// See `quantum_lateral_Heff_derivation.md` §3/§6.3.
pub fn get_hamiltonians(
    transverse_field_strength: f64,
    rates: [f64; MATRIX_DIMENSION],
    lateral_rate: f64,
) -> [numerics::Matrix; NUM_POSSIBLE_HAMILTONIANS] {
    let neg_g = Complex::new(-transverse_field_strength, 0.0);
    let complex_zero = Complex::ZERO; //new(0.0, 0.0);

    let base_matrix = numerics::Matrix {
        data: [
            [complex_zero, complex_zero, neg_g, neg_g],
            [complex_zero, complex_zero, neg_g, neg_g],
            [neg_g, neg_g, complex_zero, complex_zero],
            [neg_g, neg_g, complex_zero, complex_zero],
        ],
    };

    let mut hamiltonian = [base_matrix; NUM_POSSIBLE_HAMILTONIANS];

    for i in 0..NUM_POSSIBLE_HAMILTONIANS {
        let occ_left_neigh = (i >> 0) & 1;
        let occ_right_neigh = (i >> 1) & 1;
        let occ_upper_neigh = (i >> 2) & 1;
        let occ_lower_neigh = (i >> 3) & 1;
        let m_left = 1.0 - occ_left_neigh as f64;
        let m_right = 1.0 - occ_right_neigh as f64;
        let m_up = 1.0 - occ_upper_neigh as f64;
        let m_down = 1.0 - occ_lower_neigh as f64;
        hamiltonian[i].data[0][0] =
            -Complex::new(0.0, 0.5 * (rates[0] * m_left + lateral_rate * (m_up + m_down)));
        hamiltonian[i].data[1][1] =
            -Complex::new(0.0, 0.5 * (rates[1] * m_right + lateral_rate * (m_up + m_down)));
        hamiltonian[i].data[2][2] =
            -Complex::new(0.0, 0.5 * (rates[2] * m_up + lateral_rate * (m_left + m_right)));
        hamiltonian[i].data[3][3] =
            -Complex::new(0.0, 0.5 * (rates[3] * m_down + lateral_rate * (m_left + m_right)));
    }

    hamiltonian
}

pub fn get_eigenvalues_and_vectors(
    transverse_field_strength: f64,
    rates: [f64; MATRIX_DIMENSION],
    lateral_rate: f64,
) -> (
    [[Complex<f64>; MATRIX_DIMENSION]; NUM_POSSIBLE_HAMILTONIANS],
    [numerics::Matrix; NUM_POSSIBLE_HAMILTONIANS],
) {
    let hams = get_hamiltonians(transverse_field_strength, rates, lateral_rate);

    // Implement function to compute eigenvalues and eigenvectors
    let mut eigenvalues = [[Complex::new(0.0, 0.0); MATRIX_DIMENSION]; NUM_POSSIBLE_HAMILTONIANS];
    let mut eigenvectors = [numerics::Matrix::zeros(); NUM_POSSIBLE_HAMILTONIANS];

    for i in 0..NUM_POSSIBLE_HAMILTONIANS {
        let (vals, vecs) = hams[i].eigen();
        eigenvalues[i] = vals;
        eigenvectors[i] = vecs;
    }

    (eigenvalues, eigenvectors)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independently recomputes the expected diagonal decay rate for state
    /// index `state` (0=L,1=R,2=U,3=D) given a bitmask, mirroring the
    /// Gamma_sigma formulas of the derivation note (§5), and checks it
    /// against `get_hamiltonians`.
    fn expected_diag(rates: [f64; 4], lateral_rate: f64, bitmask: usize, state: usize) -> f64 {
        let m = |bit: usize| 1.0 - ((bitmask >> bit) & 1) as f64;
        let (m_left, m_right, m_up, m_down) = (m(0), m(1), m(2), m(3));
        let gamma = match state {
            0 => rates[0] * m_left + lateral_rate * (m_up + m_down),
            1 => rates[1] * m_right + lateral_rate * (m_up + m_down),
            2 => rates[2] * m_up + lateral_rate * (m_left + m_right),
            3 => rates[3] * m_down + lateral_rate * (m_left + m_right),
            _ => unreachable!(),
        };
        0.5 * gamma
    }

    #[test]
    fn diagonal_matches_gamma_formula_with_lateral_rate() {
        let rates = [1.0, 1.0, 1.0, 1.0];
        let lateral_rate = 0.37;
        let hams = get_hamiltonians(0.6, rates, lateral_rate);

        // A handful of representative bitmasks: all-empty, all-occupied,
        // and a few mixed cases.
        for &bitmask in &[0usize, 15, 1, 2, 4, 8, 5, 10, 3, 12, 7, 11] {
            for state in 0..4 {
                let expected = expected_diag(rates, lateral_rate, bitmask, state);
                let got = hams[bitmask].data[state][state];
                assert!(
                    (got.re - 0.0).abs() < 1e-14,
                    "real part of diagonal entry should be zero, got {:?} (bitmask {}, state {})",
                    got,
                    bitmask,
                    state
                );
                assert!(
                    (got.im - (-expected)).abs() < 1e-12,
                    "diagonal mismatch at bitmask {} state {}: got {:?}, expected -i*{}",
                    bitmask,
                    state,
                    got,
                    expected
                );
            }
        }
    }

    #[test]
    fn diagonal_with_anisotropic_rates_and_lateral_rate() {
        let rates = [0.5, 1.5, 2.0, 1.0];
        let lateral_rate = 1.25;
        let hams = get_hamiltonians(0.3, rates, lateral_rate);

        for bitmask in 0..NUM_POSSIBLE_HAMILTONIANS {
            for state in 0..4 {
                let expected = expected_diag(rates, lateral_rate, bitmask, state);
                let got = hams[bitmask].data[state][state];
                assert!((got.im - (-expected)).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn lateral_rate_zero_reproduces_forward_only_diagonal_bit_exact() {
        let rates = [1.0, 1.0, 1.0, 1.0];
        let hams_eps0 = get_hamiltonians(0.3, rates, 0.0);

        // Recompute the pristine (forward-only) formula directly.
        for bitmask in 0..NUM_POSSIBLE_HAMILTONIANS {
            let m = |bit: usize| 1.0 - ((bitmask >> bit) & 1) as f64;
            for (state, &rate) in rates.iter().enumerate() {
                let expected_im = -0.5 * rate * m(state);
                let got = hams_eps0[bitmask].data[state][state];
                assert_eq!(got.im, expected_im, "bitmask {} state {}", bitmask, state);
            }
        }
    }
}
