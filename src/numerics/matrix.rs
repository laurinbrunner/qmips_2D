use nalgebra::{SMatrix, linalg::Schur};
use num_complex::Complex;
use std::ops::{Add, Mul};

use super::MATRIX_DIMENSION;

#[derive(Debug, Clone, Copy)]
pub struct Matrix {
    pub data: [[Complex<f64>; MATRIX_DIMENSION]; MATRIX_DIMENSION],
}

impl Matrix {
    pub fn new(data: [[Complex<f64>; MATRIX_DIMENSION]; MATRIX_DIMENSION]) -> Self {
        Matrix { data }
    }

    pub fn zeros() -> Self {
        Matrix {
            data: [[Complex::new(0.0, 0.0); MATRIX_DIMENSION]; MATRIX_DIMENSION],
        }
    }

    pub fn identity() -> Self {
        let mut identity_data = [[Complex::new(0.0, 0.0); MATRIX_DIMENSION]; MATRIX_DIMENSION];
        for i in 0..MATRIX_DIMENSION {
            identity_data[i][i] = Complex::new(1.0, 0.0);
        }
        Matrix {
            data: identity_data,
        }
    }

    /// Computes the inverse using Gauss-Jordan elimination with partial pivoting.
    pub fn inverse(&self) -> Option<Matrix> {
        let n = MATRIX_DIMENSION;
        let mut a = self.data;
        let mut inv = Self::identity().data;

        for col in 0..n {
            // Partial pivoting: find row with largest element in current column
            let mut max_norm = 0.0;
            let mut pivot_row = col;
            for row in col..n {
                let norm = a[row][col].norm();
                if norm > max_norm {
                    max_norm = norm;
                    pivot_row = row;
                }
            }

            if max_norm < f64::EPSILON {
                return None;
            }

            // Swap rows
            if pivot_row != col {
                a.swap(pivot_row, col);
                inv.swap(pivot_row, col);
            }

            // Scale pivot row
            let pivot_val = a[col][col];
            let inv_pivot = Complex::new(1.0, 0.0) / pivot_val;
            for j in 0..n {
                a[col][j] *= inv_pivot;
                inv[col][j] *= inv_pivot;
            }

            // Eliminate column entries in all other rows
            for row in 0..n {
                if row != col {
                    let factor = a[row][col];
                    for j in 0..n {
                        a[row][j] -= factor * a[col][j];
                        inv[row][j] -= factor * inv[col][j];
                    }
                }
            }
        }

        Some(Matrix { data: inv })
    }

    pub fn dagger(&self) -> Self {
        let mut dagger_data = [[Complex::new(0.0, 0.0); MATRIX_DIMENSION]; MATRIX_DIMENSION];

        for i in 0..MATRIX_DIMENSION {
            for j in 0..MATRIX_DIMENSION {
                dagger_data[j][i] = self.data[i][j].conj();
            }
        }

        Matrix { data: dagger_data }
    }

    pub fn transpose(&self) -> Self {
        let mut transposed_data = [[Complex::new(0.0, 0.0); MATRIX_DIMENSION]; MATRIX_DIMENSION];

        for i in 0..MATRIX_DIMENSION {
            for j in 0..MATRIX_DIMENSION {
                transposed_data[j][i] = self.data[i][j];
            }
        }
        Matrix {
            data: transposed_data,
        }
    }

    fn to_nalgebra(&self) -> SMatrix<Complex<f64>, MATRIX_DIMENSION, MATRIX_DIMENSION> {
        let mut data = [Complex::new(0.0, 0.0); MATRIX_DIMENSION * MATRIX_DIMENSION];
        for i in 0..MATRIX_DIMENSION {
            for j in 0..MATRIX_DIMENSION {
                data[i * MATRIX_DIMENSION + j] = self.data[i][j];
            }
        }
        SMatrix::from_row_slice(&data)
    }

    fn from_nalgebra(matrix: &SMatrix<Complex<f64>, MATRIX_DIMENSION, MATRIX_DIMENSION>) -> Self {
        let mut data = [[Complex::new(0.0, 0.0); MATRIX_DIMENSION]; MATRIX_DIMENSION];
        for i in 0..MATRIX_DIMENSION {
            for j in 0..MATRIX_DIMENSION {
                data[i][j] = matrix[(i, j)];
            }
        }
        Matrix { data }
    }

    /// QR decomposition using modified Gram-Schmidt orthogonalization.
    /// Returns (Q, R) where self = Q * R, Q is unitary and R is upper triangular.
    fn qr_decompose(&self) -> (Matrix, Matrix) {
        let n = MATRIX_DIMENSION;
        let zero = Complex::new(0.0, 0.0);
        let mut q_cols = [[zero; MATRIX_DIMENSION]; MATRIX_DIMENSION];
        let mut r = Matrix::zeros();

        for j in 0..n {
            // Start with j-th column of self
            let mut v = [zero; MATRIX_DIMENSION];
            for i in 0..n {
                v[i] = self.data[i][j];
            }

            // Orthogonalize against previous columns
            for k in 0..j {
                let mut dot = zero;
                for i in 0..n {
                    dot += q_cols[k][i].conj() * v[i];
                }
                r.data[k][j] = dot;
                for i in 0..n {
                    v[i] -= dot * q_cols[k][i];
                }
            }

            // Normalize
            let norm = v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
            r.data[j][j] = Complex::new(norm, 0.0);
            if norm > 1e-14 {
                for i in 0..n {
                    q_cols[j][i] = v[i] / norm;
                }
            } else {
                // Column is linearly dependent (e.g. shifted matrix is singular).
                // Complete Q to unitary by finding an orthogonal vector.
                // Try each standard basis vector, orthogonalize, pick the best.
                let mut best_norm = 0.0_f64;
                let mut best_v = [zero; MATRIX_DIMENSION];
                for trial in 0..n {
                    let mut candidate = [zero; MATRIX_DIMENSION];
                    candidate[trial] = Complex::new(1.0, 0.0);
                    for k in 0..j {
                        let mut dot = zero;
                        for i in 0..n {
                            dot += q_cols[k][i].conj() * candidate[i];
                        }
                        for i in 0..n {
                            candidate[i] -= dot * q_cols[k][i];
                        }
                    }
                    let cnorm = candidate.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
                    if cnorm > best_norm {
                        best_norm = cnorm;
                        best_v = candidate;
                    }
                }
                if best_norm > 1e-14 {
                    for i in 0..n {
                        q_cols[j][i] = best_v[i] / best_norm;
                    }
                }
            }
        }

        // Store columns in Q matrix
        let mut q = Matrix::zeros();
        for j in 0..n {
            for i in 0..n {
                q.data[i][j] = q_cols[j][i];
            }
        }

        (q, r)
    }

    /// Solves the linear system Ax = b using Gaussian elimination with partial pivoting.
    fn solve_linear_system(
        &self,
        b: &[Complex<f64>; MATRIX_DIMENSION],
    ) -> Option<[Complex<f64>; MATRIX_DIMENSION]> {
        let n = MATRIX_DIMENSION;
        let mut a = self.data;
        let mut rhs = *b;

        // Forward elimination with partial pivoting
        for col in 0..n {
            let mut max_norm = 0.0;
            let mut pivot_row = col;
            for row in col..n {
                let norm = a[row][col].norm();
                if norm > max_norm {
                    max_norm = norm;
                    pivot_row = row;
                }
            }

            if max_norm < 1e-30 {
                return None;
            }

            a.swap(col, pivot_row);
            rhs.swap(col, pivot_row);

            for row in (col + 1)..n {
                let factor = a[row][col] / a[col][col];
                for j in col..n {
                    a[row][j] -= factor * a[col][j];
                }
                rhs[row] -= factor * rhs[col];
            }
        }

        // Back substitution
        let mut x = [Complex::new(0.0, 0.0); MATRIX_DIMENSION];
        for i in (0..n).rev() {
            x[i] = rhs[i];
            for j in (i + 1)..n {
                x[i] -= a[i][j] * x[j];
            }
            x[i] /= a[i][i];
        }

        Some(x)
    }

    /// Computes the Schur decomposition using the QR algorithm with single shifts.
    /// Returns (T, Q) where T is upper triangular and Q is unitary,
    /// such that A = Q * T * Q^H.
    fn schur_decompose(&self) -> (Matrix, Matrix) {
        let matrix = self.to_nalgebra();
        let eps = 1e-12;
        let max_niter = 0;

        if let Some(schur) = Schur::try_new(matrix, eps, max_niter) {
            let (q, t) = schur.unpack();
            return (Self::from_nalgebra(&t), Self::from_nalgebra(&q));
        }

        self.schur_decompose_qr_fallback()
    }

    fn schur_decompose_qr_fallback(&self) -> (Matrix, Matrix) {
        let n = MATRIX_DIMENSION;
        let mut t = *self;
        let mut q_total = Matrix::identity();
        let max_iter = 10000;
        let tol = 1e-12;

        for _iter in 0..max_iter {
            // Single shift: use bottom-right diagonal element
            let shift = t.data[n - 1][n - 1];
            for i in 0..n {
                t.data[i][i] -= shift;
            }

            let (q, r) = t.qr_decompose();
            t = &r * &q;

            for i in 0..n {
                t.data[i][i] += shift;
            }

            // Accumulate the unitary transformation
            q_total = &q_total * &q;

            // Check convergence: all strictly lower triangular elements should vanish
            let mut max_lower = 0.0_f64;
            for i in 1..n {
                for j in 0..i {
                    max_lower = max_lower.max(t.data[i][j].norm());
                }
            }
            if max_lower < tol {
                break;
            }
        }

        (t, q_total)
    }

    /// Extracts eigenvectors from an upper triangular (Schur form) matrix T
    /// using back-substitution. Returns a matrix whose columns are the
    /// eigenvectors of T.
    fn eigenvectors_from_schur(t: &Matrix) -> Matrix {
        let n = MATRIX_DIMENSION;
        let zero = Complex::new(0.0, 0.0);
        let mut eigvecs = Matrix::zeros();

        for idx in 0..n {
            let lambda = t.data[idx][idx];
            let mut y = [zero; MATRIX_DIMENSION];
            y[idx] = Complex::new(1.0, 0.0);

            // Back-substitution for rows above idx:
            // From (T - λI)y = 0, row j gives:
            //   (T[j][j] - λ)*y[j] + Σ_{k=j+1..=idx} T[j][k]*y[k] = 0
            for j in (0..idx).rev() {
                let mut sum = zero;
                for k in (j + 1)..=idx {
                    sum += t.data[j][k] * y[k];
                }
                let diag = t.data[j][j] - lambda;
                if diag.norm() > 1e-10 {
                    y[j] = -sum / diag;
                }
                // If diag ≈ 0 (degenerate eigenvalue), y[j] stays 0
            }

            // Normalize
            let norm = y
                .iter()
                .map(|c: &Complex<f64>| c.norm_sqr())
                .sum::<f64>()
                .sqrt();
            if norm > 1e-30 {
                for c in y.iter_mut() {
                    *c /= norm;
                }
            }

            // Store as column idx
            for j in 0..n {
                eigvecs.data[j][idx] = y[j];
            }
        }

        eigvecs
    }

    /// Computes the eigenvalues of the matrix using the QR algorithm.
    ///
    /// # Returns
    /// An array of MATRIX_DIMENSION eigenvalues (may contain duplicates for
    /// degenerate cases).
    pub fn eigenvalues(&self) -> [Complex<f64>; MATRIX_DIMENSION] {
        let (t, _) = self.schur_decompose();
        let mut result = [Complex::new(0.0, 0.0); MATRIX_DIMENSION];
        for i in 0..MATRIX_DIMENSION {
            result[i] = t.data[i][i];
        }
        result
    }

    /// Computes the eigenvectors of the matrix.
    ///
    /// # Returns
    /// An array of MATRIX_DIMENSION normalized eigenvectors, where each vᵢ corresponds
    /// to the eigenvalue λᵢ returned by `eigenvalues()` (in the same order).
    pub fn eigenvectors(&self) -> [[Complex<f64>; MATRIX_DIMENSION]; MATRIX_DIMENSION] {
        let (t, q) = self.schur_decompose();
        let eigvecs_t = Self::eigenvectors_from_schur(&t);
        let eigvecs = &q * &eigvecs_t;
        // Convert to array format where result[i] is the i-th eigenvector (column i)
        let mut result = [[Complex::new(0.0, 0.0); MATRIX_DIMENSION]; MATRIX_DIMENSION];
        for i in 0..MATRIX_DIMENSION {
            for j in 0..MATRIX_DIMENSION {
                result[i][j] = eigvecs.data[j][i];
            }
        }
        result
    }

    /// Computes both eigenvalues and eigenvectors simultaneously.
    ///
    /// This is the recommended method for obtaining the complete eigendecomposition.
    ///
    /// # Returns
    /// A tuple `(eigenvalues, eigenvectors)` where:
    /// - `eigenvalues`: array of MATRIX_DIMENSION eigenvalues
    /// - `eigenvectors`: Matrix whose columns are the corresponding eigenvectors
    ///
    /// Each eigenvector is normalized such that |vᵢ| = 1.
    pub fn eigen(&self) -> ([Complex<f64>; MATRIX_DIMENSION], Matrix) {
        let (t, q) = self.schur_decompose();

        let mut eigenvalues = [Complex::new(0.0, 0.0); MATRIX_DIMENSION];
        for i in 0..MATRIX_DIMENSION {
            eigenvalues[i] = t.data[i][i];
        }

        let eigvecs_t = Self::eigenvectors_from_schur(&t);
        let eigenvectors = &q * &eigvecs_t;

        (eigenvalues, eigenvectors)
    }
}

impl Mul<&Matrix> for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: &Matrix) -> Matrix {
        let mut result_data = [[Complex::new(0.0, 0.0); MATRIX_DIMENSION]; MATRIX_DIMENSION];

        for i in 0..MATRIX_DIMENSION {
            for j in 0..MATRIX_DIMENSION {
                for k in 0..MATRIX_DIMENSION {
                    result_data[i][j] += self.data[i][k] * rhs.data[k][j];
                }
            }
        }

        Matrix { data: result_data }
    }
}

impl Mul<&[Complex<f64>; MATRIX_DIMENSION]> for &Matrix {
    type Output = [Complex<f64>; MATRIX_DIMENSION];

    fn mul(self, rhs: &[Complex<f64>; MATRIX_DIMENSION]) -> [Complex<f64>; MATRIX_DIMENSION] {
        let mut result = [Complex::new(0.0, 0.0); MATRIX_DIMENSION];

        for i in 0..MATRIX_DIMENSION {
            for j in 0..MATRIX_DIMENSION {
                result[i] += self.data[i][j] * rhs[j];
            }
        }
        result
    }
}

impl Mul<Complex<f64>> for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: Complex<f64>) -> Matrix {
        let mut result_data = [[Complex::new(0.0, 0.0); MATRIX_DIMENSION]; MATRIX_DIMENSION];

        for i in 0..MATRIX_DIMENSION {
            for j in 0..MATRIX_DIMENSION {
                result_data[i][j] = self.data[i][j] * rhs;
            }
        }

        Matrix { data: result_data }
    }
}

impl Add<&Matrix> for &Matrix {
    type Output = Matrix;

    fn add(self, rhs: &Matrix) -> Matrix {
        let mut result_data = [[Complex::new(0.0, 0.0); MATRIX_DIMENSION]; MATRIX_DIMENSION];

        for i in 0..MATRIX_DIMENSION {
            for j in 0..MATRIX_DIMENSION {
                result_data[i][j] = self.data[i][j] + rhs.data[i][j];
            }
        }

        Matrix { data: result_data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_reconstruction_error(h: &Matrix, evals: &[Complex<f64>; MATRIX_DIMENSION], v: &Matrix) -> f64 {
        let v_inv = v.inverse().expect("eigenvector matrix should be invertible");
        let mut d = Matrix::zeros();
        for i in 0..MATRIX_DIMENSION {
            d.data[i][i] = evals[i];
        }

        let recon = &(&(*v) * &d) * &v_inv;
        let mut max_err = 0.0_f64;
        for i in 0..MATRIX_DIMENSION {
            for j in 0..MATRIX_DIMENSION {
                max_err = max_err.max((recon.data[i][j] - h.data[i][j]).norm());
            }
        }
        max_err
    }

    #[test]
    fn eigenvalues_match_known_non_hermitian_case() {
        let h = Matrix::new([
            [
                Complex::new(0.0, -0.5),
                Complex::new(0.0, 0.0),
                Complex::new(-1.32, 0.0),
                Complex::new(-2.0, 0.0),
            ],
            [
                Complex::new(0.0, 0.0),
                Complex::new(0.0, -0.5),
                Complex::new(-2.0, 0.0),
                Complex::new(-1.32, 0.0),
            ],
            [
                Complex::new(-1.32, 0.0),
                Complex::new(-2.0, 0.0),
                Complex::new(0.0, -0.5),
                Complex::new(0.0, 0.0),
            ],
            [
                Complex::new(-2.0, 0.0),
                Complex::new(-1.32, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, -0.5),
            ],
        ]);

        let (evals, evecs) = h.eigen();

        let mut got = evals.to_vec();
        got.sort_by(|a, b| a.re.total_cmp(&b.re).then(a.im.total_cmp(&b.im)));

        let mut expected: Vec<Complex<f64>> = vec![
            Complex::new(-3.32, -0.5),
            Complex::new(-0.68, -0.5),
            Complex::new(0.68, -0.5),
            Complex::new(3.32, -0.5),
        ];
        expected.sort_by(|a, b| a.re.total_cmp(&b.re).then(a.im.total_cmp(&b.im)));

        for (actual, target) in got.iter().zip(expected.iter()) {
            assert!(
                (*actual - *target).norm() < 1e-8,
                "eigenvalue mismatch: got {:?}, expected {:?}",
                actual,
                target
            );
        }

        let recon_err = max_reconstruction_error(&h, &evals, &evecs);
        assert!(
            recon_err < 1e-8,
            "reconstruction error too large: {}",
            recon_err
        );
    }
}
