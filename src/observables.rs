use crate::{
    hilbert::{self, SystemSize},
    numerics::LatticeState,
};

/// Results of quantum measurements on the system.
///
/// Contains the occupation pattern and two-point correlations measured from a quantum state.
#[derive(Clone)]
pub struct MeasurementResult {
    /// Occupation pattern - 0 or 1 for each site
    pub occupation: Vec<Vec<hilbert::Occupation>>,
    /// Two-point correlations between all pairs of sites
    pub correlation: Vec<Vec<u8>>,
}

impl MeasurementResult {
    /// Creates a new MeasurementResult from occupation and correlation data.
    ///
    /// # Arguments
    /// * `occupation` - Occupation pattern for all sites
    /// * `correlation` - Two-point correlations for all site pairs
    /// * `total_magnetization` - Total magnetization value
    /// * `total_magnetization_squared` - Squared total magnetization value
    /// * `total_magnetization_fourth` - Fourth power of total magnetization value
    pub fn new(occupation: Vec<Vec<hilbert::Occupation>>, correlation: Vec<Vec<u8>>) -> Self {
        MeasurementResult {
            occupation,
            correlation,
        }
    }
}

/// Performs measurements on the quantum state, extracting occupation and correlations.
///
/// Calculates occupations and two-point correlations from the spatial configuration.
///
/// # Arguments
/// * `_psi` - The state vector
/// * `spatial_index` - The spatial configuration index
/// * `system_size` - System size (number of sites)
/// * `num_particles` - Number of particles
///
/// # Returns
/// * `MeasurementResult` - The measurement results containing occupations and correlations
pub fn perform_measurements(
    spatial_configuration: &LatticeState,
    system_size: SystemSize,
) -> MeasurementResult {
    let mut correlation =
        vec![vec![0u8; system_size.l_x * system_size.l_y]; system_size.l_x * system_size.l_y];

    let occupation = spatial_configuration.occupation_grid();
    let occupation_flat = occupation
        .iter()
        .flatten()
        .cloned()
        .collect::<Vec<hilbert::Occupation>>();

    for i in 0..(system_size.l_x * system_size.l_y) {
        for j in 0..(system_size.l_x * system_size.l_y) {
            correlation[i][j] = occupation_flat[i] as u8 * occupation_flat[j] as u8;
        }
    }

    MeasurementResult::new(occupation, correlation)
}

/// Returns the names of observable properties measured in the system.
///
/// # Returns
/// * `Vec<&'static str>` - Names of the observable properties
pub fn get_observable_names() -> Vec<&'static str> {
    vec!["Occupation", "Correlation"]
}
