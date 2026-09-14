use crate::numerics::{Direction, MATRIX_DIMENSION, SpinWavefunction};
use crate::{config, hilbert, numerics, observables};
use num_complex::Complex;
use rand::Rng;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;

/// Results from a single quantum trajectory simulation.
///
/// Contains a sequence of measurements taken at specified time points
/// throughout the evolution of a single quantum trajectory.
#[derive(Default)]
pub struct TrajectoryResult {
    /// Measurements taken at each time point in the simulation
    pub measurements: Vec<observables::MeasurementResult>,
    pub end_spatial_configuration: Vec<Vec<Option<usize>>>,
    pub end_spin_state: Option<numerics::SpinWavefunction>,
    pub jump_times: Vec<f64>,
}

impl TrajectoryResult {
    /// Creates a new empty TrajectoryResult with preallocated capacity.
    ///
    /// # Arguments
    /// * `capacity` - Number of measurement time points to preallocate
    pub fn with_capacity(capacity: usize) -> Self {
        let measurements: Vec<observables::MeasurementResult> = Vec::with_capacity(capacity);

        TrajectoryResult {
            measurements,
            ..Default::default()
        }
    }
}

/// Number of jump channels per particle: `MATRIX_DIMENSION` forward
/// channels (one per internal direction) plus `2 * MATRIX_DIMENSION`
/// lateral ones (two perpendicular displacements per internal direction).
pub const NUM_JUMP_CHANNELS: usize = 3 * MATRIX_DIMENSION;

/// The `(internal direction sigma, displacement mu)` label of every jump
/// channel, in channel-index order. All indices are `Direction`
/// discriminants (`L=0, R=1, U=2, D=3`).
///
/// Channels `0..MATRIX_DIMENSION` are the forward channels
/// `M_{r,sigma} = sqrt(gamma_sigma) b^dag_{r+e_sigma,sigma} b_{r,sigma}`
/// (`mu == sigma`); channels `MATRIX_DIMENSION..NUM_JUMP_CHANNELS` are the
/// lateral channels
/// `M_{r,sigma,mu} = sqrt(eps) b^dag_{r+e_mu,sigma} b_{r,sigma}`, grouped by
/// `sigma`, each with its two perpendicular displacements
/// (`Direction::perpendicular`). Every operator is **rank-1** in the
/// internal space -- its internal action is the projector `P_sigma` -- so
/// the post-jump state is `|sigma>` in every case and only the displacement
/// distinguishes the two lateral channels of a given `sigma`. See
/// `quantum_lateral_Heff_derivation.md` §2/§4/§7.
const CHANNEL_LABELS: [(usize, usize); NUM_JUMP_CHANNELS] = [
    // Forward: displacement == internal direction.
    (0, 0), // L-particle hops -x
    (1, 1), // R-particle hops +x
    (2, 2), // U-particle hops +y
    (3, 3), // D-particle hops -y
    // Lateral: displacement perpendicular to the internal direction.
    (0, 2), // L-particle hops +y
    (0, 3), // L-particle hops -y
    (1, 2), // R-particle hops +y
    (1, 3), // R-particle hops -y
    (2, 0), // U-particle hops -x
    (2, 1), // U-particle hops +x
    (3, 0), // D-particle hops -x
    (3, 1), // D-particle hops +x
];

/// Per-particle, unnormalized jump-channel weights: `NUM_JUMP_CHANNELS`
/// entries per particle, laid out as `CHANNEL_LABELS`.
///
/// Because every jump operator is rank-1 (§2 of the derivation note), every
/// weight has the *same* form -- the expectation value of
/// `M^dag M = rate * Theta_{r+e_mu} n_{r,sigma}`:
///
/// ```text
/// w[(sigma, mu)] = mask[mu] * rate * |psi_sigma|^2
/// ```
///
/// with `rate = rates[sigma]` for the forward channels and
/// `rate = lateral_rate` for the lateral ones.
///
/// With `lateral_rate == 0.0` the lateral entries are exactly `0.0`, so a
/// `WeightedIndex` built from the flattened weights partitions channels
/// `0..MATRIX_DIMENSION` identically to the forward-only (pre-lateral)
/// model -- inserting exact zeros changes neither the partial sums nor the
/// selected index, keeping eps=0 trajectories bit-identical.
fn jump_channel_weights(
    psi: &SpinWavefunction,
    masks: &[[bool; MATRIX_DIMENSION]],
    rates: [f64; MATRIX_DIMENSION],
    lateral_rate: f64,
) -> Vec<[f64; NUM_JUMP_CHANNELS]> {
    masks
        .iter()
        .enumerate()
        .map(|(i, mask)| {
            let psi_i = &psi[i];
            let mut weights = [0.0; NUM_JUMP_CHANNELS];
            for (channel, &(spin, move_dir)) in CHANNEL_LABELS.iter().enumerate() {
                let rate = if channel < MATRIX_DIMENSION {
                    rates[spin]
                } else {
                    lateral_rate
                };
                weights[channel] =
                    mask[move_dir] as u32 as f64 * rate * psi_i[spin].norm_sqr();
            }
            weights
        })
        .collect()
}

pub fn linear_trajectory_evolution(
    initial_state: &numerics::LatticeState,
    initial_spins: &numerics::SpinWavefunction,
    measurement_times: &Vec<f64>,
    config: &config::Config,
    rng: &mut rand::rngs::StdRng,
    save_end_states: bool,
    save_end_spatial: bool,
) -> TrajectoryResult {
    let rates = config.physics.normalized_rates();
    let lateral_rate = config.physics.lateral_rate;
    let (eigenvalues, eigenvectors) = hilbert::get_eigenvalues_and_vectors(
        config.physics.transverse_field,
        rates,
        lateral_rate,
    );
    let inverse_eigenvectors: [numerics::Matrix; 16] = eigenvectors
        .iter()
        .map(|evec| evec.inverse().unwrap())
        .collect::<Vec<_>>() // Collect to Vec first
        .try_into() // Try to convert Vec to [T; 4]
        .expect("Iterator did not yield exactly 16 elements");

    // for i in 0..4 {
    //     inverse_eigenvectors[i] = eigenvectors[i].inverse().unwrap();
    // }

    let mut trajectory_results = TrajectoryResult::with_capacity(measurement_times.len());

    let system_size = config.physics.system_size;
    let num_particles = config.physics.num_particles;

    let mut current_spatial = initial_state.clone();
    let mut current_psi = initial_spins.clone();

    let mut current_time = 0.0;

    let normalized_psi = current_psi.normalized_clone();
    let meas_result = observables::perform_measurements(&current_spatial, system_size);

    trajectory_results.measurements.push(meas_result);

    let mut hamilton_indices = current_spatial.hamiltonian_indices();
    let mut jump_probability_mask = current_spatial.jump_masks();

    let mut r = (*rng).random::<f64>();
    // r = 0.3801957350196178;
    // let mut counter = 0;
    for i in 0..measurement_times.len() - 1 {
        let t_start_interval = measurement_times[i];
        let t_next_measure = measurement_times[i + 1];
        current_time = t_start_interval;

        let mut dt = config.numerics.time_step;
        let mut too_large_dt = -1.0;

        'time_loop: while current_time < t_next_measure {
            // panic!("Debug exit before time evolution step");
            let psi_ham_basis = SpinWavefunction::new(
                (0..num_particles)
                    .map(|n| &(inverse_eigenvectors[hamilton_indices[n]]) * &current_psi[n])
                    .collect(),
            );

            // println!("Psi in ham basis: {:?}", psi_ham_basis);

            let actual_dt = dt.min(t_next_measure - current_time);

            let psi_new = SpinWavefunction::new(
                (0..num_particles)
                    .map(|n| {
                        let ev = eigenvalues[hamilton_indices[n]];
                        let phase_factors = [
                            (-ev[0] * Complex::i() * actual_dt).exp(),
                            (-ev[1] * Complex::i() * actual_dt).exp(),
                            (-ev[2] * Complex::i() * actual_dt).exp(),
                            (-ev[3] * Complex::i() * actual_dt).exp(),
                        ];
                        let help_state = [
                            phase_factors[0] * &psi_ham_basis[n][0],
                            phase_factors[1] * &psi_ham_basis[n][1],
                            phase_factors[2] * &psi_ham_basis[n][2],
                            phase_factors[3] * &psi_ham_basis[n][3],
                        ];
                        &eigenvectors[hamilton_indices[n]] * &(help_state)
                    })
                    .collect(),
            );

            let norm_squared: f64 = psi_new.norm_squared();

            // println!(
            //     "t_current: {}, dt: {}, norm_sq: {}, r: {}",
            //     current_time, actual_dt, norm_squared, r
            // );

            // counter += 1;
            // if counter > 50 {
            //     panic!("Debug exit after 30 steps");
            // }

            if norm_squared < r {
                // Jump occured before t_current + dt
                dt = actual_dt / 2.0;
                too_large_dt = actual_dt;
                continue 'time_loop;
            } else if (norm_squared - r).abs() < 1e-6 {
                // println!("Jump time: {}", current_time + actual_dt);

                let mut normalized_psi = psi_new.normalized_clone();

                // 12 weighted channels per particle: 0-3 forward, 4-11
                // rank-1 lateral (see `jump_channel_weights` /
                // `CHANNEL_LABELS`). With `lateral_rate == 0.0` channels 4-11
                // are exactly zero-width, so the WeightedIndex boundaries for
                // channels 0-3 (and the single RNG draw below) are unchanged,
                // keeping ε=0 trajectories bit-identical to the forward-only
                // model.
                let probs = jump_channel_weights(
                    &normalized_psi,
                    &jump_probability_mask,
                    rates,
                    lateral_rate,
                );

                let mut jump_probs_flat: Vec<f64> =
                    probs.iter().flat_map(|pair| pair.iter()).cloned().collect();
                jump_probs_flat = jump_probs_flat
                    .iter()
                    .map(|&p| p / jump_probs_flat.iter().sum::<f64>())
                    .collect();

                // println!("Jump probabilities: {:?}", jump_probs_flat);

                let dist = WeightedIndex::new(&jump_probs_flat).unwrap();

                let jump_index = dist.sample(rng);
                let jump_particle_index = jump_index / NUM_JUMP_CHANNELS;
                let jump_channel_index = jump_index % NUM_JUMP_CHANNELS;
                let (spin_index, move_index) = CHANNEL_LABELS[jump_channel_index];
                let jump_spin = Direction::ALL[spin_index];
                let jump_direction = Direction::ALL[move_index];

                let particle_position = current_spatial
                    .position_of_particle(jump_particle_index)
                    .unwrap();

                // Every channel is rank-1 in the internal space: collapse the
                // particle's spin onto `jump_spin` (which equals the
                // displacement for a forward jump and is perpendicular to it
                // for a lateral one), then update the spatial configuration.
                normalized_psi.collapse_at_index(jump_particle_index, jump_spin);
                current_psi = normalized_psi.normalized_clone();

                current_spatial.move_particle(
                    particle_position.0,
                    particle_position.1,
                    jump_direction,
                );

                // Update Hamiltonian indices and jump masks after the move
                hamilton_indices = current_spatial.hamiltonian_indices();
                jump_probability_mask = current_spatial.jump_masks();

                trajectory_results.jump_times.push(current_time + actual_dt);

                too_large_dt = -1.0;
                r = (*rng).random::<f64>();
                dt = config.numerics.time_step;
            } else {
                // No jump occured yet
                current_psi = psi_new;

                if too_large_dt > 1e-14 {
                    dt = (too_large_dt - actual_dt).abs() / 2.0;
                    too_large_dt -= actual_dt;
                } else {
                    dt = config.numerics.time_step;
                }
            }

            current_time += actual_dt;
        }

        // Perform measurement at t_next_measure on NORMALIZED state
        let normalized_psi = current_psi.normalized_clone();
        let meas_result = observables::perform_measurements(&current_spatial, system_size);

        trajectory_results.measurements.push(meas_result);
    }

    if save_end_states {
        trajectory_results.end_spatial_configuration = current_spatial.grid;
        trajectory_results.end_spin_state = Some(current_psi);
    } else if save_end_spatial {
        trajectory_results.end_spatial_configuration = current_spatial.grid;
    }

    trajectory_results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerics::{RIGHT_STATE, UP_STATE};

    const ALL_EMPTY: [bool; MATRIX_DIMENSION] = [true, true, true, true];

    /// Sanity check (task item e): a single particle prepared in the pure
    /// |R> direction eigenstate, on an empty lattice (all neighbor masks
    /// true), with ε > 0. Every channel weight is `mask[mu] * rate *
    /// |psi_sigma|^2`, and |R>'s only nonzero component is R, so the only
    /// live channels are those with `sigma = R`:
    /// - the forward channel gives +x (Right) only -- every other forward
    ///   weight is exactly zero because `|psi_sigma|^2 = 0`;
    /// - the two lateral channels of the R component (displacements ±y) are
    ///   strictly positive;
    /// - every lateral channel belonging to another `sigma` -- including
    ///   both of the ±x lateral channels, which belong to U and D -- is
    ///   exactly zero.
    ///
    /// So a pure |R> particle can only ever move +x (forward) or ±y
    /// (lateral); it can never move -x, and never moves ±x via a lateral
    /// jump.
    #[test]
    fn pure_right_state_has_no_leftward_or_lateral_x_weight() {
        let psi = SpinWavefunction::new(vec![RIGHT_STATE]);
        let masks = vec![ALL_EMPTY];
        let rates = [1.0, 1.0, 1.0, 1.0];
        let lateral_rate = 0.5;

        let weights = jump_channel_weights(&psi, &masks, rates, lateral_rate);
        let w = weights[0];

        for (channel, &(spin, move_dir)) in CHANNEL_LABELS.iter().enumerate() {
            let is_r_channel = spin == usize::from(Direction::Right);
            if is_r_channel {
                assert!(
                    w[channel] > 0.0,
                    "channel {} (sigma=R, mu={:?}) must be positive for pure |R>",
                    channel,
                    Direction::ALL[move_dir]
                );
            } else {
                assert_eq!(
                    w[channel], 0.0,
                    "channel {} (sigma={:?}, mu={:?}) must be zero for pure |R>",
                    channel,
                    Direction::ALL[spin],
                    Direction::ALL[move_dir]
                );
            }
        }

        // Spelled out for the displacements that matter physically: the only
        // moves available to |R> are +x (forward) and ±y (lateral).
        let live: Vec<Direction> = CHANNEL_LABELS
            .iter()
            .enumerate()
            .filter(|(c, _)| w[*c] > 0.0)
            .map(|(_, &(_, move_dir))| Direction::ALL[move_dir])
            .collect();
        assert_eq!(live, vec![Direction::Right, Direction::Up, Direction::Down]);
    }

    /// The lateral channel table must agree with `Direction::perpendicular`:
    /// each `sigma` contributes exactly two lateral channels, and their
    /// displacements are the two directions perpendicular to `sigma`.
    #[test]
    fn channel_table_matches_perpendicular_directions() {
        for (channel, &(spin, move_dir)) in
            CHANNEL_LABELS[..MATRIX_DIMENSION].iter().enumerate()
        {
            assert_eq!(spin, channel, "forward channels are indexed by sigma");
            assert_eq!(move_dir, spin, "a forward jump moves along sigma");
        }

        for spin in Direction::ALL {
            let displacements: Vec<Direction> = CHANNEL_LABELS[MATRIX_DIMENSION..]
                .iter()
                .filter(|&&(s, _)| s == usize::from(spin))
                .map(|&(_, mu)| Direction::ALL[mu])
                .collect();
            assert_eq!(
                displacements,
                spin.perpendicular().to_vec(),
                "lateral displacements for sigma={:?}",
                spin
            );
        }
    }

    #[test]
    fn lateral_rate_zero_gives_exactly_zero_weight_lateral_channels() {
        // Any state, any masks: lateral_rate == 0.0 must zero out channels
        // 4-11 exactly (bit-for-bit), which is what the ε=0 bit-identity
        // guarantee (see module docs and `quantum_lateral_Heff_derivation.md`
        // §7) rests on.
        let psi = SpinWavefunction::new(vec![[
            Complex::new(0.3, 0.1),
            Complex::new(0.2, -0.4),
            Complex::new(-0.1, 0.5),
            Complex::new(0.6, -0.2),
        ]]);
        let masks = vec![ALL_EMPTY];
        let rates = [0.7, 1.3, 0.9, 1.1];

        let weights = jump_channel_weights(&psi, &masks, rates, 0.0);
        for &lateral_weight in &weights[0][MATRIX_DIMENSION..NUM_JUMP_CHANNELS] {
            assert_eq!(lateral_weight, 0.0);
        }
    }

    #[test]
    fn masked_neighbor_zeroes_the_corresponding_channel() {
        // If the neighbor in a given lab direction is occupied (mask false),
        // *every* channel that would move the particle in that direction --
        // the forward one and both lateral ones -- must be zero, regardless
        // of state; all other channels stay positive.
        let psi = SpinWavefunction::new(vec![[
            Complex::new(0.5, 0.0),
            Complex::new(0.5, 0.0),
            Complex::new(0.5, 0.0),
            Complex::new(0.5, 0.0),
        ]]);
        // Right neighbor occupied; all others empty.
        let masks = vec![[true, false, true, true]];
        let rates = [1.0, 1.0, 1.0, 1.0];
        let lateral_rate = 1.0;

        let weights = jump_channel_weights(&psi, &masks, rates, lateral_rate);
        let w = weights[0];

        let blocked = usize::from(Direction::Right);
        let mut blocked_channels = 0;
        for (channel, &(_, move_dir)) in CHANNEL_LABELS.iter().enumerate() {
            if move_dir == blocked {
                blocked_channels += 1;
                assert_eq!(
                    w[channel], 0.0,
                    "channel {} moves Right, which is blocked",
                    channel
                );
            } else {
                assert!(w[channel] > 0.0, "channel {} should be open", channel);
            }
        }
        // One forward channel (sigma=R) plus the two lateral ones that
        // displace +x (sigma=U and sigma=D).
        assert_eq!(blocked_channels, 3);
    }

    /// Golden ε=0 regression (task item d): a small, fully deterministic
    /// two-particle trajectory computed with `lateral_rate = 0.0` must keep
    /// producing exactly these jump times and this end-of-run occupation
    /// pattern. These values were captured once from this exact test
    /// (post-lateral-jump-implementation, with `lateral_rate` explicitly
    /// zero) after independently verifying, via the compiled CLI binary
    /// against a golden HDF5 baseline recorded from the pristine
    /// (pre-lateral-jump) code, that ε=0 trajectories are bit-identical to
    /// the original forward-only model (see the task's manual verification
    /// procedure: run both binaries on the same config/seed with
    /// `--save-end-spatial --save-jump-times` and diff the HDF5 output). If
    /// this test ever fails, the ε=0 code path has regressed.
    #[test]
    fn epsilon_zero_golden_regression() {
        use crate::hilbert;
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let l = 5;
        let system_size = hilbert::SystemSize::new(l, l);
        let mut occ =
            vec![vec![hilbert::Occupation::Empty; system_size.l_y]; system_size.l_x];
        occ[1][1] = hilbert::Occupation::Occupied;
        occ[3][2] = hilbert::Occupation::Occupied;
        let num_particles = 2;
        let initial_state = numerics::LatticeState::new(system_size, &occ, num_particles);

        let initial_spins = SpinWavefunction::new(vec![RIGHT_STATE, UP_STATE]);

        let cfg = config::Config {
            physics: config::PhysicsConfig {
                system_size,
                num_particles,
                transverse_field: 0.7,
                max_time: 4.0,
                rates: [1.0, 1.0, 1.0, 1.0],
                lateral_rate: 0.0,
            },
            numerics: config::NumericsConfig {
                time_step: 0.05,
                num_trajectories: 1,
                measurements: 5,
                seed: 7,
                num_random_init_states: 0,
            },
        };

        let measurement_times: Vec<f64> = (0..cfg.numerics.measurements)
            .map(|i| (i as f64) * cfg.physics.max_time / (cfg.numerics.measurements - 1) as f64)
            .collect();

        let mut rng = StdRng::seed_from_u64(cfg.numerics.seed);
        let result = linear_trajectory_evolution(
            &initial_state,
            &initial_spins,
            &measurement_times,
            &cfg,
            &mut rng,
            false,
            true,
        );

        let expected_jump_times: Vec<f64> =
            vec![1.7480102539062505, 2.7217163085937477, 3.3717788696289053];
        assert_eq!(
            result.jump_times.len(),
            expected_jump_times.len(),
            "number of jumps changed for the ε=0 golden case -- got {:?}",
            result.jump_times
        );
        for (got, expected) in result.jump_times.iter().zip(expected_jump_times.iter()) {
            assert_eq!(got, expected, "jump time mismatch: got {:?}, expected {:?}", result.jump_times, expected_jump_times);
        }

        let occupied: Vec<(usize, usize)> = result
            .end_spatial_configuration
            .iter()
            .enumerate()
            .flat_map(|(x, row)| {
                row.iter()
                    .enumerate()
                    .filter(|(_, c)| c.is_some())
                    .map(move |(y, _)| (x, y))
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(occupied.len(), num_particles, "particle count not conserved");
        assert_eq!(occupied, vec![(0usize, 1usize), (2usize, 1usize)]);
    }
}
