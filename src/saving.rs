use crate::config;
use crate::numerics::MATRIX_DIMENSION;
use crate::time_evolution;
use fs2::FileExt;
use hdf5::Error;
use hdf5::File;
use hdf5::types::VarLenUnicode;
use std::fs::OpenOptions;
use std::path::Path;
use std::thread;
use std::time::Duration;

const MAX_RETRIES: u32 = 10;
const INITIAL_BACKOFF_MS: u64 = 100;
const MAX_BACKOFF_MS: u64 = 10000;

struct Saver<'a> {
    group: &'a hdf5::Group,
    trajectory_results: &'a Vec<time_evolution::TrajectoryResult>,
    measurement_times: &'a Vec<f64>,
    config: &'a config::Config,
}

impl Saver<'_> {
    fn save_measurement_times(&self) -> Result<(), Error> {
        let time_dataset = self
            .group
            .new_dataset::<f64>()
            .shape(self.config.numerics.measurements as usize)
            .create("measure_time")
            .unwrap();

        time_dataset.write(&self.measurement_times).unwrap();
        Ok(())
    }

    fn save_occupation(&self) -> Result<(), Error> {
        // Implementation for saving occupation data
        let occupation_dataset = self
            .group
            .new_dataset::<f64>()
            .shape((
                self.config.numerics.measurements as usize,
                self.config.physics.system_size.l_x,
                self.config.physics.system_size.l_y,
            ))
            .create("occupation")?;

        let mut occupation_data = ndarray::Array3::<f64>::zeros((
            self.config.numerics.measurements as usize,
            self.config.physics.system_size.l_x,
            self.config.physics.system_size.l_y,
        ));
        for trajec_result in self.trajectory_results.iter() {
            for (j, meas_result) in trajec_result.measurements.iter().enumerate() {
                for (x, row) in meas_result.occupation.iter().enumerate() {
                    for (y, occ) in row.iter().enumerate() {
                        occupation_data[[j, x, y]] +=
                            (*occ as u8) as f64 / self.config.numerics.num_trajectories as f64;
                    }
                }
            }
        }
        occupation_dataset.write(&occupation_data)?;
        Ok(())
    }

    fn save_correlation(&self) -> Result<(), Error> {
        let correlation_dataset = self
            .group
            .new_dataset::<f64>()
            .shape((
                self.config.numerics.measurements as usize,
                self.config.physics.system_size.l_x * self.config.physics.system_size.l_y,
                self.config.physics.system_size.l_x * self.config.physics.system_size.l_y,
            ))
            .create("correlation")?;

        let mut correlation_data = ndarray::Array3::<f64>::zeros((
            self.config.numerics.measurements as usize,
            self.config.physics.system_size.l_x * self.config.physics.system_size.l_y,
            self.config.physics.system_size.l_x * self.config.physics.system_size.l_y,
        ));

        for trajec_result in self.trajectory_results.iter() {
            for (j, meas_result) in trajec_result.measurements.iter().enumerate() {
                for l1 in
                    0..(self.config.physics.system_size.l_x * self.config.physics.system_size.l_y)
                {
                    for l2 in 0..(self.config.physics.system_size.l_x
                        * self.config.physics.system_size.l_y)
                    {
                        correlation_data[[j, l1, l2]] += meas_result.correlation[l1][l2] as f64
                            / self.config.numerics.num_trajectories as f64;
                    }
                }
            }
        }

        correlation_dataset.write(&correlation_data)?;
        Ok(())
    }

    fn save_spatial_configs(&self) -> Result<(), Error> {
        let end_spatial_conf_dataset = self
            .group
            .new_dataset::<usize>()
            .shape((
                self.config.numerics.num_trajectories as usize,
                self.config.physics.system_size.l_x,
                self.config.physics.system_size.l_y,
            ))
            .create("end_spatial_configuration")?;

        let end_spatial_conf_data = self
            .trajectory_results
            .iter()
            .map(|traj| {
                traj.end_spatial_configuration
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|&occ| occ.unwrap_or(usize::MAX)) // Use usize::MAX to represent None
                            .collect::<Vec<usize>>()
                    })
                    .collect::<Vec<Vec<usize>>>()
            })
            .collect::<Vec<Vec<Vec<usize>>>>();

        let end_spatial_conf_data = ndarray::Array3::<usize>::from_shape_vec(
            (
                self.config.numerics.num_trajectories as usize,
                self.config.physics.system_size.l_x,
                self.config.physics.system_size.l_y,
            ),
            end_spatial_conf_data
                .into_iter()
                .flatten()
                .into_iter()
                .flatten()
                .collect(),
        )
        .unwrap();

        // ndarray::Array1::<usize>::from_iter(
        //     self.trajectory_results
        //         .iter()
        //         .map(|traj| traj.end_spatial_configuration),
        // );

        end_spatial_conf_dataset.write(&end_spatial_conf_data)?;
        Ok(())
    }

    fn save_spin_states(&self) -> Result<(), Error> {
        let end_spin_state_dataset_real = self
            .group
            .new_dataset::<f64>()
            .shape((
                self.config.numerics.num_trajectories as usize,
                self.config.physics.num_particles,
                MATRIX_DIMENSION,
            ))
            .create("end_spin_state_real")?;
        let end_spin_state_dataset_imag = self
            .group
            .new_dataset::<f64>()
            .shape((
                self.config.numerics.num_trajectories as usize,
                self.config.physics.num_particles,
                MATRIX_DIMENSION,
            ))
            .create("end_spin_state_imag")?;

        let mut end_spin_state_data = ndarray::Array3::<num_complex::Complex<f64>>::zeros((
            self.config.numerics.num_trajectories as usize,
            self.config.physics.num_particles,
            MATRIX_DIMENSION,
        ));
        for (i, traj) in self.trajectory_results.iter().enumerate() {
            if let Some(end_spin_state) = &traj.end_spin_state {
                for (j, amp) in end_spin_state.iter().enumerate() {
                    for k in 0..MATRIX_DIMENSION {
                        end_spin_state_data[[i, j, k]] = amp[k];
                    }
                }
            }
        }

        end_spin_state_dataset_real.write(&end_spin_state_data.mapv(|c| c.re))?;
        end_spin_state_dataset_imag.write(&end_spin_state_data.mapv(|c| c.im))?;
        Ok(())
    }

    fn save_jump_times(&self) -> Result<(), Error> {
        let max_jump_times_length = self
            .trajectory_results
            .iter()
            .map(|traj| traj.jump_times.len())
            .max()
            .unwrap_or(0);

        let jump_times_dataset = self
            .group
            .new_dataset::<f64>()
            .shape((
                self.config.numerics.num_trajectories as usize,
                max_jump_times_length,
            ))
            .create("jump_times")?;

        let mut jump_times_data = ndarray::Array2::<f64>::ones((
            self.config.numerics.num_trajectories as usize,
            max_jump_times_length,
        )) * (-1.0);
        for (i, traj) in self.trajectory_results.iter().enumerate() {
            for (j, jump_time) in traj.jump_times.iter().enumerate() {
                jump_times_data[[i, j]] = *jump_time;
            }
        }
        jump_times_dataset.write(&jump_times_data)?;
        Ok(())
    }

    /// Store every parameter needed to reproduce the run as group attributes,
    /// so each result group is fully self-describing: the complete config
    /// (physics + numerics, with the seed already resolved to the concrete
    /// value used) plus the version of the code that produced it.
    ///
    /// Note on reproducibility: bit-identical trajectories additionally require
    /// the same `rand` crate version (`StdRng` output is not stable across major
    /// versions), which is pinned by `Cargo.lock`.
    fn save_metadata_attributes(&self) -> Result<(), Error> {
        let physics = &self.config.physics;
        let numerics = &self.config.numerics;

        // --- Physics ---
        self.write_u64_attr("l_x", physics.system_size.l_x as u64)?;
        self.write_u64_attr("l_y", physics.system_size.l_y as u64)?;
        self.write_u64_attr("num_particles", physics.num_particles as u64)?;
        self.group
            .new_attr::<f64>()
            .create("transverse_field")?
            .write_scalar(&physics.transverse_field)?;
        self.group
            .new_attr::<f64>()
            .create("max_time")?
            .write_scalar(&physics.max_time)?;
        self.group
            .new_attr::<f64>()
            .shape(physics.rates.len())
            .create("rates")?
            .write(&physics.rates)?;
        self.group
            .new_attr::<f64>()
            .shape(physics.rates.len())
            .create("normalized_rates")?
            .write(&physics.normalized_rates())?;
        self.group
            .new_attr::<f64>()
            .create("lateral_rate")?
            .write_scalar(&physics.lateral_rate)?;

        // --- Numerics ---
        self.group
            .new_attr::<f64>()
            .create("time_step")?
            .write_scalar(&numerics.time_step)?;
        self.write_u64_attr("num_trajectories", numerics.num_trajectories as u64)?;
        self.write_u64_attr("measurements", numerics.measurements as u64)?;
        self.write_u64_attr("seed", numerics.seed)?;
        self.write_u64_attr(
            "num_random_init_states",
            numerics.num_random_init_states as u64,
        )?;

        // --- Code version ---
        self.write_string_attr("code_version", env!("CARGO_PKG_VERSION"))?;
        self.write_string_attr("git_commit", env!("GIT_COMMIT_HASH"))?;

        Ok(())
    }

    fn write_u64_attr(&self, name: &str, value: u64) -> Result<(), Error> {
        self.group
            .new_attr::<u64>()
            .create(name)?
            .write_scalar(&value)?;
        Ok(())
    }

    fn write_string_attr(&self, name: &str, value: &str) -> Result<(), Error> {
        let value: VarLenUnicode = value
            .parse()
            .expect("attribute string should be valid unicode");
        self.group
            .new_attr::<VarLenUnicode>()
            .create(name)?
            .write_scalar(&value)?;
        Ok(())
    }
}

/// Checks if a group with the given name exists in an HDF5 file.
///
/// Returns Ok(true) if the group exists, Ok(false) if it doesn't,
/// or Err if the file can't be opened.
///
/// # Arguments
/// * `output_path` - Path to the HDF5 file
/// * `group_name` - Name of the group to check for
///
/// # Returns
/// * `Result<bool, Box<dyn std::error::Error>>` - Whether the group exists
pub fn group_exists(
    output_path: &str,
    group_name: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    let file = match File::open(output_path) {
        Ok(f) => f,
        Err(_) => return Ok(false),
    };

    let exists = file
        .member_names()
        .ok()
        .map(|names| names.contains(&group_name.to_string()))
        .unwrap_or(false);

    Ok(exists)
}

/// Saves simulation results from multiple trajectories to an HDF5 file.
///
/// Aggregates results from all trajectories and saves occupations and correlations
/// as HDF5 datasets within a named group. Implements file locking to handle concurrent access.
///
/// # Arguments
/// * `trajectory_results` - Results from all individual trajectories
/// * `measurement_times` - Times at which measurements were performed
/// * `config` - Configuration parameters used in the simulation
/// * `output_path` - Path to the output HDF5 file
/// * `group_name` - Name of the HDF5 group to create
///
/// # Returns
/// * `Result<(), Box<dyn std::error::Error>>` - Success or error
pub fn save_results(
    trajectory_results: &Vec<time_evolution::TrajectoryResult>,
    measurement_times: &Vec<f64>,
    config: &config::Config,
    output_path: &str,
    group_name: &str,
    save_end_states: bool,
    save_end_spatial: bool,
    save_jump_times: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Acquire file lock with retry logic
    let lock_file_path = format!("{}.lock", output_path);
    let mut retry_count = 0;

    let lock_file = loop {
        match OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_file_path)
        {
            Ok(f) => {
                match f.try_lock_exclusive() {
                    Ok(()) => {
                        break f;
                    }
                    Err(_) => {
                        // Lock is held by another process, retry
                        retry_count += 1;
                        if retry_count >= MAX_RETRIES {
                            return Err(format!(
                                "Failed to acquire lock on {} after {} retries",
                                output_path, MAX_RETRIES
                            )
                            .into());
                        }
                        let backoff_ms = std::cmp::min(
                            INITIAL_BACKOFF_MS * 2_u64.pow(retry_count - 1),
                            MAX_BACKOFF_MS,
                        );
                        thread::sleep(Duration::from_millis(backoff_ms));
                    }
                }
            }
            Err(e) => {
                return Err(format!("Failed to open lock file: {}", e).into());
            }
        }
    };

    let result = {
        let file = if Path::new(output_path).exists() {
            File::open_rw(output_path)?
        } else {
            File::create(output_path)?
        };

        let group = file.create_group(group_name)?;

        let saver = Saver {
            group: &group,
            trajectory_results,
            measurement_times,
            config,
        };

        if save_jump_times {
            saver.save_jump_times()?;
        }

        saver.save_metadata_attributes()?;
        saver.save_measurement_times()?;
        saver.save_occupation()?;
        saver.save_correlation()?;

        if save_end_spatial || save_end_states {
            saver.save_spatial_configs()?;
        }

        if save_end_states {
            saver.save_spin_states()?;
        }

        file.close().unwrap();

        Ok(())
    };

    drop(lock_file);

    let _ = std::fs::remove_file(&lock_file_path);

    result
}
