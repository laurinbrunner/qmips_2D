use crate::hilbert::SystemSize;
use clap::Parser;
use serde::Deserialize;

/// Command line arguments for the quantum trajectories simulation.
///
/// This struct holds the paths to the configuration file and output file
/// passed as command line arguments.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct CommandLineArgs {
    /// Path to configuration toml file
    #[arg(long)]
    pub config_path: String,

    /// Path to h5 output file
    #[arg(long)]
    pub output_path: String,

    /// Print verbose information for each trajectory
    #[arg(short, long)]
    pub verbose: bool,

    /// Include entire end state per trajectory in output file
    #[arg(long)]
    pub save_end_states: bool,

    /// Include spatial end state per trajectory in output file
    #[arg(long)]
    pub save_end_spatial: bool,

    /// Include all times of quantum jumps per trajectory in output file
    #[arg(long)]
    pub save_jump_times: bool,
}

/// Complete simulation configuration combining physics, numerics, and initial state parameters.
#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub physics: PhysicsConfig,
    pub numerics: NumericsConfig,
}

/// Default jump rates: isotropic, one per direction.
///
/// Order matches the `Direction` discriminants: `[left, right, up, down]`.
fn default_rates() -> [f64; 4] {
    [1.0; 4]
}

/// Physical parameters of the quantum system.
#[derive(Debug, Deserialize, Copy, Clone)]
pub struct PhysicsConfig {
    pub system_size: SystemSize,
    pub num_particles: usize,
    pub transverse_field: f64,
    pub max_time: f64,
    /// Jump-operator rates per direction, order `[left, right, up, down]`
    /// matching the `Direction` discriminants.
    ///
    /// Only the ratios between the rates are physical: internally the rates are
    /// normalized by their mean (see [`PhysicsConfig::normalized_rates`]), so
    /// that time and the transverse field `transverse_field` are expressed in
    /// units of the mean rate. `[1, 1, 1, 1]` reproduces the isotropic model.
    #[serde(default = "default_rates")]
    pub rates: [f64; 4],
    /// Rate `ε` of the lateral (orthogonal) jump channels, in units of the
    /// mean forward hop rate.
    ///
    /// Each internal direction `σ` gets two such channels, one per lab
    /// direction `μ` perpendicular to `σ`, each with operator
    /// `M_{r,σ,μ} = √ε b†_{r+ê_μ,σ} b_{r,σ}` (rank-1: the post-jump internal
    /// state is `|σ⟩`). See `quantum_lateral_Heff_derivation.md`.
    ///
    /// Unlike `rates`, `ε` is *not* normalized together with the forward
    /// rates and is *not* multiplied by any anisotropy factor: it enters
    /// `H_eff` and the jump weights directly. `ε = 0` (the default, so that
    /// pre-existing configs without this field still load) reproduces the
    /// original forward-only model exactly, including bit-identical
    /// trajectories.
    #[serde(default)]
    pub lateral_rate: f64,
}

impl PhysicsConfig {
    /// Jump rates divided by their mean.
    ///
    /// This fixes the time unit to be the inverse mean rate, so that the
    /// transverse field and `max_time` are measured in units of the mean rate.
    /// In the isotropic case all four entries are `1.0`, recovering the
    /// original single-rate model exactly.
    pub fn normalized_rates(&self) -> [f64; 4] {
        let mean = self.rates.iter().sum::<f64>() / self.rates.len() as f64;
        assert!(
            mean > 0.0,
            "The mean jump rate must be positive, got rates {:?}",
            self.rates
        );
        assert!(
            self.lateral_rate.is_finite() && self.lateral_rate >= 0.0,
            "lateral_rate must be finite and non-negative, got {}",
            self.lateral_rate
        );
        self.rates.map(|r| r / mean)
    }
}

/// Numerical parameters for the simulation.
#[derive(Debug, Deserialize, Copy, Clone)]
pub struct NumericsConfig {
    pub time_step: f64,
    pub num_trajectories: u32,
    pub measurements: u32,
    pub seed: u64,
    #[serde(default)]
    pub num_random_init_states: usize,
}

impl std::fmt::Display for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let markers = "-".repeat(30);
        writeln!(f, "{}", "-".repeat(30))?;
        writeln!(f, "{}", self.physics)?;
        writeln!(f, "{}", self.numerics)?;
        write!(f, "{}", markers)
    }
}

impl std::fmt::Display for PhysicsConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Physical Parameters:")?;
        writeln!(f, "  System Size: {:?}", self.system_size)?;
        writeln!(f, "  Number particles: {}", self.num_particles)?;
        writeln!(
            f,
            "  Transverse field strength g: {}",
            self.transverse_field
        )?;
        writeln!(
            f,
            "  Jump rates [left, right, up, down]: {:?} (mean-normalized: {:?})",
            self.rates,
            self.normalized_rates()
        )?;
        writeln!(
            f,
            "  Lateral rate epsilon (units of mean forward rate): {}",
            self.lateral_rate
        )?;
        writeln!(f, "  Maximum time T_max: {}", self.max_time)?;
        Ok(())
    }
}

impl std::fmt::Display for NumericsConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Numerical Parameters:")?;
        writeln!(f, "  Time step: {}", self.time_step)?;
        writeln!(f, "  Number trajectories: {}", self.num_trajectories)?;
        writeln!(f, "  Measurements: {}", self.measurements)?;
        if self.num_random_init_states > 0 {
            writeln!(
                f,
                "  Number of randomly sampled initial states: {}",
                self.num_random_init_states
            )?;
        }
        write!(f, "  Seed: {}", self.seed)
    }
}

impl Config {
    /// Loads configuration from a TOML file.
    ///
    /// # Arguments
    /// * `file_path` - Path to the TOML configuration file
    ///
    /// # Returns
    /// * `Result<Self, Box<dyn std::error::Error>>` - The parsed configuration or an error
    pub fn from_file(file_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(file_path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }
}
