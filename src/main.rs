use clap::Parser;
use mips_2d_qt::numerics::{DOWN_STATE, Direction, LEFT_STATE, RIGHT_STATE, UP_STATE};
use mips_2d_qt::{config, hilbert, numerics, saving, time_evolution};
use num_complex::Complex;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    let pargs = config::CommandLineArgs::parse();
    let mut config = config::Config::from_file(&pargs.config_path)?;

    // Resolve `seed = 0` (draw random) into a concrete seed *before* anything
    // uses the RNG, so the value actually used is stored with the results and
    // the run is reproducible. A non-zero drawn value keeps `0` meaning "random".
    if config.numerics.seed == 0 {
        config.numerics.seed = rand::random_range(1..=u64::MAX);
        println!("Using random seed {}", config.numerics.seed);
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.numerics.seed);

    println!("{}", config);

    let system_size = config.physics.system_size;
    let num_particles = config.physics.num_particles;

    let measurement_times: Vec<f64> = (0..config.numerics.measurements)
        .map(|i| (i as f64) * config.physics.max_time / (config.numerics.measurements - 1) as f64)
        .collect();

    // ----------------------------------------------------------------- //
    // ------------------ Initial state preparation -------------------- //
    // ----------------------------------------------------------------- //

    let mut initial_state_lattices: Vec<numerics::LatticeState> =
        Vec::with_capacity(config.numerics.num_random_init_states);
    let mut initial_state_spin_literals: Vec<Vec<Direction>> =
        vec![vec![Direction::Left; num_particles]; config.numerics.num_random_init_states];
    let mut initial_state_spins: Vec<numerics::SpinWavefunction> =
        Vec::with_capacity(config.numerics.num_random_init_states);

    for j in 0..config.numerics.num_random_init_states {
        let mut spatial_config =
            vec![vec![hilbert::Occupation::Empty; system_size.l_y]; system_size.l_x];
        let mut pool = (0..(system_size.l_x * system_size.l_y)).collect::<Vec<usize>>();
        for i in 0..num_particles {
            let swap_idx = rng.random_range(i..(system_size.l_x * system_size.l_y));
            pool.swap(i, swap_idx);
        }
        pool.truncate(num_particles);

        for &idx in pool.iter() {
            let x = idx / system_size.l_y;
            let y = idx % system_size.l_y;
            spatial_config[x][y] = hilbert::Occupation::Occupied;
        }

        // spatial_config[0] = hilbert::Occupation::Occupied;
        // spatial_config[1] = hilbert::Occupation::Occupied;
        // spatial_config[2] = hilbert::Occupation::Occupied;
        // spatial_config[6] = hilbert::Occupation::Occupied;

        let mut spins = vec![[Complex::<f64>::ZERO; numerics::MATRIX_DIMENSION]; num_particles];
        for i in 0..num_particles {
            let s = rng.random_range(0..=3);
            initial_state_spin_literals[j][i] = match s {
                0 => Direction::Left,
                1 => Direction::Right,
                2 => Direction::Up,
                3 => Direction::Down,
                _ => unreachable!(),
            };
            match s {
                0 => spins[i] = LEFT_STATE,  // Left
                1 => spins[i] = RIGHT_STATE, // Right
                2 => spins[i] = UP_STATE,    // Up
                3 => spins[i] = DOWN_STATE,  // Down
                _ => unreachable!(),
            }
        }

        initial_state_spins.push(numerics::SpinWavefunction { psi: spins });
        // spins[0] = [1, 0];
        // spins[1] = [1, 0];
        // spins[2] = [1, 0];
        // spins[3] = [1, 0];

        initial_state_lattices.push(numerics::LatticeState::new(
            system_size,
            &spatial_config,
            num_particles,
        ));
    }
    let total_number_initial_states = initial_state_lattices.len();

    // for state in initial_state_configurations.iter() {
    //     println!("Grid:");
    //     println!("{}", state);
    //     // for row in state.grid.iter() {
    //     //     for val in row.iter() {
    //     //         match val {
    //     //             Some(idx) => print!("{} ", idx),
    //     //             None => print!(". "),
    //     //         }
    //     //     }
    //     //     print!("\n");
    //     // }
    //     println!("Neighbor occupancy indices: {:?}", state.neighbor_occupancy_index);
    // }
    // ----------------------------------------------------------------- //
    // ------------------ Main simulation loop ------------------------- //
    // ----------------------------------------------------------------- //

    for init_state_idx in 0..config.numerics.num_random_init_states {
        let init_spacial = initial_state_lattices[init_state_idx].clone();
        let init_spins = initial_state_spins[init_state_idx].clone();

        let spatial_name = init_spacial
            .grid
            .iter()
            .map(|row| {
                row.iter()
                    .map(|idx| match idx {
                        Some(idx) => format!("{}_", idx),
                        None => "E_".to_string(),
                    })
                    .collect::<Vec<String>>()
                    .join("")
            })
            .collect::<Vec<String>>()
            .join("|");

        let spin_name = initial_state_spin_literals[init_state_idx]
            .iter()
            .map(|dir| match dir {
                Direction::Left => "L_".to_string(),
                Direction::Right => "R_".to_string(),
                Direction::Up => "U_".to_string(),
                Direction::Down => "D_".to_string(),
            })
            .collect::<Vec<String>>()
            .join("");

        let group_name = format!("spatial:({})_spin:{:?}", spatial_name, spin_name);

        println!(
            "Initial state {} (of {}) prepared:",
            init_state_idx + 1,
            total_number_initial_states
        );
        println!("{}", init_spacial);
        println!("Initial spin configuration: {:?}", spin_name);

        // Draw this state's trajectory seeds *before* the resume check so a
        // skipped (already-saved) state still consumes its block of the seed
        // stream. Otherwise a resumed run would hand the next state the seeds
        // that the skipped state used in the original run, making the file
        // non-reproducible from its config and letting two groups share a
        // seed stream.
        let seeds: Vec<u64> = (0..config.numerics.num_trajectories as usize)
            .map(|_| rng.random())
            .collect();

        if saving::group_exists(&pargs.output_path, &group_name)? {
            println!(
                "Results for initial state (spatial name {}, spin name {:?}) already exist in file {}. Skipping simulation.",
                spatial_name, spin_name, pargs.output_path
            );
            println!("{}", "-".repeat(30));
            continue;
        }
        println!("{}", "-".repeat(30));

        // ----------------------------------------------------------------- //
        // ------------------ main time evolution loop --------------------- //
        // ----------------------------------------------------------------- //

        let finished_seeds_count = Arc::new(AtomicUsize::new(0));

        let all_trajectory_results: Vec<time_evolution::TrajectoryResult> = seeds
            .into_par_iter()
            .map(|seed| {
                let mut thread_rng = StdRng::seed_from_u64(seed); // Or your RNG type

                let result = time_evolution::linear_trajectory_evolution(
                    &init_spacial,
                    &init_spins,
                    &measurement_times,
                    &config,
                    &mut thread_rng, // Use thread-local mutable RNG
                    pargs.save_end_states,
                    pargs.save_end_spatial,
                );

                if pargs.verbose {
                    let thread_id = std::thread::current().id();
                    let new_count = finished_seeds_count.fetch_add(1, Ordering::SeqCst) + 1;

                    println!("Thread {:?} finished seed number {}", thread_id, new_count);
                }

                result
            })
            .collect();

        saving::save_results(
            &all_trajectory_results,
            &measurement_times,
            &config,
            &pargs.output_path,
            &group_name,
            pargs.save_end_states,
            pargs.save_end_spatial,
            pargs.save_jump_times,
        )?;
    }

    let duration = start_time.elapsed().as_secs();

    println!(
        "Total execution time: {:02}:{:02}:{:02}",
        duration / 3600,
        (duration % 3600) / 60,
        duration % 60
    );

    Ok(())
}
