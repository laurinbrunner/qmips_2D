use crate::hilbert::{Occupation, SystemSize};
use crate::numerics::MATRIX_DIMENSION;
use std::fmt;

/// A direction on the 2D square lattice.
/// The discriminant matches the spin-state index convention:
/// 0 = left, 1 = right, 2 = up, 3 = down.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Direction {
    Left = 0,
    Right = 1,
    Up = 2,
    Down = 3,
}

impl Direction {
    pub const ALL: [Direction; MATRIX_DIMENSION] = [
        Direction::Left,
        Direction::Right,
        Direction::Up,
        Direction::Down,
    ];

    pub fn opposite(self) -> Direction {
        match self {
            Direction::Left => Direction::Right,
            Direction::Right => Direction::Left,
            Direction::Up => Direction::Down,
            Direction::Down => Direction::Up,
        }
    }

    /// The two lab directions perpendicular to `self`.
    ///
    /// These are exactly the displacements available to a `self`-pointing
    /// particle through the lateral jump channels
    /// `M_{r,sigma,mu} = sqrt(eps) b^dag_{r+e_mu,sigma} b_{r,sigma}`
    /// (`mu` perpendicular to `sigma`); see
    /// `quantum_lateral_Heff_derivation.md` §2.
    pub const fn perpendicular(self) -> [Direction; 2] {
        match self {
            Direction::Left | Direction::Right => [Direction::Up, Direction::Down],
            Direction::Up | Direction::Down => [Direction::Left, Direction::Right],
        }
    }
}

impl From<Direction> for usize {
    fn from(occ: Direction) -> Self {
        occ as usize
    }
}

/// Full spatial state of the lattice: a 2D grid plus per-particle cached neighbor info.
///
/// Particles are assigned fixed indices at construction time (row-major scan order).
/// After a hop, only the moving particle and its former/new neighbors are updated — O(1).
#[derive(Clone, Debug)]
pub struct LatticeState {
    pub system_size: SystemSize,
    /// `grid[x][y] = Some(particle_index)` if occupied, `None` if empty.
    pub grid: Vec<Vec<Option<usize>>>,
    // Per-particle Hamiltonian indices (derived from neighbor occupancy).
    pub neighbor_occupancy_index: Vec<usize>,
    pub number_particles: usize,
}

impl LatticeState {
    // ------------------------------------------------------------------ //
    //  Construction
    // ------------------------------------------------------------------ //

    /// Build from a 2D occupation grid.
    pub fn new(
        system_size: SystemSize,
        occupation: &[Vec<Occupation>],
        number_particles: usize,
    ) -> Self {
        let mut grid = vec![vec![None; system_size.l_y]; system_size.l_x];
        let neighbor_occupancy_index = vec![0; number_particles];

        let mut particle_idx = 0;
        for x in 0..system_size.l_x {
            for y in 0..system_size.l_y {
                if occupation[x][y] == Occupation::Occupied {
                    grid[x][y] = Some(particle_idx);
                    particle_idx += 1;
                }
            }
        }

        let mut state = LatticeState {
            system_size,
            grid,
            neighbor_occupancy_index,
            number_particles,
        };
        state.rebuild_all_neighbor_caches();
        state
    }

    /// Recompute neighbor caches for every particle (full O(L_x * L_y) sweep).
    fn rebuild_all_neighbor_caches(&mut self) {
        for x in 0..self.system_size.l_x {
            for y in 0..self.system_size.l_y {
                if let Some(particle_idx) = self.grid[x][y] {
                    let mut neighbor_occupation: usize = 0;
                    for &d in &Direction::ALL {
                        let (nx, ny) = self.neighbor_coords(x, y, d);
                        neighbor_occupation |=
                            (self.grid[nx][ny].is_some() as usize) << (d as usize);
                    }
                    self.neighbor_occupancy_index[particle_idx] = neighbor_occupation;
                }
            }
        }
    }

    /// Update neighbor caches for a single particle (O(1)) by giving it the direction in which an occupation changed.
    fn update_neighbor_cache(&mut self, x: usize, y: usize, dir: Direction, occupied: bool) {
        if let Some(particle_idx) = self.grid[x][y] {
            let bit = 1 << (dir as usize);
            if occupied {
                self.neighbor_occupancy_index[particle_idx] |= bit;
            } else {
                self.neighbor_occupancy_index[particle_idx] &= !bit;
            }
        }
    }

    fn recalculate_neighbor_cache(&mut self, x: usize, y: usize) {
        if let Some(particle_idx) = self.grid[x][y] {
            let mut neighbor_occupation: usize = 0;
            for &d in &Direction::ALL {
                let (nx, ny) = self.neighbor_coords(x, y, d);
                neighbor_occupation |= (self.grid[nx][ny].is_some() as usize) << (d as usize);
            }
            self.neighbor_occupancy_index[particle_idx] = neighbor_occupation;
        }
    }

    // ------------------------------------------------------------------ //
    //  Geometry helpers
    // ------------------------------------------------------------------ //

    /// Neighbor coordinates with periodic boundary conditions.
    pub fn neighbor_coords(&self, x: usize, y: usize, dir: Direction) -> (usize, usize) {
        match dir {
            Direction::Left => ((x + self.system_size.l_x - 1) % self.system_size.l_x, y),
            Direction::Right => ((x + 1) % self.system_size.l_x, y),
            Direction::Up => (x, (y + 1) % self.system_size.l_y),
            Direction::Down => (x, (y + self.system_size.l_y - 1) % self.system_size.l_y),
        }
    }

    /// Total number of lattice sites.
    pub fn num_sites(&self) -> usize {
        self.system_size.num_sites()
    }

    // ------------------------------------------------------------------ //
    //  Queries (O(1) per particle)
    // ------------------------------------------------------------------ //

    /// 4-bit Hamiltonian index for a particle (matches `get_hamiltonians` encoding).
    pub fn hamiltonian_index(&self, particle_idx: usize) -> usize {
        self.neighbor_occupancy_index[particle_idx]
    }

    /// All Hamiltonian indices, one per particle.
    pub fn hamiltonian_indices(&self) -> Vec<usize> {
        self.neighbor_occupancy_index.clone()
    }

    /// Jump mask for a particle: 1 if the neighbor in that direction is empty, 0 otherwise.
    pub fn jump_mask(&self, particle_idx: usize) -> [bool; MATRIX_DIMENSION] {
        let occ_idx = self.neighbor_occupancy_index[particle_idx];
        [
            (occ_idx >> Direction::Left as usize) & 1 != 1,
            (occ_idx >> Direction::Right as usize) & 1 != 1,
            (occ_idx >> Direction::Up as usize) & 1 != 1,
            (occ_idx >> Direction::Down as usize) & 1 != 1,
        ]
    }

    /// All jump masks, one per particle.
    pub fn jump_masks(&self) -> Vec<[bool; MATRIX_DIMENSION]> {
        (0..self.number_particles)
            .map(|n| self.jump_mask(n))
            .collect()
    }

    /// Position `(x, y)` for the particle with the given index.
    /// Returns `None` if no site currently stores this index.
    pub fn position_of_particle(&self, particle_idx: usize) -> Option<(usize, usize)> {
        for x in 0..self.system_size.l_x {
            for y in 0..self.system_size.l_y {
                if self.grid[x][y] == Some(particle_idx) {
                    return Some((x, y));
                }
            }
        }
        None
    }

    // ------------------------------------------------------------------ //
    //  Mutation — the core O(1) move
    // ------------------------------------------------------------------ //

    /// Move a particle one step in `dir`.  The target site **must** be empty.
    pub fn move_particle(&mut self, x: usize, y: usize, move_dir: Direction) {
        let particle_idx =
            self.grid[x][y].expect(format!("No particle at ({}, {}) to move", x, y).as_str());

        let (new_x, new_y) = self.neighbor_coords(x, y, move_dir);

        debug_assert!(
            self.grid[new_x][new_y].is_none(),
            "Target site ({}, {}) is occupied",
            new_x,
            new_y
        );

        // 1. Update grid
        self.grid[x][y] = None;
        self.grid[new_x][new_y] = Some(particle_idx);

        // 2. Update moving particle's position and recompute its neighbor flags
        self.recalculate_neighbor_cache(new_x, new_y);

        for &d in &Direction::ALL {
            if d == move_dir {
                continue; // This neighbor's occupancy is unchanged by the move
            }
            let (neigh_old_x, neigh_old_y) = self.neighbor_coords(x, y, d);
            self.update_neighbor_cache(neigh_old_x, neigh_old_y, d.opposite(), false);
        }

        for &d in &Direction::ALL {
            if d == move_dir.opposite() {
                continue; // This neighbor's occupancy is unchanged by the move
            }
            let (neigh_new_x, neigh_new_y) = self.neighbor_coords(new_x, new_y, d);
            self.update_neighbor_cache(neigh_new_x, neigh_new_y, d.opposite(), true);
        }
    }

    pub fn occupation_grid(&self) -> Vec<Vec<Occupation>> {
        self.grid
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&cell| {
                        if cell.is_some() {
                            Occupation::Occupied
                        } else {
                            Occupation::Empty
                        }
                    })
                    .collect()
            })
            .collect()
    }
}

impl fmt::Display for LatticeState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "LatticeState: Lx={}, Ly={}",
            self.system_size.l_x, self.system_size.l_y,
        )?;
        writeln!(f, "Grid ('.' = empty, number = particle index):")?;

        for y in (0..self.system_size.l_y).rev() {
            write!(f, "  y={:>3} |", y)?;
            for x in 0..self.system_size.l_x {
                match self.grid[x][y] {
                    Some(particle_idx) => write!(f, " {:>3}", particle_idx)?,
                    None => write!(f, "   .")?,
                }
            }
            writeln!(f)?;
        }

        write!(f, "         +")?;
        for _ in 0..self.system_size.l_x {
            write!(f, "----")?;
        }
        writeln!(f)?;

        write!(f, "          ")?;
        for x in 0..self.system_size.l_x {
            write!(f, " {:>3}", x)?;
        }

        Ok(())
    }
}
