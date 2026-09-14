# mips_2d_qt — 2D MIPS Quantum Trajectories

Quantum-trajectory (Monte Carlo wavefunction) simulation of a Lindbladian
model of hard-core bosons on a 2D lattice, used to study Motility-Induced
Phase Separation (MIPS) - like clustering behaviour in a quantum driven-
dissipative system.

## Physical model

The system is a 2D square lattice of size $L_x \times L_y$ with periodic
boundary conditions, populated by $N$ hard-core bosons (each site holds at
most one particle). Every particle additionally carries an internal 4-level
"spin" degree of freedom

$$\sigma \in \{\uparrow, \downarrow, \leftarrow, \rightarrow\}$$

which encodes the direction the particle is trying to hop in. A particle at
site $\mathbf{r}$ with spin $\sigma$ hops to the neighboring site in that
direction at its full rate, and — at the separate, typically much smaller
rate $\varepsilon$ — to either of the two sites *perpendicular* to it (the
lateral channels described below). Any hop requires the target site to be
empty (hard-core / no double occupancy); a particle never hops backwards,
i.e. against its own spin direction.

**Hamiltonian** (coherent part, acts only on the internal spin, on-site):

$$H = g\sum_\mathbf{r} \sum_{\langle \sigma, \sigma'\rangle_{90°}} b_{\mathbf{r},\sigma}^\dagger b_{\mathbf{r},\sigma'}$$

where $\langle \sigma, \sigma'\rangle_{90°}$ pairs spin species related by a
90° rotation (e.g. if $\sigma = \uparrow$ then $\sigma' \in \{\rightarrow,
\leftarrow\}$, but not $\sigma' = \downarrow$). $g$ is the `transverse_field`
parameter: it coherently rotates a particle's hop direction by 90°.

**Dissipative part** (jump operators). Every jump operator acts on a single
particle and is rank-1 in the internal space; they come in two families.

*Forward hops* — one per site and spin species, moving the particle along the
direction its spin points:

$$M_{\mathbf{r}\sigma} = \sqrt{\gamma_\sigma}\,\Theta_{\mathbf{r} + \hat{\mathbf{e}}_\sigma} b^\dagger_{\mathbf{r} + \hat{\mathbf{e}}_\sigma,\sigma} b_{\mathbf{r}\sigma}$$

$\hat{\mathbf{e}}_\sigma$ is the unit vector for direction $\sigma$ (e.g.
$\hat{\mathbf{e}}_\rightarrow = (1, 0)$), and $\Theta$ is an emptiness
projector on the *target* site: the jump (an actual hop in direction $\sigma$)
can only fire if that site is unoccupied. Each direction has its own rate
$\gamma_\sigma$, `rates = [left, right, up, down]` (see below).

*Lateral hops* — for every internal direction $\sigma$, two further channels,
one per lab direction $\mu$ perpendicular to $\sigma$:

$$M_{\mathbf{r}\sigma\mu} = \sqrt{\varepsilon}\,\Theta_{\mathbf{r} + \hat{\mathbf{e}}_\mu} b^\dagger_{\mathbf{r} + \hat{\mathbf{e}}_\mu,\sigma} b_{\mathbf{r}\sigma}, \qquad \mu \perp \sigma$$

These displace the particle *sideways* with respect to where its spin points,
while leaving the internal state in $|\sigma\rangle$ — the operator's
internal action is the projector $P_\sigma$, exactly as for the forward
operators, so the displacement alone distinguishes the two lateral channels of
a given $\sigma$. An $\uparrow$ particle can thus also step $\leftarrow$ or
$\rightarrow$ at rate $\varepsilon$ and is still $\uparrow$ afterwards, but
it can never step $\downarrow$. $\varepsilon$ is the `lateral_rate`
parameter; $\varepsilon = 0$ (the default) switches the lateral channels off
and recovers the forward-only model exactly.

Each particle therefore carries $4 + 8 = 12$ jump channels: four forward (one
per $\sigma$) and eight lateral (two per $\sigma$).

**Effective Hamiltonian.** With
$H_\text{eff} = H - \tfrac{i}{2}\sum_k M_k^\dagger M_k$ and
$M^\dagger M = \text{rate}\cdot\Theta_{\mathbf{r} + \hat{\mathbf{e}}_\mu} n_{\mathbf{r}\sigma}$
for every channel, the anti-Hermitian part remains diagonal in the direction
basis, with decay rates

$$\Gamma_\leftarrow = \gamma_\leftarrow m_\leftarrow + \varepsilon\,(m_\uparrow + m_\downarrow), \qquad \Gamma_\uparrow = \gamma_\uparrow m_\uparrow + \varepsilon\,(m_\leftarrow + m_\rightarrow)$$

(and analogously for $\rightarrow$ and $\downarrow$), where
$m_d \in \{0, 1\}$ equals $1$ iff the neighbor in lab direction $d$ is empty.
A particle's decay rates thus depend only on the occupation of its four
neighbors, so there are only $2^4 = 16$ distinct effective Hamiltonians; they
are built and diagonalized once at startup (see
[src/hilbert.rs](src/hilbert.rs)).

Because the Hamiltonian and jump operators only act on individual
particles/sites (no direct particle-particle coupling), a product
(non-entangled) state stays a product state under the Lindbladian evolution:

$$H = \sum_\mathbf{r} H_\mathbf{r}, \qquad |\psi\rangle = \bigotimes_\mathbf{r} |\psi\rangle_\mathbf{r}$$

The code exploits this: each particle is propagated as an independent 4-level
system, so the effective Hilbert space dimension is $\dim(\mathcal{H}) = 4N$
rather than the full lattice Fock space.

The evolution itself is computed with the standard quantum-jump / Monte Carlo
wavefunction (MCWF) method (see [src/time_evolution.rs](src/time_evolution.rs)):
between jumps each particle's spin evolves under the non-Hermitian effective
Hamiltonian (diagonalized analytically once per neighbor-occupation
configuration, see [src/hilbert.rs](src/hilbert.rs)); when the trajectory's
norm drops below a drawn random threshold `r`, the jump time is located by
bisection, and one of the $12N$ jump channels is sampled with weight
$\langle M_k^\dagger M_k\rangle = \text{rate}\cdot m_\mu |\psi_\sigma|^2$.
The selected particle hops in that channel's displacement direction $\mu$ and
its spin collapses onto $|\sigma\rangle$ — equal to the hop direction for a
forward jump, perpendicular to it for a lateral one. Many independent
trajectories are simulated and averaged to approximate the Lindbladian's mixed
steady state.

## Configuration parameters

The simulation is configured via a TOML file (see
[example_config.toml](example_config.toml)):

```toml
[physics]
system_size=[ 4, 5 ]
num_particles=1
transverse_field=2.0
max_time=100
rates=[ 1.0, 1.0, 1.0, 1.0 ]
lateral_rate=0.0

[numerics]
time_step=5e-2
num_trajectories=1
measurements=101
seed=12
num_random_init_states=5
```

### `[physics]`

| Key | Meaning |
|---|---|
| `system_size` | `[L_x, L_y]` — lattice dimensions (periodic boundaries). Given as a 2-element array, mapped positionally to `L_x`, `L_y`. |
| `num_particles` | Number of hard-core bosons `N` placed on the lattice. |
| `transverse_field` | `g`, the coherent 90-degree spin-rotation strength in `H`. |
| `max_time` | `T_max`, total simulated time per trajectory. |
| `rates` | `[left, right, up, down]` jump rates. Only their *ratios* matter: internally they are normalized by their mean, so `time_step`/`max_time`/`transverse_field` are effectively expressed in units of the mean rate. `[1,1,1,1]` is the isotropic model. Optional, defaults to `[1,1,1,1]`. |
| `lateral_rate` | $\varepsilon$, the rate of each lateral (perpendicular) jump channel, in units of the *mean* forward rate. Unlike `rates` it is **not** mean-normalized and carries no anisotropy factor: it enters $H_\text{eff}$ and the jump weights directly, and is the same for all eight lateral channels. Optional, defaults to `0.0`, which disables the lateral channels and reproduces the forward-only model exactly (bit-identical trajectories). |

### `[numerics]`

| Key | Meaning |
|---|---|
| `time_step` | Nominal MCWF propagation step `dt`. Automatically bisected near a jump to resolve the jump time (to a norm tolerance of `1e-6`), then reset to this value afterwards. |
| `num_trajectories` | Number of stochastic quantum trajectories simulated *per initial state*, run in parallel (via `rayon`) and averaged for the reported observables. |
| `measurements` | Number of equally spaced measurement times in `[0, max_time]` (inclusive), i.e. `measurements - 1` intervals. |
| `seed` | RNG seed. If `0`, a random seed is drawn at startup, printed to stdout, and stored in the output so the run stays reproducible. |
| `num_random_init_states` | Number of independent random initial product states to simulate. For each, `num_particles` sites are chosen uniformly at random (without replacement) to be occupied, and each particle's spin is set to a uniformly random direction. Each initial state gets its own group in the output file. Defaults to `0`, in which case the program does no work — this must be set to at least `1` to actually run a simulation. |

The total number of simulated trajectories is `num_random_init_states * num_trajectories`.

## Building

Prerequisites:
- A recent Rust toolchain (edition 2024, i.e. rustc >= 1.85) via `rustup`/`cargo`.
- A C/C++ compiler and `cmake` — the HDF5 C library is compiled from source at
  build time via the `hdf5-src`/`hdf5-sys` crates (`static` feature).

Build with the provided Makefile:

```sh
make build
```

This runs `cargo rustc --release` and copies the resulting binary to
`./QuantumTrajectories_2D_MIPS` in the repo root. Equivalently:

```sh
cargo build --release
cp target/release/mips_2d_qt ./QuantumTrajectories_2D_MIPS
```

## Running

```sh
./QuantumTrajectories_2D_MIPS --config-path example_config.toml --output-path out.h5
```

Command-line flags:

| Flag | Meaning |
|---|---|
| `--config-path <PATH>` | Path to the TOML configuration file (required). |
| `--output-path <PATH>` | Path to the HDF5 file results are written/appended to (required). |
| `-v`, `--verbose` | Print a line to stdout each time a trajectory finishes. |
| `--save-end-states` | Also store each trajectory's final spatial configuration and full spin wavefunction. |
| `--save-end-spatial` | Also store each trajectory's final spatial configuration only (cheaper than `--save-end-states`). |
| `--save-jump-times` | Also store every quantum-jump time for each trajectory. |

The run prints the resolved configuration, then progress, then total wall-clock
time.

Re-running with the same config/output file skips any initial state whose
group already exists in the output file (see below), so a run can be safely
resumed/extended (e.g. after increasing `num_random_init_states`) without
recomputing earlier results. Concurrent invocations writing to the same output
file coordinate via a `.lock` file (retried with exponential backoff) so
parallel jobs targeting the same HDF5 file don't corrupt it.

## Output format

Results are written to a single HDF5 file. Each simulated **initial state**
(one random placement of particles + spins) gets its own top-level HDF5 group,
named:

```
spatial:(<grid>)_spin:(<directions>)
```

where `<grid>` encodes, row by row, either the particle index at each site or
`E` for empty, and `<directions>` lists each particle's initial spin direction
(`L`/`R`/`U`/`D`). This name is what `group_exists` checks to decide whether an
initial state has already been simulated.

Each group contains, averaged over that initial state's `num_trajectories`
trajectories:

| Dataset | Shape | Contents |
|---|---|---|
| `measure_time` | `(measurements,)` | The measurement times. |
| `occupation` | `(measurements, L_x, L_y)` | Mean site occupation (0-1) at each measurement time. |
| `correlation` | `(measurements, L_x*L_y, L_x*L_y)` | Mean two-point density correlation `<n_i n_j>` between all site pairs, at each measurement time. |

Optional datasets (only written if the corresponding CLI flag was passed),
one entry per trajectory (not averaged):

| Dataset | Shape | Contents |
|---|---|---|
| `end_spatial_configuration` | `(num_trajectories, L_x, L_y)` | Final particle index per site (`--save-end-states` or `--save-end-spatial`); empty sites are stored as `usize::MAX`. |
| `end_spin_state_real` / `end_spin_state_imag` | `(num_trajectories, num_particles, 4)` | Final per-particle spin amplitudes (`--save-end-states`). |
| `jump_times` | `(num_trajectories, max_jumps)` | Time of each quantum jump per trajectory, padded with `-1.0` (`--save-jump-times`). |

Every group also carries the full resolved configuration as HDF5 attributes,
so each group is self-describing and independently reproducible:
`l_x`, `l_y`, `num_particles`, `transverse_field`, `max_time`, `rates`,
`normalized_rates`, `lateral_rate`, `time_step`, `num_trajectories`,
`measurements`, `seed` (the concrete seed actually used),
`num_random_init_states`, `code_version` (crate version), and `git_commit`
(commit hash the binary was built from).

Note: bit-identical reproduction of a run from its stored `seed` additionally
requires using the same version of the `rand` crate, since `StdRng`'s output
stream is not guaranteed stable across `rand` major versions — this is
pinned by `Cargo.lock`.
