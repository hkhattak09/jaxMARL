"""Assembly Swarm Environment - JAX-compatible multi-agent environment.

A swarm robotics environment where agents must assemble into target formations
while avoiding collisions. Features include:
- Multiple target shapes from pickle files or procedural generation
- Domain randomization (rotation, scaling, offset)
- Trajectory tracking for visualization
- Rule-based prior policy for baselines/imitation learning
- Occupied grid tracking
- Reward sharing modes (individual, mean, max)

Compatible with MADDPG and other multi-agent RL algorithms.
"""

from typing import Tuple, Dict, Any, Optional
from enum import Enum
import jax
import jax.numpy as jnp
from jax import random
import flax.struct as struct

from environment import MultiAgentEnv, EnvState
from spaces import Box, MultiAgentActionSpace, MultiAgentObservationSpace
from physics import PhysicsParams, physics_step
from observations import (
    ObservationParams,
    compute_observations,
    get_k_nearest_neighbors_all_agents,
)
from rewards import (
    RewardParams,
    compute_rewards,
    REWARD_MODE_INDIVIDUAL,
    REWARD_MODE_SHARED_MEAN,
    REWARD_MODE_SHARED_MAX,
)
from shape_loader import (
    ShapeLibrary,
    load_shapes_from_pickle,
    create_shape_library_from_procedural,
    get_shape_from_library,
    apply_shape_transform,
)


class RewardSharingMode(str, Enum):
    """Reward sharing mode for multi-agent training."""
    INDIVIDUAL = "individual"
    SHARED_MEAN = "shared_mean"
    SHARED_MAX = "shared_max"


@struct.dataclass
class AssemblyState(EnvState):
    """State for the Assembly Swarm Environment.
    
    Attributes:
        positions: Agent positions (n_agents, 2)
        velocities: Agent velocities (n_agents, 2)
        grid_centers: Target grid cell centers (max_n_grid, 2)
        grid_mask: Valid grid cell mask (max_n_grid,)
        l_cell: Current shape's cell size
        time: Current simulation time
        step_count: Number of steps taken
        done: Whether episode is done
        
        # Trajectory tracking
        trajectory: Position history (traj_len, n_agents, 2)
        traj_idx: Current index in circular buffer
        
        # Shape info
        shape_idx: Index of current shape in library
        shape_rotation: Applied rotation angle
        shape_scale: Applied scale factor
        shape_offset: Applied translation offset (2,)
        
        # Occupancy tracking
        occupied_mask: Which grid cells are occupied (max_n_grid,)
        in_target: Which agents are in target (n_agents,)
        is_colliding: Which agents are colliding (n_agents,)
    """
    positions: jnp.ndarray
    velocities: jnp.ndarray
    grid_centers: jnp.ndarray
    grid_mask: jnp.ndarray
    l_cell: float
    time: float
    step_count: int
    done: bool
    
    # Trajectory
    trajectory: jnp.ndarray
    traj_idx: int
    
    # Shape info
    shape_idx: int
    shape_rotation: float
    shape_scale: float
    shape_offset: jnp.ndarray
    
    # Occupancy
    occupied_mask: jnp.ndarray
    in_target: jnp.ndarray
    is_colliding: jnp.ndarray


@struct.dataclass
class AssemblyParams:
    """Parameters for the Assembly Swarm Environment.
    
    Attributes:
        # Arena
        arena_size: Size of square arena
        
        # Agent
        agent_radius: Radius of each agent
        max_velocity: Maximum agent velocity
        max_acceleration: Maximum agent acceleration
        
        # Observation
        k_neighbors: Number of nearest neighbors to observe
        d_sen: Sensing distance
        
        # Physics/Observation/Reward sub-params
        physics: Physics parameters
        obs_params: Observation parameters
        reward_params: Reward parameters
        
        # Episode
        max_steps: Maximum steps per episode
        dt: Time step
        
        # Domain randomization
        randomize_shape: Whether to randomly select shape each episode
        randomize_rotation: Whether to rotate shape randomly
        rotation_range: Max rotation angle (radians)
        randomize_scale: Whether to scale shape randomly
        scale_min/scale_max: Scale range
        randomize_offset: Whether to translate shape randomly
        offset_range: Max offset from center
        
        # Trajectory
        traj_len: Length of trajectory history
        
        # Reward sharing (use integer constants from rewards.py)
        reward_mode: 0=individual, 1=shared_mean, 2=shared_max
    """
    # Arena
    arena_size: float = 5.0
    
    # Agent
    agent_radius: float = 0.1
    max_velocity: float = 0.8
    max_acceleration: float = 1.0
    
    # Observation
    k_neighbors: int = 6
    d_sen: float = 3.0
    
    # Sub-params
    physics: PhysicsParams = struct.field(default_factory=PhysicsParams)
    obs_params: ObservationParams = struct.field(default_factory=ObservationParams)
    reward_params: RewardParams = struct.field(default_factory=RewardParams)
    
    # Episode
    max_steps: int = 500
    dt: float = 0.1
    
    # Domain randomization
    randomize_shape: bool = True
    randomize_rotation: bool = True
    rotation_range: float = 3.14159  # pi
    randomize_scale: bool = True
    scale_min: float = 0.8
    scale_max: float = 1.3
    randomize_offset: bool = True
    offset_range: float = 1.2
    
    # Trajectory
    traj_len: int = 15
    
    # Prior policy
    # r_avoid for prior policy repulsion (None = compute dynamically per shape)
    # Formula: sqrt(4*n_grid/(n_agents*pi)) * l_cell
    r_avoid: float = None
    
    # Reward sharing (use integer: 0=individual, 1=shared_mean, 2=shared_max)
    reward_mode: int = 0  # REWARD_MODE_INDIVIDUAL


class AssemblySwarmEnv(MultiAgentEnv):
    """Assembly Swarm Environment - JAX-compatible multi-agent environment.
    
    A swarm robotics environment where agents must assemble into target
    formations while avoiding collisions.
    
    Features:
    - Multiple target shapes from pickle files or procedural generation
    - Domain randomization (rotation, scale, offset)
    - Trajectory tracking for visualization
    - Rule-based prior policy for baselines
    - Occupied grid tracking
    - Reward sharing modes
    
    Example:
        ```python
        # Basic usage
        env = AssemblySwarmEnv(n_agents=10)
        params = env.default_params
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key, params)
        
        # Load shapes from pickle
        library = load_shapes_from_pickle("shapes.pkl")
        env = AssemblySwarmEnv(n_agents=10, shape_library=library)
        
        # With domain randomization disabled
        params = AssemblyParams(randomize_shape=False, randomize_rotation=False)
        ```
    """
    
    def __init__(
        self,
        n_agents: int = 10,
        shape_library: Optional[ShapeLibrary] = None,
    ):
        """Initialize environment.
        
        Args:
            n_agents: Number of agents (fixed for JIT)
            shape_library: Pre-loaded shape library (or creates procedural)
        """
        super().__init__()
        self._n_agents = n_agents
        
        # Create default procedural library if none provided
        if shape_library is None:
            self._shape_library = create_shape_library_from_procedural(
                shape_types=["rectangle", "cross", "ring", "line"],
                n_cells=5,
                l_cell=0.35,
            )
        else:
            self._shape_library = shape_library
    
    @property
    def n_agents(self) -> int:
        return self._n_agents
    
    @property
    def shape_library(self) -> ShapeLibrary:
        return self._shape_library
    
    @property
    def default_params(self) -> AssemblyParams:
        return AssemblyParams()
    
    def reset(
        self,
        key: jnp.ndarray,
        params: AssemblyParams,
    ) -> Tuple[jnp.ndarray, AssemblyState]:
        """Reset environment with domain randomization.
        
        Args:
            key: JAX random key
            params: Environment parameters
            
        Returns:
            observations: Initial observations
            state: Initial state
        """
        key, pos_key, vel_key, shape_key, rot_key, scale_key, offset_key = random.split(key, 7)
        
        # Select shape
        if params.randomize_shape:
            shape_idx = random.randint(shape_key, (), 0, self._shape_library.n_shapes)
        else:
            shape_idx = 0
        
        # Get base shape
        base_grid, base_l_cell, base_mask = get_shape_from_library(
            self._shape_library, shape_idx
        )
        
        # Domain randomization
        if params.randomize_rotation:
            rotation = random.uniform(rot_key, shape=(), minval=-params.rotation_range, maxval=params.rotation_range)
        else:
            rotation = 0.0
        
        if params.randomize_scale:
            scale = random.uniform(scale_key, shape=(), minval=params.scale_min, maxval=params.scale_max)
        else:
            scale = 1.0
        
        if params.randomize_offset:
            offset = random.uniform(
                offset_key, shape=(2,), minval=-params.offset_range, maxval=params.offset_range
            )
        else:
            offset = jnp.zeros(2)
        
        # Apply transformations
        grid_centers = apply_shape_transform(
            base_grid, base_mask, rotation, scale, offset
        )
        l_cell = base_l_cell * scale
        
        # Initialize agent positions
        half_size = params.arena_size / 2
        positions = random.uniform(
            pos_key,
            shape=(self.n_agents, 2),
            minval=-half_size * 0.8,
            maxval=half_size * 0.8,
        )
        
        # Initialize velocities
        velocities = jnp.zeros((self.n_agents, 2))
        
        # Initialize trajectory buffer
        trajectory = jnp.zeros((params.traj_len, self.n_agents, 2))
        trajectory = trajectory.at[-1].set(positions)
        
        # Initialize occupancy
        occupied_mask = jnp.zeros(self._shape_library.max_n_grid, dtype=bool)
        in_target = jnp.zeros(self.n_agents, dtype=bool)
        is_colliding = jnp.zeros(self.n_agents, dtype=bool)
        
        state = AssemblyState(
            positions=positions,
            velocities=velocities,
            grid_centers=grid_centers,
            grid_mask=base_mask,
            l_cell=l_cell,
            time=0.0,
            step_count=0,
            done=False,
            trajectory=trajectory,
            traj_idx=0,
            shape_idx=shape_idx,
            shape_rotation=rotation,
            shape_scale=scale,
            shape_offset=offset,
            occupied_mask=occupied_mask,
            in_target=in_target,
            is_colliding=is_colliding,
        )
        
        # Update occupancy
        state = self._update_occupancy(state, params)
        
        obs = self._get_observations(state, params)
        
        return obs, state
    
    def step(
        self,
        key: jnp.ndarray,
        state: AssemblyState,
        actions: jnp.ndarray,
        params: AssemblyParams,
    ) -> Tuple[jnp.ndarray, AssemblyState, jnp.ndarray, jnp.ndarray, Dict]:
        """Take a step in the environment.
        
        Args:
            key: Random key
            state: Current state
            actions: Agent actions (n_agents, 2)
            params: Environment parameters
            
        Returns:
            obs, new_state, rewards, dones, info
        """
        # Clip actions
        actions = jnp.clip(actions, -params.max_acceleration, params.max_acceleration)
        
        # Physics step
        half_arena = params.arena_size / 2
        new_positions, new_velocities, _, b2b_collisions, _ = physics_step(
            state.positions,
            state.velocities,
            actions,
            params.physics,
            is_boundary=True,
            boundary_width=half_arena,
            boundary_height=half_arena,
        )
        
        # Clip velocities per-component (matches C++ MARL: np.clip(dp, -Vel_max, Vel_max))
        # This clips each velocity component independently, not the magnitude
        new_velocities = jnp.clip(new_velocities, -params.max_velocity, params.max_velocity)
        
        # Update trajectory (circular buffer)
        new_traj_idx = (state.traj_idx + 1) % params.traj_len
        new_trajectory = state.trajectory.at[new_traj_idx].set(new_positions)
        
        # Update time
        new_time = state.time + params.dt
        new_step_count = state.step_count + 1
        done = new_step_count >= params.max_steps
        
        # Create intermediate state for occupancy update
        new_state = AssemblyState(
            positions=new_positions,
            velocities=new_velocities,
            grid_centers=state.grid_centers,
            grid_mask=state.grid_mask,
            l_cell=state.l_cell,
            time=new_time,
            step_count=new_step_count,
            done=done,
            trajectory=new_trajectory,
            traj_idx=new_traj_idx,
            shape_idx=state.shape_idx,
            shape_rotation=state.shape_rotation,
            shape_scale=state.shape_scale,
            shape_offset=state.shape_offset,
            occupied_mask=state.occupied_mask,
            in_target=state.in_target,
            is_colliding=state.is_colliding,
        )
        
        # Update occupancy tracking
        new_state = self._update_occupancy(new_state, params)
        
        # Get observations
        obs = self._get_observations(new_state, params)
        
        # Compute rewards with occupancy info
        half_arena = params.arena_size / 2
        _, _, _, neighbor_indices = get_k_nearest_neighbors_all_agents(
            new_positions,
            new_velocities,
            params.obs_params.topo_nei_max,
            params.d_sen,
            False,  # is_periodic
            half_arena,
            half_arena,
        )
        
        rewards, reward_info = compute_rewards(
            new_positions,
            new_velocities,
            state.grid_centers,
            state.l_cell,
            neighbor_indices,
            params.reward_params,
            False,
            half_arena,
            half_arena,
            params.d_sen,
        )
        
        # Apply reward sharing
        rewards = self._apply_reward_sharing(rewards, params.reward_mode)
        
        dones = jnp.full((self.n_agents,), done)
        
        # Compute coverage as fraction of valid grid cells that are occupied
        # (matches MARL wrapper: grid-centric coverage over target shape only).
        # A grid cell is "occupied" if ANY agent is within r_avoid/2 of it.
        # r_avoid is computed using MARL formula: sqrt(4 * n_grid / (n_agents * pi)) * l_cell
        # Only count valid cells (within target shape) - MARL only stores valid cells.
        r_avoid = compute_r_avoid(
            new_state.grid_mask, self.n_agents, new_state.l_cell, params.r_avoid
        )
        agent_to_grid = new_state.grid_centers[None, :, :] - new_state.positions[:, None, :]
        agent_to_grid_dist = jnp.linalg.norm(agent_to_grid, axis=-1)  # (n_agents, n_grid)
        occupied_by_any = jnp.any(agent_to_grid_dist < r_avoid / 2.0, axis=0)  # (n_grid,)
        # Only count occupied cells that are valid (in the target shape)
        occupied_and_valid = occupied_by_any & new_state.grid_mask
        n_valid_cells = jnp.sum(new_state.grid_mask).astype(jnp.float32)
        n_occupied_cells = jnp.sum(occupied_and_valid).astype(jnp.float32)
        coverage_rate = jnp.where(n_valid_cells > 0.0, n_occupied_cells / n_valid_cells, 0.0)
        
        # Compute distribution uniformity (matches MARL wrapper)
        # Based on variance of minimum inter-agent distances
        distribution_uniformity = self._compute_distribution_uniformity(new_state.positions)
        
        # Compute Voronoi-based uniformity (matches MARL wrapper)
        # Based on variance of grid cells assigned to each agent
        voronoi_uniformity = self._compute_voronoi_uniformity(
            new_state.positions, new_state.grid_centers, new_state.grid_mask
        )

        info = {
            "time": new_time,
            "in_target": new_state.in_target,
            "is_colliding": new_state.is_colliding,
            "coverage_rate": coverage_rate,
            "distribution_uniformity": distribution_uniformity,
            "voronoi_uniformity": voronoi_uniformity,
            "shape_idx": state.shape_idx,
            "occupied_count": jnp.sum(new_state.occupied_mask),
        }
        
        return obs, new_state, rewards, dones, info
    
    def _compute_distribution_uniformity(self, positions: jnp.ndarray) -> jnp.ndarray:
        """Compute distribution uniformity based on minimum inter-agent distances.
        
        Matches MARL wrapper's distribution_uniformity() method.
        Measures how uniformly agents are distributed by analyzing variance
        in minimum distances between agents.
        
        Args:
            positions: Agent positions (n_agents, 2)
            
        Returns:
            Normalized uniformity metric (0 to 1), lower is more uniform
        """
        n_agents = positions.shape[0]
        
        # Compute pairwise distances: (n_agents, n_agents)
        diff = positions[:, None, :] - positions[None, :, :]  # (n_agents, n_agents, 2)
        pairwise_dist = jnp.linalg.norm(diff, axis=-1)  # (n_agents, n_agents)
        
        # Set diagonal to inf so we don't pick self-distance as minimum
        pairwise_dist = pairwise_dist + jnp.eye(n_agents) * 1e10
        
        # Minimum distance for each agent to its nearest neighbor
        min_dists = jnp.min(pairwise_dist, axis=1)  # (n_agents,)
        
        # Use coefficient of variation (std/mean) as uniformity metric
        # Lower values = more uniform distribution
        # Clamp to [0, 1] for interpretability
        mean_dist = jnp.mean(min_dists)
        std_dist = jnp.std(min_dists)
        cv = jnp.where(mean_dist > 1e-8, std_dist / mean_dist, 0.0)
        # Clamp to reasonable range
        uniformity = jnp.clip(cv, 0.0, 1.0)
        return uniformity
    
    def _compute_voronoi_uniformity(
        self,
        positions: jnp.ndarray,
        grid_centers: jnp.ndarray,
        grid_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute Voronoi-based uniformity.
        
        Matches MARL wrapper's voronoi_based_uniformity() method.
        Assigns each valid grid cell to the nearest agent and measures uniformity
        based on variance in number of cells per agent.
        
        Args:
            positions: Agent positions (n_agents, 2)
            grid_centers: Grid cell centers (n_grid, 2)
            grid_mask: Valid grid cells mask (n_grid,)
            
        Returns:
            Normalized Voronoi uniformity metric (0 to 1), lower is more uniform
        """
        n_agents = positions.shape[0]
        
        # Distance from each grid cell to each agent: (n_grid, n_agents)
        grid_to_agent = positions[None, :, :] - grid_centers[:, None, :]  # (n_grid, n_agents, 2)
        dist_to_agents = jnp.linalg.norm(grid_to_agent, axis=-1)  # (n_grid, n_agents)
        
        # Find nearest agent for each grid cell
        nearest_agent = jnp.argmin(dist_to_agents, axis=1)  # (n_grid,)
        
        # Count cells assigned to each agent (only valid cells)
        # Use one-hot encoding and sum
        agent_indices = jnp.arange(n_agents)
        cell_assignments = (nearest_agent[:, None] == agent_indices[None, :])  # (n_grid, n_agents)
        # Mask out invalid cells
        valid_assignments = cell_assignments & grid_mask[:, None]
        cells_per_agent = jnp.sum(valid_assignments, axis=0).astype(jnp.float32)  # (n_agents,)
        
        # Use coefficient of variation (std/mean) as uniformity metric
        # Lower values = more uniform distribution of cells per agent
        # Clamp to [0, 1] for interpretability
        mean_cells = jnp.mean(cells_per_agent)
        std_cells = jnp.std(cells_per_agent)
        cv = jnp.where(mean_cells > 1e-8, std_cells / mean_cells, 0.0)
        # Clamp to reasonable range
        uniformity = jnp.clip(cv, 0.0, 1.0)
        return uniformity
    
    def _update_occupancy(
        self,
        state: AssemblyState,
        params: AssemblyParams,
    ) -> AssemblyState:
        """Update occupancy tracking."""
        # Distance from each agent to each grid cell
        agent_to_grid = state.grid_centers[None, :, :] - state.positions[:, None, :]
        agent_to_grid_dist = jnp.linalg.norm(agent_to_grid, axis=-1)
        
        # In target threshold
        l_cell = state.l_cell
        in_target_threshold = jnp.sqrt(2.0) * l_cell / 2.0
        
        # Check which agents are in any target cell
        in_cell = agent_to_grid_dist < in_target_threshold
        in_valid_cell = in_cell & state.grid_mask[None, :]
        in_target = jnp.any(in_valid_cell, axis=1)
        
        # Grid cell is occupied if ANY agent is within r_avoid/2
        r_avoid = params.reward_params.collision_threshold
        occupied_by_any = jnp.any(agent_to_grid_dist < r_avoid / 2, axis=0)
        occupied_mask = occupied_by_any & state.grid_mask
        
        # Check collisions
        from rewards import compute_agent_collisions
        is_colliding, _ = compute_agent_collisions(
            state.positions,
            params.reward_params.collision_threshold,
        )
        
        return state.replace(
            in_target=in_target,
            is_colliding=is_colliding,
            occupied_mask=occupied_mask,
        )
    
    def _apply_reward_sharing(
        self,
        rewards: jnp.ndarray,
        mode: int,
    ) -> jnp.ndarray:
        """Apply reward sharing mode (JIT-compatible with jnp.where).
        
        Args:
            rewards: Individual rewards, shape (n_agents,)
            mode: 0=individual, 1=shared_mean, 2=shared_max
        """
        mean_reward = jnp.mean(rewards)
        max_reward = jnp.max(rewards)
        
        return jnp.where(
            mode == 1,  # REWARD_MODE_SHARED_MEAN
            jnp.full_like(rewards, mean_reward),
            jnp.where(
                mode == 2,  # REWARD_MODE_SHARED_MAX
                jnp.full_like(rewards, max_reward),
                rewards  # individual mode (mode == 0)
            )
        )
    
    def _get_observations(
        self,
        state: AssemblyState,
        params: AssemblyParams,
    ) -> jnp.ndarray:
        """Compute observations (internal use)."""
        return compute_observations(
            state.positions,
            state.velocities,
            state.grid_centers,
            state.l_cell,
            params.obs_params,
            False,
            params.arena_size / 2,
            params.arena_size / 2,
        )
    
    def get_obs(
        self,
        state: AssemblyState,
        params: AssemblyParams,
    ) -> jnp.ndarray:
        """Compute observations from state (public API).
        
        Args:
            state: Current environment state
            params: Environment parameters
            
        Returns:
            obs: Observations for all agents, shape (n_agents, obs_dim)
        """
        return self._get_observations(state, params)
    
    def get_trajectory(
        self,
        state: AssemblyState,
        params: AssemblyParams,
    ) -> jnp.ndarray:
        """Get ordered trajectory history.
        
        Returns:
            trajectory: (traj_len, n_agents, 2) in chronological order
        """
        indices = (jnp.arange(params.traj_len) + state.traj_idx + 1) % params.traj_len
        return state.trajectory[indices]
    
    def prior_policy(
        self,
        state: AssemblyState,
        params: AssemblyParams,
    ) -> jnp.ndarray:
        """Compute rule-based prior policy for all agents.
        
        Based on:
        1. Attraction to nearest UNOCCUPIED target cell
        2. Repulsion from nearby agents within r_avoid
        3. Velocity synchronization with neighbors
        
        r_avoid is computed dynamically using the MARL-LLM formula:
            r_avoid = sqrt(4 * n_grid / (n_agents * pi)) * l_cell
        This ensures proper spacing based on shape size and agent count.
        
        Returns:
            actions: (n_agents, 2) prior policy actions
        """
        # Compute r_avoid using MARL-LLM formula:
        # r_avoid = sqrt(4 * n_grid / (n_agents * pi)) * l_cell
        r_avoid = compute_r_avoid(
            state.grid_mask, 
            self._n_agents, 
            state.l_cell,
            params.r_avoid,
        )
        
        return compute_prior_policy(
            state.positions,
            state.velocities,
            state.grid_centers,
            state.grid_mask,
            state.l_cell,
            r_avoid,
            params.d_sen,
        )
    
    def observation_space(self, params: AssemblyParams) -> MultiAgentObservationSpace:
        """Get observation space."""
        obs_dim = self.get_obs_dim(params)
        return MultiAgentObservationSpace(
            n_agents=self.n_agents,
            obs_dim=obs_dim,
            low=-jnp.inf,
            high=jnp.inf,
        )
    
    def action_space(self, params: AssemblyParams) -> MultiAgentActionSpace:
        """Get action space."""
        return MultiAgentActionSpace(
            n_agents=self.n_agents,
            action_dim=2,
            low=-params.max_acceleration,
            high=params.max_acceleration,
        )
    
    def get_obs_dim(self, params: AssemblyParams) -> int:
        """Get observation dimension."""
        n_grid = self._shape_library.max_n_grid
        grid_obs_count = min(n_grid, params.obs_params.num_obs_grid_max)
        neighbor_count = min(params.obs_params.topo_nei_max, self.n_agents)
        return (
            4 +  # Self state
            neighbor_count * 4 +  # Neighbors
            4 +  # Target
            grid_obs_count * 2  # Grid
        )
    
    def get_action_dim(self, params: AssemblyParams) -> int:
        """Get action dimension (always 2 for 2D acceleration)."""
        return 2


def compute_r_avoid(
    grid_mask: jnp.ndarray,
    n_agents: int,
    l_cell: float,
    override_r_avoid: float = None,
) -> float:
    """Compute r_avoid using MARL-LLM formula.
    
    Formula: r_avoid = sqrt(4 * n_grid / (n_agents * pi)) * l_cell
    
    This ensures agents are spaced appropriately for the shape size.
    
    Args:
        grid_mask: Boolean mask of valid grid cells (n_grid,)
        n_agents: Number of agents
        l_cell: Grid cell size
        override_r_avoid: If provided and > 0, use this value instead
        
    Returns:
        r_avoid: Repulsion radius for prior policy
    """
    n_grid = jnp.sum(grid_mask)
    computed = jnp.sqrt(4.0 * n_grid / (n_agents * jnp.pi)) * l_cell
    
    # Use override if provided and positive, otherwise use computed
    # Python if is fine here because override_r_avoid is a static config value
    # known at trace time (not a traced JAX array)
    if override_r_avoid is not None and override_r_avoid > 0:
        return jnp.float32(override_r_avoid)  # Ensure JAX type
    return computed


def compute_prior_policy(
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    grid_centers: jnp.ndarray,
    grid_mask: jnp.ndarray,
    l_cell: float,
    r_avoid: float,
    d_sen: float,
    attraction_strength: float = 2.0,
    repulsion_strength: float = 3.0,  # Matches C++ MARL default
    sync_strength: float = 2.0,
) -> jnp.ndarray:
    """Compute rule-based prior policy for all agents (matches C++ MARL).
    
    Three-component Reynolds-style flocking policy:
    1. Attraction: Move toward nearest AVAILABLE target cell (goal-seeking)
       - A cell is "available" to agent i if agent i is closest to it, 
         OR no agent is within the in_target threshold
    2. Repulsion: Avoid nearby agents within r_avoid (separation)
    3. Synchronization: Match velocity with neighbors (alignment)
    
    Parameters match C++ implementation:
        - attraction_strength: 2.0
        - repulsion_strength: 3.0 
        - sync_strength: 2.0
        
    Key difference from naive implementation:
        - Uses dynamic r_avoid based on shape size
        - Agents attract to unoccupied cells, preventing clustering
    """
    n_agents = positions.shape[0]
    n_grid = grid_centers.shape[0]
    
    # Precompute distances from all agents to all grid cells: (n_agents, n_grid)
    agent_to_grid = grid_centers[None, :, :] - positions[:, None, :]  # (n_agents, n_grid, 2)
    dist_to_grid = jnp.linalg.norm(agent_to_grid, axis=-1)  # (n_agents, n_grid)
    
    # Set invalid cells to large distance
    dist_to_grid = jnp.where(grid_mask[None, :], dist_to_grid, 1e10)
    
    # In-target threshold
    in_target_threshold = jnp.sqrt(2.0) * l_cell / 2.0
    
    # Which agent is closest to each cell? (n_grid,)
    closest_agent_per_cell = jnp.argmin(dist_to_grid, axis=0)
    
    # Is any agent within in_target threshold of each cell? (n_grid,)
    min_dist_per_cell = jnp.min(dist_to_grid, axis=0)
    cell_is_occupied = min_dist_per_cell < in_target_threshold
    
    # For each agent, which cells are "available"?
    # A cell is available if: (agent is closest) OR (no one is in it)
    agent_indices = jnp.arange(n_agents)[:, None]  # (n_agents, 1)
    agent_is_closest = agent_indices == closest_agent_per_cell[None, :]  # (n_agents, n_grid)
    cell_available = agent_is_closest | ~cell_is_occupied[None, :]  # (n_agents, n_grid)
    cell_available = cell_available & grid_mask[None, :]  # Must be valid cell
    
    # Distance to available cells (unavailable = inf)
    dist_to_available = jnp.where(cell_available, dist_to_grid, 1e10)  # (n_agents, n_grid)
    
    # Nearest available cell for each agent
    nearest_available_idx = jnp.argmin(dist_to_available, axis=1)  # (n_agents,)
    nearest_available_dist = jnp.min(dist_to_available, axis=1)  # (n_agents,)
    
    # Target positions for each agent
    target_positions = grid_centers[nearest_available_idx]  # (n_agents, 2)
    
    def single_agent_policy(agent_idx: int) -> jnp.ndarray:
        pos = positions[agent_idx]
        vel = velocities[agent_idx]
        
        # 1. Attraction to nearest AVAILABLE target cell (precomputed)
        target_pos = target_positions[agent_idx]
        target_dist = nearest_available_dist[agent_idx]
        
        direction_to_target = target_pos - pos
        dist_to_target = jnp.maximum(target_dist, 1e-8)
        attraction_force = attraction_strength * direction_to_target / dist_to_target
        
        # If already in target cell (close enough), reduce attraction
        # This happens naturally as dist_to_target approaches 0
        in_target = target_dist < in_target_threshold
        attraction_force = jnp.where(in_target, jnp.zeros(2), attraction_force)
        
        # 2. Repulsion from nearby agents
        rel_to_agents = positions - pos
        dist_to_agents = jnp.linalg.norm(rel_to_agents, axis=-1)
        dist_to_agents = jnp.where(
            jnp.arange(n_agents) == agent_idx, 1e10, dist_to_agents
        )
        
        in_range = dist_to_agents < r_avoid
        safe_dist = jnp.maximum(dist_to_agents, 1e-8)
        repulsion_dir = -rel_to_agents / safe_dist[:, None]
        repulsion_mag = repulsion_strength * (r_avoid / safe_dist - 1.0)
        repulsion_mag = jnp.maximum(repulsion_mag, 0.0)
        
        repulsion_force = jnp.sum(
            jnp.where(in_range[:, None], repulsion_mag[:, None] * repulsion_dir, 0.0),
            axis=0
        )
        
        # 3. Velocity synchronization
        in_sensing = dist_to_agents < d_sen
        neighbor_count = jnp.sum(in_sensing)
        avg_velocity = jnp.sum(
            jnp.where(in_sensing[:, None], velocities, 0.0), axis=0
        ) / jnp.maximum(neighbor_count, 1.0)
        
        sync_force = sync_strength * (avg_velocity - vel)
        sync_force = jnp.where(neighbor_count > 0, sync_force, jnp.zeros(2))
        
        # Combine and clip
        total_force = attraction_force + repulsion_force + sync_force
        return jnp.clip(total_force, -1.0, 1.0)
    
    return jax.vmap(single_agent_policy)(jnp.arange(n_agents))


# ============================================================
# Factory functions
# ============================================================

def make_assembly_env(
    n_agents: int = 10,
    shape_library: Optional[ShapeLibrary] = None,
    shape_file: Optional[str] = None,
    **kwargs,
) -> Tuple[AssemblySwarmEnv, AssemblyParams]:
    """Create assembly environment with optional shape loading.
    
    Args:
        n_agents: Number of agents
        shape_library: Pre-loaded shape library
        shape_file: Path to pickle file with shapes
        **kwargs: Additional AssemblyParams arguments
        
    Returns:
        env: AssemblySwarmEnv instance
        params: AssemblyParams instance
    """
    if shape_file is not None and shape_library is None:
        shape_library = load_shapes_from_pickle(shape_file)
    
    env = AssemblySwarmEnv(n_agents=n_agents, shape_library=shape_library)
    params = AssemblyParams(**kwargs)
    
    return env, params


def make_vec_env(
    n_envs: int,
    n_agents: int = 10,
    shape_library: Optional[ShapeLibrary] = None,
    shape_file: Optional[str] = None,
    **kwargs,
):
    """Create vectorized assembly environment.
    
    Args:
        n_envs: Number of parallel environments
        n_agents: Number of agents per environment
        shape_library: Pre-loaded shape library
        shape_file: Path to pickle file
        **kwargs: Additional AssemblyParams arguments
        
    Returns:
        env, params, vec_reset, vec_step
    """
    env, params = make_assembly_env(
        n_agents=n_agents,
        shape_library=shape_library,
        shape_file=shape_file,
        **kwargs,
    )
    
    @jax.jit
    def vec_reset(keys: jnp.ndarray):
        return jax.vmap(lambda k: env.reset(k, params))(keys)
    
    @jax.jit
    def vec_step(keys: jnp.ndarray, states, actions: jnp.ndarray):
        return jax.vmap(lambda k, s, a: env.step(k, s, a, params))(keys, states, actions)
    
    return env, params, vec_reset, vec_step


# ============================================================
# Exports
# ============================================================

__all__ = [
    # Core classes
    "AssemblyState",
    "AssemblyParams",
    "AssemblySwarmEnv",
    # Factory functions
    "make_assembly_env",
    "make_vec_env",
    # Utilities
    "compute_prior_policy",
    "RewardSharingMode",
]


if __name__ == "__main__":
    print("Testing AssemblySwarmEnv...")
    
    # Test with procedural shapes
    env, params = make_assembly_env(n_agents=6)
    print(f"Created env with {env.shape_library.n_shapes} shapes")
    
    key = random.PRNGKey(42)
    obs, state = env.reset(key, params)
    
    print(f"Observation shape: {obs.shape}")
    print(f"Shape index: {state.shape_idx}")
    print(f"Shape rotation: {state.shape_rotation:.2f} rad")
    print(f"Shape scale: {state.shape_scale:.2f}")
    print(f"Shape offset: {state.shape_offset}")
    
    # Test prior policy
    prior_actions = env.prior_policy(state, params)
    print(f"Prior policy shape: {prior_actions.shape}")
    
    # Take a step
    key, step_key = random.split(key)
    obs, state, rewards, dones, info = env.step(step_key, state, prior_actions, params)
    print(f"Coverage rate: {info['coverage_rate']:.2%}")
    print(f"Occupied cells: {info['occupied_count']}")
    
    # Test trajectory
    traj = env.get_trajectory(state, params)
    print(f"Trajectory shape: {traj.shape}")
    
    # Test reward sharing
    params_mean = params.replace(reward_mode=1)  # REWARD_MODE_SHARED_MEAN
    _, _, rewards_mean, _, _ = env.step(step_key, state, prior_actions, params_mean)
    print(f"Shared mean rewards: all equal = {jnp.allclose(rewards_mean, rewards_mean[0])}")
    
    print("\nAssemblySwarmEnv test passed!")
