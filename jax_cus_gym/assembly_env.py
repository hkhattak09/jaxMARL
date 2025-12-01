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
from rewards import RewardParams, compute_rewards
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
        
        # Reward sharing
        reward_mode: "individual", "shared_mean", or "shared_max"
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
    
    # Reward sharing
    reward_mode: str = "individual"  # "individual", "shared_mean", "shared_max"


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
        
        # Clip velocities
        speed = jnp.linalg.norm(new_velocities, axis=-1, keepdims=True)
        speed_clipped = jnp.minimum(speed, params.max_velocity)
        new_velocities = jnp.where(
            speed > 0,
            new_velocities * speed_clipped / (speed + 1e-8),
            new_velocities,
        )
        
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
        
        info = {
            "time": new_time,
            "in_target": new_state.in_target,
            "is_colliding": new_state.is_colliding,
            "coverage_rate": jnp.mean(new_state.in_target.astype(jnp.float32)),
            "shape_idx": state.shape_idx,
            "occupied_count": jnp.sum(new_state.occupied_mask),
        }
        
        return obs, new_state, rewards, dones, info
    
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
        mode: str,
    ) -> jnp.ndarray:
        """Apply reward sharing mode."""
        if mode == "shared_mean":
            return jnp.full_like(rewards, jnp.mean(rewards))
        elif mode == "shared_max":
            return jnp.full_like(rewards, jnp.max(rewards))
        else:  # individual
            return rewards
    
    def _get_observations(
        self,
        state: AssemblyState,
        params: AssemblyParams,
    ) -> jnp.ndarray:
        """Compute observations."""
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
        1. Attraction to nearest target cell
        2. Repulsion from nearby agents
        3. Velocity synchronization with neighbors
        
        Returns:
            actions: (n_agents, 2) prior policy actions
        """
        return compute_prior_policy(
            state.positions,
            state.velocities,
            state.grid_centers,
            state.grid_mask,
            state.l_cell,
            params.reward_params.collision_threshold,
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



def compute_prior_policy(
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    grid_centers: jnp.ndarray,
    grid_mask: jnp.ndarray,
    l_cell: float,
    r_avoid: float,
    d_sen: float,
    attraction_strength: float = 2.0,
    repulsion_strength: float = 1.0,
    sync_strength: float = 2.0,
) -> jnp.ndarray:
    """Compute rule-based prior policy for all agents.
    
    Three-component policy:
    1. Attraction: Move toward nearest unoccupied target cell
    2. Repulsion: Avoid nearby agents within r_avoid
    3. Synchronization: Match velocity with neighbors
    """
    n_agents = positions.shape[0]
    
    def single_agent_policy(agent_idx: int) -> jnp.ndarray:
        pos = positions[agent_idx]
        vel = velocities[agent_idx]
        
        # 1. Attraction to nearest target
        rel_to_grid = grid_centers - pos
        dist_to_grid = jnp.linalg.norm(rel_to_grid, axis=-1)
        dist_to_grid = jnp.where(grid_mask, dist_to_grid, 1e10)
        
        nearest_idx = jnp.argmin(dist_to_grid)
        nearest_dist = dist_to_grid[nearest_idx]
        target_pos = grid_centers[nearest_idx]
        
        direction_to_target = target_pos - pos
        dist_to_target = jnp.maximum(nearest_dist, 1e-8)
        attraction_force = attraction_strength * direction_to_target / dist_to_target
        
        in_target = nearest_dist < jnp.sqrt(2.0) * l_cell / 2.0
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
    params_mean = params.replace(reward_mode="shared_mean")
    _, _, rewards_mean, _, _ = env.step(step_key, state, prior_actions, params_mean)
    print(f"Shared mean rewards: all equal = {jnp.allclose(rewards_mean, rewards_mean[0])}")
    
    print("\nAssemblySwarmEnv test passed!")
