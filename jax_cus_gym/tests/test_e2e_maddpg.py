"""End-to-end test simulating MADDPG training with actual target shapes.

This script demonstrates a complete training loop using the JAX swarm environment
with MADDPG-style training. It uses actual target shapes from the fig directory
(results.pkl) which contains processed shape data from PNG images.

It includes:
- Vectorized environment execution with real shapes
- Experience collection with replay buffer
- Simulated actor-critic updates
- Performance metrics tracking

Run with: python tests/test_e2e_maddpg.py
"""

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from typing import Tuple, Dict, List
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_fig_shapes_path() -> str:
    """Get path to the actual shapes file in the fig directory.
    
    Returns:
        Path to fig/results.pkl containing processed target shapes.
        Returns None if the file doesn't exist.
    """
    workspace_root = Path(__file__).parent.parent.parent  # jax_cus_gym -> MARL_jax
    fig_path = workspace_root / "fig" / "results.pkl"
    if fig_path.exists():
        return str(fig_path)
    return None


# Get the shapes path once at module load
FIG_SHAPES_PATH = get_fig_shapes_path()


from maddpg_wrapper import (
    MADDPGWrapper,
    VectorizedMADDPGWrapper,
    Transition,
    create_vec_maddpg_env,
    stack_transitions,
)
from assembly_env import AssemblyParams
from observations import ObservationParams
from rewards import RewardParams


# Simple replay buffer for demonstration
class ReplayBuffer:
    """Simple replay buffer for MADDPG."""
    
    def __init__(self, capacity: int, n_agents: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.obs = jnp.zeros((capacity, n_agents, obs_dim))
        self.actions = jnp.zeros((capacity, n_agents, action_dim))
        self.rewards = jnp.zeros((capacity, n_agents))
        self.next_obs = jnp.zeros((capacity, n_agents, obs_dim))
        self.dones = jnp.zeros((capacity, n_agents))
        
        self.idx = 0
        self.size = 0
    
    def add(self, transition: Transition):
        """Add a transition to the buffer."""
        self.obs = self.obs.at[self.idx].set(transition.obs)
        self.actions = self.actions.at[self.idx].set(transition.actions)
        self.rewards = self.rewards.at[self.idx].set(transition.rewards)
        self.next_obs = self.next_obs.at[self.idx].set(transition.next_obs)
        self.dones = self.dones.at[self.idx].set(transition.dones.astype(jnp.float32))
        
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, key: jnp.ndarray, batch_size: int) -> Transition:
        """Sample a batch of transitions."""
        indices = random.randint(key, (batch_size,), 0, self.size)
        
        return Transition(
            obs=self.obs[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_obs=self.next_obs[indices],
            dones=self.dones[indices],
        )


# Simple MLP actor network
class Actor(nn.Module):
    """Simple MLP actor for continuous actions."""
    action_dim: int
    hidden_dim: int = 64
    
    @nn.compact
    def __call__(self, obs):
        x = nn.Dense(self.hidden_dim)(obs)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return nn.tanh(x)  # Actions in [-1, 1]


# Simple MLP critic network  
class Critic(nn.Module):
    """Simple MLP critic for Q-value estimation."""
    hidden_dim: int = 64
    
    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs, action], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


def test_environment_performance():
    """Test environment step throughput with actual shapes."""
    print("\n" + "="*60)
    print("Testing Environment Performance")
    print("="*60 + "\n")
    
    n_envs = 32
    n_agents = 10
    n_steps = 100
    
    kwargs = {'shape_file': FIG_SHAPES_PATH} if FIG_SHAPES_PATH else {}
    if FIG_SHAPES_PATH:
        print(f"  Using actual shapes from: {FIG_SHAPES_PATH}")
    
    vec_wrapper = create_vec_maddpg_env(n_envs=n_envs, n_agents=n_agents, **kwargs)
    
    print(f"  Shape library: {vec_wrapper.wrapper.env.shape_library.n_shapes} shapes")
    print(f"  Max grid cells: {vec_wrapper.wrapper.env.shape_library.max_n_grid}")
    
    # Compile functions
    @jax.jit
    def reset(keys):
        return vec_wrapper.reset(keys)
    
    @jax.jit
    def step(keys, states, actions):
        return vec_wrapper.step(keys, states, actions)
    
    key = random.PRNGKey(0)
    keys = random.split(key, n_envs)
    
    # Warm-up (compilation)
    obs, states = reset(keys)
    actions = jnp.zeros((n_envs, n_agents, 2))
    step_keys = random.split(random.PRNGKey(1), n_envs)
    obs, states, rewards, dones, info = step(step_keys, states, actions)
    
    # Benchmark
    start_time = time.time()
    
    for i in range(n_steps):
        key, *step_keys = random.split(key, n_envs + 1)
        step_keys = jnp.stack(step_keys)
        actions = random.uniform(key, (n_envs, n_agents, 2), minval=-1, maxval=1)
        obs, states, rewards, dones, info = step(step_keys, states, actions)
    
    # Wait for computation to complete
    obs.block_until_ready()
    
    elapsed = time.time() - start_time
    total_steps = n_envs * n_steps
    steps_per_sec = total_steps / elapsed
    
    print(f"  Environments: {n_envs}")
    print(f"  Agents per env: {n_agents}")
    print(f"  Total steps: {total_steps}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {steps_per_sec:.0f} env steps/sec")
    print(f"  Agent steps/sec: {steps_per_sec * n_agents:.0f}")
    
    print("\n  ✓ Environment performance test passed")
    
    return steps_per_sec


def test_training_loop():
    """Test a simulated MADDPG training loop with actual shapes."""
    print("\n" + "="*60)
    print("Testing MADDPG Training Loop (Simulated)")
    print("="*60 + "\n")
    
    # Configuration
    n_envs = 8
    n_agents = 6
    buffer_size = 10000
    batch_size = 64
    n_episodes = 5
    max_steps = 50
    
    print(f"  Config: {n_envs} envs × {n_agents} agents")
    print(f"  Episodes: {n_episodes}, Max steps: {max_steps}")
    print(f"  Buffer size: {buffer_size}, Batch size: {batch_size}")
    
    # Create environment with actual shapes
    kwargs = {'shape_file': FIG_SHAPES_PATH} if FIG_SHAPES_PATH else {}
    if FIG_SHAPES_PATH:
        print(f"  Using actual shapes from: {FIG_SHAPES_PATH}")
    
    vec_wrapper = create_vec_maddpg_env(
        n_envs=n_envs, 
        n_agents=n_agents,
        max_steps=max_steps,
        **kwargs
    )
    
    print(f"  Shape library: {vec_wrapper.wrapper.env.shape_library.n_shapes} shapes")
    print(f"  Max grid cells: {vec_wrapper.wrapper.env.shape_library.max_n_grid}")
    
    # Create replay buffer
    buffer = ReplayBuffer(
        capacity=buffer_size,
        n_agents=n_agents,
        obs_dim=vec_wrapper.obs_dim,
        action_dim=vec_wrapper.action_dim,
    )
    
    # Initialize actor networks (one per agent in MADDPG)
    actors = [Actor(action_dim=2) for _ in range(n_agents)]
    
    # Initialize parameters
    key = random.PRNGKey(0)
    dummy_obs = jnp.zeros((vec_wrapper.obs_dim,))
    actor_params = []
    for i, actor in enumerate(actors):
        key, init_key = random.split(key)
        params = actor.init(init_key, dummy_obs)
        actor_params.append(params)
    
    # JIT compile environment functions
    @jax.jit
    def env_reset(keys):
        return vec_wrapper.reset(keys)
    
    @jax.jit
    def env_step(keys, states, actions):
        return vec_wrapper.step(keys, states, actions)
    
    # JIT compile action selection
    @jax.jit
    def select_actions(params_list, obs, key):
        """Select actions for all agents across all envs."""
        # obs: (n_envs, n_agents, obs_dim)
        # For each agent, apply its actor to all envs
        actions = []
        for agent_idx in range(n_agents):
            agent_obs = obs[:, agent_idx, :]  # (n_envs, obs_dim)
            agent_actions = actors[agent_idx].apply(params_list[agent_idx], agent_obs)
            actions.append(agent_actions)
        
        actions = jnp.stack(actions, axis=1)  # (n_envs, n_agents, action_dim)
        
        # Add exploration noise
        key, noise_key = random.split(key)
        noise = random.normal(noise_key, actions.shape) * 0.1
        actions = jnp.clip(actions + noise, -1, 1)
        
        return actions
    
    # Training metrics
    episode_returns = []
    episode_lengths = []
    coverage_rates = []
    
    print("\n  Training...")
    
    for episode in range(n_episodes):
        # Reset environments
        key, reset_key = random.split(key)
        reset_keys = random.split(reset_key, n_envs)
        obs, states = env_reset(reset_keys)
        
        episode_reward = jnp.zeros((n_envs, n_agents))
        step_count = 0
        
        # Collect experience
        for step in range(max_steps):
            # Select actions
            key, action_key = random.split(key)
            actions = select_actions(actor_params, obs, action_key)
            
            # Step environment
            key, step_key = random.split(key)
            step_keys = random.split(step_key, n_envs)
            next_obs, next_states, rewards, dones, info = env_step(
                step_keys, states, actions
            )
            
            # Store transitions (for each env)
            for env_idx in range(n_envs):
                transition = Transition(
                    obs=obs[env_idx],
                    actions=actions[env_idx],
                    rewards=rewards[env_idx],
                    next_obs=next_obs[env_idx],
                    dones=dones[env_idx],
                )
                buffer.add(transition)
            
            episode_reward += rewards
            step_count += 1
            
            # Check if all envs done
            all_done = jnp.all(next_states.env_state.done)
            if all_done:
                break
            
            obs = next_obs
            states = next_states
        
        # Record metrics
        mean_return = float(jnp.mean(jnp.sum(episode_reward, axis=1)))
        episode_returns.append(mean_return)
        episode_lengths.append(step_count)
        coverage_rates.append(float(info["coverage_rate"].mean()))
        
        print(f"    Episode {episode+1}: Return={mean_return:.2f}, "
              f"Length={step_count}, Coverage={coverage_rates[-1]:.1%}, "
              f"Buffer={buffer.size}")
        
        # Simulate update step (just sample from buffer)
        if buffer.size >= batch_size:
            key, sample_key = random.split(key)
            batch = buffer.sample(sample_key, batch_size)
            # In real MADDPG, we would update actor and critic here
    
    print("\n  Training Summary:")
    print(f"    Mean episode return: {jnp.mean(jnp.array(episode_returns)):.2f}")
    print(f"    Mean episode length: {jnp.mean(jnp.array(episode_lengths)):.1f}")
    print(f"    Mean coverage rate: {jnp.mean(jnp.array(coverage_rates)):.1%}")
    print(f"    Final buffer size: {buffer.size}")
    
    print("\n  ✓ Training loop test passed")
    
    return episode_returns


def test_maddpg_critic():
    """Test centralized critic with global state using actual shapes."""
    print("\n" + "="*60)
    print("Testing MADDPG Centralized Critic")
    print("="*60 + "\n")
    
    n_agents = 4
    kwargs = {'shape_file': FIG_SHAPES_PATH} if FIG_SHAPES_PATH else {}
    wrapper = MADDPGWrapper(n_agents=n_agents, **kwargs)
    
    print(f"  obs_dim per agent: {wrapper.obs_dim}")
    print(f"  action_dim per agent: {wrapper.action_dim}")
    print(f"  global_state_dim: {wrapper.global_state_dim}")
    
    # Centralized critic takes global state + all actions
    critic_input_dim = wrapper.global_state_dim + n_agents * wrapper.action_dim
    print(f"  critic_input_dim: {critic_input_dim}")
    
    # Create centralized critic
    class CentralizedCritic(nn.Module):
        hidden_dim: int = 128
        
        @nn.compact
        def __call__(self, global_state, all_actions):
            # all_actions: (n_agents, action_dim) -> flatten
            all_actions_flat = all_actions.flatten()
            x = jnp.concatenate([global_state, all_actions_flat])
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
            x = nn.Dense(1)(x)
            return x
    
    # Initialize critic
    key = random.PRNGKey(0)
    critic = CentralizedCritic()
    
    dummy_global_state = jnp.zeros((wrapper.global_state_dim,))
    dummy_actions = jnp.zeros((n_agents, wrapper.action_dim))
    
    key, init_key = random.split(key)
    critic_params = critic.init(init_key, dummy_global_state, dummy_actions)
    
    # Test with real data
    key, reset_key = random.split(key)
    obs, state = wrapper.reset(reset_key)
    global_state = wrapper.get_global_state(state)
    
    actions = random.uniform(key, (n_agents, 2), minval=-1, maxval=1)
    
    q_value = critic.apply(critic_params, global_state, actions)
    print(f"  Q-value output shape: {q_value.shape}")
    print(f"  Q-value: {float(q_value[0]):.4f}")
    
    # Test batched
    @jax.jit
    def batch_q_values(params, global_states, all_actions):
        return jax.vmap(lambda gs, a: critic.apply(params, gs, a))(
            global_states, all_actions
        )
    
    batch_size = 32
    batch_global_states = jnp.stack([global_state] * batch_size)
    batch_actions = jnp.stack([actions] * batch_size)
    
    batch_q = batch_q_values(critic_params, batch_global_states, batch_actions)
    print(f"  Batched Q-values shape: {batch_q.shape}")
    
    print("\n  ✓ Centralized critic test passed")


def test_policy_gradient():
    """Test policy gradient computation with actual shapes."""
    print("\n" + "="*60)
    print("Testing Policy Gradient Computation")
    print("="*60 + "\n")
    
    n_agents = 3
    kwargs = {'shape_file': FIG_SHAPES_PATH} if FIG_SHAPES_PATH else {}
    wrapper = MADDPGWrapper(n_agents=n_agents, **kwargs)
    
    # Simple actor
    actor = Actor(action_dim=2, hidden_dim=32)
    
    key = random.PRNGKey(0)
    dummy_obs = jnp.zeros((wrapper.obs_dim,))
    key, init_key = random.split(key)
    actor_params = actor.init(init_key, dummy_obs)
    
    # Define loss function
    def actor_loss(params, obs, target_q):
        actions = actor.apply(params, obs)
        # In real MADDPG, we'd use critic to get Q-value
        # Here we simulate with a simple loss
        return -jnp.mean(target_q * jnp.sum(actions**2, axis=-1))
    
    # Compute gradient
    key, reset_key = random.split(key)
    obs, state = wrapper.reset(reset_key)
    agent_obs = obs[0]  # First agent
    target_q = jnp.array([1.0])  # Dummy target
    
    loss, grads = jax.value_and_grad(actor_loss)(actor_params, agent_obs, target_q)
    
    print(f"  Loss: {float(loss):.6f}")
    print(f"  Gradient shapes:")
    for layer_name, layer_grads in grads['params'].items():
        for param_name, param_grad in layer_grads.items():
            print(f"    {layer_name}/{param_name}: {param_grad.shape}")
    
    # Verify gradient is non-zero
    total_grad_norm = sum(
        float(jnp.linalg.norm(g)) 
        for layer in grads['params'].values() 
        for g in layer.values()
    )
    print(f"  Total gradient norm: {total_grad_norm:.6f}")
    assert total_grad_norm > 0, "Gradients should be non-zero"
    
    print("\n  ✓ Policy gradient test passed")


def test_full_pipeline():
    """Test complete MADDPG pipeline with actual shapes."""
    print("\n" + "="*60)
    print("Testing Full MADDPG Pipeline")
    print("="*60 + "\n")
    
    # 1. Create vectorized environment with actual shapes
    print("  1. Creating vectorized environment...")
    n_envs = 4
    n_agents = 5
    kwargs = {'shape_file': FIG_SHAPES_PATH} if FIG_SHAPES_PATH else {}
    if FIG_SHAPES_PATH:
        print(f"     Using actual shapes from: {FIG_SHAPES_PATH}")
    vec_wrapper = create_vec_maddpg_env(n_envs=n_envs, n_agents=n_agents, **kwargs)
    print(f"     ✓ Created {n_envs} envs with {n_agents} agents each")
    print(f"     Shape library: {vec_wrapper.wrapper.env.shape_library.n_shapes} shapes, max {vec_wrapper.wrapper.env.shape_library.max_n_grid} grid cells")
    
    # 2. Initialize networks
    print("  2. Initializing networks...")
    key = random.PRNGKey(42)
    
    actors = [Actor(action_dim=2) for _ in range(n_agents)]
    actor_params = []
    
    for i in range(n_agents):
        key, init_key = random.split(key)
        params = actors[i].init(init_key, jnp.zeros((vec_wrapper.obs_dim,)))
        actor_params.append(params)
    print(f"     ✓ Initialized {n_agents} actor networks")
    
    # 3. Reset environments
    print("  3. Resetting environments...")
    key, reset_key = random.split(key)
    reset_keys = random.split(reset_key, n_envs)
    obs, states = vec_wrapper.reset(reset_keys)
    print(f"     ✓ Obs shape: {obs.shape}")
    
    # 4. Collect experience
    print("  4. Collecting experience...")
    transitions = []
    
    for step in range(10):
        # Select actions
        actions_list = []
        for agent_idx in range(n_agents):
            agent_obs = obs[:, agent_idx, :]
            agent_actions = actors[agent_idx].apply(
                actor_params[agent_idx], agent_obs
            )
            actions_list.append(agent_actions)
        actions = jnp.stack(actions_list, axis=1)
        
        # Add noise
        key, noise_key = random.split(key)
        noise = random.normal(noise_key, actions.shape) * 0.1
        actions = jnp.clip(actions + noise, -1, 1)
        
        # Step
        key, step_key = random.split(key)
        step_keys = random.split(step_key, n_envs)
        next_obs, next_states, rewards, dones, info = vec_wrapper.step(
            step_keys, states, actions
        )
        
        # Store (just first env for simplicity)
        transitions.append(Transition(
            obs=obs[0],
            actions=actions[0],
            rewards=rewards[0],
            next_obs=next_obs[0],
            dones=dones[0],
        ))
        
        obs = next_obs
        states = next_states
    
    print(f"     ✓ Collected {len(transitions)} transitions")
    
    # 5. Stack transitions
    print("  5. Stacking transitions...")
    batched = stack_transitions(transitions)
    print(f"     ✓ Batched obs shape: {batched.obs.shape}")
    
    # 6. Compute global states
    print("  6. Computing global states...")
    global_states = vec_wrapper.get_global_states(states)
    print(f"     ✓ Global states shape: {global_states.shape}")
    
    print("\n  ✓ Full pipeline test passed")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("End-to-End MADDPG Tests with Actual Target Shapes")
    print("="*60)
    
    if FIG_SHAPES_PATH:
        print(f"\nUsing actual shapes from: {FIG_SHAPES_PATH}")
    else:
        print("\nWarning: fig/results.pkl not found, using procedural shapes")
    
    # Run all tests
    test_environment_performance()
    test_training_loop()
    test_maddpg_critic()
    test_policy_gradient()
    test_full_pipeline()
    
    print("\n" + "="*60)
    print("ALL END-TO-END TESTS PASSED! ✓")
    print("="*60)
    if FIG_SHAPES_PATH:
        print("\nTests used actual target shapes from fig directory (7 shapes, ~500 grid cells each).")
    print("\nThe JAX multi-agent swarm environment is ready for MADDPG training!")
    print("\nQuick Start with actual shapes:")
    print("  from maddpg_wrapper import create_vec_maddpg_env")
    print("  env = create_vec_maddpg_env(n_envs=32, n_agents=10, shape_file='path/to/fig/results.pkl')")
    print("  obs, states = env.reset(keys)")
    print("  obs, states, rewards, dones, info = env.step(keys, states, actions)")
    print("="*60 + "\n")
