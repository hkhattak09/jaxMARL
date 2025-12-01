"""Tests for networks.py - neural network architectures for MADDPG."""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jax import random

from networks import (
    Actor,
    ActorDiscrete,
    Critic,
    CriticTwin,
    create_actor,
    create_actor_discrete,
    create_critic,
    create_critic_twin,
    create_maddpg_networks,
    create_maddpg_networks_shared_critic,
    count_parameters,
    print_network_summary,
)


def test_actor_creation():
    """Test Actor network creation and initialization."""
    print("Testing Actor creation...")
    
    key = random.PRNGKey(42)
    obs_dim = 10
    action_dim = 2
    hidden_dims = (64, 64, 64)
    
    actor, params = create_actor(key, obs_dim, action_dim, hidden_dims)
    
    # Check that params were created
    assert params is not None, "Params should not be None"
    assert 'params' in params, "Should have 'params' key"
    
    # Check structure
    param_keys = list(params['params'].keys())
    expected_keys = ['fc1', 'fc2', 'fc3', 'fc_out']
    assert param_keys == expected_keys, f"Expected {expected_keys}, got {param_keys}"
    
    print(f"   Actor created with layers: {param_keys}")
    print(f"   Total parameters: {count_parameters(params):,}")
    
    print("   Actor creation: PASSED")
    return True


def test_actor_forward():
    """Test Actor forward pass."""
    print("Testing Actor forward pass...")
    
    key = random.PRNGKey(42)
    obs_dim = 10
    action_dim = 2
    batch_size = 32
    
    actor, params = create_actor(key, obs_dim, action_dim)
    
    # Single observation
    obs_single = jnp.zeros(obs_dim)
    action_single = actor.apply(params, obs_single)
    assert action_single.shape == (action_dim,), f"Single action shape: {action_single.shape}"
    print(f"   Single obs -> action shape: {action_single.shape}")
    
    # Batched observations
    obs_batch = jnp.zeros((batch_size, obs_dim))
    action_batch = actor.apply(params, obs_batch)
    assert action_batch.shape == (batch_size, action_dim), f"Batch action shape: {action_batch.shape}"
    print(f"   Batch ({batch_size}) obs -> action shape: {action_batch.shape}")
    
    # Check output bounds (tanh should be in [-1, 1])
    key, subkey = random.split(key)
    obs_random = random.normal(subkey, (100, obs_dim))
    actions_random = actor.apply(params, obs_random)
    
    assert jnp.all(actions_random >= -1.0), "Actions should be >= -1"
    assert jnp.all(actions_random <= 1.0), "Actions should be <= 1"
    print(f"   Action bounds verified: [{jnp.min(actions_random):.4f}, {jnp.max(actions_random):.4f}]")
    
    print("   Actor forward: PASSED")
    return True


def test_actor_discrete():
    """Test discrete Actor network."""
    print("Testing ActorDiscrete...")
    
    key = random.PRNGKey(42)
    obs_dim = 10
    n_actions = 5
    batch_size = 16
    
    actor, params = create_actor_discrete(key, obs_dim, n_actions)
    
    # Forward pass
    obs = random.normal(key, (batch_size, obs_dim))
    logits = actor.apply(params, obs)
    
    assert logits.shape == (batch_size, n_actions), f"Logits shape: {logits.shape}"
    print(f"   Logits shape: {logits.shape}")
    
    # Convert to probabilities
    probs = jax.nn.softmax(logits, axis=-1)
    assert jnp.allclose(jnp.sum(probs, axis=-1), 1.0), "Probs should sum to 1"
    print(f"   Softmax probs sum: {jnp.sum(probs, axis=-1)[0]:.4f}")
    
    print("   ActorDiscrete: PASSED")
    return True


def test_critic_creation():
    """Test Critic network creation."""
    print("Testing Critic creation...")
    
    key = random.PRNGKey(42)
    obs_dim = 20  # Global state dim
    action_dim = 4  # All agents' actions
    hidden_dims = (64, 64, 64)
    
    critic, params = create_critic(key, obs_dim, action_dim, hidden_dims)
    
    # Check that params were created
    assert params is not None, "Params should not be None"
    
    # Check structure
    param_keys = list(params['params'].keys())
    expected_keys = ['fc1', 'fc2', 'fc3', 'fc_out']
    assert param_keys == expected_keys, f"Expected {expected_keys}, got {param_keys}"
    
    print(f"   Critic created with layers: {param_keys}")
    print(f"   Total parameters: {count_parameters(params):,}")
    
    print("   Critic creation: PASSED")
    return True


def test_critic_forward():
    """Test Critic forward pass."""
    print("Testing Critic forward pass...")
    
    key = random.PRNGKey(42)
    obs_dim = 20
    action_dim = 4
    batch_size = 32
    
    critic, params = create_critic(key, obs_dim, action_dim)
    
    # Single input
    obs_single = jnp.zeros(obs_dim)
    action_single = jnp.zeros(action_dim)
    q_single = critic.apply(params, obs_single, action_single)
    assert q_single.shape == (1,), f"Single Q shape: {q_single.shape}"
    print(f"   Single input -> Q shape: {q_single.shape}")
    
    # Batched input
    obs_batch = jnp.zeros((batch_size, obs_dim))
    action_batch = jnp.zeros((batch_size, action_dim))
    q_batch = critic.apply(params, obs_batch, action_batch)
    assert q_batch.shape == (batch_size, 1), f"Batch Q shape: {q_batch.shape}"
    print(f"   Batch ({batch_size}) input -> Q shape: {q_batch.shape}")
    
    # Q-values should be real numbers (no bounds)
    key, subkey = random.split(key)
    obs_random = random.normal(subkey, (100, obs_dim))
    key, subkey = random.split(key)
    action_random = random.uniform(subkey, (100, action_dim), minval=-1, maxval=1)
    q_random = critic.apply(params, obs_random, action_random)
    
    assert jnp.all(jnp.isfinite(q_random)), "Q-values should be finite"
    print(f"   Q-value range: [{jnp.min(q_random):.4f}, {jnp.max(q_random):.4f}]")
    
    print("   Critic forward: PASSED")
    return True


def test_critic_twin():
    """Test Twin Critic network."""
    print("Testing CriticTwin...")
    
    key = random.PRNGKey(42)
    obs_dim = 20
    action_dim = 4
    batch_size = 16
    
    critic, params = create_critic_twin(key, obs_dim, action_dim)
    
    # Forward pass
    obs = random.normal(key, (batch_size, obs_dim))
    key, subkey = random.split(key)
    action = random.uniform(subkey, (batch_size, action_dim), minval=-1, maxval=1)
    
    q1, q2 = critic.apply(params, obs, action)
    
    assert q1.shape == (batch_size, 1), f"Q1 shape: {q1.shape}"
    assert q2.shape == (batch_size, 1), f"Q2 shape: {q2.shape}"
    print(f"   Q1 shape: {q1.shape}, Q2 shape: {q2.shape}")
    
    # Q1 and Q2 should be different (different parameters)
    assert not jnp.allclose(q1, q2), "Q1 and Q2 should differ"
    print(f"   Q1 != Q2: verified")
    
    # Test q1 only method
    q1_only = critic.apply(params, obs, action, method=critic.q1)
    assert jnp.allclose(q1_only, q1), "q1 method should match"
    print(f"   q1() method: PASSED")
    
    print("   CriticTwin: PASSED")
    return True


def test_maddpg_networks():
    """Test MADDPG network creation utility."""
    print("Testing create_maddpg_networks...")
    
    key = random.PRNGKey(42)
    n_agents = 5
    obs_dim = 10  # Per agent
    action_dim = 2  # Per agent
    
    actor, critic, actor_params, critic_params = create_maddpg_networks(
        key, n_agents, obs_dim, action_dim
    )
    
    # Test actor (local obs -> local action)
    local_obs = jnp.zeros((1, obs_dim))
    local_action = actor.apply(actor_params, local_obs)
    assert local_action.shape == (1, action_dim), f"Local action shape: {local_action.shape}"
    print(f"   Actor: obs({obs_dim}) -> action({action_dim})")
    
    # Test critic (global state + all actions -> Q)
    global_obs = jnp.zeros((1, obs_dim * n_agents))
    all_actions = jnp.zeros((1, action_dim * n_agents))
    q_value = critic.apply(critic_params, global_obs, all_actions)
    assert q_value.shape == (1, 1), f"Q shape: {q_value.shape}"
    print(f"   Critic: global_obs({obs_dim * n_agents}) + all_actions({action_dim * n_agents}) -> Q")
    
    print(f"   Actor params: {count_parameters(actor_params):,}")
    print(f"   Critic params: {count_parameters(critic_params):,}")
    
    print("   create_maddpg_networks: PASSED")
    return True


def test_maddpg_networks_shared_critic():
    """Test MADDPG with explicit global state dimension."""
    print("Testing create_maddpg_networks_shared_critic...")
    
    key = random.PRNGKey(42)
    n_agents = 5
    obs_dim = 10  # Per agent
    action_dim = 2  # Per agent
    global_state_dim = 50  # Different from n_agents * obs_dim
    
    actor, critic, actor_params, critic_params = create_maddpg_networks_shared_critic(
        key, n_agents, obs_dim, action_dim, global_state_dim
    )
    
    # Test actor
    local_obs = jnp.zeros((1, obs_dim))
    local_action = actor.apply(actor_params, local_obs)
    assert local_action.shape == (1, action_dim), f"Local action shape: {local_action.shape}"
    print(f"   Actor: obs({obs_dim}) -> action({action_dim})")
    
    # Test critic with global state
    global_state = jnp.zeros((1, global_state_dim))
    all_actions = jnp.zeros((1, action_dim * n_agents))
    q_value = critic.apply(critic_params, global_state, all_actions)
    assert q_value.shape == (1, 1), f"Q shape: {q_value.shape}"
    print(f"   Critic: global_state({global_state_dim}) + all_actions({action_dim * n_agents}) -> Q")
    
    print("   create_maddpg_networks_shared_critic: PASSED")
    return True


def test_gradient_computation():
    """Test that gradients can be computed through networks."""
    print("Testing gradient computation...")
    
    key = random.PRNGKey(42)
    obs_dim = 10
    action_dim = 2
    batch_size = 8
    
    actor, actor_params = create_actor(key, obs_dim, action_dim)
    key, subkey = random.split(key)
    critic, critic_params = create_critic(subkey, obs_dim, action_dim)
    
    # Test actor gradient
    def actor_loss(params, obs):
        actions = actor.apply(params, obs)
        return jnp.mean(actions ** 2)  # Dummy loss
    
    obs = random.normal(key, (batch_size, obs_dim))
    actor_grad = jax.grad(actor_loss)(actor_params, obs)
    
    assert actor_grad is not None, "Actor grad should not be None"
    # Check that gradients exist for all layers
    for layer_name in actor_params['params'].keys():
        assert layer_name in actor_grad['params'], f"Missing grad for {layer_name}"
    print("   Actor gradients: computed successfully")
    
    # Test critic gradient
    def critic_loss(params, obs, action):
        q = critic.apply(params, obs, action)
        return jnp.mean(q ** 2)  # Dummy loss
    
    action = random.uniform(key, (batch_size, action_dim), minval=-1, maxval=1)
    critic_grad = jax.grad(critic_loss)(critic_params, obs, action)
    
    assert critic_grad is not None, "Critic grad should not be None"
    for layer_name in critic_params['params'].keys():
        assert layer_name in critic_grad['params'], f"Missing grad for {layer_name}"
    print("   Critic gradients: computed successfully")
    
    # Test actor-critic gradient (policy gradient style)
    def policy_gradient_loss(actor_params, critic_params, obs):
        actions = actor.apply(actor_params, obs)
        q_values = critic.apply(critic_params, obs, actions)
        return -jnp.mean(q_values)  # Maximize Q
    
    policy_grad = jax.grad(policy_gradient_loss)(actor_params, critic_params, obs)
    assert policy_grad is not None, "Policy grad should not be None"
    print("   Policy gradient through critic: computed successfully")
    
    print("   Gradient computation: PASSED")
    return True


def test_jit_compilation():
    """Test JIT compilation of network forward passes."""
    print("Testing JIT compilation...")
    
    key = random.PRNGKey(42)
    obs_dim = 10
    action_dim = 2
    batch_size = 32
    
    actor, actor_params = create_actor(key, obs_dim, action_dim)
    key, subkey = random.split(key)
    critic, critic_params = create_critic(subkey, obs_dim, action_dim)
    
    # JIT actor
    @jax.jit
    def actor_forward(params, obs):
        return actor.apply(params, obs)
    
    obs = random.normal(key, (batch_size, obs_dim))
    
    # First call compiles
    action = actor_forward(actor_params, obs)
    # Second call uses cached compilation
    action = actor_forward(actor_params, obs)
    
    assert action.shape == (batch_size, action_dim), "JIT actor failed"
    print("   Actor JIT: PASSED")
    
    # JIT critic
    @jax.jit
    def critic_forward(params, obs, action):
        return critic.apply(params, obs, action)
    
    action_input = random.uniform(key, (batch_size, action_dim), minval=-1, maxval=1)
    q = critic_forward(critic_params, obs, action_input)
    q = critic_forward(critic_params, obs, action_input)
    
    assert q.shape == (batch_size, 1), "JIT critic failed"
    print("   Critic JIT: PASSED")
    
    print("   JIT compilation: ALL PASSED")
    return True


def test_vmap_multi_agent():
    """Test vmap for parallel multi-agent forward passes."""
    print("Testing vmap for multi-agent...")
    
    key = random.PRNGKey(42)
    n_agents = 4
    obs_dim = 10
    action_dim = 2
    
    # Create separate actor params for each agent
    keys = random.split(key, n_agents)
    actors_and_params = [create_actor(k, obs_dim, action_dim) for k in keys]
    actor = actors_and_params[0][0]  # All actors have same architecture
    all_params = [ap[1] for ap in actors_and_params]
    
    # Stack params for vmap (this is a bit tricky with PyTrees)
    # For simplicity, we'll vmap over observations instead
    
    # Vmap actor over batch of agent observations
    obs_all_agents = random.normal(key, (n_agents, obs_dim))
    
    # Using single actor params, vmap over observations
    vmapped_actor = jax.vmap(lambda obs: actor.apply(all_params[0], obs))
    actions = vmapped_actor(obs_all_agents)
    
    assert actions.shape == (n_agents, action_dim), f"Vmap actions shape: {actions.shape}"
    print(f"   Vmap over {n_agents} agents: actions shape {actions.shape}")
    
    # Alternative: vmap over both params and observations (more complex)
    # This requires stacking params which is architecture-specific
    
    print("   vmap multi-agent: PASSED")
    return True


def test_different_hidden_dims():
    """Test networks with different hidden layer configurations."""
    print("Testing different hidden_dims configurations...")
    
    key = random.PRNGKey(42)
    obs_dim = 10
    action_dim = 2
    
    configs = [
        (32,),           # Single small layer
        (64, 64),        # Two layers
        (128, 128, 128), # Three larger layers
        (256, 128, 64),  # Decreasing layers
    ]
    
    for hidden_dims in configs:
        actor, params = create_actor(key, obs_dim, action_dim, hidden_dims)
        n_params = count_parameters(params)
        
        # Forward pass
        obs = random.normal(key, (1, obs_dim))
        action = actor.apply(params, obs)
        
        assert action.shape == (1, action_dim), f"Failed for hidden_dims={hidden_dims}"
        print(f"   hidden_dims={hidden_dims}: {n_params:,} params")
    
    print("   Different hidden_dims: PASSED")
    return True


def test_parameter_counting():
    """Test parameter counting utility."""
    print("Testing count_parameters...")
    
    key = random.PRNGKey(42)
    obs_dim = 10
    action_dim = 2
    hidden_dims = (64, 64)
    
    actor, params = create_actor(key, obs_dim, action_dim, hidden_dims)
    
    # Manual calculation:
    # fc1: obs_dim * 64 + 64 = 10 * 64 + 64 = 704
    # fc2: 64 * 64 + 64 = 4160
    # fc_out: 64 * action_dim + action_dim = 64 * 2 + 2 = 130
    # Total: 704 + 4160 + 130 = 4994
    
    expected_params = (obs_dim * 64 + 64) + (64 * 64 + 64) + (64 * action_dim + action_dim)
    actual_params = count_parameters(params)
    
    assert actual_params == expected_params, f"Expected {expected_params}, got {actual_params}"
    print(f"   Expected params: {expected_params}, actual: {actual_params}")
    
    print("   count_parameters: PASSED")
    return True


def test_print_network_summary():
    """Test network summary printing."""
    print("Testing print_network_summary...")
    
    key = random.PRNGKey(42)
    actor, params = create_actor(key, 10, 2, (64, 64))
    
    print_network_summary("Actor", params)
    
    print("   print_network_summary: PASSED")
    return True


def test_layer_norm():
    """Test networks with layer normalization."""
    print("Testing layer normalization...")
    
    key = random.PRNGKey(42)
    obs_dim = 10
    action_dim = 2
    batch_size = 16
    
    # Actor with layer norm
    actor = Actor(action_dim=action_dim, use_layer_norm=True)
    dummy_obs = jnp.zeros((1, obs_dim))
    params = actor.init(key, dummy_obs)
    
    # Check layer norm params exist
    assert 'ln1' in params['params'], "Layer norm 1 should exist"
    assert 'ln2' in params['params'], "Layer norm 2 should exist"
    print("   Layer norm params exist: PASSED")
    
    # Forward pass
    obs = random.normal(key, (batch_size, obs_dim))
    actions = actor.apply(params, obs)
    assert actions.shape == (batch_size, action_dim)
    print(f"   Forward with layer norm: shape {actions.shape}")
    
    # Critic with layer norm
    key, subkey = random.split(key)
    critic = Critic(use_layer_norm=True)
    dummy_action = jnp.zeros((1, action_dim))
    critic_params = critic.init(subkey, dummy_obs, dummy_action)
    
    q = critic.apply(critic_params, obs, random.uniform(key, (batch_size, action_dim)))
    assert q.shape == (batch_size, 1)
    print(f"   Critic with layer norm: Q shape {q.shape}")
    
    print("   layer normalization: PASSED")
    return True


def test_dropout():
    """Test networks with dropout."""
    print("Testing dropout...")
    
    key = random.PRNGKey(42)
    obs_dim = 10
    action_dim = 2
    batch_size = 16
    
    # Actor with dropout
    actor = Actor(action_dim=action_dim, dropout_rate=0.5)
    dummy_obs = jnp.zeros((1, obs_dim))
    params = actor.init(key, dummy_obs)
    
    obs = random.normal(key, (batch_size, obs_dim))
    
    # Training mode (dropout active) - needs rngs
    actions_train = actor.apply(params, obs, training=True, rngs={'dropout': key})
    
    # Inference mode (no dropout)
    actions_eval = actor.apply(params, obs, training=False)
    
    assert actions_train.shape == actions_eval.shape == (batch_size, action_dim)
    print(f"   Dropout training shape: {actions_train.shape}")
    print(f"   Dropout eval shape: {actions_eval.shape}")
    
    print("   dropout: PASSED")
    return True


def test_create_all_agents_networks():
    """Test creating networks for all agents."""
    print("Testing create_all_agents_networks...")
    
    from networks import create_all_agents_networks
    
    key = random.PRNGKey(42)
    n_agents = 4
    obs_dim = 10
    action_dim = 2
    
    actors, critics, actor_params, critic_params = create_all_agents_networks(
        key, n_agents, obs_dim, action_dim
    )
    
    assert len(actors) == n_agents, f"Expected {n_agents} actors"
    assert len(critics) == n_agents, f"Expected {n_agents} critics"
    assert len(actor_params) == n_agents
    assert len(critic_params) == n_agents
    
    print(f"   Created {n_agents} actor-critic pairs")
    
    # Test each agent's network
    obs = jnp.zeros((1, obs_dim))
    global_obs = jnp.zeros((1, obs_dim * n_agents))
    all_actions = jnp.zeros((1, action_dim * n_agents))
    
    for i in range(n_agents):
        action = actors[i].apply(actor_params[i], obs)
        assert action.shape == (1, action_dim)
        
        q = critics[i].apply(critic_params[i], global_obs, all_actions)
        assert q.shape == (1, 1)
    
    print("   All agent networks verified")
    
    print("   create_all_agents_networks: PASSED")
    return True


def test_shared_critic_networks():
    """Test creating networks with shared critic."""
    print("Testing create_shared_critic_networks...")
    
    from networks import create_shared_critic_networks
    
    key = random.PRNGKey(42)
    n_agents = 4
    obs_dim = 10
    action_dim = 2
    
    actors, critic, actor_params, critic_params = create_shared_critic_networks(
        key, n_agents, obs_dim, action_dim
    )
    
    assert len(actors) == n_agents
    assert len(actor_params) == n_agents
    # Single shared critic
    assert not isinstance(critic, list)
    assert not isinstance(critic_params, list)
    
    print(f"   Created {n_agents} actors with 1 shared critic")
    
    print("   create_shared_critic_networks: PASSED")
    return True


def test_actor_critic_combined():
    """Test combined ActorCritic network."""
    print("Testing ActorCritic combined network...")
    
    from networks import ActorCritic
    
    key = random.PRNGKey(42)
    obs_dim = 10
    action_dim = 2
    batch_size = 16
    
    ac = ActorCritic(action_dim=action_dim, hidden_dims=(64, 64, 32))
    dummy_obs = jnp.zeros((1, obs_dim))
    params = ac.init(key, dummy_obs)
    
    # Forward pass
    obs = random.normal(key, (batch_size, obs_dim))
    actions, values = ac.apply(params, obs)
    
    assert actions.shape == (batch_size, action_dim)
    assert values.shape == (batch_size, 1)
    
    print(f"   Actions shape: {actions.shape}")
    print(f"   Values shape: {values.shape}")
    
    # Check shared layers exist
    assert 'shared_fc1' in params['params']
    assert 'actor_fc' in params['params']
    assert 'critic_fc' in params['params']
    print("   Shared + separate heads verified")
    
    print("   ActorCritic: PASSED")
    return True


def test_critic_twin_q_min():
    """Test CriticTwin q_min method."""
    print("Testing CriticTwin q_min...")
    
    key = random.PRNGKey(42)
    obs_dim = 10
    action_dim = 4
    batch_size = 16
    
    critic, params = create_critic_twin(key, obs_dim, action_dim)
    
    obs = random.normal(key, (batch_size, obs_dim))
    key, subkey = random.split(key)
    action = random.uniform(subkey, (batch_size, action_dim))
    
    q1, q2 = critic.apply(params, obs, action)
    q_min = critic.apply(params, obs, action, method=critic.q_min)
    
    # q_min should be element-wise minimum
    expected_q_min = jnp.minimum(q1, q2)
    assert jnp.allclose(q_min, expected_q_min)
    
    print(f"   q_min verified: matches jnp.minimum(q1, q2)")
    
    print("   CriticTwin q_min: PASSED")
    return True


def test_get_activation_fn():
    """Test activation function getter."""
    print("Testing get_activation_fn...")
    
    from networks import get_activation_fn
    import flax.linen as nn
    
    assert get_activation_fn('relu') == nn.relu
    assert get_activation_fn('leaky_relu') == nn.leaky_relu
    assert get_activation_fn('tanh') == nn.tanh
    assert get_activation_fn('gelu') == nn.gelu
    assert get_activation_fn('unknown') == nn.leaky_relu  # default
    
    print("   All activation functions verified")
    
    print("   get_activation_fn: PASSED")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running networks.py tests")
    print("=" * 60)
    
    tests = [
        test_actor_creation,
        test_actor_forward,
        test_actor_discrete,
        test_critic_creation,
        test_critic_forward,
        test_critic_twin,
        test_maddpg_networks,
        test_maddpg_networks_shared_critic,
        test_gradient_computation,
        test_jit_compilation,
        test_vmap_multi_agent,
        test_different_hidden_dims,
        test_parameter_counting,
        test_print_network_summary,
        # New tests
        test_layer_norm,
        test_dropout,
        test_create_all_agents_networks,
        test_shared_critic_networks,
        test_actor_critic_combined,
        test_critic_twin_q_min,
        test_get_activation_fn,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"   FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
