"""Tests for agents.py - DDPG agent implementation."""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jax import random
import pytest

from agents import (
    DDPGAgentState,
    AgentConfig,
    create_agent,
    create_agent_with_networks,
    select_action,
    select_action_with_noise,
    select_target_action,
    compute_critic_loss,
    compute_actor_loss,
    update_critic,
    update_actor,
    update_targets,
    reset_noise,
    get_noise_scale,
    create_all_agents,
    select_all_actions,
    select_all_actions_with_noise,
    get_agent_params,
    load_agent_params,
)
from networks import Actor, Critic
from noise import OUNoiseParams


class TestAgentCreation:
    """Tests for agent creation functions."""
    
    def test_create_agent_continuous(self):
        """Test creating a continuous action agent."""
        print("Testing create_agent (continuous)...")
        
        key = random.PRNGKey(42)
        config = AgentConfig(
            obs_dim=10,
            action_dim=2,
            critic_input_dim=20,  # obs_dim (10) + total_action_dim (10)
            hidden_dims=(64, 64),
            lr_actor=1e-4,
            lr_critic=1e-3,
            discrete_action=False,
        )
        
        agent_state, actor, critic = create_agent(key, config)
        
        # Check state structure
        assert agent_state.actor_params is not None
        assert agent_state.critic_params is not None
        assert agent_state.target_actor_params is not None
        assert agent_state.target_critic_params is not None
        assert agent_state.actor_opt_state is not None
        assert agent_state.critic_opt_state is not None
        assert agent_state.step == 0
        
        # Check networks can forward pass
        dummy_obs = jnp.zeros((1, 10))
        dummy_critic_obs = jnp.zeros((1, 10))
        dummy_critic_action = jnp.zeros((1, 10))  # total_action_dim
        
        action = actor.apply(agent_state.actor_params, dummy_obs)
        assert action.shape == (1, 2)
        
        q_value = critic.apply(agent_state.critic_params, dummy_critic_obs, dummy_critic_action)
        assert q_value.shape == (1, 1)
        
        print("   create_agent (continuous): PASSED")
    
    def test_create_agent_discrete(self):
        """Test creating a discrete action agent."""
        print("Testing create_agent (discrete)...")
        
        key = random.PRNGKey(42)
        config = AgentConfig(
            obs_dim=10,
            action_dim=5,  # 5 discrete actions
            critic_input_dim=15,  # obs_dim (10) + action_dim (5)
            hidden_dims=(64, 64),
            discrete_action=True,
        )
        
        agent_state, actor, critic = create_agent(key, config)
        
        # Check discrete actor produces logits
        dummy_obs = jnp.zeros((1, 10))
        logits = actor.apply(agent_state.actor_params, dummy_obs)
        assert logits.shape == (1, 5)
        
        # Logits should sum to ~1 after softmax
        probs = jax.nn.softmax(logits)
        assert jnp.isclose(jnp.sum(probs), 1.0, atol=1e-5)
        
        print("   create_agent (discrete): PASSED")
    
    def test_create_agent_with_ou_noise(self):
        """Test creating an agent with OU noise."""
        print("Testing create_agent with OU noise...")
        
        key = random.PRNGKey(42)
        config = AgentConfig(
            obs_dim=10,
            action_dim=2,
            critic_input_dim=14,  # 10 + 4
            noise_type='ou',
        )
        
        agent_state, actor, critic = create_agent(key, config)
        
        # Should have initialized OU noise state
        assert agent_state.noise_state is not None
        assert agent_state.noise_state.state.shape == (2,)
        
        print("   create_agent with OU noise: PASSED")
    
    def test_create_agent_with_layer_norm(self):
        """Test creating an agent with layer normalization."""
        print("Testing create_agent with layer norm...")
        
        key = random.PRNGKey(42)
        config = AgentConfig(
            obs_dim=10,
            action_dim=2,
            critic_input_dim=14,
            use_layer_norm=True,
        )
        
        agent_state, actor, critic = create_agent(key, config)
        
        # Check that layer norm params exist
        param_names = str(agent_state.actor_params)
        assert 'LayerNorm' in param_names or 'ln' in param_names.lower()
        
        print("   create_agent with layer norm: PASSED")
    
    def test_create_agent_with_grad_clipping(self):
        """Test creating an agent with gradient clipping."""
        print("Testing create_agent with grad clipping...")
        
        key = random.PRNGKey(42)
        config = AgentConfig(
            obs_dim=10,
            action_dim=2,
            critic_input_dim=14,
            max_grad_norm=0.5,
        )
        
        agent_state, actor, critic = create_agent(key, config)
        
        # Agent should be created successfully
        assert agent_state.actor_opt_state is not None
        
        print("   create_agent with grad clipping: PASSED")


class TestActionSelection:
    """Tests for action selection functions."""
    
    def test_select_action_single(self):
        """Test selecting action for single observation."""
        print("Testing select_action (single)...")
        
        key = random.PRNGKey(42)
        config = AgentConfig(obs_dim=10, action_dim=2, critic_input_dim=14)
        agent_state, actor, _ = create_agent(key, config)
        
        obs = jnp.zeros(10)
        action = select_action(actor, agent_state.actor_params, obs)
        
        assert action.shape == (2,)
        assert jnp.all(action >= -1) and jnp.all(action <= 1)
        
        print("   select_action (single): PASSED")
    
    def test_select_action_batch(self):
        """Test selecting actions for batch of observations."""
        print("Testing select_action (batch)...")
        
        key = random.PRNGKey(42)
        config = AgentConfig(obs_dim=10, action_dim=2, critic_input_dim=14)
        agent_state, actor, _ = create_agent(key, config)
        
        obs = jnp.zeros((32, 10))
        actions = select_action(actor, agent_state.actor_params, obs)
        
        assert actions.shape == (32, 2)
        
        print("   select_action (batch): PASSED")
    
    def test_select_action_with_gaussian_noise(self):
        """Test action selection with Gaussian noise."""
        print("Testing select_action_with_noise (Gaussian)...")
        
        key = random.PRNGKey(42)
        config = AgentConfig(obs_dim=10, action_dim=2, critic_input_dim=14)
        agent_state, actor, _ = create_agent(key, config)
        
        obs = jnp.zeros(10)
        action, log_prob, new_noise_state = select_action_with_noise(
            key=key,
            actor=actor,
            actor_params=agent_state.actor_params,
            obs=obs,
            noise_scale=0.1,
            noise_type='gaussian',
        )
        
        assert action.shape == (2,)
        assert log_prob.shape == (1,)
        assert new_noise_state is None  # Gaussian doesn't need state
        
        print(f"   Action: {action}, Log prob: {log_prob}")
        print("   select_action_with_noise (Gaussian): PASSED")
    
    def test_select_action_with_ou_noise(self):
        """Test action selection with OU noise."""
        print("Testing select_action_with_noise (OU)...")
        
        key = random.PRNGKey(42)
        config = AgentConfig(obs_dim=10, action_dim=2, critic_input_dim=14, noise_type='ou')
        agent_state, actor, _ = create_agent(key, config)
        
        obs = jnp.zeros(10)
        action, log_prob, new_noise_state = select_action_with_noise(
            key=key,
            actor=actor,
            actor_params=agent_state.actor_params,
            obs=obs,
            noise_scale=0.1,
            noise_type='ou',
            noise_state=agent_state.noise_state,
        )
        
        assert action.shape == (2,)
        assert new_noise_state is not None
        assert new_noise_state.state.shape == (2,)
        
        print(f"   Action: {action}")
        print("   select_action_with_noise (OU): PASSED")
    
    def test_select_action_epsilon_greedy(self):
        """Test epsilon-greedy action selection."""
        print("Testing select_action_with_noise (epsilon-greedy)...")
        
        key = random.PRNGKey(42)
        config = AgentConfig(obs_dim=10, action_dim=2, critic_input_dim=14)
        agent_state, actor, _ = create_agent(key, config)
        
        obs = jnp.zeros(10)
        
        # With epsilon=1.0, should always get random action
        random_count = 0
        for i in range(100):
            k = random.PRNGKey(i)
            action, _, _ = select_action_with_noise(
                key=k,
                actor=actor,
                actor_params=agent_state.actor_params,
                obs=obs,
                noise_scale=0.01,  # Small noise
                epsilon=1.0,
            )
            # Random actions should vary more
            random_count += 1
        
        print("   select_action_with_noise (epsilon-greedy): PASSED")
    
    def test_select_target_action(self):
        """Test target action selection."""
        print("Testing select_target_action...")
        
        key = random.PRNGKey(42)
        config = AgentConfig(obs_dim=10, action_dim=2, critic_input_dim=14)
        agent_state, actor, _ = create_agent(key, config)
        
        obs = jnp.zeros((16, 10))
        target_actions = select_target_action(actor, agent_state.target_actor_params, obs)
        
        assert target_actions.shape == (16, 2)
        
        # Initially, target should equal policy
        policy_actions = select_action(actor, agent_state.actor_params, obs)
        assert jnp.allclose(target_actions, policy_actions)
        
        print("   select_target_action: PASSED")


class TestCriticUpdate:
    """Tests for critic update functions."""
    
    def test_compute_critic_loss(self):
        """Test critic loss computation."""
        print("Testing compute_critic_loss...")
        
        key = random.PRNGKey(42)
        config = AgentConfig(obs_dim=10, action_dim=2, critic_input_dim=14)  # 10 + 4 (2 agents * 2 actions)
        agent_state, actor, critic = create_agent(key, config)
        
        batch_size = 32
        
        # Create dummy data
        global_obs = random.normal(key, (batch_size, 10))
        all_actions = random.uniform(key, (batch_size, 4), minval=-1, maxval=1)  # 2 agents * 2 actions
        rewards = random.normal(key, (batch_size, 1))
        next_global_obs = random.normal(key, (batch_size, 10))
        next_all_actions = random.uniform(key, (batch_size, 4), minval=-1, maxval=1)
        dones = jnp.zeros((batch_size, 1))
        
        loss, info = compute_critic_loss(
            critic=critic,
            critic_params=agent_state.critic_params,
            target_critic_params=agent_state.target_critic_params,
            actor=actor,
            target_actor_params=agent_state.target_actor_params,
            global_obs=global_obs,
            all_actions=all_actions,
            rewards=rewards,
            next_global_obs=next_global_obs,
            next_all_actions=next_all_actions,
            dones=dones,
            gamma=0.99,
        )
        
        assert loss.shape == ()
        assert loss >= 0
        assert 'q_values' in info
        assert 'target_q_values' in info
        assert 'td_error' in info
        
        print(f"   Critic loss: {loss:.4f}")
        print("   compute_critic_loss: PASSED")
    
    def test_update_critic(self):
        """Test critic update."""
        print("Testing update_critic...")
        
        key = random.PRNGKey(42)
        config = AgentConfig(obs_dim=10, action_dim=2, critic_input_dim=14, lr_critic=1e-3)
        agent_state, actor, critic = create_agent(key, config)
        
        import optax
        optimizer = optax.adam(config.lr_critic)
        
        batch_size = 32
        global_obs = random.normal(key, (batch_size, 10))
        all_actions = random.uniform(key, (batch_size, 4), minval=-1, maxval=1)
        rewards = random.normal(key, (batch_size, 1))
        next_global_obs = random.normal(key, (batch_size, 10))
        next_all_actions = random.uniform(key, (batch_size, 4), minval=-1, maxval=1)
        dones = jnp.zeros((batch_size, 1))
        
        new_state, info = update_critic(
            agent_state=agent_state,
            critic=critic,
            actor=actor,
            optimizer=optimizer,
            global_obs=global_obs,
            all_actions=all_actions,
            rewards=rewards,
            next_global_obs=next_global_obs,
            next_all_actions=next_all_actions,
            dones=dones,
            gamma=0.99,
        )
        
        # Parameters should have changed
        old_params = jax.tree_util.tree_leaves(agent_state.critic_params)
        new_params = jax.tree_util.tree_leaves(new_state.critic_params)
        
        params_changed = any(
            not jnp.allclose(old, new) 
            for old, new in zip(old_params, new_params)
        )
        assert params_changed, "Critic params should change after update"
        
        # Target params should NOT have changed
        old_target = jax.tree_util.tree_leaves(agent_state.target_critic_params)
        new_target = jax.tree_util.tree_leaves(new_state.target_critic_params)
        targets_same = all(
            jnp.allclose(old, new)
            for old, new in zip(old_target, new_target)
        )
        assert targets_same, "Target critic params should not change"
        
        print(f"   Critic loss: {info['critic_loss']:.4f}")
        print(f"   Grad norm: {info['critic_grad_norm']:.4f}")
        print("   update_critic: PASSED")


class TestActorUpdate:
    """Tests for actor update functions."""
    
    def test_compute_actor_loss(self):
        """Test actor loss computation."""
        print("Testing compute_actor_loss...")
        
        key = random.PRNGKey(42)
        config = AgentConfig(obs_dim=10, action_dim=2, critic_input_dim=14)  # 10 + 4
        agent_state, actor, critic = create_agent(key, config)
        
        batch_size = 32
        
        # Create dummy data
        global_obs = random.normal(key, (batch_size, 10))
        agent_obs = random.normal(key, (batch_size, 10))
        other_actions = random.uniform(key, (batch_size, 2), minval=-1, maxval=1)  # One other agent
        
        loss, info = compute_actor_loss(
            actor=actor,
            actor_params=agent_state.actor_params,
            critic=critic,
            critic_params=agent_state.critic_params,
            global_obs=global_obs,
            agent_obs=agent_obs,
            all_actions_except_agent=other_actions,
            agent_action_idx=0,
            action_dim=2,
        )
        
        assert loss.shape == ()
        assert 'policy_loss' in info
        assert 'q_value_mean' in info
        
        print(f"   Actor loss: {loss:.4f}")
        print("   compute_actor_loss: PASSED")
    
    def test_compute_actor_loss_with_prior(self):
        """Test actor loss with prior regularization."""
        print("Testing compute_actor_loss with prior...")
        
        key = random.PRNGKey(42)
        config = AgentConfig(obs_dim=10, action_dim=2, critic_input_dim=14)
        agent_state, actor, critic = create_agent(key, config)
        
        batch_size = 32
        
        global_obs = random.normal(key, (batch_size, 10))
        agent_obs = random.normal(key, (batch_size, 10))
        other_actions = random.uniform(key, (batch_size, 2), minval=-1, maxval=1)
        action_prior = random.uniform(key, (batch_size, 2), minval=-1, maxval=1)
        
        # Without prior
        loss_no_prior, _ = compute_actor_loss(
            actor=actor,
            actor_params=agent_state.actor_params,
            critic=critic,
            critic_params=agent_state.critic_params,
            global_obs=global_obs,
            agent_obs=agent_obs,
            all_actions_except_agent=other_actions,
            agent_action_idx=0,
            action_dim=2,
        )
        
        # With prior
        loss_with_prior, info = compute_actor_loss(
            actor=actor,
            actor_params=agent_state.actor_params,
            critic=critic,
            critic_params=agent_state.critic_params,
            global_obs=global_obs,
            agent_obs=agent_obs,
            all_actions_except_agent=other_actions,
            agent_action_idx=0,
            action_dim=2,
            action_prior=action_prior,
            prior_weight=0.5,
        )
        
        # Loss with prior should be different
        assert not jnp.isclose(loss_no_prior, loss_with_prior)
        assert info['reg_loss'] > 0
        
        print(f"   Loss without prior: {loss_no_prior:.4f}")
        print(f"   Loss with prior: {loss_with_prior:.4f}")
        print(f"   Reg loss: {info['reg_loss']:.4f}")
        print("   compute_actor_loss with prior: PASSED")
    
    def test_update_actor(self):
        """Test actor update."""
        print("Testing update_actor...")
        
        key = random.PRNGKey(42)
        config = AgentConfig(obs_dim=10, action_dim=2, critic_input_dim=14, lr_actor=1e-4)
        agent_state, actor, critic = create_agent(key, config)
        
        import optax
        optimizer = optax.adam(config.lr_actor)
        
        batch_size = 32
        global_obs = random.normal(key, (batch_size, 10))
        agent_obs = random.normal(key, (batch_size, 10))
        other_actions = random.uniform(key, (batch_size, 2), minval=-1, maxval=1)
        
        new_state, info = update_actor(
            agent_state=agent_state,
            actor=actor,
            critic=critic,
            optimizer=optimizer,
            global_obs=global_obs,
            agent_obs=agent_obs,
            all_actions_except_agent=other_actions,
            agent_action_idx=0,
            action_dim=2,
        )
        
        # Parameters should have changed
        old_params = jax.tree_util.tree_leaves(agent_state.actor_params)
        new_params = jax.tree_util.tree_leaves(new_state.actor_params)
        
        params_changed = any(
            not jnp.allclose(old, new)
            for old, new in zip(old_params, new_params)
        )
        assert params_changed, "Actor params should change after update"
        
        print(f"   Actor loss: {info['actor_loss']:.4f}")
        print(f"   Grad norm: {info['actor_grad_norm']:.4f}")
        print("   update_actor: PASSED")


class TestTargetUpdate:
    """Tests for target network updates."""
    
    def test_update_targets(self):
        """Test soft target update."""
        print("Testing update_targets...")
        
        key = random.PRNGKey(42)
        # critic_input_dim = global_obs_dim (10) + action_dim (2) = 12
        config = AgentConfig(obs_dim=10, action_dim=2, critic_input_dim=12, tau=0.01)
        agent_state, actor, critic = create_agent(key, config)
        
        # Modify actor params to be different from target
        import optax
        optimizer = optax.adam(1e-3)
        
        # Do a fake update to make params different
        batch_size = 32
        global_obs = random.normal(key, (batch_size, 10))
        agent_obs = random.normal(key, (batch_size, 10))
        
        # all_actions_except_agent is empty since we only have this agent's action
        new_state, _ = update_actor(
            agent_state=agent_state,
            actor=actor,
            critic=critic,
            optimizer=optimizer,
            global_obs=global_obs,
            agent_obs=agent_obs,
            all_actions_except_agent=jnp.zeros((batch_size, 0)),
            agent_action_idx=0,
            action_dim=2,
        )
        
        # Now update targets
        updated_state = update_targets(new_state, tau=0.01)
        
        # Targets should have moved toward current params
        old_target = jax.tree_util.tree_leaves(new_state.target_actor_params)
        new_target = jax.tree_util.tree_leaves(updated_state.target_actor_params)
        current = jax.tree_util.tree_leaves(updated_state.actor_params)
        
        # Check soft update formula: target' = tau * current + (1-tau) * target
        for old_t, new_t, curr in zip(old_target, new_target, current):
            expected = 0.01 * curr + 0.99 * old_t
            assert jnp.allclose(new_t, expected, atol=1e-5)
        
        # Step should increment
        assert updated_state.step == new_state.step + 1
        
        print("   update_targets: PASSED")


class TestNoiseManagement:
    """Tests for noise management functions."""
    
    def test_reset_noise(self):
        """Test resetting OU noise."""
        print("Testing reset_noise...")
        
        key = random.PRNGKey(42)
        config = AgentConfig(obs_dim=10, action_dim=2, critic_input_dim=14, noise_type='ou')
        agent_state, actor, _ = create_agent(key, config)
        
        # Run some noise steps
        obs = jnp.zeros(10)
        for i in range(10):
            k = random.PRNGKey(i)
            _, _, new_noise_state = select_action_with_noise(
                key=k,
                actor=actor,
                actor_params=agent_state.actor_params,
                obs=obs,
                noise_scale=0.1,
                noise_type='ou',
                noise_state=agent_state.noise_state,
            )
            agent_state = agent_state.replace(noise_state=new_noise_state)
        
        # Noise state should have drifted from 0
        assert not jnp.allclose(agent_state.noise_state.state, 0.0)
        
        # Reset noise
        reset_state = reset_noise(agent_state, action_dim=2)
        
        # Should be back to 0
        assert jnp.allclose(reset_state.noise_state.state, 0.0)
        
        print("   reset_noise: PASSED")
    
    def test_get_noise_scale(self):
        """Test noise scale scheduling."""
        print("Testing get_noise_scale...")
        
        initial = 0.3
        final = 0.05
        total_steps = 1000
        
        # Linear schedule
        scale_0 = get_noise_scale(0, initial, final, total_steps, 'linear')
        scale_500 = get_noise_scale(500, initial, final, total_steps, 'linear')
        scale_1000 = get_noise_scale(1000, initial, final, total_steps, 'linear')
        
        assert jnp.isclose(scale_0, initial, atol=1e-5)
        assert jnp.isclose(scale_500, (initial + final) / 2, atol=1e-3)
        assert jnp.isclose(scale_1000, final, atol=1e-5)
        
        print(f"   Linear: {scale_0:.3f} -> {scale_500:.3f} -> {scale_1000:.3f}")
        
        # Cosine schedule
        scale_cos_500 = get_noise_scale(500, initial, final, total_steps, 'cosine')
        print(f"   Cosine at 500: {scale_cos_500:.3f}")
        
        print("   get_noise_scale: PASSED")


class TestMultiAgent:
    """Tests for multi-agent utilities."""
    
    def test_create_all_agents(self):
        """Test creating multiple agents."""
        print("Testing create_all_agents...")
        
        key = random.PRNGKey(42)
        
        configs = [
            AgentConfig(obs_dim=10, action_dim=2, critic_input_dim=16),  # 10 + 6 (3 agents * 2)
            AgentConfig(obs_dim=10, action_dim=2, critic_input_dim=16),
            AgentConfig(obs_dim=10, action_dim=2, critic_input_dim=16),
        ]
        
        agent_states, actors, critics = create_all_agents(key, configs)
        
        assert len(agent_states) == 3
        assert len(actors) == 3
        assert len(critics) == 3
        
        # Each agent should have different params
        params_0 = jax.tree_util.tree_leaves(agent_states[0].actor_params)
        params_1 = jax.tree_util.tree_leaves(agent_states[1].actor_params)
        
        some_different = any(
            not jnp.allclose(p0, p1)
            for p0, p1 in zip(params_0, params_1)
        )
        assert some_different, "Different agents should have different params"
        
        print("   create_all_agents: PASSED")
    
    def test_select_all_actions(self):
        """Test selecting actions for all agents."""
        print("Testing select_all_actions...")
        
        key = random.PRNGKey(42)
        
        configs = [
            AgentConfig(obs_dim=10, action_dim=2, critic_input_dim=15),  # 10 + 5 (2+3)
            AgentConfig(obs_dim=10, action_dim=3, critic_input_dim=15),
        ]
        
        agent_states, actors, _ = create_all_agents(key, configs)
        
        obs_list = [
            jnp.zeros(10),
            jnp.zeros(10),
        ]
        
        actor_params_list = [s.actor_params for s in agent_states]
        
        actions = select_all_actions(actors, actor_params_list, obs_list)
        
        assert len(actions) == 2
        assert actions[0].shape == (2,)
        assert actions[1].shape == (3,)
        
        print("   select_all_actions: PASSED")
    
    def test_select_all_actions_with_noise(self):
        """Test selecting noisy actions for all agents."""
        print("Testing select_all_actions_with_noise...")
        
        key = random.PRNGKey(42)
        
        configs = [
            AgentConfig(obs_dim=10, action_dim=2, critic_input_dim=14),
            AgentConfig(obs_dim=10, action_dim=2, critic_input_dim=14),
        ]
        
        agent_states, actors, _ = create_all_agents(key, configs)
        
        obs_list = [jnp.zeros(10), jnp.zeros(10)]
        
        actions, log_probs, new_states = select_all_actions_with_noise(
            key=key,
            actors=actors,
            agent_states=agent_states,
            obs_list=obs_list,
            noise_scale=0.1,
        )
        
        assert len(actions) == 2
        assert len(log_probs) == 2
        assert len(new_states) == 2
        
        print("   select_all_actions_with_noise: PASSED")


class TestSerialization:
    """Tests for saving/loading agent parameters."""
    
    def test_get_and_load_params(self):
        """Test getting and loading agent parameters."""
        print("Testing get_agent_params and load_agent_params...")
        
        key = random.PRNGKey(42)
        config = AgentConfig(obs_dim=10, action_dim=2, critic_input_dim=14)
        agent_state, actor, critic = create_agent(key, config)
        
        # Get params
        params_dict = get_agent_params(agent_state)
        
        assert 'actor_params' in params_dict
        assert 'critic_params' in params_dict
        assert 'target_actor_params' in params_dict
        assert 'target_critic_params' in params_dict
        assert 'actor_opt_state' in params_dict
        assert 'critic_opt_state' in params_dict
        
        # Create a new agent and load params
        key2 = random.PRNGKey(123)
        new_state, _, _ = create_agent(key2, config)
        
        # New agent should have different params
        old_leaves = jax.tree_util.tree_leaves(new_state.actor_params)
        orig_leaves = jax.tree_util.tree_leaves(agent_state.actor_params)
        
        initially_different = any(
            not jnp.allclose(o, n)
            for o, n in zip(orig_leaves, old_leaves)
        )
        assert initially_different, "New agent should have different params"
        
        # Load original params
        loaded_state = load_agent_params(new_state, params_dict)
        
        # Should now match
        loaded_leaves = jax.tree_util.tree_leaves(loaded_state.actor_params)
        
        all_match = all(
            jnp.allclose(o, l)
            for o, l in zip(orig_leaves, loaded_leaves)
        )
        assert all_match, "Loaded params should match original"
        
        print("   get_agent_params and load_agent_params: PASSED")


class TestJITCompilation:
    """Tests for JIT compilation of agent functions."""
    
    def test_select_action_jit(self):
        """Test JIT compilation of select_action."""
        print("Testing JIT compilation...")
        
        key = random.PRNGKey(42)
        config = AgentConfig(obs_dim=10, action_dim=2, critic_input_dim=14)
        agent_state, actor, _ = create_agent(key, config)
        
        # JIT compile
        select_action_jit = jax.jit(
            lambda params, obs: select_action(actor, params, obs)
        )
        
        obs = jnp.zeros(10)
        action = select_action_jit(agent_state.actor_params, obs)
        
        assert action.shape == (2,)
        print("   select_action JIT: PASSED")
    
    def test_update_targets_jit(self):
        """Test JIT compilation of update_targets."""
        key = random.PRNGKey(42)
        config = AgentConfig(obs_dim=10, action_dim=2, critic_input_dim=14)
        agent_state, _, _ = create_agent(key, config)
        
        update_targets_jit = jax.jit(lambda s: update_targets(s, tau=0.01))
        
        new_state = update_targets_jit(agent_state)
        assert new_state.step == 1
        
        print("   update_targets JIT: PASSED")
        print("   JIT compilation: ALL PASSED")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running agents.py tests")
    print("=" * 60)
    
    test_classes = [
        TestAgentCreation,
        TestActionSelection,
        TestCriticUpdate,
        TestActorUpdate,
        TestTargetUpdate,
        TestNoiseManagement,
        TestMultiAgent,
        TestSerialization,
        TestJITCompilation,
    ]
    
    passed = 0
    failed = 0
    
    for test_class in test_classes:
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    getattr(instance, method_name)()
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
