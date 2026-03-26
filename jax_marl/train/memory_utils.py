"""GPU and JAX memory reporting utilities.

Reports:
  - Actual GPU memory in use vs total (via device.memory_stats())
  - Per-component breakdown of JAX array bytes (buffer, params, env states)
  - Estimated scaling headroom for buffer_size and n_agents
"""

import jax
import jax.numpy as jnp
from typing import Any


def _bytes_of(pytree) -> int:
    """Sum nbytes of all JAX array leaves in a pytree."""
    leaves = jax.tree_util.tree_leaves(pytree)
    return sum(x.nbytes for x in leaves if hasattr(x, "nbytes"))


def _fmt(n_bytes: int) -> str:
    """Human-readable byte count."""
    if n_bytes >= 1 << 30:
        return f"{n_bytes / (1 << 30):.2f} GB"
    if n_bytes >= 1 << 20:
        return f"{n_bytes / (1 << 20):.1f} MB"
    return f"{n_bytes / (1 << 10):.1f} KB"


def get_device_memory_stats() -> dict:
    """Query JAX's own view of device memory.

    Returns dict with keys: bytes_in_use, peak_bytes_in_use, bytes_limit.
    Returns empty dict if the backend doesn't support memory_stats().
    """
    try:
        device = jax.devices()[0]
        stats = device.memory_stats()
        return {
            "bytes_in_use":      stats.get("bytes_in_use", 0),
            "peak_bytes_in_use": stats.get("peak_bytes_in_use", 0),
            "bytes_limit":       stats.get("bytes_limit", 0),
        }
    except Exception:
        return {}


def memory_report(maddpg_state, training_state, config, label: str = "") -> dict:
    """Print and return a breakdown of memory usage by component.

    Components measured:
      buffer        — replay buffer arrays (obs, actions, rewards, …)
      actor_params  — actor network weights + target weights + opt state
      critic_params — critic network weights + target weights + opt state
      env_states    — parallel environment states held in training_state
      other_jax     — live JAX bytes not accounted for above

    Args:
        maddpg_state:    MADDPGState (contains buffer_state, agent_states)
        training_state:  TrainingState (contains env_states, key, …)
        config:          AssemblyTrainConfig
        label:           Optional header label (e.g. "After init")

    Returns:
        dict with all measured values in bytes.
    """
    # ------------------------------------------------------------------ #
    # 1. Per-component pytree sizes
    # ------------------------------------------------------------------ #
    buffer_bytes = _bytes_of(maddpg_state.buffer_state)

    agent_states = maddpg_state.agent_states
    actor_bytes = _bytes_of({
        "params":        agent_states.actor_params,
        "target_params": agent_states.target_actor_params,
        "opt_state":     agent_states.actor_opt_state,
    })
    critic_bytes = _bytes_of({
        "params":        agent_states.critic_params,
        "target_params": agent_states.target_critic_params,
        "opt_state":     agent_states.critic_opt_state,
    })

    env_bytes = _bytes_of(training_state.env_states) if hasattr(training_state, "env_states") else 0
    accounted_bytes = buffer_bytes + actor_bytes + critic_bytes + env_bytes

    # ------------------------------------------------------------------ #
    # 2. Device-level memory
    # ------------------------------------------------------------------ #
    dev_stats = get_device_memory_stats()
    dev_in_use   = dev_stats.get("bytes_in_use", 0)
    dev_peak     = dev_stats.get("peak_bytes_in_use", 0)
    dev_limit    = dev_stats.get("bytes_limit", 0)
    other_bytes  = max(0, dev_in_use - accounted_bytes)

    # ------------------------------------------------------------------ #
    # 3. Scaling estimates (based on current buffer fraction)
    # ------------------------------------------------------------------ #
    scaling = {}
    if dev_limit > 0 and buffer_bytes > 0:
        free_bytes = dev_limit - dev_in_use
        # How many extra buffer slots fit in free memory?
        bytes_per_slot = buffer_bytes / max(config.buffer_size, 1)
        extra_slots = int(free_bytes * 0.85 / bytes_per_slot)  # 85% headroom
        scaling["max_buffer_size"] = config.buffer_size + extra_slots
        # Rough agent scaling (linear: obs/action dims grow with n_agents)
        if actor_bytes + critic_bytes > 0:
            bytes_per_agent_param = (actor_bytes + critic_bytes) / config.n_agents
            extra_agents = int(free_bytes * 0.85 / bytes_per_agent_param)
            scaling["max_additional_agents"] = extra_agents

    # ------------------------------------------------------------------ #
    # 4. Print
    # ------------------------------------------------------------------ #
    header = f"Memory Report — {label}" if label else "Memory Report"
    print(f"\n{'=' * 55}")
    print(f"  {header}")
    print(f"{'=' * 55}")

    if dev_limit > 0:
        pct_used = 100 * dev_in_use / dev_limit
        pct_peak = 100 * dev_peak / dev_limit
        print(f"  Device memory  ({jax.devices()[0].device_kind})")
        print(f"    In use:  {_fmt(dev_in_use):>10}  ({pct_used:.1f}% of {_fmt(dev_limit)})")
        print(f"    Peak:    {_fmt(dev_peak):>10}  ({pct_peak:.1f}%)")
        print(f"    Free:    {_fmt(dev_limit - dev_in_use):>10}")
    else:
        print("  Device memory stats unavailable")

    print(f"\n  JAX component breakdown")
    print(f"    Replay buffer:    {_fmt(buffer_bytes):>10}  "
          f"({config.buffer_size:,} slots × {config.n_agents} agents)")
    print(f"    Actor networks:   {_fmt(actor_bytes):>10}  "
          f"(params + targets + opt state, {config.n_agents} agents)")
    print(f"    Critic networks:  {_fmt(critic_bytes):>10}  "
          f"(params + targets + opt state, {config.n_agents} agents)")
    print(f"    Env states:       {_fmt(env_bytes):>10}  "
          f"({config.n_parallel_envs} parallel envs)")
    if dev_limit > 0:
        print(f"    XLA/other:        {_fmt(other_bytes):>10}  "
              f"(executables, workspace, misc)")

    if scaling:
        print(f"\n  Scaling headroom  (85% of free memory)")
        print(f"    Max buffer_size:  {scaling.get('max_buffer_size', '?'):>10,}")
        if "max_additional_agents" in scaling:
            print(f"    Extra agents fit: {scaling['max_additional_agents']:>10,}  "
                  f"(param memory only — buffer grows too)")

    print(f"{'=' * 55}\n")

    return {
        "buffer_bytes":    buffer_bytes,
        "actor_bytes":     actor_bytes,
        "critic_bytes":    critic_bytes,
        "env_bytes":       env_bytes,
        "dev_in_use":      dev_in_use,
        "dev_peak":        dev_peak,
        "dev_limit":       dev_limit,
        "other_bytes":     other_bytes,
        **scaling,
    }
