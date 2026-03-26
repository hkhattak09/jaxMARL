Code Audit Report
Actual Bugs
1. observations.py:266 — Dead variable k, silent observation shape mismatch


k = jnp.minimum(num_obs_grid_max, n_grid)  # computed but NEVER used
sorted_indices = jnp.argsort(..., axis=1)[:, :num_obs_grid_max]  # uses Python int
k is computed and thrown away. The slice [:, :num_obs_grid_max] on a matrix with only n_grid columns silently returns n_grid columns when n_grid < num_obs_grid_max. With the default procedural shapes (max 25 cells), but num_obs_grid_max=80, the grid observation block is 50 floats instead of 160 floats. This means compute_observation_dim() in observations.py returns a wrong (inflated) value — don't use it. The training code sidesteps the bug by probing the actual env output to get the real obs_dim (test_obs.shape[-1] in train_assembly.py:177), and get_obs_dim() in assembly_env.py:750 correctly uses min(n_grid, num_obs_grid_max). So the networks get the right size, but compute_observation_dim() is dead/wrong and will mislead anyone using it.

2. buffers.py:537 — Dynamic slice in get_average_rewards() breaks inside JIT


return jnp.mean(state.rewards[:state.size], axis=0)
state.size is a traced jnp.ndarray. Dynamic array slicing with a traced index like this is illegal in JAX JIT. It'll blow up with a ConcretizationTypeError if called inside any JIT-compiled function. Currently this method isn't called inside JIT in the main training path, but it's broken and will bite anyone who tries.

3. buffers.py:435 — sample_without_replacement() passes traced array to random.permutation()


indices = random.permutation(key, state.size)[:batch_size]
state.size is jnp.ndarray (traced). jax.random.permutation requires a static integer for the size argument when called inside JIT. This works in eager mode but will fail inside a JIT-compiled context. Not called in the hot path currently but is silently broken.

4. maddpg_wrapper.py:466 — rollout_episode forces device sync every step


if next_state.env_state.done:
    break
done is a JAX boolean scalar. Evaluating if on it materializes the value to Python — a full device synchronization every step. This completely breaks async GPU execution. This function isn't used in the main JIT training path, but if anyone calls it for debugging/eval, they'll get serial, sync'd execution without realizing it.

Performance Issues (not correctness, but wasteful)
5. Triple pairwise distance computation per assembly_env.step() call

Every step computes O(n_agents²) pairwise distances 3 separate times:

_update_occupancy() → compute_agent_collisions() → compute_pairwise_distances()
step() → get_k_nearest_neighbors_all_agents() → compute_pairwise_distances()
compute_rewards() → compute_agent_collisions() → compute_pairwise_distances()
For 20-30 agents this is a ~3× waste of compute on the most expensive operation in the step. The result from get_k_nearest_neighbors_all_agents() already has all the distance info needed for rewards; it's just not threaded through.

6. Double velocity clipping
integrate_dynamics() clips velocities to physics_params.vel_max, then assembly_env.step() immediately clips again to params.max_velocity. Both default to 0.8. Harmless but redundant.

7. lax.scan over updates with a Python loop over agents inside
In create_jit_train_step(), lax.scan calls maddpg.update(), which has for agent_i in range(n_agents) inside. Inside a traced context, Python loops get unrolled. With 20 agents × 30 updates_per_step, this creates an enormous XLA graph. Expect very long first-run compile time. Could cause OOM during compilation on smaller GPUs.

Design Issues / Subtle Gotchas
8. d_sen=0.4 in production config vs grid layout
The production config sets d_sen=0.4 (sensing range). With the target shape spread across a 5.0 arena and up to 25 grid cells, agents only sense grid cells that happen to be within 0.4 units. Most of the time the 80-slot grid observation block will be nearly all zeros. This isn't wrong per se — it makes the task harder and forces agents to navigate blind — but it's worth knowing this is intentional. The default ObservationParams.d_sen=3.0 would give much richer grid observations.

9. MADDPGWrapper.episode_returns never resets on episode done
After the env auto-resets (done=True), episode_returns keeps accumulating across what are now multiple episodes within one step() call sequence. You'd need to check done and manually reset if you want clean per-episode stats from the wrapper. The main training loop doesn't use this field anyway — it rolls its own accumulation — but it's a trap for anyone using MADDPGWrapper directly.

10. store_transition() in maddpg.py has Python loops over agent lists
Lines 462-511 do Python for i, obs in enumerate(observations) with jnp.pad calls inside. This runs eagerly (fine), but the len(obs_flat) check works only because shapes are static. If you ever call this inside a JIT context expecting it to trace, it'll fail on the len() call over a dynamic shape.

11. Import inside JIT-traced closure in train_assembly.py:268


from assembly_env import compute_r_avoid
Inside compute_priors_batch which is @jax.jit and called inside lax.scan. Python import inside a JIT function happens at trace time (Python-level), not XLA compile time, so it works. But it re-executes the import statement every retracing and is confusing — move it to module top.

What's Fine
The AssemblyState.trajectory circular buffer logic is correct. The traj_idx wraps correctly and get_trajectory() reorders indices properly.
lax.scan's carry doesn't replicate the replay buffer — only the current-state copy lives in memory. No memory hoarding from the scan itself.
physics.py if is_boundary checks are Python bools, static at JIT compile time — correctly excluded from tracing.
The apply_shape_transform() masking is correct — padded (invalid) cells are zeroed.
The rho_cos_dec function handles edge case delta=0 correctly: when delta=0, the cosine decay zone spans [0, r) and the z < 0 branch never triggers.
compute_pairwise_distances_periodic wrapping logic is correct — it applies the shortest-path wrap in both directions.
MADDPGState functional updates via .replace() are correct — no mutation.
The replay buffer circular overwrite is correct: (pos + 1) % capacity.
lax.cond branches in jit_train_step return matching pytree structures.
The obs_dim is determined from actual env output at create_training_state:177, not from the broken compute_observation_dim(), so networks get the right size.
Summary Table
Severity	Location	Issue
Bug	observations.py:266	k never used; compute_observation_dim() gives wrong value
Bug	buffers.py:537	Dynamic slice [:state.size] breaks in JIT
Bug	buffers.py:435	random.permutation(key, state.size) breaks in JIT
Bug	maddpg_wrapper.py:466	if done: forces device sync per step
Perf	assembly_env.py:380-454	3× redundant pairwise distance computation per step
Perf	train_assembly.py:690	Massive unrolled XLA graph from Python loop inside lax.scan
Perf	assembly_env.py:392-393	Double velocity clipping
Design	assembly_cfg.py:38	d_sen=0.4 → near-empty grid observations by default
Design	maddpg_wrapper.py:184	episode_returns doesn't reset on episode done
Design	train_assembly.py:268	Import inside JIT closure
None of these are memory leaks — JAX's XLA runtime manages device memory, and all Python-side allocations are bounded and GC'd correctly. The biggest memory concern is the pre-allocated replay buffer (buffer_size=240k × n_agents × obs_dim × float32 × ~4 arrays ≈ a few GB), which is intentional and bounded.