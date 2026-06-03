"""
On policy solver type.

Fields
======
- `agent::PolicyParams` Policy parameters ([`PolicyParams`](@ref))
- `S::AbstractSpace` State space
- `N::Int = 1000` Number of environment interactions
- `ΔN::Int = 200` Number of interactions between updates
- `max_steps::Int = 100` Maximum number of steps per episode
- `log::Union{Nothing, LoggerParams} = nothing` The logging parameters
- `i::Int = 0` The current number of environment interactions
- `param_optimizers::Dict{Any, TrainingParams} = Dict()` Training parameters for the parameters
- `a_opt::TrainingParams` Training parameters for the actor
- `c_opt::Union{Nothing, TrainingParams} = nothing` Training parameters for the critic
- `𝒫::NamedTuple = (;)` Parameters of the algorithm
- `interaction_storage = nothing` If this is initialized to an array then it will store all interactions
- `post_sample_callback = (𝒟; kwargs...) -> nothing` Callback that that happens after sampling experience
- `post_batch_callback = (𝒟; kwargs...) -> nothing` Callback that that happens after sampling a batch (before training)
- `post_train_callback = (𝒟; kwargs...) -> nothing` Callback that runs after training each iteration.
    Receives `training_info::Dict` (actor_loss, critic_loss, kl, entropy, clip_fraction, grad_norms, advantage,
    return, …) and `info::Dict` (rollout-buffer stats). Use this to log per-iteration metrics without
    intercepting the internal TBLogger.

On-policy-specific parameters
======
- `λ_gae::Float32 = 0.95` Generalized advantage estimation parameter
- `required_columns = Symbol[]` Extra data columns to store

Parameters specific to cost constraints (a separate value network)
======
- `Vc::Union{ContinuousNetwork, Nothing} = nothing` Cost value approximator
- `cost_opt::Union{Nothing, TrainingParams} = nothing` Training parameters for the cost value
"""
@with_kw mutable struct OnPolicySolver <: Solver
    agent::PolicyParams # Policy
    S::AbstractSpace # State space
    N::Int = 1000 # Number of environment interactions
    ΔN::Int = 200 # Number of interactions between updates
    max_steps::Int = 100 # Maximum number of steps per episode
    log::Union{Nothing, LoggerParams} = nothing # The logging parameters
    i::Int = 0 # The current number of environment interactions
    param_optimizers::Dict{Any, TrainingParams} = Dict() # Training parameters for the parameters
    a_opt::TrainingParams # Training parameters for the actor
    c_opt::Union{Nothing, TrainingParams} = nothing # Training parameters for the critic
    𝒫::NamedTuple = (;) # Parameters of the algorithm
    interaction_storage = nothing # If this is initialized to an array then it will store all interactions
    post_sample_callback = (𝒟; kwargs...) -> nothing # Callback that that happens after sampling experience
    post_batch_callback = (𝒟; kwargs...) -> nothing # Callback that that happens after sampling a batch (BEFORE training)
    post_train_callback = (𝒟; kwargs...) -> nothing # Callback after training — receives training_info (loss/grad/kl/entropy) + info (rollout stats)

    # On-policy-specific parameters
    λ_gae::Float32 = 0.95 # Generalized advantage estimation parameter
    required_columns = Symbol[]# Extra data columns to store

    # Parallel sampling. num_envs=1 → existing single-env code path (unchanged
    # except for the carry-state slice-end semantics applied to both paths).
    # num_envs>1 → deep-copy the MDP per env, run `Threads.@threads` env steps,
    # ONE batched policy forward per timestep. ΔN must be divisible by num_envs;
    # each env collects ΔN÷num_envs steps per rollout.
    num_envs::Int = 1
    # Per-env RNG strategy. `env_seed::Int` → samplers + eval get
    # `Xoshiro(env_seed + 1000*e)` (reproducible across runs). `nothing`
    # → fresh `Xoshiro()` per sampler (system-entropy-seeded, each run
    # different, but per-env still independent so threading is race-free).
    env_seed::Union{Int, Nothing} = nothing

    # Parameters specific to cost constraints (a separate value network)
    Vc::Union{ContinuousNetwork, Nothing} = nothing # Cost value approximator
    cost_opt::Union{Nothing, TrainingParams} = nothing # Training parameters for the cost value

    # F-head (ConstrainedZero chance-constraint failure surrogate). `Vf` is an
    # INDEPENDENT network whose raw output is a per-state failure LOGIT; it is
    # trained by `f_opt` with BCE against the :traj_failure column, fully
    # decoupled from the actor so it never enters the policy gradient (it is a
    # purely predictive head, exportable to BetaZero/ConstrainedZero — read it
    # as a probability via `failure_probability(Vf, s)`). `failure_source` /
    # `traj_failure_mode` configure the trajectory-failure label (see `Sampler`)
    # and are forwarded to every sampler at `solve` time.
    Vf::Union{ContinuousNetwork, Nothing} = nothing # Failure-probability approximator (logits)
    f_opt::Union{Nothing, TrainingParams} = nothing # Training parameters for the failure head
    failure_source::Symbol = :cost
    traj_failure_mode::Symbol = :episode
end

function policy_gradient_training(𝒮::OnPolicySolver, 𝒟)
    info = Dict()

    # Train parameters
    for (θs, p_opt) in 𝒮.param_optimizers
        batch_train!(θs, p_opt, 𝒮.𝒫, 𝒟, info=info, π_loss=𝒮.agent.π)
    end

    # Train the actor
    batch_train!(actor(𝒮.agent.π), 𝒮.a_opt, 𝒮.𝒫, 𝒟, info=info)

    # Train the critic (if applicable)
    if !isnothing(𝒮.c_opt)
        batch_train!(critic(𝒮.agent.π), 𝒮.c_opt, 𝒮.𝒫, 𝒟, info=info)
    end


    if !isnothing(𝒮.cost_opt)
        batch_train!(𝒮.Vc, 𝒮.cost_opt, 𝒮.𝒫, 𝒟, info=info)
    end

    # Train the failure surrogate Vf (if applicable). Independent network +
    # optimizer; its BCE loss differentiates only Vf, so it never contributes to
    # the actor/critic/cost-critic gradients.
    if !isnothing(𝒮.f_opt)
        batch_train!(𝒮.Vf, 𝒮.f_opt, 𝒮.𝒫, 𝒟, info=info)
    end

    return info
end

function POMDPs.solve(𝒮::OnPolicySolver, mdp)
    # Construct the training buffer, constants, and sampler
    𝒟 = ExperienceBuffer(𝒮.S, 𝒮.agent.space, 𝒮.ΔN, 𝒮.required_columns, device=device(𝒮.agent.π))
    γ, λ = Float32(discount(mdp)), 𝒮.λ_gae

    mkrng(offset) = isnothing(𝒮.env_seed) ? Random.Xoshiro() : Random.Xoshiro(𝒮.env_seed + offset)

    # Dedicated eval sampler so logging carries state across rollouts.
    if isnothing(𝒮.log.sampler)
        𝒮.log.sampler = Sampler(mdp, 𝒮.agent, S=𝒮.S,
                                max_steps=𝒮.max_steps,
                                rng=mkrng(999_999))
    end

    if 𝒮.num_envs == 1
        s = Sampler(mdp, 𝒮.agent, S=𝒮.S, required_columns=𝒮.required_columns, λ=λ, max_steps=𝒮.max_steps, Vc=𝒮.Vc, failure_source=𝒮.failure_source, traj_failure_mode=𝒮.traj_failure_mode, rng=mkrng(1000))
        run_training_loop!(𝒮, 𝒟, s)
    else
        # Parallel: share the MDP across envs, give each sampler an independent
        # RNG (seeded `Xoshiro(env_seed + 1000*e)` for reproducibility, or
        # `Xoshiro()` when `env_seed === nothing`). ΔN must split evenly
        # across envs — `steps!(samplers, …)` asserts this.
        samplers = [Sampler(mdp, 𝒮.agent, S=𝒮.S, required_columns=𝒮.required_columns,
                            λ=λ, max_steps=𝒮.max_steps, Vc=𝒮.Vc,
                            failure_source=𝒮.failure_source, traj_failure_mode=𝒮.traj_failure_mode,
                            rng=mkrng(1000 * e))
                    for e in 1:𝒮.num_envs]
        run_training_loop!(𝒮, 𝒟, samplers)
    end
    𝒮.i += 𝒮.ΔN
    𝒮.agent.π
end


# Shared training loop: logs pre-train performance, then iterates rollout →
# callbacks → training → callbacks → log. Dispatch on `s` (Sampler vs
# Vector{<:Sampler}) routes the `steps!` call to the serial or batched-parallel
# implementation. Caller is responsible for setting `𝒮.log.sampler` (the eval
# sampler) before calling.
function run_training_loop!(𝒮::OnPolicySolver, 𝒟, s)
    # Log the pre-train performance
    log(𝒮.log, 𝒮.i, 𝒮=𝒮)
    # Loop over the desired number of environment interactions
    for 𝒮.i = range(𝒮.i, stop=𝒮.i + 𝒮.N - 𝒮.ΔN, step=𝒮.ΔN)
        # Info to collect during training
        info = Dict()
        # Sample transitions into the batch buffer
        t_sample_ns = time_ns()
        steps!(s, 𝒟, Nsteps=𝒮.ΔN, explore=true, i=𝒮.i, store=𝒮.interaction_storage,
               cb=(D) -> 𝒮.post_sample_callback(D, info=info, 𝒮=𝒮), reset=true)
        info[:t_sample_seconds]       = (time_ns() - t_sample_ns) / 1e9
        info[:n_steps_in_batch] = 𝒮.ΔN
        # Post-batch callback, often used for additional training
        𝒮.post_batch_callback(𝒟, info=info, 𝒮=𝒮)
        # Train the networks
        t_train_ns = time_ns()
        training_info = policy_gradient_training(𝒮, 𝒟)
        info[:t_train_seconds] = (time_ns() - t_train_ns) / 1e9
        # Post-train callback — fires AFTER training so `training_info`
        # (actor_loss, critic_loss, kl, entropy, clip_fraction, grad norms,
        # advantage, returns) is available. Use this hook to log per-iteration
        # metrics from external systems (e.g. Wandb) instead of intercepting
        # the internal TBLogger via the `log_value` mechanism.
        𝒮.post_train_callback(𝒟, training_info=training_info, info=info, 𝒮=𝒮)
        # Log the results
        log(𝒮.log, 𝒮.i + 1:𝒮.i + 𝒮.ΔN, training_info, info, 𝒮=𝒮)
    end
end
