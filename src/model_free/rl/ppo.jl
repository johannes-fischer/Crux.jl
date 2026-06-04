"""
PPO loss function.

Flux 0.16 port: takes the differentiated model `m` (the actor) as its first
argument so it composes directly with `Flux.withgradient(m -> …, model)`.

cleanrl-style detail: advantages are whitened **per-minibatch** here (`whiten`
is applied to the slice the loss sees, not once over the whole rollout
buffer). This matches the canonical PPO implementation tricks; the
rollout-wide whitening that previously lived in `post_batch_callback` is
removed.
"""
function ppo_loss(m, 𝒫, 𝒟; info = Dict())
    new_probs = logpdf(m, 𝒟[:s], 𝒟[:a])
    logratio = new_probs .- 𝒟[:logprob]
    r = exp.(logratio)

    A_raw = 𝒟[:advantage]
    # Per-minibatch advantage normalization (cleanrl ppo.py:262)
    A = ignore_derivatives(() -> whiten(A_raw))
    p_loss = -mean(min.(r .* A, clamp.(r, (1f0 - 𝒫[:ϵ]), (1f0 + 𝒫[:ϵ])) .* A))
    e_loss = -mean(entropy(m, 𝒟[:s]))

    # Log useful information
    ignore_derivatives() do
        info[:entropy] = -e_loss
        info[:kl] = mean(𝒟[:logprob] .- new_probs)
        info[:clip_fraction] = sum((r .> 1 + 𝒫[:ϵ]) .| (r .< 1 - 𝒫[:ϵ])) / length(r)
        info[:avg_advantage] = mean(A_raw)
        info[:p_loss] = 𝒫[:λp]*p_loss
        log_ratio_stats!(info, logratio, r)
    end
    𝒫[:λp]*p_loss + 𝒫[:λe]*e_loss
end

# Importance-sampling ratio diagnostics. `logratio = new_probs - old_logprob`
# (raw); `log_ratio_max` is the overflow tell-tale, `r = exp(logratio)`.
function log_ratio_stats!(info, logratio, r)
    info[:log_ratio_mean] = mean(logratio)
    info[:log_ratio_max]  = maximum(logratio)
    info[:log_ratio_min]  = minimum(logratio)
    info[:ratio_mean] = mean(r)
    info[:ratio_max]  = maximum(r)
    info[:ratio_min]  = minimum(r)
    return info
end

"""
PPO critic (value) loss.

If the rollout stored an `:value` column (V(s) at action-selection time) and
`𝒫[:vclip]` is set, the new value is clipped to within ±vclip of the old
value before computing MSE — the cleanrl/spinningup "v_clipped" trick that
keeps the critic from drifting too far on a single update batch:

    v_clipped = v_old + clamp(v_new − v_old, −ϵ, +ϵ)
    L = mean(max((v_new − ret)², (v_clipped − ret)²))

If `:value` isn't in the buffer, falls back to plain MSE.
"""
function ppo_critic_loss(m, 𝒫, 𝒟; info = Dict(), kwargs...)
    v_new = value(m, 𝒟[:s])
    ret = 𝒟[:return]
    clip_fraction = 0f0
    if haskey(𝒫, :vclip) && !isnothing(𝒫[:vclip]) && haskey(𝒟, :value)
        v_old = 𝒟[:value]
        ϵv = 𝒫[:vclip]
        v_clipped = v_old .+ clamp.(v_new .- v_old, -ϵv, ϵv)
        loss = mean(max.((v_new .- ret) .^ 2, (v_clipped .- ret) .^ 2))
        clip_fraction = ignore_derivatives(() ->
            sum(abs.(v_new .- v_old) .> ϵv) / length(v_new))
    else
        loss = Flux.mse(v_new, ret)
    end
    ignore_derivatives() do
        info[:avg_return]           = mean(ret)            # moved from ppo_loss
        info[:mean_predicted_value] = mean(v_new)
        info[:clip_fraction_value]  = clip_fraction
    end
    loss
end

"""
LagrangePPO cost-critic (Vc) value loss.

Mirrors `ppo_critic_loss` but reads `𝒟[:cost_return]` and `𝒟[:cost_value]`
and clips with `𝒫[:vclip_cost]`. Falls back to plain MSE if either the
column or the clip range is absent.
"""
function ppo_cost_critic_loss(m, 𝒫, 𝒟; info = Dict(), kwargs...)
    v_new = value(m, 𝒟[:s])
    ret = 𝒟[:cost_return]
    clip_fraction = 0f0
    if haskey(𝒫, :vclip_cost) && !isnothing(𝒫[:vclip_cost]) && haskey(𝒟, :cost_value)
        v_old = 𝒟[:cost_value]
        ϵv = 𝒫[:vclip_cost]
        v_clipped = v_old .+ clamp.(v_new .- v_old, -ϵv, ϵv)
        loss = mean(max.((v_new .- ret) .^ 2, (v_clipped .- ret) .^ 2))
        clip_fraction = ignore_derivatives(() ->
            sum(abs.(v_new .- v_old) .> ϵv) / length(v_new))
    else
        loss = Flux.mse(v_new, ret)
    end
    ignore_derivatives() do
        info[:avg_cost_return]           = mean(ret)
        info[:mean_predicted_cost_value] = mean(v_new)
        info[:clip_fraction_cost_value]  = clip_fraction
    end
    loss
end

"""
F-head failure-probability loss (ConstrainedZero chance-constraint surrogate).

Trains a standalone failure network `Vf` (the differentiated model `m`) whose
raw output is a per-state failure LOGIT, via binary cross-entropy against the
`:traj_failure` column (1{trajectory fails}, populated by the sampler in
`fill_traj_failure!`). This approximates the chance-constraint surrogate
F(s) ≈ P(eventual failure | s) under the current rollout policy — the
undiscounted-probability analogue of the discounted cost critic `Vc` (which
estimates 𝔼[∑ γ^t cost]). Because `Vf` is trained by its own optimizer,
decoupled from the actor, this loss never contributes to the PPO policy
gradient: it is a purely predictive head we can later export to
BetaZero/ConstrainedZero (see [`failure_probability`](@ref)).

Training is done in logit space with `logitbinarycrossentropy` for numerical
stability; the sigmoid is applied only at readout in `failure_probability`.
"""
function ppo_failure_loss(m, 𝒫, 𝒟; info = Dict(), kwargs...)
    f_logits = value(m, 𝒟[:s])          # raw head output = failure logits
    targets  = 𝒟[:traj_failure]         # Float32 0/1, broadcast over the episode
    loss = Flux.logitbinarycrossentropy(f_logits, targets)
    ignore_derivatives() do
        info[:avg_traj_failure]     = mean(targets)
        info[:mean_predicted_pfail] = mean(Flux.sigmoid.(f_logits))
    end
    loss
end

"""
    failure_probability(Vf, s)

Read the F-head failure surrogate as a probability in [0,1]: applies `sigmoid`
to the raw logits produced by `value(Vf, s)`, where `Vf` is the network trained
by [`ppo_failure_loss`](@ref). Use this when exporting the surrogate into
BetaZero / ConstrainedZero, whose planner consumes a scalar
`estimate_failure(mdp, s)` — e.g.

    f = (mdp, s) -> only(failure_probability(Vf, input_representation(s)))
"""
failure_probability(Vf, s) = Flux.sigmoid.(value(Vf, s))

"""
Proximal policy optimization (PPO) solver.

cleanrl-aligned defaults:
- Per-minibatch advantage normalization inside `ppo_loss`.
- Value clipping (Schulman/cleanrl `--clip-vloss`) at `vclip = ϵ`. Set
  `vclip=nothing` to disable; in that case the `:value` column is also
  unused and falls back to plain MSE.

```julia
PPO(;
    π::ActorCritic,
    ϵ::Float32 = 0.2f0,
    λp::Float32 = 1f0,
    λe::Float32 = 0.1f0,
    target_kl = 0.012f0,
    vclip::Union{Float32, Nothing} = ϵ,
    a_opt::NamedTuple=(;),
    c_opt::NamedTuple=(;),
    log::NamedTuple=(;),
    required_columns=[],
    kwargs...)
```
"""
function PPO(;
        π::ActorCritic,
        ϵ::Float32 = 0.2f0,
        λp::Float32 = 1f0,
        λe::Float32 = 0.1f0,
        target_kl = 0.012f0,
        vclip::Union{Float32, Nothing} = ϵ,
        a_opt::NamedTuple=(;),
        c_opt::NamedTuple=(;),
        log::NamedTuple=(;),
        required_columns=[],
        kwargs...)

     function record_avgr(𝒟; info=Dict(), 𝒮)
         info[:avg_r] = sum(𝒟[:r]) / sum(𝒟[:episode_end])
     end

     # Add :value column when value clipping is on; the sampler's fill_gae!
     # writes V(s) at rollout time so ppo_critic_loss can clip the update.
     extra_cols = isnothing(vclip) ? Symbol[] : Symbol[:value]

     OnPolicySolver(;agent=PolicyParams(π),
                    𝒫=(ϵ=ϵ, λp=λp, λe=λe, vclip=vclip),
                    log = LoggerParams(;dir = "log/ppo", log...),
                    a_opt = TrainingParams(;loss = ppo_loss, early_stopping = (infos) -> (infos[end][:kl] > target_kl), name = "actor_", a_opt...),
                    c_opt = TrainingParams(;loss = ppo_critic_loss, name = "critic_", c_opt...),
                    required_columns = unique([required_columns..., :return, :logprob, :advantage, extra_cols...]),
                    post_sample_callback=record_avgr,
                    kwargs...)
end

"""
PPO loss with a penalty (Lagrange-constrained PPO).

Flux 0.16 port: same model-first signature as ppo_loss.
"""
function lagrange_ppo_loss(m, 𝒫, 𝒟; info = Dict())
    new_probs = logpdf(m, 𝒟[:s], 𝒟[:a])
    logratio = new_probs .- 𝒟[:logprob]
    r = exp.(logratio)

    A_raw = 𝒟[:advantage]
    # Per-minibatch advantage normalization (matches PPO).
    A = ignore_derivatives(() -> whiten(A_raw))
    # Same per-minibatch whitening for the cost advantage — keeps it O(1) and on
    # the same scale as the reward advantage (the reward p_loss never NaN'd with
    # the same r, the only difference being that A was whitened and this wasn't).
    # Mirrors the GAIL solvers, which whiten both advantages.
    Ac_raw = 𝒟[:cost_advantage]
    Ac = ignore_derivatives(() -> whiten(Ac_raw))
    rA = r .* A
    p_term = min.(rA, clamp.(r, (1f0 - 𝒫[:ϵ]), (1f0 + 𝒫[:ϵ])) .* A)
    p_loss = -mean(p_term)
    e_loss = -mean(entropy(m, 𝒟[:s]))

    # Read the cost penalty computed ONCE PER ITERATION by `lagrange_post_sample`
    # (the solver's post_batch_callback) over the FULL batch — see that function
    # for the PID controller. Computing it per-minibatch here previously divided
    # `Σcost / Σepisode_end` over a shuffled minibatch that could contain zero
    # episode-end markers → 0/0 = NaN, which then poisoned the persistent PID
    # state (I, smooth_Δ, smooth_Jc, Jc_prev) for every subsequent minibatch.
    penalty = ignore_derivatives(() -> 𝒫[:penalty][1])

    # cost_loss = 𝒫[:penalty_scale] * penalty * mean(r .* 𝒟[:cost_advantage])
    cost_loss = penalty * mean(max.(r .* Ac, clamp.(r, (1f0 - 𝒫[:ϵ]), (1f0 + 𝒫[:ϵ])) .* Ac))

    # Log useful information
    ignore_derivatives() do
        info[:entropy] = -e_loss
        info[:kl] = mean(𝒟[:logprob] .- new_probs)
        info[:clip_fraction] = sum((r .> 1 + 𝒫[:ϵ]) .| (r .< 1 - 𝒫[:ϵ])) / length(r)
        info[:p_loss] = 𝒫[:λp]*p_loss
        info[:cost_loss] = cost_loss
        info[:avg_advantage] = mean(A_raw)
        info[:avg_cost_advantage] = mean(Ac_raw)
        # Surface the per-iteration PID controller state (set by
        # `lagrange_post_sample`) into training_info so the eval logger keeps
        # finding `:penalty`/`:cur_cost`/`:prop_term`/`:deriv_term`/`:integral_term`.
        info[:penalty] = penalty
        info[:cur_cost] = 𝒫[:cur_cost][1]
        info[:prop_term] = 𝒫[:prop_term][1]
        info[:deriv_term] = 𝒫[:deriv_term][1]
        info[:integral_term] = 𝒫[:I][1]
        log_ratio_stats!(info, logratio, r)
    end
    (𝒫[:λp]*p_loss + 𝒫[:λe]*e_loss + cost_loss) / (1 + penalty)
end

"""
Lagrange-Constrained PPO solver.

Value clipping (PPO `--clip-vloss`) is supported separately for the reward
and cost critics via `vclip` and `vclip_cost`. Each defaults to `ϵ`; set to
`nothing` to disable. When set, the rollout caches `Vr(s)` / `Vc(s)` at
action-selection time into `:value` / `:cost_value` so the critic losses can
clip their update.

```julia
LagrangePPO(;
    π::ActorCritic,
    Vc::ContinuousNetwork, # value network for estimating cost
    ϵ::Float32 = 0.2f0,
    λp::Float32 = 1f0,
    λe::Float32 = 0.1f0,
    λ_gae = 0.95f0,
    target_kl = 0.012f0,
    target_cost = 0.025f0,
    penalty_scale = 1f0,
    penalty_max = Inf32,
    Ki_max = 10f0,
    Ki = 1f-3,
    Kp = 1,
    Kd = 0,
    ema_α = 0.95,
    vclip::Union{Float32, Nothing} = ϵ,
    vclip_cost::Union{Float32, Nothing} = ϵ,
    a_opt::NamedTuple=(;),
    c_opt::NamedTuple=(;),
    cost_opt::NamedTuple=(;),
    Vf::Union{ContinuousNetwork, Nothing} = nothing, # optional failure-probability surrogate (logit output)
    f_opt::NamedTuple=(;),
    failure_source::Symbol = :cost,        # :cost (cost>0) or :fail (isfailure column)
    traj_failure_mode::Symbol = :episode,  # :episode (BetaZero ref) or :suffix (paper Eq. 8)
    log::NamedTuple=(;),
    required_columns=[],
    kwargs...)
```

When `Vf` is supplied, an independent failure-probability head F(s) ≈ P(eventual
failure | s) is trained by binary cross-entropy against a `:traj_failure` label
(the ConstrainedZero chance-constraint surrogate). It is fully decoupled from the
actor/critics — it never enters the policy gradient — and is exported as a
probability via [`failure_probability`](@ref). With `Vf === nothing`,
`LagrangePPO` is unchanged.

"""
function LagrangePPO(;
    π::ActorCritic,
    Vc::ContinuousNetwork, # value network for estimating cost
    ϵ::Float32 = 0.2f0,
    λp::Float32 = 1f0,
    λe::Float32 = 0.1f0,
    λ_gae = 0.95f0,
    target_kl = 0.012f0,
    target_cost = 0.025f0,
    penalty_scale = 1f0,
    penalty_max = Inf32,
    Ki_max = 10f0,
    Ki = 1f-3,
    Kp = 1,
    Kd = 0,
    ema_α = 0.95,
    vclip::Union{Float32, Nothing} = ϵ,
    vclip_cost::Union{Float32, Nothing} = ϵ,
    a_opt::NamedTuple=(;),
    c_opt::NamedTuple=(;),
    cost_opt::NamedTuple=(;),
    Vf::Union{ContinuousNetwork, Nothing} = nothing, # failure-probability surrogate (raw logit output)
    f_opt::NamedTuple=(;),
    failure_source::Symbol = :cost,        # per-step failure event: :cost (cost>0) or :fail (isfailure column)
    traj_failure_mode::Symbol = :episode,  # label form: :episode (BetaZero ref) or :suffix (paper Eq. 8)
    log::NamedTuple=(;),
    required_columns=[],
    kwargs...)

     # Per-iteration callback — runs ONCE over the FULL batch (post_sample, fired
     # by `steps!` after the buffer is assembled). Computes `avg_r` AND the
     # PID-Lagrange penalty update. It MUST be the post_sample_callback, not the
     # post_batch_callback: the DDE trainer reserves `post_batch_callback` for
     # periodic eval, and a splatted kwarg there silently overrides anything set
     # here (keyword splat — last value wins), which would leave the PID state at
     # its zero init forever (penalty/cost_loss/PID terms all logging 0).
     #
     # The cost constraint is on the per-episode cumulative cost return
     # Jc = 𝔼[Σ_t c_t], estimated as Σcost / (#completed episodes). The denominator
     # is guarded with max(·,1): `episode_end` is set only at true episode ends
     # (never on slice cutoffs), so a fully-truncated / non-terminating batch would
     # otherwise give 0/0 = NaN and poison the persistent PID state. Results are
     # stashed in 𝒫 for `lagrange_ppo_loss` to read (it no longer mutates state).
     function lagrange_pid_penalty_update(𝒟; info=Dict(), 𝒮)
         𝒫 = 𝒮.𝒫
         n_ep = max(sum(𝒟[:episode_end]), 1)
         info[:avg_r] = sum(𝒟[:r]) / n_ep
         Jc = sum(𝒟[:cost]) / n_ep
         Δ = Jc - 𝒫[:target_cost]
         𝒫[:I][1] = clamp(𝒫[:I][1] + 𝒫[:Ki]*Δ, 0, 𝒫[:Ki_max])
         α = 𝒫[:ema_α]
         𝒫[:smooth_Δ][1] = α * 𝒫[:smooth_Δ][1] + (1 - α)*Δ
         𝒫[:smooth_Jc][1] = α * 𝒫[:smooth_Jc][1] + (1 - α)*Jc
         ∂ = max(0, 𝒫[:smooth_Jc][1] - 𝒫[:Jc_prev][1])
         𝒫[:Jc_prev][1] = 𝒫[:smooth_Jc][1]
         𝒫[:penalty][1] = clamp(𝒫[:Kp] * 𝒫[:smooth_Δ][1] + 𝒫[:I][1] + 𝒫[:Kd]*∂, 0, 𝒫[:penalty_max])
         𝒫[:cur_cost][1] = Jc
         𝒫[:prop_term][1] = 𝒫[:Kp] * 𝒫[:smooth_Δ][1]
         𝒫[:deriv_term][1] = ∂
         info[:penalty] = 𝒫[:penalty][1]
         info[:cur_cost] = Jc
     end

     𝒫=(ϵ=ϵ, λp=λp, λe=λe,
        target_cost=target_cost,
        penalty_scale=penalty_scale,
        penalty_max=penalty_max,
        Ki_max=Ki_max,
        I = [0f0],
        Jc_prev = [0f0],
        Ki=Ki,
        Kp=Kp,
        Kd=Kd,
        ema_α=ema_α,
        smooth_Δ = [0f0],
        smooth_Jc = [0f0],
        vclip=vclip,
        vclip_cost=vclip_cost,
        # Per-iteration PID outputs, written by `lagrange_post_sample` and read
        # by `lagrange_ppo_loss` (which no longer mutates any controller state).
        penalty = [0f0],
        cur_cost = [0f0],
        prop_term = [0f0],
        deriv_term = [0f0],
        )

     # Add :value / :cost_value columns when the respective clip is enabled;
     # the sampler's fill_gae! writes Vr(s) / Vc(s) at rollout time so the
     # critic losses can clip their updates.
     extra_cols = Symbol[]
     isnothing(vclip)      || push!(extra_cols, :value)
     isnothing(vclip_cost) || push!(extra_cols, :cost_value)

     # F-head (ConstrainedZero) wiring: only allocate the :traj_failure column
     # and build the failure optimizer when a Vf network is supplied — with
     # Vf === nothing, LagrangePPO behaves exactly as before. When the label is
     # derived from the :fail predicate, also require that column so the sampler
     # populates it via extra_functions["isfailure"].
     failure_cols = Symbol[]
     if !isnothing(Vf)
         push!(failure_cols, :traj_failure)
         failure_source == :fail && push!(failure_cols, :fail)
     end
     f_opt_tp = isnothing(Vf) ? nothing :
                TrainingParams(;loss = ppo_failure_loss, name = "failure_", f_opt...)

     OnPolicySolver(;agent=PolicyParams(π),
                    𝒫=𝒫,
                    Vc=Vc,
                    Vf=Vf,
                    f_opt=f_opt_tp,
                    failure_source=failure_source,
                    traj_failure_mode=traj_failure_mode,
                    log = LoggerParams(;dir = "log/lagrange_ppo", log...),
                    a_opt = TrainingParams(;loss = lagrange_ppo_loss, early_stopping = (infos) -> (infos[end][:kl] > target_kl), name = "actor_", a_opt...),
                    c_opt = TrainingParams(;loss = ppo_critic_loss, name = "critic_", c_opt...),
                    cost_opt = TrainingParams(;loss = ppo_cost_critic_loss, name = "cost_critic_", cost_opt...),
                    required_columns = unique([required_columns..., :return, :advantage, :logprob, :cost_advantage, :cost, :cost_return, extra_cols..., failure_cols...]),
                    post_sample_callback=lagrange_pid_penalty_update,
                    kwargs...)
end
