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
    r = exp.(new_probs .- 𝒟[:logprob])

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
    end
    𝒫[:λp]*p_loss + 𝒫[:λe]*e_loss
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
    r = exp.(new_probs .- 𝒟[:logprob])

    A_raw = 𝒟[:advantage]
    # Per-minibatch advantage normalization (matches PPO).
    A = ignore_derivatives(() -> whiten(A_raw))
    p_loss = -mean(min.(r .* A, clamp.(r, (1f0 - 𝒫[:ϵ]), (1f0 + 𝒫[:ϵ])) .* A))
    e_loss = -mean(entropy(m, 𝒟[:s]))

    #update the cost penalty
    penalty = ignore_derivatives() do
        # 𝒫[:penalty_param][1] = clamp(𝒫[:penalty_param][1], -7, 10)
        # Flux.softplus(𝒫[:penalty_param][1])

        # Average cost
        Jc = sum(𝒟[:cost]) / sum(𝒟[:episode_end])
        # Jc = maximum(𝒟[:cost])


        # Compute the error
        Δ = Jc - 𝒫[:target_cost]

        # Update integral term
        𝒫[:I][1] = clamp(𝒫[:I][1] + 𝒫[:Ki]*Δ, 0, 𝒫[:Ki_max])

        # Smooth out the values
        α = 𝒫[:ema_α]
        𝒫[:smooth_Δ][1] = α * 𝒫[:smooth_Δ][1] + (1 - α)*Δ
        𝒫[:smooth_Jc][1] = α * 𝒫[:smooth_Jc][1] + (1 - α)*Jc

        # Compute the derivative term
        ∂ = max(0, 𝒫[:smooth_Jc][1] - 𝒫[:Jc_prev][1])

        # Update the previous cost
        𝒫[:Jc_prev][1] = 𝒫[:smooth_Jc][1]

        # PID update
        penalty = clamp(𝒫[:Kp] * 𝒫[:smooth_Δ][1] + 𝒫[:I][1] + 𝒫[:Kd]*∂, 0, 𝒫[:penalty_max])

        info[:penalty] = penalty
        info[:cur_cost] = Jc
        info[:prop_term] = 𝒫[:Kp] * 𝒫[:smooth_Δ][1]
        info[:deriv_term] = ∂
        info[:integral_term] = 𝒫[:I][1]

        penalty
    end

    # cost_loss = 𝒫[:penalty_scale] * penalty * mean(r .* 𝒟[:cost_advantage])
    cost_loss = penalty * mean(max.(r .* 𝒟[:cost_advantage], clamp.(r, (1f0 - 𝒫[:ϵ]), (1f0 + 𝒫[:ϵ])) .* 𝒟[:cost_advantage]))

    # Log useful information
    ignore_derivatives() do
        info[:entropy] = -e_loss
        info[:kl] = mean(𝒟[:logprob] .- new_probs)
        info[:clip_fraction] = sum((r .> 1 + 𝒫[:ϵ]) .| (r .< 1 - 𝒫[:ϵ])) / length(r)
        info[:p_loss] = 𝒫[:λp]*p_loss
        info[:cost_loss] = cost_loss
        info[:avg_advantage] = mean(A_raw)
        info[:avg_cost_advantage] = mean(𝒟[:cost_advantage])
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
    log::NamedTuple=(;),
    required_columns=[],
    kwargs...)
```

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
    log::NamedTuple=(;),
    required_columns=[],
    kwargs...)

     function record_avgr(𝒟; info=Dict(), 𝒮)
         info[:avg_r] = sum(𝒟[:r]) / sum(𝒟[:episode_end])
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
        )

     # Add :value / :cost_value columns when the respective clip is enabled;
     # the sampler's fill_gae! writes Vr(s) / Vc(s) at rollout time so the
     # critic losses can clip their updates.
     extra_cols = Symbol[]
     isnothing(vclip)      || push!(extra_cols, :value)
     isnothing(vclip_cost) || push!(extra_cols, :cost_value)

     OnPolicySolver(;agent=PolicyParams(π),
                    𝒫=𝒫,
                    Vc=Vc,
                    log = LoggerParams(;dir = "log/lagrange_ppo", log...),
                    a_opt = TrainingParams(;loss = lagrange_ppo_loss, early_stopping = (infos) -> (infos[end][:kl] > target_kl), name = "actor_", a_opt...),
                    c_opt = TrainingParams(;loss = ppo_critic_loss, name = "critic_", c_opt...),
                    cost_opt = TrainingParams(;loss = ppo_cost_critic_loss, name = "cost_critic_", cost_opt...),
                    required_columns = unique([required_columns..., :return, :advantage, :logprob, :cost_advantage, :cost, :cost_return, extra_cols...]),
                    post_sample_callback=record_avgr,
                    kwargs...)
end
