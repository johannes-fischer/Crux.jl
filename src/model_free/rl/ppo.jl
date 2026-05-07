"""
PPO loss function.

Flux 0.16 port: takes the differentiated model `m` (the actor) as its first
argument so it composes directly with `Flux.withgradient(m -> …, model)`.
"""
function ppo_loss(m, 𝒫, 𝒟; info = Dict())
    new_probs = logpdf(m, 𝒟[:s], 𝒟[:a])
    r = exp.(new_probs .- 𝒟[:logprob])

    A = 𝒟[:advantage]
    p_loss = -mean(min.(r .* A, clamp.(r, (1f0 - 𝒫[:ϵ]), (1f0 + 𝒫[:ϵ])) .* A))
    e_loss = -mean(entropy(m, 𝒟[:s]))

    # Log useful information
    ignore_derivatives() do
        info[:entropy] = -e_loss
        info[:kl] = mean(𝒟[:logprob] .- new_probs)
        info[:clip_fraction] = sum((r .> 1 + 𝒫[:ϵ]) .| (r .< 1 - 𝒫[:ϵ])) / length(r)
        info[:avg_advantage] = mean(A)
        info[:avg_return] = mean(𝒟[:return])
    end
    𝒫[:λp]*p_loss + 𝒫[:λe]*e_loss
end

"""
Proximal policy optimization (PPO) solver.

```julia
PPO(;
    π::ActorCritic,
    ϵ::Float32 = 0.2f0,
    λp::Float32 = 1f0,
    λe::Float32 = 0.1f0,
    target_kl = 0.012f0,
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
        a_opt::NamedTuple=(;),
        c_opt::NamedTuple=(;),
        log::NamedTuple=(;),
        required_columns=[],
        kwargs...)

     function record_avgr(𝒟; info=Dict(), 𝒮)
         info[:avg_r] = sum(𝒟[:r]) / sum(𝒟[:episode_end])
     end

     OnPolicySolver(;agent=PolicyParams(π),
                    𝒫=(ϵ=ϵ, λp=λp, λe=λe),
                    log = LoggerParams(;dir = "log/ppo", log...),
                    a_opt = TrainingParams(;loss = ppo_loss, early_stopping = (infos) -> (infos[end][:kl] > target_kl), name = "actor_", a_opt...),
                    c_opt = TrainingParams(;loss = (m, 𝒫, D; kwargs...) -> Flux.mse(value(m, D[:s]), D[:return]), name = "critic_", c_opt...),
                    post_batch_callback = (𝒟; kwargs...) -> (𝒟[:advantage] .= whiten(𝒟[:advantage])),
                    required_columns = unique([required_columns..., :return, :logprob, :advantage]),
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

    A = 𝒟[:advantage]
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

        info["penalty"] = penalty
        info["cur_cost"] = Jc
        info["prop_term"] = 𝒫[:Kp] * 𝒫[:smooth_Δ][1]
        info["deriv_term"] = ∂
        info["integral term"] = 𝒫[:I][1]

        penalty
    end

    # cost_loss = 𝒫[:penalty_scale] * penalty * mean(r .* 𝒟[:cost_advantage])
    cost_loss = penalty * mean(max.(r .* 𝒟[:cost_advantage], clamp.(r, (1f0 - 𝒫[:ϵ]), (1f0 + 𝒫[:ϵ])) .* 𝒟[:cost_advantage]))

    # Log useful information
    ignore_derivatives() do
        info[:entropy] = -e_loss
        info[:kl] = mean(𝒟[:logprob] .- new_probs)
        info[:clip_fraction] = sum((r .> 1 + 𝒫[:ϵ]) .| (r .< 1 - 𝒫[:ϵ])) / length(r)
        info["p_loss"] = 𝒫[:λp]*p_loss
        info["cost_loss"] = cost_loss
        info[:avg_advantage] = mean(A)
        info[:avg_return] = mean(𝒟[:return])
    end
    (𝒫[:λp]*p_loss + 𝒫[:λe]*e_loss + cost_loss) / (1 + penalty)
end

"""
Lagrange-Constrained PPO solver.

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
        smooth_Jc = [0f0]
        )

     OnPolicySolver(;agent=PolicyParams(π),
                    𝒫=𝒫,
                    Vc=Vc,
                    log = LoggerParams(;dir = "log/lagrange_ppo", log...),
                    a_opt = TrainingParams(;loss = lagrange_ppo_loss, early_stopping = (infos) -> (infos[end][:kl] > target_kl), name = "actor_", a_opt...),
                    c_opt = TrainingParams(;loss = (m, 𝒫, D; kwargs...) -> Flux.mse(value(m, D[:s]), D[:return]), name = "critic_", c_opt...),
                    cost_opt = TrainingParams(;loss = (m, 𝒫, D; kwargs...) -> Flux.mse(value(m, D[:s]), D[:cost_return]), name = "cost_critic_", cost_opt...),
                    required_columns = unique([required_columns..., :return, :advantage, :logprob, :cost_advantage, :cost, :cost_return]),
                    post_batch_callback = (𝒟; kwargs...) -> (𝒟[:advantage] .= whiten(𝒟[:advantage])),
                    post_sample_callback=record_avgr,
                    kwargs...)
end
