"""
SAC target function. Not differentiated — `π⁻` is the target ActorCritic and
the closed-over `π` is the *current* policy (used to sample fresh actions).

Flux 0.16 port: `𝒫[:SAC_log_α]` is now a `LearnableScalar`, accessed via `.v[1]`.
"""
function sac_target(π)
    (π⁻, 𝒫, 𝒟, γ; kwargs...) -> begin
        ap, logprob = exploration(actor(π), 𝒟[:sp])
        y = 𝒟[:r] .+ γ .* (1.f0 .- 𝒟[:done]) .* (min.(value(π⁻, 𝒟[:sp], ap)...) .- exp(𝒫[:SAC_log_α].v[1]).*logprob)
    end
end

"""
Deterministic SAC target function.
"""
function sac_deterministic_target(π)
    (π⁻, 𝒫, 𝒟, γ; kwargs...) -> begin
        y = 𝒟[:r] .+ γ .* (1.f0 .- 𝒟[:done]) .* min.(value(π⁻, 𝒟[:sp], action(actor(π), 𝒟[:sp]))...)
    end
end

"""
Max-Q SAC target function.
"""
function sac_max_q_target(π)
    (π⁻, 𝒫, 𝒟, γ; kwargs...) -> begin
        error("not implemented")
        #TODO: Sample some number of actions and then choose the max
    end
end


"""
SAC actor loss function.

Flux 0.16 port: `m` is the differentiated actor (= π_loss.A); `π_loss` is the
full ActorCritic so we can backprop ∂Q/∂a through the (frozen) twin critics.
The `ignore_derivatives` wrap on π_loss avoids the actor/π_loss aliasing trap
that breaks Zygote 0.7's accum pass (see ddpg_actor_loss).
"""
function sac_actor_loss(m, 𝒫, 𝒟; info=Dict(), π_loss=m)
    π_frozen = ignore_derivatives(π_loss)
    a, logprob = exploration(m, 𝒟[:s])
    ignore_derivatives() do
        info[:entropy] = -mean(logprob)
    end
    α = exp(𝒫[:SAC_log_α].v[1])     # constant w.r.t. m
    mean(α .* logprob .- min.(value(π_frozen, 𝒟[:s], a)...))
end

"""
SAC temp-based loss function — differentiated w.r.t. the LearnableScalar log-α.

`m::LearnableScalar` is the model passed by off_policy.jl's
`for (θs, p_opt) in 𝒮.param_optimizers` loop. `π_loss` is the full ActorCritic
(frozen) for sampling fresh actions to compute the entropy target.
"""
function sac_temp_loss(m, 𝒫, 𝒟; info=Dict(), π_loss=m)
    log_α = m.v[1]
    α = exp(log_α)
    ignore_derivatives() do
        info[:SAC_alpha] = α
    end
    π_frozen = ignore_derivatives(π_loss)
    _, logprob = exploration(π_frozen.A, 𝒟[:s])
    target_α = logprob .+ 𝒫[:SAC_H_target]
    -mean(α .* target_α)
end


"""
Soft Actor Critic (SAC) solver.

```julia
SAC(;
    π::ActorCritic{T, DoubleNetwork{ContinuousNetwork, ContinuousNetwork}},
    ΔN=50,
    SAC_α::Float32=1f0,
    SAC_H_target::Float32 = Float32(-prod(dim(action_space(π)))),
    π_explore=GaussianNoiseExplorationPolicy(0.1f0),
    SAC_α_opt::NamedTuple=(;),
    a_opt::NamedTuple=(;),
    c_opt::NamedTuple=(;),
    a_loss=sac_actor_loss,
    c_loss=double_Q_loss(),
    target_fn=sac_target(π),
    prefix="",
    log::NamedTuple=(;),
    𝒫::NamedTuple=(;),
    param_optimizers=Dict(),
    kwargs...)
```
"""
function SAC(;
        π::ActorCritic{T, DoubleNetwork{ContinuousNetwork, ContinuousNetwork}},
        ΔN=50,
        SAC_α::Float32=1f0,
        SAC_H_target::Float32 = Float32(-prod(dim(action_space(π)))),
        π_explore=GaussianNoiseExplorationPolicy(0.1f0),
        SAC_α_opt::NamedTuple=(;),
        a_opt::NamedTuple=(;),
        c_opt::NamedTuple=(;),
        a_loss=sac_actor_loss,
        c_loss=double_Q_loss(),
        target_fn=sac_target(π),
        prefix="",
        log::NamedTuple=(;),
        𝒫::NamedTuple=(;),
        param_optimizers=Dict(),
        kwargs...) where T

    # Flux 0.16 port: SAC_log_α is now a LearnableScalar (not a raw [Float32]).
    # The dict key in param_optimizers IS the model train! differentiates, so it
    # needs stable identity-based equality — a mutable struct gives us that, and
    # the same instance is shared with 𝒫 so updates from sac_temp_loss are
    # immediately visible to sac_actor_loss / sac_target.
    𝒫 = (SAC_log_α=LearnableScalar(Base.log(SAC_α)), SAC_H_target=SAC_H_target, 𝒫...)
    OffPolicySolver(;agent=PolicyParams(π=π, π_explore=π_explore, π⁻=deepcopy(π)),
                     ΔN=ΔN,
                     𝒫=𝒫,
                     log=LoggerParams(;dir = "log/sac", log...),
                     param_optimizers=Dict{Any,TrainingParams}(𝒫[:SAC_log_α] => TrainingParams(;loss=sac_temp_loss, name="temp_", SAC_α_opt...), param_optimizers...),
                     a_opt=TrainingParams(;loss=a_loss, name=string(prefix, "actor_"), a_opt...),
                     c_opt=TrainingParams(;loss=c_loss, name=string(prefix, "critic_"), epochs=ΔN, c_opt...),
                     target_fn=target_fn,
                     kwargs...)
end
