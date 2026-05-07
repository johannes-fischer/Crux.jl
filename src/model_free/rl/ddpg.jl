"""
DDPG target function.

Set `yᵢ = rᵢ + γQ′(sᵢ₊₁, μ′(sᵢ₊₁ | θᵘ′) | θᶜ′)`
"""
function ddpg_target(π, 𝒫, 𝒟, γ::Float32; kwargs...)
    𝒟[:r] .+ γ .* (1.f0 .- 𝒟[:done]) .* value(π, 𝒟[:sp], action(π, 𝒟[:sp]))
end


"""
Smooth DDPG target.
"""
function smoothed_ddpg_target(π, 𝒫, 𝒟, γ::Float32; i)
    ap, _ = exploration(𝒫[:π_smooth], 𝒟[:sp], π_on=π, i=i)
    y = 𝒟[:r] .+ γ .* (1.f0 .- 𝒟[:done]) .* value(π, 𝒟[:sp], ap)
end


"""
DDPG actor loss function.

`∇_θᵘ 𝐽 ≈ 1/𝑁 Σᵢ ∇ₐQ(s, a | θᶜ)|ₛ₌ₛᵢ, ₐ₌ᵤ₍ₛᵢ₎ ∇_θᵘ μ(s | θᵘ)|ₛᵢ`

Flux 0.16 port: `m` is the differentiated actor; `π_loss` is the full
ActorCritic so we can backprop ∂Q/∂a through the (frozen) critic. We wrap
π_loss in `ignore_derivatives` because Zygote 0.7 otherwise tries to propagate
gradients into both `π_loss.A` (the closed-over copy of the actor) and `m`
(the differentiated arg) — they alias the same object on the heap, and the
internal `accum(::Ref, ::Ref)` assertion fails.
"""
function ddpg_actor_loss(m, 𝒫, 𝒟; info=Dict(), π_loss=m)
    π_frozen = ignore_derivatives(π_loss)
    -mean(value(π_frozen, 𝒟[:s], action(m, 𝒟[:s])))
end


"""
Deep deterministic policy gradient (DDPG) solver.
- T. P. Lillicrap, et al., "Continuous control with deep reinforcement learning", ICLR 2016.

```julia
DDPG(;
    π::ActorCritic, 
    ΔN=50, 
    π_explore=GaussianNoiseExplorationPolicy(0.1f0),  
    a_opt::NamedTuple=(;), 
    c_opt::NamedTuple=(;),
    a_loss=ddpg_actor_loss,
    c_loss=td_loss(),
    target_fn=ddpg_target,
    prefix="",
    log::NamedTuple=(;), 
    π_smooth=GaussianNoiseExplorationPolicy(0.1f0, ϵ_min=-0.5f0, ϵ_max=0.5f0), kwargs...)
```
"""
function DDPG(;
        π::ActorCritic, 
        ΔN=50, 
        π_explore=GaussianNoiseExplorationPolicy(0.1f0),  
        a_opt::NamedTuple=(;), 
        c_opt::NamedTuple=(;),
        a_loss=ddpg_actor_loss,
        c_loss=td_loss(),
        target_fn=ddpg_target,
        prefix="",
        log::NamedTuple=(;), 
        π_smooth=GaussianNoiseExplorationPolicy(0.1f0, ϵ_min=-0.5f0, ϵ_max=0.5f0), kwargs...)
               
    OffPolicySolver(;agent=PolicyParams(π=π, π_explore=π_explore, π⁻=deepcopy(π)), 
                     ΔN=ΔN,
                     𝒫=(π_smooth=π_smooth,),
                     log=LoggerParams(;dir = "log/ddpg", log...),
                     a_opt=TrainingParams(;loss=a_loss, name=string(prefix, "actor_"), a_opt...),
                     c_opt=TrainingParams(;loss=c_loss, name=string(prefix, "critic_"), epochs=ΔN, c_opt...),
                     target_fn=target_fn,
                     kwargs...)
end 
        

