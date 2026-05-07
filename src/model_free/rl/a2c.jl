"""
A2C loss function.
"""
function a2c_loss(m, 𝒫, 𝒟; info = Dict())  # Flux 0.16 port: model-first signature
    new_probs = logpdf(m, 𝒟[:s], 𝒟[:a])
    p_loss = -mean(new_probs .* 𝒟[:advantage])
    e_loss = -mean(entropy(m, 𝒟[:s]))
    
    # Log useful information
    ignore_derivatives() do
        info[:entropy] = -e_loss
        info[:kl] = mean(𝒟[:logprob] .- new_probs)
    end 
    
    𝒫[:λp]*p_loss + 𝒫[:λe]*e_loss
end

"""
Advantage actor critic (A2C) solver.

```julia
A2C(;
    π::ActorCritic, 
    a_opt::NamedTuple=(;), 
    c_opt::NamedTuple=(;), 
    log::NamedTuple=(;), 
    λp::Float32=1f0, 
    λe::Float32=0.1f0, 
    required_columns=[],
    kwargs...)
```
"""
function A2C(;
        π::ActorCritic, 
        a_opt::NamedTuple=(;), 
        c_opt::NamedTuple=(;), 
        log::NamedTuple=(;), 
        λp::Float32=1f0, 
        λe::Float32=0.1f0, 
        required_columns=[],
        kwargs...)
              
    OnPolicySolver(;agent=PolicyParams(π),
                    𝒫=(λp=λp, λe=λe),
                    log=LoggerParams(;dir = "log/a2c", log...),
                    a_opt=TrainingParams(;loss=a2c_loss, early_stopping = (infos) -> (infos[end][:kl] > 0.015), name = "actor_", a_opt...),
                    c_opt=TrainingParams(;loss=(m, 𝒫, D; kwargs...) -> Flux.mse(value(m, D[:s]), D[:return]), name = "critic_", c_opt...),
                    post_sample_callback=(𝒟; kwargs...) -> (𝒟[:advantage] .= whiten(𝒟[:advantage])),
                    required_columns = unique([required_columns..., :return, :logprob, :advantage]),
                    kwargs...)
end
        
    



