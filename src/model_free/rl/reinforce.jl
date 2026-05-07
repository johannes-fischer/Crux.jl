"""
REINFORCE loss function.
"""
function reinforce_loss(m, 𝒫, 𝒟; info = Dict())  # Flux 0.16 port: model-first signature
    new_probs = logpdf(m, 𝒟[:s], 𝒟[:a])

    ignore_derivatives() do
        info[:entropy] = mean(entropy(m, 𝒟[:s]))
        info[:kl] = mean(𝒟[:logprob] .- new_probs)
    end

    -mean(new_probs .* 𝒟[:return])
end

"""
REINFORCE solver.

```julia
REINFORCE(;
    π,
    a_opt::NamedTuple=(;), 
    log::NamedTuple=(;),
    required_columns=[],
    kwargs...)
```
"""
function REINFORCE(;
        π,
        a_opt::NamedTuple=(;), 
        log::NamedTuple=(;),
        required_columns=[],
        kwargs...)

    OnPolicySolver(;agent=PolicyParams(π),
                    log = LoggerParams(;dir = "log/reinforce", log...),
                    a_opt = TrainingParams(;loss = reinforce_loss, early_stopping = (infos) -> (infos[end][:kl] > 0.015), name = "actor_", a_opt...),
                    required_columns = unique([required_columns..., :return, :logprob]),
                    kwargs...)
end
        
    



