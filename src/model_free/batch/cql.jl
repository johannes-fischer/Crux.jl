# Flux 0.16 port: cql_alpha_loss differentiates the LearnableScalar log_α,
# while cql_critic_loss differentiates the critic; both share `conservative_loss`,
# so we let it accept an explicit `π_full` (frozen, used for sampling and value
# evaluation under cql_alpha_loss) and a separate `β_source` (the LearnableScalar
# that's being differentiated when called from cql_alpha_loss; otherwise read
# from 𝒫[:CQL_log_α] as a constant).
function cql_alpha_loss(m, 𝒫, 𝒟; info=Dict(), π_loss=m)
    ignore_derivatives() do
        info[:CQL_alpha] = exp(m.v[1])
    end
    -conservative_loss(ignore_derivatives(π_loss), 𝒫, 𝒟; β_source=m, info=info)
end

function importance_sampling(πsamp, π, obs, Nsamples)
    @assert ndims(obs) == 2 # does not support multidimensional observations yet
    @assert critic(π) isa DoubleNetwork # Assumes we have a double network

    rep_obs, flat_actions, logprobs = ignore_derivatives() do
        actions_and_logprobs = [exploration(πsamp, obs) for i=1:Nsamples]
        actions = cat([a for (a, _) in actions_and_logprobs]..., dims=3)
        logprobs = cat([lp for (_, lp) in actions_and_logprobs]..., dims=3)
        rep_obs = repeat(obs, 1, Nsamples)
        flat_actions = reshape(actions, size(actions)[1], :)
        rep_obs, flat_actions, logprobs
    end

    qvals = reshape(mean(value(π, rep_obs, flat_actions)), 1, :, Nsamples)

    return qvals .- logprobs
end

function conservative_loss(π, 𝒫, 𝒟; info=Dict(), β_source=𝒫[:CQL_log_α])
    obs = 𝒟[:s]
    acts = 𝒟[:a]
    pol_values = importance_sampling(π, π, obs, 𝒫[:CQL_n_action_samples])
    unif_values = importance_sampling(𝒫[:CQL_is_distribution], π, obs, 𝒫[:CQL_n_action_samples])
    combined = cat(pol_values, unif_values, dims=3)
    lse = logsumexp(combined, dims=3)
    loss = mean(lse) - mean(mean(value(π, obs, acts)))

    β = clamp(exp(β_source.v[1]), 0f0, 1f6)
    β * (5f0*loss - 𝒫[:CQL_α_thresh])
end

function cql_critic_loss(;kwargs...)
    Q2loss = double_Q_loss(;kwargs...)
    # `m` is the differentiated DoubleNetwork critic; `π_loss` is the full
    # ActorCritic so importance_sampling can run exploration on the actor.
    # The critic value() calls inside conservative_loss go through `π_loss.C`
    # which Functors-unwraps onto the actual critic m — we sidestep aliasing by
    # rebuilding a temporary ActorCritic that pairs the frozen actor with `m`.
    (m, 𝒫, 𝒟, y; info=Dict(), π_loss=m) -> begin
        td_loss = Q2loss(m, 𝒫, 𝒟, y, info=info, π_loss=π_loss)
        π_for_cql = ActorCritic(ignore_derivatives(π_loss).A, m)
        c_loss = conservative_loss(π_for_cql, 𝒫, 𝒟, info=info)
        td_loss + c_loss
    end
end


"""
Conservative Q-Learning (CQL) solver.

```julia
CQL(;
    π::ActorCritic{T, DoubleNetwork{ContinuousNetwork, ContinuousNetwork}},
    solver_type=BatchSAC,
    CQL_α::Float32=1f0,
    CQL_is_distribution=DistributionPolicy(product_distribution([Uniform(-1,1) for i=1:dim(action_space(π))[1]])),
    CQL_α_thresh::Float32=10f0,
    CQL_n_action_samples::Int=10,
    CQL_α_opt::NamedTuple=(;),
    a_opt::NamedTuple=(;), 
    c_opt::NamedTuple=(;), 
    log::NamedTuple=(;),
    kwargs...)
```
"""
function CQL(;
        π::ActorCritic{T, DoubleNetwork{ContinuousNetwork, ContinuousNetwork}},
        solver_type=BatchSAC,
        CQL_α::Float32=1f0,
        CQL_is_distribution=DistributionPolicy(product_distribution([Uniform(-1,1) for i=1:dim(action_space(π))[1]])),
        CQL_α_thresh::Float32=10f0,
        CQL_n_action_samples::Int=10,
        CQL_α_opt::NamedTuple=(;),
        a_opt::NamedTuple=(;), 
        c_opt::NamedTuple=(;), 
        log::NamedTuple=(;),
        kwargs...) where T

    # Fill the parameters. Flux 0.16 port: CQL_log_α is a LearnableScalar so
    # the param_optimizers dict has a stable identity-keyed entry for it.
    𝒫 = (CQL_log_α=LearnableScalar(Base.log(CQL_α)),
          CQL_is_distribution=CQL_is_distribution,
          CQL_n_action_samples=CQL_n_action_samples,
          CQL_α_thresh=CQL_α_thresh)
    solver_type(;
        π=π,
        𝒫=𝒫,
        log=(;dir = "log/cql", log...),
        param_optimizers=Dict{Any,TrainingParams}(𝒫[:CQL_log_α] => TrainingParams(;loss=cql_alpha_loss, name="CQL_alpha_", CQL_α_opt...)),
        a_opt=a_opt,
        c_opt=(loss=cql_critic_loss(), name="critic_", c_opt...),
        kwargs...)
end

