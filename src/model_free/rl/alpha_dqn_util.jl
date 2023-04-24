struct AlphaVec
    alpha::Vector{Float32}
    action::Any
end

# adds probabilities of terminals in b to b′ and normalizes b′
function belief_norm(pomdp, b, b′, terminals, not_terminals)
    if sum(b′[not_terminals]) != 0.
        if !isempty(terminals)
            b′[not_terminals] = b′[not_terminals] / (sum(b′[not_terminals]) / (1. - sum(b[terminals]) - sum(b′[terminals])))
            b′[terminals] += b[terminals]
        else
            b′[not_terminals] /= sum(b′[not_terminals])
        end
    else
        b′[terminals] += b[terminals]
        b′[terminals] /= sum(b′[terminals])
    end
    return b′
end

function _argmax(f, X)
    return X[argmax(map(f, X))]
end

function AlphaDQN(;π::Crux.Policy,
    N::Int,
    ΔN=4,
    π_explore=ϵGreedyPolicy(LinearDecaySchedule(1., 0.1, floor(Int, N/2)), π.outputs),
    c_opt::NamedTuple=(;),
    log::NamedTuple=(;),
    c_loss=td_Qloss(),
    target_fn=alphaDQN_target,
    prefix="",
    kwargs...)

    OffPolicySolver(;agent=PolicyParams(π=π, π_explore=π_explore, π⁻=deepcopy(π)),
            log=LoggerParams(;dir="log/adqn", log...),
            N=N,
            ΔN=ΔN,
            c_opt = TrainingParams(;loss=c_loss, name=string(prefix, "critic_"), epochs=ΔN, c_opt...),
            target_fn=target_fn,
            required_columns = [:o, :s_pomdp, :sp_pomdp],
            kwargs...)
end


