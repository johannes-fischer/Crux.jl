
### LookaheadPolicy

mutable struct LookaheadPolicy <: Policy
    mdp::MDP
    π::NetworkPolicy
end

Flux.@functor LookaheadPolicy (π,)

# Flux.trainable(π::LookaheadPolicy) = Flux.trainable(π.π)
# Flux.layers(π::LookaheadPolicy) = Flux.layers(π.π)
function POMDPs.action(π::LookaheadPolicy, svec)
    mdp = π.mdp
    pomdp = mdp.pomdp
    A = POMDPTools.ordered_actions(pomdp)
    b = convert_s(DiscreteBelief, svec, mdp)

    Qvals = map(A) do a
        R = reward(mdp, b, a)
        for o in observations(pomdp)
            b′ = POMDPs.update(mdp.updater, b, a, o)
            α = value(π.π, b′.b)
            po = sum(
                pdf(transition(pomdp, s, a), sp) * pdf(observation(pomdp, s, a, sp), o) * pdf(b, s)
                for s in states(pomdp), sp in states(pomdp)
            )
            R += discount(pomdp) * po * dot(α, convert_s(Vector, b′, mdp))
        end
        R
    end
    A[argmax(Qvals)]
end

POMDPs.value(π::LookaheadPolicy, s) = value(π.π, s)
action_space(π::LookaheadPolicy) = action_space(π.π)
actor(π::LookaheadPolicy) = π.π
critic(π::LookaheadPolicy) = π.π
### AlphaQPolicy

mutable struct AlphaQPolicy <: Policy
    mdp::MDP
    π::NetworkPolicy
end

Flux.@functor AlphaQPolicy (π,)

function POMDPs.action(π::AlphaQPolicy, svec)
    mdp = π.mdp
    pomdp = mdp.pomdp

    ns = length(states(pomdp))
    na = length(actions(pomdp))
    αQ = reshape(value(π.π, svec), (ns, na))
    idx = argmax(reshape(svec' * αQ, na))
    POMDPTools.ordered_actions(pomdp)[idx]
end

function best_value(π::AlphaQPolicy, svec)
    mdp = π.mdp
    pomdp = mdp.pomdp

    ns = length(states(pomdp))
    na = length(actions(pomdp))
    αQ = reshape(value(π.π, svec), (ns, na))
    idx = argmax(reshape(svec' * αQ, na))
    αQ[:, idx]
end

POMDPs.value(π::AlphaQPolicy, s) = value(π.π, s)
# POMDPs.value(π::AlphaQPolicy, s, a) = value(π.π, s, a)
function POMDPs.value(π::AlphaQPolicy, svec, a_oh)
    mdp = π.mdp
    pomdp = mdp.pomdp

    ns = length(states(pomdp))
    na = length(actions(pomdp))
    αQ = reshape(value(π.π, svec), (ns, na, size(svec)[2:end]...))
    sum(αQ .* reshape(a_oh, (1, na, size(svec)[2:end]...)), dims=2)
end
action_space(π::AlphaQPolicy) = action_space(π.π)
actor(π::AlphaQPolicy) = π.π
critic(π::AlphaQPolicy) = π.π
