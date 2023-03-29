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

function alphaDVN_target_single_bao(π, 𝒫, 𝒟, γ::Float32; kwargs...)
    mdp = π.mdp
    alphas = map(eachcol(𝒟[:s])) do s
        @assert length(s) == 2
        b = convert_s(DiscreteBelief, collect(s), mdp)
        pomdp = mdp.pomdp

        S = POMDPTools.ordered_states(pomdp)
        A = POMDPTools.ordered_actions(pomdp)
        O = POMDPTools.ordered_observations(pomdp)
        r = StateActionReward(pomdp)

        Γa = Vector{Vector{Float32}}(undef, length(A))

        not_terminals = [stateindex(pomdp, s) for s in S if !isterminal(pomdp, s)]
        terminals = [stateindex(pomdp, s) for s in S if isterminal(pomdp, s)]
        for a in A
            Γao = Vector{Vector{Float32}}(undef, length(O))
            trans_probs = dropdims(sum([pdf(transition(pomdp, S[is], a), sp) * b.b[is] for sp in S, is in not_terminals], dims=2), dims=2)
            if !isempty(terminals) trans_probs[terminals] .+= b.b[terminals] end

            for o in O
                # update beliefs
                obs_probs = pdf.(map(sp -> observation(pomdp, a, sp), S), [o])
                b′ = obs_probs .* trans_probs

                if sum(b′) > 0.
                    b′ = DiscreteBelief(pomdp, b.state_list, belief_norm(pomdp, b.b, b′, terminals, not_terminals))
                else
                    b′ = DiscreteBelief(pomdp, b.state_list, zeros(length(S)))
                end

                # extract optimal alpha vector at resulting belief
                # Γao[obsindex(pomdp, o)] = _argmax(α -> dot(α,b′.b), Γ)
                bao_vec = convert_s(Vector, b′, mdp)
                Γao[obsindex(pomdp, o)] = value(π, bao_vec)
            end

            # construct new alpha vectors
            Γa[actionindex(pomdp, a)] = [r(s, a) + (!isterminal(pomdp, s) ? (γ * sum(pdf(transition(pomdp, s, a), sp) * pdf(observation(pomdp, s, a, sp), o) * Γao[i][j]
                                            for (j, sp) in enumerate(S), (i, o) in enumerate(O))) : 0.)
                                            for s in S]
        end

        # find the optimal alpha vector
        idx = argmax(map(αa -> αa ⋅ b.b, Γa))
        alphavec = AlphaVec(Γa[idx], A[idx])
        return alphavec.alpha
    end
    reduce(hcat, alphas)
end

function alphaDVN_target_all_bao(π, 𝒫, 𝒟, γ::Float32; kwargs...)
    mdp = π.mdp
    alphas = map(eachcol(𝒟[:s])) do s
        @assert length(s) == 2
        b = convert_s(DiscreteBelief, collect(s), mdp)
        pomdp = mdp.pomdp

        S = ordered_states(pomdp)
        A = ordered_actions(pomdp)
        O = ordered_observations(pomdp)
        r = StateActionReward(pomdp)


        not_terminals = [stateindex(pomdp, s) for s in S if !isterminal(pomdp, s)]
        terminals = [stateindex(pomdp, s) for s in S if isterminal(pomdp, s)]

        Γnet = Vector{Vector{Float64}}(undef, length(A) * length(O))
        i = 1

        for a in A
            trans_probs = dropdims(sum([pdf(transition(pomdp, S[is], a), sp) * b.b[is] for sp in S, is in not_terminals], dims=2), dims=2)
            if !isempty(terminals) trans_probs[terminals] .+= b.b[terminals] end

            for o in O
                # update beliefs
                obs_probs = pdf.(map(sp -> observation(pomdp, a, sp), S), [o])
                b′ = obs_probs .* trans_probs

                if sum(b′) > 0.
                    b′ = DiscreteBelief(pomdp, b.state_list, belief_norm(pomdp, b.b, b′, terminals, not_terminals))
                else
                    b′ = DiscreteBelief(pomdp, b.state_list, zeros(length(S)))
                end

                # extract optimal alpha vector at resulting belief
                # Γao[obsindex(pomdp, o)] = _argmax(α -> dot(α,b′.b), Γ)
                bao_vec = convert_s(Vector, b′, mdp)
                Γnet[i] = value(π, bao_vec)
                i += 1
            end
        end

        Γa = Vector{Vector{Float32}}(undef, length(A))
        for a in A
            Γao = Vector{Vector{Float32}}(undef, length(O))
            trans_probs = dropdims(sum([pdf(transition(pomdp, S[is], a), sp) * b.b[is] for sp in S, is in not_terminals], dims=2), dims=2)
            if !isempty(terminals) trans_probs[terminals] .+= b.b[terminals] end

            for o in O
                # update beliefs
                obs_probs = pdf.(map(sp -> observation(pomdp, a, sp), S), [o])
                b′ = obs_probs .* trans_probs

                if sum(b′) > 0.
                    b′ = DiscreteBelief(pomdp, b.state_list, belief_norm(pomdp, b.b, b′, terminals, not_terminals))
                else
                    b′ = DiscreteBelief(pomdp, b.state_list, zeros(length(S)))
                end

                # extract optimal alpha vector at resulting belief
                Γao[obsindex(pomdp, o)] = _argmax(α -> dot(α,b′.b), Γnet)
            end

            # construct new alpha vectors
            Γa[actionindex(pomdp, a)] = [r(s, a) + (!isterminal(pomdp, s) ? (γ * sum(pdf(transition(pomdp, s, a), sp) * pdf(observation(pomdp, s, a, sp), o) * Γao[i][j]
                                            for (j, sp) in enumerate(S), (i, o) in enumerate(O))) : 0.)
                                            for s in S]
        end

        # find the optimal alpha vector
        idx = argmax(map(αa -> αa ⋅ b.b, Γa))
        alphavec = AlphaVec(Γa[idx], A[idx])
        return alphavec.alpha
    end
    reduce(hcat, alphas)
end

function sampledAlphaDVN(π, 𝒫, 𝒟, γ::Float32; kwargs...)
    mdp = π.mdp
    alphas = map(eachcol(𝒟[:s])) do s
        @assert length(s) == 2
        B = state_type(mdp)
        b = convert_s(B, collect(s), mdp)
        pomdp = mdp.pomdp

        S = POMDPTools.ordered_states(pomdp)
        A = POMDPTools.ordered_actions(pomdp)
        O = POMDPTools.ordered_observations(pomdp)
        r = StateActionReward(pomdp)

        Γa = Vector{Vector{Float32}}(undef, length(A))

        not_terminals = [stateindex(pomdp, s) for s in S if !isterminal(pomdp, s)]
        terminals = [stateindex(pomdp, s) for s in S if isterminal(pomdp, s)]
        for a in A
            Γao = Vector{Vector{Float32}}(undef, length(O))
            trans_probs = dropdims(sum([pdf(transition(pomdp, S[is], a), sp) * b.b[is] for sp in S, is in not_terminals], dims=2), dims=2)
            if !isempty(terminals) trans_probs[terminals] .+= b.b[terminals] end

            for o in O
                # update beliefs
                obs_probs = pdf.(map(sp -> observation(pomdp, a, sp), S), [o])
                b′ = obs_probs .* trans_probs

                if sum(b′) > 0.
                    b′ = B(pomdp, belief_norm(pomdp, b.b, b′, terminals, not_terminals))
                else
                    b′ = B(pomdp, zeros(length(S)))
                end

                # extract optimal alpha vector at resulting belief
                # Γao[obsindex(pomdp, o)] = _argmax(α -> dot(α,b′.b), Γ)
                bao_vec = convert_s(Vector, b′, mdp)
                Γao[obsindex(pomdp, o)] = value(π, bao_vec)
            end

            # construct new alpha vectors
            Γa[actionindex(pomdp, a)] = [r(s, a) + (!isterminal(pomdp, s) ? (γ * sum(pdf(transition(pomdp, s, a), sp) * pdf(observation(pomdp, s, a, sp), o) * Γao[i][j]
                                            for (j, sp) in enumerate(S), (i, o) in enumerate(O))) : 0.)
                                            for s in S]
        end

        # find the optimal alpha vector
        idx = argmax(map(αa -> αa ⋅ b.b, Γa))
        alphavec = AlphaVec(Γa[idx], A[idx])
        return alphavec.alpha
    end
    reduce(hcat, alphas)
end

function alphaDQN_target(π, 𝒫, 𝒟, γ::Float32; kwargs...)
    mdp = π.mdp
    pomdp = mdp.pomdp
    S = POMDPTools.ordered_states(pomdp)
    A = POMDPTools.ordered_actions(pomdp)
    O = POMDPTools.ordered_observations(pomdp)
    r = StateActionReward(pomdp)
    ns = length(S)
    na = length(A)

    alphas = map(zip(eachcol(𝒟[:s]), eachcol(𝒟[:a]))) do (s, a_oh)
        b = convert_s(DiscreteBelief, collect(s), mdp)

        @assert length(s) == length(states(pomdp))
        @assert length(a_oh) == length(actions(pomdp))

        not_terminals = [stateindex(pomdp, s) for s in S if !isterminal(pomdp, s)]
        terminals = [stateindex(pomdp, s) for s in S if isterminal(pomdp, s)]

        a = A[argmax(a_oh)]

        Γao = Vector{Vector{Float32}}(undef, length(O))
        trans_probs = dropdims(sum([pdf(transition(pomdp, S[is], a), sp) * b.b[is] for sp in S, is in not_terminals], dims=2), dims=2)
        if !isempty(terminals) trans_probs[terminals] .+= b.b[terminals] end

        for o in O
            # update beliefs
            obs_probs = pdf.(map(sp -> observation(pomdp, a, sp), S), [o])
            b′ = obs_probs .* trans_probs

            if sum(b′) > 0.
                b′ = DiscreteBelief(pomdp, b.state_list, belief_norm(pomdp, b.b, b′, terminals, not_terminals))
            else
                b′ = DiscreteBelief(pomdp, b.state_list, zeros(length(S)))
            end

            # extract optimal alpha vector at resulting belief
            # Γao[obsindex(pomdp, o)] = _argmax(α -> dot(α,b′.b), Γ)
            bao_vec = convert_s(Vector, b′, mdp)
            αQ = reshape(value(π.π, bao_vec), (ns, na))
            idx = argmax(map(αa -> αa ⋅ bao_vec, eachcol(αQ)))
            # alphavec = AlphaVec(Γa[idx], A[idx])
            # idx = argmax(reshape(bao_vec * αQ, na))
            Γao[obsindex(pomdp, o)] = αQ[:, idx]
        end

        # construct new alpha vectors
        α = Float32[r(s, a) + (!isterminal(pomdp, s) ? (γ * sum(pdf(transition(pomdp, s, a), sp) * pdf(observation(pomdp, s, a, sp), o) * Γao[i][j]
                                        for (j, sp) in enumerate(S), (i, o) in enumerate(O))) : 0.)
                                        for s in S]

        # find the optimal alpha vector
        # idx = argmax(map(αa -> αa ⋅ b.b, Γa))
        # alphavec = AlphaVec(Γa[idx], A[idx])
        return α
    end
    res = reduce(hcat, alphas)
    reshaped = reshape(res, (length(S), 1, length(alphas)))

    reshaped
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
                      log=LoggerParams(;dir="log/dqn", log...),
                      N=N,
                      ΔN=ΔN,
                      c_opt = TrainingParams(;loss=c_loss, name=string(prefix, "critic_"), epochs=ΔN, c_opt...),
                      target_fn=target_fn,
                      kwargs...)
end


