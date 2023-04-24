
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
    reshape(res, (length(S), 1, length(alphas)))
end

function alphaDQN_target_generalize(π, 𝒫, 𝒟, γ::Float32; kwargs...)
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

        B_discretization = 100
        na = length(A)
        Γlength = na * (B_discretization + length(O))
        Γnet = Vector{Vector{Float64}}(undef, Γlength)
        i = 1

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
            Γnet[i:i+na-1] = collect(eachcol(αQ))
            i += na
        end
        for x in LinRange(0, 1, B_discretization)
            @assert i <= Γlength
            bao_vec = [x, 1-x]
            αQ = reshape(value(π.π, bao_vec), (ns, na))
            Γnet[i:i+na-1] = collect(eachcol(αQ))
            i += na
        end

        for o in O
            # update beliefs
            obs_probs = pdf.(map(sp -> observation(pomdp, a, sp), S), [o])
            b′ = obs_probs .* trans_probs

            if sum(b′) > 0.
                b′ = DiscreteBelief(pomdp, b.state_list, belief_norm(pomdp, b.b, b′, terminals, not_terminals))
            else
                b′ = DiscreteBelief(pomdp, b.state_list, zeros(length(S)))
            end

            # # extract optimal alpha vector at resulting belief
            # # Γao[obsindex(pomdp, o)] = _argmax(α -> dot(α,b′.b), Γ)
            # bao_vec = convert_s(Vector, b′, mdp)
            # αQ = reshape(value(π.π, bao_vec), (ns, na))
            # idx = argmax(map(αa -> αa ⋅ bao_vec, eachcol(αQ)))
            # # alphavec = AlphaVec(Γa[idx], A[idx])
            # # idx = argmax(reshape(bao_vec * αQ, na))
            # Γao[obsindex(pomdp, o)] = αQ[:, idx]
            Γao[obsindex(pomdp, o)] = _argmax(α -> dot(α,b′.b), Γnet)
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
    reshape(res, (length(S), 1, length(alphas)))
end

function alphaDQN_sampleTarget(π, 𝒫, 𝒟, γ::Float32; kwargs...)
    mdp = π.mdp
    pomdp = mdp.pomdp
    S = POMDPTools.ordered_states(pomdp)
    A = POMDPTools.ordered_actions(pomdp)
    O = POMDPTools.ordered_observations(pomdp)
    r = StateActionReward(pomdp)
    ns = length(S)
    na = length(A)

    alphas = map(zip(eachcol(𝒟[:s]), eachcol(𝒟[:a]))) do (s, a_oh)
        B = statetype(mdp)
        b = convert_s(B, collect(s), mdp)

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
                b′ = DiscreteBelief(pomdp, b.state_list, zeros(ns))
            end

            # extract optimal alpha vector at resulting belief
            bao_vec = convert_s(Vector, b′, mdp)
            αQ = reshape(value(π.π, bao_vec), (ns, na))
            idx = argmax(map(αa -> αa ⋅ bao_vec, eachcol(αQ)))
            Γao[obsindex(pomdp, o)] = αQ[:, idx]
        end

        # construct new alpha vectors
        α = zeros(Float32, length(S))
        for s in S
            for _ in 1:π.m
                sp, o, r = @gen(:sp,:o,:r)(pomdp, s, a)
                αo = Γao[obsindex(pomdp, o)]
                v = r + (!isterminal(pomdp, s) ? (γ * αo[stateindex(pomdp, sp)]) : 0.)
                α[stateindex(pomdp, s)] += v / π.m
            end
        end
        α
    end
    res = reduce(hcat, alphas)
    reshape(res, (length(S), 1, length(alphas)))
end

function alphaDQN_weightedSampleTarget(π, 𝒫, 𝒟, γ::Float32; kwargs...)
    mdp = π.mdp
    pomdp = mdp.pomdp
    S = POMDPTools.ordered_states(pomdp)
    A = POMDPTools.ordered_actions(pomdp)
    O = POMDPTools.ordered_observations(pomdp)
    r = StateActionReward(pomdp)
    ns = length(S)

    alphas = map(zip(eachcol(𝒟[:s]), eachcol(𝒟[:a]), eachcol(𝒟[:o]), eachcol(𝒟[:sp]))) do (b_vec, a_oh, ovec, bp_vec)
        o = convert_o(obstype(pomdp), collect(ovec), mdp.pomdp)

        a = A[argmax(a_oh)]

        # extract optimal alpha vector at resulting belief
        αQ = reshape(value(π.π, bp_vec), (ns, na))
        idx = argmax(map(αa -> αa ⋅ bao_vec, eachcol(αQ)))
        αo = αQ[:, idx]

        # construct new alpha vectors
        α = zeros(length(S))
        for s in S
            weights = Vector{Float64}(undef, π.m)
            values = Vector{Float64}(undef, π.m)
            α[stateindex(pomdp, s)] = r(s, a)
            if !isterminal(pomdp, s)
                for i in 1:π.m
                    sp = @gen(:sp)(pomdp, s, a)
                    weights[i] = obs_weight(pomdp, s, a, sp, o)
                    values[i] = αo[stateindex(pomdp, sp)]
                end
                α[stateindex(pomdp, s)] += γ * dot(weights, values) / sum(weights)
            end
        end
        return α
    end
    reduce(hcat, alphas)
end

function alphaDQN_weightedSampleTargetAllBelief(π, 𝒫, 𝒟, γ::Float32; kwargs...)
    mdp = π.mdp
    pomdp = mdp.pomdp
    S = POMDPTools.ordered_states(pomdp)
    A = POMDPTools.ordered_actions(pomdp)
    O = POMDPTools.ordered_observations(pomdp)
    r = StateActionReward(pomdp)
    ns = length(S)

    B_discretization = 100
    na = length(A)
    Γlength = na * B_discretization
    Γnet = Vector{Vector{Float64}}(undef, Γlength)
    i=1
    for x in LinRange(0, 1, B_discretization)
        @assert i <= Γlength
        bao_vec = [x, 1-x]
        αQ = reshape(value(π.π, bao_vec), (ns, na))
        Γnet[i:i+na-1] = collect(eachcol(αQ))
        i += na
    end

    alphas = map(zip(eachcol(𝒟[:s]), eachcol(𝒟[:a]), eachcol(𝒟[:o]), eachcol(𝒟[:sp]))) do (b_vec, a_oh, ovec, bp_vec)
        o = convert_o(obstype(pomdp), collect(ovec), mdp.pomdp)

        a = A[argmax(a_oh)]

        bpvec = collect(spvec)
        α_all = _argmax(α -> dot(α, bpvec), Γnet)

        # extract optimal alpha vector at resulting belief
        αQ = reshape(value(π.π, bp_vec), (ns, na))
        αo = _argmax(αa -> αa ⋅ bpvec, collect(eachcol(αQ)))

        if dot(α_all, bpvec) > dot(αo, bpvec)
            αo = α_all
        end

        # construct new alpha vectors
        α = zeros(length(S))
        for s in S
            weights = Vector{Float64}(undef, π.m)
            values = Vector{Float64}(undef, π.m)
            α[stateindex(pomdp, s)] = r(s, a)
            if !isterminal(pomdp, s)
                for i in 1:π.m
                    sp = @gen(:sp)(pomdp, s, a)
                    weights[i] = obs_weight(pomdp, s, a, sp, o)
                    values[i] = αo[stateindex(pomdp, sp)]
                end
                α[stateindex(pomdp, s)] += γ * dot(weights, values) / sum(weights)
            end
        end
        return α
    end
    reduce(hcat, alphas)
end

function alphaDQN_singleSampleTarget(π, 𝒫, 𝒟, γ::Float32; kwargs...)
    mdp = π.mdp
    pomdp = mdp.pomdp
    S = POMDPTools.ordered_states(pomdp)
    A = POMDPTools.ordered_actions(pomdp)
    B = statetype(mdp)
    r = StateActionReward(pomdp)
    ns = length(S)
    na = length(A)

    alphas = map(zip(eachcol(𝒟[:s]), eachcol(𝒟[:a]))) do (svec, a_oh)
        b = convert_s(B, collect(svec), mdp)

        a = A[argmax(a_oh)]

        # construct new alpha vectors
        α = value(π, svec, a_oh)

        s = rand(b)
        sp, o, r = @gen(:sp,:o,:r)(pomdp, s, a)

        bp = POMDPs.update(mdp.updater, b, a, o)
        bpvec = convert_s(Vector, bp, mdp)

        αQ = reshape(value(π.π, bpvec), (ns, na))
        idx = argmax(map(αa -> αa ⋅ bpvec, eachcol(αQ)))
        αo = αQ[:, idx]
        v = r + (!isterminal(pomdp, s) ? (γ * αo[stateindex(pomdp, sp)]) : 0.)
        α[stateindex(pomdp, s)] = v
        α
    end
    res = reduce(hcat, alphas)
    reshape(res, (length(S), 1, length(alphas)))
end

function alphaDQN_weightedSampleTarget(π, 𝒫, 𝒟, γ::Float32; kwargs...)
    mdp = π.mdp
    pomdp = mdp.pomdp
    S = POMDPTools.ordered_states(pomdp)
    A = POMDPTools.ordered_actions(pomdp)
    B = statetype(mdp)
    r = StateActionReward(pomdp)
    ns = length(S)
    na = length(A)

    alphas = map(zip(eachcol(𝒟[:s]), eachcol(𝒟[:s_pomdp]), eachcol(𝒟[:a]), eachcol(𝒟[:o]), eachcol(𝒟[:sp]), eachcol(𝒟[:sp_pomdp]), 𝒟[:r])) do (svec, s_pomdp_vec, a_oh, ovec, spvec, sp_pomdp_vec, r)
        b = convert_s(B, collect(svec), mdp)

        a = A[argmax(a_oh)]

        # construct new alpha vectors
        α = value(π, svec, a_oh)

        s = rand(b)
        sp, o, r = @gen(:sp,:o,:r)(pomdp, s, a)

        bp = POMDPs.update(mdp.updater, b, a, o)
        bpvec = convert_s(Vector, bp, mdp)

        αQ = reshape(value(π.π, bpvec), (ns, na))
        idx = argmax(map(αa -> αa ⋅ bpvec, eachcol(αQ)))
        αo = αQ[:, idx]
        v = r + (!isterminal(pomdp, s) ? (γ * αo[stateindex(pomdp, sp)]) : 0.)
        α[stateindex(pomdp, s)] = v
        α
    end
    res = reduce(hcat, alphas)
    reshape(res, (length(S), 1, length(alphas)))
end

function alphaDQN_modelFreeTrueStateBackups2(π, 𝒫, 𝒟, γ::Float32; kwargs...)
    mdp = π.mdp
    pomdp = mdp.pomdp
    S = POMDPTools.ordered_states(pomdp)
    A = POMDPTools.ordered_actions(pomdp)
    B = statetype(mdp)
    r = StateActionReward(pomdp)
    ns = length(S)
    na = length(A)

    alphas = map(zip(eachcol(𝒟[:s]), eachcol(𝒟[:s_pomdp]), eachcol(𝒟[:a]), eachcol(𝒟[:o]), eachcol(𝒟[:sp]), eachcol(𝒟[:sp_pomdp]), 𝒟[:r])) do (svec, s_pomdp_vec, a_oh, ovec, spvec, sp_pomdp_vec, r)
        # b = convert_s(B, collect(svec), mdp)

        # a = A[argmax(a_oh)]

        # construct new alpha vectors
        α = value(π, svec, a_oh)

        s = convert_s(statetype(pomdp), collect(s_pomdp_vec), pomdp)
        sp = convert_s(statetype(pomdp), collect(sp_pomdp_vec), pomdp)
        # o = convert_o(obstype(pomdp), collect(ovec), mdp.pomdp)

        # bp = POMDPs.update(mdp.updater, b, a, o)
        # bpvec = convert_s(Vector, bp, mdp)
        bpvec = collect(spvec)

        αQ = reshape(value(π.π, bpvec), (ns, na))
        idx = argmax(map(αa -> αa ⋅ bpvec, eachcol(αQ)))
        αo = αQ[:, idx]
        v = r + (!isterminal(pomdp, s) ? (γ * αo[stateindex(pomdp, sp)]) : 0.)
        α[stateindex(pomdp, s)] = v
        return α
    end
    res = reduce(hcat, alphas)
    reshape(res, (length(S), 1, length(alphas)))
end

function alphaDQN_modelFreeTrueStateBackups2AllBelief(π, 𝒫, 𝒟, γ::Float32; kwargs...)
    mdp = π.mdp
    pomdp = mdp.pomdp
    S = POMDPTools.ordered_states(pomdp)
    A = POMDPTools.ordered_actions(pomdp)
    B = statetype(mdp)
    r = StateActionReward(pomdp)
    ns = length(S)
    na = length(A)

    B_discretization = 100
    na = length(A)
    Γlength = na * B_discretization
    Γnet = Vector{Vector{Float64}}(undef, Γlength)
    i=1
    for x in LinRange(0, 1, B_discretization)
        @assert i <= Γlength
        bao_vec = [x, 1-x]
        αQ = reshape(value(π.π, bao_vec), (ns, na))
        Γnet[i:i+na-1] = collect(eachcol(αQ))
        i += na
    end

    alphas = map(zip(eachcol(𝒟[:s]), eachcol(𝒟[:s_pomdp]), eachcol(𝒟[:a]), eachcol(𝒟[:o]), eachcol(𝒟[:sp]), eachcol(𝒟[:sp_pomdp]), 𝒟[:r])) do (svec, s_pomdp_vec, a_oh, ovec, spvec, sp_pomdp_vec, r)
        # b = convert_s(B, collect(svec), mdp)

        # a = A[argmax(a_oh)]

        # construct new alpha vectors
        α = value(π, svec, a_oh)

        s = convert_s(statetype(pomdp), collect(s_pomdp_vec), pomdp)
        sp = convert_s(statetype(pomdp), collect(sp_pomdp_vec), pomdp)
        # o = convert_o(obstype(pomdp), collect(ovec), mdp.pomdp)

        # bp = POMDPs.update(mdp.updater, b, a, o)
        # bpvec = convert_s(Vector, bp, mdp)
        bpvec = collect(spvec)
        α_all = _argmax(α -> dot(α, bpvec), Γnet)

        αQ = reshape(value(π.π, bpvec), (ns, na))
        αo = _argmax(αa -> αa ⋅ bpvec, collect(eachcol(αQ)))

        if dot(α_all, bpvec) > dot(αo, bpvec)
            αo = α_all
        end

        v = r + (!isterminal(pomdp, s) ? (γ * αo[stateindex(pomdp, sp)]) : 0.)
        sidx = stateindex(pomdp, s)
        α[sidx] = svec[sidx] * v + (1 - svec[sidx]) * α[sidx] # weighting happens here!
        return α
    end
    res = reduce(hcat, alphas)
    reshape(res, (length(S), 1, length(alphas)))
end

function alphaDQN_modelFreeStateBackups(π, 𝒫, 𝒟, γ::Float32; kwargs...)
    mdp = π.mdp
    pomdp = mdp.pomdp
    S = POMDPTools.ordered_states(pomdp)
    A = POMDPTools.ordered_actions(pomdp)
    O = POMDPTools.ordered_observations(pomdp)
    r = StateActionReward(pomdp)
    ns = length(S)
    na = length(A)

    alphas = map(zip(eachcol(𝒟[:s]), eachcol(𝒟[:a]), eachcol(𝒟[:o]))) do (svec, a_oh, ovec)
        B = statetype(mdp)
        b = convert_s(B, collect(svec), mdp)
        o = convert_o(obstype(pomdp), collect(ovec), mdp.pomdp)

        a = A[argmax(a_oh)]

        # construct new alpha vectors
        α = zeros(Float32, length(S))
        for s in S
            s_base = zeros(length(S))
            s_base[stateindex(pomdp, s)] = 1.
            bs = convert_s(B, s_base, mdp)
            bp = POMDPs.update(mdp.updater, bs, a, o)
            bpvec = convert_s(Vector, bp, mdp)
            # αs = value(π, bpvec)
            αQ = reshape(value(π.π, bpvec), (ns, na))
            idx = argmax(map(αa -> αa ⋅ bpvec, eachcol(αQ)))
            αs = αQ[:, idx]
            v = r(s, a) + (!isterminal(pomdp, s) ? (γ * dot(bpvec, αs)) : 0.)
            α[stateindex(pomdp, s)] = v
        end
        return α
    end
    res = reduce(hcat, alphas)
    reshape(res, (length(S), 1, length(alphas)))
end

function alphaDQN_modelFreeWeightedStateBackups(π, 𝒫, 𝒟, γ::Float32; kwargs...)
    mdp = π.mdp
    pomdp = mdp.pomdp
    S = POMDPTools.ordered_states(pomdp)
    A = POMDPTools.ordered_actions(pomdp)
    O = POMDPTools.ordered_observations(pomdp)
    r = StateActionReward(pomdp)
    ns = length(S)
    na = length(A)

    alphas = map(zip(eachcol(𝒟[:s]), eachcol(𝒟[:a]), eachcol(𝒟[:o]))) do (svec, a_oh, ovec)
        B = statetype(mdp)
        b = convert_s(B, collect(svec), mdp)
        o = convert_o(obstype(pomdp), collect(ovec), mdp.pomdp)
        svec = convert(Vector{Float32}, svec)

        a = A[argmax(a_oh)]

        # construct new alpha vectors
        α = zeros(Float32, length(S))
        for s in S
            s_base = zeros(length(S))
            s_base[stateindex(pomdp, s)] = 1.
            bs = convert_s(B, s_base, mdp)
            bp = POMDPs.update(mdp.updater, bs, a, o)
            bpvec = convert_s(Vector, bp, mdp)
            # αs = value(π, bpvec)
            αQ = reshape(value(π.π, bpvec), (ns, na))
            idx = argmax(map(αa -> αa ⋅ bpvec, eachcol(αQ)))
            αs = αQ[:, idx]
            v = r(s, a) + (!isterminal(pomdp, s) ? (γ * dot(bpvec, αs)) : 0.)
            α[stateindex(pomdp, s)] = v
        end
        α = svec .* α + (1 .- svec) .* value(π, svec, a_oh) # weighting happens here!
        return α
    end
    res = reduce(hcat, alphas)
    reshape(res, (length(S), 1, length(alphas)))
end

function alphaDQN_modelFreeTrueStateBackups(π, 𝒫, 𝒟, γ::Float32; kwargs...)
    mdp = π.mdp
    pomdp = mdp.pomdp
    S = POMDPTools.ordered_states(pomdp)
    A = POMDPTools.ordered_actions(pomdp)
    O = POMDPTools.ordered_observations(pomdp)
    r = StateActionReward(pomdp)
    ns = length(S)
    na = length(A)

    alphas = map(zip(eachcol(𝒟[:s]), eachcol(𝒟[:s_pomdp]), eachcol(𝒟[:a]), eachcol(𝒟[:o]), eachcol(𝒟[:sp_pomdp]))) do (svec, s_pomdp_vec, a_oh, ovec, sp_pomdp_vec)
        B = statetype(mdp)
        b = convert_s(B, collect(svec), mdp)
        o = convert_o(obstype(pomdp), collect(ovec), mdp.pomdp)

        a = A[argmax(a_oh)]

        # construct new alpha vectors
        α = value(π, svec, a_oh)

        s = convert_s(statetype(pomdp), collect(s_pomdp_vec), pomdp)
        sp = convert_s(statetype(pomdp), collect(sp_pomdp_vec), pomdp)

        s_base = zeros(length(S))
        s_base[stateindex(pomdp, s)] = 1.
        bs = convert_s(B, s_base, mdp)

        bp = POMDPs.update(mdp.updater, bs, a, o)
        bpvec = convert_s(Vector, bp, mdp)
        # αs = value(π, bpvec)
        αQ = reshape(value(π.π, bpvec), (ns, na))
        idx = argmax(map(αa -> αa ⋅ bpvec, eachcol(αQ)))
        αs = αQ[:, idx]
        v = r(s, a) + (!isterminal(pomdp, s) ? (γ * αs[stateindex(pomdp, sp)]) : 0.)
        α[stateindex(pomdp, s)] = v
        return α
    end
    res = reduce(hcat, alphas)
    reshape(res, (length(S), 1, length(alphas)))
end

function alphaDQN_targetImpSamp(π, 𝒫, 𝒟, γ::Float32; kwargs...)
    mdp = π.mdp
    pomdp = mdp.pomdp
    S = POMDPTools.ordered_states(pomdp)
    A = POMDPTools.ordered_actions(pomdp)
    O = POMDPTools.ordered_observations(pomdp)
    r = StateActionReward(pomdp)
    ns = length(S)
    na = length(A)

    alphas = map(zip(eachcol(𝒟[:s]), eachcol(𝒟[:a]))) do (bvec, a_oh)
        b = convert_s(DiscreteBelief, collect(bvec), mdp)

        @assert length(bvec) == length(states(pomdp))
        @assert length(a_oh) == length(actions(pomdp))

        not_terminals = [stateindex(pomdp, s) for s in S if !isterminal(pomdp, s)]
        terminals = [stateindex(pomdp, s) for s in S if isterminal(pomdp, s)]

        a = A[argmax(a_oh)]

        Γao = Vector{Vector{Float32}}(undef, length(O))
        bbao_vecs = Vector{Vector{Float32}}(undef, length(O))
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
            # idx = argmax(reshape(bao_vec * αQ, na))
            Γao[obsindex(pomdp, o)] = αQ[:, idx]
            bbao_vecs[obsindex(pomdp, o)] = bao_vec
        end

        poba(o) = sum(pdf(b, s) * sum(pdf(transition(pomdp, s, a), sp) * pdf(observation(pomdp, s, a, sp), o) for sp in S) for s in S)

        α = zeros(Float32, length(S))
        for s in S
            idx = stateindex(pomdp, s)
            if bvec[idx] ≈ zero(bvec[idx])
                α[idx] = value(π, bvec, a_oh)[idx]
            else
                # is = abs(bvec[idx]) < sqrt(eps()) ? 0. : (bpvec[idx] / bvec[idx])
                v = r(s,a)
                if !isterminal(pomdp, s)
                    dummy = 0f0
                    for (i, o) in enumerate(O)
                        bpvec = bbao_vecs[i]
                        α_bp = Γao[i]
                        is = bpvec[idx] / bvec[idx]
                        # lb = min(is, 1f1)
                        lb = min(is, 1f10)
                        # lb = is
                        po = convert(Float32, poba(o))
                        v += γ * po * lb * α_bp[idx]
                        dummy += po
                    end
                    @assert dummy ≈ 1f0
                end
                @assert isfinite(v) idx, bvec, bpvec, lb, v, α_bp, α_bp[idx]
                α[idx] = v
            end
        end

        # find the optimal alpha vector
        # idx = argmax(map(αa -> αa ⋅ b.b, Γa))
        # alphavec = AlphaVec(Γa[idx], A[idx])
        return α
    end
    res = reduce(hcat, alphas)
    reshape(res, (length(S), 1, length(alphas)))
end

function alphaDQN_targetImpSampSmoothed(π, 𝒫, 𝒟, γ::Float32; kwargs...)
    mdp = π.mdp
    pomdp = mdp.pomdp
    S = POMDPTools.ordered_states(pomdp)
    A = POMDPTools.ordered_actions(pomdp)
    O = POMDPTools.ordered_observations(pomdp)
    r = StateActionReward(pomdp)
    ns = length(S)
    na = length(A)

    alphas = map(zip(eachcol(𝒟[:s]), eachcol(𝒟[:a]))) do (bvec, a_oh)
        b = convert_s(DiscreteBelief, collect(bvec), mdp)

        @assert length(bvec) == length(states(pomdp))
        @assert length(a_oh) == length(actions(pomdp))

        not_terminals = [stateindex(pomdp, s) for s in S if !isterminal(pomdp, s)]
        terminals = [stateindex(pomdp, s) for s in S if isterminal(pomdp, s)]

        a = A[argmax(a_oh)]

        Γao = Vector{Vector{Float32}}(undef, length(O))
        bbao_vecs = Vector{Vector{Float32}}(undef, length(O))
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
            # idx = argmax(reshape(bao_vec * αQ, na))
            Γao[obsindex(pomdp, o)] = αQ[:, idx]
            bbao_vecs[obsindex(pomdp, o)] = bao_vec
        end

        poba(o) = sum(pdf(b, s) * sum(pdf(transition(pomdp, s, a), sp) * pdf(observation(pomdp, s, a, sp), o) for sp in S) for s in S)

        α = zeros(Float32, length(S))
        for s in S
            idx = stateindex(pomdp, s)
            if bvec[idx] ≈ zero(bvec[idx])
                α[idx] = value(π, bvec, a_oh)[idx]
            else
                # is = abs(bvec[idx]) < sqrt(eps()) ? 0. : (bpvec[idx] / bvec[idx])
                v = r(s,a)
                if !isterminal(pomdp, s)
                    dummy = 0f0
                    for (i, o) in enumerate(O)
                        bpvec = bbao_vecs[i]
                        α_bp = Γao[i]
                        is = bpvec[idx] / bvec[idx]
                        # lb = min(is, 1f1)
                        lb = min(is, 1f10)
                        # lb = is
                        po = convert(Float32, poba(o))
                        v += γ * po * lb * α_bp[idx]
                        dummy += po
                    end
                    @assert dummy ≈ 1f0
                end
                @assert isfinite(v) idx, bvec, bpvec, lb, v, α_bp, α_bp[idx]
                α[idx] = v
            end
        end
        α .= bvec .* α + (1f0 .- bvec) .* value(π, bvec, a_oh)

        # find the optimal alpha vector
        # idx = argmax(map(αa -> αa ⋅ b.b, Γa))
        # alphavec = AlphaVec(Γa[idx], A[idx])
        return α
    end
    res = reduce(hcat, alphas)
    reshape(res, (length(S), 1, length(alphas)))
end

function alphaDQN_targetSampledImpSamp(π, 𝒫, 𝒟, γ::Float32; kwargs...)
    mdp = π.mdp
    pomdp = mdp.pomdp
    S = POMDPTools.ordered_states(pomdp)
    A = POMDPTools.ordered_actions(pomdp)
    O = POMDPTools.ordered_observations(pomdp)
    r = StateActionReward(pomdp)
    ns = length(S)
    na = length(A)

    alphas = map(zip(eachcol(𝒟[:s]), eachcol(𝒟[:a]))) do (bvec, a_oh)
        b = convert_s(DiscreteBelief, collect(bvec), mdp)

        @assert length(bvec) == length(states(pomdp))
        @assert length(a_oh) == length(actions(pomdp))

        not_terminals = [stateindex(pomdp, s) for s in S if !isterminal(pomdp, s)]
        terminals = [stateindex(pomdp, s) for s in S if isterminal(pomdp, s)]

        a = A[argmax(a_oh)]

        Γao = Vector{Vector{Float32}}(undef, length(O))
        bbao_vecs = Vector{Vector{Float32}}(undef, length(O))
        poba = zeros(Float32, length(O))
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
            # idx = argmax(reshape(bao_vec * αQ, na))
            Γao[obsindex(pomdp, o)] = αQ[:, idx]
            bbao_vecs[obsindex(pomdp, o)] = bao_vec
            poba[obsindex(pomdp, o)] = sum(pdf(b, s) * sum(pdf(transition(pomdp, s, a), sp) * pdf(observation(pomdp, s, a, sp), o) for sp in S) for s in S)
        end

        α = zeros(Float32, length(S))
        for s in S
            idx = stateindex(pomdp, s)
            if bvec[idx] ≈ zero(bvec[idx])
                α[idx] = value(π, bvec, a_oh)[idx]
            else
                v = r(s,a)
                if !isterminal(pomdp, s)
                    for _ in 1:π.m
                        o = @gen(:o)(pomdp, rand(b), a)
                        bpvec = bbao_vecs[obsindex(pomdp, o)]
                        α_bp = Γao[obsindex(pomdp, o)]
                        is = bpvec[idx] / bvec[idx]
                        lb = min(is, 10f0)
                        po = convert(Float32, 1/π.m)
                        v += γ * po * lb * α_bp[idx]
                    end
                end
                @assert isfinite(v) idx, bvec, bpvec, lb, v, α_bp, α_bp[idx]
                α[idx] = v
            end
        end

        # find the optimal alpha vector
        # idx = argmax(map(αa -> αa ⋅ b.b, Γa))
        # alphavec = AlphaVec(Γa[idx], A[idx])
        return α
    end
    res = reduce(hcat, alphas)
    reshape(res, (length(S), 1, length(alphas)))
end

function alphaDQN_modelFree(π, 𝒫, 𝒟, γ::Float32; kwargs...)
    mdp = π.mdp
    pomdp = mdp.pomdp
    S = POMDPTools.ordered_states(pomdp)
    A = POMDPTools.ordered_actions(pomdp)
    r = StateActionReward(pomdp)

    alphas = map(zip(eachcol(𝒟[:s]), eachcol(𝒟[:a]), eachcol(𝒟[:sp]))) do (bvec, a_oh, bpvec)
        @assert length(bvec) == 2
        a = A[argmax(a_oh)]
        α_bp  = best_value(π, bpvec)
        @assert length(α_bp) == length(bpvec)

        α = zeros(Float32, length(S))
        for s in S
            idx = stateindex(pomdp, s)
            is = abs(bvec[idx]) < sqrt(eps()) ? 0. : (bpvec[idx] / bvec[idx])
            v = r(s,a) + (!isterminal(pomdp, s) ? (γ * is * α_bp[idx]) : 0.)
            @assert isfinite(v) idx, bvec, bpvec, is, v, α_bp, α_bp[idx]
            α[stateindex(pomdp, s)] = v
        end
        return α
    end
    res = reduce(hcat, alphas)
    reshape(res, (length(S), 1, length(alphas)))
end

function alphaDQN_modelFreeClamped(π, 𝒫, 𝒟, γ::Float32; kwargs...)
    mdp = π.mdp
    pomdp = mdp.pomdp
    S = POMDPTools.ordered_states(pomdp)
    A = POMDPTools.ordered_actions(pomdp)
    r = StateActionReward(pomdp)

    ϵ = 0.2

    alphas = map(zip(eachcol(𝒟[:s]), eachcol(𝒟[:a]), eachcol(𝒟[:sp]))) do (bvec, a_oh, bpvec)
        @assert length(bvec) == 2
        a = A[argmax(a_oh)]
        α_bp  = best_value(π, bpvec)
        @assert length(α_bp) == length(bpvec)

        α = zeros(Float32, length(S))
        for s in S
            idx = stateindex(pomdp, s)
            # is = abs(bvec[idx]) < sqrt(eps()) ? 0. : (bpvec[idx] / bvec[idx])
            v = r(s,a)
            if !isterminal(pomdp, s)
                is = bpvec[idx] / bvec[idx]
                lb = isnan(is) ? one(is) : clamp(is, 1-ϵ, 1+ϵ)
                v += γ * lb * α_bp[idx]
            end
            @assert isfinite(v) idx, bvec, bpvec, lb, v, α_bp, α_bp[idx]
            α[stateindex(pomdp, s)] = v
        end
        return α
    end
    res = reduce(hcat, alphas)
    reshape(res, (length(S), 1, length(alphas)))
end

function alphaDQN_modelFreeClampedMin(π, 𝒫, 𝒟, γ::Float32; kwargs...)
    mdp = π.mdp
    pomdp = mdp.pomdp
    S = POMDPTools.ordered_states(pomdp)
    A = POMDPTools.ordered_actions(pomdp)
    r = StateActionReward(pomdp)

    ϵ = 0.2

    alphas = map(zip(eachcol(𝒟[:s]), eachcol(𝒟[:a]), eachcol(𝒟[:sp]))) do (bvec, a_oh, bpvec)
        @assert length(bvec) == 2
        a = A[argmax(a_oh)]
        α_bp  = best_value(π, bpvec)
        @assert length(α_bp) == length(bpvec)

        α = zeros(Float32, length(S))
        for s in S
            idx = stateindex(pomdp, s)
            # is = abs(bvec[idx]) < sqrt(eps()) ? 0. : (bpvec[idx] / bvec[idx])
            v = r(s,a)
            if !isterminal(pomdp, s)
                is = bpvec[idx] / bvec[idx]
                lb = isfinite(is) ? one(is) : min(is * α_bp[idx], clamp(is, 1-ϵ, 1+ϵ) * α_bp[idx])
                v += γ * lb
            end
            @assert isfinite(v) idx, bvec, bpvec, is, v, lb, α_bp, α_bp[idx]
            α[stateindex(pomdp, s)] = v
        end
        return α
    end
    res = reduce(hcat, alphas)
    reshape(res, (length(S), 1, length(alphas)))
end

function alphaDQN_modelFreeWeighted(π, 𝒫, 𝒟, γ::Float32; kwargs...)
    mdp = π.mdp
    pomdp = mdp.pomdp
    S = POMDPTools.ordered_states(pomdp)
    A = POMDPTools.ordered_actions(pomdp)
    r = StateActionReward(pomdp)

    alphas = map(zip(eachcol(𝒟[:s]), eachcol(𝒟[:a]), eachcol(𝒟[:sp]))) do (bvec, a_oh, bpvec)
        @assert length(bvec) == 2
        a = A[argmax(a_oh)]
        α_bp  = best_value(π, bpvec)
        @assert length(α_bp) == length(bpvec)

        α = zeros(Float32, length(S))
        for s in S
            idx = stateindex(pomdp, s)
            is = abs(bvec[idx]) < sqrt(eps()) ? 0. : (bpvec[idx] / bvec[idx])
            v = r(s,a) + (!isterminal(pomdp, s) ? (γ * is * α_bp[idx]) : 0.)
            α[stateindex(pomdp, s)] = v
        end
        α .= bvec .* α + (1f0 .- bvec) .* value(π, bvec, a_oh)
        return α
    end
    res = reduce(hcat, alphas)
    reshape(res, (length(S), 1, length(alphas)))
end

function alphaDQN_modelFreeWeightedClamped(π, 𝒫, 𝒟, γ::Float32; kwargs...)
    mdp = π.mdp
    pomdp = mdp.pomdp
    S = POMDPTools.ordered_states(pomdp)
    A = POMDPTools.ordered_actions(pomdp)
    r = StateActionReward(pomdp)

    ϵ = 0.2

    alphas = map(zip(eachcol(𝒟[:s]), eachcol(𝒟[:a]), eachcol(𝒟[:sp]))) do (bvec, a_oh, bpvec)
        @assert length(bvec) == 2
        a = A[argmax(a_oh)]
        α_bp  = best_value(π, bpvec)
        @assert length(α_bp) == length(bpvec)

        α = zeros(Float32, length(S))
        for s in S
            idx = stateindex(pomdp, s)
            # is = abs(bvec[idx]) < sqrt(eps()) ? 0. : (bpvec[idx] / bvec[idx])
            v = r(s,a)
            if !isterminal(pomdp, s)
                is = bpvec[idx] / bvec[idx]
                lb = isnan(is) ? one(is) : clamp(is, 1-ϵ, 1+ϵ)
                v += γ * lb * α_bp[idx]
            end
            @assert isfinite(v) idx, bvec, bpvec, lb, v, α_bp, α_bp[idx]
            α[stateindex(pomdp, s)] = v
        end
        α .= bvec .* α + (1f0 .- bvec) .* value(π, bvec, a_oh)
        return α
    end
    res = reduce(hcat, alphas)
    reshape(res, (length(S), 1, length(alphas)))
end

function alphaDQN_modelFreeUpperClamped(π, 𝒫, 𝒟, γ::Float32; kwargs...)
    mdp = π.mdp
    pomdp = mdp.pomdp
    S = POMDPTools.ordered_states(pomdp)
    A = POMDPTools.ordered_actions(pomdp)
    r = StateActionReward(pomdp)

    alphas = map(zip(eachcol(𝒟[:s]), eachcol(𝒟[:a]), eachcol(𝒟[:sp]))) do (bvec, a_oh, bpvec)
        @assert length(bvec) == 2
        a = A[argmax(a_oh)]
        α_bp  = best_value(π, bpvec)
        @assert length(α_bp) == length(bpvec)

        α = zeros(Float32, length(S))
        for s in S
            idx = stateindex(pomdp, s)
            if bvec[idx] ≈ zero(bvec[idx])
                α[stateindex(pomdp, s)] = value(π, bvec, a_oh)[idx]
            else
                # is = abs(bvec[idx]) < sqrt(eps()) ? 0. : (bpvec[idx] / bvec[idx])
                v = r(s,a)
                if !isterminal(pomdp, s)
                    is = bpvec[idx] / bvec[idx]
                    lb = min(is, 10f0)
                    v += γ * lb * α_bp[idx]
                end
                @assert isfinite(v) idx, bvec, bpvec, lb, v, α_bp, α_bp[idx]
                α[stateindex(pomdp, s)] = v
            end
        end
        return α
    end
    res = reduce(hcat, alphas)
    reshape(res, (length(S), 1, length(alphas)))
end
