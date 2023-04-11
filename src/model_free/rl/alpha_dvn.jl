
function alphaDVN_target_single_bao(π, 𝒫, 𝒟, γ::Float32; kwargs...)
    mdp = π.mdp
    pomdp = mdp.pomdp
    S = POMDPTools.ordered_states(pomdp)
    A = POMDPTools.ordered_actions(pomdp)
    O = POMDPTools.ordered_observations(pomdp)
    r = StateActionReward(pomdp)

    alphas = map(eachcol(𝒟[:s])) do s
        @assert length(s) == 2
        b = convert_s(DiscreteBelief, collect(s), mdp)

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
    pomdp = mdp.pomdp
    S = POMDPTools.ordered_states(pomdp)
    A = POMDPTools.ordered_actions(pomdp)
    O = POMDPTools.ordered_observations(pomdp)
    r = StateActionReward(pomdp)

    alphas = map(eachcol(𝒟[:s])) do s
        @assert length(s) == 2
        b = convert_s(DiscreteBelief, collect(s), mdp)

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

function alphaDVN_sampleTarget(π, 𝒫, 𝒟, γ::Float32; kwargs...)
    mdp = π.mdp
    pomdp = mdp.pomdp
    S = POMDPTools.ordered_states(pomdp)
    A = POMDPTools.ordered_actions(pomdp)
    O = POMDPTools.ordered_observations(pomdp)
    r = StateActionReward(pomdp)
    ns = length(S)

    alphas = map(eachcol(𝒟[:s])) do s
        @assert length(s) == 2
        B = statetype(mdp)
        b = convert_s(B, collect(s), mdp)

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
                    b′ = DiscreteBelief(pomdp, b.state_list, zeros(ns))
                end

                # extract optimal alpha vector at resulting belief
                # Γao[obsindex(pomdp, o)] = _argmax(α -> dot(α,b′.b), Γ)
                bao_vec = convert_s(Vector, b′, mdp)
                Γao[obsindex(pomdp, o)] = value(π, bao_vec)
            end

            # construct new alpha vectors
            Γa[actionindex(pomdp, a)] = zeros(length(S))
            for s in S
                for _ in 1:π.m
                    sp, o, r = @gen(:sp,:o,:r)(pomdp, s, a)
                    αo = Γao[obsindex(pomdp, o)]
                    v = r + (!isterminal(pomdp, s) ? (γ * αo[stateindex(pomdp, sp)]) : 0.)
                    Γa[actionindex(pomdp, a)][stateindex(pomdp, s)] += v / π.m
                end
            end
        end

        # find the optimal alpha vector
        idx = argmax(map(αa -> αa ⋅ b.b, Γa))
        alphavec = AlphaVec(Γa[idx], A[idx])
        return alphavec.alpha
    end
    reduce(hcat, alphas)
end

function alphaDVN_weightedSampleTarget(π, 𝒫, 𝒟, γ::Float32; kwargs...)
    mdp = π.mdp
    pomdp = mdp.pomdp
    S = POMDPTools.ordered_states(pomdp)
    A = POMDPTools.ordered_actions(pomdp)
    O = POMDPTools.ordered_observations(pomdp)
    r = StateActionReward(pomdp)
    ns = length(S)

    alphas = map(zip(eachcol(𝒟[:s]), eachcol(𝒟[:o]))) do (s, ovec)
        @assert length(s) == 2
        b = convert_s(statetype(mdp), collect(s), mdp)
        o = convert_o(obstype(pomdp), collect(ovec), mdp.pomdp)

        Γa = Vector{Vector{Float32}}(undef, length(A))

        not_terminals = [stateindex(pomdp, s) for s in S if !isterminal(pomdp, s)]
        terminals = [stateindex(pomdp, s) for s in S if isterminal(pomdp, s)]
        for a in A
            # Γao = Vector{Vector{Float32}}(undef, length(O))
            trans_probs = dropdims(sum([pdf(transition(pomdp, S[is], a), sp) * b.b[is] for sp in S, is in not_terminals], dims=2), dims=2)
            if !isempty(terminals) trans_probs[terminals] .+= b.b[terminals] end

            # update beliefs
            obs_probs = pdf.(map(sp -> observation(pomdp, a, sp), S), [o])
            b′ = obs_probs .* trans_probs

            if sum(b′) > 0.
                b′ = DiscreteBelief(pomdp, b.state_list, belief_norm(pomdp, b.b, b′, terminals, not_terminals))
            else
                b′ = DiscreteBelief(pomdp, b.state_list, zeros(ns))
            end

            # extract optimal alpha vector at resulting belief
            # Γao[obsindex(pomdp, o)] = _argmax(α -> dot(α,b′.b), Γ)
            bao_vec = convert_s(Vector, b′, mdp)
            αo = value(π, bao_vec)

            # construct new alpha vectors
            Γa[actionindex(pomdp, a)] = zeros(length(S))
            for s in S
                if !isterminal(pomdp, s)
                    w_sum = 0.0
                    for _ in 1:π.m
                        sp = @gen(:sp)(pomdp, s, a)
                        w = obs_weight(pomdp, s, a, sp, o)
                        v = w * αo[stateindex(pomdp, sp)]
                        w_sum += w
                        Γa[actionindex(pomdp, a)][stateindex(pomdp, s)] += v
                    end
                    Γa[actionindex(pomdp, a)][stateindex(pomdp, s)] *= γ / w_sum
                end
                Γa[actionindex(pomdp, a)][stateindex(pomdp, s)] += r(s, a)
            end
        end

        # find the optimal alpha vector
        idx = argmax(map(αa -> αa ⋅ b.b, Γa))
        alphavec = AlphaVec(Γa[idx], A[idx])
        return alphavec.alpha
    end
    reduce(hcat, alphas)
end

function alphaDVN_modelFreeStateBackups(π, 𝒫, 𝒟, γ::Float32; kwargs...)
    mdp = π.mdp
    pomdp = mdp.pomdp
    S = POMDPTools.ordered_states(pomdp)
    A = POMDPTools.ordered_actions(pomdp)
    r = StateActionReward(pomdp)

    alphas = map(zip(eachcol(𝒟[:s]), eachcol(𝒟[:o]))) do (svec, ovec)
        @assert length(svec) == 2
        B = statetype(mdp)
        b = convert_s(B, collect(svec), mdp)
        o = convert_o(obstype(pomdp), collect(ovec), mdp.pomdp)

        Γa = Vector{Vector{Float32}}(undef, length(A))
        for a in A
            # construct new alpha vectors
            α = zeros(length(S))
            for s in S
                s_base = zeros(length(S))
                s_base[stateindex(pomdp, s)] = 1.
                bs = convert_s(B, s_base, mdp)
                bp = POMDPs.update(mdp.updater, bs, a, o)
                αs = value(π, convert_s(Vector, bp, mdp))
                bpvec = convert_s(Vector, bp, mdp)
                v = r(s, a) + (!isterminal(pomdp, s) ? (γ * dot(bpvec, αs)) : 0.)
                α[stateindex(pomdp, s)] = v
            end
            Γa[actionindex(pomdp, a)] = α
        end

        # find the optimal alpha vector
        idx = argmax(map(αa -> αa ⋅ b.b, Γa))
        alphavec = AlphaVec(Γa[idx], A[idx])
        return alphavec.alpha
    end
    reduce(hcat, alphas)
end

function alphaDVN_modelFreeWeightedStateBackups(π, 𝒫, 𝒟, γ::Float32; kwargs...)
    mdp = π.mdp
    pomdp = mdp.pomdp
    S = POMDPTools.ordered_states(pomdp)
    A = POMDPTools.ordered_actions(pomdp)
    r = StateActionReward(pomdp)

    alphas = map(zip(eachcol(𝒟[:s]), eachcol(𝒟[:o]))) do (svec, ovec)
        @assert length(svec) == 2
        B = statetype(mdp)
        b = convert_s(B, collect(svec), mdp)
        o = convert_o(obstype(pomdp), collect(ovec), mdp.pomdp)

        Γa = Vector{Vector{Float32}}(undef, length(A))
        for a in A
            # construct new alpha vectors
            α = zeros(length(S))
            for s in S
                s_base = zeros(length(S))
                s_base[stateindex(pomdp, s)] = 1.
                bs = convert_s(B, s_base, mdp)
                bp = POMDPs.update(mdp.updater, bs, a, o)
                αs = value(π, convert_s(Vector, bp, mdp))
                bpvec = convert_s(Vector, bp, mdp)
                v = r(s, a) + (!isterminal(pomdp, s) ? (γ * dot(bpvec, αs)) : 0.)
                α[stateindex(pomdp, s)] = v
            end
            α = svec .* α + (1 .- svec) .* value(π, svec)
            Γa[actionindex(pomdp, a)] = α
        end

        # find the optimal alpha vector
        idx = argmax(map(αa -> αa ⋅ b.b, Γa))
        alphavec = AlphaVec(Γa[idx], A[idx])
        return alphavec.alpha
    end
    reduce(hcat, alphas)
end

function alphaDVN_modelFreeBeliefBackup(π, 𝒫, 𝒟, γ::Float32; kwargs...)
    mdp = π.mdp
    pomdp = mdp.pomdp
    S = POMDPTools.ordered_states(pomdp)
    A = POMDPTools.ordered_actions(pomdp)
    r = StateActionReward(pomdp)

    alphas = map(eachcol(𝒟[:s])) do bvec
        @assert length(bvec) == 2 string(bvec)
        B = statetype(mdp)
        b = convert_s(B, collect(bvec), mdp)

        Γa = Vector{Vector{Float32}}(undef, length(A))
        for a in A
            # construct new alpha vectors
            αa = zeros(length(S))

            o = @gen(:o)(pomdp, rand(b), a)
            bp = POMDPs.update(mdp.updater, b, a, o)
            bpvec = convert_s(Vector, bp, mdp)
            α_bp  = value(π, bpvec)
            for s in S
                idx = stateindex(pomdp, s)
                is = abs(bvec[idx]) < sqrt(eps()) ? 0. : (bpvec[idx] / bvec[idx])
                v = r(s,a) + (!isterminal(pomdp, s) ? (γ * is * α_bp[idx]) : 0.)
                αa[stateindex(pomdp, s)] = v
            end
            Γa[actionindex(pomdp, a)] = αa
        end

        # find the optimal alpha vector
        idx = argmax(map(αa -> αa ⋅ b.b, Γa))
        alphavec = AlphaVec(Γa[idx], A[idx])
        return alphavec.alpha
    end
    reduce(hcat, alphas)
end

function alphaDVN_modelFreeWeightedBeliefBackup(π, 𝒫, 𝒟, γ::Float32; kwargs...)
    mdp = π.mdp
    pomdp = mdp.pomdp
    S = POMDPTools.ordered_states(pomdp)
    A = POMDPTools.ordered_actions(pomdp)
    r = StateActionReward(pomdp)

    alphas = map(eachcol(𝒟[:s])) do bvec
        @assert length(bvec) == 2 string(bvec)
        B = statetype(mdp)
        b = convert_s(B, collect(bvec), mdp)

        Γa = Vector{Vector{Float32}}(undef, length(A))
        for a in A
            # construct new alpha vectors
            αa = zeros(length(S))

            o = @gen(:o)(pomdp, rand(b), a)
            bp = POMDPs.update(mdp.updater, b, a, o)
            bpvec = convert_s(Vector, bp, mdp)
            α_bp  = value(π, bpvec)
            for s in S
                idx = stateindex(pomdp, s)
                is = abs(bvec[idx]) < sqrt(eps()) ? 0. : (bpvec[idx] / bvec[idx])
                v = r(s,a) + (!isterminal(pomdp, s) ? (γ * is * α_bp[idx]) : 0.)
                αa[stateindex(pomdp, s)] = v
            end
            αa = bvec .* αa + (1 .- bvec) .* value(π, bvec)

            Γa[actionindex(pomdp, a)] = αa
        end

        # find the optimal alpha vector
        idx = argmax(map(αa -> αa ⋅ b.b, Γa))
        alphavec = AlphaVec(Γa[idx], A[idx])
        return alphavec.alpha
    end
    reduce(hcat, alphas)
end
