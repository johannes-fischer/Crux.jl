
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

# TODO:
# - weighed clampled min
# - Use other beliefs in update (dont use value(bbao) but argmax alpha * value(b') for multiple b')
#

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
