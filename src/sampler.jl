@with_kw mutable struct Sampler{P, Pol <: Policy, T1 <: AbstractSpace, T2 <: AbstractSpace}
    mdp::P
    agent::PolicyParams{Pol, T2} # The agent
    adversary = nothing # The adversary
    s = rand(initialstate(mdp)) # Current State
    S::T1 = state_space(initial_observation(mdp, s)) # State space
    svec::AbstractArray = tovec(initial_observation(mdp, s), S) # Current observation
    max_steps::Int = 100
    required_columns::Array{Symbol} = []
    γ::Float32 = discount(mdp)
    λ::Float32 = NaN32
    episode_length::Int64 = 0
    episode_checker::Function = (data, start, stop) -> true
    was_reset = false # Used to make sure that there isn't more than 1 reset per trajectory

    # Parameters for cost constraints
    Vc::Union{ContinuousNetwork, Nothing} = nothing
    λcost::Float32 = NaN32

    # Trajectory-level measurements
    traj_weight_fn=nothing # weight of the trajectory
end

Sampler(mdp, π::T; kwargs...) where {T <: Policy} = Sampler(;mdp=mdp, agent=PolicyParams(π), kwargs...)
Sampler(mdp, agent::T; kwargs...) where {T <: PolicyParams} = Sampler(;mdp=mdp, agent=agent, kwargs...)

# Construct a vector of samplers from a vector of mdps
Sampler(mdps::AbstractVector, π::T; kwargs...) where {T <: Policy} = [Sampler(mdps[i], π; kwargs...) for i in 1:length(mdps)]
Sampler(mdps::AbstractVector, agent::T; kwargs...) where {T <: PolicyParams} = [Sampler(mdps[i], agent; kwargs...) for i in 1:length(mdps)]

function reset_sampler!(sampler::Sampler)
    sampler.was_reset && return
    if sampler.agent.π isa LatentConditionedNetwork && hasproperty(sampler.mdp, :z)
        sampler.agent.π.z = sampler.mdp.z
    end

    new_ep_reset!(sampler.agent.π)

    sampler.s = rand(initialstate(sampler.mdp))
    sampler.svec = tovec(initial_observation(sampler.mdp, sampler.s), sampler.S)
    sampler.episode_length = 0
    sampler.was_reset=true
end

function initial_observation(mdp, s)
    if mdp isa POMDP
        return convert_o(AbstractArray, rand(initialobs(mdp, s)), mdp)
    else
        return convert_s(AbstractArray, s, mdp)
    end
end

function terminate_episode!(sampler::Sampler, data, j)
    data[:episode_end][1,j] = true
    ep = j - sampler.episode_length + 1 : j
    haskey(data, :advantage) && fill_gae!(data, ep, sampler.agent.π, sampler.λ, sampler.γ)
    haskey(data, :return) && fill_returns!(data, ep, sampler.γ)
    haskey(data, :fwd_importance_weight) && fill_fwd_importance_weight!(data, ep)
    haskey(data, :cum_importance_weight) && fill_cum_importance_weight!(data, ep)
    haskey(data, :rev_importance_weight) && fill_rev_importance_weight!(data, ep)

    haskey(data, :traj_importance_weight) && (data[:traj_importance_weight][1,ep] .= sampler.traj_weight_fn(sampler.agent, data, ep))

    # Dealing with cost constraints
    haskey(data, :cost_advantage) && fill_gae!(data, ep, sampler.Vc, sampler.λ, sampler.γ, source=:cost, target=:cost_advantage)
    haskey(data, :cost_return) && fill_returns!(data, ep, sampler.γ, source=:cost, target=:cost_return)

    reset_sampler!(sampler)
end

function step!(data, j::Int, sampler::Sampler; explore=false, i=0)
    sampler.was_reset=false
    a, logprob = explore ? exploration(sampler.agent.π_explore, sampler.svec, π_on=sampler.agent.π, i=i) : (action(sampler.agent.π, sampler.svec), NaN)
    (a isa AbstractArray || a isa Tuple) && length(a) == 1 && (a = a[1])

    # This implements the ability to get cost information from safety gym
    info = Dict()
    kwargs = (haskey(data, :cost) || haskey(data, :z) || haskey(data, :grasp_success)) ? (info=info,) : ()

    args = (a,)
    if !isnothing(sampler.adversary)
        x, xlogprob = explore ? exploration(sampler.adversary.π_explore, sampler.svec, π_on=sampler.adversary.π, i=i) : (action(sampler.adversary.π, sampler.svec), NaN)
        (x isa AbstractArray || x isa Tuple) && length(x) == 1 && (x = x[1]) # disturbances always come out as an array
        data[:x][:, j:j] .= tovec(x, sampler.adversary.space)
        haskey(data, :xlogprob) && (data[:xlogprob][:, j] .= xlogprob)
        args = (a, x)
    end

    if sampler.mdp isa ExplicitBeliefMDP && haskey(data, :o)
        s_pomdp = rand(sampler.s)
        sp_pomdp, o, r = @gen(:sp, :o, :r)(sampler.mdp.pomdp, s_pomdp, args...; kwargs...)
        sp = POMDPs.update(sampler.mdp.updater, sampler.s, a, o)
        spvec = convert_s(AbstractArray, sp, sampler.mdp)

        ovec = convert_o(AbstractArray, o, sampler.mdp.pomdp)
        if size(data[:o], 1) == 0
            data[:o] = fill(ovec[1], size(ovec)..., size(data[:s], ndims(data[:s])))
        end
        bslice(data[:o], j:j) .= ovec

        if haskey(data, :s_pomdp)
            s_pomdp_vec = convert_s(AbstractArray, s_pomdp, sampler.mdp.pomdp)
            if size(data[:s_pomdp], 1) == 0
                data[:s_pomdp] = fill(s_pomdp_vec[1], size(s_pomdp_vec)..., size(data[:s], ndims(data[:s])))
            end
            bslice(data[:s_pomdp], j:j) .= s_pomdp_vec
        end
        if haskey(data, :sp_pomdp)
            sp_pomdp_vec = convert_s(AbstractArray, sp_pomdp, sampler.mdp.pomdp)
            if size(data[:sp_pomdp], 1) == 0
                data[:sp_pomdp] = fill(sp_pomdp_vec[1], size(sp_pomdp_vec)..., size(data[:s], ndims(data[:s])))
            end
            bslice(data[:sp_pomdp], j:j) .= sp_pomdp_vec
        end
    elseif sampler.mdp isa POMDP
        sp, o, r = @gen(:sp,:o,:r)(sampler.mdp, sampler.s, args...; kwargs...)
        spvec = convert_o(AbstractArray, o, sampler.mdp)
    else
        sp, r = @gen(:sp,:r)(sampler.mdp, sampler.s, args...; kwargs...)
        spvec = convert_s(AbstractArray, sp, sampler.mdp)
    end
    spvec = tovec(spvec, sampler.S)
    done = isterminal(sampler.mdp, sp)

    # Save the tuple
    bslice(data[:s], j:j) .= sampler.svec
    data[:a][:, j:j] .= tovec(a, sampler.agent.space)
    bslice(data[:sp], j:j) .= spvec
    data[:r][1, j] = r
    data[:done][1, j] = done

    # Handle optional data storage
    haskey(data, :logprob) && (data[:logprob][:, j] .= logprob)
    if haskey(data, :importance_weight)
        nom_logprob = logpdf(sampler.agent.pa, sampler.svec, tovec(a, sampler.agent.space))
        data[:importance_weight][:, j] .= exp.(nom_logprob .- logprob)
    end
    haskey(data, :t) && (data[:t][1, j] = sampler.episode_length + 1)
    haskey(data, :i) && (data[:i][1, j] = i+1)
    haskey(data, :cost) && (data[:cost][1,j] = info["cost"])
    haskey(data, :grasp_success) && (data[:grasp_success][1,j] = info["grasp_success"])
    if haskey(data, :z) && haskey(info, "z")
        z = info["z"]
        if sampler.agent.π isa LatentConditionedNetwork
            sampler.agent.π.z = z
        end

        if size(data[:z], 1) == 0
            data[:z] = fill(z[1], length(z), size(data[:s], ndims(data[:s])))
        end
        data[:z][:, j] = z
    end
    haskey(data, :fail) && (data[:fail][1,j] = extra_functions["isfailure"](sampler.mdp, sp)) #TODO Changed this to "s" instead of "sp" for the continuum world

    # Cut the episode short if needed
    sampler.episode_length += 1
    if done || sampler.episode_length >= sampler.max_steps
        terminate_episode!(sampler, data, j)
    else
        sampler.s = sp
        sampler.svec = spvec
    end
end

function steps!(sampler::Sampler, buffer=nothing; store=nothing, cb=(kwargs...)->nothing, Nsteps=1, explore=false, i=0, reset=false, return_episodes=false, return_at_episode_end=false)
    data = mdp_data(sampler.S, sampler.agent.space, Nsteps, sampler.required_columns)
    for j=1:Nsteps
        step!(data, j, sampler, explore=explore, i=i + (j-1))
        if return_at_episode_end && sampler.episode_length == 0
            trim!(data, j)
            break
        end
    end
    reset && terminate_episode!(sampler, data, Nsteps)

    cb(data) # Run the callback on the dataset before adding it
    !isnothing(store) && push!(store, data) # add it to the storage array if provided
    !isnothing(buffer) && push!(buffer, data) # Push it to the provided buffer

    return_episodes ? (data, episodes(data)) : data
end

function steps!(samplers::Vector{T}, buffer=nothing; store=nothing, cb=(kwargs...)->nothing, Nsteps=1, explore=false, i=0, reset=false, return_episodes = false) where {T<:Sampler}
    data = mdp_data(samplers[1].S, samplers[1].agent.space, Nsteps*length(samplers), samplers[1].required_columns)
    j = 1
    for s=1:Nsteps
        for sampler in samplers
            step!(data, j, sampler, explore = explore, i = i + (j-1))
            j += 1
        end
    end
    reset && terminate_episode!(sampler, data, Nsteps)

    cb(data) # Run the callback on the dataset before adding it
    !isnothing(store) && push!(store, data) # add it to the storage array if provided
    !isnothing(buffer) && push!(buffer, data) # Push it to the provided buffer

    return_episodes ? (data, episodes(data)) : data
end

function episodes!(sampler::Sampler, buffer=nothing; store=nothing, cb=(kwargs...)->nothing, Neps=1, explore=false, i=0, return_episodes=false)
    reset_sampler!(sampler)
    data = mdp_data(sampler.S, sampler.agent.space, Neps*sampler.max_steps, sampler.required_columns)
    episode_starts, episode_ends = zeros(Int, Neps), zeros(Int, Neps)

    j, k = 0, 1
    while k <= Neps
        episode_starts[k] = j+1
        while true
            j = j+1
            step!(data, j, sampler, explore=explore, i=i + (i-1))
            if sampler.episode_length == 0
                episode_ends[k] = j
                sampler.episode_checker(data, episode_starts[k], j) ? (k = k+1) : (j = episode_starts[k]-1)
                break
            end
        end
    end
    trim!(data, j)

    cb(data) # Run the callback on the dataset before adding it
    !isnothing(store) && push!(store, data) # add it to the storage array if provided
    !isnothing(buffer) && push!(buffer, data) # Push it to the provided buffer

    return_episodes ? (data, zip(episode_starts, episode_ends)) : data
end


## metric

# Recover multiple metrics from a single sampler
metrics_by_key(data, start, stop; keys) = [sum(data[key][1,start:stop]) for key in keys]

function metrics_by_key(s::Sampler; keys, Neps=100, kwargs...)
    if hasproperty(s.mdp, :logging)
        s.mdp.logging = true
    end
    data = episodes!(s, Neps=Neps; kwargs...)

    if hasproperty(s.mdp, :logging)
        s.mdp.logging = false
    end

    [sum(data[key]) / Neps for key in keys]
end

# recover a single metric
metric_by_key(data, start, stop; key) = metrics_by_key(data, start, stop; keys=[key])[1]

metric_by_key(s::Sampler; key, Neps=100, kwargs...) = metrics_by_key(s; keys=[key], Neps=Neps, kwargs...)[1]

# Get the undiscounted return
undiscounted_return(data, start, stop) = metric_by_key(data, start, stop; key=:r)
undiscounted_return(s::Sampler; Neps=100, kwargs...) = metric_by_key(s; Neps=Neps, key=:r, kwargs...)


## Discounted returns
function discounted_return(data, start, stop, γ)
    r = 0f0
    for i in reverse(start:stop)
        r = data[:r][1, i] + γ*r
    end
    r
end

function discounted_return(s::Sampler; Neps=100, kwargs...)
    data, episodes = episodes!(s, Neps=Neps, return_episodes=true; kwargs...)
    mean([discounted_return(data, e..., discount(s.mdp)) for e in episodes])
end

## Failures
failure(data, start, stop; threshold = 0.) = undiscounted_return(data, start, stop) < threshold

function failure(s::Sampler; threshold=0., Neps=100, kwargs...)
    data, episodes = episodes!(s, Neps = Neps, return_episodes = true; kwargs...)
    mean([failure(data, e..., threshold = threshold) for e in episodes])
end


## Generalized Advantage Estimation
function fill_gae!(d::ExperienceBuffer, V, λ::Float32, γ::Float32)
    eps = episodes(d)
    for ep in eps
        fill_gae!(d, ep, V, λ, γ)
    end
end

function fill_gae!(d, episode_range, V, λ::Float32, γ::Float32; source = :r, target = :advantage)
    A, c = 0f0, λ*γ
    nd = ndims(d[:s])
    for i in reverse(episode_range)
        Vsp = value(V, bslice(d[:sp], i:i))
        Vs = value(V, bslice(d[:s], i:i))
        @assert length(Vs) == 1
        A = c*A + d[source][1,i] + (1.f0 - d[:done][1,i])*γ*Vsp[1] - Vs[1]
        @assert !isnan(A)
        d[target][:, i] .= A
    end
end

function fill_returns!(data, episode_range, γ::Float32; source=:r, target=:return)
    r = 0f0
    for i in reverse(episode_range)
        r = data[source][1, i] + γ*r
        data[target][:, i] .= r
    end
end

function fill_fwd_importance_weight!(data, episode_range;)
    @assert haskey(data, :importance_weight)
    w = 1f0
    for i in episode_range
        w = data[:importance_weight][1, i] * w
        data[:fwd_importance_weight][:, i] .= w
    end
end

function fill_cum_importance_weight!(data, episode_range;)
    @assert haskey(data, :importance_weight)
    w = 1f0
    for i in episode_range
        w = data[:importance_weight][1, i] * w
    end
    data[:cum_importance_weight][:, episode_range] .= w
end

function fill_rev_importance_weight!(data, episode_range;)
    @assert haskey(data, :importance_weight)
    w=1f0
    for i in reverse(episode_range)
        w = data[:importance_weight][1, i] * w
        data[:rev_importance_weight][:, i] .= w
    end
end

# Utils
function trim!(data::Dict{Symbol, Array}, N)
    for k in keys(data)
        data[k] = bslice(data[k], 1:N)
    end
    data
end
