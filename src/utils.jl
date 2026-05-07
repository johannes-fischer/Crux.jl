# Categorical distribution that works with arbitrary objects (Rather than consecutive ints)
struct ObjectCategorical <: DiscreteUnivariateDistribution
    objs
    p
    cat
    ObjectCategorical(objs::T, p = fill(1/length(objs), length(objs))) where {T <: AbstractArray} = new(objs, p, Categorical(p))
    ObjectCategorical(objs::T) where {T <: Tuple} = ObjectCategorical([objs...])
end
Base.length(d::ObjectCategorical) = length(d.cat)

Base.rand(d::ObjectCategorical, sz::Int...) = d.objs[rand(d.cat, sz...)]
function Distributions.logpdf(d::ObjectCategorical, x::AbstractArray)
    x = mapslices(findfirst, d.objs .== reshape(x, 1,:), dims=1)
    return logpdf.(d.cat, x)
end

function Distributions.logpdf(d::ObjectCategorical, x)
    x = findfirst(d.objs .== x)
    return logpdf(d.cat, x)
end
Distributions.support(d::ObjectCategorical) = d.objs

Distributions.entropy(d::ObjectCategorical) = entropy(d.cat)

function Distributions.fit(dt::Type{D}, x, w; objs) where D <: ObjectCategorical
	cat = fit(Categorical, Flux.onecold(x), w)
	ObjectCategorical(objs, cat.p)
end

# Constant Layer
struct ConstantLayer{T}
    vec::T
end

Flux.@layer ConstantLayer
(m::ConstantLayer)(x::AbstractArray) = m.vec


# Flux 0.16 port — wrapper for trainable scalars (e.g. SAC log-α, CQL log-α).
# Old code keyed `param_optimizers::Dict{Flux.Params,TrainingParams}` on a
# Flux.Params view of a 1-element array; Flux.Params is gone in 0.16, and we
# also need a stable, identity-based dict key for the model that train! is
# differentiating. A mutable struct with `Flux.@layer` gives both: Functors
# walks `v` for gradients and Optimisers state, while the wrapper itself
# hashes by identity so the dict lookup survives in-place updates of `v`.
mutable struct LearnableScalar
    v::Vector{Float32}
end

LearnableScalar(x::Real) = LearnableScalar(Float32[Float32(x)])

Flux.@layer LearnableScalar trainable=(v,)

Base.getindex(s::LearnableScalar, i) = s.v[i]
Base.length(::LearnableScalar) = 1



## Useful functions
whiten(v) = whiten(v, mean(v), std(v))
whiten(v, μ, σ) = (v .- μ) ./ σ

to2D(W) = reshape(W, :, size(W, ndims(W))) # convert a multidimensional weight matrix to 2D

# Weighted mean aggregator
weighted_mean(weights) = (y) -> mean(y .* weights)
    
# --- Flux 0.16 port: gradient L2/Lp norm walking a NamedTuple grad tree ---
# Replaces the old `norm(::Zygote.Grads)` overload, which only existed for the
# implicit-params API. The new `Flux.gradient(model)` returns a tree shaped
# like `model`, so we walk it with `Functors.fmap` and aggregate leaves.
function global_grad_norm(grads; p::Real = 2)
    s = 0.0
    Flux.fmap(grads) do x
        if x isa AbstractArray{<:AbstractFloat}
            s += sum(abs.(x) .^ p)
        end
        x
    end
    return s^(1/p)
end

## Early stopping
# Early stopping function that terminates training on validation error increase
function stop_on_validation_increase(π, 𝒫, 𝒟_val, loss; window=5)
    k = "validation_error"
    (infos) -> begin
        ve = loss(π, 𝒫, 𝒟_val) # Compute the validation error
        infos[end][k] = ve # store it
        N = length(infos)
        if length(infos) >= 2*window
            curr_window = mean([infos[i][k] for i=N-window+1:N])
            old_window = mean([infos[i][k] for i=N-2*window+1:N-window])
            return curr_window >= old_window # check if the error has gone up
        end
        false
    end
end


## Losses
# Flux 0.16 port: all losses now take the differentiated model `m` as the first
# positional argument (Zygote differentiates with respect to it). Off-policy
# losses additionally accept `π_loss=m` — the full agent policy passed by
# off_policy.jl when a loss needs both the actor and critic in scope (e.g.
# DDPG/TD3 actor losses). Losses that don't need it ignore the kwarg.
function td_loss(;loss=Flux.mse, name=:Qavg, s_key=:s, a_key=:a, weight=nothing)
    (m, 𝒫, 𝒟, y; info=Dict(), π_loss=m) -> begin
        Q = value(m, 𝒟[s_key], 𝒟[a_key])

        # Store useful information
        ignore_derivatives() do
            info[name] = mean(Q)
        end

        loss(Q, y, agg = isnothing(weight) ? mean : weighted_mean(𝒟[weight]))
    end
end

function double_Q_loss(;name1=:Q1avg, name2=:Q2avg, kwargs...)
    l1 = td_loss(;name=name1, kwargs...)
    l2 = td_loss(;name=name2, kwargs...)

    # `m` here is the (Double-)critic itself — we differentiate through both
    # heads. The previous version reached into `π.C.N1` because π was the full
    # ActorCritic; under the new API critic(π) → π.C is what's passed to train!.
    (m, 𝒫, 𝒟, y; info=Dict(), π_loss=m) -> begin
        .5f0*(l1(m.N1, 𝒫, 𝒟, y, info=info) + l2(m.N2, 𝒫, 𝒟, y, info=info))
    end
end

function multi_td_loss(;names, indices=1:length(names), kwargs...)
    ls = [td_loss(;name=name, kwargs...) for name in names]

    (m, 𝒫, 𝒟, ys; info=Dict(), π_loss=m) -> begin
        mean([loss(net, 𝒫, 𝒟, y, info=info) for (loss, net, y) in zip(ls, m.networks[indices], ys)])
    end
end

function multi_actor_loss(actor_lf, N; indices=1:N, kwargs...)
    (m, 𝒫, 𝒟; info=Dict(), π_loss=m) -> begin
        mean([actor_lf(net, 𝒫, 𝒟, info=info) for net in m.networks[indices]])
    end
end

td_error(π, 𝒫, 𝒟, y) = abs.(value(π, 𝒟[:s], 𝒟[:a])  .- y)


## Scheduling
struct LinearDecaySchedule{R<:Real} <: Function
    start::R
    stop::R
    steps::Int
end

function (schedule::LinearDecaySchedule)(i)
    rate = (schedule.start - schedule.stop) / schedule.steps
    val = schedule.start - i*rate 
    val = max(schedule.stop, val)
end

function MultitaskDecaySchedule(steps::Int, task_ids; start = 1.0, stop = 0.1)
    schedule = LinearDecaySchedule(start, stop, steps)
    function val(i)
        taskindex = ceil(Int, i / steps)
        taskindex < 1 && return start
        taskindex > length(task_ids) && return stop
        taskid = task_ids[taskindex]
        used_steps = steps*sum(task_ids[1:taskindex-1] .== taskid)
        schedule(used_steps + mod1(i, steps))
    end
end

function logcompσ(x::Real)
    # Computes log(1 - sigmoid(x)) in a numerically stable way
    return logσ(x) - x
end


# credit to: https://mileslucas.com/posts/weighted-logsumexp/
function weighted_logsumexp(X, w)
    a = -Inf
    r = zero(eltype(X))
    for (x, wi) in zip(X, w)
        if x ≤ a
            # standard computation
            r += wi * exp(x - a)
        else
            # if new value is higher than current max
            r *= exp(a - x)
            r += wi
            a = x
        end
    end
    return a + log(r)
end

