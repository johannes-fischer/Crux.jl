@with_kw mutable struct TrainingParams
    loss
    optimizer = Adam(3f-4)
    optimizer_state = nothing       # populated lazily on first train! call (Flux 0.16 port)
    regularizer = (π) -> 0
    batch_size = 128
    epochs = 80
    update_every = 1
    early_stopping = (info) -> false
    name = ""
    max_batches = Inf
end

# --- Flux 0.16 port -----------------------------------------------------------
# Old API (Flux ≤ 0.14):
#     l, back = Flux.pullback(() -> loss(info=info), Flux.params(π))
#     grad   = back(1f0)                       # Zygote.Grads (dict-like)
#     Flux.update!(p.optimizer, params, grad)  # implicit-params update
#
# New API (Flux 0.15+):
#     l, grads = Flux.withgradient(m -> loss(m; info=info), model)
#     Flux.update!(opt_state, model, grads[1])  # explicit-tree update
#
# `opt_state` carries persistent state (Adam momentum, …) and must outlive a
# single call to train!, so we cache it on the TrainingParams. Loss functions
# now take the model `m` as their first positional argument; Zygote
# differentiates with respect to that argument.

# Per-leaf non-finite localization for a Zygote gradient tree (or any nested
# NamedTuple / Tuple / Array structure). Prints one line per leaf that is
# NaN/Inf, with its path + shape + counts, so a non-finite grad-norm can be
# traced to the specific parameter whose gradient went bad. Distilled from
# DroneDatasetAnalysis' nan_utils.jl — we only need localization here, not its
# file-dump / model-walk machinery.
function summarize_nonfinite_grads(io::IO, obj::Union{NamedTuple, Tuple}, path::String = "")
    for (key, val) in (obj isa Tuple ? enumerate(obj) : pairs(obj))
        sep = isempty(path) ? "" : "."
        summarize_nonfinite_grads(io, val, "$path$sep$key")
    end
end
function summarize_nonfinite_grads(io::IO, x::AbstractArray, path::String)
    n_nan = count(isnan, x)
    n_inf = count(v -> !isnan(v) && !isfinite(v), x)
    (n_nan > 0 || n_inf > 0) &&
        println(io, "  [BAD] $path  shape=$(size(x))  nan=$n_nan inf=$n_inf / $(length(x))")
end
summarize_nonfinite_grads(io::IO, x::Real, path::String) =
    (isnan(x) || !isfinite(x)) && println(io, "  [BAD] $path  value=$x")
summarize_nonfinite_grads(::IO, ::Any, ::String) = nothing

# Forward non-finite localizer: given `name => value` pairs, throw naming every
# one that contains NaN/Inf — so a loss NaN is attributed to its SOURCE (e.g. an
# Inf advantage from a diverged critic, or a blown-up importance ratio) BEFORE it
# propagates into the gradient. Cheap `any(!isfinite, ·)` per tensor; detailed
# counts only for offenders. List raw inputs first and derived quantities last so
# the first-named entry is the root cause. Complements `summarize_nonfinite_grads`
# (the backward-pass case: all inputs finite, but a gradient is not).
function check_finite_inputs(where::AbstractString, pairs::Pair...)
    bad = String[]
    for (name, x) in pairs
        x === nothing && continue
        if x isa Real
            isfinite(x) || push!(bad, "$name=$x")
        elseif x isa AbstractArray && any(!isfinite, x)
            n_nan = count(isnan, x)
            n_inf = count(v -> !isnan(v) && !isfinite(v), x)
            push!(bad, "$name(nan=$n_nan inf=$n_inf/$(length(x)))")
        end
    end
    isempty(bad) || error("Non-finite in $where → " * join(bad, ", "))
end

function train!(model, p::TrainingParams, loss_fn::Function; info = Dict())
    if p.optimizer_state === nothing
        p.optimizer_state = Flux.setup(p.optimizer, model)
    end
    val, grads = Flux.withgradient(m -> loss_fn(m; info = info) + p.regularizer(m), model)
    typeof(val) == Float64 && @error "Float64 loss found: computation in double precision may be slow"
    gnorm = global_grad_norm(grads[1])
    # Localize on ANY non-finite grad-norm (NaN *or* Inf). Catching Inf too is
    # the point: an Inf grad-norm previously slipped this check, got clipped to
    # NaN by ClipNorm, and corrupted the params — so the *next* step reported a
    # useless "everything is NaN". Reporting here names the culprit early.
    if !isfinite(gnorm)
        @error "Non-finite gradient norm in train!(\"$(p.name)\")" gnorm loss=val loss_is_finite=isfinite(val)
        # loss finite + grad non-finite ⇒ the NaN/Inf is born in the BACKWARD
        # pass (e.g. 0·Inf, or d/dx of √/log/clamp at a bad point). loss already
        # non-finite ⇒ it came from the FORWARD loss computation.
        println(stderr, "── non-finite gradient leaves (path  shape  counts) ──")
        summarize_nonfinite_grads(stderr, grads[1])
        error("Non-finite gradient norm ($gnorm), loss=$val. See per-leaf summary above.")
    end
    Flux.update!(p.optimizer_state, model, grads[1])
    info[Symbol(p.name, "loss")] = val
    info[Symbol(p.name, "grad_norm")] = gnorm
    info
end

# Train with minibatches and epochs.
#
# `π` is the *model* whose parameters are updated (e.g. an actor or critic).
# `π_loss` defaults to `π` and is the object passed to the user-supplied loss
# function; for SAC etc. it can wrap additional context. The differentiated
# argument is `π` (whatever object becomes `m` inside `loss_fn`).
function batch_train!(π, p::TrainingParams, 𝒫, 𝒟::ExperienceBuffer...; info=Dict(), π_loss=π)
    infos = []
    total_batches = 0
    early_stopped = false
    for epoch in 1:p.epochs
        minibatch_infos = []

        # Shuffle the experience buffers
        for D in 𝒟
            shuffle!(D)
        end

        partitions = [partition(1:length(D), p.batch_size) for D in 𝒟]
        for indices in zip(partitions...)
            mbs = [minibatch(D, i) for (D, i) in zip(𝒟, indices)]
            # `m` is the differentiated model. For losses that need the
            # full policy (π_loss ≠ π), the loss function itself wires it.
            loss_fn = (m; info = Dict()) -> p.loss(m, 𝒫, mbs...; info = info)
            push!(minibatch_infos, train!(π, p, loss_fn, info=info))
            total_batches += 1
            total_batches >= p.max_batches && break
            if p.early_stopping([infos...,  aggregate_info(minibatch_infos)])
                early_stopped = true
                break
            end
        end
        push!(infos, aggregate_info(minibatch_infos))
        if p.early_stopping(infos)
            early_stopped = true
            break
        end
        total_batches >= p.max_batches && break

    end
    info[Symbol(p.name, "batches_trained")] = total_batches
    info[Symbol(p.name, "early_stopped")] = early_stopped
    merge!(info, aggregate_info(infos))
end
