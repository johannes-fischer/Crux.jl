# Flux 0.16 port: rewrites the diagonal Fisher regularizer to walk model trees
# via `Functors.fmap` instead of relying on `Flux.params` (gone in 0.16). The
# regularizer now stores `F` and `θ⁻` as model-shaped trees (deepcopies of the
# model with parameter arrays replaced/copied), and `(R)(model)` walks all
# three trees in parallel to compute the elastic-weight-consolidation penalty.
mutable struct DiagonalFisherRegularizer
    F           # model-shaped tree of running mean squared-gradient
    N::Int
    λ::Float32
    θ⁻          # model-shaped tree of last parameter values
end

# Construct from a model (preferred). Initial F is a copy of the model with
# every float-array leaf zeroed; θ⁻ is a full deepcopy.
function DiagonalFisherRegularizer(model, λ=1)
    F0 = Flux.fmap(model) do x
        x isa AbstractArray{<:AbstractFloat} ? zero(x) : x
    end
    DiagonalFisherRegularizer(F0, 0, Float32(λ), deepcopy(model))
end

function (R::DiagonalFisherRegularizer)(model)
    tot = 0f0
    nleaves = 0
    Flux.fmap(model, R.F, R.θ⁻) do p1, F, p2
        if p1 isa AbstractArray{<:AbstractFloat} &&
           F  isa AbstractArray{<:AbstractFloat} &&
           p2 isa AbstractArray{<:AbstractFloat}
            tot += mean(F .* (p1 .- p2).^2)
            nleaves += 1
        end
        p1
    end
    R.λ * tot / max(nleaves, 1)
end

# Accumulate the diagonal Fisher information by squaring the gradient of
# `neg_loss` with respect to `model` and folding it into a running mean.
#
# `neg_loss` should be a function `m -> scalar` (the differentiated argument
# is the model). For the old `() -> loss(𝒟)` signature, wrap it as
# `m -> loss(𝒟; model=m)` or similar.
function add_fisher_information_diagonal!(R::DiagonalFisherRegularizer, neg_loss, model)
    val, grads = Flux.withgradient(neg_loss, model)
    R.N += 1
    Flux.fmap(R.F, grads[1]) do f, g
        if f isa AbstractArray{<:AbstractFloat} && g isa AbstractArray{<:AbstractFloat}
            f .+= (g.^2 .- f) ./ R.N
        end
        f
    end
    nothing
end

function update_fisher!(R::DiagonalFisherRegularizer, 𝒟, loss, model, batch_size)
    shuffle!(𝒟)
    for i in partition(1:length(𝒟), batch_size)
        mb = minibatch(𝒟, i)
        add_fisher_information_diagonal!(R, m -> loss(m, mb), model)
    end
    R.θ⁻ = deepcopy(model)
    nothing
end
