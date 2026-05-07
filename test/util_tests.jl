using Crux
using Test
using POMDPModels
using Flux
using LinearAlgebra
using CUDA
using Distributions

# bslice
v = zeros(4,4,4)
@test size(bslice(v, 2)) == (4,4)

# Constant Layer
c1 = ConstantLayer(ones(10))
@test Crux.device(c1) == cpu
@test c1(rand(100)) == c1.vec
# Flux 0.16 port: implicit `Flux.params` is gone; check the underlying field.
@test c1.vec == ones(10)

if USE_CUDA
    c2 = c1 |> gpu
    @test Crux.device(c2) == gpu
    @test c2(rand(100)) == c2.vec
end

# Distribution stuff
objs = [:up, :down]
o = ObjectCategorical(objs)
@test o isa DiscreteUnivariateDistribution
@test o.objs == objs
@test o.cat.p == Categorical(2).p


@test rand(o) in objs
@test size(rand(o, 10)) == (10,)

@test logpdf(o, [:up]) == logpdf(o, [:down])
@test size(logpdf(o, rand(o,10))) == (1,10)

## Flux Stuff — global_grad_norm helper
# Flux 0.16 port: the old `LinearAlgebra.norm(::Zygote.Grads)` overload is
# gone; Crux now exposes `global_grad_norm(grads_tree; p=2)` for explicit
# gradients. Rebuild the test in that style.
W = rand(2, 5)
b = rand(2)
x, y = rand(5), rand(2)
loss_fn(W, b) = sum(((W * x) .+ b .- y).^2)
val, grads = Flux.withgradient(loss_fn, W, b)
@test val > 0
@test Crux.global_grad_norm(grads) > 1


##  MultitaskDecay Schedule
m = MultitaskDecaySchedule(10, [1,2,3])
l = Crux.LinearDecaySchedule(1.0, 0.1, 10)

for i=1:10
    @test m(i) == l(i)
end

for i=11:20
    @test m(i) == l(i-10)
end

for i=21:30
    @test m(i) == l(i-20)
end

m = MultitaskDecaySchedule(10, [1,2,1])

for i=1:10
    @test m(i) == l(i)
end

for i=11:20
    @test m(i) == l(i-10)
end

for i=21:30
    @test m(i) == l(i-10)
end

@test m(31) == 0.1
@test m(0) == 1





