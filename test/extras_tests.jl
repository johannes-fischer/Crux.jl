using Crux
using Test
using Zygote
using Flux

## gradient penalty
m = Dense(2,1, init=ones, bias=false)
x = ones(Float32, 2, 100)
@test gradient_penalty(m, x) ≈ (sqrt(2) - 1)^2


# Other gradient-penalty test, exercised on the explicit-tree gradient API.
# Flux 0.16 port: the old version used `Flux.pullback(loss, Flux.params(m))`
# (implicit-params, removed) and pulled `.grads` off the Zygote.Grads dict.
# We now differentiate w.r.t. the model directly via `Flux.withgradient`.
let
    idim = 5
    batch_size = 8
    m = Chain(Dense(idim, 2*idim, tanh), Dense(2*idim, 1))
    x = rand(Float32, idim, batch_size)
    y = rand(Float32, 1, batch_size)
    total_loss(model) = Flux.mse(model(x), y) + gradient_penalty(model, x)
    l, grads = Flux.withgradient(total_loss, m)
    @test l > 0
    @test grads[1] !== nothing
end

# GPU variant — exercised only when CUDA is healthy in the host env.
if @isdefined(USE_CUDA) && USE_CUDA
    let
        idim = 5
        batch_size = 8
        m_g = Chain(Dense(idim, 2*idim, tanh), Dense(2*idim, 1)) |> gpu
        x_g = rand(Float32, idim, batch_size) |> gpu
        y_g = rand(Float32, 1, batch_size) |> gpu
        total_loss_g(model) = Flux.mse(model(x_g), y_g) + gradient_penalty(model, x_g)
        l, grads = Flux.withgradient(total_loss_g, m_g)
        @test l > 0
        @test grads[1] !== nothing
    end
end

