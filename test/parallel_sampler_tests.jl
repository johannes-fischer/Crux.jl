# Bit-equivalence test for the parallel sampler.
#
# Property: env e's contiguous slice in a parallel rollout, run with
# `samplers[e].rng = Xoshiro(seed)` and a deterministic MDP, is bit-identical
# (modulo BLAS reduction order in the batched matmul, which we treat as
# tolerable noise) to a single-env rollout with `sampler.rng = Xoshiro(seed)`.
#
# Test runs on both CPU and GPU policies — the carry-state +
# `batched_exploration` machinery has to work identically whether
# the network weights live on the host or on a CUDA device.
#
# We use a hand-rolled deterministic 2D point-mass MDP so the only RNG draws
# come from the policy exploration noise. With per-env RNG plumbing in
# place, those draws come from the per-env RNG in both paths.
#
# Run with:
#   julia --startup-file=no --project=. -t 2 test/parallel_sampler_tests.jl

using CUDA
using cuDNN
using Crux
using Flux
using POMDPs
using POMDPTools
using Random
using Test

# --- Toy MDP --------------------------------------------------------------
# State: (x, ẋ) ∈ R². Action: a ∈ R. Dynamics: deterministic point-mass with
# unit timestep. Reward shapes a smooth quadratic objective. Terminal at
# |x| > 5. Initial state is FIXED (no RNG) so the only randomness left in
# the rollout is the GaussianPolicy exploration noise.

struct PointMassMDP <: MDP{Vector{Float32}, Vector{Float32}} end

POMDPs.initialstate(::PointMassMDP) = Deterministic(Float32[0.0, 0.0])
POMDPs.discount(::PointMassMDP) = 0.99f0
POMDPs.actions(::PointMassMDP) = nothing  # continuous, see action_space below
POMDPs.isterminal(::PointMassMDP, s::Vector{Float32}) = abs(s[1]) > 5.0f0

function POMDPs.gen(::PointMassMDP, s::Vector{Float32}, a, rng::AbstractRNG)
    x, ẋ = s
    # action may be Float32 (after Crux scalarizes single-element actions) or
    # an array — handle both.
    a_val = a isa AbstractArray ? Float32(a[1]) : Float32(a)
    a1 = clamp(a_val, -1f0, 1f0)
    xp  = x + ẋ
    ẋp  = ẋ + a1
    sp  = Float32[xp, ẋp]
    r   = -xp^2 - 0.1f0 * a1^2
    (sp = sp, r = r)
end

# 1-arg gen (no rng): forwards to the explicit-rng method using default_rng.
function POMDPs.gen(m::PointMassMDP, s, a)
    POMDPs.gen(m, s, a, Random.default_rng())
end

# State / action space adaptors for Crux.
Crux.state_space(::PointMassMDP) = Crux.ContinuousSpace((2,); type = Float32)
Crux.convert_s(::Type{AbstractArray}, s::Vector{Float32}, ::PointMassMDP) = s

# --- Build a policy -------------------------------------------------------
const SEED = 12345
const SDIM = 2
const ADIM = 1
const HIDDEN = 16
const N_STEPS = 32      # buffer length per rollout (parallel: Nsteps_per_env)
const MAX_STEPS = 50

function build_policy(seed::Int; move_to_gpu::Bool = false)
    Random.seed!(seed)
    μ_chain = Chain(
        Dense(SDIM, HIDDEN, tanh; init = Flux.glorot_uniform(Random.default_rng())),
        Dense(HIDDEN, ADIM;       init = Flux.glorot_uniform(Random.default_rng())),
    )
    logΣ = zeros(Float32, ADIM)
    Random.seed!(seed + 100)
    critic_chain = Chain(
        Dense(SDIM, HIDDEN, tanh; init = Flux.glorot_uniform(Random.default_rng())),
        Dense(HIDDEN, 1;          init = Flux.glorot_uniform(Random.default_rng())),
    )
    if move_to_gpu
        μ_chain = μ_chain |> gpu
        critic_chain = critic_chain |> gpu
        # logΣ is wrapped in a ConstantLayer by GaussianPolicy; the policy
        # constructor takes the raw Array and builds the layer internally.
        # ContinuousNetwork picks device from the chain's leaves.
        logΣ_gpu = ContinuousNetwork(Chain(Crux.ConstantLayer(logΣ |> gpu)), length(logΣ))
        A = Crux.GaussianPolicy(Crux.ContinuousNetwork(μ_chain, ADIM), logΣ_gpu)
    else
        A = Crux.GaussianPolicy(Crux.ContinuousNetwork(μ_chain, ADIM), logΣ)
    end
    V = Crux.ContinuousNetwork(critic_chain)
    Crux.ActorCritic(A, V)
end

function run_equivalence_test(; on_gpu::Bool)
    # Build TWO IDENTICAL policy instances so independent test runs don't share
    # weight tensors. We `Random.seed!` to make their weights identical.
    π_single = build_policy(SEED; move_to_gpu = on_gpu)
    π_par    = build_policy(SEED; move_to_gpu = on_gpu)

    mdp = PointMassMDP()
    λ_gae = 0.95f0
    required_columns = [:return, :logprob, :advantage, :value]

    # --- Single-env rollout
    sampler_single = Crux.Sampler(mdp, Crux.PolicyParams(π = π_single);
                                  S = Crux.state_space(mdp),
                                  required_columns = required_columns,
                                  λ = λ_gae, max_steps = MAX_STEPS,
                                  rng = Xoshiro(SEED + 1))
    data_single = Crux.steps!(sampler_single; Nsteps = N_STEPS, explore = true,
                              i = 0, reset = true)

    # --- Parallel rollout, N = 2, env 1 seeded identically to single
    sampler_par_1 = Crux.Sampler(mdp, Crux.PolicyParams(π = π_par);
                                 S = Crux.state_space(mdp),
                                 required_columns = required_columns,
                                 λ = λ_gae, max_steps = MAX_STEPS,
                                 rng = Xoshiro(SEED + 1))
    sampler_par_2 = Crux.Sampler(mdp, Crux.PolicyParams(π = π_par);
                                 S = Crux.state_space(mdp),
                                 required_columns = required_columns,
                                 λ = λ_gae, max_steps = MAX_STEPS,
                                 rng = Xoshiro(SEED + 999))
    samplers_par = [sampler_par_1, sampler_par_2]
    data_par = Crux.steps!(samplers_par; Nsteps_per_env = N_STEPS, explore = true,
                           i = 0, reset = true)

    slice1 = 1:N_STEPS
    label = on_gpu ? "GPU" : "CPU"
    @testset "Parallel env-1 ≈ single-env at matching seed [$label]" begin
        atol = 1f-3
        for col in (:s, :a, :sp, :r, :done, :logprob, :advantage, :return, :value)
            single_col   = data_single[col]
            parallel_col = data_par[col][:, slice1]
            @test size(single_col) == size(parallel_col)
            @test isapprox(single_col, parallel_col; atol = atol)
            diff_max = maximum(abs.(single_col .- parallel_col))
            println("  [$label] ", lpad(string(col), 12), "  max|Δ| = ", diff_max)
        end
    end
end

# CPU run always
run_equivalence_test(; on_gpu = false)

# GPU run if CUDA functional. Crux's mdcall roundtrips CPU↔GPU so the
# rollout `data` is still CPU; the network forward just routes through CUDA.
if CUDA.functional()
    run_equivalence_test(; on_gpu = true)
else
    println("CUDA not functional — skipping GPU test.")
end
