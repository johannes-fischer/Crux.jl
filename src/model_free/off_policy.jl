"""
Off policy solver type.

Fields
======

- `agent::PolicyParams` # Policy parameters ([`PolicyParams`](@ref))
- `S::AbstractSpace` # State space
- `N::Int = 1000` # Number of environment interactions
- `ΔN::Int = 4` # Number of interactions between updates
- `max_steps::Int = 100` # Maximum number of steps per episode
- `log::Union{Nothing, LoggerParams} = nothing` # The logging parameters
- `i::Int = 0` # The current number of environment interactions
- `param_optimizers::Dict{Any, TrainingParams} = Dict()` # Training parameters for the parameters
- `a_opt::Union{Nothing, TrainingParams} = nothing` # Training parameters for the actor
- `c_opt::TrainingParams` # Training parameters for the critic
- `𝒫::NamedTuple = (;)` # Parameters of the algorithm
- `interaction_storage = nothing` # If this is initialized to an array then it will store all interactions
- `post_sample_callback = (𝒟; kwargs...) -> nothing` # Callback that that happens after sampling experience


Off-policy-specific parameters
======

- `post_batch_callback = (𝒟; kwargs...) -> nothing` Callback that that happens after sampling a batch
- `pre_train_callback = (𝒮; kwargs...) -> nothing` callback that gets called once prior to training
- `target_update = (π⁻, π; kwargs...) -> polyak_average!(π⁻, π, 0.005f0)` Function for updating the target network
- `target_fn` Target for critic regression with input signature `(π⁻, 𝒟, γ; i)`
- `buffer_size = 1000` Size of the buffer
- `required_columns = Symbol[]`
- `buffer = ExperienceBuffer(S, agent.space, buffer_size, required_columns)` The replay buffer
- `priority_fn = td_error` function for prioritized replay
- `buffer_init::Int = max(c_opt.batch_size, 200)` Number of observations to initialize the buffer with
- `extra_buffers = []` extra buffers (i.e. for experience replay in continual learning)
- `buffer_fractions = [1.0]` Fraction of the minibatch devoted to each buffer
"""
@with_kw mutable struct OffPolicySolver <: Solver
    agent::PolicyParams # Policy
    S::AbstractSpace # State space
    N::Int = 1000 # Number of environment interactions
    ΔN::Int = 4 # Number of interactions between updates
    max_steps::Int = 100 # Maximum number of steps per episode
    log::Union{Nothing, LoggerParams} = nothing # The logging parameters
    i::Int = 0 # The current number of environment interactions
    param_optimizers::Dict{Any, TrainingParams} = Dict() # Training parameters for the parameters
    a_opt::Union{Nothing, TrainingParams} = nothing # Training parameters for the actor
    c_opt::TrainingParams # Training parameters for the critic
    𝒫::NamedTuple = (;) # Parameters of the algorithm
	interaction_storage = nothing # If this is initialized to an array then it will store all interactions
	post_sample_callback = (𝒟; kwargs...) -> nothing # Callback that that happens after sampling experience
    
    # Off-policy-specific parameters
	post_batch_callback = (𝒟; kwargs...) -> nothing # Callback that that happens after sampling a batch
	pre_train_callback = (𝒮; kwargs...) -> nothing # callback that gets called once prior to training
    target_update = (π⁻, π; kwargs...) -> polyak_average!(π⁻, π, 0.005f0) # Function for updating the target network
    target_fn # Target for critic regression with input signature (π⁻, 𝒟, γ; i)
    buffer_size = 1000 # Size of the buffer
    required_columns = Symbol[]
    buffer = ExperienceBuffer(S, agent.space, buffer_size, required_columns) # The replay buffer
	priority_fn = td_error  # function for prioritized replay
    buffer_init::Int = max(c_opt.batch_size, 200) # Number of observations to initialize the buffer with
    extra_buffers = [] # extra buffers (i.e. for experience replay in continual learning)
    buffer_fractions = [1.0] # Fraction of the minibatch devoted to each buffer
end

function value_training(𝒮::OffPolicySolver, 𝒟, γ)
    infos = []
    # Loop over the desired number of training steps
    for epoch in 1:𝒮.c_opt.epochs
        # Sample a random minibatch of 𝑁 transitions (sᵢ, aᵢ, rᵢ, sᵢ₊₁) from 𝒟
        rand!(𝒟, 𝒮.buffer, 𝒮.extra_buffers..., fracs=𝒮.buffer_fractions, i=𝒮.i)
        
        # Dictionary to store info from the various optimization processes
        info = Dict()
        
        # Callack for potentially updating the buffer
        𝒮.post_batch_callback(𝒟, 𝒮=𝒮, info=info)
        
        # Compute target
        y = 𝒮.target_fn(𝒮.agent.π⁻, 𝒮.𝒫, 𝒟, γ, i=𝒮.i)
        
        # Update priorities (for prioritized replay)
        isprioritized(𝒮.buffer) && update_priorities!(𝒮.buffer, 𝒟.indices, cpu(𝒮.priority_fn(𝒮.agent.π, 𝒮.𝒫, 𝒟, y)))
        
        # Train parameters (Flux 0.16 port: model-first train!; loss receives
        # the differentiated `m` plus the full policy via `π_loss=…`). The
        # full π is bound into a local before the closure to keep Zygote from
        # reaching back through the solver struct (it can't follow `𝒮.agent.π`
        # without also tracking 𝒮 as a closed-over Ref, which can confuse the
        # accum pass when m is a sub-tree of π).
        let π_full = 𝒮.agent.π, 𝒫_local = 𝒮.𝒫, 𝒟_local = 𝒟, y_local = y
            for (θs, p_opt) in 𝒮.param_optimizers
                p_loss_fn = p_opt.loss
                train!(θs, p_opt,
                       (m; info=Dict()) -> p_loss_fn(m, 𝒫_local, 𝒟_local; π_loss=π_full, info=info),
                       info=info)
            end

            if ((epoch-1) % 𝒮.c_opt.update_every) == 0
                c_loss_fn = 𝒮.c_opt.loss
                train!(critic(π_full), 𝒮.c_opt,
                       (m; info=Dict()) -> c_loss_fn(m, 𝒫_local, 𝒟_local, y_local; π_loss=π_full, info=info),
                       info=info)
            end

            if !isnothing(𝒮.a_opt) && ((epoch-1) % 𝒮.a_opt.update_every) == 0
                a_loss_fn = 𝒮.a_opt.loss
                train!(actor(π_full), 𝒮.a_opt,
                       (m; info=Dict()) -> a_loss_fn(m, 𝒫_local, 𝒟_local; π_loss=π_full, info=info),
                       info=info)

                𝒮.target_update(𝒮.agent.π⁻, π_full)
            end
        end
        
        # Store the training information
        push!(infos, info)
        
    end
    # If not using a separate actor, update target networks after critic training
    isnothing(𝒮.a_opt) && 𝒮.target_update(𝒮.agent.π⁻, 𝒮.agent.π, i=𝒮.i + 1:𝒮.i + 𝒮.ΔN)
    
    aggregate_info(infos)
end

function POMDPs.solve(𝒮::OffPolicySolver, mdp)
    # Construct the training buffer, constants, and sampler
    𝒟 = buffer_like(𝒮.buffer, capacity=𝒮.c_opt.batch_size, device=device(𝒮.agent.π))
    γ = Float32(discount(mdp))
    s = Sampler(mdp, 𝒮.agent, S=𝒮.S, max_steps=𝒮.max_steps, required_columns=extra_columns(𝒮.buffer))
    isnothing(𝒮.log.sampler) && (𝒮.log.sampler = s)

    # Fill the buffer with initial observations before training
	info = Dict()
	Nfill = max(0, 𝒮.buffer_init - length(𝒮.buffer))
	istart = 𝒮.i
	if Nfill > 0
		𝒮.i += Nfill
		steps!(s, 𝒮.buffer, Nsteps=Nfill, explore=true, i=𝒮.i, store=𝒮.interaction_storage, cb=(D)->𝒮.post_sample_callback(D, 𝒮=𝒮, info=info))
	end 
	
	# Log the pre-train performance
	log(𝒮.log, 𝒮.i, info, 𝒮=𝒮)
    
    # Loop over the desired number of environment interactions
    for 𝒮.i in range(𝒮.i, stop=istart + 𝒮.N - 𝒮.ΔN, step=𝒮.ΔN)
		# Store info here
		info = Dict()
        
        # Sample transitions into the replay buffer
		steps!(s, 𝒮.buffer, Nsteps=𝒮.ΔN, explore=true, i=𝒮.i, store=𝒮.interaction_storage, cb=(D)->𝒮.post_sample_callback(D, 𝒮=𝒮, info=info))
		
		𝒮.pre_train_callback(𝒮, info=info)
        
        # Train the networks
        training_info = value_training(𝒮, 𝒟, γ)
        
        # Log the results
        log(𝒮.log, 𝒮.i + 1:𝒮.i + 𝒮.ΔN, training_info, info, 𝒮=𝒮)
    end
    𝒮.i += 𝒮.ΔN
    𝒮.agent.π
end

