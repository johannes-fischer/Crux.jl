"""
Experience replay buffer.
"""
function ExperienceReplay(;π, 
                          S, 
                          𝒫=(;),
                          A=action_space(π), 
                          N_experience_replay,
                          ER_frac = 0.5,
                          solver=TD3, 
                          required_columns=Symbol[],  
                          c_opt=(;), 
                          a_opt=(;), 
                          replay_store_weight = (D)->1f0,
                          kwargs...)
                          
    buffer_er = ExperienceBuffer(S, A, N_experience_replay, [:weight, :value, required_columns...])
    𝒫 = (;buffer_er, 𝒫...)
    function cb(D; 𝒮, info=Dict())
        if 𝒮.agent.π isa LatentConditionedNetwork
            D[:value] = mean(value(critic(𝒮.agent.π), vcat(D[:s], D[:z]), D[:a]))
        else
            D[:value] = mean(value(𝒮.agent.π, D[:s], D[:a]))
        end
        D[:weight] .= replay_store_weight(D)
        push_reservoir!(buffer_er, D, weighted=true)
        info[:experience_size] = length(buffer_er)
    end
    creg = action_value_regularization
    areg = action_regularization
    if π isa LatentConditionedNetwork
        creg = TIER_action_value_regularization
        areg = TIER_action_regularization
    end
    
    
    
    solver(;  π=π,
              S=S,
              𝒫=𝒫,
              required_columns=unique([:weight, required_columns...]),
              post_sample_callback=cb,
              c_opt=(;regularizer=BatchRegularizer(buffers=[buffer_er], batch_size=128, λ=0.5f0, loss=creg), c_opt...),
              a_opt=(;regularizer=BatchRegularizer(buffers=[buffer_er], batch_size=128, λ=0.5f0, loss=areg), a_opt...),
              extra_buffers=[buffer_er],
              buffer_fractions=[1.0-ER_frac, ER_frac],
              kwargs...
              )
end

