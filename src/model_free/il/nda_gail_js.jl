function NDA_GAIL_JS(;π, 
                       S, 
                       𝒟_demo,
                       𝒟_nda,
                       Vc::ContinuousNetwork, # value network for estimating cost
                       γ::Float32,
                       λ_gae::Float32 = 0.95f0,
                       αr::Float32=0.5f0,
                       normalize_demo::Bool=true, 
                       D::ContinuousNetwork,
                       Dnda::ContinuousNetwork,
                       solver=LagrangePPO, 
                       gan_loss::Crux.GANLoss=GAN_BCELoss(), 
                       d_opt::NamedTuple=(;),
                       d_opt_nda::NamedTuple=(;), 
                       log::NamedTuple=(;), 
                       kwargs...)
                       
    d_opt = TrainingParams(;loss = Crux.gail_d_loss(gan_loss), name="discriminator_", d_opt...)
    d_opt_nda = TrainingParams(;loss = Crux.gail_d_loss(gan_loss), name="nda_discriminator_", d_opt_nda...)
    if normalize_demo
        𝒟_demo = normalize!(deepcopy(𝒟_demo), S, action_space(π))
        𝒟_nda = normalize!(deepcopy(𝒟_nda), S, action_space(π))
    end
    𝒟_demo = 𝒟_demo |> device(π)
    𝒟_nda = 𝒟_nda |> device(π)
    
    function GAIL_callback(𝒟; info=Dict(), 𝒮)
        batch_train!(D, d_opt, (;), 𝒟_demo, deepcopy(𝒟), info=info)
        batch_train!(Dnda, d_opt_nda, (;), 𝒟_nda, deepcopy(𝒟), info=info)
        
        # Set the reward
        D_out = value(D, 𝒟[:a], 𝒟[:s]) # This is swapped because a->x and s->y and the convention for GANs is D(x,y)
        r = αr * logσ.(D_out) .- (1f0 - αr) * logcompσ.(D_out)
        ignore_derivatives() do
            minval, maxval = extrema(D_out)
            info[:disc_reward] = mean(r)
        end
        𝒟[:r] .= r 
        
        # Set the cost
        D_out_nda = value(Dnda, 𝒟[:a], 𝒟[:s])
        r_nda = αr * logσ.(D_out_nda) .- (1f0 - αr) * logcompσ.(D_out_nda)
        c = max.(0, r_nda .- r)
        # c = max.(0, σ.(D_out_nda) .- σ.(D_out))
        ignore_derivatives() do
            info[:disc_nda_cost] = sum(c) / sum(𝒟[:episode_end])
        end
        𝒟[:cost] .= c
        
        eps = episodes(𝒟)
        for eprange in eps
            ep = eprange[1]:eprange[2]
            fill_gae!(𝒟, ep, 𝒮.agent.π, λ_gae, γ)
            fill_returns!(𝒟, ep, γ)
            
            fill_gae!(𝒟, ep, Vc, λ_gae, γ, source=:cost, target=:cost_advantage)
            fill_returns!(𝒟, ep, γ, source=:cost, target=:cost_return)
        end
        𝒟[:advantage] .= whiten(𝒟[:advantage])
        𝒟[:cost_advantage] .= whiten(𝒟[:cost_advantage])
        
    end
    solver(;π=π, S=S, post_batch_callback=GAIL_callback, Vc=Vc, λ_gae=λ_gae, log=(dir="log/ndagail", period=500, log...), kwargs...)
end

