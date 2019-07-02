"""
Implements rollout utility functions to collect trajectory experiences
"""

"""
Returns an episode's worth of experience
"""

function run_episode(env::Gym.EnvWrapper,policy,num_steps::Int)
    experience = []
    
    s = reset!(env)
    for i in 1:num_steps
        a = action(policy,s)

        s_,r,_ = step!(env,a)
        push!(experience,(s,a,r,s_))
        s = s_
        if env.done
           break 
        end
    end
    experience
end

addprocs(num_processes) 

@everywhere function collect(policy,env,num_steps::Int)
    run_episode(env,policy,num_steps::Int)
end

@everywhere function episode(policy,num_steps::Int)
  env = make(policy.env_wrap.ENV_NAME,:rgb)
  env.max_episode_steps = num_steps
  return collect(policy,env,num_steps::Int)
end

function get_rollouts(policy,num_steps::Int)
    g = []
    for  w in workers()
      push!(g, episode(policy,num_steps))
    end

    fetch.(g)
end

"""
Process and extraction information from rollouts
"""

function collect_and_process_rollouts(policy,episode_buffer::Buffer,num_steps::Int,stats_buffer::Buffer)
    rollouts = get_rollouts(policy,num_steps)
    
    # Process the variables
    states = []
    actions = []
    rewards = []
    next_states = []
    advantages = []
    returns = []
    log_probs = []
    
    # Logging statistics
    rollout_returns = []
    
    for ro in rollouts
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_next_states = []
        
        for i in 1:length(ro)
             push!(episode_states,ro[i][1])
             push!(episode_actions,ro[i][2])
             push!(episode_rewards,ro[i][3])
             push!(episode_next_states,ro[i][4])

             if typeof(ro[i][2]) <: Int64
                push!(log_probs,log_prob(policy,reshape(ro[i][1],length(ro[i][1]),1),[ro[i][2]]).data)
             elseif typeof(ro[i][2]) <: Array
                push!(log_probs,log_prob(policy,reshape(ro[i][1],length(ro[i][1]),1),reshape(ro[i][2],length(ro[i][2]),1)).data)
             end
        end
        
        episode_rewards = scale_rewards(policy.env_wrap,episode_rewards)
        episode_advantages = gae(policy,episode_states,episode_actions,episode_rewards,episode_next_states,num_steps)
        episode_advantages = normalise(episode_advantages)
        
        episode_returns = disconunted_returns(episode_rewards)

        push!(states,episode_states)
        push!(actions,episode_actions)
        push!(rewards,episode_rewards)
        push!(advantages,episode_advantages)
        push!(returns,episode_returns)

        # Variables for logging
        push!(rollout_returns,episode_returns)

    end
    
    episode_buffer.exp_dict["states"] = hcat(cat(states...,dims=1)...)
    episode_buffer.exp_dict["actions"] = hcat(cat(actions...,dims=1)...)
    episode_buffer.exp_dict["rewards"] = hcat(cat(rewards...,dims=1)...)
    episode_buffer.exp_dict["advantages"] = hcat(cat(advantages...,dims=1)...)
    episode_buffer.exp_dict["returns"] = hcat(cat(returns...,dims=1)...)
    episode_buffer.exp_dict["log_probs"] = hcat(cat(log_probs...,dims=1)...)

    # Log the statistics
    add(stats_buffer,"rollout_returns",mean(cat(rollout_returns...,dims=1)))
    
    return states,actions,rewards,advantages,returns,log_probs
end
