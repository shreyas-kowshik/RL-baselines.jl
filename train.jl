using Flux, CuArrays
using OpenAIGym
import Reinforce.action
import Reinforce:run_episode
import Flux.params
using Flux.Tracker: grad, update!
using Flux: onehot
using Statistics
using Distributed
using Distributions
using LinearAlgebra
using Base.Iterators
using Random
using BSON:@save,@load
using JLD

include("policy.jl")

"""
HYPERPARAMETERS
"""
# Environment Creation #
env_name = "Pendulum-v0"
MODE = "CON" # Can be either "CON" (Continuous) or "CON" (Categorical)

# Environment Variables #
STATE_SIZE = 3
ACTION_SIZE = 1
MIN_RANGE = -2.0f0
MAX_RANGE = 2.0f0
EPISODE_LENGTH = 2000
TEST_STEPS = 10000
REWARD_SCALING = 16.2736044
# Policy parameters #
η = 3e-4 # Learning rate
STD = 0.0 # Standard deviation
HIDDEN_SIZE = 30
# GAE parameters
γ = 0.99
λ = 0.95
# Optimization parameters
PPO_EPOCHS = 10
NUM_EPISODES = 100000
BATCH_SIZE = 256
c₀ = 1.0
c₁ = 1.0
c₂ = 0.001
# PPO parameters
ϵ = 0.2
# FREQUENCIES
SAVE_FREQUENCY = 50
VERBOSE_FREQUENCY = 5
global_step = 0

# Global variable to monitor losses
reward_hist = []
policy_l = 0.0
entropy_l = 0.0
value_l = 0.0

#---------Scale rewards-------#
function scale_rewards(rewards)
    return (rewards  ./ REWARD_SCALING) .+ 2.0f0
end

function normalise(arr)
    (arr .- mean(arr))./(sqrt(var(arr) + 1e-10))
end

"""
Define the networks
"""

if MODE == "CON"
    policy_μ,policy_Σ = gaussian_policy(STATE_SIZE,HIDDEN_SIZE,ACTION_SIZE)
    value = value_fn(STATE_SIZE,HIDDEN_SIZE,ACTION_SIZE,tanh)
elseif MODE == "CAT"
    policy = categorical_policy(STATE_SIZE,HIDDEN_SIZE,ACTION_SIZE)
    value = value_fn(STATE_SIZE,HIDDEN_SIZE,ACTION_SIZE,relu)
else 
    error("MODE can only be (CON) or (CAT)...")
end

opt = ADAM(η)

"""
Functions to get rollouts
"""

function action(state)
    # Acccounting for the element type
    state = reshape(Array(state),length(state),1) 

    a = nothing
    if MODE == "CON"
        # Our policy outputs the parameters of a Normal distribution
        μ = policy_μ(state)
        μ = reshape(μ,ACTION_SIZE)
        log_std = policy_Σ
        
        σ² = (exp.(log_std)).^2
        Σ = diagm(0=>σ².data)
        
        dis = MvNormal(μ.data,Σ)
        
        a = rand(dis,ACTION_SIZE)
    else
        action_probs = policy(state).data
        action_probs = reshape(action_probs,ACTION_SIZE)
        a = sample(1:ACTION_SIZE,Weights(action_probs)) - 1
    end
    a
end

function run_episode(env)
    experience = []
    
    s = reset!(env)
    for i in 1:EPISODE_LENGTH
        a = action(s)
        # a = convert.(Float64,a)
        
        if MODE == "CON"
            a = reshape(a,ACTION_SIZE)
        end

        r,s_ = step!(env,a)
        push!(experience,(s,a,r,s_))
        s = s_
        if env.done
           break 
        end
    end
    experience
end

"""
Multi-threaded parallel rollout collection
"""

num_processes = 16 
addprocs(num_processes) 

@everywhere function collect(env)
    run_episode(env)
end

@everywhere function rollout()
  env = GymEnv(env_name)
  env.pyenv._max_episode_steps = EPISODE_LENGTH
  return collect(env)
end

function get_rollouts()
    g = []
    for  w in workers()
      push!(g, rollout())
    end

    fetch.(g)
end

"""
Generalized Adavantage Estimation
"""

function gae(states,actions,rewards,next_states)
    """
    Returns a Generalized Advantage Estimate for an episode
    """
    Â = []
    A = 0.0
    for i in reverse(1:length(states))
        if length(states) < EPISODE_LENGTH && i == length(states)
	    println("CASE")
            δ = rewards[i] - cpu.(value(states[i]).data[1])
        else
            δ = rewards[i] + γ*cpu.(value(next_states[i]).data[1]) - cpu.(value(states[i]).data[1])
        end

        A = δ + (γ*λ*A)
        push!(Â,A)
    end
    
    Â = reverse(Â)
    return Â
end

function disconunted_returns(rewards)
    r = 0.0
    returns = []
    for i in reverse(1:length(rewards))
        r = rewards[i] + γ*r
        push!(returns,r)
    end
    returns = reverse(returns)
    returns
end

"""
Calculate Log Probabilities
"""
function log_prob_from_actions(states,actions)
    """
    Returns log probabilities of the actions taken
    
    states,actions : episode vairbles in the form of a list
    """
    log_probs = []
    
    for i in 1:length(states)
        if MODE == "CON"
            μ = reshape(policy_μ(states[i]),ACTION_SIZE).data
            logΣ = policy_Σ.data |> cpu
            push!(log_probs,normal_log_prob(μ,logΣ,actions[i]))
        else
            action_probs = policy(states[i])
            prob = action_probs[actions[i],:].data
            push!(log_probs,log.(prob))
        end
    end
    
    log_probs
end


"""
Process and extraction information from rollouts
"""

function process_rollouts(rollouts)
    """
    rollouts : variable returned by calling `get_rollouts`
    
    Returns : 
    states, actions, rewards for minibatch processing
    """
    # Process the variables
    states = []
    actions = []
    rewards = []
    next_states = []
    advantages = []
    returns = []
    log_probs = []
    
    # Logging statistics
    episode_mean_returns = []
    
    for ro in rollouts
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_next_states = []
        
        for i in 1:length(ro)
             push!(episode_states,Array(ro[i][1]))
             
             if MODE == "CON"
                 push!(episode_actions,ro[i][2])
             else
                 push!(episode_actions,ro[i][2] + 1)
             end
             
             push!(episode_rewards,ro[i][3])
             push!(episode_next_states,ro[i][4])
        end
        
        episode_rewards = scale_rewards(episode_rewards)
        episode_advantages = gae(episode_states,episode_actions,episode_rewards,episode_next_states)
        episode_advantages = normalise(episode_advantages)
	# episode_rewards = normalise(episode_rewards)
         
        episode_returns = disconunted_returns(episode_rewards)
         
        push!(episode_mean_returns,mean(episode_returns))
         
        push!(states,episode_states)
        push!(actions,episode_actions)
        push!(rewards,episode_rewards)
        push!(advantages,episode_advantages)
        push!(returns,episode_returns)
        push!(log_probs,log_prob_from_actions(episode_states,episode_actions))
    end
    
    states = cat(states...,dims=1)
    actions = cat(actions...,dims=1)
    rewards = cat(rewards...,dims=1)
    advantages = cat(advantages...,dims=1)
    returns = cat(returns...,dims=1)
    log_probs = cat(log_probs...,dims=1)
    
    push!(reward_hist,mean(episode_mean_returns))
    
    if length(reward_hist) <= 100
        println("RETURNS : $(mean(episode_mean_returns))")
    else
        println("MEAN RETURNS : $(mean(reward_hist))")
        println("LAST 100 RETURNS : $(mean(reward_hist[end-100:end]))")
    end
    
    return hcat(states...),hcat(actions...),hcat(rewards...),hcat(advantages...),hcat(returns...),hcat(log_probs...)
end


"""
Loss function definition
"""
function loss(states,actions,advantages,returns,old_log_probs)
    global global_step,policy_l,entropy_l,value_l
    global_step += 1
     
    if MODE == "CON"
        μ = policy_μ(states)
        logΣ = policy_Σ 
        
        new_log_probs = normal_log_prob(μ,logΣ,actions)
    else
        action_probs = policy(states) # ACTION_SIZE x BATCH_SIZE
	# println("")
	# println(size(action_probs))
	# println(action_probs)
        actions_one_hot = zeros(ACTION_SIZE,size(action_probs)[end])
        
        for i in 1:size(action_probs)[end]
            actions_one_hot[actions[:,i][1],i] = 1.0                
        end
        
        new_log_probs = log.(sum((action_probs .+ 1f-5) .* actions_one_hot,dims=1))
    end
    
    # Surrogate loss computations
    ratio = exp.(new_log_probs .- old_log_probs)
    surr1 = ratio .* advantages
    surr2 = clamp.(ratio,(1.0 - ϵ),(1.0 + ϵ)) .* advantages
    policy_loss = mean(min.(surr1,surr2))
    
    value_predicted = value(states)
    value_loss = mean((value_predicted .- returns).^2)
    
    if MODE == "CON"
        entropy_loss = mean(normal_entropy(logΣ))
    else
        entropy_loss = mean(categorical_entropy(action_probs))
    end
    
    policy_l = policy_loss.data
    entropy_l = entropy_loss.data
    value_l = value_loss.data
    
    -c₀*policy_loss + c₁*value_loss - c₂*entropy_loss
end

"""
Optimization Function
"""
function ppo_update(states,actions,advantages,returns,old_log_probs)
    # Define model parameters
    if MODE == "CON"
        model_params = params(params(policy_μ)...,params(policy_Σ)...,params(value)...)
    else
        model_params = params(params(policy)...,params(value)...)
    end

    # Calculate gradients
    gs = Tracker.gradient(() -> loss(states,actions,advantages,returns,old_log_probs),model_params)

    # Take a step of optimisation
    update!(opt,model_params,gs)
end

"""
Train
"""

function train_step()    
    routs = get_rollouts()
    states,actions,rewards,advantages,returns,log_probs = process_rollouts(routs)

    idxs = partition(shuffle(1:size(states)[end]),BATCH_SIZE)
      
    for epoch in 1:PPO_EPOCHS
        for i in idxs
            mb_states = states[:,i] 
            mb_actions = actions[:,i] 
            mb_advantages = advantages[:,i] 
            mb_returns = returns[:,i] 
            mb_log_probs = log_probs[:,i]
            
            ppo_update(mb_states,mb_actions,mb_advantages,mb_returns,mb_log_probs)
        end
    end
end

function train()
    for i in 1:NUM_EPISODES
        println("EP : $i")
        train_step()
        println("Ep done")
        
        # Anneal learning rate
        if i%300 == 0
            if opt.eta > 1e-6
                opt.eta = opt.eta / 1.0
            end
        end
        
        if i % VERBOSE_FREQUENCY == 0
            # Show important statistics
            println("-----___Stats___-----")
            
            if MODE == "CON"
                println("Entropy : $(normal_entropy(policy_Σ))")
            end
            
            println("Policy Loss : $(policy_l)")
            println("Entropy Loss : $(entropy_l)")
            println("Value Loss : $(value_l)")
        end
        
        if i%SAVE_FREQUENCY == 0
            if MODE == "CON"
                @save "weights/policy_mu.bson" policy_μ
                @save "weights/policy_sigma.bson" policy_Σ
                @save "weights/value.bson" value
            else
                @save "weights/policy_cat.bson" policy
                @save "weights/value.bson" value
            end
            
            save("stats.jld","rewards",reward_hist)
            println("\n\n\n----MAX REWRD SO FAR : $(maximum(reward_hist))---\n\n\n")
        end
    end
end

train()
