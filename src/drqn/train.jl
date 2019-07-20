using Pkg
Pkg.activate("../../Project.toml")

using Flux
using Gym
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
using BSON
using BSON:@save,@load
using JLD

include("../common/utils.jl")
include("../common/buffer.jl")

ENV_NAME = "CartPole-v0"
env = make(ENV_NAME,:rgb)

# Exploration params
ϵ = 1.0   # Initial exploration rate
ϵ_DECAY = 0.99  # Decay rate for exploration
ϵ_MIN = 0.02    # Minimum value for exploration

# Hyperparams
SEQ_LENGTH = 8
BATCH_SIZE = 256
γ = 0.99f0
τ = 50 # Target update frequency
global update_counter = 0
NUM_EPISODES = 100000

# Optimiser params
η = 0.01f0   # Learning rate

opt = ADAM(η)

# Memory related functions
MEM_SIZE = 100000
memory = []

function add(s,a,r,s_,done)
    if length(memory) == MEM_SIZE
        deleteat!(memory,1)
    end
    
    push!(memory,(s,a,r,s_,done))
end

function is_mem_full()
    return length(memory) >= MEM_SIZE
end

# Define neural networks
STATE_SIZE = 4
ACTION_SIZE = 2

global net = Chain(Dense(STATE_SIZE,30,tanh),Dense(30,50,tanh),LSTM(50,100),
            Dense(100,ACTION_SIZE))

global target_net = deepcopy(net)

function action(state)
    Q_vals = net(state).data
    
    if rand() <= ϵ
        act = rand(1:ACTION_SIZE) 
    end
    
    return act
    
    Q_values = net(state)
    act = Flux.argmax(act_values)[1]
    
    return act
end

function init_memory()
    # Fill memory with experiences
    s = reset!(env)
    
    while true
        a = action(s)
        
        s_,r = step!(env,a)
        add(s,a,r,s_,env.done)
        
        s = s_
        
        if env.done
            s = reset!(env)
        end
        
        if is_mem_full()
            break
        end
    end
end

function episode()
    s = reset!(env)
    ep_r = 0.0
    
    while true
        a = action(s)
        
        s_,r = step!(env,a)
        add(s,a,r,s_,env.done)
        
        s = s_
        
        ep_r += r
        
        ϵ *= ϵ > ϵ_MIN ? ϵ_DECAY : 1.0f0
        
        if env.done
            break
        end
    end
    
    ep_r
end

function loss(states,next_states,rewards,actions)
    # We are concenrned only about the last timestep's information
    # The previous steps are just to encode temporal information
    Q_s = net.(states)[end]
    Q_target = target_net.(next_states)[end]
    
    target = ones(size(Q_s))
    for i in 1:size(Q_s)[end]
        max_action = argmax(Q_s[:,i])
        target[:,i] = Q_s[:,i].data
        target[actions[i],i] = rewards[i] + γ * Q_target[max_action,i].data
    end
    
    # Pseudo-Huber Loss function
    mean(sqrt.(1.0f0 .+ (Q_s .- target).^2) .- 1.0f0)
end

function train_step()
    global net,target_net,update_counter
    # Reset the hidden layer of the network #
    Flux.reset!(net.layers[3])
    Flux.reset!(target_net.layers[3])
    
    idx = Distributions.sample(1:MEM_SIZE,BATCH_SIZE,replace=false)
    
    mb_states = []
    mb_next_states = []
    mb_actions = []
    mb_rewards = []
    
    # Get timestep-wise data from different indices
    for i in idx
        start = i - SEQ_LENGTH + 1
        
        experience = memory[max(start,1):i]
        
        for (t,exp) in enumerate(experience)
            if exp[5] == true # If episode termination lies in current sequence
                experience = experience[t+1:end] # Consider the later episode in memory
                break
            end
        end
        
        # Temporary Hack
        if length(experience) < SEQ_LENGTH
            continue
        end
        
        push!(mb_states,[exp[1] for exp in experience]...)
        push!(mb_next_states,[exp[4] for exp in experience]...)
        push!(mb_rewards,[exp[3] for exp in experience]...)
        push!(mb_actions,[exp[2] for exp in experience]...)
    end

    chunk_size = div(length(mb_states),SEQ_LENGTH)
    
    states = collect(partition(Flux.batchseq(Flux.chunk(mb_states,chunk_size),zeros(4)),SEQ_LENGTH))[1]
    next_states = collect(partition(Flux.batchseq(Flux.chunk(mb_next_states,chunk_size), zeros(4)), SEQ_LENGTH))[1]
    rewards = collect(partition(Flux.batchseq(Flux.chunk(mb_rewards,chunk_size), -1.0f0), SEQ_LENGTH))[1][end]
    actions = collect(partition(Flux.batchseq(Flux.chunk(mb_actions,chunk_size), -1), SEQ_LENGTH))[1][end]

    # Compute loss and gradients #
    gs = Tracker.gradient(() -> loss(states,next_states,rewards,actions),params(net))
    
    # Take a update step
    update!(opt,params(net),gs)
    
    if update_counter % τ == 0
        # Copy parameters onto target network
        target_net = deepcopy(net)
        println("Updated target network")
    end
    
    update_counter += 1
end

if is_mem_full() == false
    init_memory()
end

function train()
    for i in 1:NUM_EPISODES
        ep_r = episode()
        println("Episode Rewards : $ep_r")
        
        train_step()
    end
end

train()