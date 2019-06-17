"""
Test
"""

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
using BSON:@save,@load
using JLD

env_name = "CartPole-v0"
MODE = "CAT"
ACTION_SIZE = 2
TEST_STEPS = 50000

# Load the policy
if MODE == "CON"
    @load "./weights/policy_mu.bson" policy_μ
    @load "./weights/policy_sigma.bson" policy_Σ
else
    @load "./weights/policy_cat.bson" policy
end

# Test Run Function
function test_run(env)
    ep_r = 0.0
    
    s = reset!(env)
    for i in 1:TEST_STEPS
        if i % 1000000 == 0
            println("Resetting...")
            s = reset!(env)
        end
        OpenAIGym.render(env)

        if MODE == "CON"
            a = policy_μ(s).data
            a = convert.(Float64,a)
            a = reshape(a,ACTION_SIZE)
        else
            action_probs = policy(s).data
            action_probs = reshape(action_probs,ACTION_SIZE)
            a = sample(1:ACTION_SIZE,Weights(action_probs)) - 1
        end

        r,s_ = step!(env,a)
        ep_r += r
        
        s = s_
        if env.done
           break 
        end
    end
    ep_r
end

env = GymEnv(env_name)
env.pyenv._max_episode_steps = TEST_STEPS

r = test_run(env)
println("---Total Steps : $TEST_STEPS ::: Total Reward : $r---")