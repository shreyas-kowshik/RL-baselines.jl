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

env_name = "Pendulum-v0"
MODE = "CON"
ACTION_SIZE = 1
TEST_STEPS = 50000 
global steps_run = 0

# Load the policy
if MODE == "CON"
    @load "./weights/policy_mu.bson" policy_μ
    @load "./weights/policy_sigma.bson" policy_Σ
    println("\n\n\n")
    println(policy_Σ)
else
    @load "./weights/policy_cat.bson" policy
end

# Test Run Function
function test_run(env)
    global steps_run 
    ep_r = 0.0
    
    s = reset!(env)
    for i in 1:TEST_STEPS
        if i % 10000 == 0
            println("Resetting...")
            s = reset!(env)
        end
        # OpenAIGym.render(env)

        if MODE == "CON"
            a = policy_μ(s).data
            a = convert.(Float64,a)
            a = reshape(a,ACTION_SIZE)
        else
            action_probs = policy(s).data
            action_probs = reshape(action_probs,ACTION_SIZE)
            a = sample(1:ACTION_SIZE,Weights(action_probs)) - 1
        end
	
	println(a)
        r,s_ = step!(env,a)
        ep_r += r
	steps_run += 1

        s = s_
        if env.done
           break 
        end
    end
    ep_r
end

env = GymEnv(env_name)
env.pyenv._max_episode_steps = TEST_STEPS

if MODE == "CAT"
	ret = []
	for _ in 1:100
		r = test_run(env)
		println("---Total Steps : $steps_run ::: Total Reward : $r---")
		push!(ret,r)
	end
	
	println("Minimum : $(minimum(ret))")
	println("Maximum : $(maximum(ret))")
	println("Mean : $(mean(ret))")
else
	r = test_run(env)	
	println("---Total Steps : $steps_run ::: Total Reward : $r---")
end
