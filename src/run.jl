using Pkg
Pkg.activate("../Project.toml")

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

include("common/policies.jl")
include("common/utils.jl")

ENV_NAME = "Pendulum-v0"
TEST_STEPS = 2000
global steps_run = 0

LOAD_PATH = "../weights/"

# Define policy
env_wrap = EnvWrap(ENV_NAME)

env = make(ENV_NAME,:rgb)
env.max_episode_steps = TEST_STEPS
policy = load_policy(env_wrap,LOAD_PATH)

# Test Run Function
function test_run(env)
	global steps_run
 	testmode!(env)
    ep_r = 0.0
    
    s = reset!(env)
    for i in 1:TEST_STEPS
    	println(i)
        # render!(env)
	a = policy.μ(s)
        s_,r,_ = step!(env,a)
	
        ep_r += r
        
		steps_run += 1

        s = s_
        if env.done
           break 
        end
    end
    ep_r
end

total_reward = test_run(env)

println("TOTAL STEPS :: $steps_run :: TOTAL REWARD :: $total_reward")
