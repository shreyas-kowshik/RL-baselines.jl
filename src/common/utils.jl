# weight initialization
function _random_normal(shape...)
    return map(Float32,rand(Normal(0,0.1),shape...))
end

function constant_init(shape...)
    return map(Float32,ones(shape...) * 0.1)
end

function normalise(arr)
    (arr .- mean(arr))./(sqrt(var(arr) + 1e-10))
end

"""
Returns a Generalized Advantage Estimate for an episode
"""

function gae(policy,states::Array,actions::Array,rewards::Array,next_states::Array,num_steps::Int;γ=0.99,λ=0.95)
    Â = []
    A = 0.0
    for i in reverse(1:length(states))
        if length(states) < num_steps && i == length(states)
            δ = rewards[i] - policy.value_net(states[i]).data[1]
        else
            δ = rewards[i] + γ*policy.value_net(next_states[i]).data[1] - policy.value_net(states[i]).data[1]
        end

        A = δ + (γ*λ*A)
        push!(Â,A)
    end
    
    Â = reverse(Â)
    return Â
end

"""
Returns the cumulative discounted returns for each timestep
"""

function disconunted_returns(rewards::Array;γ=0.99)
    r = 0.0
    returns = []
    for i in reverse(1:length(rewards))
        r = rewards[i] + γ*r
        push!(returns,r)
    end
    returns = reverse(returns)
    returns
end
