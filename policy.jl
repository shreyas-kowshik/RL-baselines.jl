# weight initialization
function _random_normal(shape...)
    return map(Float32,rand(Normal(0,0.1),shape...))
end

function constant_init(shape...)
    return map(Float32,ones(shape...) * 0.1)
end

function normal_log_prob(μ,log_std,a)
    """
    Returns the log probability of an action under a policy Gaussian policy π
    """
    σ = exp.(log_std)
    σ² = σ.^2
    -(((a .- μ).^2)./(2.0 * σ²)) .- 0.5*log.(sqrt(2 * π)) .- log.(σ)
end

function categorical_entropy(π)
	return sum(π .* log.(π .+ 1f-10),dims=1)
end

function normal_entropy(log_std)
    0.5 + 0.5 * log(2 * π) .+ log_std
end

function gaussian_policy(STATE_SIZE,HIDDEN_SIZE,ACTION_SIZE,STD=0.0)
	"""
	STD : Initial standard deviation values

	Returns : a neural network for a gaussian policy

	Feel free to modify the networks to suit your custom environment needs
	"""
	policy_μ = Chain(Dense(STATE_SIZE,HIDDEN_SIZE,tanh;initW = _random_normal,initb=constant_init),
                     Dense(HIDDEN_SIZE,ACTION_SIZE;initW = _random_normal,initb=constant_init),
                     x->tanh.(x),
                     x->param(2.0) .* x)
    policy_Σ = param(ones(ACTION_SIZE) * STD)
    
    return policy_μ,policy_Σ
end

function categorical_policy(STATE_SIZE,HIDDEN_SIZE,ACTION_SIZE)
	"""
	Returns a catrgorical policy
	"""
	policy = Chain(Dense(STATE_SIZE,HIDDEN_SIZE,relu;initW = _random_normal,initb=constant_init),
                   Dense(HIDDEN_SIZE,ACTION_SIZE;initW = _random_normal,initb=constant_init),
    			   x -> softmax(x))

	return policy
end

function value_fn(STATE_SIZE,HIDDEN_SIZE,ACTION_SIZE,activation_fn=relu)
	value = Chain(Dense(STATE_SIZE,HIDDEN_SIZE,activation_fn),
                  Dense(HIDDEN_SIZE,HIDDEN_SIZE,activation_fn),
                  Dense(HIDDEN_SIZE,1))
end
