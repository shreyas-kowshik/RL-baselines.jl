# RL-Baselines

A repository inspired along the lines of the [OpenAI Baselines](https://github.com/openai/baselines) in julia.

Algorithms supported : 

```
Trust Region Policy Optimization

Proximal Policy Optimization
```

![Sample](docs/pendulum.gif)

Tested environments : 

```
CartPole-v0 Pendulum-v0
```

## Instructions To Run

<b>Training</b>

Go to the respective algorithm folder `cd src/ppo` or `cd src/trpo` and run : 

```
julia train.jl
```

<b>Testing</b>

```
cd src/

julia run.jl
```

## Results

![CartPole](docs/cartpole.gif)

![Pendulum](docs/pendulum.gif)
