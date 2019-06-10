using Pkg
Pkg.activate("./plot_env/")

using JLD
using Plots

rh = load("stats.jld")
i = plot(rh["rewards"],fmt=:png)
savefig(i,"rewards.png")
