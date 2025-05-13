### A Pluto.jl notebook ###
# v0.20.8

using Markdown
using InteractiveUtils

# ╔═╡ 91080a5e-2ffc-11f0-0f4e-0962e0ab142e
using Pkg

# ╔═╡ e8e0cee3-de91-43b1-a750-e4b043e58a55
Pkg.add("ConformalPrediction")

# ╔═╡ 6785d835-e233-41e4-a99d-7d18d82ced5c
using MLJ

# Inputs:
N = 600
xmax = 3.0
using Distributions
d = Uniform(-xmax, xmax)
X = rand(d, N)
X = reshape(X, :, 1)

# Outputs:
noise = 0.5
fun(X) = sin(X)
ε = randn(N) .* noise
y = @.(fun(X)) + ε
y = vec(y)

# Partition:
train, test = partition(eachindex(y), 0.4, 0.4, shuffle=true)

# ╔═╡ Cell order:
# ╠═91080a5e-2ffc-11f0-0f4e-0962e0ab142e
# ╠═e8e0cee3-de91-43b1-a750-e4b043e58a55
# ╠═6785d835-e233-41e4-a99d-7d18d82ced5c
