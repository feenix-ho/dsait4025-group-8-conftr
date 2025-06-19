using Pkg
Pkg.add("ConformalPrediction")

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

regressor = @load SRRegressor pkg=SymbolicRegression
model = regressor(
    niterations=50,
    binary_operators=[+, -, *],
    unary_operators=[sin],
)

using ConformalPrediction
conf_model = conformal_model(model)
mach = machine(conf_model, X, y)
fit!(mach, rows=train)

show_first = 5
Xtest = selectrows(X, test)
ytest = y[test]
ŷ = predict(mach, Xtest)
ŷ[1:show_first]


# using Plots
# zoom = 0
# plt = plot(mach.model, mach.fitresult, Xtest, ytest, lw=5, zoom=zoom, observed_lab="Test points")
# xrange = range(-xmax+zoom,xmax-zoom,length=N)
# plot!(plt, xrange, @.(fun(xrange)), lw=2, ls=:dash, colour=:darkorange, label="Ground truth")

_eval = evaluate!(mach; measure=[emp_coverage, ssc], verbosity=0)
display(_eval)
println("Empirical coverage: $(round(_eval.measurement[1], digits=3))")
println("SSC: $(round(_eval.measurement[2], digits=3))")