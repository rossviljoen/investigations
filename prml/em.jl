using Random, Distributions, Plots, LinearAlgebra

# Helper function to plot results
function plot_em(K, X, μ, Σ)
    IJulia.clear_output(true)
    plot_range = -3:0.1:9
    pl = plot()
    scatter!(pl, eachrow(X)...)
    for k = 1:K
        P = [pdf(MvNormal(μ[:,k], Σ[:,:,k]), [j,i]) for i in plot_range, j in plot_range]
        contour!(pl, plot_range, plot_range, P)
    end
    display(pl)
end

# Create some synthetic data
samples = 200

μ1 = [0, 2]
Σ1 = [1 0.9; 0.9 1]
d1 = MvNormal(μ1, Σ1)
X1 = rand(d1, samples)

μ2 = [3, 3]
Σ2 = [1 -0.9; -0.9 1]
d2 = MvNormal(μ2, Σ2)
X2 = rand(d2, samples)

μ3 = [6, 4]
Σ3 = [1 0.9; 0.9 1]
d3 = MvNormal(μ3, Σ3)
X3 = rand(d3, samples)

p = plot()
scatter!(p, eachrow(X1)...)
scatter!(p, eachrow(X2)...)
scatter!(p, eachrow(X3)...)

X = hcat(X1, X2, X3)
X = X[:, shuffle(1:end)] # shuffle the observations
N = length(X[1, :])

scatter(eachrow(X)...)

# The EM algorithm for a mixture of Gaussians
K = 3 # Number of mixture components

# Initialise the parameters
μ = 6 * rand(2, K) # [i, k]
Σ = cat([0.5 * Matrix{Float64}(I, 2, 2) for i=1:K]...; dims=3) # [i,j,k]
π = [1/K for i=1:K]

γ = zeros(N, K)

for _ = 1:30
    plot_em(K, X, μ, Σ)
    
    # E step
    for n = 1:N
        for k = 1:K
            # Unnormalised responsibilities
            γ[n,k] = π[k] * pdf(MvNormal(μ[:,k], Σ[:,:,k]), X[:, n])
        end
        γ = mapslices(x -> x/sum(x), γ; dims=2) # Normalise
    end

    # M step
    for k = 1:K
        N_k = sum(γ[:,k])
        μ[:,k] = (1 / N_k) * sum(γ[:,k] .* transpose(X), dims=1)
        Σ[:,:,k] = Hermitian((1 / N_k) * (transpose(γ[:,k]) .* (X .- μ[:,k]) * transpose(X .- μ[:,k])))
        π[k] = N_k / N
    end

    log_lik = 0
    for n = 1:N
        inner = 0
        for k = 1:K
            inner += π[k] * pdf(MvNormal(μ[:,k], Σ[:,:,k]), X[:,n])
        end
        log_lik += log(inner)
    end
    println(log_lik)
end
