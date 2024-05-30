using LinearAlgebra, Distributions, Random, MAT, Pkg, JuMP, TSVD, Gurobi, Dates, MathOptInterface 

include("../Exact Solvers/OA.jl")

Pkg.build("RCall")
using RCall
R"library(sMTL)"
# Put your Julia path here
R"Sys.setenv(JULIA_BINDIR = '/Applications/Julia-1.9.app/Contents/Resources/julia/bin/')"


Random.seed!(0)



p = 20 # Number of predictors
rho = 0.4 # Exponential Correlation Sigma_{ij} = rho^|i-j|

vector = 1:p
Sigma = abs.( ones(p)*vector' - vector*ones(p)')
Sigma = rho.^Sigma
dist = MvNormal(Sigma)




s_true = 5 # Number of true nonzeros per task
K = 5 # Number of tasks
n = 200 # Number of samples per task
n_test = n # Number of test samples per task
q = 1 # Support heterogeneity
snr = 40 # SNR in dB
M = 1 # Big-M for Outer Approximation


# Draw X and test X
Z = zeros(p,K)
task = zeros(1,n*K)

X = rand(dist, n*K)'
X_test = rand(dist, n_test*K)'
B = zeros(p,K)
y = zeros(n*K,1)
y_test = zeros(n_test*K,1)
X = convert(Matrix,X)
# Draw the data
for k = 1:K
    Z[1:s_true-q,k] = ones(size(Z[1:s_true-q,k]))
    Z[s_true+1+q*(k-1):s_true+1+q*(k)-1,k] = ones(q,1)
    X_task = X[(k-1)*n+1:k*n,:]
    task[(k-1)*n+1:k*n] = k*ones(n,1)
    beta = rand(p).*Z[:,k]
    beta_task = beta/norm(beta)
    noise = randn(n)
    noise_sigma = (norm(X_task*beta_task))/(10^(snr/20)*norm(noise))
    noise = noise*noise_sigma
    y_task = X_task*beta_task + noise
    b = mean(y_task)
    y_task = y_task - b*ones(size(y_task))
    X_task_test = X_test[(k-1)*n_test+1:k*n_test,:]
    noise = randn(n_test)
    noise = noise*noise_sigma
    y_task_test = X_task_test*beta_task + noise
    y_task_test = y_task_test - b*ones(size(y_task_test))
    y_test[(k-1)*n_test+1:k*n_test,:] = y_task_test
    y[(k-1)*n+1:k*n,:] = y_task
    B[:,k] = beta_task
end

# Tuning grid
R"grid<-data.frame(s=rep(c(4,5,6),each=7), lambda_z=rep(c(.01, 0.025, 0.05, 0.025, .1, 0.25, 0.5),3),lambda_2=rep(0,21), lambda_1=rep(1e-6,21))"
# Cross-validation
R"tn <- cv.smtl( y=$y, X=$X, study=$task, nfolds=5, grid=grid, lambda_1=FALSE)"
# Fit the model using the approximate solvers
R"mod <- sMTL::smtl(y = $y,X = $X, study = $task, s = tn$best.1se$s, commonSupp = FALSE, LocSrch_maxIter = 10,
lambda_1 = tn$best.1se$lambda_1,
lambda_2 = 0,
lambda_z = tn$best.1se$lambda_z)"

# Get the cross-validated hyper-parameters and the approximate solution
R"beta<-mod$beta"
R"lambda_z <- tn$best.1se$lambda_z"
R"lambda_1 <- tn$best.1se$lambda_1"
R"s <- tn$best.1se$s"

B0 = @rget beta
B0 = B0[2:end,:]
lambda_z = @rget lambda_z
lambda_1 = @rget lambda_1
s_est = @rget s


# Initial Z for the outer approximation, available from the approximate solver
Z_init = zeros(p,K)
idx = findall(x -> x.>1e-5, abs.(B0))
Z_init[idx] = ones(size(Z_init[idx]))


# Run the outer approximation solver
B_OA, GAP, T_OA = OA(y, X, K, s_est, lambda_z, lambda_1, Z_init,  M)

# The results
println("------")
println("Apprxomate Solver Objective:")
println(objective(B0, y, X, K, s_est, lambda_z, lambda_1))
println("Exact Solver Objective:")
println(objective(B_OA, y, X, K, s_est, lambda_z, lambda_1))
println("------")
println("Apprxomate Solver Test RMSE:")
println(test_error(X_test, y_test, B0, K, n_test))
println("Exact Solver Test RMSE:")
println(test_error(X_test, y_test, B_OA, K, n_test))
