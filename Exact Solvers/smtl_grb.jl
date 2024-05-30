using Gurobi, JuMP

function smtl_grb(y, X, K, s, lambda_z, lambda2,  M, Z0=nothing,  eps = 0.05, max_time = 400000, verbose = 1)
# The off-the-shelf Gurobi exact solver.
# y: A vector of all observations from all tasks. It is assumed to be of length nK, 
#     with observations 1:n belonging to task 1, n+1:2n to task 2 and so on.
# X: A matrix of all data points from all tasks. It is assumed to be of size nK*p
#     with rows 1:n belonging to task 1, n+1:2n to task 2 and so on.
# K: Number of tasks.
# s: Sparsity level.
# lambda_z: Regression coefficient for Zbar penalty.
# lambda2: Regression coefficient for Ridge penalty.
# Z0: Initial binary solution. Assumed to be of size p*K.
# M: Big-M constant.
# eps: MIP gap tolerance.
# max_time: Maximum runtime.

    t0 = now()
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "NonConvex", 2)
    set_attribute(model, "OutputFlag", verbose)
    set_optimizer_attribute(model, "TimeLimit", max_time/1000)
    set_optimizer_attribute(model, "MIPGap", eps)
    N, p = size(X)
    n = Int(N/K)

    y_tensor = zeros(n,K)
    X_tensor = zeros(n,p,K)

    for k in 1:K

        task_idx = (k-1)*n + 1:k*n

        X_tensor[:,:,k] = X[task_idx,:]
        y_tensor[:,k] = y[task_idx]
    end


    @variable(model, Z[1:p,1:K],Bin)
    set_start_value.(Z, Z0)
    @variable(model, Z_bar[1:p])
    @variable(model, BETA[1:p, 1:K])
    @variable(model, XI[1:n,1:K])

    @constraint(model, [k in 1:K], sum(Z[i,k] for i=1:p) <= s)
    @constraint(model, BETA + M*Z .>= 0)
	@constraint(model, -BETA + M*Z .>= 0)
    @constraint(model, [k in 1:K], XI[:,k] -y_tensor[:,k] + X_tensor[:,:,k]*BETA[:,k] .== 0)

    sense = MOI.MIN_SENSE
	@objective(model, sense, sum(   sum((XI[ii,kk] )^2 for kk = 1:K) for ii = 1:n)/n/2 +  
        lambda_z*sum(   sum((Z[ii,kk] - Z_bar[ii] )^2 for kk = 1:K) for ii = 1:p)/2 + 
        lambda2*sum(   sum((BETA[ii,kk] )^2 for kk = 1:K) for ii = 1:p)/2 )
	status=optimize!(model)

    B = value.(BETA)

    GAP = MathOptInterface.get(model, MathOptInterface.RelativeGap())
    t2 = now()
    diff_t = Dates.value(convert(Dates.Millisecond, t2-t0))
    
    
    return B, GAP, diff_t/1000

    

    
end
