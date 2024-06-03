function OA(y, X, K, s, lambda_z, lambda2, Z0, M, verbose = 1, maxIter =100, eps = 0.05, max_time = 3*60 )

# The Outer Approximation exact solver.
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


    N,p = size(X)
    n = Int(N/K)

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "NonConvex", 2)
    set_attribute(model, "OutputFlag", 0 )
    set_attribute(model, "TimeLimit", 60 )

    @variable(model, Z[1:p, 1:K],Bin)
    @variable(model, Zbar[1:p])
    @variable(model, eta[1:K])

    
    @constraint(model, [kk in 1:K], sum(Z[ii,kk] for ii=1:p) <= s)
    @constraint(model, [ii in 1:p], sum(Z[ii,kk] for kk=1:K)/K - Zbar[ii] == 0)


    sense = MOI.MIN_SENSE
    @objective(model, sense, sum(eta[kk] for kk in 1:K)  
       + lambda_z*sum( sum( (Z[ii,kk]-Zbar[ii])^2 for kk in 1:K) for ii in 1:p)/2)


    iter = 1

    Best_UB = 1e10
    Best_B = zeros(p,K)
    GAP = 10

    t0 = now()
    diff_t = 0
    LB = -1e10
    while iter <= maxIter && GAP >= eps && diff_t <= max_time*1000

        ZBAR = sum(Z0, dims=2)/K
        ZBAR = ZBAR*ones(1,K)

        UB = lambda_z*norm(Z0 - ZBAR)^2/2
        B = zeros(p,K)

        for k in 1:K

            task_idx = (k-1)*n + 1:k*n
            X_task = X[task_idx,:]
            y_task = y[task_idx]
            idx = findall(x -> x.>0.1, Z0[:,k])
            r , beta, OBJ = subGrad(X_task[:,idx], y_task , lambda2, M)
            B[idx,k] = beta
            UB = UB + OBJ
            Gamma = abs.((-1/n)*X_task'*r - lambda2*B[:,k])
            G = -M*Gamma
            @constraint(model, eta[k] - (OBJ + sum( G[ii]*(Z[ii,k]-Z0[ii,k]) for ii in 1:p)) >= 0)


        end
        if UB < Best_UB
            Best_UB = UB
            Best_B = copy(B)
        end
        
        status=optimize!(model)
        Z0 = value.(Z)

        ETA = value.(eta)


        LB0 = objective_bound(model)
        if LB0 > LB
            LB = LB0
        end
        GAP = ((Best_UB - LB)/Best_UB)
        t2 = now()
		diff_t = Dates.value(convert(Dates.Millisecond, t2-t0))
        iter = iter + 1
        if GAP <= 0
            break
        end
        if verbose == 1
            println("----")
            println("Current Gap:")
            println(GAP)
            println("Current Upper Bound:")
            println(Best_UB)
        end



    end

    

    return Best_B, GAP, diff_t/1000


end


function subGrad(X, y, lambda, M)
    n,s = size(X)
    beta = zeros(s)
    r = -y
    OBJ = r'*r/n/2
    a2 = tsvd(X')[2][1].^2
    
    t = 1/(a2/n + lambda)

    counter = 1

    beta_old = 1000*ones(s)

    while (norm(beta - beta_old)/norm(beta_old) > 1e-2 && counter <= 500)

        beta_old = copy(beta)
        r_old = copy(r)
        OBJ_old = OBJ

        g = (1/n)*X'*r + lambda*beta

        temp = beta - t*g


        neg_idx = findall(x -> x.<0, temp)
        temp = abs.(temp)
        bad_idx = findall(x -> x.>M, temp)
        temp[bad_idx] = M*ones(length(bad_idx))
        temp[neg_idx] = -temp[neg_idx]

        beta = copy(temp)
        r = X*beta - y

        OBJ = r'*r/n/2 + lambda*beta'*beta/2

        counter = counter + 1
        
    end
 

    return r, beta, OBJ
end




function test_error(X_test, y_test, B1, K, n)

    y_test_est = zeros(n*K,1)

    for k = 1:K
        X_task_test = X_test[(k-1)*n+1:k*n,:]
        y_test_task_est = X_task_test*B1[:,k]
        y_test_est[(k-1)*n+1:k*n,:] = y_test_task_est
    end

    return sqrt(norm(y_test- y_test_est)^2/n/K)

end


function objective(B, y, X, K, s, lambda_z, lambda2)
    N,p = size(X)
    n = Int(N/K)
    Z = zeros(p,K)

    idx = findall(x -> x.>1e-5, abs.(B))

    Z[idx] = ones(size(Z[idx]))

    ZBAR = sum(Z, dims=2)/K
    ZBAR = ZBAR*ones(1,K)

    

    obj = lambda_z*norm(Z - ZBAR)^2/2

    for k in 1:K

        task_idx = (k-1)*n + 1:k*n

        X_task = X[task_idx,:]


        y_task = y[task_idx]

        r = y_task - X_task*B[:,k]

        obj = obj + r'*r/2/n
    end

    obj = obj + lambda2*norm(B)^2/2

    return obj

end


function backsolve(B, y, X, K, lambda2)
    N,p = size(X)
    n = Int(N/K)

    for k in 1:K

        idx = findall(x -> x.>1e-5, abs.(B[:,k]))
        task_idx = (k-1)*n + 1:k*n
        X_task = X[task_idx,idx]
        y_task = y[task_idx]
        B[idx,k] = inv(X_task'*X_task/n + lambda2*1* Matrix(I, length(idx), length(idx)))*X_task'*y_task/n


    end
    return B

end