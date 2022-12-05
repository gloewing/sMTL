using LinearAlgebra

# Local Search for exact common support
# beta must have exactly rho nonzero rows.
function SingleLS(; X::Matrix,
    y,
    rho,
    beta,
    lambda,
    n,
    p,
    maxIter = 50)

    s = rho; #
    B = copy(beta)
    beta = 0; # save memory
    S = zeros(p, p)
    Aq = zeros(s, p-s)
    Bq = zeros(s, p-s)
    Cq = zeros(s, p-s)

    S = X[:, 2:end]' * X[:, 2:end]

    iter = 1
    flag = 0
    cost_old = 0
    delta_cost = 0

    obj = 0
    r = zeros(n);  # list of vectors of residuals # zeros(n,K)

    r = y - X * B
    obj = obj + r' * r / (2 * n)

    if (lambda > 0)
        obj = obj + lambda * B[2:end]' * B[2:end] / 2
    end

    betaq = zeros(s, p-s)
    costq = zeros(s, p-s)

    while (iter <= maxIter)

        idx1 = findall(x -> x .> 1e-5, abs.( B[2:end] ) )
        idx2 = findall(x -> x .<= 1e-5, abs.( B[2:end] ) )
        B[1 .+ idx2] = zeros(length(idx2))

        flag = 0
        if length(idx1) > s

            println("The initial solution is not feasible. Aborting local search.")
            return B

        elseif length(idx1) < s
            stilde = length(idx1)
            Aq2 = zeros(p-stilde)
            Bq2 = zeros(p-stilde)
            Cq2 = zeros(p-stilde)

            betaq2 = zeros(p-stilde)
            costq2 = zeros(p-stilde)

            Aq2 = diag(S[idx2,idx2])./ n + lambda*ones(p-stilde)
            Bq2 = -2*r'*X[:,1 .+ idx2] ./ n



            betaq2 = -Bq2 ./ (2 * Aq2) # find argmin of each decision variable as univariate problem (thats why doing elementwise division)
            costq2 = Cq2 - (Bq2.^2) ./ (4 * Aq2) # cost function evaluated at argmin



            minc = minimum(costq2[:]) # find lowest cost

            if (minc < 0)
                # iterate until there is no improvement (i.e., minc >= 0) in objective from making swaps

                idx_best = findfirst(costq2 .== minc)
                B[ 1 .+ idx2[ idx_best ]] = betaq[ idx_best] # set the zero entry for all studies to the coefficient we swtiched on (i.e., the one with the lowest cost)
                obj = obj + minc / 2 # scale by 1/2 since we include this in the objective and we didn't above

                r = y - X * B # residual for kth study

                flag = 1
            end

        else
            # constant matrix in quadratic program (QP)
            Cq = diag( S[idx1, idx1] ) .* B[1 .+ idx1].^2 * ones(p-s)' ./ n +
            2 * X[:, 1 .+ idx1]' * r .* B[1 .+ idx1] * ones(p-s)' ./ n

            # matrix associated with quadratic term in QP
            Aq = ones(s) * diag(S[idx2, idx2])' ./ n

            # matrix associated with linear term in QP
            Bq = (-2 * ones(s) * r' * X[:, 1 .+ idx2] - 2 * S[idx1, idx2] .* (B[1 .+ idx1] * ones(p-s)')) ./ n

            if (lambda > 0)
                # if ridge penalty
                Cq = Cq - lambda * B[1 .+ idx1].^2 * ones(p-s)'
                Aq = Aq + lambda * ones(s, p-s)
            end

            # solve s x (p-s) individual univariate QPs
            betaq = -Bq ./ (2 * Aq) # find argmin of each decision variable as univariate problem (thats why doing elementwise division)
            costq = Cq - (Bq.^2) ./ (4 * Aq) # cost function evaluated at argmin

            minc = minimum(costq[:]) # find lowest cost

            if (minc < 0)
                # iterate until there is no improvement (i.e., minc >= 0) in objective from making swaps

                idx_best = findfirst(costq .== minc)
                B[ 1 .+ idx1[ idx_best[1] ] ] = 0 # zero out all K coefficients (for all studies) corresponding to the coefficient we switched off
                B[ 1 .+ idx2[ idx_best[2] ] ] = betaq[ idx_best[1], idx_best[2] ] # set the zero entry for all studies to the coefficient we swtiched on (i.e., the one with the lowest cost)
                obj = obj + minc / 2 # scale by 1/2 since we include this in the objective and we didn't above

                r = y - X * B # residual update
                flag = 1

            end
        end

        iter = iter + 1

        if(flag == 0 )
            break
        end

    end

    return B

end
