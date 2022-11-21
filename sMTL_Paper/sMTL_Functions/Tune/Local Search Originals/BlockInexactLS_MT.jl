using LinearAlgebra

# Local Search for exact non-common support
# beta must have exactly rho nonzero rows.
function BlockInexactLS_MT(; X::Matrix,
                    y,
                    s,
                    beta,
                    lambda1,
                    lambda2,
                    lambda_z,
                    #indxList,
                    K,
                    n,
                    p,
                    maxIter = 50)

    # p is number of covariates not including intercept
    #s = rho; #
    # n, p, K = size(X)
    B = copy(beta)
    #beta = 0; # save memory
    #S = zeros(p, p ,K)
    Aq = zeros(s, p-s)
    Bq = zeros(s, p-s)
    Cq = zeros(s, p-s)
    z = zeros(p, K)

    S = X[ :, 2:end]' * X[ :, 2:end]
    # for k = 1:K
    #     S[:,:,k] = X[ indxList[k], 2:end]' * X[ indxList[k], 2:end]
    # end

    idx1 = findall(x -> x.> 1e-9, abs.(B[2:end, :]))
    z[idx1] = ones(size(idx1))

    iter = 1
    flag = 0
    cost_old = 0
    delta_cost = 0

    z_bar = sum(z, dims=2) / K
    z_bar = z_bar[:]
    B_bar = sum(B[2:end, :], dims=2) / K # betaBar if needed (do not include intercept)
    B_bar = B_bar[:]

    obj = 0
    r0 = zeros(n, K) #[ zeros( nVec[i] ) for i in 1:K]; # list of vectors of residuals # zeros(n,K)

    for k = 1:K
        r0[:, k] = y[ :, k ] - X * B[:,k]
        obj = obj + r0[:, k]' * r0[:, k] / (2 * n)

        if lambda1 > 0
            obj = obj + lambda1 * B[2:end, k]' * B[2:end, k] / 2
        end

        if lambda2 > 0
            obj = obj + lambda2 * (B[2:end, k] - B_bar)' * (B[2:end, k] - B_bar) / 2
        end

        if lambda_z > 0
            obj = obj + lambda_z * (z[:,k] - z_bar)' * (z[:,k] - z_bar) / 2
        end
    end

    betaq = zeros(s, p-s, K)
    costq = zeros(s, p-s, K)

    if lambda2 > 0
        B_bar = sum( B[2:end, :], dims = 2) / K # betaBar if have needed (do not include intercept)
        B_bar = B_bar[:]
    end

    while(iter <= maxIter)

        # sumB = sum( abs.( B[2:end, :] ), dims=2 )
        # sumB = sumB[:]

        flag = 0
        X0 = X[ :, 2:end] # do not include intercept

        for k = 1:K
            idx1 = findall(x -> x.> 0.5, abs.(z[:,k]) )
            idx2 = findall(x -> x.<= 0.5, abs.(z[:,k]) )

            # X0 = X[ :, 2:end] # do not include intercept
            y0 = y[ :, k ]
            #S0 = S[:,:,k]
            beta = B[2:end, k] # ask Kayhan -- residuals include contribution of interept but other terms below do now
            r = r0[:, k] # residuals include contribution of intercept

            # constant matrix in quadratic program (QP)
            Cq = diag( S[idx1, idx1] ) .* beta[idx1].^2 * ones(p-s)' ./ n +
                        2 * X0[:,idx1]' * r .* beta[idx1] * ones(p-s)' ./ n

            # matrix associated with quadratic term in QP
            Aq = ones(s) * diag(S[idx2, idx2])' ./ n

            # matrix associated with linear term in QP
            Bq = (-2 * ones(s) * r' * X0[:, idx2] - 2 * S[idx1,idx2] .* (beta[idx1] * ones(p-s)')) ./ n

            if (lambda1 > 0)
                # if ridge penalty
                Cq = Cq - lambda1 * beta[idx1].^2 * ones(p-s)'
                Aq = Aq + lambda1 * ones(s, p-s)
            end

            if (lambda2 > 0)
                # beta - betaBar penalty

                Aq = Aq + lambda2 * ( (K-1)^2 / K^2 + (K-1) / K^2 ) * ones(s, p-s) # DOESNT MATCH -- ASK KAYHAN

                beta_cut = B[idx2 .+ 1, :] # +1 for intercept
                B_bar_cut = B_bar[idx2]

                 # DOESNT MATCH -- ASK KAYHAN
                Bq = Bq + ones(s) *
                    (2 * lambda2 * ( B_bar_cut * (K-1) - sum(beta_cut, dims=2) ) / K - 2 * lambda2 * (K-1) *  B_bar_cut / K )'
                    # I got the equivalent of (2 * lambda2 * (K-1)/K) * ( B_cut - B_bar_cut)
                # cost of beta - betaBar penalty BEFORE swap (only including fixed terms that do not depend on decision variable)
                b_bar_temp = B_bar[idx1]
                B_temp = B[idx1 .+ 1, :] # add one for intercept
                delta_cost = -lambda2 * sum( (B_temp - b_bar_temp * ones(K)').^2, dims=2) * ones( length(idx2) )'

                # cost of beta - betaBar penalty AFTER swap (only including fixed terms that do not depend on decision variable)
                b_bar_temp = B_bar[idx1] - B[idx1 .+ 1, k] / K # add one for intercept
                B_temp = B[idx1 .+ 1, :] # add one for intercept
                B_temp[:,k] = zeros( length(idx1) )
                delta_cost = delta_cost + lambda2 * sum( (B_temp - b_bar_temp * ones(K)').^2, dims=2) * ones(length(idx2))'

                 # ASK KAYHAN to confirm that we are just taking difference of cost before and cost after
                Cq = Cq + delta_cost
            end

            # solve s x (p-s) individual univariate QPs
            betaq = -Bq ./ (2 * Aq) # find argmin of each decision variable as univariate problem (thats why doing elementwise division)
            costq = Cq - (Bq.^2) ./ (4 * Aq) # cost function evaluated at argmin

            if (lambda_z > 0)

                z_bar_temp = z_bar[idx2]
                z_temp = z[idx2,:]
                delta_cost = -ones(length(idx1)) * lambda_z * sum( (z_temp - z_bar_temp * ones(K)').^2, dims=2)'

                z_bar_temp = z_bar[idx2] + ones(length(idx2)) / K
                z_temp = z[idx2,:]
                z_temp[:,k] = ones(length(idx2))
                delta_cost = delta_cost + ones(length(idx1)) * lambda_z * sum( (z_temp - z_bar_temp * ones(K)').^2, dims=2)'

                z_bar_temp = z_bar[idx1]
                z_temp = z[idx1,:]
                delta_cost = delta_cost - lambda_z * sum( (z_temp - z_bar_temp * ones(K)').^2, dims=2) * ones(length(idx2))'

                z_bar_temp = z_bar[idx1] - ones(length(idx1)) / K
                z_temp = z[idx1,:]
                z_temp[:,k] = zeros(length(idx1))
                delta_cost = delta_cost + lambda_z * sum( (z_temp - z_bar_temp * ones(K)').^2, dims=2) * ones(length(idx2))'

                costq = costq + delta_cost

            end

            minc = minimum(costq[:])

            if (minc < 0)
                idx_best = findfirst(costq .== minc)
                b_temp = B[1 .+ idx1[idx_best[1]], k]
                B[1 .+ idx1[idx_best[1]], k] = 0 # add 1 for intercept
                B[1 .+ idx2[idx_best[2]], k] = betaq[idx_best]
                z[idx1[idx_best[1]], k] = 0
                z[idx2[idx_best[2]], k] = 1
                B[2:end, k] = B[2:end, k] .* z[:,k]
                z_bar[idx1[idx_best[1]]] = z_bar[idx1[idx_best[1]]] - 1 / K
                z_bar[idx2[idx_best[2]]] = z_bar[idx2[idx_best[2]]] + 1 / K
                B_bar[idx1[idx_best[1]]] = B_bar[idx1[idx_best[1]]] - b_temp / K
                B_bar[idx2[idx_best[2]]] = B_bar[idx2[idx_best[2]]] + betaq[idx_best] / K

                obj = obj + minc / 2
                r0[:, k] = y[ :, k ] - X * B[:,k] # residual for kth study

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
