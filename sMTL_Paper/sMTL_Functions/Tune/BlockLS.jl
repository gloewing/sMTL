using LinearAlgebra

# Local Search for exact common support
# beta must have exactly rho nonzero rows.
function BlockLS(; X::Matrix,
                    y,
                    rho,
                    B,
                    lambda1,
                    lambda2,
                    indxList,
                    K,
                    nVec,
                    p,
                    maxIter = 50)

    s = rho; #
    S = zeros(p, p ,K)
    Aq = zeros(s, p-s)
    Bq = zeros(s, p-s)
    Cq = zeros(s, p-s)

    B_bar = sum( B[2:end, :], dims = 2) / K # betaBar if needed (do not include intercept)
    B_bar = B_bar[:]

    for k = 1:K
        S[:,:,k] = X[ indxList[k], 2:end]' * X[ indxList[k], 2:end]
    end

    iter = 1
    flag = 0
    cost_old = 0
    delta_cost = 0

    obj = 0
    r0 = [Vector{Any}() for i in 1:K]; # list of vectors of residuals # zeros(n,K)

    for k = 1:K
        r0[k] = y[indxList[k]] - X[indxList[k], :] * B[:,k]
        obj = obj + r0[k]' * r0[k] / (2 * nVec[k])

        if (lambda1 > 0)
            obj = obj + lambda1 * B[2:end, k]' * B[2:end, k] / 2
        end

        if (lambda2 > 0)
            B_bar = sum( B[2:end, :], dims = 2) / K # betaBar if needed (do not include intercept)
            B_bar = B_bar[:]
            obj = obj + lambda2 * (B[2:end, k] - B_bar)' * (B[2:end, k] - B_bar) / 2
        end

    end

    betaq = zeros(s, p-s, K)
    costq = zeros(s, p-s, K)

    while(iter <= maxIter)

        sumB = sum( abs.( B[2:end, :] ), dims=2 )
        sumB = sumB[:]
        idx1 = findall(x -> x .> 1e-5, sumB)
        idx2 = findall(x -> x .<= 1e-5, sumB)
        B[1 .+ idx2, :] = zeros(length(idx2),K) ############################################################################################# NEW

        flag = 0
        if length(idx1) > s
            
            println("The initial solution is not feasible. Aborting local search.")
            return B

        elseif length(idx1) < s
            # if need to add in a variable (support is less than it should be)

            stilde = length(idx1)
            Aq2 = zeros(p-stilde)
            Bq2 = zeros(p-stilde)
            Cq2 = zeros(p-stilde)

            betaq2 = zeros(p-stilde, K)
            costq2 = zeros(p-stilde, K)


            for k = 1:K
                X0 = X[ indxList[k], 2:end] # do not include intercept
                y0 = y[ indxList[k] ]
                S0 = S[:,:,k]
                beta = B[2:end, k] # ask Kayhan -- residuals include contribution of interept but other terms below do now
                r = r0[k] # residuals include contribution of intercept



                Aq2 = diag(S0[idx2,idx2])./ nVec[k] + lambda1*ones(p-stilde)
                Bq2 = -2*r'*X0[:,idx2] ./ nVec[k]
                if (lambda1 > 0)
                    Aq2 = Aq2 + lambda1*ones(p-stilde)
                end

                if (lambda2 > 0)
                    Aq2 = Aq2 + lambda2*((K-1)^2/K^2+(K-1)/K^2)*ones(p-stilde)

                    beta_cut = B[1 .+ idx2,:]
                    B_bar_cut = B_bar[idx2]

                    Bq2 = Bq2 + (2*lambda2*(B_bar_cut*(K-1) - sum(beta_cut,dims=2))/K - 2*lambda2*(K-1)*B_bar_cut/K)'

                end

                betaq2[:,k] = -Bq2 ./ (2 * Aq2) # find argmin of each decision variable as univariate problem (thats why doing elementwise division)
                costq2[:,k] = Cq2 - (Bq2.^2) ./ (4 * Aq2) # cost function evaluated at argmin

            end

            cost_agg = sum(costq2, dims = 2) # sum over the costs across studies (since we are assuming common support) to get s x (p-s) matrix
            minc = minimum(cost_agg[:]) # find lowest cost

            if (minc < 0)
                # iterate until there is no improvement (i.e., minc >= 0) in objective from making swaps

                idx_best = findfirst(cost_agg .== minc)
                B[ 1 .+ idx2[ idx_best ], :] = betaq[ idx_best,:] # set the zero entry for all studies to the coefficient we swtiched on (i.e., the one with the lowest cost)
                obj = obj + minc / 2 # scale by 1/2 since we include this in the objective and we didn't above

                if (lambda2 > 0)
                    # update beta bar
                    B_bar = sum( B[2:end, :], dims = 2) / K # betaBar if have needed (do not include intercept)
                    B_bar = B_bar[:]
                end

                for kk = 1:K
                    r0[kk] = y[indxList[kk]] - X[indxList[kk], :] * B[:,kk] # residual for kth study
                end

                flag = 1
            end



        else
            for k = 1:K

                X0 = X[ indxList[k], 2:end] # do not include intercept
                y0 = y[ indxList[k] ]
                S0 = S[:,:,k]
                beta = B[2:end, k] # ask Kayhan -- residuals include contribution of interept but other terms below do now
                r = r0[k] # residuals include contribution of intercept

                # constant matrix in quadratic program (QP)
                Cq = diag( S0[idx1, idx1] ) .* beta[idx1].^2 * ones(p-s)' ./ nVec[k] +
                            2 * X0[:,idx1]' * r .* beta[idx1] * ones(p-s)' ./ nVec[k]

                # matrix associated with quadratic term in QP
                Aq = ones(s) * diag(S0[idx2, idx2])' ./ nVec[k]

                # matrix associated with linear term in QP
                Bq = (  -2 * ones(s) * r' * X0[:,idx2] - 2 * S0[idx1,idx2] .* (beta[idx1] * ones(p-s)')   ) ./ nVec[k]

                if (lambda1 > 0)
                    # if ridge penalty
                    Cq = Cq - lambda1 * beta[idx1].^2 * ones(p-s)'
                    Aq = Aq + lambda1 * ones(s, p-s)
                end

                if (lambda2 > 0)
                    # beta - betaBar penalty

                    Aq = Aq + lambda2 * ( (K-1)^2 / K^2 + (K-1) / K^2 ) * ones(s, p-s)

                    beta_cut = B[idx2 .+ 1, :] # +1 for intercept
                    B_bar_cut = B_bar[idx2]

                    Bq = Bq + ones(s) *
                        (2 * lambda2 * ( B_bar_cut * (K-1) - sum(beta_cut, dims=2) ) / K - 2 * lambda2 * (K-1) *  B_bar_cut / K )'

                    b_bar_temp = B_bar[idx1]
                    B_temp = B[idx1 .+ 1, :] # add one for intercept
                    delta_cost = -lambda2 * sum( (B_temp - b_bar_temp * ones(K)').^2, dims=2) * ones( length(idx2) )'

                    b_bar_temp = B_bar[idx1] - B[idx1 .+ 1, k] / K # add one for intercept
                    B_temp = B[idx1 .+ 1, :] # add one for intercept
                    B_temp[:,k] = zeros( length(idx1) )
                    delta_cost = delta_cost + lambda2 * sum( (B_temp - b_bar_temp * ones(K)').^2, dims=2) * ones(length(idx2))'

                    Cq = Cq + delta_cost
                end

                # solve s x (p-s) individual univariate QPs
                betaq[:,:,k] = -Bq ./ (2 * Aq) # find argmin of each decision variable as univariate problem (thats why doing elementwise division)
                costq[:,:,k] = Cq - (Bq.^2) ./ (4 * Aq) # cost function evaluated at argmin

            end

            cost_agg = sum(costq, dims = 3) # sum over the costs across studies (since we are assuming common support) to get s x (p-s) matrix
            minc = minimum(cost_agg[:]) # find lowest cost

            if (minc < 0)
                # iterate until there is no improvement (i.e., minc >= 0) in objective from making swaps

                idx_best = findfirst(cost_agg .== minc)
                B[ 1 .+ idx1[ idx_best[1] ], :] = zeros(K) # zero out all K coefficients (for all studies) corresponding to the coefficient we switched off
                B[ 1 .+ idx2[ idx_best[2] ], :] = betaq[ idx_best[1], idx_best[2],:] # set the zero entry for all studies to the coefficient we swtiched on (i.e., the one with the lowest cost)
                obj = obj + minc / 2 # scale by 1/2 since we include this in the objective and we didn't above

                if (lambda2 > 0)
                    # update beta bar
                    B_bar = sum( B[2:end, :], dims = 2) / K # betaBar if have needed (do not include intercept)
                    B_bar = B_bar[:]
                end

                for kk = 1:K
                    r0[kk] = y[indxList[kk]] - X[indxList[kk], :] * B[:,kk] # residual for kth study
                end

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
