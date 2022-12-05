# Optimization Code

#### Active set
# sparse regression with IHT
function BlockComIHT_inexactAS_opt(; X::Matrix,
                    y,
                    rho,
                    indxList,
                    B,
                    K,
                    L,
                    idx,
                    nVec,
                    lambda1,
                    lambda2,
                    lambda_z,
                    p,
                    maxIter_in = 1000,
                    maxIter_out = 1000
                    )

    eig = L # initialize for active set
    L = L + lambda1 + lambda2 # L for complete dataset for outer loop
    idxFlag = true # indicates whether indices have changed and whether we nee to recalculate eigenvalues

    ncol = p + 1

    # initialize
    z = zeros(p, K)
    B_bar = zeros(ncol)
    z_bar = zeros(p)
    Z_bar = zeros(p, K)

    r = [zeros( nVec[i] ) for i in 1:K]; # list of vectors of indices of studies
    g = zeros(ncol, K);

    obj = 1e20;
    iter_in = 1
    iter_out = 1

    B_summed = sum( B.^2, dims = 2)
    B_summed = B_summed[:]

    # # projection step

    idxInt = Int.( cat(1, idx .+ 1, dims = 1) ) # add one for intercept
    z[idx] = ones(size(idx))
################################################################################

# outer while loop
    while (iter_out <= maxIter_out)

        objPrev = obj
        BPrev = copy(B)

        if idxFlag
            # if active set has changed (or first round) recalculate svd
            L_active = 0
            for k = 1:K
                a2 = tsvd( X[ indxList[k], idxInt] )[2][1]
                if (a2 > L_active)
                    L_active = a2
                end
            end

            L_active = L_active^2 * sqrt(K) / maximum(nVec) + lambda1 + lambda2
            #println(L_active)
            idxFlag = false # default to see if we need to recalculate eigenvalues

        end

        B_bar_active = zeros(length(idxInt))
        z_bar_active = zeros(length(idx))
        r_active = [Vector{Any}() for i in 1:K]
        g_active = zeros(length(idxInt) ,K)
        B_active = B[idxInt,:]
        z_active = z[idx, :]

        # inner while loop
########################################################
        while (iter_in <= maxIter_in)

            objPrev = obj
            B_bar_active = sum(B_active[2:end, :], dims=2) / K
            z_bar_active = sum(z_active, dims=2) / K
            z_bar_active = z_bar_active[:]
            B_bar_active = B_bar_active[:]
            f = 0

            for k = 1:K
                r_active[k] = X[ indxList[k], idxInt] * B_active[:,k] - y[ indxList[k] ]
                g_active[:, k] = (1 / nVec[k]) * X[ indxList[k], idxInt]' * r_active[k] # gradient update for whole vector
                g_active[2:end, k] = g_active[2:end, k] + lambda1 * B_active[2:end, k] + lambda2 * (B_active[2:end, k] - B_bar_active) # gradient update for non-intercept
                f = f + r_active[k]' * r_active[k] / (2 * nVec[k]) +
                        lambda1 * B_active[2:end, k]' * B_active[2:end, k] / 2 +
                        lambda2 * (B_active[2:end, k] - B_bar_active)' * (B_active[2:end, k] - B_bar_active) / 2 +
                        lambda_z * (z[idx, k] - z_bar_active)' * (z[idx,k] - z_bar_active) / 2
            end

            obj = f
            # println(obj)

            if (abs(obj - objPrev)/objPrev < 1e-6)
                break
            end

            iter_in = iter_in + 1
            Z_bar_active = z_bar_active * ones(K)'
            B_temp_active = B_active - 0.5 * (1/L) * g_active
            z_active = zeros(length(idx), K)
            idx_active = findall(x-> x.>1e-9, abs.(B_temp_active[2:end,:]) )
            z_active[idx_active] = ones(size(idx_active))
            cost_active = lambda_z / L_active * ( Z_bar_active.^2  - (z_active-Z_bar_active).^2 ) # part of cost vector, rest in for loop below
            z_active = zeros(length(idx), K)

            for k=1:K
                costAct_k = B_temp_active[2:end, k].^2 + nVec[k] * cost_active[:,k] # nVec[k] * cost_active[:,k] **** removed the nVec[k] on 10/7/21  # scale by nVec since our objective is altered
                idx1 = partialsortperm(costAct_k, 1:rho, rev=true)
                z_active[idx1, k] = ones(size(idx1))
            end


            B_active[2:end,:] = B_temp_active[2:end,:] .* z_active
            B_active[1,:] = B_temp_active[1,:] # update intercept

            B[idxInt,:] = B_active[:,:]
            z[idx,:] = z_active[:,:]
        end
        # end inner while loop
########################################################

        B_bar = sum(B[2:end, :], dims=2) / K
        z_bar = sum(z, dims=2) / K
        z_bar = z_bar[:]
        B_bar = B_bar[:]
        f = 0

        for k = 1:K

            # residual for kth study
            r[k] = X[ indxList[k], :] * B[:,k] - y[ indxList[k] ]

            # gradient for kth study
            g[:, k] = (1 / nVec[k]) * X[ indxList[k], :]' * r[k] # gradient update for whole vector
            g[2:end, k] = g[2:end, k] + lambda1 * B[2:end, k] + # gradient update for non-intercept terms
                                        lambda2 * (B[2:end, k] - B_bar)

            # objective for kth study
            f = f + r[k]' * r[k] / (2 * nVec[k]) +
                        lambda1 / 2 * B[2:end, k]' * B[2:end, k] +
                        lambda2 / 2 * (B[2:end, k] - B_bar)' * (B[2:end, k] - B_bar) +
                        lambda_z / 2 * (z[:,k] - z_bar)' * (z[:, k] - z_bar)
        end

        obj = f

        iter_out = iter_out + 1
        Z_bar = z_bar * ones(K)'
        B_temp = B - 0.5 * (1/L) * g

        z = zeros(p, K)
        idx1 = findall(x-> x.>1e-9, abs.(B_temp[2:end,:]) )
        z[idx1] = ones(size(idx1))
        cost =  lambda_z / L * (Z_bar.^2 - (z-Z_bar).^2) # scale by nVec (below) since our objective is altered
        z = zeros(p, K)
        flag = 0

        for k=1:K
            cost_k = B_temp[2:end, k].^2 + nVec[k] * cost[:,k] # scale lambda_z by nVec
            idx1 = partialsortperm(cost_k,
                                  1:rho,
                                  rev=true)
            z[idx1, k] = ones(size(idx1))

            for jj = 1:length(idx1)
                if ( sum(idx .== idx1[jj]) == 0 )
                    idx = [ idx; idx1[jj] ]
                    idxInt = [ idxInt; idx1[jj] + 1] # for intercept version
                    flag = 1
                    idxFlag = true # set to recalculate eigenvalues
                end
            end

        end

        B[2:end,:] = B_temp[2:end,:] .* z
        B[1,:] = B_temp[1,:] # update intercept

        if (flag == 0)
           break
        end

    end
    # end outer while loop
######################################################################

    return B

end
