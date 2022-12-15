# Optimization Code
## X: n x p design matrix (feature matrix)
## y: n x K outcome matrix
## rho: Sparsity level (integer)
## B: p*K initial solution
## indxList: list of task row indices
## K: number of tasks
## nVec: vector of task sample sizes
## eig: vector of max eigenvalue of each design matrix
## lambda1>=0: the ridge coefficient
## lambda2>=0: the coefficient value strength sharing coefficient (Bbar penalty)
## lambda_z>=0: the coefficient support strength sharing coefficient (Zbar penalty)
## p: number of features/covariates
## maxIter_in: number of inner coordinate descent iterations (within active sets)
## maxIter_out: number of outer coordinate descent iterations

#### Active set
# sparse regression with IHT
function BlockComIHT_inexactAS_diff_old_opt(; X::Array{Float64,2},
                    y::Array{Float64,1},
                    rho::Integer,
                    B::Array{Float64,2},
                    indxList::Array{Array{Int64,1}},
                    K::Integer,
                    nVec::Array{Int64,1},
                    eigenVec::Array{Float64,1},
                    lambda1::Float64,
                    lambda2::Float64,
                    lambda_z::Float64,
                    p::Integer,
                    maxIter_in::Integer = 1000,
                    maxIter_out::Integer = 1000
                    )::Array{Float64,2}

    idxFlag = true # indicates whether indices have changed and whether we nee to recalculate eigenvalues
    idx = [ Vector{Int64}(undef, rho) for i in 1:K]; # list of vectors of indices of studies

    ncol = p + 1

    # initialize
    B_bar = zeros(ncol)

    iter_in = 1
    iter_out = 1

    B_summed = sum( B.^2, dims = 2)
    B_summed = B_summed[:]

    # projection step # NOTE on change to active set: Comment this out and use idx as argument for _opt function -- can get this for good active set

################################################################################
    for k = 1:K

        L = eigenVec[k]^2 / nVec[k] + lambda1 + lambda2 # L for complete dataset for outer loop

        B_summed = sum( B.^2, dims = 2)
        B_summed = B_summed[:]

        #### in this version "_opt_oldTest.jl" -- the indices are selected based on the warm start of B
        # projection step
        idx[k] = partialsortperm(B[2:end, k].^2, # do not include intercept
                              1:rho,
                              by=abs,
                              rev=true)

        idxInt = Int.( [1; idx[k] .+ 1 ] ) #  add one for intercept # AS_Change - change to be matrix for each K

        r = zeros( nVec[k] ); # list of vectors of indices of studies
        g = zeros(ncol);

        obj = 1e20;
################################################################################
    # outer while loop
        while (iter_out <= maxIter_out)

            objPrev = obj
            BPrev = copy(B)

            if idxFlag
                L_active = tsvd( X[ indxList[k], idxInt ]  )[2][1];

                L_active = L_active^2 / nVec[k] + lambda1 + lambda2

                idxFlag = false # default to see if we need to recalculate eigenvalues

            end

            B_bar_active = zeros(length(idxInt))
            r_active = zeros(nVec[k])
            g_active = zeros(ncol)
            B_active = B[idxInt, k]

            # inner while loop
    ########################################################
            while (iter_in <= maxIter_in)

                objPrev = obj
                B_bar_active = sum(B_active[2:end, :], dims=2) / K
                B_bar_active = B_bar_active[:]
                f = 0

                r_active = X[ indxList[k], idxInt ] * B_active - y[ indxList[k] ]
                g_active = (1 / nVec[k]) * X[ indxList[k], idxInt ]' * r_active # gradient update for whole vector
                g_active[2:end] = g_active[2:end] + lambda1 * B_active[2:end] + lambda2 * (B_active[2:end] - B_bar_active) # gradient update for non-intercept
                f = f + r_active' * r_active / (2 * nVec[k]) +
                        lambda1 * B_active[2:end]' * B_active[2:end] / 2 +
                        lambda2 * (B_active[2:end] - B_bar_active)' * (B_active[2:end] - B_bar_active) / 2


                obj = f

                if (abs(obj - objPrev)/objPrev < 1e-6)
                    break
                end

                iter_in = iter_in + 1
                B_temp_active = B_active - 0.5 * (1/L) * g_active
                idx1 = partialsortperm( abs.(B_temp_active[2:end]), 1:rho, rev=true)

                #
                B_active = zeros( size(idxInt)[1] ) # set to zero for thresholding
                B_active[idx1] = B_temp_active[idx1]
                B_active[1] = B_temp_active[1] # update intercept
                #
                B[ idxInt, k] = B_active[:]
            end
            # end inner while loop
    ########################################################

            B_bar = sum(B[2:end, :], dims=2) / K
            B_bar = B_bar[:]
            f = 0

            # residual for kth study
            r = X[ indxList[k], :] * B[:,k] - y[ indxList[k] ]

            # gradient for kth study
            g = (1 / nVec[k]) * X[ indxList[k], :]' * r # gradient update for whole vector
            g[2:end] = g[2:end] + lambda1 * B[2:end, k] + # gradient update for non-intercept terms
                                        lambda2 * (B[2:end, k] - B_bar)

            # objective for kth study
            f = f + r' * r / (2 * nVec[k]) +
                        lambda1 / 2 * B[2:end, k]' * B[2:end, k] +
                        lambda2 / 2 * (B[2:end, k] - B_bar)' * (B[2:end, k] - B_bar) #+

            obj = f

            iter_out = iter_out + 1
            B_temp = B[:,k] - 0.5 * (1/L) * g

            idx1 = partialsortperm(B_temp[2:end].^2,
                                  1:rho,
                                  rev=true)

            flag = 0

            for jj = 1:length(idx1)
                if ( sum(idx[k] .== idx1[jj]) == 0 )
                    idx[k] = [ idx[k]; idx1[jj] ]
                    idxInt = [ idxInt; idx1[jj] + 1] # for intercept version
                    flag = 1
                    idxFlag = true # set to recalculate eigenvalues
                end
            end

            B[:,k] = zeros(ncol)
            B[idx1 .+ 1, k] = B_temp[idx1 .+ 1]
            B[1, k] = B_temp[1] # update intercept

            if (flag == 0)
               break
            end

        end
        # end outer while loop

    end
    # end for loop across studies for individual active sets
######################################################################

    return B

end
