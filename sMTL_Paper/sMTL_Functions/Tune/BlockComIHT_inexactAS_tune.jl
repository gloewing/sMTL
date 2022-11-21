## Block IHT INEXACT for the NO common support problem with strength sharing (problem (3) in the write-up)
## shared active set support
## b: n*K observation matrix
## A: n*p*K data tensor
## s: Sparsity level (integer)
## x0: p*K initial solution
## lambda1>=0 the ridge coefficient
## lambda2>=0 the strength sharing coefficient
## lambda_z>=0 the strength sharing coefficient for support

using TSVD, Statistics #, LinearAlgebra, Statistics

include("BlockComIHT_inexactAS_opt.jl") # IHT algorithm
include("BlockInexactLS.jl") # local search algorithm
include("l0_IHT_opt.jl") # individual L0 regressions to find active set

# sparse regression with IHT and local search
function BlockComIHT_inexactAS(; X,
                    y,
                    rho,
                    study,
                    beta = 0,
                    scale = true,
                    lambda1 = 0,
                    lambda2 = 0,
                    lambda_z = 0,
                    maxIter = 5000,
                    localIter = 50,
                    maxIter_in = nothing,
                    maxIter_out = nothing,
                    eig = nothing,
                    eigenVec = nothing,
                    idx = nothing,
                    ASmultiplier = 4
                    )

    # rho is number of non-zero coefficient
    # beta is a feasible initial solution
    # scale -- if true then scale covaraites before fitting model
    # maxIter is maximum number of iterations
    # max eigenvalue for Lipschitz constant
    # localIter is a vector as long as lambda1/lambda2 and specifies the number of local search iterations for each lambda
    # idx -- if nothing then fit individual l0 regressions to find it
    # ASmultiplier is number that we multiple rho by to get size of initial active set for first lambda in path
    # eigenVec is a dummy and is never used for this -- only used for diffAS version (different active sets for each study version)

    y = Array(y);
    X = Matrix(X);
    n, p = size(X); # number of covaraites
    beta = Matrix(beta); # initial value
    rho = Int64(rho);
    study = Int.(study);
    K = length( unique(study) ); # number of studies
    indxList = [Vector{Any}() for i in 1:K]; # list of vectors of indices of studies
    nVec = zeros(K); # vector of sample sizes of studies

    if isnothing(maxIter_in)
        maxIter_in = maxIter
    end

    if isnothing(maxIter_out)
        maxIter_out = maxIter
    end

    # scale covariates
    if scale
        # scale covariates like glmnet
        sdMat = ones(p, K); # K x p matrix to save std of covariates of each study
        Ysd = ones(K); # K x 1 matrix to save std of Ys

        for i = 1:K
            indx = findall(x -> x == i, study); # indices of rows for ith study
            indxList[i] = indx; # save indices
            n_k = length(indx); # study k sample size
            nVec[i] = Int(n_k); # save sample size
            Xsd = std(X[indx,:], dims=1) .* (n_k - 1) / n_k; # glmnet style MLE of sd
            sdMat[:,i] = Xsd[1,:]; # save std of covariates of ith study in ith row of matrix
            X[indx,:] .= X[indx,:] ./ Xsd; # standardize ith study's covariates
            # Ysd[i] = std(y[indx]) * (n_k - 1) / n_k; # glmnet style MLE of sd of y_k

        end

        sdMat = vcat(1, sdMat); # add row of ones so standardize intercept by ones
        beta = beta .* sdMat; # scale warm start solution

        # lambda1 = lambda1 / mean(Ysd); # scale tuning parameter for L2 norm by average std of y_k

    else
        # otherwise just make this a vector of ones for multiplication
        # by coefficient estimates later
        # sdMat = ones(p, K); # K x p matrix to save std of covariates of each study

        for i = 1:K
            indxList[i] = findall(x -> x == i, study); # indices of rows for ith study
            indx = indxList[i];
            n_k = length(indx); # study k sample size
            nVec[i] = Int(n_k); # save sample size
        end

    end

    ## intercept
    # add column of 1s for intercept
    X = hcat(ones(n), X);
    ncol = size(X)[2]; # num coefficients (including intercept)
    eigenVec = zeros(K) # store max eigenvalues
    nVec = Int.(nVec)

    # Lipschitz constant
    if isnothing(eig)

        eig = 0;
        # if not provided by user
        for i = 1:K
            # indx = findall(x -> x == i, study); # indices of rows for ith study
            eigenVec[i] = tsvd( X[ indxList[i], :] )[2][1]; # max eigenvalue of X^T X
            if (eigenVec[i] > eig)
                eig = eigenVec[i]
            end
        end

    else
        eig = Float64(eig)

        if isnothing(idx)
            # if idx not given then we need the max singular values for the the individual regressions
            for i = 1:K
                # indx = findall(x -> x == i, study); # indices of rows for ith study
                eigenVec[i] = tsvd( X[ indxList[i], :] )[2][1]; # max eigenvalue of X^T X
            end

        end

    end

    L = eig^2 * sqrt(K) / maximum(nVec) # L without regularization terms (updated in optimization fn below)

    # optimization
    vals = length(lambda1)
    βmat = zeros(ncol, K, vals) # store all of them -- last index is tuning value

    # number of local search iterations
    if length(localIter) < vals
        # if number of local iterations for each tuning not specified just choose first
        localIter = fill( localIter[1], vals ) # just use the first value of local search iterations for each value of lambda
    end

    ####################################################################
    # find initial active set with individual L0 regressions
    ####################################################################
    if isnothing(idx)
        # if no active set inidices provided, use individual sparse regressions to get an initial active set
        # global idx = zeros(p)
        rhoStar = min(rho * ASmultiplier, p); # active set set is bigger than actual rho

        for i = 1:K
            # this alters
            beta[:,i] = L0_iht_opt(X = X[indxList[i], :],
                                    y = y[ indxList[i] ],
                                    rho = rhoStar,
                                    beta = beta[:,i],
                                    L = eigenVec[i].^2 / nVec[i],
                                    n = nVec[i],
                                    maxIter = maxIter,
                                    lambda = lambda1[1], # ridge term for first value in path
                                    p = p
                                    )

            # idx[i] = findall(x-> x.>1e-9, abs.( beta[2:end, i] ) ) # save non-zero elements (not including intercept)
        end

        idx = Int.( partialsortperm( sum( abs.( beta[2:end,:] ), dims = 2)[:], 1:rhoStar, rev=true) )

    end

    ###############
    # optimization
    ###############

    for v = 1:vals
        # use warm starts as previous value for both beta and the active set
        beta = BlockComIHT_inexactAS_opt(X = X,
                                        y = y,
                                        rho = rho,
                                        indxList = indxList,
                                        B = beta,
                                        K = K,
                                        L = L,
                                        nVec = nVec,
                                        maxIter_in = maxIter_in,
                                        maxIter_out = maxIter_out,
                                        lambda1 = lambda1[v],
                                        lambda2 = lambda2[v],
                                        lambda_z = lambda_z[v],
                                        idx = idx,
                                        p = p
                                        )
        ###############
        # local search
        ###############
        if localIter[v] > 0
            # run local search if positive number of local search iterations for this lambda1/lambda2 value
            beta = BlockInexactLS(X = X,
                                y = y,
                                s = rho,
                                beta = beta,
                                lambda1 = lambda1[v],
                                lambda2 = lambda2[v],
                                lambda_z = lambda_z[v],
                                indxList = indxList,
                                K = K,
                                nVec = nVec,
                                p = p,
                                maxIter = localIter[v] )
        end


        if scale
            βmat[:,:,v] = beta ./ sdMat; # rescale by sd
        else
            βmat[:,:,v] = beta
        end
    end

    if vals == 1
        # if only one tuning value, just return a matrix
        return βmat[:,:,1];
    else
        return βmat;
    end

end
# # #
# using CSV, DataFrames
# #
# # # # # #
# dat = CSV.read("/Users/gabeloewinger/Desktop/Research/dat_ms", DataFrame);
# X = Matrix(dat[:,3:end]);
# y = (dat[:,2]);
#
# itrs = 4
# lambda1 = 1 #ones(itrs)
# lambda2 = 0 #ones(itrs)
# lambda_z = 0.01 #ones(itrs) * 0.01
# fit = BlockComIHT_inexactAS(X = X,
#         y = y,
#         study = dat[:,1],
#                     beta =  ones(51, 2),#beta;#
#                     rho = 5,
#                     lambda1 = lambda1,
#                     maxIter = 5000,
#                     lambda2 = lambda2,
#                     lambda_z = lambda_z,
#                     localIter = [50],
#                     scale = true,
#                     eig = nothing,
#                     idx = nothing
# )
#
# include("objFun.jl") # local search
#
# itr = 1
# objFun( X = X,
#         y = y,
#         study = dat[:,1],
#                     beta = fit[:,:, itr],
#                     lambda1 = lambda1[ itr ],
#                     lambda2 = lambda2[ itr ],
#                     lambda_z = 0,
#                     )

# # number of non-zeros per study (not including intercept)
# size(findall(x -> x.> 1e-9, abs.(fit[2:end, :,1])))[1] / K

# X2 = randn(size(X))
# y2 = randn(size(y))
