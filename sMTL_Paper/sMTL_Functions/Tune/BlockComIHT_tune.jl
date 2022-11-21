## Block IHT for the common support problem with strength sharing (problem (3) in the write-up)
## b: n*K observation matrix
## A: n*p*K data tensor
## s: Sparsity level (integer)
## x0: p*K initial solution
## lambda1>=0 the ridge coefficient
## lambda2>=0 the strength sharing coefficient

using TSVD, Statistics #, LinearAlgebra, Statistics

include("BlockComIHT_opt.jl") # IHT
include("BlockLS.jl") # local search

# sparse regression with IHT
function BlockComIHT(; X,
                    y,
                    rho,
                    study,
                    beta = 0,
                    scale = true,
                    lambda1 = 0,
                    lambda2 = 0,
                    maxIter = 5000,
                    eig = nothing,
                    localIter = 50,
                    )

    # rho is number of non-zero coefficient
    # beta is a feasible initial solution
    # scale -- if true then scale covaraites before fitting model
    # maxIter is maximum number of iterations
    # max eigenvalue for Lipschitz constant
    # localIter is a vector as long as lambda1/lambda2 and specifies the number of local search iterations for each lambda

    y = Array(y);
    X = Matrix(X);
    n, p = size(X); # number of covaraites
    beta = Matrix(beta); # initial value
    rho = Int64(rho);
    study = Int.(study);
    localIter = Int.(localIter);
    K = length( unique(study) ); # number of studies
    indxList = [Vector{Any}() for i in 1:K]; # list of vectors of indices of studies
    nVec = zeros(K); # vector of sample sizes of studies

    # scale covariates
    if scale
        # scale covariates like glmnet
        sdMat = ones(p, K); # K x p matrix to save std of covariates of each study
        Ysd = ones(K); # K x 1 matrix to save std of Ys

        for i = 1:K
            indx = findall(x -> x == i, study); # indices of rows for ith study
            indxList[i] = indx; # save indices
            n_k = length(indx); # study k sample size
            nVec[i] = n_k; # save sample size
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
            nVec[i] = n_k; # save sample size
        end

    end

    ## intercept
    # add column of 1s for intercept
    X = hcat(ones(n), X);
    ncol = size(X)[2]; # num coefficients (including intercept)

    # Lipschitz constant
    if isnothing(eig)
        eig = 0;
        # if not provided by user
        for i = 1:K
            indx = findall(x -> x == i, study); # indices of rows for ith study
            a2 = tsvd(X[indx,:])[2][1]; # max eigenvalue of X^T X
            if (a2 > eig)
                eig = a2
            end
        end
    end

    L = eig^2 * K / maximum(nVec) # L without regularization terms (updated in optimization fn below)

    # optimization
    vals = length(lambda1)
    βmat = zeros(ncol, K, vals) # store all of them -- last index is tuning value

    if length(localIter) < vals
        # if number of local iterations for each tuning not specified just choose first
        localIter = fill( localIter[1], vals ) # just use the first value of local search iterations for each value of lambda
    end

    for v = 1:vals
        ###############
        # IHT
        ###############
        # use warm starts as previous value
        beta = BlockComIHT_opt(X = X,
                                        y = y,
                                        rho = rho,
                                        indxList = indxList,
                                        βhat = beta,
                                        K = K,
                                        L = L,
                                        nVec = nVec,
                                        maxIter = maxIter,
                                        lambda1 = lambda1[v],
                                        lambda2 = lambda2[v],
                                        p = p
                                        )
        ###############
        # local search
        ###############
        if localIter[v] > 0
            # run local search if positive number of local search iterations for this lambda1/lambda2 value
            beta = BlockLS(X = X,
                            y = y,
                            rho = rho,
                            indxList = indxList,
                            B = beta,
                            K = K,
                            nVec = nVec,
                            lambda1 = lambda1[v],
                            lambda2 = lambda2[v],
                            p = p,
                            maxIter = localIter[v])
        end

        #############################################
        # rescale betas back to original scale
        #############################################
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

# using CSV, DataFrames
#
# # # #
# dat = CSV.read("/Users/gabeloewinger/Desktop/Research/dat_ms", DataFrame);
# X = Matrix(dat[:,3:end]);
# y = (dat[:,2]);
# lambda1 = [1 2 3]
# lambda2 = [1 2 3]
# fit = BlockComIHT(X = X,
#         y = y,
#         study = dat[:,1],
#                     beta =  zeros(51, 2),#beta;#
#                     rho = 9,
#                     lambda1 = lambda1,
#                     lambda2 = lambda2,
#                     scale = true,
#                     eig = nothing,
#                     localIter = [100 100 100])
#
#
# lambda1 =  ones(4)
# lambda2 = ones(4) #ones(4)
#
# fit = BlockComIHT(X = X,
#         y = y,
#         study = dat[:,1],
#                     beta =  zeros(51, 2),#beta;#
#                     rho = 8,
#                     lambda1 = lambda1,
#                     lambda2 = lambda2,
#                     maxIter = 10000,
#                     scale = true,
#                     eig = nothing,
#                     localIter = [0 10 0 10])
#
# include("objFun.jl") # local search
#
# itr = 2
# objFun( X = X,
#         y = y,
#         study = dat[:,1],
#                     beta = fit[:,:, itr],
#                     lambda1 = lambda1[ itr ],
#                     lambda2 = lambda2[ itr ],
#                     lambda_z = 0,
#                     )
#
# # number of non-zeros per study (not including intercept)
# size(findall(x -> x.> 1e-9, abs.(fit[2:end, :,1])))[1] / K
