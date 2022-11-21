## Block IHT INEXACT for the common support problem with strength sharing (problem (3) in the write-up)
## b: n*K observation matrix
## A: n*p*K data tensor
## s: Sparsity level (integer)
## x0: p*K initial solution
## lambda1>=0 the ridge coefficient
## lambda2>=0 the strength sharing coefficient

using TSVD, Statistics #, LinearAlgebra, Statistics

include("BlockComIHT_inexact_opt_MT.jl")
include("BlockInexactLS_MT.jl") # local search algorithm

# sparse regression with IHT
function BlockComIHT_inexact(; X,
                    y,
                    rho,
                    study = nothing, # dummy variable
                    beta = 0,
                    scale = true,
                    lambda1 = 0,
                    lambda2 = 0,
                    lambda_z = 0,
                    localIter = 50,
                    maxIter = 5000,
                    eig = nothing,
                    eigenVec = nothing,
                    WSmethod = 1,
                    ASpass = false
                    )

    # rho is number of non-zero coefficient
    # beta is a feasible initial solution
    # scale -- if true then scale covaraites before fitting model
    # maxIter is maximum number of iterations
    # max eigenvalue for Lipschitz constant
    # eigenVec is a dummy argument and is never actually used. only to make coding for the tuning code easier

    y = Matrix(y);
    X = Matrix(X);
    n, p = size(X); # number of covaraites
    beta = Matrix(beta); # initial value
    rho = Int64(rho);
    #study = Int.(study);
    K = size(y, 2); #length( unique(study) ); # number of studies
    indxList = [Vector{Any}() for i in 1:K]; # list of vectors of indices of studies
    #nVec = zeros(K); # vector of sample sizes of studies

    # scale covariates
    if scale
        # scale covariates like glmnet
        sdMat = ones(p); # K x p matrix to save std of covariates of each study
        Ysd = ones(K); # K x 1 matrix to save std of Ys

        Xsd = std(X, dims=1) .* (n - 1) / n; # glmnet style MLE of sd
        sdMat = Xsd; # save std of covariates of ith study in ith row of matrix
        X .= X ./ Xsd; # standardize ith study's covariates
        #
        # for i = 1:K
        #     # indx = findall(x -> x == i, study); # indices of rows for ith study
        #     # indxList[i] = indx; # save indices
        #     # n_k = length(indx); # study k sample size
        #     # nVec[i] = n_k; # save sample size
        #
        #     # Ysd[i] = std(y[indx]) * (n_k - 1) / n_k; # glmnet style MLE of sd of y_k
        # end

        sdMat = hcat(1, sdMat); # add row of ones so standardize intercept by ones
        beta = beta .* sdMat'; # current solution β

        # lambda = lambda / mean(Ysd); # scale tuning parameter for L2 norm by average std of y_k

    # else
    #     # otherwise just make this a vector of ones for multiplication
    #     # by coefficient estimates later
    #     # sdMat = ones(p, K); # K x p matrix to save std of covariates of each study
    #
    #     for i = 1:K
    #         indxList[i] = findall(x -> x == i, study); # indices of rows for ith study
    #         indx = indxList[i];
    #         n_k = length(indx); # study k sample size
    #         nVec[i] = n_k; # save sample size
    #     end

    end

    ## intercept
    # add column of 1s for intercept
    X = hcat(ones(n), X);
    ncol = size(X)[2]; # num coefficients (including intercept)

    # Lipschitz constant
    if isnothing(eig)
        eig = 0;
        eig = tsvd(X)[2][1]; # max eigenvalue of X^T X
        # if not provided by user
        # for i = 1:K
        #     indx = findall(x -> x == i, study); # indices of rows for ith study
        #     a2 = tsvd(X[indx,:])[2][1]; # max eigenvalue of X^T X
        #     if (a2 > eig)
        #         eig = a2
        #     end
        # end
    else
        eig = Float64(eig)
    end

    L = eig^2 * sqrt(K) / n #maximum(nVec) # L without regularization terms (updated in optimization fn below)

    # optimization
    vals = length(lambda1)
    βmat = zeros(ncol, K, vals) # store all of them -- last index is tuning value

    # number of local search iterations
    if length(localIter) < vals
        # if number of local iterations for each tuning not specified just choose first
        localIter = fill( localIter[1], vals ) # just use the first value of local search iterations for each value of lambda
    end

    for v = 1:vals

        # use warm starts as previous value
        beta = BlockComIHT_inexact_opt_MT(X = X,
                                        y = y,
                                        s = rho,
                                        #indxList = indxList,
                                        B = beta,
                                        K = K,
                                        L = L,
                                        n = n,
                                        #nVec = nVec,
                                        maxIter = maxIter,
                                        lambda1 = lambda1[v],
                                        lambda2 = lambda2[v],
                                        lambda_z = lambda_z[v],
                                        p = p
                                        )

        ###############
        # local search
        ###############
        if localIter[v] > 0
            # run local search if positive number of local search iterations for this lambda1/lambda2 value
            beta = BlockInexactLS_MT(X = X,
                                y = y,
                                s = rho,
                                beta = beta,
                                lambda1 = lambda1[v],
                                lambda2 = lambda2[v],
                                lambda_z = lambda_z[v],
                                #cindxList = indxList,
                                K = K,
                                #nVec = nVec,
                                n = n,
                                p = p,
                                maxIter = localIter[v] )
        end



        if scale
            βmat[:,:,v] = beta ./ sdMat'; # rescale by sd
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
#
# using CSV, DataFrames
#
# # # # # # #
# dat = CSV.read("/Users/gabeloewinger/Desktop/Research/dat_ms", DataFrame);
# X = Matrix(dat[:,4:end]);
# y = (dat[:,2:3]);
# lambda1 = [1 2 3]
# lambda2 = [1 2 3]
# lambda_z = [1 2 3]
# fit = BlockComIHT_inexact(X = X,
#         y = y,
#         #study = dat[:,1],
#                     beta =  zeros(50, 2),#beta;#
#                     rho = 9,
#                     lambda1 = lambda1,
#                     lambda2 = lambda2,
#                     lambda_z = lambda_z,
#                     scale = true,
#                     localIter = 50,
#                     eig = nothing)
